package tensor


import l "../linalg"
import "../util"
import "core:fmt"
import "core:math"
import "core:mem"


// tensor_backward backpropagates from `root`.
// - Scalar roots (losses) are always seeded with dL/dL = 1.
// - Non-scalar roots use a gradient already written into root.grad (legacy
//   pattern: tensor_ensure_grad + fill), or all ones if none was written.
//   Prefer tensor_backward_with_grad to pass the upstream gradient explicitly.
tensor_backward :: proc(root: ^Tensor, allocator: mem.Allocator = context.allocator) {
	if !root.requires_grad {return}

	tensor_ensure_grad(root)
	seeded := false
	if len(root.data.data) > 1 {
		for v in root.grad.data {
			if v != 0.0 {
				seeded = true
				break
			}
		}
	}
	if !seeded {
		for i in 0 ..< len(root.grad.data) {root.grad.data[i] = 1.0}
	}
	_backward_from_root(root, allocator)
}

// tensor_backward_with_grad backpropagates from `root` using `grad_output`
// (dL/droot, one value per element of root) as the upstream gradient.
tensor_backward_with_grad :: proc(
	root: ^Tensor,
	grad_output: []f64,
	allocator: mem.Allocator = context.allocator,
	loc := #caller_location,
) {
	if !root.requires_grad {return}
	fmt.assertf(
		len(grad_output) == len(root.data.data),
		"tensor_backward_with_grad: %d gradient values for a root with %d elements",
		len(grad_output),
		len(root.data.data),
		loc = loc,
	)
	tensor_ensure_grad(root)
	copy(root.grad.data, grad_output)
	_backward_from_root(root, allocator)
}

// _backward_from_root runs the reverse pass once root.grad holds the seed.
_backward_from_root :: proc(root: ^Tensor, allocator: mem.Allocator) {
	// 1. Build the topological sort
	topo := make([dynamic]^Tensor, 0, root.allocator)
	defer delete(topo) // ✅ FIX: Removed allocator

	visited := make(map[^Tensor]bool, root.allocator)
	defer delete(visited) // ✅ FIX: Removed allocator

	_build_topo(root, &topo, &visited)

	// Op outputs are created without gradient buffers; allocate them now for
	// every node that takes part in the backward pass.
	for node in topo {
		if node.requires_grad {tensor_ensure_grad(node)}
	}

	// 3. Iterate in reverse topological order
	for i := len(topo) - 1; i >= 0; i -= 1 {
		node := topo[i]

		if !node.requires_grad {continue}
		// ✅ CRITICAL: Ensure this node has a gradient allocated
		tensor_ensure_grad(node)

		fmt.assertf(
			len(node.grad.data) == len(node.data.data),
			"backward: %v node has %d gradient values for %d data values",
			node.op,
			len(node.grad.data),
			len(node.data.data),
		)
		switch node.op {
		case .Add:
			for input in node.inputs {
				if input.requires_grad {
					tensor_ensure_grad(input)
					if len(input.grad.data) > 0 {
						l.axpy_simd(1.0, node.grad.data, input.grad.data)
					}
				}
			}
		case .Clamp:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				lo := node.f64_metadata[0]
				hi := node.f64_metadata[1]
				for i in 0 ..< len(a_in.grad.data) {
					v := a_in.data.data[i]
					if v >= lo && v <= hi {
						a_in.grad.data[i] += node.grad.data[i]
					}
					// outside [lo,hi] → gradient is 0 (clamped)
				}
			}

		case .PermuteLOB:
			x_in := node.inputs[0]
			if x_in.requires_grad && len(x_in.grad.data) > 0 {
				batch := node.int_metadata[0]
				c_out := node.int_metadata[1]
				t_out := node.int_metadata[2]
				l_out := node.int_metadata[3]
				feat_dim := c_out * l_out

				// Reverse the permutation to route gradients back to [B, C, T, L]
				for b in 0 ..< batch {
					for tt in 0 ..< t_out {
						for c in 0 ..< c_out {
							for ll in 0 ..< l_out {
								src_idx := (b * t_out + tt) * feat_dim + c * l_out + ll
								dst_idx :=
									b * (c_out * t_out * l_out) +
									c * (t_out * l_out) +
									tt * l_out +
									ll
								x_in.grad.data[dst_idx] += node.grad.data[src_idx]
							}
						}
					}
				}
			}
		case .SumDim1:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				// Output is [batch, 1]; the summed width comes from the input
				batch_size := node.data.rows
				num_assets := len(a_in.data.data) / batch_size
				// The gradient of a sum is just the gradient of the output, broadcasted to all summed elements
				for b in 0 ..< batch_size {
					grad_val := node.grad.data[b]
					for asset in 0 ..< num_assets {
						a_in.grad.data[b * num_assets + asset] += grad_val
					}
				}
			}
		case .Mul:
			a_in := node.inputs[0]
			b_in := node.inputs[1]

			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 {
					l.vec_fma_inplace_simd(node.grad.data, b_in.data.data, a_in.grad.data)
				}
			}
			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				if len(b_in.grad.data) > 0 {
					l.vec_fma_inplace_simd(node.grad.data, a_in.data.data, b_in.grad.data)
				}
			}
		case .Sub:
			a_in := node.inputs[0]
			b_in := node.inputs[1]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(node.grad.data) > 0 {
					l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
				}
			}
			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				if len(b_in.grad.data) > 0 && len(node.grad.data) > 0 {
					l.vec_sub_inplace_simd(b_in.grad.data, node.grad.data) // ✅ SIMD
				}
			}

		case .Mean:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				n := f64(len(a_in.data.data))
				scalar_grad := node.grad.data[0] / n
				l.vec_broadcast_add_simd(scalar_grad, a_in.grad.data) // ✅ SIMD
			}

		case .Neg:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(node.grad.data) > 0 {
					l.vec_sub_inplace_simd(a_in.grad.data, node.grad.data) // ✅ SIMD
				}
			}

		case .MatMul:
			a_in := node.inputs[0]
			b_in := node.inputs[1]
			// Both paths are C[N, out] = A[N, in] @ B[in, out]; the flattened
			// [1, N*in] path just reinterprets A and C as N rows.
			N := a_in.data.rows
			in_features := a_in.data.cols
			out_features := b_in.data.cols
			if len(node.int_metadata) == 3 {
				N = node.int_metadata[0]
				in_features = node.int_metadata[1]
				out_features = node.int_metadata[2]
			}
			grad_view := l.Matrix(f64) {
				rows = N,
				cols = out_features,
				data = node.grad.data,
			}

			// dA += dC @ B^T
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				grad_a_view := l.Matrix(f64) {
					rows = N,
					cols = in_features,
					data = a_in.grad.data,
				}
				l.matmul_nt_acc_simd(&grad_view, &b_in.data, &grad_a_view)
			}

			// dB += A^T @ dC
			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				a_view := l.Matrix(f64) {
					rows = N,
					cols = in_features,
					data = a_in.data.data,
				}
				l.matmul_tn_acc_simd(&a_view, &grad_view, &b_in.grad)
			}
		case .Sum:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				scalar_grad := node.grad.data[0]
				l.vec_broadcast_add_simd(scalar_grad, a_in.grad.data) // ✅ SIMD
			}
		case .KLDivergence:
			mu_in := node.inputs[0]
			log_var_in := node.inputs[1]

			scalar_grad := node.grad.data[0]
			n := f64(len(mu_in.data.data))

			// ∂KL/∂mu = -0.5 * (-2 * mu) / n = mu / n
			if mu_in.requires_grad {
				tensor_ensure_grad(mu_in)
				for i in 0 ..< len(mu_in.grad.data) {
					mu_in.grad.data[i] += scalar_grad * mu_in.data.data[i] / n
				}
			}

			// ∂KL/∂log_var = -0.5 * (1 - exp(log_var)) / n = (exp(log_var) - 1) / (2n)
			if log_var_in.requires_grad {
				tensor_ensure_grad(log_var_in)
				for i in 0 ..< len(log_var_in.grad.data) {
					log_var_in.grad.data[i] +=
						scalar_grad * (math.exp(log_var_in.data.data[i]) - 1.0) / (2.0 * n)
				}
			}
		case .MaskedScaledDotProductAttention:
			// The forward stores the exact additive mask in f64_metadata
			_attention_backward(node, node.f64_metadata[:], allocator)
		case .Relu:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				_assert_same_len(node, a_in)
				l.vec_relu_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data)
			}
		case .Sigmoid:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				_assert_same_len(node, a_in)
				l.vec_sigmoid_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data) // ✅ SIMD
			}

		case .Tanh:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				_assert_same_len(node, a_in)
				l.vec_tanh_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data) // ✅ SIMD
			}
		case .Exp:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					// d/dx exp(x) = exp(x) = node.data[i]
					a_in.grad.data[i] += node.grad.data[i] * node.data.data[i]
				}
			}

		case .Sqrt:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					// d/dx sqrt(x) = 1 / (2*sqrt(x)) = 1 / (2*out)
					a_in.grad.data[i] += node.grad.data[i] / (2.0 * node.data.data[i])
				}
			}

		case .Log:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					// d/dx log(x) = 1/x
					a_in.grad.data[i] += node.grad.data[i] / a_in.data.data[i]
				}
			}

		case .Div:
			a_in := node.inputs[0]
			b_in := node.inputs[1]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					// d/da (a/b) = 1/b
					a_in.grad.data[i] += node.grad.data[i] / b_in.data.data[i]
				}
			}
			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				for i in 0 ..< len(b_in.grad.data) {
					// d/db (a/b) = -a/b²
					b_val := b_in.data.data[i]
					b_in.grad.data[i] -= node.grad.data[i] * a_in.data.data[i] / (b_val * b_val)
				}
			}

		case .NormCDF:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				inv_sqrt_2pi := 0.3989422804014327
				for i in 0 ..< len(a_in.grad.data) {
					// d/dx N(x) = φ(x) = (1/√(2π)) * exp(-x²/2)
					x := a_in.data.data[i]
					phi := math.exp(-0.5 * x * x) * inv_sqrt_2pi
					a_in.grad.data[i] += node.grad.data[i] * phi
				}
			}
		case .PermuteMHA:
			x_in := node.inputs[0]
			if x_in.requires_grad && len(x_in.grad.data) > 0 {
				batch := node.int_metadata[0]
				seq_len := node.int_metadata[1]
				num_heads := node.int_metadata[2]
				head_dim := node.int_metadata[3]
				d_model := num_heads * head_dim

				for b in 0 ..< batch {
					for s in 0 ..< seq_len {
						for h in 0 ..< num_heads {
							for d in 0 ..< head_dim {
								src :=
									(b * num_heads + h) * (seq_len * head_dim) + s * head_dim + d
								dst := b * (seq_len * d_model) + s * d_model + h * head_dim + d
								x_in.grad.data[dst] += node.grad.data[src]
							}
						}
					}
				}
			}

		case .PermuteMHAInverse:
			x_in := node.inputs[0]
			if x_in.requires_grad && len(x_in.grad.data) > 0 {
				batch := node.int_metadata[0]
				seq_len := node.int_metadata[1]
				num_heads := node.int_metadata[2]
				head_dim := node.int_metadata[3]
				d_model := num_heads * head_dim

				for b in 0 ..< batch {
					for s in 0 ..< seq_len {
						for h in 0 ..< num_heads {
							for d in 0 ..< head_dim {
								src := b * (seq_len * d_model) + s * d_model + h * head_dim + d
								dst :=
									(b * num_heads + h) * (seq_len * head_dim) + s * head_dim + d
								x_in.grad.data[dst] += node.grad.data[src]
							}
						}
					}
				}
			}
		case .LeakyReLU:
			// d/dx = 1 if x > 0 else α
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				alpha := node.f64_metadata[0]
				for i in 0 ..< len(a_in.grad.data) {
					if a_in.data.data[i] > 0.0 {
						a_in.grad.data[i] += node.grad.data[i]
					} else {
						a_in.grad.data[i] += node.grad.data[i] * alpha
					}
				}
			}
		case .Reparameterize:
			mu_in := node.inputs[0]
			log_var_in := node.inputs[1]

			if mu_in.requires_grad {
				tensor_ensure_grad(mu_in)
				// ∂z/∂mu = 1, so gradient flows directly
				for i in 0 ..< len(mu_in.grad.data) {
					mu_in.grad.data[i] += node.grad.data[i]
				}
			}

			if log_var_in.requires_grad {
				tensor_ensure_grad(log_var_in)
				// ∂z/∂log_var = 0.5 * exp(0.5 * log_var) * epsilon
				// But we don't have epsilon stored, so we use the chain rule approximation
				// ∂z/∂log_var ≈ 0.5 * (z - mu) / exp(0.5 * log_var) * exp(0.5 * log_var)
				// Simplified: ∂z/∂log_var = 0.5 * std * epsilon = 0.5 * (z - mu)
				for i in 0 ..< len(log_var_in.grad.data) {
					mu_val := mu_in.data.data[i]
					z_val := node.data.data[i]
					log_var_val := log_var_in.data.data[i]
					log_var_val = max(-10.0, min(10.0, log_var_val))
					std := math.exp(0.5 * log_var_val)

					// Gradient through std
					log_var_in.grad.data[i] +=
						node.grad.data[i] * 0.5 * std * (z_val - mu_val) / std
				}
			}
		case .AddBias:
			a_in := node.inputs[0]
			bias_in := node.inputs[1]

			// Output is N rows of width D (the flattened path stores N, D explicitly)
			N := node.grad.rows
			D := node.grad.cols
			if len(node.int_metadata) == 2 {
				N = node.int_metadata[0]
				D = node.int_metadata[1]
			}
			fmt.assertf(
				N * D == len(node.grad.data),
				"AddBias backward: grad has %d elements, expected %d x %d",
				len(node.grad.data),
				N,
				D,
			)

			if a_in.requires_grad {
				l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
			}
			if bias_in.requires_grad {
				fmt.assertf(
					len(bias_in.grad.data) == D,
					"AddBias backward: bias has %d elements, expected %d",
					len(bias_in.grad.data),
					D,
				)
				// Sum the gradient over the N rows
				for i in 0 ..< N {
					l.axpy_simd(1.0, node.grad.data[i * D:(i + 1) * D], bias_in.grad.data)
				}
			}
		case .Embedding:
			input_in := node.inputs[0] // indices (no grad)
			weight_in := node.inputs[1] // weight matrix

			if len(node.grad.data) == 0 || !weight_in.requires_grad {continue}

			batch := input_in.shape[0]
			seq_len := input_in.shape[1]
			vocab_size := weight_in.shape[0]
			embed_dim := weight_in.shape[1]

			// Ensure gradient matrix is allocated
			if weight_in.grad.data == nil {
				weight_in.grad = l.matrix_new(f64, vocab_size, embed_dim, weight_in.allocator)
			}

			// ✅ OPTIMIZATION: Scatter-add using SIMD vector addition
			for b in 0 ..< batch {
				for s in 0 ..< seq_len {
					idx := int(input_in.data.data[b * seq_len + s])
					if idx < 0 || idx >= vocab_size {continue} 	// Safety

					src_offset := (b * seq_len + s) * embed_dim
					dst_offset := idx * embed_dim

					// weight_in.grad[idx] += node.grad[b, s]
					l.vec_add_simd(
						weight_in.grad.data[dst_offset:dst_offset + embed_dim],
						node.grad.data[src_offset:src_offset + embed_dim],
						weight_in.grad.data[dst_offset:dst_offset + embed_dim],
					)
				}
			}
		// In tensor_backward, find case .ScaledDotProductAttention and update it:
		case .LogSumExpDim1:
			q_in := node.inputs[0]
			if q_in.requires_grad && len(q_in.grad.data) > 0 {
				num_actions := node.int_metadata[0]
				batch := node.data.rows
				for i in 0 ..< batch {
					max_val := -math.F64_MAX
					for a in 0 ..< num_actions {
						val := q_in.data.data[i * num_actions + a]
						if val > max_val {max_val = val}
					}
					sum_exp := 0.0
					for a in 0 ..< num_actions {
						sum_exp += math.exp(q_in.data.data[i * num_actions + a] - max_val)
					}
					grad_out := node.grad.data[i]
					for a in 0 ..< num_actions {
						prob := math.exp(q_in.data.data[i * num_actions + a] - max_val) / sum_exp
						q_in.grad.data[i * num_actions + a] += grad_out * prob
					}
				}
			}
		case .Softplus:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				for i in 0 ..< len(a_in.grad.data) {
					x := a_in.data.data[i]
					sig := 1.0 / (1.0 + math.exp(-x))
					a_in.grad.data[i] += node.grad.data[i] * sig
				}
			}

		case .SSM:
			_ssm_backward(node, allocator)
		case .ScaledDotProductAttention:
			_attention_backward(node, nil, allocator)
		case .MSELoss:
			// L = mean((pred - target)^2)
			// dL/dpred = 2 * (pred - target) / N
			pred_in := node.inputs[0]
			target_in := node.inputs[1]

			// dL/dtarget = -dL/dpred
			if pred_in.requires_grad || target_in.requires_grad {
				n := f64(len(pred_in.data.data))
				scale := 2.0 * node.grad.data[0] / n

				diff := make([]f64, len(pred_in.data.data), allocator)
				defer delete(diff, allocator)
				l.vec_sub_simd(pred_in.data.data, target_in.data.data, diff)

				if pred_in.requires_grad {l.axpy_simd(scale, diff, pred_in.grad.data)}
				if target_in.requires_grad {l.axpy_simd(-scale, diff, target_in.grad.data)}
			}
		case .LayerNorm:
			input_in := node.inputs[0]
			gamma_in := node.inputs[1]
			beta_in := node.inputs[2]

			if len(node.grad.data) == 0 {continue}

			// ✅ FIX: Use stored metadata
			N := node.int_metadata[0]
			d_model := node.int_metadata[1]
			eps := 1e-5

			// ✅ FIX: Ensure gradient matrices are allocated before accumulating
			if input_in.requires_grad {tensor_ensure_grad(input_in)}
			if gamma_in.requires_grad {tensor_ensure_grad(gamma_in)}
			if beta_in.requires_grad {tensor_ensure_grad(beta_in)}

			// Temporary buffers
			centered := make([]f64, d_model, allocator)
			x_hat := make([]f64, d_model, allocator)
			dx_hat := make([]f64, d_model, allocator)
			dx_row := make([]f64, d_model, allocator)
			inv_std_vec := make([]f64, d_model, allocator)
			defer {
				delete(centered, allocator)
				delete(x_hat, allocator)
				delete(dx_hat, allocator)
				delete(dx_row, allocator)
				delete(inv_std_vec, allocator)
			}

			for i in 0 ..< N {
				row_start := i * d_model
				row := input_in.data.data[row_start:row_start + d_model]
				dy_row := node.grad.data[row_start:row_start + d_model]
				gamma_row := gamma_in.data.data

				// Recompute forward stats
				mean := l.sum_simd(row) / f64(d_model)
				for j in 0 ..< d_model {
					centered[j] = row[j] - mean
				}
				var := l.dot_simd(centered, centered) / f64(d_model)
				std := math.sqrt(var + eps)
				inv_std := 1.0 / std

				for j in 0 ..< d_model {inv_std_vec[j] = inv_std}
				l.vec_mul_simd(centered, inv_std_vec, x_hat)

				// dx_hat = dy * gamma (SIMD)
				l.vec_mul_simd(dy_row, gamma_row, dx_hat)

				// Stable O(N) backward formula
				sum_dx_hat := l.sum_simd(dx_hat)
				sum_dx_hat_x_hat := l.dot_simd(dx_hat, x_hat)

				// term1 = d_model * dx_hat
				d_model_f := f64(d_model)
				for j in 0 ..< d_model {dx_row[j] = d_model_f * dx_hat[j]}

				// term1 -= sum_dx_hat
				for j in 0 ..< d_model {dx_row[j] -= sum_dx_hat}

				// term1 -= x_hat * sum_dx_hat_x_hat
				for j in 0 ..< d_model {dx_row[j] -= x_hat[j] * sum_dx_hat_x_hat}

				// dx = dx_row / (d_model * std)
				inv_denom := 1.0 / (d_model_f * std)
				for j in 0 ..< d_model {dx_row[j] *= inv_denom}

				// ✅ Accumulate dx to input gradient
				if input_in.requires_grad && len(input_in.grad.data) > 0 {
					dx_grad_row := input_in.grad.data[row_start:row_start + d_model]
					l.vec_add_simd(dx_grad_row, dx_row, dx_grad_row)
				}

				// ✅ Accumulate dgamma: dgamma += dy * x_hat
				if gamma_in.requires_grad && len(gamma_in.grad.data) > 0 {
					dgamma_row := gamma_in.grad.data
					for j in 0 ..< d_model {
						dgamma_row[j] += dy_row[j] * x_hat[j]
					}
				}

				// ✅ Accumulate dbeta: dbeta += dy
				if beta_in.requires_grad && len(beta_in.grad.data) > 0 {
					dbeta_row := beta_in.grad.data
					l.vec_add_simd(dbeta_row, dy_row, dbeta_row)
				}
			}
		case .NormalizeTime:
			input_in := node.inputs[0]
			if input_in.requires_grad && len(input_in.grad.data) > 0 {
				N := node.int_metadata[0]
				C := node.int_metadata[1]
				T := node.int_metadata[2]
				L := node.int_metadata[3]
				eps := node.f64_metadata[0]

				T_f := f64(T)
				// Per-slice buffers, allocated once for all (n, c, l) slices
				grad_buf := make([]f64, T, allocator)
				norm_buf := make([]f64, T, allocator)
				temp := make([]f64, T, allocator)
				in_buf := make([]f64, T, allocator)
				in_centered := make([]f64, T, allocator)
				defer {
					delete(grad_buf, allocator)
					delete(norm_buf, allocator)
					delete(temp, allocator)
					delete(in_buf, allocator)
					delete(in_centered, allocator)
				}

				for n: int = 0; n < N; n += 1 {
					for c: int = 0; c < C; c += 1 {
						for el: int = 0; el < L; el += 1 {
							// Extract gradients and normalized values
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								grad_buf[t] = node.grad.data[idx]
								norm_buf[t] = node.data.data[idx] // This is the normalized input
							}

							// Gradient of Z-score: dx = (dy - mean(dy) - y * dot(dy, y) / T) / std
							sum_grad := l.sum_simd(grad_buf)
							mean_grad := sum_grad / T_f
							dot_grad_y := l.dot_simd(grad_buf, norm_buf)

							for t: int = 0; t < T; t += 1 {temp[t] = grad_buf[t] - mean_grad}

							scalar := dot_grad_y / T_f
							for t: int = 0; t < T; t += 1 {
								temp[t] -= norm_buf[t] * scalar
							}

							// Recompute std of input for this slice to divide the gradient
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								in_buf[t] = input_in.data.data[idx]
							}
							in_mean := l.sum_simd(in_buf) / T_f
							for t: int = 0; t < T; t += 1 {in_centered[t] = in_buf[t] - in_mean}
							in_var := l.dot_simd(in_centered, in_centered) / T_f
							in_std := math.sqrt(in_var + eps)
							inv_std := 1.0 / in_std

							for t: int = 0; t < T; t += 1 {temp[t] *= inv_std}

							// Accumulate to input gradient
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								input_in.grad.data[idx] += temp[t]
							}
						}
					}
				}
			}
		case .GRU:
			_gru_backward(node, allocator)
		case .LSTM:
			_lstm_backward(node, allocator)
		case .Scale:
			a_in := node.inputs[0]
			scalar := node.f64_metadata[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				// a_in.grad += scalar * node.grad
				l.axpy_simd(scalar, node.grad.data, a_in.grad.data)
			}
		case .CrossEntropy:
			// The magical simplified gradient: grad = (softmax_prob - target_one_hot) / N
			logits_in := node.inputs[0]
			scalar_grad := node.grad.data[0] // Usually 1.0

			// Stored by the forward: [N, C, target_0 .. target_{N-1}]
			N := node.int_metadata[0]
			C := node.int_metadata[1]
			fmt.assertf(
				len(node.int_metadata) == 2 + N && N * C == len(logits_in.data.data),
				"CrossEntropy backward: metadata (%d entries) does not match N=%d, C=%d",
				len(node.int_metadata),
				N,
				C,
			)

			for i in 0 ..< N {
				// Recompute softmax for stability
				max_val := -math.F64_MAX
				for j in 0 ..< C {
					v := logits_in.data.data[i * C + j]
					if v > max_val {max_val = v}
				}

				sum_exp := 0.0
				for j in 0 ..< C {
					sum_exp += math.exp(logits_in.data.data[i * C + j] - max_val)
				}

				target_class := node.int_metadata[2 + i]

				for j in 0 ..< C {
					prob := math.exp(logits_in.data.data[i * C + j] - max_val) / sum_exp
					target_val := 0.0
					if j == target_class {target_val = 1.0}

					// Apply chain rule
					logit_grad := (prob - target_val) * scalar_grad / f64(N)
					logits_in.grad.data[i * C + j] += logit_grad
				}
			}
		case .Dropout:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// ✅ SIMD Optimization: grad_a += node.grad * mask
				// vec_fma_inplace_simd does: c += a * b
				l.vec_fma_inplace_simd(node.grad.data, node.dropout_mask, a_in.grad.data)
			}
		case .Flatten:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// Same flat layout: accumulate gradient 1:1
				tensor_ensure_grad(a_in)
				l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
			}
		case .BCELoss:
			prediction := node.inputs[0]
			target := node.inputs[1]

			if prediction.requires_grad {
				tensor_ensure_grad(prediction)
				n := len(prediction.data.data)
				grad_scalar := node.grad.data[0]

				// Gradient of BCE loss w.r.t. prediction:
				// dL/dp = -(t/p - (1-t)/(1-p)) / n
				for i in 0 ..< n {
					p := prediction.data.data[i]
					t := target.data.data[i]

					// Clamp prediction for numerical stability
					p = math.max(1e-7, math.min(1.0 - 1e-7, p))

					// Gradient: -(t/p - (1-t)/(1-p))
					grad := -(t / p - (1.0 - t) / (1.0 - p)) / f64(n)
					prediction.grad.data[i] += grad_scalar * grad
				}
			}
			if target.requires_grad {
				n := len(prediction.data.data)
				grad_scalar := node.grad.data[0]
				// dL/dt = (ln(1-p) - ln p) / n, with the same clamping as the forward
				for i in 0 ..< n {
					p := math.max(1e-7, math.min(1.0 - 1e-7, prediction.data.data[i]))
					target.grad.data[i] += grad_scalar * (math.ln(1.0 - p) - math.ln(p)) / f64(n)
				}
			}
		case .MaxPool2d:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// int_metadata holds the argmax input index of every output element
				fmt.assertf(
					len(node.int_metadata) == len(node.grad.data),
					"MaxPool2d backward: %d argmax indices for %d outputs",
					len(node.int_metadata),
					len(node.grad.data),
				)
				for i in 0 ..< len(node.grad.data) {
					a_in.grad.data[node.int_metadata[i]] += node.grad.data[i]
				}
			}
		case .BatchNorm2d:
			input_in := node.inputs[0]
			weight_in := node.inputs[1]
			bias_in := node.inputs[2]

			if len(node.grad.data) == 0 {continue}

			N := input_in.shape[0]
			C := input_in.shape[1]
			H := input_in.shape[2]
			W := input_in.shape[3]
			N_hw := f64(N * H * W)
			channel_size := N * H * W
			eps := 1e-5

			// Contiguous per-channel buffers, allocated once for all channels
			x_buf := make([]f64, channel_size, allocator)
			dout_buf := make([]f64, channel_size, allocator)
			centered := make([]f64, channel_size, allocator)
			x_hat := make([]f64, channel_size, allocator)
			dout_gamma := make([]f64, channel_size, allocator)
			defer {
				delete(x_buf, allocator)
				delete(dout_buf, allocator)
				delete(centered, allocator)
				delete(x_hat, allocator)
				delete(dout_gamma, allocator)
			}

			for c in 0 ..< C {
				// 1. Extract channel data and grad to contiguous buffers
				for n in 0 ..< N {
					for h in 0 ..< H {
						for w in 0 ..< W {
							idx := n * (C * H * W) + c * (H * W) + h * W + w
							buf_idx := n * (H * W) + h * W + w
							x_buf[buf_idx] = input_in.data.data[idx]
							dout_buf[buf_idx] = node.grad.data[idx]
						}
					}
				}

				// 2. Compute mean (SIMD sum)
				mean := l.sum_simd(x_buf) / N_hw

				// 3. Compute variance (SIMD)
				for i in 0 ..< channel_size {centered[i] = x_buf[i] - mean}

				var := l.dot_simd(centered, centered) / N_hw
				std := math.sqrt(var + eps)
				inv_std := 1.0 / std
				gamma := weight_in.data.data[c]

				// 4. Compute x_hat
				for i in 0 ..< channel_size {x_hat[i] = centered[i] * inv_std}

				// 5. Compute weight gradient: dgamma = dot(dout, x_hat) (SIMD)
				if weight_in.requires_grad && len(weight_in.grad.data) > 0 {
					dgamma := l.dot_simd(dout_buf, x_hat)
					weight_in.grad.data[c] += dgamma
				}

				// 6. Compute bias gradient: dbeta = sum(dout) (SIMD)
				if bias_in.requires_grad && len(bias_in.grad.data) > 0 {
					dbeta := l.sum_simd(dout_buf)
					bias_in.grad.data[c] += dbeta
				}

				// 7. Compute input gradient
				if input_in.requires_grad && len(input_in.grad.data) > 0 {
					for i in 0 ..< channel_size {dout_gamma[i] = dout_buf[i] * gamma}

					// Precompute sums (SIMD)
					sum_dout_gamma := l.sum_simd(dout_gamma)
					sum_dout_gamma_x_hat := l.dot_simd(dout_gamma, x_hat)

					// dx = (N_hw * dout_gamma - sum_dout_gamma - x_hat * sum_dout_gamma_x_hat) / (N_hw * std)
					inv_denom := 1.0 / (N_hw * std)
					for n in 0 ..< N {
						for h in 0 ..< H {
							for w in 0 ..< W {
								idx := n * (C * H * W) + c * (H * W) + h * W + w
								i := n * (H * W) + h * W + w
								dx :=
									((dout_gamma[i] * N_hw) - sum_dout_gamma) -
									(x_hat[i] * sum_dout_gamma_x_hat)
								input_in.grad.data[idx] += dx * inv_denom
							}
						}
					}
				}
			}
		case .SharpeLoss:
			ret_in := node.inputs[0]
			if ret_in.requires_grad && len(ret_in.grad.data) > 0 {
				n := f64(node.int_metadata[0])
				rf := node.f64_metadata[0]
				mean := node.f64_metadata[1]
				std := node.f64_metadata[2]

				// Analytical gradient of L = -Sharpe w.r.t x_i:
				// dL/dx_i = ( (mean - rf) * (x_i - mean) - std^2 ) / (n * std^3)
				scalar_grad := node.grad.data[0]
				std_sq := std * std
				std_cube := std_sq * std
				denominator := n * std_cube
				excess_mean := mean - rf

				for i in 0 ..< len(ret_in.grad.data) {
					x_i := ret_in.data.data[i]
					grad := ((excess_mean * (x_i - mean)) - std_sq) / denominator
					ret_in.grad.data[i] += scalar_grad * grad
				}
			}
		case .RNN:
			_rnn_backward(node, allocator)
		case .BinaryCrossEntropy:
			pred_in := node.inputs[0]
			target_in := node.inputs[1]

			if pred_in.requires_grad {
				eps := 1e-7
				n := f64(len(pred_in.data.data))
				scalar_grad := node.grad.data[0]

				for i in 0 ..< len(pred_in.data.data) {
					p := math.max(math.min(pred_in.data.data[i], 1.0 - eps), eps)
					t := target_in.data.data[i]
					grad := scalar_grad * (-t / p + (1.0 - t) / (1.0 - p)) / n
					pred_in.grad.data[i] += grad
				}
			}
			if target_in.requires_grad {
				eps := 1e-7
				n := f64(len(pred_in.data.data))
				scalar_grad := node.grad.data[0]
				for i in 0 ..< len(pred_in.data.data) {
					p := math.max(math.min(pred_in.data.data[i], 1.0 - eps), eps)
					target_in.grad.data[i] += scalar_grad * (math.ln(1.0 - p) - math.ln(p)) / n
				}
			}
		case .AvgPool2d:
			// Distribute gradient equally to all elements in the pooling window
			a_in := node.inputs[0]
			if a_in.requires_grad {
				N := a_in.shape[0]
				C := a_in.shape[1]
				H := a_in.shape[2]
				W := a_in.shape[3]
				kH := node.pool_params.kH
				kW := node.pool_params.kW
				stride := node.pool_params.stride
				out_h := node.shape[2]
				out_w := node.shape[3]
				pool_size := f64(kH * kW)

				for n in 0 ..< N {
					for c in 0 ..< C {
						for oh in 0 ..< out_h {
							for ow in 0 ..< out_w {
								out_idx :=
									n * (C * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow
								grad := node.grad.data[out_idx] / pool_size

								for kh in 0 ..< kH {
									for kw in 0 ..< kW {
										ih := oh * stride + kh
										iw := ow * stride + kw
										in_idx := n * (C * H * W) + c * (H * W) + ih * W + iw
										a_in.grad.data[in_idx] += grad
									}
								}
							}
						}
					}
				}
			}
		case .Softmax:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				// Per-row Jacobian-vector product: dx = p * (g - sum(p * g))
				C := node.int_metadata[0]
				N := len(node.data.data) / C
				for r in 0 ..< N {
					p := node.data.data[r * C:(r + 1) * C]
					g := node.grad.data[r * C:(r + 1) * C]
					dx := a_in.grad.data[r * C:(r + 1) * C]
					sum_p_grad := l.dot_simd(p, g)
					for j in 0 ..< C {
						dx[j] += p[j] * (g[j] - sum_p_grad)
					}
				}
			}
		case .Entropy:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				scalar_grad := node.grad.data[0]
				for i in 0 ..< len(a_in.grad.data) {
					p := a_in.data.data[i]
					if p > 1e-10 {
						// d(-sum p ln p)/dp = -(1 + ln p)
						a_in.grad.data[i] += scalar_grad * (-(1.0 + math.ln_f64(p)))
					}
				}
			}
		case .Concat:
			dim := node.int_metadata[0]
			num_tensors := len(node.inputs)

			// Calculate total dim size for source indexing
			total_dim_size := 0
			for j in 0 ..< num_tensors {
				total_dim_size += node.int_metadata[1 + j]
			}

			offset := 0
			for i in 0 ..< num_tensors {
				t_in := node.inputs[i]
				// ✅ FIX: Declare t_dim_size outside the if block so it's visible for offset update
				t_dim_size := node.int_metadata[1 + i]

				if t_in.requires_grad && len(t_in.grad.data) > 0 {
					outer_blocks := 1
					for d in 0 ..< dim {
						outer_blocks *= t_in.shape[d]
					}

					inner_block_size := 1
					for d in dim + 1 ..< 4 {
						inner_block_size *= t_in.shape[d]
					}

					for ob in 0 ..< outer_blocks {
						for c in 0 ..< t_dim_size {
							src_start :=
								ob * (total_dim_size * inner_block_size) +
								(offset + c) * inner_block_size
							dst_start :=
								ob * (t_dim_size * inner_block_size) + c * inner_block_size

							for k in 0 ..< inner_block_size {
								t_in.grad.data[dst_start + k] += node.grad.data[src_start + k]
							}
						}
					}
				}
				// ✅ FIX: Now t_dim_size is in scope here
				offset += t_dim_size
			}
		case .Conv2d:
			_conv2d_backward(node, allocator)
		case .FlashAttention:
			Q_in := node.inputs[0]
			K_in := node.inputs[1]
			V_in := node.inputs[2]
			if len(node.grad.data) == 0 {continue}

			is_causal := false
			if len(node.int_metadata) > 0 && node.int_metadata[0] == 1 {
				is_causal = true
			}

			batch := Q_in.shape[0]
			seq_len := Q_in.shape[1]
			d_k := Q_in.shape[2]
			d_v := V_in.shape[2]
			scale := 1.0 / math.sqrt(f64(d_k))

			if Q_in.requires_grad {tensor_ensure_grad(Q_in)}
			if K_in.requires_grad {tensor_ensure_grad(K_in)}
			if V_in.requires_grad {tensor_ensure_grad(V_in)}

			Bc := 64
			if seq_len < Bc {Bc = seq_len}

			s_tile := make([]f64, Bc, allocator)
			p_tile := make([]f64, Bc, allocator)
			defer {
				delete(s_tile, allocator)
				delete(p_tile, allocator)
			}

			for b in 0 ..< batch {
				for i in 0 ..< seq_len {
					q_offset := (b * seq_len + i) * d_k
					q_row := Q_in.data.data[q_offset:q_offset + d_k]

					dO_offset := (b * seq_len + i) * d_v
					dO_row := node.grad.data[dO_offset:dO_offset + d_v]

					// Pass 1: Recompute Softmax stats (m_i, l_i) and D_i
					m_i: f64 = -math.F64_MAX
					l_i: f64 = 0.0
					D_i: f64 = 0.0

					for j_start := 0; j_start < seq_len; j_start += Bc {
						if is_causal && j_start > i {break}
						j_end := j_start + Bc
						if j_end > seq_len {j_end = seq_len}
						curr_Bc := j_end - j_start

						block_max := -math.F64_MAX
						for x in 0 ..< curr_Bc {
							j_idx := j_start + x
							if is_causal && j_idx > i {
								s_tile[x] = -math.F64_MAX
							} else {
								k_offset := (b * seq_len + j_idx) * d_k
								k_row := K_in.data.data[k_offset:k_offset + d_k]
								s_val := l.dot_simd(q_row, k_row) * scale
								s_tile[x] = s_val
								if s_val > block_max {block_max = s_val}
							}
						}

						new_m := m_i
						if block_max > new_m {new_m = block_max}
						correction := math.exp(m_i - new_m)

						block_sum: f64 = 0.0
						block_D: f64 = 0.0

						for x in 0 ..< curr_Bc {
							j_idx := j_start + x
							p_val := math.exp(s_tile[x] - new_m)
							p_tile[x] = p_val

							if !(is_causal && j_idx > i) {
								block_sum += p_val
								v_offset := (b * seq_len + j_idx) * d_v
								v_row := V_in.data.data[v_offset:v_offset + d_v]
								block_D += p_val * l.dot_simd(dO_row, v_row)
							}
						}

						D_i = D_i * correction + block_D
						l_i = l_i * correction + block_sum
						m_i = new_m
					}


					// Pass 2: Compute dQ_i, and scatter dK_j, dV_j
					inv_l := 1.0 / l_i
					D_i *= inv_l // ✅ FIX: Normalize D_i by the softmax denominator
					dQ_row := Q_in.grad.data[q_offset:q_offset + d_k]

					for j_start := 0; j_start < seq_len; j_start += Bc {
						if is_causal && j_start > i {break}
						j_end := j_start + Bc
						if j_end > seq_len {j_end = seq_len}
						curr_Bc := j_end - j_start

						for x in 0 ..< curr_Bc {
							j_idx := j_start + x
							if is_causal && j_idx > i {continue}

							k_offset := (b * seq_len + j_idx) * d_k
							k_row := K_in.data.data[k_offset:k_offset + d_k]

							s_val := l.dot_simd(q_row, k_row) * scale
							p_val := math.exp(s_val - m_i) * inv_l

							v_offset := (b * seq_len + j_idx) * d_v
							v_row := V_in.data.data[v_offset:v_offset + d_v]

							dot_dO_V := l.dot_simd(dO_row, v_row)
							dS_ij := p_val * (dot_dO_V - D_i) * scale

							// ✅ SIMD Optimization: Replaced scalar loops with axpy_simd
							if V_in.requires_grad {
								dV_row := V_in.grad.data[v_offset:v_offset + d_v]
								l.axpy_simd(p_val, dO_row, dV_row)
							}
							if Q_in.requires_grad {
								l.axpy_simd(dS_ij, k_row, dQ_row)
							}
							if K_in.requires_grad {
								dK_row := K_in.grad.data[k_offset:k_offset + d_k]
								l.axpy_simd(dS_ij, q_row, dK_row)
							}
						}
					}
				}
			}
		case .GateMul:
			x_in := node.inputs[0]
			gate_in := node.inputs[1]

			if x_in.requires_grad {
				tensor_ensure_grad(x_in)
				for n in 0 ..< x_in.shape[0] {
					g := gate_in.data.data[n]
					for d in 0 ..< x_in.shape[1] {
						idx := n * x_in.shape[1] + d
						x_in.grad.data[idx] += node.grad.data[idx] * g
					}
				}
			}

			if gate_in.requires_grad {
				tensor_ensure_grad(gate_in)
				for n in 0 ..< x_in.shape[0] {
					sum_d: f64 = 0.0
					for d in 0 ..< x_in.shape[1] {
						idx := n * x_in.shape[1] + d
						sum_d += node.grad.data[idx] * x_in.data.data[idx]
					}
					gate_in.grad.data[n] += sum_d
				}
			}
		case .Reshape:
			input_in := node.inputs[0]
			if input_in.requires_grad {
				tensor_ensure_grad(input_in)
				// Gradients flow back 1:1 because the memory layout is unchanged
				l.vec_add_simd(node.grad.data, input_in.grad.data, input_in.grad.data)
			}
		case .SliceGate:
			gates_in := node.inputs[0]
			if gates_in.requires_grad {
				tensor_ensure_grad(gates_in)
				expert_idx := node.int_metadata[0]
				E := node.int_metadata[1]
				N := gates_in.data.rows
				// Scatter the gradient back to the exact column it came from
				for n in 0 ..< N {
					gates_in.grad.data[n * E + expert_idx] += node.grad.data[n]
				}
			}
		// ============================================================================
		// Prospect Value Backward
		// ============================================================================
		case .ProspectValue:
			x_in := node.inputs[0]
			if x_in.requires_grad {
				tensor_ensure_grad(x_in)
				alpha := node.f64_metadata[0]
				beta := node.f64_metadata[1]
				lambda := node.f64_metadata[2]

				for i in 0 ..< len(x_in.data.data) {
					x := x_in.data.data[i]
					grad_out := node.grad.data[i]

					if x > 1e-10 {
						// dv/dx = α · x^(α-1)
						x_in.grad.data[i] += grad_out * alpha * math.pow(x, alpha - 1.0)
					} else if x < -1e-10 {
						// dv/dx = λ · β · (-x)^(β-1)
						x_in.grad.data[i] += grad_out * lambda * beta * math.pow(-x, beta - 1.0)
					}
					// At x ≈ 0: subgradient = 0
				}
			}

		// ============================================================================
		// Probability Weighting Backward
		// ============================================================================
		case .ProbWeighting:
			p_in := node.inputs[0]
			if p_in.requires_grad {
				tensor_ensure_grad(p_in)
				gamma := node.f64_metadata[0]
				eps := 1e-7

				for i in 0 ..< len(p_in.data.data) {
					pi := math.max(eps, math.min(1.0 - eps, p_in.data.data[i]))
					w := node.data.data[i] // w(p) from forward pass

					p_gamma := math.pow(pi, gamma)
					q_gamma := math.pow(1.0 - pi, gamma)
					denom_inner := p_gamma + q_gamma

					// dw/dp = w · [γ/p - (p^(γ-1) - (1-p)^(γ-1)) / (p^γ + (1-p)^γ)]
					term1 := gamma / pi
					term2 :=
						(math.pow(pi, gamma - 1.0) - math.pow(1.0 - pi, gamma - 1.0)) / denom_inner
					dw_dp := w * (term1 - term2)

					p_in.grad.data[i] += node.grad.data[i] * dw_dp
				}
			}

		// ============================================================================
		// Prospect Utility Backward (fused, like SharpeLoss)
		// ============================================================================
		case .ProspectUtility:
			x_in := node.inputs[0]
			if x_in.requires_grad {
				tensor_ensure_grad(x_in)
				alpha := node.f64_metadata[0]
				beta := node.f64_metadata[1]
				lambda := node.f64_metadata[2]
				n := f64(len(x_in.data.data))

				// d(-U)/dx_i = -dv/dx_i / n
				for i in 0 ..< len(x_in.data.data) {
					x := x_in.data.data[i]
					grad_out := node.grad.data[0] // scalar gradient flows to all elements

					if x > 1e-10 {
						x_in.grad.data[i] += grad_out * (-alpha * math.pow(x, alpha - 1.0)) / n
					} else if x < -1e-10 {
						x_in.grad.data[i] +=
							grad_out * (-lambda * beta * math.pow(-x, beta - 1.0)) / n
					}
				}
			}
		case .None, .TopKMask:
		// Leaf node, nothing to do
		}
	}
}

// ============================================================================
// Layer backward helpers
// All matmuls go through the accumulating linalg kernels, so no transposes
// or temporary result matrices are created, and gradients are written
// straight into the input .grad buffers.
// ============================================================================

// _mat returns a [rows, cols] row-major view over existing data (no allocation)
_mat :: proc(rows, cols: int, data: []f64) -> l.Matrix(f64) {
	return l.Matrix(f64){rows = rows, cols = cols, data = data}
}

// _grad_view returns a [rows, cols] row-major view over a tensor's gradient
_grad_view :: proc(t: ^Tensor, rows, cols: int) -> l.Matrix(f64) {
	return _mat(rows, cols, t.grad.data)
}

// _attention_backward handles both plain and masked scaled dot-product attention.
// mask is an additive [seq_q, seq_k] mask (nil for none), identical to the forward.
_attention_backward :: proc(node: ^Tensor, mask: []f64, allocator: mem.Allocator) {
	Q_in := node.inputs[0]
	K_in := node.inputs[1]
	V_in := node.inputs[2]

	batch := Q_in.shape[0]
	seq_q := Q_in.shape[1]
	seq_k := K_in.shape[1]
	d_k := Q_in.shape[2]
	d_v := V_in.shape[2]
	scale := 1.0 / math.sqrt(f64(d_k))

	p_buf := make([]f64, seq_q * seq_k, allocator)
	ds_buf := make([]f64, seq_q * seq_k, allocator)
	defer {
		delete(p_buf, allocator)
		delete(ds_buf, allocator)
	}

	for b in 0 ..< batch {
		q_b := _mat(seq_q, d_k, Q_in.data.data[b * seq_q * d_k:(b + 1) * seq_q * d_k])
		k_b := _mat(seq_k, d_k, K_in.data.data[b * seq_k * d_k:(b + 1) * seq_k * d_k])
		v_b := _mat(seq_k, d_v, V_in.data.data[b * seq_k * d_v:(b + 1) * seq_k * d_v])
		dO_b := _mat(seq_q, d_v, node.grad.data[b * seq_q * d_v:(b + 1) * seq_q * d_v])

		// 1. Recompute P = softmax(scale * Q K^T + mask) (checkpointing)
		P := _mat(seq_q, seq_k, p_buf)
		mem.zero_slice(p_buf)
		l.matmul_nt_acc_simd(&q_b, &k_b, &P)
		for i in 0 ..< seq_q * seq_k {
			p_buf[i] *= scale
			if mask != nil {p_buf[i] += mask[i]}
		}
		for i in 0 ..< seq_q {
			row := p_buf[i * seq_k:(i + 1) * seq_k]
			max_val := row[0]
			for v in row[1:] {if v > max_val {max_val = v}}
			sum_exp := 0.0
			for j in 0 ..< seq_k {
				row[j] = math.exp(row[j] - max_val)
				sum_exp += row[j]
			}
			inv_sum := 1.0 / sum_exp
			for j in 0 ..< seq_k {row[j] *= inv_sum}
		}

		// 2. dV += P^T @ dO
		if V_in.requires_grad {
			dV_b := _mat(seq_k, d_v, V_in.grad.data[b * seq_k * d_v:(b + 1) * seq_k * d_v])
			l.matmul_tn_acc_simd(&P, &dO_b, &dV_b)
		}

		if !Q_in.requires_grad && !K_in.requires_grad {continue}

		// 3. dP = dO @ V^T, then dS = scale * P * (dP - rowsum(dP * P))
		dS := _mat(seq_q, seq_k, ds_buf)
		mem.zero_slice(ds_buf)
		l.matmul_nt_acc_simd(&dO_b, &v_b, &dS)
		for i in 0 ..< seq_q {
			p_row := p_buf[i * seq_k:(i + 1) * seq_k]
			ds_row := ds_buf[i * seq_k:(i + 1) * seq_k]
			row_dot := l.dot_simd(ds_row, p_row)
			for j in 0 ..< seq_k {
				ds_row[j] = scale * p_row[j] * (ds_row[j] - row_dot)
			}
		}

		// 4. dQ += dS @ K,  dK += dS^T @ Q
		if Q_in.requires_grad {
			dQ_b := _mat(seq_q, d_k, Q_in.grad.data[b * seq_q * d_k:(b + 1) * seq_q * d_k])
			l.matmul_nn_acc_simd(&dS, &k_b, &dQ_b)
		}
		if K_in.requires_grad {
			dK_b := _mat(seq_k, d_k, K_in.grad.data[b * seq_k * d_k:(b + 1) * seq_k * d_k])
			l.matmul_tn_acc_simd(&dS, &q_b, &dK_b)
		}
	}
}

// _scatter_add_step adds src [batch, width] into timestep s of a [batch, seq_len, width] sequence
_scatter_add_step :: proc(dst, src: []f64, batch, seq_len, width, s: int) {
	for b in 0 ..< batch {
		off := (b * seq_len + s) * width
		l.vec_add_simd(dst[off:off + width], src[b * width:(b + 1) * width], dst[off:off + width])
	}
}

// Recurrent backward passes read the states and gates the forward pass saved
// in node.f64_metadata (see tensor_rnn / tensor_gru / tensor_lstm), then:
//   1. backpropagate through time; only dh_prev = d_hh @ W_hh^T is sequential.
//      The pre-activation gate gradients of every step are kept;
//   2. the parameter and input gradients for all steps at once (_recurrent_grads).
// All per-step buffers are stored "reverse step-major": block p holds timestep
// seq_len-1-p, the order in which backpropagation visits the steps. The stacked
// products then add up their terms in exactly the order the per-step updates
// did, so the gradients are bit-identical to accumulating step by step.

// _Recurrent_Bufs: the per-step data phase 2 needs, all reverse step-major
_Recurrent_Bufs :: struct {
	xs:     []f64, // inputs        [S*B, in_size]
	h_prev: []f64, // states entering each step [S*B, H]
	d_ih:   []f64, // gate grads on the input path  [S*B, G]
	d_hh:   []f64, // gate grads on the hidden path [S*B, G] (may alias d_ih)
}

// _recurrent_grads accumulates, over all steps at once:
//   dbias += sum d_ih,   dW_ih += xs^T @ d_ih,   dW_hh += h_prev^T @ d_hh,
//   dx[s] += d_ih[s] @ W_ih^T
_recurrent_grads :: proc(
	x_in, w_ih_in, w_hh_in, bias_in: ^Tensor,
	bufs: _Recurrent_Bufs,
	batch, seq_len, in_size, H, G: int,
	allocator: mem.Allocator,
) {
	rows := seq_len * batch
	d_ih_mat := _mat(rows, G, bufs.d_ih)

	if bias_in.requires_grad {
		db := bias_in.grad.data[0:G]
		for r in 0 ..< rows {
			l.vec_add_simd(db, bufs.d_ih[r * G:(r + 1) * G], db)
		}
	}
	if w_ih_in.requires_grad {
		xs_mat := _mat(rows, in_size, bufs.xs)
		dw := _grad_view(w_ih_in, in_size, G)
		l.matmul_tn_acc_simd(&xs_mat, &d_ih_mat, &dw)
	}
	if w_hh_in.requires_grad {
		h_mat := _mat(rows, H, bufs.h_prev)
		d_hh_mat := _mat(rows, G, bufs.d_hh)
		dw := _grad_view(w_hh_in, H, G)
		l.matmul_tn_acc_simd(&h_mat, &d_hh_mat, &dw)
	}
	if x_in.requires_grad {
		dxs := make([]f64, rows * in_size, allocator)
		defer delete(dxs, allocator)
		dxs_mat := _mat(rows, in_size, dxs)
		w_ih_mat := _mat(in_size, G, w_ih_in.data.data)
		l.matmul_nt_acc_simd(&d_ih_mat, &w_ih_mat, &dxs_mat)
		BI := batch * in_size
		for p in 0 ..< seq_len {
			_scatter_add_step(x_in.grad.data, dxs[p * BI:(p + 1) * BI], batch, seq_len, in_size, seq_len - 1 - p)
		}
	}
}

// _reverse_states copies hs[0:seq_len] (forward order, BH each) into reverse
// step-major order: block p = hs[seq_len-1-p]
_reverse_states :: proc(dst, hs: []f64, seq_len, BH: int) {
	for p in 0 ..< seq_len {
		s := seq_len - 1 - p
		copy(dst[p * BH:(p + 1) * BH], hs[s * BH:(s + 1) * BH])
	}
}

// _hidden_backprop: dh_prev = d_hh [batch, G] @ W_hh^T (overwrites dh_prev)
_hidden_backprop :: proc(d_hh, w_hh, dh_prev: []f64, batch, H, G: int) {
	mem.zero_slice(dh_prev)
	d_mat := _mat(batch, G, d_hh)
	w_mat := _mat(H, G, w_hh)
	dh_mat := _mat(batch, H, dh_prev)
	l.matmul_nt_acc_simd(&d_mat, &w_mat, &dh_mat)
}

// _saved_blocks splits a node's saved forward data into `n_states` state rows
// of (seq_len + 1) * BH followed by `n_steps` per-step blocks of seq_len * BH
_saved_blocks :: proc(node: ^Tensor, seq_len, BH, n_states, n_steps: int) -> (states: [2][]f64, steps: [5][]f64) {
	saved := node.f64_metadata[:]
	S1, SB := (seq_len + 1) * BH, seq_len * BH
	fmt.assertf(
		len(saved) == n_states * S1 + n_steps * SB,
		"backward: %v node has %d saved values, expected %d",
		node.op,
		len(saved),
		n_states * S1 + n_steps * SB,
	)
	for k in 0 ..< n_states {states[k] = saved[k * S1:(k + 1) * S1]}
	base := n_states * S1
	for k in 0 ..< n_steps {steps[k] = saved[base + k * SB:base + (k + 1) * SB]}
	return
}

// _rnn_backward: h_t = tanh(x_t W_ih + h_{t-1} W_hh + b)
_rnn_backward :: proc(node: ^Tensor, allocator: mem.Allocator) {
	x_in, h_0_in, w_ih_in, w_hh_in, bias_in :=
		node.inputs[0], node.inputs[1], node.inputs[2], node.inputs[3], node.inputs[4]
	batch, seq_len, in_size := x_in.shape[0], x_in.shape[1], x_in.shape[2]
	H := w_ih_in.shape[1]
	BH := batch * H

	// hs[s] holds the state entering step s (hs[0] = h_0)
	states, _ := _saved_blocks(node, seq_len, BH, 1, 0)
	hs := states[0]

	xs := make([]f64, seq_len * batch * in_size, allocator)
	h_rev := make([]f64, seq_len * BH, allocator)
	d_all := make([]f64, seq_len * BH, allocator)
	dh_next := make([]f64, BH, allocator)
	dh_prev := make([]f64, BH, allocator)
	defer {
		delete(xs, allocator); delete(h_rev, allocator); delete(d_all, allocator)
		delete(dh_next, allocator); delete(dh_prev, allocator)
	}

	// 1. Backpropagate through time
	for s := seq_len - 1; s >= 0; s -= 1 {
		p := seq_len - 1 - s
		h_cur := hs[(s + 1) * BH:(s + 2) * BH]
		d_pre := d_all[p * BH:(p + 1) * BH]
		for b in 0 ..< batch {
			off := (b * seq_len + s) * H
			for i in 0 ..< H {
				idx := b * H + i
				dh := node.grad.data[off + i] + dh_next[idx]
				d_pre[idx] = dh * (1.0 - h_cur[idx] * h_cur[idx])
			}
		}
		_hidden_backprop(d_pre, w_hh_in.data.data, dh_prev, batch, H, H)
		copy(dh_next, dh_prev)
	}

	// 2. Parameter and input gradients for all steps
	_gather_steps(xs, x_in.data.data, batch, seq_len, in_size, true)
	_reverse_states(h_rev, hs, seq_len, BH)
	_recurrent_grads(
		x_in,
		w_ih_in,
		w_hh_in,
		bias_in,
		{xs = xs, h_prev = h_rev, d_ih = d_all, d_hh = d_all},
		batch,
		seq_len,
		in_size,
		H,
		H,
		allocator,
	)

	if h_0_in.requires_grad {
		l.vec_add_simd(h_0_in.grad.data, dh_next, h_0_in.grad.data)
	}
}

// _gru_backward: gates [r, z, n]; bias is added to the input projection only.
//   r = σ(ih_r + hh_r), z = σ(ih_z + hh_z), n = tanh(ih_n + r * hh_n)
//   h = (1 - z) * n + z * h_prev
_gru_backward :: proc(node: ^Tensor, allocator: mem.Allocator) {
	x_in, h_0_in, w_ih_in, w_hh_in, bias_in :=
		node.inputs[0], node.inputs[1], node.inputs[2], node.inputs[3], node.inputs[4]
	batch, seq_len, in_size := x_in.shape[0], x_in.shape[1], x_in.shape[2]
	H := w_ih_in.shape[1] / 3
	H3 := 3 * H
	BH := batch * H
	BG := batch * H3

	states, steps := _saved_blocks(node, seq_len, BH, 1, 4)
	hs := states[0]
	rs, zs, ns, hh_n := steps[0], steps[1], steps[2], steps[3]

	xs := make([]f64, seq_len * batch * in_size, allocator)
	h_rev := make([]f64, seq_len * BH, allocator)
	d_ih := make([]f64, seq_len * BG, allocator)
	d_hh := make([]f64, seq_len * BG, allocator)
	dh_next := make([]f64, BH, allocator)
	dh_prev := make([]f64, BH, allocator)
	defer {
		delete(xs, allocator); delete(h_rev, allocator); delete(d_ih, allocator)
		delete(d_hh, allocator); delete(dh_next, allocator); delete(dh_prev, allocator)
	}

	// 1. Backpropagate through time
	for s := seq_len - 1; s >= 0; s -= 1 {
		p := seq_len - 1 - s
		h_prev := hs[s * BH:(s + 1) * BH]
		r := rs[s * BH:(s + 1) * BH]
		z := zs[s * BH:(s + 1) * BH]
		n := ns[s * BH:(s + 1) * BH]
		hn := hh_n[s * BH:(s + 1) * BH]
		d_ih_p := d_ih[p * BG:(p + 1) * BG]
		d_hh_p := d_hh[p * BG:(p + 1) * BG]

		for b in 0 ..< batch {
			off := (b * seq_len + s) * H
			for i in 0 ..< H {
				idx := b * H + i
				dh := node.grad.data[off + i] + dh_next[idx]

				dz := dh * (h_prev[idx] - n[idx])
				dn_pre := dh * (1.0 - z[idx]) * (1.0 - n[idx] * n[idx])
				dr := dn_pre * hn[idx]

				dr_pre := dr * r[idx] * (1.0 - r[idx])
				dz_pre := dz * z[idx] * (1.0 - z[idx])

				d_ih_p[b * H3 + i] = dr_pre
				d_ih_p[b * H3 + H + i] = dz_pre
				d_ih_p[b * H3 + 2 * H + i] = dn_pre
				d_hh_p[b * H3 + i] = dr_pre
				d_hh_p[b * H3 + H + i] = dz_pre
				d_hh_p[b * H3 + 2 * H + i] = dn_pre * r[idx]

				// Direct path h_prev -> h through the update gate
				dh_next[idx] = dh * z[idx]
			}
		}

		_hidden_backprop(d_hh_p, w_hh_in.data.data, dh_prev, batch, H, H3)
		l.vec_add_simd(dh_next, dh_prev, dh_next)
	}

	// 2. Parameter and input gradients for all steps
	_gather_steps(xs, x_in.data.data, batch, seq_len, in_size, true)
	_reverse_states(h_rev, hs, seq_len, BH)
	_recurrent_grads(
		x_in,
		w_ih_in,
		w_hh_in,
		bias_in,
		{xs = xs, h_prev = h_rev, d_ih = d_ih, d_hh = d_hh},
		batch,
		seq_len,
		in_size,
		H,
		H3,
		allocator,
	)

	if h_0_in.requires_grad {
		l.vec_add_simd(h_0_in.grad.data, dh_next, h_0_in.grad.data)
	}
}

// _lstm_backward: gates [i, f, g, o]
//   c = f * c_prev + i * g,  h = o * tanh(c)
_lstm_backward :: proc(node: ^Tensor, allocator: mem.Allocator) {
	x_in, h_0_in, c_0_in := node.inputs[0], node.inputs[1], node.inputs[2]
	w_ih_in, w_hh_in, bias_in := node.inputs[3], node.inputs[4], node.inputs[5]
	batch, seq_len, in_size := x_in.shape[0], x_in.shape[1], x_in.shape[2]
	H := w_ih_in.shape[1] / 4
	H4 := 4 * H
	BH := batch * H
	BG := batch * H4

	states, steps := _saved_blocks(node, seq_len, BH, 2, 5)
	hs, cs := states[0], states[1]
	i_all, f_all, g_all, o_all, tc_all := steps[0], steps[1], steps[2], steps[3], steps[4]

	xs := make([]f64, seq_len * batch * in_size, allocator)
	h_rev := make([]f64, seq_len * BH, allocator)
	d_gate := make([]f64, seq_len * BG, allocator)
	dh_next := make([]f64, BH, allocator)
	dc_next := make([]f64, BH, allocator)
	dh_prev := make([]f64, BH, allocator)
	defer {
		delete(xs, allocator); delete(h_rev, allocator); delete(d_gate, allocator)
		delete(dh_next, allocator); delete(dc_next, allocator); delete(dh_prev, allocator)
	}

	// 1. Backpropagate through time
	for s := seq_len - 1; s >= 0; s -= 1 {
		p := seq_len - 1 - s
		c_prev := cs[s * BH:(s + 1) * BH]
		ig := i_all[s * BH:(s + 1) * BH]
		fg := f_all[s * BH:(s + 1) * BH]
		gg := g_all[s * BH:(s + 1) * BH]
		og := o_all[s * BH:(s + 1) * BH]
		tcs := tc_all[s * BH:(s + 1) * BH]
		dg := d_gate[p * BG:(p + 1) * BG]

		for b in 0 ..< batch {
			off := (b * seq_len + s) * H
			for i in 0 ..< H {
				idx := b * H + i
				dh := node.grad.data[off + i] + dh_next[idx]
				tc := tcs[idx]

				d_o := dh * tc
				dc := dh * og[idx] * (1.0 - tc * tc) + dc_next[idx]

				dg[b * H4 + i] = dc * gg[idx] * ig[idx] * (1.0 - ig[idx])
				dg[b * H4 + H + i] = dc * c_prev[idx] * fg[idx] * (1.0 - fg[idx])
				dg[b * H4 + 2 * H + i] = dc * ig[idx] * (1.0 - gg[idx] * gg[idx])
				dg[b * H4 + 3 * H + i] = d_o * og[idx] * (1.0 - og[idx])

				dc_next[idx] = dc * fg[idx]
			}
		}

		_hidden_backprop(dg, w_hh_in.data.data, dh_prev, batch, H, H4)
		copy(dh_next, dh_prev)
	}

	// 2. Parameter and input gradients for all steps
	_gather_steps(xs, x_in.data.data, batch, seq_len, in_size, true)
	_reverse_states(h_rev, hs, seq_len, BH)
	_recurrent_grads(
		x_in,
		w_ih_in,
		w_hh_in,
		bias_in,
		{xs = xs, h_prev = h_rev, d_ih = d_gate, d_hh = d_gate},
		batch,
		seq_len,
		in_size,
		H,
		H4,
		allocator,
	)

	if h_0_in.requires_grad {
		l.vec_add_simd(h_0_in.grad.data, dh_next, h_0_in.grad.data)
	}
	if c_0_in.requires_grad {
		l.vec_add_simd(c_0_in.grad.data, dc_next, c_0_in.grad.data)
	}
}

// _Ssm_Bwd_Job: shared buffers of the SSM backward; every batch row only
// touches its own slices, so rows can run on different threads.
_Ssm_Bwd_Job :: struct {
	x, h_0, A, B, C, Delta, D, dy: []f64,
	// [batch, seq_len, d_model * d_state]: abar = exp(delta * A) of each step,
	// saved by the forward pass. hs receives the state entering each step and
	// is overwritten with that step's dA term once it is consumed.
	abar, hs:                      []f64,
	dh:                            []f64, // [batch, d_model * d_state], ends as dL/dh_0
	dx, dB, dC, dDelta:            []f64, // zeroed, same layout as the inputs
	seq_len, d_model, d_state:     int,
}

// _ssm_backward_rows recomputes the states of batch rows [b0, b1) from the
// saved exp(delta * A) and backpropagates through them. dA terms are left in
// hs for an ordered reduction.
_ssm_backward_rows :: proc(data: rawptr, b0, b1: int) #no_bounds_check {
	j := (^_Ssm_Bwd_Job)(data)
	S, Dm, N := j.seq_len, j.d_model, j.d_state
	DN := Dm * N
	// Local copies: stores through the slices cannot alias the job struct
	X, A, Bm, C, Dl, D, dy := j.x, j.A, j.B, j.C, j.Delta, j.D, j.dy
	hs, abar := j.hs, j.abar
	dx, dB, dC, dDelta := j.dx, j.dB, j.dC, j.dDelta
	for b in b0 ..< b1 {
		// 1. States entering every step (same expression as the forward scan)
		if S > 0 {copy(hs[b * S * DN:(b * S + 1) * DN], j.h_0[b * DN:(b + 1) * DN])}
		for s in 0 ..< S - 1 {
			row_x := (b * S + s) * Dm
			row_bc := (b * S + s) * N
			st := (b * S + s) * DN
			for d in 0 ..< Dm {
				x_val := X[row_x + d]
				delta := Dl[row_x + d]
				for n in 0 ..< N {
					B_bar := delta * Bm[row_bc + n]
					hs[st + DN + d * N + n] = abar[st + d * N + n] * hs[st + d * N + n] + B_bar * x_val
				}
			}
		}

		// 2. Backpropagate through time
		dh := j.dh[b * DN:(b + 1) * DN]
		for s := S - 1; s >= 0; s -= 1 {
			row_x := (b * S + s) * Dm
			row_bc := (b * S + s) * N
			st := (b * S + s) * DN
			for d in 0 ..< Dm {
				x_val := X[row_x + d]
				delta := Dl[row_x + d]
				dy_val := dy[row_x + d]

				// dx / dDelta accumulate in registers, in the same order of terms
				dx_acc := dx[row_x + d] + dy_val * D[d]
				dDelta_acc := dDelta[row_x + d]

				for n in 0 ..< N {
					A_val := A[d * N + n]
					A_bar := abar[st + d * N + n]
					B_val := Bm[row_bc + n]
					B_bar := delta * B_val

					h_prev_val := hs[st + d * N + n]
					dh_n := dh[d * N + n] + dy_val * C[row_bc + n]

					h_n := A_bar * h_prev_val + B_bar * x_val
					dC[row_bc + n] += dy_val * h_n
					dh[d * N + n] = dh_n * A_bar
					hs[st + d * N + n] = dh_n * h_prev_val * delta * A_bar // dA term
					dB[row_bc + n] += dh_n * x_val * delta
					dx_acc += dh_n * B_bar
					dDelta_acc += dh_n * (h_prev_val * A_val * A_bar + x_val * B_val)
				}
				dx[row_x + d] = dx_acc
				dDelta[row_x + d] = dDelta_acc
			}
		}
	}
}

// _Ssm_Reduce_Job sums the dA terms of every (step, batch row) per element,
// in the order a sequential backward pass adds them: steps last to first,
// batch rows in order.
_Ssm_Reduce_Job :: struct {
	terms, dA:              []f64,
	batch, seq_len, DN:     int,
}

_ssm_reduce_dA :: proc(data: rawptr, e0, e1: int) #no_bounds_check {
	j := (^_Ssm_Reduce_Job)(data)
	terms, dA := j.terms, j.dA[e0:e1]
	// Sweep contiguous rows of terms; each element still adds its terms in
	// (step descending, batch row ascending) order
	for s := j.seq_len - 1; s >= 0; s -= 1 {
		for b in 0 ..< j.batch {
			row := terms[(b * j.seq_len + s) * j.DN + e0:][:e1 - e0]
			for e in 0 ..< len(dA) {dA[e] += row[e]}
		}
	}
}

// _ssm_backward: h_n = exp(delta * A) * h_prev + (delta * B) * x,
// y = x * D + sum_n C * h_n. Gradients for all 7 inputs.
_ssm_backward :: proc(node: ^Tensor, allocator: mem.Allocator) {
	x_in := node.inputs[0]
	h_0_in := node.inputs[1]
	A_in := node.inputs[2]
	B_in := node.inputs[3]
	C_in := node.inputs[4]
	Delta_in := node.inputs[5]
	D_in := node.inputs[6]

	if len(node.grad.data) == 0 {return}
	batch := node.shape[0]
	seq_len := node.shape[1]
	d_model := node.shape[2]
	d_state := A_in.shape[1]
	DN := d_model * d_state

	dx := make([]f64, len(x_in.data.data), allocator)
	dA := make([]f64, len(A_in.data.data), allocator)
	dB := make([]f64, len(B_in.data.data), allocator)
	dC := make([]f64, len(C_in.data.data), allocator)
	dDelta := make([]f64, len(Delta_in.data.data), allocator)
	dD := make([]f64, len(D_in.data.data), allocator)
	dh := make([]f64, batch * DN, allocator)
	// States, then dA terms: every element is written, so no zeroing
	hs_buf := make([dynamic]f64, 0, allocator)
	non_zero_resize(&hs_buf, batch * seq_len * DN)
	defer {
		delete(dx, allocator); delete(dA, allocator); delete(dB, allocator)
		delete(dC, allocator); delete(dDelta, allocator); delete(dD, allocator)
		delete(dh, allocator); delete(hs_buf)
	}

	// exp(delta * A) saved by tensor_ssm
	fmt.assertf(
		len(node.f64_metadata) == batch * seq_len * DN,
		"backward: SSM node has %d saved values, expected %d",
		len(node.f64_metadata),
		batch * seq_len * DN,
	)

	job := _Ssm_Bwd_Job {
		x       = x_in.data.data,
		h_0     = h_0_in.data.data,
		A       = A_in.data.data,
		B       = B_in.data.data,
		C       = C_in.data.data,
		Delta   = Delta_in.data.data,
		D       = D_in.data.data,
		dy      = node.grad.data,
		abar    = node.f64_metadata[:],
		hs      = hs_buf[:],
		dh      = dh,
		dx      = dx,
		dB      = dB,
		dC      = dC,
		dDelta  = dDelta,
		seq_len = seq_len,
		d_model = d_model,
		d_state = d_state,
	}
	util.parallel_for(batch, &job, _ssm_backward_rows, _ssm_threads(batch, seq_len * DN))

	red := _Ssm_Reduce_Job{hs_buf[:], dA, batch, seq_len, DN}
	util.parallel_for(DN, &red, _ssm_reduce_dA, _ssm_threads(DN, batch * seq_len))

	// dD[d] = sum of dy * x, in the same (step, batch row) order
	for s := seq_len - 1; s >= 0; s -= 1 {
		for b in 0 ..< batch {
			row := (b * seq_len + s) * d_model
			for d in 0 ..< d_model {
				dD[d] += node.grad.data[row + d] * x_in.data.data[row + d]
			}
		}
	}

	if x_in.requires_grad {l.vec_add_simd(x_in.grad.data, dx, x_in.grad.data)}
	if h_0_in.requires_grad {l.vec_add_simd(h_0_in.grad.data, dh, h_0_in.grad.data)}
	if A_in.requires_grad {l.vec_add_simd(A_in.grad.data, dA, A_in.grad.data)}
	if B_in.requires_grad {l.vec_add_simd(B_in.grad.data, dB, B_in.grad.data)}
	if C_in.requires_grad {l.vec_add_simd(C_in.grad.data, dC, C_in.grad.data)}
	if Delta_in.requires_grad {l.vec_add_simd(Delta_in.grad.data, dDelta, Delta_in.grad.data)}
	if D_in.requires_grad {l.vec_add_simd(D_in.grad.data, dD, D_in.grad.data)}
}

// _conv2d_backward: the forward is out_n = W[C_out, C_in*kH*kW] @ im2col(x_n) + b
_conv2d_backward :: proc(node: ^Tensor, allocator: mem.Allocator) {
	input_in := node.inputs[0]
	weight_in := node.inputs[1]
	bias_in: ^Tensor = nil
	if len(node.inputs) > 2 {bias_in = node.inputs[2]}

	N, C_in, H, W := input_in.shape[0], input_in.shape[1], input_in.shape[2], input_in.shape[3]
	C_out := weight_in.shape[0]
	kH, kW := node.conv_params.kH, node.conv_params.kW
	stride, pad := node.conv_params.stride, node.conv_params.pad
	out_h, out_w := node.shape[2], node.shape[3]
	col_h := out_h * out_w
	col_w := C_in * kH * kW

	// View the weight by its logical shape, exactly like the forward pass
	w_view := _mat(C_out, col_w, weight_in.data.data)

	// dx = col2im(W^T @ dY_n)
	if input_in.requires_grad {
		grad_col := make([]f64, N * col_w * col_h, allocator)
		defer delete(grad_col, allocator)
		for n in 0 ..< N {
			dy_n := _mat(C_out, col_h, node.grad.data[n * C_out * col_h:(n + 1) * C_out * col_h])
			dcol_n := _mat(col_w, col_h, grad_col[n * col_w * col_h:(n + 1) * col_w * col_h])
			l.matmul_tn_acc_simd(&w_view, &dy_n, &dcol_n)
		}
		grad_input := _col2im(
			grad_col,
			N,
			C_in,
			H,
			W,
			kH,
			kW,
			stride,
			pad,
			out_h,
			out_w,
			allocator,
		)
		defer delete(grad_input, allocator)
		l.vec_add_simd(input_in.grad.data, grad_input, input_in.grad.data)
	}

	// dW += dY_n @ im2col(x_n)^T
	if weight_in.requires_grad {
		col, _, _ := _im2col(input_in.data.data, N, C_in, H, W, kH, kW, stride, pad, allocator)
		defer delete(col, allocator)
		dw := _grad_view(weight_in, C_out, col_w)
		for n in 0 ..< N {
			dy_n := _mat(C_out, col_h, node.grad.data[n * C_out * col_h:(n + 1) * C_out * col_h])
			col_n := _mat(col_w, col_h, col[n * col_w * col_h:(n + 1) * col_w * col_h])
			l.matmul_nt_acc_simd(&dy_n, &col_n, &dw)
		}
	}

	// db += sum of dY over batch and spatial positions
	if bias_in != nil && bias_in.requires_grad {
		for n in 0 ..< N {
			for c in 0 ..< C_out {
				off := (n * C_out + c) * col_h
				bias_in.grad.data[c] += l.sum_simd(node.grad.data[off:off + col_h])
			}
		}
	}
}

// _assert_same_len checks that an element-wise op's output, input and input
// gradient all have the same number of elements.
_assert_same_len :: proc(node, input: ^Tensor, loc := #caller_location) {
	fmt.assertf(
		len(node.grad.data) == len(input.data.data) &&
		len(input.grad.data) == len(input.data.data),
		"%v backward: output grad %d, input %d, input grad %d elements",
		node.op,
		len(node.grad.data),
		len(input.data.data),
		len(input.grad.data),
		loc = loc,
	)
}
