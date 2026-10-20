package tensor


import l "../linalg"
import "core:fmt"
import "core:math"
import "core:mem"


tensor_backward :: proc(root: ^Tensor, allocator: mem.Allocator = context.allocator) {
	if !root.requires_grad {return}

	// 1. Set the gradient of the root node to 1.0
	tensor_ensure_grad(root)
	has_custom_grad := false
	for i in 0 ..< len(root.grad.data) {
		if root.grad.data[i] != 0.0 {
			has_custom_grad = true
			break
		}
	}

	// Only set to 1.0 if user hasn't set a custom gradient
	if !has_custom_grad {
		for i in 0 ..< len(root.grad.data) {
			root.grad.data[i] = 1.0
		}
	}

	// 2. Build the topological sort
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

		// ✅ CRITICAL: Skip if gradient is still empty (shouldn't happen, but defensive)
		if len(node.grad.data) == 0 {
			fmt.printf("WARNING: Skipping node with op %v - empty gradient\n", node.op)
			continue
		}
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
				if len(node.grad.data) == 0 {
					fmt.println("ERROR: Sum grad.data is empty")
					continue
				}
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
				// ✅ FIX: Add defensive length checks
				grad_out_len := len(node.grad.data)
				input_len := len(a_in.data.data)
				grad_in_len := len(a_in.grad.data)

				if grad_out_len != input_len || grad_out_len != grad_in_len {
					fmt.printf(
						"WARNING: ReLU backward length mismatch at epoch - " +
						"grad_out=%d, input=%d, grad_in=%d\n",
						grad_out_len,
						input_len,
						grad_in_len,
					)
					continue
				}

				l.vec_relu_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data)
			}
		case .Gelu:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)

				// Constants for Hastings approximation of CDF and PDF
				inv_sqrt_2pi := 0.3989422804014327
				p := 0.2316419
				b1 := 0.319381530
				b2 := -0.356563782
				b3 := 1.781477937
				b4 := -1.821255978
				b5 := 1.330274429

				for i in 0 ..< len(a_in.grad.data) {
					x := a_in.data.data[i]

					// 1. Compute PDF: φ(x) = (1/√(2π)) * e^(-x²/2)
					phi := math.exp(-0.5 * x * x) * inv_sqrt_2pi

					// 2. Compute CDF: Φ(x) using Hastings approximation
					ax := math.abs(x)
					t_val := 1.0 / (1.0 + p * ax)
					t2 := t_val * t_val
					t3 := t2 * t_val
					t4 := t3 * t_val
					t5 := t4 * t_val
					poly := b1 * t_val + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5
					cdf := 1.0 - phi * poly
					if x < 0.0 {
						cdf = 1.0 - cdf
					}

					// 3. GELU derivative: Φ(x) + x * φ(x)
					grad := cdf + x * phi

					a_in.grad.data[i] += node.grad.data[i] * grad
				}
			}

		case .Sigmoid:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				if len(node.grad.data) != len(a_in.data.data) ||
				   len(node.grad.data) != len(a_in.grad.data) {
					fmt.printf("WARNING: Sigmoid backward length mismatch\n")
					continue
				}
				l.vec_sigmoid_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data) // ✅ SIMD
			}

		case .Tanh:
			a_in := node.inputs[0]
			if a_in.requires_grad {
				if len(node.grad.data) != len(a_in.data.data) ||
				   len(node.grad.data) != len(a_in.grad.data) {
					fmt.printf("WARNING: Tanh backward length mismatch\n")
					continue
				}
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

			// ✅ CRITICAL: Check gradient dimensions
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: AddBias node has empty gradient, skipping")
				continue
			}
			if len(node.int_metadata) == 2 {
				N := node.int_metadata[0]
				D := node.int_metadata[1]

				if a_in.requires_grad {
					tensor_ensure_grad(a_in)
					l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
				}

				if bias_in.requires_grad {
					tensor_ensure_grad(bias_in)
					// Sum gradients over the N rows
					if len(bias_in.grad.data) >= D {
						bias_grad_slice := bias_in.grad.data[0:D]
						// Sum gradients over the N rows
						for i in 0 ..< N {
							row_grad := node.grad.data[i * D:(i + 1) * D]
							l.axpy_simd(1.0, row_grad, bias_grad_slice)
						}
					}
				}
				continue // Skip standard 2D backward
			}
			N := node.grad.rows
			D := node.grad.cols

			// ✅ CRITICAL: Verify gradient size matches expected dimensions
			if len(node.grad.data) != N * D {
				fmt.printf(
					"WARNING: AddBias gradient size mismatch: %d != %d * %d\n",
					len(node.grad.data),
					N,
					D,
				)
				continue
			}

			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(a_in.grad.data) == len(node.grad.data) {
					l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
				}
			}

			if bias_in.requires_grad {
				tensor_ensure_grad(bias_in)
				if len(bias_in.grad.data) > 0 && len(bias_in.grad.data) == D {
					bias_grad_slice := bias_in.grad.data[0:D]
					for i in 0 ..< N {
						start_idx := i * D
						end_idx := (i + 1) * D

						// ✅ CRITICAL: Bounds check before slicing
						if end_idx > len(node.grad.data) {
							fmt.printf(
								"ERROR: AddBias slice out of bounds: %d:%d > %d\n",
								start_idx,
								end_idx,
								len(node.grad.data),
							)
							continue
						}

						row_grad := node.grad.data[start_idx:end_idx]
						l.axpy_simd(1.0, row_grad, bias_grad_slice)
					}
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
			x_in := node.inputs[0]
			h_0_in := node.inputs[1]
			A_in := node.inputs[2]
			B_in := node.inputs[3]
			C_in := node.inputs[4]
			Delta_in := node.inputs[5]
			D_in := node.inputs[6]

			if len(node.grad.data) == 0 {continue}
			batch := node.shape[0]
			seq_len := node.shape[1]
			d_model := node.shape[2]
			d_state := A_in.shape[1]

			dx := make([]f64, len(x_in.data.data), allocator)
			dh_0 := make([]f64, len(h_0_in.data.data), allocator)
			dA := make([]f64, len(A_in.data.data), allocator)
			dB := make([]f64, len(B_in.data.data), allocator)
			dC := make([]f64, len(C_in.data.data), allocator)
			dDelta := make([]f64, len(Delta_in.data.data), allocator)
			dD := make([]f64, len(D_in.data.data), allocator)

			h_t := make([]f64, batch * d_model * d_state, allocator)
			copy(h_t, h_0_in.data.data)
			h_prev := make([]f64, batch * d_model * d_state, allocator)
			h_next := make([]f64, batch * d_model * d_state, allocator)
			y_t := make([]f64, batch * d_model, allocator)

			x_s := make([]f64, batch * d_model, allocator)
			B_s := make([]f64, batch * d_state, allocator)
			C_s := make([]f64, batch * d_state, allocator)
			Delta_s := make([]f64, batch * d_model, allocator)

			dh_next := make([]f64, batch * d_model * d_state, allocator)
			dh_prev := make([]f64, batch * d_model * d_state, allocator)
			dy_t := make([]f64, batch * d_model, allocator)

			dx_s := make([]f64, batch * d_model, allocator)
			dB_s := make([]f64, batch * d_state, allocator)
			dC_s := make([]f64, batch * d_state, allocator)
			dDelta_s := make([]f64, batch * d_model, allocator)

			defer {
				delete(dx, allocator); delete(dh_0, allocator); delete(dA, allocator)
				delete(
					dB,
					allocator,
				); delete(dC, allocator); delete(dDelta, allocator); delete(dD, allocator)
				delete(h_t, allocator); delete(h_prev, allocator); delete(h_next, allocator)
				delete(y_t, allocator); delete(x_s, allocator); delete(B_s, allocator)
				delete(C_s, allocator); delete(Delta_s, allocator); delete(dh_next, allocator)
				delete(dy_t, allocator); delete(dx_s, allocator); delete(dB_s, allocator)
				delete(dC_s, allocator); delete(dDelta_s, allocator); delete(dh_prev, allocator)
			}

			h_states := make([]f64, seq_len * batch * d_model * d_state, allocator)
			defer delete(h_states, allocator)

			// 1. Recompute all forward states (Checkpointing)
			copy(h_t, h_0_in.data.data)
			for s in 0 ..< seq_len {
				copy(
					h_states[s * batch * d_model * d_state:(s + 1) * batch * d_model * d_state],
					h_t,
				)
				for b in 0 ..< batch {
					src_x := b * seq_len * d_model + s * d_model
					copy(x_s[b * d_model:(b + 1) * d_model], x_in.data.data[src_x:src_x + d_model])
					src_B := b * seq_len * d_state + s * d_state
					copy(B_s[b * d_state:(b + 1) * d_state], B_in.data.data[src_B:src_B + d_state])
					src_C := b * seq_len * d_state + s * d_state
					copy(C_s[b * d_state:(b + 1) * d_state], C_in.data.data[src_C:src_C + d_state])
					src_D := b * seq_len * d_model + s * d_model
					copy(
						Delta_s[b * d_model:(b + 1) * d_model],
						Delta_in.data.data[src_D:src_D + d_model],
					)
				}
				_ssm_step_forward(
					x_s,
					h_t,
					A_in.data.data,
					B_s,
					C_s,
					Delta_s,
					D_in.data.data,
					h_next,
					y_t,
					batch,
					d_model,
					d_state,
				)
				copy(h_t, h_next)
			}

			// 2. Backward pass
			for s := seq_len - 1; s >= 0; s -= 1 {
				copy(
					h_prev,
					h_states[s * batch * d_model * d_state:(s + 1) * batch * d_model * d_state],
				)

				for b in 0 ..< batch {
					src_x := b * seq_len * d_model + s * d_model
					copy(x_s[b * d_model:(b + 1) * d_model], x_in.data.data[src_x:src_x + d_model])
					src_B := b * seq_len * d_state + s * d_state
					copy(B_s[b * d_state:(b + 1) * d_state], B_in.data.data[src_B:src_B + d_state])
					src_C := b * seq_len * d_state + s * d_state
					copy(C_s[b * d_state:(b + 1) * d_state], C_in.data.data[src_C:src_C + d_state])
					src_D := b * seq_len * d_model + s * d_model
					copy(
						Delta_s[b * d_model:(b + 1) * d_model],
						Delta_in.data.data[src_D:src_D + d_model],
					)

					src_dy := b * seq_len * d_model + s * d_model
					copy(
						dy_t[b * d_model:(b + 1) * d_model],
						node.grad.data[src_dy:src_dy + d_model],
					)
				}

				for i in 0 ..< len(dx_s) {dx_s[i] = 0.0}
				for i in 0 ..< len(dB_s) {dB_s[i] = 0.0}
				for i in 0 ..< len(dC_s) {dC_s[i] = 0.0}
				for i in 0 ..< len(dDelta_s) {dDelta_s[i] = 0.0}

				for b in 0 ..< batch {
					for d in 0 ..< d_model {
						idx_x := b * d_model + d
						x_val := x_s[idx_x]
						delta := Delta_s[idx_x]
						dy_val := dy_t[idx_x]
						D_val := D_in.data.data[d]

						dx_s[idx_x] += dy_val * D_val
						dD[d] += dy_val * x_val

						for n in 0 ..< d_state {
							idx_h := (b * d_model + d) * d_state + n
							idx_A := d * d_state + n
							idx_BC := b * d_state + n

							A_val := A_in.data.data[idx_A]
							A_bar := math.exp(delta * A_val)
							B_bar := delta * B_s[idx_BC]

							h_prev_val := h_prev[idx_h]
							dh_n := dh_next[idx_h] + dy_val * C_s[idx_BC]

							h_n := A_bar * h_prev_val + B_bar * x_val
							dC_s[idx_BC] += dy_val * h_n
							dh_prev[idx_h] = dh_n * A_bar
							dA[idx_A] += dh_n * h_prev_val * delta * A_bar
							dB_s[idx_BC] += dh_n * x_val * delta
							dx_s[idx_x] += dh_n * B_bar
							dDelta_s[idx_x] +=
								dh_n * (h_prev_val * A_val * A_bar + x_val * B_s[idx_BC])
						}
					}
				}

				for b in 0 ..< batch {
					src_x := b * seq_len * d_model + s * d_model
					for i in 0 ..< d_model {dx[src_x + i] += dx_s[b * d_model + i]}
					src_B := b * seq_len * d_state + s * d_state
					for i in 0 ..< d_state {dB[src_B + i] += dB_s[b * d_state + i]}
					src_C := b * seq_len * d_state + s * d_state
					for i in 0 ..< d_state {dC[src_C + i] += dC_s[b * d_state + i]}
					src_D := b * seq_len * d_model + s * d_model
					for i in 0 ..< d_model {dDelta[src_D + i] += dDelta_s[b * d_model + i]}
				}
				copy(dh_next, dh_prev)
			}

			copy(dh_0, dh_prev)

			if x_in.requires_grad &&
			   len(x_in.grad.data) > 0 {l.vec_add_simd(x_in.grad.data, dx, x_in.grad.data)}
			if h_0_in.requires_grad &&
			   len(h_0_in.grad.data) > 0 {l.vec_add_simd(h_0_in.grad.data, dh_0, h_0_in.grad.data)}
			if A_in.requires_grad &&
			   len(A_in.grad.data) > 0 {l.vec_add_simd(A_in.grad.data, dA, A_in.grad.data)}
			if B_in.requires_grad &&
			   len(B_in.grad.data) > 0 {l.vec_add_simd(B_in.grad.data, dB, B_in.grad.data)}
			if C_in.requires_grad &&
			   len(C_in.grad.data) > 0 {l.vec_add_simd(C_in.grad.data, dC, C_in.grad.data)}
			if Delta_in.requires_grad &&
			   len(Delta_in.grad.data) >
				   0 {l.vec_add_simd(Delta_in.grad.data, dDelta, Delta_in.grad.data)}
			if D_in.requires_grad &&
			   len(D_in.grad.data) > 0 {l.vec_add_simd(D_in.grad.data, dD, D_in.grad.data)}
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
				// Temporary buffers
				grad_buf := make([]f64, T, allocator)
				norm_buf := make([]f64, T, allocator)
				mean_vec := make([]f64, T, allocator)
				defer {
					delete(grad_buf, allocator)
					delete(norm_buf, allocator)
					delete(mean_vec, allocator)
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

							for i: int = 0; i < T; i += 1 {mean_vec[i] = mean_grad}

							temp := make([]f64, T, allocator)
							l.vec_sub_simd(grad_buf, mean_vec, temp)

							scalar := dot_grad_y / T_f
							for t: int = 0; t < T; t += 1 {
								temp[t] -= norm_buf[t] * scalar
							}

							// Recompute std of input for this slice to divide the gradient
							in_buf := make([]f64, T, allocator)
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								in_buf[t] = input_in.data.data[idx]
							}
							in_mean := l.sum_simd(in_buf) / T_f
							for i: int = 0; i < T; i += 1 {mean_vec[i] = in_mean}

							in_centered := make([]f64, T, allocator)
							l.vec_sub_simd(in_buf, mean_vec, in_centered)
							in_var := l.dot_simd(in_centered, in_centered) / T_f
							in_std := math.sqrt(in_var + eps)
							inv_std := 1.0 / in_std

							for t: int = 0; t < T; t += 1 {temp[t] *= inv_std}

							// Accumulate to input gradient
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								input_in.grad.data[idx] += temp[t]
							}

							delete(in_buf, allocator)
							delete(in_centered, allocator)
							delete(temp, allocator)
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
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: CrossEntropy node has empty gradient, skipping")
				continue
			}
			scalar_grad := node.grad.data[0] // Usually 1.0

			// ✅ FIX: Use stored dimensions
			if len(node.int_metadata) < 2 {
				fmt.println("ERROR: CrossEntropy missing dimension metadata")
				continue
			}
			N := node.int_metadata[0]
			C := node.int_metadata[1]

			if len(node.int_metadata) < 2 + N {
				fmt.printf(
					"ERROR: CrossEntropy int_metadata length %d < %d\n",
					len(node.int_metadata),
					2 + N,
				)
				continue
			}

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
				// ✅ ADD: Bounds check
				if len(node.int_metadata) != len(node.grad.data) {
					fmt.printf(
						"ERROR: MaxPool2d int_metadata length %d != grad length %d\n",
						len(node.int_metadata),
						len(node.grad.data),
					)
					continue
				}

				for i in 0 ..< len(node.grad.data) {
					idx := node.int_metadata[i]
					// ✅ ADD: Bounds check for idx
					if idx >= len(a_in.grad.data) {
						fmt.printf(
							"ERROR: MaxPool2d idx %d >= grad length %d\n",
							idx,
							len(a_in.grad.data),
						)
						continue
					}
					a_in.grad.data[idx] += node.grad.data[i]
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

			// ✅ Temporary contiguous buffers for SIMD operations
			x_buf := make([]f64, channel_size, allocator)
			dout_buf := make([]f64, channel_size, allocator)
			defer delete(x_buf, allocator)
			defer delete(dout_buf, allocator)

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
				mean_vec := make([]f64, channel_size, allocator)
				for i in 0 ..< channel_size {mean_vec[i] = mean}
				centered := make([]f64, channel_size, allocator)
				l.vec_sub_simd(x_buf, mean_vec, centered)
				delete(mean_vec, allocator)

				var := l.dot_simd(centered, centered) / N_hw
				std := math.sqrt(var + eps)
				inv_std := 1.0 / std
				gamma := weight_in.data.data[c]

				// 4. Compute x_hat (SIMD mul)
				x_hat := make([]f64, channel_size, allocator)
				inv_std_vec := make([]f64, channel_size, allocator)
				for i in 0 ..< channel_size {inv_std_vec[i] = inv_std}
				l.vec_mul_simd(centered, inv_std_vec, x_hat)
				delete(inv_std_vec, allocator)

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

				// 7. Compute input gradient (SIMD)
				if input_in.requires_grad && len(input_in.grad.data) > 0 {
					// dout_gamma = dout * gamma (SIMD)
					dout_gamma := make([]f64, channel_size, allocator)
					gamma_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {gamma_vec[i] = gamma}
					l.vec_mul_simd(dout_buf, gamma_vec, dout_gamma)
					delete(gamma_vec, allocator)

					// Precompute sums (SIMD)
					sum_dout_gamma := l.sum_simd(dout_gamma)
					sum_dout_gamma_x_hat := l.dot_simd(dout_gamma, x_hat)

					// dx = (1 / (N_hw * std)) * (N_hw * dout_gamma - sum_dout_gamma - x_hat * sum_dout_gamma_x_hat)

					// term1 = N_hw * dout_gamma (SIMD)
					term1 := make([]f64, channel_size, allocator)
					n_hw_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {n_hw_vec[i] = N_hw}
					l.vec_mul_simd(dout_gamma, n_hw_vec, term1)
					delete(n_hw_vec, allocator)
					delete(dout_gamma, allocator)

					// term2 = x_hat * sum_dout_gamma_x_hat (SIMD)
					term2 := make([]f64, channel_size, allocator)
					s2_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {s2_vec[i] = sum_dout_gamma_x_hat}
					l.vec_mul_simd(x_hat, s2_vec, term2)
					delete(s2_vec, allocator)
					delete(x_hat, allocator)

					// combined = term1 - sum_dout_gamma (SIMD)
					sum_dg_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {sum_dg_vec[i] = sum_dout_gamma}
					l.vec_sub_simd(term1, sum_dg_vec, term1)
					delete(sum_dg_vec, allocator)

					// combined = combined - term2 (SIMD)
					l.vec_sub_simd(term1, term2, term1)
					delete(term2, allocator)

					// dx = combined / (N_hw * std) (SIMD)
					inv_denom := 1.0 / (N_hw * std)
					inv_denom_vec := make([]f64, channel_size, allocator)
					for i in 0 ..< channel_size {inv_denom_vec[i] = inv_denom}
					l.vec_mul_simd(term1, inv_denom_vec, term1)
					delete(inv_denom_vec, allocator)

					// Write dx back to input gradient (de-interleave)
					for n in 0 ..< N {
						for h in 0 ..< H {
							for w in 0 ..< W {
								idx := n * (C * H * W) + c * (H * W) + h * W + w
								buf_idx := n * (H * W) + h * W + w
								input_in.grad.data[idx] += term1[buf_idx]
							}
						}
					}
					delete(term1, allocator)
				} else {
					delete(x_hat, allocator)
				}

				delete(centered, allocator)
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
		case .None, .Constant, .TopKMask:
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

// _gather_step copies timestep s of a [batch, seq_len, width] sequence into dst [batch, width]
_gather_step :: proc(dst, src: []f64, batch, seq_len, width, s: int) {
	for b in 0 ..< batch {
		off := (b * seq_len + s) * width
		copy(dst[b * width:(b + 1) * width], src[off:off + width])
	}
}

// _scatter_add_step adds src [batch, width] into timestep s of a [batch, seq_len, width] sequence
_scatter_add_step :: proc(dst, src: []f64, batch, seq_len, width, s: int) {
	for b in 0 ..< batch {
		off := (b * seq_len + s) * width
		l.vec_add_simd(dst[off:off + width], src[b * width:(b + 1) * width], dst[off:off + width])
	}
}

// _recurrent_step_grads accumulates the gradients shared by all recurrent cells
// for one timestep, given the pre-activation gate gradients (G = gates * H):
//   dW_ih += x_t^T @ d_ih,   dW_hh += h_prev^T @ d_hh,   dbias += sum_b d_ih
//   dx[s] += d_ih @ W_ih^T,  dh_prev = d_hh @ W_hh^T  (dh_prev is overwritten)
_recurrent_step_grads :: proc(
	x_in, w_ih_in, w_hh_in, bias_in: ^Tensor,
	x_t, h_prev, d_ih, d_hh, dx_t, dh_prev: []f64,
	batch, seq_len, in_size, H, G, s: int,
) {
	x_mat := _mat(batch, in_size, x_t)
	h_prev_mat := _mat(batch, H, h_prev)
	d_ih_mat := _mat(batch, G, d_ih)
	d_hh_mat := _mat(batch, G, d_hh)
	w_ih_mat := _mat(in_size, G, w_ih_in.data.data)
	w_hh_mat := _mat(H, G, w_hh_in.data.data)

	if bias_in.requires_grad {
		db := bias_in.grad.data[0:G]
		for b in 0 ..< batch {
			l.vec_add_simd(db, d_ih[b * G:(b + 1) * G], db)
		}
	}
	if w_ih_in.requires_grad {
		dw := _grad_view(w_ih_in, in_size, G)
		l.matmul_tn_acc_simd(&x_mat, &d_ih_mat, &dw)
	}
	if w_hh_in.requires_grad {
		dw := _grad_view(w_hh_in, H, G)
		l.matmul_tn_acc_simd(&h_prev_mat, &d_hh_mat, &dw)
	}
	if x_in.requires_grad {
		mem.zero_slice(dx_t)
		dx_mat := _mat(batch, in_size, dx_t)
		l.matmul_nt_acc_simd(&d_ih_mat, &w_ih_mat, &dx_mat)
		_scatter_add_step(x_in.grad.data, dx_t, batch, seq_len, in_size, s)
	}
	mem.zero_slice(dh_prev)
	dh_prev_mat := _mat(batch, H, dh_prev)
	l.matmul_nt_acc_simd(&d_hh_mat, &w_hh_mat, &dh_prev_mat)
}

// _rnn_backward: h_t = tanh(x_t W_ih + h_{t-1} W_hh + b)
_rnn_backward :: proc(node: ^Tensor, allocator: mem.Allocator) {
	x_in, h_0_in, w_ih_in, w_hh_in, bias_in :=
		node.inputs[0], node.inputs[1], node.inputs[2], node.inputs[3], node.inputs[4]
	batch, seq_len, in_size := x_in.shape[0], x_in.shape[1], x_in.shape[2]
	H := w_ih_in.shape[1]
	BH := batch * H
	BI := batch * in_size

	// hs[s] holds the state entering step s (hs[0] = h_0)
	hs := make([]f64, (seq_len + 1) * BH, allocator)
	xs := make([]f64, seq_len * BI, allocator)
	d_pre := make([]f64, BH, allocator)
	dh_next := make([]f64, BH, allocator)
	dh_prev := make([]f64, BH, allocator)
	dx_t := make([]f64, BI, allocator)
	defer {
		delete(hs, allocator); delete(xs, allocator); delete(d_pre, allocator)
		delete(dh_next, allocator); delete(dh_prev, allocator); delete(dx_t, allocator)
	}

	// 1. Recompute and store all hidden states, in forward order
	copy(hs[0:BH], h_0_in.data.data)
	for s in 0 ..< seq_len {
		x_t := xs[s * BI:(s + 1) * BI]
		_gather_step(x_t, x_in.data.data, batch, seq_len, in_size, s)
		_rnn_step_forward(
			x_t,
			hs[s * BH:(s + 1) * BH],
			w_ih_in.data.data,
			w_hh_in.data.data,
			bias_in.data.data,
			hs[(s + 1) * BH:(s + 2) * BH],
			batch,
			in_size,
			H,
			allocator,
		)
	}

	// 2. Backpropagate through time
	for s := seq_len - 1; s >= 0; s -= 1 {
		h_cur := hs[(s + 1) * BH:(s + 2) * BH]
		for b in 0 ..< batch {
			off := (b * seq_len + s) * H
			for i in 0 ..< H {
				idx := b * H + i
				dh := node.grad.data[off + i] + dh_next[idx]
				d_pre[idx] = dh * (1.0 - h_cur[idx] * h_cur[idx])
			}
		}
		_recurrent_step_grads(
			x_in, w_ih_in, w_hh_in, bias_in,
			xs[s * BI:(s + 1) * BI], hs[s * BH:(s + 1) * BH],
			d_pre, d_pre, dx_t, dh_prev,
			batch, seq_len, in_size, H, H, s,
		)
		copy(dh_next, dh_prev)
	}

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
	BI := batch * in_size

	hs := make([]f64, (seq_len + 1) * BH, allocator)
	xs := make([]f64, seq_len * BI, allocator)
	rs := make([]f64, seq_len * BH, allocator)
	zs := make([]f64, seq_len * BH, allocator)
	ns := make([]f64, seq_len * BH, allocator)
	gate_hh := make([]f64, batch * H3, allocator)
	d_ih := make([]f64, batch * H3, allocator)
	d_hh := make([]f64, batch * H3, allocator)
	dh_next := make([]f64, BH, allocator)
	dh_prev := make([]f64, BH, allocator)
	dx_t := make([]f64, BI, allocator)
	defer {
		delete(hs, allocator); delete(xs, allocator); delete(rs, allocator)
		delete(zs, allocator); delete(ns, allocator); delete(gate_hh, allocator)
		delete(d_ih, allocator); delete(d_hh, allocator); delete(dh_next, allocator)
		delete(dh_prev, allocator); delete(dx_t, allocator)
	}

	// 1. Recompute and store states and gates, in forward order
	copy(hs[0:BH], h_0_in.data.data)
	for s in 0 ..< seq_len {
		x_t := xs[s * BI:(s + 1) * BI]
		_gather_step(x_t, x_in.data.data, batch, seq_len, in_size, s)
		_gru_step_forward(
			x_t,
			hs[s * BH:(s + 1) * BH],
			w_ih_in.data.data,
			w_hh_in.data.data,
			bias_in.data.data,
			hs[(s + 1) * BH:(s + 2) * BH],
			rs[s * BH:(s + 1) * BH],
			zs[s * BH:(s + 1) * BH],
			ns[s * BH:(s + 1) * BH],
			batch,
			in_size,
			H,
			allocator,
		)
	}

	w_hh_mat := _mat(H, H3, w_hh_in.data.data)

	// 2. Backpropagate through time
	for s := seq_len - 1; s >= 0; s -= 1 {
		h_prev := hs[s * BH:(s + 1) * BH]
		r := rs[s * BH:(s + 1) * BH]
		z := zs[s * BH:(s + 1) * BH]
		n := ns[s * BH:(s + 1) * BH]

		// hh_n = (h_prev @ W_hh)[:, 2H:3H] is needed for dr
		mem.zero_slice(gate_hh)
		h_prev_mat := _mat(batch, H, h_prev)
		gate_hh_mat := _mat(batch, H3, gate_hh)
		l.matmul_nn_acc_simd(&h_prev_mat, &w_hh_mat, &gate_hh_mat)

		for b in 0 ..< batch {
			off := (b * seq_len + s) * H
			for i in 0 ..< H {
				idx := b * H + i
				dh := node.grad.data[off + i] + dh_next[idx]

				dz := dh * (h_prev[idx] - n[idx])
				dn_pre := dh * (1.0 - z[idx]) * (1.0 - n[idx] * n[idx])
				dr := dn_pre * gate_hh[b * H3 + 2 * H + i]

				dr_pre := dr * r[idx] * (1.0 - r[idx])
				dz_pre := dz * z[idx] * (1.0 - z[idx])

				d_ih[b * H3 + i] = dr_pre
				d_ih[b * H3 + H + i] = dz_pre
				d_ih[b * H3 + 2 * H + i] = dn_pre
				d_hh[b * H3 + i] = dr_pre
				d_hh[b * H3 + H + i] = dz_pre
				d_hh[b * H3 + 2 * H + i] = dn_pre * r[idx]

				// Direct path h_prev -> h through the update gate
				dh_next[idx] = dh * z[idx]
			}
		}

		_recurrent_step_grads(
			x_in, w_ih_in, w_hh_in, bias_in,
			xs[s * BI:(s + 1) * BI], h_prev,
			d_ih, d_hh, dx_t, dh_prev,
			batch, seq_len, in_size, H, H3, s,
		)
		l.vec_add_simd(dh_next, dh_prev, dh_next)
	}

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
	BI := batch * in_size

	hs := make([]f64, (seq_len + 1) * BH, allocator)
	cs := make([]f64, (seq_len + 1) * BH, allocator)
	xs := make([]f64, seq_len * BI, allocator)
	i_all := make([]f64, seq_len * BH, allocator)
	f_all := make([]f64, seq_len * BH, allocator)
	g_all := make([]f64, seq_len * BH, allocator)
	o_all := make([]f64, seq_len * BH, allocator)
	d_gate := make([]f64, batch * H4, allocator)
	dh_next := make([]f64, BH, allocator)
	dc_next := make([]f64, BH, allocator)
	dh_prev := make([]f64, BH, allocator)
	dx_t := make([]f64, BI, allocator)
	defer {
		delete(hs, allocator); delete(cs, allocator); delete(xs, allocator)
		delete(i_all, allocator); delete(f_all, allocator); delete(g_all, allocator)
		delete(o_all, allocator); delete(d_gate, allocator); delete(dh_next, allocator)
		delete(dc_next, allocator); delete(dh_prev, allocator); delete(dx_t, allocator)
	}

	// 1. Recompute and store states and gates, in forward order
	copy(hs[0:BH], h_0_in.data.data)
	copy(cs[0:BH], c_0_in.data.data)
	for s in 0 ..< seq_len {
		x_t := xs[s * BI:(s + 1) * BI]
		_gather_step(x_t, x_in.data.data, batch, seq_len, in_size, s)
		_lstm_step_forward(
			x_t,
			hs[s * BH:(s + 1) * BH],
			cs[s * BH:(s + 1) * BH],
			w_ih_in.data.data,
			w_hh_in.data.data,
			bias_in.data.data,
			hs[(s + 1) * BH:(s + 2) * BH],
			cs[(s + 1) * BH:(s + 2) * BH],
			i_all[s * BH:(s + 1) * BH],
			f_all[s * BH:(s + 1) * BH],
			g_all[s * BH:(s + 1) * BH],
			o_all[s * BH:(s + 1) * BH],
			batch,
			in_size,
			H,
			allocator,
		)
	}

	// 2. Backpropagate through time
	for s := seq_len - 1; s >= 0; s -= 1 {
		c_prev := cs[s * BH:(s + 1) * BH]
		c_cur := cs[(s + 1) * BH:(s + 2) * BH]
		ig := i_all[s * BH:(s + 1) * BH]
		fg := f_all[s * BH:(s + 1) * BH]
		gg := g_all[s * BH:(s + 1) * BH]
		og := o_all[s * BH:(s + 1) * BH]

		for b in 0 ..< batch {
			off := (b * seq_len + s) * H
			for i in 0 ..< H {
				idx := b * H + i
				dh := node.grad.data[off + i] + dh_next[idx]
				tc := math.tanh(c_cur[idx])

				d_o := dh * tc
				dc := dh * og[idx] * (1.0 - tc * tc) + dc_next[idx]

				d_gate[b * H4 + i] = dc * gg[idx] * ig[idx] * (1.0 - ig[idx])
				d_gate[b * H4 + H + i] = dc * c_prev[idx] * fg[idx] * (1.0 - fg[idx])
				d_gate[b * H4 + 2 * H + i] = dc * ig[idx] * (1.0 - gg[idx] * gg[idx])
				d_gate[b * H4 + 3 * H + i] = d_o * og[idx] * (1.0 - og[idx])

				dc_next[idx] = dc * fg[idx]
			}
		}

		_recurrent_step_grads(
			x_in, w_ih_in, w_hh_in, bias_in,
			xs[s * BI:(s + 1) * BI], hs[s * BH:(s + 1) * BH],
			d_gate, d_gate, dx_t, dh_prev,
			batch, seq_len, in_size, H, H4, s,
		)
		copy(dh_next, dh_prev)
	}

	if h_0_in.requires_grad {
		l.vec_add_simd(h_0_in.grad.data, dh_next, h_0_in.grad.data)
	}
	if c_0_in.requires_grad {
		l.vec_add_simd(c_0_in.grad.data, dc_next, c_0_in.grad.data)
	}
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
		grad_input := _col2im(grad_col, N, C_in, H, W, kH, kW, stride, pad, out_h, out_w, allocator)
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
