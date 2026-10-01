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
				lo := f64(node.int_metadata[0]) / 1_000_000.0
				hi := f64(node.int_metadata[1]) / 1_000_000.0
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
				batch_size := node.shape[0]
				num_assets := node.shape[1]
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
			if len(node.int_metadata) == 3 {
				N := node.int_metadata[0]
				in_features := node.int_metadata[1]
				out_features := node.int_metadata[2]

				a_view := l.Matrix(f64) {
					rows = N,
					cols = in_features,
					data = a_in.data.data,
				}
				b_view := b_in.data
				grad_view := l.Matrix(f64) {
					rows = N,
					cols = out_features,
					data = node.grad.data,
				}

				if a_in.requires_grad {
					tensor_ensure_grad(a_in)
					b_t := _matrix_transpose(b_view, allocator)
					grad_a_view := l.matmul_dyn_simd(&grad_view, &b_t, allocator)
					l.matrix_free(&b_t)

					// Copy flattened gradient back
					copy(a_in.grad.data, grad_a_view.data)
					l.matrix_free(&grad_a_view)
				}

				if b_in.requires_grad {
					tensor_ensure_grad(b_in)
					a_t := _matrix_transpose(a_view, allocator)
					grad_b := l.matmul_dyn_simd(&a_t, &grad_view, allocator)
					l.matrix_free(&a_t)

					l.vec_add_simd(b_in.grad.data, grad_b.data, b_in.grad.data)
					l.matrix_free(&grad_b)
				}
				continue // Skip the standard 2D backward pass
			}
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(node.grad.data) > 0 {
					// ✅ FIX: Use allocator for ALL temporary matrices
					bt := _matrix_transpose(b_in.data, allocator)
					grad_a := l.matmul_dyn_simd(&node.grad, &bt, allocator)
					l.vec_add_simd(a_in.grad.data, grad_a.data, a_in.grad.data)
					l.matrix_free(&bt)
					l.matrix_free(&grad_a)
				}
			}

			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				if len(b_in.grad.data) > 0 && len(node.grad.data) > 0 {
					// ✅ FIX: Use allocator
					at := _matrix_transpose(a_in.data, allocator)
					grad_b := l.matmul_dyn_simd(&at, &node.grad, allocator)
					l.vec_add_simd(b_in.grad.data, grad_b.data, b_in.grad.data)
					l.matrix_free(&at)
					l.matrix_free(&grad_b)
				}
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

			// ∂KL/∂log_var = -0.5 * (1 - exp(log_var)) / n
			if log_var_in.requires_grad {
				tensor_ensure_grad(log_var_in)
				for i in 0 ..< len(log_var_in.grad.data) {
					log_var_in.grad.data[i] +=
						scalar_grad * (1.0 - math.exp(log_var_in.data.data[i])) / (2.0 * n)
				}
			}
		case .MaskedScaledDotProductAttention:
			Q_in := node.inputs[0]
			K_in := node.inputs[1]
			V_in := node.inputs[2]

			if len(node.grad.data) == 0 {continue}

			batch := Q_in.shape[0]
			seq_q := Q_in.shape[1]
			seq_k := K_in.shape[1]
			d_k := Q_in.shape[2]
			d_v := V_in.shape[2]
			scale := 1.0 / math.sqrt(f64(d_k))

			// Retrieve mask from metadata
			mask := make([]f64, seq_q * seq_k, allocator)
			neg_inf: f64 = -1e9
			for i in 0 ..< len(mask) {
				if node.int_metadata[i] == 1 {
					mask[i] = neg_inf
				} else {
					mask[i] = 0.0
				}
			}
			defer delete(mask, allocator)

			dQ := make([]f64, len(Q_in.data.data), allocator)
			dK := make([]f64, len(K_in.data.data), allocator)
			dV := make([]f64, len(V_in.data.data), allocator)
			defer {
				delete(dQ, allocator)
				delete(dK, allocator)
				delete(dV, allocator)
			}

			for b in 0 ..< batch {
				q_b := l.Matrix(f64) {
					rows = seq_q,
					cols = d_k,
					data = Q_in.data.data[b * seq_q * d_k:(b + 1) * seq_q * d_k],
				}
				k_b := l.Matrix(f64) {
					rows = seq_k,
					cols = d_k,
					data = K_in.data.data[b * seq_k * d_k:(b + 1) * seq_k * d_k],
				}
				v_b := l.Matrix(f64) {
					rows = seq_k,
					cols = d_v,
					data = V_in.data.data[b * seq_k * d_v:(b + 1) * seq_k * d_v],
				}
				dO_b := l.Matrix(f64) {
					rows = seq_q,
					cols = d_v,
					data = node.grad.data[b * seq_q * d_v:(b + 1) * seq_q * d_v],
				}

				// Recompute S_b and P_b with mask
				k_b_t := _matrix_transpose(k_b, allocator)
				s_b := l.matmul_dyn_simd(&q_b, &k_b_t, allocator)
				l.matrix_free(&k_b_t)

				p_b := l.Matrix(f64) {
					rows = seq_q,
					cols = seq_k,
					data = s_b.data,
				}
				for i in 0 ..< seq_q * seq_k {
					p_b.data[i] *= scale
				}

				// Apply mask
				for i in 0 ..< seq_q {
					for j in 0 ..< seq_k {
						if mask[i * seq_k + j] != 0.0 {
							p_b.data[i * seq_k + j] += mask[i * seq_k + j]
						}
					}
				}

				// Softmax
				for i in 0 ..< seq_q {
					row_start := i * seq_k
					max_val := p_b.data[row_start]
					for j in 1 ..< seq_k {
						if p_b.data[row_start + j] > max_val {max_val = p_b.data[row_start + j]}
					}
					sum_exp := 0.0
					for j in 0 ..< seq_k {
						p_b.data[row_start + j] = math.exp(p_b.data[row_start + j] - max_val)
						sum_exp += p_b.data[row_start + j]
					}
					inv_sum := 1.0 / sum_exp
					for j in 0 ..< seq_k {
						p_b.data[row_start + j] *= inv_sum
					}
				}

				// dV = P_b^T @ dO_b
				p_b_t := _matrix_transpose(p_b, allocator)
				dV_b := l.matmul_dyn_simd(&p_b_t, &dO_b, allocator)
				copy(dV[b * seq_k * d_v:(b + 1) * seq_k * d_v], dV_b.data)
				l.matrix_free(&p_b_t)
				l.matrix_free(&dV_b)

				// dP = dO_b @ V_b^T
				v_b_t := _matrix_transpose(v_b, allocator)
				dP_b := l.matmul_dyn_simd(&dO_b, &v_b_t, allocator)

				// ✅ FIX: Free v_b_t immediately after use
				l.matrix_free(&v_b_t)

				// dS = P_b * (dP_b - sum(dP_b * P_b, dim=-1))
				for i in 0 ..< seq_q {
					row_start := i * seq_k
					sum := 0.0
					for j in 0 ..< seq_k {
						sum += dP_b.data[row_start + j] * p_b.data[row_start + j]
					}
					for j in 0 ..< seq_k {
						dP_b.data[row_start + j] =
							p_b.data[row_start + j] * (dP_b.data[row_start + j] - sum)
					}
				}

				// dQ = (dP_b * scale) @ K_b
				for i in 0 ..< seq_q * seq_k {
					dP_b.data[i] *= scale
				}
				dQ_b := l.matmul_dyn_simd(&dP_b, &k_b, allocator)
				copy(dQ[b * seq_q * d_k:(b + 1) * seq_q * d_k], dQ_b.data)
				l.matrix_free(&dQ_b)

				// dK = (dP_b * scale)^T @ Q_b
				dP_b_t := _matrix_transpose(dP_b, allocator)
				dK_b := l.matmul_dyn_simd(&dP_b_t, &q_b, allocator)
				copy(dK[b * seq_k * d_k:(b + 1) * seq_k * d_k], dK_b.data)
				l.matrix_free(&dP_b_t)
				l.matrix_free(&dK_b)

				l.matrix_free(&dP_b)
				l.matrix_free(&s_b)
			}

			if Q_in.requires_grad &&
			   len(Q_in.grad.data) > 0 {l.vec_add_simd(Q_in.grad.data, dQ, Q_in.grad.data)}
			if K_in.requires_grad &&
			   len(K_in.grad.data) > 0 {l.vec_add_simd(K_in.grad.data, dK, K_in.grad.data)}
			if V_in.requires_grad &&
			   len(V_in.grad.data) > 0 {l.vec_add_simd(V_in.grad.data, dV, V_in.grad.data)}
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
					_ = a_in.grad.data[i] // just to reference (not used)
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
				alpha := 0.01 // or store in Tensor struct
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
				batch := node.shape[0]
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
			Q_in := node.inputs[0]
			K_in := node.inputs[1]
			V_in := node.inputs[2]

			if len(node.grad.data) == 0 {continue}

			batch := Q_in.shape[0]
			seq_q := Q_in.shape[1]
			seq_k := K_in.shape[1]
			d_k := Q_in.shape[2]
			d_v := V_in.shape[2]
			scale := 1.0 / math.sqrt(f64(d_k))

			dQ := make([]f64, len(Q_in.data.data), allocator)
			dK := make([]f64, len(K_in.data.data), allocator)
			dV := make([]f64, len(V_in.data.data), allocator)
			defer {
				delete(dQ, allocator)
				delete(dK, allocator)
				delete(dV, allocator)
			}

			for b in 0 ..< batch {
				q_b := l.Matrix(f64) {
					rows = seq_q,
					cols = d_k,
					data = Q_in.data.data[b * seq_q * d_k:(b + 1) * seq_q * d_k],
				}
				k_b := l.Matrix(f64) {
					rows = seq_k,
					cols = d_k,
					data = K_in.data.data[b * seq_k * d_k:(b + 1) * seq_k * d_k],
				}
				v_b := l.Matrix(f64) {
					rows = seq_k,
					cols = d_v,
					data = V_in.data.data[b * seq_k * d_v:(b + 1) * seq_k * d_v],
				}
				dO_b := l.Matrix(f64) {
					rows = seq_q,
					cols = d_v,
					data = node.grad.data[b * seq_q * d_v:(b + 1) * seq_q * d_v],
				}

				// 1. Recompute S_b and P_b (Checkpointing)
				k_b_t := _matrix_transpose(k_b, allocator)
				s_b := l.matmul_dyn_simd(&q_b, &k_b_t, allocator)
				l.matrix_free(&k_b_t) // ✅ Free immediately

				p_b := l.Matrix(f64) {
					rows = seq_q,
					cols = seq_k,
					data = s_b.data,
				}
				for i in 0 ..< seq_q * seq_k {
					p_b.data[i] *= scale
				}

				// Numerically stable softmax
				for i in 0 ..< seq_q {
					row_start := i * seq_k
					max_val := p_b.data[row_start]
					for j in 1 ..< seq_k {
						if p_b.data[row_start + j] > max_val {max_val = p_b.data[row_start + j]}
					}
					sum_exp := 0.0
					for j in 0 ..< seq_k {
						p_b.data[row_start + j] = math.exp(p_b.data[row_start + j] - max_val)
						sum_exp += p_b.data[row_start + j]
					}
					inv_sum := 1.0 / sum_exp
					for j in 0 ..< seq_k {
						p_b.data[row_start + j] *= inv_sum
					}
				}

				// 2. dV = P_b^T @ dO_b
				p_b_t := _matrix_transpose(p_b, allocator)
				dV_b := l.matmul_dyn_simd(&p_b_t, &dO_b, allocator)
				copy(dV[b * seq_k * d_v:(b + 1) * seq_k * d_v], dV_b.data)
				l.matrix_free(&p_b_t) // ✅ Free immediately
				l.matrix_free(&dV_b) // ✅ Free immediately

				// 3. dP = dO_b @ V_b^T
				v_b_t := _matrix_transpose(v_b, allocator)
				dP_b := l.matmul_dyn_simd(&dO_b, &v_b_t, allocator)
				l.matrix_free(&v_b_t) // ✅ Free immediately

				// 4. dS = P_b * (dP_b - sum(dP_b * P_b, dim=-1))
				for i in 0 ..< seq_q {
					row_start := i * seq_k
					sum := 0.0
					for j in 0 ..< seq_k {
						sum += dP_b.data[row_start + j] * p_b.data[row_start + j]
					}
					for j in 0 ..< seq_k {
						dP_b.data[row_start + j] =
							p_b.data[row_start + j] * (dP_b.data[row_start + j] - sum)
					}
				}

				// 5. dQ = (dP_b * scale) @ K_b
				for i in 0 ..< seq_q * seq_k {
					dP_b.data[i] *= scale
				}
				dQ_b := l.matmul_dyn_simd(&dP_b, &k_b, allocator)
				copy(dQ[b * seq_q * d_k:(b + 1) * seq_q * d_k], dQ_b.data)
				l.matrix_free(&dQ_b) // ✅ Free immediately

				// 6. dK = (dP_b * scale)^T @ Q_b
				dP_b_t := _matrix_transpose(dP_b, allocator)
				dK_b := l.matmul_dyn_simd(&dP_b_t, &q_b, allocator)
				copy(dK[b * seq_k * d_k:(b + 1) * seq_k * d_k], dK_b.data)
				l.matrix_free(&dP_b_t) // ✅ Free immediately
				l.matrix_free(&dK_b) // ✅ Free immediately

				l.matrix_free(&dP_b) // ✅ Free immediately
				l.matrix_free(&s_b) // ✅ Free immediately
			}

			if Q_in.requires_grad && len(Q_in.grad.data) > 0 {
				l.vec_add_simd(Q_in.grad.data, dQ, Q_in.grad.data)
			}
			if K_in.requires_grad && len(K_in.grad.data) > 0 {
				l.vec_add_simd(K_in.grad.data, dK, K_in.grad.data)
			}
			if V_in.requires_grad && len(V_in.grad.data) > 0 {
				l.vec_add_simd(V_in.grad.data, dV, V_in.grad.data)
			}
		case .MSELoss:
			// L = mean((pred - target)^2)
			// dL/dpred = 2 * (pred - target) / N
			pred_in := node.inputs[0]
			target_in := node.inputs[1]

			if pred_in.requires_grad {
				if len(node.grad.data) == 0 {
					fmt.println("WARNING: MSELoss node has empty gradient, skipping")
					continue
				}
				n := f64(len(pred_in.data.data))
				scalar_grad := node.grad.data[0] // Usually 1.0
				scale := 2.0 * scalar_grad / n

				// Re-calculate (pred - target) using temp allocator
				diff := make([]f64, len(pred_in.data.data), allocator)
				l.vec_sub_simd(pred_in.data.data, target_in.data.data, diff)

				// ✅ SIMD Optimization: grad_pred += scale * diff
				l.axpy_simd(scale, diff, pred_in.grad.data)
				delete(diff, allocator)
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
				eps := f64(node.int_metadata[4]) / 1_000_000_000.0

				T_f := f64(T)
				// Temporary buffers
				grad_buf := make([]f64, T, context.allocator)
				norm_buf := make([]f64, T, context.allocator)
				mean_vec := make([]f64, T, context.allocator)
				defer {
					delete(grad_buf, context.allocator)
					delete(norm_buf, context.allocator)
					delete(mean_vec, context.allocator)
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

							temp := make([]f64, T, context.allocator)
							l.vec_sub_simd(grad_buf, mean_vec, temp)

							scalar := dot_grad_y / T_f
							for t: int = 0; t < T; t += 1 {
								temp[t] -= norm_buf[t] * scalar
							}

							// Recompute std of input for this slice to divide the gradient
							in_buf := make([]f64, T, context.allocator)
							for t: int = 0; t < T; t += 1 {
								idx := n * (C * T * L) + c * (T * L) + t * L + el
								in_buf[t] = input_in.data.data[idx]
							}
							in_mean := l.sum_simd(in_buf) / T_f
							for i: int = 0; i < T; i += 1 {mean_vec[i] = in_mean}

							in_centered := make([]f64, T, context.allocator)
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

							delete(in_buf, context.allocator)
							delete(in_centered, context.allocator)
							delete(temp, context.allocator)
						}
					}
				}
			}
		case .GRU:
			x_in := node.inputs[0]
			h_0_in := node.inputs[1]
			w_ih_in := node.inputs[2]
			w_hh_in := node.inputs[3]
			bias_in := node.inputs[4]

			if len(node.grad.data) == 0 {continue}

			batch := x_in.shape[0]
			seq_len := x_in.shape[1]
			in_size := x_in.shape[2]
			hidden_size := w_ih_in.shape[1] / 3
			H := hidden_size
			H3 := 3 * H

			dx := make([]f64, len(x_in.data.data), allocator)
			dw_ih := make([]f64, len(w_ih_in.data.data), allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), allocator)
			dbias := make([]f64, len(bias_in.data.data), allocator)

			h_t := make([]f64, batch * H, allocator)
			copy(h_t, h_0_in.data.data)
			h_prev := make([]f64, batch * H, allocator)
			h_next := make([]f64, batch * H, allocator)
			x_t := make([]f64, batch * in_size, allocator)

			r_buf := make([]f64, batch * H, allocator)
			z_buf := make([]f64, batch * H, allocator)
			n_buf := make([]f64, batch * H, allocator)
			n_hh_buf := make([]f64, batch * H, allocator)

			dh_next := make([]f64, batch * H, allocator)
			dh_prev := make([]f64, batch * H, allocator)
			d_gate_ih := make([]f64, batch * H3, allocator)
			d_gate_hh := make([]f64, batch * H3, allocator)

			defer {
				delete(dx, allocator); delete(dw_ih, allocator)
				delete(dw_hh, allocator); delete(dbias, allocator)
				delete(h_t, allocator); delete(h_prev, allocator)
				delete(h_next, allocator); delete(x_t, allocator)
				delete(r_buf, allocator); delete(z_buf, allocator)
				delete(n_buf, allocator); delete(n_hh_buf, allocator)
				delete(dh_next, allocator); delete(dh_prev, allocator)
				delete(d_gate_ih, allocator); delete(d_gate_hh, allocator)
			}

			for s := seq_len - 1; s >= 0; s -= 1 {
				copy(h_prev, h_t)
				for b in 0 ..< batch {
					src := b * seq_len * in_size + s * in_size
					dst := b * in_size
					copy(x_t[dst:dst + in_size], x_in.data.data[src:src + in_size])
				}

				// Recompute forward
				_gru_step_forward(
					x_t,
					h_prev,
					w_ih_in.data.data,
					w_hh_in.data.data,
					bias_in.data.data,
					h_next,
					r_buf,
					z_buf,
					n_buf,
					batch,
					in_size,
					hidden_size,
					allocator,
				)

				// Recompute n_hh for backward
				h_prev_mat := l.Matrix(f64) {
					rows = batch,
					cols = H,
					data = h_prev,
				}
				w_hh_mat := l.Matrix(f64) {
					rows = H,
					cols = H3,
					data = w_hh_in.data.data,
				}
				gate_hh := l.matmul_dyn_simd(&h_prev_mat, &w_hh_mat, allocator)
				for b in 0 ..< batch {
					for i in 0 ..< H {
						n_hh_buf[b * H + i] = gate_hh.data[b * H3 + 2 * H + i]
					}
				}
				l.matrix_free(&gate_hh)

				copy(h_t, h_next)

				for b in 0 ..< batch {
					src := b * seq_len * H + s * H
					dst := b * H
					for i in 0 ..< H {
						dh_next[dst + i] = node.grad.data[src + i] + dh_prev[dst + i]
					}
				}

				// Backprop through h_t = (1 - z) * n + z * h_prev
				dz := make([]f64, batch * H, allocator)
				dn := make([]f64, batch * H, allocator)
				dr_from_n := make([]f64, batch * H, allocator)

				for i in 0 ..< batch * H {
					dh_prev[i] += dh_next[i] * z_buf[i]
					dz[i] = dh_next[i] * (h_prev[i] - n_buf[i])
					dn[i] = dh_next[i] * (1.0 - z_buf[i])
				}

				// Backprop through n = tanh(n_ih + r * n_hh)
				d_n_pre := make([]f64, batch * H, allocator)
				for i in 0 ..< batch * H {
					d_n_pre[i] = dn[i] * (1.0 - n_buf[i] * n_buf[i])
					dr_from_n[i] = d_n_pre[i] * n_hh_buf[i]
				}

				// Backprop through sigmoid gates
				dr_pre := make([]f64, batch * H, allocator)
				dz_pre := make([]f64, batch * H, allocator)
				for i in 0 ..< batch * H {
					dr_pre[i] = dr_from_n[i] * r_buf[i] * (1.0 - r_buf[i])
					dz_pre[i] = dz[i] * z_buf[i] * (1.0 - z_buf[i])
				}

				// Pack into d_gate matrices
				for b in 0 ..< batch {
					for i in 0 ..< H {
						idx_r := b * H3 + i
						idx_z := b * H3 + H + i
						idx_n := b * H3 + 2 * H + i

						d_gate_ih[idx_r] = dr_pre[b * H + i]
						d_gate_ih[idx_z] = dz_pre[b * H + i]
						d_gate_ih[idx_n] = d_n_pre[b * H + i]

						d_gate_hh[idx_r] = dr_pre[b * H + i]
						d_gate_hh[idx_z] = dz_pre[b * H + i]
						d_gate_hh[idx_n] = d_n_pre[b * H + i] * r_buf[b * H + i]
					}
				}

				delete(dz, allocator); delete(dn, allocator)
				delete(dr_from_n, allocator); delete(d_n_pre, allocator)
				delete(dr_pre, allocator); delete(dz_pre, allocator)

				for b in 0 ..< batch {
					for i in 0 ..< H3 {dbias[i] += d_gate_ih[b * H3 + i]}
				}

				// ✅ SIMD Matmul Backward
				x_mat := l.Matrix(f64) {
					rows = batch,
					cols = in_size,
					data = x_t,
				}
				h_prev_mat_bwd := l.Matrix(f64) {
					rows = batch,
					cols = H,
					data = h_prev,
				}
				d_gate_ih_mat := l.Matrix(f64) {
					rows = batch,
					cols = H3,
					data = d_gate_ih,
				}
				d_gate_hh_mat := l.Matrix(f64) {
					rows = batch,
					cols = H3,
					data = d_gate_hh,
				}

				x_t_t := _matrix_transpose(x_mat, allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &d_gate_ih_mat, allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t); l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat_bwd, allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &d_gate_hh_mat, allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t); l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, allocator)
				dx_t_res := l.matmul_dyn_simd(&d_gate_ih_mat, &w_ih_t, allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {dx[dst + i] += dx_t_res.data[src + i]}
				}
				l.matrix_free(&w_ih_t); l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, allocator)
				dh_prev_res := l.matmul_dyn_simd(&d_gate_hh_mat, &w_hh_t, allocator)
				for i in 0 ..< len(dh_prev) {dh_prev[i] = dh_prev_res.data[i]}
				l.matrix_free(&w_hh_t); l.matrix_free(&dh_prev_res)
			}

			if x_in.requires_grad &&
			   len(x_in.grad.data) > 0 {l.vec_add_simd(x_in.grad.data, dx, x_in.grad.data)}
			if h_0_in.requires_grad &&
			   len(h_0_in.grad.data) >
				   0 {l.vec_add_simd(h_0_in.grad.data, dh_prev, h_0_in.grad.data)}
			if w_ih_in.requires_grad &&
			   len(w_ih_in.grad.data) >
				   0 {l.vec_add_simd(w_ih_in.grad.data, dw_ih, w_ih_in.grad.data)}
			if w_hh_in.requires_grad &&
			   len(w_hh_in.grad.data) >
				   0 {l.vec_add_simd(w_hh_in.grad.data, dw_hh, w_hh_in.grad.data)}
			if bias_in.requires_grad &&
			   len(bias_in.grad.data) >
				   0 {l.vec_add_simd(bias_in.grad.data, dbias, bias_in.grad.data)}
		case .LSTM:
			x_in := node.inputs[0]
			h_0_in := node.inputs[1]
			c_0_in := node.inputs[2]
			w_ih_in := node.inputs[3]
			w_hh_in := node.inputs[4]
			bias_in := node.inputs[5]

			if len(node.grad.data) == 0 {continue}

			batch := x_in.shape[0]
			seq_len := x_in.shape[1]
			in_size := x_in.shape[2]
			hidden_size := w_ih_in.shape[1] / 4
			H := hidden_size
			H4 := 4 * H

			dx := make([]f64, len(x_in.data.data), allocator)
			dw_ih := make([]f64, len(w_ih_in.data.data), allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), allocator)
			dbias := make([]f64, len(bias_in.data.data), allocator)

			h_t := make([]f64, batch * H, allocator)
			c_t := make([]f64, batch * H, allocator)
			copy(h_t, h_0_in.data.data)
			copy(c_t, c_0_in.data.data)

			h_prev := make([]f64, batch * H, allocator)
			c_prev := make([]f64, batch * H, allocator)
			h_next := make([]f64, batch * H, allocator)
			c_next := make([]f64, batch * H, allocator)
			x_t := make([]f64, batch * in_size, allocator)

			i_buf := make([]f64, batch * H, allocator)
			f_buf := make([]f64, batch * H, allocator)
			g_buf := make([]f64, batch * H, allocator)
			o_buf := make([]f64, batch * H, allocator)

			dh_next := make([]f64, batch * H, allocator)
			dc_next := make([]f64, batch * H, allocator)
			dh_prev := make([]f64, batch * H, allocator)
			dc_prev := make([]f64, batch * H, allocator)
			d_gate_ih := make([]f64, batch * H4, allocator)
			d_gate_hh := make([]f64, batch * H4, allocator)

			defer {
				delete(dx, allocator); delete(dw_ih, allocator)
				delete(dw_hh, allocator); delete(dbias, allocator)
				delete(h_t, allocator); delete(c_t, allocator)
				delete(h_prev, allocator); delete(c_prev, allocator)
				delete(h_next, allocator); delete(c_next, allocator)
				delete(x_t, allocator)
				delete(i_buf, allocator); delete(f_buf, allocator)
				delete(g_buf, allocator); delete(o_buf, allocator)
				delete(dh_next, allocator); delete(dc_next, allocator)
				delete(dh_prev, allocator); delete(dc_prev, allocator)
				delete(d_gate_ih, allocator); delete(d_gate_hh, allocator)
			}

			for s := seq_len - 1; s >= 0; s -= 1 {
				copy(h_prev, h_t)
				copy(c_prev, c_t)

				for b in 0 ..< batch {
					src := b * seq_len * in_size + s * in_size
					dst := b * in_size
					copy(x_t[dst:dst + in_size], x_in.data.data[src:src + in_size])
				}

				// Recompute forward
				_lstm_step_forward(
					x_t,
					h_prev,
					c_prev,
					w_ih_in.data.data,
					w_hh_in.data.data,
					bias_in.data.data,
					h_next,
					c_next,
					i_buf,
					f_buf,
					g_buf,
					o_buf,
					batch,
					in_size,
					hidden_size,
					allocator,
				)
				copy(h_t, h_next)
				copy(c_t, c_next)

				// Accumulate gradients from output and future timestep
				for b in 0 ..< batch {
					src := b * seq_len * H + s * H
					dst := b * H
					for i in 0 ..< H {
						dh_next[dst + i] = node.grad.data[src + i] + dh_prev[dst + i]
					}
				}

				// Backprop through h_t = o * tanh(c_t)
				tanh_c := make([]f64, batch * H, allocator)
				dov := make([]f64, batch * H, allocator)
				dc := make([]f64, batch * H, allocator)

				for i in 0 ..< batch * H {
					tanh_c[i] = math.tanh(c_t[i])
					dov[i] = dh_next[i] * tanh_c[i]
					dc[i] = dh_next[i] * o_buf[i] * (1.0 - tanh_c[i] * tanh_c[i]) + dc_next[i]
				}

				// Backprop through c_t = f * c_prev + i * g
				df := make([]f64, batch * H, allocator)
				di := make([]f64, batch * H, allocator)
				dg := make([]f64, batch * H, allocator)

				for i in 0 ..< batch * H {
					df[i] = dc[i] * c_prev[i]
					di[i] = dc[i] * g_buf[i]
					dg[i] = dc[i] * i_buf[i]
					dc_prev[i] = dc[i] * f_buf[i]
				}

				// Backprop through activation functions
				di_pre := make([]f64, batch * H, allocator)
				df_pre := make([]f64, batch * H, allocator)
				dg_pre := make([]f64, batch * H, allocator)
				do_pre := make([]f64, batch * H, allocator)

				for i in 0 ..< batch * H {
					di_pre[i] = di[i] * i_buf[i] * (1.0 - i_buf[i])
					df_pre[i] = df[i] * f_buf[i] * (1.0 - f_buf[i])
					dg_pre[i] = dg[i] * (1.0 - g_buf[i] * g_buf[i])
					do_pre[i] = dov[i] * o_buf[i] * (1.0 - o_buf[i])
				}

				// Pack into d_gate matrices
				for b in 0 ..< batch {
					for i in 0 ..< H {
						idx_i := b * H4 + i
						idx_f := b * H4 + H + i
						idx_g := b * H4 + 2 * H + i
						idx_o := b * H4 + 3 * H + i

						d_gate_ih[idx_i] = di_pre[b * H + i]
						d_gate_ih[idx_f] = df_pre[b * H + i]
						d_gate_ih[idx_g] = dg_pre[b * H + i]
						d_gate_ih[idx_o] = do_pre[b * H + i]

						d_gate_hh[idx_i] = di_pre[b * H + i]
						d_gate_hh[idx_f] = df_pre[b * H + i]
						d_gate_hh[idx_g] = dg_pre[b * H + i]
						d_gate_hh[idx_o] = do_pre[b * H + i]
					}
				}

				delete(tanh_c, allocator)
				delete(dov, allocator)
				delete(dc, allocator)
				delete(df, allocator)
				delete(di, allocator); delete(dg, allocator)
				delete(di_pre, allocator); delete(df_pre, allocator)
				delete(dg_pre, allocator); delete(do_pre, allocator)

				// Accumulate bias gradients
				for b in 0 ..< batch {
					for i in 0 ..< H4 {dbias[i] += d_gate_ih[b * H4 + i]}
				}

				// ✅ SIMD Matmul Backward
				x_mat := l.Matrix(f64) {
					rows = batch,
					cols = in_size,
					data = x_t,
				}
				h_prev_mat_bwd := l.Matrix(f64) {
					rows = batch,
					cols = H,
					data = h_prev,
				}
				d_gate_ih_mat := l.Matrix(f64) {
					rows = batch,
					cols = H4,
					data = d_gate_ih,
				}
				d_gate_hh_mat := l.Matrix(f64) {
					rows = batch,
					cols = H4,
					data = d_gate_hh,
				}

				x_t_t := _matrix_transpose(x_mat, allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &d_gate_ih_mat, allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t); l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat_bwd, allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &d_gate_hh_mat, allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t); l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, allocator)
				dx_t_res := l.matmul_dyn_simd(&d_gate_ih_mat, &w_ih_t, allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {dx[dst + i] += dx_t_res.data[src + i]}
				}
				l.matrix_free(&w_ih_t); l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, allocator)
				dh_prev_res := l.matmul_dyn_simd(&d_gate_hh_mat, &w_hh_t, allocator)
				for i in 0 ..< len(dh_prev) {dh_prev[i] = dh_prev_res.data[i]}
				l.matrix_free(&w_hh_t); l.matrix_free(&dh_prev_res)

				copy(dc_next, dc_prev)
			}

			if x_in.requires_grad &&
			   len(x_in.grad.data) > 0 {l.vec_add_simd(x_in.grad.data, dx, x_in.grad.data)}
			if h_0_in.requires_grad &&
			   len(h_0_in.grad.data) >
				   0 {l.vec_add_simd(h_0_in.grad.data, dh_prev, h_0_in.grad.data)}
			if c_0_in.requires_grad &&
			   len(c_0_in.grad.data) >
				   0 {l.vec_add_simd(c_0_in.grad.data, dc_prev, c_0_in.grad.data)}
			if w_ih_in.requires_grad &&
			   len(w_ih_in.grad.data) >
				   0 {l.vec_add_simd(w_ih_in.grad.data, dw_ih, w_ih_in.grad.data)}
			if w_hh_in.requires_grad &&
			   len(w_hh_in.grad.data) >
				   0 {l.vec_add_simd(w_hh_in.grad.data, dw_hh, w_hh_in.grad.data)}
			if bias_in.requires_grad &&
			   len(bias_in.grad.data) >
				   0 {l.vec_add_simd(bias_in.grad.data, dbias, bias_in.grad.data)}
		case .Scale:
			a_in := node.inputs[0]
			scalar := f64(node.int_metadata[0]) / 1000000.0
			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				// a_in.grad += node.grad * scalar
				// Use fma: out += a * b where a=node.grad, b=scalar (broadcast)
				// Since we don't have a vec_fma_scalar, do it in two steps
				temp := make([]f64, len(node.grad.data), allocator)
				l.vec_scale_simd(node.grad.data, scalar, temp)
				l.vec_add_simd(a_in.grad.data, temp, a_in.grad.data)
				delete(temp, allocator)
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
				// Just reshape the gradient back to 4D
				copy(a_in.grad.data, node.grad.data)
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
				rf := f64(node.int_metadata[1]) / 1_000_000.0
				mean := f64(node.int_metadata[2]) / 1_000_000.0
				std := f64(node.int_metadata[3]) / 1_000_000.0

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
			x_in := node.inputs[0]
			h_0_in := node.inputs[1]
			w_ih_in := node.inputs[2]
			w_hh_in := node.inputs[3]
			bias_in := node.inputs[4]

			if len(node.grad.data) == 0 {continue}

			batch := x_in.shape[0]
			seq_len := x_in.shape[1]
			in_size := x_in.shape[2]
			hidden_size := w_ih_in.shape[1]

			dx := make([]f64, len(x_in.data.data), allocator)
			dh_0 := make([]f64, len(h_0_in.data.data), allocator) // ✅ FIX: typo corrected
			dw_ih := make([]f64, len(w_ih_in.data.data), allocator)
			dw_hh := make([]f64, len(w_hh_in.data.data), allocator)
			dbias := make([]f64, len(bias_in.data.data), allocator)

			h_t := make([]f64, batch * hidden_size, allocator)
			copy(h_t, h_0_in.data.data)
			h_prev := make([]f64, batch * hidden_size, allocator)
			h_next := make([]f64, batch * hidden_size, allocator)
			x_t := make([]f64, batch * in_size, allocator)

			dh_next := make([]f64, batch * hidden_size, allocator)
			dx_t := make([]f64, batch * in_size, allocator)
			dh_prev := make([]f64, batch * hidden_size, allocator)
			dw_ih_step := make([]f64, in_size * hidden_size, allocator)
			dw_hh_step := make([]f64, hidden_size * hidden_size, allocator)
			dbias_step := make([]f64, hidden_size, allocator)

			defer {
				delete(dx, allocator)
				delete(dh_0, allocator)
				delete(dw_ih, allocator)
				delete(dw_hh, allocator)
				delete(dbias, allocator)
				delete(h_t, allocator)
				delete(h_prev, allocator)
				delete(h_next, allocator)
				delete(x_t, allocator)
				delete(dh_next, allocator)
				delete(dx_t, allocator)
				delete(dh_prev, allocator)
				delete(dw_ih_step, allocator)
				delete(dw_hh_step, allocator)
				delete(dbias_step, allocator)
			}

			for s := seq_len - 1; s >= 0; s -= 1 {
				copy(h_prev, h_t)
				for b in 0 ..< batch {
					src := b * seq_len * in_size + s * in_size
					dst := b * in_size
					copy(x_t[dst:dst + in_size], x_in.data.data[src:src + in_size])
				}
				_rnn_step_forward(
					x_t,
					h_prev,
					w_ih_in.data.data,
					w_hh_in.data.data,
					bias_in.data.data,
					h_next,
					batch,
					in_size,
					hidden_size,
					allocator,
				)
				copy(h_t, h_next)

				for b in 0 ..< batch {
					src := b * seq_len * hidden_size + s * hidden_size
					dst := b * hidden_size
					for i in 0 ..< hidden_size {
						dh_next[dst + i] = node.grad.data[src + i] + dh_prev[dst + i]
					}
				}

				for i in 0 ..< batch * hidden_size {
					one_minus_h2 := 1.0 - h_next[i] * h_next[i]
					dh_next[i] *= one_minus_h2
				}

				for b in 0 ..< batch {
					for i in 0 ..< hidden_size {
						dbias_step[i] += dh_next[b * hidden_size + i]
					}
				}

				// ✅ FIX: Use '=' for struct initialization
				x_mat := l.Matrix(f64) {
					rows = batch,
					cols = in_size,
					data = x_t,
				}
				h_prev_mat := l.Matrix(f64) {
					rows = batch,
					cols = hidden_size,
					data = h_prev,
				}
				dh_mat := l.Matrix(f64) {
					rows = batch,
					cols = hidden_size,
					data = dh_next,
				}

				x_t_t := _matrix_transpose(x_mat, allocator)
				dw_ih_res := l.matmul_dyn_simd(&x_t_t, &dh_mat, allocator)
				for i in 0 ..< len(dw_ih) {dw_ih[i] += dw_ih_res.data[i]}
				l.matrix_free(&x_t_t)
				l.matrix_free(&dw_ih_res)

				h_prev_t := _matrix_transpose(h_prev_mat, allocator)
				dw_hh_res := l.matmul_dyn_simd(&h_prev_t, &dh_mat, allocator)
				for i in 0 ..< len(dw_hh) {dw_hh[i] += dw_hh_res.data[i]}
				l.matrix_free(&h_prev_t)
				l.matrix_free(&dw_hh_res)

				w_ih_t := _matrix_transpose(w_ih_in.data, allocator)
				dx_t_res := l.matmul_dyn_simd(&dh_mat, &w_ih_t, allocator)
				for b in 0 ..< batch {
					src := b * in_size
					dst := b * seq_len * in_size + s * in_size
					for i in 0 ..< in_size {
						dx[dst + i] += dx_t_res.data[src + i]
					}
				}
				l.matrix_free(&w_ih_t)
				l.matrix_free(&dx_t_res)

				w_hh_t := _matrix_transpose(w_hh_in.data, allocator)
				dh_prev_res := l.matmul_dyn_simd(&dh_mat, &w_hh_t, allocator)
				for i in 0 ..< len(dh_prev) {
					dh_prev[i] = dh_prev_res.data[i]
				}
				l.matrix_free(&w_hh_t)
				l.matrix_free(&dh_prev_res)
			}

			if x_in.requires_grad && len(x_in.grad.data) > 0 {
				l.vec_add_simd(x_in.grad.data, dx, x_in.grad.data)
			}
			if h_0_in.requires_grad && len(h_0_in.grad.data) > 0 {
				l.vec_add_simd(h_0_in.grad.data, dh_prev, h_0_in.grad.data)
			}
			if w_ih_in.requires_grad && len(w_ih_in.grad.data) > 0 {
				l.vec_add_simd(w_ih_in.grad.data, dw_ih, w_ih_in.grad.data)
			}
			if w_hh_in.requires_grad && len(w_hh_in.grad.data) > 0 {
				l.vec_add_simd(w_hh_in.grad.data, dw_hh, w_hh_in.grad.data)
			}
			if bias_in.requires_grad && len(bias_in.grad.data) > 0 {
				l.vec_add_simd(bias_in.grad.data, dbias, bias_in.grad.data)
			}
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
				sum_p_grad := 0.0
				for j in 0 ..< len(node.data.data) {
					sum_p_grad += node.data.data[j] * node.grad.data[j]
				}
				for i in 0 ..< len(a_in.grad.data) {
					a_in.grad.data[i] += node.data.data[i] * (node.grad.data[i] - sum_p_grad)
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
			input_in := node.inputs[0]
			weight_in := node.inputs[1]
			bias_in: ^Tensor = nil
			if len(node.inputs) > 2 {bias_in = node.inputs[2]}

			// ✅ ADD: Comprehensive checks for empty data
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: Conv2d node has empty gradient, skipping")
				continue
			}
			if len(input_in.data.data) == 0 {
				fmt.println("WARNING: Conv2d input has empty data, skipping")
				continue
			}
			if len(weight_in.data.data) == 0 {
				fmt.println("WARNING: Conv2d weight has empty data, skipping")
				continue
			}

			N := input_in.shape[0]
			C_in := input_in.shape[1]
			H := input_in.shape[2]
			W := input_in.shape[3]

			C_out := weight_in.shape[0]
			kH := node.conv_params.kH
			kW := node.conv_params.kW
			stride := node.conv_params.stride
			pad := node.conv_params.pad

			out_h := node.shape[2]
			out_w := node.shape[3]
			col_h := out_h * out_w
			col_w := C_in * kH * kW

			// ✅ ADD: Skip if dimensions are 0
			if col_h == 0 || col_w == 0 || C_out == 0 || N == 0 {
				fmt.printf(
					"WARNING: Conv2d has zero dimensions (col_h=%d, col_w=%d, C_out=%d, N=%d), skipping\n",
					col_h,
					col_w,
					C_out,
					N,
				)
				continue
			}

			col, _, _ := _im2col(input_in.data.data, N, C_in, H, W, kH, kW, stride, pad, allocator)

			// ✅ ADD: Check if col is empty
			if len(col) == 0 {
				fmt.println("WARNING: Conv2d im2col returned empty slice, skipping")
				continue
			}

			if input_in.requires_grad {
				tensor_ensure_grad(input_in)
				if len(input_in.grad.data) > 0 {
					grad_input_col := make([]f64, N * col_w * col_h, allocator)
					weight_2d_t := _matrix_transpose(weight_in.data, allocator)

					for n in 0 ..< N {
						grad_out_start := n * C_out * col_h
						if grad_out_start + C_out * col_h > len(node.grad.data) {
							fmt.printf(
								"WARNING: Conv2d grad_out slice out of bounds: %d:%d > %d\n",
								grad_out_start,
								grad_out_start + C_out * col_h,
								len(node.grad.data),
							)
							continue
						}

						grad_out_2d := l.Matrix(f64) {
							rows = C_out,
							cols = col_h,
							data = node.grad.data[grad_out_start:grad_out_start + C_out * col_h],
						}

						// ✅ ADD: Check if grad_out_2d is empty
						if len(grad_out_2d.data) == 0 {
							fmt.println("WARNING: Conv2d grad_out_2d is empty, skipping")
							continue
						}

						grad_col_2d := l.matmul_dyn_simd(&weight_2d_t, &grad_out_2d, allocator)

						col_start := n * col_w * col_h
						if col_start + col_w * col_h <= len(grad_input_col) {
							copy(
								grad_input_col[col_start:col_start + col_w * col_h],
								grad_col_2d.data,
							)
						}
						l.matrix_free(&grad_col_2d)
					}

					l.matrix_free(&weight_2d_t)

					grad_input := _col2im(
						grad_input_col,
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

					for i in 0 ..< len(input_in.grad.data) {
						if i < len(grad_input) {
							input_in.grad.data[i] += grad_input[i]
						}
					}

					delete(grad_input_col, allocator)
					delete(grad_input, allocator)
				}
			}

			if weight_in.requires_grad {
				tensor_ensure_grad(weight_in)
				if len(weight_in.grad.data) > 0 {
					grad_weight := make([]f64, len(weight_in.data.data), allocator)

					for n in 0 ..< N {
						grad_out_start := n * C_out * col_h
						if grad_out_start + C_out * col_h > len(node.grad.data) {continue}

						grad_out_2d := l.Matrix(f64) {
							rows = C_out,
							cols = col_h,
							data = node.grad.data[grad_out_start:grad_out_start + C_out * col_h],
						}

						col_start := n * col_w * col_h
						if col_start + col_w * col_h > len(col) {
							fmt.printf(
								"WARNING: Conv2d col slice out of bounds: %d:%d > %d\n",
								col_start,
								col_start + col_w * col_h,
								len(col),
							)
							continue
						}

						col_2d := l.Matrix(f64) {
							rows = col_w,
							cols = col_h,
							data = col[col_start:col_start + col_w * col_h],
						}

						col_2d_t := _matrix_transpose(col_2d, allocator)
						grad_weight_2d := l.matmul_dyn_simd(&grad_out_2d, &col_2d_t, allocator)

						for i in 0 ..< len(grad_weight) {
							if i < len(grad_weight_2d.data) {
								grad_weight[i] += grad_weight_2d.data[i]
							}
						}

						l.matrix_free(&col_2d_t)
						l.matrix_free(&grad_weight_2d)
					}

					for i in 0 ..< len(weight_in.grad.data) {
						weight_in.grad.data[i] += grad_weight[i]
					}

					delete(grad_weight, allocator)
				}
			}

			if bias_in != nil && bias_in.requires_grad {
				tensor_ensure_grad(bias_in)
				if len(bias_in.grad.data) > 0 {
					for n in 0 ..< N {
						for c in 0 ..< C_out {
							offset := n * C_out * col_h + c * col_h
							if offset + col_h > len(node.grad.data) {continue}
							sum := 0.0
							for i in 0 ..< col_h {sum += node.grad.data[offset + i]}
							bias_in.grad.data[c] += sum
						}
					}
				}
			}

			delete(col, allocator)

		// We don't calculate gradients for the target data.
		case .FlashAttention:
			Q_in := node.inputs[0]
			K_in := node.inputs[1]
			V_in := node.inputs[2]
			if len(node.grad.data) == 0 {continue}

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

			s_tile := make([]f64, Bc, context.allocator)
			p_tile := make([]f64, Bc, context.allocator)

			defer {
				delete(s_tile, context.allocator)
				delete(p_tile, context.allocator)
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
					D_i: f64 = 0.0 // sum_j P_ij (dO_i . V_j)

					for j_start := 0; j_start < seq_len; j_start += Bc {
						j_end := j_start + Bc
						if j_end > seq_len {j_end = seq_len}
						curr_Bc := j_end - j_start

						block_max := -math.F64_MAX
						for x in 0 ..< curr_Bc {
							k_offset := (b * seq_len + j_start + x) * d_k
							k_row := K_in.data.data[k_offset:k_offset + d_k]
							s_val := l.dot_simd(q_row, k_row) * scale
							s_tile[x] = s_val
							if s_val > block_max {block_max = s_val}
						}

						new_m := m_i
						if block_max > new_m {new_m = block_max}
						correction := math.exp(m_i - new_m)

						block_sum: f64 = 0.0
						block_D: f64 = 0.0
						for x in 0 ..< curr_Bc {
							p_val := math.exp(s_tile[x] - new_m)
							p_tile[x] = p_val
							block_sum += p_val

							v_offset := (b * seq_len + j_start + x) * d_v
							v_row := V_in.data.data[v_offset:v_offset + d_v]
							block_D += p_val * l.dot_simd(dO_row, v_row)
						}

						D_i = D_i * correction + block_D
						l_i = l_i * correction + block_sum
						m_i = new_m
					}

					// Pass 2: Compute dQ_i, and scatter dK_j, dV_j
					inv_l := 1.0 / l_i
					dQ_row := Q_in.grad.data[q_offset:q_offset + d_k]

					for j_start := 0; j_start < seq_len; j_start += Bc {
						j_end := j_start + Bc
						if j_end > seq_len {j_end = seq_len}
						curr_Bc := j_end - j_start

						for x in 0 ..< curr_Bc {
							k_offset := (b * seq_len + j_start + x) * d_k
							k_row := K_in.data.data[k_offset:k_offset + d_k]
							s_val := l.dot_simd(q_row, k_row) * scale
							p_val := math.exp(s_val - m_i) * inv_l

							v_offset := (b * seq_len + j_start + x) * d_v
							v_row := V_in.data.data[v_offset:v_offset + d_v]
							dot_dO_V := l.dot_simd(dO_row, v_row)

							dS_ij := p_val * (dot_dO_V - D_i) * scale

							if V_in.requires_grad {
								dV_row := V_in.grad.data[v_offset:v_offset + d_v]
								for d in 0 ..< d_v {dV_row[d] += p_val * dO_row[d]}
							}

							if Q_in.requires_grad {
								for d in 0 ..< d_k {dQ_row[d] += dS_ij * k_row[d]}
							}

							if K_in.requires_grad {
								dK_row := K_in.grad.data[k_offset:k_offset + d_k]
								for d in 0 ..< d_k {dK_row[d] += dS_ij * q_row[d]}
							}
						}
					}
				}
			}
		case .None, .Constant:
		// Leaf node, nothing to do
		}
	}
}
