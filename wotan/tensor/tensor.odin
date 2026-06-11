
package tensor

import l "../linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
// ============================================================================
// 1. The Tensor Struct & Graph Nodes
// ============================================================================

// Op defines the operation that created this tensor.
// We use an enum instead of function pointers to keep it Odin-idiomatic.
Op :: enum {
	None,
	Add,
	Mul,
	MatMul,
	Sum,
	Relu,
	Sigmoid, // ✅ ADD
	Tanh, // ✅ ADD
	LeakyReLU, // ✅ ADD
	AddBias,
	MSELoss,
	CrossEntropy,
	Dropout,
	Conv2d,
	Flatten,
	MaxPool2d,
	AvgPool2d,
}

PoolParams :: struct {
	kH:     int,
	kW:     int,
	stride: int,
	pad:    int,
}

Tensor :: struct {
	data:          l.Matrix(f64),
	grad:          l.Matrix(f64),
	requires_grad: bool,
	op:            Op,
	inputs:        [dynamic]^Tensor,
	allocator:     mem.Allocator,
	int_metadata:  [dynamic]int,
	dropout_mask:  []f64,
	shape:         [4]int,
	conv_params:   ConvParams,
	pool_params:   PoolParams, // ✅ ADD THIS
}

ConvParams :: struct {
	kH:     int,
	kW:     int,
	stride: int,
	pad:    int,
}
// ============================================================================
// 2. Construction & Lifecycle
// ============================================================================

tensor_new :: proc(
	data: l.Matrix(f64),
	requires_grad: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	if len(data.data) == 0 {
		fmt.printf(
			"WARNING: Creating tensor with empty data! rows=%d, cols=%d\n",
			data.rows,
			data.cols,
		)
	}
	t := new(Tensor, allocator)
	t.data = data
	t.requires_grad = requires_grad
	t.op = .None
	t.allocator = allocator

	// Initialize gradients to zero
	if requires_grad {
		t.grad = l.matrix_new(f64, data.rows, data.cols, allocator)
		// (In a real implementation, we'd zero it out here, but matrix_new usually does)
	}

	return t
}

tensor_free :: proc(t: ^Tensor) {
	if t == nil {return}
	if t.data.data != nil {l.matrix_free(&t.data)}
	if t.grad.data != nil {l.matrix_free(&t.grad)}

	delete(t.inputs)
	delete(t.int_metadata)
	if t.dropout_mask != nil {delete(t.dropout_mask, t.allocator)} 	// ✅ ADD THIS
	free(t, t.allocator)
}

tensor_zero_grad :: proc(t: ^Tensor) {
	if t.grad.data != nil {
		// Odin's compiler will auto-vectorize this simple zero loop perfectly.
		for i in 0 ..< len(t.grad.data) {
			t.grad.data[i] = 0.0
		}
	}
}

// tensor_dropout randomly zeroes elements with probability drop_prob.
// Uses Inverted Dropout: scales remaining elements by 1/(1-p) during training.
tensor_dropout :: proc(a: ^Tensor, drop_prob: f64, training: bool) -> ^Tensor {
	// If not training, or drop_prob is 0, just return the input tensor directly.
	// This is safe for tensor_free_graph because 'a' will have op == .None (or its original op).
	if !training || drop_prob <= 0.0 {
		return a
	}

	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	n := len(a.data.data)

	// Allocate the mask to remember what we dropped for the backward pass
	mask := make([]f64, n, a.allocator)
	scale := 1.0 / (1.0 - drop_prob)

	for i in 0 ..< n {
		if rand.float64() > drop_prob {
			out_data.data[i] = a.data.data[i] * scale
			mask[i] = scale
		} else {
			out_data.data[i] = 0.0
			mask[i] = 0.0
		}
	}

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.dropout_mask = mask

	if out.requires_grad {
		out.op = .Dropout
		append(&out.inputs, a)
	}

	return out
}
// tensor_sum reduces a tensor to a single scalar value (1x1 matrix)
tensor_sum :: proc(a: ^Tensor) -> ^Tensor {
	// ✅ SIMD Optimization: Fast reduction sum
	total := l.sum_simd(a.data.data)

	// Create a 1x1 matrix for the scalar output
	out_data := l.matrix_new(f64, 1, 1, a.allocator)
	out_data.data[0] = total

	// 2. Create output tensor
	out := tensor_new(out_data, a.requires_grad, a.allocator)

	// 3. Record graph
	if out.requires_grad {
		out.op = .Sum
		append(&out.inputs, a)
	}

	return out
}

// ============================================================================
// 3. Operations (Building the Graph)
// ============================================================================

tensor_add :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {
	// ✅ FIX: Check dimensions
	if a.data.rows != b.data.rows || a.data.cols != b.data.cols {
		panic("tensor_add: dimension mismatch")
	}

	// 1. Perform the forward pass using your existing SIMD add!
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_add_simd(a.data.data, b.data.data, out_data.data)

	// 2. Create the output tensor
	out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)

	// 3. Record the graph
	if out.requires_grad {
		out.op = .Add
		append(&out.inputs, a)
		append(&out.inputs, b)
	}

	return out
}

// tensor_relu applies the Rectified Linear Unit: max(0, x)
tensor_relu :: proc(a: ^Tensor) -> ^Tensor {

	// 1. Forward pass
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_relu_simd(a.data.data, out_data.data) // ✅ SIMD ReLU

	// 2. Create output tensor
	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape
	// 3. Record graph
	if out.requires_grad {
		out.op = .Relu
		append(&out.inputs, a)
	}

	return out
}


// tensor_mul creates a new tensor C = A * B (element-wise)
tensor_mul :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {
	if a.data.rows != b.data.rows || a.data.cols != b.data.cols {
		panic("tensor_mul: dimension mismatch")
	}

	// 1. Forward pass using SIMD
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	l.vec_mul_simd(a.data.data, b.data.data, out_data.data)

	// 2. Create output tensor
	out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)

	// 3. Record graph
	if out.requires_grad {
		out.op = .Mul
		append(&out.inputs, a)
		append(&out.inputs, b)
	}

	return out
}
// tensor_add_bias adds a 1xD bias vector to every row of an NxD matrix
// a: [N, D], bias: [1, D] -> out: [N, D]
tensor_add_bias :: proc(a: ^Tensor, bias: ^Tensor) -> ^Tensor {
	if a.data.cols != bias.data.cols {
		panic("tensor_add_bias: column mismatch")
	}
	if bias.data.rows != 1 {
		panic("tensor_add_bias: bias must be a 1xD row vector")
	}

	N := a.data.rows
	D := a.data.cols

	// 1. Forward pass
	out_data := l.matrix_new(f64, N, D, a.allocator)

	for i in 0 ..< N {
		row_out := out_data.data[i * D:(i + 1) * D]
		row_a := a.data.data[i * D:(i + 1) * D]

		// ✅ SIMD Optimization: out_row = a_row + bias
		l.vec_add_simd(row_a, bias.data.data, row_out)
	}

	// 2. Create output tensor
	out := tensor_new(out_data, a.requires_grad || bias.requires_grad, a.allocator)

	// 3. Record graph
	if out.requires_grad {
		out.op = .AddBias
		append(&out.inputs, a)
		append(&out.inputs, bias)
	}

	return out
}


// tensor_mse_loss calculates Mean Squared Error: mean((pred - target)^2)
// This is a Fused SIMD operation that avoids creating intermediate Sub/Mul tensors!
// tensor_mse_loss calculates Mean Squared Error: mean((pred - target)^2)
tensor_mse_loss :: proc(pred: ^Tensor, target: ^Tensor) -> ^Tensor {
	if pred.data.rows != target.data.rows || pred.data.cols != target.data.cols {
		panic("tensor_mse_loss: shape mismatch")
	}

	n := f64(len(pred.data.data))

	diff := make([]f64, len(pred.data.data), context.temp_allocator)
	l.vec_sub_simd(pred.data.data, target.data.data, diff)
	squared_sum := l.dot_simd(diff, diff)
	defer delete(diff, context.temp_allocator)

	loss_val := squared_sum / n

	out_data := l.matrix_new(f64, 1, 1, pred.allocator)
	out_data.data[0] = loss_val

	out := tensor_new(out_data, pred.requires_grad, pred.allocator)

	// ✅ FIX: Restored the correct MSE graph recording!
	if out.requires_grad {
		out.op = .MSELoss
		append(&out.inputs, pred)
		append(&out.inputs, target)
	}
	return out
}

// tensor_sigmoid applies σ(x) = 1 / (1 + exp(-x))
tensor_sigmoid :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	n := len(a.data.data)

	for i in 0 ..< n {
		out_data.data[i] = 1.0 / (1.0 + math.exp(-a.data.data[i]))
	}

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape

	if out.requires_grad {
		out.op = .Sigmoid
		append(&out.inputs, a)
	}
	return out
}

// tensor_tanh applies tanh(x)
tensor_tanh :: proc(a: ^Tensor) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	n := len(a.data.data)

	for i in 0 ..< n {
		out_data.data[i] = math.tanh(a.data.data[i])
	}

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape

	if out.requires_grad {
		out.op = .Tanh
		append(&out.inputs, a)
	}
	return out
}

// tensor_leaky_relu applies max(α*x, x) where α is small (e.g., 0.01)
tensor_leaky_relu :: proc(a: ^Tensor, alpha: f64 = 0.01) -> ^Tensor {
	out_data := l.matrix_new(f64, a.data.rows, a.data.cols, a.allocator)
	n := len(a.data.data)

	for i in 0 ..< n {
		v := a.data.data[i]
		out_data.data[i] = v > 0.0 ? v : alpha * v
	}

	out := tensor_new(out_data, a.requires_grad, a.allocator)
	out.shape = a.shape

	if out.requires_grad {
		out.op = .LeakyReLU
		append(&out.inputs, a)
	}
	return out
}


// tensor_cross_entropy_loss calculates the loss for multi-class classification.
tensor_cross_entropy_loss :: proc(logits: ^Tensor, target_indices: []int) -> ^Tensor {
	if len(target_indices) != logits.data.rows {
		panic("tensor_cross_entropy_loss: batch size mismatch")
	}

	N := logits.data.rows
	C := logits.data.cols

	loss := 0.0
	for i in 0 ..< N {
		max_val := -math.F64_MAX
		for j in 0 ..< C {
			v := logits.data.data[i * C + j]
			if v > max_val {max_val = v}
		}

		sum_exp := 0.0
		for j in 0 ..< C {
			sum_exp += math.exp(logits.data.data[i * C + j] - max_val)
		}

		// Note: Use math.log for natural logarithm in Odin
		log_sum_exp := math.ln(sum_exp) + max_val

		target_class := target_indices[i]
		log_prob := logits.data.data[i * C + target_class] - log_sum_exp
		loss -= log_prob
	}

	loss /= f64(N)

	out_data := l.matrix_new(f64, 1, 1, logits.allocator)
	out_data.data[0] = loss

	out := tensor_new(out_data, logits.requires_grad, logits.allocator)
	if out.requires_grad {
		out.op = .CrossEntropy
		append(&out.inputs, logits)
		// ✅ Save indices for the backward pass
		for idx in target_indices {
			append(&out.int_metadata, idx)
		}
	}
	return out
}


// tensor_flatten reshapes a 4D tensor (N, C, H, W) to 2D (N, C*H*W)
tensor_flatten :: proc(input: ^Tensor) -> ^Tensor {
	N := input.shape[0]
	features := input.shape[1] * input.shape[2] * input.shape[3]

	// Just create a new tensor with the same data but different shape
	out_data := l.matrix_new(f64, N, features, input.allocator)
	copy(out_data.data, input.data.data)

	out := tensor_new(out_data, input.requires_grad, input.allocator)
	out.shape = [4]int{N, features, 1, 1} // Treat as 2D

	if out.requires_grad {
		out.op = .Flatten
		append(&out.inputs, input)
	}

	return out
}


// ============================================================================
// 4. The Backward Engine (Chain Rule)
// ============================================================================

// Helper: Topological Sort to determine the order of operations
_build_topo :: proc(node: ^Tensor, topo: ^[dynamic]^Tensor, visited: ^map[^Tensor]bool) {
	if node == nil {return}
	if visited[node] {return}

	visited[node] = true

	for input in node.inputs {
		_build_topo(input, topo, visited)
	}

	append(topo, node)
}
// tensor_ensure_grad ensures a tensor has a gradient matrix allocated
tensor_ensure_grad :: proc(t: ^Tensor) {
	if t.requires_grad && t.grad.data == nil {
		t.grad = l.matrix_new(f64, t.data.rows, t.data.cols, t.allocator)
	}
}

tensor_backward :: proc(root: ^Tensor) {
	if !root.requires_grad {return}

	// 1. Set the gradient of the root node to 1.0
	tensor_ensure_grad(root)
	for i in 0 ..< len(root.grad.data) {
		root.grad.data[i] = 1.0
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
		case .MatMul:
			a_in := node.inputs[0]
			b_in := node.inputs[1]

			if a_in.requires_grad {
				tensor_ensure_grad(a_in)
				if len(a_in.grad.data) > 0 && len(node.grad.data) > 0 {
					bt := _matrix_transpose(b_in.data, node.allocator)
					grad_a := l.matmul_dyn_simd(&node.grad, &bt, node.allocator)
					l.vec_add_simd(a_in.grad.data, grad_a.data, a_in.grad.data)
					l.matrix_free(&bt)
					l.matrix_free(&grad_a)
				}
			}

			if b_in.requires_grad {
				tensor_ensure_grad(b_in)
				if len(b_in.grad.data) > 0 && len(node.grad.data) > 0 {
					at := _matrix_transpose(a_in.data, node.allocator)
					grad_b := l.matmul_dyn_simd(&at, &node.grad, node.allocator)
					l.vec_add_simd(b_in.grad.data, grad_b.data, b_in.grad.data)
					l.matrix_free(&at)
					l.matrix_free(&grad_b)
				}
			}
		case .Sum:
			// L = sum(A)
			// dL/dA_ij = 1 for all i, j.
			// We just broadcast the incoming scalar gradient to the shape of A.
			a_in := node.inputs[0]
			if a_in.requires_grad {
				if len(node.grad.data) == 0 {
					fmt.println("ERROR: Sum grad.data is empty")
					continue
				}
				scalar_grad := node.grad.data[0]
				for j in 0 ..< len(a_in.grad.data) {
					a_in.grad.data[j] += scalar_grad
				}
			}
		case .Relu:
			// y = max(0, x)
			// dy/dx = 1 if x > 0 else 0
			a_in := node.inputs[0]
			if a_in.requires_grad {
				// ✅ SIMD Optimization: Use the SIMD backward function
				l.vec_relu_backward_simd(node.grad.data, a_in.data.data, a_in.grad.data)
			}
		case .Sigmoid:
			// d/dx σ(x) = σ(x) * (1 - σ(x))
			a_in := node.inputs[0]
			if a_in.requires_grad {
				for i in 0 ..< len(a_in.grad.data) {
					s := a_in.data.data[i] // already σ(x) from forward
					a_in.grad.data[i] += node.grad.data[i] * s * (1.0 - s)
				}
			}

		case .Tanh:
			// d/dx tanh(x) = 1 - tanh²(x)
			a_in := node.inputs[0]
			if a_in.requires_grad {
				for i in 0 ..< len(a_in.grad.data) {
					t := a_in.data.data[i] // already tanh(x) from forward
					a_in.grad.data[i] += node.grad.data[i] * (1.0 - t * t)
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
		case .AddBias:
			// C = A + bias (broadcasted)
			a_in := node.inputs[0]
			bias_in := node.inputs[1]
			N := node.grad.rows
			D := node.grad.cols

			// dL/dA = dL/dC (just pass it through)
			if a_in.requires_grad {
				l.vec_add_simd(a_in.grad.data, node.grad.data, a_in.grad.data)
			}

			// dL/dbias = sum(dL/dC, axis=0)
			if bias_in.requires_grad {
				for i in 0 ..< N {
					row_grad := node.grad.data[i * D:(i + 1) * D]
					// ✅ SIMD Optimization: bias_grad += 1.0 * row_grad
					l.axpy_simd(1.0, row_grad, bias_in.grad.data)
				}
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
				diff := make([]f64, len(pred_in.data.data), context.temp_allocator)
				l.vec_sub_simd(pred_in.data.data, target_in.data.data, diff)

				// ✅ SIMD Optimization: grad_pred += scale * diff
				l.axpy_simd(scale, diff, pred_in.grad.data)
				delete(diff, context.temp_allocator)
			}
		case .CrossEntropy:
			// The magical simplified gradient: grad = (softmax_prob - target_one_hot) / N
			logits_in := node.inputs[0]
			N := logits_in.data.rows
			C := logits_in.data.cols
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: CrossEntropy node has empty gradient, skipping")
				continue
			}
			scalar_grad := node.grad.data[0] // Usually 1.0
			if len(node.int_metadata) < N {
				fmt.printf(
					"ERROR: CrossEntropy int_metadata length %d < N %d\n",
					len(node.int_metadata),
					N,
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

				target_class := node.int_metadata[i]

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
		case .Conv2d:
			input_in := node.inputs[0]
			weight_in := node.inputs[1]
			bias_in: ^Tensor = nil
			if len(node.inputs) > 2 {
				bias_in = node.inputs[2]
			}

			// ✅ CRITICAL: Check that we have valid gradient data
			if len(node.grad.data) == 0 {
				fmt.println("WARNING: Conv2d node has empty gradient, skipping")
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

			col, _, _ := _im2col(
				input_in.data.data,
				N,
				C_in,
				H,
				W,
				kH,
				kW,
				stride,
				pad,
				context.temp_allocator,
			)
			defer delete(col, context.temp_allocator)

			if input_in.requires_grad {
				tensor_ensure_grad(input_in)
				if len(input_in.grad.data) > 0 {
					grad_input_col := make([]f64, N * col_w * col_h, context.temp_allocator)

					weight_2d_t := _matrix_transpose(weight_in.data, context.temp_allocator)
					defer l.matrix_free(&weight_2d_t)

					for n in 0 ..< N {
						grad_out_start := n * C_out * col_h
						// ✅ CRITICAL: Bounds check before slicing
						if grad_out_start + C_out * col_h > len(node.grad.data) {
							fmt.printf(
								"ERROR: Conv2d grad_out slice out of bounds: %d:%d > %d\n",
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

						grad_col_2d := l.matmul_dyn_simd(
							&weight_2d_t,
							&grad_out_2d,
							context.temp_allocator,
						)

						col_start := n * col_w * col_h
						copy(grad_input_col[col_start:col_start + col_w * col_h], grad_col_2d.data)
						l.matrix_free(&grad_col_2d)
					}

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
						context.temp_allocator,
					)
					defer delete(grad_input, context.temp_allocator)

					for i in 0 ..< len(input_in.grad.data) {
						input_in.grad.data[i] += grad_input[i]
					}

					delete(grad_input_col, context.temp_allocator)
				}
			}

			if weight_in.requires_grad {
				tensor_ensure_grad(weight_in)
				if len(weight_in.grad.data) > 0 {
					grad_weight := make([]f64, len(weight_in.data.data), context.temp_allocator)
					defer delete(grad_weight, context.temp_allocator)

					for n in 0 ..< N {
						grad_out_start := n * C_out * col_h
						if grad_out_start + C_out * col_h > len(node.grad.data) {
							continue
						}

						grad_out_2d := l.Matrix(f64) {
							rows = C_out,
							cols = col_h,
							data = node.grad.data[grad_out_start:grad_out_start + C_out * col_h],
						}

						col_start := n * col_w * col_h
						col_2d := l.Matrix(f64) {
							rows = col_w,
							cols = col_h,
							data = col[col_start:col_start + col_w * col_h],
						}

						col_2d_t := _matrix_transpose(col_2d, context.temp_allocator)
						grad_weight_2d := l.matmul_dyn_simd(
							&grad_out_2d,
							&col_2d_t,
							context.temp_allocator,
						)

						for i in 0 ..< len(grad_weight) {
							grad_weight[i] += grad_weight_2d.data[i]
						}

						l.matrix_free(&col_2d_t)
						l.matrix_free(&grad_weight_2d)
					}

					for i in 0 ..< len(weight_in.grad.data) {
						weight_in.grad.data[i] += grad_weight[i]
					}
				}
			}

			if bias_in != nil && bias_in.requires_grad {
				tensor_ensure_grad(bias_in)
				if len(bias_in.grad.data) > 0 {
					for n in 0 ..< N {
						for c in 0 ..< C_out {
							offset := n * C_out * col_h + c * col_h
							if offset + col_h > len(node.grad.data) {
								continue
							}
							sum := 0.0
							for i in 0 ..< col_h {
								sum += node.grad.data[offset + i]
							}
							bias_in.grad.data[c] += sum
						}
					}
				}
			}
		// We don't calculate gradients for the target data.
		case .None:
		// Leaf node, nothing to do
		}
	}
}
// tensor_max_pool2d performs max pooling.
// input: (N, C, H, W) -> output: (N, C, H_out, W_out)
tensor_max_pool2d :: proc(input: ^Tensor, kH, kW, stride: int) -> ^Tensor {
	N := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]

	out_h := (H - kH) / stride + 1
	out_w := (W - kW) / stride + 1

	out_data := l.matrix_new(f64, 1, N * C * out_h * out_w, input.allocator)

	// We need to store the indices of the max values for the backward pass
	indices := make([]int, N * C * out_h * out_w, input.allocator)

	for n in 0 ..< N {
		for c in 0 ..< C {
			for oh in 0 ..< out_h {
				for ow in 0 ..< out_w {
					max_val := -math.F64_MAX
					max_idx := 0

					for kh in 0 ..< kH {
						for kw in 0 ..< kW {
							ih := oh * stride + kh
							iw := ow * stride + kw
							idx := n * (C * H * W) + c * (H * W) + ih * W + iw
							val := input.data.data[idx]
							if val > max_val {
								max_val = val
								max_idx = idx
							}
						}
					}

					out_idx := n * (C * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow
					out_data.data[out_idx] = max_val
					indices[out_idx] = max_idx
				}
			}
		}
	}

	out := tensor_new(out_data, input.requires_grad, input.allocator)
	out.shape = [4]int{N, C, out_h, out_w}

	if out.requires_grad {
		out.op = .MaxPool2d
		append(&out.inputs, input)
		out.pool_params = PoolParams{kH, kW, stride, 0}

		// Save indices to int_metadata for backward pass
		for idx in indices {
			append(&out.int_metadata, idx)
		}
	}

	delete(indices, input.allocator)
	return out
}
// tensor_avg_pool2d performs average pooling.
tensor_avg_pool2d :: proc(input: ^Tensor, kH, kW, stride: int) -> ^Tensor {
	N := input.shape[0]
	C := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]

	out_h := (H - kH) / stride + 1
	out_w := (W - kW) / stride + 1

	out_data := l.matrix_new(f64, 1, N * C * out_h * out_w, input.allocator)
	pool_size := f64(kH * kW)

	for n in 0 ..< N {
		for c in 0 ..< C {
			for oh in 0 ..< out_h {
				for ow in 0 ..< out_w {
					sum := 0.0
					for kh in 0 ..< kH {
						for kw in 0 ..< kW {
							ih := oh * stride + kh
							iw := ow * stride + kw
							idx := n * (C * H * W) + c * (H * W) + ih * W + iw
							sum += input.data.data[idx]
						}
					}
					out_idx := n * (C * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow
					out_data.data[out_idx] = sum / pool_size
				}
			}
		}
	}

	out := tensor_new(out_data, input.requires_grad, input.allocator)
	out.shape = [4]int{N, C, out_h, out_w}

	if out.requires_grad {
		out.op = .AvgPool2d
		append(&out.inputs, input)
		out.pool_params = PoolParams{kH, kW, stride, 0}
	}

	return out
}
// tensor_conv2d performs 2D convolution using im2col + SIMD matmul
// input: (N, C_in, H, W)
// weight: (C_out, C_in, kH, kW)
// bias: (C_out,) or nil
// output: (N, C_out, H_out, W_out)
tensor_conv2d :: proc(
	input: ^Tensor,
	weight: ^Tensor,
	bias: ^Tensor = nil,
	stride: int = 1,
	pad: int = 0,
) -> ^Tensor {
	// Extract shapes
	N := input.shape[0]
	C_in := input.shape[1]
	H := input.shape[2]
	W := input.shape[3]

	C_out := weight.shape[0]
	kH := weight.shape[2]
	kW := weight.shape[3]

	// Calculate output dimensions
	out_h := (H + 2 * pad - kH) / stride + 1
	out_w := (W + 2 * pad - kW) / stride + 1

	// 1. im2col: unfold input patches
	col, _, _ := _im2col(
		input.data.data,
		N,
		C_in,
		H,
		W,
		kH,
		kW,
		stride,
		pad,
		context.temp_allocator,
	)

	// 2. Reshape weight to (C_out, C_in*kH*kW)
	// Weight is already in the right layout, just treat it as a matrix

	// 3. Matrix multiply: (C_out, C_in*kH*kW) @ (C_in*kH*kW, H_out*W_out) per sample
	// For simplicity, we'll do this per-sample in the batch
	out_data := l.matrix_new(f64, 1, N * C_out * out_h * out_w, input.allocator)

	col_w := C_in * kH * kW
	col_h := out_h * out_w

	// Flatten weight to 2D: (C_out, col_w)
	weight_2d := l.Matrix(f64) {
		rows = C_out,
		cols = col_w,
		data = weight.data.data,
	}

	for n in 0 ..< N {
		// Extract this sample's columns: (col_w, col_h)
		col_start := n * col_w * col_h
		col_2d := l.Matrix(f64) {
			rows = col_w,
			cols = col_h,
			data = col[col_start:col_start + col_w * col_h],
		}

		// Matmul: (C_out, col_w) @ (col_w, col_h) -> (C_out, col_h)
		out_2d := l.matmul_dyn_simd(&weight_2d, &col_2d, context.temp_allocator)

		// Copy to output
		out_start := n * C_out * col_h
		copy(out_data.data[out_start:out_start + C_out * col_h], out_2d.data)
		l.matrix_free(&out_2d)
	}

	delete(col, context.temp_allocator)

	// 4. Add bias if provided
	if bias != nil {
		for n in 0 ..< N {
			for c in 0 ..< C_out {
				b := bias.data.data[c]
				offset := n * C_out * col_h + c * col_h
				for i in 0 ..< col_h {
					out_data.data[offset + i] += b
				}
			}
		}
	}

	// 5. Create output tensor
	out := tensor_new(out_data, input.requires_grad || weight.requires_grad, input.allocator)
	out.shape = [4]int{N, C_out, out_h, out_w}

	if out.requires_grad {
		out.op = .Conv2d
		append(&out.inputs, input)
		append(&out.inputs, weight)
		if bias != nil {
			append(&out.inputs, bias)
		}
		out.conv_params = ConvParams{kH, kW, stride, pad}
	}

	return out
}

// Internal helper: Transpose a matrix (needed for MatMul backward pass)
_matrix_transpose :: proc(m: l.Matrix(f64), alloc: mem.Allocator) -> l.Matrix(f64) {
	out := l.matrix_new(f64, m.cols, m.rows, alloc)
	for r in 0 ..< m.rows {
		for c in 0 ..< m.cols {
			out.data[c * out.cols + r] = m.data[r * m.cols + c]
		}
	}
	return out
}

// tensor_matmul creates a new tensor C = A @ B (Matrix Multiplication)
tensor_matmul :: proc(a: ^Tensor, b: ^Tensor) -> ^Tensor {
	if a.data.cols != b.data.rows {
		panic("tensor_matmul: dimension mismatch")
	}

	// 1. Forward pass using your blazing fast SIMD MatMul!
	out_data := l.matmul_dyn_simd(&a.data, &b.data, a.allocator)

	// 2. Create output tensor
	out := tensor_new(out_data, a.requires_grad || b.requires_grad, a.allocator)

	// 3. Record graph
	if out.requires_grad {
		out.op = .MatMul
		append(&out.inputs, a)
		append(&out.inputs, b)
	}

	return out
}

// _tensor_free_graph_impl is the recursive helper
_tensor_free_graph_impl :: proc(node: ^Tensor, visited: ^map[^Tensor]bool) {
	if node == nil {return}
	if visited[node] {return} 	// Prevent double-free in DAGs
	visited[node] = true

	// ✅ CRITICAL: Do NOT free leaf nodes (op == .None).
	// Leaf nodes are user-managed (e.g., input data, weights).
	if node.op == .None {
		return
	}

	// Recursively free inputs first
	for input in node.inputs {
		_tensor_free_graph_impl(input, visited)
	}

	// Free this intermediate node
	if node.data.data != nil {l.matrix_free(&node.data)}
	if node.grad.data != nil {l.matrix_free(&node.grad)}
	delete(node.inputs)
	delete(node.int_metadata)
	if node.dropout_mask != nil {delete(node.dropout_mask, node.allocator)}
	free(node, node.allocator)
}

// tensor_free_graph automatically cleans up all intermediate tensors
// in the computation graph, starting from 'root'.
// It protects leaf nodes (inputs/weights) from being freed.
tensor_free_graph :: proc(root: ^Tensor) {
	if root == nil {return}
	visited := make(map[^Tensor]bool)
	defer delete(visited)
	_tensor_free_graph_impl(root, &visited)
}


// _im2col unfolds image patches into columns for matrix multiplication
// Input shape: (N, C, H, W)
// Output shape: (N, C*kH*kW, H_out*W_out)
_im2col :: proc(
	input: []f64,
	N, C, H, W: int,
	kH, kW, stride, pad: int,
	allocator: mem.Allocator,
) -> (
	col: []f64,
	out_h: int,
	out_w: int,
) {
	out_h = (H + 2 * pad - kH) / stride + 1
	out_w = (W + 2 * pad - kW) / stride + 1

	col_size := N * (C * kH * kW) * (out_h * out_w)
	col = make([]f64, col_size, allocator)

	for n in 0 ..< N {
		for c in 0 ..< C {
			for kh in 0 ..< kH {
				for kw in 0 ..< kW {
					// Which row in the column matrix is this?
					col_row := c * (kH * kW) + kh * kW + kw

					for oh in 0 ..< out_h {
						for ow in 0 ..< out_w {
							// Which column in the column matrix?
							col_col := oh * out_w + ow

							// Input coordinates
							ih := oh * stride - pad + kh
							iw := ow * stride - pad + kw

							// Handle padding (zero-fill)
							val := 0.0
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								val = input[n * (C * H * W) + c * (H * W) + ih * W + iw]
							}

							// Write to col matrix: (N, C*kH*kW, H_out*W_out)
							idx :=
								n * ((C * kH * kW) * (out_h * out_w)) +
								col_row * (out_h * out_w) +
								col_col
							col[idx] = val
						}
					}
				}
			}
		}
	}

	return col, out_h, out_w
}

// _col2im is the backward pass of im2col: accumulates gradients back to input shape
_col2im :: proc(
	col: []f64,
	N, C, H, W: int,
	kH, kW, stride, pad, out_h, out_w: int,
	allocator: mem.Allocator,
) -> []f64 {
	grad_input := make([]f64, N * C * H * W, allocator)

	for n in 0 ..< N {
		for c in 0 ..< C {
			for kh in 0 ..< kH {
				for kw in 0 ..< kW {
					col_row := c * (kH * kW) + kh * kW + kw

					for oh in 0 ..< out_h {
						for ow in 0 ..< out_w {
							col_col := oh * out_w + ow

							ih := oh * stride - pad + kh
							iw := ow * stride - pad + kw

							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								idx_col :=
									n * ((C * kH * kW) * (out_h * out_w)) +
									col_row * (out_h * out_w) +
									col_col
								idx_in := n * (C * H * W) + c * (H * W) + ih * W + iw
								grad_input[idx_in] += col[idx_col]
							}
						}
					}
				}
			}
		}
	}

	return grad_input
}
tensor_new_4d :: proc(
	N, C, H, W: int,
	requires_grad: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
	data := l.matrix_new(f64, 1, N * C * H * W, allocator)
	t := tensor_new(data, requires_grad, allocator)
	t.shape = [4]int{N, C, H, W}
	return t
}
