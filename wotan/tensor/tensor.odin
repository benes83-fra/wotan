
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
	AddBias,
	MSELoss,
	CrossEntropy,
	Dropout, // ✅ ADD THIS
}

Tensor :: struct {
	data:          l.Matrix(f64),
	grad:          l.Matrix(f64),
	requires_grad: bool,
	op:            Op,
	inputs:        [dynamic]^Tensor,
	allocator:     mem.Allocator,
	int_metadata:  [dynamic]int,
	dropout_mask:  []f64, // ✅ ADD THIS: Stores the 0.0 or scale values
}
// ============================================================================
// 2. Construction & Lifecycle
// ============================================================================

tensor_new :: proc(
	data: l.Matrix(f64),
	requires_grad: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> ^Tensor {
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
} // ============================================================================
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

tensor_backward :: proc(root: ^Tensor) {
	if !root.requires_grad {return}

	// 1. Set the gradient of the root node to 1.0
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

		switch node.op {
		case .Add:
			for input in node.inputs {
				if input.requires_grad {
					// ✅ SIMD Optimization: y += 1.0 * x
					l.axpy_simd(1.0, node.grad.data, input.grad.data)
				}
			}
		case .Mul:
			a_in := node.inputs[0]
			b_in := node.inputs[1]

			if a_in.requires_grad {
				// ✅ SIMD Optimization: grad_a += grad_c * b
				l.vec_fma_inplace_simd(node.grad.data, b_in.data.data, a_in.grad.data)
			}
			if b_in.requires_grad {
				// ✅ SIMD Optimization: grad_b += grad_c * a
				l.vec_fma_inplace_simd(node.grad.data, a_in.data.data, b_in.grad.data)
			}
		case .MatMul:
			// C = A @ B
			// dA = dC @ B^T
			// dB = A^T @ dC
			a_in := node.inputs[0]
			b_in := node.inputs[1]

			if a_in.requires_grad {
				bt := _matrix_transpose(b_in.data, node.allocator)
				grad_a := l.matmul_dyn_simd(&node.grad, &bt, node.allocator)

				// Add to existing gradient
				l.vec_add_simd(a_in.grad.data, grad_a.data, a_in.grad.data)

				l.matrix_free(&bt)
				l.matrix_free(&grad_a)
			}

			if b_in.requires_grad {
				at := _matrix_transpose(a_in.data, node.allocator)
				grad_b := l.matmul_dyn_simd(&at, &node.grad, node.allocator)

				// Add to existing gradient
				l.vec_add_simd(b_in.grad.data, grad_b.data, b_in.grad.data)

				l.matrix_free(&at)
				l.matrix_free(&grad_b)
			}
		case .Sum:
			// L = sum(A)
			// dL/dA_ij = 1 for all i, j.
			// We just broadcast the incoming scalar gradient to the shape of A.
			a_in := node.inputs[0]
			if a_in.requires_grad {
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
			scalar_grad := node.grad.data[0] // Usually 1.0

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
		// We don't calculate gradients for the target data.
		case .None:
		// Leaf node, nothing to do
		}
	}
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
