
package autograd

import l "../linalg"
import "core:fmt"
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
	MatMul, // ✅ ADD THIS: Matrix multiplication
	Sum,
}
Tensor :: struct {
	data:          l.Matrix(f64), // The actual values
	grad:          l.Matrix(f64), // The gradients (same shape as data)
	requires_grad: bool, // Do we track history for this?
	op:            Op, // How was this tensor created?
	inputs:        [dynamic]^Tensor, // Pointers to the tensors that created this one
	allocator:     mem.Allocator,
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

	// ✅ FIX: Odin's delete for dynamic arrays doesn't take an allocator
	delete(t.inputs)
	free(t, t.allocator)
}

tensor_zero_grad :: proc(t: ^Tensor) {
	if t.grad.data != nil {
		for i in 0 ..< len(t.grad.data) {
			t.grad.data[i] = 0.0
		}
	}
}


// tensor_sum reduces a tensor to a single scalar value (1x1 matrix)
tensor_sum :: proc(a: ^Tensor) -> ^Tensor {
	// 1. Forward pass: sum all elements
	total := 0.0
	for v in a.data.data {
		total += v
	}

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
			// If C = A + B, then dC/dA = 1 and dC/dB = 1.
			for input in node.inputs {
				if input.requires_grad {
					for j in 0 ..< len(input.grad.data) {
						input.grad.data[j] += node.grad.data[j]
					}
				}
			}
		case .Mul:
			// C = A * B
			// dL/dA += dL/dC * B
			// dL/dB += dL/dC * A
			a_in := node.inputs[0]
			b_in := node.inputs[1]

			if a_in.requires_grad {
				for j in 0 ..< len(a_in.grad.data) {
					a_in.grad.data[j] += node.grad.data[j] * b_in.data.data[j]
				}
			}
			if b_in.requires_grad {
				for j in 0 ..< len(b_in.grad.data) {
					b_in.grad.data[j] += node.grad.data[j] * a_in.data.data[j]
				}
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
