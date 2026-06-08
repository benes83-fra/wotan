
package tests

import autograd "../wotan/autograd"
import l "../wotan/linalg"
import "core:fmt"
import "core:mem"
autograd_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Engine ===")

	// 1. Create two 2x2 matrices (Leaf nodes)
	// A = [[1, 2], [3, 4]]
	data_a := l.matrix_new(f64, 2, 2, allocator)
	data_a.data[0] = 1; data_a.data[1] = 2
	data_a.data[2] = 3; data_a.data[3] = 4
	a := autograd.tensor_new(data_a, true, allocator)

	// B = [[5, 6], [7, 8]]
	data_b := l.matrix_new(f64, 2, 2, allocator)
	data_b.data[0] = 5; data_b.data[1] = 6
	data_b.data[2] = 7; data_b.data[3] = 8
	b := autograd.tensor_new(data_b, true, allocator)

	// 2. Build the graph: C = A + B
	c := autograd.tensor_add(a, b)

	// 3. Run Backward
	autograd.tensor_backward(c)

	// 4. Check Gradients
	// Since C = A + B, the derivative of C with respect to A is 1, and B is 1.
	// So a.grad and b.grad should be filled with 1.0s.
	fmt.println("Gradient of A:")
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			fmt.printf("%.1f ", a.grad.data[i * 2 + j])
		}
		fmt.println("")
	}

	fmt.println("Gradient of B:")
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			fmt.printf("%.1f ", b.grad.data[i * 2 + j])
		}
		fmt.println("")
	}

	// Cleanup
	autograd.tensor_free(a)
	autograd.tensor_free(b)
	autograd.tensor_free(c)
}
