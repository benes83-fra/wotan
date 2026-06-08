
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


autograd_mul_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Multiplication ===")

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

	// C = A * B (Element-wise)
	// C should be [[5, 12], [21, 32]]
	c := autograd.tensor_mul(a, b)

	// To test backward, we need a scalar output.
	// Let's just pretend 'c' is our final loss for a moment by setting its grad to 1.0
	// (In a real NN, we'd have a sum() or mean() operation here).
	autograd.tensor_backward(c)

	// Math check:
	// dC/dA = B. Since we set dL/dC = 1, dL/dA should equal B.
	fmt.println("Gradient of A (Should match B: 5, 6, 7, 8):")
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			fmt.printf("%.1f ", a.grad.data[i * 2 + j])
		}
		fmt.println("")
	}

	// dC/dB = A. Since we set dL/dC = 1, dL/dB should equal A.
	fmt.println("Gradient of B (Should match A: 1, 2, 3, 4):")
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
autograd_matmul_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Matrix Multiplication ===")

	// A = [[1, 2, 3],
	//      [4, 5, 6]]  (2x3)
	data_a := l.matrix_new(f64, 2, 3, allocator)
	data_a.data[0] = 1; data_a.data[1] = 2; data_a.data[2] = 3
	data_a.data[3] = 4; data_a.data[4] = 5; data_a.data[5] = 6
	a := autograd.tensor_new(data_a, true, allocator)

	// B = [[7, 8],
	//      [9, 10],
	//      [11, 12]] (3x2)
	data_b := l.matrix_new(f64, 3, 2, allocator)
	data_b.data[0] = 7; data_b.data[1] = 8
	data_b.data[2] = 9; data_b.data[3] = 10
	data_b.data[4] = 11; data_b.data[5] = 12
	b := autograd.tensor_new(data_b, true, allocator)

	// C = A @ B (2x2 output)
	c := autograd.tensor_matmul(a, b)

	// Trigger backward pass
	autograd.tensor_backward(c)

	fmt.println("Gradient of A (2x3):")
	for i in 0 ..< 2 {
		for j in 0 ..< 3 {
			fmt.printf("%.1f ", a.grad.data[i * 3 + j])
		}
		fmt.println("")
	}

	fmt.println("Gradient of B (3x2):")
	for i in 0 ..< 3 {
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

autograd_sum_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Sum (Scalar Loss) ===")

	// A = [[1, 2, 3], [4, 5, 6]] (2x3)
	data_a := l.matrix_new(f64, 2, 3, allocator)
	data_a.data[0] = 1; data_a.data[1] = 2; data_a.data[2] = 3
	data_a.data[3] = 4; data_a.data[4] = 5; data_a.data[5] = 6
	a := autograd.tensor_new(data_a, true, allocator)

	// B = [[7, 8], [9, 10], [11, 12]] (3x2)
	data_b := l.matrix_new(f64, 3, 2, allocator)
	data_b.data[0] = 7; data_b.data[1] = 8
	data_b.data[2] = 9; data_b.data[3] = 10
	data_b.data[4] = 11; data_b.data[5] = 12
	b := autograd.tensor_new(data_b, true, allocator)

	// C = A @ B (2x2)
	c := autograd.tensor_matmul(a, b)

	// ✅ L = sum(C) (1x1 scalar)
	loss := autograd.tensor_sum(c)

	fmt.printf("Loss value (Sum of C): %.1f\n", loss.data.data[0])
	// Expected: 58 + 64 + 139 + 154 = 415.0

	// ✅ Trigger backward pass from the SCALAR loss
	autograd.tensor_backward(loss)

	// The gradients should be exactly the same as the matmul test!
	fmt.println("Gradient of A (2x3):")
	for i in 0 ..< 2 {
		for j in 0 ..< 3 {
			fmt.printf("%.1f ", a.grad.data[i * 3 + j])
		}
		fmt.println("")
	}

	fmt.println("Gradient of B (3x2):")
	for i in 0 ..< 3 {
		for j in 0 ..< 2 {
			fmt.printf("%.1f ", b.grad.data[i * 2 + j])
		}
		fmt.println("")
	}

	// Cleanup
	autograd.tensor_free(a)
	autograd.tensor_free(b)
	autograd.tensor_free(c)
	autograd.tensor_free(loss)
}
