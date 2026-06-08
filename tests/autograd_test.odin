
package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
autograd_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Engine ===")

	// 1. Create two 2x2 matrices (Leaf nodes)
	// A = [[1, 2], [3, 4]]
	data_a := l.matrix_new(f64, 2, 2, allocator)
	data_a.data[0] = 1; data_a.data[1] = 2
	data_a.data[2] = 3; data_a.data[3] = 4
	a := t.tensor_new(data_a, true, allocator)

	// B = [[5, 6], [7, 8]]
	data_b := l.matrix_new(f64, 2, 2, allocator)
	data_b.data[0] = 5; data_b.data[1] = 6
	data_b.data[2] = 7; data_b.data[3] = 8
	b := t.tensor_new(data_b, true, allocator)

	// 2. Build the graph: C = A + B
	c := t.tensor_add(a, b)

	// 3. Run Backward
	t.tensor_backward(c)

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
	t.tensor_free(a)
	t.tensor_free(b)
	t.tensor_free(c)
}


autograd_mul_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Multiplication ===")

	// A = [[1, 2], [3, 4]]
	data_a := l.matrix_new(f64, 2, 2, allocator)
	data_a.data[0] = 1; data_a.data[1] = 2
	data_a.data[2] = 3; data_a.data[3] = 4
	a := t.tensor_new(data_a, true, allocator)

	// B = [[5, 6], [7, 8]]
	data_b := l.matrix_new(f64, 2, 2, allocator)
	data_b.data[0] = 5; data_b.data[1] = 6
	data_b.data[2] = 7; data_b.data[3] = 8
	b := t.tensor_new(data_b, true, allocator)

	// C = A * B (Element-wise)
	// C should be [[5, 12], [21, 32]]
	c := t.tensor_mul(a, b)

	// To test backward, we need a scalar output.
	// Let's just pretend 'c' is our final loss for a moment by setting its grad to 1.0
	// (In a real NN, we'd have a sum() or mean() operation here).
	t.tensor_backward(c)

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
	t.tensor_free(a)
	t.tensor_free(b)
	t.tensor_free(c)
}
autograd_matmul_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Matrix Multiplication ===")

	// A = [[1, 2, 3],
	//      [4, 5, 6]]  (2x3)
	data_a := l.matrix_new(f64, 2, 3, allocator)
	data_a.data[0] = 1; data_a.data[1] = 2; data_a.data[2] = 3
	data_a.data[3] = 4; data_a.data[4] = 5; data_a.data[5] = 6
	a := t.tensor_new(data_a, true, allocator)

	// B = [[7, 8],
	//      [9, 10],
	//      [11, 12]] (3x2)
	data_b := l.matrix_new(f64, 3, 2, allocator)
	data_b.data[0] = 7; data_b.data[1] = 8
	data_b.data[2] = 9; data_b.data[3] = 10
	data_b.data[4] = 11; data_b.data[5] = 12
	b := t.tensor_new(data_b, true, allocator)

	// C = A @ B (2x2 output)
	c := t.tensor_matmul(a, b)

	// Trigger backward pass
	t.tensor_backward(c)

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
	t.tensor_free(a)
	t.tensor_free(b)
	t.tensor_free(c)
}

autograd_sum_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Sum (Scalar Loss) ===")

	// A = [[1, 2, 3], [4, 5, 6]] (2x3)
	data_a := l.matrix_new(f64, 2, 3, allocator)
	data_a.data[0] = 1; data_a.data[1] = 2; data_a.data[2] = 3
	data_a.data[3] = 4; data_a.data[4] = 5; data_a.data[5] = 6
	a := t.tensor_new(data_a, true, allocator)

	// B = [[7, 8], [9, 10], [11, 12]] (3x2)
	data_b := l.matrix_new(f64, 3, 2, allocator)
	data_b.data[0] = 7; data_b.data[1] = 8
	data_b.data[2] = 9; data_b.data[3] = 10
	data_b.data[4] = 11; data_b.data[5] = 12
	b := t.tensor_new(data_b, true, allocator)

	// C = A @ B (2x2)
	c := t.tensor_matmul(a, b)

	// ✅ L = sum(C) (1x1 scalar)
	loss := t.tensor_sum(c)

	fmt.printf("Loss value (Sum of C): %.1f\n", loss.data.data[0])
	// Expected: 58 + 64 + 139 + 154 = 415.0

	// ✅ Trigger backward pass from the SCALAR loss
	t.tensor_backward(loss)

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
	t.tensor_free(a)
	t.tensor_free(b)
	t.tensor_free(c)
	t.tensor_free(loss)
}
autograd_relu_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd ReLU ===")

	// A = [[-2, 3],
	//      [-1, 4]]
	data_a := l.matrix_new(f64, 2, 2, allocator)
	data_a.data[0] = -2; data_a.data[1] = 3
	data_a.data[2] = -1; data_a.data[3] = 4
	a := t.tensor_new(data_a, true, allocator)

	// B = ReLU(A) -> [[0, 3], [0, 4]]
	b := t.tensor_relu(a)

	// L = sum(B) -> 0 + 3 + 0 + 4 = 7
	loss := t.tensor_sum(b)

	fmt.printf("Loss value (Sum of ReLU): %.1f\n", loss.data.data[0])

	// Trigger backward pass
	t.tensor_backward(loss)

	// Gradient of A should be 0 where A was negative, and 1 where A was positive.
	fmt.println("Gradient of A (Should be 0, 1, 0, 1):")
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			fmt.printf("%.1f ", a.grad.data[i * 2 + j])
		}
		fmt.println("")
	}

	// Cleanup
	t.tensor_free(a)
	t.tensor_free(b)
	t.tensor_free(loss)
}

autograd_bias_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Autograd Bias Addition ===")

	// A = [[1, 2],
	//      [3, 4]] (2x2)
	data_a := l.matrix_new(f64, 2, 2, allocator)
	data_a.data[0] = 1; data_a.data[1] = 2
	data_a.data[2] = 3; data_a.data[3] = 4
	a := t.tensor_new(data_a, true, allocator)

	// bias = [[10, 20]] (1x2)
	data_bias := l.matrix_new(f64, 1, 2, allocator)
	data_bias.data[0] = 10; data_bias.data[1] = 20
	bias := t.tensor_new(data_bias, true, allocator)

	// C = A + bias -> [[11, 22], [13, 24]]
	c := t.tensor_add_bias(a, bias)

	// L = sum(C) -> 11 + 22 + 13 + 24 = 70
	loss := t.tensor_sum(c)

	fmt.printf("Loss value (Sum of A + bias): %.1f\n", loss.data.data[0])

	// Trigger backward pass
	t.tensor_backward(loss)

	// Gradient of A should be all 1s (since dL/dC is all 1s)
	fmt.println("Gradient of A (Should be all 1s):")
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			fmt.printf("%.1f ", a.grad.data[i * 2 + j])
		}
		fmt.println("")
	}

	// Gradient of bias should be the sum of columns of dL/dC.
	// Since dL/dC is all 1s, and there are 2 rows, the sum is [2, 2].
	fmt.println("Gradient of bias (Should be 2, 2):")
	fmt.printf("%.1f %.1f\n", bias.grad.data[0], bias.grad.data[1])

	// Cleanup
	t.tensor_free(a)
	t.tensor_free(bias)
	t.tensor_free(c)
	t.tensor_free(loss)
}
