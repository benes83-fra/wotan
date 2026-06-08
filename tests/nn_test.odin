
package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
nn_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Neural Network Linear Layer ===")

	// Create a Linear Layer: 2 inputs -> 3 outputs
	layer := nn.linear_layer_new(2, 3, allocator)

	// Create a batch of 2 samples, each with 2 features
	// X = [[1.0, 2.0],
	//      [3.0, 4.0]]
	x_data := l.matrix_new(f64, 2, 2, allocator)
	x_data.data[0] = 1.0; x_data.data[1] = 2.0
	x_data.data[2] = 3.0; x_data.data[3] = 4.0
	x := t.tensor_new(x_data, true, allocator)

	defer t.tensor_free(x) // User manages input lifecycle

	// 3. Forward Pass (Clean API!)
	out := nn.linear_forward(&layer, x)
	fmt.printf("Input shape:  %dx%d\n", x.data.rows, x.data.cols)
	fmt.printf("Output shape: %dx%d\n", out.data.rows, out.data.cols)

	fmt.println("\nOutput values (Y = X @ W + b):")
	for i in 0 ..< out.data.rows {
		for j in 0 ..< out.data.cols {
			fmt.printf("%.4f ", out.data.data[i * out.data.cols + j])
		}
		fmt.println("")
	}

	// Cleanup
	t.tensor_free_graph(out)
}

optim_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing SGD Optimizer ===")

	// 1. Setup Layer and Optimizer
	layer := nn.linear_layer_new(2, 2, allocator)
	defer nn.linear_layer_free(&layer)

	opt := nn.sgd_new(0.1, allocator) // Learning rate = 0.1
	defer nn.sgd_free(&opt)

	// ✅ Register the layer's parameters with the optimizer
	nn.sgd_add_param(&opt, layer.weights)
	nn.sgd_add_param(&opt, layer.bias)

	// 2. Create Input (1 sample, 2 features)
	x_data := l.matrix_new(f64, 1, 2, allocator)
	x_data.data[0] = 1.0; x_data.data[1] = 1.0
	x := t.tensor_new(x_data, true, allocator)
	defer t.tensor_free(x)

	fmt.println("Initial Weights:")
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			fmt.printf("%.4f ", layer.weights.data.data[i * 2 + j])
		}
		fmt.println("")
	}

	// 3. Forward Pass
	out := nn.linear_forward(&layer, x)

	// 4. Dummy Loss (just sum the output to get a scalar to backprop from)
	loss := t.tensor_sum(out)
	fmt.printf("\nInitial Loss: %.4f\n", loss.data.data[0])

	// 5. Backward Pass (Calculates gradients)
	t.tensor_backward(loss)

	fmt.println("\nGradients of Weights:")
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			fmt.printf("%.4f ", layer.weights.grad.data[i * 2 + j])
		}
		fmt.println("")
	}

	// 6. Optimizer Step (Updates weights using gradients)
	nn.sgd_step(&opt)

	fmt.println("\nUpdated Weights (after SGD step):")
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			fmt.printf("%.4f ", layer.weights.data.data[i * 2 + j])
		}
		fmt.println("")
	}

	// 7. Cleanup the computation graph
	t.tensor_free_graph(loss)
}


mse_train_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Training a Layer with MSE Loss ===")

	// 1. Setup: 1 input -> 1 output
	layer := nn.linear_layer_new(1, 1, allocator)
	defer nn.linear_layer_free(&layer)

	opt := nn.sgd_new(0.1, allocator) // High learning rate for fast demo
	defer nn.sgd_free(&opt)
	nn.sgd_add_param(&opt, layer.weights)
	nn.sgd_add_param(&opt, layer.bias)

	// Target function: y = 2x
	n_samples := 4
	x_data := l.matrix_new(f64, n_samples, 1, allocator)
	y_data := l.matrix_new(f64, n_samples, 1, allocator)

	for i in 0 ..< n_samples {
		x_val := f64(i)
		x_data.data[i] = x_val
		y_data.data[i] = 2.0 * x_val // Target
	}

	x := t.tensor_new(x_data, false, allocator) // Inputs don't need grad
	target := t.tensor_new(y_data, false, allocator) // Targets don't need grad
	defer t.tensor_free(x)
	defer t.tensor_free(target)

	// 2. Training Loop
	epochs := 50
	for epoch in 0 ..< epochs {
		nn.sgd_zero_grad(&opt)

		// Forward
		pred := nn.linear_forward(&layer, x)
		loss := t.tensor_mse_loss(pred, target)

		if epoch % 10 == 0 {
			fmt.printf(
				"Epoch %d | Loss: %.4f | Weight: %.4f | Bias: %.4f\n",
				epoch,
				loss.data.data[0],
				layer.weights.data.data[0],
				layer.bias.data.data[0],
			)
		}

		// Backward
		t.tensor_backward(loss)

		// Step
		nn.sgd_step(&opt)

		// ✅ Graph Cleanup: This single call frees 'pred' and 'matmul_out'
		// without touching 'x', 'target', or the 'layer' weights!
		t.tensor_free_graph(loss)
	}

	fmt.println("\nFinal Learned Parameters:")
	fmt.printf("Weight: %.4f (Target: 2.0)\n", layer.weights.data.data[0])
	fmt.printf("Bias:   %.4f (Target: 0.0)\n", layer.bias.data.data[0])
}
