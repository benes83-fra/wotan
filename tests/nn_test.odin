
package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
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
xor_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Training MLP on XOR Problem ===")

	// 1. Create Network: 2 inputs -> 4 hidden neurons -> 1 output
	sizes := []int{2, 4, 1}
	net := nn.mlp_new(sizes, allocator = allocator)
	defer nn.mlp_free(&net)

	// 2. Setup Optimizer (XOR requires a slightly higher learning rate)
	opt := nn.sgd_new(0.5, allocator)
	defer nn.sgd_free(&opt)
	nn.mlp_add_to_opt(&net, &opt)

	// 3. XOR Dataset
	x_data := l.matrix_new(f64, 4, 2, allocator)
	y_data := l.matrix_new(f64, 4, 1, allocator)

	x_data.data[0] = 0; x_data.data[1] = 0; y_data.data[0] = 0
	x_data.data[2] = 0; x_data.data[3] = 1; y_data.data[1] = 1
	x_data.data[4] = 1; x_data.data[5] = 0; y_data.data[2] = 1
	x_data.data[6] = 1; x_data.data[7] = 1; y_data.data[3] = 0

	x := t.tensor_new(x_data, false, allocator)
	target := t.tensor_new(y_data, false, allocator)
	defer t.tensor_free(x)
	defer t.tensor_free(target)

	// 4. Training Loop
	epochs := 2000
	for epoch in 0 ..< epochs {
		nn.sgd_zero_grad(&opt)

		// Forward pass through the entire MLP
		pred := nn.mlp_forward(&net, x)
		loss := t.tensor_mse_loss(pred, target)

		if epoch % 500 == 0 {
			fmt.printf("Epoch %d | Loss: %.6f\n", epoch, loss.data.data[0])
		}

		// Backward pass (Chain rule through MatMul -> Relu -> MatMul -> MSE)
		t.tensor_backward(loss)

		// Optimizer step
		nn.sgd_step(&opt)

		// ✅ Graph Cleanup:
		// This single call frees the 'pred', 'relu_out', and 'matmul_out'
		// tensors created during this epoch, without touching 'x', 'target',
		// or the network's weights/biases!
		t.tensor_free_graph(loss)
	}

	// 5. Test the learned function
	fmt.println("\nTesting XOR predictions:")
	pred_final := nn.mlp_forward(&net, x)
	for i in 0 ..< 4 {
		fmt.printf(
			"Input: [%.0f, %.0f] | Target: %.0f | Predicted: %.4f\n",
			x.data.data[i * 2],
			x.data.data[i * 2 + 1],
			target.data.data[i],
			pred_final.data.data[i],
		)
	}
	t.tensor_free_graph(pred_final)
}
classification_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Classification (Cross Entropy) ===")

	// 1. Create Network: 2 inputs -> 8 hidden -> 3 classes
	sizes := []int{2, 8, 3}
	net := nn.mlp_new(sizes, allocator = allocator)
	defer nn.mlp_free(&net)

	opt := nn.sgd_new(0.1, allocator)
	defer nn.sgd_free(&opt)
	nn.mlp_add_to_opt(&net, &opt)

	// 2. Create 3 Clusters of Data
	n_samples := 30 // 10 per class
	x_data := l.matrix_new(f64, n_samples, 2, allocator)
	targets := make([]int, n_samples, allocator)
	defer delete(targets, allocator)

	for i in 0 ..< n_samples {
		class_idx := i % 3
		targets[i] = class_idx

		// Cluster 0: around (-2, -2)
		// Cluster 1: around (2, 2)
		// Cluster 2: around (-2, 2)
		if class_idx == 0 {
			x_data.data[i * 2] = -2.0 + rand.float64_normal(0, 0.5)
			x_data.data[i * 2 + 1] = -2.0 + rand.float64_normal(0, 0.5)
		} else if class_idx == 1 {
			x_data.data[i * 2] = 2.0 + rand.float64_normal(0, 0.5)
			x_data.data[i * 2 + 1] = 2.0 + rand.float64_normal(0, 0.5)
		} else {
			x_data.data[i * 2] = -2.0 + rand.float64_normal(0, 0.5)
			x_data.data[i * 2 + 1] = 2.0 + rand.float64_normal(0, 0.5)
		}
	}

	x := t.tensor_new(x_data, false, allocator)
	defer t.tensor_free(x)

	// 3. Training Loop
	epochs := 500
	for epoch in 0 ..< epochs {
		nn.sgd_zero_grad(&opt)

		pred := nn.mlp_forward(&net, x)
		// ✅ Use the new Cross Entropy Loss
		loss := t.tensor_cross_entropy_loss(pred, targets)

		if epoch % 100 == 0 {
			fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])
		}

		t.tensor_backward(loss)
		nn.sgd_step(&opt)
		t.tensor_free_graph(loss)
	}

	// 4. Check Accuracy
	fmt.println("\nFinal Predictions:")
	pred_final := nn.mlp_forward(&net, x)
	correct := 0
	for i in 0 ..< n_samples {
		// Find class with highest probability
		max_prob := -math.F64_MAX
		pred_class := 0
		for j in 0 ..< 3 {
			v := pred_final.data.data[i * 3 + j]
			if v > max_prob {
				max_prob = v
				pred_class = j
			}
		}
		if pred_class == targets[i] {correct += 1}
	}

	fmt.printf(
		"Accuracy: %d / %d (%.1f%%)\n",
		correct,
		n_samples,
		f64(correct) / f64(n_samples) * 100.0,
	)
	t.tensor_free_graph(pred_final)
}


adam_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Adam Optimizer vs SGD ===")

	// Create Network: 2 inputs -> 4 hidden -> 1 output
	sizes := []int{2, 4, 1}

	// XOR Dataset
	x_data := l.matrix_new(f64, 4, 2, allocator)
	y_data := l.matrix_new(f64, 4, 1, allocator)

	x_data.data[0] = 0; x_data.data[1] = 0; y_data.data[0] = 0
	x_data.data[2] = 0; x_data.data[3] = 1; y_data.data[1] = 1
	x_data.data[4] = 1; x_data.data[5] = 0; y_data.data[2] = 1
	x_data.data[6] = 1; x_data.data[7] = 1; y_data.data[3] = 0

	x := t.tensor_new(x_data, false, allocator)
	target := t.tensor_new(y_data, false, allocator)
	defer t.tensor_free(x)
	defer t.tensor_free(target)

	// Test 1: SGD
	fmt.println("\n--- Training with SGD (lr=0.5) ---")
	net_sgd := nn.mlp_new(sizes, allocator = allocator)
	opt_sgd := nn.sgd_new(0.5, allocator)
	nn.mlp_add_to_opt(&net_sgd, &opt_sgd)

	epochs := 500
	for epoch in 0 ..< epochs {
		nn.sgd_zero_grad(&opt_sgd)
		pred := nn.mlp_forward(&net_sgd, x)
		loss := t.tensor_mse_loss(pred, target)

		if epoch % 100 == 0 {
			fmt.printf("Epoch %d | Loss: %.6f\n", epoch, loss.data.data[0])
		}

		t.tensor_backward(loss)
		nn.sgd_step(&opt_sgd)
		t.tensor_free_graph(loss)
	}
	nn.mlp_free(&net_sgd)
	nn.sgd_free(&opt_sgd)

	// Test 2: Adam
	fmt.println("\n--- Training with Adam (lr=0.01) ---")
	net_adam := nn.mlp_new(sizes, allocator = allocator)
	opt_adam := nn.adam_new(0.01, allocator = allocator) // Lower learning rate, Adam is more stable
	nn.mlp_add_to_opt(&net_adam, &opt_adam)

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt_adam)
		pred := nn.mlp_forward(&net_adam, x)
		loss := t.tensor_mse_loss(pred, target)

		if epoch % 100 == 0 {
			fmt.printf("Epoch %d | Loss: %.6f\n", epoch, loss.data.data[0])
		}

		t.tensor_backward(loss)
		nn.adam_step(&opt_adam)
		t.tensor_free_graph(loss)
	}
	nn.mlp_free(&net_adam)
	nn.adam_free(&opt_adam)
}

adam_classification_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Adam vs SGD on Classification ===")

	// Create Network: 2 inputs -> 16 hidden -> 3 classes
	sizes := []int{2, 16, 3}

	// Create 3 Clusters of Data
	n_samples := 90 // 30 per class
	x_data := l.matrix_new(f64, n_samples, 2, allocator)
	targets := make([]int, n_samples, allocator)
	defer delete(targets, allocator)

	for i in 0 ..< n_samples {
		class_idx := i % 3
		targets[i] = class_idx

		if class_idx == 0 {
			x_data.data[i * 2] = -3.0 + rand.float64_normal(0, 0.8)
			x_data.data[i * 2 + 1] = -3.0 + rand.float64_normal(0, 0.8)
		} else if class_idx == 1 {
			x_data.data[i * 2] = 3.0 + rand.float64_normal(0, 0.8)
			x_data.data[i * 2 + 1] = 3.0 + rand.float64_normal(0, 0.8)
		} else {
			x_data.data[i * 2] = -3.0 + rand.float64_normal(0, 0.8)
			x_data.data[i * 2 + 1] = 3.0 + rand.float64_normal(0, 0.8)
		}
	}

	x := t.tensor_new(x_data, false, allocator)
	defer t.tensor_free(x)

	// Test SGD
	fmt.println("\n--- SGD (lr=0.1) ---")
	net_sgd := nn.mlp_new(sizes, allocator = allocator)
	opt_sgd := nn.sgd_new(0.1, allocator)
	nn.mlp_add_to_opt(&net_sgd, &opt_sgd)

	epochs := 200
	for epoch in 0 ..< epochs {
		nn.sgd_zero_grad(&opt_sgd)
		pred := nn.mlp_forward(&net_sgd, x)
		loss := t.tensor_cross_entropy_loss(pred, targets)

		if epoch % 50 == 0 {
			fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])
		}

		t.tensor_backward(loss)
		nn.sgd_step(&opt_sgd)
		t.tensor_free_graph(loss)
	}
	nn.mlp_free(&net_sgd)
	nn.sgd_free(&opt_sgd)

	// Test Adam
	fmt.println("\n--- Adam (lr=0.01) ---")
	net_adam := nn.mlp_new(sizes, allocator = allocator)
	opt_adam := nn.adam_new(0.01, allocator = allocator)
	nn.mlp_add_to_opt(&net_adam, &opt_adam)

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt_adam)
		pred := nn.mlp_forward(&net_adam, x)
		loss := t.tensor_cross_entropy_loss(pred, targets)

		if epoch % 50 == 0 {
			fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])
		}

		t.tensor_backward(loss)
		nn.adam_step(&opt_adam)
		t.tensor_free_graph(loss)
	}
	nn.mlp_free(&net_adam)
	nn.adam_free(&opt_adam)
}
dropout_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Dropout Regularization ===")

	sizes := []int{2, 32, 3} // Larger network to force overfitting
	n_samples := 90
	x_data := l.matrix_new(f64, n_samples, 2, allocator)
	targets := make([]int, n_samples, allocator)
	defer delete(targets, allocator)

	for i in 0 ..< n_samples {
		class_idx := i % 3
		targets[i] = class_idx
		if class_idx == 0 {
			x_data.data[i * 2] = -3.0 + rand.float64_normal(0, 0.8)
			x_data.data[i * 2 + 1] = -3.0 + rand.float64_normal(0, 0.8)
		} else if class_idx == 1 {
			x_data.data[i * 2] = 3.0 + rand.float64_normal(0, 0.8)
			x_data.data[i * 2 + 1] = 3.0 + rand.float64_normal(0, 0.8)
		} else {
			x_data.data[i * 2] = -3.0 + rand.float64_normal(0, 0.8)
			x_data.data[i * 2 + 1] = 3.0 + rand.float64_normal(0, 0.8)
		}
	}

	x := t.tensor_new(x_data, false, allocator)
	defer t.tensor_free(x)

	// Test 1: No Dropout
	fmt.println("\n--- No Dropout (Overfitting likely) ---")
	net1 := nn.mlp_new(sizes, allocator = allocator)
	opt1 := nn.adam_new(0.01, allocator = allocator)
	nn.mlp_add_to_opt(&net1, &opt1)

	for epoch in 0 ..< 100 {
		nn.adam_zero_grad(&opt1)
		// ✅ training = true, drop_prob = 0.0
		pred := nn.mlp_forward(&net1, x, 0.0, true)
		loss := t.tensor_cross_entropy_loss(pred, targets)
		if epoch % 25 == 0 {fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])}
		t.tensor_backward(loss)
		nn.adam_step(&opt1)
		t.tensor_free_graph(loss)
	}
	nn.mlp_free(&net1)
	nn.adam_free(&opt1)

	// Test 2: With Dropout
	fmt.println("\n--- With Dropout (p=0.3) ---")
	net2 := nn.mlp_new(sizes, allocator = allocator)
	opt2 := nn.adam_new(0.01, allocator = allocator)
	nn.mlp_add_to_opt(&net2, &opt2)

	for epoch in 0 ..< 100 {
		nn.adam_zero_grad(&opt2)
		// ✅ training = true, drop_prob = 0.3
		pred := nn.mlp_forward(&net2, x, 0.3, true)
		loss := t.tensor_cross_entropy_loss(pred, targets)
		if epoch % 25 == 0 {fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])}
		t.tensor_backward(loss)
		nn.adam_step(&opt2)
		t.tensor_free_graph(loss)
	}
	nn.mlp_free(&net2)
	nn.adam_free(&opt2)
}
flexible_network_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Flexible Network Architecture ===")

	// Test 1: MLP with Tanh activation
	fmt.println("\n--- MLP with Tanh ---")
	net_tanh := nn.mlp_new([]int{2, 8, 1}, .Tanh, allocator)
	defer nn.mlp_free(&net_tanh)

	x_data := l.matrix_new(f64, 4, 2, allocator)
	x_data.data[0] = 0; x_data.data[1] = 0
	x_data.data[2] = 0; x_data.data[3] = 1
	x_data.data[4] = 1; x_data.data[5] = 0
	x_data.data[6] = 1; x_data.data[7] = 1
	x := t.tensor_new(x_data, false, allocator)
	defer t.tensor_free(x)

	y_data := l.matrix_new(f64, 4, 1, allocator)
	y_data.data[0] = 0; y_data.data[1] = 1; y_data.data[2] = 1; y_data.data[3] = 0
	y := t.tensor_new(y_data, false, allocator)
	defer t.tensor_free(y)

	opt := nn.adam_new(0.01, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.mlp_add_to_opt(&net_tanh, &opt)

	for epoch in 0 ..< 100 {
		nn.adam_zero_grad(&opt)
		pred := nn.mlp_forward(&net_tanh, x)
		loss := t.tensor_mse_loss(pred, y)
		if epoch % 25 == 0 {
			fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])
		}
		t.tensor_backward(loss)
		nn.adam_step(&opt)
		t.tensor_free_graph(loss)
	}

	// Test 2: Manual CNN composition
	fmt.println("\n--- Manual CNN Composition ---")
	conv1 := nn.conv2d_layer_new(1, 8, 3, 1, 0, true, allocator)
	defer nn.conv2d_layer_free(&conv1)

	pool1 := nn.maxpool2d_layer_new(2, 2)

	// Create a dummy 4x4 input
	input := t.tensor_new_4d(1, 1, 4, 4, false, allocator)
	defer t.tensor_free(input)
	for i in 0 ..< 16 {
		input.data.data[i] = f64(i) / 16.0
	}

	// Forward pass: Conv → ReLU → Pool
	out := nn.conv2d_layer_forward(&conv1, input)
	out = t.tensor_relu(out)
	out = nn.maxpool2d_layer_forward(&pool1, out)

	fmt.printf(
		"Input shape: %dx%dx%dx%d\n",
		input.shape[0],
		input.shape[1],
		input.shape[2],
		input.shape[3],
	)
	fmt.printf(
		"Output shape after Conv→ReLU→Pool: %dx%dx%dx%d\n",
		out.shape[0],
		out.shape[1],
		out.shape[2],
		out.shape[3],
	)

	t.tensor_free_graph(out)
}
