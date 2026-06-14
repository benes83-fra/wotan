package tests

import data "../wotan/data"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"

// Test 1: Basic BatchNorm forward pass
batchnorm_forward_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing BatchNorm2d Forward Pass ===")

	// Create a simple input: (N=2, C=2, H=2, W=2)
	input := t.tensor_new_4d(2, 2, 2, 2, false, allocator)
	defer t.tensor_free(input)

	// Fill with known values
	// Channel 0: [1, 2, 3, 4, 5, 6, 7, 8]
	// Channel 1: [10, 20, 30, 40, 50, 60, 70, 80]
	for n in 0 ..< 2 {
		for c in 0 ..< 2 {
			for h in 0 ..< 2 {
				for w in 0 ..< 2 {
					idx := n * (2 * 2 * 2) + c * (2 * 2) + h * 2 + w
					if c == 0 {
						input.data.data[idx] = f64(n * 4 + h * 2 + w + 1)
					} else {
						input.data.data[idx] = f64((n * 4 + h * 2 + w + 1) * 10)
					}
				}
			}
		}
	}

	fmt.println("Input shape:", input.shape)
	fmt.println("Input data (first 8 values):")
	for i in 0 ..< 8 {
		fmt.printf("%.2f ", input.data.data[i])
	}
	fmt.println("")

	// Create BatchNorm layer
	bn_layer := nn.batch_norm_2d_layer_new(2, 1e-5, 0.1, allocator)
	defer nn.batch_norm_2d_layer_free(&bn_layer)

	// Test training mode
	fmt.println("\n--- Training Mode ---")
	output_train := t.tensor_batch_norm_2d(
		input,
		bn_layer.weight,
		bn_layer.bias,
		bn_layer.running_mean,
		bn_layer.running_var,
		true, // training mode
		0.1,
		1e-5,
	)
	defer t.tensor_free(output_train)

	fmt.println("Output shape:", output_train.shape)
	fmt.println("Output data (first 8 values):")
	for i in 0 ..< 8 {
		fmt.printf("%.4f ", output_train.data.data[i])
	}
	fmt.println("")

	fmt.println("Running mean after training:")
	for i in 0 ..< 2 {
		fmt.printf("%.4f ", bn_layer.running_mean.data.data[i])
	}
	fmt.println("")

	fmt.println("Running var after training:")
	for i in 0 ..< 2 {
		fmt.printf("%.4f ", bn_layer.running_var.data.data[i])
	}
	fmt.println("")

	// Test eval mode
	fmt.println("\n--- Eval Mode ---")
	output_eval := t.tensor_batch_norm_2d(
		input,
		bn_layer.weight,
		bn_layer.bias,
		bn_layer.running_mean,
		bn_layer.running_var,
		false, // eval mode
		0.1,
		1e-5,
	)
	defer t.tensor_free(output_eval)

	fmt.println("Output data (first 8 values):")
	for i in 0 ..< 8 {
		fmt.printf("%.4f ", output_eval.data.data[i])
	}
	fmt.println("")

	fmt.println("✓ BatchNorm forward pass test completed")
}

// Test 2: BatchNorm backward pass
batchnorm_backward_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing BatchNorm2d Backward Pass ===")

	// Create input with gradients
	input := t.tensor_new_4d(2, 2, 2, 2, true, allocator)
	defer t.tensor_free(input)

	// Fill with values
	for i in 0 ..< len(input.data.data) {
		input.data.data[i] = f64(i + 1)
	}

	// Create BatchNorm layer
	bn_layer := nn.batch_norm_2d_layer_new(2, 1e-5, 0.1, allocator)
	defer nn.batch_norm_2d_layer_free(&bn_layer)

	// Forward pass
	output := t.tensor_batch_norm_2d(
		input,
		bn_layer.weight,
		bn_layer.bias,
		bn_layer.running_mean,
		bn_layer.running_var,
		true,
		0.1,
		1e-5,
	)

	// Set output gradients to 1.0
	for i in 0 ..< len(output.grad.data) {
		output.grad.data[i] = 1.0
	}

	// Backward pass
	t.tensor_backward(output)

	fmt.println("Input gradients (first 8 values):")
	for i in 0 ..< 8 {
		fmt.printf("%.4f ", input.grad.data[i])
	}
	fmt.println("")

	fmt.println("Weight gradients:")
	for i in 0 ..< 2 {
		fmt.printf("%.4f ", bn_layer.weight.grad.data[i])
	}
	fmt.println("")

	fmt.println("Bias gradients:")
	for i in 0 ..< 2 {
		fmt.printf("%.4f ", bn_layer.bias.grad.data[i])
	}
	fmt.println("")

	t.tensor_free(output)
	fmt.println("✓ BatchNorm backward pass test completed")
}

// Test 3: BatchNorm in Sequential container
batchnorm_sequential_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing BatchNorm in Sequential ===")

	// Create a simple model: Conv2d -> BatchNorm -> ReLU
	model := nn.sequential_new(allocator)
	defer nn.sequential_free(model)

	nn.sequential_add(
		model,
		nn.conv2d_layer_new(1, 4, 3, 1, 1, true, allocator),
		nn.batch_norm_2d_layer_new(4, 1e-5, 0.1, allocator),
		nn.Activation.ReLU,
	)

	// Create input: (N=2, C=1, H=4, W=4)
	input := t.tensor_new_4d(2, 1, 4, 4, false, allocator)
	defer t.tensor_free(input)

	// Fill with values
	for i in 0 ..< len(input.data.data) {
		input.data.data[i] = f64(i) / f64(len(input.data.data))
	}

	fmt.println("Input shape:", input.shape)

	// Forward pass
	output := nn.sequential_forward(model, input)
	defer t.tensor_free(output)

	fmt.println("Output shape:", output.shape)
	fmt.println("Output data (first 8 values):")
	for i in 0 ..< 8 {
		fmt.printf("%.4f ", output.data.data[i])
	}
	fmt.println("")

	fmt.println("✓ BatchNorm in Sequential test completed")
}

// Test 4: BatchNorm with MNIST
batchnorm_mnist_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing BatchNorm with MNIST ===")

	// Load a small subset of MNIST
	train_set, ok := data.mnist_load(
		"data/mnist/train-images-idx3-ubyte",
		"data/mnist/train-labels-idx1-ubyte",
		allocator,
	)
	if !ok {
		fmt.println("Failed to load MNIST")
		return
	}
	defer data.mnist_free(&train_set)

	// Create model with BatchNorm
	model := nn.sequential_new(allocator)
	defer nn.sequential_free(model)

	nn.sequential_add(
		model,
		nn.conv2d_layer_new(1, 8, 3, 1, 1, true, allocator),
		nn.batch_norm_2d_layer_new(8, 1e-5, 0.1, allocator),
		nn.Activation.ReLU,
		nn.maxpool2d_layer_new(2, 2),
		nn.conv2d_layer_new(8, 16, 3, 1, 1, true, allocator),
		nn.batch_norm_2d_layer_new(16, 1e-5, 0.1, allocator),
		nn.Activation.ReLU,
		nn.maxpool2d_layer_new(2, 2),
		nn.FlattenLayer{},
		// ✅ FIX: Removed `true, allocator` to match your linear_layer_new signature
		nn.linear_layer_new(16 * 7 * 7, 64),
		nn.linear_layer_new(64, 10),
	)

	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)
	// ✅ FIX: Pass pointer to opt
	nn.sequential_add_to_opt(model, &opt)

	// Train for 1 epoch on a small batch
	batch_size := 32
	batch_imgs, batch_labs := data.mnist_get_batch(&train_set, 0, batch_size, allocator)
	defer t.tensor_free(batch_imgs)
	defer delete(batch_labs, allocator)

	fmt.println("Training on batch of", batch_size, "samples...")

	// Forward pass
	nn.adam_zero_grad(&opt)
	output := nn.sequential_forward(model, batch_imgs)
	loss := t.tensor_cross_entropy_loss(output, batch_labs)

	fmt.printf("Initial loss: %.4f\n", loss.data.data[0])

	// Backward pass
	t.tensor_backward(loss)
	nn.adam_step(&opt)

	t.tensor_free_graph(loss)
	t.tensor_free(output)

	fmt.println("✓ BatchNorm MNIST test completed")
}

// Test 5: Training vs Eval mode
batchnorm_train_eval_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Training vs Eval Mode ===")

	// Create model
	model := nn.sequential_new(allocator)
	defer nn.sequential_free(model)

	nn.sequential_add(
		model,
		nn.conv2d_layer_new(1, 4, 3, 1, 1, true, allocator),
		nn.batch_norm_2d_layer_new(4, 1e-5, 0.1, allocator),
	)

	// Create input
	input := t.tensor_new_4d(2, 1, 4, 4, false, allocator)
	defer t.tensor_free(input)

	for i in 0 ..< len(input.data.data) {
		input.data.data[i] = f64(i) / 10.0
	}

	// Training mode
	nn.sequential_train(model)
	output_train := nn.sequential_forward(model, input)
	defer t.tensor_free(output_train)

	fmt.println("Training mode output (first 4 values):")
	for i in 0 ..< 4 {
		fmt.printf("%.4f ", output_train.data.data[i])
	}
	fmt.println("")

	// Eval mode
	nn.sequential_eval(model)
	output_eval := nn.sequential_forward(model, input)
	defer t.tensor_free(output_eval)

	fmt.println("Eval mode output (first 4 values):")
	for i in 0 ..< 4 {
		fmt.printf("%.4f ", output_eval.data.data[i])
	}
	fmt.println("")

	// Outputs should be different (training uses batch stats, eval uses running stats)
	fmt.println("✓ Training vs Eval mode test completed")
}

// Run all BatchNorm tests
// Run all BatchNorm tests
run_all_batchnorm_tests :: proc(allocator: mem.Allocator) {
	fmt.println("")
	fmt.println("============================================================")
	fmt.println("RUNNING ALL BATCHNORM TESTS")
	fmt.println("============================================================")

	batchnorm_forward_test(allocator)
	batchnorm_backward_test(allocator)
	batchnorm_sequential_test(allocator)
	batchnorm_train_eval_test(allocator)
	batchnorm_mnist_test(allocator)

	fmt.println("")
	fmt.println("============================================================")
	fmt.println("ALL BATCHNORM TESTS COMPLETED SUCCESSFULLY")
	fmt.println("============================================================")
}
