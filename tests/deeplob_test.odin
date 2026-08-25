package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

deeplob_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== DeepLOB Training Step Test ===")

	config := ml_fin.DeepLOBConfig {
		time_steps   = 100,
		price_levels = 10,
		num_classes  = 3,
		hidden_dim   = 64,
	}

	model := ml_fin.deeplob_new(config, allocator)
	defer ml_fin.deeplob_free(model)

	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)

	ml_fin.deeplob_add_to_adam(model, &opt)

	// Dummy input: [Batch=8, Channels=4, Time=100, Levels=10]
	// Wotan stores 4D tensors as flat 2D matrices under the hood.
	input_data := l.matrix_new(f64, 8, 4 * 100 * 10, allocator)
	input_tensor := t.tensor_new(input_data, true, allocator)
	input_tensor.shape = [4]int{8, 4, 100, 10}

	// Forward pass
	logits := ml_fin.deeplob_forward(model, input_tensor)

	// Dummy targets for Cross Entropy Loss
	targets := []int{0, 1, 2, 0, 1, 2, 0, 1}
	loss := t.tensor_cross_entropy_loss(logits, targets)

	fmt.printf("Initial Loss: %.4f\n", loss.data.data[0])

	// Backward pass & Optimizer Step
	nn.adam_zero_grad(&opt)
	t.tensor_backward(loss)
	nn.adam_step(&opt)

	// Cleanup the ENTIRE computation graph safely
	t.tensor_free_graph(loss)
	t.tensor_free(input_tensor)

	fmt.println("✓ DeepLOB training step complete!")
}
deeplob_talkative_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== DeepLOB (Synthetic LOB) Talkative Training Test ===")

	// 1. Hyperparameters
	batch_size := 32
	seq_len := 100 // Time steps (e.g., 100 order book snapshots)
	features := 40 // Features (e.g., 10 bid prices, 10 bid vols, 10 ask prices, 10 ask vols)
	n_classes := 3 // Up, Down, Flat

	// 2. Generate Synthetic "Limit Order Book" Data
	// Shape: (batch_size, 1, seq_len, features) for Conv2d
	fmt.printf(
		"Generating synthetic LOB data: batch=%d, seq=%d, features=%d\n",
		batch_size,
		seq_len,
		features,
	)

	x_data := l.matrix_new(f64, batch_size, 1 * seq_len * features, allocator)
	// Fill with random noise (simulating normalized order book imbalances)
	for i in 0 ..< len(x_data.data) {
		x_data.data[i] = rand.float64_normal(0.0, 1.0)
	}
	x_tensor := t.tensor_new(x_data, false, allocator)
	x_tensor.shape = [4]int{batch_size, 1, seq_len, features}

	// Generate random 3-class targets
	y_targets := make([]int, batch_size, allocator)
	for i in 0 ..< batch_size {
		y_targets[i] = rand.int_range(0, n_classes)
	}

	// 3. Build a Mini-DeepLOB Architecture using existing NN layers
	// Real DeepLOB uses Inception modules, but we'll use a standard CNN pipeline
	// to demonstrate the training loop and loss reduction using your existing library.
	fmt.println("Building Mini-DeepLOB (CNN) architecture...")
	model := nn.sequential_new(allocator)
	defer nn.sequential_free(model)

	// Conv Block 1: Extract local temporal/feature patterns
	nn.sequential_add(model, nn.conv2d_layer_new(1, 16, 3, 1, 1, true, allocator)) // (N, 16, 100, 40)
	nn.sequential_add(model, nn.Activation.ReLU)

	// Conv Block 2
	nn.sequential_add(model, nn.conv2d_layer_new(16, 32, 3, 1, 1, true, allocator)) // (N, 32, 100, 40)
	nn.sequential_add(model, nn.Activation.ReLU)

	// Pooling to reduce sequence length
	nn.sequential_add(model, nn.maxpool2d_layer_new(2, 2)) // (N, 32, 50, 20)

	// Flatten and classify
	nn.sequential_add(model, nn.FlattenLayer{})
	nn.sequential_add(model, nn.linear_layer_new(32 * 50 * 20, 64, allocator))
	nn.sequential_add(model, nn.Activation.ReLU)
	nn.sequential_add(model, nn.linear_layer_new(64, n_classes, allocator))

	// 4. Setup Optimizer
	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(model, &opt)

	// 5. Training Loop
	epochs := 20
	fmt.println("\nStarting Training...")
	fmt.println("Epoch | Loss    | Accuracy | Status")
	fmt.println("------|---------|----------|-------------------------")

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Forward pass
		logits := nn.sequential_forward(model, x_tensor)

		// Compute Cross Entropy Loss
		loss := t.tensor_cross_entropy_loss(logits, y_targets)

		// Backward pass
		t.tensor_backward(loss)

		// Optimizer step
		nn.adam_step(&opt)

		// Calculate Accuracy
		correct := 0
		for i in 0 ..< batch_size {
			// Find argmax for this sample
			max_val := -math.F64_MAX
			max_idx := 0
			for c in 0 ..< n_classes {
				val := logits.data.data[i * n_classes + c]
				if val > max_val {
					max_val = val
					max_idx = c
				}
			}
			if max_idx == y_targets[i] {
				correct += 1
			}
		}
		acc := f64(correct) / f64(batch_size) * 100.0

		// Print progress
		status := ""
		if epoch == 0 {
			status = "(Initial random guess ~ln(3))"
		} else if loss.data.data[0] < 1.00 {
			status = "(Learning! Loss < 1.0)"
		} else if acc > 40.0 {
			status = "(Overfitting synthetic data)"
		}

		fmt.printf(" %3d  | %.5f | %6.2f%%  | %s\n", epoch + 1, loss.data.data[0], acc, status)

		// Cleanup graph for this step
		t.tensor_free_graph(loss)
		t.tensor_free_graph(logits)
	}

	fmt.println("\n✓ DeepLOB training step complete!")
	fmt.println("Note: As expected, the loss started at ~1.0986 (ln(3)) and should have")
	fmt.println("decreased as the CNN learned to map the synthetic noise to the targets.")

	// Cleanup
	t.tensor_free(x_tensor)
	delete(y_targets, allocator)
}
