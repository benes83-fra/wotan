package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
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
