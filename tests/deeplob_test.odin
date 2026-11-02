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
	seed_test(#procedure)
	fmt.println("=== DeepLOB Training Step Test ===")

	config := ml_fin.DeepLOBConfig {
		time_steps   = 100,
		price_levels = 10,
		num_classes  = 3,
		hidden_dim   = 64,
		seq_backend  = .Mamba,
	}

	model := ml_fin.deeplob_new(config, allocator)
	defer ml_fin.deeplob_free(model)

	opt := nn.adam_new(0.001, allocator = allocator)
	defer nn.adam_free(&opt)

	ml_fin.deeplob_add_to_adam(model, &opt)

	// ✅ FIX: Dummy input now has 3 channels (Mid-Price, Spread, OBI)
	// Shape: [Batch=8, Channels=3, Time=100, Levels=10]
	input_data := l.matrix_new(f64, 8, 3 * 100 * 10, allocator)
	input_tensor := t.tensor_new(input_data, true, allocator)
	input_tensor.shape = [4]int{8, 3, 100, 10}

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
	seed_test(#procedure)
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

	// One arena per training step, reused across epochs
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for epoch in 0 ..< epochs {
		t.graph_begin(&scope)
		defer t.graph_end(&scope)
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

// DeepLOB gradients against central finite differences, for every sequence
// backend: every parameter (CNN, inception block, sequence layer, head) gets a
// gradient, and it matches. The last time step used to be copied out of the
// sequence layer without a graph link, so only the head was trained.
deeplob_gradient_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== DeepLOB gradients vs finite differences ===")
	context.allocator = allocator
	passed, total := 0, 0

	@(static) R: []f64 // weights of the scalar loss Σ R ⊙ logits
	loss :: proc(model: ^ml_fin.DeepLOB, x: ^t.Tensor) -> ^t.Tensor {
		logits := ml_fin.deeplob_forward(model, x)
		alloc := t._scope_or(context.allocator) // the active scope
		w := t.tensor_new(
			l.matrix_new(f64, logits.data.rows, logits.data.cols, alloc),
			false,
			alloc,
		)
		w.shape = logits.shape
		copy(w.data.data, R)
		return t.tensor_sum(t.tensor_mul(logits, w))
	}

	for backend in nn.SeqBackend {
		config := ml_fin.DeepLOBConfig {
			time_steps   = 8,
			price_levels = 4,
			num_classes  = 3,
			hidden_dim   = 4,
			seq_backend  = backend,
		}
		model := ml_fin.deeplob_new(config)
		defer ml_fin.deeplob_free(model)
		ml_fin.deeplob_eval(model) // no dropout
		opt := nn.adam_new(0.001)
		defer nn.adam_free(&opt)
		ml_fin.deeplob_add_to_adam(model, &opt)

		B := 2
		xd := l.matrix_new(f64, B, 3 * 8 * 4)
		for &v in xd.data {v = rand.float64_range(-1, 1)}
		x := t.tensor_new(xd, false)
		x.shape = {B, 3, 8, 4}
		defer t.tensor_free(x)
		R = make([]f64, B * 3)
		defer delete(R)
		for &v in R {v = rand.float64_range(-1, 1)}

		scope: t.Graph_Scope
		defer t.graph_scope_destroy(&scope)
		value :: proc(model: ^ml_fin.DeepLOB, x: ^t.Tensor, scope: ^t.Graph_Scope) -> f64 {
			t.graph_begin(scope, no_grad = true)
			defer t.graph_end(scope)
			return loss(model, x).data.data[0]
		}

		nn.adam_zero_grad(&opt)
		t.graph_begin(&scope)
		t.tensor_backward(loss(model, x))
		t.graph_end(&scope)

		with_grad, max_err := 0, 0.0
		h := 1e-6
		for p in opt.parameters {
			nonzero := false
			for g in p.grad.data {nonzero ||= g != 0}
			if nonzero {with_grad += 1}
			n := len(p.data.data)
			for i in ([]int{0, n / 2, n - 1}) {
				x0 := p.data.data[i]
				p.data.data[i] = x0 + h
				up := value(model, x, &scope)
				p.data.data[i] = x0 - h
				down := value(model, x, &scope)
				p.data.data[i] = x0
				fd := (up - down) / (2 * h)
				max_err = max(max_err, abs(fd - p.grad.data[i]) / max(1, abs(fd)))
			}
		}
		n_params := len(opt.parameters)
		ok := with_grad == n_params && max_err < 1e-6
		total += 1
		if ok {passed += 1}
		fmt.printf(
			"%s DeepLOB with %v: %d/%d parameters get a gradient, max error vs finite differences %.1e\n",
			"OK  " if ok else "FAIL",
			backend,
			with_grad,
			n_params,
			max_err,
		)
	}
	fmt.printf("DeepLOB gradients: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
