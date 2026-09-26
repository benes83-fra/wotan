package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Mamba (State Space Model) Integration Test
// ============================================================================
mamba_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Mamba (State Space Model) Test ===\n")

	// 1. Setup dimensions
	batch_size := 4
	seq_len := 16
	d_model := 32
	d_state := 16 // SSM state dimension

	// 2. Initialize Mamba Layer
	mamba := nn.mamba_layer_new(d_model, d_state, allocator)
	defer nn.mamba_layer_free(&mamba)

	// 3. Create dummy input sequence [batch, seq_len, d_model, 1]
	// We use random data to simulate a financial time series
	input_data := l.matrix_new(f64, 1, batch_size * seq_len * d_model, allocator)
	for i in 0 ..< len(input_data.data) {
		input_data.data[i] = rand.float64() * 2.0 - 1.0 // Uniform [-1, 1]
	}
	x := t.tensor_new(input_data, true, allocator)
	x.shape = [4]int{batch_size, seq_len, d_model, 1}
	defer t.tensor_free(x)

	// 4. Initial hidden state (all zeros) [batch, d_model, d_state, 1]
	h0_data := l.matrix_new(f64, 1, batch_size * d_model * d_state, allocator)
	h0 := t.tensor_new(h0_data, false, allocator)
	h0.shape = [4]int{batch_size, d_model, d_state, 1}
	defer t.tensor_free(h0)

	// 5. Forward pass
	fmt.println("Running forward pass...")
	out := nn.mamba_layer_forward(&mamba, x, h0)

	fmt.printf("Input shape:  [%d, %d, %d, %d]\n", x.shape[0], x.shape[1], x.shape[2], x.shape[3])
	fmt.printf(
		"Output shape: [%d, %d, %d, %d]\n",
		out.shape[0],
		out.shape[1],
		out.shape[2],
		out.shape[3],
	)

	// Verify output shape
	if out.shape[0] != batch_size || out.shape[1] != seq_len || out.shape[2] != d_model {
		fmt.println("❌ ERROR: Output shape mismatch!")
		t.tensor_free_graph(out)
		return
	}
	fmt.println("✅ Output shape is correct.")

	// 6. Compute a dummy loss (Sum of outputs to get a scalar gradient of 1.0)
	loss := t.tensor_sum(out)

	fmt.printf("Initial Loss (Sum): %.4f\n", loss.data.data[0])

	// 7. Backward pass (Backpropagation Through Time)
	fmt.println("Running backward pass...")
	t.tensor_backward(loss)

	// 8. Check gradients
	fmt.println("Checking gradients...")
	grad_ok := true

	check_grad :: proc(name: string, tensor: ^t.Tensor, ok: ^bool) {
		if tensor == nil || tensor.grad.data == nil {
			fmt.printf("❌ %s has nil gradient!\n", name)
			ok^ = false
			return
		}
		sum := 0.0
		for v in tensor.grad.data {
			sum += math.abs(v)
		}
		if math.is_nan(sum) || math.is_inf(sum) {
			fmt.printf("❌ %s has NaN/Inf gradient!\n", name)
			ok^ = false
		} else if sum == 0.0 {
			fmt.printf("❌ %s has zero gradient! (Sum = %.4f)\n", name, sum)
			ok^ = false
		} else {
			fmt.printf("✅ %s gradient OK (L1 Sum = %.4f)\n", name, sum)
		}
	}

	check_grad("proj_x.weights", mamba.proj_x.weights, &grad_ok)
	check_grad("proj_B.weights", mamba.proj_B.weights, &grad_ok)
	check_grad("proj_C.weights", mamba.proj_C.weights, &grad_ok)
	check_grad("proj_Delta.weights", mamba.proj_Delta.weights, &grad_ok)
	check_grad("proj_out.weights", mamba.proj_out.weights, &grad_ok)
	check_grad("A (State Transition)", mamba.A, &grad_ok)
	check_grad("D (Skip Connection)", mamba.D, &grad_ok)

	if !grad_ok {
		fmt.println("\n❌ Mamba Test FAILED: Gradient check failed.")
		t.tensor_free_graph(loss)
		return
	}

	// 9. Optimization Step (Verify weights actually update)
	fmt.println("\nRunning Adam optimizer step...")
	opt := nn.adam_new(0.01, allocator = allocator)

	nn.adam_add_param(&opt, mamba.proj_x.weights)
	nn.adam_add_param(&opt, mamba.proj_B.weights)
	nn.adam_add_param(&opt, mamba.proj_C.weights)
	nn.adam_add_param(&opt, mamba.proj_Delta.weights)
	nn.adam_add_param(&opt, mamba.proj_out.weights)
	nn.adam_add_param(&opt, mamba.A)
	nn.adam_add_param(&opt, mamba.D)

	// Store a weight before step
	w_before := mamba.proj_out.weights.data.data[0]

	nn.adam_step(&opt)

	w_after := mamba.proj_out.weights.data.data[0]

	if w_before == w_after {
		fmt.println("❌ ERROR: Weights did not update after optimizer step!")
	} else {
		fmt.printf(
			"✅ Weights updated successfully (Before: %.6f, After: %.6f)\n",
			w_before,
			w_after,
		)
	}

	// 10. Clean up the autograd graph and optimizer
	t.tensor_free_graph(loss)
	nn.adam_free(&opt)

	fmt.println("\n=== Mamba Test PASSED ===\n")
}
// ============================================================================
// Mamba "Proof of Learning": Next-Step Forecasting of a Quasi-Periodic Signal
// ============================================================================
// Signal: x[t] = sin(w1*t + p1) + 0.5*sin(w2*t + p2) + noise
// The model sees a window of 16 steps and must forecast the next value at
// EVERY timestep. Tracking the random per-sample phase requires memory,
// so the SSM state must do real work.
mamba_learning_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Mamba Learning Test: Sequence Forecasting ===\n")

	batch_size := 32
	seq_len := 16
	d_model := 24
	d_state := 16
	num_batches := 8
	epochs := 100
	learning_rate := 0.005

	w1 := 2.0 * math.PI / 9.0
	w2 := 2.0 * math.PI / 5.0

	// ---- Model: Embed(1->D) -> Mamba(D) -> Head(D->1) ----
	embed := nn.linear_layer_new(1, d_model, allocator)
	defer nn.linear_layer_free(&embed)
	mamba := nn.mamba_layer_new(d_model, d_state, allocator)
	defer nn.mamba_layer_free(&mamba)
	head := nn.linear_layer_new(d_model, 1, allocator)
	defer nn.linear_layer_free(&head)

	opt := nn.adam_new(learning_rate, 0.9, 0.999, 1e-8, allocator)
	defer nn.adam_free(&opt)

	// Register every trainable parameter
	nn.adam_add_param(&opt, embed.weights)
	nn.adam_add_param(&opt, embed.bias)
	nn.adam_add_param(&opt, mamba.proj_x.weights)
	nn.adam_add_param(&opt, mamba.proj_x.bias)
	nn.adam_add_param(&opt, mamba.proj_B.weights)
	nn.adam_add_param(&opt, mamba.proj_B.bias)
	nn.adam_add_param(&opt, mamba.proj_C.weights)
	nn.adam_add_param(&opt, mamba.proj_C.bias)
	nn.adam_add_param(&opt, mamba.proj_Delta.weights)
	nn.adam_add_param(&opt, mamba.proj_Delta.bias)
	nn.adam_add_param(&opt, mamba.proj_out.weights)
	nn.adam_add_param(&opt, mamba.proj_out.bias)
	nn.adam_add_param(&opt, mamba.A)
	nn.adam_add_param(&opt, mamba.D)
	nn.adam_add_param(&opt, head.weights)
	nn.adam_add_param(&opt, head.bias)

	// ---- Dataset: fixed set of quasi-periodic series ----
	n_samples := batch_size * num_batches
	x_data := l.matrix_new(f64, 1, n_samples * seq_len, allocator)
	y_data := l.matrix_new(f64, 1, n_samples * seq_len, allocator)
	defer l.matrix_free(&x_data)
	defer l.matrix_free(&y_data)

	for s in 0 ..< n_samples {
		p1 := rand.float64() * 2.0 * math.PI
		p2 := rand.float64() * 2.0 * math.PI
		// Generate seq_len+1 steps: input = first seq_len, target = shifted by 1
		for tt in 0 ..< seq_len + 1 {
			val := math.sin(w1 * f64(tt) + p1) + 0.5 * math.sin(w2 * f64(tt) + p2)
			val += 0.05 * (rand.float64() * 2.0 - 1.0)
			if tt < seq_len {x_data.data[s * seq_len + tt] = val}
			if tt > 0 {y_data.data[s * seq_len + (tt - 1)] = val}
		}
	}

	// Memoryless baseline: always predict the global mean
	target_mean := 0.0
	for v in y_data.data {target_mean += v}
	target_mean /= f64(len(y_data.data))
	var_sum := 0.0
	for v in y_data.data {
		d := v - target_mean
		var_sum += d * d
	}
	baseline_mse := var_sum / f64(len(y_data.data))
	fmt.printf("Dataset: %d samples | Memoryless baseline MSE: %.4f\n", n_samples, baseline_mse)

	// ---- Training loop ----
	fmt.println("\nEpoch | Train MSE | vs Baseline")
	fmt.println("------+-----------+------------")
	final_loss := 0.0
	for epoch in 0 ..< epochs {
		epoch_loss := 0.0
		for b in 0 ..< num_batches {
			xb := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
			yb := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
			copy(xb.data, x_data.data[b * batch_size * seq_len:(b + 1) * batch_size * seq_len])
			copy(yb.data, y_data.data[b * batch_size * seq_len:(b + 1) * batch_size * seq_len])

			x := t.tensor_new(xb, false, allocator)
			x.shape = [4]int{batch_size, seq_len, 1, 1}
			y := t.tensor_new(yb, false, allocator)
			y.shape = [4]int{batch_size, seq_len, 1, 1}

			h0m := l.matrix_new(f64, 1, batch_size * d_model * d_state, allocator)
			h0 := t.tensor_new(h0m, false, allocator)
			h0.shape = [4]int{batch_size, d_model, d_state, 1}

			// Forward: embed -> SSM scan -> head
			emb := nn.linear_forward(&embed, x)
			mout := nn.mamba_layer_forward(&mamba, emb, h0)
			pred := nn.linear_forward(&head, mout)
			loss := t.tensor_mse_loss(pred, y)

			// Backward + optimizer
			t.tensor_backward(loss)
			nn.clip_grad_norm(&opt, 1.0)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)

			epoch_loss += loss.data.data[0]

			// Cleanup: graph intermediates first, then leaf tensors
			t.tensor_free_graph(loss)
			t.tensor_free(x)
			t.tensor_free(y)
			t.tensor_free(h0)
		}
		final_loss = epoch_loss / f64(num_batches)
		if epoch % 5 == 0 || epoch == epochs - 1 {
			fmt.printf(" %4d | %9.4f | %.2fx\n", epoch, final_loss, final_loss / baseline_mse)
		}
	}

	// ---- Qualitative peek: one forward pass of forecasts ----
	xb := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
	yb := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
	copy(xb.data, x_data.data[0:batch_size * seq_len])
	copy(yb.data, y_data.data[0:batch_size * seq_len])
	x := t.tensor_new(xb, false, allocator)
	x.shape = [4]int{batch_size, seq_len, 1, 1}
	y := t.tensor_new(yb, false, allocator)
	y.shape = [4]int{batch_size, seq_len, 1, 1}
	h0m := l.matrix_new(f64, 1, batch_size * d_model * d_state, allocator)
	h0 := t.tensor_new(h0m, false, allocator)
	h0.shape = [4]int{batch_size, d_model, d_state, 1}

	emb := nn.linear_forward(&embed, x)
	mout := nn.mamba_layer_forward(&mamba, emb, h0)
	pred := nn.linear_forward(&head, mout)

	fmt.println("\nSample forecasts (series 0, second half of window):")
	for tt in 8 ..< seq_len {
		fmt.printf(
			"  t=%2d  actual=%+.3f  predicted=%+.3f\n",
			tt,
			y.data.data[tt],
			pred.data.data[tt],
		)
	}

	t.tensor_free_graph(pred)
	t.tensor_free(x)
	t.tensor_free(y)
	t.tensor_free(h0)

	// ---- Verdict ----
	fmt.println("\n--- Result ---")
	fmt.printf("Memoryless baseline MSE : %.4f\n", baseline_mse)
	fmt.printf("Trained Mamba MSE       : %.4f\n", final_loss)
	if final_loss < 0.3 * baseline_mse {
		fmt.println("✅ SUCCESS: Mamba learned the temporal structure (loss << baseline).")
	} else {
		fmt.println("❌ FAILED: Mamba did not beat the memoryless baseline.")
	}
	fmt.println("\n=== Mamba Learning Test Complete ===\n")
}
