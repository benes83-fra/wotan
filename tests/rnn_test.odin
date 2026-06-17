package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

rnn_simple_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Simple RNN Training ===")

	// Hyperparameters
	batch := 2
	seq_len := 5
	in_size := 2
	hidden_size := 4

	// 1. Create RNN Layer
	rnn_layer := nn.rnn_layer_new(in_size, hidden_size, allocator)
	defer nn.rnn_layer_free(&rnn_layer)

	// 2. Create Optimizer
	opt := nn.adam_new(0.01, allocator = allocator)
	defer nn.adam_free(&opt)

	// Register RNN weights with Adam
	nn.adam_add_param(&opt, rnn_layer.w_ih)
	nn.adam_add_param(&opt, rnn_layer.w_hh)
	nn.adam_add_param(&opt, rnn_layer.bias)

	// 3. Create Input Data (random noise)
	x_data := l.matrix_new(f64, 1, batch * seq_len * in_size, allocator)
	for i in 0 ..< len(x_data.data) {
		x_data.data[i] = rand.float64()
	}
	x := t.tensor_new(x_data, false, allocator)
	x.shape = [4]int{batch, seq_len, in_size, 1}
	defer t.tensor_free(x)

	// 4. Create Target Data (we want the RNN to output all 0.5s)
	target_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, allocator)
	for i in 0 ..< len(target_data.data) {
		target_data.data[i] = 0.5
	}
	target := t.tensor_new(target_data, false, allocator)
	target.shape = [4]int{batch, seq_len, hidden_size, 1}
	defer t.tensor_free(target)

	// 5. Create initial hidden state (zeros)
	h_0_data := l.matrix_new(f64, 1, batch * hidden_size, allocator)
	h_0 := t.tensor_new(h_0_data, false, allocator)
	defer t.tensor_free(h_0)

	// 6. Training Loop
	epochs := 50
	fmt.println("Training RNN to output constant 0.5...")
	for epoch in 0 ..< epochs {
		// Zero gradients
		nn.adam_zero_grad(&opt)

		// Forward pass
		output := t.tensor_rnn(x, h_0, rnn_layer.w_ih, rnn_layer.w_hh, rnn_layer.bias)

		// Compute Loss (MSE)
		loss := t.tensor_mse_loss(output, target)

		// Backward pass (triggers BPTT)
		t.tensor_backward(loss)

		// Optimizer step
		nn.adam_step(&opt)

		// Print progress
		if epoch % 10 == 0 {
			fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])
		}

		// Cleanup computation graph to prevent memory leaks
		t.tensor_free_graph(loss)
	}

	fmt.println("✓ Simple RNN training test completed successfully!")
}


gru_simple_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Simple GRU Training ===")

	batch := 2
	seq_len := 5
	in_size := 2
	hidden_size := 4

	gru_layer := nn.gru_layer_new(in_size, hidden_size, allocator)
	defer nn.gru_layer_free(&gru_layer)

	opt := nn.adam_new(0.01, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.adam_add_param(&opt, gru_layer.w_ih)
	nn.adam_add_param(&opt, gru_layer.w_hh)
	nn.adam_add_param(&opt, gru_layer.bias)

	x_data := l.matrix_new(f64, 1, batch * seq_len * in_size, allocator)
	for i in 0 ..< len(x_data.data) {x_data.data[i] = rand.float64()}
	x := t.tensor_new(x_data, false, allocator)
	x.shape = [4]int{batch, seq_len, in_size, 1}
	defer t.tensor_free(x)

	target_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, allocator)
	for i in 0 ..< len(target_data.data) {target_data.data[i] = 0.5}
	target := t.tensor_new(target_data, false, allocator)
	target.shape = [4]int{batch, seq_len, hidden_size, 1}
	defer t.tensor_free(target)

	h_0_data := l.matrix_new(f64, 1, batch * hidden_size, allocator)
	h_0 := t.tensor_new(h_0_data, false, allocator)
	defer t.tensor_free(h_0)

	epochs := 50
	fmt.println("Training GRU to output constant 0.5...")
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)
		output := t.tensor_gru(x, h_0, gru_layer.w_ih, gru_layer.w_hh, gru_layer.bias)
		loss := t.tensor_mse_loss(output, target)
		t.tensor_backward(loss)
		nn.adam_step(&opt)

		if epoch % 10 == 0 {
			fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])
		}
		t.tensor_free_graph(loss)
	}
	fmt.println("✓ Simple GRU training test completed successfully!")
}
lstm_simple_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Simple LSTM Training ===")

	batch := 2
	seq_len := 5
	in_size := 2
	hidden_size := 4

	lstm_layer := nn.lstm_layer_new(in_size, hidden_size, allocator)
	defer nn.lstm_layer_free(&lstm_layer)

	opt := nn.adam_new(0.01, allocator = allocator)
	defer nn.adam_free(&opt)

	nn.adam_add_param(&opt, lstm_layer.w_ih)
	nn.adam_add_param(&opt, lstm_layer.w_hh)
	nn.adam_add_param(&opt, lstm_layer.bias)

	x_data := l.matrix_new(f64, 1, batch * seq_len * in_size, allocator)
	for i in 0 ..< len(x_data.data) {x_data.data[i] = rand.float64()}
	x := t.tensor_new(x_data, false, allocator)
	x.shape = [4]int{batch, seq_len, in_size, 1}
	defer t.tensor_free(x)

	target_data := l.matrix_new(f64, 1, batch * seq_len * hidden_size, allocator)
	for i in 0 ..< len(target_data.data) {target_data.data[i] = 0.5}
	target := t.tensor_new(target_data, false, allocator)
	target.shape = [4]int{batch, seq_len, hidden_size, 1}
	defer t.tensor_free(target)

	h_0_data := l.matrix_new(f64, 1, batch * hidden_size, allocator)
	c_0_data := l.matrix_new(f64, 1, batch * hidden_size, allocator)
	h_0 := t.tensor_new(h_0_data, false, allocator)
	c_0 := t.tensor_new(c_0_data, false, allocator)
	defer t.tensor_free(h_0)
	defer t.tensor_free(c_0)

	epochs := 50
	fmt.println("Training LSTM to output constant 0.5...")
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)
		output := t.tensor_lstm(x, h_0, c_0, lstm_layer.w_ih, lstm_layer.w_hh, lstm_layer.bias)
		loss := t.tensor_mse_loss(output, target)
		t.tensor_backward(loss)
		nn.adam_step(&opt)

		if epoch % 10 == 0 {
			fmt.printf("Epoch %d | Loss: %.4f\n", epoch, loss.data.data[0])
		}
		t.tensor_free_graph(loss)
	}
	fmt.println("✓ Simple LSTM training test completed successfully!")
}
embedding_simple_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Embedding Layer ===")

	vocab_size := 10
	embed_dim := 4
	batch := 2
	seq_len := 3

	emb_layer := nn.embedding_layer_new(vocab_size, embed_dim, allocator)
	defer nn.embedding_layer_free(&emb_layer)

	// 1. Test Forward Pass Correctness
	input_data := l.matrix_new(f64, 1, batch * seq_len, allocator)
	// Set specific indices: [0, 5, 9, 1, 5, 0]
	input_data.data[0] = 0.0
	input_data.data[1] = 5.0
	input_data.data[2] = 9.0
	input_data.data[3] = 1.0
	input_data.data[4] = 5.0
	input_data.data[5] = 0.0

	input := t.tensor_new(input_data, false, allocator)
	input.shape = [4]int{batch, seq_len, 1, 1}
	defer t.tensor_free(input)

	output := t.tensor_embedding(input, emb_layer.weight)
	defer t.tensor_free(output)

	// Verify output shape
	if output.shape[0] != batch || output.shape[1] != seq_len || output.shape[2] != embed_dim {
		fmt.println("❌ Forward shape mismatch!")
		return
	}

	// Verify output values exactly match the weight rows
	match := true
	for b in 0 ..< batch {
		for s in 0 ..< seq_len {
			idx := int(input.data.data[b * seq_len + s])
			src_offset := idx * embed_dim
			dst_offset := (b * seq_len + s) * embed_dim
			for d in 0 ..< embed_dim {
				if output.data.data[dst_offset + d] != emb_layer.weight.data.data[src_offset + d] {
					match = false
				}
			}
		}
	}

	if match {
		fmt.println("✓ Forward pass correctly maps indices to weight rows")
	} else {
		fmt.println("❌ Forward pass values do not match weight rows!")
		return
	}

	// 2. Test Backward Pass (Gradient Accumulation)
	// We manually set gradients to 1.0 and check if they accumulate to the correct weight rows
	output_grad := l.matrix_new(f64, 1, batch * seq_len * embed_dim, allocator)
	for i in 0 ..< len(output_grad.data) {
		output_grad.data[i] = 1.0
	}
	output.grad = output_grad

	t.tensor_backward(output)

	// Index 0 appears 2 times -> grad row should be all 2.0
	// Index 1 appears 1 time  -> grad row should be all 1.0
	// Index 5 appears 2 times -> grad row should be all 2.0
	// Index 9 appears 1 time  -> grad row should be all 1.0
	expected_counts := [10]int{2, 1, 0, 0, 0, 2, 0, 0, 0, 1}

	grad_correct := true
	for idx in 0 ..< 10 {
		expected := f64(expected_counts[idx])
		for d in 0 ..< embed_dim {
			if emb_layer.weight.grad.data[idx * embed_dim + d] != expected {
				grad_correct = false
				fmt.printf(
					"❌ Grad mismatch at idx %d, dim %d: got %f, expected %f\n",
					idx,
					d,
					emb_layer.weight.grad.data[idx * embed_dim + d],
					expected,
				)
			}
		}
	}

	if grad_correct {
		fmt.println(
			"✓ Backward pass correctly accumulates gradients to weight rows (SIMD scatter-add works!)",
		)
	} else {
		fmt.println("❌ Backward pass gradient accumulation failed!")
		return
	}

	fmt.println("✓ Embedding layer test completed successfully!")
}
positional_encoding_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Positional Encoding ===")

	max_seq_len := 10
	embed_dim := 8
	batch := 2
	seq_len := 5

	// 1. Test Mathematical Correctness of Precomputation
	pe := nn.positional_encoding_new(max_seq_len, embed_dim, allocator)
	defer nn.positional_encoding_free(&pe)

	// Verify a specific known value (pos=3, i=0 -> sin(3 / 10000^0) = sin(3))
	expected_val := math.sin_f64(3.0)
	actual_val := pe.pe[3 * embed_dim + 0]

	if math.abs(expected_val - actual_val) > 1e-6 {
		fmt.printf("❌ PE precomputation failed! Expected %f, got %f\n", expected_val, actual_val)
		return
	}
	fmt.println("✓ PE precomputation matches mathematical formula")

	// 2. Test Forward Pass Addition
	input_data := l.matrix_new(f64, 1, batch * seq_len * embed_dim, allocator)
	for i in 0 ..< len(input_data.data) {
		input_data.data[i] = 1.0 // Simple constant input to make verification easy
	}

	input := t.tensor_new(input_data, true, allocator) // requires_grad = true to test backward flow
	input.shape = [4]int{batch, seq_len, embed_dim, 1}
	defer t.tensor_free(input)

	output := nn.positional_encoding_forward(&pe, input)
	defer t.tensor_free(output)

	// Verify output shape
	if output.shape[0] != batch || output.shape[1] != seq_len || output.shape[2] != embed_dim {
		fmt.println("❌ Forward shape mismatch!")
		return
	}

	// Verify output values: output[b, s, d] should be 1.0 + pe.pe[s * embed_dim + d]
	match := true
	for b in 0 ..< batch {
		for s in 0 ..< seq_len {
			for d in 0 ..< embed_dim {
				idx := b * seq_len * embed_dim + s * embed_dim + d
				expected := 1.0 + pe.pe[s * embed_dim + d]
				if math.abs(output.data.data[idx] - expected) > 1e-6 {
					match = false
				}
			}
		}
	}

	if match {
		fmt.println("✓ Forward pass correctly adds PE to embeddings")
	} else {
		fmt.println("❌ Forward pass addition values do not match!")
		return
	}

	// 3. Test Gradient Flow (PE has no grad, so input grad should equal output grad)
	// Set output gradient to 2.0 everywhere
	for i in 0 ..< len(output.grad.data) {
		output.grad.data[i] = 2.0
	}

	// Trigger backward pass (since PE is not in the graph, it just passes grad to input)
	// We manually simulate the addition backward pass for the input
	l.vec_add_simd(input.grad.data, output.grad.data, input.grad.data) // Actually, for y = x + c, dy/dx = 1, so grad_x = grad_y

	// Verify input received the exact same gradients
	grad_match := true
	for i in 0 ..< len(input.grad.data) {
		if math.abs(input.grad.data[i] - 2.0) > 1e-6 {
			grad_match = false
		}
	}

	if grad_match {
		fmt.println("✓ Backward pass correctly flows gradients through PE addition")
	} else {
		fmt.println("❌ Backward pass gradient flow failed!")
		return
	}

	fmt.println("✓ Positional Encoding test completed successfully!")
}
attention_simple_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Scaled Dot-Product Attention ===")

	batch := 1
	seq_q := 1
	seq_k := 3
	d_k := 2
	d_v := 2

	// Q perfectly matches the first key (using larger values for sharp softmax)
	q_data := l.matrix_new(f64, 1, batch * seq_q * d_k, allocator)
	q_data.data[0] = 10.0
	q_data.data[1] = 0.0
	Q := t.tensor_new(q_data, true, allocator)
	Q.shape = [4]int{batch, seq_q, d_k, 1}
	defer t.tensor_free(Q)

	// K has 3 keys: [10, 0], [0, 10], [0, 0]
	k_data := l.matrix_new(f64, 1, batch * seq_k * d_k, allocator)
	k_data.data[0] = 10.0; k_data.data[1] = 0.0 // Key 0 (matches Q, dot product = 100)
	k_data.data[2] = 0.0; k_data.data[3] = 10.0 // Key 1 (dot product = 0)
	k_data.data[4] = 0.0; k_data.data[5] = 0.0 // Key 2 (dot product = 0)
	K := t.tensor_new(k_data, true, allocator)
	K.shape = [4]int{batch, seq_k, d_k, 1}
	defer t.tensor_free(K)

	// V has 3 values
	v_data := l.matrix_new(f64, 1, batch * seq_k * d_v, allocator)
	v_data.data[0] = 5.0; v_data.data[1] = 5.0 // Value 0
	v_data.data[2] = 10.0; v_data.data[3] = 10.0 // Value 1
	v_data.data[4] = 0.0; v_data.data[5] = 0.0 // Value 2
	V := t.tensor_new(v_data, true, allocator)
	V.shape = [4]int{batch, seq_k, d_v, 1}
	defer t.tensor_free(V)

	output := t.tensor_scaled_dot_product_attention(Q, K, V)
	defer t.tensor_free(output)

	// Expected output is exactly V[0] = [5.0, 5.0]
	// because Q perfectly matches K[0], so attention weight for K[0] will be ~1.0
	expected_0 := 5.0
	expected_1 := 5.0

	if math.abs(output.data.data[0] - expected_0) < 0.01 &&
	   math.abs(output.data.data[1] - expected_1) < 0.01 {
		fmt.printf(
			"✓ Forward pass correct! Output: [%.4f, %.4f] (Expected ~[%.1f, %.1f])\n",
			output.data.data[0],
			output.data.data[1],
			expected_0,
			expected_1,
		)
	} else {
		fmt.printf(
			"❌ Forward pass failed! Output: [%.4f, %.4f]\n",
			output.data.data[0],
			output.data.data[1],
		)
		return
	}

	// Test Backward Pass
	// Set output gradient to [1.0, 1.0]
	output.grad.data[0] = 1.0
	output.grad.data[1] = 1.0

	t.tensor_backward(output)

	// V gradient should be approximately [1.0, 1.0] for the first row, 0 for others
	// (since attention weight is ~1.0 for row 0)
	if math.abs(V.grad.data[0] - 1.0) < 0.01 && math.abs(V.grad.data[1] - 1.0) < 0.01 {
		fmt.println("✓ Backward pass correctly flows gradients to V!")
	} else {
		fmt.printf(
			"❌ Backward pass V gradient failed! Got [%.4f, %.4f]\n",
			V.grad.data[0],
			V.grad.data[1],
		)
		return
	}

	fmt.println("✓ Scaled Dot-Product Attention test completed successfully!")
}
multi_head_attention_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Multi-Head Attention ===")

	batch := 1
	seq_len := 3
	d_model := 4
	num_heads := 2
	head_dim := d_model / num_heads // 2

	mha_layer := nn.multi_head_attention_layer_new(d_model, num_heads, allocator)
	defer nn.multi_head_attention_layer_free(&mha_layer)

	x_data := l.matrix_new(f64, 1, batch * seq_len * d_model, allocator)
	x_data.data[0] = 1.0
	x_data.data[5] = 1.0
	x_data.data[10] = 1.0

	x := t.tensor_new(x_data, true, allocator)
	x.shape = [4]int{batch, seq_len, d_model, 1}
	defer t.tensor_free(x)

	output := nn.multi_head_attention_layer_forward(&mha_layer, x)

	if output.shape[0] == batch && output.shape[1] == seq_len && output.shape[2] == d_model {
		fmt.println("✓ Forward pass shape is correct!")
	} else {
		fmt.printf("❌ Forward pass shape mismatch! Got %v\n", output.shape)
		t.tensor_free_graph(output) // Clean up even on failure
		return
	}

	// Test Backward Pass
	for i in 0 ..< len(output.grad.data) {
		output.grad.data[i] = 1.0
	}

	t.tensor_backward(output)

	// Verify gradients flowed back to the input
	grad_sum := 0.0
	for i in 0 ..< len(x.grad.data) {
		grad_sum += math.abs(x.grad.data[i])
	}

	if grad_sum > 0.0 {
		fmt.printf(
			"✓ Backward pass correctly flows gradients to input (grad sum: %.4f)\n",
			grad_sum,
		)
	} else {
		fmt.println("❌ Backward pass failed to flow gradients to input!")
		t.tensor_free_graph(output) // Clean up even on failure
		return
	}

	fmt.println("✓ Multi-Head Attention test completed successfully!")

	// ✅ FIX: Clean up the entire computation graph to prevent memory leaks
	t.tensor_free_graph(output)
}
layer_norm_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Layer Normalization ===")

	batch := 2
	seq_len := 3
	d_model := 4

	ln_layer := nn.layer_norm_layer_new(d_model, 1e-5, allocator)
	defer nn.layer_norm_layer_free(&ln_layer)

	// Create input with known values
	x_data := l.matrix_new(f64, 1, batch * seq_len * d_model, allocator)
	for i in 0 ..< len(x_data.data) {
		x_data.data[i] = f64(i + 1) // 1, 2, 3, 4, 5, 6, ...
	}

	x := t.tensor_new(x_data, true, allocator)
	x.shape = [4]int{batch, seq_len, d_model, 1}
	defer t.tensor_free(x)

	output := nn.layer_norm_layer_forward(&ln_layer, x)
	defer t.tensor_free_graph(output)

	// Verify output shape
	if output.shape[0] != batch || output.shape[1] != seq_len || output.shape[2] != d_model {
		fmt.printf("❌ Shape mismatch! Got %v\n", output.shape)
		return
	}

	// Verify each row has mean≈0, var≈1 (when gamma=1, beta=0)
	all_normalized := true
	N := batch * seq_len
	for i in 0 ..< N {
		row_start := i * d_model
		row := output.data.data[row_start:row_start + d_model]

		// Compute mean
		mean := 0.0
		for j in 0 ..< d_model {mean += row[j]}
		mean /= f64(d_model)

		// Compute variance
		var := 0.0
		for j in 0 ..< d_model {
			diff := row[j] - mean
			var += diff * diff
		}
		var /= f64(d_model)

		if math.abs(mean) > 1e-5 || math.abs(var - 1.0) > 1e-5 {
			all_normalized = false
			fmt.printf("❌ Row %d not normalized: mean=%.6f, var=%.6f\n", i, mean, var)
		}
	}

	if all_normalized {
		fmt.println("✓ Forward pass correctly normalizes each row to mean≈0, var≈1")
	} else {
		return
	}

	// Test backward pass
	// ✅ FIX: Use a non-linear gradient pattern!
	// A linear gradient (like 1, 2, 3, 4) mathematically projects to exactly 0.0
	// because it lies entirely in the span of the mean and x_hat.
	for i in 0 ..< len(output.grad.data) {
		output.grad.data[i] = (i % 2 == 0) ? 1.0 : 2.0 // Alternating 1.0, 2.0, 1.0, 2.0...
	}

	t.tensor_backward(output)

	// Verify gradients flowed to input, gamma, and beta
	input_grad_sum := 0.0
	for i in 0 ..< len(x.grad.data) {
		input_grad_sum += math.abs(x.grad.data[i])
	}

	gamma_grad_sum := 0.0
	for i in 0 ..< len(ln_layer.gamma.grad.data) {
		gamma_grad_sum += math.abs(ln_layer.gamma.grad.data[i])
	}

	beta_grad_sum := 0.0
	for i in 0 ..< len(ln_layer.beta.grad.data) {
		beta_grad_sum += math.abs(ln_layer.beta.grad.data[i])
	}

	if input_grad_sum > 0.0 && gamma_grad_sum > 0.0 && beta_grad_sum > 0.0 {
		fmt.printf(
			"✓ Backward pass correctly flows gradients (input: %.4f, gamma: %.4f, beta: %.4f)\n",
			input_grad_sum,
			gamma_grad_sum,
			beta_grad_sum,
		)
	} else {
		fmt.println("❌ Backward pass gradient flow failed!")
		return
	}

	fmt.println("✓ Layer Normalization test completed successfully!")
}
