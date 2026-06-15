package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
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
