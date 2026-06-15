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
