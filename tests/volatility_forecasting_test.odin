package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:mem"

volatility_forecasting_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Volatility Forecasting Test ===")

	input_size := 3
	hidden_size := 64
	seq_len := 20
	batch_size := 16
	epochs := 100
	learning_rate := 0.001

	flat_input := seq_len * input_size

	fmt.printf(
		"Training MLP (Flat Input: %d, Hidden: %d) for %d epochs...\n",
		flat_input,
		hidden_size,
		epochs,
	)

	forecaster := ml_fin.volatility_forecaster_new(flat_input, hidden_size, allocator)
	defer ml_fin.volatility_forecaster_free(&forecaster)

	opt := nn.adam_new(learning_rate, 0.9, 0.999, 1e-8, allocator)
	defer nn.adam_free(&opt)

	volatility_forecaster_add_to_optimizer(&forecaster, &opt)

	num_samples := 500
	X_data := make([]f64, num_samples * flat_input, allocator)
	Y_data := make([]f64, num_samples, allocator)
	defer {delete(X_data, allocator); delete(Y_data, allocator)}

	current_vol := 0.01
	for s in 0 ..< num_samples {
		shock := rand.float64() * 0.02
		current_vol = 0.9 * current_vol + 0.1 * shock + 0.01

		for t in 0 ..< seq_len {
			offset := s * flat_input + t * input_size
			X_data[offset + 0] = current_vol * (1.0 - f64(t) * 0.02)
			X_data[offset + 1] = current_vol * 1.5 + (rand.float64() - 0.5) * 0.01
			X_data[offset + 2] = (rand.float64() - 0.5) * 2.0
		}
		Y_data[s] = current_vol
	}

	for epoch in 0 ..< epochs {
		epoch_loss := 0.0

		for b in 0 ..< num_samples / batch_size {
			batch_start := b * batch_size

			// ✅ Framework convention: [1, batch * features]
			x_batch_data := l.matrix_new(f64, 1, batch_size * flat_input, allocator)
			copy(
				x_batch_data.data,
				X_data[batch_start * flat_input:(batch_start + batch_size) * flat_input],
			)
			x_batch := t.tensor_new(x_batch_data, true, allocator)
			x_batch.shape = [4]int{batch_size, flat_input, 1, 1}

			// ✅ Framework convention: [1, batch]
			y_batch_data := l.matrix_new(f64, 1, batch_size, allocator)
			for i in 0 ..< batch_size {
				y_batch_data.data[i] = Y_data[batch_start + i]
			}
			y_batch := t.tensor_new(y_batch_data, false, allocator)
			y_batch.shape = [4]int{batch_size, 1, 1, 1}

			preds := ml_fin.volatility_forecaster_forward(&forecaster, x_batch)
			loss := t.tensor_mse_loss(preds, y_batch)

			t.tensor_backward(loss, allocator)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)

			epoch_loss += loss.data.data[0]

			// ✅ Free graph first, then leaf nodes
			t.tensor_free_graph(loss)
			t.tensor_free(x_batch)
			t.tensor_free(y_batch)
		}

		if epoch % 10 == 0 {
			fmt.printf(
				"  Epoch %d | Loss: %.6f\n",
				epoch,
				epoch_loss / f64(num_samples / batch_size),
			)
		}
	}

	fmt.println("\n✓ Volatility Forecasting Test Complete!")
}

volatility_forecaster_add_to_optimizer :: proc(
	model: ^ml_fin.VolatilityForecaster,
	opt: ^nn.Adam,
) {
	nn.adam_add_param(opt, model.fc1.weights)
	nn.adam_add_param(opt, model.fc1.bias)
	nn.adam_add_param(opt, model.fc2.weights)
	nn.adam_add_param(opt, model.fc2.bias)
	nn.adam_add_param(opt, model.fc3.weights)
	nn.adam_add_param(opt, model.fc3.bias)
}
lstm_volatility_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== LSTM Volatility Forecasting Test ===")

	input_size := 3
	hidden_size := 32
	seq_len := 20
	batch_size := 16
	epochs := 100
	learning_rate := 0.001

	fmt.printf(
		"Training LSTM (Input: %d, Hidden: %d, Seq: %d) for %d epochs...\n",
		input_size,
		hidden_size,
		seq_len,
		epochs,
	)

	forecaster := ml_fin.lstm_volatility_forecaster_new(
		input_size,
		hidden_size,
		seq_len,
		allocator,
	)
	defer ml_fin.lstm_volatility_forecaster_free(&forecaster)

	opt := nn.adam_new(learning_rate, 0.9, 0.999, 1e-8, allocator)
	defer nn.adam_free(&opt)

	lstm_volatility_add_to_optimizer(&forecaster, &opt)

	// ----------------------------------------------------------------
	// Synthetic GARCH-like data: 3 features per timestep
	// ----------------------------------------------------------------
	num_samples := 500
	X_data := make([]f64, num_samples * seq_len * input_size, allocator)
	Y_data := make([]f64, num_samples, allocator)
	defer {delete(X_data, allocator); delete(Y_data, allocator)}

	current_vol := 0.01
	for s in 0 ..< num_samples {
		shock := rand.float64() * 0.02
		current_vol = 0.9 * current_vol + 0.1 * shock + 0.01

		for step in 0 ..< seq_len {
			offset := s * seq_len * input_size + step * input_size
			X_data[offset + 0] = current_vol * (1.0 - f64(step) * 0.02)
			X_data[offset + 1] = current_vol * 1.5 + (rand.float64() - 0.5) * 0.01
			X_data[offset + 2] = (rand.float64() - 0.5) * 2.0
		}
		Y_data[s] = current_vol
	}

	// ----------------------------------------------------------------
	// Training loop
	// ----------------------------------------------------------------
	for epoch in 0 ..< epochs {
		epoch_loss := 0.0

		for b in 0 ..< num_samples / batch_size {
			batch_start := b * batch_size

			// Input: [1, batch*seq_len*input_size], shape [batch, seq_len, input_size, 1]
			x_batch_data := l.matrix_new(f64, 1, batch_size * seq_len * input_size, allocator)
			copy(
				x_batch_data.data,
				X_data[batch_start *
				seq_len *
				input_size:(batch_start + batch_size) *
				seq_len *
				input_size],
			)
			x_batch := t.tensor_new(x_batch_data, true, allocator)
			x_batch.shape = [4]int{batch_size, seq_len, input_size, 1}

			// h_0, c_0: zeros [1, batch*hidden_size]
			h0_data := l.matrix_new(f64, 1, batch_size * hidden_size, allocator)
			h_0 := t.tensor_new(h0_data, false, allocator)
			h_0.shape = [4]int{batch_size, 1, hidden_size, 1}

			c0_data := l.matrix_new(f64, 1, batch_size * hidden_size, allocator)
			c_0 := t.tensor_new(c0_data, false, allocator)
			c_0.shape = [4]int{batch_size, 1, hidden_size, 1}

			// Target: [batch, 1], shape [batch, 1, 1, 1]
			y_batch_data := l.matrix_new(f64, batch_size, 1, allocator)
			for i in 0 ..< batch_size {
				y_batch_data.data[i] = Y_data[batch_start + i]
			}
			y_batch := t.tensor_new(y_batch_data, false, allocator)
			y_batch.shape = [4]int{batch_size, 1, 1, 1}

			// Forward → Loss → Backward → Step
			preds := ml_fin.lstm_volatility_forecaster_forward(&forecaster, x_batch, h_0, c_0)
			loss := t.tensor_mse_loss(preds, y_batch)

			t.tensor_backward(loss, allocator)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)

			epoch_loss += loss.data.data[0]

			// Free graph first, then leaf nodes
			t.tensor_free_graph(loss)
			t.tensor_free(x_batch)
			t.tensor_free(h_0)
			t.tensor_free(c_0)
			t.tensor_free(y_batch)
		}

		if epoch % 10 == 0 {
			fmt.printf(
				"  Epoch %d | Loss: %.6f\n",
				epoch,
				epoch_loss / f64(num_samples / batch_size),
			)
		}
	}

	fmt.println("\n✓ LSTM Volatility Forecasting Test Complete!")
}

lstm_volatility_add_to_optimizer :: proc(model: ^ml_fin.LSTMVolatilityForecaster, opt: ^nn.Adam) {
	nn.adam_add_param(opt, model.lstm.w_ih)
	nn.adam_add_param(opt, model.lstm.w_hh)
	nn.adam_add_param(opt, model.lstm.bias)
	nn.adam_add_param(opt, model.fc1.weights)
	nn.adam_add_param(opt, model.fc1.bias)
	nn.adam_add_param(opt, model.fc2.weights)
	nn.adam_add_param(opt, model.fc2.bias)
}
