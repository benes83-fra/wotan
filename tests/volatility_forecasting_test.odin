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
