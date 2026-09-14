package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
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
lstm_volatility_real_data_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== LSTM Volatility (Real Market Data Pipeline) ===")

	// 1. Fetch Real Data (5 Years of SPY and VIX)
	fmt.println("--- Fetching Market Data ---")
	spy_df := yahoo.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&spy_df)

	vix_df := yahoo.read_yahoo("^VIX", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&vix_df)

	n_spy := spy_df.rows
	n_vix := vix_df.rows
	n_common := min(n_spy, n_vix)
	num_days := n_common - 1
	fmt.printf("Aligned %d days of SPY and VIX data\n", num_days)

	// 2. Compute Features & Targets
	num_features := 4 // [Return, AbsReturn, RV_20d, VIX_Level]
	features := make([]f64, num_days * num_features, allocator)
	targets := make([]f64, num_days, allocator)
	defer {delete(features, allocator); delete(targets, allocator)}

	returns := make([]f64, num_days, allocator)
	vix_levels := make([]f64, num_days, allocator)
	defer {delete(returns, allocator); delete(vix_levels, allocator)}

	for i in 1 ..< n_common {
		prev_close, _ := w.column_at_float(&spy_df.columns[4], i - 1)
		curr_close, _ := w.column_at_float(&spy_df.columns[4], i)
		returns[i - 1] = math.ln_f64(curr_close / prev_close)

		vix_close, _ := w.column_at_float(&vix_df.columns[4], i)
		vix_levels[i - 1] = vix_close / 100.0 // Scale VIX to ~0.20 instead of 20.0
	}

	window := 20
	for i in 0 ..< num_days {
		features[i * num_features + 0] = returns[i]
		features[i * num_features + 1] = math.abs(returns[i])

		if i < window {
			features[i * num_features + 2] = 0.0
		} else {
			sum := 0.0; sum_sq := 0.0
			for j in (i - window) ..< i {
				sum += returns[j]
				sum_sq += returns[j] * returns[j]
			}
			m := sum / f64(window)
			features[i * num_features + 2] = math.sqrt_f64((sum_sq / f64(window)) - m * m)
		}

		features[i * num_features + 3] = vix_levels[i]

		// Target: Next day's absolute return (volatility proxy)
		if i + 1 < num_days {
			targets[i] = math.abs(returns[i + 1])
		}
	}

	// 3. Train/Val Split & Standardization (Prevents Look-Ahead Bias)
	start_idx := window
	end_idx := num_days - 1
	valid_days := end_idx - start_idx

	train_ratio := 0.8
	train_days := int(f64(valid_days) * train_ratio)

	means := make([]f64, num_features, allocator)
	stds := make([]f64, num_features, allocator)
	defer {delete(means, allocator); delete(stds, allocator)}

	// Fit ONLY on training data
	for day in 0 ..< train_days {
		idx := start_idx + day
		for f in 0 ..< num_features {means[f] += features[idx * num_features + f]}
	}
	for f in 0 ..< num_features {means[f] /= f64(train_days)}

	for day in 0 ..< train_days {
		idx := start_idx + day
		for f in 0 ..< num_features {
			diff := features[idx * num_features + f] - means[f]
			stds[f] += diff * diff
		}
	}
	for f in 0 ..< num_features {
		stds[f] = math.sqrt(stds[f] / f64(train_days))
		if stds[f] < 1e-8 {stds[f] = 1.0}
	}

	// Apply to ALL valid data
	for day in 0 ..< valid_days {
		idx := start_idx + day
		for f in 0 ..< num_features {
			features[idx * num_features + f] =
				(features[idx * num_features + f] - means[f]) / stds[f]
		}
	}

	// 4. Create Sliding Window Sequences
	seq_len := 20
	num_samples := valid_days - seq_len
	X_seq := make([]f64, num_samples * seq_len * num_features, allocator)
	Y_seq := make([]f64, num_samples, allocator)
	defer {delete(X_seq, allocator); delete(Y_seq, allocator)}

	for i in 0 ..< num_samples {
		src_start := (start_idx + i) * num_features
		dst_start := i * seq_len * num_features
		copy(
			X_seq[dst_start:dst_start + seq_len * num_features],
			features[src_start:src_start + seq_len * num_features],
		)
		Y_seq[i] = targets[start_idx + i + seq_len] // Predict the day immediately following the sequence
	}

	num_train := int(f64(num_samples) * train_ratio)
	num_val := num_samples - num_train
	fmt.printf("Data split: %d Train sequences, %d Validation sequences\n", num_train, num_val)

	// 5. Initialize Model
	input_size := num_features
	hidden_size := 32
	batch_size := 32
	epochs := 50
	learning_rate := 0.001

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

	// 6. Training & Validation Loop
	for epoch in 0 ..< epochs {
		epoch_train_loss := 0.0
		epoch_val_loss := 0.0

		// --- TRAINING PHASE ---
		for b in 0 ..< num_train / batch_size {
			batch_start := b * batch_size

			x_batch_data := l.matrix_new(f64, 1, batch_size * seq_len * input_size, allocator)
			copy(
				x_batch_data.data,
				X_seq[batch_start *
				seq_len *
				input_size:(batch_start + batch_size) *
				seq_len *
				input_size],
			)
			x_batch := t.tensor_new(x_batch_data, true, allocator)
			x_batch.shape = [4]int{batch_size, seq_len, input_size, 1}

			h0_data := l.matrix_new(f64, 1, batch_size * hidden_size, allocator)
			h_0 := t.tensor_new(h0_data, false, allocator)
			c0_data := l.matrix_new(f64, 1, batch_size * hidden_size, allocator)
			c_0 := t.tensor_new(c0_data, false, allocator)

			y_batch_data := l.matrix_new(f64, batch_size, 1, allocator)
			copy(y_batch_data.data, Y_seq[batch_start:batch_start + batch_size])
			y_batch := t.tensor_new(y_batch_data, false, allocator)
			y_batch.shape = [4]int{batch_size, 1, 1, 1}

			preds := ml_fin.lstm_volatility_forecaster_forward(&forecaster, x_batch, h_0, c_0)
			loss := t.tensor_mse_loss(preds, y_batch)

			t.tensor_backward(loss, allocator)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)

			epoch_train_loss += loss.data.data[0]

			t.tensor_free_graph(loss)
			t.tensor_free(x_batch); t.tensor_free(h_0); t.tensor_free(c_0); t.tensor_free(y_batch)
		}

		// --- VALIDATION PHASE ---
		for b in 0 ..< num_val / batch_size {
			batch_start := num_train + b * batch_size

			x_batch_data := l.matrix_new(f64, 1, batch_size * seq_len * input_size, allocator)
			copy(
				x_batch_data.data,
				X_seq[batch_start *
				seq_len *
				input_size:(batch_start + batch_size) *
				seq_len *
				input_size],
			)

			// requires_grad = true ensures intermediates are added to the graph so tensor_free_graph cleans them up
			x_batch := t.tensor_new(x_batch_data, true, allocator)
			x_batch.shape = [4]int{batch_size, seq_len, input_size, 1}

			h0_data := l.matrix_new(f64, 1, batch_size * hidden_size, allocator)
			h_0 := t.tensor_new(h0_data, false, allocator)
			c0_data := l.matrix_new(f64, 1, batch_size * hidden_size, allocator)
			c_0 := t.tensor_new(c0_data, false, allocator)

			y_batch_data := l.matrix_new(f64, batch_size, 1, allocator)
			copy(y_batch_data.data, Y_seq[batch_start:batch_start + batch_size])
			y_batch := t.tensor_new(y_batch_data, false, allocator)
			y_batch.shape = [4]int{batch_size, 1, 1, 1}

			preds := ml_fin.lstm_volatility_forecaster_forward(&forecaster, x_batch, h_0, c_0)
			loss := t.tensor_mse_loss(preds, y_batch)
			epoch_val_loss += loss.data.data[0]

			// Free graph cleans up all intermediates (lstm_out, flat, h1, etc.)
			t.tensor_free_graph(loss)
			t.tensor_free(x_batch); t.tensor_free(h_0); t.tensor_free(c_0); t.tensor_free(y_batch)
		}

		if epoch % 5 == 0 {
			avg_train := epoch_train_loss / f64(num_train / batch_size)
			avg_val := epoch_val_loss / f64(num_val / batch_size)
			fmt.printf("Epoch %02d | Train MSE: %.6f | Val MSE: %.6f\n", epoch, avg_train, avg_val)
		}
	}

	fmt.println("\n✓ Real Market Data Pipeline Test Complete!")
}
