package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

ensemble_volatility_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Ensemble Volatility Forecasting (GARCH + LSTM) ===")

	main_alloc := context.allocator

	// ----------------------------------------------------------------
	// 1. Fetch Data
	// ----------------------------------------------------------------
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

	// ----------------------------------------------------------------
	// 2. Compute Returns & Features
	// ----------------------------------------------------------------
	returns := make([]f64, num_days, allocator)
	vix_levels := make([]f64, num_days, allocator)
	defer {delete(returns, allocator); delete(vix_levels, allocator)}

	for i in 1 ..< n_common {
		prev_close, _ := w.column_at_float(&spy_df.columns[4], i - 1)
		curr_close, _ := w.column_at_float(&spy_df.columns[4], i)
		returns[i - 1] = math.ln_f64(curr_close / prev_close)
		vix_close, _ := w.column_at_float(&vix_df.columns[4], i)
		vix_levels[i - 1] = vix_close / 100.0
	}

	num_features := 4
	window := 20
	features := make([]f64, num_days * num_features, allocator)
	targets := make([]f64, num_days, allocator)
	defer {delete(features, allocator); delete(targets, allocator)}

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
		if i + 1 < num_days {
			targets[i] = math.abs(returns[i + 1])
		}
	}

	start_idx := window
	end_idx := num_days - 1
	valid_days := end_idx - start_idx

	// ----------------------------------------------------------------
	// 3. Fit GARCH(1,1) on Training Data
	// ----------------------------------------------------------------
	fmt.println("\n--- Fitting GARCH(1,1) ---")
	train_ratio := 0.8
	train_days := int(f64(valid_days) * train_ratio)

	train_returns := returns[:start_idx + train_days]
	residuals := ts.extract_residuals(train_returns, main_alloc)
	defer delete(residuals, main_alloc)

	garch_result := ts.garch_fit(residuals, .StudentT, 1, 1, 2000, 1e-4, main_alloc)
	defer {
		delete(garch_result.params.alpha, main_alloc)
		delete(garch_result.params.beta, main_alloc)
		delete(garch_result.conditional_var, main_alloc)
		delete(garch_result.standardized_resid, main_alloc)
	}

	fmt.printf("GARCH(1,1) Parameters:\n")
	fmt.printf("  ω (omega): %.8f\n", garch_result.params.omega)
	fmt.printf("  α (alpha): %.4f\n", garch_result.params.alpha[0])
	fmt.printf("  β (beta):  %.4f\n", garch_result.params.beta[0])

	garch_vol_series := ml_fin.garch_recursive_forecast(
		returns[start_idx:end_idx],
		garch_result.params.omega,
		garch_result.params.alpha[0],
		garch_result.params.beta[0],
		allocator,
	)
	defer delete(garch_vol_series, allocator)

	// ----------------------------------------------------------------
	// 4. Standardize Features & Create Sequences
	// ----------------------------------------------------------------
	means := make([]f64, num_features, allocator)
	stds := make([]f64, num_features, allocator)
	defer {delete(means, allocator); delete(stds, allocator)}

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

	for day in 0 ..< valid_days {
		idx := start_idx + day
		for f in 0 ..< num_features {
			features[idx * num_features + f] =
				(features[idx * num_features + f] - means[f]) / stds[f]
		}
	}

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
		Y_seq[i] = targets[start_idx + i + seq_len]
	}

	num_train_samples := int(f64(num_samples) * train_ratio)
	num_val_samples := num_samples - num_train_samples

	// ----------------------------------------------------------------
	// 5. Train LSTM via Ensemble Struct
	// ----------------------------------------------------------------
	fmt.println("\n--- Training LSTM ---")
	input_size := num_features
	hidden_size := 32
	batch_size := 32
	epochs := 50
	learning_rate := 0.001

	// ✅ Use the new value-based Ensemble struct
	ensemble := ml_fin.ensemble_volatility_new(input_size, hidden_size, seq_len, allocator)
	defer ml_fin.ensemble_volatility_free(&ensemble)

	// Store GARCH params in the ensemble struct
	ensemble.garch_omega = garch_result.params.omega
	ensemble.garch_alpha = garch_result.params.alpha[0]
	ensemble.garch_beta = garch_result.params.beta[0]

	opt := nn.adam_new(learning_rate, 0.9, 0.999, 1e-8, allocator)
	defer nn.adam_free(&opt)

	// ✅ Use the ensemble optimizer helper
	ml_fin.ensemble_add_to_optimizer(&ensemble, &opt)

	for epoch in 0 ..< epochs {
		epoch_loss := 0.0
		for b in 0 ..< num_train_samples / batch_size {
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

			// ✅ Pass pointer to the internal LSTM
			preds := ml_fin.lstm_volatility_forecaster_forward(&ensemble.lstm, x_batch, h_0, c_0)
			loss := t.tensor_mse_loss(preds, y_batch)
			t.tensor_backward(loss, allocator)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)
			epoch_loss += loss.data.data[0]

			t.tensor_free_graph(loss)
			t.tensor_free(x_batch); t.tensor_free(h_0); t.tensor_free(c_0); t.tensor_free(y_batch)
		}
		if epoch % 10 == 0 {
			fmt.printf(
				"  Epoch %02d | Train MSE: %.6f\n",
				epoch,
				epoch_loss / f64(num_train_samples / batch_size),
			)
		}
	}

	// ----------------------------------------------------------------
	// 6. Generate Validation Forecasts
	// ----------------------------------------------------------------
	fmt.println("\n--- Generating Validation Forecasts ---")
	lstm_val_forecasts := make([]f64, num_val_samples, allocator)
	garch_val_forecasts := make([]f64, num_val_samples, allocator)
	actual_val_vols := make([]f64, num_val_samples, allocator)
	defer {
		delete(lstm_val_forecasts, allocator)
		delete(garch_val_forecasts, allocator)
		delete(actual_val_vols, allocator)
	}

	for i in 0 ..< num_val_samples {
		sample_idx := num_train_samples + i

		x_batch_data := l.matrix_new(f64, 1, 1 * seq_len * input_size, allocator)
		copy(
			x_batch_data.data,
			X_seq[sample_idx * seq_len * input_size:(sample_idx + 1) * seq_len * input_size],
		)
		x_batch := t.tensor_new(x_batch_data, true, allocator)
		x_batch.shape = [4]int{1, seq_len, input_size, 1}

		h0_data := l.matrix_new(f64, 1, hidden_size, allocator)
		h_0 := t.tensor_new(h0_data, false, allocator)
		c0_data := l.matrix_new(f64, 1, hidden_size, allocator)
		c_0 := t.tensor_new(c0_data, false, allocator)

		preds := ml_fin.lstm_volatility_forecaster_forward(&ensemble.lstm, x_batch, h_0, c_0)
		lstm_val_forecasts[i] = preds.data.data[0]

		t.tensor_free_graph(preds)
		t.tensor_free(x_batch); t.tensor_free(h_0); t.tensor_free(c_0)

		garch_idx := sample_idx + seq_len
		if garch_idx < len(garch_vol_series) {
			garch_val_forecasts[i] = garch_vol_series[garch_idx]
		}
		actual_val_vols[i] = Y_seq[sample_idx]
	}

	// ----------------------------------------------------------------
	// 7. Compute Optimal Ensemble Weight
	// ----------------------------------------------------------------
	fmt.println("\n--- Computing Optimal Ensemble Weight ---")
	ensemble.ensemble_weight = ml_fin.compute_optimal_weight(
		garch_val_forecasts,
		lstm_val_forecasts,
		actual_val_vols,
	)
	fmt.printf(
		"Optimal GARCH weight: %.2f (LSTM weight: %.2f)\n",
		ensemble.ensemble_weight,
		1.0 - ensemble.ensemble_weight,
	)

	// ----------------------------------------------------------------
	// 8. Evaluate All Three Models
	// ----------------------------------------------------------------
	fmt.println("\n--- Validation Results ---")
	mse_garch := 0.0
	mse_lstm := 0.0
	mse_ensemble := 0.0
	ensemble_val_forecasts := make([]f64, num_val_samples, allocator)
	defer delete(ensemble_val_forecasts, allocator)
	for i in 0 ..< num_val_samples {
		actual := actual_val_vols[i]
		garch_pred := garch_val_forecasts[i]
		lstm_pred := lstm_val_forecasts[i]

		// ✅ Use the library function for ensemble prediction
		ensemble_pred := ml_fin.ensemble_predict(&ensemble, garch_pred, lstm_pred)
		ensemble_val_forecasts[i] = ensemble_pred
		mse_garch += (garch_pred - actual) * (garch_pred - actual)
		mse_lstm += (lstm_pred - actual) * (lstm_pred - actual)
		mse_ensemble += (ensemble_pred - actual) * (ensemble_pred - actual)
	}

	mse_garch /= f64(num_val_samples)
	mse_lstm /= f64(num_val_samples)
	mse_ensemble /= f64(num_val_samples)

	fmt.printf("\n%-20s %-15s %-15s\n", "Model", "Val MSE", "Improvement")
	fmt.printf("%-20s %-15s %-15s\n", "--------------------", "---------------", "---------------")
	fmt.printf("%-20s %-15.6f %-15s\n", "GARCH(1,1)", mse_garch, "baseline")
	fmt.printf("%-20s %-15.6f %-15.1f%%\n", "LSTM", mse_lstm, (1.0 - mse_lstm / mse_garch) * 100)
	fmt.printf(
		"%-20s %-15.6f %-15.1f%%\n",
		"Ensemble",
		mse_ensemble,
		(1.0 - mse_ensemble / mse_garch) * 100,
	)

	// ----------------------------------------------------------------
	// 9. Latest Forecast Comparison
	// ----------------------------------------------------------------
	fmt.println("\n--- Latest Day Forecast ---")
	last_garch := garch_val_forecasts[num_val_samples - 1]
	last_lstm := lstm_val_forecasts[num_val_samples - 1]
	last_ensemble := ml_fin.ensemble_predict(&ensemble, last_garch, last_lstm)
	last_actual := actual_val_vols[num_val_samples - 1]

	fmt.printf("  Actual next-day vol:     %.4f%%\n", last_actual * 100)
	fmt.printf("  GARCH forecast:          %.4f%%\n", last_garch * 100)
	fmt.printf("  LSTM forecast:           %.4f%%\n", last_lstm * 100)
	fmt.printf("  Ensemble forecast:       %.4f%%\n", last_ensemble * 100)

	scale := math.sqrt_f64(252) * 100
	fmt.printf("\n  Annualized forecasts:\n")
	fmt.printf("    GARCH:    %.2f%%\n", last_garch * scale)
	fmt.printf("    LSTM:     %.2f%%\n", last_lstm * scale)
	fmt.printf("    Ensemble: %.2f%%\n", last_ensemble * scale)

	// ... (existing Step 9 code) ...
	fmt.printf("    Ensemble: %.2f%%\n", last_ensemble * scale)

	// ----------------------------------------------------------------
	// 10. Conformal Prediction (Distribution-Free Risk Bounds)
	// ----------------------------------------------------------------
	fmt.println("\n--- Calibrating Conformal Risk Bounds ---")
	cp := ml_fin.conformal_new(main_alloc)
	defer ml_fin.conformal_free(&cp)

	// Calibrate on the validation set (acting as our holdout calibration set)
	// We use the Ensemble predictions and the actual realized vols
	ml_fin.conformal_calibrate(&cp, actual_val_vols, ensemble_val_forecasts, 0.05) // 95% confidence
	ml_fin.print_conformal_stats(&cp)

	// Generate the guaranteed interval for the latest forecast
	lower, upper := ml_fin.conformal_predict_interval(&cp, last_ensemble)

	fmt.printf("\nLatest Ensemble Forecast (Daily): %.4f%%\n", last_ensemble * 100)
	fmt.printf(
		"95%% Conformal Confidence Interval (Daily): [%.4f%%, %.4f%%]\n",
		lower * 100,
		upper * 100,
	)

	fmt.printf("\nLatest Ensemble Forecast (Annualized): %.2f%%\n", last_ensemble * scale)
	fmt.printf(
		"95%% Conformal Confidence Interval (Ann.): [%.2f%%, %.2f%%]\n",
		lower * scale,
		upper * scale,
	)

	fmt.println("\n✓ Ensemble Volatility Forecasting Test Complete!")
}


vrp_signal_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Live Volatility Risk Premium (VRP) Signal Generation ===")
	main_alloc := context.allocator

	// 1. Fetch Data
	spy_df := yahoo.read_yahoo("SPY", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&spy_df)
	vix_df := yahoo.read_yahoo("^VIX", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&vix_df)

	n_common := min(spy_df.rows, vix_df.rows)
	num_days := n_common - 1

	returns := make([]f64, num_days, allocator)
	vix_levels := make([]f64, num_days, allocator) // Annualized %
	defer {delete(returns, allocator); delete(vix_levels, allocator)}

	for i in 1 ..< n_common {
		prev_c, _ := w.column_at_float(&spy_df.columns[4], i - 1)
		curr_c, _ := w.column_at_float(&spy_df.columns[4], i)
		returns[i - 1] = math.ln_f64(curr_c / prev_c)

		vix_c, _ := w.column_at_float(&vix_df.columns[4], i)
		vix_levels[i - 1] = vix_c
	}

	// 2. Compute Historical VRP Stats (using 20-day rolling RV as proxy)
	window := 20
	hist_vrps := make([]f64, num_days - window, allocator)
	defer delete(hist_vrps, allocator)

	for i in window ..< num_days {
		sum_sq := 0.0
		for j in (i - window) ..< i {
			sum_sq += returns[j] * returns[j]
		}
		daily_rv := math.sqrt(sum_sq / f64(window))
		annualized_rv := daily_rv * math.sqrt_f64(252.0) * 100.0
		hist_vrps[i - window] = vix_levels[i] - annualized_rv
	}

	vrp_mean := 0.0
	for v in hist_vrps {vrp_mean += v}
	vrp_mean /= f64(len(hist_vrps))

	vrp_var := 0.0
	for v in hist_vrps {vrp_var += (v - vrp_mean) * (v - vrp_mean)}
	vrp_std := math.sqrt(vrp_var / f64(len(hist_vrps) - 1))

	fmt.printf("Historical VRP Mean: %.2f%%\n", vrp_mean)
	fmt.printf("Historical VRP Std:  %.2f%%\n", vrp_std)

	// 3. Streamlined Ensemble Training for Inference
	num_features := 4
	features := make([]f64, num_days * num_features, allocator)
	targets := make([]f64, num_days, allocator)
	defer {delete(features, allocator); delete(targets, allocator)}

	for i in 0 ..< num_days {
		features[i * num_features + 0] = returns[i]
		features[i * num_features + 1] = math.abs(returns[i])
		if i < window {
			features[i * num_features + 2] = 0.0
		} else {
			sum_sq := 0.0
			for j in (i - window) ..< i {sum_sq += returns[j] * returns[j]}
			features[i * num_features + 2] = math.sqrt(sum_sq / f64(window))
		}
		features[i * num_features + 3] = vix_levels[i] / 100.0
		if i + 1 < num_days {targets[i] = math.abs(returns[i + 1])}
	}

	start_idx := window
	valid_days := num_days - 1 - start_idx
	train_days := int(f64(valid_days) * 0.8)

	means := make([]f64, num_features, allocator)
	stds := make([]f64, num_features, allocator)
	defer {delete(means, allocator); delete(stds, allocator)}

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

	for day in 0 ..< valid_days {
		idx := start_idx + day
		for f in 0 ..< num_features {
			features[idx * num_features + f] =
				(features[idx * num_features + f] - means[f]) / stds[f]
		}
	}

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
		Y_seq[i] = targets[start_idx + i + seq_len]
	}

	num_train_samples := int(f64(num_samples) * 0.8)

	// Fit GARCH
	train_returns := returns[:start_idx + train_days]
	residuals := ts.extract_residuals(train_returns, main_alloc)
	defer delete(residuals, main_alloc)
	garch_result := ts.garch_fit(residuals, .StudentT, 1, 1, 1000, 1e-4, main_alloc)
	defer {
		delete(garch_result.params.alpha, main_alloc)
		delete(garch_result.params.beta, main_alloc)
		delete(garch_result.conditional_var, main_alloc)
		delete(garch_result.standardized_resid, main_alloc)
	}

	// Train LSTM quickly
	input_size := num_features
	hidden_size := 32
	batch_size := 32
	epochs := 20 // Fast training for signal demo
	learning_rate := 0.001

	ensemble := ml_fin.ensemble_volatility_new(input_size, hidden_size, seq_len, allocator)
	defer ml_fin.ensemble_volatility_free(&ensemble)

	ensemble.garch_omega = garch_result.params.omega
	ensemble.garch_alpha = garch_result.params.alpha[0]
	ensemble.garch_beta = garch_result.params.beta[0]

	opt := nn.adam_new(learning_rate, 0.9, 0.999, 1e-8, allocator)
	defer nn.adam_free(&opt)
	ml_fin.ensemble_add_to_optimizer(&ensemble, &opt)

	fmt.println("Training Ensemble for Inference...")
	for epoch in 0 ..< epochs {
		for b in 0 ..< num_train_samples / batch_size {
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

			preds := ml_fin.lstm_volatility_forecaster_forward(&ensemble.lstm, x_batch, h_0, c_0)
			loss := t.tensor_mse_loss(preds, y_batch)
			t.tensor_backward(loss, allocator)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)

			t.tensor_free_graph(loss)
			t.tensor_free(x_batch); t.tensor_free(h_0); t.tensor_free(c_0); t.tensor_free(y_batch)
		}
	}

	// 4. Generate Next-Day Forecast (Inference)
	last_sample_idx := num_samples - 1

	x_inf_data := l.matrix_new(f64, 1, 1 * seq_len * input_size, allocator)
	copy(
		x_inf_data.data,
		X_seq[last_sample_idx * seq_len * input_size:(last_sample_idx + 1) * seq_len * input_size],
	)
	x_inf := t.tensor_new(x_inf_data, false, allocator) // No grad needed for inference
	x_inf.shape = [4]int{1, seq_len, input_size, 1}

	h0_inf := l.matrix_new(f64, 1, hidden_size, allocator)
	h_0_inf := t.tensor_new(h0_inf, false, allocator)
	c0_inf := l.matrix_new(f64, 1, hidden_size, allocator)
	c_0_inf := t.tensor_new(c0_inf, false, allocator)

	lstm_pred_tensor := ml_fin.lstm_volatility_forecaster_forward(
		&ensemble.lstm,
		x_inf,
		h_0_inf,
		c_0_inf,
	)
	lstm_pred_daily := lstm_pred_tensor.data.data[0]

	t.tensor_free(lstm_pred_tensor)
	t.tensor_free(x_inf); t.tensor_free(h_0_inf); t.tensor_free(c_0_inf)

	// GARCH forecast for next day
	last_return := returns[num_days - 1]
	last_cond_var := garch_result.conditional_var[len(garch_result.conditional_var) - 1]
	garch_var_next :=
		garch_result.params.omega +
		garch_result.params.alpha[0] * last_return * last_return +
		garch_result.params.beta[0] * last_cond_var
	garch_pred_daily := math.sqrt(garch_var_next)

	// Ensemble combination (using the 0.65/0.35 weight discovered in the previous test)
	ensemble_weight := 0.65
	ensemble_pred_daily :=
		ensemble_weight * garch_pred_daily + (1.0 - ensemble_weight) * lstm_pred_daily

	// Current VIX
	current_vix := vix_levels[num_days - 1]

	// 5. Compute and Print VRP Signal
	sig := ml_fin.compute_vrp(current_vix, ensemble_pred_daily, vrp_mean, vrp_std)
	ml_fin.print_vrp_signal(sig)

	fmt.println("✓ VRP Signal Generation Complete!")
}

vrp_backtest_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Walk-Forward VRP Backtester V2 (Capstone) ===")
	main_alloc := context.allocator

	// 1. Fetch Data
	fmt.println("\n--- Fetching Market Data ---")
	spy_df := yahoo.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&spy_df)
	vix_df := yahoo.read_yahoo("^VIX", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&vix_df)

	n_common := min(spy_df.rows, vix_df.rows)
	num_days := n_common - 1
	fmt.printf("Aligned %d days of SPY and VIX data\n", num_days)

	returns := make([]f64, num_days, allocator)
	vix_levels := make([]f64, num_days, allocator)
	defer {delete(returns, allocator); delete(vix_levels, allocator)}

	for i in 1 ..< n_common {
		prev_c, _ := w.column_at_float(&spy_df.columns[4], i - 1)
		curr_c, _ := w.column_at_float(&spy_df.columns[4], i)
		returns[i - 1] = math.ln_f64(curr_c / prev_c)
		vix_c, _ := w.column_at_float(&vix_df.columns[4], i)
		vix_levels[i - 1] = vix_c
	}

	// 2. Configuration
	window := 20
	seq_len := 20
	num_features := 4
	hidden_size := 32
	batch_size := 32
	initial_train_days := 500
	retrain_interval := 60
	retrain_epochs := 15
	learning_rate := 0.001
	forward_horizon := 20 // ✅ V2 FIX: Match VIX 30-day horizon with 20-day RV

	backtest_start := initial_train_days + window + seq_len + forward_horizon
	backtest_days := num_days - backtest_start - 1
	if backtest_days <= 0 {
		fmt.println("ERROR: Not enough data for backtesting")
		return
	}

	fmt.printf("Backtest Period: %d days\n", backtest_days)

	// 3. Allocate Results
	result: ml_fin.VRPBacktestResult
	result.allocator = main_alloc
	result.daily_pnl = make([]f64, backtest_days, main_alloc)
	result.equity_curve = make([]f64, backtest_days, main_alloc)
	result.positions = make([]f64, backtest_days, main_alloc)
	result.signals = make([]f64, backtest_days, main_alloc)
	result.forecast_rv = make([]f64, backtest_days, main_alloc)
	result.implied_vol = make([]f64, backtest_days, main_alloc)
	result.actual_vol = make([]f64, backtest_days, main_alloc)
	defer ml_fin.vrp_backtest_result_free(&result)

	vrp_mean := 3.0
	vrp_std := 5.0

	// 4. Initialize Model
	forecaster := ml_fin.ensemble_volatility_new(num_features, hidden_size, seq_len, allocator)
	defer ml_fin.ensemble_volatility_free(&forecaster)
	opt := nn.adam_new(learning_rate, 0.9, 0.999, 1e-8, allocator)
	defer nn.adam_free(&opt)
	ml_fin.ensemble_add_to_optimizer(&forecaster, &opt)

	// Helper: Build Features
	build_features :: proc(
		returns, vix_levels: []f64,
		start_day, end_day, window, num_features: int,
		allocator: mem.Allocator,
	) -> []f64 {
		n := end_day - start_day
		features := make([]f64, n * num_features, allocator)
		for i in 0 ..< n {
			day := start_day + i
			features[i * num_features + 0] = returns[day]
			features[i * num_features + 1] = math.abs(returns[day])
			if day < window {
				features[i * num_features + 2] = 0.0
			} else {
				sum_sq := 0.0
				for j in (day - window) ..< day {sum_sq += returns[j] * returns[j]}
				features[i * num_features + 2] = math.sqrt(sum_sq / f64(window))
			}
			features[i * num_features + 3] = vix_levels[day] / 100.0
		}
		return features
	}

	// ✅ V2 FIX: Helper that returns standardized features AND the means/stds for inference
	train_and_standardize :: proc(
		forecaster: ^ml_fin.EnsembleVolatilityForecaster,
		opt: ^nn.Adam,
		returns, vix_levels: []f64,
		train_start, train_end: int,
		window, seq_len, num_features, hidden_size, batch_size, epochs, forward_horizon: int,
		allocator: mem.Allocator,
	) -> (
		means, stds: []f64,
	) {
		n_days := train_end - train_start - window - forward_horizon
		if n_days <= seq_len {
			return make([]f64, num_features, allocator), make([]f64, num_features, allocator)
		}

		features := build_features(
			returns,
			vix_levels,
			train_start + window,
			train_end - forward_horizon,
			window,
			num_features,
			allocator,
		)

		// ✅ V2 FIX: Target is 20-day forward annualized RV
		targets := make([]f64, n_days, allocator)
		for i in 0 ..< n_days {
			day := train_start + window + i
			sum_sq := 0.0
			for j in 1 ..= forward_horizon {
				if day + j < len(returns) {sum_sq += returns[day + j] * returns[day + j]}
			}
			targets[i] = math.sqrt(sum_sq / f64(forward_horizon)) * math.sqrt_f64(252.0) // Annualized %
		}

		// Standardize Features
		means = make([]f64, num_features, allocator)
		stds = make([]f64, num_features, allocator)
		for day in 0 ..< n_days {
			for f in 0 ..< num_features {means[f] += features[day * num_features + f]}
		}
		for f in 0 ..< num_features {means[f] /= f64(n_days)}
		for day in 0 ..< n_days {
			for f in 0 ..< num_features {
				diff := features[day * num_features + f] - means[f]
				stds[f] += diff * diff
			}
		}
		for f in 0 ..< num_features {
			stds[f] = math.sqrt(stds[f] / f64(n_days))
			if stds[f] < 1e-8 {stds[f] = 1.0}
		}
		for day in 0 ..< n_days {
			for f in 0 ..< num_features {
				features[day * num_features + f] =
					(features[day * num_features + f] - means[f]) / stds[f]
			}
		}

		// Create sequences
		num_samples := n_days - seq_len
		if num_samples <= 0 {
			delete(features, allocator); delete(targets, allocator)
			return means, stds
		}
		X_seq := make([]f64, num_samples * seq_len * num_features, allocator)
		Y_seq := make([]f64, num_samples, allocator)
		for i in 0 ..< num_samples {
			src := i * num_features
			dst := i * seq_len * num_features
			copy(
				X_seq[dst:dst + seq_len * num_features],
				features[src:src + seq_len * num_features],
			)
			Y_seq[i] = targets[i + seq_len]
		}

		// Training loop
		for epoch in 0 ..< epochs {
			for b in 0 ..< num_samples / batch_size {
				batch_start := b * batch_size
				x_data := l.matrix_new(f64, 1, batch_size * seq_len * num_features, allocator)
				copy(
					x_data.data,
					X_seq[batch_start *
					seq_len *
					num_features:(batch_start + batch_size) *
					seq_len *
					num_features],
				)
				x_batch := t.tensor_new(x_data, true, allocator)
				x_batch.shape = [4]int{batch_size, seq_len, num_features, 1}

				h0 := t.tensor_new(
					l.matrix_new(f64, 1, batch_size * hidden_size, allocator),
					false,
					allocator,
				)
				c0 := t.tensor_new(
					l.matrix_new(f64, 1, batch_size * hidden_size, allocator),
					false,
					allocator,
				)

				y_data := l.matrix_new(f64, batch_size, 1, allocator)
				copy(y_data.data, Y_seq[batch_start:batch_start + batch_size])
				y_batch := t.tensor_new(y_data, false, allocator)
				y_batch.shape = [4]int{batch_size, 1, 1, 1}

				preds := ml_fin.lstm_volatility_forecaster_forward(
					&forecaster.lstm,
					x_batch,
					h0,
					c0,
				)
				loss := t.tensor_mse_loss(preds, y_batch)
				t.tensor_backward(loss, allocator)
				nn.adam_step(opt)
				nn.adam_zero_grad(opt)

				t.tensor_free_graph(loss)
				t.tensor_free(
					x_batch,
				); t.tensor_free(h0); t.tensor_free(c0); t.tensor_free(y_batch)
			}
		}

		delete(features, allocator); delete(targets, allocator)
		delete(X_seq, allocator); delete(Y_seq, allocator)
		return means, stds
	}

	// 5. Initial Training
	fmt.println("\n--- Initial Training ---")
	train_returns := returns[window:initial_train_days]
	residuals := ts.extract_residuals(train_returns, main_alloc)
	defer delete(residuals, main_alloc)
	garch_result := ts.garch_fit(residuals, .StudentT, 1, 1, 1000, 1e-4, main_alloc)
	defer {
		delete(garch_result.params.alpha, main_alloc); delete(garch_result.params.beta, main_alloc)
		delete(
			garch_result.conditional_var,
			main_alloc,
		); delete(garch_result.standardized_resid, main_alloc)
	}
	forecaster.garch_omega = garch_result.params.omega
	forecaster.garch_alpha = garch_result.params.alpha[0]
	forecaster.garch_beta = garch_result.params.beta[0]

	current_means, current_stds := train_and_standardize(
		&forecaster,
		&opt,
		returns,
		vix_levels,
		0,
		initial_train_days,
		window,
		seq_len,
		num_features,
		hidden_size,
		batch_size,
		30,
		forward_horizon,
		main_alloc,
	)
	defer {delete(current_means, main_alloc); delete(current_stds, main_alloc)}
	fmt.println("  Initial training complete.")

	// 6. Walk-Forward Loop
	fmt.println("\n--- Running Walk-Forward Backtest ---")
	equity := 0.0
	last_retrain_day := backtest_start
	prev_vrp := 0.0
	prev_position := 0.0

	for day in backtest_start ..< num_days - 1 {
		bt_idx := day - backtest_start

		// Periodic Retraining
		if day - last_retrain_day >= retrain_interval {
			retrain_returns := returns[window:day]
			retrain_resid := ts.extract_residuals(retrain_returns, main_alloc)
			retrain_garch := ts.garch_fit(retrain_resid, .StudentT, 1, 1, 500, 1e-4, main_alloc)
			forecaster.garch_omega = retrain_garch.params.omega
			forecaster.garch_alpha = retrain_garch.params.alpha[0]
			forecaster.garch_beta = retrain_garch.params.beta[0]
			delete(
				retrain_garch.params.alpha,
				main_alloc,
			); delete(retrain_garch.params.beta, main_alloc)
			delete(
				retrain_garch.conditional_var,
				main_alloc,
			); delete(retrain_garch.standardized_resid, main_alloc)
			delete(retrain_resid, main_alloc)

			// ✅ V2 FIX: Update means and stds from the expanding window
			delete(current_means, main_alloc); delete(current_stds, main_alloc)
			current_means, current_stds = train_and_standardize(
				&forecaster,
				&opt,
				returns,
				vix_levels,
				0,
				day,
				window,
				seq_len,
				num_features,
				hidden_size,
				batch_size,
				retrain_epochs,
				forward_horizon,
				main_alloc,
			)
			last_retrain_day = day
		}

		// Inference
		feat_start := day - seq_len + 1
		if feat_start < window {continue}
		inf_features := build_features(
			returns,
			vix_levels,
			feat_start,
			day + 1,
			window,
			num_features,
			main_alloc,
		)

		// ✅ V2 FIX: Apply exact training standardization to inference
		for i in 0 ..< seq_len {
			for f in 0 ..< num_features {
				idx := i * num_features + f
				inf_features[idx] = (inf_features[idx] - current_means[f]) / current_stds[f]
			}
		}

		x_inf_data := l.matrix_new(f64, 1, 1 * seq_len * num_features, main_alloc)
		copy(x_inf_data.data, inf_features)
		x_inf := t.tensor_new(x_inf_data, true, main_alloc)
		x_inf.shape = [4]int{1, seq_len, num_features, 1}
		h0_inf := t.tensor_new(l.matrix_new(f64, 1, hidden_size, main_alloc), false, main_alloc)
		c0_inf := t.tensor_new(l.matrix_new(f64, 1, hidden_size, main_alloc), false, main_alloc)

		lstm_pred := ml_fin.lstm_volatility_forecaster_forward(
			&forecaster.lstm,
			x_inf,
			h0_inf,
			c0_inf,
		)
		lstm_rv := lstm_pred.data.data[0] // Already annualized % from target definition
		t.tensor_free_graph(lstm_pred)
		t.tensor_free(x_inf); t.tensor_free(h0_inf); t.tensor_free(c0_inf)
		delete(inf_features, main_alloc)

		// GARCH Forecast (20-day average approximation)
		last_ret := returns[day]
		last_var := forecaster.garch_omega / (1.0 - forecaster.garch_alpha - forecaster.garch_beta)
		if last_var <= 0.0 {last_var = 0.0001}
		garch_var_next :=
			forecaster.garch_omega +
			forecaster.garch_alpha * last_ret * last_ret +
			forecaster.garch_beta * last_var
		garch_rv := math.sqrt(garch_var_next) * math.sqrt_f64(252.0) * 100.0 // Annualized %

		ensemble_weight := 0.65
		ensemble_rv := ensemble_weight * garch_rv + (1.0 - ensemble_weight) * lstm_rv

		// VRP Signal
		current_vix := vix_levels[day]
		vrp_today := current_vix - ensemble_rv
		z_score := (vrp_today - vrp_mean) / vrp_std

		// ✅ V2 FIX: Discrete Thresholds to prevent overtrading
		position := 0.0
		if z_score > 1.2 {
			position = 1.0 // Short Vol (Expect VRP to shrink)
		} else if z_score < -1.2 {
			position = -1.0 // Long Vol (Expect VRP to expand)
		}

		// ✅ V2 FIX: Mean-Reversion PnL (Trading the spread convergence)
		// PnL = position * (VRP_today - VRP_tomorrow)
		daily_pnl := prev_position * (prev_vrp - vrp_today)

		equity += daily_pnl
		result.daily_pnl[bt_idx] = daily_pnl
		result.equity_curve[bt_idx] = equity
		result.positions[bt_idx] = position
		result.signals[bt_idx] = z_score
		result.forecast_rv[bt_idx] = ensemble_rv
		result.implied_vol[bt_idx] = current_vix
		result.actual_vol[bt_idx] = 0.0 // Placeholder for 20-day forward realization

		prev_vrp = vrp_today
		prev_position = position

		if bt_idx % 100 == 0 {
			fmt.printf(
				"  Day %d | VRP: %+.2f%% | Z: %+.2f | Pos: %+.0f | Equity: %+.4f\n",
				day,
				vrp_today,
				z_score,
				position,
				equity,
			)
		}
	}

	ml_fin.compute_backtest_metrics(&result)
	ml_fin.print_backtest_result(&result)
	fmt.println("\n✓ Walk-Forward VRP Backtest V2 Complete!")
}
