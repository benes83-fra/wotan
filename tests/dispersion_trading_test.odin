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

dispersion_trading_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Cross-Asset Dispersion Trading Test ===")
	main_alloc := context.allocator

	// 1. Define the Basket (Index + 3 Mega-Cap Components)
	tickers := []string{"SPY", "AAPL", "MSFT", "NVDA"}
	n_assets := len(tickers)

	// ✅ Realistic SPY index weights for top holdings (approximate)
	// These will be normalized to sum to 1.0 for the sub-basket
	raw_weights := []f64{0.064, 0.069, 0.062} // AAPL, MSFT, NVDA
	basket_weights := ml_fin.normalize_weights(raw_weights, main_alloc)
	defer delete(basket_weights, main_alloc)

	fmt.println("\n--- Fetching Market Data ---")
	dfs := make([]w.DataFrame, n_assets, main_alloc)
	defer {
		for i in 0 ..< n_assets {w.destroy_dataframe(&dfs[i])}
		delete(dfs, main_alloc)
	}

	for i in 0 ..< n_assets {
		dfs[i] = yahoo.read_yahoo(tickers[i], .Daily, .TwoYears, allocator)
		fmt.printf("Loaded %d days for %s\n", dfs[i].rows, tickers[i])
	}

	// Align lengths
	n_common := dfs[0].rows
	for i in 1 ..< n_assets {
		if dfs[i].rows < n_common {n_common = dfs[i].rows}
	}
	num_days := n_common - 1

	// 2. Compute Returns
	returns_matrix := make([][]f64, n_assets, main_alloc)
	defer {
		for i in 0 ..< n_assets {delete(returns_matrix[i], main_alloc)}
		delete(returns_matrix, main_alloc)
	}

	for i in 0 ..< n_assets {
		returns_matrix[i] = make([]f64, num_days, main_alloc)
		for d in 1 ..< n_common {
			prev_c, _ := w.column_at_float(&dfs[i].columns[4], d - 1)
			curr_c, _ := w.column_at_float(&dfs[i].columns[4], d)
			returns_matrix[i][d - 1] = math.ln_f64(curr_c / prev_c)
		}
	}

	// Fetch VIX for Index IV proxy
	vix_df := yahoo.read_yahoo("^VIX", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&vix_df)
	vix_levels := make([]f64, num_days, main_alloc)
	defer delete(vix_levels, main_alloc)
	for d in 1 ..< min(n_common, vix_df.rows) {
		vix_c, _ := w.column_at_float(&vix_df.columns[4], d)
		vix_levels[d - 1] = vix_c / 100.0
	}

	// 3. Train Ensembles
	fmt.println("\n--- Training Ensembles ---")
	window := 20
	seq_len := 20
	num_features := 4
	hidden_size := 16
	batch_size := 32
	epochs := 15
	learning_rate := 0.001

	// Nested helper: ALL dependencies passed as explicit parameters (no closures in Odin)
	train_and_predict :: proc(
		returns: []f64,
		vix_proxy: []f64,
		window: int,
		seq_len: int,
		num_features: int,
		hidden_size: int,
		batch_size: int,
		epochs: int,
		lr: f64,
		allocator: mem.Allocator,
	) -> (
		rv_dec: f64,
	) {
		n_days := len(returns)

		// Features & Targets
		features := make([]f64, n_days * num_features, allocator)
		targets := make([]f64, n_days, allocator)
		defer {delete(features, allocator); delete(targets, allocator)}

		for i in 0 ..< n_days {
			features[i * num_features + 0] = returns[i]
			features[i * num_features + 1] = math.abs(returns[i])
			if i < window {
				features[i * num_features + 2] = 0.0
			} else {
				sum_sq := 0.0
				for j in (i - window) ..< i {sum_sq += returns[j] * returns[j]}
				features[i * num_features + 2] = math.sqrt_f64(sum_sq / f64(window))
			}
			features[i * num_features + 3] = vix_proxy[i]
			if i + 1 < n_days {targets[i] = math.abs(returns[i + 1])}
		}

		start_idx := window
		valid_days := n_days - 1 - start_idx
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
			stds[f] = math.sqrt_f64(stds[f] / f64(train_days))
			if stds[f] < 1e-8 {stds[f] = 1.0}
		}
		for day in 0 ..< valid_days {
			idx := start_idx + day
			for f in 0 ..< num_features {
				features[idx * num_features + f] =
					(features[idx * num_features + f] - means[f]) / stds[f]
			}
		}

		num_samples := valid_days - seq_len
		if num_samples <= seq_len {return 0.0}

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
		if num_train_samples < batch_size {return 0.0}

		// Fit GARCH
		train_returns := returns[window:start_idx + train_days]
		residuals := ts.extract_residuals(train_returns, allocator)
		defer delete(residuals, allocator)
		garch_result := ts.garch_fit(residuals, .StudentT, 1, 1, 500, 1e-4, allocator)
		defer {
			delete(garch_result.params.alpha, allocator)
			delete(garch_result.params.beta, allocator)
			delete(garch_result.conditional_var, allocator)
			delete(garch_result.standardized_resid, allocator)
		}

		// Train LSTM Ensemble
		ensemble := ml_fin.ensemble_volatility_new(num_features, hidden_size, seq_len, allocator)
		defer ml_fin.ensemble_volatility_free(&ensemble)
		ensemble.garch_omega = garch_result.params.omega
		ensemble.garch_alpha = garch_result.params.alpha[0]
		ensemble.garch_beta = garch_result.params.beta[0]

		opt := nn.adam_new(lr, 0.9, 0.999, 1e-8, allocator)
		defer nn.adam_free(&opt)
		ml_fin.ensemble_add_to_optimizer(&ensemble, &opt)

		for epoch in 0 ..< epochs {
			for b in 0 ..< num_train_samples / batch_size {
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

				preds := ml_fin.lstm_volatility_forecaster_forward(&ensemble.lstm, x_batch, h0, c0)
				loss := t.tensor_mse_loss(preds, y_batch)
				t.tensor_backward(loss, allocator)
				nn.adam_step(&opt)
				nn.adam_zero_grad(&opt)

				t.tensor_free_graph(loss)
				t.tensor_free(x_batch)
				t.tensor_free(h0)
				t.tensor_free(c0)
				t.tensor_free(y_batch)
			}
		}

		// Inference on last sequence
		last_sample_idx := num_samples - 1
		x_inf_data := l.matrix_new(f64, 1, 1 * seq_len * num_features, allocator)
		copy(
			x_inf_data.data,
			X_seq[last_sample_idx *
			seq_len *
			num_features:(last_sample_idx + 1) *
			seq_len *
			num_features],
		)
		for i in 0 ..< seq_len {
			for f in 0 ..< num_features {
				idx := i * num_features + f
				x_inf_data.data[idx] = (x_inf_data.data[idx] - means[f]) / stds[f]
			}
		}
		x_inf := t.tensor_new(x_inf_data, false, allocator)
		x_inf.shape = [4]int{1, seq_len, num_features, 1}
		h0_inf := t.tensor_new(l.matrix_new(f64, 1, hidden_size, allocator), false, allocator)
		c0_inf := t.tensor_new(l.matrix_new(f64, 1, hidden_size, allocator), false, allocator)

		lstm_pred_tensor := ml_fin.lstm_volatility_forecaster_forward(
			&ensemble.lstm,
			x_inf,
			h0_inf,
			c0_inf,
		)
		lstm_rv_dec := lstm_pred_tensor.data.data[0]

		t.tensor_free(lstm_pred_tensor)
		t.tensor_free(x_inf)
		t.tensor_free(h0_inf)
		t.tensor_free(c0_inf)

		last_cond_var := garch_result.conditional_var[len(garch_result.conditional_var) - 1]
		garch_rv_dec := math.sqrt_f64(last_cond_var) * math.sqrt_f64(252.0)

		// Ensemble blend
		ensemble_rv := 0.65 * garch_rv_dec + 0.35 * lstm_rv_dec

		return ensemble_rv
	}

	// Run for Index (SPY)
	fmt.println("Training Index (SPY)...")
	index_rv := train_and_predict(
		returns_matrix[0],
		vix_levels,
		window,
		seq_len,
		num_features,
		hidden_size,
		batch_size,
		epochs,
		learning_rate,
		main_alloc,
	)

	// Run for Components
	comp_rvs := make([]f64, n_assets - 1, main_alloc)
	defer delete(comp_rvs, main_alloc)

	for i in 1 ..< n_assets {
		fmt.printf("Training Component (%s)...\n", tickers[i])
		rv := train_and_predict(
			returns_matrix[i],
			vix_levels,
			window,
			seq_len,
			num_features,
			hidden_size,
			batch_size,
			epochs,
			learning_rate,
			main_alloc,
		)
		comp_rvs[i - 1] = rv
	}

	// 4. Compute Dispersion Signal
	index_iv := vix_levels[num_days - 1]

	comp_ivs := make([]f64, n_assets - 1, main_alloc)
	defer delete(comp_ivs, main_alloc)
	for i in 0 ..< len(comp_rvs) {
		comp_ivs[i] = comp_rvs[i] * 1.15 // Approximate VRP premium
	}

	hist_corr_mean := 0.35
	hist_corr_std := 0.15

	sig := ml_fin.compute_dispersion_signal(
		index_iv,
		index_rv,
		comp_ivs,
		comp_rvs,
		basket_weights, // ✅ Pass realistic weights
		hist_corr_mean,
		hist_corr_std,
	)

	ml_fin.print_dispersion_signal(sig)

	fmt.println("\n✓ Cross-Asset Dispersion Trading Test Complete!")
}
