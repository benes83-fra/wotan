package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

volatility_arbitrage_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Volatility Arbitrage: Market IV vs Ensemble RV ===")
	main_alloc := context.allocator

	// 1. Fetch Data
	fmt.println("\n--- Fetching Market Data ---")
	spy_df := yahoo.read_yahoo("SPY", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&spy_df)
	vix_df := yahoo.read_yahoo("^VIX", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&vix_df)

	n_common := min(spy_df.rows, vix_df.rows)
	num_days := n_common - 1

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

	// 2. Train Ensemble (Streamlined for Inference)
	fmt.println("\n--- Training Ensemble Forecaster ---")
	window := 20
	seq_len := 20
	num_features := 4
	hidden_size := 32
	batch_size := 32
	epochs := 20
	learning_rate := 0.001
	forward_horizon := 20

	// Feature generation & standardization (abbreviated for test brevity, same as VRP test)
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
	train_returns := returns[window:start_idx + train_days]
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

	// Train LSTM
	ensemble := ml_fin.ensemble_volatility_new(num_features, hidden_size, seq_len, allocator)
	defer ml_fin.ensemble_volatility_free(&ensemble)
	ensemble.garch_omega = garch_result.params.omega
	ensemble.garch_alpha = garch_result.params.alpha[0]
	ensemble.garch_beta = garch_result.params.beta[0]

	opt := nn.adam_new(learning_rate, 0.9, 0.999, 1e-8, allocator)
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
			t.tensor_free(x_batch); t.tensor_free(h0); t.tensor_free(c0); t.tensor_free(y_batch)
		}
	}

	// 3. Generate Next-Day Forecast
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

	// Apply standardization
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
	lstm_rv_dec := lstm_pred_tensor.data.data[0] // Already annualized decimal from target definition

	t.tensor_free(lstm_pred_tensor)
	t.tensor_free(x_inf); t.tensor_free(h0_inf); t.tensor_free(c0_inf)

	// 4. Price the 30-Day ATM Straddle
	S := math.exp(returns[num_days - 1]) * 450.0 // Approximate SPY price
	K := S // At-The-Money
	T_years := 30.0 / 252.0 // 30 days to expiration
	r := 0.05 // 5% risk-free rate

	current_vix := vix_levels[num_days - 1]
	market_iv_dec := current_vix / 100.0

	last_cond_var := garch_result.conditional_var[len(garch_result.conditional_var) - 1]
	ensemble_weight := 0.65

	// A. Market Pricing (Using VIX as flat IV)
	market_call_price, market_call_greeks := fin.price_and_greeks(
		S,
		K,
		T_years,
		r,
		market_iv_dec,
		.Call,
		main_alloc,
	)
	market_put_price, market_put_greeks := fin.price_and_greeks(
		S,
		K,
		T_years,
		r,
		market_iv_dec,
		.Put,
		main_alloc,
	)
	market_straddle_price := market_call_price + market_put_price

	// B. Model Pricing (Using Ensemble Hybrid Term Structure)
	garch_rv_dec := math.sqrt(last_cond_var) * math.sqrt_f64(252.0)
	lstm_blend_dec := ensemble_weight * garch_rv_dec + (1.0 - ensemble_weight) * lstm_rv_dec

	model_call_price, model_call_greeks := ml_fin.ensemble_price_and_greeks(
		S,
		K,
		T_years,
		r,
		lstm_blend_dec,
		ensemble.garch_omega,
		ensemble.garch_alpha,
		ensemble.garch_beta,
		last_cond_var,
		.Call,
		main_alloc,
	)
	model_put_price, model_put_greeks := ml_fin.ensemble_price_and_greeks(
		S,
		K,
		T_years,
		r,
		lstm_blend_dec,
		ensemble.garch_omega,
		ensemble.garch_alpha,
		ensemble.garch_beta,
		last_cond_var,
		.Put,
		main_alloc,
	)
	model_straddle_price := model_call_price + model_put_price

	// 5. Output the Arbitrage Dashboard
	fmt.println(
		"\n╔══════════════════════════════════════════════════════════════╗",
	)
	fmt.println("║           30-DAY ATM STRADDLE ARBITRAGE DASHBOARD            ║")
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.printf("║  Underlying (SPY):      $%7.2f                              ║\n", S)
	fmt.printf("║  Strike (K):            $%7.2f                              ║\n", K)
	fmt.printf(
		"║  Time to Expiration:    %7.2f Years (%d Days)                ║\n",
		T_years,
		30,
	)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║  Market Implied Vol:    %7.2f%% (VIX)                        ║\n",
		current_vix,
	)
	fmt.printf(
		"║  Model Expected RV:     %7.2f%% (Ensemble Term Structure)    ║\n",
		lstm_blend_dec * 100.0,
	)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║  MARKET STRADDLE PRICE: $%7.2f                              ║\n",
		market_straddle_price,
	)
	fmt.printf(
		"║  MODEL STRADDLE PRICE:  $%7.2f                              ║\n",
		model_straddle_price,
	)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)

	edge := market_straddle_price - model_straddle_price
	if edge > 0.10 {
		fmt.printf(
			"║  💰 THEORETICAL EDGE:   $%+7.2f per share (SHORT STRADDLE)   ║\n",
			edge,
		)
		fmt.println("║  ACTION: Sell the Straddle. Market is overpricing vol.      ║")
	} else if edge < -0.10 {
		fmt.printf(
			"║  💰 THEORETICAL EDGE:   $%+7.2f per share (LONG STRADDLE)    ║\n",
			-edge,
		)
		fmt.println("║  ACTION: Buy the Straddle. Market is underpricing vol.      ║")
	} else {
		fmt.printf(
			"║  ⚖️  THEORETICAL EDGE:   $%+7.2f per share (NEUTRAL)          ║\n",
			edge,
		)
		fmt.println("║  ACTION: No edge. Fairly priced.                            ║")
	}
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.println("║  MODEL GREEKS (Call)                                         ║")
	fmt.printf(
		"║    Delta: %+7.4f   Gamma: %7.4f   Vega: %7.4f                ║\n",
		model_call_greeks.delta,
		model_call_greeks.gamma,
		model_call_greeks.vega,
	)
	fmt.printf(
		"║    Theta: %+7.4f   Rho:   %+7.4f                            ║\n",
		model_call_greeks.theta,
		model_call_greeks.rho,
	)
	fmt.println(
		"╚══════════════════════════════════════════════════════════════╝",
	)

	fmt.println("\n✓ Volatility Arbitrage Test Complete!")
}
