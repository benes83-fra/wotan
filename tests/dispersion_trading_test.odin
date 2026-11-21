package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// Dispersion trading on a basket of AAPL, AMZN and GOOGL (equal weights), which is
// also the index: compute_dispersion_signal's formula assumes that the index is the
// weighted basket of its components.
// - Implied side: the components' implied volatilities are CBOE's single-stock
//   volatility indices (VXAPL, VXAZN, VXGOG); the implied correlation is CBOE's
//   COR1M, the market's one-month implied correlation of the 50 largest S&P 500
//   stocks. No market quotes the basket's own implied volatility, so it is built
//   from COR1M and the components' implied volatilities; the formula must then
//   return COR1M.
// - Realised side: GARCH-Mamba forecasts of the next 21 days' volatility of the
//   basket and of each component.
// - Premium history: on every day of the sample, COR1M minus the basket's
//   correlation over the following 21 trading days (the correlation risk premium);
//   its mean and standard deviation z-score today's premium. The three stocks are
//   more correlated than a typical pair of the top 50, so the premium has a
//   negative mean; the z-score removes that level.
// Yahoo serves only the latest value of the CBOE indices, so their histories come
// from CBOE (net.read_cboe_index).

@(private = "file")
DT_HORIZON :: 21 // trading days: the horizon of COR1M and of the forecasts

// The basket's realised correlation from returns (one row per day, one column per
// component): the CBOE formula on mean squared returns (not demeaned, as the
// forecasts), basket = weighted sum of the components
@(private = "file")
_dt_realised_corr :: proc(R: [][3]f64, weights: []f64) -> f64 {
	comp_ms: [3]f64
	basket_ms := 0.0
	for r in R {
		b := 0.0
		for i in 0 ..< 3 {
			comp_ms[i] += r[i] * r[i]
			b += weights[i] * r[i]
		}
		basket_ms += b * b
	}
	n := f64(len(R))
	for &v in comp_ms {v /= n}
	return ml_fin.calculate_pairwise_correlation(basket_ms / n, comp_ms[:], weights)
}

// A GARCH(1,1)-t and Mamba ensemble's forecast of the annualised volatility of the
// next `horizon` returns. Features of day i: r_i, |r_i|, the standard deviation of
// the 20 returns before, and an implied volatility (decimal). The network learns
// the logarithm of the realised volatility, standardised with the training samples'
// mean and standard deviation; its forecast exp(ymean + ysd · output) is positive.
@(private = "file")
_dt_train_and_predict :: proc(
	returns: []f64,
	implied: []f64,
	horizon: int,
	allocator: mem.Allocator,
) -> f64 {
	window, seq_len, num_features, hidden_size := 20, 20, 4, 16
	batch_size, epochs, lr := 32, 40, 0.0005
	n_days := len(returns)

	features := make([]f64, n_days * num_features, allocator)
	targets := make([]f64, n_days, allocator)
	defer {delete(features, allocator); delete(targets, allocator)}

	max_target_idx := n_days - horizon
	if max_target_idx <= window + seq_len {return 0.0}

	for i in 0 ..< n_days {
		features[i * num_features + 0] = returns[i]
		features[i * num_features + 1] = math.abs(returns[i])
		if i >= window {
			sum_sq := 0.0
			for j in (i - window) ..< i {sum_sq += returns[j] * returns[j]}
			features[i * num_features + 2] = math.sqrt(sum_sq / f64(window))
		}
		features[i * num_features + 3] = implied[i]
		if i < max_target_idx {
			sum_sq_fwd := 0.0
			for j in 1 ..= horizon {sum_sq_fwd += returns[i + j] * returns[i + j]}
			targets[i] = math.sqrt(sum_sq_fwd / f64(horizon)) * math.sqrt(252.0)
		}
	}

	start_idx := window
	valid_days := max_target_idx - start_idx
	train_days := int(f64(valid_days) * 0.8)

	means := make([]f64, num_features, allocator)
	stds := make([]f64, num_features, allocator)
	defer {delete(means, allocator); delete(stds, allocator)}
	for day in 0 ..< train_days {
		for f in 0 ..< num_features {means[f] += features[(start_idx + day) * num_features + f]}
	}
	for f in 0 ..< num_features {means[f] /= f64(train_days)}
	for day in 0 ..< train_days {
		for f in 0 ..< num_features {
			diff := features[(start_idx + day) * num_features + f] - means[f]
			stds[f] += diff * diff
		}
	}
	for f in 0 ..< num_features {
		stds[f] = math.sqrt(stds[f] / f64(train_days))
		if stds[f] < 1e-8 {stds[f] = 1.0}
	}
	// standardised with the training statistics and winsorised at 4 sd (the Mamba
	// block normalises its input itself; the clamp keeps single crash days from
	// dominating the linear adapter in front of it)
	for day in 0 ..< valid_days {
		idx := start_idx + day
		for f in 0 ..< num_features {
			z := (features[idx * num_features + f] - means[f]) / stds[f]
			features[idx * num_features + f] = clamp(z, -4, 4)
		}
	}

	num_samples := valid_days - seq_len
	if num_samples <= seq_len {return 0.0}
	X_seq := make([]f64, num_samples * seq_len * num_features, allocator)
	Y_seq := make([]f64, num_samples, allocator)
	defer {delete(X_seq, allocator); delete(Y_seq, allocator)}
	for i in 0 ..< num_samples {
		src := (start_idx + i) * num_features
		copy(
			X_seq[i * seq_len * num_features:][:seq_len * num_features],
			features[src:][:seq_len * num_features],
		)
		Y_seq[i] = math.ln(max(targets[start_idx + i + seq_len], 1e-12))
	}
	num_train_samples := int(f64(num_samples) * 0.8)
	if num_train_samples < batch_size {return 0.0}
	ymean, ysd := 0.0, 0.0
	for y in Y_seq[:num_train_samples] {ymean += y}
	ymean /= f64(num_train_samples)
	for y in Y_seq[:num_train_samples] {ysd += (y - ymean) * (y - ymean)}
	ysd = max(math.sqrt(ysd / f64(num_train_samples)), 1e-12)
	for &y in Y_seq {y = (y - ymean) / ysd}

	train_returns := returns[window:start_idx + train_days]
	residuals := ts.extract_residuals(train_returns, allocator)
	defer delete(residuals, allocator)
	garch_result := ts.garch_fit(residuals, .StudentT, 1, 1, 500, 1e-4, allocator)
	defer ts.garch_result_free(&garch_result, allocator)

	ensemble := ml_fin.ensemble_volatility_new(
		num_features,
		hidden_size,
		seq_len,
		allocator,
		.Mamba,
	)
	defer ml_fin.ensemble_volatility_free(&ensemble)
	ensemble.garch_omega = garch_result.params.omega
	ensemble.garch_alpha = garch_result.params.alpha[0]
	ensemble.garch_beta = garch_result.params.beta[0]

	opt := nn.adam_new(lr, 0.9, 0.999, 1e-8, allocator)
	defer nn.adam_free(&opt)
	ml_fin.ensemble_add_to_optimizer(&ensemble, &opt)

	// One arena per training step / forecast, reused across them
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)

	for epoch in 0 ..< epochs {
		epoch_loss := 0.0
		for b in 0 ..< num_train_samples / batch_size {
			bs := b * batch_size
			x_data := l.matrix_new(f64, 1, batch_size * seq_len * num_features, allocator)
			copy(
				x_data.data,
				X_seq[bs * seq_len * num_features:][:batch_size * seq_len * num_features],
			)
			x_batch := t.tensor_new(x_data, true, allocator)
			x_batch.shape = [4]int{batch_size, seq_len, num_features, 1}
			y_data := l.matrix_new(f64, batch_size, 1, allocator)
			copy(y_data.data, Y_seq[bs:][:batch_size])
			y_batch := t.tensor_new(y_data, false, allocator)
			y_batch.shape = [4]int{batch_size, 1, 1, 1}

			t.graph_begin(&scope)
			preds := ml_fin.lstm_volatility_forecaster_forward(&ensemble.lstm, x_batch)
			loss := t.tensor_mse_loss(preds, y_batch)
			t.tensor_backward(loss, allocator)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)
			epoch_loss += loss.data.data[0]
			t.tensor_free_graph(loss)
			t.graph_end(&scope)
			t.tensor_free(x_batch)
			t.tensor_free(y_batch)
		}
		if epoch % 10 == 0 || epoch == epochs - 1 {
			fmt.printf(
				"    epoch %02d | MSE (standardised log vol): %.4f\n",
				epoch,
				epoch_loss / f64(num_train_samples / batch_size),
			)
		}
	}

	// the newest sample (features already standardised)
	last := num_samples - 1
	x_inf_data := l.matrix_new(f64, 1, seq_len * num_features, allocator)
	copy(x_inf_data.data, X_seq[last * seq_len * num_features:][:seq_len * num_features])
	x_inf := t.tensor_new(x_inf_data, false, allocator)
	x_inf.shape = [4]int{1, seq_len, num_features, 1}
	t.graph_begin(&scope)
	pred := ml_fin.lstm_volatility_forecaster_forward(&ensemble.lstm, x_inf)
	model_vol := math.exp(ymean + ysd * pred.data.data[0])
	t.tensor_free_graph(pred)
	t.graph_end(&scope)
	t.tensor_free(x_inf)

	garch_vol := math.sqrt(
		garch_result.conditional_var[len(garch_result.conditional_var) - 1] * 252,
	)
	return 0.65 * garch_vol + 0.35 * model_vol
}

dispersion_trading_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Cross-Asset Dispersion Trading Test ===")
	main_alloc := context.allocator
	failed := 0
	check :: proc(failed: ^int, ok: bool, name: string) {
		if !ok {failed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	stocks := []string{"AAPL", "AMZN", "GOOGL"}
	vol_indices := []string{"VXAPL", "VXAZN", "VXGOG"}
	weights := []f64{1.0 / 3, 1.0 / 3, 1.0 / 3}

	// 1. Data: the stocks and the VIX from Yahoo, the implied volatilities and
	// COR1M from CBOE, matched by date
	fmt.println("\n--- Fetching Market Data ---")
	symbols := []string{"AAPL", "AMZN", "GOOGL", "^VIX"}
	NS := len(symbols)
	panel, panel_ok, bad := net.read_yahoo_panel(symbols, .Daily, .TwoYears, allocator)
	defer net.yahoo_panel_free(&panel)
	if !panel_ok {
		what := "a common date" if bad < 0 else symbols[bad]
		fail_missing_data(#procedure, fmt.tprintf("Yahoo data for %s", what))
		return
	}
	cboe_syms := []string{"VXAPL", "VXAZN", "VXGOG", "COR1M"}
	cboe: [4]map[w.Date]f64
	defer for &m in cboe {delete(m)}
	for sym, k in cboe_syms {
		s, ok := net.read_cboe_index(sym, main_alloc)
		defer net.cboe_series_free(&s)
		if !ok {
			fail_missing_data(#procedure, fmt.tprintf("the CBOE history of %s", sym))
			return
		}
		cboe[k] = make(map[w.Date]f64, main_alloc)
		for d, i in s.dates {cboe[k][d] = s.close[i]}
	}
	// closes only: a running session's row is left out
	n_rows := len(panel.dates) - (1 if panel.last_is_live else 0)
	rows := make([dynamic]int, main_alloc)
	defer delete(rows)
	for r in 0 ..< n_rows {
		all := true
		for m in cboe {all &&= panel.dates[r] in m}
		if all {append(&rows, r)}
	}
	T := len(rows)
	if T < 300 {
		fail_missing_data(#procedure, fmt.tprintf("enough common days (%d)", T))
		return
	}
	fmt.printf(
		"%d common days, %s .. %s\n",
		T,
		w.date_to_string(panel.dates[rows[0]], context.temp_allocator),
		w.date_to_string(panel.dates[rows[T - 1]], context.temp_allocator),
	)

	// day k of the sample is panel row rows[k]; return k is from day k to day k + 1
	n_ret := T - 1
	R := make([][3]f64, n_ret, main_alloc) // component returns
	defer delete(R, main_alloc)
	basket := make([]f64, n_ret, main_alloc)
	defer delete(basket, main_alloc)
	comp_ret: [3][]f64
	comp_iv: [3][]f64 // implied volatility at the close ending return k, decimal
	for i in 0 ..< 3 {
		comp_ret[i] = make([]f64, n_ret, main_alloc)
		comp_iv[i] = make([]f64, n_ret, main_alloc)
	}
	defer for i in 0 ..< 3 {delete(comp_ret[i], main_alloc); delete(comp_iv[i], main_alloc)}
	vix := make([]f64, n_ret, main_alloc)
	defer delete(vix, main_alloc)
	cor := make([]f64, T, main_alloc) // COR1M at the close of day k, decimal
	defer delete(cor, main_alloc)
	for k in 0 ..< T {cor[k] = cboe[3][panel.dates[rows[k]]] / 100}
	for k in 0 ..< n_ret {
		a, b := rows[k], rows[k + 1]
		for i in 0 ..< 3 {
			r := math.ln(panel.adj_close[b * NS + i] / panel.adj_close[a * NS + i])
			R[k][i] = r
			comp_ret[i][k] = r
			basket[k] += weights[i] * r
			comp_iv[i][k] = cboe[i][panel.dates[b]] / 100
		}
		vix[k] = panel.adj_close[b * NS + 3] / 100
	}

	// 2. The formula against a direct estimate: over the last 120 days, the
	// realised correlation from the basket's and the components' volatilities is
	// the weighted average pairwise correlation Σ_{i≠j} w_i w_j S_ij / Σ_{i≠j} w_i w_j
	// √(S_ii S_jj), S = the mean of r_i r_j
	{
		win := R[n_ret - 120:]
		S: [3][3]f64
		for r in win {for i in 0 ..< 3 {for j in 0 ..< 3 {S[i][j] += r[i] * r[j] / 120}}}
		num, den := 0.0, 0.0
		for i in 0 ..< 3 {
			for j in 0 ..< 3 {
				if i == j do continue
				num += weights[i] * weights[j] * S[i][j]
				den += weights[i] * weights[j] * math.sqrt(S[i][i] * S[j][j])
			}
		}
		got := _dt_realised_corr(win, weights)
		check(
			&failed,
			math.abs(got - num / den) < 1e-10,
			fmt.tprintf(
				"realised correlation, last 120 days: formula %.6f = weighted pairwise %.6f",
				got,
				num / den,
			),
		)
	}

	// 3. The correlation risk premium on every day whose next 21 days are known
	hist := make([dynamic]f64, main_alloc)
	defer delete(hist)
	for k in 0 ..= n_ret - DT_HORIZON {
		append(&hist, cor[k] - _dt_realised_corr(R[k:k + DT_HORIZON], weights))
	}
	hmean, hsd := 0.0, 0.0
	for v in hist {hmean += v}
	hmean /= f64(len(hist))
	for v in hist {hsd += (v - hmean) * (v - hmean)}
	hsd = math.sqrt(hsd / f64(len(hist) - 1))
	fmt.printf(
		"correlation premium (COR1M - next 21 days' basket correlation), %d days: mean %+.4f, sd %.4f\n",
		len(hist),
		hmean,
		hsd,
	)

	// 4. Forecasts of the next 21 days' volatility
	fmt.println("\n--- Training Ensembles ---")
	fmt.println("Basket (equal weights)...")
	basket_rv := _dt_train_and_predict(basket, vix, DT_HORIZON, main_alloc)
	comp_rvs: [3]f64
	for i in 0 ..< 3 {
		fmt.printf("%s...\n", stocks[i])
		comp_rvs[i] = _dt_train_and_predict(comp_ret[i], comp_iv[i], DT_HORIZON, main_alloc)
	}

	fmt.println("\n--- Dispersion Signal ---")
	forecasts_ok := basket_rv > 0 && basket_rv < 3
	for rv in comp_rvs {forecasts_ok = forecasts_ok && rv > 0 && rv < 3}
	check(
		&failed,
		forecasts_ok,
		fmt.tprintf(
			"forecast vols positive and below 300%%: basket %.4f, components %.4f",
			basket_rv,
			comp_rvs,
		),
	)

	// 5. Today's signal. The basket's implied volatility from COR1M and the
	// components' implied volatilities, σ_B² = Σ w²σ² + ρ [(Σ wσ)² - Σ w²σ²]
	rho := cor[T - 1]
	ivs: [3]f64
	for i in 0 ..< 3 {ivs[i] = comp_iv[i][n_ret - 1]}
	sw2, swv := 0.0, 0.0
	for i in 0 ..< 3 {
		sw2 += weights[i] * weights[i] * ivs[i] * ivs[i]
		swv += weights[i] * ivs[i]
	}
	basket_iv := math.sqrt(sw2 + rho * (swv * swv - sw2))
	fmt.printf(
		"COR1M %.2f%%, implied vols %s %.2f%% %s %.2f%% %s %.2f%%, basket implied vol %.2f%%\n",
		100 * rho,
		vol_indices[0],
		100 * ivs[0],
		vol_indices[1],
		100 * ivs[1],
		vol_indices[2],
		100 * ivs[2],
		100 * basket_iv,
	)
	sig := ml_fin.compute_dispersion_signal(
		basket_iv,
		basket_rv,
		ivs[:],
		comp_rvs[:],
		weights,
		hmean,
		hsd,
	)
	ml_fin.print_dispersion_signal(sig)
	z := (sig.correlation_premium - hmean) / hsd
	fmt.printf("premium %+.4f against its history: z = %+.2f\n", sig.correlation_premium, z)
	check(
		&failed,
		math.abs(sig.implied_corr - rho) < 1e-12,
		fmt.tprintf(
			"implied correlation from the basket implied vol = COR1M (%.6f vs %.6f)",
			sig.implied_corr,
			rho,
		),
	)
	in_range :: proc(x: f64) -> bool {return x >= -1 && x <= 1} 	// false for NaN
	check(
		&failed,
		in_range(sig.realized_corr) && in_range(sig.signal) && len(hist) >= 250 && hsd > 0,
		fmt.tprintf(
			"forecast correlation %.4f and signal %.4f in [-1, 1], from %d days of premium history",
			sig.realized_corr,
			sig.signal,
			len(hist),
		),
	)

	// 6. Factor risk of the components
	fmt.println("\n--- Factor Risk Decomposition ---")
	comp_returns := make([][]f64, n_ret, main_alloc)
	defer {
		for r in comp_returns {delete(r, main_alloc)}
		delete(comp_returns, main_alloc)
	}
	for k in 0 ..< n_ret {
		comp_returns[k] = make([]f64, 3, main_alloc)
		for i in 0 ..< 3 {comp_returns[k][i] = R[k][i]}
	}
	risk := fin.decompose_risk(comp_returns, 1, main_alloc) // 1 factor = market
	defer {
		delete(risk.factor_variance, main_alloc)
		delete(risk.idiosyncratic_var, main_alloc)
		delete(risk.total_variance, main_alloc)
		for row in risk.factor_exposure {delete(row, main_alloc)}
		delete(risk.factor_exposure, main_alloc)
	}
	fmt.printf("  %-8s %-12s %-12s %-12s\n", "Asset", "Systematic", "Idiosync.", "Total")
	for i in 0 ..< 3 {
		fmt.printf(
			"  %-8s %10.2f%% %10.2f%% %10.2f%%\n",
			stocks[i],
			risk.factor_variance[i] * 100.0,
			risk.idiosyncratic_var[i] * 100.0,
			risk.total_variance[i] * 100.0,
		)
	}

	record_failures(failed)
	fmt.printf("\nCross-asset dispersion trading: %d/4 checks passed\n", 4 - failed)
}
