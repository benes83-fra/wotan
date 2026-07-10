package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import fin "../wotan/finance"
import yahoo "../wotan/net"
import p "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

garch_real_data_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GARCH Real Data Test (SPY) ===\n")

	main_alloc := context.allocator

	// 1. Fetch SPY data (2 years)
	fmt.println("--- Fetching SPY Data ---")
	spy_df := yahoo.read_yahoo("SPY", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&spy_df)

	fmt.printf("Loaded %d days of SPY data\n", spy_df.rows)

	// 2. Compute log returns
	fmt.println("\n--- Computing Returns ---")
	n := spy_df.rows
	returns := make([]f64, n - 1, main_alloc)
	defer delete(returns, main_alloc)

	dates := make([]f64, n - 1, main_alloc)
	defer delete(dates, main_alloc)

	for i in 1 ..< n {
		prev_close, _ := w.column_at_float(&spy_df.columns[4], i - 1)
		curr_close, _ := w.column_at_float(&spy_df.columns[4], i)
		returns[i - 1] = math.ln_f64(curr_close / prev_close)
		dates[i - 1] = f64(i)
	}

	// Compute mean and std
	mean_ret := 0.0
	for r in returns {
		mean_ret += r
	}
	mean_ret /= f64(len(returns))

	var_sum := 0.0
	for r in returns {
		var_sum += (r - mean_ret) * (r - mean_ret)
	}
	std_ret := math.sqrt_f64(var_sum / f64(len(returns) - 1))

	fmt.printf("Mean daily return: %.4f%%\n", mean_ret * 100)
	fmt.printf("Std deviation: %.4f%%\n", std_ret * 100)
	fmt.printf("Annualized vol: %.2f%%\n", std_ret * math.sqrt_f64(252) * 100)

	// 3. Fit GARCH(1,1) manually for plotting
	fmt.println("\n--- Fitting GARCH(1,1) with Student-t Errors---")
	residuals := ts.extract_residuals(returns, main_alloc)
	defer delete(residuals, main_alloc)

	garch_result := ts.garch_fit(residuals, .StudentT, 1, 1, 2000, 1e-4, main_alloc)
	defer {
		delete(garch_result.params.alpha, main_alloc)
		delete(garch_result.params.beta, main_alloc)
		delete(garch_result.conditional_var, main_alloc)
		delete(garch_result.standardized_resid, main_alloc)
	}

	fmt.printf("\nGARCH(1,1) Parameters:\n")
	fmt.printf("  ω (omega): %.8f\n", garch_result.params.omega)
	fmt.printf("  α (alpha): %.4f\n", garch_result.params.alpha[0])
	fmt.printf("  β (beta):  %.4f\n", garch_result.params.beta[0])
	fmt.printf("  Persistence (α+β): %.4f\n", garch_result.persistence)
	long_run_vol :=
		math.sqrt_f64(garch_result.params.omega / (1.0 - garch_result.persistence)) *
		math.sqrt_f64(252) *
		100
	fmt.printf("  Long-run annualized vol: %.2f%%\n", long_run_vol)
	fmt.printf(
		"  Converged: %v (iterations: %d)\n",
		garch_result.converged,
		garch_result.n_iterations,
	)

	// 4. Compute volatility series for comparison
	fmt.println("\n--- Computing Volatility Series ---")

	// Rolling 20-day vol
	rolling_vol := make([]f64, len(returns), main_alloc)
	defer delete(rolling_vol, main_alloc)
	window := 20
	for i in 0 ..< len(returns) {
		if i < window {
			rolling_vol[i] = 0.0
		} else {
			sum := 0.0
			sum_sq := 0.0
			for j in (i - window) ..< i {
				sum += returns[j]
				sum_sq += returns[j] * returns[j]
			}
			m := sum / f64(window)
			rolling_vol[i] = math.sqrt_f64((sum_sq / f64(window)) - m * m)
		}
	}

	// EWMA vol (alpha = 0.06, RiskMetrics standard)
	ewma_vol := make([]f64, len(returns), main_alloc)
	defer delete(ewma_vol, main_alloc)
	ewma_alpha := 0.06
	ewma_var := returns[0] * returns[0]
	for i in 0 ..< len(returns) {
		ewma_var = ewma_alpha * returns[i] * returns[i] + (1.0 - ewma_alpha) * ewma_var
		ewma_vol[i] = math.sqrt_f64(ewma_var)
	}

	// GARCH vol
	garch_vol := make([]f64, len(returns), main_alloc)
	defer delete(garch_vol, main_alloc)
	for i in 0 ..< len(returns) {
		garch_vol[i] = math.sqrt_f64(garch_result.conditional_var[i])
	}

	// 5. Compute VaR series using new finance API
	fmt.println("\n--- Computing VaR Series ---")
	var_95, var_99 := ts.garch_var_series(&garch_result, residuals, main_alloc)
	defer {
		delete(var_95, main_alloc)
		delete(var_99, main_alloc)
	}

	// 6. Backtest VaR using new finance API (skip first 100 for warmup)
	fmt.println("\n--- VaR Backtesting ---")
	start_idx := 100
	returns_bt := returns[start_idx:]
	var_95_bt := var_95[start_idx:]
	var_99_bt := var_99[start_idx:]

	// Use new backtest_var API (separate calls for 95% and 99%)
	backtest_95 := fin.backtest_var(returns_bt, var_95_bt, 0.95)
	backtest_99 := fin.backtest_var(returns_bt, var_99_bt, 0.99)

	fmt.printf("\n95%% VaR Backtest:\n")
	fmt.printf("  Observations: %d\n", backtest_95.n_obs)
	fmt.printf(
		"  Breaches: %d (expected: %.1f)\n",
		backtest_95.n_breaches,
		backtest_95.expected_breaches,
	)
	fmt.printf("  Breach rate: %.2f%% (expected: 5.00%%)\n", backtest_95.breach_rate * 100)
	fmt.printf("  Kupiec statistic: %.4f\n", backtest_95.kupiec_stat)
	pass_95 := "PASS"
	if !backtest_95.passes_test {
		pass_95 = "FAIL"
	}
	fmt.printf("  p-value: %.4f (%s)\n", backtest_95.kupiec_pvalue, pass_95)

	fmt.printf("\n99%% VaR Backtest:\n")
	fmt.printf("  Observations: %d\n", backtest_99.n_obs)
	fmt.printf(
		"  Breaches: %d (expected: %.1f)\n",
		backtest_99.n_breaches,
		backtest_99.expected_breaches,
	)
	fmt.printf("  Breach rate: %.2f%% (expected: 1.00%%)\n", backtest_99.breach_rate * 100)
	fmt.printf("  Kupiec statistic: %.4f\n", backtest_99.kupiec_stat)
	pass_99 := "PASS"
	if !backtest_99.passes_test {
		pass_99 = "FAIL"
	}
	fmt.printf("  p-value: %.4f (%s)\n", backtest_99.kupiec_pvalue, pass_99)

	// 7. Compare with Historical VaR using new finance API
	fmt.println("\n--- Historical vs GARCH VaR ---")
	hist_var_95 := fin.var_historical(returns_bt, 0.95)
	hist_var_99 := fin.var_historical(returns_bt, 0.99)
	hist_cvar_95 := fin.conditional_var(returns_bt, 0.95)
	hist_cvar_99 := fin.conditional_var(returns_bt, 0.99)

	fmt.printf("%-20s %-15s %-15s\n", "Method", "95% VaR", "99% VaR")
	fmt.printf("%-20s %-15.4f%% %-15.4f%%\n", "Historical", hist_var_95 * 100, hist_var_99 * 100)

	// GARCH VaR for latest observation
	last_cond_var := garch_result.conditional_var[len(garch_result.conditional_var) - 1]
	garch_var_95 := fin.var_garch_single(last_cond_var, 0.95)
	garch_var_99 := fin.var_garch_single(last_cond_var, 0.99)
	fmt.printf(
		"%-20s %-15.4f%% %-15.4f%%\n",
		"GARCH-Normal",
		garch_var_95 * 100,
		garch_var_99 * 100,
	)

	// 8. Current risk metrics using new finance API
	fmt.println("\n--- Current Risk Metrics ---")
	last_vol := garch_vol[len(garch_vol) - 1]

	// Compute VaR and CVaR manually using norm_inv
	z_95 := fin.norm_inv(0.95)
	z_99 := fin.norm_inv(0.99)
	inv_sqrt_2pi := 0.3989422804014327
	phi_95 := inv_sqrt_2pi * math.exp_f64(-0.5 * z_95 * z_95)
	phi_99 := inv_sqrt_2pi * math.exp_f64(-0.5 * z_99 * z_99)

	var_95_current := -z_95 * last_vol
	var_99_current := -z_99 * last_vol
	cvar_95_current := -last_vol * phi_95 / 0.05
	cvar_99_current := -last_vol * phi_99 / 0.01

	fmt.printf("Latest GARCH volatility (daily): %.4f%%\n", last_vol * 100)
	fmt.printf(
		"Latest GARCH volatility (annualized): %.2f%%\n",
		last_vol * math.sqrt_f64(252) * 100,
	)
	fmt.printf("95%% VaR (1-day): %.4f%%\n", var_95_current * 100)
	fmt.printf("99%% VaR (1-day): %.4f%%\n", var_99_current * 100)
	fmt.printf("95%% CVaR (1-day): %.4f%%\n", cvar_95_current * 100)
	fmt.printf("99%% CVaR (1-day): %.4f%%\n", cvar_99_current * 100)

	// For $1M portfolio
	portfolio := 1_000_000.0
	fmt.printf("\nFor $1,000,000 portfolio:\n")
	fmt.printf("  95%% VaR: $%.2f\n", portfolio * -var_95_current)
	fmt.printf("  99%% VaR: $%.2f\n", portfolio * -var_99_current)
	fmt.printf("  95%% CVaR: $%.2f\n", portfolio * -cvar_95_current)
	fmt.printf("  99%% CVaR: $%.2f\n", portfolio * -cvar_99_current)

	// 9. Visualization
	fmt.println("\n--- Generating Visualizations ---")

	// Plot 1: Returns with GARCH volatility bands
	plot_returns_with_vol(dates, returns, garch_vol, "garch_returns_vol.png", allocator)

	// Plot 2: Volatility comparison
	plot_volatility_comparison(
		dates,
		rolling_vol,
		ewma_vol,
		garch_vol,
		"garch_vol_comparison.png",
		allocator,
	)

	fmt.println("\n✓ GARCH Real Data test completed!")
}

plot_returns_with_vol :: proc(
	dates: []f64,
	returns: []f64,
	garch_vol: []f64,
	output_path: string,
	allocator: mem.Allocator,
) {
	n := len(dates)

	// Upper and lower bands (±2σ)
	upper := make([]f64, n, allocator)
	lower := make([]f64, n, allocator)
	defer {
		delete(upper, allocator)
		delete(lower, allocator)
	}

	for i in 0 ..< n {
		upper[i] = 2.0 * garch_vol[i]
		lower[i] = -2.0 * garch_vol[i]
	}

	lines := []p.LineData {
		p.LineData{xs = dates, ys = returns, color = p.BLUE, style = .Solid, label = "Returns"},
		p.LineData{xs = dates, ys = upper, color = p.RED, style = .Dashed, label = "±2σ GARCH"},
		p.LineData{xs = dates, ys = lower, color = p.RED, style = .Dashed, label = ""},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "SPY Returns with GARCH Volatility Bands"
	config.x_label = "Time (days)"
	config.y_label = "Daily Log Return"
	config.show_grid = true

	p.multi_line_png(lines, output_path, config, allocator)
	fmt.printf("✓ Saved: %s\n", output_path)
}

plot_volatility_comparison :: proc(
	dates: []f64,
	rolling_vol: []f64,
	ewma_vol: []f64,
	garch_vol: []f64,
	output_path: string,
	allocator: mem.Allocator,
) {
	// Convert to annualized %
	n := len(dates)
	scale := math.sqrt_f64(252) * 100

	rolling_ann := make([]f64, n, allocator)
	ewma_ann := make([]f64, n, allocator)
	garch_ann := make([]f64, n, allocator)
	defer {
		delete(rolling_ann, allocator)
		delete(ewma_ann, allocator)
		delete(garch_ann, allocator)
	}

	for i in 0 ..< n {
		rolling_ann[i] = rolling_vol[i] * scale
		ewma_ann[i] = ewma_vol[i] * scale
		garch_ann[i] = garch_vol[i] * scale
	}

	lines := []p.LineData {
		p.LineData {
			xs = dates,
			ys = rolling_ann,
			color = p.BLUE,
			style = .Solid,
			label = "Rolling 20d",
		},
		p.LineData {
			xs = dates,
			ys = ewma_ann,
			color = p.RED,
			style = .Solid,
			label = "EWMA (α=0.06)",
		},
		p.LineData {
			xs = dates,
			ys = garch_ann,
			color = p.Color{0, 150, 0, 255},
			style = .Solid,
			label = "GARCH(1,1)",
		},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Volatility Estimates Comparison (Annualized %)"
	config.x_label = "Time (days)"
	config.y_label = "Volatility (%)"
	config.show_grid = true

	p.multi_line_png(lines, output_path, config, allocator)
	fmt.printf("✓ Saved: %s\n", output_path)
}
