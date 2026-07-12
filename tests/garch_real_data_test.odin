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
	// 3.b Fit GARCH(1,1) with GED distribution
	fmt.println("\n--- Fitting GARCH(1,1) with GED Errors ---")
	residuals2 := ts.extract_residuals(returns, main_alloc)
	defer delete(residuals2, main_alloc)

	garch_result2 := ts.garch_fit(residuals2, .GED, 1, 1, 2000, 1e-4, main_alloc)
	defer {
		delete(garch_result2.params.alpha, main_alloc)
		delete(garch_result2.params.beta, main_alloc)
		delete(garch_result2.conditional_var, main_alloc)
		delete(garch_result2.standardized_resid, main_alloc)
	}

	fmt.printf("\nGARCH(1,1) GED Parameters:\n")
	fmt.printf("  ω (omega): %.8f\n", garch_result2.params.omega)
	fmt.printf("  α (alpha): %.4f\n", garch_result2.params.alpha[0])
	fmt.printf("  β (beta):  %.4f\n", garch_result2.params.beta[0])
	fmt.printf("  Shape (p): %.4f\n", garch_result2.params.shape)
	fmt.printf("  Persistence (α+β): %.4f\n", garch_result2.persistence)

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
	// 6b. Historical Simulation (Rolling Window)
	fmt.println("\n--- Historical Simulation VaR ---")
	hist_window := 252 // 1 year rolling window
	var_95_hist := fin.historical_var_rolling(returns, 0.95, hist_window, main_alloc)
	var_99_hist := fin.historical_var_rolling(returns, 0.99, hist_window, main_alloc)
	cvar_95_hist := fin.historical_cvar_rolling(returns, 0.95, hist_window, main_alloc)
	cvar_99_hist := fin.historical_cvar_rolling(returns, 0.99, hist_window, main_alloc)
	defer {
		delete(var_95_hist, main_alloc)
		delete(var_99_hist, main_alloc)
		delete(cvar_95_hist, main_alloc)
		delete(cvar_99_hist, main_alloc)
	}

	// Backtest Historical Simulation
	start_idx_hist := hist_window // Use window size as warmup
	returns_bt_hist := returns[start_idx_hist:]
	var_95_hist_bt := var_95_hist[start_idx_hist:]
	var_99_hist_bt := var_99_hist[start_idx_hist:]

	backtest_hist_95 := fin.backtest_var(returns_bt_hist, var_95_hist_bt, 0.95)
	backtest_hist_99 := fin.backtest_var(returns_bt_hist, var_99_hist_bt, 0.99)

	fmt.printf("\n%-30s %-15s %-15s\n", "Method", "95% VaR", "99% VaR")
	fmt.printf(
		"%-30s %-15s %-15s\n",
		"----------------------------",
		"---------------",
		"---------------",
	)

	// GARCH results
	pass_g95 := "PASS"
	if !backtest_95.passes_test {pass_g95 = "FAIL"}
	pass_g99 := "PASS"
	if !backtest_99.passes_test {pass_g99 = "FAIL"}
	fmt.printf(
		"%-30s %d (%.2f%%) %s   %d (%.2f%%) %s\n",
		"GARCH(1,1) Student-t",
		backtest_95.n_breaches,
		backtest_95.breach_rate * 100,
		pass_g95,
		backtest_99.n_breaches,
		backtest_99.breach_rate * 100,
		pass_g99,
	)

	// Historical results
	pass_h95 := "PASS"
	if !backtest_hist_95.passes_test {pass_h95 = "FAIL"}
	pass_h99 := "PASS"
	if !backtest_hist_99.passes_test {pass_h99 = "FAIL"}
	fmt.printf(
		"%-30s %d (%.2f%%) %s   %d (%.2f%%) %s\n",
		"Historical Simulation",
		backtest_hist_95.n_breaches,
		backtest_hist_95.breach_rate * 100,
		pass_h95,
		backtest_hist_99.n_breaches,
		backtest_hist_99.breach_rate * 100,
		pass_h99,
	)
	// 6c. Extreme Value Theory (EVT)
	fmt.println("\n--- Extreme Value Theory (EVT) ---")
	evt_result := fin.evt_fit(returns, 0.95, main_alloc) // 95th percentile threshold

	fmt.printf("\nEVT Model Parameters:\n")
	fmt.printf("  Threshold (u): %.4f%%\n", evt_result.threshold * 100)
	fmt.printf("  Shape (ξ): %.4f\n", evt_result.xi)
	fmt.printf("  Scale (β): %.6f\n", evt_result.beta)
	fmt.printf(
		"  Exceedances: %d / %d (%.2f%%)\n",
		evt_result.n_exceedances,
		evt_result.n_total,
		f64(evt_result.n_exceedances) / f64(evt_result.n_total) * 100,
	)
	fmt.printf("  Converged: %v\n", evt_result.converged)

	fmt.printf("\nEVT Risk Metrics:\n")
	fmt.printf("  95%% VaR:  %.4f%%\n", evt_result.var_95 * 100)
	fmt.printf("  99%% VaR:  %.4f%%\n", evt_result.var_99 * 100)
	fmt.printf("  95%% CVaR: %.4f%%\n", evt_result.cvar_95 * 100)
	fmt.printf("  99%% CVaR: %.4f%%\n", evt_result.cvar_99 * 100)
	// 6d. GARCH-EVT Hybrid Model
	fmt.println("\n--- GARCH-EVT Hybrid Model ---")
	// Use 90th percentile threshold for the standardized residuals
	hybrid_result := fin.garch_evt_fit(returns, 0.90, main_alloc)
	defer {
		delete(hybrid_result.var_95_series, main_alloc)
		delete(hybrid_result.var_99_series, main_alloc)
	}

	fmt.printf("\nGARCH-EVT Parameters:\n")
	fmt.printf("  GARCH ω: %.8f\n", hybrid_result.garch_omega)
	fmt.printf("  GARCH α: %.4f\n", hybrid_result.garch_alpha)
	fmt.printf("  GARCH β: %.4f\n", hybrid_result.garch_beta)
	fmt.printf("  EVT ξ (shape): %.4f\n", hybrid_result.evt_xi)
	fmt.printf("  EVT β (scale): %.6f\n", hybrid_result.evt_beta)
	fmt.printf("  Threshold (u): %.4f\n", hybrid_result.evt_threshold)
	fmt.printf("  Exceedances: %d / %d\n", hybrid_result.n_exceedances, len(returns))

	fmt.printf("\nGARCH-EVT Backtesting:\n")

	pass_h95h := "PASS"
	if !hybrid_result.backtest_95.passes_test {pass_h95h = "FAIL"}
	pass_h99h := "PASS"
	if !hybrid_result.backtest_99.passes_test {pass_h99h = "FAIL"}

	fmt.printf(
		"  95%% VaR: %d breaches (%.2f%%) - p-value: %.4f (%s)\n",
		hybrid_result.backtest_95.n_breaches,
		hybrid_result.backtest_95.breach_rate * 100,
		hybrid_result.backtest_95.kupiec_pvalue,
		pass_h95h,
	)
	fmt.printf(
		"  99%% VaR: %d breaches (%.2f%%) - p-value: %.4f (%s)\n",
		hybrid_result.backtest_99.n_breaches,
		hybrid_result.backtest_99.breach_rate * 100,
		hybrid_result.backtest_99.kupiec_pvalue,
		pass_h99h,
	)

	// Add Hybrid to the final comparison table
	fmt.printf(
		"\n%-25s %-15s %-15s %-15s %-15s\n",
		"Metric",
		"GARCH",
		"Historical",
		"EVT",
		"GARCH-EVT",
	)
	fmt.printf(
		"%-25s %-15s %-15s %-15s %-15s\n",
		"-------------------------",
		"---------------",
		"---------------",
		"---------------",
		"---------------",
	)

	last_idx := len(returns) - 1
	hybrid_var_95 := hybrid_result.var_95_series[last_idx]
	hybrid_var_99 := hybrid_result.var_99_series[last_idx]

	fmt.printf(
		"%-25s %-15.4f%% %-15.4f%% %-15.4f%% %-15.4f%%\n",
		"95% VaR",
		var_95[last_idx] * 100,
		var_95_hist[last_idx] * 100,
		evt_result.var_95 * 100,
		hybrid_var_95 * 100,
	)
	fmt.printf(
		"%-25s %-15.4f%% %-15.4f%% %-15.4f%% %-15.4f%%\n",
		"99% VaR",
		var_99[last_idx] * 100,
		var_99_hist[last_idx] * 100,
		evt_result.var_99 * 100,
		hybrid_var_99 * 100,
	)
	// 6e. Copula Analysis (Bivariate Dependence)
	fmt.println("\n--- Copula Analysis ---")

	// Fetch VIX data for bivariate analysis
	fmt.println("Fetching VIX data...")
	vix_df := yahoo.read_yahoo("^VIX", .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&vix_df)

	// Compute VIX returns
	n_vix := vix_df.rows
	vix_returns := make([]f64, n_vix - 1, main_alloc)
	defer delete(vix_returns, main_alloc)

	for i in 1 ..< n_vix {
		prev_close, _ := w.column_at_float(&vix_df.columns[4], i - 1)
		curr_close, _ := w.column_at_float(&vix_df.columns[4], i)
		vix_returns[i - 1] = math.ln_f64(curr_close / prev_close)
	}

	// Align lengths
	n_common := min(len(returns), len(vix_returns))
	spy_aligned := returns[:n_common]
	vix_aligned := vix_returns[:n_common]

	fmt.printf("Aligned %d observations\n", n_common)

	// Transform to uniform marginals
	u_spy := fin.pit_empirical(spy_aligned, main_alloc)
	u_vix := fin.pit_empirical(vix_aligned, main_alloc)
	defer {
		delete(u_spy, main_alloc)
		delete(u_vix, main_alloc)
	}

	// Compute empirical dependence measures
	tau := fin.kendall_tau(spy_aligned, vix_aligned)
	rho_spearman := fin.spearman_rho(spy_aligned, vix_aligned, main_alloc)
	ltd := fin.lower_tail_dependence(u_spy, u_vix, 0.05)
	utd := fin.upper_tail_dependence(u_spy, u_vix, 0.95)

	fmt.printf("\nEmpirical Dependence Measures:\n")
	fmt.printf("  Kendall's τ: %.4f\n", tau)
	fmt.printf("  Spearman's ρ: %.4f\n", rho_spearman)
	fmt.printf("  Lower Tail Dependence (5%%): %.4f\n", ltd)
	fmt.printf("  Upper Tail Dependence (95%%): %.4f\n", utd)

	// Fit Gaussian copula
	fmt.println("\nFitting Gaussian Copula...")
	gauss_copula := fin.fit_gaussian_copula(u_spy, u_vix, main_alloc)
	defer delete(gauss_copula.parameters, main_alloc)

	fmt.printf("\nGaussian Copula:\n")
	fmt.printf("  ρ (correlation): %.4f\n", gauss_copula.parameters[0])
	fmt.printf("  Log-Likelihood: %.2f\n", gauss_copula.log_likelihood)
	fmt.printf("  AIC: %.2f\n", gauss_copula.aic)
	fmt.printf("  BIC: %.2f\n", gauss_copula.bic)

	// Fit Student-t copula
	fmt.println("\nFitting Student-t Copula...")
	student_copula := fin.fit_student_t_copula(u_spy, u_vix, main_alloc)
	defer delete(student_copula.parameters, main_alloc)

	fmt.printf("\nStudent-t Copula:\n")
	fmt.printf("  ρ (correlation): %.4f\n", student_copula.parameters[0])
	fmt.printf("  ν (degrees of freedom): %.2f\n", student_copula.parameters[1])
	fmt.printf("  Log-Likelihood: %.2f\n", student_copula.log_likelihood)
	fmt.printf("  AIC: %.2f\n", student_copula.aic)
	fmt.printf("  BIC: %.2f\n", student_copula.bic)
	// Fit Clayton Copula
	fmt.println("\nFitting Clayton Copula...")
	clayton_copula := fin.fit_clayton(u_spy, u_vix, main_alloc)
	defer delete(clayton_copula.parameters, main_alloc)

	if clayton_copula.converged {
		fmt.printf("\nClayton Copula:\n")
		fmt.printf("  θ (dependence): %.4f\n", clayton_copula.parameters[0])
		fmt.printf("  Log-Likelihood: %.2f\n", clayton_copula.log_likelihood)
		fmt.printf("  AIC: %.2f\n", clayton_copula.aic)
	} else {
		fmt.printf("\nClayton Copula: Skipped (requires positive dependence)\n")
	}

	// Fit Gumbel Copula
	fmt.println("\nFitting Gumbel Copula...")
	gumbel_copula := fin.fit_gumbel(u_spy, u_vix, main_alloc)
	defer delete(gumbel_copula.parameters, main_alloc)

	if gumbel_copula.converged {
		fmt.printf("\nGumbel Copula:\n")
		fmt.printf("  θ (dependence): %.4f\n", gumbel_copula.parameters[0])
		fmt.printf("  Log-Likelihood: %.2f\n", gumbel_copula.log_likelihood)
		fmt.printf("  AIC: %.2f\n", gumbel_copula.aic)
	} else {
		fmt.printf("\nGumbel Copula: Skipped (requires positive dependence)\n")
	}

	// Fit Frank Copula
	fmt.println("\nFitting Frank Copula...")
	frank_copula := fin.fit_frank(u_spy, u_vix, main_alloc)
	defer delete(frank_copula.parameters, main_alloc)

	fmt.printf("\nFrank Copula:\n")
	fmt.printf("  θ (dependence): %.4f\n", frank_copula.parameters[0])
	fmt.printf("  Log-Likelihood: %.2f\n", frank_copula.log_likelihood)
	fmt.printf("  AIC: %.2f\n", frank_copula.aic)
	fmt.printf("  BIC: %.2f\n", frank_copula.bic)

	// Comprehensive Comparison Table
	fmt.printf("\n%-15s %-15s %-15s %-15s\n", "Copula", "θ / ρ", "LogLik", "AIC")
	fmt.printf(
		"%-15s %-15s %-15s %-15s\n",
		"---------------",
		"---------------",
		"---------------",
		"---------------",
	)

	// Gaussian
	fmt.printf(
		"%-15s ρ=%.4f          %-15.2f %-15.2f\n",
		"Gaussian",
		gauss_copula.parameters[0],
		gauss_copula.log_likelihood,
		gauss_copula.aic,
	)

	// Student-t
	fmt.printf(
		"%-15s =%.4f, ν=%.1f  %-15.2f %-15.2f\n",
		"Student-t",
		student_copula.parameters[0],
		student_copula.parameters[1],
		student_copula.log_likelihood,
		student_copula.aic,
	)

	// Clayton (if converged)
	if clayton_copula.converged {
		fmt.printf(
			"%-15s θ=%.4f          %-15.2f %-15.2f\n",
			"Clayton",
			clayton_copula.parameters[0],
			clayton_copula.log_likelihood,
			clayton_copula.aic,
		)
	} else {
		fmt.printf("%-15s %-15s %-15s %-15s\n", "Clayton", "N/A (neg dep)", "-", "-")
	}

	// Gumbel (if converged)
	if gumbel_copula.converged {
		fmt.printf(
			"%-15s θ=%.4f          %-15.2f %-15.2f\n",
			"Gumbel",
			gumbel_copula.parameters[0],
			gumbel_copula.log_likelihood,
			gumbel_copula.aic,
		)
	} else {
		fmt.printf("%-15s %-15s %-15s %-15s\n", "Gumbel", "N/A (neg dep)", "-", "-")
	}

	// Frank
	fmt.printf(
		"%-15s θ=%.4f          %-15.2f %-15.2f\n",
		"Frank",
		frank_copula.parameters[0],
		frank_copula.log_likelihood,
		frank_copula.aic,
	)

	fmt.printf("\nNote: Clayton/Gumbel model only positive dependence. Frank handles negative.\n")
	// Model comparison
	fmt.printf("\n%-20s %-15s %-15s %-15s\n", "Copula", "Parameters", "AIC", "BIC")
	fmt.printf(
		"%-20s %-15s %-15s %-15s\n",
		"--------------------",
		"---------------",
		"---------------",
		"---------------",
	)
	fmt.printf(
		"%-20s ρ=%.4f          %-15.2f %-15.2f\n",
		"Gaussian",
		gauss_copula.parameters[0],
		gauss_copula.aic,
		gauss_copula.bic,
	)
	fmt.printf(
		"%-20s ρ=%.4f, ν=%.1f  %-15.2f %-15.2f\n",
		"Student-t",
		student_copula.parameters[0],
		student_copula.parameters[1],
		student_copula.aic,
		student_copula.bic,
	)

	// Interpretation
	fmt.printf("\nInterpretation:\n")
	if student_copula.aic < gauss_copula.aic {
		fmt.printf("  ✓ Student-t copula fits better (lower AIC)\n")
		fmt.printf(
			"  ✓ Evidence of tail dependence (ν = %.1f < ∞)\n",
			student_copula.parameters[1],
		)
	} else {
		fmt.printf("  ✓ Gaussian copula fits better (lower AIC)\n")
		fmt.printf("  ✓ No significant tail dependence\n")
	}

	if ltd > 0.05 {
		fmt.printf("  ⚠ Significant lower tail dependence (%.2f)\n", ltd)
		fmt.printf("    Assets tend to crash together!\n")
	}
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
	// 8b. Compare Current Risk Metrics (with EVT)
	fmt.println("\n--- Current Risk Metrics Comparison ---")
	last_idxh := len(returns) - 1

	fmt.printf("\n%-25s %-15s %-15s %-15s\n", "Metric", "GARCH", "Historical", "EVT")
	fmt.printf(
		"%-25s %-15s %-15s %-15s\n",
		"-------------------------",
		"---------------",
		"---------------",
		"---------------",
	)
	fmt.printf(
		"%-25s %-15.4f%% %-15.4f%% %-15.4f%%\n",
		"95% VaR",
		var_95[last_idxh] * 100,
		var_95_hist[last_idxh] * 100,
		evt_result.var_95 * 100,
	)
	fmt.printf(
		"%-25s %-15.4f%% %-15.4f%% %-15.4f%%\n",
		"99% VaR",
		var_99[last_idxh] * 100,
		var_99_hist[last_idxh] * 100,
		evt_result.var_99 * 100,
	)
	fmt.printf(
		"%-25s %-15.4f%% %-15.4f%% %-15.4f%%\n",
		"95% CVaR",
		cvar_95_current * 100,
		cvar_95_hist[last_idxh] * 100,
		evt_result.cvar_95 * 100,
	)
	fmt.printf(
		"%-25s %-15.4f%% %-15.4f%% %-15.4f%%\n",
		"99% CVaR",
		cvar_99_current * 100,
		cvar_99_hist[last_idxh] * 100,
		evt_result.cvar_99 * 100,
	)
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
	plot_var_comparison(dates, returns, var_95, var_95_hist, "var_comparison.png", allocator)

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
plot_var_comparison :: proc(
	dates: []f64,
	returns: []f64,
	var_garch: []f64,
	var_hist: []f64,
	output_path: string,
	allocator: mem.Allocator,
) {
	n := len(dates)

	// Convert to negative for display (losses)
	var_garch_neg := make([]f64, n, allocator)
	var_hist_neg := make([]f64, n, allocator)
	defer {
		delete(var_garch_neg, allocator)
		delete(var_hist_neg, allocator)
	}

	for i in 0 ..< n {
		var_garch_neg[i] = -var_garch[i]
		var_hist_neg[i] = -var_hist[i]
	}

	lines := []p.LineData {
		p.LineData{xs = dates, ys = returns, color = p.BLUE, style = .Solid, label = "Returns"},
		p.LineData {
			xs = dates,
			ys = var_garch_neg,
			color = p.RED,
			style = .Solid,
			label = "GARCH 95% VaR",
		},
		p.LineData {
			xs = dates,
			ys = var_hist_neg,
			color = p.Color{0, 150, 0, 255},
			style = .Dashed,
			label = "Historical 95% VaR",
		},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "SPY Returns: GARCH vs Historical Simulation VaR"
	config.x_label = "Time (days)"
	config.y_label = "Daily Return / VaR Threshold"
	config.show_grid = true

	p.multi_line_png(lines, output_path, config, allocator)
	fmt.printf("✓ Saved: %s\n", output_path)
}
