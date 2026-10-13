package tests

import bfin "../wotan/behavioral_finance"
import w "../wotan/core"
import fin "../wotan/finance"
import net "../wotan/net"
import plot "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Helper: Plot Behavioral Heston Curves
// ============================================================================
plot_behavioral_heston_curves :: proc(
	market_params: bfin.Prospect_Params,
	allocator: mem.Allocator,
) {
	fmt.println("\n=== Generating Behavioral Heston PNGs ===")

	tk92 := bfin.Prospect_Params {
		alpha  = 0.88,
		beta   = 0.88,
		lambda = 2.25,
		gamma  = 0.61,
	}
	n_points := 200

	// --- 1. Value Function v(x) ---
	x_vals := make([]f64, n_points, allocator)
	y_tk92_val := make([]f64, n_points, allocator)
	y_market_val := make([]f64, n_points, allocator)
	defer {delete(x_vals, allocator); delete(y_tk92_val, allocator)
		delete(y_market_val, allocator)}

	for i in 0 ..< n_points {
		x := -3.0 + 6.0 * f64(i) / f64(n_points - 1)
		x_vals[i] = x

		if x >= 0 {
			y_tk92_val[i] = math.pow(x, tk92.alpha)
			y_market_val[i] = math.pow(x, market_params.alpha)
		} else {
			y_tk92_val[i] = -tk92.lambda * math.pow(-x, tk92.beta)
			y_market_val[i] = -market_params.lambda * math.pow(-x, market_params.beta)
		}
	}

	val_lines := []plot.LineData {
		{xs = x_vals, ys = y_tk92_val, color = plot.BLUE, style = .Solid, label = "TK92 Baseline"},
		{
			xs = x_vals,
			ys = y_market_val,
			color = plot.RED,
			style = .Dashed,
			label = "Behav. Heston (Live)",
		},
	}

	val_config := plot.DEFAULT_PLOT_CONFIG
	val_config.title = "Behavioral Heston: Value Function v(x)"
	val_config.x_label = "Payoff (x)"
	val_config.y_label = "Subjective Value v(x)"
	val_config.width = 800
	val_config.height = 600
	val_config.show_grid = true

	if plot.multi_line_png(val_lines, "b_heston_value_curve.png", val_config, allocator) {
		fmt.println("Saved: b_heston_value_curve.png")
	}

	// --- 2. Probability Weighting w(p) ---
	p_vals := make([]f64, n_points, allocator)
	y_tk92_prob := make([]f64, n_points, allocator)
	y_market_prob := make([]f64, n_points, allocator)
	y_rational := make([]f64, n_points, allocator)
	defer {delete(p_vals, allocator); delete(y_tk92_prob, allocator)
		delete(y_market_prob, allocator)
		delete(y_rational, allocator)}

	for i in 0 ..< n_points {
		p := 0.001 + 0.998 * f64(i) / f64(n_points - 1)
		p_vals[i] = p
		y_rational[i] = p

		p_g_tk := math.pow(p, tk92.gamma)
		q_g_tk := math.pow(1.0 - p, tk92.gamma)
		y_tk92_prob[i] = p_g_tk / math.pow(p_g_tk + q_g_tk, 1.0 / tk92.gamma)

		p_g_mk := math.pow(p, market_params.gamma)
		q_g_mk := math.pow(1.0 - p, market_params.gamma)
		y_market_prob[i] = p_g_mk / math.pow(p_g_mk + q_g_mk, 1.0 / market_params.gamma)
	}

	prob_lines := []plot.LineData {
		{
			xs = p_vals,
			ys = y_rational,
			color = plot.Color{128, 128, 128, 255},
			style = .Dotted,
			label = "Rational (w=p)",
		},
		{
			xs = p_vals,
			ys = y_tk92_prob,
			color = plot.BLUE,
			style = .Solid,
			label = "TK92 Baseline",
		},
		{
			xs = p_vals,
			ys = y_market_prob,
			color = plot.RED,
			style = .Dashed,
			label = "Behav. Heston (Live)",
		},
	}

	prob_config := plot.DEFAULT_PLOT_CONFIG
	prob_config.title = "Behavioral Heston: Probability Weighting w(p)"
	prob_config.x_label = "Objective Probability (p)"
	prob_config.y_label = "Subjective Weight w(p)"
	prob_config.width = 800
	prob_config.height = 600
	prob_config.show_grid = true

	if plot.multi_line_png(prob_lines, "b_heston_weighting_curve.png", prob_config, allocator) {
		fmt.println("Saved: b_heston_weighting_curve.png")
	}
}

// ============================================================================
// Main Test: Behavioral Heston (Physics + Psychology)
// ============================================================================
behavioral_heston_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Behavioral Heston Test (Physics + Psychology) ===")
	fmt.println("Separating Stochastic Volatility (Physics) from Investor Bias (Psychology).")
	fmt.println("WARNING: This involves heavy Fourier inversion. Compile with `-o:speed`!")

	symbol := "SPY"
	r := 0.05

	// 1. Fetch Underlying Data for S and Historical Volatility
	fmt.printf("- Fetching %s historical data...\n", symbol)
	df := net.read_yahoo(symbol, .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows < 10 {
		fmt.println("Failed to fetch sufficient historical data.")
		return
	}

	close_col := w.column(&df, "Close")
	n_days := df.rows
	S, _ := w.column_at_float(close_col, n_days - 1)
	fmt.printf("Current Underlying Price (S): $%.2f\n", S)

	// Calculate Historical Volatility
	returns := make([]f64, n_days, allocator)
	defer delete(returns, allocator)
	ret_idx := 0
	prev_close := 0.0
	for i in 0 ..< n_days {
		c, _ := w.column_at_float(close_col, i)
		if i > 0 && prev_close > 0 {
			returns[ret_idx] = math.ln_f64(c / prev_close)
			ret_idx += 1
		}
		prev_close = c
	}
	valid_returns := returns[:ret_idx]
	mean_ret := 0.0
	for ret in valid_returns {mean_ret += ret}
	if len(valid_returns) > 0 {mean_ret /= f64(len(valid_returns))}
	var_ret := 0.0
	for ret in valid_returns {var_ret += (ret - mean_ret) * (ret - mean_ret)}
	if len(valid_returns) > 0 {var_ret /= f64(len(valid_returns))}
	true_sigma := math.sqrt(var_ret) * math.sqrt_f64(252.0)
	fmt.printf("Historical Volatility: %.2f%%\n", true_sigma * 100)

	// 2. Fetch Live Options Chain
	fmt.printf("- Fetching %s live options chain...\n", symbol)
	chain := fin.fetch_yahoo_options(symbol, allocator)
	defer {
		if chain.strikes != nil {delete(chain.strikes, allocator)}
		if chain.expiries != nil {delete(chain.expiries, allocator)}
		if chain.implied_vols != nil {delete(chain.implied_vols, allocator)}
		if chain.market_prices != nil {delete(chain.market_prices, allocator)}
		if chain.option_types != nil {delete(chain.option_types, allocator)}
	}

	if chain.n_options == 0 {
		fmt.println("Failed to fetch options chain.")
		return
	}

	// Filter to ~45 DTE Calls
	target_exp := 0.0
	min_time_diff: f64 = 1e9
	target_dte := 45.0
	for i in 0 ..< chain.n_options {
		dte := chain.expiries[i] * 365.25
		diff: f64 = math.abs(dte - target_dte)
		if diff < min_time_diff && chain.option_types[i] == "call" {
			min_time_diff = diff
			target_exp = chain.expiries[i]
		}
	}

	market_data: [dynamic]fin.VolSurfacePoint
	defer delete(market_data)

	for i in 0 ..< chain.n_options {
		if math.abs(chain.expiries[i] - target_exp) < 1e-5 && chain.option_types[i] == "call" {
			moneyness := chain.strikes[i] / S
			if moneyness >= 0.85 && moneyness <= 1.15 && chain.market_prices[i] > 0.05 {
				append(
					&market_data,
					fin.VolSurfacePoint {
						strike = chain.strikes[i],
						expiry = chain.expiries[i],
						market_price = chain.market_prices[i],
						implied_vol = chain.implied_vols[i],
					},
				)
			}
		}
	}

	if len(market_data) < 3 {
		fmt.println("Not enough liquid options.")
		return
	}
	fmt.printf("Filtered to %d liquid Call options.\n", len(market_data))

	// 3. Calibrate Pure Heston (The Physics)
	fmt.println("\n--- Step 1: Calibrating Pure Heston (Stochastic Volatility) ---")
	// ✅ PASS true_sigma to seed the optimizer intelligently
	heston_result := fin.calibrate_heston(market_data[:], S, r, true_sigma, allocator)
	fmt.printf("Pure Heston RMSE: $%.4f\n", heston_result.rmse)
	fmt.printf(
		"Heston Params: v0=%.4f, kappa=%.2f, theta=%.4f, sigma=%.2f, rho=%.2f\n",
		heston_result.params.v0,
		heston_result.params.kappa,
		heston_result.params.theta,
		heston_result.params.sigma,
		heston_result.params.rho,
	)

	// 4. Calibrate Behavioral Heston (The Psychology)
	fmt.println("\n--- Step 2: Calibrating Behavioral Heston (Residual Psychology) ---")
	fmt.println("(Running Adam optimizer over Heston characteristic functions...)")
	b_heston_result := bfin.calibrate_behavioral_heston(
		market_data[:],
		S,
		r,
		heston_result.params,
		allocator,
	)

	// 5. Print Results
	fmt.println("\n--- Behavioral Heston Results ---")
	fmt.printf(
		"Behavioral Heston RMSE: $%.4f (Pure Heston was $%.4f)\n",
		b_heston_result.rmse,
		heston_result.rmse,
	)
	fmt.println("\nParameter | TK92 Baseline | Behav. Heston | Interpretation")
	fmt.println("----------|---------------|---------------|---------------")
	fmt.printf(
		"Alpha     |     0.88      |     %.2f      | Gain sensitivity\n",
		b_heston_result.params.alpha,
	)
	fmt.printf(
		"Beta      |     0.88      |     %.2f      | Loss sensitivity\n",
		b_heston_result.params.beta,
	)
	fmt.printf(
		"Lambda    |     2.25      |     %.2f      | Loss Aversion\n",
		b_heston_result.params.lambda,
	)
	fmt.printf(
		"Gamma     |     0.61      |     %.2f      | Probability Weighting\n",
		b_heston_result.params.gamma,
	)

	// 6. Insight
	fmt.println("\n💡 Quant Insight:")
	if b_heston_result.params.gamma > 0.70 {
		fmt.println(
			"SUCCESS: Heston absorbed the structural skew! The residual gamma is near rational.",
		)
		fmt.println(
			"The options market's skew is primarily driven by stochastic volatility (leverage effect),",
		)
		fmt.println("not irrational crashophobia.")
	} else if b_heston_result.params.gamma < 0.55 {
		fmt.println(
			"WARNING: Even after accounting for stochastic volatility, investors are irrationally",
		)
		fmt.println("overweighting tail risk. The fear is purely psychological.")
	} else {
		fmt.println(
			"The skew is a mix of both stochastic volatility physics and moderate behavioral bias.",
		)
	}

	// 7. Plotting
	plot_behavioral_heston_curves(b_heston_result.params, allocator)

	fmt.println("\n=== Behavioral Heston Test Complete ===\n")
}
