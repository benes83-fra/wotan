package tests

import bfin "../wotan/behavoral_finance"
import w "../wotan/core"
import fin "../wotan/finance"
import net "../wotan/net"
import plot "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

market_implied_psychology_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Market-Implied Psychology Calibration ===")
	fmt.println("Reverse-engineering investor bias from a synthetic option chain.")

	S := 100.0
	r := 0.05
	T := 0.5 // 6 months
	true_sigma := 0.20 // The actual, physical volatility of the asset

	// 1. Simulate a "Panicked" Market
	// The true vol is 20%, but investors are terrified of a crash.
	// They distort probabilities with gamma = 0.40 (extreme tail overweighting).
	panic_params := bfin.Prospect_Params {
		alpha  = 0.88,
		beta   = 0.88,
		lambda = 2.25,
		gamma  = 0.40,
	}

	strikes := []f64{70, 80, 90, 100, 110, 120, 130}
	market_data := make([]fin.VolSurfacePoint, len(strikes), allocator)
	defer delete(market_data, allocator)

	fmt.println("\nGenerating synthetic 'Panicked' market prices...")
	for K, i in strikes {
		// The market prices options using distorted probabilities
		panicked_price := bfin.prospect_theory_price(
			S,
			K,
			T,
			r,
			true_sigma,
			panic_params,
			.Call,
			500,
			allocator,
		)

		market_data[i] = fin.VolSurfacePoint {
			strike       = K,
			expiry       = T,
			market_price = panicked_price,
			implied_vol  = 0.0, // We will let the calibrator use the raw price
		}
	}

	// 2. Calibrate Prospect Theory to the Market
	fmt.println("Running Adam optimizer to extract Market-Implied Psychology...")
	result := bfin.calibrate_prospect_theory(market_data, S, r, true_sigma, allocator)

	fmt.println("\n--- Calibration Results ---")
	fmt.printf("Converged: %v (RMSE: $%.4f)\n", result.converged, result.rmse)
	fmt.println("\nParameter | True (TK92) | Market-Implied | Interpretation")
	fmt.println("----------|-------------|----------------|---------------")
	fmt.printf(
		"Alpha     |    0.88     |      %.2f      | Gain sensitivity\n",
		result.params.alpha,
	)
	fmt.printf("Beta      |    0.88     |      %.2f      | Loss sensitivity\n", result.params.beta)
	fmt.printf("Lambda    |    2.25     |      %.2f      | Loss Aversion\n", result.params.lambda)
	fmt.printf(
		"Gamma     |    0.61     |      %.2f      | Probability Weighting\n",
		result.params.gamma,
	)

	fmt.println("\n💡 Insight:")
	if result.params.gamma < 0.55 {
		fmt.println("The market is exhibiting EXTREME CRASHOPHOBIA.")
		fmt.println("Investors are heavily overweighting the probability of a left-tail event.")
	} else if result.params.gamma > 0.70 {
		fmt.println("The market is highly rational / risk-neutral.")
	}

	fmt.println("\n=== Market-Implied Psychology Test Complete ===\n")
}

real_market_psychology_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Real Market-Implied Psychology (Live SPY Options) ===")
	fmt.println("Extracting investor bias from live Yahoo Finance data.")

	symbol := "SPY"
	r := 0.05 // Approximate current risk-free rate (e.g., 5%)

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

	// Calculate Historical Volatility (annualized) to use as the "True" physical sigma
	// Calculate Historical Volatility (annualized) to use as the "True" physical sigma
	// ✅ FIX: Pre-allocate a fixed slice and track the index instead of using append on a slice
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

	// Slice to the actual number of valid returns
	valid_returns := returns[:ret_idx]

	mean_ret := 0.0
	for ret in valid_returns {mean_ret += ret}
	if len(valid_returns) > 0 {
		mean_ret /= f64(len(valid_returns))
	}

	var_ret := 0.0
	for ret in valid_returns {var_ret += (ret - mean_ret) * (ret - mean_ret)}
	if len(valid_returns) > 0 {
		var_ret /= f64(len(valid_returns))
	}

	daily_vol := math.sqrt(var_ret)
	true_sigma := daily_vol * math.sqrt_f64(252.0)
	fmt.printf("Historical Volatility (true_sigma): %.2f%%\n", true_sigma * 100)

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
		fmt.println(
			"Failed to fetch options chain. (Yahoo Finance API might be blocking the request).",
		)
		return
	}
	fmt.printf("Fetched %d total options.\n", chain.n_options)

	// 3. Filter to a single expiry and Calls (to match the calibrator's .Call assumption)
	// We'll pick the expiry closest to ~45 days out to capture standard monthly behavior
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

	if target_exp == 0.0 {
		fmt.println("Could not find a suitable call option expiration.")
		return
	}

	fmt.printf("Target Expiry: %.2f years (~%.0f days)\n", target_exp, target_exp * 365.25)

	market_data := make([dynamic]fin.VolSurfacePoint, 0, allocator)
	defer delete(market_data)

	for i in 0 ..< chain.n_options {
		// Filter: Same expiry, Call options, and within 85% to 115% moneyness
		// to avoid illiquid deep OTM/ITM where bid-ask spread ruins the mid-price
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
		fmt.println("Not enough liquid options in the target range to calibrate.")
		return
	}

	fmt.printf("Filtered to %d liquid Call options for calibration.\n", len(market_data))

	// 4. Calibrate Prospect Theory to the Real Market
	fmt.println("- Running Adam optimizer to extract Market-Implied Psychology...")
	// Note: We pass market_data[:] to convert the dynamic array to a slice
	result := bfin.calibrate_prospect_theory(market_data[:], S, r, true_sigma, allocator)

	fmt.println("\n--- Live Market Calibration Results ---")
	fmt.printf("Converged: %v (RMSE: $%.4f)\n", result.converged, result.rmse)
	fmt.println("\nParameter | TK92 Baseline | Live Market (SPY) | Interpretation")
	fmt.println("----------|---------------|-------------------|---------------")
	fmt.printf(
		"Alpha     |     0.88      |       %.2f        | Gain sensitivity\n",
		result.params.alpha,
	)
	fmt.printf(
		"Beta      |     0.88      |       %.2f        | Loss sensitivity\n",
		result.params.beta,
	)
	fmt.printf(
		"Lambda    |     2.25      |       %.2f        | Loss Aversion\n",
		result.params.lambda,
	)
	fmt.printf(
		"Gamma     |     0.61      |       %.2f        | Probability Weighting\n",
		result.params.gamma,
	)

	fmt.println("\n💡 Live Market Insight:")
	if result.params.gamma < 0.55 {
		fmt.println("The options market is pricing in EXTREME CRASHOPHOBIA.")
		fmt.println("Investors are paying a massive premium for tail protection,")
		fmt.println("which the model interprets as heavy left-tail probability weighting.")
	} else if result.params.gamma > 0.70 {
		fmt.println("The options market is relatively rational / close to risk-neutral.")
		fmt.println("The volatility smile is mild.")
	} else {
		fmt.println(
			"The market exhibits standard behavioral biases (moderate tail overweighting).",
		)
	}
	plot_prospect_curves(result.params, allocator)
	fmt.println("\n=== Real Market-Implied Psychology Test Complete ===\n")
}

plot_prospect_curves :: proc(market_params: bfin.Prospect_Params, allocator: mem.Allocator) {
	fmt.println("\n=== Generating High-Res Prospect Curve PNGs ===")

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
			label = "Live Market (SPY)",
		},
	}

	val_config := plot.DEFAULT_PLOT_CONFIG
	val_config.title = "Prospect Theory Value Function v(x)"
	val_config.x_label = "Payoff (x)"
	val_config.y_label = "Subjective Value v(x)"
	val_config.width = 800
	val_config.height = 600
	val_config.show_grid = true

	if plot.multi_line_png(val_lines, "prospect_value_curve.png", val_config, allocator) {
		fmt.println("Saved: prospect_value_curve.png")
	} else {
		fmt.println("Failed to save value curve PNG.")
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
		// Avoid exactly 0.0 or 1.0 to prevent math.pow domain errors with fractional exponents
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
			label = "Live Market (SPY)",
		},
	}

	prob_config := plot.DEFAULT_PLOT_CONFIG
	prob_config.title = "Probability Weighting Function w(p)"
	prob_config.x_label = "Objective Probability (p)"
	prob_config.y_label = "Subjective Weight w(p)"
	prob_config.width = 800
	prob_config.height = 600
	prob_config.show_grid = true

	if plot.multi_line_png(prob_lines, "prospect_weighting_curve.png", prob_config, allocator) {
		fmt.println("Saved: prospect_weighting_curve.png")
	} else {
		fmt.println("Failed to save weighting curve PNG.")
	}

	fmt.println("=== PNG Generation Complete ===\n")
}
