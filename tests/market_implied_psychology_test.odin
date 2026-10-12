package tests

import bfin "../wotan/behavoral_finance"
import w "../wotan/core"
import fin "../wotan/finance"
import net "../wotan/net"
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

	fmt.println("\n=== Real Market-Implied Psychology Test Complete ===\n")
}
