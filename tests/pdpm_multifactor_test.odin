package tests

import w "../wotan/core"
import fin "../wotan/finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

pdpm_multifactor_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== PDPM Multi-Factor Test with Real Data ===\n")

	// 1. Fetch real market data
	fmt.println("--- Fetching Market Data from Yahoo Finance ---")

	spy_df := yahoo.read_yahoo("SPY", .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&spy_df)

	vix_df := yahoo.read_yahoo("^VIX", .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&vix_df)

	fmt.printf("Loaded %d days of SPY data\n", spy_df.rows)
	fmt.printf("Loaded %d days of VIX data\n", vix_df.rows)

	// 2. Extract returns
	fmt.println("\n--- Computing Returns ---")

	n_days := min(spy_df.rows, vix_df.rows)
	spy_returns := make([]f64, n_days - 1, allocator)
	vix_returns := make([]f64, n_days - 1, allocator)
	defer {
		delete(spy_returns, allocator)
		delete(vix_returns, allocator)
	}
	for i in 1 ..< n_days {
		spy_close_prev, _ := w.column_at_float(&spy_df.columns[4], i - 1)
		spy_close_curr, _ := w.column_at_float(&spy_df.columns[4], i)
		spy_returns[i - 1] = (spy_close_curr - spy_close_prev) / spy_close_prev

		vix_close_prev, _ := w.column_at_float(&vix_df.columns[4], i - 1)
		vix_close_curr, _ := w.column_at_float(&vix_df.columns[4], i)
		vix_returns[i - 1] = (vix_close_curr - vix_close_prev) / vix_close_prev
	}

	// Compute historical volatility
	var_sum := 0.0
	var_sum_sq := 0.0
	for i in 0 ..< len(spy_returns) {
		ret := spy_returns[i]
		var_sum += ret
		var_sum_sq += ret * ret
	}
	mean_ret := var_sum / f64(len(spy_returns))
	variance := (var_sum_sq / f64(len(spy_returns))) - (mean_ret * mean_ret)
	hist_vol := math.sqrt_f64(variance * 252.0)

	fmt.printf("SPY Historical Volatility: %.2f%%\n", hist_vol * 100)
	fmt.printf("SPY Mean Return: %.4f%%\n", mean_ret * 100)

	// Get current spot price from the last SPY close
	spot_price, _ := w.column_at_float(&spy_df.columns[4], n_days - 1)
	fmt.printf("Current SPY Price: $%.2f\n", spot_price)

	// 3. Fetch REAL Option Chain from Yahoo Finance
	fmt.println("\n--- Fetching Real Option Chain ---")

	option_chain := fin.fetch_yahoo_options("SPY", allocator)

	// We need to defer delete the slices inside the option chain
	defer {
		delete(option_chain.strikes, allocator)
		delete(option_chain.expiries, allocator)
		delete(option_chain.implied_vols, allocator)
		delete(option_chain.market_prices, allocator)
		delete(option_chain.option_types, allocator)
	}

	fmt.printf("Loaded %d real options contracts\n", option_chain.n_options)

	// DECLARE risk_free_rate HERE so it's available for synthetic options
	risk_free_rate := 0.05

	if option_chain.n_options == 0 {
		fmt.println("\n⚠ Yahoo Options API blocked - generating synthetic option chain")

		// Clean up empty slices first
		if option_chain.strikes != nil {delete(option_chain.strikes, allocator)}
		if option_chain.expiries != nil {delete(option_chain.expiries, allocator)}
		if option_chain.implied_vols != nil {delete(option_chain.implied_vols, allocator)}
		if option_chain.market_prices != nil {delete(option_chain.market_prices, allocator)}
		if option_chain.option_types != nil {delete(option_chain.option_types, allocator)}

		// Generate synthetic options using Black-Scholes
		n_options := 12
		syn_strikes := make([]f64, n_options, allocator)
		syn_expiries := make([]f64, n_options, allocator)
		syn_ivs := make([]f64, n_options, allocator)
		syn_prices := make([]f64, n_options, allocator)
		syn_types := make([]string, n_options, allocator)

		// Use historical vol + 2% risk premium as implied vol (realistic)
		base_vol := hist_vol + 0.02
		time_to_expiry := 30.0 / 365.0 // 30 days

		for i in 0 ..< n_options {
			if i < 6 {
				syn_types[i] = "call"
				syn_strikes[i] = spot_price * (0.95 + f64(i) * 0.02)
			} else {
				syn_types[i] = "put"
				syn_strikes[i] = spot_price * (0.95 + f64(i - 6) * 0.02)
			}

			syn_expiries[i] = time_to_expiry
			syn_ivs[i] = base_vol

			// Black-Scholes pricing (use explicit _f64 suffixes)
			d1 :=
				(math.ln_f64(spot_price / syn_strikes[i]) +
					(risk_free_rate + 0.5 * base_vol * base_vol) * time_to_expiry) /
				(base_vol * math.sqrt_f64(time_to_expiry))
			d2 := d1 - base_vol * math.sqrt_f64(time_to_expiry)
			nd1 := 0.5 * (1.0 + math.erf_f64(d1 / math.sqrt_f64(2.0)))
			nd2 := 0.5 * (1.0 + math.erf_f64(d2 / math.sqrt_f64(2.0)))

			if syn_types[i] == "call" {
				syn_prices[i] =
					spot_price * nd1 -
					syn_strikes[i] * math.exp_f64(-risk_free_rate * time_to_expiry) * nd2
			} else {
				syn_prices[i] =
					syn_strikes[i] * math.exp_f64(-risk_free_rate * time_to_expiry) * (1.0 - nd2) -
					spot_price * (1.0 - nd1)
			}
		}

		option_chain = fin.OptionChain {
			strikes       = syn_strikes,
			expiries      = syn_expiries,
			implied_vols  = syn_ivs,
			market_prices = syn_prices,
			option_types  = syn_types,
			n_options     = n_options,
		}

		fmt.printf("Generated %d synthetic options (calls + puts)\n", n_options)
		fmt.printf("Implied Volatility: %.2f%% (hist vol + 2%% risk premium)\n", base_vol * 100)
	} else {
		// Real data was fetched successfully
		fmt.printf(
			"Strike Range: $%.2f - $%.2f\n",
			option_chain.strikes[0],
			option_chain.strikes[option_chain.n_options - 1],
		)

		iv_sum := 0.0
		for iv in option_chain.implied_vols {
			iv_sum += iv
		}
		avg_iv := (iv_sum / f64(option_chain.n_options)) * 100
		fmt.printf("Avg Implied Vol: %.2f%%\n", avg_iv)
	}

	// 4. Calibrate Multi-Factor SDF
	fmt.println("\n--- Calibrating Multi-Factor SDF ---")

	calibration := fin.calibrate_sdf_multifactor_2d(
		spy_returns,
		vix_returns,
		risk_free_rate,
		&option_chain,
		spot_price,
		allocator,
	)
	defer {
		delete(calibration.sdf.values, allocator)
		delete(calibration.sdf.parameters, allocator)
	}

	fmt.printf("\nMulti-Factor Calibration Results:\n")
	fmt.printf("  RMSE: $%.4f\n", calibration.rmse)
	fmt.printf(
		"  SDF Mean: %.4f (Target: %.4f)\n",
		calibration.sdf.mean,
		1.0 / (1.0 + risk_free_rate),
	)

	// 5. Compare with Single-Factor Model
	fmt.println("\n--- Comparing with Single-Factor Model ---")

	calibration_single := fin.calibrate_sdf_linear(
		spy_returns,
		risk_free_rate,
		&option_chain,
		spot_price,
		200,
		1e-6,
		allocator,
	)
	defer {
		delete(calibration_single.sdf.values, allocator)
		delete(calibration_single.sdf.parameters, allocator)
	}

	fmt.printf("Single-Factor RMSE: $%.4f\n", calibration_single.rmse)
	fmt.printf("Multi-Factor RMSE:  $%.4f\n", calibration.rmse)

	if calibration_single.rmse > 0 {
		improvement :=
			(calibration_single.rmse - calibration.rmse) / calibration_single.rmse * 100.0
		fmt.printf("Improvement: %.2f%%\n", improvement)
	}

	fmt.println("\n✓ PDPM Multi-Factor test completed!")
}
