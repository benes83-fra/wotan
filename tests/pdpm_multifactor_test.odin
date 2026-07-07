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

	// 3. Create synthetic option chain (since we can't easily fetch options)
	fmt.println("\n--- Creating Synthetic Option Chain ---")

	spot_price, _ := w.column_at_float(&spy_df.columns[4], n_days - 1)
	fmt.printf("Current SPY Price: $%.2f\n", spot_price)

	// Create ATM and OTM options
	n_options := 6
	strikes := make([]f64, n_options, allocator)
	expiries := make([]f64, n_options, allocator)
	implied_vols := make([]f64, n_options, allocator)
	market_prices := make([]f64, n_options, allocator)
	option_types := make([]string, n_options, allocator)
	defer {
		delete(strikes, allocator)
		delete(expiries, allocator)
		delete(implied_vols, allocator)
		delete(market_prices, allocator)
		delete(option_types, allocator)
	}

	// Use Black-Scholes to generate "market" prices
	risk_free_rate := 0.05
	time_to_expiry := 30.0 / 365.0 // 30 days

	for i in 0 ..< n_options {
		strikes[i] = spot_price * (0.95 + f64(i) * 0.02) // 95% to 105% of spot
		expiries[i] = time_to_expiry
		implied_vols[i] = 0.20 // 20% implied vol
		option_types[i] = "call"

		// Black-Scholes price
		d1 :=
			(math.ln_f64(spot_price / strikes[i]) +
				(risk_free_rate + 0.5 * implied_vols[i] * implied_vols[i]) * time_to_expiry) /
			(implied_vols[i] * math.sqrt_f64(time_to_expiry))
		d2 := d1 - implied_vols[i] * math.sqrt_f64(time_to_expiry)
		nd1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
		nd2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
		market_prices[i] =
			spot_price * nd1 - strikes[i] * math.exp(-risk_free_rate * time_to_expiry) * nd2
	}

	option_chain := fin.OptionChain {
		strikes       = strikes,
		expiries      = expiries,
		implied_vols  = implied_vols,
		market_prices = market_prices,
		option_types  = option_types,
		n_options     = n_options,
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

	improvement := (calibration_single.rmse - calibration.rmse) / calibration_single.rmse * 100.0
	fmt.printf("Improvement: %.2f%%\n", improvement)

	fmt.println("\n✓ PDPM Multi-Factor test completed!")
}
