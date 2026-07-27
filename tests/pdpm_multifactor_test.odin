package tests

import w "../wotan/core"
import fin "../wotan/finance"
import yahoo "../wotan/net"
import p "../wotan/plot"
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
		vix_close_prev, _ := w.column_at_float(&vix_df.columns[4], i - 1)
		vix_close_curr, _ := w.column_at_float(&vix_df.columns[4], i)
		if spy_close_prev == 0.0 || vix_close_prev == 0.0 {
			continue
		}
		spy_returns[i - 1] = (spy_close_curr - spy_close_prev) / spy_close_prev


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

	// Get current spot price
	spot_price, _ := w.column_at_float(&spy_df.columns[4], n_days - 1)
	fmt.printf("Current SPY Price: $%.2f\n", spot_price)

	// 3. Fetch or generate option chain
	fmt.println("\n--- Fetching Real Option Chain ---")

	option_chain := fin.fetch_yahoo_options("SPY", allocator)

	defer {
		delete(option_chain.strikes, allocator)
		delete(option_chain.expiries, allocator)
		delete(option_chain.implied_vols, allocator)
		delete(option_chain.market_prices, allocator)
		delete(option_chain.option_types, allocator)
	}

	fmt.printf("Loaded %d real options contracts\n", option_chain.n_options)

	risk_free_rate := 0.05

	if option_chain.n_options == 0 {
		fmt.println("\n⚠ Yahoo Options API blocked - generating synthetic option chain")

		if option_chain.strikes != nil {delete(option_chain.strikes, allocator)}
		if option_chain.expiries != nil {delete(option_chain.expiries, allocator)}
		if option_chain.implied_vols != nil {delete(option_chain.implied_vols, allocator)}
		if option_chain.market_prices != nil {delete(option_chain.market_prices, allocator)}
		if option_chain.option_types != nil {delete(option_chain.option_types, allocator)}

		n_options := 12
		syn_strikes := make([]f64, n_options, allocator)
		syn_expiries := make([]f64, n_options, allocator)
		syn_ivs := make([]f64, n_options, allocator)
		syn_prices := make([]f64, n_options, allocator)
		syn_types := make([]string, n_options, allocator)

		base_vol := hist_vol + 0.02
		time_to_expiry := 30.0 / 365.0

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

		fmt.printf("Generated %d synthetic options\n", n_options)
		fmt.printf("Implied Volatility: %.2f%%\n", base_vol * 100)
	}

	// 4. Calculate Momentum and Macro Factors
	fmt.println("\n--- Calculating Additional Factors ---")

	momentum_factor := fin.calculate_momentum(spy_returns, 252, allocator)
	defer delete(momentum_factor, allocator)

	macro_factor := fin.calculate_macro_factor(spy_returns, 126, allocator)
	defer delete(macro_factor, allocator)

	fmt.println("✓ Calculated momentum (12-month rolling return)")
	fmt.println("✓ Calculated macro factor (6-month rolling average)")

	// 5. Calibrate 2-Factor Model
	fmt.println("\n--- Calibrating 2-Factor SDF ---")

	calibration_2f := fin.calibrate_sdf_multifactor_2d(
		spy_returns,
		vix_returns,
		risk_free_rate,
		&option_chain,
		spot_price,
		allocator,
	)
	defer {
		delete(calibration_2f.sdf.values, allocator)
		delete(calibration_2f.sdf.parameters, allocator)
	}

	// 6. Calibrate 4-Factor Model
	fmt.println("\n--- Calibrating 4-Factor SDF ---")

	calibration_4f := fin.calibrate_sdf_multifactor_4d(
		spy_returns,
		vix_returns,
		momentum_factor,
		macro_factor,
		risk_free_rate,
		&option_chain,
		spot_price,
		allocator,
	)
	defer {
		delete(calibration_4f.sdf.values, allocator)
		delete(calibration_4f.sdf.parameters, allocator)
	}

	// 7. Compare Models
	fmt.println("\n--- Model Comparison ---")
	fmt.printf("2-Factor RMSE: $%.4f\n", calibration_2f.rmse)
	fmt.printf("4-Factor RMSE: $%.4f\n", calibration_4f.rmse)

	if calibration_2f.rmse > 0 {
		improvement := (calibration_2f.rmse - calibration_4f.rmse) / calibration_2f.rmse * 100.0
		fmt.printf("Improvement: %.2f%%\n", improvement)
	}

	// 8. Generate Visualizations
	fmt.println("\n--- Generating Visualizations ---")

	// Plot SDF vs Market Returns for 2-factor model
	n_states := len(spy_returns)
	xs_2f := make([]f64, n_states, allocator)
	ys_2f := make([]f64, n_states, allocator)
	defer {
		delete(xs_2f, allocator)
		delete(ys_2f, allocator)
	}

	copy(xs_2f, spy_returns)
	copy(ys_2f, calibration_2f.sdf.values)

	// Sort by market return
	for i in 0 ..< n_states - 1 {
		for j in 0 ..< n_states - i - 1 {
			if xs_2f[j] > xs_2f[j + 1] {
				xs_2f[j], xs_2f[j + 1] = xs_2f[j + 1], xs_2f[j]
				ys_2f[j], ys_2f[j + 1] = ys_2f[j + 1], ys_2f[j]
			}
		}
	}

	// Plot SDF vs Market Returns for 4-factor model
	xs_4f := make([]f64, n_states, allocator)
	ys_4f := make([]f64, n_states, allocator)
	defer {
		delete(xs_4f, allocator)
		delete(ys_4f, allocator)
	}

	copy(xs_4f, spy_returns)
	copy(ys_4f, calibration_4f.sdf.values)

	// Sort by market return
	for i in 0 ..< n_states - 1 {
		for j in 0 ..< n_states - i - 1 {
			if xs_4f[j] > xs_4f[j + 1] {
				xs_4f[j], xs_4f[j + 1] = xs_4f[j + 1], xs_4f[j]
				ys_4f[j], ys_4f[j + 1] = ys_4f[j + 1], ys_4f[j]
			}
		}
	}

	// Create comparison plot
	lines := []p.LineData {
		p.LineData{xs = xs_2f, ys = ys_2f, color = p.BLUE, style = .Solid, label = "2-Factor SDF"},
		p.LineData{xs = xs_4f, ys = ys_4f, color = p.RED, style = .Dashed, label = "4-Factor SDF"},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "SDF vs Market Return: 2-Factor vs 4-Factor"
	config.x_label = "Market Return"
	config.y_label = "SDF Value M"
	config.show_grid = true

	p.multi_line_png(lines, "sdf_comparison.png", config, allocator)
	fmt.println("✓ SDF comparison plot saved to: sdf_comparison.png")

	fmt.println("\n✓ PDPM Multi-Factor test completed!")
}
