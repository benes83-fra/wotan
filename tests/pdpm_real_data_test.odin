// wotan/tests/pdpm_real_data_test.odin
package tests

import w "../wotan/core"
import fin "../wotan/finance"
import p "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

pdpm_real_data_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== PDPM Real Market Data Test ===\n")

	// 1. Load historical price data
	fmt.println("--- Loading Market Data ---")
	market_data := fin.load_market_data("data/spy_prices.csv", allocator)
	defer {
		delete(market_data.dates, allocator)
		delete(market_data.prices, allocator)
		delete(market_data.volumes, allocator)
		delete(market_data.returns, allocator)
	}

	fmt.printf("Loaded %d days of price data\n", market_data.n_obs)
	fmt.printf(
		"Price range: $%.2f - $%.2f\n",
		market_data.prices[0],
		market_data.prices[market_data.n_obs - 1],
	)

	// Compute historical volatility
	var_sum := 0.0
	var_sum_sq := 0.0
	for i in 1 ..< market_data.n_obs {
		ret := market_data.returns[i]
		var_sum += ret
		var_sum_sq += ret * ret
	}
	mean_ret := var_sum / f64(market_data.n_obs - 1)
	variance := (var_sum_sq / f64(market_data.n_obs - 1)) - (mean_ret * mean_ret)
	hist_vol := math.sqrt_f64(variance * 252.0)
	fmt.printf("Historical Volatility: %.2f%%\n\n", hist_vol * 100)

	// 2. Load option chain
	fmt.println("--- Loading Option Chain ---")
	current_date := market_data.dates[market_data.n_obs - 1]
	option_chain := fin.load_option_chain("data/spy_options.csv", current_date, allocator)
	defer {
		delete(option_chain.strikes, allocator)
		delete(option_chain.expiries, allocator)
		delete(option_chain.implied_vols, allocator)
		delete(option_chain.market_prices, allocator)
		delete(option_chain.option_types, allocator)
	}

	fmt.printf("Loaded %d options\n", option_chain.n_options)
	fmt.printf(
		"Strike range: $%.0f - $%.0f\n",
		option_chain.strikes[0],
		option_chain.strikes[option_chain.n_options - 1],
	)
	fmt.printf("Time to expiry: %.2f years\n\n", option_chain.expiries[0])

	// 3. Calibrate SDF to market prices
	fmt.println("--- Calibrating SDF ---")
	risk_free_rate := 0.05
	spot_price := market_data.prices[market_data.n_obs - 1]

	// UPDATED: New signature only takes 6 arguments
	// In pdpm_real_data_test.odin, after loading market_data:

	// Calculate Implied Volatility (average from option chain)
	avg_iv := 0.0
	for i in 0 ..< option_chain.n_options {
		avg_iv += option_chain.implied_vols[i]
	}
	avg_iv /= f64(option_chain.n_options)

	fmt.printf("Historical Vol: %.2f%%, Implied Vol: %.2f%%\n", hist_vol * 100, avg_iv * 100)

	// Scale historical returns to match implied volatility
	vol_scale := avg_iv / hist_vol
	scaled_returns := make([]f64, len(market_data.returns) - 1, allocator)
	defer delete(scaled_returns, allocator)

	for i in 0 ..< len(scaled_returns) {
		scaled_returns[i] = market_data.returns[i + 1] * vol_scale
	}

	// Now calibrate using scaled_returns instead of market_data.returns[1:]
	calibration := fin.calibrate_sdf_linear(
		scaled_returns, // <-- Use scaled returns here
		risk_free_rate,
		&option_chain,
		spot_price,
		1000,
		1e-3,
		allocator,
	)
	defer {
		delete(calibration.sdf.values, allocator)
		delete(calibration.sdf.parameters, allocator)
	}

	fmt.printf("\nCalibration Results:\n")
	fmt.printf("  Converged: %v\n", calibration.converged)
	fmt.printf("  RMSE: $%.4f\n", calibration.rmse)
	fmt.printf("  Iterations: %d\n", calibration.n_iterations)
	fmt.printf(
		"  SDF Mean: %.4f (Target: %.4f)\n",
		calibration.sdf.mean,
		1.0 / (1.0 + risk_free_rate),
	)

	// 4. Compare prices
	// 4. Compare prices
	fmt.println("\n--- Price Comparison ---")
	fmt.printf("%-10s %-10s %-10s %-10s\n", "Strike", "Market", "PDPM", "BS")

	// Use historical returns as terminal states, NOT Monte Carlo
	n_returns := len(market_data.returns) - 1 // Skip first undefined return
	terminal_prices := make([]f64, n_returns, allocator)
	defer delete(terminal_prices, allocator)

	for i in 0 ..< n_returns {
		terminal_prices[i] = spot_price * (1.0 + market_data.returns[i + 1])
	}

	for i in 0 ..< option_chain.n_options {
		if option_chain.option_types[i] == "call" {
			pdpm_price, bs_price := fin.compare_to_black_scholes(
				spot_price,
				option_chain.strikes[i],
				option_chain.expiries[i],
				risk_free_rate,
				option_chain.implied_vols[i],
				&calibration.sdf,
				terminal_prices, // Now has 29 values, matching SDF!
				context.temp_allocator,
			)

			fmt.printf(
				"$%-9.0f $%-9.2f $%-9.2f $%-9.2f\n",
				option_chain.strikes[i],
				option_chain.market_prices[i],
				pdpm_price,
				bs_price,
			)
		}
	}
	// In pdpm_real_data_test.odin, after the linear calibration:

	// Test Power Utility SDF
	fmt.println("\n--- Power Utility SDF Calibration ---")
	calibration_power := fin.calibrate_sdf_power_utility(
		scaled_returns,
		risk_free_rate,
		&option_chain,
		spot_price,
		allocator,
	)
	defer {
		delete(calibration_power.sdf.values, allocator)
		delete(calibration_power.sdf.parameters, allocator)
	}

	fmt.printf("Power Utility Results:\n")
	fmt.printf("  Risk Aversion (γ): %.4f\n", calibration_power.sdf.parameters[0])
	fmt.printf("  RMSE: $%.4f\n", calibration_power.rmse)
	fmt.printf("  SDF Mean: %.4f\n", calibration_power.sdf.mean)

	// Test Quadratic SDF
	fmt.println("\n--- Quadratic SDF Calibration ---")
	calibration_quad := fin.calibrate_sdf_quadratic(
		scaled_returns,
		risk_free_rate,
		&option_chain,
		spot_price,
		allocator,
	)
	defer {
		delete(calibration_quad.sdf.values, allocator)
		delete(calibration_quad.sdf.parameters, allocator)
	}

	fmt.printf("Quadratic SDF Results:\n")
	fmt.printf(
		"  Parameters: a=%.4f, b=%.4f, c=%.4f\n",
		calibration_quad.sdf.parameters[0],
		calibration_quad.sdf.parameters[1],
		calibration_quad.sdf.parameters[2],
	)
	fmt.printf("  RMSE: $%.4f\n", calibration_quad.rmse)
	fmt.printf("  SDF Mean: %.4f\n", calibration_quad.sdf.mean)

	// Compare all three models
	fmt.println("\n--- Model Comparison ---")
	fmt.printf(
		"%-20s %-15s %-15s %-15s\n",
		"Strike",
		"Linear RMSE",
		"Power RMSE",
		"Quadratic RMSE",
	)
	fmt.printf(
		"%-20s $%-14.4f $%-14.4f $%-14.4f\n",
		"Overall",
		calibration.rmse,
		calibration_power.rmse,
		calibration_quad.rmse,
	)
	// Test Exponential SDF
	fmt.println("\n--- Exponential SDF Calibration ---")
	calibration_exp := fin.calibrate_sdf_exponential(
		scaled_returns,
		risk_free_rate,
		&option_chain,
		spot_price,
		allocator,
	)
	defer {
		delete(calibration_exp.sdf.values, allocator)
		delete(calibration_exp.sdf.parameters, allocator)
	}

	fmt.printf("Exponential SDF Results:\n")
	fmt.printf("  Risk Aversion (γ): %.4f\n", calibration_exp.sdf.parameters[0])
	fmt.printf("  RMSE: $%.4f\n", calibration_exp.rmse)
	fmt.printf("  SDF Mean: %.4f\n", calibration_exp.sdf.mean)

	// Update Model Comparison
	fmt.println("\n--- Final Model Comparison ---")
	fmt.printf("%-20s %-15s %-15s %-15s %-15s\n", "Model", "RMSE", "E[M]", "Parameters", "Type")
	fmt.printf(
		"%-20s $%-14.4f %-14.4f a=%.2f, b=%.2f       Linear\n",
		"Linear",
		calibration.rmse,
		calibration.sdf.mean,
		calibration.sdf.parameters[0],
		calibration.sdf.parameters[1],
	)
	fmt.printf(
		"%-20s $%-14.4f %-14.4f γ=%.2f              Power Utility\n",
		"Power Utility",
		calibration_power.rmse,
		calibration_power.sdf.mean,
		calibration_power.sdf.parameters[0],
	)
	fmt.printf(
		"%-20s $%-14.4f %-14.4f γ=%.2f              Exponential\n",
		"Exponential",
		calibration_exp.rmse,
		calibration_exp.sdf.mean,
		calibration_exp.sdf.parameters[0],
	)
	fmt.printf(
		"%-20s $%-14.4f %-14.4f a=%.2f, b=%.2f, c=%.2f Quadratic\n",
		"Quadratic",
		calibration_quad.rmse,
		calibration_quad.sdf.mean,
		calibration_quad.sdf.parameters[0],
		calibration_quad.sdf.parameters[1],
		calibration_quad.sdf.parameters[2],
	)

	// 7. Generate Visualizations
	fmt.println("\n--- Generating Visualizations ---")

	// For 2D surface, we need a 2-factor model. Use the best model we have.
	// For now, let's just plot the SDF vs returns for the quadratic model (best performer)
	fmt.println("Plotting SDF vs Market Returns (Quadratic Model)...")

	// Create a simple 1D plot of SDF values
	n_states := len(scaled_returns)
	xs := make([]f64, n_states, allocator)
	ys := make([]f64, n_states, allocator)
	defer {
		delete(xs, allocator)
		delete(ys, allocator)
	}

	// Copy and sort
	copy(xs, scaled_returns)
	copy(ys, calibration_quad.sdf.values)

	// Sort by x
	for i in 0 ..< n_states - 1 {
		for j in 0 ..< n_states - i - 1 {
			if xs[j] > xs[j + 1] {
				xs[j], xs[j + 1] = xs[j + 1], xs[j]
				ys[j], ys[j + 1] = ys[j + 1], ys[j]
			}
		}
	}

	lines := []p.LineData {
		p.LineData{xs = xs, ys = ys, color = p.RED, style = .Solid, label = "Quadratic SDF"},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Stochastic Discount Factor vs Market Return (Quadratic Model)"
	config.x_label = "Market Return"
	config.y_label = "SDF Value M"
	config.show_grid = true

	p.multi_line_png(lines, "sdf_vs_returns.png", config, allocator)
	fmt.println("✓ SDF plot saved to: sdf_vs_returns.png")


	fmt.println("\n✓ PDPM Real Market Data test completed!")
}
