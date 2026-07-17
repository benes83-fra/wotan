package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

vol_surface_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Volatility Surface Calibration ===\n")
	fin.debug_heston()

	S := 450.0
	r := 0.05

	// Use slice with allocator for append
	market_data := make([dynamic]fin.VolSurfacePoint, 0, allocator)
	defer delete(market_data)

	expiries := []f64{30.0 / 365.0, 90.0 / 365.0, 180.0 / 365.0}
	strikes := []f64{0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15}

	for T in expiries {
		for moneyness in strikes {
			K := S * moneyness

			base_vol := 0.20
			skew := -0.15 * (moneyness - 1.0)
			smile := 0.05 * (moneyness - 1.0) * (moneyness - 1.0)
			term_structure := 0.02 * math.sqrt_f64(T * 365.0 / 30.0)

			implied_vol := base_vol + skew + smile + term_structure

			// ✅ FIX: Remove the allocator argument. The slice already has it.
			append(
				&market_data,
				fin.VolSurfacePoint {
					strike = K,
					expiry = T,
					implied_vol = implied_vol,
					market_price = fin.black_scholes_call(S, K, T, r, implied_vol),
				},
			)
		}
	}

	fmt.printf("Generated %d market data points\n", len(market_data))
	fmt.printf("Spot: $%.2f, Risk-free rate: %.2f%%\n\n", S, r * 100)

	fmt.println("--- Market Volatility Surface ---")
	fmt.printf("%-10s %-10s %-12s\n", "Expiry", "Strike", "Implied Vol")
	fmt.printf("%-10s %-10s %-12s\n", "----------", "----------", "------------")
	for point in market_data {
		fmt.printf(
			"%-10.1f %-10.1f %-12.4f\n",
			point.expiry * 365.0,
			point.strike,
			point.implied_vol * 100,
		)
	}

	fmt.println("\n--- Calibrating SABR Model ---")
	sabr_result := fin.calibrate_sabr(market_data[:], S, r, allocator)
	fmt.printf("Converged: %v (iterations: %d)\n", sabr_result.converged, sabr_result.iterations)
	fmt.printf("RMSE: %.6f\n\n", sabr_result.rmse)
	fmt.println("SABR Parameters:")
	fmt.printf("  α (alpha): %.6f\n", sabr_result.params.alpha)
	fmt.printf("  β (beta):  %.6f\n", sabr_result.params.beta)
	fmt.printf("  ρ (rho):   %.6f\n", sabr_result.params.rho)
	fmt.printf("  ν (nu):    %.6f\n\n", sabr_result.params.nu)

	fmt.println("--- Calibrating Heston Model ---")
	heston_result := fin.calibrate_heston(market_data[:], S, r, allocator)
	fmt.printf(
		"Converged: %v (iterations: %d)\n",
		heston_result.converged,
		heston_result.iterations,
	)
	fmt.printf("RMSE: %.6f\n\n", heston_result.rmse)
	fmt.println("Heston Parameters:")
	fmt.printf("  v₀ (initial var):     %.6f\n", heston_result.params.v0)
	fmt.printf("  κ (mean reversion):   %.6f\n", heston_result.params.kappa)
	fmt.printf("  θ (long-run var):     %.6f\n", heston_result.params.theta)
	fmt.printf("  σ (vol of vol):       %.6f\n", heston_result.params.sigma)
	fmt.printf("  ρ (correlation):      %.6f\n\n", heston_result.params.rho)

	fmt.println("--- Model Comparison ---")
	fmt.printf("%-10s %-12s %-12s %-12s\n", "Strike", "Market Vol", "SABR Vol", "Heston Vol")
	fmt.printf(
		"%-10s %-12s %-12s %-12s\n",
		"----------",
		"------------",
		"------------",
		"------------",
	)

	test_strikes := []f64{S * 0.90, S * 0.95, S * 1.00, S * 1.05, S * 1.10}
	T_test := 90.0 / 365.0

	for K in test_strikes {
		F := S * math.exp(r * T_test)
		sabr_vol := fin.sabr_implied_vol(F, K, T_test, sabr_result.params)

		// ✅ FIX: 1. Price the option using the CALIBRATED Heston parameters
		heston_price_val := fin.heston_price(S, K, T_test, r, heston_result.params, .Call, 1000)

		// ✅ FIX: 2. Find the implied volatility of THAT Heston price
		heston_vol, _, _ := fin.implied_volatility(
			heston_price_val,
			S,
			K,
			T_test,
			r,
			.Call,
			allocator,
		)

		market_vol := 0.20
		for point in market_data {
			if math.abs(point.strike - K) < 1.0 && math.abs(point.expiry - T_test) < 0.01 {
				market_vol = point.implied_vol
				break
			}
		}

		fmt.printf(
			"%-10.1f %-12.4f %-12.4f %-12.4f\n",
			K,
			market_vol * 100,
			sabr_vol * 100,
			heston_vol * 100,
		)
	}

	fmt.println("\n✓ Volatility Surface Calibration test completed!")
}
