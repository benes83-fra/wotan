package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

derivatives_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Derivatives Pricing Test ===\n")

	// ====================================================================
	// Test 1: Black-Scholes Pricing (known values)
	// ====================================================================
	fmt.println("--- Test 1: Black-Scholes Pricing ---")

	// Standard test case: S=100, K=100, T=1, r=5%, σ=20%
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.20

	call_price, call_greeks := fin.price_and_greeks(S, K, T, r, sigma, .Call, allocator)
	put_price, put_greeks := fin.price_and_greeks(S, K, T, r, sigma, .Put, allocator)

	fmt.printf("ATM Call (S=100, K=100, T=1, r=5%%, σ=20%%):\n")
	fmt.printf("  Price: %.4f  (expected ~10.4506)\n", call_price)
	fmt.printf("  Delta: %.4f  (expected ~0.6368)\n", call_greeks.delta)
	fmt.printf("  Gamma: %.4f  (expected ~0.0187)\n", call_greeks.gamma)
	fmt.printf("  Vega:  %.4f  (expected ~0.1870 per 1%%)\n", call_greeks.vega)
	fmt.printf("  Theta: %.4f  (expected ~-0.0163 per day)\n", call_greeks.theta)
	fmt.printf("  Rho:   %.4f  (expected ~0.0532 per 1%%)\n", call_greeks.rho)

	fmt.printf("\nATM Put:\n")
	fmt.printf("  Price: %.4f  (expected ~5.5735)\n", put_price)
	fmt.printf("  Delta: %.4f  (expected ~-0.3632)\n", put_greeks.delta)

	// Put-Call Parity check: C - P = S - K*exp(-r*T)
	pcp_rhs := S - K * math.exp(-r * T)
	pcp_lhs := call_price - put_price
	fmt.printf("\nPut-Call Parity:\n")
	fmt.printf("  C - P = %.4f\n", pcp_lhs)
	fmt.printf("  S - K*exp(-rT) = %.4f\n", pcp_rhs)
	fmt.printf("  Error: %.2e\n", math.abs(pcp_lhs - pcp_rhs))

	// ====================================================================
	// Test 2: Moneyness (ITM, ATM, OTM)
	// ====================================================================
	fmt.println("\n--- Test 2: Moneyness ---")

	strikes := []f64{80.0, 90.0, 100.0, 110.0, 120.0}
	labels := []string{"ITM", "ITM", "ATM", "OTM", "OTM"}

	fmt.printf(
		"  %-5s  K=%-6s  Call=%-10s  Put=%-10s  CallΔ=%-8s  PutΔ=%-8s\n",
		"Type",
		"Strike",
		"Price",
		"Price",
		"Delta",
		"Delta",
	)

	for i in 0 ..< len(strikes) {
		cp, cg := fin.price_and_greeks(S, strikes[i], T, r, sigma, .Call, allocator)
		pp, pg := fin.price_and_greeks(S, strikes[i], T, r, sigma, .Put, allocator)
		fmt.printf(
			"  %-5s  K=%-6.0f  Call=%-10.4f  Put=%-10.4f  CallΔ=%-8.4f  PutΔ=%-8.4f\n",
			labels[i],
			strikes[i],
			cp,
			pp,
			cg.delta,
			pg.delta,
		)
	}

	// ====================================================================
	// Test 3: Implied Volatility
	// ====================================================================
	fmt.println("\n--- Test 3: Implied Volatility ---")

	// Given a market price, find the implied vol
	market_call := 10.45 // approximately ATM call price
	iv, converged, iters := fin.implied_volatility(market_call, S, K, T, r, .Call, allocator)

	fmt.printf("  Market Call Price: %.2f\n", market_call)
	fmt.printf("  Implied Vol:       %.4f (%.2f%%)\n", iv, iv * 100)
	fmt.printf("  Converged:         %v in %d iterations\n", converged, iters)

	// Verify: price at implied vol should match market price
	verify_price, _ := fin.price_and_greeks(S, K, T, r, iv, .Call, allocator)
	fmt.printf(
		"  Verify Price:      %.4f (error: %.2e)\n",
		verify_price,
		math.abs(verify_price - market_call),
	)

	// Test with different market prices
	fmt.println("\n  Vol Surface Scan:")
	test_prices := []f64{5.0, 8.0, 10.45, 13.0, 16.0}
	for mp in test_prices {
		iv2, conv2, it2 := fin.implied_volatility(mp, S, K, T, r, .Call, allocator)
		fmt.printf(
			"    Market=%.2f  →  IV=%.4f (%.2f%%)  [%v in %d iter]\n",
			mp,
			iv2,
			iv2 * 100,
			conv2,
			it2,
		)
	}

	// ====================================================================
	// Test 4: Greeks Sanity Checks
	// ====================================================================
	fmt.println("\n--- Test 4: Greeks Sanity Checks ---")

	cg := fin.compute_greeks(S, K, T, r, sigma, .Call, allocator)
	pg := fin.compute_greeks(S, K, T, r, sigma, .Put, allocator)

	// Call delta should be in (0, 1)
	fmt.printf("  Call delta in (0,1): %v (%.4f)\n", cg.delta > 0 && cg.delta < 1, cg.delta)

	// Put delta should be in (-1, 0)
	fmt.printf("  Put delta in (-1,0): %v (%.4f)\n", pg.delta < 0 && pg.delta > -1, pg.delta)

	// Call delta - Put delta ≈ 1 (for European options)
	delta_diff := cg.delta - pg.delta
	fmt.printf(
		"  CallΔ - PutΔ ≈ 1:  %v (%.4f)\n",
		math.abs(delta_diff - 1.0) < 0.01,
		delta_diff,
	)

	// Gamma should be positive and same for call/put
	fmt.printf("  Gamma > 0:          %v (%.4f)\n", cg.gamma > 0, cg.gamma)
	fmt.printf(
		"  Call γ ≈ Put γ:     %v (diff: %.2e)\n",
		math.abs(cg.gamma - pg.gamma) < 1e-6,
		math.abs(cg.gamma - pg.gamma),
	)

	// Vega should be positive and same for call/put
	fmt.printf("  Vega > 0:           %v (%.4f)\n", cg.vega > 0, cg.vega)
	fmt.printf(
		"  Call ν ≈ Put ν:     %v (diff: %.2e)\n",
		math.abs(cg.vega - pg.vega) < 1e-6,
		math.abs(cg.vega - pg.vega),
	)

	fmt.println("\n✓ Derivatives test completed!")
}
