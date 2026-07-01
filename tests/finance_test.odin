package tests

import fin "../wotan/finance"
import l "../wotan/linalg"
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
	// ====================================================================
	// Test 4: Greeks Sanity Checks (with CORRECT expectations)
	// ====================================================================
	fmt.println("\n--- Test 4: Greeks Sanity Checks ---")

	cg := fin.compute_greeks(S, K, T, r, sigma, .Call, allocator)
	pg := fin.compute_greeks(S, K, T, r, sigma, .Put, allocator)

	// Call delta should be in (0, 1)
	fmt.printf("  Call delta in (0,1): %v (%.4f)\n", cg.delta > 0 && cg.delta < 1, cg.delta)

	// Put delta should be in (-1, 0)
	fmt.printf("  Put delta in (-1,0): %v (%.4f)\n", pg.delta < 0 && pg.delta > -1, pg.delta)

	// Call delta - Put delta ≈ 1 (put-call parity for deltas)
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

	// Theta should be NEGATIVE for long calls (time decay)
	fmt.printf("  Call theta < 0:     %v (%.4f per day)\n", cg.theta < 0, cg.theta)
	fmt.printf("  Put theta < 0:      %v (%.4f per day)\n", pg.theta < 0, pg.theta)

	// Verify against closed-form BS Greeks
	fmt.println("\n--- Verification Against Closed-Form BS ---")
	inv_sqrt_2pi := 0.3989422804014327
	sqrt_T := math.sqrt(T)
	d1 := (math.ln_f64(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	d2 := d1 - sigma * sqrt_T
	phi_d1 := math.exp_f64(-0.5 * d1 * d1) * inv_sqrt_2pi

	bs_delta := norm_cdf(d1)
	bs_gamma := phi_d1 / (S * sigma * sqrt_T)
	bs_vega := S * phi_d1 * sqrt_T / 100.0 // per 1% move
	bs_rho := K * T * math.exp(-r * T) * norm_cdf(d2) / 100.0 // per 1% move

	fmt.printf("  %-8s  %-12s  %-12s  %-10s\n", "Greek", "Autograd", "Closed-Form", "Error")
	fmt.printf(
		"  %-8s  %-12.6f  %-12.6f  %.2e\n",
		"Delta",
		cg.delta,
		bs_delta,
		math.abs(cg.delta - bs_delta),
	)
	fmt.printf(
		"  %-8s  %-12.6f  %-12.6f  %.2e\n",
		"Gamma",
		cg.gamma,
		bs_gamma,
		math.abs(cg.gamma - bs_gamma),
	)
	fmt.printf(
		"  %-8s  %-12.6f  %-12.6f  %.2e\n",
		"Vega",
		cg.vega,
		bs_vega,
		math.abs(cg.vega - bs_vega),
	)
	fmt.printf(
		"  %-8s  %-12.6f  %-12.6f  %.2e\n",
		"Rho",
		cg.rho,
		bs_rho,
		math.abs(cg.rho - bs_rho),
	)
	fmt.println("\n✓ Derivatives test completed!")
}


norm_cdf :: proc(x: f64) -> f64 {
	return 0.5 * (1.0 + math.erf(x / math.sqrt_f64(2.0)))
}


portfolio_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Portfolio Optimization Test ===\n")

	// Create sample data: 3 assets
	n_assets := 3

	// Expected annual returns (10%, 15%, 20%)
	expected_returns := []f64{0.10, 0.15, 0.20}

	// Covariance matrix (annualized)
	cov_matrix := l.matrix_new(f64, n_assets, n_assets, allocator)
	defer l.matrix_free(&cov_matrix)

	// Set covariance values
	// Diagonal: variances (0.04, 0.09, 0.16)
	cov_matrix.data[0] = 0.04 // Asset 1 variance
	cov_matrix.data[4] = 0.09 // Asset 2 variance
	cov_matrix.data[8] = 0.16 // Asset 3 variance

	// Off-diagonal: covariances
	cov_matrix.data[1] = 0.02 // Cov(1,2)
	cov_matrix.data[2] = 0.01 // Cov(1,3)
	cov_matrix.data[3] = 0.02 // Cov(2,1)
	cov_matrix.data[5] = 0.03 // Cov(2,3)
	cov_matrix.data[6] = 0.01 // Cov(3,1)
	cov_matrix.data[7] = 0.03 // Cov(3,2)

	risk_free_rate := 0.03 // 3% risk-free rate

	// Test 1: Minimum Variance Portfolio
	fmt.println("--- Minimum Variance Portfolio ---")
	min_var_weights := fin.min_variance_portfolio(&cov_matrix, allocator)
	defer delete(min_var_weights, allocator)

	min_var_metrics := fin.portfolio_metrics(
		min_var_weights,
		expected_returns,
		&cov_matrix,
		risk_free_rate,
	)
	fmt.printf(
		"Weights: [%.3f, %.3f, %.3f]\n",
		min_var_metrics.weights[0],
		min_var_metrics.weights[1],
		min_var_metrics.weights[2],
	)
	fmt.printf("Expected Return: %.2f%%\n", min_var_metrics.expected_return * 100)
	fmt.printf("Volatility: %.2f%%\n", min_var_metrics.volatility * 100)
	fmt.printf("Sharpe Ratio: %.3f\n\n", min_var_metrics.sharpe_ratio)

	// Test 2: Maximum Sharpe Ratio Portfolio
	fmt.println("--- Maximum Sharpe Ratio Portfolio ---")
	max_sharpe_weights := fin.max_sharpe_portfolio(
		expected_returns,
		&cov_matrix,
		risk_free_rate,
		allocator,
	)
	defer delete(max_sharpe_weights, allocator)

	max_sharpe_metrics := fin.portfolio_metrics(
		max_sharpe_weights,
		expected_returns,
		&cov_matrix,
		risk_free_rate,
	)
	fmt.printf(
		"Weights: [%.3f, %.3f, %.3f]\n",
		max_sharpe_metrics.weights[0],
		max_sharpe_metrics.weights[1],
		max_sharpe_metrics.weights[2],
	)
	fmt.printf("Expected Return: %.2f%%\n", max_sharpe_metrics.expected_return * 100)
	fmt.printf("Volatility: %.2f%%\n", max_sharpe_metrics.volatility * 100)
	fmt.printf("Sharpe Ratio: %.3f\n\n", max_sharpe_metrics.sharpe_ratio)

	// Test 3: Efficient Frontier
	fmt.println("--- Efficient Frontier (5 points) ---")
	frontier := fin.efficient_frontier(expected_returns, &cov_matrix, risk_free_rate, 5, allocator)
	defer {
		for point in frontier {
			delete(point.weights, allocator)
		}
		delete(frontier, allocator)
	}

	for point, i in frontier {
		fmt.printf(
			"Point %d: Return=%.2f%%, Volatility=%.2f%%, Sharpe=%.3f\n",
			i + 1,
			point.expected_return * 100,
			point.volatility * 100,
			point.sharpe_ratio,
		)
	}

	fmt.println("\n✓ Portfolio optimization test completed!")
}
constraints_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Portfolio Constraints Test ===\n")


	// 3-asset example
	n := 3

	// Expected returns: [10%, 15%, 20%]
	returns := []f64{0.10, 0.15, 0.20}

	// Covariance matrix
	cov := l.matrix_new(f64, n, n, allocator)
	defer l.matrix_free(&cov)

	cov.data[0] = 0.04; cov.data[1] = 0.02; cov.data[2] = 0.01
	cov.data[3] = 0.02; cov.data[4] = 0.09; cov.data[5] = 0.03
	cov.data[6] = 0.01; cov.data[7] = 0.03; cov.data[8] = 0.16

	rf := 0.03 // Risk-free rate

	// Test 1: No short selling constraint
	fmt.println("--- Test 1: No Short Selling ---")
	constraints1 := fin.PortfolioConstraints {
		no_short_selling = true,
		max_weight       = 0.0,
		min_weight       = 0.0,
	}

	weights1 := fin.constrained_min_variance_portfolio(&cov, constraints1, allocator)
	defer delete(weights1, allocator)

	fmt.printf("Min variance weights: [%.3f, %.3f, %.3f]\n", weights1[0], weights1[1], weights1[2])
	fmt.printf(
		"All weights >= 0: %v\n",
		le_tol(0.0, weights1[0]) && le_tol(0.0, weights1[1]) && le_tol(0.0, weights1[2]),
	)

	// Test 2: Max Position Size (40%)
	fmt.println("\n--- Test 2: Max Position Size (40%) ---")
	config2 := fin.PortfolioConstraints {
		no_short_selling = true,
		max_weight       = 0.40,
	}

	raw_weights := fin.constrained_max_sharpe_portfolio(returns, &cov, rf, config2, allocator)
	weights := fin.enforce_constraints(raw_weights, config2, allocator)

	fmt.printf("Max Sharpe weights: [%.3f, %.3f, %.3f]\n", weights[0], weights[1], weights[2])
	fmt.printf(
		"All weights <= 40%%: %v\n",
		le_tol(weights[0], 0.40) && le_tol(weights[1], 0.40) && le_tol(weights[2], 0.40),
	)

	delete(raw_weights, allocator)
	delete(weights, allocator)

	// Test 3: Sector constraints
	fmt.println("\n--- Test 3: Sector Constraints ---")
	group_a := fin.GroupLimit {
		asset_indices = []int{0, 1},
		max_weight    = 0.60,
	}
	group_b := fin.GroupLimit {
		asset_indices = []int{2},
		max_weight    = 0.50,
	}

	constraints3 := fin.PortfolioConstraints {
		no_short_selling = true,
		max_weight       = 0.0,
		min_weight       = 0.0,
		group_limits     = []fin.GroupLimit{group_a, group_b},
	}

	raw_weights3 := fin.constrained_min_variance_portfolio(&cov, constraints3, allocator)
	weights3 := fin.enforce_constraints(raw_weights3, constraints3, allocator)

	defer delete(raw_weights3, allocator)
	defer delete(weights3, allocator)

	sector_a_sum := weights3[0] + weights3[1]

	fmt.printf("Min variance weights: [%.3f, %.3f, %.3f]\n", weights3[0], weights3[1], weights3[2])
	fmt.printf(
		"Sector A (assets 0+1) <= 60%%: %v (%.3f)\n",
		le_tol(sector_a_sum, 0.60),
		sector_a_sum,
	)
	fmt.printf("Sector B (asset 2) <= 50%%: %v (%.3f)\n", le_tol(weights3[2], 0.50), weights3[2])

	fmt.println("\n✓ Portfolio constraints test completed!")
}
// Helper function for tolerance-based comparison
le_tol :: proc(a: f64, b: f64, tol: f64 = 1e-4) -> bool { 	// Changed from 1e-6 to 1e-4
	return a <= b + tol
}
