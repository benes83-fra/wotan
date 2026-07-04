// wotan/tests/pdpm_test.odin
package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
pdpm_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Dybvig PDPM Test ===\n")

	// 1. Generate synthetic market data (Physical Measure P)
	n_states := 10000
	market_returns := make([]f64, n_states, allocator)
	defer delete(market_returns, allocator)

	// Assume market has 8% annual return and 15% volatility
	mean_rm := 0.08
	std_rm := 0.15
	for i in 0 ..< n_states {
		market_returns[i] = rand.float64_normal(mean_rm, std_rm)
	}

	// 2. Construct Stochastic Discount Factors (SDF)
	risk_free_rate := 0.04

	fmt.println("--- Stochastic Discount Factor Construction ---")

	// Linear SDF: M = a + b*R_m
	sdf_linear := fin.sdf_linear_factor(market_returns, risk_free_rate, allocator)
	defer {
		delete(sdf_linear.values, allocator)
		delete(sdf_linear.parameters, allocator)
	}
	fmt.printf(
		"Linear SDF: E[M] = %.4f (Target 1/(1+rf): %.4f)\n",
		sdf_linear.mean,
		1.0 / (1.0 + risk_free_rate),
	)

	// 3. Generate Terminal Prices FROM THE SAME MARKET RETURNS
	// This is critical for consistency!
	spot_price := 100.0
	terminal_prices := make([]f64, n_states, allocator)
	defer delete(terminal_prices, allocator)

	for i in 0 ..< n_states {
		terminal_prices[i] = spot_price * (1.0 + market_returns[i])
	}

	// 4. Payoff Distribution Analysis
	fmt.println("\n--- Payoff Distribution Analysis ---")
	dist := fin.analyze_payoff_distribution(terminal_prices, nil, allocator)
	fmt.printf("Terminal Price Distribution:\n")
	fmt.printf("  Mean:    %.2f\n", dist.mean)
	fmt.printf("  Std Dev: %.2f\n", math.sqrt(dist.variance))

	// 5. Core PDPM Pricing (Options)
	fmt.println("\n--- Option Pricing (PDPM) ---")

	strike := 105.0

	// Price Call using Linear SDF
	call_res := fin.price_european_call(
		spot_price,
		strike,
		1.0,
		&sdf_linear,
		terminal_prices,
		allocator,
	)
	defer delete(call_res.state_prices, allocator)
	fmt.printf("European Call (K=%.0f) Price: $%.4f\n", strike, call_res.price)

	// Price Put
	put_res := fin.price_european_put(
		spot_price,
		strike,
		1.0,
		&sdf_linear,
		terminal_prices,
		allocator,
	)
	defer delete(put_res.state_prices, allocator)
	fmt.printf("European Put  (K=%.0f) Price: $%.4f\n", strike, put_res.price)

	// Check Put-Call Parity: C - P = S - K * E[M]
	parity_lhs := call_res.price - put_res.price
	parity_rhs := spot_price - strike * sdf_linear.mean
	fmt.printf("Put-Call Parity Check:\n")
	fmt.printf("  C - P        = %.4f\n", parity_lhs)
	fmt.printf("  S - K * E[M] = %.4f\n", parity_rhs)
	fmt.printf("  Difference   = %.6f (should be ~0)\n", math.abs(parity_lhs - parity_rhs))

	// 6. Dybvig's Distributional Pricing
	fmt.println("\n--- Dybvig Distributional Pricing ---")

	dybvig_res := fin.dybvig_distributional_price(terminal_prices, &sdf_linear, 50, allocator)
	defer delete(dybvig_res.state_prices, allocator)

	fmt.printf("Dybvig Price of Underlying: $%.4f (Spot: $%.2f)\n", dybvig_res.price, spot_price)
	fmt.printf("Expected Payoff (Physical): $%.4f\n", dybvig_res.expected_payoff)

	// Add to pdpm_test after the pricing section:

	// 7. Visualize the complete PDPM analysis
	fmt.println("\n--- Generating PDPM Visualizations ---")

	// Plot complete analysis
	ok1 := fin.plot_pdpm_analysis(
		terminal_prices,
		&sdf_linear,
		market_returns,
		"Linear SDF Analysis",
		"pdpm_linear_sdf",
		allocator,
	)
	if ok1 {
		fmt.println("✓ Generated 4 PDPM analysis plots:")
		fmt.println("  - pdpm_linear_sdf_1_payoff_dist.png")
		fmt.println("  - pdpm_linear_sdf_2_sdf_values.png")
		fmt.println("  - pdpm_linear_sdf_3_state_price_density.png")
		fmt.println("  - pdpm_linear_sdf_4_risk_neutral_dist.png")
	}

	// Plot option payoff profiles
	ok2 := fin.plot_option_payoff_profile(
		terminal_prices,
		strike,
		"call",
		&sdf_linear,
		"pdpm_call_payoff.png",
		allocator,
	)

	ok3 := fin.plot_option_payoff_profile(
		terminal_prices,
		strike,
		"put",
		&sdf_linear,
		"pdpm_put_payoff.png",
		allocator,
	)

	if ok2 && ok3 {
		fmt.println("✓ Generated option payoff profiles:")
		fmt.println("  - pdpm_call_payoff.png")
		fmt.println("  - pdpm_put_payoff.png")
	}
	fmt.println("\n✓ Dybvig PDPM test completed!")
}

pdpm2_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== PDPM (Payoff Distribution Pricing Model) Test ===\n")

	// ========================================================================
	// Test 1: SDF Construction - Power Utility
	// ========================================================================
	fmt.println("--- Test 1: Power Utility SDF ---")

	n_states := 1000
	consumption_growth := make([]f64, n_states, allocator)
	defer delete(consumption_growth, allocator)

	// Simulate consumption growth with mean 1.02 and std 0.05
	for i in 0 ..< n_states {
		consumption_growth[i] = rand.float64_normal(1.02, 0.05)
	}

	sdf_power := fin.sdf_power_utility(
		consumption_growth,
		2.0, // γ = 2 (moderate risk aversion)
		0.99, // β = 0.99
		allocator,
	)
	defer {
		delete(sdf_power.values, allocator)
		delete(sdf_power.parameters, allocator)
	}

	fmt.printf("Power Utility SDF:\n")
	fmt.printf("  Mean: %.4f\n", sdf_power.mean)
	fmt.printf("  Variance: %.6f\n", sdf_power.variance)
	fmt.printf("  Risk Aversion (γ): %.2f\n", sdf_power.risk_aversion)
	fmt.printf("  Discount Factor (β): %.2f\n", sdf_power.discount_factor)

	// ========================================================================
	// Test 2: SDF Construction - Linear Factor Model
	// ========================================================================
	fmt.println("\n--- Test 2: Linear Factor SDF ---")

	market_returns := make([]f64, n_states, allocator)
	defer delete(market_returns, allocator)

	// Simulate market returns with mean 0.08 and std 0.15
	for i in 0 ..< n_states {
		market_returns[i] = rand.float64_normal(0.08, 0.15)
	}

	sdf_linear := fin.sdf_linear_factor(
		market_returns,
		0.02, // Risk-free rate = 2%
		allocator,
	)
	defer {
		delete(sdf_linear.values, allocator)
		delete(sdf_linear.parameters, allocator)
	}

	fmt.printf("Linear Factor SDF:\n")
	fmt.printf("  Mean: %.4f\n", sdf_linear.mean)
	fmt.printf("  Variance: %.6f\n", sdf_linear.variance)
	fmt.printf(
		"  Parameters: a=%.4f, b=%.4f\n",
		sdf_linear.parameters[0],
		sdf_linear.parameters[1],
	)

	// ========================================================================
	// Test 3: Payoff Distribution Analysis
	// ========================================================================
	fmt.println("\n--- Test 3: Payoff Distribution Analysis ---")

	payoffs := make([]f64, n_states, allocator)
	defer delete(payoffs, allocator)

	// Create a payoff with positive skewness (lottery-like)
	for i in 0 ..< n_states {
		if rand.float64() < 0.9 {
			payoffs[i] = rand.float64_normal(0.0, 1.0)
		} else {
			payoffs[i] = rand.float64_normal(5.0, 2.0) // Occasional large payoff
		}
	}

	dist := fin.analyze_payoff_distribution(payoffs, nil, allocator)

	fmt.printf("Payoff Distribution Metrics:\n")
	fmt.printf("  Mean: %.4f\n", dist.mean)
	fmt.printf("  Variance: %.4f\n", dist.variance)
	fmt.printf("  Skewness: %.4f\n", dist.skewness)
	fmt.printf("  Kurtosis: %.4f\n", dist.kurtosis)
	fmt.printf("  Range: [%.4f, %.4f]\n", dist.min_payoff, dist.max_payoff)

	// ========================================================================
	// Test 4: Basic PDPM Pricing
	// ========================================================================
	fmt.println("\n--- Test 4: Basic PDPM Pricing ---")

	result := fin.price_payoff(payoffs, &sdf_power, nil, allocator)
	defer {
		delete(result.state_prices, allocator)
	}

	fmt.printf("PDPM Pricing Results:\n")
	fmt.printf("  Fair Price: %.4f\n", result.price)
	fmt.printf("  Expected Payoff: %.4f\n", result.expected_payoff)
	fmt.printf("  Risk Premium: %.4f\n", result.risk_premium)
	fmt.printf("  Correlation with SDF: %.4f\n", result.correlation_with_sdf)
	fmt.printf("  Certainty Equivalent: %.4f\n", result.certainty_equivalent)

	// ========================================================================
	// Test 5: European Option Pricing
	// ========================================================================
	fmt.println("\n--- Test 5: European Option Pricing ---")

	spot_price := 100.0
	strike_price := 105.0
	time_to_maturity := 1.0
	drift := 0.08
	volatility := 0.20

	terminal_prices := fin.simulate_terminal_prices(
		spot_price,
		drift,
		volatility,
		time_to_maturity,
		n_states,
		allocator,
	)
	defer delete(terminal_prices, allocator)

	// Price European Call
	call_result := fin.price_european_call(
		spot_price,
		strike_price,
		time_to_maturity,
		&sdf_linear,
		terminal_prices,
		allocator,
	)
	defer delete(call_result.state_prices, allocator)

	fmt.printf(
		"European Call Option (S=%.0f, K=%.0f, T=%.1f):\n",
		spot_price,
		strike_price,
		time_to_maturity,
	)
	fmt.printf("  Price: %.4f\n", call_result.price)
	fmt.printf("  Expected Payoff: %.4f\n", call_result.expected_payoff)
	fmt.printf("  Risk Premium: %.4f\n", call_result.risk_premium)

	// Price European Put
	put_result := fin.price_european_put(
		spot_price,
		strike_price,
		time_to_maturity,
		&sdf_linear,
		terminal_prices,
		allocator,
	)
	defer delete(put_result.state_prices, allocator)

	fmt.printf(
		"\nEuropean Put Option (S=%.0f, K=%.0f, T=%.1f):\n",
		spot_price,
		strike_price,
		time_to_maturity,
	)
	fmt.printf("  Price: %.4f\n", put_result.price)
	fmt.printf("  Expected Payoff: %.4f\n", put_result.expected_payoff)
	fmt.printf("  Risk Premium: %.4f\n", put_result.risk_premium)

	// ========================================================================
	// Test 6: Structured Product - Bull Spread
	// ========================================================================
	fmt.println("\n--- Test 6: Bull Spread Pricing ---")

	strike_low := 95.0
	strike_high := 110.0

	bull_spread_result := fin.price_bull_spread(
		spot_price,
		strike_low,
		strike_high,
		&sdf_linear,
		terminal_prices,
		allocator,
	)
	defer delete(bull_spread_result.state_prices, allocator)

	fmt.printf("Bull Spread (Long Call K=%.0f, Short Call K=%.0f):\n", strike_low, strike_high)
	fmt.printf("  Price: %.4f\n", bull_spread_result.price)
	fmt.printf("  Max Payoff: %.4f\n", strike_high - strike_low)
	fmt.printf("  Risk Premium: %.4f\n", bull_spread_result.risk_premium)

	// ========================================================================
	// Test 7: Dybvig's Distributional Pricing
	// ========================================================================
	fmt.println("\n--- Test 7: Dybvig's Distributional Pricing ---")

	dybvig_result := fin.dybvig_distributional_price(
		payoffs,
		&sdf_power,
		50, // 50 quantiles
		allocator,
	)
	defer delete(dybvig_result.state_prices, allocator)

	fmt.printf("Dybvig Distributional Pricing:\n")
	fmt.printf("  Price: %.4f\n", dybvig_result.price)
	fmt.printf("  Expected Payoff: %.4f\n", dybvig_result.expected_payoff)
	fmt.printf("  Risk Premium: %.4f\n", dybvig_result.risk_premium)
	fmt.printf("  Certainty Equivalent: %.4f\n", dybvig_result.certainty_equivalent)

	// ========================================================================
	// Test 8: State Price Density Estimation
	// ========================================================================
	fmt.println("\n--- Test 8: State Price Density Estimation ---")

	spd := fin.estimate_state_price_density(
		&sdf_linear,
		market_returns,
		0.02, // Bandwidth
		50, // 50 grid points
		allocator,
	)
	defer {
		delete(spd.states, allocator)
		delete(spd.densities, allocator)
	}

	fmt.printf("State Price Density (first 5 states):\n")
	for i in 0 ..< min(5, spd.n_points) {
		fmt.printf("  State %.4f: Density %.6f\n", spd.states[i], spd.densities[i])
	}

	// ========================================================================
	// Test 9: Distribution Metrics
	// ========================================================================
	fmt.println("\n--- Test 9: Distribution Metrics ---")

	metrics := fin.compute_distribution_metrics(market_returns, allocator)

	fmt.printf("Market Return Distribution:\n")
	fmt.printf("  Mean: %.4f\n", metrics.mean)
	fmt.printf("  Variance: %.4f\n", metrics.variance)
	fmt.printf("  Skewness: %.4f\n", metrics.skewness)
	fmt.printf("  Kurtosis: %.4f\n", metrics.kurtosis)
	fmt.printf("  VaR (95%%): %.4f\n", metrics.value_at_risk_95)
	fmt.printf("  VaR (99%%): %.4f\n", metrics.value_at_risk_99)
	fmt.printf("  Expected Shortfall (95%%): %.4f\n", metrics.expected_shortfall_95)

	// ========================================================================
	// Test 10: Continuous Pricing with State Price Density
	// ========================================================================
	fmt.println("\n--- Test 10: Continuous Pricing ---")

	// Define a call option payoff function
	call_payoff_fn :: proc(state: f64) -> f64 {
		return math.max(state - 105.0, 0.0)
	}

	continuous_result := fin.price_payoff_continuous(call_payoff_fn, &spd, allocator)
	defer delete(continuous_result.state_prices, allocator)

	fmt.printf("Continuous Call Option Pricing:\n")
	fmt.printf("  Price: %.4f\n", continuous_result.price)
	fmt.printf("  Expected Payoff: %.4f\n", continuous_result.expected_payoff)

	fmt.println("\n✓ PDPM test completed!")
}
