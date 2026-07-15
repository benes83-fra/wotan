package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:mem"

advanced_derivatives_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Advanced Derivatives: Differentiable Monte Carlo ===\n")

	// Asian Call Option Parameters
	S_0 := 100.0 // Initial spot price
	K := 100.0 // Strike price (At-The-Money)
	T := 1.0 // 1 year to maturity
	r := 0.05 // 5% risk-free rate
	sigma := 0.20 // 20% volatility

	n_paths := 10000 // Number of Monte Carlo paths
	n_steps := 252 // Daily time steps

	fmt.printf("Pricing Asian Call Option:\n")
	fmt.printf("  S_0 = %.2f, K = %.2f, T = %.2f, r = %.2f, σ = %.2f\n", S_0, K, T, r, sigma)
	fmt.printf("  Monte Carlo: %d paths, %d steps\n\n", n_paths, n_steps)

	fmt.println(
		"Running Differentiable Monte Carlo (computing Price, Delta, Vega simultaneously)...",
	)

	price, delta, vega := fin.monte_carlo_asian_option(
		S_0,
		K,
		T,
		r,
		sigma,
		n_paths,
		n_steps,
		allocator,
	)

	fmt.println("\n--- Results ---")
	fmt.printf("  Option Price: $%.4f\n", price)
	fmt.printf("  Delta (∂V/∂S): %.4f\n", delta)
	fmt.printf("  Vega  (∂V/∂σ): %.4f\n", vega)


	fmt.println("\n✓ Advanced Derivatives test completed!")
}
exotic_options_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Exotic Options Pricing ===\n")

	// Test 1: Up-and-Out Barrier Asian
	fmt.println("--- Up-and-Out Barrier Asian Call ---")
	fmt.println("Parameters: S=100, K=100, T=1, r=0.05, σ=0.2, Barrier=120")

	barrier_price, barrier_delta, barrier_vega := fin.monte_carlo_barrier_asian_option(
		100.0,
		100.0,
		1.0,
		0.05,
		0.20,
		120.0,
		10000,
		252,
		allocator,
	)

	fmt.printf("  Price: $%.4f\n", barrier_price)
	fmt.printf("  Delta: %.4f\n", barrier_delta)
	fmt.printf("  Vega:  %.4f\n", barrier_vega)
	fmt.println("  Note: Barrier reduces price vs vanilla Asian (knockout risk)")

	// Test 2: 2-Asset Basket Call
	fmt.println("\n--- 2-Asset Basket Call ---")
	fmt.println("Parameters: S1=100, S2=100, K=100, T=1, r=0.05, σ1=0.2, σ2=0.25, ρ=0.5")

	basket_price, basket_delta1, basket_delta2, basket_vega1, basket_vega2 :=
		fin.monte_carlo_basket_option(
			100.0,
			100.0,
			100.0,
			1.0,
			0.05,
			0.20,
			0.25,
			0.5,
			0.5,
			0.5,
			10000,
			252,
			allocator,
		)

	fmt.printf("  Price: $%.4f\n", basket_price)
	fmt.printf("  Delta1: %.4f\n", basket_delta1)
	fmt.printf("  Delta2: %.4f\n", basket_delta2)
	fmt.printf("  Vega1:  %.4f\n", basket_vega1)
	fmt.printf("  Vega2:  %.4f\n", basket_vega2)
	fmt.println("  Note: Diversification reduces price vs single-asset option")
	fmt.println("\n--- Fixed Strike Lookback Call ---")
	fmt.println("Parameters: S=100, K=100, T=1, r=0.05, σ=0.2")

	lookback_price, lookback_delta, lookback_vega := fin.monte_carlo_lookback_call_option(
		100.0,
		100.0,
		1.0,
		0.05,
		0.20,
		10000,
		252,
		allocator,
	)
	fmt.printf(" Price: $%.4f\n", lookback_price)
	fmt.printf("  Delta: %.4f\n", lookback_delta)
	fmt.printf("  Vega: %.4f\n", lookback_vega)

	fmt.println("\n✓ Exotic Options test completed!")
}
