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

	fmt.println("\n✓ Why this is powerful:")
	fmt.println("  • No 'bump and revalue' (finite difference) noise.")
	fmt.println("  • Exact gradients computed through the entire simulation loop.")
	fmt.println("  • Easily extensible to Barrier, Lookback, or Basket options.")

	fmt.println("\n✓ Advanced Derivatives test completed!")
}
