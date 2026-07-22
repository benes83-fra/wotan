package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

american_options_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("         AMERICAN OPTIONS: BINOMIAL TREE PRICING")
	fmt.println("======================================================================\n")

	// Test parameters
	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20
	q := 0.0 // No dividends
	n_steps := 1000

	fmt.println("1. American Put Option (Early Exercise Premium)")
	fmt.println("   Parameters: S=100, K=100, T=1Y, r=5%, σ=20%, q=0%")
	fmt.println("   ----------------------------------------------------------------------")

	put_result := fin.american_put_binomial(S, K, T, r, sigma, q, n_steps, allocator)

	fmt.printf("   %-30s | $%8.4f\n", "American Put Price", put_result.price)
	fmt.printf(
		"   %-30s | $%8.4f\n",
		"European Put Price (implied)",
		put_result.price - put_result.early_exercise_premium,
	)
	fmt.printf("   %-30s | $%8.4f\n", "Early Exercise Premium", put_result.early_exercise_premium)
	fmt.printf("   %-30s | %8.4f\n", "Delta", put_result.delta)
	fmt.printf("   %-30s | %8.4f\n", "Gamma", put_result.gamma)
	fmt.printf("   %-30s | %8.4f\n", "Theta (per year)", put_result.theta)

	fmt.println("\n   💡 American puts have early exercise premium because it's optimal")
	fmt.println("   to exercise early when the put is deep in-the-money (you receive")
	fmt.println("   the strike immediately and can earn interest on it).")

	// Test 2: American Call with Dividends
	fmt.println("\n2. American Call with Dividends")
	fmt.println("   Parameters: S=100, K=100, T=1Y, r=5%, σ=20%, q=3%")
	fmt.println("   ----------------------------------------------------------------------")

	q_div := 0.03
	call_result := fin.american_call_binomial(S, K, T, r, sigma, q_div, n_steps, allocator)

	fmt.printf("   %-30s | $%8.4f\n", "American Call Price", call_result.price)
	fmt.printf("   %-30s | $%8.4f\n", "Early Exercise Premium", call_result.early_exercise_premium)

	fmt.println("\n   💡 American calls on dividend-paying stocks have early exercise premium")
	fmt.println("   because it's optimal to exercise just before an ex-dividend date to")
	fmt.println("   capture the dividend.")

	// Test 3: Convergence analysis
	fmt.println("\n3. Convergence Analysis (American Put)")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-15s | %-15s | %-15s\n", "n_steps", "Price", "Early Ex. Prem.")
	fmt.println("   ----------------------------------------------------------------------")

	steps := []int{100, 500, 1000, 2000, 5000}
	for n in steps {
		result := fin.american_put_binomial(S, K, T, r, sigma, q, n, allocator)
		fmt.printf(
			"   %-15d | $%12.4f | $%12.4f\n",
			n,
			result.price,
			result.early_exercise_premium,
		)
	}

	fmt.println("\n   💡 The binomial tree converges to the true American option price")
	fmt.println("   as n_steps → ∞. With 1000+ steps, the price is accurate to ~4 decimals.")

	fmt.println("======================================================================\n")
}
