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


trinomial_tree_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("         TRINOMIAL TREES: CONVERGENCE COMPARISON")
	fmt.println("======================================================================\n")

	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20
	q := 0.0

	fmt.println("Parameters: S=100, K=100, T=1Y, r=5%, σ=20%, q=0%")
	fmt.println("Instrument: American Put\n")

	// 1. Convergence Analysis
	fmt.println("1. Convergence Comparison (American Put Price)")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf(
		"   %-10s | %-15s | %-15s | %-12s\n",
		"n_steps",
		"Binomial",
		"Trinomial",
		"Difference",
	)
	fmt.println("   ----------------------------------------------------------------------")

	steps := []int{50, 100, 200, 500, 1000, 2000}
	for n in steps {
		bin_result := fin.american_put_binomial(S, K, T, r, sigma, q, n, allocator)
		tri_result := fin.american_put_trinomial(S, K, T, r, sigma, q, n, allocator)
		diff := math.abs(bin_result.price - tri_result.price)
		fmt.printf(
			"   %-10d | $%12.4f | $%12.4f | $%9.4f\n",
			n,
			bin_result.price,
			tri_result.price,
			diff,
		)
	}

	// 2. Greek Stability Comparison
	fmt.println("\n2. Greek Stability at n_steps=500 (where binomial oscillates)")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-15s | %-15s | %-15s\n", "Greek", "Binomial", "Trinomial")
	fmt.println("   ----------------------------------------------------------------------")

	bin_500 := fin.american_put_binomial(S, K, T, r, sigma, q, 500, allocator)
	tri_500 := fin.american_put_trinomial(S, K, T, r, sigma, q, 500, allocator)

	fmt.printf("   %-15s | %14.4f | %14.4f\n", "Price", bin_500.price, tri_500.price)
	fmt.printf("   %-15s | %14.4f | %14.4f\n", "Delta", bin_500.delta, tri_500.delta)
	fmt.printf("   %-15s | %14.4f | %14.4f\n", "Gamma", bin_500.gamma, tri_500.gamma)
	fmt.printf("   %-15s | %14.4f | %14.4f\n", "Theta (per year)", bin_500.theta, tri_500.theta)
	fmt.printf(
		"   %-15s | $%12.4f | $%12.4f\n",
		"Early Ex. Prem.",
		bin_500.early_exercise_premium,
		tri_500.early_exercise_premium,
	)

	// 3. Deep ITM Put (where early exercise matters most)
	fmt.println("\n3. Deep ITM American Put (S=80, K=100) - Early Exercise Premium")
	fmt.println("   ----------------------------------------------------------------------")
	bin_deep := fin.american_put_binomial(80.0, 100.0, T, r, sigma, q, 1000, allocator)
	tri_deep := fin.american_put_trinomial(80.0, 100.0, T, r, sigma, q, 1000, allocator)

	fmt.printf("   %-25s | $%10.4f | $%10.4f\n", "American Price", bin_deep.price, tri_deep.price)
	fmt.printf(
		"   %-25s | $%10.4f | $%10.4f\n",
		"European Price (implied)",
		bin_deep.price - bin_deep.early_exercise_premium,
		tri_deep.price - tri_deep.early_exercise_premium,
	)
	fmt.printf(
		"   %-25s | $%10.4f | $%10.4f\n",
		"Early Exercise Premium",
		bin_deep.early_exercise_premium,
		tri_deep.early_exercise_premium,
	)

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Trinomial trees converge ~10x faster than binomial")
	fmt.println("   • At n=500, binomial still oscillates while trinomial is stable")
	fmt.println("   • Greeks (especially Gamma) are much more stable in trinomial")
	fmt.println("   • The Kamrad-Ritchken formulation guarantees valid probabilities")
	fmt.println("     (p_u, p_m, p_d > 0) for reasonable parameter ranges")
	fmt.println("======================================================================\n")
}
