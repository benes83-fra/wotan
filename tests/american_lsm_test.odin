package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

american_lsm_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    LONGSTAFF-SCHWARTZ MONTE CARLO: AMERICAN OPTIONS")
	fmt.println("======================================================================\n")

	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20

	fmt.println("Parameters: S=100, K=100, T=1Y, r=5%, σ=20%")
	fmt.println("Instrument: American Put Option\n")

	// 1. Baseline: Binomial Tree (2000 steps)
	fmt.println("1. Baseline: Binomial Tree (2000 steps)")
	fmt.println("   ----------------------------------------------------------------------")
	bin_result := fin.american_put_binomial(S, K, T, r, sigma, 0.0, 2000, allocator)

	fmt.printf("   %-20s | $%12.4f\n", "American Put Price", bin_result.price)
	fmt.printf("   %-20s | %13.4f\n", "Delta", bin_result.delta)
	fmt.printf("   %-20s | %13.4f\n", "Gamma", bin_result.gamma)

	// 2. Longstaff-Schwartz Monte Carlo (50,000 paths, 50 exercise dates)
	fmt.println("2. Longstaff-Schwartz Monte Carlo (50,000 paths, 50 exercise dates)")
	fmt.println("   ----------------------------------------------------------------------")

	// n_paths = 50000, n_steps = 100, n_exercise_dates = 50, poly_degree = 3
	lsm_result := fin.lsm_american_put(S, K, T, r, sigma, 50000, 100, 50, 3, allocator)

	fmt.printf("   %-20s | $%12.4f\n", "American Put Price", lsm_result.price)
	fmt.printf("   %-20s | %13.4f\n", "Delta", lsm_result.delta)
	fmt.printf("   %-20s | %13.4f\n", "Gamma", lsm_result.gamma)
	fmt.printf("   %-20s | %13.4f\n", "Vega", lsm_result.vega)

	// 3. Error Analysis (LSM vs Binomial)
	fmt.println("\n3. Error Analysis (LSM vs Binomial)")
	fmt.println("   ----------------------------------------------------------------------")
	price_err := (lsm_result.price - bin_result.price) / bin_result.price * 100.0
	delta_err := (lsm_result.delta - bin_result.delta) / math.abs(bin_result.delta) * 100.0

	fmt.printf("   %-20s | %13.4f%%\n", "Price Error", price_err)
	fmt.printf("   %-20s | %13.4f%%\n", "Delta Error", delta_err)

	// 4. Convergence Analysis
	fmt.println("\n4. Convergence Analysis (Increasing Paths)")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-15s | %-15s | %-15s\n", "n_paths", "LSM Price", "Error (%)")
	fmt.println("   ----------------------------------------------------------------------")

	paths := []int{1000, 5000, 10000, 20000, 50000}
	for n in paths {
		res := fin.lsm_american_put(S, K, T, r, sigma, n, 50, 10, 2, allocator)
		err := (res.price - bin_result.price) / bin_result.price * 100.0
		fmt.printf("   %-15d | $%12.4f | %13.4f%%\n", n, res.price, err)
	}

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • LSM converges to the true American option price as n_paths → ∞")
	fmt.println("   • With 10,000+ paths, LSM matches binomial trees to within ~1-2%")
	fmt.println("   • LSM enables American pricing for ANY stochastic process")
	fmt.println("   • This unlocks American Asians, American Lookbacks, American Barriers")
	fmt.println("======================================================================\n")
}
