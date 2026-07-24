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

american_asian_lsm_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    AMERICAN ASIAN OPTIONS: LONGSTAFF-SCHWARTZ MONTE CARLO")
	fmt.println("======================================================================\n")

	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20

	fmt.println("Parameters: S=100, K=100, T=1Y, r=5%, σ=20%")
	fmt.println("Instrument: American Asian Call Option (Arithmetic Average)\n")

	// 1. Price American Asian
	fmt.println("1. American Asian Call (LSM, 50,000 paths, 50 exercise dates)")
	fmt.println("   ----------------------------------------------------------------------")

	// n_paths=50000, n_steps=100, n_exercise_dates=50, poly_degree=3
	asian_result := fin.lsm_american_asian_call(S, K, T, r, sigma, 50000, 100, 50, 3, allocator)

	fmt.printf("   %-20s | $%12.4f\n", "American Asian Price", asian_result.price)
	fmt.printf("   %-20s | %13.4f\n", "Delta", asian_result.delta)
	fmt.printf("   %-20s | %13.4f\n", "Gamma", asian_result.gamma)
	fmt.printf("   %-20s | %13.4f\n", "Vega", asian_result.vega)

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • The American Asian Call has a small but positive early exercise")
	fmt.println("     premium because averaging reduces volatility, making deep ITM")
	fmt.println("     states more likely to be optimal for early exercise.")
	fmt.println("   • This would be virtually impossible to price with Binomial Trees")
	fmt.println("     due to the non-recombining nature of the arithmetic average.")
	fmt.println("======================================================================\n")
}


variance_reduction_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    VARIANCE REDUCTION: ANTITHETIC + CONTROL VARIATE")
	fmt.println("======================================================================\n")

	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20

	fmt.println("Parameters: S=100, K=100, T=1Y, r=5%, σ=20%")
	fmt.println("Instrument: American Put Option\n")

	// 1. Baseline: Plain LSM (5,000 paths)
	fmt.println("1. Plain LSM (5,000 paths, no variance reduction)")
	fmt.println("   ----------------------------------------------------------------------")
	plain := fin.lsm_american_put(S, K, T, r, sigma, 5000, 100, 50, 3, allocator)
	fmt.printf("   %-20s | $%12.4f\n", "Price", plain.price)
	fmt.printf("   %-20s | %13.4f\n", "Delta", plain.delta)

	// 2. Variance-Reduced LSM (5,000 paths = 2,500 original + 2,500 antithetic)
	fmt.println("\n2. Variance-Reduced LSM (5,000 paths, antithetic + control variate)")
	fmt.println("   ----------------------------------------------------------------------")
	reduced := fin.lsm_american_vanilla_reduced(
		S,
		K,
		T,
		r,
		sigma,
		.Put,
		5000,
		100,
		50,
		3,
		allocator,
	)
	fmt.printf("   %-20s | $%12.4f\n", "Price", reduced.price)
	fmt.printf("   %-20s | %13.4f\n", "Delta", reduced.delta)
	fmt.printf("   %-20s | %12.2fx\n", "Variance Reduction", reduced.variance_reduction)

	// 3. Binomial Tree baseline (2000 steps)
	fmt.println("\n3. Binomial Tree Baseline (2000 steps)")
	fmt.println("   ----------------------------------------------------------------------")
	bin := fin.american_put_binomial(S, K, T, r, sigma, 0.0, 2000, allocator)
	fmt.printf("   %-20s | $%12.4f\n", "Price", bin.price)
	fmt.printf("   %-20s | %13.4f\n", "Delta", bin.delta)

	// 4. Error comparison
	fmt.println("\n4. Error Comparison (vs Binomial Tree)")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-25s | %-12s | %-12s\n", "Method", "Price Error", "Delta Error")
	fmt.println("   ----------------------------------------------------------------------")

	plain_price_err := math.abs(plain.price - bin.price) / bin.price * 100.0
	plain_delta_err := math.abs(plain.delta - bin.delta) / math.abs(bin.delta) * 100.0
	fmt.printf(
		"   %-25s | %11.4f%% | %11.4f%%\n",
		"Plain LSM (5k paths)",
		plain_price_err,
		plain_delta_err,
	)

	reduced_price_err := math.abs(reduced.price - bin.price) / bin.price * 100.0
	reduced_delta_err := math.abs(reduced.delta - bin.delta) / math.abs(bin.delta) * 100.0
	fmt.printf(
		"   %-25s | %11.4f%% | %11.4f%%\n",
		"VR LSM (5k paths)",
		reduced_price_err,
		reduced_delta_err,
	)

	// 5. Convergence Analysis
	fmt.println("\n5. Convergence Analysis (Variance-Reduced LSM)")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-12s | %-12s | %-12s | %-12s\n", "n_paths", "Price", "Price Err", "Var Red")
	fmt.println("   ----------------------------------------------------------------------")

	paths := []int{1000, 2500, 5000, 10000, 25000}
	for n in paths {
		res := fin.lsm_american_vanilla_reduced(S, K, T, r, sigma, .Put, n, 100, 50, 3, allocator)
		err := math.abs(res.price - bin.price) / bin.price * 100.0
		fmt.printf(
			"   %-12d | $%10.4f | %11.4f%% | %11.2fx\n",
			n,
			res.price,
			err,
			res.variance_reduction,
		)
	}

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Antithetic Variates: For every path Z, also use -Z.")
	fmt.println("     This halves variance for monotone payoffs (free 2x speedup).")
	fmt.println("   • Control Variate: Use BS European price as a control.")
	fmt.println("     The adjustment β×(MC_euro - BS_euro) removes systematic bias,")
	fmt.println("     yielding 10-100x variance reduction for ATM options.")
	fmt.println("   • Combined: You can achieve <0.5% error with just 5,000 paths,")
	fmt.println("     where plain LSM would need 500,000+ paths for the same accuracy.")
	fmt.println("======================================================================\n")
}
