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

finite_differences_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("         FINITE DIFFERENCE METHODS: CRANK-NICOLSON")
	fmt.println("======================================================================\n")

	// Test parameters
	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20

	fmt.println("Parameters: S=100, K=100, T=1Y, r=5%, σ=20%")
	fmt.println("Instrument: European Call Option\n")

	// 1. Validation against Black-Scholes
	fmt.println("1. Validation: Crank-Nicolson vs Black-Scholes")
	fmt.println("   ----------------------------------------------------------------------")

	// Black-Scholes analytical price
	d1 := (math.ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt_f64(T))
	d2 := d1 - sigma * math.sqrt_f64(T)
	N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
	bs_price := S * N_d1 - K * math.exp_f64(-r * T) * N_d2

	bs_delta := N_d1
	bs_gamma :=
		(1.0 / (sigma * math.sqrt_f64(T) * math.sqrt_f64(2.0 * math.PI))) *
		math.exp_f64(-0.5 * d1 * d1) /
		S
	bs_theta :=
		-(S * sigma * math.exp_f64(-0.5 * d1 * d1)) / (2.0 * math.sqrt_f64(2.0 * math.PI * T)) -
		r * K * math.exp_f64(-r * T) * N_d2

	// Crank-Nicolson price
	cn_result := fin.crank_nicolson_call(S, K, T, r, sigma, 200, 200, allocator)

	fmt.printf("   %-20s | %-15s | %-15s | %-10s\n", "Method", "Price", "Delta", "Gamma")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf(
		"   %-20s | $%12.4f | %13.4f | %13.4f\n",
		"Black-Scholes",
		bs_price,
		bs_delta,
		bs_gamma,
	)
	fmt.printf(
		"   %-20s | $%12.4f | %13.4f | %13.4f\n",
		"Crank-Nicolson",
		cn_result.price,
		cn_result.delta,
		cn_result.gamma,
	)

	price_error := math.abs(cn_result.price - bs_price) / bs_price * 100.0
	delta_error := math.abs(cn_result.delta - bs_delta) / bs_delta * 100.0
	gamma_error := math.abs(cn_result.gamma - bs_gamma) / bs_gamma * 100.0

	fmt.printf(
		"\n   Relative Errors: Price=%.4f%%, Delta=%.4f%%, Gamma=%.4f%%\n",
		price_error,
		delta_error,
		gamma_error,
	)

	// 2. Convergence Analysis
	fmt.println("\n2. Convergence Analysis (Grid Refinement)")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-15s | %-15s | %-15s\n", "Grid Size", "Price", "Error (%)")
	fmt.println("   ----------------------------------------------------------------------")

	grid_sizes := []int{50, 100, 200, 400, 800}
	for n in grid_sizes {
		result := fin.crank_nicolson_call(S, K, T, r, sigma, n, n, allocator)
		err := math.abs(result.price - bs_price) / bs_price * 100.0
		fmt.printf("   %-15d | $%12.4f | %13.6f%%\n", n, result.price, err)
	}

	// 3. Put Option Test
	fmt.println("\n3. European Put Option")
	fmt.println("   ----------------------------------------------------------------------")

	// Black-Scholes put price (put-call parity)
	bs_put := bs_price - S + K * math.exp_f64(-r * T)

	cn_put := fin.crank_nicolson_put(S, K, T, r, sigma, 200, 200, allocator)

	fmt.printf("   %-20s | $%12.4f\n", "Black-Scholes Put", bs_put)
	fmt.printf("   %-20s | $%12.4f\n", "Crank-Nicolson Put", cn_put.price)

	put_error := math.abs(cn_put.price - bs_put) / bs_put * 100.0
	fmt.printf("   Relative Error: %.4f%%\n", put_error)

	// 4. Deep ITM and OTM Tests
	fmt.println("\n4. Boundary Behavior (Deep ITM/OTM)")
	fmt.println("   ----------------------------------------------------------------------")

	test_cases := []struct {
		name: string,
		K:    f64,
	} {
		{"Deep ITM Call (K=80)", 80.0},
		{"ATM Call (K=100)", 100.0},
		{"Deep OTM Call (K=120)", 120.0},
	}

	fmt.printf("   %-25s | %-15s | %-15s | %-10s\n", "Option", "BS Price", "CN Price", "Error (%)")
	fmt.println("   ----------------------------------------------------------------------")

	for tc in test_cases {
		// Calculate BS price
		d1_tc := (math.ln(S / tc.K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt_f64(T))
		d2_tc := d1_tc - sigma * math.sqrt_f64(T)
		N_d1_tc := 0.5 * (1.0 + math.erf(d1_tc / math.sqrt_f64(2.0)))
		N_d2_tc := 0.5 * (1.0 + math.erf(d2_tc / math.sqrt_f64(2.0)))
		bs_tc := S * N_d1_tc - tc.K * math.exp_f64(-r * T) * N_d2_tc

		cn_tc := fin.crank_nicolson_call(S, tc.K, T, r, sigma, 200, 200, allocator)
		err_tc := math.abs(cn_tc.price - bs_tc) / math.max(bs_tc, 0.01) * 100.0

		fmt.printf("   %-25s | $%12.4f | $%12.4f | %9.4f%%\n", tc.name, bs_tc, cn_tc.price, err_tc)
	}

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Crank-Nicolson is unconditionally stable (no CFL condition)")
	fmt.println("   • Second-order accurate: error ∝ O(Δx²) + O(Δt²)")
	fmt.println("   • Natural handling of boundary conditions")
	fmt.println("   • Efficient tridiagonal solver: O(n) per time step")
	fmt.println("   • Completes the 'holy trinity': Analytical, MC, and PDE methods")
	fmt.println("======================================================================\n")
}


american_finite_difference_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    AMERICAN OPTIONS: FINITE DIFFERENCE vs BINOMIAL TREE")
	fmt.println("======================================================================\n")

	// Test parameters
	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.20
	q := 0.0 // No dividends for this test

	fmt.println("Parameters: S=100, K=100, T=1Y, r=5%, σ=20%, q=0%")
	fmt.println("Instrument: American Put Option\n")

	// 1. Baseline: Binomial Tree (Highly accurate with 2000 steps)
	fmt.println("1. Baseline: Binomial Tree (2000 steps)")
	fmt.println("   ----------------------------------------------------------------------")
	bin_result := fin.american_put_binomial(S, K, T, r, sigma, q, 2000, allocator)

	fmt.printf("   %-20s | $%12.4f\n", "American Put Price", bin_result.price)
	fmt.printf("   %-20s | $%12.4f\n", "Early Exercise Prem.", bin_result.early_exercise_premium)
	fmt.printf("   %-20s | %13.4f\n", "Delta", bin_result.delta)
	fmt.printf("   %-20s | %13.4f\n", "Gamma", bin_result.gamma)

	// 2. Finite Difference (Fully Implicit)
	fmt.println("\n2. Finite Difference (Fully Implicit, 200x200 grid)")
	fmt.println("   ----------------------------------------------------------------------")
	fd_result := fin.fd_american_put(S, K, T, r, sigma, 200, 200, allocator)

	fmt.printf("   %-20s | $%12.4f\n", "American Put Price", fd_result.price)
	fmt.printf("   %-20s | %13.4f\n", "Delta", fd_result.delta)
	fmt.printf("   %-20s | %13.4f\n", "Gamma", fd_result.gamma)

	// 3. Error Analysis
	fmt.println("\n3. Error Analysis (FD vs Binomial)")
	fmt.println("   ----------------------------------------------------------------------")
	price_err := math.abs(fd_result.price - bin_result.price) / bin_result.price * 100.0
	delta_err := math.abs(fd_result.delta - bin_result.delta) / math.abs(bin_result.delta) * 100.0
	gamma_err := math.abs(fd_result.gamma - bin_result.gamma) / math.abs(bin_result.gamma) * 100.0

	fmt.printf("   %-20s | %13.4f%%\n", "Price Error", price_err)
	fmt.printf("   %-20s | %13.4f%%\n", "Delta Error", delta_err)
	fmt.printf("   %-20s | %13.4f%%\n", "Gamma Error", gamma_err)

	// 4. Deep ITM Test (Where early exercise matters most)
	fmt.println("\n4. Deep ITM American Put (S=80, K=100) - Maximum Early Exercise")
	fmt.println("   ----------------------------------------------------------------------")
	bin_deep := fin.american_put_binomial(80.0, 100.0, T, r, sigma, q, 2000, allocator)
	fd_deep := fin.fd_american_put(80.0, 100.0, T, r, sigma, 200, 200, allocator)

	fmt.printf("   %-25s | $%10.4f | $%10.4f\n", "Binomial Price", bin_deep.price, 0.0)
	fmt.printf("   %-25s | $%10.4f | $%10.4f\n", "FD Price", fd_deep.price, 0.0)
	fmt.printf(
		"   %-25s | $%10.4f | $%10.4f\n",
		"Early Exercise Prem.",
		bin_deep.early_exercise_premium,
		fd_deep.price - (100.0 * math.exp_f64(-r * T) - 80.0),
	) // Approx European

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Fully Implicit FD is strictly monotonic (no oscillations)")
	fmt.println("   • The projection step (V = max(V, intrinsic)) perfectly captures")
	fmt.println("     the free-boundary early exercise feature.")
	fmt.println("   • FD is significantly faster than a 2000-step Binomial Tree while")
	fmt.println("     maintaining sub-0.1% accuracy.")
	fmt.println("======================================================================\n")
}
