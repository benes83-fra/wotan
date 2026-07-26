package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

unified_mc_and_swaps_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    UNIFIED MONTE CARLO ENGINE + INTEREST RATE SWAPS")
	fmt.println("======================================================================\n")

	// ========================================================================
	// PART 1: UNIFIED MC ENGINE DEMONSTRATION
	// ========================================================================
	fmt.println("1. Unified Monte Carlo Engine - Shared Primitives")
	fmt.println("   ----------------------------------------------------------------------")

	S_0 := 100.0
	r := 0.05
	sigma := 0.20
	T := 1.0
	n_paths := 10000
	n_steps := 252

	// Generate paths using the unified primitive
	norm_data := fin.mc_generate_normals(n_paths * n_steps, allocator)
	defer delete(norm_data, allocator)

	S_paths := fin.mc_generate_gbm_paths(S_0, r, sigma, T, n_paths, n_steps, norm_data, allocator)
	defer delete(S_paths, allocator)

	// Price a vanilla European call using the paths
	total_payoff := 0.0
	K := 100.0
	for path in 0 ..< n_paths {
		S_T := S_paths[path * (n_steps + 1) + n_steps]
		payoff := math.max(S_T - K, 0.0)
		total_payoff += payoff
	}
	mc_price := (total_payoff / f64(n_paths)) * math.exp_f64(-r * T)

	// Compare with Black-Scholes
	bs_price := fin._bs_call_price(S_0, K, T, r, sigma)

	fmt.printf("   %-30s | $%10.4f\n", "MC Price (unified engine)", mc_price)
	fmt.printf("   %-30s | $%10.4f\n", "Black-Scholes Price", bs_price)
	fmt.printf(
		"   %-30s | %10.4f%%\n",
		"MC Error",
		math.abs(mc_price - bs_price) / bs_price * 100.0,
	)

	// Demonstrate Asian averaging primitive
	fmt.println("\n   Asian Average (first path):")
	asian_avg := fin.mc_compute_asian_average(S_paths, 0, n_steps)
	fmt.printf("   %-30s | $%10.4f\n", "Arithmetic Average", asian_avg)

	// Demonstrate barrier checking primitive
	barrier := 120.0
	breached := fin.mc_check_barrier_breached(S_paths, 0, n_steps, barrier, true)
	fmt.printf("   %-30s | %v\n", "Up-and-Out Breached (first path)", breached)

	// Demonstrate lookback primitive
	max_S := fin.mc_compute_lookback_max(S_paths, 0, n_steps)
	fmt.printf("   %-30s | $%10.4f\n", "Running Maximum (first path)", max_S)

	// ========================================================================
	// PART 2: INTEREST RATE SWAPS
	// ========================================================================
	fmt.println("\n2. Interest Rate Swaps (IRS)")
	fmt.println("   ----------------------------------------------------------------------")

	notional := 100_000_000.0 // $100M
	r_swap := 0.03 // 3% flat curve

	// Compute par swap rate for 5Y and 10Y
	par_5y := fin.compute_par_swap_rate(5.0, r_swap, 0.25, .ACT_360)
	par_10y := fin.compute_par_swap_rate(10.0, r_swap, 0.25, .ACT_360)

	fmt.printf("   %-30s | %.4f%%\n", "5Y Par Swap Rate", par_5y * 100.0)
	fmt.printf("   %-30s | %.4f%%\n", "10Y Par Swap Rate", par_10y * 100.0)

	// Create and price a payer swap (pay fixed, receive floating)
	payer_swap := fin.create_payer_swap(notional, 0.035, 10.0, 0.25, .ACT_360)
	payer_result := fin.price_swap(payer_swap, r_swap, allocator)

	fmt.println("\n   10Y Payer Swap (pay 3.5% fixed, receive floating):")
	fmt.printf("   %-30s | $%15.2f\n", "NPV", payer_result.npv)
	fmt.printf("   %-30s | %.4f%%\n", "Par Swap Rate", payer_result.par_swap_rate * 100.0)
	fmt.printf("   %-30s | $%15.2f\n", "PV01 (per 1bp)", payer_result.pv01)
	fmt.printf("   %-30s | %10.4f\n", "Modified Duration", payer_result.modified_duration)

	// Create and price a receiver swap
	receiver_swap := fin.create_receiver_swap(notional, 0.025, 5.0, 0.25, .ACT_360)
	receiver_result := fin.price_swap(receiver_swap, r_swap, allocator)

	fmt.println("\n   5Y Receiver Swap (receive 2.5% fixed, pay floating):")
	fmt.printf("   %-30s | $%15.2f\n", "NPV", receiver_result.npv)
	fmt.printf("   %-30s | %.4f%%\n", "Par Swap Rate", receiver_result.par_swap_rate * 100.0)
	fmt.printf("   %-30s | $%15.2f\n", "PV01 (per 1bp)", receiver_result.pv01)

	// ========================================================================
	// PART 3: SWAP RISK ANALYSIS
	// ========================================================================
	fmt.println("\n3. Swap Risk Metrics Across Maturities")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-8s | %-12s | %-12s | %-12s\n", "Maturity", "Par Rate", "PV01 ($)", "Duration")
	fmt.println("   ----------------------------------------------------------------------")

	maturities := []f64{1.0, 2.0, 5.0, 10.0, 30.0}
	for mat in maturities {
		par_rate := fin.compute_par_swap_rate(mat, r_swap, 0.25, .ACT_360)
		pv01 := fin.compute_swap_pv01(mat, notional, r_swap, 0.25, .ACT_360)
		duration := fin.compute_swap_duration(mat, r_swap, 0.25, .ACT_360)

		fmt.printf(
			"   %-8.0fY | %11.4f%% | $%11.2f | %11.4f\n",
			mat,
			par_rate * 100.0,
			pv01,
			duration,
		)
	}

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Unified MC engine eliminates ~20 duplications of path generation code")
	fmt.println("   • Shared primitives: GBM, Heston, MJD, Hull-White path generators")
	fmt.println("   • Shared payoff primitives: Asian averaging, barrier checking, lookback")
	fmt.println("   • Shared LSM primitives: regression steps for American exercise")
	fmt.println("   • Swaps are the foundation of fixed income (Calypso uses these everywhere)")
	fmt.println("   • Par swap rate = (1 - DF_maturity) / annuity")
	fmt.println("   • PV01 = annuity × notional × 0.0001 (risk per 1bp shift)")
	fmt.println("======================================================================\n")
}
