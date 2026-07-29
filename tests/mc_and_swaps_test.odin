package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

unified_mc_test :: proc(allocator: mem.Allocator) {
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
}
swaps_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    INTEREST RATE SWAPS (CURVE-AWARE)")
	fmt.println("======================================================================\n")

	notional := 100_000_000.0 // $100M

	// 1. Define market par swap rates (upward sloping curve)
	tenors := []f64{1.0, 2.0, 3.0, 5.0, 7.0, 10.0}
	market_rates := []f64{0.0450, 0.0460, 0.0465, 0.0475, 0.0480, 0.0490}

	fmt.println("1. Bootstrapping Yield Curve from Market Par Rates")
	fmt.println("   ----------------------------------------------------------------------")
	for i in 0 ..< len(tenors) {
		fmt.printf("   %-5.1fY Par Swap Rate: %6.4f%%\n", tenors[i], market_rates[i] * 100.0)
	}

	// ✅ FIX: Corrected arguments to match the 5-parameter signature
	curve := fin.bootstrap_yield_curve_from_swaps(
		tenors,
		market_rates,
		0.25, // Quarterly payments
		.ACT_360,
		allocator,
	)
	defer fin.free_yield_curve(&curve, allocator)

	fmt.println("\n   Bootstrapped Zero Rates:")
	for i in 0 ..< len(curve.tenors) {
		fmt.printf(
			"   %-5.1fY Zero Rate:     %6.4f%%\n",
			curve.tenors[i],
			curve.zero_rates[i] * 100.0,
		)
	}

	// 2. Price a specific swap using the bootstrapped curve
	fmt.println("\n2. Pricing a 5-Year Payer Swap (Pay 4.80% Fixed)")
	fmt.println("   ----------------------------------------------------------------------")

	// Create a swap that pays 4.80% fixed (slightly above the 5Y par rate of 4.75%)
	payer_swap := fin.create_payer_swap(notional, 0.0480, 5.0, 0.25, .ACT_360)

	// Price it against the curve
	result := fin.price_swap(payer_swap, &curve, allocator)

	fmt.printf("   %-25s | $%15.2f\n", "NPV (Receiver - Payer)", result.npv)
	fmt.printf("   %-25s | %15.4f%%\n", "Market Par Swap Rate", result.par_swap_rate * 100.0)
	fmt.printf("   %-25s | $%15.2f\n", "PV01 (per 1bp)", result.pv01)
	fmt.printf("   %-25s | %15.4f\n", "Modified Duration", result.modified_duration)

	// 3. Verify Par Swap Rate calculation
	fmt.println("\n3. Curve Metrics Verification")
	fmt.println("   ----------------------------------------------------------------------")

	// The par rate for 5Y should be very close to the input 4.75%
	verified_par := fin.compute_par_swap_rate(5.0, &curve, 0.25, .ACT_360)
	fmt.printf("   %-25s | %15.4f%%\n", "Computed 5Y Par Rate", verified_par * 100.0)

	verified_pv01 := fin.compute_swap_pv01(5.0, notional, &curve, 0.25, .ACT_360)
	fmt.printf("   %-25s | $%15.2f\n", "Computed 5Y PV01", verified_pv01)

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • The floating leg is valued using the exact no-arbitrage identity:")
	fmt.println("     PV_float = Notional × (DF_start - DF_end)")
	fmt.println("   • This avoids interpolation errors and perfectly matches market practice.")
	fmt.println(
		"   • Because we pay 4.80% but the market par rate is ~4.75%, the NPV is negative.",
	)
	fmt.println("======================================================================\n")
}
