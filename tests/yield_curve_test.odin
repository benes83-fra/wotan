package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:mem"

yield_curve_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    YIELD CURVE BOOTSTRAPPING & INTERPOLATION")
	fmt.println("======================================================================\n")

	// Market par swap rates (e.g., realistic USD SOFR swap curve)
	tenors := []f64{1.0, 2.0, 3.0, 5.0, 7.0, 10.0}
	par_rates := []f64{0.0450, 0.0460, 0.0465, 0.0475, 0.0480, 0.0490} // 4.50% to 4.90%

	fmt.println("1. Market Input (Par Swap Rates)")
	fmt.println("   ----------------------------------------------------------------------")
	for i in 0 ..< len(tenors) {
		fmt.printf("   %-5.1f Year Swap Rate: %6.4f%%\n", tenors[i], par_rates[i] * 100.0)
	}

	// Bootstrap the curve
	curve := fin.bootstrap_yield_curve_from_swaps(tenors, par_rates, 0.25, .ACT_360, allocator)
	defer fin.free_yield_curve(&curve, allocator)

	fmt.println("\n2. Bootstrapped Zero Curve (Continuously Compounded)")
	fmt.println("   ----------------------------------------------------------------------")
	for i in 0 ..< len(curve.tenors) {
		fmt.printf(
			"   %-5.1f Year Zero Rate: %6.4f%%\n",
			curve.tenors[i],
			curve.zero_rates[i] * 100.0,
		)
	}

	fmt.println("\n3. Curve Queries (Interpolation & Forward Rates)")
	fmt.println("   ----------------------------------------------------------------------")

	// Query at 4 years (interpolated between 3Y and 5Y)
	t_query := 4.0
	z_rate := fin.yield_curve_zero_rate(&curve, t_query)
	df := fin.yield_curve_discount_factor(&curve, t_query)
	fmt.printf("   4.0Y Zero Rate:          %6.4f%%\n", z_rate * 100.0)
	fmt.printf("   4.0Y Discount Factor:    %6.6f\n", df)

	// Forward rate between 4Y and 5Y
	fwd_rate := fin.yield_curve_forward_rate(&curve, 4.0, 5.0)
	fmt.printf("   4Y5Y Forward Rate:       %6.4f%%\n", fwd_rate * 100.0)

	fmt.println("\n💡 Key Insights:")
	fmt.println(
		"   • Bootstrapping iteratively solves for zero rates that price par swaps to 0 NPV.",
	)
	fmt.println(
		"   • Linear interpolation on zero rates is standard and prevents static arbitrage.",
	)
	fmt.println(
		"   • This curve can now replace the flat 'r' in the Swap pricer for realistic valuation.",
	)
	fmt.println("======================================================================\n")
}
