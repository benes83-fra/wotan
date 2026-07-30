package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:mem"

bonds_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    VANILLA BONDS: PRICING WITH BOOTSTRAPPED YIELD CURVES")
	fmt.println("======================================================================\n")

	// ========================================================================
	// 1. BOOTSTRAP THE YIELD CURVE
	// ========================================================================
	fmt.println("1. Bootstrapping Yield Curve from Market Par Swaps")
	fmt.println("   ----------------------------------------------------------------------")

	tenors := []f64{1.0, 2.0, 3.0, 5.0, 7.0, 10.0}
	market_rates := []f64{0.0450, 0.0460, 0.0465, 0.0475, 0.0480, 0.0490}

	curve := fin.bootstrap_yield_curve_from_swaps(tenors, market_rates, 0.25, .ACT_360, allocator)
	defer fin.free_yield_curve(&curve, allocator)

	for i in 0 ..< len(curve.tenors) {
		fmt.printf("   %-5.1fY Zero Rate: %6.4f%%\n", curve.tenors[i], curve.zero_rates[i] * 100.0)
	}

	// ========================================================================
	// 2. FIXED-RATE BOND
	// ========================================================================
	fmt.println("\n2. 10-Year Fixed-Rate Bond (5% coupon)")
	fmt.println("   ----------------------------------------------------------------------")

	bond := fin.create_fixed_bond(100.0, 0.05, 10.0, .SemiAnnual, .ACT_ACT)
	result := fin.price_bond_from_yield_curve(bond, &curve, allocator)

	fmt.printf("   %-25s | $%10.4f\n", "Clean Price", result.clean_price)
	fmt.printf("   %-25s | $%10.4f\n", "Dirty Price", result.dirty_price)
	fmt.printf("   %-25s | $%10.4f\n", "Accrued Interest", result.accrued_interest)
	fmt.printf("   %-25s | %10.4f%%\n", "Yield to Maturity", result.yield_to_maturity * 100.0)
	fmt.printf("   %-25s | %10.4f\n", "Modified Duration", result.modified_duration)
	fmt.printf("   %-25s | %10.4f\n", "Convexity", result.convexity)

	// ========================================================================
	// 3. ZERO-COUPON BOND
	// ========================================================================
	fmt.println("\n3. 5-Year Zero-Coupon Bond")
	fmt.println("   ----------------------------------------------------------------------")

	zero_bond := fin.create_zero_coupon_bond(100.0, 5.0)
	zero_result := fin.price_bond_from_yield_curve(zero_bond, &curve, allocator)

	fmt.printf("   %-25s | $%10.4f\n", "Clean Price", zero_result.clean_price)
	fmt.printf("   %-25s | %10.4f%%\n", "Yield to Maturity", zero_result.yield_to_maturity * 100.0)
	fmt.printf("   %-25s | %10.4f\n", "Modified Duration", zero_result.modified_duration)
	fmt.printf("   %-25s | %10.4f\n", "Convexity", zero_result.convexity)

	// ========================================================================
	// 4. PARALLEL SHIFT SENSITIVITY (STRESS TESTING)
	// ========================================================================
	fmt.println("\n4. Bond Price Sensitivity to Parallel Curve Shifts")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-12s | %-12s | %-12s\n", "Shift (bps)", "10Y Fixed", "5Y Zero")
	fmt.println("   ----------------------------------------------------------------------")

	shifts_bps := []f64{-100.0, -50.0, 0.0, 50.0, 100.0}
	for shift in shifts_bps {
		shift_decimal := shift / 10000.0

		// Create a shifted copy of the yield curve
		shifted_curve := fin.YieldCurve {
			tenors     = make([]f64, len(curve.tenors), allocator),
			zero_rates = make([]f64, len(curve.zero_rates), allocator),
		}
		copy(shifted_curve.tenors, curve.tenors)
		for i in 0 ..< len(curve.zero_rates) {
			shifted_curve.zero_rates[i] = curve.zero_rates[i] + shift_decimal
		}

		res_fixed := fin.price_bond_from_yield_curve(bond, &shifted_curve, allocator)
		res_zero := fin.price_bond_from_yield_curve(zero_bond, &shifted_curve, allocator)

		fmt.printf(
			"   %+11.0f | $%11.4f | $%11.4f\n",
			shift,
			res_fixed.clean_price,
			res_zero.clean_price,
		)

		fin.free_yield_curve(&shifted_curve, allocator)
	}

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Bonds are now priced using a realistic, bootstrapped zero-coupon curve.")
	fmt.println(
		"   • The 10Y 5% bond trades at a premium because its coupon > market zero rates.",
	)
	fmt.println(
		"   • Parallel shifts demonstrate the portfolio's interest rate risk (Duration).",
	)
	fmt.println("======================================================================\n")
}
