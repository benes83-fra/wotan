package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

bonds_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    VANILLA BONDS: PRICING & ANALYTICS")
	fmt.println("======================================================================\n")

	// ========================================================================
	// 1. FIXED-RATE BOND
	// ========================================================================
	fmt.println("1. Fixed-Rate Bond (10-year, 5% coupon)")
	fmt.println("   ----------------------------------------------------------------------")

	bond := fin.create_fixed_bond(100.0, 0.05, 10.0, .SemiAnnual, .ACT_ACT)

	// ✅ FIX: Use the struct and rawptr pattern instead of a local closure
	params := fin.FlatCurveParams {
		rate = 0.04,
	}
	result := fin.price_bond_from_curve(bond, fin.flat_curve_proc, rawptr(&params), allocator)

	fmt.printf("   %-25s | $%10.4f\n", "Clean Price", result.clean_price)
	fmt.printf("   %-25s | $%10.4f\n", "Dirty Price", result.dirty_price)
	fmt.printf("   %-25s | $%10.4f\n", "Accrued Interest", result.accrued_interest)
	fmt.printf("   %-25s | %10.4f%%\n", "Yield to Maturity", result.yield_to_maturity * 100.0)
	fmt.printf("   %-25s | %10.4f\n", "Modified Duration", result.modified_duration)
	fmt.printf("   %-25s | %10.4f\n", "Convexity", result.convexity)

	// ========================================================================
	// 2. ZERO-COUPON BOND
	// ========================================================================
	fmt.println("\n2. Zero-Coupon Bond (5-year)")
	fmt.println("   ----------------------------------------------------------------------")

	zero_bond := fin.create_zero_coupon_bond(100.0, 5.0)
	zero_params := fin.FlatCurveParams {
		rate = 0.04,
	}
	zero_result := fin.price_bond_from_curve(
		zero_bond,
		fin.flat_curve_proc,
		rawptr(&zero_params),
		allocator,
	)

	fmt.printf("   %-25s | $%10.4f\n", "Clean Price", zero_result.clean_price)
	fmt.printf("   %-25s | %10.4f%%\n", "Yield to Maturity", zero_result.yield_to_maturity * 100.0)
	fmt.printf("   %-25s | %10.4f\n", "Modified Duration", zero_result.modified_duration)
	fmt.printf("   %-25s | %10.4f\n", "Convexity", zero_result.convexity)

	// ========================================================================
	// 3. YIELD CURVE SENSITIVITY
	// ========================================================================
	fmt.println("\n3. Bond Prices Across Yield Curve")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-10s | %-12s | %-12s | %-12s\n", "Yield", "Price", "Duration", "Convexity")
	fmt.println("   ----------------------------------------------------------------------")

	yields := []f64{0.02, 0.03, 0.04, 0.05, 0.06, 0.07}
	for y in yields {
		curve_params := fin.FlatCurveParams {
			rate = y,
		}
		res := fin.price_bond_from_curve(
			bond,
			fin.flat_curve_proc,
			rawptr(&curve_params),
			allocator,
		)
		fmt.printf(
			"   %9.2f%% | $%11.4f | %11.4f | %11.4f\n",
			y * 100.0,
			res.clean_price,
			res.modified_duration,
			res.convexity,
		)
	}

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Clean price excludes accrued interest (used for quoting)")
	fmt.println("   • Dirty price includes accrued interest (actual settlement price)")
	fmt.println("   • YTM is the internal rate of return if held to maturity")
	fmt.println("   • Modified duration measures price sensitivity to yield changes")
	fmt.println("   • Convexity captures the curvature in the price-yield relationship")
	fmt.println("======================================================================\n")
}
