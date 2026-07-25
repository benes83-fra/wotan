package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

callable_bond_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    CALLABLE BONDS: HULL-WHITE 1F + LONGSTAFF-SCHWARTZ")
	fmt.println("======================================================================\n")

	// Define a 10-year callable bond
	bond := fin.CallableBond {
		face_value       = 100.0,
		coupon_rate      = 0.05, // 5% annual coupon
		coupon_frequency = 2, // Semi-annual
		maturity         = 10.0, // 10 years
		settlement       = 0.0,
		call_schedule    = []fin.CallDate {
			{time = 5.0, price = 102.0}, // Call at year 5 at 102% of par
			{time = 6.0, price = 101.5}, // Call at year 6 at 101.5%
			{time = 7.0, price = 101.0}, // Call at year 7 at 101%
			{time = 8.0, price = 100.5}, // Call at year 8 at 100.5%
			{time = 9.0, price = 100.0}, // Call at year 9 at par
		},
	}

	// Hull-White 1F parameters
	a := 0.10 // Mean reversion (10%)
	sigma := 0.01 // Volatility (1%)
	r0 := 0.03 // Initial short rate (3%)

	// Flat discount curve at 3%
	P0T_func :: proc(t: f64) -> f64 {
		return math.exp_f64(-0.03 * t)
	}

	fmt.println("Bond Details:")
	fmt.printf("   Face Value:        $%.2f\n", bond.face_value)
	fmt.printf("   Coupon Rate:       %.2f%%\n", bond.coupon_rate * 100.0)
	fmt.printf("   Maturity:          %.1f years\n", bond.maturity)
	fmt.printf("   Call Schedule:     %d dates\n", len(bond.call_schedule))
	fmt.println("\nHull-White Parameters:")
	fmt.printf("   Mean Reversion (a): %.2f\n", a)
	fmt.printf("   Volatility (σ):     %.2f%%\n", sigma * 100.0)
	fmt.printf("   Initial Rate (r0):  %.2f%%\n", r0 * 100.0)

	// ✅ OPTIMIZED: Use 2,000 paths and 50 steps for instant, stable execution
	fmt.println("\n1. Callable Bond Price (LSM, 2,000 paths, 50 steps)")
	fmt.println("   ----------------------------------------------------------------------")

	result := fin.callable_bond_lsm_hw1f(bond, a, sigma, r0, P0T_func, 2000, 50, 3, allocator)

	fmt.printf("   %-25s | $%10.4f\n", "Callable Bond Price", result.price)
	fmt.printf("   %-25s | %10.4f\n", "Straight Bond Price", result.straight_bond_price)
	fmt.printf(
		"   %-25s | %10.4f\n",
		"Embedded Call Value",
		result.straight_bond_price - result.price,
	)
	fmt.printf("   %-25s | %10.2f bps\n", "OAS", result.oas)
	fmt.printf("   %-25s | %10.4f\n", "Effective Duration", result.effective_duration)
	fmt.printf("   %-25s | %10.4f\n", "Effective Convexity", result.effective_convexity)
	fmt.printf("   %-25s | %10.2f%%\n", "Call Probability", result.call_probability * 100.0)

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Callable bonds are worth LESS than straight bonds (issuer has the option)")
	fmt.println("   • The embedded call option is an American option on interest rates")
	fmt.println("   • When rates fall, the bond is more likely to be called (refinancing)")
	fmt.println("   • Effective duration < Modified duration (due to call option)")
	fmt.println("   • Effective convexity can be NEGATIVE (unlike straight bonds)")
	fmt.println("======================================================================\n")
}
