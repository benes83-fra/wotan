package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

callable_bond_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    CALLABLE BONDS: HULL-WHITE 1F + LONGSTAFF-SCHWARTZ + OAS")
	fmt.println("======================================================================\n")

	bond := fin.CallableBond {
		face_value       = 100.0,
		coupon_rate      = 0.05,
		coupon_frequency = 2,
		maturity         = 10.0,
		settlement       = 0.0,
		call_schedule    = []fin.CallDate {
			{time = 5.0, price = 102.0},
			{time = 6.0, price = 101.5},
			{time = 7.0, price = 101.0},
			{time = 8.0, price = 100.5},
			{time = 9.0, price = 100.0},
		},
	}

	a := 0.10
	sigma := 0.01
	r0 := 0.03

	P0T_func :: proc(t: f64) -> f64 {
		return math.exp_f64(-0.03 * t)
	}

	fmt.println("Bond Details:")
	fmt.printf("   Face Value:        $%.2f\n", bond.face_value)
	fmt.printf("   Coupon Rate:       %.2f%%\n", bond.coupon_rate * 100.0)
	fmt.printf("   Maturity:          %.1f years\n", bond.maturity)

	// 1. Base Pricing
	fmt.println("\n1. Base Model Pricing (LSM, 2,000 paths)")
	fmt.println("   ----------------------------------------------------------------------")
	result := fin.callable_bond_lsm_hw1f(bond, a, sigma, r0, P0T_func, 2000, 50, 3, allocator)

	fmt.printf("   %-25s | $%10.4f\n", "Model Price (Z-spread=0)", result.price)
	fmt.printf("   %-25s | $%10.4f\n", "Straight Bond Price", result.straight_bond_price)
	fmt.printf("   %-25s | $%10.4f\n", "Embedded Call Value", result.embedded_call_value)
	fmt.printf("   %-25s | %10.2f%%\n", "Call Probability", result.call_probability * 100.0)

	// 2. OAS Solver
	// Let's assume the market is pricing this bond at $115.50 (cheaper than the model's $116.73)
	// This implies the market demands extra yield (OAS) for the call risk.
	market_price := 115.50
	fmt.println("\n2. Option-Adjusted Spread (OAS) Solver")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-25s | $%10.2f\n", "Target Market Price", market_price)

	// Note: OAS solving is computationally heavier (runs MC ~15 times).
	// We use 2,000 paths here for a fast demo. In production, use 5,000-10,000.
	oas_decimal := fin.callable_bond_oas(
		bond,
		a,
		sigma,
		r0,
		P0T_func,
		market_price,
		2000,
		50,
		3,
		allocator,
	)
	oas_bps := oas_decimal * 10000.0

	fmt.printf("   %-25s | %10.2f bps\n", "Calculated OAS", oas_bps)

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • OAS is the constant spread added to the short rate that makes")
	fmt.println("     the model price equal the market price.")
	fmt.println("   • A positive OAS means the market prices the call risk higher")
	fmt.println("     than our base Hull-White calibration assumes.")
	fmt.println("   • Bisection method guarantees convergence in ~15 iterations.")
	fmt.println("======================================================================\n")
}

puttable_bond_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    PUTTABLE BONDS: HULL-WHITE 1F + LONGSTAFF-SCHWARTZ + OAS")
	fmt.println("======================================================================\n")

	// Define a 10-year puttable bond
	// Lower coupon (2%) and higher initial rate (5%) makes the put option ITM
	bond := fin.PuttableBond {
		face_value       = 100.0,
		coupon_rate      = 0.02, // 2% annual coupon (below market rate)
		coupon_frequency = 2,
		maturity         = 10.0,
		settlement       = 0.0,
		put_schedule     = []fin.PutDate {
			{time = 5.0, price = 100.0}, // Put at year 5 at par
			{time = 6.0, price = 100.0}, // Put at year 6 at par
			{time = 7.0, price = 100.0}, // Put at year 7 at par
			{time = 8.0, price = 100.0}, // Put at year 8 at par
			{time = 9.0, price = 100.0}, // Put at year 9 at par
		},
	}

	a := 0.10
	sigma := 0.02 // 2% volatility for realistic rate movements
	r0 := 0.05 // 5% initial rate (higher than the 2% coupon)

	// ✅ FIXED: Hardcode the rate to avoid Odin closure capture quirks
	P0T_func :: proc(t: f64) -> f64 {
		return math.exp_f64(-0.05 * t)
	}

	fmt.println("Bond Details:")
	fmt.printf("   Face Value:        $%.2f\n", bond.face_value)
	fmt.printf("   Coupon Rate:       %.2f%%\n", bond.coupon_rate * 100.0)
	fmt.printf("   Maturity:          %.1f years\n", bond.maturity)
	fmt.printf("   Put Schedule:      %d dates (at par)\n", len(bond.put_schedule))
	fmt.println("\nMarket Environment:")
	fmt.printf("   Initial Rate (r0): %.2f%%\n", r0 * 100.0)
	fmt.printf("   Volatility (σ):    %.2f%%\n", sigma * 100.0)

	// 1. Base Pricing
	fmt.println("\n1. Base Model Pricing (LSM, 5,000 paths)")
	fmt.println("   ----------------------------------------------------------------------")
	result := fin.puttable_bond_lsm_hw1f(bond, a, sigma, r0, P0T_func, 5000, 100, 3, allocator)

	fmt.printf("   %-25s | $%10.4f\n", "Model Price (Z-spread=0)", result.price)
	fmt.printf("   %-25s | $%10.4f\n", "Straight Bond Price", result.straight_bond_price)
	fmt.printf("   %-25s | $%10.4f\n", "Embedded Put Value", result.embedded_put_value)
	fmt.printf("   %-25s | %10.2f%%\n", "Put Probability", result.put_probability * 100.0)

	// 2. OAS Solver
	// Puttable bonds trade at a PREMIUM to straight bonds (investor has the option)
	// Let's assume market price is $96.00 (above the straight bond price of ~$84)
	market_price := 96.00
	fmt.println("\n2. Option-Adjusted Spread (OAS) Solver")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-25s | $%10.2f\n", "Target Market Price", market_price)

	oas_decimal := fin.puttable_bond_oas(
		bond,
		a,
		sigma,
		r0,
		P0T_func,
		market_price,
		5000,
		100,
		3,
		allocator,
	)
	oas_bps := oas_decimal * 10000.0

	fmt.printf("   %-25s | %10.2f bps\n", "Calculated OAS", oas_bps)

	fmt.println("\n💡 Key Insights:")
	fmt.println(
		"   • Puttable bonds are worth MORE than straight bonds (investor has the option)",
	)
	fmt.println("   • Because the coupon (2%) < market rate (5%), the straight bond trades at a")
	fmt.println("     deep discount (~$84). The put option guarantees the investor can get $100.")
	fmt.println("   • When rates rise further, the investor is highly likely to put the bond")
	fmt.println("     and reinvest at higher rates.")
	fmt.println("   • The embedded put option provides massive downside protection.")
	fmt.println("======================================================================\n")
}
