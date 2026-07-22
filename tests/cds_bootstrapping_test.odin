package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:mem"

cds_bootstrapping_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("         CDS CURVE BOOTSTRAPPING & UPFRONT PRICING")
	fmt.println("======================================================================\n")

	// 1. Market Data: Typical BBB-rated corporate CDS spreads
	tenors := []f64{1.0, 3.0, 5.0, 7.0, 10.0}
	market_spreads := []f64{0.0040, 0.0060, 0.0080, 0.0095, 0.0110}

	recovery := 0.40
	r := 0.04
	payment_freq := 0.25

	fmt.println("1. Market Input Data:")
	fmt.println("   Tenor  | Market Spread (bps)")
	fmt.println("   -------|--------------------")
	for i in 0 ..< len(tenors) {
		fmt.printf("   %5.1f  | %18.1f\n", tenors[i], market_spreads[i] * 10000.0)
	}

	// 2. Bootstrap the Hazard Rate Curve
	fmt.println("\n2. Bootstrapping Piecewise Hazard Rate Curve...")
	curve := fin.bootstrap_cds_curve(tenors, market_spreads, recovery, r, payment_freq, allocator)
	defer {
		delete(curve.tenors, allocator)
		delete(curve.hazards, allocator)
	}

	fmt.println("   Tenor  | Hazard Rate (λ) | Cumulative Default Prob")
	fmt.println("   -------|-----------------|------------------------")
	for i in 0 ..< len(curve.tenors) {
		cum_def := 1.0 - fin.get_survival_prob(curve.tenors[i], curve)
		fmt.printf(
			"   %5.1f  | %15.4f | %21.2f%%\n",
			curve.tenors[i],
			curve.hazards[i],
			cum_def * 100.0,
		)
	}

	// 3. Price a 5Y CDS with a Standardized Coupon
	notional := 10_000_000.0
	fixed_coupon := 0.0100 // 100 bps standardized coupon
	maturity := 5.0

	fmt.printf(
		"\n3. Pricing 5Y CDS (Notional: $%.0f, Standard Coupon: %.0f bps)...\n",
		notional,
		fixed_coupon * 10000.0,
	)

	result := fin.price_cds_full(notional, fixed_coupon, maturity, curve, payment_freq)

	fmt.println("   ----------------------------------------------------------------------")
	fmt.println("   Metric                         |        Value")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-30s | %11.2f bps\n", "Market 5Y Spread", market_spreads[2] * 10000.0)
	fmt.printf("   %-30s | %11.2f bps\n", "Standard Fixed Coupon", fixed_coupon * 10000.0)
	fmt.printf("   %-30s | $%10.2f\n", "Upfront Payment", result.upfront_pct * notional)
	fmt.printf("   %-30s | %11.4f%%\n", "Upfront (as % of Notional)", result.upfront_pct * 100.0)
	fmt.printf("   %-30s | $%10.2f\n", "Risky PV01 (per $1M notional)", result.rpv01)
	fmt.println("   ----------------------------------------------------------------------")

	if result.upfront_pct < 0.0 {
		fmt.println("\n💡 Interpretation: The market spread (80 bps) is LOWER than the")
		fmt.println("   standard coupon (100 bps). The protection buyer is overpaying")
		fmt.println("   in running coupons, so the protection seller must rebate the")
		fmt.println("   difference as an UPFRONT payment to the buyer at inception.")
	} else {
		fmt.println("\n💡 Interpretation: The market spread is HIGHER than the standard")
		fmt.println("   coupon. The protection buyer must pay an upfront fee to compensate")
		fmt.println("   the seller for the higher-than-coupon credit risk.")
	}
	fmt.println("======================================================================\n")
}
