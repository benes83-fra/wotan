package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

fx_derivatives_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    FX DERIVATIVES: GARMAN-KOHLHAGEN + VANNA-VOLGA")
	fmt.println("======================================================================\n")

	// EUR/USD: Spot = 1.10, meaning 1 EUR = 1.10 USD
	// Domestic = USD, Foreign = EUR
	S := 1.10
	r_d := 0.05 // USD rate 5%
	r_f := 0.02 // EUR rate 2%
	T := 1.0 // 1 year expiry

	// Realistic FX volatility smile (typical EUR/USD 1Y)
	smile := fin.FXSmile {
		atm_vol           = 0.085, // 8.5% ATM vol
		risk_reversal_25d = -0.015, // -1.5% (puts more expensive than calls, typical for EUR/USD)
		butterfly_25d     = 0.004, // +0.4% (smile convexity)
	}

	vols := fin.fx_smile_to_vols(smile)

	fmt.println("Market Setup:")
	fmt.printf("   Spot (EUR/USD):    %.4f\n", S)
	fmt.printf("   Domestic (USD):    %.2f%%\n", r_d * 100.0)
	fmt.printf("   Foreign (EUR):     %.2f%%\n", r_f * 100.0)
	fmt.printf("   Time to Expiry:    %.1f year\n", T)
	fmt.println("\nVolatility Smile (1Y EUR/USD):")
	fmt.printf("   ATM Vol:           %.2f%%\n", smile.atm_vol * 100.0)
	fmt.printf("   25d Risk Reversal: %.2f%%\n", smile.risk_reversal_25d * 100.0)
	fmt.printf("   25d Butterfly:     %.2f%%\n", smile.butterfly_25d * 100.0)
	fmt.println("\nDerived Volatilities:")
	fmt.printf("   25d Put Vol:       %.2f%%\n", vols.vol_25d_put * 100.0)
	fmt.printf("   ATM Vol:           %.2f%%\n", vols.vol_atm * 100.0)
	fmt.printf("   25d Call Vol:      %.2f%%\n", vols.vol_25d_call * 100.0)

	// ========================================================================
	// 1. VANILLA FX OPTIONS (Garman-Kohlhagen)
	// ========================================================================
	fmt.println("\n1. Vanilla FX Options (Garman-Kohlhagen)")
	fmt.println("   ----------------------------------------------------------------------")

	K_atm_strike := S * math.exp_f64((r_d - r_f) * T) // ATM delta-neutral strike
	call_res := fin.fx_gk_call(S, K_atm_strike, T, r_d, r_f, vols.vol_atm)
	put_res := fin.fx_gk_put(S, K_atm_strike, T, r_d, r_f, vols.vol_atm)

	fmt.printf(
		"   %-25s | Call %10.4f | Put %10.4f\n",
		"Price (pips)",
		call_res.price * 10000.0,
		put_res.price * 10000.0,
	)
	fmt.printf("   %-25s | %10.4f    | %10.4f\n", "Spot Delta", call_res.delta, put_res.delta)
	fmt.printf(
		"   %-25s | %10.4f    | %10.4f\n",
		"Forward Delta",
		call_res.delta_f,
		put_res.delta_f,
	)
	fmt.printf(
		"   %-25s | %10.4f    | %10.4f\n",
		"Vega (per 1%)",
		call_res.vega / 100.0,
		put_res.vega / 100.0,
	)

	// ========================================================================
	// 2. EUROPEAN DIGITAL OPTION (with Vanna-Volga)
	// ========================================================================
	fmt.println("\n2. European Digital Call (Cash-or-Nothing, pays $1 if S_T > K)")
	fmt.println("   ----------------------------------------------------------------------")

	K_digital := 1.12 // 200 pips OTM

	// BS price with ATM vol
	bs_price := fin.fx_digital_call_price(S, K_digital, T, r_d, r_f, vols.vol_atm)
	vanna_dig, volga_dig := fin.fx_digital_call_sensitivities(
		S,
		K_digital,
		T,
		r_d,
		r_f,
		vols.vol_atm,
	)

	// Market prices of the 3 hedging vanillas (using smile vols)
	K_25d_put := fin.fx_strike_from_delta(S, -0.25, T, r_d, r_f, vols.vol_25d_put, .Put)
	K_25d_call := fin.fx_strike_from_delta(S, 0.25, T, r_d, r_f, vols.vol_25d_call, .Call)

	mkt_25d_put := fin.fx_gk_price(S, K_25d_put, T, r_d, r_f, vols.vol_25d_put, .Put)
	mkt_atm := fin.fx_gk_price(S, K_atm_strike, T, r_d, r_f, vols.vol_atm, .Call)
	mkt_25d_call := fin.fx_gk_price(S, K_25d_call, T, r_d, r_f, vols.vol_25d_call, .Call)

	vv_result := fin.fx_vanna_volga_adjust(
		S,
		T,
		r_d,
		r_f,
		smile,
		bs_price,
		vanna_dig,
		volga_dig,
		mkt_25d_put,
		mkt_atm,
		mkt_25d_call,
		allocator,
	)

	fmt.printf("   %-25s | %10.6f\n", "BS Price (ATM vol)", bs_price)
	fmt.printf("   %-25s | %10.6f\n", "Vanna-Volga Price", vv_result.price_vv)
	fmt.printf("   %-25s | %10.6f\n", "Smile Adjustment", vv_result.adjustment)
	fmt.printf(
		"   %-25s | %10.2f%%\n",
		"Adjustment as % of BS",
		vv_result.adjustment / bs_price * 100.0,
	)

	// ========================================================================
	// 3. DOWN-AND-OUT DIGITAL CALL (with Vanna-Volga)
	// ========================================================================
	fmt.println("\n3. Down-and-Out Digital Call (pays $1 if S_T > K and S_t > L for all t)")
	fmt.println("   ----------------------------------------------------------------------")

	K_do_digital := 1.12 // Strike
	L_barrier := 1.05 // Lower barrier

	// Analytical price with ATM vol
	bs_do_digital := fin.fx_down_and_out_digital_call_price(
		S,
		K_do_digital,
		L_barrier,
		T,
		r_d,
		r_f,
		vols.vol_atm,
	)
	vanna_do, volga_do := fin.fx_down_and_out_digital_call_sensitivities(
		S,
		K_do_digital,
		L_barrier,
		T,
		r_d,
		r_f,
		vols.vol_atm,
	)

	vv_do := fin.fx_vanna_volga_adjust(
		S,
		T,
		r_d,
		r_f,
		smile,
		bs_do_digital,
		vanna_do,
		volga_do,
		mkt_25d_put,
		mkt_atm,
		mkt_25d_call,
		allocator,
	)

	fmt.printf("   Barriers/Strike:     L = %.4f, K = %.4f\n", L_barrier, K_do_digital)
	fmt.printf("   %-25s | %10.6f\n", "BS Price (ATM vol)", bs_do_digital)
	fmt.printf("   %-25s | %10.6f\n", "Vanna-Volga Price", vv_do.price_vv)
	fmt.printf("   %-25s | %10.6f\n", "Smile Adjustment", vv_do.adjustment)
	fmt.printf(
		"   %-25s | %10.2f%%\n",
		"Adjustment as % of BS",
		vv_do.adjustment / bs_do_digital * 100.0,
	)

	fmt.println("\n💡 Key Insights:")
	fmt.println("   • Garman-Kohlhagen is the FX Black-Scholes (r_f acts as dividend yield)")
	fmt.println("   • The FX volatility smile is quoted via ATM, Risk Reversal, Butterfly")
	fmt.println("   • Vanna-Volga adjusts exotic prices to match the market smile")
	fmt.println("   • For the Digital, the smile adjustment is ~5-15% of the BS price")
	fmt.println("   • For Barriers, the smile adjustment is typically 20-40% due to high Volga")
	fmt.println("   • Analytical formulas are used for barrier Greeks to avoid MC noise")
	fmt.println("======================================================================\n")
}
