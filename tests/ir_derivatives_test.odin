package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import rand "core:math/rand"
import "core:mem"

ir_derivatives_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("         INTEREST RATE DERIVATIVES: HULL-WHITE MODEL")
	fmt.println("======================================================================\n")

	// Simulate a 10-year Cap strip (1Y to 10Y)
	n_caplets := 10
	T_start_arr := make([]f64, n_caplets, allocator)
	T_end_arr := make([]f64, n_caplets, allocator)
	F_arr := make([]f64, n_caplets, allocator)
	K_arr := make([]f64, n_caplets, allocator)
	P_arr := make([]f64, n_caplets, allocator)
	delta_arr := make([]f64, n_caplets, allocator)
	market_prices := make([]f64, n_caplets, allocator)

	defer {
		delete(T_start_arr, allocator)
		delete(T_end_arr, allocator)
		delete(F_arr, allocator)
		delete(K_arr, allocator)
		delete(P_arr, allocator)
		delete(delta_arr, allocator)
		delete(market_prices, allocator)
	}

	r0 := 0.04 // 4% flat initial yield curve for demo
	true_a := 0.08
	true_sigma := 0.012

	fmt.println("1. Generating synthetic market caplet strip (1Y - 10Y) with noise...")
	for i in 0 ..< n_caplets {
		T_start_arr[i] = f64(i + 1)
		T_end_arr[i] = f64(i + 2)
		delta_arr[i] = 1.0 // Annual payments

		F_arr[i] = r0
		K_arr[i] = r0 // ATM cap

		P_arr[i] = math.exp_f64(-r0 * T_end_arr[i])

		// Add 2% realistic market noise
		noise := 1.0 + 0.02 * (rand.float64() - 0.5)

		b_delta := (1.0 - math.exp_f64(-true_a * delta_arr[i])) / true_a
		var_factor := (1.0 - math.exp_f64(-2.0 * true_a * T_start_arr[i])) / (2.0 * true_a)
		sigma_P := true_sigma * b_delta * math.sqrt_f64(var_factor)

		d1 := (math.ln(F_arr[i] / K_arr[i]) + 0.5 * sigma_P * sigma_P) / sigma_P
		d2 := d1 - sigma_P
		N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
		N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))

		true_price := P_arr[i] * (F_arr[i] * N_d1 - K_arr[i] * N_d2)
		market_prices[i] = true_price * noise
	}

	fmt.println("2. Calibrating Hull-White to 10Y Cap Strip using Tensor SIMD...")
	hw_res := fin.calibrate_hull_white(
		market_prices,
		T_start_arr,
		T_end_arr,
		F_arr,
		K_arr,
		P_arr,
		n_caplets,
		allocator,
	)

	fmt.printf(
		"   Calibrated HW Params: a = %.4f, sigma = %.4f\n",
		hw_res.params.a,
		hw_res.params.sigma,
	)
	fmt.printf("   True HW Params:       a = %.4f, sigma = %.4f\n", true_a, true_sigma)
	fmt.printf("   Calibration RMSE:     %.6f\n\n", hw_res.rmse)

	fmt.println("3. Pricing a 5Y ATM Cap using Monte Carlo (CRN Greeks)...")
	cap_n := 5
	cap_T_start := make([]f64, cap_n, allocator)
	cap_T_end := make([]f64, cap_n, allocator)
	cap_delta := make([]f64, cap_n, allocator)

	for i in 0 ..< cap_n {
		cap_T_start[i] = f64(i + 1)
		cap_T_end[i] = f64(i + 2)
		cap_delta[i] = 1.0
	}
	defer {
		delete(cap_T_start, allocator)
		delete(cap_T_end, allocator)
		delete(cap_delta, allocator)
	}

	hw_cap_price, hw_cap_delta, hw_cap_vega := fin.hw_mc_cap_option(
		r0,
		r0,
		cap_T_start,
		cap_T_end,
		cap_delta,
		cap_n,
		hw_res.params,
		20000,
		100,
		allocator,
	)

	fmt.println("----------------------------------------------------------------------")
	fmt.printf(" %-35s | $%8.4f\n", "Hull-White MC Cap Price", hw_cap_price)
	fmt.printf(
		" %-35s | Rho (dS): %6.4f | Vega (dVol): %6.4f\n",
		"HW Cap Greeks",
		hw_cap_delta,
		hw_cap_vega,
	)
	fmt.println("----------------------------------------------------------------------")

	fmt.println("\n💡 The IR Insight: Hull-White perfectly captures the mean-reverting")
	fmt.println("   nature of interest rates. Calibrating to a full strip via SIMD tensors")
	fmt.println("   ensures the model respects the entire term structure of volatility,")
	fmt.println(
		"   making it the gold standard for pricing Bermudan swaptions and callable bonds.",
	)
	fmt.println("======================================================================\n")
}
// ========================================================================
// NEW: Hull-White 1F Floor Option Test
// ========================================================================
ir_derivatives_floor_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    INTEREST RATE DERIVATIVES: HULL-WHITE 1F FLOOR OPTION")
	fmt.println("======================================================================\n")

	r0 := 0.04
	K := 0.04 // 4% strike

	// 5-year floor with annual payments
	n_caplets := 5
	T_start := make([]f64, n_caplets, allocator)
	T_end := make([]f64, n_caplets, allocator)
	delta := make([]f64, n_caplets, allocator)
	defer {
		delete(T_start, allocator)
		delete(T_end, allocator)
		delete(delta, allocator)
	}

	for i in 0 ..< n_caplets {
		T_start[i] = f64(i + 1)
		T_end[i] = f64(i + 2)
		delta[i] = 1.0
	}

	params := fin.HW_Params {
		a     = 0.10,
		sigma = 0.012,
	}

	n_paths := 20000
	n_steps := 100

	fmt.printf("   %-30s | %10.4f%%\n", "Initial Short Rate (r0)", r0 * 100.0)
	fmt.printf("   %-30s | %10.4f%%\n", "Floor Strike (K)", K * 100.0)
	fmt.printf("   %-30s | %10d\n", "Number of Floorlets", n_caplets)
	fmt.printf("   %-30s | %10.4f%%\n", "Mean Reversion (a)", params.a * 100.0)
	fmt.printf("   %-30s | %10.4f%%\n", "Volatility (sigma)", params.sigma * 100.0)
	fmt.println()

	floor_price, delta_r, vega := fin.hw_mc_floor_option(
		r0,
		K,
		T_start,
		T_end,
		delta,
		n_caplets,
		params,
		n_paths,
		n_steps,
		allocator,
	)

	fmt.println("   Monte Carlo Pricing Results (1F Floor):")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-30s | $%10.4f\n", "Floor Price", floor_price)
	fmt.printf("   %-30s | %10.4f\n", "Delta (dPrice/dr0)", delta_r)
	fmt.printf("   %-30s | %10.4f\n", "Vega (dPrice/dsigma)", vega)
	fmt.println("======================================================================\n")
}

// ========================================================================
// NEW: Hull-White 2F Swaption Strip Test
// ========================================================================
ir_derivatives_swaption_strip_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    INTEREST RATE DERIVATIVES: HULL-WHITE 2F SWAPTION STRIP")
	fmt.println("======================================================================\n")

	r0 := 0.04
	params := fin.HW2_Params {
		a     = 0.10,
		b     = 0.03,
		sigma = 0.010,
		eta   = 0.008,
		rho   = -0.50,
	}

	n_swaptions := 5
	swaptions := make([]fin.SwaptionSpec, n_swaptions, allocator)
	defer delete(swaptions, allocator)

	fmt.println("   Swaption Strip Setup (ATM Payer Swaptions):")
	fmt.println("   ----------------------------------------------------------------------")
	for i in 0 ..< n_swaptions {
		T_exp := f64(i + 1)
		swap_length := 5 // 5-year underlying swap

		// Simplified annuity calculation for flat curve
		annuity := 0.0
		for k in 1 ..< swap_length + 1 {
			annuity += math.exp_f64(-r0 * (T_exp + f64(k)))
		}

		swaptions[i] = fin.SwaptionSpec {
			T_exp       = T_exp,
			swap_length = swap_length,
			F           = r0, // ATM forward rate
			K           = r0, // ATM strike
			annuity     = annuity,
		}
		fmt.printf(
			"   %-5.1fY x %-2dY ATM Swaption | Annuity: %8.4f\n",
			T_exp,
			swap_length,
			annuity,
		)
	}
	fmt.println()

	prices := fin.hw2f_swaption_price_strip(params, swaptions, n_swaptions, r0, allocator)
	defer delete(prices, allocator)

	fmt.println("   Analytical Pricing Results (Black's Formula with HW2F Vol):")
	fmt.println("   ----------------------------------------------------------------------")
	for i in 0 ..< n_swaptions {
		fmt.printf(
			"   %-5.1fY x %-2dY Swaption Price | $%10.6f\n",
			swaptions[i].T_exp,
			swaptions[i].swap_length,
			prices[i],
		)
	}
	fmt.println("======================================================================\n")
}
