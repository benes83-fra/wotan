package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import rand "core:math/rand"
import "core:mem"

ir_derivatives_2f_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    INTEREST RATE DERIVATIVES: HULL-WHITE 2-FACTOR MODEL")
	fmt.println("======================================================================\n")

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

	r0 := 0.04
	true_a := 0.15
	true_b := 0.02
	true_sigma := 0.008
	true_eta := 0.006
	true_rho := -0.4

	fmt.println("1. Generating synthetic 2-factor caplet strip (1Y - 10Y)...")
	for i in 0 ..< n_caplets {
		T_start_arr[i] = f64(i + 1)
		T_end_arr[i] = f64(i + 2)
		delta_arr[i] = 1.0

		F_arr[i] = r0
		K_arr[i] = r0
		P_arr[i] = math.exp_f64(-r0 * T_end_arr[i])

		noise := 1.0 + 0.02 * (rand.float64() - 0.5)

		// 2-factor variance formula
		B_a := (1.0 - math.exp_f64(-true_a * delta_arr[i])) / true_a
		B_b := (1.0 - math.exp_f64(-true_b * delta_arr[i])) / true_b

		term1 := true_sigma * true_sigma * B_a * B_a * T_start_arr[i]
		term2 :=
			true_eta *
			true_eta *
			B_b *
			B_b *
			(1.0 - math.exp_f64(-2.0 * true_b * T_start_arr[i])) /
			(2.0 * true_b)
		term3 :=
			2.0 *
			true_rho *
			true_sigma *
			true_eta *
			B_a *
			B_b *
			(1.0 - math.exp_f64(-(true_a + true_b) * T_start_arr[i])) /
			(true_a + true_b)

		sigma_P := math.sqrt_f64(term1 + term2 + term3)

		d1 := (math.ln(F_arr[i] / K_arr[i]) + 0.5 * sigma_P * sigma_P) / sigma_P
		d2 := d1 - sigma_P
		N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
		N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))

		true_price := P_arr[i] * (F_arr[i] * N_d1 - K_arr[i] * N_d2)
		market_prices[i] = true_price * noise
	}

	fmt.println("2. Calibrating HW 2-Factor to 10Y Cap Strip (5 parameters, SIMD)...")
	hw2_res := fin.calibrate_hull_white_2f(
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
		"   Calibrated: a=%.4f, b=%.4f, σ=%.4f, η=%.4f, ρ=%.4f\n",
		hw2_res.params.a,
		hw2_res.params.b,
		hw2_res.params.sigma,
		hw2_res.params.eta,
		hw2_res.params.rho,
	)
	fmt.printf(
		"   True:       a=%.4f, b=%.4f, σ=%.4f, η=%.4f, ρ=%.4f\n",
		true_a,
		true_b,
		true_sigma,
		true_eta,
		true_rho,
	)
	fmt.printf("   RMSE:       %.6f\n\n", hw2_res.rmse)

	fmt.println("3. Pricing 5Y ATM Cap using 2-Factor Monte Carlo...")
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

	hw2_cap_price, hw2_cap_delta, hw2_cap_vega := fin.hw2_mc_cap_option(
		r0,
		r0,
		cap_T_start,
		cap_T_end,
		cap_delta,
		cap_n,
		hw2_res.params,
		20000,
		100,
		allocator,
	)

	fmt.println("----------------------------------------------------------------------")
	fmt.printf(" %-35s | $%8.4f\n", "HW 2-Factor MC Cap Price", hw2_cap_price)
	fmt.printf(
		" %-35s | Rho: %6.4f | Vega: %6.4f\n",
		"HW 2F Cap Greeks",
		hw2_cap_delta,
		hw2_cap_vega,
	)
	fmt.println("----------------------------------------------------------------------")

	fmt.println("\n💡 The 2-Factor Insight: Hull-White 2-Factor captures both short-term")
	fmt.println("   rate volatility (σ, fast mean reversion a) and long-term structural")
	fmt.println("   shifts (η, slow mean reversion b). The correlation ρ allows the model")
	fmt.println("   to capture complex term structure dynamics that 1-factor models miss.")
	fmt.println("======================================================================\n")
}


ir_derivatives_2f_swaption_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    HW 2-FACTOR JOINT CALIBRATION: CAPS + SWAPTIONS")
	fmt.println("======================================================================\n")

	r0 := 0.04
	true_a := 0.15
	true_b := 0.02
	true_sigma := 0.008
	true_eta := 0.006
	true_rho := -0.4
	true_params := fin.HW2_Params {
		a     = true_a,
		b     = true_b,
		sigma = true_sigma,
		eta   = true_eta,
		rho   = true_rho,
	}

	// =====================================================================
	// 1. Generate 10 Caplets
	// =====================================================================
	n_caplets := 10
	T_start_arr := make([]f64, n_caplets, allocator)
	T_end_arr := make([]f64, n_caplets, allocator)
	F_arr := make([]f64, n_caplets, allocator)
	K_arr := make([]f64, n_caplets, allocator)
	P_arr := make([]f64, n_caplets, allocator)
	delta_arr := make([]f64, n_caplets, allocator)
	cap_market := make([]f64, n_caplets, allocator)
	defer {
		delete(T_start_arr, allocator); delete(T_end_arr, allocator)
		delete(F_arr, allocator); delete(K_arr, allocator)
		delete(P_arr, allocator); delete(delta_arr, allocator)
		delete(cap_market, allocator)
	}

	for i in 0 ..< n_caplets {
		T_start_arr[i] = f64(i + 1)
		T_end_arr[i] = f64(i + 2)
		delta_arr[i] = 1.0
		F_arr[i] = r0
		K_arr[i] = r0
		P_arr[i] = math.exp_f64(-r0 * T_end_arr[i])

		noise := 1.0 + 0.02 * (rand.float64() - 0.5)
		B_a := (1.0 - math.exp_f64(-true_a * delta_arr[i])) / true_a
		B_b := (1.0 - math.exp_f64(-true_b * delta_arr[i])) / true_b
		term1 := true_sigma * true_sigma * B_a * B_a * T_start_arr[i]
		term2 :=
			true_eta *
			true_eta *
			B_b *
			B_b *
			(1.0 - math.exp_f64(-2.0 * true_b * T_start_arr[i])) /
			(2.0 * true_b)
		term3 :=
			2.0 *
			true_rho *
			true_sigma *
			true_eta *
			B_a *
			B_b *
			(1.0 - math.exp_f64(-(true_a + true_b) * T_start_arr[i])) /
			(true_a + true_b)
		sigma_P := math.sqrt_f64(math.max(term1 + term2 + term3, 1e-12))
		d1 := (math.ln(F_arr[i] / K_arr[i]) + 0.5 * sigma_P * sigma_P) / sigma_P
		d2 := d1 - sigma_P
		N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
		N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
		cap_market[i] = P_arr[i] * (F_arr[i] * N_d1 - K_arr[i] * N_d2) * noise
	}

	// =====================================================================
	// 2. Generate 8 Co-terminal Swaptions (1Yx9Y through 8Yx2Y)
	// =====================================================================
	n_swaptions := 8
	swaption_specs := make([]fin.SwaptionSpec, n_swaptions, allocator)
	swaption_market := make([]f64, n_swaptions, allocator)
	defer {delete(swaption_specs, allocator); delete(swaption_market, allocator)}

	fmt.println("1. Generating market data: 10 Caplets + 8 Swaptions...")
	for i in 0 ..< n_swaptions {
		T_exp := f64(i + 1)
		swap_length := 10 - i // Co-terminal at year 10

		// Compute annuity (PV01)
		annuity := 0.0
		for k in 1 ..< swap_length + 1 {
			T_k := T_exp + f64(k)
			annuity += math.exp_f64(-r0 * T_k)
		}

		swaption_specs[i] = fin.SwaptionSpec {
			T_exp       = T_exp,
			swap_length = swap_length,
			F           = r0,
			K           = r0,
			annuity     = annuity,
		}

		// Generate true price using HW2F
		swap_var := fin.compute_hw2f_swap_variance(swaption_specs[i], true_params, r0)
		swap_vol := math.sqrt_f64(math.max(swap_var, 1e-12))
		ln_F_K := math.ln(r0 / r0) // ATM so this is 0
		d1 := (ln_F_K + 0.5 * swap_var) / swap_vol
		d2 := d1 - swap_vol
		N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
		N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
		true_price := annuity * (r0 * N_d1 - r0 * N_d2)

		noise := 1.0 + 0.02 * (rand.float64() - 0.5)
		swaption_market[i] = true_price * noise
	}

	// =====================================================================
	// 3. Calibrate: Caps Only (baseline)
	// =====================================================================
	fmt.println("2. Calibrating to CAPS ONLY (baseline)...")
	caps_only_res := fin.calibrate_hull_white_2f(
		cap_market,
		T_start_arr,
		T_end_arr,
		F_arr,
		K_arr,
		P_arr,
		n_caplets,
		allocator,
	)
	fmt.printf(
		"   Caps-Only:  a=%.4f, b=%.4f, σ=%.4f, η=%.4f, ρ=%.4f | RMSE=%.6f\n",
		caps_only_res.params.a,
		caps_only_res.params.b,
		caps_only_res.params.sigma,
		caps_only_res.params.eta,
		caps_only_res.params.rho,
		caps_only_res.rmse,
	)

	// =====================================================================
	// 4. Calibrate: Caps + Swaptions (joint)
	// =====================================================================
	fmt.println("3. Calibrating to CAPS + SWAPTIONS (joint)...")
	joint_res := fin.calibrate_hull_white_2f_joint(
		cap_market,
		T_start_arr,
		T_end_arr,
		F_arr,
		K_arr,
		P_arr,
		n_caplets,
		swaption_specs,
		swaption_market,
		n_swaptions,
		r0,
		allocator, // <-- pass r0
	)
	fmt.printf(
		"   Joint:      a=%.4f, b=%.4f, σ=%.4f, η=%.4f, ρ=%.4f | RMSE=%.6f\n",
		joint_res.params.a,
		joint_res.params.b,
		joint_res.params.sigma,
		joint_res.params.eta,
		joint_res.params.rho,
		joint_res.rmse,
	)

	// =====================================================================
	// 5. Comparison
	// =====================================================================
	fmt.println("\n4. Parameter Recovery Comparison:")
	fmt.println("----------------------------------------------------------------------")
	fmt.printf(" %-12s | %8s | %8s | %8s\n", "Parameter", "True", "Caps Only", "Joint")
	fmt.println("----------------------------------------------------------------------")
	fmt.printf(
		" %-12s | %8.4f | %8.4f | %8.4f\n",
		"a (fast MR)",
		true_a,
		caps_only_res.params.a,
		joint_res.params.a,
	)
	fmt.printf(
		" %-12s | %8.4f | %8.4f | %8.4f\n",
		"b (slow MR)",
		true_b,
		caps_only_res.params.b,
		joint_res.params.b,
	)
	fmt.printf(
		" %-12s | %8.4f | %8.4f | %8.4f\n",
		"σ (fast vol)",
		true_sigma,
		caps_only_res.params.sigma,
		joint_res.params.sigma,
	)
	fmt.printf(
		" %-12s | %8.4f | %8.4f | %8.4f\n",
		"η (slow vol)",
		true_eta,
		caps_only_res.params.eta,
		joint_res.params.eta,
	)
	fmt.printf(
		" %-12s | %8.4f | %8.4f | %8.4f\n",
		"ρ (corr)",
		true_rho,
		caps_only_res.params.rho,
		joint_res.params.rho,
	)
	fmt.println("----------------------------------------------------------------------")

	// Compute parameter error
	caps_err :=
		math.abs(caps_only_res.params.a - true_a) +
		math.abs(caps_only_res.params.b - true_b) +
		math.abs(caps_only_res.params.sigma - true_sigma) +
		math.abs(caps_only_res.params.eta - true_eta) +
		math.abs(caps_only_res.params.rho - true_rho)
	joint_err :=
		math.abs(joint_res.params.a - true_a) +
		math.abs(joint_res.params.b - true_b) +
		math.abs(joint_res.params.sigma - true_sigma) +
		math.abs(joint_res.params.eta - true_eta) +
		math.abs(joint_res.params.rho - true_rho)

	fmt.printf("\n Total Parameter Error (L1):\n")
	fmt.printf("   Caps Only: %.4f\n", caps_err)
	fmt.printf("   Joint:     %.4f\n", joint_err)

	improvement := (1.0 - joint_err / caps_err) * 100.0
	if improvement > 0 {
		fmt.printf("\n✅ Joint calibration reduced parameter error by %.1f%%\n", improvement)
	} else {
		fmt.printf("\n⚠️ Joint calibration error change: %.1f%%\n", improvement)
	}

	fmt.println("\n💡 Adding swaptions breaks the parameter non-identifiability by providing")
	fmt.println("   cross-tenor correlation information that caplets alone cannot capture.")
	fmt.println("======================================================================\n")
}


bermudan_swaption_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("         BERMUDAN SWAPTION: 1-FACTOR vs 2-FACTOR")
	fmt.println("======================================================================\n")

	r0 := 0.04
	K := 0.04 // ATM

	// 5Y Bermudan into a 10Y swap (exercise dates: 1Y, 2Y, 3Y, 4Y, 5Y)
	exercise_dates := make([]f64, 5, allocator)
	for i in 0 ..< 5 {
		exercise_dates[i] = f64(i + 1)
	}
	maturity := 10.0
	delta := 1.0

	defer delete(exercise_dates, allocator)

	// Calibrated parameters (from your previous joint calibration)
	params_1f := fin.HW_Params {
		a     = 0.0514,
		sigma = 0.0096,
	}

	params_2f := fin.HW2_Params {
		a     = 0.0514, // Fast mean reversion
		b     = 0.0201, // Slow mean reversion
		sigma = 0.0096,
		eta   = 0.0050,
		rho   = -0.3023,
	}

	fmt.println("1. Pricing 5Y Bermudan Payer Swaption into 10Y Swap...")
	fmt.println("   (Using calibrated parameters from joint Caps+Swaptions fit)\n")

	// Price with 1-Factor (using a simplified MC for comparison)
	// Note: For a true 1F Bermudan, you'd use a similar MC with 1 factor.
	// Here we approximate the 1F Bermudan value by noting it lacks the slow factor's flexibility.
	// We will simulate it using the 2F engine but with eta=0 to show the pure 1F baseline.
	params_1f_mc := fin.HW2_Params {
		a     = params_1f.a,
		b     = 0.01, // Dummy slow factor
		sigma = params_1f.sigma,
		eta   = 0.0001, // Effectively zero volatility for the second factor
		rho   = 0.0,
	}

	price_1f := fin.hw2f_mc_bermudan_option(
		r0,
		K,
		exercise_dates,
		maturity,
		delta,
		params_1f_mc,
		20000,
		100,
		allocator,
	)

	price_2f := fin.hw2f_mc_bermudan_option(
		r0,
		K,
		exercise_dates,
		maturity,
		delta,
		params_2f,
		20000,
		100,
		allocator,
	)

	fmt.println("----------------------------------------------------------------------")
	fmt.printf(" %-35s | $%8.4f\n", "1-Factor HW Bermudan Price", price_1f)
	fmt.printf(" %-35s | $%8.4f\n", "2-Factor HW Bermudan Price", price_2f)
	fmt.println("----------------------------------------------------------------------")

	diff := price_2f - price_1f
	pct_diff := (diff / price_1f) * 100.0

	fmt.printf("\n💡 The 2-Factor Premium: +$%.4f (%.2f%%)\n", diff, pct_diff)
	fmt.println(
		"   Reason: The 2-factor model captures long-term structural shifts (via 'b' and 'eta')",
	)
	fmt.println("   that make the swap rate more volatile over long horizons. This increases the")
	fmt.println(
		"   probability of the swaption finishing deep in-the-money at later exercise dates,",
	)
	fmt.println("   which the 1-factor model systematically underestimates.")
	fmt.println("======================================================================\n")
}
