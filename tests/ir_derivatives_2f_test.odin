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
