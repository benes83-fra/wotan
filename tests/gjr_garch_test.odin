package tests

import ts "../wotan/analytics"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// GJR-GARCH contracts:
// - fitting a simulated GJR(1,1) process recovers the leverage term γ
// - the GJR fit beats a symmetric GARCH fit by a likelihood-ratio margin
// - persistence is α + β + γ/2 and stays below 1
// - the 1-step forecast matches the GJR recursion, long horizons revert to
//   the unconditional variance ω / (1 - persistence)

gjr_garch_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== GJR-GARCH Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	true_omega, true_alpha, true_gamma, true_beta := 1e-5, 0.03, 0.12, 0.88
	n := 3000
	eps := make([]f64, n)
	defer delete(eps)
	h := true_omega / (1.0 - true_alpha - true_beta - 0.5 * true_gamma)
	prev := 0.0
	for i in 0 ..< n {
		if i > 0 {
			h = true_omega + true_alpha * prev * prev + true_beta * h
			if prev < 0.0 {h += true_gamma * prev * prev}
		}
		prev = math.sqrt_f64(h) * rand.norm_float64()
		eps[i] = prev
	}

	free_result :: proc(r: ^ts.GARCH_Result) {
		delete(r.params.alpha)
		delete(r.params.beta)
		delete(r.conditional_var)
		delete(r.standardized_resid)
	}
	gjr := ts.garch_fit(eps, .GJR_GARCH, 1, 1, 2000, 1e-7, allocator)
	defer free_result(&gjr)
	sym := ts.garch_fit(eps, .GARCH, 1, 1, 2000, 1e-7, allocator)
	defer free_result(&sym)

	g := gjr.params
	fmt.printf("GJR fit: ω=%.2e α=%.4f γ=%.4f β=%.4f (truth 1.0e-05 / 0.03 / 0.12 / 0.88)\n",
		g.omega, g.alpha[0], g.gamma, g.beta[0])

	// 1. leverage term recovered
	expect(&passed, &total, abs(g.gamma - true_gamma) < 0.05 && abs(g.alpha[0] - true_alpha) < 0.04 && abs(g.beta[0] - true_beta) < 0.04,
		fmt.tprintf("recovers γ=%.4f α=%.4f β=%.4f", g.gamma, g.alpha[0], g.beta[0]))

	// 2. asymmetry is detected: LR = 2 (ll_gjr - ll_garch) far above χ²(1) 5% = 3.84
	lr := 2.0 * (gjr.log_likelihood - sym.log_likelihood)
	expect(&passed, &total, lr > 10.0, fmt.tprintf("GJR beats symmetric GARCH (LR = %.1f)", lr))

	// 3. persistence includes γ/2 and is stationary
	pers := g.alpha[0] + g.beta[0] + 0.5 * g.gamma
	expect(&passed, &total, abs(gjr.persistence - pers) < 1e-15 && pers < 1.0,
		fmt.tprintf("persistence = α + β + γ/2 = %.4f < 1", gjr.persistence))

	// 4. forecasts: 1-step follows the recursion, long horizon reverts to ω / (1 - persistence)
	last := eps[n - 1]
	manual := g.omega + g.alpha[0] * last * last + g.beta[0] * gjr.conditional_var[n - 1]
	if last < 0.0 {manual += g.gamma * last * last}
	f1 := ts.garch_forecast(&gjr, eps, 1, allocator)
	expect(&passed, &total, abs(f1.variance_forecast / manual - 1.0) < 1e-12,
		fmt.tprintf("1-step forecast matches GJR recursion (%.3e)", f1.variance_forecast))

	uncond := g.omega / (1.0 - gjr.persistence)
	f5 := ts.garch_forecast(&gjr, eps, 5, allocator)
	f_long := ts.garch_forecast(&gjr, eps, 5000, allocator)
	expect(&passed, &total, abs(f_long.variance_forecast / uncond - 1.0) < 1e-6 && f5.variance_forecast != f1.variance_forecast,
		fmt.tprintf("forecast reverts to ω/(1-persistence) = %.3e (h=5000: %.3e)", uncond, f_long.variance_forecast))

	fmt.printf("GJR-GARCH summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
