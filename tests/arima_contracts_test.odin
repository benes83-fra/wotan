package tests

import ts "../wotan/analytics"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ARIMA / Nelder-Mead / SARIMA fitting contracts:
// - arima_fit is scale-free: the same series at variance 1 and 1e-4 gives the
//   same (phi, theta) and sigma2 scaled by 1e-4
// - the fit is the MLE (a long Nelder-Mead run from the truth finds nothing better)
// - nothing but the returned phi/theta stays allocated, and `converged` is set
// - nelder_mead returns the best vertex it holds, even when max_iter runs out
// - sarima_fit does not depend on the global RNG state

arima_contracts_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ARIMA Fit Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// ARMA(1,1), phi = 0.6, theta = 0.3, at unit and return scale (same shocks)
	n := 500
	y := ts.simulate_arima_pdq([]f64{0.6}, 0, []f64{0.3}, 1.0, n, allocator)
	y_small := make([]f64, n)
	defer delete(y_small)
	for v, i in y {y_small[i] = 0.01 * v}

	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, allocator)
	defer mem.tracking_allocator_destroy(&track)
	fit := ts.arima_fit(y, 1, 0, 1, mem.tracking_allocator(&track))
	live := len(track.allocation_map)
	fit_small := ts.arima_fit(y_small, 1, 0, 1, allocator)

	// 1. scale invariance
	expect(&passed, &total,
		abs(fit.phi[0] - fit_small.phi[0]) < 1e-3 && abs(fit.theta[0] - fit_small.theta[0]) < 1e-3 &&
		abs(fit_small.sigma2 / (1e-4 * fit.sigma2) - 1.0) < 1e-3,
		fmt.tprintf("scale-free: phi %.4f / %.4f, theta %.4f / %.4f, sigma2 ratio %.6f",
			fit.phi[0], fit_small.phi[0], fit.theta[0], fit_small.theta[0], fit_small.sigma2 / fit.sigma2))

	// 2. MLE: a long run started at the truth finds no better likelihood
	ctx := ts.ArimaObjectiveCtx{y = y_small, p = 1, q = 1, d = 0, allocator = allocator}
	x_true := []f64{math.atanh(0.6 / 0.95), math.atanh(0.3 / 0.95), math.ln_f64(1e-4)}
	x_ref, f_ref, _ := ts.nelder_mead(ts.arima_neg_loglik_obj, &ctx, x_true, 5000, 1e-12, allocator)
	defer delete(x_ref)
	gap := -f_ref - fit_small.loglik
	expect(&passed, &total, gap < 1e-4, fmt.tprintf("return-scale fit is the MLE (loglik %.4f, reference gap %.1e)", fit_small.loglik, gap))

	// 3. allocations and convergence flag
	expect(&passed, &total, live == 2 && fit.converged && fit_small.converged,
		fmt.tprintf("only phi/theta stay allocated (%d live), converged = %v", live, fit.converged))

	// 4. nelder_mead hands back its best vertex when the budget runs out
	bowl :: proc(x: []f64, _: rawptr) -> f64 {return (x[0] - 3) * (x[0] - 3) + 10 * (x[1] + 1) * (x[1] + 1)}
	x_nm, f_nm, conv_nm := ts.nelder_mead(bowl, nil, []f64{0, 0}, 7, 1e-12, allocator)
	defer delete(x_nm)
	expect(&passed, &total, f_nm == bowl(x_nm, nil) && f_nm < bowl([]f64{0, 0}, nil) && !conv_nm,
		fmt.tprintf("Nelder-Mead budget stop returns a consistent best point (f = %.4f)", f_nm))

	// 5. SARIMA(1,0,0)(1,0,0)_4 fit is independent of the global RNG state
	ys := ts.simulate_sarima_pdqPDQ([]f64{0.5}, 0, nil, []f64{0.4}, 0, nil, 4, 1.0, 240, allocator)
	fit_a := ts.sarima_fit(ys, 1, 0, 0, 1, 0, 0, 4, allocator)
	for _ in 0 ..< 1000 {_ = rand.float64()}
	fit_b := ts.sarima_fit(ys, 1, 0, 0, 1, 0, 0, 4, allocator)
	expect(&passed, &total, fit_a.phi[0] == fit_b.phi[0] && fit_a.Phi[0] == fit_b.Phi[0] && fit_a.loglik == fit_b.loglik,
		fmt.tprintf("SARIMA fit is deterministic (phi %.4f, Phi %.4f)", fit_a.phi[0], fit_a.Phi[0]))

	fmt.printf("ARIMA contracts summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
