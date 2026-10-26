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
// - likelihood, MLE, AIC/BIC, forecasts and intervals match statsmodels

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
	x_true, x_ok := ts.arima_unconstrain_params([]f64{0.6}, []f64{0.3}, 1e-4, allocator)
	defer delete(x_true)
	x_ref, f_ref, _ := ts.nelder_mead(ts.arima_neg_loglik_obj, &ctx, x_true, 5000, 1e-12, allocator)
	defer delete(x_ref)
	gap := -f_ref - fit_small.loglik
	expect(&passed, &total, x_ok && gap < 1e-4, fmt.tprintf("return-scale fit is the MLE (loglik %.4f, reference gap %.1e)", fit_small.loglik, gap))

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

	// 6. statsmodels 0.13.5 reference values (ARIMA / SARIMAX, trend = 'n')
	arima_statsmodels_checks(&passed, &total, allocator)

	fmt.printf("ARIMA contracts summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// y_t = Σ phi_i y_{t-i} + e_t + Σ theta_j e_{t-j} from zero pre-sample values
_sm_ref_arma :: proc(phi, theta, e: []f64, allocator: mem.Allocator) -> []f64 {
	y := make([]f64, len(e), allocator)
	for t in 0 ..< len(e) {
		v := e[t]
		for p, i in phi {if t - 1 - i >= 0 {v += p * y[t - 1 - i]}}
		for q, j in theta {if t - 1 - j >= 0 {v += q * e[t - 1 - j]}}
		y[t] = v
	}
	return y
}

// Checks against statsmodels 0.13.5 (numpy 1.23) on series built from gcl_fill shocks
// (uniform * sqrt(3)); the generating Python is the same recursion. Conventions:
// - loglik is the exact Gaussian likelihood of the differenced series, sigma2 not
//   concentrated; statsmodels' ARIMA(d > 0) llf uses an approximate-diffuse prior for the
//   integrated states and is ~1e-5 lower, so d > 0 references are ARIMA(diff(y), d = 0).
// - AIC/BIC count sigma2; n = len(y) - d - D*s.
// - forecasts: exactly h values on the original scale, get_forecast(h).conf_int(alpha).
arima_statsmodels_checks :: proc(passed, total: ^int, allocator: mem.Allocator) {
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	max_err :: proc(got, want: []f64) -> f64 {
		if len(got) != len(want) {return math.INF_F64}
		e := 0.0
		for v, i in got {e = max(e, abs(v - want[i]))}
		return e
	}
	fc_err :: proc(mean, lower, upper: []f64, want_mean, want_lo, want_hi: []f64) -> f64 {
		return max(max_err(mean, want_mean), max_err(lower, want_lo), max_err(upper, want_hi))
	}
	// everything below is scratch in `allocator` (the callers pass an arena / temp allocator)
	context.allocator = allocator

	e := make([]f64, 200)
	gcl_fill(e, 4242)
	for &v in e {v *= math.sqrt_f64(3.0)}
	A := _sm_ref_arma({0.6}, {0.3}, e, allocator)
	B := make([]f64, 200)
	acc := 5.0
	for v, i in A {acc += v; B[i] = acc}

	eC := make([]f64, 150)
	gcl_fill(eC, 4343)
	for &v in eC {v *= math.sqrt_f64(3.0)}
	C := _sm_ref_arma({}, {0.5}, eC, allocator)
	for _ in 0 ..< 2 {
		acc = 0.0
		for &v in C {acc += v; v = acc}
	}

	// airline-type (0,1,1)(0,1,1)_4: z = (1 - 0.4L)(1 - 0.5L^4) e, (1-L)(1-L^4) y = z
	eD := make([]f64, 120)
	gcl_fill(eD, 4444)
	for &v in eD {v *= 0.5 * math.sqrt_f64(3.0)}
	zD := _sm_ref_arma({}, {-0.4, 0, 0, -0.5, 0.2}, eD, allocator)
	yD := make([]f64, 125)
	for t in 0 ..< 120 {n := t + 5; yD[n] = zD[t] + yD[n - 1] + yD[n - 4] - yD[n - 5]}
	D := yD[5:]
	for &v in D {v += 50.0}

	// --- ARMA(1,1): exact loglik at fixed params, MLE, AIC/BIC, 90% forecast ---
	ll := ts.arima_loglik(A, {0.6}, {0.3}, 0, 1.0)
	expect(passed, total, abs(ll - -285.02074235129413) < 1e-7, fmt.tprintf("ARMA(1,1) exact loglik vs statsmodels (err %.1e)", ll - -285.02074235129413))
	fit := ts.arima_fit(A, 1, 0, 1)
	sm_fit := []f64{0.61667688633452966, 0.30646535366913735, 1.0070149717878563}
	got_fit := []f64{fit.phi[0], fit.theta[0], fit.sigma2}
	expect(passed, total, max_err(got_fit, sm_fit) < 1e-3 && fit.loglik >= -284.94915615901402 - 1e-6 &&
		abs(fit.aic - (-2 * fit.loglik + 6)) < 1e-9 && abs(fit.aic - 575.89831231802805) < 1e-4 &&
		abs(fit.bic - 585.79326441767216) < 1e-4,
		fmt.tprintf("ARMA(1,1) MLE vs statsmodels (params err %.1e, loglik %.6f, aic %.4f, bic %.4f)",
			max_err(got_fit, sm_fit), fit.loglik, fit.aic, fit.bic))
	fixed := ts.ArimaFitResult{phi = {0.6}, theta = {0.3}, sigma2 = 1.0}
	fcA := ts.arima_forecast(A, fixed, 1, 0, 1, 4, 0.1)
	errA := fc_err(fcA.mean, fcA.lower, fcA.upper,
		{0.84574181620274058, 0.50744508972164437, 0.30446705383298661, 0.18268023229979197},
		{-0.79911181091740335, -1.7054791412705632, -2.0800601788063959, -2.2606755037300501},
		{2.4905954433228845, 2.7203693207138522, 2.6889942864723695, 2.6260359683296337})
	expect(passed, total, errA < 1e-8, fmt.tprintf("ARMA(1,1) forecast + 90%% interval vs statsmodels (err %.1e)", errA))

	// --- ARIMA(1,1,1): h values on the level scale, integrated-variance 80% interval ---
	fcB := ts.arima_forecast(B, fixed, 1, 1, 1, 4, 0.2)
	errB := fc_err(fcB.mean, fcB.lower, fcB.upper,
		{27.672123831913346, 28.179568921634992, 28.484035975467979, 28.66671620776777},
		{26.390572266237328, 25.42796101811695, 24.318776514981685, 23.198935339500107},
		{28.953675397589365, 30.931176825153035, 32.649295435954272, 34.13449707603543})
	expect(passed, total, len(fcB.mean) == 4 && errB < 1e-7, fmt.tprintf("ARIMA(1,1,1) forecast (len %d) + 80%% interval vs statsmodels (err %.1e)", len(fcB.mean), errB))
	fc0 := ts.arima_forecast(B, fixed, 1, 1, 1, 0, 0.05)
	expect(passed, total, len(fc0.mean) == 0, "arima_forecast with h = 0 returns no values")

	// --- ARIMA(0,2,1): MA(1) likelihood (the state carries theta) and d = 2 forecast ---
	Cd := ts.difference(C, 2)
	llC := ts.arima_loglik(Cd, {}, {0.5}, 2, 1.0)
	expect(passed, total, abs(llC - -202.59772728457321) < 1e-7, fmt.tprintf("MA(1) exact loglik vs statsmodels (err %.1e)", llC - -202.59772728457321))
	fcC := ts.arima_forecast(C, ts.ArimaFitResult{theta = {0.5}, sigma2 = 1.0}, 0, 2, 1, 4, 0.05)
	errC := fc_err(fcC.mean, fcC.lower, fcC.upper,
		{-104.08837266555487, -116.08439557410081, -128.08041848264676, -140.0764413911927},
		{-106.04833665026605, -121.36176011075314, -137.53102256848041, -154.41234202274944},
		{-102.12840868084369, -110.80703103744848, -118.62981439681312, -125.74054075963598})
	expect(passed, total, errC < 1e-7, fmt.tprintf("ARIMA(0,2,1) forecast + 95%% interval vs statsmodels (err %.1e)", errC))

	// --- SARIMA(0,1,1)(0,1,1)_4: forecast, exact loglik, innovations, MLE ---
	sfc := ts.sarima_forecast(D, {}, {-0.4}, {}, {-0.5}, 1, 1, 4, 6, 0.25, 0.05)
	errD := fc_err(sfc.mean, sfc.lower, sfc.upper,
		{31.482902779328338, 31.082774341441144, 30.107369211884084, 31.349759205404098, 29.837760350561137, 29.437631912673936},
		{30.502920786562765, 29.939928769962325, 28.822134877471598, 29.936409075716188, 28.060234011952016, 27.453319614580671},
		{32.462884772093915, 32.22561991291996, 31.39260354629657, 32.763109335092011, 31.615286689170258, 31.421944210767201})
	expect(passed, total, len(sfc.mean) == 6 && errD < 1e-7, fmt.tprintf("SARIMA(0,1,1)(0,1,1)_4 forecast + 95%% interval vs statsmodels (err %.1e)", errD))
	z := ts.sarima_difference(D, 1, 1, 4)
	ar, ma := ts.sarima_expand_to_arma({}, {}, {-0.4}, {-0.5}, 4)
	llD := ts.arima_loglik(z, ar, ma, 0, 0.25)
	innov := ts.sarima_innovations(D, {}, {-0.4}, {}, {-0.5}, 1, 1, 4, 0.25)
	errI := max_err(innov[:3], {-0.57963602661986613, -0.84565056861475207, 0.37292571175730382})
	expect(passed, total, abs(llD - -78.51445819739115) < 1e-7 && len(innov) == len(z) && errI < 1e-10,
		fmt.tprintf("SARIMA exact loglik (err %.1e) and innovations (err %.1e) vs statsmodels", llD - -78.51445819739115, errI))
	sfit := ts.sarima_fit(D, 0, 1, 1, 0, 1, 1, 4)
	sm_s := []f64{-0.48179387407895008, -0.47813484191536221, 0.22377779628657049}
	got_s := []f64{sfit.theta[0], sfit.Theta[0], sfit.sigma2}
	expect(passed, total, max_err(got_s, sm_s) < 1e-3 && sfit.loglik >= -77.770514611504623 - 1e-6 &&
		abs(sfit.aic - 161.54102922300925) < 1e-4 && abs(sfit.bic - 169.77582560809901) < 1e-4 && sfit.converged,
		fmt.tprintf("SARIMA MLE vs statsmodels (params err %.1e, loglik %.6f, aic %.4f)", max_err(got_s, sm_s), sfit.loglik, sfit.aic))
}
