package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

garch_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== GARCH Model Test ===\n")

	main_alloc := context.allocator

	// Generate proper GARCH(1,1) data
	fmt.println("--- Generating Synthetic GARCH(1,1) Returns ---")
	n := 1000 // More data for better estimation
	returns := make([]f64, n, main_alloc)
	defer delete(returns, main_alloc)

	true_omega := 0.00001
	true_alpha := 0.1
	true_beta := 0.85

	cond_var := make([]f64, n, context.temp_allocator)
	defer delete(cond_var, context.temp_allocator)

	// Initialize
	cond_var[0] = true_omega / (1.0 - true_alpha - true_beta)
	returns[0] = 0.0

	// Generate returns with proper Box-Muller
	for i in 1 ..< n {
		cond_var[i] =
			true_omega + true_alpha * returns[i - 1] * returns[i - 1] + true_beta * cond_var[i - 1]
		std_dev := math.sqrt_f64(cond_var[i])

		// Box-Muller transform
		u1 := rand.float64()
		u2 := rand.float64()
		if u1 < 1e-10 {u1 = 1e-10}
		z := math.sqrt_f64(-2.0 * math.ln_f64(u1)) * math.cos_f64(2.0 * math.PI * u2)
		returns[i] = z * std_dev
	}

	fmt.printf("Generated %d returns\n", n)
	fmt.printf(
		"True parameters: ω=%.6f, α=%.4f, β=%.4f, α+β=%.4f\n",
		true_omega,
		true_alpha,
		true_beta,
		true_alpha + true_beta,
	)

	// Extract residuals
	residuals := ts.extract_residuals(returns, main_alloc)
	defer delete(residuals, main_alloc)

	// Fit GARCH(1,1)
	fmt.println("\n--- Fitting GARCH(1,1) Model ---")
	result := ts.garch_fit(residuals, .GARCH, 1, 1, 1000, 1e-6, main_alloc)
	defer {
		delete(result.params.alpha, main_alloc)
		delete(result.params.beta, main_alloc)
		delete(result.conditional_var, main_alloc)
		delete(result.standardized_resid, main_alloc)
	}

	fmt.printf("\n=== GARCH(1,1) Results ===\n")
	fmt.printf("  Omega: %.6f (true: %.6f)\n", result.params.omega, true_omega)
	fmt.printf("  Alpha: %.6f (true: %.6f)\n", result.params.alpha[0], true_alpha)
	fmt.printf("  Beta:  %.6f (true: %.6f)\n", result.params.beta[0], true_beta)
	fmt.printf(
		"  Persistence (α+β): %.4f (true: %.4f)\n",
		result.persistence,
		true_alpha + true_beta,
	)
	fmt.printf("  Log-Likelihood: %.2f\n", result.log_likelihood)
	fmt.printf("  AIC: %.2f\n", result.aic)
	fmt.printf("  BIC: %.2f\n", result.bic)
	fmt.printf("  Converged: %v (iterations: %d)\n", result.converged, result.n_iterations)

	// Forecast
	fmt.println("\n--- Volatility Forecast ---")
	forecast_1 := ts.garch_forecast(&result, residuals, 1, main_alloc)
	forecast_5 := ts.garch_forecast(&result, residuals, 5, main_alloc)
	forecast_10 := ts.garch_forecast(&result, residuals, 10, main_alloc)

	fmt.printf(
		"1-day ahead:  σ²=%.6f, σ=%.6f\n",
		forecast_1.variance_forecast,
		forecast_1.std_forecast,
	)
	fmt.printf(
		"5-day ahead:  σ²=%.6f, σ=%.6f\n",
		forecast_5.variance_forecast,
		forecast_5.std_forecast,
	)
	fmt.printf(
		"10-day ahead: σ²=%.6f, σ=%.6f\n",
		forecast_10.variance_forecast,
		forecast_10.std_forecast,
	)

	fmt.println("\n✓ GARCH test completed!")

	garch_reference_checks(main_alloc)
}

// Deterministic GARCH/GJR shocks (splitmix64 + Box-Muller; Student-t(nu) innovations
// scaled to unit variance when nu > 0) so the reference values below, computed with
// numpy/scipy on the identical series, can be hardcoded.
garch_ref_uniform :: proc(state: ^u64) -> f64 {
	state^ += 0x9E3779B97F4A7C15
	z := state^
	z = (z ~ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ~ (z >> 27)) * 0x94D049BB133111EB
	z ~= z >> 31
	return (f64(z >> 11) + 0.5) * 1.1102230246251565e-16 // 2^-53
}

garch_ref_normal :: proc(state: ^u64) -> f64 {
	u1 := garch_ref_uniform(state)
	u2 := garch_ref_uniform(state)
	return math.sqrt_f64(-2.0 * math.ln_f64(u1)) * math.cos_f64(2.0 * math.PI * u2)
}

garch_ref_shocks :: proc(n: int, omega, alpha, beta, gamma: f64, nu: int, seed: u64, allocator: mem.Allocator) -> []f64 {
	state := seed
	normal :: garch_ref_normal
	e := make([]f64, n, allocator)
	h := omega / (1.0 - alpha - beta - 0.5 * gamma)
	prev := 0.0
	for t in 0 ..< n {
		if t > 0 {
			h = omega + alpha * prev * prev + beta * h
			if prev < 0.0 {h += gamma * prev * prev}
		}
		z := normal(&state)
		if nu > 0 {
			c := 0.0
			for _ in 0 ..< nu {x := normal(&state); c += x * x}
			z = z / math.sqrt_f64(c / f64(nu)) * math.sqrt_f64(f64(nu - 2) / f64(nu))
		}
		prev = math.sqrt_f64(h) * z
		e[t] = prev
	}
	return e
}

// GARCH reference checks (numpy / scipy 1.10 / mpmath values hardcoded below):
// - exact negative log-likelihood (normal, GJR, unit-variance Student-t and GED)
//   at fixed parameters; variance seeded with mean(ε²), first obs not scored
// - garch_fit reaches the scipy SLSQP + Nelder-Mead optimum
// - standardized-error quantiles (VaR, forecast interval) are those of the fitted
//   error distribution, not the normal
// - multi-step forecast E[σ²_{n+h}] = ω + (α+β+γ/2) E[σ²_{n+h-1}], AIC/BIC counts
garch_reference_checks :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GARCH Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	rel :: proc(a, b: f64) -> f64 {return abs(a / b - 1.0)}
	free_result :: proc(r: ^ts.GARCH_Result) {
		delete(r.params.alpha)
		delete(r.params.beta)
		delete(r.conditional_var)
		delete(r.standardized_resid)
	}

	A := garch_ref_shocks(1500, 1e-5, 0.08, 0.9, 0.0, 0, 11, allocator)
	defer delete(A)
	B := garch_ref_shocks(1500, 1e-5, 0.03, 0.88, 0.12, 0, 12, allocator)
	defer delete(B)
	C := garch_ref_shocks(1500, 1e-5, 0.08, 0.9, 0.0, 6, 13, allocator)
	defer delete(C)
	expect(&passed, &total, rel(A[0], -0.002633451058836295) < 1e-12 && rel(A[1499], -0.038240103729992772) < 1e-9,
		fmt.tprintf("reference series reproduced (A[1499] = %.17g)", A[1499]))

	// 1. likelihood at fixed parameters
	alpha, beta := []f64{0.07}, []f64{0.88}
	pn := ts.GARCH_Params{model_type = .GARCH, p = 1, q = 1, omega = 2e-5, alpha = alpha, beta = beta}
	pg := pn
	pg.model_type, pg.gamma = .GJR_GARCH, 0.1
	pt := pn
	pt.model_type, pt.nu = .StudentT, 7.0
	pe := pn
	pe.model_type, pe.shape = .GED, 1.4
	ln := ts._garch_neg_log_likelihood(&pn, A, allocator)
	lg := ts._garch_neg_log_likelihood(&pg, B, allocator)
	lt := ts._garch_neg_log_likelihood(&pt, C, allocator)
	le := ts._garch_neg_log_likelihood(&pe, C, allocator)
	expect(&passed, &total,
		rel(ln, -3730.491491023519) < 1e-11 && rel(lg, -3937.013277836470) < 1e-11 &&
		rel(lt, -3937.333033168709) < 1e-11 && rel(le, -3931.813316354995) < 1e-11,
		fmt.tprintf("NLL normal %.9f, GJR %.9f, t %.9f, GED %.9f match numpy", ln, lg, lt, le))

	// 2. fits reach the scipy optimum (log-likelihoods and parameters)
	fa := ts.garch_fit(A, .GARCH, 1, 1, 2000, 1e-4, allocator)
	defer free_result(&fa)
	fb := ts.garch_fit(B, .GJR_GARCH, 1, 1, 2000, 1e-4, allocator)
	defer free_result(&fb)
	ft := ts.garch_fit(C, .StudentT, 1, 1, 2000, 1e-4, allocator)
	defer free_result(&ft)
	fe := ts.garch_fit(C, .GED, 1, 1, 2000, 1e-4, allocator)
	defer free_result(&fe)
	expect(&passed, &total,
		fa.log_likelihood > 3730.9377787268 - 1e-5 && fb.log_likelihood > 4003.9978707631 - 1e-5 &&
		ft.log_likelihood > 3941.0213101953 - 1e-5 && fe.log_likelihood > 3934.8978257394 - 1e-5,
		fmt.tprintf("fitted LL reach scipy (%.6f / %.6f / %.6f / %.6f)", fa.log_likelihood, fb.log_likelihood, ft.log_likelihood, fe.log_likelihood))
	expect(&passed, &total,
		rel(fa.params.omega, 1.633046271e-05) < 1e-3 && abs(fa.params.alpha[0] - 0.07394053481) < 1e-4 && abs(fa.params.beta[0] - 0.8881745425) < 1e-4 &&
		abs(fb.params.gamma - 0.1291771499) < 1e-4 && abs(fb.params.alpha[0] - 0.03723160666) < 1e-4 &&
		abs(ft.params.nu - 6.434370286) < 1e-2 && abs(fe.params.shape - 1.402325767) < 1e-3,
		fmt.tprintf("fitted params: α %.5f β %.5f | GJR γ %.5f | ν %.4f | GED p %.4f",
			fa.params.alpha[0], fa.params.beta[0], fb.params.gamma, ft.params.nu, fe.params.shape))

	// 3. AIC/BIC count ω, α, β (+γ, +ν/p): k = 3 or 4
	expect(&passed, &total,
		abs(fa.aic - (6.0 - 2.0 * fa.log_likelihood)) < 1e-9 &&
		abs(ft.bic - (4.0 * math.ln_f64(1500) - 2.0 * ft.log_likelihood)) < 1e-9 &&
		abs(fb.aic - (8.0 - 2.0 * fb.log_likelihood)) < 1e-9 && fa.n_iterations > 0 && fa.n_iterations <= 2000,
		fmt.tprintf("AIC/BIC parameter counts (AIC %.4f, %d iterations)", fa.aic, fa.n_iterations))

	// 4. standardized-error quantiles (mpmath references)
	q_t := ts.GARCH_Params{model_type = .StudentT, nu = 6.5}
	q_g := ts.GARCH_Params{model_type = .GED, shape = 1.4}
	q_n := ts.GARCH_Params{model_type = .GARCH}
	expect(&passed, &total,
		abs(ts.garch_std_quantile(&q_t, 0.95) - 1.5948206627112243) < 1e-10 &&
		abs(ts.garch_std_quantile(&q_t, 0.99) - 2.5488741748091449) < 1e-10 &&
		abs(ts.garch_std_quantile(&q_g, 0.95) - 1.6522327018752828) < 1e-10 &&
		abs(ts.garch_std_quantile(&q_g, 0.99) - 2.5422389384158263) < 1e-10 &&
		abs(ts.garch_std_quantile(&q_n, 0.975) - 1.959963984540054) < 1e-12 &&
		abs(ts.garch_std_quantile(&q_t, 0.01) + 2.5488741748091449) < 1e-10,
		"unit-variance t / GED / normal quantiles match mpmath")
	v95, v99 := ts.garch_var_series(&ft, C, allocator)
	defer {delete(v95); delete(v99)}
	sd := math.sqrt_f64(ft.conditional_var[1499])
	expect(&passed, &total,
		rel(v99[1499], ts.garch_std_quantile(&ft.params, 0.99) * sd) < 1e-14 &&
		v99[1499] / sd > 2.4 && v95[1499] / sd < 1.62,
		fmt.tprintf("Student-t VaR uses the t quantile (z99 = %.4f, normal 2.3263)", v99[1499] / sd))

	// 5. forecasts: one step from the data, then mean reversion at rate α+β+γ/2
	g := fb.params
	last := B[1499]
	h1 := g.omega + g.alpha[0] * last * last + g.beta[0] * fb.conditional_var[1499]
	if last < 0.0 {h1 += g.gamma * last * last}
	h := h1
	for _ in 2 ..= 10 {h = g.omega + fb.persistence * h}
	f10 := ts.garch_forecast(&fb, B, 10, allocator)
	expect(&passed, &total,
		rel(f10.variance_forecast, h) < 1e-12 && abs(fb.persistence - (g.alpha[0] + g.beta[0] + 0.5 * g.gamma)) < 1e-15 &&
		rel(f10.confidence_95[1], 1.959963984540054 * math.sqrt_f64(h)) < 1e-10,
		fmt.tprintf("10-step GJR forecast %.6e follows the recursion", f10.variance_forecast))
	z_last := C[1499] / math.sqrt_f64(ft.conditional_var[1499])
	expect(&passed, &total, rel(ft.standardized_resid[1499], z_last) < 1e-14, "standardized residuals are ε / σ")

	// 6. volatility models: NaN warm-up for rolling / EWMA
	{
		df := w.dataframe_new()
		defer w.destroy_dataframe(&df)
		w.add_column(&df, w.column_from_floats("r", A[:40]))
		df.rows = 40
		rm := ts.create_rolling_volatility(&df, "r", 5)
		em := ts.create_ewma_volatility(&df, "r", 0.1, 3)
		ok := len(rm.conditional_var) == 40 && len(em.conditional_var) == 40
		if ok {
			for i in 0 ..< 4 {ok &= math.is_nan(rm.conditional_var[i])}
			for i in 4 ..< 40 {ok &= rm.conditional_var[i] > 0}
			for i in 0 ..< 2 {ok &= math.is_nan(em.conditional_var[i])}
			for i in 2 ..< 40 {ok &= !math.is_nan(em.conditional_var[i])}
		}
		ts.destroy_garch_volatility(&rm)
		ts.destroy_garch_volatility(&em)
		expect(&passed, &total, ok, "rolling / EWMA volatility warm-up rows are NaN, not 0")
	}

	fmt.printf("GARCH reference summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
