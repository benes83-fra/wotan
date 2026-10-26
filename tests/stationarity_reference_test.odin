package tests

import ts "../wotan/analytics"
import "core:fmt"
import "core:mem"

// Time-series diagnostics against statsmodels 0.13.5 / scipy 1.10 reference
// values (hardcoded, computed on the series below):
// - acf (adjusted=False), pacf (method="ldb" = Durbin-Levinson on the biased acf)
// - acorr_ljungbox (model_df), scipy.stats.jarque_bera, chi-square tail
// - adfuller: fixed lag, AIC/BIC autolag on the common sample, MacKinnon
//   p-values and finite-sample critical values, maxlag cap on short series
// - kpss: Hobijn auto bandwidth, fixed lags, interpolated p-value

stationarity_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Stationarity / Diagnostics vs statsmodels ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	close :: proc(a, b: f64, tol := 1e-9) -> bool {
		return abs(a - b) <= tol * max(1.0, abs(b))
	}

	// ARMA(1,1), phi = 0.85, theta = 0.5, n = 80 (numpy default_rng(0), rounded)
	// odinfmt: disable
	y := []f64{0, -0.069, 0.516, 0.863, 0.251, 0.307, 1.746, 3.083, 2.39, 0.414, -0.904, -1.039, -3.187, -4.09, -4.832, -5.463, -5.554, -5.309, -4.259, -2.372, -1.623, -0.078, -0.048, -0.022, 1.061, 1.447, 0.534, -0.84, -1.632, -1.396, -2.086, -2.487, -2.378, -1.56, -0.841, -0.252, -0.69, -1.043, -0.168, 1.743, 0.969, 1.708, 3.555, 4.476, 4.459, 3.609, 4.369, 6.403, 8.224, 9.206, 8.84, 6.485, 4.903, 4.822, 3.139, 2.419, 2.683, 3.192, 1.877, 0.342, -0.477, -1.793, -0.37, 0.059, 0.131, 0.018, 1.469, 3.361, 4.15, 1.641, 0.345, 1.003, 2.198, 1.753, 3.003, 2.143, 0.5, 1.029, 1.391, 3.21}
	acf_ref := []f64{0.927806512282544, 0.808296106071757, 0.696543681754225, 0.598686882435524, 0.490512607878718}
	pacf_ref := []f64{0.927806512282544, -0.377429779527646, 0.115061635648885, -0.0387125486116585, -0.189124060172343}
	// odinfmt: enable

	// --- acf / pacf ---
	ac := ts.acf(y, 5)
	pc := ts.pacf(y, 5)
	ok_ac, ok_pc := true, true
	for k in 1 ..= 5 {
		if !close(ac[k], acf_ref[k - 1], 1e-12) do ok_ac = false
		if !close(pc[k], pacf_ref[k - 1], 1e-12) do ok_pc = false
	}
	expect(&passed, &total, ok_ac, fmt.tprintf("acf matches statsmodels (r1 = %.12f)", ac[1]))
	expect(&passed, &total, ok_pc, fmt.tprintf("pacf matches statsmodels ldb (phi22 = %.12f)", pc[2]))

	// --- Ljung-Box (10 lags, model_df 2): tiny p-value keeps relative precision ---
	Q, df, p_lb := ts.ljung_box(y, 10, 2)
	expect(&passed, &total,
		close(Q, 242.45790160995) && df == 8 && abs(p_lb / 6.82958468748923e-48 - 1) < 1e-8,
		fmt.tprintf("ljung_box Q = %.10f, df = %d, p = %.6e", Q, df, p_lb))
	Q_long, _, p_long := ts.ljung_box(y[:12], 20, 0)
	expect(&passed, &total, Q_long == Q_long && p_long == p_long, "ljung_box with max_lag >= n stays finite")

	// --- Jarque-Bera and chi-square tail ---
	JB, p_jb := ts.jarque_bera(y)
	expect(&passed, &total, close(JB, 2.72963397568836) && close(p_jb, 0.255427418033196),
		fmt.tprintf("jarque_bera JB = %.10f, p = %.10f", JB, p_jb))
	expect(&passed, &total,
		close(ts.chi2_sf(1000, 1000), 0.49405285382924) && close(ts.chi2_sf(300, 400), 0.999942903114258) &&
		abs(ts.chi2_sf(150, 10) / 3.7274850550625e-27 - 1) < 1e-10,
		"chi2_sf for large df and tiny tails")

	// --- ADF ---
	{
		s, p, lags, nobs, c1, c5, c10 := ts.adf_test(y, 2, .Constant, .Fixed)
		expect(&passed, &total,
			close(s, -1.93640198829787) && close(p, 0.315134917907817) && lags == 2 && nobs == 77 &&
			close(c1, -3.51828113466058) && close(c5, -2.89987818519143) && close(c10, -2.58722299375949),
			fmt.tprintf("adf c, fixed lag 2: stat %.10f, p %.10f, crit %.6f", s, p, c5))
	}
	{
		s, p, lags, nobs, c1, c5, c10 := ts.adf_test(y, 6, .ConstantTrend, .AIC)
		expect(&passed, &total,
			close(s, -2.9076334420339) && close(p, 0.159759708001583) && lags == 1 && nobs == 78 &&
			close(c1, -4.07979066664981) && close(c5, -3.46835800097776) && close(c10, -3.16089101299752),
			fmt.tprintf("adf ct, AIC <= 6: stat %.10f, p %.10f, lag %d, nobs %d", s, p, lags, nobs))
	}
	{
		s, p, lags, nobs, _, c5, _ := ts.adf_test(y, 6, .None, .BIC)
		expect(&passed, &total,
			close(s, -2.29877554529116) && close(p, 0.0207098725452723) && lags == 1 && nobs == 78 &&
			close(c5, -1.94493088512955),
			fmt.tprintf("adf n, BIC <= 6: stat %.10f, p %.10f, lag %d", s, p, lags))
	}
	{
		// 20 points: maxlag 12 is capped at 20/2 - 1 - 1 = 8 (statsmodels maxlag=8)
		s, p, lags, nobs, _, _, _ := ts.adf_test(y[:20], 12, .Constant, .AIC)
		expect(&passed, &total,
			close(s, -2.69817112796952) && close(p, 0.074355278811514) && lags == 8 && nobs == 11,
			fmt.tprintf("adf c, AIC on a short series: stat %.10f, lag %d, nobs %d", s, lags, nobs))
	}

	// --- KPSS ---
	{
		s, p, _, _, _, _ := ts.kpss_test(y, .Level, -1)
		expect(&passed, &total, close(s, 0.445939393587291) && close(p, 0.0573537096606505),
			fmt.tprintf("kpss level, auto lags (5): stat %.10f, p %.10f", s, p))
		s, p, _, _, _, _ = ts.kpss_test(y, .Trend, 4)
		expect(&passed, &total, close(s, 0.12646322902656) && close(p, 0.0861792055063702),
			fmt.tprintf("kpss trend, 4 lags: stat %.10f, p %.10f", s, p))
		s, p, _, _, _, _ = ts.kpss_test(y, .Level, 12)
		expect(&passed, &total, close(s, 0.292075756605375) && p == 0.10,
			fmt.tprintf("kpss level, 12 lags (statsmodels legacy): stat %.10f, p clamped %.2f", s, p))
	}

	fmt.printf("Stationarity reference: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
