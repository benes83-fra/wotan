package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import l "../wotan/linalg"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Memory regression tests for wotan/analytics (and the linalg Cholesky solvers it
// uses): every procedure below once leaked internal scratch when called with the
// default allocator. Each area runs under its own Tracking_Allocator wrapping the
// default heap allocator, frees exactly the documented outputs, and must end with
// no live allocations and no bad frees. Also: PCA reads Int / Bool columns by
// type (it used to reinterpret Int bits as f64).

@(private = "file")
_mt_begin :: proc(track: ^mem.Tracking_Allocator) -> mem.Allocator {
	mem.tracking_allocator_init(track, runtime.default_allocator())
	track.bad_free_callback = mem.tracking_allocator_bad_free_callback_add_to_array
	return mem.tracking_allocator(track)
}

// true when nothing is live and nothing was freed twice; destroys the tracker
@(private = "file")
_mt_end :: proc(track: ^mem.Tracking_Allocator, name: string) -> bool {
	live, bad := len(track.allocation_map), len(track.bad_free_array)
	if live > 0 || bad > 0 {
		fmt.printf("     %s: %d live allocation(s), %d bad free(s)\n", name, live, bad)
		shown := 0
		for _, e in track.allocation_map {
			if shown < 3 {fmt.printf("       allocated at %v\n", e.location)}
			shown += 1
		}
	}
	mem.tracking_allocator_destroy(track)
	return live == 0 && bad == 0
}

// ARMA(1,1), phi = 0.85, theta = 0.5 (the series of stationarity_reference_test)
@(private = "file")
MT_Y := [?]f64 {
	0,
	-0.069,
	0.516,
	0.863,
	0.251,
	0.307,
	1.746,
	3.083,
	2.39,
	0.414,
	-0.904,
	-1.039,
	-3.187,
	-4.09,
	-4.832,
	-5.463,
	-5.554,
	-5.309,
	-4.259,
	-2.372,
	-1.623,
	-0.078,
	-0.048,
	-0.022,
	1.061,
	1.447,
	0.534,
	-0.84,
	-1.632,
	-1.396,
	-2.086,
	-2.487,
	-2.378,
	-1.56,
	-0.841,
	-0.252,
	-0.69,
	-1.043,
	-0.168,
	1.743,
	0.969,
	1.708,
	3.555,
	4.476,
	4.459,
	3.609,
	4.369,
	6.403,
	8.224,
	9.206,
	8.84,
	6.485,
	4.903,
	4.822,
	3.139,
	2.419,
	2.683,
	3.192,
	1.877,
	0.342,
	-0.477,
	-1.793,
	-0.37,
	0.059,
	0.131,
	0.018,
	1.469,
	3.361,
	4.15,
	1.641,
	0.345,
	1.003,
	2.198,
	1.753,
	3.003,
	2.143,
	0.5,
	1.029,
	1.391,
	3.21,
}

@(private = "file")
_mt_f :: proc(x: [2]f64) -> [2]f64 {return {
		x[0] + 0.1 * x[1],
		0.95 * x[1] + 0.05 * math.sin(x[0]),
	}}
@(private = "file")
_mt_h :: proc(x: [2]f64) -> [1]f64 {return {x[0] * x[0] / 10 + x[0]}}
@(private = "file")
_mt_fu :: proc(x: [2]f64, u: [1]f64) -> [2]f64 {return {x[0] + 0.1 * x[1] + u[0], 0.95 * x[1]}}
@(private = "file")
_mt_hu :: proc(x: [2]f64, u: [1]f64) -> [1]f64 {return {x[0] + 0.5 * u[0]}}

analytics_memory_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Analytics memory (no leaks with the default allocator) ===")
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	y := MT_Y[:]

	// --- linalg: Cholesky solve and SPD inverse (the scratch factor was leaked) ---
	{
		track: mem.Tracking_Allocator
		ok := true
		{
			context.allocator = _mt_begin(&track)
			A := l.matrix_new(f64, 3, 3)
			copy(A.data, []f64{4, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 2})
			x := l.solve_spd_cholesky(&A, []f64{1, 2, 3})
			inv := l.spd_inverse(&A)
			// A x = b and A inv = I
			for i in 0 ..< 3 {
				s := 0.0
				for j in 0 ..< 3 {s += A.data[i * 3 + j] * x[j]}
				if abs(s - f64(i + 1)) > 1e-12 do ok = false
				for k in 0 ..< 3 {
					e := 0.0
					for j in 0 ..< 3 {e += A.data[i * 3 + j] * inv.data[j * 3 + k]}
					if abs(e - (1 if i == k else 0)) > 1e-12 do ok = false
				}
			}
			delete(x)
			l.matrix_free(&inv)
			l.matrix_free(&A)
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "cholesky") && ok,
			"solve_spd_cholesky / spd_inverse: correct, no leaks",
		)
	}

	// --- ACF / PACF / Ljung-Box and their DataFrame versions ---
	{
		track: mem.Tracking_Allocator
		pacf2 := 0.0
		{
			context.allocator = _mt_begin(&track)
			ac := ts.acf(y, 5)
			pc := ts.pacf(y, 5)
			pacf2 = pc[2]
			_, _, _ = ts.ljung_box(y, 10, 2)
			delete(ac)
			delete(pc)
			dfy := w.df_from(w.column_from_floats("y", y))
			c1 := ts.df_acf(&dfy, "y", 4)
			c2 := ts.df_pacf(&dfy, "y", 4)
			w.destroy_column(&c1)
			w.destroy_column(&c2)
			w.destroy_dataframe(&dfy)
			frames := [?]w.DataFrame {
				ts.df_ljung_box(y, 10, 2),
				ts.df_residual_diagnostics(y, 10, 2),
				ts.df_residual_acf(y, 5),
				ts.df_residual_pacf(y, 5),
			}
			for &f in frames {w.destroy_dataframe(&f)}
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "acf/pacf/ljung-box") && abs(pacf2 - -0.377429779527646) < 1e-12,
			"pacf, df_acf, df_pacf, ljung_box, df_ljung_box, df_residual_*: no leaks (pacf unchanged)",
		)
	}

	// --- ADF / KPSS / combined stationarity test ---
	{
		track: mem.Tracking_Allocator
		adf_s, kpss_s := 0.0, 0.0
		{
			context.allocator = _mt_begin(&track)
			adf_s, _, _, _, _, _, _ = ts.adf_test(y, 2, .Constant, .Fixed)
			_, _, _, _, _, _, _ = ts.adf_test(y, 6, .ConstantTrend, .AIC)
			kpss_s, _, _, _, _, _ = ts.kpss_test(y, .Level, -1)
			_, _, _, _, _, _ = ts.kpss_test(y, .Trend, 4)
			_, _, _, _, _ = ts.stationarity_test(y)
			_ = ts.auto_arima_d_from_tests(y)
			frames := [?]w.DataFrame{ts.df_adf(y, 2), ts.df_kpss(y), ts.df_stationarity(y)}
			for &f in frames {w.destroy_dataframe(&f)}
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "adf/kpss") &&
			abs(adf_s - -1.93640198829787) < 1e-9 &&
			abs(kpss_s - 0.445939393587291) < 1e-9,
			"adf_test, kpss_test, stationarity_test and df_* versions: no leaks (statistics unchanged)",
		)
	}

	// --- differencing ---
	{
		track: mem.Tracking_Allocator
		ok := true
		{
			context.allocator = _mt_begin(&track)
			x := []f64{1, 4, 9, 16, 25, 36}
			d2 := ts.difference(x, 2)
			d3 := ts.difference(x, 3)
			ok = ok && len(d2) == 4 && d2[0] == 2 && d2[3] == 2 && len(d3) == 3 && d3[0] == 0
			q := []f64{1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 5, 6, 7, 8}
			sd2 := ts.seasonal_difference(q, 2, 4)
			ok = ok && len(sd2) == 8 && sd2[0] == 0
			short := ts.seasonal_difference(q[:3], 1, 4) // too short: a copy, not the input
			ok = ok && len(short) == 3 && raw_data(short) != raw_data(q)
			sdd := ts.sarima_difference(q, 1, 1, 4)
			ok = ok && len(sdd) == 11
			back := ts.sarima_inverse_difference(sdd, q[:1], q[1:5], 1, 1, 4)
			delete(d2)
			delete(d3)
			delete(sd2)
			delete(short)
			delete(sdd)
			delete(back)
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "differencing") && ok,
			"difference (d >= 2), seasonal_difference, sarima_(inverse_)difference: correct, no leaks",
		)
	}

	// --- automatic ARIMA order selection ---
	{
		track: mem.Tracking_Allocator
		p, d, q := -1, -1, -1
		{
			context.allocator = _mt_begin(&track)
			a := ts.arima_auto(y, 1, 1, 1, "aic")
			delete(a.fit.phi)
			delete(a.fit.theta)
			b := ts.arima_auto_with_tests(y, 2, 2, "aic")
			p, d, q = b.p, b.d, b.q
			delete(b.fit.phi)
			delete(b.fit.theta)
			f := ts.df_arima_auto_with_tests(y, 1, 1)
			w.destroy_dataframe(&f)
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "arima auto") && p == 0 && d == 1 && q == 1,
			fmt.tprintf(
				"arima_auto, arima_auto_with_tests, df_arima_auto_with_tests: no leaks (best ARIMA(%d,%d,%d))",
				p,
				d,
				q,
			),
		)
	}

	// --- SARIMA: auto search, residual diagnostics, state space ---
	n_s := 48
	ys := make([]f64, n_s, allocator)
	for t in 0 ..< n_s {ys[t] = 10 + f64(t % 4) + 0.05 * f64(t) + rand.float64_normal(0, 0.5)}
	{
		track: mem.Tracking_Allocator
		{
			context.allocator = _mt_begin(&track)
			r := ts.sarima_auto_with_tests(ys, 1, 1, 1, 1, 1, 1, 4)
			delete(r.fit.phi)
			delete(r.fit.theta)
			delete(r.fit.Phi)
			delete(r.fit.Theta)
		}
		expect(&passed, &total, _mt_end(&track, "sarima auto"), "sarima_auto_with_tests: no leaks")
	}
	{
		track: mem.Tracking_Allocator
		{
			context.allocator = _mt_begin(&track)
			dg := ts.sarima_residual_diagnostics(ys, {0.3}, {0.2}, {}, {-0.4}, 1, 1, 4, 0.25, 8)
			delete(dg.acf_vals)
			delete(dg.pacf_vals)
			f := ts.df_sarima_residual_diagnostics(ys, {0.3}, {0.2}, {}, {-0.4}, 1, 1, 4, 0.25, 8)
			w.destroy_dataframe(&f)
			F, Q, P0, H, R, x0, _ := ts.sarima_state_space({0.3}, {0.2}, {0.5}, {-0.4}, 4, 0.25)
			for s in ([][]f64{F, Q, P0, H, R, x0}) {delete(s)}
			ar, ma := ts.sarima_expand_to_arma({0.3}, {0.5}, {0.2}, {-0.4}, 4)
			delete(ar)
			delete(ma)
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "sarima diag"),
			"sarima_residual_diagnostics, df_sarima_residual_diagnostics, sarima_state_space, sarima_expand_to_arma: no leaks",
		)
	}

	// --- residual DataFrame of an ARIMA fit ---
	{
		track: mem.Tracking_Allocator
		rows0, rows1 := 0, 0
		{
			context.allocator = _mt_begin(&track)
			f0 := ts.df_residuals(
				y,
				ts.ArimaFitResult{phi = {0.85}, theta = {0.5}, sigma2 = 1},
				1,
				0,
				1,
			)
			f1 := ts.df_residuals(y, ts.ArimaFitResult{phi = {0.3}, sigma2 = 1}, 1, 1, 0)
			rows0, rows1 = f0.rows, f1.rows
			w.destroy_dataframe(&f0)
			w.destroy_dataframe(&f1)
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "df_residuals") && rows0 == 80 && rows1 == 79,
			"df_residuals (d = 0 and d = 1): no leaks",
		)
	}

	// --- simulators ---
	{
		track: mem.Tracking_Allocator
		n1, n2 := 0, 0
		{
			context.allocator = _mt_begin(&track)
			s1 := ts.simulate_arima_pdq({0.5}, 1, {0.2}, 1.0, 30, context.allocator)
			s2 := ts.simulate_sarima_pdqPDQ(
				{0.5},
				1,
				{0.2},
				{0.3},
				1,
				{-0.4},
				4,
				1.0,
				30,
				context.allocator,
			)
			n1, n2 = len(s1), len(s2)
			delete(s1)
			delete(s2)
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "simulators") && n1 == 30 && n2 == 30,
			"simulate_arima_pdq, simulate_sarima_pdqPDQ: no leaks",
		)
	}

	// --- UKF with the default allocator ---
	{
		track: mem.Tracking_Allocator
		ok := true
		{
			context.allocator = _mt_begin(&track)
			params := ts.UKF_Params {
				alpha = 0.5,
				beta  = 2,
				kappa = 0,
			}
			Q := matrix[2, 2]f64{
				0.01, 0,
				0, 0.01,
			}
			R := matrix[1, 1]f64{
				0.1,
			}
			x := [2]f64{0.5, 0.1}
			P := matrix[2, 2]f64{
				1, 0,
				0, 1,
			}
			for k in 0 ..< 50 {
				z := [1]f64{0.5 + 0.01 * f64(k)}
				u := [1]f64{0.01}
				x, P = ts.ukf_predict(x, P, _mt_f, Q, params, context.allocator)
				x, P = ts.ukf_update(x, P, z, _mt_h, R, params, context.allocator)
				x, P = ts.ukf_predict_control(x, P, u, _mt_fu, Q, params, context.allocator)
				x, P = ts.ukf_update_control(x, P, z, u, _mt_hu, R, params, context.allocator)
			}
			ok = !math.is_nan(x[0]) && P[0, 0] > 0
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "ukf") && ok,
			"ukf_predict/update (+ _control): sigma-point scratch freed",
		)
	}

	// --- GARCH volatility model with an EGARCH result ---
	{
		r := make([]f64, 300, allocator)
		s2, prev := 1.0, 0.0
		for t in 0 ..< len(r) {
			s2 = 0.05 + 0.1 * prev * prev + 0.85 * s2
			prev = math.sqrt(s2) * rand.float64_normal(0, 1)
			r[t] = prev
		}
		track: mem.Tracking_Allocator
		{
			context.allocator = _mt_begin(&track)
			g := ts.garch_fit_spec(
				r,
				ts.GARCH_Spec{variance = .EGARCH, dist = .Normal, p = 1, q = 1},
				max_iter = 200,
			)
			vm := ts.VolatilityModel {
				model_type      = .GARCH,
				garch_result    = g,
				conditional_var = g.conditional_var,
			}
			ts.destroy_garch_volatility(&vm)
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "garch volatility"),
			"destroy_garch_volatility frees the EGARCH coefficients",
		)
	}

	// --- PCA reads Int and Bool columns by type ---
	{
		track: mem.Tracking_Allocator
		same, bool_ok, nan_ok := true, false, false
		{
			context.allocator = _mt_begin(&track)
			di := w.df_from(
				w.column_from_ints("a", []int{1, 2, 3, 4, 5}),
				w.column_from_ints("b", []int{2, 4, 5, 4, 5}),
			)
			df := w.df_from(
				w.column_from_floats("a", []f64{1, 2, 3, 4, 5}),
				w.column_from_floats("b", []f64{2, 4, 5, 4, 5}),
			)
			pi := ts.pca_dataframe(&di, {"a", "b"}, context.allocator)
			pf := ts.pca_dataframe(&df, {"a", "b"}, context.allocator)
			for k in 0 ..< 2 {
				if abs(pi.eigenvalues[k] - pf.eigenvalues[k]) > 1e-12 || pi.mean[k] != pf.mean[k] do same = false
				for j in 0 ..< 2 {if abs(pi.eigenvectors[k][j] - pf.eigenvectors[k][j]) > 1e-12 do same = false}
			}
			ts.destroy_pca_result(pi)
			ts.destroy_pca_result(pf)

			// Bool -> 0 / 1, null -> 0 (documented convention), NaN stays NaN
			c := w.column_new("f", .Float, 3)
			w.append_float(&c, math.nan_f64())
			w.append_null(&c)
			w.append_float(&c, 2.5)
			mixed := w.df_from(w.column_from_bools("b", []bool{true, false, true}), c)
			m := ts.extract_numeric_matrix(&mixed, {"b", "f"})
			bool_ok = m[0][0] == 1 && m[1][0] == 0 && m[2][0] == 1
			nan_ok = math.is_nan(m[0][1]) && m[1][1] == 0 && m[2][1] == 2.5
			ts.destroy_matrix(m)
			w.destroy_dataframe(&mixed)
			w.destroy_dataframe(&di)
			w.destroy_dataframe(&df)
		}
		expect(
			&passed,
			&total,
			_mt_end(&track, "pca") && same && bool_ok && nan_ok,
			"pca_dataframe on Int columns = on Float columns; Bool read as 0/1, null as 0, NaN kept",
		)
	}

	fmt.printf("Analytics memory: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
