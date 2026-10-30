package tests

import ts "../wotan/analytics"
import w "../wotan/core"
import "core:fmt"
import "core:math"
import "core:mem"

// Analytics edge cases that used to crash (bounds panics, Cholesky panics,
// use-after-free, double free). Each check only has to return sensibly:
// - acf/pacf with max_lag 0 or negative, and on an empty series
// - ADF on constant, all-zero and exact-trend series (singular X'X) and with negative lags
// - differencing / forecasting series shorter than the differencing order
// - GARCH on too-short input, forecasting from empty residuals, volatility-model cleanup
// - PCA on an empty covariance and inverse transform from truncated scores
// - rolling Count / bool Mean / quantile outside [0,1] / no aggregators / EWM_Std

analytics_edge_cases_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Analytics Edge Cases ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	ys := make([]f64, 60)
	for i in 0 ..< 60 do ys[i] = math.sin(f64(i) * 0.7) + 0.01 * f64(i * i % 7)
	constant := make([]f64, 100)
	for &v in constant do v = 3.7
	trend := make([]f64, 60)
	for i in 0 ..< 60 do trend[i] = 2 + 0.5 * f64(i)

	// --- acf / pacf ---
	expect(&passed, &total, len(ts.pacf(ys, 0)) == 1, "pacf with max_lag 0")
	expect(&passed, &total, len(ts.pacf([]f64{}, 3)) == 4, "pacf of an empty series")
	expect(&passed, &total, len(ts.acf(ys, -3)) == 1, "acf with negative max_lag")
	racf := ts.df_residual_acf([]f64{}, 3)
	expect(&passed, &total, racf.rows == 4, "df_residual_acf of an empty series")

	// --- ADF: singular regressions are rejected, not a Cholesky panic ---
	stat, _, _, _, _, _, _ := ts.adf_test(constant, 2)
	expect(&passed, &total, stat == 0, "adf_test on a constant series")
	zeros := make([]f64, 50)
	stat, _, _, _, _, _, _ = ts.adf_test(zeros, 0)
	expect(&passed, &total, stat == 0, "adf_test on an all-zero series")
	stat, _, _, _, _, _, _ = ts.adf_test(trend, 4, .Constant, .AIC)
	expect(&passed, &total, !math.is_nan(stat), "adf_test with AIC on an exact linear trend")
	_, _, lags, _, _, _, _ := ts.adf_test(ys, -1)
	expect(&passed, &total, lags == 0, "adf_test with negative max_lags")
	ts.stationarity_test(constant)
	expect(&passed, &total, true, "stationarity_test on a constant series")

	// --- differencing and forecasting short series ---
	expect(&passed, &total, len(ts.difference([]f64{}, 1)) == 0, "difference of an empty series")
	_, hist := ts.difference_with_history([]f64{1.0}, 2)
	expect(&passed, &total, len(hist) == 1, "difference_with_history shorter than d")
	fit := ts.arima_fit(ys, 1, 1, 1)
	fc_short := ts.arima_forecast([]f64{1.0}, fit, 1, 2, 1, 3, 0.05)
	expect(
		&passed,
		&total,
		len(fc_short.mean) == 3 && math.is_nan(fc_short.mean[0]),
		"arima_forecast on a series shorter than d gives NaN",
	)
	fc := ts.arima_forecast(ys, fit, 1, 1, 1, 5, 0.05)
	fc2 := ts.arima_forecast(ys, fit, 1, 1, 1, 5, 0.05)
	same := len(fc.mean) == len(fc2.mean)
	for v, i in fc.mean do if !same || v != fc2.mean[i] || math.is_nan(v) do same = false
	expect(&passed, &total, same, "arima_forecast is finite and repeatable")
	sfc := ts.sarima_forecast(ys[:5], {0.5}, {}, {0.3}, {}, 0, 1, 12, 3, 1.0)
	expect(&passed, &total, len(sfc.mean) == 3, "sarima_forecast on a series shorter than s")

	// --- GARCH / volatility ---
	gempty := ts.garch_fit([]f64{})
	expect(&passed, &total, !gempty.converged, "garch_fit on an empty series")
	gfit := ts.garch_fit(ys)
	gfc := ts.garch_forecast(&gfit, []f64{}, 3)
	expect(
		&passed,
		&total,
		math.is_nan(gfc.variance_forecast),
		"garch_forecast from empty residuals",
	)
	vm := ts.create_garch_volatility(ys)
	ts.destroy_garch_volatility(&vm)
	expect(&passed, &total, vm.conditional_var == nil, "destroy_garch_volatility frees once")

	// --- PCA ---
	pempty := ts.pca_from_cov(nil)
	expect(&passed, &total, len(pempty.eigenvalues) == 0, "pca_from_cov on an empty covariance")
	data := [][]f64{{1, 2, 3}, {2, 1, 0}, {0, 3, 1}, {4, 1, 2}}
	pca := ts.pca_from_cov(ts.covariance_matrix(data))
	scores := ts.pca_transform(data, pca)
	top2 := make([][]f64, len(scores))
	for r, i in scores do top2[i] = r[:2]
	recon := ts.pca_inverse_transform(top2, pca)
	expect(
		&passed,
		&total,
		len(recon) == 4 && len(recon[0]) == 3,
		"pca_inverse_transform from top-2 scores",
	)

	// --- rolling ---
	df := w.df_from(
		w.column_from_floats("x", []f64{1, 2, 3, 4, 5}),
		w.column_from_bools("b", []bool{true, false, true, true, false}),
		w.column_from_ints("i", []int{1, 2, 3, 4, 5}),
	)
	cnt := ts.rolling_apply(
		ts.rolling_window(&df, "b", 3, 1),
		w.Aggregator{name = "n", kind = .Count},
	)
	expect(&passed, &total, cnt.type == .Int && cnt.len == 5, "rolling Count on a Bool column")
	bmean := ts.rolling_apply(
		ts.rolling_window(&df, "b", 3, 1),
		w.Aggregator{name = "m", kind = .Mean},
	)
	expect(
		&passed,
		&total,
		bmean.type == .Float && bmean.len == 5,
		"rolling Mean on a Bool column",
	)
	bmax := ts.rolling_apply(
		ts.rolling_window(&df, "b", 3, -1),
		w.Aggregator{name = "mx", kind = .Max},
	)
	expect(&passed, &total, bmax.len == 5, "rolling Max with negative min_periods")
	q := ts.rolling_apply(ts.rolling_window(&df, "x", 3, 1), w.make_quantile_agg("q", "x", 1.5))
	qv, _ := w.column_at_float(&q, 4)
	expect(&passed, &total, qv == 5, "rolling quantile above 1 clamps to the max")
	imean := ts.rolling_apply(
		ts.rolling_window(&df, "i", 3, 1),
		w.Aggregator{name = "m", kind = .Mean},
	)
	iv, _ := w.column_at_float(&imean, 4)
	expect(
		&passed,
		&total,
		imean.type == .Float && iv == 4,
		"rolling int Mean uses the window (mean of 3,4,5)",
	)
	none := ts.rolling_apply_many(ts.rolling_window(&df, "x", 3, 1), {})
	expect(&passed, &total, none.rows == 0, "rolling_apply_many with no aggregators")
	many := ts.rolling_apply_many(
		ts.rolling_window(&df, "x", 3, 1),
		{
			w.Aggregator{name = "m", kind = .Mean},
			w.Aggregator{name = "s", kind = .EWM_Std, alpha = 0.5},
		},
	)
	expect(&passed, &total, many.rows == 5, "rolling_apply_many with Mean + EWM_Std")

	fmt.printf("Analytics edge cases summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
