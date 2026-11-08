package tests

import ml "../wotan/analytics/ML"
import bfin "../wotan/behavioral_finance"
import l "../wotan/linalg"
import "core:math"

// Cross-sectional test of a return predictor: each period the low-minus-high
// portfolio (long the lowest `fraction` of the signal, short the highest, equally
// weighted: anchoring_portfolio on -signal) and the Fama-MacBeth slope of the return
// on the signal. R, S [asset][period], S[i][s] predicts R[i][s]; periods first ..
// end. Assets with a NaN signal or return in a period are left out of it.
signal_sort :: proc(
	R, S: [][]f64,
	first: int,
	fraction := 0.2,
) -> (
	spread_mean, spread_t, fm_coef, fm_t: f64,
	periods: int,
) {
	N, M := len(R), len(R[0])
	col := make([]f64, N, context.temp_allocator)
	spreads := make([dynamic]f64, context.temp_allocator)
	Xs := make([dynamic]l.Matrix(f64), context.temp_allocator)
	Ys := make([dynamic][]f64, context.temp_allocator)
	defer {
		delete(col, context.temp_allocator)
		delete(spreads)
		for i in 0 ..< len(Xs) {l.matrix_free(&Xs[i]); delete(Ys[i], context.temp_allocator)}
		delete(Xs)
		delete(Ys)
	}
	for s in first ..< M {
		n := 0
		for i in 0 ..< N {
			ok := !math.is_nan(S[i][s]) && !math.is_nan(R[i][s])
			col[i] = -S[i][s] if ok else math.nan_f64()
			if ok {n += 1}
		}
		w := bfin.anchoring_portfolio(col, fraction, context.temp_allocator)
		sp := 0.0
		for i in 0 ..< N {if w[i] != 0 {sp += w[i] * R[i][s]}}
		delete(w, context.temp_allocator)
		append(&spreads, sp)
		X := l.matrix_new(f64, n, 2, context.temp_allocator)
		y := make([]f64, n, context.temp_allocator)
		k := 0
		for i in 0 ..< N {
			if math.is_nan(col[i]) {continue}
			X.data[2 * k], X.data[2 * k + 1], y[k] = 1, S[i][s], R[i][s]
			k += 1
		}
		append(&Xs, X)
		append(&Ys, y)
	}
	periods = len(spreads)
	for v in spreads {spread_mean += v / f64(periods)}
	ss := 0.0
	for v in spreads {ss += (v - spread_mean) * (v - spread_mean)}
	spread_t = spread_mean / math.sqrt(ss / f64(periods - 1) / f64(periods))
	fm := ml.fama_macbeth(Xs[:], Ys[:], allocator = context.temp_allocator)
	fm_coef, fm_t = fm.coef[1], fm.t_stat[1]
	ml.fama_macbeth_result_free(&fm)
	return
}

// monthly compounded returns [asset][month] from daily returns [asset][day]: months[k]
// .. months[k+1] - 1 are the days of month k
monthly_compound :: proc(R: [][]f64, months: []int) -> [][]f64 {
	out := make([][]f64, len(R))
	for r, i in R {
		out[i] = make([]f64, len(months) - 1)
		for k in 0 ..< len(months) - 1 {
			g := 1.0
			for v in r[months[k]:months[k + 1]] {g *= 1 + v}
			out[i][k] = g - 1
		}
	}
	return out
}

// mean of a series and its t-statistic, Newey-West with `lags` lags (0: classical)
// as statsmodels OLS on a constant, cov_type="HAC" without correction; NaN entries
// are left out
mean_t :: proc(series: []f64, lags := 0) -> (mean, t: f64) {
	v := make([dynamic]f64, context.temp_allocator)
	defer delete(v)
	for x in series {if !math.is_nan(x) {append(&v, x)}}
	T := len(v)
	if T < 2 {return math.nan_f64(), math.nan_f64()}
	ones := l.matrix_new(f64, T, 1, context.temp_allocator)
	defer l.matrix_free(&ones)
	for &o in ones.data {o = 1}
	fit :=
		ml.ols_fit_robust(&ones, v[:], {type = .HAC, maxlags = lags}, .Cholesky, context.temp_allocator) if lags > 0 else ml.ols_fit(&ones, v[:], .Cholesky, context.temp_allocator)
	mean, t = fit.beta[0], fit.tvalues[0]
	ml._ols_result_free(&fit, context.temp_allocator)
	return
}

// [period][asset] from [asset][period]
transpose_panel :: proc(X: [][]f64) -> [][]f64 {
	out := make([][]f64, len(X[0]))
	for t in 0 ..< len(X[0]) {
		out[t] = make([]f64, len(X))
		for i in 0 ..< len(X) {out[t][i] = X[i][t]}
	}
	return out
}

// mean monthly spread of the overlapping strategy with weights W [month][asset] on
// returns Rt [month][asset] from month `first` on, Newey-West t (`lags` lags)
overlapping_spread_t :: proc(W, Rt: [][]f64, hold, first: int, lags := 6) -> (mean, t: f64) {
	o := bfin.overlapping_strategy_returns(W, Rt, hold, allocator = context.temp_allocator)
	defer bfin.overlapping_returns_free(&o)
	return mean_t(o.spread[first:], lags)
}

// JT / DBT weights per month from a signal panel S [asset][month]: long the top
// `fraction`, short the bottom one
long_short_weights :: proc(S: [][]f64, fraction := 0.1) -> [][]f64 {
	W := make([][]f64, len(S[0]))
	col := make([]f64, len(S))
	defer delete(col)
	for t in 0 ..< len(S[0]) {
		for i in 0 ..< len(S) {col[i] = S[i][t]}
		W[t] = bfin.long_short_portfolio(col, fraction)
	}
	return W
}

free_panel :: proc(X: [][]f64) {for r in X {delete(r)}; delete(X)}
