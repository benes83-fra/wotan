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
