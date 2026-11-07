package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Fama-MacBeth (1973) regressions
// ============================================================================
// One cross-sectional OLS per period, y_t = X_t b_t + e_t; the estimate is the time
// series mean of the b_t, its standard error the standard deviation of the b_t over
// sqrt(T) (sample standard deviation, T - 1), or with nw_lags > 0 the Newey-West
// standard error of that mean (OLS of b_t on a constant with HAC covariance, as
// statsmodels cov_type="HAC" without the small-sample correction). Periods with fewer
// rows than columns, or whose least squares fails, are skipped.

Fama_MacBeth_Result :: struct {
	coef:      []f64,
	std_err:   []f64,
	t_stat:    []f64,
	n_periods: int, // periods used
	allocator: mem.Allocator,
}

fama_macbeth_result_free :: proc(r: ^Fama_MacBeth_Result) {
	delete(r.coef, r.allocator)
	delete(r.std_err, r.allocator)
	delete(r.t_stat, r.allocator)
}

// X[t]: the period-t design (rows = assets, columns = regressors; include a column of
// ones for an intercept), y[t]: the period-t returns
fama_macbeth :: proc(
	X: []l.Matrix(f64),
	y: [][]f64,
	nw_lags := 0,
	allocator := context.allocator,
) -> Fama_MacBeth_Result {
	assert(
		len(X) == len(y) && len(X) > 0,
		"fama_macbeth: one design and one return vector per period",
	)
	k := X[0].cols
	coefs := make([dynamic]f64, context.temp_allocator) // T x k
	defer delete(coefs)
	for &Xt, t in X {
		assert(
			Xt.cols == k && Xt.rows == len(y[t]),
			"fama_macbeth: inconsistent period dimensions",
		)
		if Xt.rows < k {continue}
		r := l.lstsq(&Xt, y[t], .Auto, context.temp_allocator)
		ok := r.success && r.rank == k
		if ok {append(&coefs, ..r.beta[:k])}
		l.lstsq_result_free(&r, context.temp_allocator)
	}
	T := len(coefs) / k
	res := Fama_MacBeth_Result {
		coef      = make([]f64, k, allocator),
		std_err   = make([]f64, k, allocator),
		t_stat    = make([]f64, k, allocator),
		n_periods = T,
		allocator = allocator,
	}
	if T < 2 {
		for j in 0 ..< k {res.coef[j], res.std_err[j], res.t_stat[j] = math.nan_f64(), math.nan_f64(), math.nan_f64()}
		return res
	}
	series := make([]f64, T, context.temp_allocator)
	defer delete(series, context.temp_allocator)
	ones := l.matrix_new(f64, T, 1, context.temp_allocator)
	defer l.matrix_free(&ones)
	for &v in ones.data {v = 1}
	for j in 0 ..< k {
		m := 0.0
		for t in 0 ..< T {
			series[t] = coefs[t * k + j]
			m += series[t] / f64(T)
		}
		se: f64
		if nw_lags > 0 {
			fit := ols_fit_robust(
				&ones,
				series,
				{type = .HAC, maxlags = nw_lags},
				.Cholesky,
				context.temp_allocator,
			)
			se = fit.stderr[0]
			_ols_result_free(&fit, context.temp_allocator)
		} else {
			ss := 0.0
			for v in series {ss += (v - m) * (v - m)}
			se = math.sqrt(ss / f64(T - 1) / f64(T))
		}
		res.coef[j], res.std_err[j], res.t_stat[j] = m, se, m / se
	}
	return res
}
