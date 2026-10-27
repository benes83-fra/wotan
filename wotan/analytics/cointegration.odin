package analytics

import l "../linalg"
import ml "./ML"
import "core:math"
import "core:mem"

// ============================================================================
// Cointegration tests: Engle-Granger (statsmodels coint) and Johansen
// (statsmodels coint_johansen), with MacKinnon p-values / critical values.
// ============================================================================

// Deterministic terms of the cointegrating regression (statsmodels trend)
CointTrend :: enum {
	None, // "n"
	Constant, // "c"
	ConstantTrend, // "ct"
	ConstantTrendSq, // "ctt"
}

// MacKinnon (1994) approximate asymptotic p-value of a unit-root / cointegration
// t-statistic for N = 1 … 6 integrated variables (statsmodels mackinnonp)
mackinnon_pvalue :: proc(stat: f64, trend: CointTrend, n_vars: int = 1) -> f64 {
	if n_vars < 1 || n_vars > 6 do panic("mackinnon_pvalue: n_vars must be 1 … 6")
	if math.is_nan(stat) do return math.nan_f64()
	r, i := int(trend), n_vars - 1
	if stat > _MK_TAU_MAX[r][i] do return 1
	if stat < _MK_TAU_MIN[r][i] do return 0
	z := 0.0
	if stat <= _MK_TAU_STAR[r][i] {
		c := _MK_SMALLP[r][i]
		z = c[0] + stat * (c[1] + stat * c[2])
	} else {
		c := _MK_LARGEP[r][i]
		z = c[0] + stat * (c[1] + stat * (c[2] + stat * c[3]))
	}
	return 0.5 * math.erfc(-z / math.SQRT_TWO)
}

// MacKinnon (2010) finite-sample critical values (1 %, 5 %, 10 %) for N = 1 … 12
// variables and nobs observations (nobs <= 0: asymptotic); statsmodels mackinnoncrit
mackinnon_critical_values :: proc(n_vars: int, trend: CointTrend, nobs: int = 0) -> [3]f64 {
	if n_vars < 1 || n_vars > 12 do panic("mackinnon_critical_values: n_vars must be 1 … 12")
	out: [3]f64
	tab := _MK_TAU_2010[int(trend)][n_vars - 1]
	for q in 0 ..< 3 {
		c := tab[q]
		if nobs <= 0 {
			out[q] = c[0]
		} else {
			x := 1 / f64(nobs)
			out[q] = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
		}
	}
	return out
}

// ----------------------------------------------------------------------------
// Engle-Granger two-step test
// ----------------------------------------------------------------------------

EngleGrangerResult :: struct {
	statistic: f64, // ADF t-statistic of the residuals (-Inf if y0 is a perfect fit)
	p_value:   f64, // MacKinnon asymptotic p-value (N = 1 + number of regressors)
	crit:      [3]f64, // 1 %, 5 %, 10 % critical values (NaN for trend .None)
	lags_used: int, // ADF lags chosen by AIC
	beta:      []f64, // cointegrating regression: coefficients of y1 series, then const / trend terms
	residuals: []f64, // y0 - fitted (the spread)
	allocator: mem.Allocator,
}

// H0: no cointegration between y0 and the series y1[0], y1[1], …
// (statsmodels coint(y0, y1, trend, autolag="aic")): OLS of y0 on [y1 …, deterministic
// terms], then an ADF test without deterministic terms on the residuals, lag
// length by AIC up to max_lags (< 0: ⌈12 (n/100)^¼⌉, capped at n/2 - 1).
// A small statistic / p-value rejects H0, i.e. the series are cointegrated.
engle_granger :: proc(
	y0: []f64,
	y1: [][]f64,
	trend: CointTrend = .Constant,
	max_lags: int = -1,
	allocator: mem.Allocator = context.allocator,
) -> EngleGrangerResult {
	n := len(y0)
	m := len(y1)
	if m < 1 do panic("engle_granger: need at least one series in y1")
	for s in y1 {if len(s) != n do panic("engle_granger: all series need the length of y0")}
	n_det := int(trend) // 0, 1, 2, 3 deterministic columns
	P := m + n_det
	if n <= P + 2 do panic("engle_granger: series too short")

	X := l.matrix_new(f64, n, P, context.temp_allocator)
	defer l.matrix_free(&X)
	for t in 0 ..< n {
		for j in 0 ..< m {X.data[t * P + j] = y1[j][t]}
		tt := f64(t + 1)
		if n_det >= 1 do X.data[t * P + m] = 1
		if n_det >= 2 do X.data[t * P + m + 1] = tt
		if n_det >= 3 do X.data[t * P + m + 2] = tt * tt
	}
	res := EngleGrangerResult {
		allocator = allocator,
	}
	res.beta = make([]f64, P, allocator)
	res.residuals = make([]f64, n, allocator)
	_lstsq_resid(&X, y0, res.beta, res.residuals)

	// R² (centred with a constant, uncentred without, as statsmodels OLS)
	ssr, tss, mean := 0.0, 0.0, 0.0
	for v in y0 {mean += v}
	mean /= f64(n)
	for v, t in y0 {
		ssr += res.residuals[t] * res.residuals[t]
		d := v - mean if n_det >= 1 else v
		tss += d * d
	}
	r2 := 1 - ssr / tss
	if r2 < 1 - 100 * math.sqrt_f64(math.F64_EPSILON) {
		ml_lags := max_lags
		if ml_lags < 0 do ml_lags = int(math.ceil(12 * math.pow(f64(n) / 100, 0.25)))
		ml_lags = min(n / 2 - 1, ml_lags)
		if ml_lags < 0 do panic("engle_granger: series too short for the ADF regression")
		stat, _, lags, _, _, _, _ := adf_test(res.residuals, ml_lags, .None, .AIC, context.temp_allocator)
		res.statistic, res.lags_used = stat, lags
	} else {
		res.statistic = math.inf_f64(-1) // (almost) perfectly collinear: not reliable
	}
	res.p_value = mackinnon_pvalue(res.statistic, trend, min(m + 1, 6)) if m + 1 <= 6 else math.nan_f64()
	if trend == .None {
		res.crit = {math.nan_f64(), math.nan_f64(), math.nan_f64()}
	} else {
		// nobs - 1, as statsmodels (to match Stata's egranger)
		res.crit = mackinnon_critical_values(m + 1, trend, n - 1)
	}
	return res
}

engle_granger_free :: proc(r: ^EngleGrangerResult) {
	delete(r.beta, r.allocator)
	delete(r.residuals, r.allocator)
}

// ----------------------------------------------------------------------------
// Johansen test
// ----------------------------------------------------------------------------

JohansenResult :: struct {
	eig:          []f64, // eigenvalues λ_1 ≥ … ≥ λ_k
	evec:         l.Matrix(f64), // k × k, column i = cointegrating vector of λ_i, normalised vᵀ S₁₁ v = 1
	trace_stat:   []f64, // trace statistic for H0: rank ≤ r, r = 0 … k-1 (statsmodels lr1)
	max_eig_stat: []f64, // max-eigenvalue statistic for H0: rank = r vs r + 1 (lr2)
	trace_crit:   [][3]f64, // 90 %, 95 %, 99 % critical values (cvt; NaN beyond 12 variables)
	max_eig_crit: [][3]f64, // (cvm)
	rank_trace:   int, // cointegration rank by the sequential trace test at 5 %
	rank_max_eig: int, // … by the sequential max-eigenvalue test at 5 %
	nobs:         int, // observations in the reduced-rank regression
	allocator:    mem.Allocator,
}

// Johansen cointegration test on X (n × k, one column per series), as statsmodels
// coint_johansen(X, det_order, k_ar_diff):
//   det_order -1: no deterministic terms, 0: constant, 1: linear trend
//   k_ar_diff: lagged differences in the VECM
// The levels are detrended (order det_order), Δx_t and x_{t-k_ar_diff} are
// regressed on the lagged differences (demeaned for det_order >= 0); the
// eigenvalues solve |λ S₁₁ - S₁₀ S₀₀⁻¹ S₀₁| = 0. Eigenvectors: each column is
// scaled to vᵀS₁₁v = 1 with its first non-zero entry positive (statsmodels fixes
// only the overall sign, so other columns can differ from it in sign).
johansen_test :: proc(
	X: ^l.Matrix(f64),
	det_order: int,
	k_ar_diff: int,
	allocator: mem.Allocator = context.allocator,
) -> JohansenResult {
	n, k := X.rows, X.cols
	if det_order < -1 || det_order > 1 do panic("johansen_test: det_order must be -1, 0 or 1")
	if k_ar_diff < 0 do panic("johansen_test: k_ar_diff must be >= 0")
	T := n - 1 - k_ar_diff
	if T <= k * (k_ar_diff + 1) + 1 do panic("johansen_test: series too short")
	f := 0 if det_order > -1 else -1

	x := l.matrix_new(f64, n, k, context.temp_allocator)
	defer l.matrix_free(&x)
	copy(x.data, X.data)
	_detrend_cols(&x, det_order)

	// Δx (rows k_ar_diff … n-2 of the differences), z = lagged differences,
	// lx = x_{t-k_ar_diff}: rows 1 … n-k_ar_diff-1 of the levels
	q := k * k_ar_diff
	dx := l.matrix_new(f64, T, k, context.temp_allocator)
	z := l.matrix_new(f64, T, q, context.temp_allocator)
	lx := l.matrix_new(f64, T, k, context.temp_allocator)
	defer {
		l.matrix_free(&dx)
		l.matrix_free(&z)
		l.matrix_free(&lx)
	}
	for r in 0 ..< T {
		i := r + k_ar_diff // index into the differences d_i = x_{i+1} - x_i
		for v in 0 ..< k {
			dx.data[r * k + v] = x.data[(i + 1) * k + v] - x.data[i * k + v]
			lx.data[r * k + v] = x.data[(r + 1) * k + v]
		}
		for j in 1 ..= k_ar_diff {
			for v in 0 ..< k {
				z.data[r * q + (j - 1) * k + v] = x.data[(i - j + 1) * k + v] - x.data[(i - j) * k + v]
			}
		}
	}
	_detrend_cols(&z, f)
	_detrend_cols(&dx, f)
	_detrend_cols(&lx, f)
	r0 := _resid_on(&dx, &z)
	rk := _resid_on(&lx, &z)
	defer {
		l.matrix_free(&r0)
		l.matrix_free(&rk)
	}

	skk := _crossprod(&rk, &rk)
	sk0 := _crossprod(&rk, &r0)
	s00 := _crossprod(&r0, &r0)
	defer {
		l.matrix_free(&skk)
		l.matrix_free(&sk0)
		l.matrix_free(&s00)
	}
	// sig = S_k0 S_00⁻¹ S_0k
	s00i := l.matrix_new(f64, k, k, context.temp_allocator)
	defer l.matrix_free(&s00i)
	ml._ols_inverse(&s00, &s00i)
	sig := l.matrix_new(f64, k, k, context.temp_allocator)
	defer l.matrix_free(&sig)
	for a in 0 ..< k {
		for b in 0 ..< k {
			s := 0.0
			for c in 0 ..< k {
				for d in 0 ..< k {s += sk0.data[a * k + c] * s00i.data[c * k + d] * sk0.data[b * k + d]}
			}
			sig.data[a * k + b] = s
		}
	}
	// S_kk = L Lᵀ:  C = L⁻¹ sig L⁻ᵀ symmetric, eigenvalues λ, v = L⁻ᵀ w
	L := l.matrix_new(f64, k, k, context.temp_allocator)
	defer l.matrix_free(&L)
	copy(L.data, skk.data)
	if !l.cholesky_try(&L) do panic("johansen_test: S_kk is not positive definite (collinear series?)")
	Li := _lower_inverse(&L)
	defer l.matrix_free(&Li)
	C := l.matrix_new(f64, k, k, context.temp_allocator)
	defer l.matrix_free(&C)
	for a in 0 ..< k {
		for b in 0 ..< k {
			s := 0.0
			for c in 0 ..< k {
				for d in 0 ..< k {s += Li.data[a * k + c] * sig.data[c * k + d] * Li.data[b * k + d]}
			}
			C.data[a * k + b] = s
		}
	}
	for a in 0 ..< k {
		for b in a + 1 ..< k {
			avg := 0.5 * (C.data[a * k + b] + C.data[b * k + a])
			C.data[a * k + b], C.data[b * k + a] = avg, avg
		}
	}
	W, V := l.eigh(&C, .Descending, context.temp_allocator)
	defer {
		delete(W, context.temp_allocator)
		l.matrix_free(&V)
	}

	res := JohansenResult {
		allocator = allocator,
		nobs      = T,
	}
	res.eig = make([]f64, k, allocator)
	res.evec = l.matrix_new(f64, k, k, allocator)
	res.trace_stat = make([]f64, k, allocator)
	res.max_eig_stat = make([]f64, k, allocator)
	res.trace_crit = make([][3]f64, k, allocator)
	res.max_eig_crit = make([][3]f64, k, allocator)
	for i in 0 ..< k {res.eig[i] = W[i]}
	// v = L⁻ᵀ w
	for col in 0 ..< k {
		for a in 0 ..< k {
			s := 0.0
			for c in 0 ..< k {s += Li.data[c * k + a] * V.data[c * k + col]}
			res.evec.data[a * k + col] = s
		}
		first := 0.0
		for a in 0 ..< k {if res.evec.data[a * k + col] != 0 {first = res.evec.data[a * k + col]; break}}
		if first < 0 {for a in 0 ..< k {res.evec.data[a * k + col] = -res.evec.data[a * k + col]}}
	}
	Tf := f64(T)
	res.rank_trace, res.rank_max_eig = -1, -1
	for i in 0 ..< k {
		s := 0.0
		for j in i ..< k {s += math.ln(1 - res.eig[j])}
		res.trace_stat[i] = -Tf * s
		res.max_eig_stat[i] = -Tf * math.ln(1 - res.eig[i])
		nn := k - i
		if nn <= 12 {
			res.trace_crit[i] = _JOH_TRACE[det_order + 1][nn - 1]
			res.max_eig_crit[i] = _JOH_MAXEIG[det_order + 1][nn - 1]
		} else {
			nan := math.nan_f64()
			res.trace_crit[i], res.max_eig_crit[i] = {nan, nan, nan}, {nan, nan, nan}
		}
		// sequential tests: the rank is the first r whose H0 is not rejected at 5 %
		if res.rank_trace < 0 && !(res.trace_stat[i] > res.trace_crit[i][1]) do res.rank_trace = i
		if res.rank_max_eig < 0 && !(res.max_eig_stat[i] > res.max_eig_crit[i][1]) do res.rank_max_eig = i
	}
	if res.rank_trace < 0 do res.rank_trace = k
	if res.rank_max_eig < 0 do res.rank_max_eig = k
	return res
}

johansen_free :: proc(r: ^JohansenResult) {
	a := r.allocator
	delete(r.eig, a)
	l.matrix_free(&r.evec)
	delete(r.trace_stat, a)
	delete(r.max_eig_stat, a)
	delete(r.trace_crit, a)
	delete(r.max_eig_crit, a)
}

// ----------------------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------------------

// beta = least squares of y on X, resid = y - X beta
_lstsq_resid :: proc(X: ^l.Matrix(f64), y: []f64, beta: []f64, resid: []f64) {
	n, P := X.rows, X.cols
	G := l.matrix_new(f64, P, P, context.temp_allocator)
	Gi := l.matrix_new(f64, P, P, context.temp_allocator)
	defer {
		l.matrix_free(&G)
		l.matrix_free(&Gi)
	}
	xy := make([]f64, P, context.temp_allocator)
	defer delete(xy, context.temp_allocator)
	for a in 0 ..< P {
		for b in 0 ..< P {
			s := 0.0
			for t in 0 ..< n {s += X.data[t * P + a] * X.data[t * P + b]}
			G.data[a * P + b] = s
		}
		s := 0.0
		for t in 0 ..< n {s += X.data[t * P + a] * y[t]}
		xy[a] = s
	}
	ml._ols_inverse(&G, &Gi)
	for a in 0 ..< P {
		s := 0.0
		for b in 0 ..< P {s += Gi.data[a * P + b] * xy[b]}
		beta[a] = s
	}
	for t in 0 ..< n {
		f := 0.0
		for a in 0 ..< P {f += X.data[t * P + a] * beta[a]}
		resid[t] = y[t] - f
	}
}

// Y - Z (ZᵀZ)⁺ ZᵀY (Y unchanged if Z has no columns)
_resid_on :: proc(Y: ^l.Matrix(f64), Z: ^l.Matrix(f64)) -> l.Matrix(f64) {
	T, m, q := Y.rows, Y.cols, Z.cols
	out := l.matrix_new(f64, T, m, context.temp_allocator)
	copy(out.data, Y.data)
	if q == 0 do return out
	G := _crossprod(Z, Z)
	Gi := l.matrix_new(f64, q, q, context.temp_allocator)
	defer {
		l.matrix_free(&G)
		l.matrix_free(&Gi)
	}
	// _crossprod divides by T: undo for the projection
	for &v in G.data {v *= f64(T)}
	ml._ols_inverse(&G, &Gi)
	ZtY := make([]f64, q * m, context.temp_allocator)
	B := make([]f64, q * m, context.temp_allocator)
	defer {
		delete(ZtY, context.temp_allocator)
		delete(B, context.temp_allocator)
	}
	for a in 0 ..< q {
		for c in 0 ..< m {
			s := 0.0
			for t in 0 ..< T {s += Z.data[t * q + a] * Y.data[t * m + c]}
			ZtY[a * m + c] = s
		}
	}
	for a in 0 ..< q {
		for c in 0 ..< m {
			s := 0.0
			for b in 0 ..< q {s += Gi.data[a * q + b] * ZtY[b * m + c]}
			B[a * m + c] = s
		}
	}
	for t in 0 ..< T {
		for c in 0 ..< m {
			s := 0.0
			for a in 0 ..< q {s += Z.data[t * q + a] * B[a * m + c]}
			out.data[t * m + c] -= s
		}
	}
	return out
}

// Residuals of every column on a polynomial in t ∈ linspace(-1, 1, T) of degree
// `order` (statsmodels detrend in coint_johansen); order -1 leaves Y unchanged
_detrend_cols :: proc(Y: ^l.Matrix(f64), order: int) {
	if order < 0 || Y.cols == 0 do return
	T := Y.rows
	V := l.matrix_new(f64, T, order + 1, context.temp_allocator)
	defer l.matrix_free(&V)
	for t in 0 ..< T {
		u := -1 + 2 * f64(t) / f64(T - 1) if T > 1 else 0
		for d in 0 ..= order {V.data[t * (order + 1) + d] = math.pow(u, f64(order - d))}
	}
	R := _resid_on(Y, &V)
	copy(Y.data, R.data)
	l.matrix_free(&R)
}

// AᵀB / T
_crossprod :: proc(A: ^l.Matrix(f64), B: ^l.Matrix(f64)) -> l.Matrix(f64) {
	T, p, q := A.rows, A.cols, B.cols
	out := l.matrix_new(f64, p, q, context.temp_allocator)
	for a in 0 ..< p {
		for b in 0 ..< q {
			s := 0.0
			for t in 0 ..< T {s += A.data[t * p + a] * B.data[t * q + b]}
			out.data[a * q + b] = s / f64(T)
		}
	}
	return out
}

// Inverse of a lower-triangular matrix (forward substitution)
_lower_inverse :: proc(L: ^l.Matrix(f64)) -> l.Matrix(f64) {
	k := L.rows
	out := l.matrix_new(f64, k, k, context.temp_allocator)
	for col in 0 ..< k {
		for i in 0 ..< k {
			s := 1.0 if i == col else 0.0
			for j in 0 ..< i {s -= L.data[i * k + j] * out.data[j * k + col]}
			out.data[i * k + col] = s / L.data[i * k + i]
		}
	}
	return out
}
