package analytics

import l "../linalg"
import ml "./ML"
import "core:math"

// ============================================================================
// Regression residual diagnostics (statsmodels.stats.diagnostic / stattools)
// ============================================================================
// Every LM test returns the Lagrange-multiplier statistic with its χ² p-value
// and the F version of the same auxiliary regression with its p-value.

LMTestResult :: struct {
	lm:        f64, // LM statistic
	lm_pvalue: f64, // χ²(df) tail probability
	f_stat:    f64, // F statistic of the auxiliary regression
	f_pvalue:  f64,
	df:        f64, // degrees of freedom of the LM statistic
}

// Durbin-Watson statistic Σ (e_t - e_{t-1})² / Σ e_t² (≈ 2(1 - ρ̂₁): 2 = no
// first-order autocorrelation, < 2 positive, > 2 negative); statsmodels durbin_watson
durbin_watson :: proc(resid: []f64) -> f64 {
	num, den := 0.0, 0.0
	for e, t in resid {
		den += e * e
		if t > 0 {
			d := e - resid[t - 1]
			num += d * d
		}
	}
	return num / den
}

// Engle's ARCH-LM test (statsmodels het_arch): regress e_t² on a constant and
// e²_{t-1} … e²_{t-q}; LM = (T - ddof) R² ~ χ²(q), T = n - q. H0: no ARCH effects
// (a small p-value suggests volatility clustering, i.e. a GARCH model).
// nlags < 0: min(10, n / 5).
arch_lm_test :: proc(resid: []f64, nlags: int = -1, ddof: int = 0) -> LMTestResult {
	n := len(resid)
	sq := make([]f64, n, context.temp_allocator)
	defer delete(sq, context.temp_allocator)
	for e, t in resid {sq[t] = e * e}
	q := nlags if nlags >= 0 else min(10, n / 5)
	if q < 1 do panic("arch_lm_test: need at least one lag")
	T := n - q
	if T <= q + 1 do panic("arch_lm_test: too few observations for nlags")
	Z := l.matrix_new(f64, T, q + 1, context.temp_allocator)
	defer l.matrix_free(&Z)
	y := make([]f64, T, context.temp_allocator)
	defer delete(y, context.temp_allocator)
	for r in 0 ..< T {
		t := r + q
		y[r] = sq[t]
		Z.data[r * (q + 1)] = 1
		for j in 1 ..= q {Z.data[r * (q + 1) + j] = sq[t - j]}
	}
	res := ml.ols_fit(&Z, y, allocator = context.temp_allocator)
	defer ml._ols_result_free(&res, context.temp_allocator)
	lm := f64(T - ddof) * res.r2
	return {lm = lm, lm_pvalue = chi2_sf(lm, f64(q)), f_stat = res.fstat, f_pvalue = res.f_pvalue, df = f64(q)}
}

// Breusch-Pagan test for heteroskedasticity (statsmodels het_breuschpagan):
// regress e² on exog_het (which must contain a constant column).
//   robust = true (default, Koenker's studentised version): LM = n R²
//   robust = false (original): LM = ESS / 2 of e² / mean(e²) on exog_het
// LM ~ χ²(k - 1). H0: homoskedastic errors.
breusch_pagan_test :: proc(resid: []f64, exog_het: ^l.Matrix(f64), robust := true) -> LMTestResult {
	n := len(resid)
	if exog_het.rows != n do panic("breusch_pagan_test: exog_het needs one row per residual")
	y := make([]f64, n, context.temp_allocator)
	defer delete(y, context.temp_allocator)
	mean := 0.0
	for e, t in resid {
		y[t] = e * e
		mean += y[t]
	}
	mean /= f64(n)
	if !robust {for &v in y {v /= mean}}
	res := ml.ols_fit(exog_het, y, allocator = context.temp_allocator)
	defer ml._ols_result_free(&res, context.temp_allocator)
	lm := f64(n) * res.r2
	if !robust {
		// ESS = Σ (ŷ - ȳ)²
		ybar := 0.0
		for v in y {ybar += v}
		ybar /= f64(n)
		ess := 0.0
		for v in res.fitted {ess += (v - ybar) * (v - ybar)}
		lm = ess / 2
	}
	df := f64(exog_het.cols - 1)
	return {lm = lm, lm_pvalue = chi2_sf(lm, df), f_stat = res.fstat, f_pvalue = res.f_pvalue, df = df}
}

// White's test for heteroskedasticity (statsmodels het_white): regress e² on all
// products x_i x_j (i ≤ j) of the columns of exog (which must contain a constant,
// so the levels and squares are included); LM = n R² ~ χ²(rank - 1).
white_test :: proc(resid: []f64, exog: ^l.Matrix(f64)) -> LMTestResult {
	n, k := exog.rows, exog.cols
	if n != len(resid) do panic("white_test: exog needs one row per residual")
	if k < 2 do panic("white_test: exog needs a constant and at least one regressor")
	m := k * (k + 1) / 2
	Z := l.matrix_new(f64, n, m, context.temp_allocator)
	defer l.matrix_free(&Z)
	for t in 0 ..< n {
		c := 0
		for i in 0 ..< k {
			for j in i ..< k {
				Z.data[t * m + c] = exog.data[t * k + i] * exog.data[t * k + j]
				c += 1
			}
		}
	}
	y := make([]f64, n, context.temp_allocator)
	defer delete(y, context.temp_allocator)
	for e, t in resid {y[t] = e * e}
	res := ml.ols_fit(&Z, y, allocator = context.temp_allocator)
	defer ml._ols_result_free(&res, context.temp_allocator)
	lm := f64(n) * res.r2
	df := f64(res.rank - 1) // products can be collinear (e.g. dummies): use the rank
	return {lm = lm, lm_pvalue = chi2_sf(lm, df), f_stat = res.fstat, f_pvalue = res.f_pvalue, df = df}
}

// Breusch-Godfrey test for serial correlation up to order q (statsmodels
// acorr_breusch_godfrey): regress e_t on the original regressors exog (may be
// nil), a constant and e_{t-1} … e_{t-q} (pre-sample residuals set to 0);
// LM = n R² ~ χ²(q), F = classical F test of the q lag coefficients.
// nlags < 0: min(10, n / 5). Valid with lagged dependent variables, unlike DW.
breusch_godfrey_test :: proc(resid: []f64, exog: ^l.Matrix(f64), nlags: int = -1) -> LMTestResult {
	n := len(resid)
	q := nlags if nlags >= 0 else min(10, n / 5)
	if q < 1 do panic("breusch_godfrey_test: need at least one lag")
	k0 := 0
	if exog != nil {
		if exog.rows != n do panic("breusch_godfrey_test: exog needs one row per residual")
		k0 = exog.cols
	}
	P := k0 + 1 + q
	if n <= P do panic("breusch_godfrey_test: too few observations")
	Z := l.matrix_new(f64, n, P, context.temp_allocator)
	defer l.matrix_free(&Z)
	for t in 0 ..< n {
		for j in 0 ..< k0 {Z.data[t * P + j] = exog.data[t * k0 + j]}
		Z.data[t * P + k0] = 1
		for j in 1 ..= min(q, t) {Z.data[t * P + k0 + j] = resid[t - j]} // pre-sample lags stay 0
	}
	res := ml.ols_fit(&Z, resid, allocator = context.temp_allocator)
	defer ml._ols_result_free(&res, context.temp_allocator)
	lm := f64(n) * res.r2

	// F = (Rb)ᵀ (R V Rᵀ)⁻¹ (Rb) / q for the last q coefficients
	M := l.matrix_new(f64, q, q, context.temp_allocator)
	Mi := l.matrix_new(f64, q, q, context.temp_allocator)
	defer {
		l.matrix_free(&M)
		l.matrix_free(&Mi)
	}
	off := P - q
	for a in 0 ..< q {
		for b in 0 ..< q {M.data[a * q + b] = res.vcov.data[(off + a) * P + off + b]}
	}
	ml._ols_inverse(&M, &Mi)
	w := 0.0
	for a in 0 ..< q {
		for b in 0 ..< q {w += res.beta[off + a] * Mi.data[a * q + b] * res.beta[off + b]}
	}
	F := w / f64(q)
	return {lm = lm, lm_pvalue = chi2_sf(lm, f64(q)), f_stat = F, f_pvalue = ml.f_dist_sf(F, f64(q), f64(res.df_resid)), df = f64(q)}
}
