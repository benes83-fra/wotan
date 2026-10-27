package analytics

import l "../linalg"
import ml "./ML"
import "core:math"
import "core:mem"

// ============================================================================
// Vector autoregression VAR(p) with intercept (statsmodels VAR(endog).fit(p,
// trend="c")), estimated equation by equation by OLS:
//   y_t = c + A_1 y_{t-1} + … + A_p y_{t-p} + u_t,   u_t ~ (0, Σ_u)
// Data: Y is n × k (one column per variable, rows in time order).
// ============================================================================

VARResult :: struct {
	k, p:        int, // variables, lags
	nobs:        int, // effective observations n - p
	df_resid:    int, // nobs - (k·p + 1)
	intercept:   []f64, // c (k)
	coefs:       []l.Matrix(f64), // A_1 … A_p (k × k); A_j[i, m] = effect of variable m at lag j on equation i
	params:      l.Matrix(f64), // (1 + k·p) × k, statsmodels layout: rows const, lag-1 vars, lag-2 vars, …; column = equation
	stderr:      l.Matrix(f64), // standard errors of params, sqrt((ZᵀZ)⁻¹_rr · Σ_u,cc)
	sigma_u:     l.Matrix(f64), // residual covariance, divided by df_resid
	sigma_u_mle: l.Matrix(f64), // residual covariance, divided by nobs
	resid:       l.Matrix(f64), // nobs × k
	ZtZ_inv:     l.Matrix(f64), // (1 + k·p)²
	y_last:      l.Matrix(f64), // last p observations (p × k), the start of var_forecast
	llf:         f64, // Gaussian log-likelihood at Σ_u,mle
	aic, bic, hqic, fpe: f64, // statsmodels info_criteria
	allocator:   mem.Allocator,
}

// Fit a VAR(p) with intercept on all rows of Y (the first p rows are only lags).
var_fit :: proc(Y: ^l.Matrix(f64), p: int, allocator: mem.Allocator = context.allocator) -> VARResult {
	if p < 0 do panic("var_fit: p must be >= 0")
	return _var_estimate(Y, p, 0, allocator)
}

// Lag order chosen by each information criterion over p = 0 … max_lags, every
// model fitted on the same sample (the last n - max_lags rows), as statsmodels
// VAR.select_order. aic / bic / hqic / fpe[p] hold the criterion values.
VARLagSelection :: struct {
	aic_lag, bic_lag, hqic_lag, fpe_lag: int,
	aic, bic, hqic, fpe:                 []f64,
	allocator:                           mem.Allocator,
}

var_select_order :: proc(Y: ^l.Matrix(f64), max_lags: int, allocator: mem.Allocator = context.allocator) -> VARLagSelection {
	if max_lags < 0 do panic("var_select_order: max_lags must be >= 0")
	s := VARLagSelection {
		aic       = make([]f64, max_lags + 1, allocator),
		bic       = make([]f64, max_lags + 1, allocator),
		hqic      = make([]f64, max_lags + 1, allocator),
		fpe       = make([]f64, max_lags + 1, allocator),
		allocator = allocator,
	}
	for p in 0 ..= max_lags {
		r := _var_estimate(Y, p, max_lags - p, context.temp_allocator)
		s.aic[p], s.bic[p], s.hqic[p], s.fpe[p] = r.aic, r.bic, r.hqic, r.fpe
		var_free(&r)
	}
	argmin :: proc(v: []f64) -> int {
		b := 0
		for x, i in v {if x < v[b] do b = i}
		return b
	}
	s.aic_lag, s.bic_lag, s.hqic_lag, s.fpe_lag = argmin(s.aic), argmin(s.bic), argmin(s.hqic), argmin(s.fpe)
	return s
}

var_lag_selection_free :: proc(s: ^VARLagSelection) {
	delete(s.aic, s.allocator)
	delete(s.bic, s.allocator)
	delete(s.hqic, s.allocator)
	delete(s.fpe, s.allocator)
}

// h-step point forecasts (steps × k) from the end of the fitted sample
var_forecast :: proc(res: ^VARResult, steps: int, allocator: mem.Allocator = context.allocator) -> l.Matrix(f64) {
	k, p := res.k, res.p
	out := l.matrix_new(f64, steps, k, allocator)
	// history: the last p observations followed by the forecasts
	hist := make([]f64, (p + steps) * k, context.temp_allocator)
	defer delete(hist, context.temp_allocator)
	copy(hist[:p * k], res.y_last.data)
	for h in 0 ..< steps {
		t := p + h
		for i in 0 ..< k {
			v := res.intercept[i]
			for j in 1 ..= p {
				A := &res.coefs[j - 1]
				for m in 0 ..< k {v += A.data[i * k + m] * hist[(t - j) * k + m]}
			}
			hist[t * k + i] = v
			out.data[h * k + i] = v
		}
	}
	return out
}

// Forecasts with (1 - alpha) intervals point ± z·sqrt(diag MSE(h)),
// MSE(h) = Σ_{i<h} Φ_i Σ_u Φ_iᵀ (statsmodels forecast_interval; no
// parameter-uncertainty term)
var_forecast_interval :: proc(
	res: ^VARResult,
	steps: int,
	alpha: f64 = 0.05,
	allocator: mem.Allocator = context.allocator,
) -> (
	point, lower, upper: l.Matrix(f64),
) {
	k := res.k
	point = var_forecast(res, steps, allocator)
	lower = l.matrix_new(f64, steps, k, allocator)
	upper = l.matrix_new(f64, steps, k, allocator)
	phi := var_irf(res, max(steps - 1, 0), false, context.temp_allocator)
	defer _free_matrices(phi, context.temp_allocator)
	z := _arima_norm_ppf(1 - alpha / 2)
	mse := make([]f64, k * k, context.temp_allocator)
	tmp := make([]f64, k * k, context.temp_allocator)
	defer {
		delete(mse, context.temp_allocator)
		delete(tmp, context.temp_allocator)
	}
	for h in 0 ..< steps {
		// mse += Φ_h Σ_u Φ_hᵀ
		F := &phi[h]
		for i in 0 ..< k {
			for j in 0 ..< k {
				s := 0.0
				for m in 0 ..< k {s += F.data[i * k + m] * res.sigma_u.data[m * k + j]}
				tmp[i * k + j] = s
			}
		}
		for i in 0 ..< k {
			for j in 0 ..< k {
				s := 0.0
				for m in 0 ..< k {s += tmp[i * k + m] * F.data[j * k + m]}
				mse[i * k + j] += s
			}
		}
		for i in 0 ..< k {
			sd := math.sqrt(mse[i * k + i])
			lower.data[h * k + i] = point.data[h * k + i] - z * sd
			upper.data[h * k + i] = point.data[h * k + i] + z * sd
		}
	}
	return
}

// Impulse responses Φ_0 = I, Φ_h = Σ_{j=1}^{min(h,p)} A_j Φ_{h-j} (the MA
// representation, statsmodels irf().irfs), or orthogonalised Φ_h P with
// P = chol(Σ_u) lower (irf().orth_irfs). Result: periods + 1 matrices k × k,
// entry [i, m] = response of variable i to a shock in variable m.
var_irf :: proc(
	res: ^VARResult,
	periods: int,
	orthogonalized := false,
	allocator: mem.Allocator = context.allocator,
) -> []l.Matrix(f64) {
	k, p := res.k, res.p
	phi := make([]l.Matrix(f64), periods + 1, allocator)
	for h in 0 ..= periods {
		phi[h] = l.matrix_new(f64, k, k, allocator)
		if h == 0 {
			for i in 0 ..< k {phi[0].data[i * k + i] = 1}
			continue
		}
		for j in 1 ..= min(h, p) {
			A := &res.coefs[j - 1]
			B := &phi[h - j]
			for i in 0 ..< k {
				for c in 0 ..< k {
					s := 0.0
					for m in 0 ..< k {s += A.data[i * k + m] * B.data[m * k + c]}
					phi[h].data[i * k + c] += s
				}
			}
		}
	}
	if orthogonalized {
		P := l.matrix_new(f64, k, k, context.temp_allocator)
		defer l.matrix_free(&P)
		copy(P.data, res.sigma_u.data)
		if !l.cholesky_try(&P) do panic("var_irf: Σ_u is not positive definite")
		row := make([]f64, k, context.temp_allocator)
		defer delete(row, context.temp_allocator)
		for h in 0 ..= periods {
			for i in 0 ..< k {
				for c in 0 ..< k {
					s := 0.0
					for m in 0 ..< k {s += phi[h].data[i * k + m] * P.data[m * k + c]}
					row[c] = s
				}
				copy(phi[h].data[i * k:(i + 1) * k], row)
			}
		}
	}
	return phi
}

// Stability: all eigenvalues of the companion matrix inside the unit circle
// (statsmodels is_stable). Returns the largest modulus as well.
var_is_stable :: proc(res: ^VARResult) -> (stable: bool, max_modulus: f64) {
	k, p := res.k, res.p
	if p == 0 do return true, 0
	K := k * p
	Cm := l.matrix_new(f64, K, K, context.temp_allocator)
	defer l.matrix_free(&Cm)
	for j in 0 ..< p {
		for i in 0 ..< k {
			for m in 0 ..< k {Cm.data[i * K + j * k + m] = res.coefs[j].data[i * k + m]}
		}
	}
	for i in k ..< K {Cm.data[i * K + i - k] = 1}
	max_modulus = l.eigvals_modulus_max(&Cm)
	return max_modulus < 1, max_modulus
}

// Granger causality inside the VAR (statsmodels test_causality): H0: the lags of
// the `causing` variables have zero coefficients in the equations of `caused`.
// Wald statistic λ = (Cb)ᵀ (C V Cᵀ)⁻¹ (Cb), V = (ZᵀZ)⁻¹ ⊗ Σ_u;
// kind .F: λ / q ~ F(q, k·df_resid); .Wald: λ ~ χ²(q), q = #restrictions.
VARCausalityKind :: enum {
	F,
	Wald,
}

VARCausalityResult :: struct {
	statistic: f64,
	p_value:   f64,
	df_num:    f64,
	df_den:    f64, // F test only (else NaN)
}

var_test_causality :: proc(res: ^VARResult, caused: []int, causing: []int, kind: VARCausalityKind = .F) -> VARCausalityResult {
	k, p := res.k, res.p
	if p == 0 do panic("var_test_causality: VAR(0) has no lags")
	for v in caused {if v < 0 || v >= k do panic("var_test_causality: variable index out of range")}
	for v in causing {if v < 0 || v >= k do panic("var_test_causality: variable index out of range")}
	q := len(caused) * len(causing) * p
	// restricted coefficients: (param row r, equation e)
	rows := make([]int, q, context.temp_allocator)
	eqs := make([]int, q, context.temp_allocator)
	defer {
		delete(rows, context.temp_allocator)
		delete(eqs, context.temp_allocator)
	}
	n := 0
	for j in 0 ..< p {
		for ing in causing {
			for ed in caused {
				rows[n], eqs[n] = 1 + ing + k * j, ed
				n += 1
			}
		}
	}
	// Cb and C V Cᵀ (V = (ZᵀZ)⁻¹ ⊗ Σ_u: cov(b_re, b_r'e') = (ZᵀZ)⁻¹_rr' Σ_u,ee')
	P := res.ZtZ_inv.cols
	M := l.matrix_new(f64, q, q, context.temp_allocator)
	Minv := l.matrix_new(f64, q, q, context.temp_allocator)
	defer {
		l.matrix_free(&M)
		l.matrix_free(&Minv)
	}
	for a in 0 ..< q {
		for b in 0 ..< q {
			M.data[a * q + b] = res.ZtZ_inv.data[rows[a] * P + rows[b]] * res.sigma_u.data[eqs[a] * k + eqs[b]]
		}
	}
	ml._ols_inverse(&M, &Minv)
	lam := 0.0
	for a in 0 ..< q {
		ca := res.params.data[rows[a] * k + eqs[a]]
		for b in 0 ..< q {
			lam += ca * Minv.data[a * q + b] * res.params.data[rows[b] * k + eqs[b]]
		}
	}
	if kind == .Wald {
		return {statistic = lam, p_value = chi2_sf(lam, f64(q)), df_num = f64(q), df_den = math.nan_f64()}
	}
	d1, d2 := f64(q), f64(k * res.df_resid)
	F := lam / d1
	return {statistic = F, p_value = ml.f_dist_sf(F, d1, d2), df_num = d1, df_den = d2}
}

var_free :: proc(res: ^VARResult) {
	a := res.allocator
	delete(res.intercept, a)
	for &m in res.coefs {l.matrix_free(&m)}
	delete(res.coefs, a)
	l.matrix_free(&res.params)
	l.matrix_free(&res.stderr)
	l.matrix_free(&res.sigma_u)
	l.matrix_free(&res.sigma_u_mle)
	l.matrix_free(&res.resid)
	l.matrix_free(&res.ZtZ_inv)
	l.matrix_free(&res.y_last)
}

var_irf_free :: proc(phi: []l.Matrix(f64), allocator: mem.Allocator = context.allocator) {
	_free_matrices(phi, allocator)
}

// ============================================================================
// Bivariate Granger causality (statsmodels grangercausalitytests): does x help
// to predict y? For every lag order 1 … max_lag, OLS of y_t on a constant and
// y_{t-1..t-p} (restricted) and additionally x_{t-1..t-p} (unrestricted), on the
// rows t = p … n-1.
// ============================================================================

GrangerLagResult :: struct {
	lag:                 int,
	f_stat, f_pvalue:    f64, // ssr_ftest: ((SSR_r - SSR_u)/p) / (SSR_u/(T - 2p - 1)) ~ F(p, T - 2p - 1)
	df_den:              f64,
	chi2_stat, chi2_pvalue: f64, // ssr_chi2test: T (SSR_r - SSR_u) / SSR_u ~ χ²(p)
	lr_stat, lr_pvalue:  f64, // lrtest: 2 (llf_u - llf_r) ~ χ²(p)
}

granger_causality_test :: proc(
	y: []f64,
	x: []f64,
	max_lag: int,
	allocator: mem.Allocator = context.allocator,
) -> []GrangerLagResult {
	n := len(y)
	if len(x) != n do panic("granger_causality_test: y and x need the same length")
	if max_lag < 1 do panic("granger_causality_test: max_lag must be >= 1")
	if n - 3 * max_lag - 1 <= 0 do panic("granger_causality_test: series too short for max_lag")
	out := make([]GrangerLagResult, max_lag, allocator)
	for p in 1 ..= max_lag {
		T := n - p
		Zr := l.matrix_new(f64, T, 1 + p, context.temp_allocator)
		Zu := l.matrix_new(f64, T, 1 + 2 * p, context.temp_allocator)
		yt := make([]f64, T, context.temp_allocator)
		for t in p ..< n {
			r := t - p
			yt[r] = y[t]
			Zr.data[r * (1 + p)] = 1
			Zu.data[r * (1 + 2 * p)] = 1
			for j in 1 ..= p {
				Zr.data[r * (1 + p) + j] = y[t - j]
				Zu.data[r * (1 + 2 * p) + j] = y[t - j]
				Zu.data[r * (1 + 2 * p) + p + j] = x[t - j]
			}
		}
		ssr_r := _ols_ssr(&Zr, yt)
		ssr_u := _ols_ssr(&Zu, yt)
		df_u := f64(T - 2 * p - 1)
		res := GrangerLagResult {
			lag    = p,
			df_den = df_u,
		}
		res.f_stat = ((ssr_r - ssr_u) / f64(p)) / (ssr_u / df_u)
		res.f_pvalue = ml.f_dist_sf(res.f_stat, f64(p), df_u)
		res.chi2_stat = f64(T) * (ssr_r - ssr_u) / ssr_u
		res.chi2_pvalue = chi2_sf(res.chi2_stat, f64(p))
		// llf = -T/2 (log(2π SSR/T) + 1)  →  2 (llf_u - llf_r) = T log(SSR_r / SSR_u)
		res.lr_stat = f64(T) * math.ln(ssr_r / ssr_u)
		res.lr_pvalue = chi2_sf(res.lr_stat, f64(p))
		out[p - 1] = res
		l.matrix_free(&Zr)
		l.matrix_free(&Zu)
		delete(yt, context.temp_allocator)
	}
	return out
}

// ============================================================================
// Internal
// ============================================================================

// VAR(p) on rows offset … n-1 of Y (offset lets select_order use a common sample)
_var_estimate :: proc(Y: ^l.Matrix(f64), p: int, offset: int, allocator: mem.Allocator) -> VARResult {
	n_all, k := Y.rows, Y.cols
	n := n_all - offset
	T := n - p
	P := 1 + k * p
	if k < 1 do panic("var_fit: Y has no columns")
	if T <= P do panic("var_fit: too few observations for this lag order")
	base := offset // first usable row of Y

	Z := l.matrix_new(f64, T, P, context.temp_allocator)
	Yt := l.matrix_new(f64, T, k, context.temp_allocator)
	defer {
		l.matrix_free(&Z)
		l.matrix_free(&Yt)
	}
	for r in 0 ..< T {
		t := base + p + r
		Z.data[r * P] = 1
		for j in 1 ..= p {
			for m in 0 ..< k {Z.data[r * P + 1 + (j - 1) * k + m] = Y.data[(t - j) * k + m]}
		}
		for m in 0 ..< k {Yt.data[r * k + m] = Y.data[t * k + m]}
	}

	res: VARResult
	res.allocator = allocator
	res.k, res.p, res.nobs = k, p, T
	res.df_resid = T - P

	// (ZᵀZ)⁻¹ and B = (ZᵀZ)⁻¹ ZᵀY
	ZtZ := l.matrix_new(f64, P, P, context.temp_allocator)
	defer l.matrix_free(&ZtZ)
	for a in 0 ..< P {
		for b in a ..< P {
			s := 0.0
			for r in 0 ..< T {s += Z.data[r * P + a] * Z.data[r * P + b]}
			ZtZ.data[a * P + b] = s
			ZtZ.data[b * P + a] = s
		}
	}
	res.ZtZ_inv = l.matrix_new(f64, P, P, allocator)
	ml._ols_inverse(&ZtZ, &res.ZtZ_inv)
	ZtY := make([]f64, P * k, context.temp_allocator)
	defer delete(ZtY, context.temp_allocator)
	for a in 0 ..< P {
		for m in 0 ..< k {
			s := 0.0
			for r in 0 ..< T {s += Z.data[r * P + a] * Yt.data[r * k + m]}
			ZtY[a * k + m] = s
		}
	}
	res.params = l.matrix_new(f64, P, k, allocator)
	for a in 0 ..< P {
		for m in 0 ..< k {
			s := 0.0
			for b in 0 ..< P {s += res.ZtZ_inv.data[a * P + b] * ZtY[b * k + m]}
			res.params.data[a * k + m] = s
		}
	}

	// residuals and covariances
	res.resid = l.matrix_new(f64, T, k, allocator)
	for r in 0 ..< T {
		for m in 0 ..< k {
			f := 0.0
			for a in 0 ..< P {f += Z.data[r * P + a] * res.params.data[a * k + m]}
			res.resid.data[r * k + m] = Yt.data[r * k + m] - f
		}
	}
	res.sigma_u = l.matrix_new(f64, k, k, allocator)
	res.sigma_u_mle = l.matrix_new(f64, k, k, allocator)
	for i in 0 ..< k {
		for j in 0 ..< k {
			s := 0.0
			for r in 0 ..< T {s += res.resid.data[r * k + i] * res.resid.data[r * k + j]}
			res.sigma_u_mle.data[i * k + j] = s / f64(T)
			res.sigma_u.data[i * k + j] = s / f64(res.df_resid)
		}
	}
	res.stderr = l.matrix_new(f64, P, k, allocator)
	for a in 0 ..< P {
		for m in 0 ..< k {
			res.stderr.data[a * k + m] = math.sqrt(res.ZtZ_inv.data[a * P + a] * res.sigma_u.data[m * k + m])
		}
	}

	// unpacked coefficients
	res.intercept = make([]f64, k, allocator)
	for m in 0 ..< k {res.intercept[m] = res.params.data[m]}
	res.coefs = make([]l.Matrix(f64), p, allocator)
	for j in 0 ..< p {
		res.coefs[j] = l.matrix_new(f64, k, k, allocator)
		for i in 0 ..< k {
			for m in 0 ..< k {res.coefs[j].data[i * k + m] = res.params.data[(1 + j * k + m) * k + i]}
		}
	}
	res.y_last = l.matrix_new(f64, p, k, allocator)
	copy(res.y_last.data, Y.data[(n_all - p) * k:n_all * k])

	// likelihood and information criteria (statsmodels VARResults)
	ld := _logdet_spd(&res.sigma_u_mle)
	Tf, kf := f64(T), f64(k)
	res.llf = -(Tf * kf / 2) * math.ln_f64(2 * math.PI) - Tf / 2 * ld - Tf * kf / 2
	free_params := f64(p * k * k + k)
	df_model := f64(k * p + 1)
	res.aic = ld + (2 / Tf) * free_params
	res.bic = ld + (math.ln(Tf) / Tf) * free_params
	res.hqic = ld + (2 * math.ln(math.ln(Tf)) / Tf) * free_params
	res.fpe = math.pow((Tf + df_model) / (Tf - df_model), kf) * math.exp(ld)
	return res
}

// log det of a symmetric positive definite matrix (via Cholesky); -Inf if singular
_logdet_spd :: proc(A: ^l.Matrix(f64)) -> f64 {
	L := l.matrix_new(f64, A.rows, A.cols, context.temp_allocator)
	defer l.matrix_free(&L)
	copy(L.data, A.data)
	if !l.cholesky_try(&L) do return math.inf_f64(-1)
	s := 0.0
	for i in 0 ..< A.rows {s += 2 * math.ln(L.data[i * A.cols + i])}
	return s
}

// SSR of the OLS fit of y on Z
_ols_ssr :: proc(Z: ^l.Matrix(f64), y: []f64) -> f64 {
	T, P := Z.rows, Z.cols
	G := l.matrix_new(f64, P, P, context.temp_allocator)
	Gi := l.matrix_new(f64, P, P, context.temp_allocator)
	defer {
		l.matrix_free(&G)
		l.matrix_free(&Gi)
	}
	zy := make([]f64, P, context.temp_allocator)
	defer delete(zy, context.temp_allocator)
	for a in 0 ..< P {
		for b in 0 ..< P {
			s := 0.0
			for r in 0 ..< T {s += Z.data[r * P + a] * Z.data[r * P + b]}
			G.data[a * P + b] = s
		}
		s := 0.0
		for r in 0 ..< T {s += Z.data[r * P + a] * y[r]}
		zy[a] = s
	}
	ml._ols_inverse(&G, &Gi)
	ssr := 0.0
	for r in 0 ..< T {
		f := 0.0
		for a in 0 ..< P {
			b := 0.0
			for c in 0 ..< P {b += Gi.data[a * P + c] * zy[c]}
			f += Z.data[r * P + a] * b
		}
		ssr += (y[r] - f) * (y[r] - f)
	}
	return ssr
}

_free_matrices :: proc(ms: []l.Matrix(f64), allocator: mem.Allocator) {
	for &m in ms {l.matrix_free(&m)}
	delete(ms, allocator)
}
