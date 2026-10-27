package analytics

import l "../linalg"
import ml "./ML"
import "base:runtime"
import "core:math"
import "core:mem"

// ============================================================================
// ARIMAX: regression with ARIMA errors (statsmodels SARIMAX(y, exog, order))
//   y_t = x_tᵀ β + u_t,   u_t ~ ARIMA(p, d, q)
// i.e. Δ^d y_t = (Δ^d x_t)ᵀ β + ARMA(p, q) errors. Exact Gaussian MLE on the
// differenced data (as arima_fit; statsmodels' d > 0 approximate-diffuse prior
// differs by ~1e-5 in the log-likelihood).
// β is concentrated out: for given ARMA parameters the GLS estimate follows
// from Kalman-filtering y and every column of X with the same filter, so the
// optimiser only searches over (φ, θ, σ²).
// Add a column of ones to X for a constant (d = 0) and a time trend t for a
// drift (d = 1): columns that vanish after differencing are rejected.
// ============================================================================

ArimaxFitResult :: struct {
	phi:       []f64,
	theta:     []f64,
	beta:      []f64, // regression coefficients (one per column of X)
	sigma2:    f64,
	loglik:    f64,
	aic:       f64, // k = p + q + len(beta) + 1
	bic:       f64,
	converged: bool,
}

arimax_fit :: proc(
	y: []f64,
	X: ^l.Matrix(f64),
	p, d, q: int,
	allocator := context.allocator,
) -> ArimaxFitResult {
	n, k := len(y), X.cols
	if X.rows != n do panic("arimax_fit: X needs one row per observation")
	if k < 1 do panic("arimax_fit: X has no columns (use arima_fit)")
	ctx := _arimax_ctx(y, X, p, d, q)
	defer _arimax_ctx_free(&ctx)

	// start: log variance of the OLS residuals of Δy on ΔX
	beta0 := make([]f64, k, context.temp_allocator)
	resid0 := make([]f64, len(ctx.z), context.temp_allocator)
	defer {
		delete(beta0, context.temp_allocator)
		delete(resid0, context.temp_allocator)
	}
	_lstsq_resid(&ctx.Xd, ctx.z, beta0, resid0)
	var0 := 0.0
	for r in resid0 {var0 += r * r}
	var0 /= f64(max(len(resid0) - k, 1))
	log_sig2_0 := math.ln_f64(var0) if var0 > 0 else 0

	starts := [][4]f64{{0.0, 0.0, 0, 0}, {0.5, 0.0, 0, 0}, {0.0, 0.5, 0, 0}, {0.5, -0.5, 0, 0}, {-0.5, 0.5, 0, 0}}
	best, best_f, converged := _arima_nm_fit(_arimax_obj, &ctx, {p, q}, {false, true}, starts, log_sig2_0, context.temp_allocator)
	defer delete(best, context.temp_allocator)

	res: ArimaxFitResult
	res.phi = make([]f64, p, allocator)
	res.theta = make([]f64, q, allocator)
	res.beta = make([]f64, k, allocator)
	_arima_constrain(best[:p], res.phi, false)
	_arima_constrain(best[p:p + q], res.theta, true)
	res.sigma2 = math.exp_f64(best[p + q])
	ll := _arimax_profile(&ctx, res.phi, res.theta, res.sigma2, res.beta)
	res.loglik = ll
	_ = best_f
	res.converged = converged
	kk := f64(p + q + k + 1)
	res.aic = -2 * res.loglik + 2 * kk
	res.bic = -2 * res.loglik + kk * math.ln(f64(len(ctx.z)))
	return res
}

arimax_free :: proc(fit: ^ArimaxFitResult, allocator := context.allocator) {
	delete(fit.phi, allocator)
	delete(fit.theta, allocator)
	delete(fit.beta, allocator)
}

// h-step forecasts on the original scale given the future regressors X_future
// (h × k): x_{T+j}ᵀ β + the ARIMA forecast of u = y - Xβ, with (1 - alpha)
// intervals (parameter uncertainty in β not included, as statsmodels).
arimax_forecast :: proc(
	y: []f64,
	X: ^l.Matrix(f64),
	fit: ArimaxFitResult,
	d: int,
	X_future: ^l.Matrix(f64),
	alpha: f64 = 0.05,
	allocator := context.allocator,
) -> ArimaForecastResult {
	n, k := len(y), X.cols
	h := X_future.rows
	if X.rows != n || X_future.cols != k || len(fit.beta) != k do panic("arimax_forecast: dimension mismatch")
	if d > 0 && n <= d {
		mean, lower, upper := _nan_forecast(h, allocator)
		return ArimaForecastResult{mean = mean, lower = lower, upper = upper}
	}
	u := make([]f64, n, context.temp_allocator)
	defer delete(u, context.temp_allocator)
	for t in 0 ..< n {
		s := 0.0
		for j in 0 ..< k {s += X.data[t * k + j] * fit.beta[j]}
		u[t] = y[t] - s
	}
	c := _arima_diff_poly(d, 0, 0, context.temp_allocator)
	defer delete(c, context.temp_allocator)
	mean, lower, upper := _arima_forecast_core(u, fit.phi, fit.theta, fit.sigma2, c, h, alpha, allocator)
	for i in 0 ..< h {
		s := 0.0
		for j in 0 ..< k {s += X_future.data[i * k + j] * fit.beta[j]}
		mean[i] += s
		lower[i] += s
		upper[i] += s
	}
	return ArimaForecastResult{mean = mean, lower = lower, upper = upper}
}

// Exact log-likelihood of y - Xβ ~ ARIMA(φ, d, θ, σ²) at given β (no estimation)
arimax_loglik :: proc(y: []f64, X: ^l.Matrix(f64), phi, theta, beta: []f64, d: int, sigma2: f64) -> f64 {
	ctx := _arimax_ctx(y, X, len(phi), d, len(theta))
	defer _arimax_ctx_free(&ctx)
	return _arimax_loglik_at(&ctx, phi, theta, sigma2, beta)
}

// ----------------------------------------------------------------------------
// Internal
// ----------------------------------------------------------------------------

_ArimaxCtx :: struct {
	z:    []f64, // Δ^d y
	Xd:   l.Matrix(f64), // Δ^d X
	p, q: int,
}

_arimax_ctx :: proc(y: []f64, X: ^l.Matrix(f64), p, d, q: int) -> _ArimaxCtx {
	n, k := len(y), X.cols
	c := _arima_diff_poly(d, 0, 0, context.temp_allocator)
	defer delete(c, context.temp_allocator)
	m := len(c) - 1
	T := n - m
	if T <= p + q + k + 2 do panic("arimax_fit: too few observations")
	ctx := _ArimaxCtx {
		z  = make([]f64, T, context.temp_allocator),
		Xd = l.matrix_new(f64, T, k, context.temp_allocator),
		p  = p,
		q  = q,
	}
	for t in m ..< n {
		s := 0.0
		for i in 0 ..= m {s += c[i] * y[t - i]}
		ctx.z[t - m] = s
		for j in 0 ..< k {
			s = 0.0
			for i in 0 ..= m {s += c[i] * X.data[(t - i) * k + j]}
			ctx.Xd.data[(t - m) * k + j] = s
		}
	}
	for j in 0 ..< k {
		zero := true
		for t in 0 ..< T {if ctx.Xd.data[t * k + j] != 0 {zero = false; break}}
		if zero do panic("arimax_fit: a column of X is constant after differencing (drop it, or use a trend for a drift)")
	}
	return ctx
}

_arimax_ctx_free :: proc(ctx: ^_ArimaxCtx) {
	delete(ctx.z, context.temp_allocator)
	l.matrix_free(&ctx.Xd)
}

// Innovations of z and of every column of Xd through the ARMA Kalman filter
// (shared variances S_t); then the GLS β and the log-likelihood at it
_arimax_profile :: proc(ctx: ^_ArimaxCtx, phi, theta: []f64, sigma2: f64, beta_out: []f64) -> f64 {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	T, k := len(ctx.z), ctx.Xd.cols
	vz := make([]f64, T, context.temp_allocator)
	S := make([]f64, T, context.temp_allocator)
	vx := make([]f64, T * k, context.temp_allocator)
	col := make([]f64, T, context.temp_allocator)
	vc := make([]f64, T, context.temp_allocator)
	if !_arimax_filter(ctx.z, phi, theta, sigma2, vz, S) do return math.inf_f64(-1)
	for j in 0 ..< k {
		for t in 0 ..< T {col[t] = ctx.Xd.data[t * k + j]}
		if !_arimax_filter(col, phi, theta, sigma2, vc, nil) do return math.inf_f64(-1)
		for t in 0 ..< T {vx[t * k + j] = vc[t]}
	}
	// weighted least squares of vz on vx with weights 1/S
	G := l.matrix_new(f64, k, k, context.temp_allocator)
	Gi := l.matrix_new(f64, k, k, context.temp_allocator)
	g := make([]f64, k, context.temp_allocator)
	for t in 0 ..< T {
		w := 1 / S[t]
		for a in 0 ..< k {
			g[a] += w * vx[t * k + a] * vz[t]
			for b in 0 ..< k {G.data[a * k + b] += w * vx[t * k + a] * vx[t * k + b]}
		}
	}
	ml._ols_inverse(&G, &Gi)
	for a in 0 ..< k {
		s := 0.0
		for b in 0 ..< k {s += Gi.data[a * k + b] * g[b]}
		beta_out[a] = s
	}
	ll := 0.0
	ln_2pi := math.ln_f64(2 * math.PI)
	for t in 0 ..< T {
		e := vz[t]
		for a in 0 ..< k {e -= vx[t * k + a] * beta_out[a]}
		ll += -0.5 * (ln_2pi + math.ln_f64(S[t]) + e * e / S[t])
	}
	return ll
}

_arimax_loglik_at :: proc(ctx: ^_ArimaxCtx, phi, theta: []f64, sigma2: f64, beta: []f64) -> f64 {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	T, k := len(ctx.z), ctx.Xd.cols
	u := make([]f64, T, context.temp_allocator)
	for t in 0 ..< T {
		s := ctx.z[t]
		for j in 0 ..< k {s -= ctx.Xd.data[t * k + j] * beta[j]}
		u[t] = s
	}
	return arima_loglik(u, phi, theta, 0, sigma2, context.temp_allocator)
}

// Kalman filter of one series under ARMA(phi, theta, sigma2): innovations and variances
_arimax_filter :: proc(z: []f64, phi, theta: []f64, sigma2: f64, v_out, S_out: []f64) -> bool {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	F, Q, P0, _, _, x0, N := arma_pq_fundamental_state_space(phi, theta, sigma2, context.temp_allocator)
	ll := _arma_filter(z, F, Q, N, x0, P0, v_out, S_out)
	return !math.is_inf(ll) && !math.is_nan(ll)
}

// Negative profile log-likelihood over [x_phi(p), x_theta(q), log σ²] (as arima_neg_loglik_obj)
_arimax_obj :: proc(params: []f64, ctx_ptr: rawptr) -> f64 {
	ctx := (^_ArimaxCtx)(ctx_ptr)
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	p, q := ctx.p, ctx.q
	if len(params) != p + q + 1 do return 1e12
	phi := make([]f64, p, context.temp_allocator)
	theta := make([]f64, q, context.temp_allocator)
	beta := make([]f64, ctx.Xd.cols, context.temp_allocator)
	_arima_constrain(params[:p], phi, false)
	_arima_constrain(params[p:p + q], theta, true)
	sigma2 := math.exp_f64(params[p + q])
	if !(sigma2 > 0) || math.is_inf(sigma2) do return 1e12
	ll := _arimax_profile(ctx, phi, theta, sigma2, beta)
	if math.is_nan(ll) || math.is_inf(ll) do return 1e12
	return -ll
}
