package analytics

import w "../core"
import "core:math"
import "base:runtime"
import "core:mem"

sarima_difference :: proc(
	y: []f64,
	d: int,
	D: int,
	s: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {

	y1 := difference(y, d, allocator)
	y2 := seasonal_difference(y1, D, s, allocator)
	return y2
}
sarima_inverse_difference :: proc(
	diff: []f64,
	hist_ns: []f64, // non-seasonal history (for d)
	hist_s: []f64, // seasonal history (for D,s)
	d: int,
	D: int,
	s: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {

	// 1) undo seasonal differencing
	y_seasonal := seasonal_inverse_difference(diff, hist_s, D, s, allocator)

	// 2) undo non-seasonal differencing
	y_full := inverse_difference(y_seasonal, hist_ns, d, allocator)

	return y_full
}
SarimaForecastResult :: struct {
	mean:  []f64,
	lower: []f64,
	upper: []f64,
}

// Forecast h steps ahead on the original scale of y: exactly h values with (1 - alpha)
// Gaussian prediction intervals, as statsmodels' SARIMAX get_forecast(h).conf_int(alpha).
// The integration (1 - L)^d (1 - L^s)^D is carried in the forecast state, so the
// interval width accumulates the psi-weights of the integrated process.
sarima_forecast :: proc(
	y: []f64,
	phi: []f64, // non-seasonal AR
	theta: []f64, // non-seasonal MA
	Phi: []f64, // seasonal AR
	Theta: []f64, // seasonal MA
	d: int,
	D: int,
	s: int,
	h: int,
	sigma2: f64,
	alpha: f64 = 0.05,
	allocator: mem.Allocator = context.allocator,
) -> SarimaForecastResult {
	if s < 0 do panic("sarima_forecast: season length s must be >= 0")
	h := max(h, 0)
	// Integrating back needs d + D*s observations: too short to forecast.
	if len(y) <= max(d, 0) + max(D, 0) * s && (d > 0 || (D > 0 && s > 0)) {
		mean, lower, upper := _nan_forecast(h, allocator)
		return SarimaForecastResult{mean = mean, lower = lower, upper = upper}
	}

	// scratch is freed explicitly (no temp guard: `allocator` may be the temp allocator)
	ar, ma := sarima_expand_to_arma(phi, Phi, theta, Theta, s, context.temp_allocator)
	c := _arima_diff_poly(d, D, s, context.temp_allocator)
	defer {
		delete(ar, context.temp_allocator)
		delete(ma, context.temp_allocator)
		delete(c, context.temp_allocator)
	}
	mean, lower, upper := _arima_forecast_core(y, ar, ma, sigma2, c, h, alpha, allocator)
	return SarimaForecastResult{mean = mean, lower = lower, upper = upper}
}

simulate_sarima_pdqPDQ :: proc(
	phi: []f64, // non-seasonal AR
	d: int,
	theta: []f64, // non-seasonal MA
	Phi: []f64, // seasonal AR
	D: int,
	Theta: []f64, // seasonal MA
	s: int,
	sigma2: f64,
	T: int,
	allocator: mem.Allocator,
) -> []f64 {
	// 1) expand to ARMA on differenced scale
	ar, ma := sarima_expand_to_arma(phi, Phi, theta, Theta, s, allocator)

	// 2) simulate ARMA on z_t = (1-L)^d (1-L^s)^D y_t
	extra := d + D * s
	if extra < 0 {
		extra = 0
	}
	z_len := T + extra
	z := arma22_simulate(ar, ma, sigma2, z_len, allocator)

	// 3) invert differencing to get y_t
	hist_ns := make([]f64, d, allocator)
	hist_s := make([]f64, D * s, allocator)
	// histories are zero-initialized

	y_full := sarima_inverse_difference(z, hist_ns, hist_s, d, D, s, allocator)

	// 4) drop warmup prefix
	drop := extra
	if drop > len(y_full) {
		drop = len(y_full)
	}

	out_len := T
	if drop + out_len > len(y_full) {
		out_len = len(y_full) - drop
	}

	out := make([]f64, out_len, allocator)
	for t in 0 ..< out_len {
		out[t] = y_full[drop + t]
	}

	return out
}
SarimaAutoResult :: struct {
	p, d, q: int,
	P, D, Q: int,
	s:       int,
	fit:     SarimaFitResult,
}

// Grid search over (p, q, P, Q) by AIC/BIC with d (ADF + KPSS) and D (stationarity
// test on the seasonal difference) chosen first: information criteria are only
// comparable between models fitted to the same differenced series.
sarima_auto_with_tests :: proc(
	y: []f64,
	max_p: int,
	max_d: int,
	max_q: int,
	max_P: int,
	max_D: int,
	max_Q: int,
	s: int,
	criterion: string = "aic",
	allocator: mem.Allocator = context.allocator,
) -> SarimaAutoResult {

	// --- 1) Determine non-seasonal differencing d ---
	d := min(auto_arima_d_from_tests(y, allocator), max(max_d, 0))

	// --- 2) Determine seasonal differencing D ---
	// Simple rule: if the seasonal difference still tests as difference-stationary, D = 1
	D := 0
	if s > 0 && max_D > 0 && len(y) > 2 * s {
		y_lag := make([]f64, len(y) - s, allocator)
		for i in s ..< len(y) {
			y_lag[i - s] = y[i] - y[i - s]
		}
		dec, _, _, _, _ := stationarity_test(y_lag, 10, .Constant, .AIC, .Level, allocator)
		if dec == .DifferenceStationary {
			D = 1
		}
	}

	best: SarimaAutoResult
	best_score := math.INF_F64
	n_diff := len(y) - d - D * s

	// --- 3) Grid search at fixed (d, D) ---
	for p in 0 ..= max_p {
		for q in 0 ..= max_q {
			for P in 0 ..= max_P {
				for Q in 0 ..= max_Q {
					// skip trivial model
					if p == 0 && q == 0 && P == 0 && Q == 0 {
						continue
					}
					if n_diff < (p + q + P + Q + 5) {
						continue
					}

					fit := sarima_fit(y, p, d, q, P, D, Q, s, allocator)
					if !fit.converged {
						continue
					}

					score := fit.aic
					if criterion == "bic" {
						score = fit.bic
					}

					if score < best_score {
						best_score = score
						best = SarimaAutoResult {
							p   = p,
							d   = d,
							q   = q,
							P   = P,
							D   = D,
							Q   = Q,
							s   = s,
							fit = fit,
						}
					}
				}
			}
		}
	}

	return best
}
SarimaObjectiveCtx :: struct {
	y:         []f64,
	y_diff:    []f64, // precomputed (1-L)^d (1-L^s)^D y_t
	p, d, q:   int,
	P, D, Q:   int,
	s:         int,
	allocator: mem.Allocator,
}

// Unpack [x_phi(p), x_theta(q), x_Phi(P), x_Theta(Q), log sigma2]: each AR block is
// mapped to a stationary and each MA block to an invertible polynomial (statsmodels
// SARIMAX transform). Slices are allocated in `allocator`.
_sarima_unpack :: proc(
	params: []f64,
	p, q, P, Q: int,
	allocator: mem.Allocator,
) -> (
	phi, theta, Phi, Theta: []f64,
	sigma2: f64,
) {
	phi = make([]f64, p, allocator)
	theta = make([]f64, q, allocator)
	Phi = make([]f64, P, allocator)
	Theta = make([]f64, Q, allocator)
	_arima_constrain(params[0:p], phi, false)
	_arima_constrain(params[p:p + q], theta, true)
	_arima_constrain(params[p + q:p + q + P], Phi, false)
	_arima_constrain(params[p + q + P:p + q + P + Q], Theta, true)
	sigma2 = math.exp_f64(params[p + q + P + Q])
	return
}

sarima_neg_loglik_obj :: proc(params: []f64, ctx: rawptr) -> f64 {
	data := (^SarimaObjectiveCtx)(ctx)

	p := data.p
	q := data.q
	P := data.P
	Q := data.Q

	// total parameters = p + q + P + Q + 1 (log_sigma2)
	if len(params) != p + q + P + Q + 1 {
		return 1e12
	}

	// Everything below is per-evaluation scratch: release it on return
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	allocator := context.temp_allocator

	phi, theta, Phi, Theta, sigma2 := _sarima_unpack(params, p, q, P, Q, allocator)
	if !(sigma2 > 0.0) || math.is_inf(sigma2) {
		return 1e12
	}

	// differenced series is precomputed by sarima_fit
	y2 := data.y_diff
	if len(y2) < (p + q + P + Q + 5) {
		return 1e12
	}

	// exact likelihood of the expanded ARMA
	ar, ma := sarima_expand_to_arma(phi, Phi, theta, Theta, data.s, allocator)
	ll := arima_loglik(y2, ar, ma, 0, sigma2, allocator)
	if math.is_nan(ll) || math.is_inf(ll) {
		return 1e12
	}

	return -ll
}
SarimaFitResult :: struct {
	phi, theta: []f64,
	Phi, Theta: []f64,
	sigma2:     f64,
	loglik:     f64,
	aic:        f64,
	bic:        f64,
	converged:  bool,
}
// Exact-Gaussian MLE of SARIMA(p,d,q)(P,D,Q)_s with stationary / invertible
// polynomials (statsmodels SARIMAX / ARIMA with trend = 'n'). AIC/BIC count sigma2 and
// use n = len(y) - d - D*s.
sarima_fit :: proc(
	y: []f64,
	p, d, q: int,
	P, D, Q: int,
	s: int,
	allocator: mem.Allocator = context.allocator,
) -> SarimaFitResult {

	y1 := difference(y, d, context.temp_allocator)
	y2 := seasonal_difference(y1, D, s, context.temp_allocator)
	defer {
		delete(y1, context.temp_allocator)
		delete(y2, context.temp_allocator)
	}
	// without a season length there are no seasonal terms
	P := P if s > 0 else 0
	Q := Q if s > 0 else 0

	ctx := SarimaObjectiveCtx {
		y         = y,
		y_diff    = y2,
		p         = p,
		d         = d,
		q         = q,
		P         = P,
		D         = D,
		Q         = Q,
		s         = s,
		allocator = allocator,
	}

	// sigma2 starts at the sample variance of the differenced series (scale-free);
	// deterministic starts so the fit does not depend on the global RNG state.
	// Each start is the first-lag {phi, theta, Phi, Theta}.
	mean2, var2 := 0.0, 0.0
	for v in y2 {mean2 += v}
	mean2 /= f64(max(len(y2), 1))
	for v in y2 {var2 += (v - mean2) * (v - mean2)}
	var2 /= f64(max(len(y2) - 1, 1))
	log_sig2_0 := math.ln_f64(var2) if var2 > 0.0 else 0.0

	starts := [][4]f64 {
		{0.0, 0.0, 0.0, 0.0},
		{0.5, 0.0, 0.3, 0.0},
		{0.0, 0.5, 0.0, 0.3},
		{0.5, -0.5, 0.3, -0.3},
		{-0.5, 0.5, -0.3, 0.3},
	}
	best_params, best_f, converged := _arima_nm_fit(
		sarima_neg_loglik_obj,
		&ctx,
		{p, q, P, Q},
		{false, true, false, true},
		starts,
		log_sig2_0,
		context.temp_allocator,
	)
	defer delete(best_params, context.temp_allocator)

	phi, theta, Phi, Theta, sigma2 := _sarima_unpack(best_params, p, q, P, Q, allocator)
	ll := -best_f

	// AIC/BIC (k counts sigma2, as statsmodels)
	k := p + q + P + Q + 1
	n := len(y2)

	aic := -2.0 * ll + 2.0 * f64(k)
	bic := -2.0 * ll + f64(k) * math.ln(f64(n))

	return SarimaFitResult {
		phi = phi,
		theta = theta,
		Phi = Phi,
		Theta = Theta,
		sigma2 = sigma2,
		loglik = ll,
		aic = aic,
		bic = bic,
		converged = converged,
	}
}


// One-step-ahead innovations v_t = z_t - E[z_t | z_<t] of the differenced series
// z = (1-L)^d (1-L^s)^D y under the given SARIMA parameters (exact Kalman filter,
// stationary initialisation). One value per differenced observation, as statsmodels'
// `resid` after its d + D*s burn-in; the first few have variance above sigma2.
sarima_innovations :: proc(
	y: []f64,
	phi: []f64,
	theta: []f64,
	Phi: []f64,
	Theta: []f64,
	d: int,
	D: int,
	s: int,
	sigma2: f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	y1 := difference(y, d, context.temp_allocator)
	z := seasonal_difference(y1, D, s, context.temp_allocator)
	defer {
		delete(y1, context.temp_allocator)
		delete(z, context.temp_allocator)
	}

	v := make([]f64, len(z), allocator)
	if len(z) == 0 {
		return v
	}

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	ar, ma := sarima_expand_to_arma(phi, Phi, theta, Theta, s, context.temp_allocator)
	F, Q, P0, _, _, x0, N := arma_pq_fundamental_state_space(ar, ma, sigma2, context.temp_allocator)
	_arma_filter(z, F, Q, N, x0, P0, v)
	return v
}

SarimaDiagResult :: struct {
	acf_vals:  []f64,
	pacf_vals: []f64,
	Q:         f64,
	df:        int,
	p_lb:      f64,
	JB:        f64,
	p_jb:      f64,
}

sarima_residual_diagnostics :: proc(
	y: []f64,
	phi: []f64,
	theta: []f64,
	Phi: []f64,
	Theta: []f64,
	d: int,
	D: int,
	s: int,
	sigma2: f64,
	max_lag: int,
	allocator: mem.Allocator = context.allocator,
) -> SarimaDiagResult {

	v := sarima_innovations(y, phi, theta, Phi, Theta, d, D, s, sigma2, allocator)

	ac := acf(v, max_lag, allocator)
	pc := pacf(v, max_lag, allocator)

	dof_adj := len(phi) + len(theta) + len(Phi) + len(Theta)
	Q, df, p_lb := ljung_box(v, max_lag, dof_adj, allocator)
	JB, p_jb := jarque_bera(v, allocator)

	return SarimaDiagResult {
		acf_vals = ac,
		pacf_vals = pc,
		Q = Q,
		df = df,
		p_lb = p_lb,
		JB = JB,
		p_jb = p_jb,
	}
}


df_sarima_residual_diagnostics :: proc(
	y: []f64,
	phi: []f64,
	theta: []f64,
	Phi: []f64,
	Theta: []f64,
	d: int,
	D: int,
	s: int,
	sigma2: f64,
	max_lag: int,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {

	v := sarima_innovations(y, phi, theta, Phi, Theta, d, D, s, sigma2, allocator)

	Q, df, p_lb := ljung_box(v, max_lag, len(phi) + len(theta) + len(Phi) + len(Theta), allocator)
	JB, p_jb := jarque_bera(v, allocator)

	df_out := w.dataframe_new()
	w.add_column(&df_out, w.column_from_floats("Q", []f64{Q}))
	w.add_column(&df_out, w.column_from_ints("df", []int{df}))
	w.add_column(&df_out, w.column_from_floats("p_lb", []f64{p_lb}))
	w.add_column(&df_out, w.column_from_floats("JB", []f64{JB}))
	w.add_column(&df_out, w.column_from_floats("p_jb", []f64{p_jb}))
	df_out.rows = 1
	return df_out
}
// Residuals of a fixed SARIMA: the one-step innovations (see sarima_innovations).
sarima_residuals :: proc(
	y: []f64,
	phi: []f64,
	theta: []f64,
	Phi: []f64,
	Theta: []f64,
	d: int,
	D: int,
	s: int,
	sigma2: f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	return sarima_innovations(y, phi, theta, Phi, Theta, d, D, s, sigma2, allocator)
}
