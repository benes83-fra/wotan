package analytics

import w "../core"
import "core:math"
import "core:math/rand"
import "base:runtime"
import "core:mem"

// Conventions (match statsmodels.tsa.arima.model.ARIMA / SARIMAX with trend = 'n'):
//   δ(L) y_t = z_t,  δ(L) = (1 - L)^d (1 - L^s)^D
//   z_t = Σ phi_i z_{t-i} + e_t + Σ theta_j e_{t-j},  e_t ~ N(0, sigma2)
// (MA terms enter with a + sign). There is no mean / constant term. The likelihood is
// the exact Gaussian likelihood of the differenced series z (stationary initial state),
// which equals the exact-diffuse likelihood of y; statsmodels' approximate-diffuse
// (1e6) initialisation of the integrated states differs from it by ~1e-5 for d > 0.
// sigma2 is a free parameter (not concentrated out) and counts in AIC/BIC.

arima_state_space :: proc(
	phi, theta: []f64,
	d: int,
	sigma2: f64,
	allocator := context.allocator,
) -> (
	F, Q, P0: []f64,
	H: []f64,
	R: []f64,
	x0: []f64,
	N: int,
) {
	// Differencing is handled outside; here we are purely ARMA(p,q)
	return arma_pq_fundamental_state_space(phi, theta, sigma2, allocator)
}


// difference(series, d) -> differenced series
difference :: proc(series: []f64, d: int, allocator := context.allocator) -> []f64 {
	if d <= 0 {
		// No differencing
		out := make([]f64, len(series), allocator)
		for i in 0 ..< len(series) {
			out[i] = series[i]
		}
		return out
	}

	if len(series) == 0 {
		return make([]f64, 0, allocator)
	}

	// First-order difference
	diff := make([]f64, len(series) - 1, allocator)
	for i in 1 ..< len(series) {
		diff[i - 1] = series[i] - series[i - 1]
	}

	// Higher-order differencing: recurse
	for k in 1 ..< d {
		if len(diff) <= 1 {
			break
		}
		next := make([]f64, len(diff) - 1, allocator)
		for i in 1 ..< len(diff) {
			next[i - 1] = diff[i] - diff[i - 1]
		}
		diff = next
	}

	return diff
}
difference_with_history :: proc(
	series: []f64,
	d: int,
	allocator := context.allocator,
) -> (
	diff: []f64,
	history: []f64,
) {

	if d <= 0 {
		diff = make([]f64, len(series), allocator)
		for i in 0 ..< len(series) {
			diff[i] = series[i]
		}
		history = make([]f64, 0, allocator)
		return
	}

	// Save first d values for inverse differencing (fewer if the series is shorter)
	history = make([]f64, min(d, len(series)), allocator)
	for i in 0 ..< len(history) {
		history[i] = series[i]
	}

	diff = difference(series, d, allocator)
	return
}

// Coefficients c[0..m] of the differencing polynomial (1 - L)^d (1 - L^s)^D, c[0] = 1,
// m = d + D*s. The seasonal factor is dropped when s <= 0.
_arima_diff_poly :: proc(d, D, s: int, allocator: mem.Allocator) -> []f64 {
	d := max(d, 0)
	D := max(D, 0) if s > 0 else 0
	c := make([]f64, d + D * s + 1, allocator)
	c[0] = 1.0
	deg := 0
	for _ in 0 ..< d { 	// multiply by (1 - L), in place (descending index reads old values)
		for i := deg + 1; i >= 1; i -= 1 {c[i] -= c[i - 1]}
		deg += 1
	}
	for _ in 0 ..< D { 	// multiply by (1 - L^s)
		for i := deg + s; i >= s; i -= 1 {c[i] -= c[i - s]}
		deg += s
	}
	return c
}

// Undo δ(L) y_t = z_t given the first len(history) original values:
// y_t = z_t - Σ_{k>=1} c_k y_{t-k} (lags before the history count as 0).
// Returns history followed by the reconstructed values.
_arima_undifference :: proc(diff, history, c: []f64, allocator: mem.Allocator) -> []f64 {
	nh := len(history)
	out := make([]f64, len(diff) + nh, allocator)
	copy(out, history)
	for i in 0 ..< len(diff) {
		t := nh + i
		v := diff[i]
		for k in 1 ..< len(c) {
			if t - k >= 0 {v -= c[k] * out[t - k]}
		}
		out[t] = v
	}
	return out
}

// Inverse of `difference`: history = the first d original values (see
// difference_with_history); returns history followed by the reconstructed series.
inverse_difference :: proc(
	diff: []f64,
	history: []f64,
	d: int,
	allocator := context.allocator,
) -> []f64 {

	if d <= 0 {
		out := make([]f64, len(diff), allocator)
		for i in 0 ..< len(diff) {
			out[i] = diff[i]
		}
		return out
	}

	c := _arima_diff_poly(d, 0, 0, context.temp_allocator)
	defer delete(c, context.temp_allocator)
	return _arima_undifference(diff, history, c, allocator)
}

// Stationary state covariance: solves P = F P Fᵀ + Q (P = Σ_k F^k Q F'^k) by the
// doubling algorithm. Returns false (P unspecified) if F is not stable.
_arima_stationary_cov :: proc(F, Q: []f64, N: int, P: []f64) -> bool {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	A := make([]f64, N * N, context.temp_allocator)
	T1 := make([]f64, N * N, context.temp_allocator)
	copy(A, F)
	copy(P, Q)
	for _ in 0 ..< 100 {
		// P += A P Aᵀ
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {s += A[i * N + k] * P[k * N + j]}
				T1[i * N + j] = s
			}
		}
		incr, scale := 0.0, 0.0
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {s += T1[i * N + k] * A[j * N + k]}
				P[i * N + j] += s
				incr = max(incr, abs(s))
				scale = max(scale, abs(P[i * N + j]))
			}
		}
		if math.is_nan(incr) || math.is_inf(incr) || scale > 1e300 {return false}
		if incr <= 1e-17 * scale {return true}
		// A = A A
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {s += A[i * N + k] * A[k * N + j]}
				T1[i * N + j] = s
			}
		}
		copy(A, T1)
	}
	return false
}

// Fill P0 with the stationary covariance, or an approximate-diffuse 1e6·I when F has a
// unit / explosive root (only reachable with user-supplied parameters).
_arima_init_P0 :: proc(F, Q: []f64, N: int, P0: []f64) {
	if !_arima_stationary_cov(F, Q, N, P0) {
		for i in 0 ..< N * N {P0[i] = 0.0}
		for i in 0 ..< N {P0[i * N + i] = 1e6}
	}
}

// Exact Gaussian Kalman filter for the Harvey ARMA form of
// arma_pq_fundamental_state_space (F = AR coefficients in column 0 and ones on the
// superdiagonal, H = e_0, R = 0), O(N²) per step. On entry a, P hold the predicted
// state for the first observation (x0 = 0, stationary P0); on return the filtered
// state a_{T|T}, P_{T|T}. v_out / S_out (optional, length len(z)) receive the
// innovations and their variances. Returns the log-likelihood.
_arma_filter :: proc(z: []f64, F, Q: []f64, N: int, a, P: []f64, v_out: []f64 = nil, S_out: []f64 = nil) -> f64 {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	col := make([]f64, N, context.temp_allocator)
	TP := make([]f64, N * N, context.temp_allocator)
	ll := 0.0
	ln_2pi := math.ln_f64(2.0 * math.PI)
	for t in 0 ..< len(z) {
		if t > 0 {
			// a = F a
			a0 := a[0]
			for i in 0 ..< N - 1 {a[i] = F[i * N] * a0 + a[i + 1]}
			a[N - 1] = F[(N - 1) * N] * a0
			// TP = F P
			for i in 0 ..< N {
				fi := F[i * N]
				for j in 0 ..< N {
					TP[i * N + j] = fi * P[j] + (P[(i + 1) * N + j] if i + 1 < N else 0.0)
				}
			}
			// P = TP Fᵀ + Q,  (TP Fᵀ)_ij = TP_i0 F_j0 + TP_i,j+1
			for i in 0 ..< N {
				for j in 0 ..< N {
					P[i * N + j] = TP[i * N] * F[j * N] + (TP[i * N + j + 1] if j + 1 < N else 0.0) + Q[i * N + j]
				}
			}
		}
		S := P[0]
		v := z[t] - a[0]
		if v_out != nil {v_out[t] = v}
		if S_out != nil {S_out[t] = S}
		if !(S > 0.0) {return math.inf_f64(-1)}
		ll += -0.5 * (ln_2pi + math.ln_f64(S) + v * v / S)
		for i in 0 ..< N {col[i] = P[i * N]}
		for i in 0 ..< N {a[i] += col[i] / S * v}
		for i in 0 ..< N {
			for j in 0 ..< N {P[i * N + j] -= col[i] * col[j] / S}
		}
	}
	return ll
}

// Exact Gaussian log-likelihood of the (already differenced) series y under
// ARMA(phi, theta, sigma2). `d` is ignored: differencing is handled outside.
arima_loglik :: proc(
	y: []f64,
	phi, theta: []f64,
	d: int,
	sigma2: f64,
	allocator := context.allocator,
) -> f64 {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	F, Q, P0, _, _, x0, N := arma_pq_fundamental_state_space(phi, theta, sigma2, context.temp_allocator)
	return _arma_filter(y, F, Q, N, x0, P0)
}

// statsmodels' constrain_stationary_univariate (Jones 1980 / Monahan 1984): maps
// unconstrained x to a stationary AR polynomial through the partial autocorrelations
// r_k = x_k / sqrt(1 + x_k²). negate = true gives an invertible MA polynomial
// (theta = -phi), as SARIMAX does with enforce_stationarity / enforce_invertibility.
_arima_constrain :: proc(x: []f64, out: []f64, negate: bool) {
	n := len(x)
	if n == 0 {return}
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	y := make([]f64, n * n, context.temp_allocator)
	for k in 0 ..< n {
		r := x[k] / math.sqrt_f64(1.0 + x[k] * x[k])
		for i in 0 ..< k {
			y[k * n + i] = y[(k - 1) * n + i] + r * y[(k - 1) * n + k - i - 1]
		}
		y[k * n + k] = r
	}
	for i in 0 ..< n {out[i] = y[(n - 1) * n + i] if negate else -y[(n - 1) * n + i]}
}

// Inverse of _arima_constrain; false if c is not stationary (invertible for MA).
_arima_unconstrain :: proc(c: []f64, out: []f64, negate: bool) -> bool {
	n := len(c)
	if n == 0 {return true}
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	y := make([]f64, n * n, context.temp_allocator)
	for i in 0 ..< n {y[(n - 1) * n + i] = c[i] if negate else -c[i]}
	for k := n - 1; k > 0; k -= 1 {
		rk := y[k * n + k]
		den := 1.0 - rk * rk
		if !(den > 0.0) {return false}
		for i in 0 ..< k {
			y[(k - 1) * n + i] = (y[k * n + i] - rk * y[k * n + k - i - 1]) / den
		}
	}
	for k in 0 ..< n {
		r := y[k * n + k]
		if !(abs(r) < 1.0) {return false}
		out[k] = r / math.sqrt_f64(1.0 - r * r)
	}
	return true
}

// Optimiser coordinates [x_phi(p), x_theta(q), log sigma2] of an ARMA(p,q) parameter
// set (the inverse of what arima_neg_loglik_obj applies). ok = false if phi is not
// stationary or theta not invertible.
arima_unconstrain_params :: proc(
	phi, theta: []f64,
	sigma2: f64,
	allocator := context.allocator,
) -> (
	x: []f64,
	ok: bool,
) {
	p, q := len(phi), len(theta)
	x = make([]f64, p + q + 1, allocator)
	ok = _arima_unconstrain(phi, x[:p], false) && _arima_unconstrain(theta, x[p:p + q], true) && sigma2 > 0.0
	x[p + q] = math.ln_f64(sigma2) if sigma2 > 0.0 else 0.0
	return
}


ArimaObjectiveCtx :: struct {
	y:         []f64,
	p, q:      int,
	d:         int,
	allocator: mem.Allocator,
}
arima_neg_loglik_obj :: proc(params: []f64, ctx: rawptr) -> f64 {
	data := (^ArimaObjectiveCtx)(ctx)
	// Everything below is per-evaluation scratch: release it on return
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	scratch := context.temp_allocator

	p := data.p
	q := data.q

	if len(params) != p + q + 1 {
		return 1e12
	}

	// unpack params: [x_phi(p), x_theta(q), log_sigma2] -> stationary phi, invertible theta
	phi := make([]f64, p, scratch)
	theta := make([]f64, q, scratch)
	_arima_constrain(params[:p], phi, false)
	_arima_constrain(params[p:p + q], theta, true)

	sigma2 := math.exp_f64(params[p + q])
	if !(sigma2 > 0.0) || math.is_inf(sigma2) {
		return 1e12
	}

	// differencing if needed
	y_eff := data.y
	if data.d > 0 {
		y_eff = difference(data.y, data.d, scratch)
		if len(y_eff) < (p + q + 5) {
			return 1e12
		}
	}

	ll := arima_loglik(y_eff, phi, theta, data.d, sigma2, scratch)
	if math.is_nan(ll) || math.is_inf(ll) {
		return 1e12
	}

	// minimize negative log-likelihood
	return -ll
}

ArimaFitResult :: struct {
	phi:       []f64,
	theta:     []f64,
	sigma2:    f64,
	loglik:    f64,
	aic:       f64,
	bic:       f64,
	converged: bool,
}

// Multi-start Nelder-Mead on a negative log-likelihood, then restarts from the best
// point until it stops improving. Parameter groups (in params order) have sizes
// counts[g] and are MA blocks if is_ma[g]; each start gives the constrained first-lag
// value of every group (higher lags start at 0). Returns the best optimiser point.
_arima_nm_fit :: proc(
	obj: ObjectiveWithCtx,
	ctx: rawptr,
	counts: []int,
	is_ma: []bool,
	starts: [][4]f64,
	log_sig2_0: f64,
	allocator: mem.Allocator,
) -> (
	best_params: []f64,
	best_f: f64,
	converged: bool,
) {
	n_params := 1
	for c in counts {n_params += c}
	best_params = make([]f64, n_params, allocator)
	best_f = math.INF_F64

	max_iter := 2000 + 500 * n_params
	tol := 1e-10 // objective spread
	xtol := 1e-7 // simplex size (optimiser coordinates)

	x0 := make([]f64, n_params, context.temp_allocator)
	defer delete(x0, context.temp_allocator)
	for st in starts {
		idx := 0
		for g in 0 ..< len(counts) {
			if counts[g] > 0 {
				c := make([]f64, counts[g], context.temp_allocator)
				c[0] = st[g]
				if !_arima_unconstrain(c, x0[idx:idx + counts[g]], is_ma[g]) {
					for k in 0 ..< counts[g] {x0[idx + k] = 0.0}
				}
				delete(c, context.temp_allocator)
			}
			idx += counts[g]
		}
		x0[n_params - 1] = log_sig2_0

		x_s, f_s, conv := nelder_mead(obj, ctx, x0, max_iter, tol, context.temp_allocator, xtol)
		if f_s < best_f {
			best_f = f_s
			copy(best_params, x_s)
			converged = conv
		}
		delete(x_s, context.temp_allocator)
	}

	// Polish: restart from the best vertex with a fresh simplex until no gain
	for _ in 0 ..< 10 {
		x_s, f_s, conv := nelder_mead(obj, ctx, best_params, max_iter, tol, context.temp_allocator, xtol)
		gain := best_f - f_s
		if f_s <= best_f {
			best_f = f_s
			copy(best_params, x_s)
			converged = conv
		}
		delete(x_s, context.temp_allocator)
		if gain < 1e-9 {break}
	}
	return
}

arima_fit :: proc(y: []f64, p, d, q: int, allocator := context.allocator) -> ArimaFitResult {
	ctx := ArimaObjectiveCtx {
		y         = y,
		p         = p,
		q         = q,
		d         = d,
		allocator = allocator,
	}

	// sigma2 starts at the sample variance of the (differenced) series, so the
	// fit is scale-free (returns ~1e-4 vs unit-variance data)
	y_eff := y
	if d > 0 {y_eff = difference(y, d, context.temp_allocator)}
	n_eff := len(y_eff)
	mean_eff, var_eff := 0.0, 0.0
	for v in y_eff {mean_eff += v}
	mean_eff /= f64(max(n_eff, 1))
	for v in y_eff {var_eff += (v - mean_eff) * (v - mean_eff)}
	var_eff /= f64(max(n_eff - 1, 1))
	if d > 0 {delete(y_eff, context.temp_allocator)}
	log_sig2_0 := math.ln_f64(var_eff) if var_eff > 0.0 else 0.0

	// Deterministic starts: first-lag (phi_1, theta_1), higher lags 0
	starts := [][4]f64{{0.0, 0.0, 0, 0}, {0.5, 0.0, 0, 0}, {0.0, 0.5, 0, 0}, {0.5, -0.5, 0, 0}, {-0.5, 0.5, 0, 0}}
	best_params, best_f, converged := _arima_nm_fit(
		arima_neg_loglik_obj,
		&ctx,
		{p, q},
		{false, true},
		starts,
		log_sig2_0,
		context.temp_allocator,
	)
	defer delete(best_params, context.temp_allocator)

	res: ArimaFitResult
	res.phi = make([]f64, p, allocator)
	res.theta = make([]f64, q, allocator)
	_arima_constrain(best_params[:p], res.phi, false)
	_arima_constrain(best_params[p:p + q], res.theta, true)

	res.sigma2 = math.exp_f64(best_params[p + q])
	res.loglik = -best_f
	res.converged = converged
	// AIC / BIC as statsmodels: k counts sigma2, n = observations after differencing
	k := p + q + 1
	n := n_eff

	res.aic = -2.0 * res.loglik + 2.0 * f64(k)
	res.bic = -2.0 * res.loglik + f64(k) * math.ln(f64(n))

	return res
}
ArimaForecastResult :: struct {
	mean:  []f64,
	lower: []f64,
	upper: []f64,
}

// NaN-filled forecast arrays, returned when the series is too short to forecast.
_nan_forecast :: proc(h: int, allocator: mem.Allocator) -> (mean, lower, upper: []f64) {
	mean = make([]f64, h, allocator)
	lower = make([]f64, h, allocator)
	upper = make([]f64, h, allocator)
	for i in 0 ..< h {
		mean[i] = math.nan_f64()
		lower[i] = math.nan_f64()
		upper[i] = math.nan_f64()
	}
	return
}

// Standard normal quantile: Acklam's rational approximation refined by one Halley
// step on erfc (full double precision).
_arima_norm_ppf :: proc(p: f64) -> f64 {
	if p <= 0.0 {return math.inf_f64(-1)}
	if p >= 1.0 {return math.inf_f64(1)}
	a := [6]f64{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00}
	b := [5]f64{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01}
	c := [6]f64{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00}
	d := [4]f64{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00}
	x: f64
	if p < 0.02425 || p > 1.0 - 0.02425 {
		q := math.sqrt_f64(-2.0 * math.ln_f64(p if p < 0.5 else 1.0 - p))
		x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
		if p > 0.5 {x = -x}
	} else {
		q := p - 0.5
		r := q * q
		x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
	}
	e := 0.5 * math.erfc(-x / math.SQRT_TWO) - p
	u := e * math.sqrt_f64(2.0 * math.PI) * math.exp_f64(0.5 * x * x)
	return x - u / (1.0 + 0.5 * x * u)
}

// h-step forecasts on the original scale of y for ARMA(ar, ma, sigma2) on the
// differenced series δ(L) y (c = coefficients of δ, see _arima_diff_poly).
// The integration is carried in the state, [α_t (N) | y_t, y_{t-1}, …], with
// y_t = α_t[0] - Σ_{k>=1} c_k y_{t-k}; the observed lags are known exactly at T, so the
// forecast variance accumulates the psi-weights of the integrated process (as
// statsmodels' ARIMA / SARIMAX get_forecast). Intervals: mean ± z_{1-alpha/2}·sd.
// Requires len(y) > len(c) - 1 (or no differencing).
_arima_forecast_core :: proc(
	y, ar, ma: []f64,
	sigma2: f64,
	c: []f64,
	h: int,
	alpha: f64,
	allocator: mem.Allocator,
) -> (
	mean, lower, upper: []f64,
) {
	mean = make([]f64, h, allocator)
	lower = make([]f64, h, allocator)
	upper = make([]f64, h, allocator)

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	scratch := context.temp_allocator
	m := len(c) - 1
	n := len(y)

	// 1) differenced series z_t = Σ c_k y_{t-k}, then filter it
	z := make([]f64, max(n - m, 0), scratch)
	for t in m ..< n {
		s := 0.0
		for k in 0 ..= m {s += c[k] * y[t - k]}
		z[t - m] = s
	}
	F, Q, P, _, _, a, N := arma_pq_fundamental_state_space(ar, ma, sigma2, scratch)
	_arma_filter(z, F, Q, N, a, P)

	// 2) augmented state [α | y_t, …, y_{t-L+1}]
	L := max(m, 1)
	M := N + L
	G := make([]f64, M * M, scratch)
	Qa := make([]f64, M * M, scratch)
	xa := make([]f64, M, scratch)
	Pa := make([]f64, M * M, scratch)
	T1 := make([]f64, M * M, scratch)
	xn := make([]f64, M, scratch)
	for i in 0 ..< N {
		for j in 0 ..< N {
			G[i * M + j] = F[i * N + j]
			Qa[i * M + j] = Q[i * N + j]
			Pa[i * M + j] = P[i * N + j]
		}
		// y_{t+1} = α_{t+1}[0] - Σ c_k y_{t+1-k} = F[0,:] α_t + η_0 - Σ c_k y_{t+1-k}
		G[N * M + i] = F[i]
		Qa[N * M + i] = Q[i]
		Qa[i * M + N] = Q[i]
		xa[i] = a[i]
	}
	Qa[N * M + N] = Q[0]
	for k in 0 ..< m {G[N * M + N + k] = -c[k + 1]}
	for k in 1 ..< L {G[(N + k) * M + N + k - 1] = 1.0}
	for k in 0 ..< m {xa[N + k] = y[n - 1 - k]}

	al := alpha if alpha > 0.0 && alpha < 1.0 else 0.05
	zq := _arima_norm_ppf(1.0 - 0.5 * al)

	for step in 0 ..< h {
		for i in 0 ..< M {
			s := 0.0
			for j in 0 ..< M {s += G[i * M + j] * xa[j]}
			xn[i] = s
		}
		copy(xa, xn)
		for i in 0 ..< M {
			for j in 0 ..< M {
				s := 0.0
				for k in 0 ..< M {s += G[i * M + k] * Pa[k * M + j]}
				T1[i * M + j] = s
			}
		}
		for i in 0 ..< M {
			for j in 0 ..< M {
				s := 0.0
				for k in 0 ..< M {s += T1[i * M + k] * G[j * M + k]}
				Pa[i * M + j] = s + Qa[i * M + j]
			}
		}
		mu := xa[N]
		sd := math.sqrt_f64(max(Pa[N * M + N], 0.0))
		mean[step] = mu
		lower[step] = mu - zq * sd
		upper[step] = mu + zq * sd
	}
	return
}

// Forecast h steps ahead on the original scale of y: exactly h values (none for
// h <= 0) with (1 - alpha) Gaussian prediction intervals, as statsmodels'
// get_forecast(h).conf_int(alpha). p and q are implied by fit.phi / fit.theta.
arima_forecast :: proc(
	y: []f64,
	fit: ArimaFitResult,
	p, d, q: int,
	h: int,
	alpha: f64, // e.g. 0.05 for 95% PI
	allocator := context.allocator,
) -> ArimaForecastResult {
	h := max(h, 0)
	// Integrating back needs the last d observations: too short to forecast.
	if d > 0 && len(y) <= d {
		mean, lower, upper := _nan_forecast(h, allocator)
		return ArimaForecastResult{mean = mean, lower = lower, upper = upper}
	}
	c := _arima_diff_poly(d, 0, 0, context.temp_allocator)
	defer delete(c, context.temp_allocator)
	mean, lower, upper := _arima_forecast_core(y, fit.phi, fit.theta, fit.sigma2, c, h, alpha, allocator)
	return ArimaForecastResult{mean = mean, lower = lower, upper = upper}
}
arma11_state_space :: proc(
	phi: f64,
	theta: f64,
	sigma2: f64,
	allocator := context.allocator,
) -> (
	F, Q, P0: []f64,
	H: []f64,
	R: []f64,
	x0: []f64,
	N: int,
) {

	N = 2 // state = [y_t, e_t]

	F = make([]f64, N * N, allocator)
	Q = make([]f64, N * N, allocator)
	P0 = make([]f64, N * N, allocator)
	H = make([]f64, N, allocator)
	R = make([]f64, 1, allocator)
	x0 = make([]f64, N, allocator)

	// Transition matrix
	// [ φ   θ ]
	// [ 0   0 ]
	F[0] = phi
	F[1] = theta
	F[2] = 0.0
	F[3] = 0.0

	// Noise covariance Q = σ² * [[1,1],[1,1]]
	Q[0] = sigma2
	Q[1] = sigma2
	Q[2] = sigma2
	Q[3] = sigma2

	// Observation y_t = [1 0] α_t
	H[0] = 1.0
	H[1] = 0.0

	// No measurement noise
	R[0] = 0.0

	// Stationary initial covariance (approximate diffuse only if |φ| >= 1)
	_arima_init_P0(F, Q, N, P0)

	return
}

// ARMA(p,q) in Harvey's state-space form (the one statsmodels uses):
//   N = max(p, q + 1),  y_t = α_t[0]  (H = e_0, R = 0),
//   α_{t+1} = F α_t + g e_{t+1},  F = [phi | I_{N-1}; 0 … 0]  (phi zero-padded to N),
//   g = [1, theta_1, …, theta_{N-1}],  Q = sigma2 g gᵀ,
//   x0 = 0, P0 = stationary covariance (P0 = F P0 Fᵀ + Q).
arma_pq_fundamental_state_space :: proc(
	phi, theta: []f64,
	sigma2: f64,
	allocator := context.allocator,
) -> (
	F, Q, P0: []f64,
	H: []f64,
	R: []f64,
	x0: []f64,
	N: int,
) {
	p := len(phi)
	q := len(theta)
	N = max(p, q + 1)

	F = make([]f64, N * N, allocator)
	Q = make([]f64, N * N, allocator)
	P0 = make([]f64, N * N, allocator)
	H = make([]f64, N, allocator)
	R = make([]f64, 1, allocator)
	x0 = make([]f64, N, allocator)

	for i in 0 ..< p {F[i * N] = phi[i]}
	for i in 0 ..< N - 1 {F[i * N + i + 1] = 1.0}

	g :: proc(theta: []f64, i: int) -> f64 {return 1.0 if i == 0 else (theta[i - 1] if i - 1 < len(theta) else 0.0)}
	for i in 0 ..< N {
		for j in 0 ..< N {
			Q[i * N + j] = sigma2 * g(theta, i) * g(theta, j)
		}
	}

	H[0] = 1.0
	R[0] = 0.0
	_arima_init_P0(F, Q, N, P0)
	return
}

// Simulate T values of the general ARMA(p,q)
//   y_t = Σ phi_i y_{t-i} + e_t + Σ theta_j e_{t-j},  e_t ~ N(0, sigma2)
// (any p, q despite the name). The initial state is drawn from the stationary
// distribution, so the path is stationary from t = 0; for non-stationary phi it
// starts from zero pre-sample values instead.
arma22_simulate :: proc(
	phi: []f64,
	theta: []f64,
	sigma2: f64,
	T: int,
	allocator: mem.Allocator,
) -> []f64 {
	y := make([]f64, max(T, 0), allocator)
	if T <= 0 {return y}
	sigma := math.sqrt_f64(max(sigma2, 0.0))

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	scratch := context.temp_allocator
	// Harvey form: y_t = α_t[0], α_{t+1} = F α_t + g e_{t+1}, g = [1, theta…]
	F, _, _, _, _, a, N := arma_pq_fundamental_state_space(phi, theta, 1.0, scratch)
	P := make([]f64, N * N, scratch)
	Qs := make([]f64, N * N, scratch)
	g := make([]f64, N, scratch)
	g[0] = 1.0
	for j in 0 ..< len(theta) {if j + 1 < N {g[j + 1] = theta[j]}}
	for i in 0 ..< N {for j in 0 ..< N {Qs[i * N + j] = sigma2 * g[i] * g[j]}}

	stationary := _arima_stationary_cov(F, Qs, N, P)
	if stationary {
		// α_0 ~ N(0, P): Cholesky factor (pivots that are numerically 0 are skipped,
		// P is singular when q + 1 < p or p < q + 1)
		Lc := make([]f64, N * N, scratch)
		scale := 0.0
		for i in 0 ..< N {scale = max(scale, P[i * N + i])}
		for j in 0 ..< N {
			s := P[j * N + j]
			for k in 0 ..< j {s -= Lc[j * N + k] * Lc[j * N + k]}
			if s <= 1e-13 * scale {continue}
			Lc[j * N + j] = math.sqrt_f64(s)
			for i in j + 1 ..< N {
				v := P[i * N + j]
				for k in 0 ..< j {v -= Lc[i * N + k] * Lc[j * N + k]}
				Lc[i * N + j] = v / Lc[j * N + j]
			}
		}
		u := make([]f64, N, scratch)
		for i in 0 ..< N {u[i] = rand.float64_normal(0.0, 1.0)}
		for i in 0 ..< N {
			s := 0.0
			for k in 0 ..= i {s += Lc[i * N + k] * u[k]}
			a[i] = s
		}
	}

	for t in 0 ..< T {
		if t > 0 {
			e := rand.float64_normal(0.0, sigma)
			a0 := a[0]
			for i in 0 ..< N - 1 {a[i] = F[i * N] * a0 + a[i + 1] + g[i] * e}
			a[N - 1] = F[(N - 1) * N] * a0 + g[N - 1] * e
		} else if !stationary {
			// non-stationary: zero pre-sample, first value is the first shock
			e := rand.float64_normal(0.0, sigma)
			for i in 0 ..< N {a[i] = g[i] * e}
		}
		y[t] = a[0]
	}

	return y
}
simulate_arima_pdq :: proc(
	phi: []f64,
	d: int,
	theta: []f64,
	sigma2: f64,
	T: int,
	allocator: mem.Allocator,
) -> []f64 {
	// simulate ARMA(p,q) on differenced series z_t
	z := arma22_simulate(phi, theta, sigma2, T + d, allocator) // reuse general ARMA sim
	// integrate d times to get y_t
	history := make([]f64, d, allocator)
	for i in 0 ..< d {
		history[i] = 0.0
	}
	y := inverse_difference(z, history, d, allocator)
	// drop initial d to get length T
	out := make([]f64, T, allocator)
	for t in 0 ..< T {
		out[t] = y[d + t]
	}
	return out
}

ArimaAutoResult :: struct {
	p, d, q: int,
	fit:     ArimaFitResult,
}

// Order selection by AIC/BIC over p <= max_p, q <= max_q. AIC/BIC are only comparable
// between models of the same differenced series (a d = 1 likelihood is a density of
// n - 1 differences, a d = 0 one of n levels), so d is fixed first by the ADF + KPSS
// rule of auto_arima_d_from_tests (capped at max_d), as pmdarima / R auto.arima do.
arima_auto :: proc(
	y: []f64,
	max_p: int,
	max_d: int,
	max_q: int,
	criterion: string, // "aic" or "bic"
	allocator := context.allocator,
) -> ArimaAutoResult {

	best: ArimaAutoResult
	best_score := math.INF_F64
	d_sel := min(auto_arima_d_from_tests(y, allocator), max(max_d, 0))

	for p in 0 ..= max_p {
		for d in d_sel ..= d_sel {
			for q in 0 ..= max_q {

				// skip the trivial (0,0,0)
				if p == 0 && d == 0 && q == 0 {
					continue
				}

				fit := arima_fit(y, p, d, q, allocator)
				if !fit.converged {
					continue
				}
				score: f64
				if criterion == "bic" {
					score = fit.bic
				} else {
					score = fit.aic
				}

				if score < best_score {
					best_score = score
					best = ArimaAutoResult {
						p   = p,
						d   = d,
						q   = q,
						fit = fit,
					}
				}
			}
		}
	}

	return best
}


auto_arima_d_from_tests :: proc(y: []f64, allocator: mem.Allocator = context.allocator) -> int {
	decision, _, _, _, _ := stationarity_test(
		y,
		10, // ADF max lags
		.Constant, // ADF regression type
		.AIC, // ADF lag selection
		.Level, // KPSS type
		allocator,
	)

	switch decision {
	case .Stationary:
		return 0
	case .TrendStationary:
		return 0
	case .DifferenceStationary:
		return 1
	case .Inconclusive:
		return 1
	}

	return 1
}

arima_auto_with_tests :: proc(
	y: []f64,
	max_p: int,
	max_q: int,
	criterion: string = "aic",
	allocator: mem.Allocator = context.allocator,
) -> ArimaAutoResult {

	// --- 1) Determine differencing order using ADF + KPSS ---
	d := auto_arima_d_from_tests(y, allocator)

	// --- 2) Run ARIMA auto-selection with fixed d ---
	best: ArimaAutoResult
	best_score := math.INF_F64

	for p in 0 ..= max_p {
		for q in 0 ..= max_q {

			// skip trivial ARIMA(0,d,0)
			if p == 0 && q == 0 {
				continue
			}

			fit := arima_fit(y, p, d, q, allocator)
			if !fit.converged {
				continue
			}

			score: f64
			if criterion == "bic" {
				score = fit.bic
			} else {
				score = fit.aic
			}

			if score < best_score {
				best_score = score
				best = ArimaAutoResult {
					p   = p,
					d   = d,
					q   = q,
					fit = fit,
				}
			}
		}
	}

	return best
}
df_arima_auto_with_tests :: proc(
	y: []f64,
	max_p: int,
	max_q: int,
	criterion: string = "aic",
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {

	result := arima_auto_with_tests(y, max_p, max_q, criterion, allocator)

	df := w.dataframe_new()
	w.add_column(&df, w.column_from_ints("p", []int{result.p}))
	w.add_column(&df, w.column_from_ints("d", []int{result.d}))
	w.add_column(&df, w.column_from_ints("q", []int{result.q}))
	w.add_column(&df, w.column_from_floats("aic", []f64{result.fit.aic}))
	w.add_column(&df, w.column_from_floats("bic", []f64{result.fit.bic}))
	w.add_column(&df, w.column_from_floats("loglik", []f64{result.fit.loglik}))
	df.rows = 1
	return df
}
// Expand multiplicative SARIMA(p,d,q)(P,D,Q)_s into equivalent ARMA(P*,Q*)
// using your convention: y_t = sum(phi[i] * y_{t-i}) + e_t + sum(theta[j] * e_{t-j})
sarima_expand_to_arma :: proc(
	phi: []f64, // non-seasonal AR (length p)
	Phi: []f64, // seasonal AR (length P)
	theta: []f64, // non-seasonal MA (length q)
	Theta: []f64, // seasonal MA (length Q)
	s: int, // season length
	allocator: mem.Allocator = context.allocator,
) -> (
	ar: []f64,
	ma: []f64,
) {
	p := len(phi)
	P := len(Phi) if s > 0 else 0 // no seasonal terms without a season length
	q := len(theta)
	Q := len(Theta) if s > 0 else 0

	// --- AR polynomial: (1 - φ(L)) (1 - Φ(L^s)) ---
	// poly_ar(k) corresponds to coefficient of L^k
	max_ar_deg := max(p, 0) + max(P, 0) * s
	poly_ar := make([]f64, max_ar_deg + 1, allocator)
	poly_ar[0] = 1.0

	// non-seasonal AR: (1 - φ_1 L - ... - φ_p L^p)
	for i in 0 ..< p {
		poly_ar[i + 1] += -phi[i]
	}

	// seasonal AR: (1 - Φ_1 L^s - ... - Φ_P L^{Ps})
	poly_sar := make([]f64, max_ar_deg + 1, allocator)
	poly_sar[0] = 1.0
	for i in 0 ..< P {
		k := (i + 1) * s
		if k <= max_ar_deg {
			poly_sar[k] += -Phi[i]
		}
	}

	// convolve poly_ar * poly_sar
	poly_ar_full := make([]f64, max_ar_deg + 1, allocator)
	for i in 0 ..< len(poly_ar) {
		for j in 0 ..< len(poly_sar) {
			k := i + j
			if k > max_ar_deg {
				break
			}
			poly_ar_full[k] += poly_ar[i] * poly_sar[j]
		}
	}

	// convert back to AR coefficients in your convention:
	// y_t = sum(ar[k-1] * y_{t-k}) + ...
	ar_deg := max_ar_deg
	if ar_deg <= 0 {
		ar = make([]f64, 0, allocator)
	} else {
		ar = make([]f64, ar_deg, allocator)
		for k in 1 ..= ar_deg {
			ar[k - 1] = -poly_ar_full[k]
		}
	}

	// --- MA polynomial: (1 + θ(L)) (1 + Θ(L^s)) ---
	max_ma_deg := max(q, 0) + max(Q, 0) * s
	poly_ma := make([]f64, max_ma_deg + 1, allocator)
	poly_ma[0] = 1.0

	// non-seasonal MA: (1 + θ_1 L + ... + θ_q L^q)
	for i in 0 ..< q {
		poly_ma[i + 1] += theta[i]
	}

	// seasonal MA: (1 + Θ_1 L^s + ... + Θ_Q L^{Qs})
	poly_sma := make([]f64, max_ma_deg + 1, allocator)
	poly_sma[0] = 1.0
	for i in 0 ..< Q {
		k := (i + 1) * s
		if k <= max_ma_deg {
			poly_sma[k] += Theta[i]
		}
	}

	// convolve poly_ma * poly_sma
	poly_ma_full := make([]f64, max_ma_deg + 1, allocator)
	for i in 0 ..< len(poly_ma) {
		for j in 0 ..< len(poly_sma) {
			k := i + j
			if k > max_ma_deg {
				break
			}
			poly_ma_full[k] += poly_ma[i] * poly_sma[j]
		}
	}

	// convert back to MA coefficients in your convention:
	// ... + e_t + sum(ma[k-1] * e_{t-k})
	ma_deg := max_ma_deg
	if ma_deg <= 0 {
		ma = make([]f64, 0, allocator)
	} else {
		ma = make([]f64, ma_deg, allocator)
		for k in 1 ..= ma_deg {
			ma[k - 1] = poly_ma_full[k]
		}
	}

	return
}

// SARIMA state-space: differencing (d,D,s) is handled outside;
// here we only encode the ARMA part implied by (p,q)(P,Q)_s.
sarima_state_space :: proc(
	phi: []f64, // non-seasonal AR
	theta: []f64, // non-seasonal MA
	Phi: []f64, // seasonal AR
	Theta: []f64, // seasonal MA
	s: int, // season length
	sigma2: f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	F, Q, P0: []f64,
	H: []f64,
	R: []f64,
	x0: []f64,
	N: int,
) {
	ar, ma := sarima_expand_to_arma(phi, Phi, theta, Theta, s, allocator)
	return arma_pq_fundamental_state_space(ar, ma, sigma2, allocator)
}
seasonal_difference :: proc(
	series: []f64,
	D: int,
	s: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {

	if D <= 0 || s <= 0 { 	// s <= 0: no seasonal differencing
		out := make([]f64, len(series), allocator)
		for i in 0 ..< len(series) {
			out[i] = series[i]
		}
		return out
	}

	diff := series

	for d in 0 ..< D {
		if len(diff) <= s {
			break
		}

		next := make([]f64, len(diff) - s, allocator)
		for i in s ..< len(diff) {
			next[i - s] = diff[i] - diff[i - s]
		}
		diff = next
	}

	return diff
}

seasonal_difference_with_history :: proc(
	series: []f64,
	D: int,
	s: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	diff: []f64,
	history: []f64,
) {

	if D <= 0 {
		diff = make([]f64, len(series), allocator)
		for i in 0 ..< len(series) {
			diff[i] = series[i]
		}
		history = make([]f64, 0, allocator)
		return
	}

	// store last D*s values
	hist_len := D * s
	if hist_len > len(series) {
		hist_len = len(series)
	}

	history = make([]f64, hist_len, allocator)
	for i in 0 ..< hist_len {
		history[i] = series[i]
	}

	diff = seasonal_difference(series, D, s, allocator)
	return
}

seasonal_inverse_difference :: proc(
	diff: []f64,
	history: []f64,
	D: int,
	s: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {

	if D <= 0 {
		out := make([]f64, len(diff), allocator)
		for i in 0 ..< len(diff) {
			out[i] = diff[i]
		}
		return out
	}

	// history = the first D*s original values; y_t = z_t - Σ c_k y_{t-k} with
	// c = (1 - L^s)^D (lags before the stored history count as 0)
	c := _arima_diff_poly(0, D, s, context.temp_allocator)
	defer delete(c, context.temp_allocator)
	return _arima_undifference(diff, history, c, allocator)
}
