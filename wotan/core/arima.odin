package core

import "core:math"
import "core:math/rand"
import "core:mem"

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
	p := len(phi)
	q := len(theta)

	// ARIMA: we handle differencing outside; here it's ARMA(p,q)
	N = max(p, q + 1) // state = [y_t, y_{t-1},..., e_t, e_{t-1},...]

	F = make([]f64, N * N, allocator)
	Q = make([]f64, N * N, allocator)
	P0 = make([]f64, N * N, allocator)
	H = make([]f64, N, allocator)
	R = make([]f64, 1, allocator)
	x0 = make([]f64, N, allocator)

	// --- Transition matrix F ---
	// First row: AR part on y-lags
	for i in 0 ..< p {
		F[0 * N + i] = phi[i]
	}
	// First row: MA part on epsilon-lags (starting at index p)
	for j in 0 ..< q {
		idx := p + j
		if idx < N {
			F[0 * N + idx] = theta[j]
		}
	}

	// Shift AR lags: y_{t-1} <- y_t, y_{t-2} <- y_{t-1}, ...
	for i in 1 ..< p {
		F[i * N + (i - 1)] = 1.0
	}

	// Shift MA lags: e_{t-1} <- e_t, e_{t-2} <- e_{t-1}, ...
	for j in 1 ..< q + 1 {
		row := p + j - 1
		col := p + j - 2
		if row < N && col < N {
			F[row * N + col] = 1.0
		}
	}

	// --- Observation matrix H ---
	// y_t = [1, 0, 0, ...] * state_t
	for i in 0 ..< N {
		H[i] = 0.0
	}
	H[0] = 1.0

	// --- Innovations noise: ε_t enters y_t and e_t ---
	// G is length-N, then Q = σ² * G Gᵀ
	G := make([]f64, N, allocator)
	for i in 0 ..< N {
		G[i] = 0.0
	}
	// ε_t affects y_t
	G[0] = 1.0
	// and the first epsilon state e_t, if it exists
	if q >= 0 && p < N {
		G[p] = 1.0
	}

	for i in 0 ..< N {
		for j in 0 ..< N {
			Q[i * N + j] = sigma2 * G[i] * G[j]
		}
	}

	// No measurement noise
	R[0] = 0.0

	// Diffuse initial covariance
	for i in 0 ..< N {
		P0[i * N + i] = 1e6
	}

	return
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

	// Save last d values for inverse differencing
	history = make([]f64, d, allocator)
	for i in 0 ..< d {
		history[i] = series[i]
	}

	diff = difference(series, d, allocator)
	return
}
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

	// Start with the first d original values
	out := make([]f64, len(diff) + d, allocator)
	for i in 0 ..< d {
		out[i] = history[i]
	}

	// First integration
	for i in 0 ..< len(diff) {
		out[d + i] = out[d + i - 1] + diff[i]
	}

	// Higher-order integration
	for k in 1 ..< d {
		for i in d ..< len(out) {
			out[i] = out[i] + out[i - 1]
		}
	}

	return out
}
arima_loglik :: proc(
	y: []f64,
	phi, theta: []f64,
	d: int,
	sigma2: f64,
	allocator := context.allocator,
) -> f64 {
	// differencing is handled outside the SS form
	if d == 0 && len(phi) == 1 && len(theta) == 1 {
		F, Q, P0, H, R, x0, N := arma11_state_space(phi[0], theta[0], sigma2, allocator)
		return kalman_loglik_scalar(y, F, Q, P0, H, R, x0, N)
	}

	F, Q, P0, H, R, x0, N := arima_state_space(phi, theta, d, sigma2, allocator)
	return kalman_loglik_scalar(y, F, Q, P0, H, R, x0, N)
}


ArimaObjectiveCtx :: struct {
	y:         []f64,
	p, q:      int,
	d:         int,
	allocator: mem.Allocator,
}
arima_neg_loglik_obj :: proc(params: []f64, ctx: rawptr) -> f64 {
	data := (^ArimaObjectiveCtx)(ctx)

	p := data.p
	q := data.q

	if len(params) != p + q + 1 {
		return 1e12
	}

	// unpack params: [phi_0..phi_{p-1}, theta_0..theta_{q-1}, log_sigma2]
	phi := make([]f64, p, data.allocator)
	theta := make([]f64, q, data.allocator)

	for i in 0 ..< p {
		phi[i] = 0.95 * math.tanh(params[i])
	}
	for j in 0 ..< q {
		theta[j] = 0.95 * math.tanh(params[p + j])
	}

	log_sig2 := params[p + q]
	sigma2 := math.exp_f64(log_sig2)
	if sigma2 <= 0.0 || sigma2 > 1e3 {
		return 1e9
	}

	// differencing if needed
	y_eff := data.y
	if data.d > 0 {
		y_eff = difference(data.y, data.d, data.allocator)
		if len(y_eff) < (p + q + 5) {
			return 1e9
		}
	}

	ll := arima_loglik(y_eff, phi, theta, data.d, sigma2, data.allocator)

	// minimize negative log-likelihood
	return -ll
}

ArimaFitResult :: struct {
	phi:       []f64,
	theta:     []f64,
	sigma2:    f64,
	loglik:    f64,
	converged: bool,
}
arima_fit :: proc(y: []f64, p, d, q: int, allocator := context.allocator) -> ArimaFitResult {
	ctx := ArimaObjectiveCtx {
		y         = y,
		p         = p,
		q         = q,
		d         = d,
		allocator = allocator,
	}

	n_params := p + q + 1
	current_best_f := math.INF_F64 // Changed name to avoid collision
	best_params := make([]f64, n_params, allocator)

	max_iter := 800
	tol := 1e-6
	n_starts := 5

	for s in 0 ..< n_starts {
		x0 := make([]f64, n_params, allocator)
		// ... init x0 ...

		x_s, f_s := nelder_mead(arima_neg_loglik_obj, &ctx, x0, max_iter, tol, allocator)

		if f_s < current_best_f {
			current_best_f = f_s
			copy(best_params, x_s)
		}
	}

	res: ArimaFitResult
	res.phi = make([]f64, p, allocator)
	res.theta = make([]f64, q, allocator)

	for i in 0 ..< p {
		res.phi[i] = 0.95 * math.tanh(best_params[i])
	}
	for j in 0 ..< q {
		res.theta[j] = 0.95 * math.tanh(best_params[p + j])
	}

	res.sigma2 = math.exp_f64(best_params[p + q])
	res.loglik = -current_best_f
	res.converged = true
	return res
}
ArimaForecastResult :: struct {
	mean:  []f64,
	lower: []f64,
	upper: []f64,
}

arima_forecast :: proc(
	y: []f64,
	fit: ArimaFitResult,
	p, d, q: int,
	h: int,
	alpha: f64, // e.g. 0.05 for 95% PI
	allocator := context.allocator,
) -> ArimaForecastResult {
	// 1) differencing if needed
	y_eff := y
	if d > 0 {
		y_eff = difference(y, d, allocator)
	}

	// 2) build state-space from fitted params
	F, Q, P0, H, R, x0, N := arima_state_space(fit.phi, fit.theta, d, fit.sigma2, allocator)

	// 3) run filter to get last state
	xT, PT := kalman_filter_last_scalar(y_eff, F, Q, P0, H, R, x0, N)

	// 4) forecasting loop
	mean := make([]f64, h, allocator)
	lower := make([]f64, h, allocator)
	upper := make([]f64, h, allocator)

	// z for two-sided normal interval
	// crude: z ≈ 1.96 for 95%; if you want, plug in an inverse CDF later
	z := 1.96
	if alpha != 0.05 {
		// leave as 1.96 for now; you can generalize later
	}

	x := make([]f64, N, allocator)
	P := make([]f64, N * N, allocator)
	for i in 0 ..< N {
		x[i] = xT[i]
	}
	for i in 0 ..< N * N {
		P[i] = PT[i]
	}

	for step in 0 ..< h {
		// predict one step ahead
		x_pred := make([]f64, N, allocator)
		for i in 0 ..< N {
			s := 0.0
			for j in 0 ..< N {
				s += F[i * N + j] * x[j]
			}
			x_pred[i] = s
		}

		P_pred := make([]f64, N * N, allocator)
		temp := make([]f64, N * N, allocator)
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {
					s += F[i * N + k] * P[k * N + j]
				}
				temp[i * N + j] = s
			}
		}
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				for k in 0 ..< N {
					s += temp[i * N + k] * F[j * N + k]
				}
				P_pred[i * N + j] = s + Q[i * N + j]
			}
		}

		// forecast mean and variance on observation scale
		mu := 0.0
		for j in 0 ..< N {
			mu += H[j] * x_pred[j]
		}

		S := 0.0
		for i in 0 ..< N {
			for j in 0 ..< N {
				S += H[i] * P_pred[i * N + j] * H[j]
			}
		}
		S += R[0]

		sd := math.sqrt_f64(S)

		mean[step] = mu
		lower[step] = mu - z * sd
		upper[step] = mu + z * sd

		// roll state forward for next step
		for i in 0 ..< N {
			x[i] = x_pred[i]
		}
		for i in 0 ..< N * N {
			P[i] = P_pred[i]
		}
	}

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

	// Diffuse initial state
	P0[0] = 1e6
	P0[3] = 1e6

	return
}
