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
	aic:       f64,
	bic:       f64,
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
	// compute AIC and BIC
	k := p + q + 1
	n := len(y)
	if d > 0 {
		n = len(difference(y, d, allocator))
	}

	res.aic = -2.0 * res.loglik + 2.0 * f64(k)
	res.bic = -2.0 * res.loglik + f64(k) * math.ln(f64(n))

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

	// If d > 0, integrate forecasts back to original scale
	if d > 0 {
		history := make([]f64, d, allocator)
		for i in 0 ..< d {
			history[i] = y[len(y) - d + i]
		}

		mean = inverse_difference(mean, history, d, allocator)
		lower = inverse_difference(lower, history, d, allocator)
		upper = inverse_difference(upper, history, d, allocator)
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

	// --- Special cases: pure AR or pure MA ---

	if q == 0 {
		// Pure AR(p): standard companion form, noise directly in y_t
		N = max(p, 1)

		F = make([]f64, N * N, allocator)
		Q = make([]f64, N * N, allocator)
		P0 = make([]f64, N * N, allocator)
		H = make([]f64, N, allocator)
		R = make([]f64, 1, allocator)
		x0 = make([]f64, N, allocator)

		// First row: AR coeffs
		for i in 0 ..< p {
			F[0 * N + i] = phi[i]
		}
		// Shift y-lags
		for i in 1 ..< p {
			F[i * N + (i - 1)] = 1.0
		}

		// Observation: y_t = state[0]
		H[0] = 1.0

		// Noise: innovation enters y_t directly
		G := make([]f64, N, allocator)
		G[0] = 1.0
		for i in 0 ..< N {
			for j in 0 ..< N {
				Q[i * N + j] = sigma2 * G[i] * G[j]
			}
		}

		R[0] = 0.0
		for i in 0 ..< N {
			P0[i * N + i] = 1e6
		}
		return
	}

	if p == 0 {
		// Pure MA(q): state = [ε_t, ε_{t-1}, ..., ε_{t-q+1}]
		N = q

		F = make([]f64, N * N, allocator)
		Q = make([]f64, N * N, allocator)
		P0 = make([]f64, N * N, allocator)
		H = make([]f64, N, allocator)
		R = make([]f64, 1, allocator)
		x0 = make([]f64, N, allocator)

		// ε_{t+1} = η_{t+1}
		// shift ε-lags
		for j in 1 ..< q {
			F[j * N + (j - 1)] = 1.0
		}

		// Observation: y_t = ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q}
		// state[0] = ε_t, state[1] = ε_{t-1}, ...
		H[0] = 1.0
		for j in 0 ..< q {
			// θ_j corresponds to ε_{t-j-1} in state index j+1
			if j + 1 < N {
				H[j + 1] += theta[j]
			}
		}

		// Noise: innovation only in ε_{t+1} (next state[0])
		G := make([]f64, N, allocator)
		G[0] = 1.0
		for i in 0 ..< N {
			for j in 0 ..< N {
				Q[i * N + j] = sigma2 * G[i] * G[j]
			}
		}

		R[0] = 0.0
		for i in 0 ..< N {
			P0[i * N + i] = 1e6
		}
		return
	}

	// --- General ARMA(p,q) ---

	N = p + q

	F = make([]f64, N * N, allocator)
	Q = make([]f64, N * N, allocator)
	P0 = make([]f64, N * N, allocator)
	H = make([]f64, N, allocator)
	R = make([]f64, 1, allocator)
	x0 = make([]f64, N, allocator)

	// State layout:
	// [ y_t, y_{t-1}, ..., y_{t-p+1}, ε_t, ε_{t-1}, ..., ε_{t-q+1} ]
	// indices: 0 .. p-1 for y-lags, p .. p+q-1 for ε-lags

	// --- Transition for y_{t+1} (row 0) ---

	// AR part on y-lags
	for i in 0 ..< p {
		F[0 * N + i] = phi[i]
	}

	// MA part on ε-lags: θ_1 ε_t + ... + θ_q ε_{t-q+1}
	// ε_t is state[p + 0], ε_{t-1} is state[p + 1], ...
	for j in 0 ..< q {
		F[0 * N + (p + j)] = theta[j]
	}
	// The current innovation ε_{t+1} = η_{t+1} will enter via G[0] = 1.0

	// --- Shift y-lags ---
	// y_{t} -> y_{t+1-1}, y_{t-1} -> y_{t+1-2}, ...
	for i in 1 ..< p {
		F[i * N + (i - 1)] = 1.0
	}

	// --- ε_{t+1} dynamics and shifts ---
	// ε_{t+1} = η_{t+1}  (no dependence on previous state)
	// so row p is all zeros; noise will enter via G[p] = 1.0

	// shift ε-lags: ε_t -> ε_{t+1-1}, ε_{t-1} -> ε_{t+1-2}, ...
	for j in 1 ..< q {
		row := p + j
		col := p + j - 1
		F[row * N + col] = 1.0
	}

	// --- Observation matrix ---
	// y_t = state[0]
	H[0] = 1.0

	// --- Process noise: η_{t+1} = ε_{t+1} ---
	// enters ε_{t+1} (state index p) and y_{t+1} with coefficient 1
	G := make([]f64, N, allocator)
	G[0] = 1.0 // innovation contributes directly to y_{t+1}
	G[p] = 1.0 // and defines ε_{t+1}

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

arma22_simulate :: proc(
	phi: []f64,
	theta: []f64,
	sigma2: f64,
	T: int,
	allocator: mem.Allocator,
) -> []f64 {
	y := make([]f64, T, allocator)
	e_prev := make([]f64, len(theta) + 1, allocator) // e_t, e_{t-1}, ...
	y_prev := make([]f64, len(phi) + 1, allocator) // y_t, y_{t-1}, ...

	sigma := math.sqrt_f64(sigma2)

	for t in 0 ..< T {
		e := rand.float64_normal(0.0, sigma)

		// AR part
		ar := 0.0
		for i in 0 ..< len(phi) {
			ar += phi[i] * y_prev[i]
		}

		// MA part
		ma := e
		for j in 0 ..< len(theta) {
			ma += theta[j] * e_prev[j]
		}

		y[t] = ar + ma

		// shift histories
		for i := len(y_prev) - 1; i > 0; i -= 1 {
			y_prev[i] = y_prev[i - 1]
		}
		y_prev[0] = y[t]

		for j := len(e_prev) - 1; j > 0; j -= 1 {
			e_prev[j] = e_prev[j - 1]
		}
		e_prev[0] = e
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

	for p in 0 ..= max_p {
		for d in 0 ..= max_d {
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
) -> DataFrame {

	result := arima_auto_with_tests(y, max_p, max_q, criterion, allocator)

	df := dataframe_new()
	add_column(&df, column_from_ints("p", []int{result.p}))
	add_column(&df, column_from_ints("d", []int{result.d}))
	add_column(&df, column_from_ints("q", []int{result.q}))
	add_column(&df, column_from_floats("aic", []f64{result.fit.aic}))
	add_column(&df, column_from_floats("bic", []f64{result.fit.bic}))
	add_column(&df, column_from_floats("loglik", []f64{result.fit.loglik}))
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
	P := len(Phi)
	q := len(theta)
	Q := len(Theta)

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

	if D <= 0 {
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

	// Start with stored history
	out := make([]f64, len(diff) + len(history), allocator)
	for i in 0 ..< len(history) {
		out[i] = history[i]
	}

	// First seasonal integration
	for i in 0 ..< len(diff) {
		out[len(history) + i] = diff[i] + out[len(history) + i - s]
	}

	// Higher-order seasonal integration
	for d in 1 ..< D {
		for i in len(history) ..< len(out) {
			out[i] = out[i] + out[i - s]
		}
	}

	return out
}
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
	hist_ns: []f64, // non-seasonal history
	hist_s: []f64, // seasonal history
	d: int,
	D: int,
	s: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {

	y1 := inverse_difference(diff, hist_ns, d, allocator)
	y2 := seasonal_inverse_difference(y1, hist_s, D, s, allocator)
	return y2
}
