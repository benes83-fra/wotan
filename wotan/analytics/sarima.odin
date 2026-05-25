package analytics

import w "../core"
import l "../linalg"
import "core:math"
import "core:math/rand"
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

	// 1) build differenced series z_t
	y_ns := difference(y, d, allocator)
	z := seasonal_difference(y_ns, D, s, allocator)

	// 2) state-space for SARIMA ARMA part
	F, Q, P0, H, R, x0, N := sarima_state_space(phi, theta, Phi, Theta, s, sigma2, allocator)

	// 3) filter to get last state
	xT, PT := kalman_filter_last_scalar(z, F, Q, P0, H, R, x0, N)

	// 4) forecasting on differenced scale
	mean_d := make([]f64, h, allocator)
	lower_d := make([]f64, h, allocator)
	upper_d := make([]f64, h, allocator)

	z_state := make([]f64, N, allocator)
	P := make([]f64, N * N, allocator)
	copy(z_state, xT)
	copy(P, PT)

	// crude z for alpha (same as arima_forecast)
	z_alpha := 1.96
	_ = alpha // keep signature; you can plug proper quantile later

	for step in 0 ..< h {
		// --- Optimized Predict Step: x_pred = F*x, P_pred = F*P*Fᵀ + Q ---
		if _kalman_use_optimized(N) {
			// Convert flat arrays to dynamic matrices for optimized linalg
			F_mat := _matrix_from_flat(F, N, N, context.temp_allocator)

			P_mat := _matrix_from_flat(P, N, N, context.temp_allocator)

			Q_mat := _matrix_from_flat(Q, N, N, context.temp_allocator)

			// x_pred = F * x using SIMD matvec
			x_pred_vec := l.matvec_dyn_simd(&F_mat, z_state, context.temp_allocator)

			// P_pred = F * P * Fᵀ + Q
			// Step 1: FP = F * P (P is symmetric, so Pᵀ = P)
			FP := l.matmul_dyn_simd(&F_mat, &P_mat, context.temp_allocator)

			// Step 2: P_pred = FP * Fᵀ + Q (matmul_dyn_simd computes FP * Fᵀ directly)
			P_pred_mat := l.matmul_dyn_simd(&FP, &F_mat, context.temp_allocator)

			// Add Q
			for i in 0 ..< N * N {P_pred_mat.data[i] += Q_mat.data[i]}

			// Copy results back to flat arrays for next iteration
			copy(z_state, x_pred_vec)
			for i in 0 ..< N * N {P[i] = P_pred_mat.data[i]}

			// Clean up temporaries immediately (no defer in loop)
			l.matrix_free(&F_mat)
			l.matrix_free(&P_mat)
			l.matrix_free(&Q_mat)
			delete(x_pred_vec, context.temp_allocator)
			l.matrix_free(&FP)
			l.matrix_free(&P_pred_mat)
		} else {
			// Built-in Odin ops for small N (≤8) - keep original manual loops
			// x_pred = F * x
			x_pred := make([]f64, N, allocator)
			for i in 0 ..< N {
				sv := 0.0
				for j in 0 ..< N {
					sv += F[i * N + j] * z_state[j]
				}
				x_pred[i] = sv
			}

			// P_pred = F * P * Fᵀ + Q
			P_pred := make([]f64, N * N, allocator)
			temp := make([]f64, N * N, allocator)
			for i in 0 ..< N {
				for j in 0 ..< N {
					sv := 0.0
					for k in 0 ..< N {
						sv += F[i * N + k] * P[k * N + j]
					}
					temp[i * N + j] = sv
				}
			}
			for i in 0 ..< N {
				for j in 0 ..< N {
					sv := 0.0
					for k in 0 ..< N {
						sv += temp[i * N + k] * F[j * N + k]
					}
					P_pred[i * N + j] = sv + Q[i * N + j]
				}
			}

			// Copy results back
			copy(z_state, x_pred)
			copy(P, P_pred)

			// Clean up
			delete(x_pred, allocator)
			delete(P_pred, allocator)
			delete(temp, allocator)
		}

		// --- observation mean/variance on differenced scale ---
		// (unchanged: this is O(N) and not a bottleneck)
		mu := 0.0
		for j in 0 ..< N {
			mu += H[j] * z_state[j]
		}

		Sv := 0.0
		for i in 0 ..< N {
			for j in 0 ..< N {
				Sv += H[i] * P[i * N + j] * H[j]
			}
		}
		Sv += R[0]

		sd := math.sqrt_f64(Sv)

		mean_d[step] = mu
		lower_d[step] = mu - z_alpha * sd
		upper_d[step] = mu + z_alpha * sd
	}

	// 5) build histories for inverse differencing

	// non-seasonal history: last d values of original y
	hist_ns := make([]f64, d, allocator)
	for i in 0 ..< d {
		if d == 0 {
			break
		}
		hist_ns[i] = y[len(y) - d + i]
	}

	// seasonal history: last min(D*s, len(y_ns)) values of non-seasonally differenced series
	hist_s_len := D * s
	if hist_s_len > len(y_ns) {
		hist_s_len = len(y_ns)
	}
	hist_s := make([]f64, hist_s_len, allocator)
	for i in 0 ..< hist_s_len {
		hist_s[i] = y_ns[len(y_ns) - hist_s_len + i]
	}

	// 6) invert differencing back to original scale
	mean_full := sarima_inverse_difference(mean_d, hist_ns, hist_s, d, D, s, allocator)
	lower_full := sarima_inverse_difference(lower_d, hist_ns, hist_s, d, D, s, allocator)
	upper_full := sarima_inverse_difference(upper_d, hist_ns, hist_s, d, D, s, allocator)

	// keep only the forecast horizon (drop the history prefix)
	start_idx := len(hist_ns) + len(hist_s) // prefix length in reconstruction
	if start_idx > len(mean_full) {
		start_idx = len(mean_full)
	}

	out_mean := make([]f64, h, allocator)
	out_lower := make([]f64, h, allocator)
	out_upper := make([]f64, h, allocator)

	for i in 0 ..< h {
		idx := start_idx + i
		if idx >= len(mean_full) {
			break
		}
		out_mean[i] = mean_full[idx]
		out_lower[i] = lower_full[idx]
		out_upper[i] = upper_full[idx]
	}

	return SarimaForecastResult{mean = out_mean, lower = out_lower, upper = out_upper}
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
	fit:     ArimaFitResult,
}

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
	d := auto_arima_d_from_tests(y, allocator)

	// --- 2) Determine seasonal differencing D ---
	// Simple rule: if KPSS rejects on seasonal lag s, set D=1
	D := 0
	if len(y) > 2 * s {
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

	// --- 3) Full grid search ---
	for p in 0 ..= max_p {
		for q in 0 ..= max_q {
			for P in 0 ..= max_P {
				for Q in 0 ..= max_Q {
					for D_try in 0 ..= max_D {

						// skip trivial model
						if p == 0 && q == 0 && P == 0 && Q == 0 {
							continue
						}

						// 3a) difference the data
						y_d := sarima_difference(y, d, D_try, s, allocator)

						if len(y_d) < (p + q + P + Q + 5) {
							continue
						}

						// 3b) expand SARIMA to ARMA
						ar, ma := sarima_expand_to_arma(
							make([]f64, p, allocator),
							make([]f64, P, allocator),
							make([]f64, q, allocator),
							make([]f64, Q, allocator),
							s,
							allocator,
						)

						// 3c) fit ARMA on differenced data
						fit := arima_fit(y_d, len(ar), 0, len(ma), allocator)
						if !fit.converged {
							continue
						}

						// 3d) compute score
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
								D   = D_try,
								Q   = Q,
								s   = s,
								fit = fit,
							}
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

	allocator := data.allocator

	// unpack parameters
	phi := make([]f64, p, allocator)
	theta := make([]f64, q, allocator)
	Phi := make([]f64, P, allocator)
	Theta := make([]f64, Q, allocator)

	idx := 0
	for i in 0 ..< p {
		phi[i] = 0.95 * math.tanh(params[idx])
		idx += 1
	}
	for i in 0 ..< q {
		theta[i] = 0.95 * math.tanh(params[idx])
		idx += 1
	}
	for i in 0 ..< P {
		Phi[i] = 0.95 * math.tanh(params[idx])
		idx += 1
	}
	for i in 0 ..< Q {
		Theta[i] = 0.95 * math.tanh(params[idx])
		idx += 1
	}

	log_sig2 := params[idx]
	sigma2 := math.exp_f64(log_sig2)
	if sigma2 <= 0.0 || sigma2 > 1e3 {
		return 1e9
	}

	// differencing
	y1 := difference(data.y, data.d, allocator)
	y2 := data.y_diff
	if len(y2) < (p + q + P + Q + 5) {
		return 1e9
	}


	// state-space
	F, Qm, P0, H, R, x0, N := sarima_state_space(phi, theta, Phi, Theta, data.s, sigma2, allocator)

	ll := kalman_loglik_scalar(y2, F, Qm, P0, H, R, x0, N)

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
sarima_fit :: proc(
	y: []f64,
	p, d, q: int,
	P, D, Q: int,
	s: int,
	allocator: mem.Allocator = context.allocator,
) -> SarimaFitResult {

	y1 := difference(y, d, allocator)
	y2 := seasonal_difference(y1, D, s, allocator)

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

	n_params := p + q + P + Q + 1
	best_params := make([]f64, n_params, allocator)
	best_f := math.INF_F64

	max_iter := 800
	tol := 1e-6
	n_starts := 5

	for s_i in 0 ..< n_starts {
		x0 := make([]f64, n_params, allocator)
		// random init
		for i in 0 ..< n_params {
			x0[i] = rand.float64_normal(0.0, 0.3)
		}

		x_opt, f_opt := nelder_mead(sarima_neg_loglik_obj, &ctx, x0, max_iter, tol, allocator)

		if f_opt < best_f {
			best_f = f_opt
			copy(best_params, x_opt)
		}
	}

	// unpack best parameters
	idx := 0
	phi := make([]f64, p, allocator)
	theta := make([]f64, q, allocator)
	Phi := make([]f64, P, allocator)
	Theta := make([]f64, Q, allocator)

	for i in 0 ..< p {
		phi[i] = 0.95 * math.tanh(best_params[idx])
		idx += 1
	}
	for i in 0 ..< q {
		theta[i] = 0.95 * math.tanh(best_params[idx])
		idx += 1
	}
	for i in 0 ..< P {
		Phi[i] = 0.95 * math.tanh(best_params[idx])
		idx += 1
	}
	for i in 0 ..< Q {
		Theta[i] = 0.95 * math.tanh(best_params[idx])
		idx += 1
	}

	sigma2 := math.exp_f64(best_params[idx])

	// compute loglik
	// y1 := difference(y, d, allocator)
	// y2 := seasonal_difference(y1, D, s, allocator)

	F, Qm, P0, H, R, x0, N := sarima_state_space(phi, theta, Phi, Theta, s, sigma2, allocator)

	ll := kalman_loglik_scalar(y2, F, Qm, P0, H, R, x0, N)

	// AIC/BIC
	k := n_params
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
		converged = true,
	}
}


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
	// 1) differenced series
	y1 := difference(y, d, allocator)
	z := seasonal_difference(y1, D, s, allocator)

	// 2) state-space
	F, Q, P0, H, R, x0, N := sarima_state_space(phi, theta, Phi, Theta, s, sigma2, allocator)

	T := len(z)
	if T == 0 {
		return make([]f64, 0, allocator)
	}

	burn_in := 20
	if burn_in >= T {
		burn_in = 0
	}

	// state
	x := make([]f64, N, allocator)
	P := make([]f64, N * N, allocator)
	for i in 0 ..< N {
		x[i] = x0[i]
	}
	for i in 0 ..< N * N {
		P[i] = P0[i]
	}

	// store innovations after burn-in
	out_len := T - burn_in
	if out_len < 0 {
		out_len = 0
	}
	v_out := make([]f64, out_len, allocator)
	idx := 0

	for t in 0 ..< T {
		// --- Predict ---
		x_pred := make([]f64, N, allocator)
		for i in 0 ..< N {
			ssum := 0.0
			for j in 0 ..< N {
				ssum += F[i * N + j] * x[j]
			}
			x_pred[i] = ssum
		}

		P_pred := make([]f64, N * N, allocator)
		temp := make([]f64, N * N, allocator)
		for i in 0 ..< N {
			for j in 0 ..< N {
				ssum := 0.0
				for k in 0 ..< N {
					ssum += F[i * N + k] * P[k * N + j]
				}
				temp[i * N + j] = ssum
			}
		}
		for i in 0 ..< N {
			for j in 0 ..< N {
				ssum := 0.0
				for k in 0 ..< N {
					ssum += temp[i * N + k] * F[j * N + k]
				}
				P_pred[i * N + j] = ssum + Q[i * N + j]
			}
		}

		// --- Innovation ---
		y_pred := 0.0
		for j in 0 ..< N {
			y_pred += H[j] * x_pred[j]
		}
		v := z[t] - y_pred

		// S = H P_pred Hᵀ + R
		S := 0.0
		for i in 0 ..< N {
			for j in 0 ..< N {
				S += H[i] * P_pred[i * N + j] * H[j]
			}
		}
		S += R[0]

		// store innovation (raw, not standardized) after burn-in
		if t >= burn_in && idx < out_len {
			v_out[idx] = v
			idx += 1
		}

		// --- Update ---
		K := make([]f64, N, allocator)
		for i in 0 ..< N {
			ssum := 0.0
			for j in 0 ..< N {
				ssum += P_pred[i * N + j] * H[j]
			}
			K[i] = ssum / S
		}

		for i in 0 ..< N {
			x[i] = x_pred[i] + K[i] * v
		}

		for i in 0 ..< N {
			for j in 0 ..< N {
				P[i * N + j] = P_pred[i * N + j] - K[i] * S * K[j]
			}
		}
	}

	return v_out[:idx]
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
	// 1) differenced series
	y1 := difference(y, d, allocator)
	y2 := seasonal_difference(y1, D, s, allocator)

	if len(y2) == 0 {
		return make([]f64, 0, allocator)
	}

	// 2) state-space
	F, Q, P0, H, R, x0, N := sarima_state_space(phi, theta, Phi, Theta, s, sigma2, allocator)

	T := len(y2)
	burn_in := 20
	if burn_in >= T {
		burn_in = 0
	}

	// state
	x := make([]f64, N, allocator)
	P := make([]f64, N * N, allocator)
	for i in 0 ..< N {
		x[i] = x0[i]
	}
	for i in 0 ..< N * N {
		P[i] = P0[i]
	}

	// residuals after burn-in
	out_len := T - burn_in
	if out_len < 0 {
		out_len = 0
	}
	res := make([]f64, out_len, allocator)
	idx := 0

	for t in 0 ..< T {
		// --- Predict ---
		x_pred := make([]f64, N, allocator)
		for i in 0 ..< N {
			ssum := 0.0
			for j in 0 ..< N {
				ssum += F[i * N + j] * x[j]
			}
			x_pred[i] = ssum
		}

		P_pred := make([]f64, N * N, allocator)
		temp := make([]f64, N * N, allocator)
		for i in 0 ..< N {
			for j in 0 ..< N {
				ssum := 0.0
				for k in 0 ..< N {
					ssum += F[i * N + k] * P[k * N + j]
				}
				temp[i * N + j] = ssum
			}
		}
		for i in 0 ..< N {
			for j in 0 ..< N {
				ssum := 0.0
				for k in 0 ..< N {
					ssum += temp[i * N + k] * F[j * N + k]
				}
				P_pred[i * N + j] = ssum + Q[i * N + j]
			}
		}

		// --- Innovation ---
		y_pred := 0.0
		for j in 0 ..< N {
			y_pred += H[j] * x_pred[j]
		}
		v := y2[t] - y_pred

		if t >= burn_in && idx < out_len {
			res[idx] = v
			idx += 1
		}

		// S and Kalman gain
		S := 0.0
		for i in 0 ..< N {
			for j in 0 ..< N {
				S += H[i] * P_pred[i * N + j] * H[j]
			}
		}
		S += R[0]

		K := make([]f64, N, allocator)
		for i in 0 ..< N {
			ssum := 0.0
			for j in 0 ..< N {
				ssum += P_pred[i * N + j] * H[j]
			}
			K[i] = ssum / S
		}

		// update
		for i in 0 ..< N {
			x[i] = x_pred[i] + K[i] * v
		}
		for i in 0 ..< N {
			for j in 0 ..< N {
				P[i * N + j] = P_pred[i * N + j] - K[i] * S * K[j]
			}
		}
	}

	return res
}
