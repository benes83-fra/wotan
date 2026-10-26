package analytics

import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Regime-Switching GARCH(1,1) - 2 Regimes
// ============================================================================

RS_GARCH_Params :: struct {
	// Regime 1 (e.g., Low Volatility / "Calm")
	omega_1: f64,
	alpha_1: f64,
	beta_1:  f64,

	// Regime 2 (e.g., High Volatility / "Crisis")
	omega_2: f64,
	alpha_2: f64,
	beta_2:  f64,

	// Transition Probabilities
	// P(S_t = 1 | S_{t-1} = 1)
	p_11:    f64,
	// P(S_t = 2 | S_{t-1} = 2)
	p_22:    f64,
}

RS_GARCH_Result :: struct {
	params:           RS_GARCH_Params,
	filtered_probs:   [][]f64, // [time][regime] Probability of being in regime j at time t
	log_likelihood:   f64,
	converged:        bool,
	most_likely_path: []int, // Viterbi path: 0 or 1 for each time step
}

// ============================================================================
// Hamilton Filter (The Core Math)
// ============================================================================

// Regime variances follow the Haas-Mittnik-Paolella (2004) specification:
// each regime runs its own GARCH(1,1) recursion on the common returns,
//   σ²_{j,t} = ω_j + α_j y²_{t-1} + β_j σ²_{j,t-1},  σ²_{j,0} = mean(y²),
// so the state is just S_t and the standard Hamilton filter applies.
// The filter starts from the ergodic distribution π_1 = (1-p22)/(2-p11-p22)
// and works in log space so the filtered probabilities always sum to 1.
_rs_garch_regime_vars :: proc(params: RS_GARCH_Params, returns: []f64, t: int, var_1, var_2: ^f64) {
	if t > 0 {
		prev_y_sq := returns[t - 1] * returns[t - 1]
		var_1^ = params.omega_1 + params.alpha_1 * prev_y_sq + params.beta_1 * var_1^
		var_2^ = params.omega_2 + params.alpha_2 * prev_y_sq + params.beta_2 * var_2^
	}
	// Ensure variances are positive
	if var_1^ < 1e-10 {var_1^ = 1e-10}
	if var_2^ < 1e-10 {var_2^ = 1e-10}
}

_rs_garch_sample_var :: proc(returns: []f64) -> f64 {
	sample_var := 0.0
	for r in returns {sample_var += r * r}
	return sample_var / f64(len(returns))
}

_rs_garch_log_normal :: proc(y, v: f64) -> f64 {
	return -0.5 * (math.ln_f64(2.0 * math.PI * v) + y * y / v)
}

// Computes the log-likelihood and optionally filtered probabilities for a given set of parameters.
_rs_garch_hamilton_filter :: proc(
	params: RS_GARCH_Params,
	returns: []f64,
	allocator: mem.Allocator,
	compute_probs: bool = false, // Only allocate probs if true
) -> (
	loglik: f64,
	filtered_probs: [][]f64,
) {
	n := len(returns)
	if n < 2 {return 0.0, [][]f64{}}

	// Initialize filtered probabilities (ergodic / unconditional)
	denom := 2.0 - params.p_11 - params.p_22
	prob_1 := (1.0 - params.p_22) / denom if denom > 0.0 else 0.5
	prob_2 := 1.0 - prob_1

	// Only allocate filtered_probs if requested
	filtered_probs = [][]f64{}
	if compute_probs {
		filtered_probs = make([][]f64, n, allocator)
		for i in 0 ..< n {
			filtered_probs[i] = make([]f64, 2, allocator)
		}
	}

	// Initialize variances (using sample variance as a proxy)
	var_1 := _rs_garch_sample_var(returns)
	var_2 := var_1

	loglik = 0.0

	for t in 0 ..< n {
		// 1. Update Variances for each regime
		_rs_garch_regime_vars(params, returns, t, &var_1, &var_2)

		// 2. Predicted Probabilities P(S_t = j | y_{1:t-1})
		pred_1 := params.p_11 * prob_1 + (1.0 - params.p_22) * prob_2
		pred_2 := (1.0 - params.p_11) * prob_1 + params.p_22 * prob_2

		// 3. Joint log density: log f(y_t, S_t=j | y_{1:t-1})
		lj_1 := _rs_garch_log_normal(returns[t], var_1) + math.ln_f64(pred_1)
		lj_2 := _rs_garch_log_normal(returns[t], var_2) + math.ln_f64(pred_2)

		// 4. Total log likelihood at t (log-sum-exp)
		m := max(lj_1, lj_2)
		log_total := m + math.ln_f64(math.exp_f64(lj_1 - m) + math.exp_f64(lj_2 - m))
		loglik += log_total

		// 5. Filtered Probabilities: P(S_t=j | y_{1:t})
		prob_1 = math.exp_f64(lj_1 - log_total)
		prob_2 = 1.0 - prob_1

		// Only store probs if requested
		if compute_probs {
			filtered_probs[t][0] = prob_1
			filtered_probs[t][1] = prob_2
		}
	}

	return loglik, filtered_probs
}

// ============================================================================
// Viterbi Algorithm (Find Most Likely Regime Path)
// ============================================================================

// Most likely regime sequence argmax_S P(S_{1:n} | y_{1:n}) (max-product recursion
// with backtracking, started from the ergodic distribution). filtered_probs is
// unused (kept for API compatibility): the argmax of the filtered probabilities is
// a different, pointwise estimate.
_viterbi_path :: proc(
	params: RS_GARCH_Params,
	returns: []f64,
	filtered_probs: [][]f64,
	allocator: mem.Allocator,
) -> []int {
	n := len(returns)
	path := make([]int, n, allocator)
	if n == 0 {return path}

	// back[t][j] = best predecessor of regime j at t
	back := make([][2]int, n, context.temp_allocator)
	defer delete(back, context.temp_allocator)

	log_p := [2][2]f64 {
		{math.ln_f64(params.p_11), math.ln_f64(1.0 - params.p_11)},
		{math.ln_f64(1.0 - params.p_22), math.ln_f64(params.p_22)},
	}
	denom := 2.0 - params.p_11 - params.p_22
	pi_1 := (1.0 - params.p_22) / denom if denom > 0.0 else 0.5

	var_1 := _rs_garch_sample_var(returns)
	var_2 := var_1
	delta: [2]f64
	for t in 0 ..< n {
		_rs_garch_regime_vars(params, returns, t, &var_1, &var_2)
		ld := [2]f64{_rs_garch_log_normal(returns[t], var_1), _rs_garch_log_normal(returns[t], var_2)}
		if t == 0 {
			delta = {math.ln_f64(pi_1) + ld[0], math.ln_f64(1.0 - pi_1) + ld[1]}
			continue
		}
		next: [2]f64
		for j in 0 ..< 2 {
			from_0 := delta[0] + log_p[0][j]
			from_1 := delta[1] + log_p[1][j]
			if from_1 > from_0 {
				next[j], back[t][j] = from_1 + ld[j], 1
			} else {
				next[j], back[t][j] = from_0 + ld[j], 0
			}
		}
		delta = next
	}

	path[n - 1] = 1 if delta[1] > delta[0] else 0
	for t := n - 1; t > 0; t -= 1 {
		path[t - 1] = back[t][path[t]]
	}
	return path
}

// ============================================================================
// Fitting Procedure (Grid Search + Refinement)
// ============================================================================

rs_garch_fit :: proc(
	returns: []f64,
	allocator: mem.Allocator = context.allocator,
) -> RS_GARCH_Result {
	result: RS_GARCH_Result
	n := len(returns)
	if n < 50 {return result}

	// 1. Initialize with Standard GARCH parameters
	// We run a standard GARCH fit to get a baseline for omega, alpha, beta
	// Then we split them: Regime 1 gets lower vol, Regime 2 gets higher vol
	std_garch := garch_fit(returns, .GARCH, 1, 1, 2000, 1e-4, allocator)
	defer {
		delete(std_garch.params.alpha, allocator)
		delete(std_garch.params.beta, allocator)
		delete(std_garch.conditional_var, allocator)
		delete(std_garch.standardized_resid, allocator)
	}

	base_omega := std_garch.params.omega
	base_alpha := std_garch.params.alpha[0]
	base_beta := std_garch.params.beta[0]

	// 2. Grid Search over Transition Probabilities
	// RS-GARCH is highly non-convex. Grid search is the most robust way to find the global optimum.
	best_loglik := -math.INF_F64
	best_params: RS_GARCH_Params

	// Grid: p_11 and p_22 from 0.80 to 0.99
	for p11_i in 80 ..< 100 {
		p11 := f64(p11_i) / 100.0
		for p22_i in 80 ..< 100 {
			p22 := f64(p22_i) / 100.0

			// Heuristic initialization for GARCH params based on transition probs
			// Regime 1 (Calm): Lower omega, higher persistence
			// Regime 2 (Crisis): Higher omega, lower persistence
			params := RS_GARCH_Params {
				omega_1 = base_omega * 0.5,
				alpha_1 = base_alpha * 0.8,
				beta_1  = base_beta * 1.05,
				omega_2 = base_omega * 2.0,
				alpha_2 = base_alpha * 1.2,
				beta_2  = base_beta * 0.9,
				p_11    = p11,
				p_22    = p22,
			}

			loglik, _ := _rs_garch_hamilton_filter(params, returns, allocator, false)

			if loglik > best_loglik {
				best_loglik = loglik
				best_params = params
			}
		}
	}

	// 3. Joint MLE of all 8 parameters (the grid only fixes p11/p22 with heuristic
	// GARCH splits): Nelder-Mead from the best grid point, with restarts.
	// Unconstrained x: ln ω_j, logit(α_j+β_j), logit(α_j/(α_j+β_j)), logit(p_jj)
	x := make([]f64, 8, allocator)
	defer delete(x, allocator)
	_rs_garch_pack(best_params, x)
	nm_obj :: proc(x: []f64, ctx: rawptr) -> f64 {
		ll, _ := _rs_garch_hamilton_filter(_rs_garch_unpack(x), (^[]f64)(ctx)^, context.temp_allocator)
		return -ll if !math.is_nan(ll) else math.F64_MAX
	}
	rets := returns
	best_nll := nm_obj(x, &rets)
	converged := false
	// Restart from the best point until a restart stops improving the likelihood
	for _ in 0 ..< 10 {
		x_nm, f_nm, conv := nelder_mead(nm_obj, &rets, x, 4000, 1e-10, allocator)
		improvement := best_nll - f_nm
		if f_nm < best_nll {
			best_nll = f_nm
			copy(x, x_nm)
		}
		delete(x_nm, allocator)
		converged = conv
		if conv && improvement < 1e-7 {break}
	}
	best_params = _rs_garch_unpack(x)

	// Label regimes so that regime 1 has the lower unconditional variance
	uv_1 := best_params.omega_1 / (1.0 - best_params.alpha_1 - best_params.beta_1)
	uv_2 := best_params.omega_2 / (1.0 - best_params.alpha_2 - best_params.beta_2)
	if uv_2 < uv_1 {
		bp := best_params
		best_params = RS_GARCH_Params {
			omega_1 = bp.omega_2,
			alpha_1 = bp.alpha_2,
			beta_1  = bp.beta_2,
			omega_2 = bp.omega_1,
			alpha_2 = bp.alpha_1,
			beta_2  = bp.beta_1,
			p_11    = bp.p_22,
			p_22    = bp.p_11,
		}
	}

	// 4. Final Evaluation
	final_loglik, final_probs := _rs_garch_hamilton_filter(best_params, returns, allocator, true)
	final_path := _viterbi_path(best_params, returns, final_probs, allocator)

	result.params = best_params
	result.filtered_probs = final_probs
	result.log_likelihood = final_loglik
	result.converged = converged
	result.most_likely_path = final_path

	return result
}

_rs_garch_logit :: proc(p: f64) -> f64 {
	q := clamp(p, 1e-6, 1.0 - 1e-6)
	return math.ln_f64(q / (1.0 - q))
}

_rs_garch_sigmoid :: proc(x: f64) -> f64 {
	return 1.0 / (1.0 + math.exp_f64(-x))
}

_rs_garch_pack :: proc(params: RS_GARCH_Params, x: []f64) {
	pack_regime :: proc(omega, alpha, beta: f64, x: []f64) {
		pers := clamp(alpha + beta, 1e-4, 0.998) // grid heuristics may be non-stationary
		x[0] = math.ln_f64(max(omega, 1e-300))
		x[1] = _rs_garch_logit(pers)
		x[2] = _rs_garch_logit(alpha / (alpha + beta) if alpha + beta > 0.0 else 0.5)
	}
	pack_regime(params.omega_1, params.alpha_1, params.beta_1, x[0:3])
	pack_regime(params.omega_2, params.alpha_2, params.beta_2, x[3:6])
	x[6] = _rs_garch_logit(params.p_11)
	x[7] = _rs_garch_logit(params.p_22)
}

_rs_garch_unpack :: proc(x: []f64) -> RS_GARCH_Params {
	pers_1, share_1 := _rs_garch_sigmoid(x[1]), _rs_garch_sigmoid(x[2])
	pers_2, share_2 := _rs_garch_sigmoid(x[4]), _rs_garch_sigmoid(x[5])
	return RS_GARCH_Params {
		omega_1 = math.exp_f64(x[0]),
		alpha_1 = pers_1 * share_1,
		beta_1 = pers_1 * (1.0 - share_1),
		omega_2 = math.exp_f64(x[3]),
		alpha_2 = pers_2 * share_2,
		beta_2 = pers_2 * (1.0 - share_2),
		p_11 = _rs_garch_sigmoid(x[6]),
		p_22 = _rs_garch_sigmoid(x[7]),
	}
}
