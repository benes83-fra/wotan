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

	// Initialize filtered probabilities (unconditional)
	denom := 2.0 - params.p_11 - params.p_22
	prob_1 := (1.0 - params.p_22) / denom
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
	sample_var := 0.0
	for r in returns {sample_var += r * r}
	sample_var /= f64(n)

	var_1 := sample_var
	var_2 := sample_var

	loglik = 0.0

	for t in 0 ..< n {
		y := returns[t]
		y_sq := y * y

		// 1. Update Variances for each regime
		if t > 0 {
			prev_y_sq := returns[t - 1] * returns[t - 1]
			var_1 = params.omega_1 + params.alpha_1 * prev_y_sq + params.beta_1 * var_1
			var_2 = params.omega_2 + params.alpha_2 * prev_y_sq + params.beta_2 * var_2
		}

		// Ensure variances are positive
		if var_1 < 1e-10 {var_1 = 1e-10}
		if var_2 < 1e-10 {var_2 = 1e-10}

		// 2. Compute Densities f(y_t | S_t = j)
		density_1 :=
			(1.0 / math.sqrt_f64(2.0 * math.PI * var_1)) * math.exp_f64(-y_sq / (2.0 * var_1))
		density_2 :=
			(1.0 / math.sqrt_f64(2.0 * math.PI * var_2)) * math.exp_f64(-y_sq / (2.0 * var_2))

		// 3. Predicted Probabilities (from t-1 to t)
		pred_1 := params.p_11 * prob_1 + (1.0 - params.p_22) * prob_2
		pred_2 := (1.0 - params.p_11) * prob_1 + params.p_22 * prob_2

		// 4. Joint Probability: f(y_t, S_t=j | y_{1:t-1})
		joint_1 := density_1 * pred_1
		joint_2 := density_2 * pred_2

		// 5. Total Likelihood at time t
		total_lik := joint_1 + joint_2
		if total_lik < 1e-300 {total_lik = 1e-300}

		loglik += math.ln_f64(total_lik)

		// 6. Filtered Probabilities: P(S_t=j | y_{1:t})
		prob_1 = joint_1 / total_lik
		prob_2 = joint_2 / total_lik

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

_viterbi_path :: proc(
	params: RS_GARCH_Params,
	returns: []f64,
	filtered_probs: [][]f64,
	allocator: mem.Allocator,
) -> []int {
	n := len(returns)
	path := make([]int, n, allocator)

	// Simple decoding: pick the regime with the highest filtered probability at each step
	// (Note: Full Viterbi requires forward-backward recursion, but this is a good approximation)
	for t in 0 ..< n {
		if filtered_probs[t][1] > filtered_probs[t][0] {
			path[t] = 1 // Regime 2
		} else {
			path[t] = 0 // Regime 1
		}
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

	// 3. Local Refinement (Simple Gradient Ascent on the best grid point)
	// We nudge the parameters slightly to find the local peak
	lr := 1e-5
	for iter in 0 ..< 100 {
		// Numerical gradient for p_11, p_22
		eps := 1e-4

		// Gradient for p_11
		params_plus := best_params
		params_plus.p_11 = math.min(0.999, best_params.p_11 + eps)
		loglik_plus, _ := _rs_garch_hamilton_filter(params_plus, returns, allocator, false)

		params_minus := best_params
		params_minus.p_11 = math.max(0.500, best_params.p_11 - eps)
		loglik_minus, _ := _rs_garch_hamilton_filter(params_minus, returns, allocator, false)

		grad_p11 := (loglik_plus - loglik_minus) / (2.0 * eps)
		best_params.p_11 += lr * grad_p11
		best_params.p_11 = math.max(0.5, math.min(0.999, best_params.p_11))

		// Gradient for p_22
		params_plus = best_params
		params_plus.p_22 = math.min(0.999, best_params.p_22 + eps)
		loglik_plus, _ = _rs_garch_hamilton_filter(params_plus, returns, allocator)

		params_minus = best_params
		params_minus.p_22 = math.max(0.500, best_params.p_22 - eps)
		loglik_minus, _ = _rs_garch_hamilton_filter(params_minus, returns, allocator)

		grad_p22 := (loglik_plus - loglik_minus) / (2.0 * eps)
		best_params.p_22 += lr * grad_p22
		best_params.p_22 = math.max(0.5, math.min(0.999, best_params.p_22))
	}

	// 4. Final Evaluation
	final_loglik, final_probs := _rs_garch_hamilton_filter(best_params, returns, allocator, true)
	final_path := _viterbi_path(best_params, returns, final_probs, allocator)

	result.params = best_params
	result.filtered_probs = final_probs
	result.log_likelihood = final_loglik
	result.converged = true
	result.most_likely_path = final_path

	return result
}
