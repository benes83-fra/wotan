package analytics

import l "../linalg"
import opt "../optimize"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// GARCH Model Structures
// ============================================================================

GARCH_Model_Type :: enum {
	ARCH,
	GARCH,
	GJR_GARCH, // Asymmetric GARCH (leverage effect)
}

GARCH_Params :: struct {
	omega:      f64,
	alpha:      []f64,
	beta:       []f64,
	gamma:      f64,
	model_type: GARCH_Model_Type,
	p:          int,
	q:          int,
}

GARCH_Result :: struct {
	params:             GARCH_Params,
	log_likelihood:     f64,
	aic:                f64,
	bic:                f64,
	conditional_var:    []f64,
	standardized_resid: []f64,
	converged:          bool,
	n_iterations:       int,
	persistence:        f64, // α + β (volatility persistence)
}

GARCH_Forecast :: struct {
	variance_forecast: f64,
	std_forecast:      f64,
	confidence_95:     [2]f64,
}

// ============================================================================
// Parameter Transformation (Fixed: allows α+β < 1)
// ============================================================================

unpack_params :: proc(x: []f64, params: ^GARCH_Params) {
	// Omega: use exp to ensure positivity
	params.omega = math.exp(x[0] - 10.0)

	// Alpha and Beta: use sigmoid to constrain to (0, 1)
	// Then scale to ensure sum < 1
	sum_raw := 0.0
	for i in 0 ..< params.p {
		val := 1.0 / (1.0 + math.exp(-x[1 + i])) // sigmoid
		params.alpha[i] = val
		sum_raw += val
	}
	for j in 0 ..< params.q {
		val := 1.0 / (1.0 + math.exp(-x[1 + params.p + j])) // sigmoid
		params.beta[j] = val
		sum_raw += val
	}

	// Scale to ensure sum < 0.999 (leave room for stationarity)
	if sum_raw > 0.999 {
		scale := 0.999 / sum_raw
		for i in 0 ..< params.p {params.alpha[i] *= scale}
		for j in 0 ..< params.q {params.beta[j] *= scale}
	}

	// Gamma for GJR-GARCH
	if params.model_type == .GJR_GARCH && len(x) > 1 + params.p + params.q {
		params.gamma = x[1 + params.p + params.q] * 0.1
	}
}

// ============================================================================
// GARCH Log-Likelihood Function
// ============================================================================

_garch_neg_log_likelihood :: proc(
	params: ^GARCH_Params,
	residuals: []f64,
	allocator: mem.Allocator,
) -> f64 {
	n := len(residuals)
	p := params.p
	q := params.q

	cond_var := make([]f64, n, allocator)
	defer delete(cond_var, allocator)

	// Initialize with unconditional variance
	uncond_var := 0.0
	for i in 0 ..< n {
		uncond_var += residuals[i] * residuals[i]
	}
	uncond_var /= f64(n)

	for i in 0 ..< max(p, q) {
		cond_var[i] = uncond_var
	}

	// Compute conditional variances with SIMD-friendly loop
	for t in max(p, q) ..< n {
		variance := params.omega

		// ARCH terms
		for i in 0 ..< p {
			if t - i - 1 >= 0 {
				variance += params.alpha[i] * residuals[t - i - 1] * residuals[t - i - 1]
			}
		}

		// GARCH terms
		for j in 0 ..< q {
			if t - j - 1 >= 0 {
				variance += params.beta[j] * cond_var[t - j - 1]
			}
		}

		// GJR-GARCH asymmetry
		if params.model_type == .GJR_GARCH && params.gamma != 0.0 {
			for i in 0 ..< p {
				if t - i - 1 >= 0 && residuals[t - i - 1] < 0.0 {
					variance += params.gamma * residuals[t - i - 1] * residuals[t - i - 1]
				}
			}
		}

		if variance < 1e-8 {
			variance = 1e-8
		}

		cond_var[t] = variance
	}

	// Compute log-likelihood
	log_lik := 0.0
	for t in max(p, q) ..< n {
		log_lik +=
			-0.5 *
			(math.ln_f64(2.0 * math.PI) +
					math.ln_f64(cond_var[t]) +
					residuals[t] * residuals[t] / cond_var[t])
	}

	return -log_lik
}

// ============================================================================
// GARCH Parameter Estimation (Fixed convergence and optimizer)
// ============================================================================

garch_fit :: proc(
	residuals: []f64,
	model_type: GARCH_Model_Type = .GARCH,
	p: int = 1,
	q: int = 1,
	max_iter: int = 2000, // Increased from 1000
	tolerance: f64 = 1e-4, // Relaxed from 1e-6
	allocator: mem.Allocator = context.allocator,
) -> GARCH_Result {
	n := len(residuals)

	params := GARCH_Params {
		model_type = model_type,
		p          = p,
		q          = q,
	}

	params.alpha = make([]f64, p, allocator)
	params.beta = make([]f64, q, allocator)

	n_params := 1 + p + q
	if model_type == .GJR_GARCH {
		n_params += 1
	}

	// 1. Initialize optimizer with higher learning rate
	opt_config := opt.optimizer_default_config(.Adam)
	opt_config.learning_rate = 0.05
	opt_config.beta1 = 0.9
	opt_config.beta2 = 0.999

	optimizer := opt.optimizer_init(opt_config, n_params, allocator)
	defer opt.optimizer_free(&optimizer)

	param_vec := make([]f64, n_params, allocator)
	gradient := make([]f64, n_params, allocator)
	defer {
		delete(param_vec, allocator)
		delete(gradient, allocator)
	}

	// Initial values in unconstrained space
	param_vec[0] = -5.0
	for i in 1 ..< 1 + p + q {
		param_vec[i] = 0.0
	}
	if model_type == .GJR_GARCH {
		param_vec[1 + p + q] = 0.0
	}

	best_loss := math.F64_MAX
	best_params := make([]f64, n_params, allocator)
	defer delete(best_params, allocator)

	param_vec_plus := make([]f64, n_params, context.temp_allocator)
	param_vec_minus := make([]f64, n_params, context.temp_allocator)
	defer {
		delete(param_vec_plus, context.temp_allocator)
		delete(param_vec_minus, context.temp_allocator)
	}

	prev_loss := math.F64_MAX
	converged := false

	for iter in 0 ..< max_iter {
		unpack_params(param_vec, &params)
		loss := _garch_neg_log_likelihood(&params, residuals, allocator)

		if loss < best_loss {
			best_loss = loss
			copy(best_params, param_vec)
		}

		// 4. Convergence check inside the loop
		if iter > 100 && math.abs(prev_loss - loss) < tolerance {
			fmt.printf(
				"GARCH converged at iteration %d (loss change: %.8f)\n",
				iter,
				math.abs(prev_loss - loss),
			)
			converged = true
			break
		}

		// 5. Early stopping if loss INCREASES (becomes less negative)
		// Since loss is negative, "increasing" means loss > prev_loss
		if iter > 100 && loss > prev_loss {
			// Only stop if it's been increasing for a while
			if loss > prev_loss + 1.0 { 	// Increased by more than 1.0
				fmt.println("Loss increasing, stopping early")
				break
			}
		}

		prev_loss = loss

		prev_loss = loss

		// Numerical gradient
		eps := 1e-5
		for i in 0 ..< n_params {
			copy(param_vec_plus, param_vec)
			copy(param_vec_minus, param_vec)

			param_vec_plus[i] += eps
			param_vec_minus[i] -= eps

			unpack_params(param_vec_plus, &params)
			loss_plus := _garch_neg_log_likelihood(&params, residuals, allocator)

			unpack_params(param_vec_minus, &params)
			loss_minus := _garch_neg_log_likelihood(&params, residuals, allocator)

			gradient[i] = (loss_plus - loss_minus) / (2.0 * eps)
		}

		opt.optimizer_step(&optimizer, param_vec, gradient, loss)

		if iter % 100 == 0 {
			unpack_params(param_vec, &params)
			persistence := 0.0
			for i in 0 ..< p {persistence += params.alpha[i]}
			for j in 0 ..< q {persistence += params.beta[j]}

			alpha_0: f64 = 0.0
			if p > 0 {
				alpha_0 = params.alpha[0]
			}

			beta_0: f64 = 0.0
			if q > 0 {
				beta_0 = params.beta[0]
			}

			fmt.printf(
				"GARCH iter %d: loss=%.4f, ω=%.6f, α=%.4f, β=%.4f, α+β=%.4f\n",
				iter,
				loss,
				params.omega,
				alpha_0,
				beta_0,
				persistence,
			)
		}
	}

	// Unpack best parameters
	unpack_params(best_params, &params)

	// Compute final conditional variances
	cond_var := make([]f64, n, allocator)
	std_resid := make([]f64, n, allocator)

	uncond_var := 0.0
	for i in 0 ..< n {
		uncond_var += residuals[i] * residuals[i]
	}
	uncond_var /= f64(n)

	for i in 0 ..< max(p, q) {
		cond_var[i] = uncond_var
		std_resid[i] = residuals[i] / math.sqrt_f64(cond_var[i])
	}

	for t in max(p, q) ..< n {
		variance := params.omega
		for i in 0 ..< p {
			if t - i - 1 >= 0 {
				variance += params.alpha[i] * residuals[t - i - 1] * residuals[t - i - 1]
			}
		}
		for j in 0 ..< q {
			if t - j - 1 >= 0 {
				variance += params.beta[j] * cond_var[t - j - 1]
			}
		}
		if params.model_type == .GJR_GARCH && params.gamma != 0.0 {
			for i in 0 ..< p {
				if t - i - 1 >= 0 && residuals[t - i - 1] < 0.0 {
					variance += params.gamma * residuals[t - i - 1] * residuals[t - i - 1]
				}
			}
		}
		if variance < 1e-8 {
			variance = 1e-8
		}
		cond_var[t] = variance
		std_resid[t] = residuals[t] / math.sqrt_f64(variance)
	}

	log_lik := -best_loss
	k := f64(n_params)
	aic := 2.0 * k - 2.0 * log_lik
	bic := k * math.ln_f64(f64(n)) - 2.0 * log_lik

	persistence := 0.0
	for i in 0 ..< p {persistence += params.alpha[i]}
	for j in 0 ..< q {persistence += params.beta[j]}

	return GARCH_Result {
		params = params,
		log_likelihood = log_lik,
		aic = aic,
		bic = bic,
		conditional_var = cond_var,
		standardized_resid = std_resid,
		converged = converged,
		n_iterations = max_iter,
		persistence = persistence,
	}
}

// ============================================================================
// GARCH Forecasting
// ============================================================================

garch_forecast :: proc(
	result: ^GARCH_Result,
	residuals: []f64,
	horizon: int,
	allocator: mem.Allocator = context.allocator,
) -> GARCH_Forecast {
	n := len(residuals)
	p := result.params.p
	q := result.params.q

	last_residual := residuals[n - 1]
	last_cond_var := result.conditional_var[n - 1]

	forecast_var := last_cond_var

	for h in 1 ..= horizon {
		new_var := result.params.omega

		for i in 0 ..< p {
			if i == 0 {
				new_var += result.params.alpha[i] * last_residual * last_residual
			} else {
				new_var += result.params.alpha[i] * forecast_var
			}
		}

		for j in 0 ..< q {
			if j == 0 {
				new_var += result.params.beta[j] * last_cond_var
			} else {
				new_var += result.params.beta[j] * forecast_var
			}
		}

		forecast_var = new_var
	}

	if forecast_var < 1e-8 {
		forecast_var = 1e-8
	}

	forecast_std := math.sqrt_f64(forecast_var)

	confidence_95 := [2]f64{-1.96 * forecast_std, 1.96 * forecast_std}

	return GARCH_Forecast {
		variance_forecast = forecast_var,
		std_forecast = forecast_std,
		confidence_95 = confidence_95,
	}
}

// ============================================================================
// Utility Functions
// ============================================================================

extract_residuals :: proc(series: []f64, allocator: mem.Allocator = context.allocator) -> []f64 {
	n := len(series)
	residuals := make([]f64, n, allocator)

	mean := 0.0
	for v in series {
		mean += v
	}
	mean /= f64(n)

	for i in 0 ..< n {
		residuals[i] = series[i] - mean
	}

	return residuals
}
