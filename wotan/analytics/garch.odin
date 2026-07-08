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
	omega:      f64, // Constant term
	alpha:      []f64, // ARCH coefficients (lagged squared residuals)
	beta:       []f64, // GARCH coefficients (lagged conditional variances)
	gamma:      f64, // Asymmetry parameter (for GJR-GARCH)
	model_type: GARCH_Model_Type,
	p:          int, // ARCH order
	q:          int, // GARCH order
}

GARCH_Result :: struct {
	params:             GARCH_Params,
	log_likelihood:     f64,
	aic:                f64,
	bic:                f64,
	conditional_var:    []f64, // σ²_t for each time point
	standardized_resid: []f64, // ε_t / σ_t
	converged:          bool,
	n_iterations:       int,
}

GARCH_Forecast :: struct {
	mean_forecast:     f64,
	variance_forecast: f64,
	std_forecast:      f64,
	confidence_95:     [2]f64, // [lower, upper]
}

// ============================================================================
// Parameter Transformation (Enforces Constraints)
// ============================================================================

// Transforms unconstrained parameters x into valid GARCH parameters
// Ensures: omega > 0, alpha/beta >= 0, and sum(alpha) + sum(beta) < 0.99
unpack_params :: proc(x: []f64, params: ^GARCH_Params) {
	// Omega: use exp to ensure positivity, scaled to typical variance range
	params.omega = math.exp(x[0] - 10.0)

	// Alpha and Beta: use exp for positivity, then scale to ensure stationarity
	sum_exp := 0.0
	for i in 0 ..< params.p {sum_exp += math.exp(x[1 + i])}
	for j in 0 ..< params.q {sum_exp += math.exp(x[1 + params.p + j])}

	// Scale factor to ensure sum(alpha) + sum(beta) <= 0.99
	scale := 0.99 / (sum_exp + 1e-8)

	for i in 0 ..< params.p {
		params.alpha[i] = math.exp(x[1 + i]) * scale
	}
	for j in 0 ..< params.q {
		params.beta[j] = math.exp(x[1 + params.p + j]) * scale
	}

	// Gamma for GJR-GARCH (can be positive or negative)
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

	// Compute conditional variances
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

		// GJR-GARCH asymmetry term
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
// GARCH Parameter Estimation
// ============================================================================

garch_fit :: proc(
	residuals: []f64,
	model_type: GARCH_Model_Type = .GARCH,
	p: int = 1,
	q: int = 1,
	max_iter: int = 1000,
	tolerance: f64 = 1e-6,
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

	opt_config := opt.optimizer_default_config(.Adam)
	opt_config.learning_rate = 0.01 // Slightly higher for transformed space
	optimizer := opt.optimizer_init(opt_config, n_params, allocator)
	defer opt.optimizer_free(&optimizer)

	param_vec := make([]f64, n_params, allocator)
	gradient := make([]f64, n_params, allocator)
	defer {
		delete(param_vec, allocator)
		delete(gradient, allocator)
	}

	// Initial values in unconstrained space
	param_vec[0] = 0.79 // exp(0.79 - 10) ≈ 0.0001
	for i in 1 ..< 1 + p + q {
		param_vec[i] = 0.0 // exp(0) = 1, will be scaled
	}
	if model_type == .GJR_GARCH {
		param_vec[1 + p + q] = 0.0
	}

	best_loss := math.F64_MAX
	best_params := make([]f64, n_params, allocator)
	defer delete(best_params, allocator)

	// FIX: Allocate gradient buffers ONCE outside the loop
	param_vec_plus := make([]f64, n_params, context.temp_allocator)
	param_vec_minus := make([]f64, n_params, context.temp_allocator)
	defer {
		delete(param_vec_plus, context.temp_allocator)
		delete(param_vec_minus, context.temp_allocator)
	}

	for iter in 0 ..< max_iter {
		unpack_params(param_vec, &params)

		loss := _garch_neg_log_likelihood(&params, residuals, allocator)

		if loss < best_loss {
			best_loss = loss
			copy(best_params, param_vec)
		}

		if iter > 0 && math.abs(loss - best_loss) < tolerance {
			fmt.printf("GARCH converged at iteration %d\n", iter)
			break
		}

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
			fmt.printf("GARCH iteration %d: loss = %.6f\n", iter, loss)
		}
	}

	// Unpack best parameters
	unpack_params(best_params, &params)

	// Compute final conditional variances and standardized residuals
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

	return GARCH_Result {
		params = params,
		log_likelihood = log_lik,
		aic = aic,
		bic = bic,
		conditional_var = cond_var,
		standardized_resid = std_resid,
		converged = true,
		n_iterations = max_iter,
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
		mean_forecast = 0.0,
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

rolling_volatility :: proc(
	series: []f64,
	window: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(series)
	rolling_var := make([]f64, n, allocator)

	for i in 0 ..< n {
		if i < window {
			rolling_var[i] = 0.0
		} else {
			sum := 0.0
			sum_sq := 0.0
			for j in (i - window) ..< i {
				sum += series[j]
				sum_sq += series[j] * series[j]
			}
			mean := sum / f64(window)
			variance := sum_sq / f64(window) - mean * mean
			rolling_var[i] = variance
		}
	}

	return rolling_var
}
