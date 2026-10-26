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
	StudentT,
	GED,
}

GARCH_Params :: struct {
	omega:      f64,
	alpha:      []f64,
	beta:       []f64,
	gamma:      f64,
	nu:         f64,
	shape:      f64,
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
	persistence:        f64, // α + β (+ γ/2 for GJR): volatility persistence
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

	// Gamma for GJR-GARCH: sigmoid keeps it in (0, 1); with symmetric shocks
	// the leverage term adds γ/2 to the persistence
	is_gjr := params.model_type == .GJR_GARCH && len(x) > 1 + params.p + params.q
	if is_gjr {
		params.gamma = 1.0 / (1.0 + math.exp(-x[1 + params.p + params.q]))
		sum_raw += 0.5 * params.gamma
	}

	// Scale to ensure persistence < 0.999 (leave room for stationarity)
	if sum_raw > 0.999 {
		scale := 0.999 / sum_raw
		for i in 0 ..< params.p {params.alpha[i] *= scale}
		for j in 0 ..< params.q {params.beta[j] *= scale}
		if is_gjr {params.gamma *= scale}
	}
	if params.model_type == .StudentT {
		nu_idx := 1 + params.p + params.q
		if params.model_type == .GJR_GARCH {nu_idx += 1}
		params.nu = 2.0 + math.exp(x[nu_idx])
	}
	if params.model_type == .GED {
		shape_idx := 1 + params.p + params.q
		if params.model_type == .GJR_GARCH {shape_idx += 1}
		params.shape = math.exp(x[shape_idx])
	}
}

// ============================================================================
// GARCH Log-Likelihood Function
// ============================================================================

// Conditional variance at t from residuals and cond_var before t:
// ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j} (+ γ Σ ε²_{t-i} 1[ε_{t-i} < 0] for GJR)
_garch_variance_at :: proc(params: ^GARCH_Params, residuals, cond_var: []f64, t: int) -> f64 {
	variance := params.omega
	for i in 0 ..< params.p {
		if t - i - 1 >= 0 {variance += params.alpha[i] * residuals[t - i - 1] * residuals[t - i - 1]}
	}
	for j in 0 ..< params.q {
		if t - j - 1 >= 0 {variance += params.beta[j] * cond_var[t - j - 1]}
	}
	if params.model_type == .GJR_GARCH && params.gamma != 0.0 {
		for i in 0 ..< params.p {
			if t - i - 1 >= 0 && residuals[t - i - 1] < 0.0 {
				variance += params.gamma * residuals[t - i - 1] * residuals[t - i - 1]
			}
		}
	}
	if variance < 1e-8 {variance = 1e-8}
	return variance
}

// Negative log-likelihood. Conventions (checked against a numpy reference):
// - variance initialisation: σ²_t = mean(ε²) (uncentred sample variance) for
//   t < max(p, q); those first max(p, q) observations only seed the recursion
//   and are not part of the likelihood (conditional MLE)
// - StudentT: standardized t with unit variance,
//   log f = lnΓ((ν+1)/2) - lnΓ(ν/2) - ½ln((ν-2)π) - ½ln σ² - (ν+1)/2 · ln(1 + z²/(ν-2))
// - GED: unit-variance GED, f(z) = p / (2 λ Γ(1/p)) exp(-|z/λ|^p), λ = sqrt(Γ(1/p)/Γ(3/p))
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

	uncond_var := 0.0
	for i in 0 ..< n {uncond_var += residuals[i] * residuals[i]}
	uncond_var /= f64(n)

	for i in 0 ..< max(p, q) {cond_var[i] = uncond_var}

	for t in max(p, q) ..< n {cond_var[t] = _garch_variance_at(params, residuals, cond_var, t)}

	log_lik := 0.0
	is_student_t := params.model_type == .StudentT
	is_ged := params.model_type == .GED
	nu := params.nu
	shape := params.shape

	for t in max(p, q) ..< n {
		z := residuals[t] / math.sqrt_f64(cond_var[t])

		if is_student_t && nu > 2.0 {
			// Change math.ln_gamma to math.lgamma
			term1, _ := math.lgamma((nu + 1.0) / 2.0)
			term2, _ := math.lgamma(nu / 2.0)
			term3 := 0.5 * math.ln_f64((nu - 2.0) * math.PI)
			term4 := 0.5 * math.ln_f64(cond_var[t])
			term5 := ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + (z * z) / (nu - 2.0))

			log_lik += term1 - term2 - term3 - term4 - term5
		} else if is_ged && shape > 0.0 {
			// GED log-likelihood
			// λ_p = sqrt(Γ(1/p) / Γ(3/p))
			lg1_p, _ := math.lgamma(1.0 / shape)
			lg3_p, _ := math.lgamma(3.0 / shape)
			lambda_p := math.exp_f64(0.5 * (lg1_p - lg3_p))

			// Standardized residual
			u := z / lambda_p

			// log f = log(p) - log(2) - log(Γ(1/p)) - log(λ_p) - log(σ_t) - |u|^p
			log_p := math.ln_f64(shape)
			log_2 := math.ln_f64(2.0)
			log_gamma, _ := math.lgamma(1.0 / shape)
			log_lambda := math.ln_f64(lambda_p)
			log_sigma := 0.5 * math.ln_f64(cond_var[t])
			power_term := math.pow(math.abs(u), shape)

			log_lik += log_p - log_2 - log_gamma - log_lambda - log_sigma - power_term
		} else {
			log_lik += -0.5 * (math.ln_f64(2.0 * math.PI) + math.ln_f64(cond_var[t]) + z * z)
		}
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

	// Need at least one observation beyond the lag order to fit anything
	if p < 0 || q < 0 || n <= max(p, q) {
		return GARCH_Result{params = params}
	}

	params.alpha = make([]f64, p, allocator)
	params.beta = make([]f64, q, allocator)

	n_params := 1 + p + q
	if model_type == .GJR_GARCH {
		n_params += 1
	}
	if model_type == .StudentT {n_params += 1}
	if model_type == .GED {n_params += 1}
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
	gamma_idx := 1 + p + q
	if model_type == .GJR_GARCH {
		param_vec[gamma_idx] = math.ln_f64(0.05 / 0.95) // gamma = 0.05
		gamma_idx += 1
	}
	if model_type == .StudentT {
		param_vec[gamma_idx] = math.ln_f64(3.0) // Initial nu = 2 + exp(ln(3)) = 5.0
	}
	if model_type == .GED {
		param_vec[gamma_idx] = math.ln_f64(2.0) // shape = 2 (Normal)
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
	n_iter := 0

	for iter in 0 ..< max_iter {
		unpack_params(param_vec, &params)
		loss := _garch_neg_log_likelihood(&params, residuals, allocator)

		if loss < best_loss {
			best_loss = loss
			copy(best_params, param_vec)
		}

		// 4. Convergence check inside the loop
		if iter > 100 && math.abs(prev_loss - loss) < tolerance {
			n_iter = iter + 1
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
		n_iter = iter + 1

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

	// Adam with a fixed step stalls ~1e-2 log-likelihood units short of the
	// optimum: polish the best point with Nelder-Mead (restarted once)
	_Garch_NM_Ctx :: struct {
		params:    ^GARCH_Params,
		residuals: []f64,
		allocator: mem.Allocator,
	}
	nm_obj :: proc(x: []f64, ctx: rawptr) -> f64 {
		c := (^_Garch_NM_Ctx)(ctx)
		unpack_params(x, c.params)
		return _garch_neg_log_likelihood(c.params, c.residuals, c.allocator)
	}
	nm_ctx := _Garch_NM_Ctx{&params, residuals, allocator}
	for _ in 0 ..< 2 {
		x_nm, f_nm, _ := nelder_mead(nm_obj, &nm_ctx, best_params, 2000, 1e-10, allocator)
		if f_nm < best_loss {
			best_loss = f_nm
			copy(best_params, x_nm)
		}
		delete(x_nm, allocator)
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
		cond_var[t] = _garch_variance_at(&params, residuals, cond_var, t)
		std_resid[t] = residuals[t] / math.sqrt_f64(cond_var[t])
	}

	log_lik := -best_loss
	k := f64(n_params)
	aic := 2.0 * k - 2.0 * log_lik
	bic := k * math.ln_f64(f64(n)) - 2.0 * log_lik

	persistence := 0.0
	for i in 0 ..< p {persistence += params.alpha[i]}
	for j in 0 ..< q {persistence += params.beta[j]}
	if model_type == .GJR_GARCH {persistence += 0.5 * params.gamma}

	return GARCH_Result {
		params = params,
		log_likelihood = log_lik,
		aic = aic,
		bic = bic,
		conditional_var = cond_var,
		standardized_resid = std_resid,
		converged = converged,
		n_iterations = n_iter, // Adam iterations (the Nelder-Mead polish is not counted)
		persistence = persistence,
	}
}

// ============================================================================
// GARCH Forecasting
// ============================================================================

// h-step-ahead conditional variance. The first step uses the observed lagged
// residuals and variances; later steps replace ε² by its expectation σ² (and
// ε² 1[ε < 0] by σ²/2 for GJR), so the forecast reverts to ω / (1 - persistence).
garch_forecast :: proc(
	result: ^GARCH_Result,
	residuals: []f64,
	horizon: int,
	allocator: mem.Allocator = context.allocator,
) -> GARCH_Forecast {
	n := len(residuals)
	if n == 0 || len(result.conditional_var) == 0 {
		nan := math.nan_f64()
		return GARCH_Forecast{variance_forecast = nan, std_forecast = nan, confidence_95 = {nan, nan}}
	}
	if len(result.conditional_var) < n {
		panic("garch_forecast: residuals are longer than the fitted conditional variance")
	}
	params := &result.params
	p, q := params.p, params.q
	is_gjr := params.model_type == .GJR_GARCH

	// Timeline: index k < n is observed, k >= n is forecast
	total := n + max(horizon, 1)
	sq := make([]f64, total, context.temp_allocator) // ε² or its expectation
	neg_sq := make([]f64, total, context.temp_allocator) // ε² 1[ε < 0] or its expectation
	s2 := make([]f64, total, context.temp_allocator)
	defer {
		delete(sq, context.temp_allocator)
		delete(neg_sq, context.temp_allocator)
		delete(s2, context.temp_allocator)
	}
	for k in 0 ..< n {
		sq[k] = residuals[k] * residuals[k]
		neg_sq[k] = sq[k] if residuals[k] < 0.0 else 0.0
		s2[k] = result.conditional_var[k]
	}

	forecast_var := result.conditional_var[n - 1]
	for k in n ..< n + horizon {
		v := params.omega
		for i in 0 ..< p {
			if k - i - 1 >= 0 {
				v += params.alpha[i] * sq[k - i - 1]
				if is_gjr {v += params.gamma * neg_sq[k - i - 1]}
			}
		}
		for j in 0 ..< q {
			if k - j - 1 >= 0 {v += params.beta[j] * s2[k - j - 1]}
		}
		s2[k], sq[k], neg_sq[k] = v, v, 0.5 * v
		forecast_var = v
	}

	if forecast_var < 1e-8 {
		forecast_var = 1e-8
	}

	forecast_std := math.sqrt_f64(forecast_var)

	// 95% interval of ε_{n+h} under the fitted standardized error distribution
	z_975 := garch_std_quantile(params, 0.975)
	confidence_95 := [2]f64{-z_975 * forecast_std, z_975 * forecast_std}

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
// ============================================================================
// GARCH-Based VaR Series
// ============================================================================

// Compute VaR series from GARCH model
// Returns 95% and 99% VaR for each time point (assuming mean ≈ 0)
garch_var_series :: proc(
	result: ^GARCH_Result,
	residuals: []f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	var_95: []f64,
	var_99: []f64,
) {
	n := min(len(residuals), len(result.conditional_var))
	var_95 = make([]f64, n, allocator)
	var_99 = make([]f64, n, allocator)

	// Quantiles of the fitted standardized (unit-variance) error distribution:
	// normal, Student-t scaled by sqrt((ν-2)/ν), or GED
	z_95 := garch_std_quantile(&result.params, 0.95)
	z_99 := garch_std_quantile(&result.params, 0.99)

	for i in 0 ..< n {
		std := math.sqrt_f64(result.conditional_var[i])
		var_95[i] = z_95 * std // Assuming mean ≈ 0
		var_99[i] = z_99 * std
	}

	return var_95, var_99
}

// ============================================================================
// Standardized Error Quantiles
// ============================================================================

// Quantile of the unit-variance error distribution of a fitted model:
// N(0,1); Student-t(ν)·sqrt((ν-2)/ν); GED(p) with λ = sqrt(Γ(1/p)/Γ(3/p))
garch_std_quantile :: proc(params: ^GARCH_Params, prob: f64) -> f64 {
	if prob <= 0.0 {return -math.INF_F64}
	if prob >= 1.0 {return math.INF_F64}
	cdf: proc(x, shape: f64) -> f64
	shape := 0.0
	scale := 1.0
	switch {
	case params.model_type == .StudentT && params.nu > 2.0:
		shape, scale = params.nu, math.sqrt_f64((params.nu - 2.0) / params.nu)
		cdf = proc(x, nu: f64) -> f64 {
			// F(x) = 1 - ½ I_{ν/(ν+x²)}(ν/2, ½) for x ≥ 0; near 0 use the
			// complement 1 - I_{x²/(ν+x²)}(½, ν/2) to avoid cancellation
			tail: f64
			if x * x < nu {
				tail = 0.5 - 0.5 * _garch_reg_inc_beta(0.5, 0.5 * nu, x * x / (nu + x * x))
			} else {
				tail = 0.5 * _garch_reg_inc_beta(0.5 * nu, 0.5, nu / (nu + x * x))
			}
			return 1.0 - tail if x >= 0.0 else tail
		}
	case params.model_type == .GED && params.shape > 0.0:
		lg1, _ := math.lgamma(1.0 / params.shape)
		lg3, _ := math.lgamma(3.0 / params.shape)
		shape, scale = params.shape, math.exp_f64(0.5 * (lg1 - lg3))
		cdf = proc(x, p: f64) -> f64 {
			// F(x) = ½ + ½ sign(x) P(1/p, |x|^p) for the λ = 1 density
			half := 0.5 * _garch_reg_lower_gamma(1.0 / p, math.pow(math.abs(x), p))
			return 0.5 + half if x >= 0.0 else 0.5 - half
		}
	case:
		cdf = proc(x, _: f64) -> f64 {return 0.5 * math.erfc(-x / math.SQRT_TWO)}
	}

	// Bisection on the (λ = 1 / unscaled) CDF, then rescale to unit variance
	lo, hi := -1.0, 1.0
	for cdf(lo, shape) > prob && lo > -1e12 {lo *= 2.0}
	for cdf(hi, shape) < prob && hi < 1e12 {hi *= 2.0}
	for _ in 0 ..< 200 {
		mid := 0.5 * (lo + hi)
		if mid == lo || mid == hi {break}
		if cdf(mid, shape) < prob {lo = mid} else {hi = mid}
	}
	return scale * 0.5 * (lo + hi)
}

// Regularized incomplete beta I_x(a, b) (continued fraction, Numerical Recipes betacf)
_garch_reg_inc_beta :: proc(a, b, x: f64) -> f64 {
	if x <= 0.0 {return 0.0}
	if x >= 1.0 {return 1.0}
	lab, _ := math.lgamma(a + b)
	la, _ := math.lgamma(a)
	lb, _ := math.lgamma(b)
	front := math.exp_f64(lab - la - lb + a * math.ln_f64(x) + b * math.ln_f64(1.0 - x))
	// Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) where the fraction converges fast
	if x > (a + 1.0) / (a + b + 2.0) {
		return 1.0 - front * _garch_betacf(b, a, 1.0 - x) / b
	}
	return front * _garch_betacf(a, b, x) / a
}

_garch_betacf :: proc(a, b, x: f64) -> f64 {
	TINY :: 1e-300
	qab, qap, qam := a + b, a + 1.0, a - 1.0
	c, d := 1.0, 1.0 - qab * x / qap
	if math.abs(d) < TINY {d = TINY}
	d = 1.0 / d
	h := d
	for m in 1 ..= 300 {
		fm := f64(m)
		m2 := 2.0 * fm
		aa := fm * (b - fm) * x / ((qam + m2) * (a + m2))
		d = 1.0 + aa * d
		if math.abs(d) < TINY {d = TINY}
		c = 1.0 + aa / c
		if math.abs(c) < TINY {c = TINY}
		d = 1.0 / d
		h *= d * c
		aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2))
		d = 1.0 + aa * d
		if math.abs(d) < TINY {d = TINY}
		c = 1.0 + aa / c
		if math.abs(c) < TINY {c = TINY}
		d = 1.0 / d
		del := d * c
		h *= del
		if math.abs(del - 1.0) < 1e-15 {break}
	}
	return h
}

// Regularized lower incomplete gamma P(a, x) (series for x < a+1, else continued fraction)
_garch_reg_lower_gamma :: proc(a, x: f64) -> f64 {
	if x <= 0.0 {return 0.0}
	lga, _ := math.lgamma(a)
	if x < a + 1.0 {
		ap, sum := a, 1.0 / a
		del := sum
		for _ in 0 ..< 1000 {
			ap += 1.0
			del *= x / ap
			sum += del
			if math.abs(del) < math.abs(sum) * 1e-16 {break}
		}
		return sum * math.exp_f64(-x + a * math.ln_f64(x) - lga)
	}
	TINY :: 1e-300
	b := x + 1.0 - a
	c := 1.0 / TINY
	d := 1.0 / b
	h := d
	for i in 1 ..= 1000 {
		an := -f64(i) * (f64(i) - a)
		b += 2.0
		d = an * d + b
		if math.abs(d) < TINY {d = TINY}
		c = b + an / c
		if math.abs(c) < TINY {c = TINY}
		d = 1.0 / d
		del := d * c
		h *= del
		if math.abs(del - 1.0) < 1e-16 {break}
	}
	return 1.0 - math.exp_f64(-x + a * math.ln_f64(x) - lga) * h
}
