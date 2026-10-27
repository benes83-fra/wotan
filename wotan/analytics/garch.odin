package analytics

import l "../linalg"
import opt "../optimize"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// GARCH Model Structures
// ============================================================================

// Legacy model selector (garch_fit): mixes the variance equation and the error
// distribution. garch_fit_spec chooses them independently (GARCH_Spec).
GARCH_Model_Type :: enum {
	ARCH,
	GARCH,
	GJR_GARCH, // Asymmetric GARCH (leverage effect)
	StudentT,
	GED,
}

// Variance equation
GARCH_Variance :: enum {
	GARCH, // σ²_t = ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}  (q = 0: ARCH)
	GJR, // + γ Σ_i ε²_{t-i} 1[ε_{t-i} < 0]  (leverage effect)
	EGARCH, // ln σ²_t = ω + Σ α_i (|z_{t-i}| - √(2/π)) + Σ γ_i z_{t-i} + Σ β_j ln σ²_{t-j}
}

// Standardised (unit-variance) error distribution
GARCH_Dist :: enum {
	Normal,
	StudentT, // ν > 2 degrees of freedom
	GED, // shape p (2 = normal)
}

GARCH_Spec :: struct {
	variance: GARCH_Variance,
	dist:     GARCH_Dist,
	p:        int, // ARCH order (≤ 0: 1)
	q:        int, // GARCH order (< 0: 1; 0 = ARCH)
}

GARCH_Params :: struct {
	omega:        f64,
	alpha:        []f64,
	beta:         []f64,
	gamma:        f64, // GJR leverage coefficient (all ARCH lags)
	egarch_gamma: []f64, // EGARCH asymmetry coefficient per ARCH lag
	nu:           f64,
	shape:        f64,
	model_type:   GARCH_Model_Type, // legacy; variance / dist take precedence when set
	variance:     GARCH_Variance,
	dist:         GARCH_Dist,
	p:            int,
	q:            int,
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
	persistence:        f64, // α + β (+ γ/2 for GJR); Σ β for EGARCH (log-variance persistence)
}

GARCH_Forecast :: struct {
	variance_forecast: f64,
	std_forecast:      f64,
	confidence_95:     [2]f64,
}

// Effective variance equation / distribution: the explicit fields, or the
// legacy model_type for parameters built by hand
_garch_variance_kind :: proc(params: ^GARCH_Params) -> GARCH_Variance {
	if params.variance != .GARCH do return params.variance
	return .GJR if params.model_type == .GJR_GARCH else .GARCH
}

_garch_dist_kind :: proc(params: ^GARCH_Params) -> GARCH_Dist {
	if params.dist != .Normal do return params.dist
	#partial switch params.model_type {
	case .StudentT:
		return .StudentT
	case .GED:
		return .GED
	}
	return .Normal
}

// Number of free parameters of a spec
_garch_n_params :: proc(variance: GARCH_Variance, dist: GARCH_Dist, p, q: int) -> int {
	n := 1 + p + q
	if variance == .GJR do n += 1
	if variance == .EGARCH do n += p
	if dist != .Normal do n += 1
	return n
}

_EGARCH_EABS :: 0.7978845608028654 // E|z| = √(2/π) for N(0, 1), as the arch package

// ============================================================================
// Parameter Transformation (unconstrained x → model parameters)
// ============================================================================
// GARCH / GJR: ω = exp(x₀ - 10); α, β, γ = sigmoid, rescaled so that the
// persistence α + β (+ γ/2) stays below 0.999.
// EGARCH: ω, α, γ unconstrained; β = tanh, rescaled so Σ|β| < 0.999.
// Then ν = 2 + exp(x) (Student-t) or shape = exp(x) (GED).

unpack_params :: proc(x: []f64, params: ^GARCH_Params) {
	p, q := params.p, params.q
	vk := _garch_variance_kind(params)
	idx := 1 + p + q
	if vk == .EGARCH {
		params.omega = x[0]
		for i in 0 ..< p {params.alpha[i] = x[1 + i]}
		sum_abs := 0.0
		for j in 0 ..< q {
			params.beta[j] = math.tanh(x[1 + p + j])
			sum_abs += math.abs(params.beta[j])
		}
		if sum_abs > 0.999 {
			for j in 0 ..< q {params.beta[j] *= 0.999 / sum_abs}
		}
		for i in 0 ..< p {params.egarch_gamma[i] = x[idx + i]}
		idx += p
	} else {
		params.omega = math.exp(x[0] - 10.0)
		sum_raw := 0.0
		for i in 0 ..< p {
			params.alpha[i] = 1.0 / (1.0 + math.exp(-x[1 + i]))
			sum_raw += params.alpha[i]
		}
		for j in 0 ..< q {
			params.beta[j] = 1.0 / (1.0 + math.exp(-x[1 + p + j]))
			sum_raw += params.beta[j]
		}
		// GJR: with symmetric shocks the leverage term adds γ/2 to the persistence
		if vk == .GJR {
			params.gamma = 1.0 / (1.0 + math.exp(-x[idx]))
			sum_raw += 0.5 * params.gamma
			idx += 1
		}
		if sum_raw > 0.999 {
			scale := 0.999 / sum_raw
			for i in 0 ..< p {params.alpha[i] *= scale}
			for j in 0 ..< q {params.beta[j] *= scale}
			if vk == .GJR {params.gamma *= scale}
		}
	}
	switch _garch_dist_kind(params) {
	case .Normal:
	case .StudentT:
		params.nu = 2.0 + math.exp(x[idx])
	case .GED:
		params.shape = math.exp(x[idx])
	}
}

// ============================================================================
// Conditional variance and log-likelihood
// ============================================================================

// Conditional variance at t from residuals and cond_var before t (GARCH / GJR):
// ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j} (+ γ Σ ε²_{t-i} 1[ε_{t-i} < 0] for GJR)
_garch_variance_at :: proc(params: ^GARCH_Params, residuals, cond_var: []f64, t: int) -> f64 {
	if _garch_variance_kind(params) == .EGARCH {
		// ln σ²_t = ω + Σ α_i (|z| - E|z|) + Σ γ_i z + Σ β_j ln σ²_{t-j}
		lv := params.omega
		for i in 0 ..< params.p {
			if t - i - 1 >= 0 {
				z := residuals[t - i - 1] / math.sqrt(cond_var[t - i - 1])
				lv += params.alpha[i] * (math.abs(z) - _EGARCH_EABS) + params.egarch_gamma[i] * z
			}
		}
		for j in 0 ..< params.q {
			if t - j - 1 >= 0 {lv += params.beta[j] * math.ln(cond_var[t - j - 1])}
		}
		return math.exp(math.clamp(lv, -50, 50))
	}
	variance := params.omega
	for i in 0 ..< params.p {
		if t - i - 1 >= 0 {variance += params.alpha[i] * residuals[t - i - 1] * residuals[t - i - 1]}
	}
	for j in 0 ..< params.q {
		if t - j - 1 >= 0 {variance += params.beta[j] * cond_var[t - j - 1]}
	}
	if _garch_variance_kind(params) == .GJR && params.gamma != 0.0 {
		for i in 0 ..< params.p {
			if t - i - 1 >= 0 && residuals[t - i - 1] < 0.0 {
				variance += params.gamma * residuals[t - i - 1] * residuals[t - i - 1]
			}
		}
	}
	if variance < 1e-8 {variance = 1e-8}
	return variance
}

// Log density of a standardised residual z with variance σ² under the fitted distribution
_garch_log_density :: proc(params: ^GARCH_Params, z, var: f64) -> f64 {
	switch _garch_dist_kind(params) {
	case .StudentT:
		nu := params.nu
		if nu > 2.0 {
			t1, _ := math.lgamma((nu + 1.0) / 2.0)
			t2, _ := math.lgamma(nu / 2.0)
			return t1 - t2 - 0.5 * math.ln_f64((nu - 2.0) * math.PI) - 0.5 * math.ln_f64(var) - ((nu + 1.0) / 2.0) * math.ln_f64(1.0 + (z * z) / (nu - 2.0))
		}
	case .GED:
		shape := params.shape
		if shape > 0.0 {
			// unit-variance GED: f(z) = p / (2 λ Γ(1/p)) exp(-|z/λ|^p), λ = sqrt(Γ(1/p)/Γ(3/p))
			lg1, _ := math.lgamma(1.0 / shape)
			lg3, _ := math.lgamma(3.0 / shape)
			lambda := math.exp_f64(0.5 * (lg1 - lg3))
			return math.ln_f64(shape) - math.ln_f64(2.0) - lg1 - math.ln_f64(lambda) - 0.5 * math.ln_f64(var) - math.pow(math.abs(z / lambda), shape)
		}
	case .Normal:
	}
	return -0.5 * (math.ln_f64(2.0 * math.PI) + math.ln_f64(var) + z * z)
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
	m := max(params.p, params.q)

	cond_var := make([]f64, n, allocator)
	defer delete(cond_var, allocator)

	uncond_var := 0.0
	for i in 0 ..< n {uncond_var += residuals[i] * residuals[i]}
	uncond_var /= f64(n)

	for i in 0 ..< m {cond_var[i] = uncond_var}
	for t in m ..< n {cond_var[t] = _garch_variance_at(params, residuals, cond_var, t)}

	log_lik := 0.0
	for t in m ..< n {
		z := residuals[t] / math.sqrt_f64(cond_var[t])
		log_lik += _garch_log_density(params, z, cond_var[t])
	}
	return -log_lik
}

// ============================================================================
// GARCH Parameter Estimation
// ============================================================================

// Legacy entry point: model_type selects variance equation and distribution
// (ARCH / GARCH → GARCH + normal, GJR_GARCH → GJR + normal, StudentT → GARCH +
// Student-t, GED → GARCH + GED). Use garch_fit_spec to combine them freely.
garch_fit :: proc(
	residuals: []f64,
	model_type: GARCH_Model_Type = .GARCH,
	p: int = 1,
	q: int = 1,
	max_iter: int = 2000,
	tolerance: f64 = 1e-4,
	allocator: mem.Allocator = context.allocator,
) -> GARCH_Result {
	spec := GARCH_Spec {
		p = p,
		q = q,
	}
	#partial switch model_type {
	case .GJR_GARCH:
		spec.variance = .GJR
	case .StudentT:
		spec.dist = .StudentT
	case .GED:
		spec.dist = .GED
	}
	res := _garch_fit_impl(residuals, spec, max_iter, tolerance, allocator, true)
	res.params.model_type = model_type
	return res
}

// Fit any combination of variance equation (GARCH, GJR, EGARCH) and error
// distribution (normal, Student-t, GED) by conditional maximum likelihood:
// Adam on the unconstrained parameters, then a Nelder-Mead polish.
// Free the result with garch_result_free.
garch_fit_spec :: proc(
	residuals: []f64,
	spec: GARCH_Spec,
	max_iter: int = 2000,
	tolerance: f64 = 1e-4,
	allocator: mem.Allocator = context.allocator,
	verbose := false,
) -> GARCH_Result {
	s := spec
	if s.p <= 0 do s.p = 1
	if s.q < 0 do s.q = 1
	return _garch_fit_impl(residuals, s, max_iter, tolerance, allocator, verbose)
}

garch_result_free :: proc(res: ^GARCH_Result, allocator: mem.Allocator = context.allocator) {
	delete(res.params.alpha, allocator)
	delete(res.params.beta, allocator)
	delete(res.params.egarch_gamma, allocator)
	delete(res.conditional_var, allocator)
	delete(res.standardized_resid, allocator)
}

_garch_fit_impl :: proc(
	residuals: []f64,
	spec: GARCH_Spec,
	max_iter: int,
	tolerance: f64,
	allocator: mem.Allocator,
	verbose: bool,
) -> GARCH_Result {
	n := len(residuals)
	p, q := spec.p, spec.q

	params := GARCH_Params {
		variance = spec.variance,
		dist     = spec.dist,
		p        = p,
		q        = q,
	}
	#partial switch spec.variance {
	case .GJR:
		params.model_type = .GJR_GARCH
	case:
		params.model_type = .GARCH
	}

	// Need at least one observation beyond the lag order to fit anything
	if p < 0 || q < 0 || n <= max(p, q) {
		return GARCH_Result{params = params}
	}

	params.alpha = make([]f64, p, allocator)
	params.beta = make([]f64, q, allocator)
	if spec.variance == .EGARCH do params.egarch_gamma = make([]f64, p, allocator)

	n_params := _garch_n_params(spec.variance, spec.dist, p, q)
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
	idx := 1 + p + q
	if spec.variance == .EGARCH {
		// ln σ² ≈ ln var(ε): ω = (1 - Σβ) ln var, α = 0.1, β = 0.9 / q, γ = 0
		v := 0.0
		for e in residuals {v += e * e}
		v = max(v / f64(n), 1e-300)
		b := 0.9 / f64(max(q, 1))
		param_vec[0] = (1 - 0.9 * f64(min(q, 1))) * math.ln(v)
		for i in 0 ..< p {param_vec[1 + i] = 0.1}
		for j in 0 ..< q {param_vec[1 + p + j] = math.atanh(b)}
		idx += p // γ_i start at 0
	} else {
		param_vec[0] = -5.0
		if spec.variance == .GJR {
			param_vec[idx] = math.ln_f64(0.05 / 0.95) // gamma = 0.05
			idx += 1
		}
	}
	switch spec.dist {
	case .Normal:
	case .StudentT:
		param_vec[idx] = math.ln_f64(3.0) // ν = 2 + 3 = 5
	case .GED:
		param_vec[idx] = math.ln_f64(2.0) // shape = 2 (normal)
	}
	best_loss := math.F64_MAX
	best_params := make([]f64, n_params, allocator)
	defer delete(best_params, allocator)
	copy(best_params, param_vec)

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

		if iter > 100 && math.abs(prev_loss - loss) < tolerance {
			n_iter = iter + 1
			if verbose do fmt.printf("GARCH converged at iteration %d (loss change: %.8f)\n", iter, math.abs(prev_loss - loss))
			converged = true
			break
		}
		// stop if the loss keeps increasing markedly
		if iter > 100 && loss > prev_loss + 1.0 {
			if verbose do fmt.println("Loss increasing, stopping early")
			break
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

		if verbose && iter % 100 == 0 {
			unpack_params(param_vec, &params)
			fmt.printf("GARCH iter %d: loss=%.4f, ω=%.6f\n", iter, loss, params.omega)
		}
	}

	// Adam with a fixed step stalls ~1e-2 log-likelihood units short of the
	// optimum: polish the best point with Nelder-Mead (restarted)
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
	n_restarts := 2 if spec.variance != .EGARCH && spec.dist == .Normal else 4
	for _ in 0 ..< n_restarts {
		x_nm, f_nm, _ := nelder_mead(nm_obj, &nm_ctx, best_params, 2000 * max(n_params / 4, 1), 1e-10, allocator)
		if f_nm < best_loss {
			best_loss = f_nm
			copy(best_params, x_nm)
		}
		delete(x_nm, allocator)
	}

	unpack_params(best_params, &params)

	// Final conditional variances
	m := max(p, q)
	cond_var := make([]f64, n, allocator)
	std_resid := make([]f64, n, allocator)
	uncond_var := 0.0
	for i in 0 ..< n {uncond_var += residuals[i] * residuals[i]}
	uncond_var /= f64(n)
	for i in 0 ..< m {
		cond_var[i] = uncond_var
		std_resid[i] = residuals[i] / math.sqrt_f64(cond_var[i])
	}
	for t in m ..< n {
		cond_var[t] = _garch_variance_at(&params, residuals, cond_var, t)
		std_resid[t] = residuals[t] / math.sqrt_f64(cond_var[t])
	}

	log_lik := -best_loss
	k := f64(n_params)
	aic := 2.0 * k - 2.0 * log_lik
	bic := k * math.ln_f64(f64(n)) - 2.0 * log_lik

	persistence := 0.0
	if spec.variance != .EGARCH {for i in 0 ..< p {persistence += params.alpha[i]}}
	for j in 0 ..< q {persistence += params.beta[j]}
	if spec.variance == .GJR {persistence += 0.5 * params.gamma}

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
// EGARCH: the log variance is forecast (E|z| - √(2/π) = 0 and E z = 0 for future
// shocks) and exponentiated, exact for h = 1 and exp(E ln σ²) ≤ E σ² beyond.
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
	vk := _garch_variance_kind(params)
	is_gjr := vk == .GJR

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
		v: f64
		if vk == .EGARCH {
			lv := params.omega
			for i in 0 ..< p {
				if k - i - 1 >= 0 && k - i - 1 < n {
					z := residuals[k - i - 1] / math.sqrt(s2[k - i - 1])
					lv += params.alpha[i] * (math.abs(z) - _EGARCH_EABS) + params.egarch_gamma[i] * z
				}
			}
			for j in 0 ..< q {
				if k - j - 1 >= 0 {lv += params.beta[j] * math.ln(s2[k - j - 1])}
			}
			v = math.exp(math.clamp(lv, -50, 50))
		} else {
			v = params.omega
			for i in 0 ..< p {
				if k - i - 1 >= 0 {
					v += params.alpha[i] * sq[k - i - 1]
					if is_gjr {v += params.gamma * neg_sq[k - i - 1]}
				}
			}
			for j in 0 ..< q {
				if k - j - 1 >= 0 {v += params.beta[j] * s2[k - j - 1]}
			}
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
	dk := _garch_dist_kind(params)
	switch {
	case dk == .StudentT && params.nu > 2.0:
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
	case dk == .GED && params.shape > 0.0:
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
