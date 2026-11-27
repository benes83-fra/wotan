package analytics

import l "../linalg"
import opt "../optimize"
import st "../stats"
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
// GARCH / GJR: ω = exp(x₀ - 10). The components of the persistence, α_i, β_j
// and γ/2, are P·w_k: P = 0.999·sigmoid(x₁) is their sum and w = softmax(0,
// x₂, ..., x_c) their shares. Smooth and one-to-one onto the region with all
// components positive and persistence below 0.999.
// EGARCH: ω, α, γ unconstrained; β_j = 0.999·x_j / (1 + Σ|x|), so Σ|β| < 0.999.
// Then ν = 2 + exp(x) (Student-t) or shape = exp(x) (GED).

unpack_params :: proc(x: []f64, params: ^GARCH_Params) {
	p, q := params.p, params.q
	vk := _garch_variance_kind(params)
	idx := 1 + p + q
	if vk == .EGARCH {
		params.omega = x[0]
		for i in 0 ..< p {params.alpha[i] = x[1 + i]}
		sum_abs := 0.0
		for j in 0 ..< q {sum_abs += math.abs(x[1 + p + j])}
		for j in 0 ..< q {params.beta[j] = 0.999 * x[1 + p + j] / (1 + sum_abs)}
		for i in 0 ..< p {params.egarch_gamma[i] = x[idx + i]}
		idx += p
	} else {
		params.omega = math.exp(x[0] - 10.0)
		c := p + q + (1 if vk == .GJR else 0)
		P := 0.999 / (1.0 + math.exp(-x[1]))
		// softmax with the first logit fixed at 0, shifted by the largest logit
		top := 0.0
		for k in 1 ..< c {top = max(top, x[1 + k])}
		total := math.exp(-top)
		for k in 1 ..< c {total += math.exp(x[1 + k] - top)}
		for k in 0 ..< c {
			share := math.exp((0 if k == 0 else x[1 + k]) - top) / total
			switch {
			case k < p:
				params.alpha[k] = P * share
			case k < p + q:
				params.beta[k - p] = P * share
			case:
				params.gamma = 2 * P * share // γ/2 is the component
			}
		}
		if vk == .GJR do idx += 1
	}
	switch _garch_dist_kind(params) {
	case .Normal:
	case .StudentT:
		params.nu = 2.0 + math.exp(x[idx])
	case .GED:
		params.shape = math.exp(x[idx])
	}
}

// x for given GARCH / GJR components (α_i, β_j, then γ/2 for GJR; all > 0 with
// sum < 0.999) and ω: the inverse of unpack_params' persistence block
_garch_pack_components :: proc(x: []f64, omega: f64, comps: []f64) {
	x[0] = math.ln(omega) + 10
	P := 0.0
	for v in comps {P += v}
	x[1] = math.ln(P / (0.999 - P))
	for k in 1 ..< len(comps) {x[1 + k] = math.ln(comps[k] / comps[0])}
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
		if t - i - 1 >=
		   0 {variance += params.alpha[i] * residuals[t - i - 1] * residuals[t - i - 1]}
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
			return(
				t1 -
				t2 -
				0.5 * math.ln_f64((nu - 2.0) * math.PI) -
				0.5 * math.ln_f64(var) -
				((nu + 1.0) / 2.0) * math.ln_f64(1.0 + (z * z) / (nu - 2.0)) \
			)
		}
	case .GED:
		shape := params.shape
		if shape > 0.0 {
			// unit-variance GED: f(z) = p / (2 λ Γ(1/p)) exp(-|z/λ|^p), λ = sqrt(Γ(1/p)/Γ(3/p))
			lg1, _ := math.lgamma(1.0 / shape)
			lg3, _ := math.lgamma(3.0 / shape)
			lambda := math.exp_f64(0.5 * (lg1 - lg3))
			return(
				math.ln_f64(shape) -
				math.ln_f64(2.0) -
				lg1 -
				math.ln_f64(lambda) -
				0.5 * math.ln_f64(var) -
				math.pow(math.abs(z / lambda), shape) \
			)
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
// verbose prints the optimizer's progress.
garch_fit :: proc(
	residuals: []f64,
	model_type: GARCH_Model_Type = .GARCH,
	p: int = 1,
	q: int = 1,
	max_iter: int = 2000,
	tolerance: f64 = 1e-4,
	allocator: mem.Allocator = context.allocator,
	verbose := false,
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
	res := _garch_fit_impl(residuals, spec, max_iter, tolerance, allocator, verbose)
	res.params.model_type = model_type
	return res
}

// Fit any combination of variance equation (GARCH, GJR, EGARCH) and error
// distribution (normal, Student-t, GED) by conditional maximum likelihood:
// L-BFGS (optimize.lbfgs_minimize) with the analytic gradient on the
// unconstrained parameters, from three starting points. max_iter bounds the
// iterations per start; the fit has converged when no gradient component
// exceeds tolerance.
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
	k_var := n_params - (0 if spec.dist == .Normal else 1)
	work := _Garch_Work {
		params    = &params,
		residuals = residuals,
		k_var     = k_var,
		cond_var  = make([]f64, n, allocator),
		deriv     = make([]f64, n * k_var, allocator),
		g_nat     = make([]f64, n_params, allocator),
		jac       = make([]f64, n_params * n_params, allocator),
	}
	defer _garch_work_free(&work, allocator)

	// Starting points in the unconstrained space: typical daily-return
	// dynamics at the scale of the data, and a low- and a high-persistence
	// variant. L-BFGS runs from each; the best fit wins.
	v := 0.0
	for e in residuals {v += e * e}
	v = max(v / f64(n), 1e-300)
	Start :: struct {
		alpha, beta, gamma: f64, // totals over the lags
	}
	starts := [?]Start{{0.05, 0.90, 0.06}, {0.15, 0.60, 0.10}, {0.03, 0.94, 0.04}}
	x := make([]f64, n_params, allocator)
	best_x := make([]f64, n_params, allocator)
	defer delete(x, allocator)
	defer delete(best_x, allocator)
	best_loss := math.INF_F64
	converged := false
	n_iter := 0
	for st, si in starts {
		for &e in x {e = 0}
		idx := 1 + p + q
		a := st.alpha / f64(p)
		b := st.beta / f64(max(q, 1))
		if spec.variance == .EGARCH {
			// ln σ² ≈ ln var(ε): ω = (1 - Σβ) ln var, γ = 0; β_j = b from x_j = b (1 + s) / 0.999
			x[0] = (1 - st.beta * f64(min(q, 1))) * math.ln(v)
			for i in 0 ..< p {x[1 + i] = 2 * a}
			B := b * f64(q)
			sx := B / (0.999 - B)
			for j in 0 ..< q {x[1 + p + j] = b * (1 + sx) / 0.999}
			idx += p
		} else {
			// ARCH (q = 0): α totals 0.5
			if q == 0 {a = 0.5 / f64(p)}
			comps := make([]f64, p + q + 1, context.temp_allocator)
			for i in 0 ..< p {comps[i] = a}
			for j in 0 ..< q {comps[p + j] = b}
			c := p + q
			if spec.variance == .GJR {
				comps[c] = 0.5 * st.gamma
				c += 1
				idx += 1
			}
			persistence := 0.0
			for e in comps[:c] {persistence += e}
			_garch_pack_components(x, v * (1 - persistence), comps[:c])
		}
		switch spec.dist {
		case .Normal:
		case .StudentT:
			x[idx] = math.ln(6.0) // ν = 8
		case .GED:
			x[idx] = math.ln(1.5)
		}
		fit := opt.lbfgs_minimize(
			_garch_value_grad,
			x,
			&work,
			max_iter,
			tolerance,
			allocator = allocator,
		)
		f, iters, conv := fit.f, fit.n_iter, fit.converged
		if verbose {
			fmt.printf(
				"GARCH start %d: -loglik %.6f after %d L-BFGS iterations, converged %v\n",
				si,
				f,
				iters,
				conv,
			)
		}
		// fits within rounding of each other are ties: prefer one that converged
		tie := 1e-9 * max(1, math.abs(f))
		if f < best_loss - tie || (f <= best_loss + tie && conv && !converged) {
			best_loss = f
			copy(best_x, x)
			converged = conv
			n_iter = iters
		}
	}

	unpack_params(best_x, &params)
	best_loss = _garch_neg_log_likelihood(&params, residuals, allocator)

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
		params             = params,
		log_likelihood     = log_lik,
		aic                = aic,
		bic                = bic,
		conditional_var    = cond_var,
		standardized_resid = std_resid,
		converged          = converged,
		n_iterations       = n_iter, // L-BFGS iterations of the winning start
		persistence        = persistence,
	}
}

// ============================================================================
// Likelihood gradient
// ============================================================================

// Buffers for the negative log-likelihood and its gradient
_Garch_Work :: struct {
	params:    ^GARCH_Params,
	residuals: []f64,
	k_var:     int, // parameters of the variance equation (all but ν / shape)
	cond_var:  []f64, // σ²_t
	deriv:     []f64, // n × k_var: dσ²_t/dθ (GARCH, GJR) or d ln σ²_t/dθ (EGARCH)
	g_nat:     []f64, // gradient in the model parameters θ
	jac:       []f64, // dθ_k / dx_l, row k
}

_garch_work_free :: proc(w: ^_Garch_Work, allocator: mem.Allocator) {
	delete(w.cond_var, allocator)
	delete(w.deriv, allocator)
	delete(w.g_nat, allocator)
	delete(w.jac, allocator)
}

// ψ(x) for x > 0: recurrence up to x ≥ 6, then the asymptotic series
_garch_digamma :: proc(x0: f64) -> f64 {
	x, r := x0, 0.0
	for x < 6 {r -= 1 / x; x += 1}
	f := 1 / (x * x)
	return(
		r +
		math.ln(x) -
		0.5 / x -
		f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132)))) \
	)
}

// Jacobian of unpack_params at x: w.jac[k * n_params + l] = dθ_k / dx_l, with θ
// in the layout of x (ω, α, β, γ or EGARCH γ, then ν / shape)
_garch_unpack_jacobian :: proc(w: ^_Garch_Work, x: []f64) {
	params := w.params
	p, q := params.p, params.q
	k := len(x)
	J := w.jac
	for &e in J {e = 0}
	vk := _garch_variance_kind(params)
	idx := 1 + p + q
	if vk == .EGARCH {
		J[0] = 1
		for i in 0 ..< p {
			J[(1 + i) * k + 1 + i] = 1
			J[(idx + i) * k + idx + i] = 1
		}
		// β_j = 0.999 x_j / (1 + s), s = Σ|x|
		s := 0.0
		for j in 0 ..< q {s += math.abs(x[1 + p + j])}
		for j in 0 ..< q {
			for l in 0 ..< q {
				xl := x[1 + p + l]
				sgn := 1.0 if xl > 0 else (-1.0 if xl < 0 else 0.0)
				v := -x[1 + p + j] * sgn / ((1 + s) * (1 + s))
				if j == l do v += 1 / (1 + s)
				J[(1 + p + j) * k + 1 + p + l] = 0.999 * v
			}
		}
		idx += p
	} else {
		J[0] = params.omega
		// component_a = P·w_a: dP/dx₁ = P (1 - P/0.999), dw_a/dx_b = w_a (δ_ab - w_b)
		c := p + q + (1 if vk == .GJR else 0)
		comp := make([]f64, c, context.temp_allocator)
		P := 0.0
		for a in 0 ..< c {
			switch {
			case a < p:
				comp[a] = params.alpha[a]
			case a < p + q:
				comp[a] = params.beta[a - p]
			case:
				comp[a] = 0.5 * params.gamma
			}
			P += comp[a]
		}
		for a in 0 ..< c {
			wa := comp[a] / P
			scale := 2.0 if a == p + q else 1.0 // θ for the γ component is 2·(γ/2)
			J[(1 + a) * k + 1] = scale * comp[a] * (1 - P / 0.999)
			for b in 1 ..< c {
				wb := comp[b] / P
				J[(1 + a) * k + 1 + b] = scale * P * wa * ((1 if a == b else 0) - wb)
			}
		}
		if vk == .GJR do idx += 1
	}
	switch _garch_dist_kind(params) {
	case .Normal:
	case .StudentT:
		J[idx * k + idx] = params.nu - 2
	case .GED:
		J[idx * k + idx] = params.shape
	}
}

// opt.ValueGradFunc for lbfgs_minimize; user_data is the ^_Garch_Work
_garch_value_grad :: proc(x, grad: []f64, user_data: rawptr) -> f64 {
	return _garch_nll_and_grad((^_Garch_Work)(user_data), x, grad)
}

// Negative log-likelihood at the unconstrained x (as _garch_neg_log_likelihood)
// and its gradient with respect to x. The derivatives of σ²_t follow the
// variance recursion forward in time.
_garch_nll_and_grad :: proc(w: ^_Garch_Work, x, grad: []f64) -> f64 {
	params := w.params
	unpack_params(x, params)
	e := w.residuals
	n := len(e)
	p, q := params.p, params.q
	vk := _garch_variance_kind(params)
	dk := _garch_dist_kind(params)
	kv := w.k_var
	m := max(p, q)
	s2 := w.cond_var
	D := w.deriv
	g := w.g_nat
	for &v in g {v = 0}

	uncond := 0.0
	for v in e {uncond += v * v}
	uncond /= f64(n)
	for t in 0 ..< m {
		s2[t] = uncond
		for k in 0 ..< kv {D[t * kv + k] = 0}
	}

	// constants of the error density and their derivatives in ν / shape
	nu, c_t, dc_t := params.nu, 0.0, 0.0
	if dk == .StudentT && nu <= 2 do dk = .Normal
	if dk == .StudentT {
		t1, _ := math.lgamma((nu + 1) / 2)
		t2, _ := math.lgamma(nu / 2)
		c_t = t1 - t2 - 0.5 * math.ln((nu - 2) * math.PI)
		dc_t = 0.5 * _garch_digamma((nu + 1) / 2) - 0.5 * _garch_digamma(nu / 2) - 0.5 / (nu - 2)
	}
	sh, lam, c_g, dlnlam, dc_g := params.shape, 0.0, 0.0, 0.0, 0.0
	if dk == .GED && sh <= 0 do dk = .Normal
	if dk == .GED {
		lg1, _ := math.lgamma(1 / sh)
		lg3, _ := math.lgamma(3 / sh)
		lam = math.exp(0.5 * (lg1 - lg3))
		c_g = math.ln(sh) - math.ln(2.0) - lg1 - math.ln(lam)
		psi1, psi3 := _garch_digamma(1 / sh), _garch_digamma(3 / sh)
		dlnlam = 0.5 * (-psi1 + 3 * psi3) / (sh * sh)
		dc_g = 1 / sh + psi1 / (sh * sh) - dlnlam
	}
	g_dist := 0.0

	nll := 0.0
	for t in m ..< n {
		Dt := D[t * kv:][:kv]
		for k in 0 ..< kv {Dt[k] = 0}
		if vk == .EGARCH {
			lv := params.omega
			Dt[0] += 1
			for i in 0 ..< p {
				s := t - i - 1
				z := e[s] / math.sqrt(s2[s])
				lv += params.alpha[i] * (math.abs(z) - _EGARCH_EABS) + params.egarch_gamma[i] * z
				Dt[1 + i] += math.abs(z) - _EGARCH_EABS
				Dt[1 + p + q + i] += z
				// z depends on θ through σ²_s: dz = -z/2 · d ln σ²_s
				sgn := 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)
				c := -0.5 * z * (params.alpha[i] * sgn + params.egarch_gamma[i])
				Ds := D[s * kv:][:kv]
				for k in 0 ..< kv {Dt[k] += c * Ds[k]}
			}
			for j in 0 ..< q {
				s := t - j - 1
				ls := math.ln(s2[s])
				lv += params.beta[j] * ls
				Dt[1 + p + j] += ls
				Ds := D[s * kv:][:kv]
				for k in 0 ..< kv {Dt[k] += params.beta[j] * Ds[k]}
			}
			if lv < -50 || lv > 50 {for k in 0 ..< kv {Dt[k] = 0}}
			s2[t] = math.exp(math.clamp(lv, -50, 50))
		} else {
			v := params.omega
			Dt[0] += 1
			for i in 0 ..< p {
				s := t - i - 1
				v += params.alpha[i] * e[s] * e[s]
				Dt[1 + i] += e[s] * e[s]
			}
			for j in 0 ..< q {
				s := t - j - 1
				v += params.beta[j] * s2[s]
				Dt[1 + p + j] += s2[s]
				Ds := D[s * kv:][:kv]
				for k in 0 ..< kv {Dt[k] += params.beta[j] * Ds[k]}
			}
			if vk == .GJR {
				for i in 0 ..< p {
					s := t - i - 1
					if e[s] < 0 {
						if params.gamma != 0 {v += params.gamma * e[s] * e[s]}
						Dt[1 + p + q] += e[s] * e[s]
					}
				}
			}
			if v < 1e-8 {
				v = 1e-8
				for k in 0 ..< kv {Dt[k] = 0}
			}
			s2[t] = v
		}

		// log density ℓ_t, a = dℓ_t/dσ²_t and dℓ_t/d(ν or shape)
		v := s2[t]
		z := e[t] / math.sqrt(v)
		ell, a: f64
		switch dk {
		case .StudentT:
			u := z * z / (nu - 2)
			ell = c_t - 0.5 * math.ln(v) - ((nu + 1) / 2) * math.ln(1 + u)
			a = (-0.5 + 0.5 * (nu + 1) * u / (1 + u)) / v
			g_dist -= dc_t - 0.5 * math.ln(1 + u) + 0.5 * (nu + 1) * u / ((1 + u) * (nu - 2))
		case .GED:
			A := math.pow(math.abs(z / lam), sh)
			ell = c_g - 0.5 * math.ln(v) - A
			a = (-0.5 + 0.5 * sh * A) / v
			dA := A * (math.ln(math.abs(z / lam)) - sh * dlnlam) if A > 0 else 0
			g_dist -= dc_g - dA
		case .Normal:
			ell = -0.5 * (math.ln(2 * math.PI) + math.ln(v) + z * z)
			a = 0.5 * (z * z - 1) / v
		}
		nll -= ell
		// EGARCH tracks d ln σ², so dσ² = σ² · Dt
		sc := a * v if vk == .EGARCH else a
		for k in 0 ..< kv {g[k] -= sc * Dt[k]}
	}
	if len(g) > kv do g[kv] = g_dist

	// chain rule through the parameter transformation
	_garch_unpack_jacobian(w, x)
	nx := len(x)
	for l in 0 ..< nx {
		s := 0.0
		for k in 0 ..< nx {s += w.jac[k * nx + l] * g[k]}
		grad[l] = s
	}
	return nll
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
		return GARCH_Forecast {
			variance_forecast = nan,
			std_forecast = nan,
			confidence_95 = {nan, nan},
		}
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
					lv +=
						params.alpha[i] * (math.abs(z) - _EGARCH_EABS) + params.egarch_gamma[i] * z
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
			return st.student_t_cdf(st.Student_T{nu}, x)
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
