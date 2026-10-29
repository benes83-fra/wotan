// wotan/finance/risk.odin
package finance

import a "../analytics"
import ml "../analytics/ML"
import w "../core"
import l "../linalg"
import opt "../optimize"
import "base:intrinsics"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:simd"
import "core:slice"

// ============================================================================
// Risk Metrics Structures
// ============================================================================

RiskMetrics :: struct {
	var_95:          f64,
	var_99:          f64,
	cvar_95:         f64,
	cvar_99:         f64,
	volatility:      f64,
	volatility_ewma: f64,
	max_drawdown:    f64,
	sharpe_ratio:    f64,
	sortino_ratio:   f64,
	beta:            f64,
	correlation:     f64,
	tracking_error:  f64,
	calmar_ratio:    f64,
	win_rate:        f64,
	profit_factor:   f64,
	avg_win:         f64,
	avg_loss:        f64,
}

VaRMethod :: enum {
	Historical,
	Parametric,
	MonteCarlo,
	GARCH, // ✅ NEW: GARCH-based dynamic VaR
}

// ✅ NEW: GARCH Risk Metrics (extends RiskMetrics with GARCH-specific fields)
GARCH_RiskMetrics :: struct {
	// Standard metrics
	base:              RiskMetrics,
	// GARCH-specific
	garch_omega:       f64,
	garch_alpha:       f64,
	garch_beta:        f64,
	garch_nu:          f64,
	garch_shape:       f64,
	garch_persistence: f64,
	long_run_vol:      f64,
	// Current GARCH-based VaR
	current_vol:       f64,
	current_var_95:    f64,
	current_var_99:    f64,
	current_cvar_95:   f64,
	current_cvar_99:   f64,
	// Backtesting results
	backtest_95:       VaR_BacktestResult,
	backtest_99:       VaR_BacktestResult,
	converged:         bool,
	n_iterations:      int,
}

// ✅ NEW: VaR Backtesting Result
VaR_BacktestResult :: struct {
	n_obs:             int,
	n_breaches:        int,
	expected_breaches: f64,
	breach_rate:       f64,
	kupiec_stat:       f64,
	kupiec_pvalue:     f64,
	passes_test:       bool, // True if p-value > 0.05
}
// ============================================================================
// Extreme Value Theory (EVT) - Peak-over-Threshold Method
// ============================================================================

EVT_Result :: struct {
	threshold:     f64, // Threshold u
	xi:            f64, // Shape parameter ξ
	beta:          f64, // Scale parameter β
	n_exceedances: int, // Number of exceedances N_u
	n_total:       int, // Total observations n
	var_95:        f64, // 95% VaR
	var_99:        f64, // 99% VaR
	cvar_95:       f64, // 95% CVaR (Expected Shortfall)
	cvar_99:       f64, // 99% CVaR
	converged:     bool,
}

GARCH_EVT_Result :: struct {
	// GARCH parameters
	garch_omega:   f64,
	garch_alpha:   f64,
	garch_beta:    f64,
	// EVT parameters (fitted on standardized residuals)
	evt_xi:        f64,
	evt_beta:      f64,
	evt_threshold: f64,
	n_exceedances: int,
	// Dynamic VaR series
	var_95_series: []f64,
	var_99_series: []f64,
	// Backtesting results
	backtest_95:   VaR_BacktestResult,
	backtest_99:   VaR_BacktestResult,
}

// GPD (Generalized Pareto Distribution) PDF
// f(y) = (1/β) * (1 + ξ*y/β)^(-1/ξ - 1) for ξ ≠ 0
// f(y) = (1/β) * exp(-y/β) for ξ = 0 (Exponential)
gpd_pdf :: proc(y: f64, xi: f64, beta: f64) -> f64 {
	if beta <= 0.0 {return 0.0}
	if y < 0.0 {return 0.0}

	if math.abs(xi) < 1e-10 {
		// Exponential case (ξ → 0)
		return (1.0 / beta) * math.exp_f64(-y / beta)
	}

	term := 1.0 + xi * y / beta
	if term <= 0.0 {return 0.0}

	return (1.0 / beta) * math.pow(term, -1.0 / xi - 1.0)
}

// GPD CDF
// F(y) = 1 - (1 + ξ*y/β)^(-1/ξ) for ξ ≠ 0
// F(y) = 1 - exp(-y/β) for ξ = 0
gpd_cdf :: proc(y: f64, xi: f64, beta: f64) -> f64 {
	if beta <= 0.0 {return 0.0}
	if y < 0.0 {return 0.0}

	if math.abs(xi) < 1e-10 {
		// Exponential case
		return 1.0 - math.exp_f64(-y / beta)
	}

	term := 1.0 + xi * y / beta
	if term <= 0.0 {return 1.0}

	return 1.0 - math.pow(term, -1.0 / xi)
}

// GPD Quantile (Inverse CDF)
// Q(p) = (β/ξ) * [(1-p)^(-ξ) - 1] for ξ ≠ 0
// Q(p) = -β * ln(1-p) for ξ = 0
gpd_quantile :: proc(p: f64, xi: f64, beta: f64) -> f64 {
	if beta <= 0.0 {return 0.0}
	if p <= 0.0 {return 0.0}
	if p >= 1.0 {return math.INF_F64}

	if math.abs(xi) < 1e-10 {
		// Exponential case
		return -beta * math.ln_f64(1.0 - p)
	}

	return (beta / xi) * (math.pow(1.0 - p, -xi) - 1.0)
}

// GPD Negative Log-Likelihood for MLE
_gpd_neg_log_likelihood :: proc(params: []f64, exceedances: []f64) -> f64 {
	xi := params[0]
	beta := params[1]

	if beta <= 0.0 {return math.INF_F64}

	n := f64(len(exceedances))
	log_lik := 0.0

	for y in exceedances {
		if y < 0.0 {return math.INF_F64}

		if math.abs(xi) < 1e-10 {
			// Exponential case
			log_lik += -math.ln_f64(beta) - y / beta
		} else {
			term := 1.0 + xi * y / beta
			if term <= 0.0 {return math.INF_F64}
			log_lik += -math.ln_f64(beta) - (1.0 / xi + 1.0) * math.ln_f64(term)
		}
	}

	return -log_lik / n // Negative for minimization
}

_gpd_objective :: proc(x: []f64, user_data: rawptr) -> f64 {
	return _gpd_neg_log_likelihood(x, (^[]f64)(user_data)^)
}

// Fit GPD by maximum likelihood. beta is optimised on log scale (it spans
// orders of magnitude with the data units); `converged` reports that the
// likelihood stopped improving, not that the iteration budget ran out.
gpd_fit :: proc(
	exceedances: []f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	xi: f64,
	beta: f64,
	converged: bool,
) {
	n := len(exceedances)
	if n < 10 {return 0.0, 0.0, false}

	mean_exc := 0.0
	for y in exceedances {mean_exc += y}
	mean_exc /= f64(n)
	if !(mean_exc > 0.0) {return 0.0, 0.0, false}

	params := []opt.Calibration_Param {
		{"xi", 0.1, -0.9, 2.0, .Linear},
		{"beta", mean_exc, 1e-6 * mean_exc, 1e3 * mean_exc, .Log},
	}
	config := opt.Calibration_Config {
		optimizer      = {type = .Adam, learning_rate = 0.02, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8},
		max_iter       = 2000,
		loss_threshold = math.inf_f64(-1),
		fd_step        = 1e-6,
		stall_iters    = 100,
		stall_rel_tol  = 1e-10,
	}
	data := exceedances
	res := opt.calibrate(params, _gpd_objective, &data, config, allocator)
	defer opt.calibration_result_free(&res)
	return res.x[0], res.x[1], res.converged
}

// Left-tail EVT VaR (as a return) at confidence p from a GPD fitted to the
// N_u exceedances below threshold u out of n observations:
//   VaR = u - (β/ξ) [((n/N_u)(1-p))^(-ξ) - 1],  ξ → 0: u + β ln((n/N_u)(1-p))
_evt_tail_var :: proc(threshold, xi, beta: f64, n, n_exceedances: int, p: f64) -> f64 {
	ratio := f64(n) / f64(n_exceedances) * (1.0 - p)
	if math.abs(xi) > 1e-10 {
		return threshold - (beta / xi) * (math.pow(ratio, -xi) - 1.0)
	}
	return threshold + beta * math.ln_f64(ratio)
}

// Fit EVT model using Peak-over-Threshold method
evt_fit :: proc(
	returns: []f64,
	threshold_percentile: f64 = 0.95, // Use 95th percentile as threshold
	allocator: mem.Allocator = context.allocator,
) -> EVT_Result {
	n := len(returns)
	if n < 100 {return EVT_Result{}}

	// Sort returns to find threshold (left tail)
	sorted := make([]f64, n, context.temp_allocator)
	copy(sorted, returns)
	slice.sort(sorted)

	// Threshold is the (1-threshold_percentile) quantile
	threshold_idx := int((1.0 - threshold_percentile) * f64(n))
	if threshold_idx < 0 {threshold_idx = 0}
	threshold := sorted[threshold_idx]

	// Extract exceedances (positive values: how much returns fall below threshold)
	exceedances := make([dynamic]f64, 0, allocator)
	defer delete(exceedances)

	for r in returns {
		if r < threshold {
			// Y = u - X, where X is the return and u is threshold
			// This gives positive exceedances
			append(&exceedances, threshold - r)
		}
	}

	n_exceedances := len(exceedances)
	if n_exceedances < 10 {
		return EVT_Result{}
	}

	// Fit GPD to exceedances
	xi, beta, converged := gpd_fit(exceedances[:], allocator)

	// Calculate VaR using EVT formula for left tail
	// VaR_p = u - (β/ξ) * [(n/N_u * (1-p))^(-ξ) - 1]
	// Note the negative sign because we're modeling the left tail
	p_95 := 0.95
	p_99 := 0.99

	var_95 := _evt_tail_var(threshold, xi, beta, n, n_exceedances, p_95)
	var_99 := _evt_tail_var(threshold, xi, beta, n, n_exceedances, p_99)

	// Calculate CVaR (Expected Shortfall)
	// In losses L = -r (threshold u_L = -u, VaR_L = -VaR):
	//   ES_L = VaR_L / (1-ξ) + (β - ξ u_L) / (1-ξ)
	// so for returns: CVaR = VaR / (1-ξ) - (β + ξ u) / (1-ξ)
	cvar_95 := 0.0
	cvar_99 := 0.0

	if xi < 1.0 { 	// Finite mean condition
		cvar_95 = var_95 / (1.0 - xi) - (beta + xi * threshold) / (1.0 - xi)
		cvar_99 = var_99 / (1.0 - xi) - (beta + xi * threshold) / (1.0 - xi)
	} else {
		// Infinite mean - use VaR as proxy with safety margin
		cvar_95 = var_95 * 1.2
		cvar_99 = var_99 * 1.2
	}

	return EVT_Result {
		threshold = threshold,
		xi = xi,
		beta = beta,
		n_exceedances = n_exceedances,
		n_total = n,
		var_95 = var_95,
		var_99 = var_99,
		cvar_95 = cvar_95,
		cvar_99 = cvar_99,
		converged = converged,
	}
}
// ============================================================================
// Helper: Inverse Normal CDF (Quantile Function)
// ============================================================================

norm_inv :: proc(p: f64) -> f64 {
	a1 := -3.969683028665376e+01
	a2 := 2.209460984245205e+02
	a3 := -2.759285104469687e+02
	a4 := 1.383577518672690e+02
	a5 := -3.066479806614716e+01
	a6 := 2.506628277459239e+00

	b1 := -5.447609879822406e+01
	b2 := 1.615858368580409e+02
	b3 := -1.556989798598866e+02
	b4 := 6.680131188771972e+01
	b5 := -1.328068155288572e+01

	c1 := -7.784894002430293e-03
	c2 := -3.223964580411365e-01
	c3 := -2.400758277161838e+00
	c4 := -2.549732539343734e+00
	c5 := 4.374664141464968e+00
	c6 := 2.938163982698783e+00

	d1 := 7.784695709041462e-03
	d2 := 3.224671290700398e-01
	d3 := 2.445134137142996e+00
	d4 := 3.754408661907416e+00

	p_low := 0.02425
	p_high := 1.0 - p_low

	if p <= 0.0 {return -math.INF_F64}
	if p >= 1.0 {return math.INF_F64}

	q, r, x: f64

	if p < p_low {
		q = math.sqrt_f64(-2.0 * math.ln_f64(p))
		x =
			(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
			((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
	} else if p <= p_high {
		q = p - 0.5
		r = q * q
		x =
			(((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) *
			q /
			(((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
	} else {
		q = math.sqrt_f64(-2.0 * math.ln_f64(1.0 - p))
		x =
			-(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
			((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
	}

	return x
}


// Student-t quantile (inverse CDF) for nu ≥ 1: Hill's approximation (CACM
// algorithm 396, as in R's qt), refined by Halley steps on the exact upper tail
// (regularized incomplete beta). ~1e-14 relative also deep in the tails.
t_quantile :: proc(p: f64, nu: f64) -> f64 {
	if math.is_nan(p) || !(nu > 0.0) {return math.nan_f64()}
	if p <= 0.0 {return -math.INF_F64}
	if p >= 1.0 {return math.INF_F64}
	if p == 0.5 {return 0.0}
	if nu > 1e7 {return _bs_ppf(p)}

	P := 2.0 * min(p, 1.0 - p) // two-sided tail probability
	q: f64
	if math.abs(nu - 2.0) < 1e-12 {
		q = math.sqrt(2.0 / (P * (2.0 - P)) - 2.0)
	} else if nu < 1.0 + 1e-12 {
		q = 1.0 / math.tan(P * math.PI / 2.0) // Cauchy
	} else {
		a := 1.0 / (nu - 0.5)
		b := 48.0 / (a * a)
		c := ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36
		d := ((94.5 / (b + c) - 3.0) / b + 1.0) * math.sqrt(a * math.PI / 2.0) * nu
		log_y := 2.0 / nu * math.ln(d * P)
		y := math.exp(log_y)
		if y > 0.05 + a {
			// Asymptotic inverse expansion about the normal
			x := _bs_ppf(0.5 * P)
			y = x * x
			if nu < 5.0 {c += 0.3 * (nu - 4.5) * (x + 0.6)}
			c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c
			y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x
			q = math.sqrt_f64(nu * _expm1(a * y * y))
		} else if y > 1e-300 {
			y =
				((1.0 / (((nu + 6.0) / (nu * y) - 0.089 * d - 0.822) * (nu + 2.0) * 3.0) +
							0.5 / (nu + 4.0)) *
						y -
					1.0) *
					(nu + 1.0) /
					(nu + 2.0) +
				1.0 / y
			q = math.sqrt(nu * y)
		} else {
			q = math.sqrt(nu) * math.exp(-0.5 * log_y) // leading term, y underflows
		}
	}

	// Halley steps on the upper tail: P(T > q) = P/2
	lg1, _ := math.lgamma(0.5 * (nu + 1.0))
	lg2, _ := math.lgamma(0.5 * nu)
	log_c := lg1 - lg2 - 0.5 * math.ln(nu * math.PI)
	for _ in 0 ..< 10 {
		dens := math.exp(log_c - 0.5 * (nu + 1.0) * math.log1p(q * q / nu))
		if !(dens > 0.0) {break}
		x := (0.5 * ml.student_t_sf2(q, nu) - 0.5 * P) / dens
		if math.is_nan(x) || math.is_inf(x) {break}
		q += x * (1.0 + x * q * (nu + 1.0) / (2.0 * (q * q + nu)))
		if math.abs(x) <= 1e-14 * math.abs(q) {break}
	}
	return q if p > 0.5 else -q
}

// ============================================================================
// Value at Risk (VaR) and Expected Shortfall (ES)
// ============================================================================
// Sign conventions (historical, kept for compatibility):
//   return level, negative = loss: var_historical, var_parametric,
//     var_monte_carlo, value_at_risk, var_garch_single, conditional_var,
//     expected_shortfall, cvar_garch
//   positive loss magnitude: var_garch, cvar_garch_series,
//     historical_var_rolling, historical_cvar_rolling, GARCH_RiskMetrics.current_*
// With α = 1 - confidence, the empirical VaR is the inverse empirical CDF at α
// (numpy quantile(method="inverted_cdf")): the k-th smallest return, k = ⌈αn⌉.
// The empirical ES is the Acerbi-Tasche estimator, the mean of the worst αn
// returns with the boundary observation weighted by the fractional part:
// (Σ_{i≤⌊αn⌋} x_(i) + (αn - ⌊αn⌋)·x_(⌊αn⌋+1)) / (αn).

// 0-based index of the inverse-ECDF α-quantile in a sorted sample of size n
@(private = "file")
_ecdf_index :: proc(alpha: f64, n: int) -> int {
	// tolerance: (1 - 0.95)·100 = 5.000000000000004 must give k = 5
	k := int(math.ceil(alpha * f64(n) - 1e-9))
	return clamp(k - 1, 0, n - 1)
}

// Acerbi-Tasche tail mean of an ascending sample (a return level)
@(private = "file")
_tail_mean_sorted :: proc(sorted: []f64, alpha: f64) -> f64 {
	n := len(sorted)
	m := clamp(alpha * f64(n), 0.0, f64(n))
	k := min(int(math.floor(m + 1e-9)), n)
	frac := max(m - f64(k), 0.0)
	if k == 0 && frac == 0.0 {return sorted[0]}
	s := 0.0
	for i in 0 ..< k {s += sorted[i]}
	if k < n {s += frac * sorted[k]}
	return s / (f64(k) + frac)
}

// Sample mean and standard deviation (n - 1)
@(private = "file")
_mean_std :: proc(x: []f64) -> (mean: f64, std: f64) {
	n := len(x)
	for v in x {mean += v}
	mean /= f64(n)
	ss := 0.0
	for v in x {ss += (v - mean) * (v - mean)}
	return mean, math.sqrt(ss / f64(n - 1))
}

// Quantile of the unit-variance error distribution of a GARCH model type
@(private = "file")
_std_quantile :: proc(model_type: a.GARCH_Model_Type, nu, shape, prob: f64) -> f64 {
	if model_type == .StudentT && nu > 2.0 {
		tp := a.GARCH_Params {
			model_type = .StudentT,
			nu         = nu,
		}
		return a.garch_std_quantile(&tp, prob)
	} else if model_type == .GED && shape > 0.0 {
		return ged_quantile(prob, shape)
	}
	return _bs_ppf(prob)
}

// ES multiplier of the unit-variance error distribution: E[-Z | Z < -z_c] > 0
@(private = "file")
_std_es :: proc(model_type: a.GARCH_Model_Type, nu, shape, confidence: f64) -> f64 {
	alpha := 1.0 - confidence
	if model_type == .StudentT && nu > 2.0 {
		// ES of the raw t(ν), rescaled to the unit-variance errors GARCH-t uses
		scale := math.sqrt((nu - 2.0) / nu)
		z := _std_quantile(.StudentT, nu, 0, confidence) / scale
		lg1, _ := math.lgamma((nu + 1.0) / 2.0)
		lg2, _ := math.lgamma(nu / 2.0)
		pdf := math.exp(lg1 - lg2 - 0.5 * math.ln(nu * math.PI) - 0.5 * (nu + 1.0) * math.log1p(z * z / nu))
		return scale * (pdf / alpha) * (nu + z * z) / (nu - 1.0)
	} else if model_type == .GED && shape > 0.0 {
		return -ged_cvar(confidence, shape)
	}
	z := _bs_ppf(confidence)
	return 0.3989422804014327 * math.exp(-0.5 * z * z) / alpha
}

var_historical :: proc(returns: []f64, confidence: f64 = 0.95) -> f64 {
	if len(returns) == 0 {return 0.0}
	sorted := slice.clone(returns, context.temp_allocator)
	defer delete(sorted, context.temp_allocator)
	slice.sort(sorted)
	return sorted[_ecdf_index(1.0 - confidence, len(sorted))]
}

// Normal VaR: μ - z_c·s
var_parametric :: proc(returns: []f64, confidence: f64 = 0.95) -> f64 {
	if len(returns) < 2 {return 0.0}
	mean, std := _mean_std(returns)
	return mean - _bs_ppf(confidence) * std
}

// Simulated normal VaR of the `horizon`-period sum of returns
var_monte_carlo :: proc(
	returns: []f64,
	confidence: f64 = 0.95,
	n_simulations: int = 10000,
	horizon: int = 1,
) -> f64 {
	if len(returns) < 2 || n_simulations < 1 {return 0.0}
	mean, std := _mean_std(returns)

	simulated := make([]f64, n_simulations, context.temp_allocator)
	defer delete(simulated, context.temp_allocator)
	for sim in 0 ..< n_simulations {
		total := 0.0
		for _ in 0 ..< horizon {total += mean + std * rand.float64_normal(0.0, 1.0)}
		simulated[sim] = total
	}
	slice.sort(simulated)
	return simulated[_ecdf_index(1.0 - confidence, n_simulations)]
}

// VaR series from GARCH conditional variances: z_c·σ_t, a positive loss
// magnitude (mean ≈ 0), z_c from the model's unit-variance error distribution
var_garch :: proc(
	cond_var: []f64,
	confidence: f64 = 0.95,
	model_type: a.GARCH_Model_Type = .GARCH,
	nu: f64 = 0.0,
	shape: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	var_series := make([]f64, len(cond_var), allocator)
	z := _std_quantile(model_type, nu, shape, confidence)
	for v, i in cond_var {var_series[i] = z * math.sqrt(v)}
	return var_series
}

// Normal VaR for one conditional variance (return level, negative)
var_garch_single :: proc(cond_variance: f64, confidence: f64 = 0.95) -> f64 {
	return -_bs_ppf(confidence) * math.sqrt(cond_variance)
}

value_at_risk :: proc(
	returns: []f64,
	confidence: f64 = 0.95,
	method: VaRMethod = .Historical,
	n_simulations: int = 10000,
	horizon: int = 1,
) -> f64 {
	switch method {
	case .Historical:
		return var_historical(returns, confidence)
	case .Parametric:
		return var_parametric(returns, confidence)
	case .MonteCarlo:
		return var_monte_carlo(returns, confidence, n_simulations, horizon)
	case .GARCH:
		// Single point: zero-mean normal VaR with the unconditional variance
		if len(returns) == 0 {return 0.0}
		variance := 0.0
		for r in returns {variance += r * r}
		variance /= f64(len(returns))
		return var_garch_single(variance, confidence)
	}
	return 0.0
}

// ============================================================================
// Conditional Value at Risk (CVaR) - Expected Shortfall
// ============================================================================

// Empirical ES (Acerbi-Tasche), a return level (negative = loss)
conditional_var :: proc(returns: []f64, confidence: f64 = 0.95) -> f64 {
	if len(returns) == 0 {return 0.0}
	sorted := slice.clone(returns, context.temp_allocator)
	defer delete(sorted, context.temp_allocator)
	slice.sort(sorted)
	return _tail_mean_sorted(sorted, 1.0 - confidence)
}

expected_shortfall :: proc(returns: []f64, confidence: f64 = 0.95) -> f64 {
	return conditional_var(returns, confidence)
}

// Normal ES series -σ_t·φ(z_c)/(1-c) (return level, negative); see
// cvar_garch_series for Student-t / GED errors
cvar_garch :: proc(cond_var: []f64, confidence: f64 = 0.95) -> []f64 {
	cvar_series := make([]f64, len(cond_var), context.allocator)
	mult := _std_es(.GARCH, 0, 0, confidence)
	for v, i in cond_var {cvar_series[i] = -mult * math.sqrt(v)}
	return cvar_series
}

// ============================================================================
// VaR Backtesting (Kupiec POF Test)
// ============================================================================
// A breach is a return below -VaR (var_series holds positive loss magnitudes)
backtest_var :: proc(
	returns: []f64,
	var_series: []f64,
	confidence: f64 = 0.95,
) -> VaR_BacktestResult {
	n := min(len(returns), len(var_series))
	if n == 0 {return VaR_BacktestResult{}}

	n_breaches := 0
	for i in 0 ..< n {
		if returns[i] < -var_series[i] {n_breaches += 1}
	}

	kupiec_stat := _kupiec_pof_test(n, n_breaches, 1.0 - confidence)
	p_value := _chi_squared_pvalue_1df(kupiec_stat)
	return VaR_BacktestResult {
		n_obs = n,
		n_breaches = n_breaches,
		expected_breaches = f64(n) * (1.0 - confidence),
		breach_rate = f64(n_breaches) / f64(n),
		kupiec_stat = kupiec_stat,
		kupiec_pvalue = p_value,
		passes_test = p_value > 0.05,
	}
}

// Kupiec proportion-of-failures likelihood ratio for k breaches in n with
// breach probability p: LR = 2[ℓ(k/n) - ℓ(p)], ℓ(π) = k ln π + (n-k) ln(1-π),
// χ²(1) under H0. k = 0 or k = n are the most extreme outcomes (0·ln 0 = 0),
// not a perfect fit.
_kupiec_pof_test :: proc(n: int, k: int, p: f64) -> f64 {
	if n == 0 || !(p > 0.0 && p < 1.0) {return 0.0}
	kf, mf := f64(k), f64(n - k)
	p_hat := kf / f64(n)
	ll0 := kf * math.ln(p) + mf * math.log1p(-p)
	ll1 := 0.0
	if k > 0 {ll1 += kf * math.ln(p_hat)}
	if k < n {ll1 += mf * math.log1p(-p_hat)}
	return max(2.0 * (ll1 - ll0), 0.0)
}

// P(χ²(1) > x) = erfc(sqrt(x/2))
_chi_squared_pvalue_1df :: proc(x: f64) -> f64 {
	if x <= 0.0 {return 1.0}
	return math.erfc(math.sqrt(x / 2.0))
}

// ============================================================================
// Comprehensive GARCH Risk Analysis
// ============================================================================

// Fits a GARCH model to the demeaned returns. current_vol is the one-step-ahead
// forecast σ_{n+1|n}; current VaR / CVaR are positive loss magnitudes from the
// model's unit-variance error distribution (mean ≈ 0); long_run_vol is the
// per-period unconditional σ = sqrt(ω / (1 - persistence)). The backtests are
// in sample (parameters fitted on all data), skipping the first 100 points.
garch_risk_metrics :: proc(
	returns: []f64,
	model_type: a.GARCH_Model_Type = .GARCH,
	p: int = 1,
	q: int = 1,
	risk_free_rate: f64 = 0.02,
	allocator: mem.Allocator = context.allocator,
) -> GARCH_RiskMetrics {
	metrics: GARCH_RiskMetrics
	metrics.base = calculate_risk_metrics(returns, nil, risk_free_rate)
	n := len(returns)
	if n < 2 {return metrics}

	residuals := a.extract_residuals(returns, allocator)
	defer delete(residuals, allocator)

	garch_result := a.garch_fit(residuals, model_type, p, q, 2000, 1e-4, allocator)
	defer {
		delete(garch_result.params.alpha, allocator)
		delete(garch_result.params.beta, allocator)
		delete(garch_result.conditional_var, allocator)
		delete(garch_result.standardized_resid, allocator)
	}
	prm := &garch_result.params

	metrics.garch_omega = prm.omega
	if len(prm.alpha) > 0 {metrics.garch_alpha = prm.alpha[0]}
	if len(prm.beta) > 0 {metrics.garch_beta = prm.beta[0]}
	if model_type == .StudentT {metrics.garch_nu = prm.nu}
	if model_type == .GED {metrics.garch_shape = prm.shape}
	metrics.garch_persistence = garch_result.persistence
	metrics.converged = garch_result.converged
	metrics.n_iterations = garch_result.n_iterations
	if garch_result.persistence < 1.0 && prm.omega > 0.0 {
		metrics.long_run_vol = math.sqrt(prm.omega / (1.0 - garch_result.persistence))
	} else {
		metrics.long_run_vol = math.INF_F64
	}

	fc := a.garch_forecast(&garch_result, residuals, 1, allocator)
	metrics.current_vol = fc.std_forecast
	metrics.current_var_95 = _std_quantile(model_type, prm.nu, prm.shape, 0.95) * fc.std_forecast
	metrics.current_var_99 = _std_quantile(model_type, prm.nu, prm.shape, 0.99) * fc.std_forecast
	metrics.current_cvar_95 = _std_es(model_type, prm.nu, prm.shape, 0.95) * fc.std_forecast
	metrics.current_cvar_99 = _std_es(model_type, prm.nu, prm.shape, 0.99) * fc.std_forecast

	warmup := min(100, n - 1)
	var_95 := var_garch(garch_result.conditional_var, 0.95, model_type, prm.nu, prm.shape, allocator)
	var_99 := var_garch(garch_result.conditional_var, 0.99, model_type, prm.nu, prm.shape, allocator)
	defer {
		delete(var_95, allocator)
		delete(var_99, allocator)
	}
	metrics.backtest_95 = backtest_var(returns[warmup:], var_95[warmup:], 0.95)
	metrics.backtest_99 = backtest_var(returns[warmup:], var_99[warmup:], 0.99)
	return metrics
}
// ============================================================================
// Volatility Measures (Integrating Analytics Module)
// ============================================================================

historical_volatility_rolling :: proc(
	df: ^w.DataFrame,
	returns_col: string,
	window: int = 252,
	min_periods: int = 20,
	periods_per_year: f64 = 252.0,
) -> w.Column {
	rw := a.rolling_window(df, returns_col, window, min_periods)
	agg := w.make_var("variance", returns_col)
	var_col := a.rolling_apply(rw, agg)

	out := w.column_new("volatility", .Float, var_col.len)
	for i in 0 ..< var_col.len {
		v, is_null := w.column_at_float(&var_col, i)
		if is_null {
			w.append_null(&out)
		} else {
			w.append_float(&out, math.sqrt_f64(v * periods_per_year))
		}
	}

	w.destroy_column(&var_col)
	return out
}

ewma_volatility_df :: proc(
	df: ^w.DataFrame,
	returns_col: string,
	alpha: f64 = 0.06,
	min_periods: int = 20,
	bias: bool = false,
	periods_per_year: f64 = 252.0,
) -> w.Column {
	rw := a.rolling_window(df, returns_col, 0, min_periods)
	agg := a.make_ewm_var("ewm_var", returns_col, alpha, bias)
	var_col := a.rolling_apply(rw, agg)

	out := w.column_new("ewma_volatility", .Float, var_col.len)
	for i in 0 ..< var_col.len {
		v, is_null := w.column_at_float(&var_col, i)
		if is_null {
			w.append_null(&out)
		} else {
			w.append_float(&out, math.sqrt_f64(v * periods_per_year))
		}
	}

	w.destroy_column(&var_col)
	return out
}

historical_volatility :: proc(returns: []f64, periods_per_year: f64 = 252.0) -> f64 {
	if len(returns) < 2 {return 0.0}

	mean := 0.0
	for r in returns {mean += r}
	mean /= f64(len(returns))

	variance := 0.0
	for r in returns {
		diff := r - mean
		variance += diff * diff
	}
	variance /= f64(len(returns) - 1)

	return math.sqrt_f64(variance * periods_per_year)
}

ewma_volatility :: proc(returns: []f64, lambda: f64 = 0.94, periods_per_year: f64 = 252.0) -> f64 {
	if len(returns) < 2 {return 0.0}

	ewma_var := returns[0] * returns[0]
	alpha := 1.0 - lambda

	for i in 1 ..< len(returns) {
		r := returns[i]
		ewma_var = (1.0 - alpha) * ewma_var + alpha * r * r
	}

	return math.sqrt_f64(ewma_var * periods_per_year)
}

// ============================================================================
// Drawdown Analysis
// ============================================================================

drawdown_series :: proc(returns: []f64) -> []f64 {
	if len(returns) == 0 {return []f64{}}

	wealth := make([]f64, len(returns) + 1, context.temp_allocator)
	wealth[0] = 1.0

	for i in 0 ..< len(returns) {
		wealth[i + 1] = wealth[i] * (1.0 + returns[i])
	}

	running_max := make([]f64, len(wealth), context.temp_allocator)
	running_max[0] = wealth[0]

	for i in 1 ..< len(wealth) {
		if wealth[i] > running_max[i - 1] {
			running_max[i] = wealth[i]
		} else {
			running_max[i] = running_max[i - 1]
		}
	}

	drawdowns := make([]f64, len(returns), context.temp_allocator)
	for i in 0 ..< len(returns) {
		drawdowns[i] = (wealth[i + 1] - running_max[i + 1]) / running_max[i + 1]
	}

	return drawdowns
}

max_drawdown :: proc(returns: []f64) -> f64 {
	if len(returns) == 0 {return 0.0}

	drawdowns := drawdown_series(returns)
	max_dd := 0.0
	for dd in drawdowns {
		if dd < max_dd {max_dd = dd}
	}

	return max_dd
}

avg_drawdown :: proc(returns: []f64) -> f64 {
	if len(returns) == 0 {return 0.0}

	drawdowns := drawdown_series(returns)
	sum := 0.0
	count := 0
	for dd in drawdowns {
		if dd < 0.0 {
			sum += dd
			count += 1
		}
	}

	if count == 0 {return 0.0}
	return sum / f64(count)
}

// ============================================================================
// Risk-Adjusted Performance Metrics
// ============================================================================

sharpe_ratio_from_returns :: proc(
	returns: []f64,
	risk_free_rate: f64 = 0.0,
	periods_per_year: f64 = 252.0,
) -> f64 {
	if len(returns) < 2 {return 0.0}

	excess_returns := make([]f64, len(returns), context.temp_allocator)
	for i in 0 ..< len(returns) {
		excess_returns[i] = returns[i] - risk_free_rate / periods_per_year
	}

	mean_excess := 0.0
	for r in excess_returns {mean_excess += r}
	mean_excess /= f64(len(excess_returns))

	variance := 0.0
	for r in excess_returns {
		diff := r - mean_excess
		variance += diff * diff
	}
	variance /= f64(len(excess_returns) - 1)
	std_dev := math.sqrt_f64(variance)

	if std_dev == 0.0 {return 0.0}

	return (mean_excess * periods_per_year) / (std_dev * math.sqrt_f64(periods_per_year))
}

sortino_ratio_from_returns :: proc(
	returns: []f64,
	risk_free_rate: f64 = 0.0,
	periods_per_year: f64 = 252.0,
) -> f64 {
	if len(returns) < 2 {return 0.0}

	excess_returns := make([]f64, len(returns), context.temp_allocator)
	for i in 0 ..< len(returns) {
		excess_returns[i] = returns[i] - risk_free_rate / periods_per_year
	}

	mean_excess := 0.0
	for r in excess_returns {mean_excess += r}
	mean_excess /= f64(len(excess_returns))

	downside_variance := 0.0
	count := 0
	for r in excess_returns {
		if r < 0.0 {
			downside_variance += r * r
			count += 1
		}
	}

	if count == 0 {return 0.0}

	downside_variance /= f64(count)
	downside_dev := math.sqrt_f64(downside_variance)

	if downside_dev == 0.0 {return 0.0}

	return (mean_excess * periods_per_year) / (downside_dev * math.sqrt_f64(periods_per_year))
}

calmar_ratio :: proc(returns: []f64, periods_per_year: f64 = 252.0) -> f64 {
	if len(returns) == 0 {return 0.0}

	cumulative_return := 1.0
	for r in returns {cumulative_return *= (1.0 + r)}
	annualized_return := math.pow(cumulative_return, periods_per_year / f64(len(returns))) - 1.0

	max_dd := max_drawdown(returns)
	if max_dd == 0.0 {return 0.0}

	return annualized_return / math.abs(max_dd)
}

// ============================================================================
// Beta and Correlation (Using Analytics Module)
// ============================================================================

rolling_correlation_df :: proc(
	df: ^w.DataFrame,
	returns_col: string,
	benchmark_col: string,
	window: int = 252,
	min_periods: int = 20,
) -> w.Column {
	rw := a.rolling_window(df, returns_col, window, min_periods)
	agg := w.make_corr("corr", returns_col, benchmark_col)
	return a.rolling_apply(rw, agg)
}

rolling_beta_df :: proc(
	df: ^w.DataFrame,
	returns_col: string,
	benchmark_col: string,
	window: int = 252,
	min_periods: int = 20,
) -> w.Column {
	rw := a.rolling_window(df, returns_col, window, min_periods)

	cov_agg := w.make_cov("cov", returns_col, benchmark_col)
	cov_col := a.rolling_apply(rw, cov_agg)

	// beta = cov(r, b) / var(b): the variance window must run over the benchmark
	rw_b := a.rolling_window(df, benchmark_col, window, min_periods)
	var_agg := w.make_var("var", benchmark_col)
	var_col := a.rolling_apply(rw_b, var_agg)

	out := w.column_new("beta", .Float, cov_col.len)
	for i in 0 ..< cov_col.len {
		cv, n1 := w.column_at_float(&cov_col, i)
		v, n2 := w.column_at_float(&var_col, i)

		if n1 || n2 || v <= 0 {
			w.append_null(&out)
		} else {
			w.append_float(&out, cv / v)
		}
	}

	w.destroy_column(&cov_col)
	w.destroy_column(&var_col)
	return out
}

beta :: proc(returns: []f64, benchmark_returns: []f64) -> f64 {
	if len(returns) != len(benchmark_returns) || len(returns) < 2 {return 0.0}

	n := len(returns)
	mean_r := 0.0
	mean_b := 0.0
	for i in 0 ..< n {
		mean_r += returns[i]
		mean_b += benchmark_returns[i]
	}
	mean_r /= f64(n)
	mean_b /= f64(n)

	covariance := 0.0
	benchmark_var := 0.0
	for i in 0 ..< n {
		diff_r := returns[i] - mean_r
		diff_b := benchmark_returns[i] - mean_b
		covariance += diff_r * diff_b
		benchmark_var += diff_b * diff_b
	}

	if benchmark_var == 0.0 {return 0.0}
	return covariance / benchmark_var
}

correlation :: proc(returns: []f64, benchmark_returns: []f64) -> f64 {
	if len(returns) != len(benchmark_returns) || len(returns) < 2 {return 0.0}

	n := len(returns)
	mean_r := 0.0
	mean_b := 0.0
	for i in 0 ..< n {
		mean_r += returns[i]
		mean_b += benchmark_returns[i]
	}
	mean_r /= f64(n)
	mean_b /= f64(n)

	covariance := 0.0
	var_r := 0.0
	var_b := 0.0
	for i in 0 ..< n {
		diff_r := returns[i] - mean_r
		diff_b := benchmark_returns[i] - mean_b
		covariance += diff_r * diff_b
		var_r += diff_r * diff_r
		var_b += diff_b * diff_b
	}

	std_r := math.sqrt_f64(var_r / f64(n - 1))
	std_b := math.sqrt_f64(var_b / f64(n - 1))

	if std_r == 0.0 || std_b == 0.0 {return 0.0}

	return (covariance / f64(n - 1)) / (std_r * std_b)
}

tracking_error :: proc(returns: []f64, benchmark_returns: []f64) -> f64 {
	if len(returns) != len(benchmark_returns) || len(returns) < 2 {return 0.0}

	active_returns := make([]f64, len(returns), context.temp_allocator)
	for i in 0 ..< len(returns) {
		active_returns[i] = returns[i] - benchmark_returns[i]
	}

	mean_active := 0.0
	for r in active_returns {mean_active += r}
	mean_active /= f64(len(active_returns))

	variance := 0.0
	for r in active_returns {
		diff := r - mean_active
		variance += diff * diff
	}
	variance /= f64(len(active_returns) - 1)

	return math.sqrt_f64(variance)
}

// ============================================================================
// Trading Statistics
// ============================================================================

win_rate :: proc(returns: []f64) -> f64 {
	if len(returns) == 0 {return 0.0}
	wins := 0
	for r in returns {
		if r > 0.0 {wins += 1}
	}
	return f64(wins) / f64(len(returns))
}

profit_factor :: proc(returns: []f64) -> f64 {
	if len(returns) == 0 {return 0.0}

	gross_profit := 0.0
	gross_loss := 0.0

	for r in returns {
		if r > 0.0 {
			gross_profit += r
		} else {
			gross_loss += math.abs(r)
		}
	}

	if gross_loss == 0.0 {return 0.0}
	return gross_profit / gross_loss
}

avg_win :: proc(returns: []f64) -> f64 {
	if len(returns) == 0 {return 0.0}

	sum := 0.0
	count := 0
	for r in returns {
		if r > 0.0 {
			sum += r
			count += 1
		}
	}

	if count == 0 {return 0.0}
	return sum / f64(count)
}

avg_loss :: proc(returns: []f64) -> f64 {
	if len(returns) == 0 {return 0.0}

	sum := 0.0
	count := 0
	for r in returns {
		if r < 0.0 {
			sum += r
			count += 1
		}
	}

	if count == 0 {return 0.0}
	return sum / f64(count)
}

// ============================================================================
// Comprehensive Risk Analysis
// ============================================================================

calculate_risk_metrics :: proc(
	returns: []f64,
	benchmark_returns: []f64 = nil,
	risk_free_rate: f64 = 0.02,
) -> RiskMetrics {
	metrics: RiskMetrics

	metrics.var_95 = value_at_risk(returns, 0.95, .Historical)
	metrics.var_99 = value_at_risk(returns, 0.99, .Historical)
	metrics.cvar_95 = conditional_var(returns, 0.95)
	metrics.cvar_99 = conditional_var(returns, 0.99)

	metrics.volatility = historical_volatility(returns)
	metrics.volatility_ewma = ewma_volatility(returns)

	metrics.max_drawdown = max_drawdown(returns)

	metrics.sharpe_ratio = sharpe_ratio_from_returns(returns, risk_free_rate)
	metrics.sortino_ratio = sortino_ratio_from_returns(returns, risk_free_rate)
	metrics.calmar_ratio = calmar_ratio(returns)

	if benchmark_returns != nil && len(benchmark_returns) == len(returns) {
		metrics.beta = beta(returns, benchmark_returns)
		metrics.correlation = correlation(returns, benchmark_returns)
		metrics.tracking_error = tracking_error(returns, benchmark_returns)
	}

	metrics.win_rate = win_rate(returns)
	metrics.profit_factor = profit_factor(returns)
	metrics.avg_win = avg_win(returns)
	metrics.avg_loss = avg_loss(returns)

	return metrics
}

// ============================================================================
// DataFrame Integration (Using Analytics Module)
// ============================================================================

risk_metrics_from_df :: proc(
	df: ^w.DataFrame,
	column_name: string,
	benchmark_column: string = "",
	risk_free_rate: f64 = 0.02,
	allocator: mem.Allocator = context.allocator,
) -> RiskMetrics {
	returns_col := w.column(df, column_name)
	returns := make([dynamic]f64, 0, allocator)

	for i in 0 ..< df.rows {
		val, is_null := w.column_at_float(returns_col, i)
		if !is_null {
			append(&returns, val)
		}
	}

	benchmark_returns: []f64 = nil
	if benchmark_column != "" {
		benchmark_col := w.column(df, benchmark_column)
		bench_dyn := make([dynamic]f64, 0, allocator)

		for i in 0 ..< df.rows {
			val, is_null := w.column_at_float(benchmark_col, i)
			if !is_null {
				append(&bench_dyn, val)
			}
		}
		benchmark_returns = bench_dyn[:]
	}

	return calculate_risk_metrics(returns[:], benchmark_returns, risk_free_rate)
}

rolling_risk_metrics_df :: proc(
	df: ^w.DataFrame,
	returns_col: string,
	benchmark_col: string = "",
	window: int = 252,
	min_periods: int = 60,
	risk_free_rate: f64 = 0.02,
) -> w.DataFrame {
	out := w.dataframe_new()

	vol_col := historical_volatility_rolling(df, returns_col, window, min_periods)
	w.add_column(&out, vol_col)

	ewma_vol_col := ewma_volatility_df(df, returns_col, 0.06, min_periods)
	w.add_column(&out, ewma_vol_col)

	if benchmark_col != "" {
		beta_col := rolling_beta_df(df, returns_col, benchmark_col, window, min_periods)
		w.add_column(&out, beta_col)

		corr_col := rolling_correlation_df(df, returns_col, benchmark_col, window, min_periods)
		w.add_column(&out, corr_col)
	}

	out.rows = out.columns[0].len
	return out
}

// ============================================================================
// Statistical Tests Integration
// ============================================================================

returns_normality_test :: proc(returns: []f64) -> (jb_stat: f64, p_value: f64) {
	return a.jarque_bera(returns)
}

returns_autocorrelation_test :: proc(
	returns: []f64,
	max_lag: int = 10,
) -> (
	q_stat: f64,
	p_value: f64,
) {
	q, _, p := a.ljung_box(returns, max_lag, 0)
	return q, p
}

returns_stationarity_test :: proc(
	returns: []f64,
) -> (
	stat: f64,
	p_value: f64,
	is_stationary: bool,
) {
	s, p, _, _, _, _, _ := a.adf_test(
		returns,
		10,
		a.RegressionType.Constant,
		a.LagSelection.AIC,
		context.allocator,
	)
	is_stationary = p < 0.05
	return s, p, is_stationary
}
// ============================================================================
// Historical Simulation (Rolling Window)
// ============================================================================

// Rolling historical VaR as a positive loss magnitude: entry i uses the
// `window` returns before i (returns[i-window:i]); the first `window` entries are 0
historical_var_rolling :: proc(
	returns: []f64,
	confidence: f64 = 0.95,
	window: int = 252, // 1 year of trading days
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(returns)
	var_series := make([]f64, n, allocator)
	if window < 1 {return var_series}
	buf := make([]f64, window, context.temp_allocator)
	defer delete(buf, context.temp_allocator)
	idx := _ecdf_index(1.0 - confidence, window)
	for i in window ..< n {
		copy(buf, returns[i - window:i])
		slice.sort(buf)
		var_series[i] = -buf[idx]
	}
	return var_series
}

// Rolling historical ES (Acerbi-Tasche) as a positive loss magnitude, same
// windows as historical_var_rolling
historical_cvar_rolling :: proc(
	returns: []f64,
	confidence: f64 = 0.95,
	window: int = 252,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(returns)
	cvar_series := make([]f64, n, allocator)
	if window < 1 {return cvar_series}
	buf := make([]f64, window, context.temp_allocator)
	defer delete(buf, context.temp_allocator)
	for i in window ..< n {
		copy(buf, returns[i - window:i])
		slice.sort(buf)
		cvar_series[i] = -_tail_mean_sorted(buf, 1.0 - confidence)
	}
	return cvar_series
}
// ============================================================================
// GED (Generalized Error Distribution) Functions
// ============================================================================
// Unit-variance GED with shape p (p = 2: normal, p = 1: Laplace):
// f(z) = p / (2·Γ(1/p)·λ) · exp(-|z/λ|^p),  λ = sqrt(Γ(1/p) / Γ(3/p))
// (the parameterization of the GARCH-GED likelihood). With t = (|z|/λ)^p,
// P(|Z| > |z|) = Q(1/p, t) and ∫_{|z|}^∞ x f(x) dx = λ·Γ(2/p)·Q(2/p, t) / (2Γ(1/p)),
// Q the regularized upper incomplete gamma, so CDF and ES are closed form.

@(private = "file")
_ged_lambda :: proc(shape: f64) -> f64 {
	lg1, _ := math.lgamma(1.0 / shape)
	lg3, _ := math.lgamma(3.0 / shape)
	return math.exp(0.5 * (lg1 - lg3))
}

ged_pdf :: proc(z: f64, shape: f64) -> f64 {
	if shape <= 0.0 {return 0.0}
	lambda := _ged_lambda(shape)
	lg1, _ := math.lgamma(1.0 / shape)
	return math.exp(
		math.ln(shape) - math.LN2 - lg1 - math.ln(lambda) - math.pow(math.abs(z) / lambda, shape),
	)
}

ged_cdf :: proc(z: f64, shape: f64) -> f64 {
	if shape <= 0.0 {return 0.5}
	if z == 0.0 {return 0.5}
	_, Q := a._gamma_inc_regularized(1.0 / shape, math.pow(math.abs(z) / _ged_lambda(shape), shape))
	tail := 0.5 * Q // P(Z < -|z|)
	return 1.0 - tail if z > 0.0 else tail
}

ged_quantile :: proc(prob: f64, shape: f64) -> f64 {
	if shape <= 0.0 {return 0.0}
	if prob <= 0.0 {return -math.INF_F64}
	if prob >= 1.0 {return math.INF_F64}
	if prob == 0.5 {return 0.0}
	tp := a.GARCH_Params {
		model_type = .GED,
		shape      = shape,
	}
	return a.garch_std_quantile(&tp, prob)
}

// Lower-tail ES E[Z | Z < z_{1-c}] of the unit-variance GED (negative)
ged_cvar :: proc(confidence: f64, shape: f64) -> f64 {
	if shape <= 0.0 {return 0.0}
	alpha := 1.0 - confidence
	lambda := _ged_lambda(shape)
	q := -ged_quantile(alpha, shape)
	_, Q2 := a._gamma_inc_regularized(2.0 / shape, math.pow(q / lambda, shape))
	lg1, _ := math.lgamma(1.0 / shape)
	lg2, _ := math.lgamma(2.0 / shape)
	return -lambda * math.exp(lg2 - lg1) * Q2 / (2.0 * alpha)
}
// Fit GARCH-EVT Hybrid Model
garch_evt_fit :: proc(
	returns: []f64,
	threshold_percentile: f64 = 0.90, // Threshold for EVT on standardized residuals
	allocator: mem.Allocator = context.allocator,
) -> GARCH_EVT_Result {
	result: GARCH_EVT_Result

	n := len(returns)
	if n < 100 {return result}

	// 1. Fit GARCH(1,1) to capture volatility clustering
	residuals := a.extract_residuals(returns, allocator)
	defer delete(residuals, allocator)

	garch_res := a.garch_fit(residuals, .StudentT, 1, 1, 2000, 1e-4, allocator)
	defer {
		delete(garch_res.params.alpha, allocator)
		delete(garch_res.params.beta, allocator)
		delete(garch_res.conditional_var, allocator)
		delete(garch_res.standardized_resid, allocator)
	}

	result.garch_omega = garch_res.params.omega
	result.garch_alpha = garch_res.params.alpha[0]
	result.garch_beta = garch_res.params.beta[0]

	// 2. Extract standardized residuals and focus on the left tail (losses)
	// We model losses as positive numbers: L_t = -Z_t
	losses := make([]f64, n, context.temp_allocator)
	defer delete(losses, context.temp_allocator)

	for i in 0 ..< n {
		losses[i] = -garch_res.standardized_resid[i]
	}

	// 3. Find threshold for EVT (e.g., 90th percentile of standardized losses)
	sorted_losses := make([]f64, n, context.temp_allocator)
	copy(sorted_losses, losses)
	slice.sort(sorted_losses)

	threshold_idx := int(threshold_percentile * f64(n))
	if threshold_idx >= n {threshold_idx = n - 1}
	u := sorted_losses[threshold_idx]
	result.evt_threshold = u

	// 4. Extract exceedances beyond threshold
	exceedances := make([dynamic]f64, 0, allocator)
	defer delete(exceedances)

	for l in losses {
		if l > u {
			append(&exceedances, l - u)
		}
	}

	result.n_exceedances = len(exceedances)
	if result.n_exceedances < 10 {return result}

	// 5. Fit GPD to exceedances
	xi, beta, _ := gpd_fit(exceedances[:], allocator)
	result.evt_xi = xi
	result.evt_beta = beta

	// 6. Calculate GPD quantiles for standardized residuals
	// Formula: q_p = u + (beta/xi) * [ ((1-p) / (N_u/N))^(-xi) - 1 ]
	N_u := f64(result.n_exceedances)
	N := f64(n)
	F_u := N_u / N // Empirical probability of being in the tail

	calc_q :: proc(p: f64, u_val: f64, xi_val: f64, beta_val: f64, F_u_val: f64) -> f64 {
		if math.abs(xi_val) < 1e-10 {
			return u_val - beta_val * math.ln_f64((1.0 - p) / F_u_val)
		}
		return u_val + (beta_val / xi_val) * (math.pow((1.0 - p) / F_u_val, -xi_val) - 1.0)
	}

	q_95 := calc_q(0.95, u, xi, beta, F_u)
	q_99 := calc_q(0.99, u, xi, beta, F_u)

	// 7. Generate dynamic VaR series: VaR_t = sigma_t * q_p
	result.var_95_series = make([]f64, n, allocator)
	result.var_99_series = make([]f64, n, allocator)

	for i in 0 ..< n {
		sigma_t := math.sqrt_f64(garch_res.conditional_var[i])
		result.var_95_series[i] = sigma_t * q_95
		result.var_99_series[i] = sigma_t * q_99
	}

	// 8. Backtest the hybrid model
	warmup := min(100, n - 1)
	if n > warmup {
		returns_bt := returns[warmup:]
		var_95_bt := result.var_95_series[warmup:]
		var_99_bt := result.var_99_series[warmup:]

		result.backtest_95 = backtest_var(returns_bt, var_95_bt, 0.95)
		result.backtest_99 = backtest_var(returns_bt, var_99_bt, 0.99)
	}

	return result
}
// ============================================================================
// Expected Shortfall (CVaR) Backtesting
// ============================================================================

ES_BacktestResult :: struct {
	n_obs:             int,
	n_violations:      int,
	violation_rate:    f64,
	mean_excess_loss:  f64, // mean of (loss - predicted ES) over the VaR violations
	z_statistic:       f64, // exceedance-residual t statistic (McNeil & Frey 2000)
	p_value:           f64, // one-sided, H1: ES too small
	passes_test:       bool, // p_value > 0.05
	acerbi_szekely_z2: f64, // Acerbi & Szekely (2014) Z2; ≈ 0 under H0, 5% critical value ≈ -0.70
}

// Expected Shortfall backtest. Losses L_t = -r_t; var_series / es_series hold
// positive loss magnitudes and a violation is L_t > VaR_t.
//  - Exceedance residuals e_t = L_t - ES_t over the k violations have mean 0
//    under a correct ES; z = ē / (s/√k) with the sample standard deviation s,
//    one-sided normal p-value (needs k ≥ 2, else z = 0).
//  - Z2 = 1 - Σ_t L_t·1[violation] / (T·α·ES_t), α = 1 - confidence: negative
//    when ES is underestimated or violations are too frequent. Its critical
//    values depend only weakly on the distribution (≈ -0.70 at 5%, ≈ -1.8 at 0.01%).
backtest_es :: proc(
	returns: []f64,
	var_series: []f64,
	es_series: []f64,
	confidence: f64 = 0.95,
) -> ES_BacktestResult {
	n := min(len(returns), min(len(var_series), len(es_series)))
	if n == 0 {
		return ES_BacktestResult{}
	}
	alpha := 1.0 - confidence

	k := 0
	sum_e, as_sum := 0.0, 0.0
	for i in 0 ..< n {
		loss := -returns[i]
		if loss > var_series[i] {
			k += 1
			sum_e += loss - es_series[i]
			as_sum += loss / es_series[i]
		}
	}

	mean_e := sum_e / f64(k) if k > 0 else 0.0
	z := 0.0
	if k >= 2 {
		ss := 0.0
		for i in 0 ..< n {
			loss := -returns[i]
			if loss > var_series[i] {
				d := loss - es_series[i] - mean_e
				ss += d * d
			}
		}
		if sd := math.sqrt(ss / f64(k - 1)); sd > 0.0 {
			z = mean_e / (sd / math.sqrt(f64(k)))
		}
	}
	p_value := 0.5 * math.erfc(z / math.SQRT_TWO)

	return ES_BacktestResult {
		n_obs = n,
		n_violations = k,
		violation_rate = f64(k) / f64(n),
		mean_excess_loss = mean_e,
		z_statistic = z,
		p_value = p_value,
		passes_test = p_value > 0.05,
		acerbi_szekely_z2 = 1.0 - as_sum / (f64(n) * alpha),
	}
}

// ES series σ_t·ES_c(Z) from GARCH conditional variances, a positive loss
// magnitude, for normal, Student-t (unit variance) or GED errors
cvar_garch_series :: proc(
	cond_var: []f64,
	confidence: f64 = 0.95,
	model_type: a.GARCH_Model_Type = .GARCH,
	nu: f64 = 0.0,
	shape: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	cvar_series := make([]f64, len(cond_var), allocator)
	mult := _std_es(model_type, nu, shape, confidence)
	for v, i in cond_var {cvar_series[i] = mult * math.sqrt(v)}
	return cvar_series
}
