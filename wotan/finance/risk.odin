// wotan/finance/risk.odin
package finance

import a "../analytics"
import w "../core"
import l "../linalg"
import "core:math"
import "core:math/rand"
import "core:mem"
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

// ============================================================================
// Value at Risk (VaR)
// ============================================================================

var_historical :: proc(returns: []f64, confidence: f64 = 0.95) -> f64 {
	if len(returns) == 0 {return 0.0}

	sorted := make([]f64, len(returns), context.temp_allocator)
	copy(sorted, returns)
	slice.sort(sorted)

	index := int((1.0 - confidence) * f64(len(sorted)))
	if index < 0 {index = 0}
	if index >= len(sorted) {index = len(sorted) - 1}

	return sorted[index]
}

var_parametric :: proc(returns: []f64, confidence: f64 = 0.95) -> f64 {
	if len(returns) == 0 {return 0.0}

	mean := 0.0
	for r in returns {mean += r}
	mean /= f64(len(returns))

	variance := 0.0
	for r in returns {
		diff := r - mean
		variance += diff * diff
	}
	variance /= f64(len(returns) - 1)
	std_dev := math.sqrt_f64(variance)

	z_score := norm_inv(confidence)
	return mean - z_score * std_dev
}

var_monte_carlo :: proc(
	returns: []f64,
	confidence: f64 = 0.95,
	n_simulations: int = 10000,
	horizon: int = 1,
) -> f64 {
	if len(returns) == 0 {return 0.0}

	mean := 0.0
	for r in returns {mean += r}
	mean /= f64(len(returns))

	variance := 0.0
	for r in returns {
		diff := r - mean
		variance += diff * diff
	}
	variance /= f64(len(returns) - 1)
	std_dev := math.sqrt_f64(variance)

	simulated_returns := make([]f64, n_simulations, context.temp_allocator)

	for sim in 0 ..< n_simulations {
		total_return := 0.0
		for day in 0 ..< horizon {
			daily_return := mean + std_dev * rand.float64_normal(0.0, 1.0)
			total_return += daily_return
		}
		simulated_returns[sim] = total_return
	}

	slice.sort(simulated_returns)

	index := int((1.0 - confidence) * f64(n_simulations))
	if index < 0 {index = 0}
	if index >= n_simulations {index = n_simulations - 1}

	return simulated_returns[index]
}

// ✅ NEW: GARCH-based VaR using conditional variance series
// Returns negative value (loss is positive when negated)
var_garch :: proc(cond_var: []f64, confidence: f64 = 0.95) -> []f64 {
	n := len(cond_var)
	var_series := make([]f64, n, context.allocator)

	z_score := norm_inv(confidence)

	for i in 0 ..< n {
		std := math.sqrt_f64(cond_var[i])
		var_series[i] = -z_score * std // Negative = potential loss
	}

	return var_series
}

// ✅ NEW: Single-point GARCH VaR
var_garch_single :: proc(cond_variance: f64, confidence: f64 = 0.95) -> f64 {
	z_score := norm_inv(confidence)
	return -z_score * math.sqrt_f64(cond_variance)
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
		// For single-point GARCH, use unconditional variance
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

conditional_var :: proc(returns: []f64, confidence: f64 = 0.95) -> f64 {
	if len(returns) == 0 {return 0.0}

	sorted := make([]f64, len(returns), context.temp_allocator)
	copy(sorted, returns)
	slice.sort(sorted)

	cutoff_index := int((1.0 - confidence) * f64(len(sorted)))
	if cutoff_index < 1 {cutoff_index = 1}

	sum := 0.0
	for i in 0 ..< cutoff_index {sum += sorted[i]}

	return sum / f64(cutoff_index)
}

expected_shortfall :: proc(returns: []f64, confidence: f64 = 0.95) -> f64 {
	return conditional_var(returns, confidence)
}

// ✅ NEW: GARCH-based CVaR (Expected Shortfall)
// Under normal distribution: CVaR_α = -μ + σ * φ(z_α) / (1-α)
cvar_garch :: proc(cond_var: []f64, confidence: f64 = 0.95) -> []f64 {
	n := len(cond_var)
	cvar_series := make([]f64, n, context.allocator)

	z_score := norm_inv(confidence)
	// Standard normal PDF at z_score: φ(z) = (1/√(2π)) * exp(-z²/2)
	inv_sqrt_2pi := 0.3989422804014327
	phi_z := inv_sqrt_2pi * math.exp_f64(-0.5 * z_score * z_score)
	scale := phi_z / (1.0 - confidence)

	for i in 0 ..< n {
		std := math.sqrt_f64(cond_var[i])
		cvar_series[i] = -std * scale // Negative = expected loss
	}

	return cvar_series
}

// ============================================================================
// ✅ NEW: VaR Backtesting (Kupiec POF Test)
// ============================================================================
// Backtest VaR: count breaches and compute Kupiec test statistic
// A breach occurs when actual loss exceeds VaR
// var_series contains POSITIVE values (magnitude of loss)
// So we check if return < -var_series (return is more negative than the loss threshold)
backtest_var :: proc(
	returns: []f64,
	var_series: []f64,
	confidence: f64 = 0.95,
) -> VaR_BacktestResult {
	n := min(len(returns), len(var_series))
	if n == 0 {
		return VaR_BacktestResult{}
	}

	n_breaches := 0
	for i in 0 ..< n {
		// Breach: actual return is worse (more negative) than the VaR threshold
		// var_series[i] is positive (e.g., +1.6%), so -var_series[i] is the threshold (e.g., -1.6%)
		if returns[i] < -var_series[i] {
			n_breaches += 1
		}
	}

	expected := f64(n) * (1.0 - confidence)
	breach_rate := f64(n_breaches) / f64(n)
	kupiec_stat := _kupiec_pof_test(n, n_breaches, 1.0 - confidence)
	p_value := _chi_squared_pvalue_1df(kupiec_stat)

	return VaR_BacktestResult {
		n_obs = n,
		n_breaches = n_breaches,
		expected_breaches = expected,
		breach_rate = breach_rate,
		kupiec_stat = kupiec_stat,
		kupiec_pvalue = p_value,
		passes_test = p_value > 0.05,
	}
}

// Kupiec Proportion of Failures (POF) test statistic
// Tests H0: actual breach rate = expected breach rate
// Statistic follows chi-squared(1) under H0
_kupiec_pof_test :: proc(n: int, k: int, p: f64) -> f64 {
	if k == 0 || k == n || n == 0 {
		return 0.0
	}

	p_hat := f64(k) / f64(n)

	// LR = 2 * [k * ln(p_hat/p) + (n-k) * ln((1-p_hat)/(1-p))]
	term1 := f64(k) * math.ln_f64(p_hat / p)
	term2 := f64(n - k) * math.ln_f64((1.0 - p_hat) / (1.0 - p))

	return 2.0 * (term1 + term2)
}

// Chi-squared p-value for df=1
// Uses relationship: P(χ²(1) > x) = erfc(sqrt(x/2))
_chi_squared_pvalue_1df :: proc(x: f64) -> f64 {
	if x <= 0.0 {
		return 1.0
	}
	return math.erfc_f64(math.sqrt_f64(x / 2.0))
}

// ============================================================================
// ✅ NEW: Comprehensive GARCH Risk Analysis
// ============================================================================

// Fit GARCH and compute comprehensive risk metrics
garch_risk_metrics :: proc(
	returns: []f64,
	p: int = 1,
	q: int = 1,
	risk_free_rate: f64 = 0.02,
	allocator: mem.Allocator = context.allocator,
) -> GARCH_RiskMetrics {
	metrics: GARCH_RiskMetrics

	// Compute base metrics
	metrics.base = calculate_risk_metrics(returns, nil, risk_free_rate)

	// Fit GARCH(1,1)
	residuals := a.extract_residuals(returns, allocator)
	defer delete(residuals, allocator)

	garch_result := a.garch_fit(residuals, .GARCH, p, q, 2000, 1e-4, allocator)
	defer {
		delete(garch_result.params.alpha, allocator)
		delete(garch_result.params.beta, allocator)
		delete(garch_result.conditional_var, allocator)
		delete(garch_result.standardized_resid, allocator)
	}

	// Store GARCH parameters
	if p > 0 {
		metrics.garch_omega = garch_result.params.omega
		metrics.garch_alpha = garch_result.params.alpha[0]
	}
	if q > 0 {
		metrics.garch_beta = garch_result.params.beta[0]
	}
	metrics.garch_persistence = garch_result.persistence
	metrics.converged = garch_result.converged
	metrics.n_iterations = garch_result.n_iterations

	// Long-run volatility (annualized)
	if metrics.garch_persistence < 1.0 {
		long_run_var := metrics.garch_omega / (1.0 - metrics.garch_persistence)
		metrics.long_run_vol = math.sqrt_f64(long_run_var * 252.0)
	}

	// Current GARCH-based metrics (last observation)
	n := len(returns)
	if n > 0 {
		last_var := garch_result.conditional_var[n - 1]
		metrics.current_vol = math.sqrt_f64(last_var * 252.0) // Annualized
		metrics.current_var_95 = var_garch_single(last_var, 0.95)
		metrics.current_var_99 = var_garch_single(last_var, 0.99)

		// CVaR under normal
		z_95 := norm_inv(0.95)
		z_99 := norm_inv(0.99)
		inv_sqrt_2pi := 0.3989422804014327
		std := math.sqrt_f64(last_var)

		phi_95 := inv_sqrt_2pi * math.exp_f64(-0.5 * z_95 * z_95)
		phi_99 := inv_sqrt_2pi * math.exp_f64(-0.5 * z_99 * z_99)

		metrics.current_cvar_95 = -std * phi_95 / 0.05
		metrics.current_cvar_99 = -std * phi_99 / 0.01
	}

	// Backtest (skip first 100 observations for warmup)
	warmup := min(100, n - 1)
	if n > warmup {
		returns_bt := returns[warmup:]
		var_95_series := var_garch(garch_result.conditional_var, 0.95)
		var_99_series := var_garch(garch_result.conditional_var, 0.99)
		defer {
			delete(var_95_series, allocator)
			delete(var_99_series, allocator)
		}

		var_95_bt := var_95_series[warmup:]
		var_99_bt := var_99_series[warmup:]

		metrics.backtest_95 = backtest_var(returns_bt, var_95_bt, 0.95)
		metrics.backtest_99 = backtest_var(returns_bt, var_99_bt, 0.99)
	}

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

	var_agg := w.make_var("var", benchmark_col)
	var_col := a.rolling_apply(rw, var_agg)

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
