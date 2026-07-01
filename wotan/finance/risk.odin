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

// ============================================================================
// Volatility Measures (Integrating Analytics Module)
// ============================================================================

// Historical volatility using your analytics rolling variance
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

// EWMA volatility using your analytics EWM infrastructure
ewma_volatility_df :: proc(
	df: ^w.DataFrame,
	returns_col: string,
	alpha: f64 = 0.06, // lambda = 0.94 -> alpha = 0.06
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

// Simple historical volatility (for backward compatibility)
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

// Simple EWMA volatility (for backward compatibility)
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

// Rolling correlation using analytics module
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

// Rolling beta using analytics rolling covariance and variance
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

// Simple beta (for backward compatibility)
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
	allocator: mem.Allocator = context.allocator, // ✅ ADD THIS
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

// Rolling risk metrics using analytics infrastructure
rolling_risk_metrics_df :: proc(
	df: ^w.DataFrame,
	returns_col: string,
	benchmark_col: string = "",
	window: int = 252,
	min_periods: int = 60,
	risk_free_rate: f64 = 0.02,
) -> w.DataFrame {
	out := w.dataframe_new()

	// Rolling volatility
	vol_col := historical_volatility_rolling(df, returns_col, window, min_periods)
	w.add_column(&out, vol_col)

	// Rolling EWMA volatility
	ewma_vol_col := ewma_volatility_df(df, returns_col, 0.06, min_periods)
	w.add_column(&out, ewma_vol_col)

	// Rolling beta if benchmark provided
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

// Test if returns are normally distributed using Jarque-Bera
returns_normality_test :: proc(returns: []f64) -> (jb_stat: f64, p_value: f64) {
	return a.jarque_bera(returns)
}

// Test for autocorrelation in returns using Ljung-Box
returns_autocorrelation_test :: proc(
	returns: []f64,
	max_lag: int = 10,
) -> (
	q_stat: f64,
	p_value: f64,
) {
	q, _, p := a.ljung_box(returns, max_lag, 0) // ✅ Handle 3 return values
	return q, p
}

// Test for stationarity using ADF
returns_stationarity_test :: proc(
	returns: []f64,
) -> (
	stat: f64,
	p_value: f64,
	is_stationary: bool,
) {
	// ✅ Call a.adf_test directly with all required parameters
	s, p, _, _, _, _, _ := a.adf_test(
		returns,
		10, // max_lags
		a.RegressionType.Constant, // ✅ Use full enum path
		a.LagSelection.AIC, // ✅ Use full enum path
		context.allocator,
	)
	is_stationary = p < 0.05
	return s, p, is_stationary
}

// ✅ Remove these wrapper functions - they're causing conflicts
// The functions above now call analytics directly

// ============================================================================
// Statistical Tests Integration
// ============================================================================


// Test for stationarity using ADF
