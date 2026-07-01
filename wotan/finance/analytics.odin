// wotan/finance/analytics.odin
package finance

import ana "../analytics"
import w "../core"
import l "../linalg"
import p "../plot"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// 1. Risk Decomposition (Marginal & Component Contribution to Risk)
// ============================================================================

// RiskDecomp holds the breakdown of portfolio risk into individual asset contributions.
RiskDecomp :: struct {
	weights:    []f64, // Original weights
	mctr:       []f64, // Marginal Contribution to Risk
	ctr:        []f64, // Component Contribution to Risk (sums to total risk)
	total_risk: f64, // Total portfolio volatility
}

// risk_decomposition calculates how much each asset contributes to the total portfolio risk.
// Uses SIMD-optimized matrix-vector multiplication.
risk_decomposition :: proc(
	weights: []f64,
	cov: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> RiskDecomp {
	n := len(weights)

	// 1. Calculate Sigma * w (SIMD)
	sigma_w := l.matvec_dyn_simd(cov, weights, allocator)
	defer delete(sigma_w, allocator)

	// 2. Calculate portfolio variance: w^T * (Sigma * w)
	port_var := l.dot_simd(weights, sigma_w)
	port_vol := math.sqrt(port_var)

	// 3. Calculate Marginal Contribution to Risk (MCTR) = (Sigma * w) / port_vol
	mctr := make([]f64, n, allocator)
	if port_vol > 1e-10 {
		inv_vol := 1.0 / port_vol
		for i in 0 ..< n {
			mctr[i] = sigma_w[i] * inv_vol
		}
	}

	// 4. Calculate Component Contribution to Risk (CTR) = w_i * MCTR_i
	ctr := make([]f64, n, allocator)
	for i in 0 ..< n {
		ctr[i] = weights[i] * mctr[i]
	}

	return RiskDecomp{weights = weights, mctr = mctr, ctr = ctr, total_risk = port_vol}
}

// ============================================================================
// 2. Rolling Analytics
// ============================================================================

// rolling_volatility calculates the annualized standard deviation over a rolling window.
// Assumes the input column contains simple returns (not prices).
rolling_volatility :: proc(
	df: ^w.DataFrame,
	col: string,
	window: int,
	min_periods: int,
	periods_per_year: f64 = 252.0,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	rw := ana.rolling_window(df, col, window, min_periods)
	agg := w.Aggregator {
		name   = "variance",
		column = col,
		kind   = .Var,
	}

	var_col := ana.rolling_apply_float_var(rw, agg, allocator)
	defer w.destroy_column(&var_col)

	// Convert variance to volatility (sqrt) and annualize
	out := w.column_new("volatility", .Float, 0)
	for i in 0 ..< var_col.len {
		v, is_null := w.column_at_float(&var_col, i)
		if is_null {
			w.append_null(&out)
		} else {
			// sqrt(variance * periods_per_year)
			vol := math.sqrt(v * periods_per_year)
			w.append_float(&out, vol)
		}
	}

	return out
}

// rolling_sharpe calculates the rolling Sharpe ratio.
rolling_sharpe :: proc(
	df: ^w.DataFrame,
	col: string,
	window: int,
	rf_annual: f64 = 0.0,
	periods: f64 = 252.0,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	src := w.column(df, col)
	out := w.column_new(fmt.tprintf("rolling_sharpe_%d", window), .Float, src.len)
	rf_daily := rf_annual / periods

	for i in 0 ..< src.len {
		if i < window - 1 {
			w.append_null(&out)
			continue
		}

		sum_ret := 0.0
		sum_excess_sq := 0.0
		count := 0

		for j in i - window + 1 ..= i {
			v, is_null := w.column_at_float(src, j)
			if !is_null {
				excess := v - rf_daily
				sum_ret += excess
				sum_excess_sq += excess * excess
				count += 1
			}
		}

		if count < 2 {
			w.append_null(&out)
			continue
		}

		mean_excess := sum_ret / f64(count)
		variance :=
			(sum_excess_sq / f64(count - 1)) -
			(mean_excess * mean_excess * f64(count) / f64(count - 1))

		if variance < 0.0 {variance = 0.0} 	// Numerical stability
		std_dev := math.sqrt(variance)

		if std_dev > 1e-10 {
			sharpe := (mean_excess / std_dev) * math.sqrt(periods)
			w.append_float(&out, sharpe)
		} else {
			w.append_null(&out)
		}
	}

	return out
}
ewma_rolling_volatility :: proc(
	df: ^w.DataFrame,
	col: string,
	alpha: f64,
	min_periods: int,
	bias: bool = false,
	periods_per_year: f64 = 252.0,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	rw := ana.rolling_window(df, col, 0, min_periods) // EWM ignores window
	agg := ana.make_ewm_var("ewm_var", col, alpha, bias)

	var_col := ana.rolling_apply_float_ewm_var(rw, agg, allocator)
	defer w.destroy_column(&var_col)

	// Convert variance to volatility and annualize
	out := w.column_new("ewma_volatility", .Float, 0)
	for i in 0 ..< var_col.len {
		v, is_null := w.column_at_float(&var_col, i)
		if is_null {
			w.append_null(&out)
		} else {
			vol := math.sqrt(v * periods_per_year)
			w.append_float(&out, vol)
		}
	}

	return out
}
// rolling_beta calculates the rolling beta of a portfolio against a benchmark.
rolling_beta :: proc(
	df: ^w.DataFrame,
	port_col: string,
	bench_col: string,
	window: int,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	src_port := w.column(df, port_col)
	src_bench := w.column(df, bench_col)
	out := w.column_new(fmt.tprintf("rolling_beta_%d", window), .Float, src_port.len)

	for i in 0 ..< src_port.len {
		if i < window - 1 {
			w.append_null(&out)
			continue
		}

		sum_port := 0.0
		sum_bench := 0.0
		sum_prod := 0.0
		sum_bench_sq := 0.0
		count := 0

		for j in i - window + 1 ..= i {
			p, np := w.column_at_float(src_port, j)
			b, nb := w.column_at_float(src_bench, j)

			if !np && !nb {
				sum_port += p
				sum_bench += b
				sum_prod += p * b
				sum_bench_sq += b * b
				count += 1
			}
		}

		if count < 2 {
			w.append_null(&out)
			continue
		}

		mean_port := sum_port / f64(count)
		mean_bench := sum_bench / f64(count)

		// Covariance
		cov := (sum_prod / f64(count)) - (mean_port * mean_bench)
		// Variance of benchmark
		var_bench := (sum_bench_sq / f64(count)) - (mean_bench * mean_bench)

		if var_bench > 1e-10 {
			beta := cov / var_bench
			w.append_float(&out, beta)
		} else {
			w.append_null(&out)
		}
	}

	return out
}

// ============================================================================
// 3. Visualizations (Leveraging wotan/plot)
// ============================================================================
rolling_corr_matrix :: proc(
	df: ^w.DataFrame,
	cols: []string,
	window: int,
	min_periods: int,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	return ana.rolling_corr_matrix(df, cols, window, min_periods, allocator)
}

rolling_cov_matrix :: proc(
	df: ^w.DataFrame,
	cols: []string,
	window: int,
	min_periods: int,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	return ana.rolling_cov_matrix(df, cols, window, min_periods, allocator)
}

// EWMA covariance matrix (already exists!)
ewm_cov_matrix :: proc(
	df: ^w.DataFrame,
	cols: []string,
	alpha: f64,
	min_periods: int,
	bias: bool,
	adjust: bool,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	return ana.ewm_cov_matrix(df, cols, alpha, min_periods, bias, adjust, allocator)
}

// plot_cumulative_returns takes price columns, normalizes them to start at 100, and plots them.
plot_cumulative_returns :: proc(
	df: ^w.DataFrame,
	cols: []string,
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(cols) == 0 {return false}

	lines := make([]p.LineData, len(cols), allocator)
	defer delete(lines, allocator)

	// Generate distinct colors for the lines
	colors := []p.Color{p.RED, p.BLUE, p.Color{0, 150, 0, 255}, p.Color{200, 100, 0, 255}}

	for col_name, c_idx in cols {
		src := w.column(df, col_name)

		xs := make([]f64, src.len, allocator)
		ys := make([]f64, src.len, allocator)

		// Find first valid price to normalize
		base_price := 0.0
		for i in 0 ..< src.len {
			v, is_null := w.column_at_float(src, i)
			if !is_null && v > 0.0 {
				base_price = v
				break
			}
		}

		if base_price == 0.0 {
			delete(xs, allocator)
			delete(ys, allocator)
			continue
		}

		for i in 0 ..< src.len {
			v, is_null := w.column_at_float(src, i)
			xs[i] = f64(i)
			if is_null {
				ys[i] = 0.0 // Or handle nulls differently
			} else {
				ys[i] = (v / base_price) * 100.0 // Normalize to 100
			}
		}

		color_idx := c_idx % len(colors)
		lines[c_idx] = p.LineData {
			xs    = xs,
			ys    = ys,
			color = colors[color_idx],
			style = .Solid,
			label = col_name,
		}
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Cumulative Returns (Base = 100)"
	config.y_label = "Index Value"
	config.x_label = "Time"
	config.show_grid = true

	return p.multi_line_png(lines, path, config, allocator)
}

// plot_drawdown calculates and plots the drawdown series for a given price column.
plot_drawdown :: proc(
	df: ^w.DataFrame,
	col: string,
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	src := w.column(df, col)

	xs := make([]f64, src.len, allocator)
	ys := make([]f64, src.len, allocator)
	defer delete(xs, allocator)
	defer delete(ys, allocator)

	peak := -math.F64_MAX
	for i in 0 ..< src.len {
		v, is_null := w.column_at_float(src, i)
		xs[i] = f64(i)

		if is_null {
			ys[i] = 0.0
			continue
		}

		if v > peak {peak = v}

		// Drawdown is negative percentage from peak
		ys[i] = ((v - peak) / peak) * 100.0
	}

	lines := []p.LineData {
		p.LineData{xs = xs, ys = ys, color = p.RED, style = .Solid, label = "Drawdown %"},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = fmt.tprintf("Drawdown: %s", col)
	config.y_label = "Drawdown %"
	config.x_label = "Time"
	config.show_grid = true

	return p.multi_line_png(lines, path, config, allocator)
}

// plot_risk_decomposition creates a bar chart of Component Contribution to Risk (CTR).
plot_risk_decomposition :: proc(
	decomp: RiskDecomp,
	asset_names: []string,
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(decomp.ctr) != len(asset_names) {return false}

	// Convert CTR to percentages for better readability
	values := make([]f64, len(decomp.ctr), allocator)
	defer delete(values, allocator)

	for i in 0 ..< len(decomp.ctr) {
		values[i] = decomp.ctr[i] * 100.0 // Percentage contribution
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Risk Decomposition (Component Contribution to Risk)"
	config.y_label = "Risk Contribution (%)"
	config.bar_color = p.BLUE

	return p.bar_png(asset_names, values, path, config, allocator)
}
