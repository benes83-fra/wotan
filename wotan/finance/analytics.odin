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
	weights:    []f64, // The caller's weights (not copied)
	mctr:       []f64, // Marginal contribution to risk ∂σ/∂w_i = (Σw)_i / σ
	ctr:        []f64, // Component contribution w_i · MCTR_i (sums to total_risk)
	total_risk: f64, // Portfolio volatility sqrt(wᵀΣw)
}

// Euler decomposition of portfolio volatility; MCTR and CTR are 0 when the
// portfolio variance is 0
risk_decomposition :: proc(
	weights: []f64,
	cov: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> RiskDecomp {
	n := len(weights)
	assert(cov.rows == n && cov.cols == n, "risk_decomposition: covariance must be n x n")

	sigma_w := l.matvec_dyn_simd(cov, weights, allocator)
	defer delete(sigma_w, allocator)
	port_var := l.dot_simd(weights, sigma_w)
	port_vol := math.sqrt(math.max(port_var, 0.0))

	mctr := make([]f64, n, allocator)
	ctr := make([]f64, n, allocator)
	if port_vol > 0.0 {
		for i in 0 ..< n {
			mctr[i] = sigma_w[i] / port_vol
			ctr[i] = weights[i] * mctr[i]
		}
	}
	return RiskDecomp{weights = weights, mctr = mctr, ctr = ctr, total_risk = port_vol}
}

// ============================================================================
// 2. Rolling Analytics
// ============================================================================

// Annualised rolling sample volatility of a returns column (pandas
// rolling(window, min_periods).std() · sqrt(periods_per_year)); same as
// historical_volatility_rolling
rolling_volatility :: proc(
	df: ^w.DataFrame,
	col: string,
	window: int,
	min_periods: int,
	periods_per_year: f64 = 252.0,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	context.allocator = allocator
	return historical_volatility_rolling(df, col, window, min_periods, periods_per_year)
}

// Rolling annualised Sharpe ratio of a returns column: mean / sample std of the
// excess returns r - rf_annual/periods over the non-null values of each window.
// The first window - 1 rows are null, as are windows with fewer than 2 values or
// a constant excess return
rolling_sharpe :: proc(
	df: ^w.DataFrame,
	col: string,
	window: int,
	rf_annual: f64 = 0.0,
	periods: f64 = 252.0,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	src := w.column(df, col)
	out := w.column_new(fmt.tprintf("rolling_sharpe_%d", window), .Float, src.len, allocator)
	rf := rf_annual / periods
	ann := math.sqrt(periods)

	for i in 0 ..< src.len {
		if i < window - 1 {
			w.append_null(&out)
			continue
		}
		lo := max(i - window + 1, 0)
		sum, count := 0.0, 0
		for j in lo ..= i {
			if v, is_null := w.column_at_float(src, j); !is_null {
				sum += v - rf
				count += 1
			}
		}
		if count < 2 {
			w.append_null(&out)
			continue
		}
		mean := sum / f64(count)
		ss, scale := 0.0, 0.0
		for j in lo ..= i {
			if v, is_null := w.column_at_float(src, j); !is_null {
				d := v - rf - mean
				ss += d * d
				scale = math.max(scale, math.abs(v - rf))
			}
		}
		std := math.sqrt(ss / f64(count - 1))
		// a constant series leaves only rounding noise in std
		if std <= 1e-13 * scale {
			w.append_null(&out)
		} else {
			w.append_float(&out, mean / std * ann)
		}
	}
	return out
}

// Annualised EWM volatility (pandas ewm(alpha, adjust=False).var(bias)); same as
// ewma_volatility_df
ewma_rolling_volatility :: proc(
	df: ^w.DataFrame,
	col: string,
	alpha: f64,
	min_periods: int,
	bias: bool = false,
	periods_per_year: f64 = 252.0,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	context.allocator = allocator
	return ewma_volatility_df(df, col, alpha, min_periods, bias, periods_per_year)
}

// Rolling OLS beta cov(r, b) / var(b) over the rows of each window where both
// columns are non-null, so both moments use the same observations. With
// full_window the first window - 1 rows are null; results need ≥ max(min_obs, 2)
// pairs and a non-constant benchmark
@(private)
_rolling_pair_beta :: proc(
	df: ^w.DataFrame,
	port_col, bench_col: string,
	window, min_obs: int,
	full_window: bool,
	name: string,
	allocator: mem.Allocator,
) -> w.Column {
	src_p := w.column(df, port_col)
	src_b := w.column(df, bench_col)
	out := w.column_new(name, .Float, df.rows, allocator)

	for i in 0 ..< df.rows {
		if full_window && i < window - 1 {
			w.append_null(&out)
			continue
		}
		lo := max(i - window + 1, 0)
		sp, sb, count := 0.0, 0.0, 0
		for j in lo ..= i {
			pv, np := w.column_at_float(src_p, j)
			bv, nb := w.column_at_float(src_b, j)
			if !np && !nb {
				sp += pv
				sb += bv
				count += 1
			}
		}
		if count < max(min_obs, 2) {
			w.append_null(&out)
			continue
		}
		mp, mb := sp / f64(count), sb / f64(count)
		sxy, syy, scale := 0.0, 0.0, 0.0
		for j in lo ..= i {
			pv, np := w.column_at_float(src_p, j)
			bv, nb := w.column_at_float(src_b, j)
			if !np && !nb {
				sxy += (pv - mp) * (bv - mb)
				syy += (bv - mb) * (bv - mb)
				scale = math.max(scale, math.abs(bv))
			}
		}
		// a constant benchmark leaves only rounding noise in syy
		if syy <= 1e-26 * scale * scale * f64(count) {
			w.append_null(&out)
		} else {
			w.append_float(&out, sxy / syy)
		}
	}
	return out
}

// Rolling beta of a portfolio against a benchmark over the last `window` rows
// (pairwise-complete rows; the first window - 1 rows are null)
rolling_beta :: proc(
	df: ^w.DataFrame,
	port_col: string,
	bench_col: string,
	window: int,
	allocator: mem.Allocator = context.allocator,
) -> w.Column {
	return _rolling_pair_beta(
		df,
		port_col,
		bench_col,
		window,
		2,
		true,
		fmt.tprintf("rolling_beta_%d", window),
		allocator,
	)
}

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

// ============================================================================
// 3. Visualizations (Leveraging wotan/plot)
// ============================================================================

// Price columns rebased to 100 at their first positive value; a null repeats the
// previous point. Columns without a positive price are skipped
plot_cumulative_returns :: proc(
	df: ^w.DataFrame,
	cols: []string,
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(cols) == 0 {return false}

	lines := make([dynamic]p.LineData, 0, len(cols), allocator)
	defer {
		for ln in lines {
			delete(ln.xs, allocator)
			delete(ln.ys, allocator)
		}
		delete(lines)
	}
	colors := []p.Color{p.RED, p.BLUE, p.Color{0, 150, 0, 255}, p.Color{200, 100, 0, 255}}

	for col_name in cols {
		src := w.column(df, col_name)
		base_price := 0.0
		for i in 0 ..< src.len {
			v, is_null := w.column_at_float(src, i)
			if !is_null && v > 0.0 {
				base_price = v
				break
			}
		}
		if base_price == 0.0 {continue}

		xs := make([]f64, src.len, allocator)
		ys := make([]f64, src.len, allocator)
		last := 100.0
		for i in 0 ..< src.len {
			xs[i] = f64(i)
			if v, is_null := w.column_at_float(src, i); !is_null {last = v / base_price * 100.0}
			ys[i] = last
		}
		append(
			&lines,
			p.LineData {
				xs = xs,
				ys = ys,
				color = colors[len(lines) % len(colors)],
				style = .Solid,
				label = col_name,
			},
		)
	}
	if len(lines) == 0 {return false}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Cumulative Returns (Base = 100)"
	config.y_label = "Index Value"
	config.x_label = "Time"
	config.show_grid = true

	return p.multi_line_png(lines[:], path, config, allocator)
}

// Drawdown (%) of a price column from its running peak; a null or non-positive
// price repeats the previous point
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

	peak, last := 0.0, 0.0
	for i in 0 ..< src.len {
		xs[i] = f64(i)
		if v, is_null := w.column_at_float(src, i); !is_null && v > 0.0 {
			peak = math.max(peak, v)
			last = (v - peak) / peak * 100.0
		}
		ys[i] = last
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

	// CTR in volatility percentage points (the bars sum to the portfolio vol)
	values := make([]f64, len(decomp.ctr), allocator)
	defer delete(values, allocator)

	for i in 0 ..< len(decomp.ctr) {
		values[i] = decomp.ctr[i] * 100.0
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Risk Decomposition (Component Contribution to Risk)"
	config.y_label = "Contribution to volatility (% pts)"
	config.bar_color = p.BLUE

	return p.bar_png(asset_names, values, path, config, allocator)
}
