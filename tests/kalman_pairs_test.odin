package tests

import w "../wotan/core"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import plot "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"
kalman_pairs_real_data_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== Kalman Filter Pairs Trading (Real Data) ===")

	// Fetch real market data using existing yahoo infrastructure
	df := net.read_yahoo("KO", .Daily, .TwoYears, allocator)
	if df.rows == 0 {
		fmt.println("ERROR: Failed to fetch KO data")
		return
	}
	defer w.destroy_dataframe(&df)

	df2 := net.read_yahoo("PEP", .Daily, .TwoYears, allocator)
	if df2.rows == 0 {
		fmt.println("ERROR: Failed to fetch PEP data")
		return
	}
	defer w.destroy_dataframe(&df2)

	// ========================================================================
	// CRITICAL FIX: Align dataframes by DATE, not by index!
	// ========================================================================
	aligned_df := w.dataframe_new()
	defer w.destroy_dataframe(&aligned_df)

	col_date := w.column_new("Date", .Date, 0)
	col_ko := w.column_new("Asset_A", .Float, 0)
	col_pep := w.column_new("Asset_B", .Float, 0)

	date_col1 := w.column(&df, "Date")
	close_col1 := w.column(&df, "Close")

	date_col2 := w.column(&df2, "Date")
	close_col2 := w.column(&df2, "Close")

	// Two-pointer approach to align by date
	i1 := 0
	i2 := 0
	for i1 < df.rows && i2 < df2.rows {
		date1, _ := w.column_at_date(date_col1, i1)
		date2, _ := w.column_at_date(date_col2, i2)

		cmp := w.date_compare(date1, date2)
		if cmp == 0 {
			// Dates match: append to aligned dataframe
			w.append_date(&col_date, date1)

			val1, _ := w.column_at_float(close_col1, i1)
			w.append_float(&col_ko, val1)

			val2, _ := w.column_at_float(close_col2, i2)
			w.append_float(&col_pep, val2)

			i1 += 1
			i2 += 1
		} else if cmp < 0 {
			// date1 is earlier, advance i1
			i1 += 1
		} else {
			// date2 is earlier, advance i2
			i2 += 1
		}
	}

	w.add_column(&aligned_df, col_date)
	w.add_column(&aligned_df, col_ko)
	w.add_column(&aligned_df, col_pep)
	aligned_df.rows = col_date.len

	fmt.printf("Loaded %d properly aligned days of real KO/PEP data\n", aligned_df.rows)

	// Run Kalman Filter Pairs Trading Strategy
	window := 60
	initial_hedge_ratio := 1.0

	strategy_result := ml_fin.kalman_pairs_strategy(
		&aligned_df,
		"Asset_A",
		"Asset_B",
		window,
		initial_hedge_ratio,
		allocator,
	)

	// Analyze Results
	fmt.println("\n--- Strategy Performance ---")
	if len(strategy_result.returns) > 0 {
		total_return := 1.0
		for r in strategy_result.returns {
			total_return *= (1.0 + r)
		}
		total_return -= 1.0

		mean_ret := 0.0
		var_ret := 0.0
		count := len(strategy_result.returns)
		for r in strategy_result.returns {
			mean_ret += r
		}
		mean_ret /= f64(count)

		for r in strategy_result.returns {
			diff := r - mean_ret
			var_ret += diff * diff
		}
		var_ret /= f64(count - 1)
		std_dev := math.sqrt_f64(var_ret)

		sharpe := 0.0
		if std_dev > 0.0 {
			sharpe = (mean_ret / std_dev) * math.sqrt_f64(252.0)
		}

		max_dd := 0.0
		peak := 1.0
		cumulative := 1.0
		for r in strategy_result.returns {
			cumulative *= (1.0 + r)
			if cumulative > peak {
				peak = cumulative
			}
			dd := (peak - cumulative) / peak
			if dd > max_dd {
				max_dd = dd
			}
		}

		fmt.printf("Total Return:      %.2f%%\n", total_return * 100.0)
		fmt.printf("Annualized Sharpe: %.2f\n", sharpe)
		fmt.printf("Max Drawdown:      %.2f%%\n", max_dd * 100.0)
		fmt.printf("Total Trades:      %d\n", len(strategy_result.trades))
	} else {
		fmt.println("No trades generated or error in strategy execution.")
	}
	// ========================================================================
	// 4. Visualization: Z-Score and Equity Curve
	// ========================================================================
	n_steps := len(strategy_result.z_scores)
	if n_steps > 0 {
		xs := make([]f64, n_steps, allocator)
		for i in 0 ..< n_steps {xs[i] = f64(i)}

		// --- Plot 1: Z-Score with Thresholds ---
		thresh_pos := make([]f64, n_steps, allocator)
		thresh_neg := make([]f64, n_steps, allocator)
		thresh_exit_pos := make([]f64, n_steps, allocator)
		thresh_exit_neg := make([]f64, n_steps, allocator)
		for i in 0 ..< n_steps {
			thresh_pos[i] = 2.0
			thresh_neg[i] = -2.0
			thresh_exit_pos[i] = 0.5
			thresh_exit_neg[i] = -0.5
		}

		z_lines := []plot.LineData {
			plot.LineData {
				xs = xs,
				ys = strategy_result.z_scores,
				color = plot.BLUE,
				style = .Solid,
				label = "Z-Score",
			},
			plot.LineData {
				xs = xs,
				ys = thresh_pos,
				color = plot.RED,
				style = .Dashed,
				label = "Entry (+2.0)",
			},
			plot.LineData {
				xs = xs,
				ys = thresh_neg,
				color = plot.RED,
				style = .Dashed,
				label = "Entry (-2.0)",
			},
			plot.LineData {
				xs = xs,
				ys = thresh_exit_pos,
				color = plot.GREEN,
				style = .Dotted,
				label = "Exit (+0.5)",
			},
			plot.LineData {
				xs = xs,
				ys = thresh_exit_neg,
				color = plot.GREEN,
				style = .Dotted,
				label = "Exit (-0.5)",
			},
		}

		z_config := plot.DEFAULT_PLOT_CONFIG
		z_config.title = "Kalman Pairs Trading: Z-Score"
		z_config.x_label = "Time Step"
		z_config.y_label = "Z-Score"
		z_config.show_grid = true
		plot.multi_line_png(z_lines, "kalman_zscore.png", z_config, allocator)
		fmt.println("✓ Saved: kalman_zscore.png")

		// --- Plot 2: Equity Curve ---
		equity := make([]f64, n_steps, allocator)
		cum := 1.0
		for i in 0 ..< n_steps {
			cum *= (1.0 + strategy_result.returns[i])
			equity[i] = cum
		}

		eq_lines := []plot.LineData {
			plot.LineData {
				xs = xs,
				ys = equity,
				color = plot.BLUE,
				style = .Solid,
				label = "Cumulative Return",
			},
		}

		eq_config := plot.DEFAULT_PLOT_CONFIG
		eq_config.title = "Kalman Pairs Trading: Equity Curve"
		eq_config.x_label = "Time Step"
		eq_config.y_label = "Portfolio Value"
		eq_config.show_grid = true
		plot.multi_line_png(eq_lines, "kalman_equity.png", eq_config, allocator)
		fmt.println("✓ Saved: kalman_equity.png")

		// Cleanup plot arrays
		delete(xs, allocator)
		delete(thresh_pos, allocator)
		delete(thresh_neg, allocator)
		delete(thresh_exit_pos, allocator)
		delete(thresh_exit_neg, allocator)
		delete(equity, allocator)
	}
}
