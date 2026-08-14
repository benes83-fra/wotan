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

	// ========================================================================
	// CONFIGURABLE STRATEGY PARAMETERS
	// ========================================================================
	config := ml_fin.kalman_pairs_default_config()

	// Customize parameters for your strategy
	config.warmup_window = 60 // OLS estimation window
	config.process_noise = 1e-5 // Kalman filter process noise
	config.measurement_noise = 1e-4 // Kalman filter measurement noise
	config.initial_hedge_ratio = 1.0 // Initial beta estimate

	// Trading thresholds
	config.entry_threshold = 1.00 // Enter when |z-score| > 2.0
	config.exit_threshold = 0.5 // Exit when |z-score| < 0.5
	config.stop_loss_threshold = 3.0 // Stop loss at |z-score| > 3.5
	config.min_hold_days = 1 // Minimum 3 days holding period

	// Run Kalman Filter Pairs Trading Strategy with config
	strategy_result := ml_fin.kalman_pairs_strategy(
		&aligned_df,
		"Asset_A",
		"Asset_B",
		config, // Pass the configuration struct
		allocator,
	)

	// Analyze Results
	fmt.println("\n--- Strategy Performance ---")
	fmt.println("--- Configuration ---")
	fmt.printf("Warmup Window:     %d days\n", config.warmup_window)
	fmt.printf("Entry Threshold:   ±%.2f\n", config.entry_threshold)
	fmt.printf("Exit Threshold:    ±%.2f\n", config.exit_threshold)
	fmt.printf("Stop Loss:         ±%.2f\n", config.stop_loss_threshold)
	fmt.printf("Min Hold Days:     %d\n", config.min_hold_days)
	fmt.printf("Process Noise:     %.2e\n", config.process_noise)
	fmt.println()

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

		// --- Plot 1: Z-Score with Configurable Thresholds ---
		thresh_pos := make([]f64, n_steps, allocator)
		thresh_neg := make([]f64, n_steps, allocator)
		thresh_exit_pos := make([]f64, n_steps, allocator)
		thresh_exit_neg := make([]f64, n_steps, allocator)
		for i in 0 ..< n_steps {
			thresh_pos[i] = config.entry_threshold
			thresh_neg[i] = -config.entry_threshold
			thresh_exit_pos[i] = config.exit_threshold
			thresh_exit_neg[i] = -config.exit_threshold
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
				label = fmt.tprintf("Entry (+%.1f)", config.entry_threshold),
			},
			plot.LineData {
				xs = xs,
				ys = thresh_neg,
				color = plot.RED,
				style = .Dashed,
				label = fmt.tprintf("Entry (-%.1f)", config.entry_threshold),
			},
			plot.LineData {
				xs = xs,
				ys = thresh_exit_pos,
				color = plot.GREEN,
				style = .Dotted,
				label = fmt.tprintf("Exit (+%.1f)", config.exit_threshold),
			},
			plot.LineData {
				xs = xs,
				ys = thresh_exit_neg,
				color = plot.GREEN,
				style = .Dotted,
				label = fmt.tprintf("Exit (-%.1f)", config.exit_threshold),
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

kalman_pairs_real_data_grid_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== Kalman Filter Grid Pairs Trading (Real Data) ===")

	// Fetch real market data
	df := net.read_yahoo("KO", .Daily, .TwoYears, allocator)
	if df.rows == 0 {fmt.println("ERROR: Failed to fetch KO data"); return}
	defer w.destroy_dataframe(&df)

	df2 := net.read_yahoo("PEP", .Daily, .TwoYears, allocator)
	if df2.rows == 0 {fmt.println("ERROR: Failed to fetch PEP data"); return}
	defer w.destroy_dataframe(&df2)

	// Align dataframes by DATE
	aligned_df := w.dataframe_new()
	defer w.destroy_dataframe(&aligned_df)

	col_date := w.column_new("Date", .Date, 0)
	col_ko := w.column_new("Asset_A", .Float, 0)
	col_pep := w.column_new("Asset_B", .Float, 0)

	date_col1 := w.column(&df, "Date")
	close_col1 := w.column(&df, "Close")
	date_col2 := w.column(&df2, "Date")
	close_col2 := w.column(&df2, "Close")

	i1 := 0
	i2 := 0
	for i1 < df.rows && i2 < df2.rows {
		date1, _ := w.column_at_date(date_col1, i1)
		date2, _ := w.column_at_date(date_col2, i2)
		cmp := w.date_compare(date1, date2)
		if cmp == 0 {
			w.append_date(&col_date, date1)
			val1, _ := w.column_at_float(close_col1, i1)
			w.append_float(&col_ko, val1)
			val2, _ := w.column_at_float(close_col2, i2)
			w.append_float(&col_pep, val2)
			i1 += 1
			i2 += 1
		} else if cmp < 0 {
			i1 += 1
		} else {
			i2 += 1
		}
	}
	w.add_column(&aligned_df, col_date)
	w.add_column(&aligned_df, col_ko)
	w.add_column(&aligned_df, col_pep)
	aligned_df.rows = col_date.len
	fmt.printf("Loaded %d properly aligned days of real KO/PEP data\n", aligned_df.rows)

	// --- Grid Search ---
	fmt.println("\n--- Running Grid Search ---")
	entry_thresholds := []f64{1.5, 2.0, 2.5}
	exit_thresholds := []f64{0.0, 0.5}
	process_noises := []f64{1e-6, 1e-5}
	transaction_cost := 0.001 // 10 bps per trade

	best_config, best_sharpe := ml_fin.kalman_pairs_grid_search(
		&aligned_df,
		"Asset_A",
		"Asset_B",
		entry_thresholds,
		exit_thresholds,
		process_noises,
		transaction_cost,
		allocator,
	)

	fmt.printf("Best Config Found:\n")
	fmt.printf("  Entry Threshold:  %.2f\n", best_config.entry_threshold)
	fmt.printf("  Exit Threshold:   %.2f\n", best_config.exit_threshold)
	fmt.printf("  Process Noise:    %.2e\n", best_config.process_noise)
	fmt.printf("  Transaction Cost: %.4f\n", best_config.transaction_cost)
	fmt.printf("  Best Sharpe:      %.4f\n", best_sharpe)

	// --- Run Final Strategy with Best Config ---
	fmt.println("\n--- Running Final Strategy with Best Config ---")
	final_result := ml_fin.kalman_pairs_strategy(
		&aligned_df,
		"Asset_A",
		"Asset_B",
		best_config,
		allocator,
	)

	// Analyze Results
	if len(final_result.returns) > 0 {
		total_return := 1.0
		for r in final_result.returns {total_return *= (1.0 + r)}
		total_return -= 1.0

		mean_ret := 0.0
		for r in final_result.returns {mean_ret += r}
		mean_ret /= f64(len(final_result.returns))

		var_ret := 0.0
		for r in final_result.returns {var_ret += (r - mean_ret) * (r - mean_ret)}
		var_ret /= f64(len(final_result.returns) - 1)
		std_dev := math.sqrt_f64(var_ret)

		sharpe := 0.0
		if std_dev > 1e-10 {
			sharpe = (mean_ret / std_dev) * math.sqrt_f64(252.0)
		}

		max_dd := 0.0
		peak := 1.0
		cumulative := 1.0
		for r in final_result.returns {
			cumulative *= (1.0 + r)
			if cumulative > peak {peak = cumulative}
			dd := (peak - cumulative) / peak
			if dd > max_dd {max_dd = dd}
		}

		fmt.printf("Total Return:      %.2f%%\n", total_return * 100.0)
		fmt.printf("Annualized Sharpe: %.2f\n", sharpe)
		fmt.printf("Max Drawdown:      %.2f%%\n", max_dd * 100.0)
		fmt.printf("Total Trades:      %d\n", len(final_result.trades))
	}

	// Visualization
	n_steps := len(final_result.z_scores)
	if n_steps > 0 {
		xs := make([]f64, n_steps, allocator)
		for i in 0 ..< n_steps {xs[i] = f64(i)}

		// Plot 1: Z-Score
		z_lines := []plot.LineData {
			plot.LineData {
				xs = xs,
				ys = final_result.z_scores,
				color = plot.BLUE,
				style = .Solid,
				label = "Z-Score",
			},
		}
		z_config := plot.DEFAULT_PLOT_CONFIG
		z_config.title = "Kalman Pairs Trading: Z-Score"
		z_config.x_label = "Time Step"
		z_config.y_label = "Z-Score"
		z_config.show_grid = true
		plot.multi_line_png(z_lines, "kalman_grid_zscore.png", z_config, allocator)
		fmt.println("✓ Saved: kalman_grid_zscore.png")

		// Plot 2: Equity Curve
		equity := make([]f64, n_steps, allocator)
		cum := 1.0
		for i in 0 ..< n_steps {
			cum *= (1.0 + final_result.returns[i])
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
		plot.multi_line_png(eq_lines, "kalman_grid_equity.png", eq_config, allocator)
		fmt.println("✓ Saved: kalman_grid_equity.png")

		delete(xs, allocator)
		delete(equity, allocator)
	}
}
