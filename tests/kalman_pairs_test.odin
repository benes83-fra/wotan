package tests

import analytic "../wotan/analytics"
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

	// 1. Higher entry thresholds to ensure the spread deviation is large enough to overcome 20bps round-trip costs
	entry_thresholds := []f64{2.0, 2.5, 3.0}

	// 2. Tighter exit thresholds to capture the mean reversion quickly before it reverses
	exit_thresholds := []f64{0.0, 0.25}

	// 3. Much lower process noise. KO/PEP are stable; the hedge ratio shouldn't jump around.
	// 1e-3 is way too high and causes the filter to "chase" the price.
	process_noises := []f64{1e-6, 1e-7}

	min_hold_days_list := []int{3, 5}
	cooldown_days_list := []int{3, 5}

	// Note: 0.001 = 10 bps per leg. A round trip is ~20 bps.
	// This is a realistic friction, but it demands high z-score moves to be profitable.
	transaction_cost := 0.001

	best_config, best_sharpe := ml_fin.kalman_pairs_grid_search(
		&aligned_df,
		"Asset_A",
		"Asset_B",
		entry_thresholds,
		exit_thresholds,
		process_noises,
		min_hold_days_list,
		cooldown_days_list,
		transaction_cost,
		allocator,
	)
	fmt.printf("Best Config Found:\n")
	fmt.printf("  Entry Threshold:  %.2f\n", best_config.entry_threshold)
	fmt.printf("  Exit Threshold:   %.2f\n", best_config.exit_threshold)
	fmt.printf("  Process Noise:    %.2e\n", best_config.process_noise)
	fmt.printf("  Min Hold Days:    %d\n", best_config.min_hold_days) // ✅ NEW
	fmt.printf("  Cooldown Days:    %d\n", best_config.cooldown_days)
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
kalman_pairs_semiconductor_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== Kalman Filter Pairs Trading (Semiconductors: NVDA & AMD) ===")

	// Fetch real market data
	df1 := net.read_yahoo("NVDA", .Daily, .TwoYears, allocator)
	if df1.rows == 0 {
		fmt.println("ERROR: Failed to fetch NVDA data")
		return
	}
	defer w.destroy_dataframe(&df1)

	df2 := net.read_yahoo("AMD", .Daily, .TwoYears, allocator)
	if df2.rows == 0 {
		fmt.println("ERROR: Failed to fetch AMD data")
		return
	}
	defer w.destroy_dataframe(&df2)

	// Align dataframes by DATE
	aligned_df := w.dataframe_new()
	defer w.destroy_dataframe(&aligned_df)

	col_date := w.column_new("Date", .Date, 0)
	col_nvda := w.column_new("Asset_A", .Float, 0)
	col_amd := w.column_new("Asset_B", .Float, 0)

	date_col1 := w.column(&df1, "Date")
	close_col1 := w.column(&df1, "Close")

	date_col2 := w.column(&df2, "Date")
	close_col2 := w.column(&df2, "Close")

	i1 := 0
	i2 := 0
	for i1 < df1.rows && i2 < df2.rows {
		date1, _ := w.column_at_date(date_col1, i1)
		date2, _ := w.column_at_date(date_col2, i2)

		cmp := w.date_compare(date1, date2)
		if cmp == 0 {
			w.append_date(&col_date, date1)

			val1, _ := w.column_at_float(close_col1, i1)
			w.append_float(&col_nvda, val1)

			val2, _ := w.column_at_float(close_col2, i2)
			w.append_float(&col_amd, val2)

			i1 += 1
			i2 += 1
		} else if cmp < 0 {
			i1 += 1
		} else {
			i2 += 1
		}
	}

	w.add_column(&aligned_df, col_date)
	w.add_column(&aligned_df, col_nvda)
	w.add_column(&aligned_df, col_amd)
	aligned_df.rows = col_date.len

	fmt.printf("Loaded %d properly aligned days of real NVDA/AMD data\n", aligned_df.rows)

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
	config.entry_threshold = 1.50 // Enter when |z-score| > 1.5
	config.exit_threshold = 0.50 // Exit when |z-score| < 0.5
	config.stop_loss_threshold = 3.0 // Stop loss at |z-score| > 3.0
	config.min_hold_days = 1 // Minimum 1 days holding period

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
		z_config.title = "Kalman Pairs Trading: Z-Score (NVDA/AMD)"
		z_config.x_label = "Time Step"
		z_config.y_label = "Z-Score"
		z_config.show_grid = true
		plot.multi_line_png(z_lines, "kalman_zscore_semiconductor.png", z_config, allocator)
		fmt.println("✓ Saved: kalman_zscore_semiconductor.png")

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
		eq_config.title = "Kalman Pairs Trading: Equity Curve (NVDA/AMD)"
		eq_config.x_label = "Time Step"
		eq_config.y_label = "Portfolio Value"
		eq_config.show_grid = true
		plot.multi_line_png(eq_lines, "kalman_equity_semiconductor.png", eq_config, allocator)
		fmt.println("✓ Saved: kalman_equity_semiconductor.png")

		// Cleanup plot arrays
		delete(xs, allocator)
		delete(thresh_pos, allocator)
		delete(thresh_neg, allocator)
		delete(thresh_exit_pos, allocator)
		delete(thresh_exit_neg, allocator)
		delete(equity, allocator)
	}
}
kalman_pairs_semiconductor_adf_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== Kalman Filter Pairs Trading (Semiconductors with ADF Check) ===")

	// 1. Fetch NVDA
	df_nvda := net.read_yahoo("NVDA", .Daily, .TwoYears, allocator)
	if df_nvda.rows == 0 {
		fmt.println("ERROR: Failed to fetch NVDA data")
		return
	}
	defer w.destroy_dataframe(&df_nvda)

	// 2. Fetch AMD
	df_amd := net.read_yahoo("AMD", .Daily, .TwoYears, allocator)
	if df_amd.rows == 0 {
		fmt.println("ERROR: Failed to fetch AMD data")
		return
	}
	defer w.destroy_dataframe(&df_amd)

	// Helper proc to align data and run ADF test on the spread
	test_pair_cointegration :: proc(
		df1: ^w.DataFrame,
		df2: ^w.DataFrame,
		sym1: string,
		sym2: string,
		alloc: mem.Allocator,
	) -> (
		bool,
		w.DataFrame,
	) {
		aligned := w.dataframe_new()
		col_date := w.column_new("Date", .Date, 0)
		col_a := w.column_new("Asset_A", .Float, 0)
		col_b := w.column_new("Asset_B", .Float, 0)

		date1 := w.column(df1, "Date")
		close1 := w.column(df1, "Close")
		date2 := w.column(df2, "Date")
		close2 := w.column(df2, "Close")

		i1, i2 := 0, 0
		for i1 < df1.rows && i2 < df2.rows {
			d1, _ := w.column_at_date(date1, i1)
			d2, _ := w.column_at_date(date2, i2)
			cmp := w.date_compare(d1, d2)
			if cmp == 0 {
				w.append_date(&col_date, d1)
				v1, _ := w.column_at_float(close1, i1)
				v2, _ := w.column_at_float(close2, i2)
				w.append_float(&col_a, v1)
				w.append_float(&col_b, v2)
				i1 += 1
				i2 += 1
			} else if cmp < 0 {
				i1 += 1
			} else {
				i2 += 1
			}
		}
		w.add_column(&aligned, col_date)
		w.add_column(&aligned, col_a)
		w.add_column(&aligned, col_b)
		aligned.rows = col_date.len

		if aligned.rows < 100 {
			w.destroy_dataframe(&aligned)
			return false, w.dataframe_new()
		}

		// Compute spread for ADF test
		spread := make([]f64, aligned.rows, alloc)
		for i in 0 ..< aligned.rows {
			v1, _ := w.column_at_float(w.column(&aligned, "Asset_A"), i)
			v2, _ := w.column_at_float(w.column(&aligned, "Asset_B"), i)
			spread[i] = v1 - v2
		}

		// Run ADF test on the spread
		_, p_val, lags, n_obs, _, _, _ := analytic.adf_test(spread, 10, .Constant, .AIC, alloc)
		delete(spread, alloc)

		fmt.printf(
			"  ADF Test for %s & %s: p-value = %.4f (Lags: %d, Obs: %d)\n",
			sym1,
			sym2,
			p_val,
			lags,
			n_obs,
		)

		is_cointegrated := p_val < 0.05
		if is_cointegrated {
			fmt.printf("  => %s and %s ARE cointegrated (p < 0.05).\n", sym1, sym2)
		} else {
			fmt.printf("  => %s and %s are NOT cointegrated (p >= 0.05).\n", sym1, sym2)
		}

		return is_cointegrated, aligned
	}

	// Test NVDA & AMD
	is_coin_amd, aligned_amd := test_pair_cointegration(
		&df_nvda,
		&df_amd,
		"NVDA",
		"AMD",
		allocator,
	)

	final_aligned: w.DataFrame
	sym_a, sym_b: string

	if is_coin_amd {
		fmt.println("  Proceeding with NVDA & AMD pair.")
		final_aligned = aligned_amd
		sym_a = "NVDA"
		sym_b = "AMD"
	} else {
		fmt.println("  Switching AMD for TSM (TSMC ADR)...")
		w.destroy_dataframe(&aligned_amd)

		df_tsm := net.read_yahoo("TSM", .Daily, .TwoYears, allocator)
		if df_tsm.rows == 0 {
			fmt.println("ERROR: Failed to fetch TSM data")
			return
		}
		defer w.destroy_dataframe(&df_tsm)

		is_coin_tsm, aligned_tsm := test_pair_cointegration(
			&df_nvda,
			&df_tsm,
			"NVDA",
			"TSM",
			allocator,
		)
		if is_coin_tsm {
			fmt.println("  Proceeding with NVDA & TSM pair.")
			final_aligned = aligned_tsm
			sym_a = "NVDA"
			sym_b = "TSM"
		} else {
			fmt.println("  ERROR: NVDA & TSM are also not cointegrated. Aborting.")
			w.destroy_dataframe(&aligned_tsm)
			return
		}
	}
	defer w.destroy_dataframe(&final_aligned)

	fmt.printf(
		"\nLoaded %d properly aligned days of real %s/%s data\n",
		final_aligned.rows,
		sym_a,
		sym_b,
	)

	// ========================================================================
	// CONFIGURABLE STRATEGY PARAMETERS
	// ========================================================================
	config := ml_fin.kalman_pairs_default_config()
	config.warmup_window = 60
	config.process_noise = 1e-5
	config.measurement_noise = 1e-4
	config.initial_hedge_ratio = 1.0
	config.entry_threshold = 1.50
	config.exit_threshold = 0.50
	config.stop_loss_threshold = 3.0
	config.min_hold_days = 1

	strategy_result := ml_fin.kalman_pairs_strategy(
		&final_aligned,
		"Asset_A",
		"Asset_B",
		config,
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
		for r in strategy_result.returns {total_return *= (1.0 + r)}
		total_return -= 1.0

		mean_ret := 0.0
		var_ret := 0.0
		count := len(strategy_result.returns)
		for r in strategy_result.returns {mean_ret += r}
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
			if cumulative > peak {peak = cumulative}
			dd := (peak - cumulative) / peak
			if dd > max_dd {max_dd = dd}
		}

		fmt.printf("Total Return:      %.2f%%\n", total_return * 100.0)
		fmt.printf("Annualized Sharpe: %.2f\n", sharpe)
		fmt.printf("Max Drawdown:      %.2f%%\n", max_dd * 100.0)
		fmt.printf("Total Trades:      %d\n", len(strategy_result.trades))
	} else {
		fmt.println("No trades generated or error in strategy execution.")
	}

	// Visualization
	n_steps := len(strategy_result.z_scores)
	if n_steps > 0 {
		xs := make([]f64, n_steps, allocator)
		for i in 0 ..< n_steps {xs[i] = f64(i)}

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
		z_config.title = fmt.tprintf("Kalman Pairs Trading: Z-Score (%s/%s)", sym_a, sym_b)
		z_config.x_label = "Time Step"
		z_config.y_label = "Z-Score"
		z_config.show_grid = true
		plot.multi_line_png(
			z_lines,
			fmt.tprintf("kalman_zscore_%s_%s.png", sym_a, sym_b),
			z_config,
			allocator,
		)
		fmt.printf("✓ Saved: kalman_zscore_%s_%s.png\n", sym_a, sym_b)

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
		eq_config.title = fmt.tprintf("Kalman Pairs Trading: Equity Curve (%s/%s)", sym_a, sym_b)
		eq_config.x_label = "Time Step"
		eq_config.y_label = "Portfolio Value"
		eq_config.show_grid = true
		plot.multi_line_png(
			eq_lines,
			fmt.tprintf("kalman_equity_%s_%s.png", sym_a, sym_b),
			eq_config,
			allocator,
		)
		fmt.printf("✓ Saved: kalman_equity_%s_%s.png\n", sym_a, sym_b)

		delete(xs, allocator)
		delete(thresh_pos, allocator)
		delete(thresh_neg, allocator)
		delete(thresh_exit_pos, allocator)
		delete(thresh_exit_neg, allocator)
		delete(equity, allocator)
	}
}
kalman_pairs_auto_select_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== Kalman Filter Pairs Auto-Selection + Grid Search ===")

	// Candidate pairs (Symbol 1, Symbol 2)
	candidates := []struct {
		sym1: string,
		sym2: string,
	} {
		{"XOM", "CVX"}, // Energy (Exxon / Chevron)
		{"JPM", "BAC"}, // Financials (JPMorgan / Bank of America)
		{"HD", "LOW"}, // Home Improvement (Home Depot / Lowe's)
		{"UNH", "CI"}, // Healthcare (UnitedHealth / Cigna)
		{"KO", "PEP"}, // Beverages (Coca-Cola / Pepsi)
		{"NVDA", "AMD"}, // Semiconductors (NVIDIA / AMD)
	}

	selected_sym1 := ""
	selected_sym2 := ""
	found_cointegrated := false

	// 1. Find a cointegrated pair
	fmt.println("\n--- Step 1: Testing Cointegration ---")
	for candidate in candidates {
		fmt.printf("Testing cointegration for %s & %s...\n", candidate.sym1, candidate.sym2)

		df1 := net.read_yahoo(candidate.sym1, .Daily, .TwoYears, allocator)
		if df1.rows == 0 {continue}
		defer w.destroy_dataframe(&df1)

		df2 := net.read_yahoo(candidate.sym2, .Daily, .TwoYears, allocator)
		if df2.rows == 0 {continue}
		defer w.destroy_dataframe(&df2)

		// Extract prices for spread calculation
		n := min(df1.rows, df2.rows)
		if n < 100 {continue}

		// Calculate log prices for cointegration test
		ret1 := make([]f64, n, context.temp_allocator)
		ret2 := make([]f64, n, context.temp_allocator)

		for i in 1 ..< n {
			v1, _ := w.column_at_float(w.column(&df1, "Close"), i)
			v1_prev, _ := w.column_at_float(w.column(&df1, "Close"), i - 1)
			v2, _ := w.column_at_float(w.column(&df2, "Close"), i)
			v2_prev, _ := w.column_at_float(w.column(&df2, "Close"), i - 1)
			if v1 > 0 && v2 > 0 {
				ret1[i - 1] = math.ln_f64(v1 / v1_prev)
				ret2[i - 1] = math.ln_f64(v2 / v2_prev)
			}
		}

		// Run ADF on the spread of log prices
		spread := make([]f64, n, context.temp_allocator)
		for i in 0 ..< n - 1 {
			spread[i] = ret1[i] - ret2[i]
		}

		_, p_val, lags, n_obs, _, _, _ := analytic.adf_test(
			spread,
			10,
			.Constant,
			.AIC,
			context.temp_allocator,
		)

		fmt.printf("  ADF p-value: %.4f (Lags: %d, Obs: %d)\n", p_val, lags, n_obs)

		if p_val < 0.05 {
			fmt.printf(
				"  => ✓ SUCCESS: %s & %s are cointegrated!\n",
				candidate.sym1,
				candidate.sym2,
			)
			selected_sym1 = candidate.sym1
			selected_sym2 = candidate.sym2
			found_cointegrated = true
			break
		} else {
			fmt.printf("  => ✗ FAILED: Not cointegrated (p >= 0.05).\n")
		}
	}

	if !found_cointegrated {
		fmt.println("\nERROR: No cointegrated pairs found in the candidate list. Aborting.")
		return
	}

	fmt.printf("\n✓ Selected pair: %s & %s\n", selected_sym1, selected_sym2)

	// 2. Fetch full data for the selected pair
	fmt.println("\n--- Step 2: Fetching Data ---")
	df1 := net.read_yahoo(selected_sym1, .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&df1)

	df2 := net.read_yahoo(selected_sym2, .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&df2)

	// 3. Align dataframes by DATE
	fmt.println("\n--- Step 3: Aligning Data ---")
	aligned_df := w.dataframe_new()
	defer w.destroy_dataframe(&aligned_df)

	col_date := w.column_new("Date", .Date, 0)
	col_a := w.column_new("Asset_A", .Float, 0)
	col_b := w.column_new("Asset_B", .Float, 0)

	date_col1 := w.column(&df1, "Date")
	close_col1 := w.column(&df1, "Close")
	date_col2 := w.column(&df2, "Date")
	close_col2 := w.column(&df2, "Close")

	i1, i2 := 0, 0
	for i1 < df1.rows && i2 < df2.rows {
		date1, _ := w.column_at_date(date_col1, i1)
		date2, _ := w.column_at_date(date_col2, i2)
		cmp := w.date_compare(date1, date2)
		if cmp == 0 {
			w.append_date(&col_date, date1)
			val1, _ := w.column_at_float(close_col1, i1)
			w.append_float(&col_a, val1)
			val2, _ := w.column_at_float(close_col2, i2)
			w.append_float(&col_b, val2)
			i1 += 1
			i2 += 1
		} else if cmp < 0 {
			i1 += 1
		} else {
			i2 += 1
		}
	}
	w.add_column(&aligned_df, col_date)
	w.add_column(&aligned_df, col_a)
	w.add_column(&aligned_df, col_b)
	aligned_df.rows = col_date.len

	fmt.printf(
		"✓ Loaded %d aligned days of %s/%s data\n",
		aligned_df.rows,
		selected_sym1,
		selected_sym2,
	)

	// 4. Grid Search for Optimal Parameters
	fmt.println("\n--- Step 4: Running Grid Search ---")

	// Define search space
	entry_thresholds := []f64{1.5, 2.0, 2.5}
	exit_thresholds := []f64{0.0, 0.5}
	process_noises := []f64{1e-6, 1e-5, 1e-4}
	min_hold_days_list := []int{1, 3, 5}
	cooldown_days_list := []int{0, 3}
	transaction_cost := 0.001 // 10 bps per leg

	best_config, best_sharpe := ml_fin.kalman_pairs_grid_search(
		&aligned_df,
		"Asset_A",
		"Asset_B",
		entry_thresholds,
		exit_thresholds,
		process_noises,
		min_hold_days_list,
		cooldown_days_list,
		transaction_cost,
		allocator,
	)

	fmt.println("\n--- Best Configuration Found ---")
	fmt.printf("  Entry Threshold:   ±%.2f\n", best_config.entry_threshold)
	fmt.printf("  Exit Threshold:    ±%.2f\n", best_config.exit_threshold)
	fmt.printf("  Process Noise:     %.2e\n", best_config.process_noise)
	fmt.printf("  Min Hold Days:     %d\n", best_config.min_hold_days)
	fmt.printf("  Cooldown Days:     %d\n", best_config.cooldown_days)
	fmt.printf(
		"  Transaction Cost:  %.4f (%.0f bps)\n",
		best_config.transaction_cost,
		best_config.transaction_cost * 10000,
	)
	fmt.printf("  Best Sharpe:       %.4f\n", best_sharpe)

	// 5. Run Final Strategy with Best Config
	fmt.println("\n--- Step 5: Running Final Strategy ---")
	final_result := ml_fin.kalman_pairs_strategy(
		&aligned_df,
		"Asset_A",
		"Asset_B",
		best_config,
		allocator,
	)

	// 6. Analyze Results
	fmt.println("\n--- Final Performance Metrics ---")
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

		// Calculate Sortino Ratio (downside deviation)
		downside_returns := make([dynamic]f64, 0, allocator)
		for r in final_result.returns {
			if r < 0 {
				append(&downside_returns, r)
			}
		}
		downside_var := 0.0
		if len(downside_returns) > 0 {
			for r in downside_returns {
				downside_var += r * r
			}
			downside_var /= f64(len(downside_returns))
		}
		downside_std := math.sqrt_f64(downside_var)
		sortino := 0.0
		if downside_std > 1e-10 {
			sortino = (mean_ret / downside_std) * math.sqrt_f64(252.0)
		}
		delete(downside_returns)

		// Max Drawdown
		max_dd := 0.0
		peak := 1.0
		cumulative := 1.0
		for r in final_result.returns {
			cumulative *= (1.0 + r)
			if cumulative > peak {peak = cumulative}
			dd := (peak - cumulative) / peak
			if dd > max_dd {max_dd = dd}
		}

		fmt.printf("Total Return:      %+.2f%%\n", total_return * 100.0)
		fmt.printf("Annualized Sharpe: %.2f\n", sharpe)
		fmt.printf("Sortino Ratio:     %.2f\n", sortino)
		fmt.printf("Max Drawdown:      %.2f%%\n", max_dd * 100.0)
		fmt.printf("Total Trades:      %d\n", len(final_result.trades))
		fmt.printf("Avg Daily Return:  %.4f%%\n", mean_ret * 100.0)
	} else {
		fmt.println("✗ No trades generated or error in strategy execution.")
	}

	// 7. Visualization
	fmt.println("\n--- Step 6: Generating Visualizations ---")
	n_steps := len(final_result.z_scores)
	if n_steps > 0 {
		xs := make([]f64, n_steps, allocator)
		for i in 0 ..< n_steps {xs[i] = f64(i)}

		// Plot 1: Z-Score with thresholds
		thresh_pos := make([]f64, n_steps, allocator)
		thresh_neg := make([]f64, n_steps, allocator)
		thresh_exit_pos := make([]f64, n_steps, allocator)
		thresh_exit_neg := make([]f64, n_steps, allocator)
		for i in 0 ..< n_steps {
			thresh_pos[i] = best_config.entry_threshold
			thresh_neg[i] = -best_config.entry_threshold
			thresh_exit_pos[i] = best_config.exit_threshold
			thresh_exit_neg[i] = -best_config.exit_threshold
		}

		z_lines := []plot.LineData {
			plot.LineData {
				xs = xs,
				ys = final_result.z_scores,
				color = plot.BLUE,
				style = .Solid,
				label = "Z-Score",
			},
			plot.LineData {
				xs = xs,
				ys = thresh_pos,
				color = plot.RED,
				style = .Dashed,
				label = fmt.tprintf("Entry (+%.1f)", best_config.entry_threshold),
			},
			plot.LineData {
				xs = xs,
				ys = thresh_neg,
				color = plot.RED,
				style = .Dashed,
				label = fmt.tprintf("Entry (-%.1f)", best_config.entry_threshold),
			},
			plot.LineData {
				xs = xs,
				ys = thresh_exit_pos,
				color = plot.GREEN,
				style = .Dotted,
				label = fmt.tprintf("Exit (+%.1f)", best_config.exit_threshold),
			},
			plot.LineData {
				xs = xs,
				ys = thresh_exit_neg,
				color = plot.GREEN,
				style = .Dotted,
				label = fmt.tprintf("Exit (-%.1f)", best_config.exit_threshold),
			},
		}

		z_config := plot.DEFAULT_PLOT_CONFIG
		z_config.title = fmt.tprintf("Kalman Pairs: Z-Score (%s/%s)", selected_sym1, selected_sym2)
		z_config.x_label = "Time Step"
		z_config.y_label = "Z-Score"
		z_config.show_grid = true

		z_filename := fmt.tprintf("kalman_auto_zscore_%s_%s.png", selected_sym1, selected_sym2)
		plot.multi_line_png(z_lines, z_filename, z_config, allocator)
		fmt.printf("✓ Saved: %s\n", z_filename)

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
		eq_config.title = fmt.tprintf(
			"Kalman Pairs: Equity Curve (%s/%s)",
			selected_sym1,
			selected_sym2,
		)
		eq_config.x_label = "Time Step"
		eq_config.y_label = "Portfolio Value"
		eq_config.show_grid = true

		eq_filename := fmt.tprintf("kalman_auto_equity_%s_%s.png", selected_sym1, selected_sym2)
		plot.multi_line_png(eq_lines, eq_filename, eq_config, allocator)
		fmt.printf("✓ Saved: %s\n", eq_filename)

		// Cleanup
		delete(xs, allocator)
		delete(thresh_pos, allocator)
		delete(thresh_neg, allocator)
		delete(thresh_exit_pos, allocator)
		delete(thresh_exit_neg, allocator)
		delete(equity, allocator)
	}

	fmt.println("\n✓ Kalman Pairs Auto-Selection + Grid Search completed!")
}
