package tests

import bfin "../wotan/behavioral_finance"
import w "../wotan/core"
import net "../wotan/net"
import plot "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:mem"

anchoring_backtest_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Anchoring & 52-Week High Backtest ===")
	fmt.println("Testing the George & Hwang (2004) anchoring anomaly.")
	fmt.println("Strategy: Long stocks near 52-week high, short stocks far below.")

	// 1. Define universe of ETFs (diversified asset classes)
	// 1. Define universe: S&P 500 Sectors (High dispersion, sentiment-driven)
	symbols := []string {
		"XLK", // Technology
		"XLF", // Financials
		"XLE", // Energy
		"XLV", // Health Care
		"XLI", // Industrials
		"XLY", // Consumer Discretionary
		"XLP", // Consumer Staples
		"XLB", // Materials
		"XLRE", // Real Estate
		"XLU", // Utilities
		"XLC", // Communication Services
	}

	n_assets := len(symbols)

	// 2. Fetch historical data
	fmt.println("- Fetching historical data for universe...")
	prices := make([][]f64, n_assets, allocator)
	defer {
		for a in 0 ..< n_assets {
			if prices[a] != nil {
				delete(prices[a], allocator)
			}
		}
		delete(prices, allocator)
	}

	n_time := 0
	for a in 0 ..< n_assets {
		fmt.printf("  Fetching %s...\n", symbols[a])
		df := net.read_yahoo(symbols[a], .Daily, .FiveYears, allocator)

		// Note: We must copy the data out before destroying the dataframe
		if df.rows >= 252 {
			close_col := w.column(&df, "Close")
			prices[a] = make([]f64, df.rows, allocator)
			for i in 0 ..< df.rows {
				prices[a][i], _ = w.column_at_float(close_col, i)
			}
			if df.rows > n_time {
				n_time = df.rows
			}
		} else {
			fmt.printf("  Warning: %s has insufficient data (%d rows)\n", symbols[a], df.rows)
		}

		w.destroy_dataframe(&df)
	}

	// Align all series to same length (use minimum)
	min_time := n_time
	for a in 0 ..< n_assets {
		if prices[a] != nil && len(prices[a]) < min_time {
			min_time = len(prices[a])
		}
	}

	fmt.printf("Aligned to %d trading days (~%.1f years)\n", min_time, f64(min_time) / 252.0)

	// 3. Compute anchoring signal
	fmt.println("- Computing 52-week high proximity signal...")
	signal := bfin.compute_anchoring_signal(prices, 252, allocator)
	defer {
		for a in 0 ..< n_assets {
			if signal[a] != nil {
				delete(signal[a], allocator)
			}
		}
		delete(signal, allocator)
	}

	// 4. Monthly rebalancing backtest
	fmt.println("- Running monthly rebalancing backtest...")

	rebalance_days := 21 // ~1 month

	// ✅ FIX 1: Pre-allocate slices instead of using dynamic arrays and `append`
	max_periods := (min_time - 252) / rebalance_days + 2

	portfolio_returns := make([]f64, max_periods, allocator)
	defer delete(portfolio_returns, allocator)

	benchmark_returns := make([]f64, max_periods, allocator)
	defer delete(benchmark_returns, allocator)

	portfolio_values := make([]f64, max_periods, allocator)
	defer delete(portfolio_values, allocator)

	benchmark_values := make([]f64, max_periods, allocator)
	defer delete(benchmark_values, allocator)

	portfolio_value := 100000.0
	benchmark_value := 100000.0

	period_idx := 0
	portfolio_values[period_idx] = portfolio_value
	benchmark_values[period_idx] = benchmark_value
	period_idx += 1

	// Pre-allocate loop buffer to prevent memory leaks
	current_signal := make([]f64, n_assets, allocator)
	defer delete(current_signal, allocator)

	for t in 252 ..< min_time - rebalance_days {
		if t % rebalance_days != 0 {
			continue
		}

		// Get current signal
		for a in 0 ..< n_assets {
			if signal[a] != nil {
				current_signal[a] = signal[a][t]
			} else {
				current_signal[a] = 0.0
			}
		}

		// ✅ PIVOT: Long-Only Breakout Strategy
		// Buy sectors crossing the 0.90 threshold, hold Cash for the rest.
		weights := bfin.generate_long_only_portfolio(current_signal, 0.90, allocator)
		// ✅ FIX 2: DO NOT use `defer` inside a loop! It will leak memory.
		// We explicitly delete it at the end of the loop body.

		// Compute returns over next rebalance period
		port_ret := 0.0
		bench_ret := 0.0

		for a in 0 ..< n_assets {
			if prices[a] == nil {
				continue
			}
			asset_ret := (prices[a][t + rebalance_days] / prices[a][t]) - 1.0
			port_ret += weights[a] * asset_ret
			bench_ret += asset_ret / f64(n_assets) // Equal-weight benchmark
		}

		delete(weights, allocator) // ✅ Explicit cleanup inside loop

		portfolio_returns[period_idx - 1] = port_ret
		benchmark_returns[period_idx - 1] = bench_ret

		portfolio_value *= (1.0 + port_ret)
		benchmark_value *= (1.0 + bench_ret)

		portfolio_values[period_idx] = portfolio_value
		benchmark_values[period_idx] = benchmark_value
		period_idx += 1
	}

	// 5. Performance metrics
	n_periods := period_idx - 1
	total_port_ret := (portfolio_value / 100000.0 - 1.0) * 100.0
	total_bench_ret := (benchmark_value / 100000.0 - 1.0) * 100.0

	mean_port_ret := 0.0
	mean_bench_ret := 0.0
	for i in 0 ..< n_periods {
		mean_port_ret += portfolio_returns[i]
		mean_bench_ret += benchmark_returns[i]
	}
	if n_periods > 0 {
		mean_port_ret /= f64(n_periods)
		mean_bench_ret /= f64(n_periods)
	}

	var_port_ret := 0.0
	var_bench_ret := 0.0
	for i in 0 ..< n_periods {
		d_p := portfolio_returns[i] - mean_port_ret
		var_port_ret += d_p * d_p
		d_b := benchmark_returns[i] - mean_bench_ret
		var_bench_ret += d_b * d_b
	}
	if n_periods > 1 {
		var_port_ret /= f64(n_periods - 1)
		var_bench_ret /= f64(n_periods - 1)
	}

	std_port_ret := math.sqrt_f64(var_port_ret)
	std_bench_ret := math.sqrt_f64(var_bench_ret)

	sharpe_port := 0.0
	if std_port_ret > 1e-9 {sharpe_port = (mean_port_ret / std_port_ret) * math.sqrt_f64(12.0)}
	sharpe_bench := 0.0
	if std_bench_ret > 1e-9 {sharpe_bench = (mean_bench_ret / std_bench_ret) * math.sqrt_f64(12.0)}

	// Max drawdown
	peak_port := 0.0
	max_dd_port := 0.0
	for i in 0 ..< period_idx {
		v := portfolio_values[i]
		if v > peak_port {peak_port = v}
		if peak_port > 0 {
			dd := (peak_port - v) / peak_port
			if dd > max_dd_port {max_dd_port = dd}
		}
	}

	fmt.println("\n--- Backtest Results (Monthly Rebalancing) ---")
	fmt.printf("Period:              %d months (~%.1f years)\n", n_periods, f64(n_periods) / 12.0)
	fmt.printf("Benchmark Return:    %+7.2f%%\n", total_bench_ret)
	fmt.printf("Anchoring Return:    %+7.2f%%\n", total_port_ret)
	fmt.printf("Alpha:               %+7.2f%%\n", total_port_ret - total_bench_ret)
	fmt.printf("Absolute Return:     %+7.2f%% (vs Cash)\n", total_port_ret)
	fmt.printf("Benchmark Sharpe:    %7.2f\n", sharpe_bench)
	fmt.printf("Anchoring Sharpe:    %7.2f\n", sharpe_port)
	fmt.printf("Max Drawdown:        %7.2f%%\n", max_dd_port * 100.0)

	// 6. Plot results
	fmt.println("\n- Generating plots...")

	months := make([]f64, period_idx, allocator)
	norm_port := make([]f64, period_idx, allocator)
	norm_bench := make([]f64, period_idx, allocator)
	defer {
		delete(months, allocator)
		delete(norm_port, allocator)
		delete(norm_bench, allocator)
	}

	for i in 0 ..< period_idx {
		months[i] = f64(i)
		norm_port[i] = portfolio_values[i] / 100000.0
		norm_bench[i] = benchmark_values[i] / 100000.0
	}

	lines := []plot.LineData {
		{
			xs = months,
			ys = norm_bench,
			color = plot.BLUE,
			style = .Solid,
			label = "Equal-Weight Benchmark",
		},
		{
			xs = months,
			ys = norm_port,
			color = plot.RED,
			style = .Solid,
			label = "Anchoring Strategy",
		},
	}

	config := plot.DEFAULT_PLOT_CONFIG
	config.title = "52-Week High Anchoring Strategy vs Benchmark"
	config.x_label = "Months"
	config.y_label = "Normalized Value"
	config.show_grid = true

	if plot.multi_line_png(lines, "plots/anchoring_backtest.png", config, allocator) {
		fmt.println("Saved: anchoring_backtest.png")
	}

	// 7. Analyze signal distribution
	fmt.println("\n- Analyzing signal distribution...")

	high_proximity_days := 0
	low_proximity_days := 0
	total_observations := 0

	for a in 0 ..< n_assets {
		if signal[a] == nil {
			continue
		}
		for t in 252 ..< min_time {
			total_observations += 1
			if signal[a][t] > 0.95 {
				high_proximity_days += 1
			} else if signal[a][t] < 0.5 {
				low_proximity_days += 1
			}
		}
	}

	fmt.printf("Total observations:  %d\n", total_observations)
	if total_observations > 0 {
		fmt.printf(
			"High proximity (>0.95): %d (%.1f%%)\n",
			high_proximity_days,
			100.0 * f64(high_proximity_days) / f64(total_observations),
		)
		fmt.printf(
			"Low proximity (<0.5):   %d (%.1f%%)\n",
			low_proximity_days,
			100.0 * f64(low_proximity_days) / f64(total_observations),
		)
	}

	fmt.println("\n=== Anchoring Backtest Complete ===\n")
}
