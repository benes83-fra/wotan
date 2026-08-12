package tests

import w "../wotan/core"
import fin "../wotan/finance"
import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Helper to generate cointegrated GBM series
generate_cointegrated_series :: proc(
	n: int,
	mean1: f64,
	std1: f64,
	mean2: f64,
	std2: f64,
	correlation: f64,
	cointegration_strength: f64, // How strongly they revert to each other
	allocator: mem.Allocator,
) -> (
	series1: []f64,
	series2: []f64,
) {
	s1 := make([]f64, n, allocator)
	s2 := make([]f64, n, allocator)

	// Initialize prices
	s1[0] = 100.0
	s2[0] = 50.0

	for i in 1 ..< n {
		// Generate independent normal random variables
		z1 := rand.float64_normal(0.0, 1.0)
		z2 := rand.float64_normal(0.0, 1.0)
		z_spread := rand.float64_normal(0.0, 1.0) // Noise for the spread

		// Correlate them: y2 = corr * z1 + sqrt(1-corr^2) * z2
		y1 := z1
		y2 := correlation * z1 + math.sqrt(1.0 - correlation * correlation) * z2

		// Update prices using GBM
		s1[i] = s1[i - 1] * math.exp((mean1 - 0.5 * std1 * std1) + std1 * y1)

		// Make s2 cointegrated with s1
		// Log-price of s2 follows: log(s2[i]) = log(s1[i]) + spread[i]
		// spread[i] = spread[i-1] * (1 - cointegration_strength) + noise
		log_s1 := math.ln(s1[i])
		log_s2_prev := math.ln(s2[i - 1])

		// Simple error correction model
		spread_error := log_s1 - log_s2_prev
		new_log_s2 :=
			log_s2_prev +
			(mean2 - 0.5 * std2 * std2) +
			std2 * y2 +
			cointegration_strength * spread_error

		s2[i] = math.exp(new_log_s2)
	}

	return s1[:], s2[:]
}

kalman_pairs_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== Kalman Filter Pairs Trading Test ===")

	// 1. Generate Cointegrated Data
	n_days := 500
	s1, s2 := generate_cointegrated_series(
		n_days,
		0.0005, // daily mean return
		0.01, // daily vol
		0.0005,
		0.015,
		0.85, // correlation
		0.05, // cointegration strength (adjust this!)
		allocator,
	)
	defer {
		delete(s1, allocator)
		delete(s2, allocator)
	}

	// Create DataFrame
	df := w.dataframe_new()
	w.add_column(&df, w.column_from_floats("Asset_A", s1))
	w.add_column(&df, w.column_from_floats("Asset_B", s2))

	fmt.printf("Generated %d days of cointegrated data\n", n_days)

	// 2. Run Kalman Filter Pairs Trading Strategy
	window := 60
	initial_hedge_ratio := 1.0 // Start with 1:1

	strategy_result := ml_fin.kalman_pairs_strategy(
		&df,
		"Asset_A",
		"Asset_B",
		window,
		initial_hedge_ratio,
		allocator,
	)

	// 3. Analyze Results
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

	w.destroy_dataframe(&df)
}
