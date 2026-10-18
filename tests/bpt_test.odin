package tests

import bfin "../wotan/behavioral_finance"
import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

bpt_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Behavioral Portfolio Theory (Mental Accounting) ===")
	fmt.println("Building the Shefrin & Statman 3-Layer Pyramid.")
	fmt.println("Layer 1: Safety (Min Variance)")
	fmt.println("Layer 2: Aspiration (Max Sharpe)")
	fmt.println("Layer 3: Speculative (Risk-Seeking Prospect)")

	// Universe: Sectors + Bonds + Gold + Broad Market
	symbols := []string {
		"TLT", // Bonds (Safety)
		"GLD", // Gold (Safety)
		"XLU", // Utilities (Safety)
		"SPY", // Broad Market (Aspiration)
		"XLF", // Financials (Aspiration)
		"XLI", // Industrials (Aspiration)
		"XLK", // Tech (Speculative/Aspiration)
		"XLE", // Energy (Speculative)
		"XLY", // Discretionary (Speculative)
	}
	n_assets := len(symbols)

	// 1. Fetch Data
	fmt.println("- Fetching historical data...")
	prices := make([][]f64, n_assets, allocator)
	defer {
		for a in 0 ..< n_assets {if prices[a] != nil {delete(prices[a], allocator)}}
		delete(prices, allocator)
	}

	n_time := 0
	for a in 0 ..< n_assets {
		df := net.read_yahoo(symbols[a], .Daily, .FiveYears, allocator)
		if df.rows >= 252 {
			close_col := w.column(&df, "Close")
			prices[a] = make([]f64, df.rows, allocator)
			for i in 0 ..< df.rows {prices[a][i], _ = w.column_at_float(close_col, i)}
			if df.rows > n_time {n_time = df.rows}
		}
		w.destroy_dataframe(&df)
	}

	min_time := n_time
	for a in 0 ..< n_assets {
		if prices[a] != nil && len(prices[a]) < min_time {min_time = len(prices[a])}
	}

	// 2. Compute Monthly Returns (Avoid Myopic Loss Aversion)
	rebalance_days := 21
	n_months := (min_time - 1) / rebalance_days
	monthly_returns := l.matrix_new(f64, n_months, n_assets, allocator)
	defer l.matrix_free(&monthly_returns)

	m_idx := 0
	for t in rebalance_days ..< min_time {
		if t % rebalance_days == 0 && m_idx < n_months {
			for a in 0 ..< n_assets {
				monthly_returns.data[m_idx * n_assets + a] = math.ln_f64(
					prices[a][t] / prices[a][t - rebalance_days],
				)
			}
			m_idx += 1
		}
	}

	// 3. Run BPT Optimizer
	// Typical retail investor allocation: 40% Safety, 50% Aspiration, 10% Speculative
	fmt.println("- Optimizing Mental Accounting Layers...")
	combined, w_safety, w_asp, w_spec := bfin.bpt_optimize(
		&monthly_returns,
		0.40,
		0.50,
		0.10,
		allocator,
	)
	defer {
		delete(combined, allocator)
		delete(w_safety, allocator)
		delete(w_asp, allocator)
		delete(w_spec, allocator)
	}

	// 4. Print the Pyramid
	fmt.println("\n--- The Mental Accounting Pyramid ---")
	fmt.printf(
		"%-5s | %-10s | %-10s | %-10s | %-10s\n",
		"Asset",
		"Safety",
		"Aspiration",
		"Speculative",
		"COMBINED",
	)
	fmt.println("------|------------|------------|-------------|-----------")
	for a in 0 ..< n_assets {
		fmt.printf(
			"%-5s | %9.2f%% | %9.2f%% | %10.2f%% | %9.2f%%\n",
			symbols[a],
			w_safety[a] * 100.0,
			w_asp[a] * 100.0,
			w_spec[a] * 100.0,
			combined[a] * 100.0,
		)
	}

	// 5. Evaluate Combined Metrics
	metrics := bfin.prospect_metrics(
		combined,
		&monthly_returns,
		bfin.ProspectPortfolioParams{0.88, 0.88, 2.25, 0.0},
	)

	fmt.println("\n--- Combined Portfolio Metrics (Monthly) ---")
	fmt.printf("Expected Return:   %+8.4f%%\n", metrics.expected_return * 100.0)
	fmt.printf("Volatility:        %8.4f%%\n", metrics.volatility * 100.0)
	fmt.printf("Loss Probability:  %8.2f%%\n", metrics.loss_probability * 100.0)
	fmt.printf("Prospect Utility:  %8.4f\n", metrics.prospect_utility)

	fmt.println("\n💡 Insight: BPT perfectly separates human psychological needs.")
	fmt.println("The Safety layer hides in Bonds/Gold, Aspiration rides the Market,")
	fmt.println("and Speculative chases high-beta Tech/Energy without risking the core.")

	fmt.println("\n=== Behavioral Portfolio Theory Test Complete ===\n")
}
