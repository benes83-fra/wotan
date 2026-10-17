package tests

import bfin "../wotan/behavioral_finance"
import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

prospect_portfolio_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Prospect Theory Portfolio Optimization ===")
	fmt.println("Comparing Markowitz Max-Sharpe vs Behavioral Prospect Optimization.")

	symbols := []string {
		"XLK",
		"XLF",
		"XLE",
		"XLV",
		"XLI",
		"XLY",
		"XLP",
		"XLB",
		"XLRE",
		"XLU",
		"XLC",
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

	// 2. Compute Daily Returns Matrix
	returns_mat := l.matrix_new(f64, min_time - 1, n_assets, allocator)
	defer l.matrix_free(&returns_mat)

	for a in 0 ..< n_assets {
		for t in 1 ..< min_time {
			returns_mat.data[(t - 1) * n_assets + a] = math.ln_f64(prices[a][t] / prices[a][t - 1])
		}
	}

	// 3. Markowitz Max Sharpe
	fmt.println("- Computing Markowitz Max-Sharpe Portfolio...")
	mu := fin.expected_returns_from_history(&returns_mat, allocator)
	defer delete(mu, allocator)

	cov := fin.covariance_matrix(&returns_mat, allocator)
	defer l.matrix_free(&cov)

	w_sharpe := fin.max_sharpe_portfolio(mu, &cov, 0.0, allocator)
	defer delete(w_sharpe, allocator)

	metrics_sharpe := bfin.prospect_metrics(
		w_sharpe,
		&returns_mat,
		bfin.ProspectPortfolioParams{0.88, 0.88, 2.25, 0.0},
	)

	// 4. Prospect Theory Optimization
	fmt.println("- Computing Prospect Theory Portfolio (Loss-Averse)...")
	params := bfin.ProspectPortfolioParams {
		alpha     = 0.88,
		beta      = 0.88,
		lambda    = 2.25,
		ref_point = 0.0, // Beating cash
	}

	w_prospect := bfin.prospect_portfolio_optimize(&returns_mat, params, 0.5, 2000, allocator)
	defer delete(w_prospect, allocator)

	metrics_prospect := bfin.prospect_metrics(w_prospect, &returns_mat, params)

	// 5. Print Comparison
	fmt.println("\n--- Asset Allocation Comparison ---")
	fmt.printf("%-5s | %-10s | %-10s\n", "Asset", "Max Sharpe", "Prospect")
	fmt.println("------|------------|-----------")
	for a in 0 ..< n_assets {
		fmt.printf(
			"%-5s | %9.2f%% | %9.2f%%\n",
			symbols[a],
			w_sharpe[a] * 100.0,
			w_prospect[a] * 100.0,
		)
	}

	fmt.println("\n--- Performance Metrics (Daily) ---")
	fmt.printf(
		"Expected Return:   %+8.4f%%  |  %+8.4f%%\n",
		metrics_sharpe.expected_return * 100.0,
		metrics_prospect.expected_return * 100.0,
	)
	fmt.printf(
		"Volatility:        %8.4f%%  |  %8.4f%%\n",
		metrics_sharpe.volatility * 100.0,
		metrics_prospect.volatility * 100.0,
	)
	fmt.printf(
		"Loss Probability:  %8.2f%%  |  %8.2f%%\n",
		metrics_sharpe.loss_probability * 100.0,
		metrics_prospect.loss_probability * 100.0,
	)
	fmt.printf(
		"Prospect Utility:  %8.4f   |  %8.4f\n",
		metrics_sharpe.prospect_utility,
		metrics_prospect.prospect_utility,
	)

	fmt.println("\n💡 Insight: Prospect Theory heavily penalizes assets with 'fat left tails'")
	fmt.println(
		"(severe drawdowns), naturally shifting weight toward smoother, compounding assets.",
	)

	fmt.println("\n=== Prospect Portfolio Test Complete ===\n")
}
