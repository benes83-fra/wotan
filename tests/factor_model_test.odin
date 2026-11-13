package tests

import ts "../wotan/analytics"
import fin "../wotan/finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

factor_model_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Factor Model Covariance Test (Fama-French Proxies) ===\n")

	main_alloc := context.allocator

	// 1. Define Assets and Factor Proxies
	assets := []string{"AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG"}
	factors := []string{"SPY", "IWM", "VTV", "MTUM"}

	// Assets and factor proxies in one panel, matched by date
	fmt.println("Fetching asset and factor proxy data...")
	tickers := make([]string, len(assets) + len(factors), main_alloc)
	defer delete(tickers, main_alloc)
	copy(tickers, assets)
	copy(tickers[len(assets):], factors)
	NT := len(tickers)

	panel, panel_ok, failed := yahoo.read_yahoo_panel(tickers, .Daily, .TwoYears, allocator)
	defer yahoo.yahoo_panel_free(&panel)
	if !panel_ok {
		what := "Yahoo data shared by all assets and factors"
		if failed >= 0 {what = fmt.tprintf("Yahoo data for %s", tickers[failed])}
		fail_missing_data(#procedure, what)
		return
	}

	// 2. Compute Log Returns (adjusted closes) on the common dates
	panel_returns := ts.log_returns(panel.adj_close, NT, allocator) // [day][ticker]
	defer delete(panel_returns, allocator)
	min_len := len(panel.dates) - 1

	if min_len < 50 {
		fmt.println("Error: Insufficient data for factor model.")
		return
	}

	// Extract returns
	assets_returns := make([][]f64, min_len, main_alloc)
	defer {
		for i in 0 ..< min_len {delete(assets_returns[i], main_alloc)}
		delete(assets_returns, main_alloc)
	}
	for i in 0 ..< min_len {
		assets_returns[i] = make([]f64, len(assets), main_alloc)
	}

	factors_returns := make([][]f64, min_len, main_alloc)
	defer {
		for i in 0 ..< min_len {delete(factors_returns[i], main_alloc)}
		delete(factors_returns, main_alloc)
	}
	for i in 0 ..< min_len {
		factors_returns[i] = make([]f64, len(factors), main_alloc)
	}

	// Populate returns (row t is the return from common day t to day t + 1)
	for t in 0 ..< min_len {
		for a in 0 ..< len(assets) {
			assets_returns[t][a] = panel_returns[t * NT + a]
		}
		for f in 0 ..< len(factors) {
			factors_returns[t][f] = panel_returns[t * NT + len(assets) + f]
		}
	}

	fmt.printf(
		"Aligned %d daily observations for %d assets and %d factors.\n",
		min_len,
		len(assets),
		len(factors),
	)

	// 3. Compute Factor Model Covariance
	fmt.println("\nComputing Factor Model Covariance...")
	fm_cov := fin.factor_model_covariance(assets_returns, factors_returns, main_alloc)
	defer fin.destroy_factor_model_covariance(&fm_cov, main_alloc)

	// 4. Display Results
	fmt.println("\n--- Factor Loadings (Betas) ---")
	fmt.printf("%-8s ", "Asset")
	for f in factors {
		fmt.printf("%-8s ", f)
	}
	fmt.println()
	fmt.printf("%-8s ", "--------")
	for _ in factors {
		fmt.printf("%-8s ", "--------")
	}
	fmt.println()

	for ticker, a in assets {
		fmt.printf("%-8s ", ticker)
		for f in 0 ..< len(factors) {
			fmt.printf("%-8.3f ", fm_cov.betas[a][f])
		}
		fmt.println()
	}

	fmt.println("\n--- Risk Decomposition ---")
	fmt.printf("%-8s %-15s %-15s %-15s\n", "Asset", "Total Var", "Sys Var", "Idio Var (%)")
	fmt.printf(
		"%-8s %-15s %-15s %-15s\n",
		"--------",
		"---------------",
		"---------------",
		"---------------",
	)
	for ticker, a in assets {
		total_var := fm_cov.sample_cov[a][a]
		sys_var := total_var - fm_cov.idiosyncratic_var[a]
		idio_pct := (fm_cov.idiosyncratic_var[a] / total_var) * 100.0
		fmt.printf("%-8s %-15.6f %-15.6f %-14.2f%%\n", ticker, total_var, sys_var, idio_pct)
	}

	// 5. Compare Sample vs Reconstructed Covariance
	fmt.println("\n--- Covariance Matrix Comparison (Sample vs Reconstructed) ---")
	// Show AAPL (0) vs MSFT (1)
	a1, a2 := 0, 1
	sample_cov_12 := fm_cov.sample_cov[a1][a2]
	recon_cov_12 := fm_cov.reconstructed_cov[a1][a2]

	fmt.printf("Covariance between %s and %s:\n", assets[a1], assets[a2])
	fmt.printf("  Sample Covariance:       %.6f\n", sample_cov_12)
	fmt.printf("  Reconstructed Covariance: %.6f\n", recon_cov_12)
	fmt.printf("  Difference:              %.6f\n", math.abs(sample_cov_12 - recon_cov_12))

	fmt.println("\n✓ Factor Model Covariance test completed!")
}
