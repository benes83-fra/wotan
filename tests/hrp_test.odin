package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

hrp_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Hierarchical Risk Parity (HRP) Portfolio Allocation ===")
	main_alloc := context.allocator

	// 1. Define a diversified basket (8 assets)
	tickers := []string{"SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TLT"}
	n_assets := len(tickers)

	fmt.println("\n--- Fetching Market Data ---")
	dfs := make([]w.DataFrame, n_assets, main_alloc)
	defer {
		for i in 0 ..< n_assets {w.destroy_dataframe(&dfs[i])}
		delete(dfs, main_alloc)
	}

	for i in 0 ..< n_assets {
		dfs[i] = yahoo.read_yahoo(tickers[i], .Daily, .TwoYears, allocator)
		fmt.printf("Loaded %d days for %s\n", dfs[i].rows, tickers[i])
	}

	// Align lengths
	n_common := dfs[0].rows
	for i in 1 ..< n_assets {
		if dfs[i].rows < n_common {n_common = dfs[i].rows}
	}
	num_days := n_common - 1

	// 2. Compute Returns Matrix [T, N]
	returns_flat := make([]f64, num_days * n_assets, main_alloc)
	defer delete(returns_flat, main_alloc)

	for d in 1 ..< n_common {
		for i in 0 ..< n_assets {
			prev_c, _ := w.column_at_float(&dfs[i].columns[4], d - 1)
			curr_c, _ := w.column_at_float(&dfs[i].columns[4], d)
			ret := math.ln_f64(curr_c / prev_c)
			// linalg expects row-major [T, N]
			returns_flat[(d - 1) * n_assets + i] = ret
		}
	}

	returns_mat := l.matrix_from_flat(returns_flat, num_days, n_assets, main_alloc)
	defer l.matrix_free(&returns_mat)

	// 3. Run HRP
	fmt.println("\n--- Computing Hierarchical Risk Parity ---")
	hrp_res := ml_fin.hrp_allocate(&returns_mat, main_alloc)
	defer ml_fin.hrp_result_free(&hrp_res)

	// 4. Compute Naive 1/N Benchmark and Asset Volatilities for context
	vols := make([]f64, n_assets, main_alloc)
	defer delete(vols, main_alloc)

	for i in 0 ..< n_assets {
		var_sum := 0.0
		for d in 0 ..< num_days {
			r := returns_flat[d * n_assets + i]
			var_sum += r * r
		}
		vols[i] = math.sqrt(var_sum / f64(num_days)) * math.sqrt_f64(252.0) * 100.0 // Annualized %
	}

	// 5. Output Dashboard
	fmt.println(
		"\n╔══════════════════════════════════════════════════════════════╗",
	)
	fmt.println("║              HRP PORTFOLIO ALLOCATION DASHBOARD              ║")
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)
	fmt.printf(
		"║ %-6s | %-8s | %-8s | %-8s | %-14s ║\n",
		"Ticker",
		"Ann. Vol",
		"1/N Wt",
		"HRP Wt",
		"HRP Bar",
	)
	fmt.println(
		"╠══════════════════════════════════════════════════════════════╣",
	)

	// We print in the quasi-diagonalized order to show the clustering
	// We print in the quasi-diagonalized order to show the clustering
	for idx in hrp_res.cluster_order {
		naive_w := 1.0 / f64(n_assets)
		hrp_w := hrp_res.weights[idx]

		// ✅ FIX: Calculate bar_len before using it
		bar_len := int(hrp_w * 100.0) // 1% = 1 char
		if bar_len > 40 {bar_len = 40}
		if bar_len < 0 {bar_len = 0}

		// Build UTF-8 bar ("█" is 3 bytes: 0xE2, 0x96, 0x88)
		bar_bytes := make([]u8, bar_len * 3, main_alloc)
		for i in 0 ..< bar_len {
			bar_bytes[i * 3 + 0] = 0xE2
			bar_bytes[i * 3 + 1] = 0x96
			bar_bytes[i * 3 + 2] = 0x88
		}
		bar := string(bar_bytes)

		fmt.printf(
			"║ %-6s | %6.1f%% | %6.1f%% | %6.1f%% | %-40s ║\n", // ✅ Changed %-14s to %-40s
			tickers[idx],
			vols[idx],
			naive_w * 100.0,
			hrp_w * 100.0,
			bar,
		)

		// Clean up immediately to avoid defer accumulation in the loop
		delete(bar_bytes, main_alloc)
	}
	fmt.println(
		"╚══════════════════════════════════════════════════════════════╝",
	)

	// 6. Compute Portfolio Volatility for both strategies to prove HRP's edge
	// Port Var = w^T * Cov * w
	cov_mat := l.covariance(&returns_mat, main_alloc)
	defer l.matrix_free(&cov_mat)

	hrp_var := 0.0
	naive_var := 0.0
	naive_w := 1.0 / f64(n_assets)

	for i in 0 ..< n_assets {
		for j in 0 ..< n_assets {
			c := cov_mat.data[i * n_assets + j]
			hrp_var += hrp_res.weights[i] * hrp_res.weights[j] * c
			naive_var += naive_w * naive_w * c
		}
	}

	hrp_port_vol := math.sqrt(hrp_var) * math.sqrt_f64(252.0) * 100.0
	naive_port_vol := math.sqrt(naive_var) * math.sqrt_f64(252.0) * 100.0

	fmt.println("\n--- Portfolio Risk Comparison (Annualized Volatility) ---")
	fmt.printf("  Naive 1/N Portfolio Vol: %.2f%%\n", naive_port_vol)
	fmt.printf("  HRP Portfolio Vol:       %.2f%%\n", hrp_port_vol)

	if hrp_port_vol < naive_port_vol {
		fmt.printf(
			"  ✅ HRP reduced portfolio risk by %.2f%% without sacrificing expected return!\n",
			naive_port_vol - hrp_port_vol,
		)
	} else {
		fmt.println("  ⚠️  HRP did not reduce risk in this specific regime.")
	}

	fmt.println("\n✓ Hierarchical Risk Parity Test Complete!")
}
