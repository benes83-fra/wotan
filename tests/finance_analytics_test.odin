// wotan/tests/finance_analytics_test.odin
package tests

import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"

finance_analytics_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Finance Analytics Test ===\n")

	// ========================================================================
	// Test 1: Risk Decomposition
	// ========================================================================
	fmt.println("--- Test 1: Risk Decomposition ---")

	// Create a 3x3 covariance matrix
	cov := l.matrix_new(f64, 3, 3, allocator)
	defer l.matrix_free(&cov)

	// Symmetric positive definite covariance matrix
	cov.data[0] = 0.04 // Asset 1 variance (20% vol)
	cov.data[1] = 0.02 // Cov(1,2)
	cov.data[2] = 0.01 // Cov(1,3)
	cov.data[3] = 0.02 // Cov(2,1)
	cov.data[4] = 0.09 // Asset 2 variance (30% vol)
	cov.data[5] = 0.03 // Cov(2,3)
	cov.data[6] = 0.01 // Cov(3,1)
	cov.data[7] = 0.03 // Cov(3,2)
	cov.data[8] = 0.16 // Asset 3 variance (40% vol)

	// Equal weights
	weights := []f64{0.333, 0.333, 0.334}

	decomp := fin.risk_decomposition(weights, &cov, allocator)
	defer {
		delete(decomp.mctr, allocator)
		delete(decomp.ctr, allocator)
	}

	fmt.printf("Portfolio Volatility: %.4f%%\n", decomp.total_risk * 100)
	fmt.printf("Marginal Contributions to Risk (MCTR):\n")
	for i in 0 ..< len(decomp.mctr) {
		fmt.printf("  Asset %d: %.4f%%\n", i + 1, decomp.mctr[i] * 100)
	}
	fmt.printf("Component Contributions to Risk (CTR):\n")
	ctr_sum := 0.0
	for i in 0 ..< len(decomp.ctr) {
		fmt.printf("  Asset %d: %.4f%%\n", i + 1, decomp.ctr[i] * 100)
		ctr_sum += decomp.ctr[i]
	}
	fmt.printf("Sum of CTR: %.4f%% (should equal total risk)\n", ctr_sum * 100)
	fmt.printf("Verification: %.2e\n", math.abs(ctr_sum - decomp.total_risk))

	// ========================================================================
	// Test 2: Rolling Analytics
	// ========================================================================
	fmt.println("\n--- Test 2: Rolling Analytics ---")

	// Create a DataFrame with synthetic return data
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	// Generate 100 days of synthetic returns
	n_days := 100
	returns_col := w.column_new("returns", .Float, n_days)
	price_col := w.column_new("price", .Float, n_days)
	benchmark_col := w.column_new("benchmark", .Float, n_days)

	// Generate random returns with mean ~0.001 (10% annual) and vol ~0.02 (20% annual)
	price := 100.0
	benchmark_price := 100.0
	for i in 0 ..< n_days {
		ret := rand.float64_normal(0.001, 0.02)
		bench_ret := rand.float64_normal(0.0005, 0.015)

		w.append_float(&returns_col, ret)
		w.append_float(&benchmark_col, bench_ret)

		price *= (1.0 + ret)
		benchmark_price *= (1.0 + bench_ret)

		w.append_float(&price_col, price)
	}

	w.add_column(&df, returns_col)
	w.add_column(&df, price_col)
	w.add_column(&df, benchmark_col)
	df.rows = n_days

	// Test rolling volatility
	fmt.println("Rolling Volatility (20-day window):")
	rolling_vol := fin.rolling_volatility(&df, "returns", 20, 10, 252.0, allocator)
	defer w.destroy_column(&rolling_vol)

	// Print last 5 values
	fmt.printf("  Last 5 values: ")
	for i := max(0, rolling_vol.len - 5); i < rolling_vol.len; i += 1 {
		v, is_null := w.column_at_float(&rolling_vol, i)
		if !is_null {
			fmt.printf("%.2f%% ", v * 100)
		}
	}
	fmt.println()

	// Test rolling Sharpe
	fmt.println("\nRolling Sharpe Ratio (20-day window):")
	rolling_sharpe := fin.rolling_sharpe(&df, "returns", 20, 0.0, 252.0, allocator)
	defer w.destroy_column(&rolling_sharpe)

	fmt.printf("  Last 5 values: ")
	for i := max(0, rolling_sharpe.len - 5); i < rolling_sharpe.len; i += 1 {
		v, is_null := w.column_at_float(&rolling_sharpe, i)
		if !is_null {
			fmt.printf("%.2f ", v)
		}
	}
	fmt.println()

	// Test EWMA volatility
	fmt.println("\nEWMA Volatility (alpha=0.06):")
	ewma_vol := fin.ewma_rolling_volatility(&df, "returns", 0.06, 10, false, 252.0, allocator)
	defer w.destroy_column(&ewma_vol)

	fmt.printf("  Last 5 values: ")
	for i := max(0, ewma_vol.len - 5); i < ewma_vol.len; i += 1 {
		v, is_null := w.column_at_float(&ewma_vol, i)
		if !is_null {
			fmt.printf("%.2f%% ", v * 100)
		}
	}
	fmt.println()

	// Test rolling beta
	fmt.println("\nRolling Beta (20-day window):")
	rolling_beta := fin.rolling_beta(&df, "returns", "benchmark", 20, allocator)
	defer w.destroy_column(&rolling_beta)

	fmt.printf("  Last 5 values: ")
	for i := max(0, rolling_beta.len - 5); i < rolling_beta.len; i += 1 {
		v, is_null := w.column_at_float(&rolling_beta, i)
		if !is_null {
			fmt.printf("%.3f ", v)
		}
	}
	fmt.println()

	// ========================================================================
	// Test 3: Visualizations
	// ========================================================================
	fmt.println("\n--- Test 3: Visualizations ---")

	// Test cumulative returns plot
	fmt.println("Generating cumulative returns plot...")
	ok1 := fin.plot_cumulative_returns(
		&df,
		[]string{"price", "benchmark"},
		"cumulative_returns.png",
		allocator,
	)
	fmt.printf("  cumulative_returns.png: %v\n", ok1)

	// Test drawdown plot
	fmt.println("Generating drawdown plot...")
	ok2 := fin.plot_drawdown(&df, "price", "drawdown.png", allocator)
	fmt.printf("  drawdown.png: %v\n", ok2)

	// Test risk decomposition plot
	fmt.println("Generating risk decomposition plot...")
	asset_names := []string{"Asset 1", "Asset 2", "Asset 3"}
	ok3 := fin.plot_risk_decomposition(decomp, asset_names, "risk_decomposition.png", allocator)
	fmt.printf("  risk_decomposition.png: %v\n", ok3)

	// Verify files exist
	fmt.println("\nVerifying generated files:")
	files := []string{"cumulative_returns.png", "drawdown.png", "risk_decomposition.png"}
	for file in files {
		_, err := os.stat(file, allocator)
		if err == nil {
			fmt.printf("  ✓ %s exists\n", file)
		} else {
			fmt.printf("  ✗ %s missing\n", file)
		}
	}

	fmt.println("\n✓ Finance analytics test completed!")
}
