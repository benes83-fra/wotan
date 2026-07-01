package tests

import w "../wotan/core"
import fin "../wotan/finance"
import "core:fmt"
import "core:math/rand"
import "core:mem"

risk_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== RISK METRICS TEST ===\n")

	// 1. Generate synthetic daily returns (e.g., 252 days = 1 year)
	n_days := 252
	returns := make([]f64, n_days, allocator)
	defer delete(returns, allocator)

	// Simulate returns with mean ~0.0004 (approx 10% annualized) and std ~0.015 (approx 24% annualized)
	mean_daily := 0.0004
	std_daily := 0.015

	for i in 0 ..< n_days {
		returns[i] = rand.float64_normal(mean_daily, std_daily)
	}

	// 2. Test VaR and CVaR
	fmt.println("--- Value at Risk (VaR) & Conditional VaR (CVaR) ---")
	var_95 := fin.var_historical(returns, 0.95)
	var_99 := fin.var_historical(returns, 0.99)
	cvar_95 := fin.conditional_var(returns, 0.95)

	fmt.printf("Historical VaR (95%%):  %.4f%%\n", var_95 * 100)
	fmt.printf("Historical VaR (99%%):  %.4f%%\n", var_99 * 100)
	fmt.printf("Historical CVaR (95%%): %.4f%%\n", cvar_95 * 100)

	// 3. Test Volatility
	fmt.println("\n--- Volatility ---")
	hist_vol := fin.historical_volatility(returns, 252.0)
	fmt.printf("Annualized Historical Volatility: %.4f%%\n", hist_vol * 100)

	// 4. Test Max Drawdown
	fmt.println("\n--- Maximum Drawdown ---")
	max_dd := fin.max_drawdown(returns)
	fmt.printf("Maximum Drawdown: %.4f%%\n", max_dd * 100)

	// 5. Test Sharpe Ratio
	fmt.println("\n--- Sharpe Ratio ---")
	// Assume risk-free rate is 4% annualized
	rf_annual := 0.04
	sharpe := fin.sharpe_ratio_from_returns(returns, rf_annual, 252.0)
	fmt.printf("Annualized Sharpe Ratio (Rf=%.2f%%): %.4f\n", rf_annual * 100, sharpe)

	// 6. Test DataFrame Integration
	fmt.println("\n--- DataFrame Interface Test ---")
	df := w.dataframe_new()
	col := w.column_from_floats("returns", returns)
	w.add_column(&df, col)

	fmt.println("DataFrame Head (first 5 rows):")
	w.df_head(&df, 5)
	w.destroy_dataframe(&df)

	fmt.println("\n=== RISK METRICS TEST COMPLETE ===\n")
}
