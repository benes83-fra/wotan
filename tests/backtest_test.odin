// wotan/tests/backtest_test.odin
package tests

import w "../wotan/core"
import fin "../wotan/finance"
import "core:fmt"
import "core:math/rand"
import "core:mem"

test_symbol :: "TEST"
test_symbol2 :: "TEST2"

sma_strategy :: proc(ctx: ^fin.StrategyContext) {
	fin.sma_crossover_strategy(ctx, test_symbol, 10, 30, 0.95)
}

mean_reversion_strategy_wrapper :: proc(ctx: ^fin.StrategyContext) {
	fin.mean_reversion_strategy(ctx, test_symbol, 20, 2.0, 0.95)
}

momentum_strategy_wrapper :: proc(ctx: ^fin.StrategyContext) {
	fin.momentum_strategy(ctx, test_symbol, 20, 0.95)
}

pairs_trading_wrapper :: proc(ctx: ^fin.StrategyContext) {
	fin.pairs_trading_strategy(ctx, test_symbol, test_symbol2, 50, 2.0, 0.5, 0.95)
}
// Wrapper procs to match the PortfolioStrategyFn signature
eq_weight_wrapper :: proc(ctx: ^fin.StrategyContext, allocator: mem.Allocator) -> map[string]f64 {
	return fin.portfolio_equal_weight(ctx, allocator)
}

inv_vol_wrapper :: proc(ctx: ^fin.StrategyContext, allocator: mem.Allocator) -> map[string]f64 {
	return fin.portfolio_inverse_volatility(ctx, allocator, 60)
}

min_var_wrapper :: proc(ctx: ^fin.StrategyContext, allocator: mem.Allocator) -> map[string]f64 {
	return fin.portfolio_min_variance(ctx, allocator, 120)
}
backtest_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Backtest Test ===\n")

	n_days := 500

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	// Generate two correlated assets
	price_col1 := w.column_new(test_symbol, .Float, n_days)
	price_col2 := w.column_new(test_symbol2, .Float, n_days)

	price1 := 100.0
	price2 := 100.0

	for i in 0 ..< n_days {
		// Common market factor
		market_factor := rand.float64_normal(0.0003, 0.015)

		// Asset-specific noise
		noise1 := rand.float64_normal(0.0, 0.01)
		noise2 := rand.float64_normal(0.0, 0.01)

		// Prices are correlated through market factor
		ret1 := market_factor + noise1
		ret2 := market_factor + noise2

		price1 *= (1.0 + ret1)
		price2 *= (1.0 + ret2)

		w.append_float(&price_col1, price1)
		w.append_float(&price_col2, price2)
	}

	w.add_column(&df, price_col1)
	w.add_column(&df, price_col2)
	df.rows = n_days

	fmt.printf("Generated %d days of correlated price data\n", n_days)
	fmt.printf("Asset 1: $%.2f → $%.2f\n", 100.0, price1)
	fmt.printf("Asset 2: $%.2f → $%.2f\n", 100.0, price2)

	config := fin.DEFAULT_BACKTEST_CONFIG
	config.initial_capital = 100000.0

	// Test 1: SMA Crossover
	fmt.println("\n--- Strategy 1: SMA Crossover (10/30) ---")
	result1 := fin.backtest_run(&df, []string{test_symbol}, sma_strategy, config, allocator)
	defer {
		delete(result1.equity_curve, allocator)
		delete(result1.trades, allocator)
	}
	fmt.printf(
		"Total Return: %.2f%%, Sharpe: %.3f, Win Rate: %.1f%%\n",
		result1.total_return * 100,
		result1.sharpe_ratio,
		result1.win_rate * 100,
	)

	// Test 2: Mean Reversion
	fmt.println("\n--- Strategy 2: Mean Reversion (Bollinger Bands) ---")
	result2 := fin.backtest_run(
		&df,
		[]string{test_symbol},
		mean_reversion_strategy_wrapper,
		config,
		allocator,
	)
	defer {
		delete(result2.equity_curve, allocator)
		delete(result2.trades, allocator)
	}
	fmt.printf(
		"Total Return: %.2f%%, Sharpe: %.3f, Win Rate: %.1f%%\n",
		result2.total_return * 100,
		result2.sharpe_ratio,
		result2.win_rate * 100,
	)

	// Test 3: Momentum
	fmt.println("\n--- Strategy 3: Momentum (Breakout) ---")
	result3 := fin.backtest_run(
		&df,
		[]string{test_symbol},
		momentum_strategy_wrapper,
		config,
		allocator,
	)
	defer {
		delete(result3.equity_curve, allocator)
		delete(result3.trades, allocator)
	}
	fmt.printf(
		"Total Return: %.2f%%, Sharpe: %.3f, Win Rate: %.1f%%\n",
		result3.total_return * 100,
		result3.sharpe_ratio,
		result3.win_rate * 100,
	)

	// Test 4: Pairs Trading
	fmt.println("\n--- Strategy 4: Pairs Trading (Statistical Arbitrage) ---")
	result4 := fin.backtest_run(
		&df,
		[]string{test_symbol, test_symbol2},
		pairs_trading_wrapper,
		config,
		allocator,
	)
	defer {
		delete(result4.equity_curve, allocator)
		delete(result4.trades, allocator)
	}
	fmt.printf(
		"Total Return: %.2f%%, Sharpe: %.3f, Win Rate: %.1f%%\n",
		result4.total_return * 100,
		result4.sharpe_ratio,
		result4.win_rate * 100,
	)
	fmt.printf("Total Trades: %d\n", result4.total_trades)

	fmt.println("\n✓ Backtest completed!")
}


backtest_portfolio_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Portfolio Backtest Test ===\n")

	n_days := 500
	symbols := []string{"ASSET_A", "ASSET_B", "ASSET_C"}

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	// Generate 3 correlated assets
	prices := [3]f64{100.0, 100.0, 100.0}
	cols := make([]w.Column, 3)
	for i in 0 ..< 3 {
		cols[i] = w.column_new(symbols[i], .Float, n_days)
	}

	for i in 0 ..< n_days {
		market := rand.float64_normal(0.0003, 0.012) // Common market factor

		// Asset specific noise (Asset C is more volatile)
		n1 := rand.float64_normal(0.0, 0.008)
		n2 := rand.float64_normal(0.0, 0.010)
		n3 := rand.float64_normal(0.0, 0.020)

		prices[0] *= (1.0 + market + n1)
		prices[1] *= (1.0 + market + n2)
		prices[2] *= (1.0 + market + n3)

		w.append_float(&cols[0], prices[0])
		w.append_float(&cols[1], prices[1])
		w.append_float(&cols[2], prices[2])
	}

	for c in cols {w.add_column(&df, c)}
	df.rows = n_days

	fmt.printf("Generated %d days for 3 assets\n", n_days)
	fmt.printf("Final Prices: A=$%.2f, B=$%.2f, C=$%.2f\n\n", prices[0], prices[1], prices[2])

	config := fin.DEFAULT_BACKTEST_CONFIG
	config.initial_capital = 100000.0
	config.commission_rate = 0.001 // 10 bps

	// Test 1: Equal Weight
	fmt.println("--- Strategy 1: Equal Weight (Monthly Rebalance) ---")
	r1 := fin.backtest_portfolio_run(&df, symbols, eq_weight_wrapper, 21, config, allocator)
	fmt.printf(
		"Return: %6.2f%% | Sharpe: %5.3f | Max DD: %6.2f%% | Trades: %d\n\n",
		r1.total_return * 100,
		r1.sharpe_ratio,
		r1.max_drawdown * 100,
		r1.total_trades,
	)
	defer {delete(r1.equity_curve, allocator); delete(r1.trades, allocator)}

	// Test 2: Inverse Volatility
	fmt.println("--- Strategy 2: Inverse Volatility (Risk Parity) ---")
	r2 := fin.backtest_portfolio_run(&df, symbols, inv_vol_wrapper, 21, config, allocator)
	fmt.printf(
		"Return: %6.2f%% | Sharpe: %5.3f | Max DD: %6.2f%% | Trades: %d\n\n",
		r2.total_return * 100,
		r2.sharpe_ratio,
		r2.max_drawdown * 100,
		r2.total_trades,
	)
	defer {delete(r2.equity_curve, allocator); delete(r2.trades, allocator)}

	// Test 3: Minimum Variance
	fmt.println("--- Strategy 3: Rolling Minimum Variance (Markowitz) ---")
	r3 := fin.backtest_portfolio_run(&df, symbols, min_var_wrapper, 21, config, allocator)
	fmt.printf(
		"Return: %6.2f%% | Sharpe: %5.3f | Max DD: %6.2f%% | Trades: %d\n\n",
		r3.total_return * 100,
		r3.sharpe_ratio,
		r3.max_drawdown * 100,
		r3.total_trades,
	)
	defer {delete(r3.equity_curve, allocator); delete(r3.trades, allocator)}

	fmt.println("✓ Portfolio Backtest completed!")
}
