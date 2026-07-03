// wotan/tests/backtest_test.odin
package tests

import w "../wotan/core"
import fin "../wotan/finance"
import "core:fmt"
import "core:math/rand"
import "core:mem"

test_symbol :: "TEST"

sma_strategy :: proc(ctx: ^fin.StrategyContext) {
	fin.sma_crossover_strategy(ctx, test_symbol, 10, 30, 0.95)
}

mean_reversion_strategy_wrapper :: proc(ctx: ^fin.StrategyContext) {
	fin.mean_reversion_strategy(ctx, test_symbol, 20, 2.0, 0.95)
}

momentum_strategy_wrapper :: proc(ctx: ^fin.StrategyContext) {
	fin.momentum_strategy(ctx, test_symbol, 20, 0.95)
}

backtest_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Backtest Test ===\n")

	n_days := 500
	symbol := test_symbol

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	price_col := w.column_new(symbol, .Float, n_days)

	price := 100.0
	for i in 0 ..< n_days {
		ret := rand.float64_normal(0.0003, 0.02)
		price *= (1.0 + ret)
		w.append_float(&price_col, price)
	}

	w.add_column(&df, price_col)
	df.rows = n_days

	fmt.printf("Generated %d days of synthetic price data\n", n_days)
	fmt.printf("Starting price: $%.2f\n", 100.0)
	fmt.printf("Ending price: $%.2f\n", price)

	config := fin.DEFAULT_BACKTEST_CONFIG
	config.initial_capital = 100000.0

	// Test 1: SMA Crossover
	fmt.println("\n--- Strategy 1: SMA Crossover (10/30) ---")
	result1 := fin.backtest_run(&df, []string{symbol}, sma_strategy, config, allocator)
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
		[]string{symbol},
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
		[]string{symbol},
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

	fmt.println("\n✓ Backtest completed!")
}
