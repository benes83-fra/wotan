// wotan/tests/backtest_test.odin
package tests

import w "../wotan/core"
import fin "../wotan/finance"
import "core:fmt"
import "core:math/rand"
import "core:mem"

// Define strategy at package level to avoid scope issues
test_symbol :: "TEST"

sma_strategy :: proc(ctx: ^fin.StrategyContext) {
	fin.sma_crossover_strategy(ctx, test_symbol, 10, 30, 0.95)
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
	config.commission_rate = 0.001
	config.slippage_rate = 0.0005

	fmt.println("\n--- Running SMA Crossover Strategy (10/30) ---")
	result := fin.backtest_run(&df, []string{symbol}, sma_strategy, config, allocator)
	defer {
		delete(result.equity_curve, allocator)
		delete(result.trades, allocator)
	}

	fmt.println("\n--- Backtest Results ---")
	fmt.printf("Initial Capital:    $%.2f\n", config.initial_capital)
	fmt.printf("Final Value:        $%.2f\n", result.equity_curve[len(result.equity_curve) - 1])
	fmt.printf("Total Return:       %.2f%%\n", result.total_return * 100)
	fmt.printf("Annual Return:      %.2f%%\n", result.annual_return * 100)
	fmt.printf("Sharpe Ratio:       %.3f\n", result.sharpe_ratio)
	fmt.printf("Max Drawdown:       %.2f%%\n", result.max_drawdown * 100)
	fmt.printf("Total Trades:       %d\n", result.total_trades)
	fmt.printf("Win Rate:           %.1f%%\n", result.win_rate * 100)
	fmt.printf("Profit Factor:      %.2f\n", result.profit_factor)

	fmt.println("\n--- Generating Plots ---")
	ok1 := fin.plot_equity_curve(&result, "backtest_equity.png", allocator)
	fmt.printf("Equity curve plot: %v\n", ok1)

	ok2 := fin.plot_backtest_drawdown(&result, "backtest_drawdown.png", allocator)
	fmt.printf("Drawdown plot: %v\n", ok2)

	fmt.println("\n✓ Backtest completed!")
}
