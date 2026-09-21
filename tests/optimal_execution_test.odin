package tests

import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:mem"

optimal_execution_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Optimal Execution & Market Making ===")
	main_alloc := context.allocator

	// 1. Almgren-Chriss Liquidation
	fmt.println("\n--- Almgren-Chriss Optimal Liquidation ---")
	ac_params := ml_fin.AlmgrenChrissParams {
		initial_shares = 100000.0, // 100k shares to sell
		n_steps        = 10, // 10 trading intervals
		T              = 1.0, // 1 day horizon
		sigma          = 2.0, // $2.00 daily vol (on a ~$100 stock)
		eta            = 1e-6, // Temporary impact
		gamma          = 1e-7, // Permanent impact
		lambda         = 1e-6, // Risk aversion
	}

	traj := ml_fin.almgren_chriss_solve(ac_params, main_alloc)
	defer ml_fin.almgren_chriss_free(&traj)

	fmt.println("  Interval | Time (t) | Shares Held | Trade Size")
	fmt.println("  ---------|----------|-------------|-----------")
	for j in 0 ..< ac_params.n_steps {
		fmt.printf(
			"  %8d | %8.3f | %11.0f | %10.0f\n",
			j,
			traj.times[j],
			traj.shares_held[j],
			traj.trade_sizes[j],
		)
	}
	fmt.printf(
		"  %8d | %8.3f | %11.0f | %10s\n",
		ac_params.n_steps,
		traj.times[ac_params.n_steps],
		traj.shares_held[ac_params.n_steps],
		"-",
	)

	fmt.printf("\n  Expected Implementation Shortfall: $%.2f\n", traj.expected_cost)
	fmt.printf("  Variance of Cost:                  %.2f\n", traj.variance)

	// 2. Avellaneda-Stoikov Market Making
	fmt.println("\n--- Avellaneda-Stoikov Market Making ---")
	as_params := ml_fin.AvellanedaStoikovParams {
		sigma = 2.0, // $2.00 daily vol
		gamma = 0.1, // Risk aversion
		k     = 1.5, // Order book density
		T     = 1.0, // 1 day horizon
	}

	mid_price := 100.0

	fmt.println("\n  Inventory | Mid Price | Res. Price | Bid Quote | Ask Quote | Spread")
	fmt.println("  ----------|-----------|------------|-----------|-----------|-------")

	inventories := []f64{-100, -50, 0, 50, 100}
	current_time := 0.9 // 90% of the day has passed (T-t = 0.1)

	for q in inventories {
		quotes := ml_fin.avellaneda_stoikov_quote(mid_price, q, current_time, as_params)
		fmt.printf(
			"  %9.0f | %9.2f | %10.4f | %9.4f | %9.4f | %6.4f\n",
			q,
			mid_price,
			quotes.reservation_price,
			quotes.bid_quote,
			quotes.ask_quote,
			quotes.optimal_spread,
		)
	}

	fmt.println("\n  Interpretation:")
	fmt.println(
		"  - Long Inventory (+100): Quotes shift DOWN to encourage selling (hitting bids).",
	)
	fmt.println(
		"  - Short Inventory (-100): Quotes shift UP to encourage buying (lifting offers).",
	)
	fmt.println("  - Flat Inventory (0): Quotes are symmetric around the mid-price.")

	fmt.println("\n✓ Optimal Execution Test Complete!")
}
