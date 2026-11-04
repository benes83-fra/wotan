package tests

import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:math"
import "core:mem"

// Almgren-Chriss against an independent minimiser of E + lambda V (E evaluated from the
// price model itself, the quadratic solved exactly; mpmath for the closed form), and
// Avellaneda-Stoikov against the closed form of the 2008 paper.
optimal_execution_reference_checks :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Optimal Execution Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}
	ac_err :: proc(tr: ml_fin.AlmgrenChrissTrajectory, x: []f64, E, V: f64) -> f64 {
		if len(tr.shares_held) != len(x) {return math.inf_f64(1)}
		e := max(rel(tr.expected_cost, E), rel(tr.variance, V))
		for v, j in x {e = max(e, abs(tr.shares_held[j] - v) / x[0])}
		return e
	}
	ac :: proc(
		X: f64,
		N: int,
		T, sigma, eta, gamma, lambda: f64,
	) -> ml_fin.AlmgrenChrissTrajectory {
		return ml_fin.almgren_chriss_solve({X, N, T, sigma, eta, gamma, lambda})
	}

	{ 	// the example of the test below
		tr := ac(100000, 10, 1, 2, 1e-6, 1e-7, 1e-6)
		defer ml_fin.almgren_chriss_free(&tr)
		x := []f64 {
			100000.0,
			81110.00386240843,
			65480.7113976775,
			52483.80934089341,
			41596.809167160296,
			32382.042527785397,
			24469.06654279383,
			17539.771624849247,
			11315.593154134787,
			5546.312900671977,
			0.0,
		}
		err := ac_err(tr, x, 12309.473842044754, 6986072314.858859)
		expect(
			&passed,
			&total,
			err < 1e-12,
			fmt.tprintf("AC trajectory, E and V (max rel err %.1e)", err),
		)
	}
	{ 	// Almgren & Chriss (2000) example: X = 1e6, T = 5 days, N = 5; eta~ = eta - gamma tau / 2
		tr := ac(1e6, 5, 5, 0.95, 2.5e-6, 2.5e-7, 2e-6)
		defer ml_fin.almgren_chriss_free(&tr)
		x := []f64 {
			1000000.0,
			428598.84574701736,
			182932.81426176775,
			76295.72161546166,
			27643.3773969064,
			0.0,
		}
		err := ac_err(tr, x, 1078215.1670497854, 201931287150.52435)
		expect(
			&passed,
			&total,
			err < 1e-12,
			fmt.tprintf("AC paper example uses eta~ (max rel err %.1e)", err),
		)
	}
	{ 	// lambda = 0: TWAP (sinh(0) / sinh(0) before)
		tr := ac(1e6, 5, 5, 0.95, 2.5e-6, 2.5e-7, 0)
		defer ml_fin.almgren_chriss_free(&tr)
		err := ac_err(tr, {1e6, 8e5, 6e5, 4e5, 2e5, 0}, 600000.0, 1082999999999.9999)
		expect(
			&passed,
			&total,
			err < 1e-14,
			fmt.tprintf("AC lambda = 0 is TWAP (max rel err %.1e)", err),
		)
	}
	{ 	// tiny lambda: kappa ~ 6e-6, no cancellation in acosh or sinh
		tr := ac(1e6, 5, 5, 0.95, 2.5e-6, 2.5e-7, 1e-16)
		defer ml_fin.almgren_chriss_free(&tr)
		x := []f64{1000000.0, 799999.9999544, 599999.9999392, 399999.9999468, 199999.9999696, 0.0}
		err := ac_err(tr, x, 600000.0, 1082999999818.9222)
		expect(
			&passed,
			&total,
			err < 1e-12,
			fmt.tprintf("AC lambda = 1e-16 (max rel err %.1e)", err),
		)
	}
	{ 	// kappa T = 1820: sinh overflows in f64 (NaN before)
		tr := ac(1e6, 500, 5, 0.95, 2.5e-6, 2.5e-7, 1)
		defer ml_fin.almgren_chriss_free(&tr)
		ok :=
			len(tr.shares_held) == 501 &&
			rel(tr.shares_held[1], 26252.36468301332) < 1e-12 &&
			rel(tr.shares_held[2], 689.1866514499249) < 1e-12 &&
			rel(tr.shares_held[3], 18.09277930852822) < 1e-11 &&
			tr.shares_held[250] == 0 &&
			rel(tr.expected_cost, 237215991.21051648) < 1e-12 &&
			rel(tr.variance, 6224199.164315584) < 1e-12
		expect(&passed, &total, ok, "AC kappa T = 1820, no overflow")
	}
	{ 	// invalid input: empty trajectory, NaN costs
		tr0 := ac(1e6, 0, 5, 0.95, 2.5e-6, 2.5e-7, 1e-6)
		tr1 := ac(1e6, 5, 5, 0.95, 1e-7, 2.5e-7, 1e-6) // eta~ < 0
		ok :=
			tr0.shares_held == nil &&
			math.is_nan(tr0.expected_cost) &&
			tr1.shares_held == nil &&
			math.is_nan(tr1.variance)
		expect(&passed, &total, ok, "AC N = 0 and eta~ <= 0 rejected")
	}

	{ 	// Avellaneda-Stoikov: spread = gamma sigma^2 (T-t) + (2/gamma) ln(1 + gamma/k)
		p := ml_fin.AvellanedaStoikovParams {
			sigma = 2.0,
			gamma = 0.1,
			k     = 1.5,
			T     = 1.0,
		}
		ref := [3][4]f64 {
			{-100, 104.0, 103.3346147886243, 104.6653852113757},
			{0, 100.0, 99.3346147886243, 100.6653852113757},
			{50, 98.0, 97.3346147886243, 98.6653852113757},
		}
		err := 0.0
		for c in ref {
			qt := ml_fin.avellaneda_stoikov_quote(100.0, c[0], 0.9, p)
			err = max(
				err,
				rel(qt.reservation_price, c[1]),
				rel(qt.bid_quote, c[2]),
				rel(qt.ask_quote, c[3]),
			)
		}
		past := ml_fin.avellaneda_stoikov_quote(100.0, 10, 1.2, p) // T - t = 0
		err = max(
			err,
			rel(past.bid_quote, 99.35461478862429),
			rel(past.ask_quote, 100.64538521137571),
		)
		expect(
			&passed,
			&total,
			err < 1e-15,
			fmt.tprintf("AS quotes incl. the inventory-risk spread (max rel err %.1e)", err),
		)
		p.gamma = 1e-12 // log1p: the half-spread tends to 1/k
		lo := ml_fin.avellaneda_stoikov_quote(100.0, 10, 0.5, p)
		p.gamma = 0
		z := ml_fin.avellaneda_stoikov_quote(100.0, 10, 0.5, p)
		ok :=
			rel(lo.bid_quote, 99.33333333331257) < 1e-15 &&
			rel(lo.ask_quote, 100.66666666664744) < 1e-15 &&
			z.reservation_price == 100 &&
			rel(z.optimal_spread, 2.0 / 1.5) < 1e-13
		expect(&passed, &total, ok, "AS gamma -> 0 and gamma = 0")
	}
	fmt.printf("Optimal execution reference checks: %d/%d passed\n", passed, total)
}

optimal_execution_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	optimal_execution_reference_checks(allocator)
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
