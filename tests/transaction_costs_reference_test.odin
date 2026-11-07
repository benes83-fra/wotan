package tests

import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:math"
import "core:mem"

// The transaction-cost model (ml_finance/transaction_costs.odin) against a Python
// reference (numpy for the cost formula, pandas rolling(20) for the liquidity estimates)
// on a 60-day LCG path of closes and volumes, and its linear-impact case against the
// Almgren-Chriss solver: with impact_exponent 1 and eta = almgren_chriss_eta, the
// solver's expected shortfall (gamma 0) equals the model's cost of each slice traded
// against the volume of its interval, for TWAP and for a risk-averse schedule.

@(private = "file")
_tc_rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}

transaction_costs_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Transaction Cost Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	N :: 60
	close := make([]f64, N)
	defer delete(close)
	volume := make([]f64, N)
	defer delete(volume)
	{
		x: u64 = 7
		nxt :: proc(x: ^u64) -> f64 {
			x^ = 6364136223846793005 * x^ + 1442695040888963407
			return f64(x^ >> 11) * 1.1102230246251565e-16
		}
		close[0] = 100
		for i in 0 ..< N {
			if i > 0 {close[i] = close[i - 1] * (1 + 0.03 * (nxt(&x) - 0.5))}
			volume[i] = 1e6 * (1 + nxt(&x))
		}
	}

	// A. liquidity estimates = pandas pct_change().rolling(20).std(), (close * vol).rolling(20).mean()
	sigma, adv := ml_fin.liquidity_estimates(close, volume, 20)
	defer delete(sigma)
	defer delete(adv)
	{
		nan_s, nan_a := 0, 0
		ss, sa := 0.0, 0.0
		for t in 0 ..< N {
			if math.is_nan(sigma[t]) {nan_s += 1} else {ss += sigma[t]}
			if math.is_nan(adv[t]) {nan_a += 1} else {sa += adv[t]}
		}
		err := max(
			_tc_rel(sigma[20], 0.0089709270086038781),
			_tc_rel(sigma[59], 0.0097601400753683232),
			_tc_rel(ss, 0.34596836497511929),
			_tc_rel(adv[19], 146133890.32244387),
			_tc_rel(adv[59], 140670104.28846249),
			_tc_rel(sa, 5800877638.1803045),
		)
		expect(
			&passed,
			&total,
			nan_s == 20 && nan_a == 19 && err < 1e-12,
			fmt.tprintf(
				"liquidity estimates (20-day window) = pandas rolling std of the returns and rolling mean of the traded value: NaN for the first 20 / 19 days, relative error %.1e",
				err,
			),
		)
	}
	// B. single trades: square-root, exponent 0.6, linear, and no impact without sigma
	{
		want := [4]f64{3097.3665961010279, 209949.21742868557, 105, 100}
		got := [4]f64 {
			ml_fin.trade_cost({0.0005, 0.0001, 1, 0.5}, 2e6, 0.015, 5e8),
			ml_fin.trade_cost({0, 0.0002, 0.7, 0.6}, -3e7, 0.02, 1e8),
			ml_fin.trade_cost({0.001, 0, 1, 1}, 1e5, 0.01, 2e7),
			ml_fin.trade_cost({0, 0.0001, 1, 0.5}, 1e6, 0, 1e8),
		}
		err := 0.0
		for v, i in got {err = max(err, _tc_rel(v, want[i]))}
		expect(
			&passed,
			&total,
			err < 1e-13,
			fmt.tprintf(
				"trade cost = |q| (commission + half spread + Y sigma (|q| / ADV)^b) for b 0.5, 0.6, 1 and without sigma (no impact): relative error %.1e",
				err,
			),
		)
	}
	// C. rebalancing four assets at 5e7
	{
		before := []f64{0.25, 0.25, 0.3, 0.2}
		after := []f64{0.4, 0.1, 0.3, 0.15}
		s4 := []f64{0.012, 0.02, 0.009, 0.015}
		v4 := []f64{3e9, 4e8, 1e9, 2e8}
		m := ml_fin.CostModel{0.0003, 0.0001, 0.8, 0.5}
		c := ml_fin.rebalance_cost(m, before, after, 5e7, s4, v4)
		c0 := ml_fin.rebalance_cost(m, before, after, 5e7)
		err := max(_tc_rel(c, 0.00060771557382809332), _tc_rel(c0, 0.00014000000000000001))
		expect(
			&passed,
			&total,
			err < 1e-13,
			fmt.tprintf(
				"rebalance cost of four assets (fraction of capital) with and without liquidity data: relative error %.1e",
				err,
			),
		)
	}
	// D. a position series (sin 0.3 t) at 1e9 with the estimates of A (none before day 20)
	{
		pos := make([]f64, N)
		defer delete(pos)
		for &p, t in pos {p = math.sin(0.3 * f64(t))}
		s0 := make([]f64, N)
		defer delete(s0)
		a0 := make([]f64, N)
		defer delete(a0)
		for t in 0 ..< N {
			s0[t] = 0 if math.is_nan(sigma[t]) else sigma[t]
			a0[t] = 0 if math.is_nan(adv[t]) else adv[t]
		}
		pc := ml_fin.position_costs(ml_fin.DEFAULT_COST_MODEL, pos, 1e9, s0, a0)
		defer delete(pc)
		sum := 0.0
		for v in pc {sum += v}
		gross := make([]f64, N)
		defer delete(gross)
		net := ml_fin.net_of_costs(gross, pc)
		defer delete(net)
		err := max(_tc_rel(sum, 0.085028964087970851), _tc_rel(pc[59], 0.00059768752382776487))
		expect(
			&passed,
			&total,
			err < 1e-12 && pc[0] == 0 && net[59] == -pc[59],
			fmt.tprintf(
				"position costs of a 60-day position series at 1e9 (default model) and net returns: relative error %.1e",
				err,
			),
		)
	}
	// E. linear impact = Almgren-Chriss temporary impact
	{
		S, sig, ADV :: 50.0, 0.02, 1e8
		m := ml_fin.CostModel {
			impact          = 1,
			impact_exponent = 1,
		}
		eta := ml_fin.almgren_chriss_eta(m, S, sig, ADV)
		err := 0.0
		for lambda in ([]f64{0, 1e-6}) {
			p := ml_fin.AlmgrenChrissParams {
				initial_shares = 2e5,
				n_steps        = 10,
				T              = 1,
				sigma          = sig * S,
				eta            = eta,
				gamma          = 0,
				lambda         = lambda,
			}
			traj := ml_fin.almgren_chriss_solve(p)
			defer ml_fin.almgren_chriss_free(&traj)
			tau := p.T / f64(p.n_steps)
			c := 0.0
			for n in traj.trade_sizes {c += ml_fin.trade_cost(m, n * S, sig, ADV * tau)}
			err = max(err, _tc_rel(c, traj.expected_cost))
		}
		expect(
			&passed,
			&total,
			err < 1e-12,
			fmt.tprintf(
				"impact exponent 1 with eta = Y sigma S^2 / ADV is Almgren-Chriss: summed slice costs = the solver's expected shortfall (TWAP and lambda 1e-6, 10 slices), relative error %.1e",
				err,
			),
		)
	}
	fmt.printf("Transaction cost reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
