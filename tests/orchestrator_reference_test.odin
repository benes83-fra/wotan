package tests

import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:mem"

// Trading orchestrator rules against hand-computed values: action, confidence and
// multiplier; the allocation (equity fraction over the equities by HRP weight, the
// bond the rest) scaled by the exposure, which now matters (it used to be renormalised
// away); trades from the weights actually held (they used to be from the initial equal
// weights every day); Liquidate flattens; the P&L and the weight drift of
// mark-to-market; the quotes' inventory in shares. And hmm_filter, the causal regime
// probabilities, against numpy's forward recursion.

@(private = "file")
HMM_FILTER_REF := [16]f64 {
	0.8187596148998378,
	0.1812403851001622,
	0.9198501308078778,
	0.0801498691921222,
	0.9079996698750745,
	0.09200033012492537,
	0.01007871650687672,
	0.9899212834931234,
	0.004603630068070132,
	0.9953963699319298,
	0.24812301672066175,
	0.7518769832793383,
	0.5763282781766478,
	0.4236717218233522,
	0.8048798725196753,
	0.1951201274803247,
}

orchestrator_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Orchestrator Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	close :: proc(a, b: []f64, tol: f64) -> bool {
		if len(a) != len(b) {return false}
		for v, i in a {if abs(v - b[i]) > tol {return false}}
		return true
	}

	CAP :: 1_000_000.0
	prices := []f64{100, 50, 200}
	hrp := []f64{0.3, 0.2, 0.5}
	asp := ml_fin.AvellanedaStoikovParams {
		sigma = 2,
		gamma = 1e-4,
		k     = 1.5,
		T     = 1,
	}
	state := ml_fin.orchestrator_init(3)
	defer ml_fin.orchestrator_free(&state)

	// A. Aggressive (mean 20%, vol 10%), VRP +0.5, interval ±15%: confidence
	// 1 - 0.06 / 0.4 = 0.85, multiplier 1.5 · 1.15 · 0.925 = 1.596, exposure capped at 1;
	// 85% equities split 0.3 : 0.2, 15% bond; trades from the initial 1/3 each
	{
		d := ml_fin.orchestrator_step(
			&state,
			0,
			0.20,
			0.10,
			0.20,
			0.17,
			0.23,
			0.5,
			hrp,
			prices,
			CAP,
			asp,
		)
		defer ml_fin.decision_free(&d, context.allocator)
		w := []f64{0.51, 0.34, 0.15}
		trades := []f64 {
			(0.51 - 1.0 / 3) * CAP / 100,
			(0.34 - 1.0 / 3) * CAP / 50,
			(0.15 - 1.0 / 3) * CAP / 200,
		}
		mult := 1.5 * 1.15 * (0.5 + 0.5 * (1 - 0.06 / (0.4 + 1e-8)))
		expect(
			&passed,
			&total,
			d.action == .Aggressive &&
			abs(d.position_mult - mult) < 1e-12 &&
			d.exposure == 1 &&
			close(d.target_weights, w, 1e-12) &&
			close(d.trades, trades, 1e-6),
			fmt.tprintf(
				"aggressive: multiplier %.4f capped at exposure %.0f%%, weights %.4f, trades %.1f shares",
				d.position_mult,
				100 * d.exposure,
				d.target_weights,
				d.trades,
			),
		)
	}
	// B. Reduce (vol 27%), no signal, zero-width interval: multiplier 0.5 halves the
	// 40% / 60% allocation, trades from step A's weights
	{
		d := ml_fin.orchestrator_step(
			&state,
			1,
			0.05,
			0.27,
			0.20,
			0.20,
			0.20,
			0,
			hrp,
			prices,
			CAP,
			asp,
		)
		defer ml_fin.decision_free(&d, context.allocator)
		w := []f64{0.12, 0.08, 0.30}
		trades := []f64 {
			(0.12 - 0.51) * CAP / 100,
			(0.08 - 0.34) * CAP / 50,
			(0.30 - 0.15) * CAP / 200,
		}
		bid_ok := true
		for i in 0 ..< 3 {
			q := ml_fin.avellaneda_stoikov_quote(prices[i], w[i] * CAP / prices[i], 0.5, asp)
			bid_ok &&=
				abs(d.bid_quotes[i] - q.bid_quote) < 1e-12 &&
				abs(d.ask_quotes[i] - q.ask_quote) < 1e-12
		}
		expect(
			&passed,
			&total,
			d.action == .Reduce &&
			d.exposure == 0.5 &&
			close(d.target_weights, w, 1e-12) &&
			close(d.trades, trades, 1e-6) &&
			bid_ok,
			fmt.tprintf(
				"reduce: exposure 50%% (cash 50%%), weights %.4f, trades from the weights held %.1f; quotes with the shares held as inventory %v",
				d.target_weights,
				d.trades,
				bid_ok,
			),
		)
	}
	// C. mark-to-market of B's weights with returns +10%, -5%, +2%
	{
		r := ml_fin.orchestrator_mark_to_market(&state, []f64{0.10, -0.05, 0.02}, CAP)
		rp := 0.12 * 0.10 - 0.08 * 0.05 + 0.30 * 0.02
		drift := []f64{0.12 * 1.10 / (1 + rp), 0.08 * 0.95 / (1 + rp), 0.30 * 1.02 / (1 + rp)}
		expect(
			&passed,
			&total,
			abs(r - rp) < 1e-15 &&
			abs(state.daily_pnl - CAP * rp) < 1e-8 &&
			abs(state.cumulative_pnl - CAP * rp) < 1e-8 &&
			close(state.current_weights, drift, 1e-15),
			fmt.tprintf(
				"mark-to-market: portfolio return %.4f, P&L %.0f, weights drift to %.4f",
				r,
				state.daily_pnl,
				state.current_weights,
			),
		)
	}
	// D. Liquidate (vol 35%, mean -10%): everything sold
	{
		held := make([]f64, 3)
		copy(held, state.current_weights)
		d := ml_fin.orchestrator_step(
			&state,
			2,
			-0.10,
			0.35,
			0.30,
			0.25,
			0.35,
			0,
			hrp,
			prices,
			CAP,
			asp,
		)
		defer ml_fin.decision_free(&d, context.allocator)
		trades := []f64{-held[0] * CAP / 100, -held[1] * CAP / 50, -held[2] * CAP / 200}
		expect(
			&passed,
			&total,
			d.action == .Liquidate &&
			d.exposure == 0 &&
			close(d.target_weights, {0, 0, 0}, 0) &&
			close(d.trades, trades, 1e-6),
			fmt.tprintf(
				"liquidate: weights %.2f, trades %.1f (all held shares sold)",
				d.target_weights,
				d.trades,
			),
		)
	}

	// E. hmm_filter vs numpy; causal: filtering a prefix gives the same rows
	{
		hmm := ml_fin.hmm_new(2)
		defer ml_fin.hmm_free(&hmm)
		copy(hmm.transition, []f64{0.95, 0.05, 0.10, 0.90})
		copy(hmm.emission_mu, []f64{0.001, -0.002})
		copy(hmm.emission_var, []f64{0.0001, 0.0009})
		copy(hmm.initial, []f64{0.6, 0.4})
		obs := []f64{0.002, -0.001, 0.015, -0.04, -0.03, 0.005, 0.001, -0.002}
		f := ml_fin.hmm_filter(&hmm, obs)
		prefix := ml_fin.hmm_filter(&hmm, obs[:5])
		err := 0.0
		for v, i in HMM_FILTER_REF {err = max(err, abs(f[i] - v))}
		same := true
		for v, i in prefix {same &&= v == f[i]}
		expect(
			&passed,
			&total,
			err < 1e-14 && same,
			fmt.tprintf(
				"hmm_filter vs numpy's forward recursion: max err %.1e; the first 5 rows do not depend on later observations %v",
				err,
				same,
			),
		)
	}
	fmt.printf("Orchestrator reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
