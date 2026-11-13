package tests

import ts "../wotan/analytics"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"

// The closed-loop orchestrator walked forward over the last year of SPY, QQQ, AAPL, MSFT
// and TLT (dates aligned across the five and the VIX), every input known at the
// decision's close:
//   regime      argmax of the HMM's filtered probabilities (hmm_filter; the old test
//               used the Viterbi path of the whole sample, which looks ahead)
//   vol         GARCH(1,1) fitted before the backtest, its one-day forecast updated
//               every day (the old recursion skipped the 15 days before the window)
//   interval    the forecast times the 10% / 90% quantiles of realised 21-day vol over
//               the forecast in the fitting period (was a fixed ±15%)
//   VRP signal  (VIX - forecast vol) / 10 points, clipped to [-1, 1] (was a vol threshold)
//   weights     HRP on the trailing 252 days
// Each decision earns the next day's returns. Reported next to 60/40 SPY/TLT, equal
// weights and SPY, each with its PSR against 0 and the minimum track record length (how
// many years of such returns a 95% PSR would take); the checks are on the accounting (P&L reconciles with the weights
// held, exposure within [0, 1], no short weights), not on beating the benchmarks.

@(private = "file")
_orch_metrics :: proc(name: string, r: []f64) {
	n := f64(len(r))
	m, m2, eq, peak, mdd := 0.0, 0.0, 1.0, 1.0, 0.0
	for x in r {
		m += x / n
		m2 += x * x / n
		eq *= 1 + x
		peak = max(peak, eq)
		mdd = max(mdd, 1 - eq / peak)
	}
	vol := math.sqrt(max(m2 - m * m, 0) * 252)
	// PSR against 0 and the years of returns the PSR would need to reach 95%
	s := ml_fin.sharpe_stats(r)
	psr := ml_fin.probabilistic_sharpe_ratio(s.sharpe, 0, s.n, s.skewness, s.kurtosis)
	trl := ml_fin.min_track_record_length(s.sharpe, 0, s.skewness, s.kurtosis) / 252
	fmt.printf(
		"   %-22s return %+6.1f%%  vol %5.1f%%  Sharpe %5.2f  max drawdown %5.1f%%  PSR %.2f  95%% needs %s\n",
		name,
		100 * (eq - 1),
		100 * vol,
		m * 252 / vol if vol > 0 else 0,
		100 * mdd,
		psr,
		fmt.tprintf("%.1f years", trl) if !math.is_inf(trl) else "a positive SR",
	)
}

orchestrator_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Closed-loop orchestrator, walked forward over the last year ===")
	tickers := []string{"SPY", "QQQ", "AAPL", "MSFT", "TLT"}
	N := len(tickers)
	CAP :: 1_000_000.0

	// prices on the dates present for every asset and the VIX
	symbols := []string{"SPY", "QQQ", "AAPL", "MSFT", "TLT", "^VIX"}
	NS := len(symbols)
	panel, panel_ok, failed := yahoo.read_yahoo_panel(symbols, .Daily, .FiveYears, allocator)
	defer yahoo.yahoo_panel_free(&panel)
	if !panel_ok {
		what := "Yahoo data shared by the assets and the VIX"
		if failed >= 0 {what = fmt.tprintf("Yahoo data for %s", symbols[failed])}
		fail_missing_data(#procedure, what)
		return
	}
	if len(panel.dates) < 600 {
		fail_missing_data(
			#procedure,
			fmt.tprintf("600 common days of Yahoo data (got %d)", len(panel.dates)),
		)
		return
	}
	T := len(panel.dates)
	price := make([]f64, T * N)
	vix := make([]f64, T)
	for t in 0 ..< T {
		for i in 0 ..< N {price[t * N + i] = panel.adj_close[t * NS + i]}
		vix[t] = panel.adj_close[t * NS + N] / 100
	}
	// liquidity known at each close (trailing 20 days): daily vol and ADV per asset, [T, N]
	// (liquidity_estimates takes close * volume as the dollar volume, so the volume passed
	// is the panel's dollar volume over the adjusted close)
	liq_sigma, liq_adv := make([]f64, T * N), make([]f64, T * N)
	for i in 0 ..< N {
		pc, vc := make([]f64, T), make([]f64, T)
		for t in 0 ..< T {
			pc[t] = panel.adj_close[t * NS + i]
			vc[t] = panel.dollar_volume[t * NS + i] / pc[t]
		}
		s, a := ml_fin.liquidity_estimates(pc, vc, 20)
		for t in 0 ..< T {liq_sigma[t * N + i], liq_adv[t * N + i] = s[t], a[t]}
	}
	// returns from day t-1 to t (row t, t >= 1; row 0 stays 0)
	ret := make([]f64, T * N)
	logret := make([]f64, T * N)
	{
		r := ts.simple_returns(price, N)
		lr := ts.log_returns(price, N)
		defer {delete(r); delete(lr)}
		copy(ret[N:], r)
		copy(logret[N:], lr)
	}
	spy := make([]f64, T)
	for t in 0 ..< T {spy[t] = logret[t * N]}
	TEST :: 250
	start := T - TEST - 1 // decisions on days start ..< T - 1
	fmt.printf(
		"%d common trading days; models fitted on the %d before the last %d decisions\n",
		T,
		start - 1,
		TEST,
	)

	// HMM fitted on the fitting period (returns clipped to ±5%), filtered over all days
	clipped := make([]f64, T)
	for v, t in spy {clipped[t] = clamp(v, -0.05, 0.05)}
	hmm := ml_fin.hmm_new(3)
	defer ml_fin.hmm_free(&hmm)
	ml_fin.hmm_init_simple(&hmm, clipped[1:start])
	ml_fin.hmm_fit(&hmm, clipped[1:start], 500, 1e-6)
	filt := ml_fin.hmm_filter(&hmm, clipped[1:]) // row t - 1 is day t
	for s in 0 ..< 3 {
		fmt.printf(
			"   regime %d: mean %+6.1f%%, vol %5.1f%% a year\n",
			s,
			100 * 252 * hmm.emission_mu[s],
			100 * math.sqrt(252 * hmm.emission_var[s]),
		)
	}

	// GARCH(1,1) on the demeaned fitting-period returns; sigma2[t] = forecast of day t's
	// variance made at the close of t - 1
	mean_r := 0.0
	for v in spy[1:start] {mean_r += v / f64(start - 1)}
	resid := make([]f64, start - 1)
	for v, k in spy[1:start] {resid[k] = v - mean_r}
	g := ts.garch_fit(resid, .GARCH, 1, 1, 1000, 1e-6)
	om, al, be := g.params.omega, g.params.alpha[0], g.params.beta[0]
	sigma2 := make([]f64, T + 1)
	for v, k in g.conditional_var {sigma2[k + 1] = v}
	for t in start ..= T {
		e := spy[t - 1] - mean_r
		sigma2[t] = om + al * e * e + be * sigma2[t - 1]
	}
	fmt.printf("   GARCH(1,1): ω %.2e, α %.3f, β %.3f\n", om, al, be)
	// interval: quantiles of realised 21-day vol / forecast over the fitting period
	ratios := make([dynamic]f64)
	for t in 1 ..< start - 21 {
		rv := 0.0
		for k in t ..< t + 21 {rv += (spy[k] - mean_r) * (spy[k] - mean_r) / 21}
		append(&ratios, math.sqrt(rv / sigma2[t]))
	}
	slice.sort(ratios[:])
	q_lo, q_hi := ratios[len(ratios) / 10], ratios[len(ratios) * 9 / 10]
	fmt.printf(
		"   vol interval: forecast × [%.2f, %.2f] (realised 21-day vol over forecast, 10%% / 90%%)\n",
		q_lo,
		q_hi,
	)

	state := ml_fin.orchestrator_init(N)
	defer ml_fin.orchestrator_free(&state)
	asp := ml_fin.AvellanedaStoikovParams {
		sigma = 1,
		gamma = 1e-5,
		k     = 1.5,
		T     = 1,
	}
	strat, b6040, beq, bspy :=
		make([]f64, TEST), make([]f64, TEST), make([]f64, TEST), make([]f64, TEST)
	pnl_check, exposure_sum, turnover := 0.0, 0.0, 0.0
	// costs (fraction of capital) of each day's trades at three capitals; the benchmarks
	// rebalance to their fixed weights every day from the weights the prices drifted to
	COST_CAPS := [3]f64{CAP, 1e9, 1e10}
	strat_cost: [3][]f64
	for &c in strat_cost {c = make([]f64, TEST)}
	t6040, teq := make([]f64, N), make([]f64, N)
	t6040[0], t6040[N - 1] = 0.6, 0.4
	for &v in teq {v = 1 / f64(N)}
	h6040, heq := slice.clone(t6040), slice.clone(teq) // held (drifted) weights
	c6040, ceq := make([]f64, TEST), make([]f64, TEST)
	bad_weights := 0
	actions: [ml_fin.RegimeAction]int
	for k in 0 ..< TEST {
		t := start + k // decision at the close of day t
		probs := filt[(t - 1) * 3:][:3]
		regime := 0
		for s in 1 ..< 3 {if probs[s] > probs[regime] {regime = s}}
		vol := math.sqrt(252 * sigma2[t + 1])
		vrp := clamp((vix[t] - vol) / 0.10, -1, 1)
		hist := l.matrix_new(f64, 252, N)
		copy(hist.data, logret[(t - 251) * N:(t + 1) * N])
		hrp := ml_fin.hrp_allocate(&hist)
		before := make([]f64, N)
		copy(before, state.current_weights)
		d := ml_fin.orchestrator_step(
			&state,
			regime,
			252 * hmm.emission_mu[regime],
			math.sqrt(252 * hmm.emission_var[regime]),
			vol,
			vol * q_lo,
			vol * q_hi,
			vrp,
			hrp.weights,
			price[t * N:][:N],
			CAP,
			asp,
		)
		actions[d.action] += 1
		exposure_sum += d.exposure
		sw := 0.0
		for v, i in d.target_weights {
			if v < 0 {bad_weights += 1}
			sw += v
			turnover += abs(v - before[i])
		}
		if sw > 1 + 1e-12 {bad_weights += 1}
		next := ret[(t + 1) * N:][:N]
		sg_t, adv_t := liq_sigma[t * N:][:N], liq_adv[t * N:][:N]
		for cap, c in COST_CAPS {
			strat_cost[c][k] = ml_fin.rebalance_cost(
				ml_fin.DEFAULT_COST_MODEL,
				before,
				d.target_weights,
				cap,
				sg_t,
				adv_t,
			)
		}
		for b in 0 ..< 2 {
			target, held := t6040 if b == 0 else teq, h6040 if b == 0 else heq
			cb := ml_fin.rebalance_cost(ml_fin.DEFAULT_COST_MODEL, held, target, CAP, sg_t, adv_t)
			if b == 0 {c6040[k] = cb} else {ceq[k] = cb}
			r_b := 0.0
			for v, i in target {r_b += v * next[i]}
			for v, i in target {held[i] = v * (1 + next[i]) / (1 + r_b)}
		}
		expected := 0.0
		for v, i in d.target_weights {expected += v * next[i]}
		strat[k] = ml_fin.orchestrator_mark_to_market(&state, next, CAP)
		pnl_check += CAP * expected
		b6040[k] = 0.6 * next[0] + 0.4 * next[4]
		for v in next {beq[k] += v / f64(N)}
		bspy[k] = next[0]
		if k == TEST - 1 {ml_fin.print_decision(&d, tickers, t)}
		ml_fin.decision_free(&d, context.allocator)
		ml_fin.hrp_result_free(&hrp)
	}
	fmt.printf(
		"   actions over %d days: hold %d, aggressive %d, reduce %d, liquidate %d; mean exposure %.0f%%; turnover %.1f× capital\n",
		TEST,
		actions[.Hold],
		actions[.Aggressive],
		actions[.Reduce],
		actions[.Liquidate],
		100 * exposure_sum / TEST,
		turnover,
	)
	_orch_metrics("orchestrator", strat)
	_orch_metrics("60/40 SPY/TLT", b6040)
	_orch_metrics("equal weights", beq)
	_orch_metrics("SPY", bspy)
	fmt.println(
		"   net of costs at $1m (1 bp half spread, square-root impact on trailing 20-day vol and ADV):",
	)
	_orch_metrics("orchestrator", ml_fin.net_of_costs(strat, strat_cost[0]))
	_orch_metrics("60/40 SPY/TLT", ml_fin.net_of_costs(b6040, c6040))
	_orch_metrics("equal weights", ml_fin.net_of_costs(beq, ceq))
	costs_ok := true
	for cap, c in COST_CAPS {
		drag := 0.0
		for v, k in strat_cost[c] {
			drag += v
			costs_ok &&= v >= 0 && (c == 0 || v >= strat_cost[c - 1][k])
		}
		fmt.printf(
			"   orchestrator costs at $%.0e: %.2f%% of capital over the year\n",
			cap,
			100 * drag,
		)
	}
	fmt.println("   (one year, one path: not evidence of an edge either way)")

	ok := abs(state.cumulative_pnl - pnl_check) < 1e-6 * CAP && bad_weights == 0 && costs_ok
	fmt.printf(
		"%s P&L %.0f reconciles with the weights held times the next day's returns (%.0f); weights >= 0 and exposure <= 1 on all %d days; costs >= 0 and growing with the capital\n",
		"OK  " if ok else "FAIL",
		state.cumulative_pnl,
		pnl_check,
		TEST,
	)
	record_failures(int(!ok))
}
