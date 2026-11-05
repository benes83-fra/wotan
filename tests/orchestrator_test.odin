package tests

import ts "../wotan/analytics"
import w "../wotan/core"
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
// weights and SPY; the checks are on the accounting (P&L reconciles with the weights
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
	fmt.printf(
		"   %-22s return %+6.1f%%  vol %5.1f%%  Sharpe %5.2f  max drawdown %5.1f%%\n",
		name,
		100 * (eq - 1),
		100 * vol,
		m * 252 / vol if vol > 0 else 0,
		100 * mdd,
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
	maps := make([]map[w.Date]f64, len(symbols))
	spy_dates := make([dynamic]w.Date)
	for sym, k in symbols {
		df := yahoo.read_yahoo(sym, .Daily, .FiveYears, allocator)
		defer w.destroy_dataframe(&df)
		if df.rows < 600 {
			fmt.printf("Not enough %s data (offline?): skipped\n", sym)
			return
		}
		maps[k] = make(map[w.Date]f64)
		dc, pc := w.column(&df, "Date"), w.column(&df, "AdjClose")
		for i in 0 ..< df.rows {
			d, dn := w.column_at_date(dc, i)
			p, pn := w.column_at_float(pc, i)
			if dn || pn || p <= 0 {continue}
			maps[k][d] = p
			if k == 0 {append(&spy_dates, d)}
		}
	}
	dates := make([dynamic]w.Date)
	for d in spy_dates {
		all := true
		for m in maps {all &&= d in m}
		if all {append(&dates, d)}
	}
	T := len(dates)
	price := make([]f64, T * N)
	vix := make([]f64, T)
	for d, t in dates {
		for i in 0 ..< N {price[t * N + i] = maps[i][d]}
		vix[t] = maps[N][d] / 100
	}
	// returns from day t-1 to t (row t, t >= 1)
	ret := make([]f64, T * N)
	logret := make([]f64, T * N)
	for t in 1 ..< T {
		for i in 0 ..< N {
			ret[t * N + i] = price[t * N + i] / price[(t - 1) * N + i] - 1
			logret[t * N + i] = math.ln(price[t * N + i] / price[(t - 1) * N + i])
		}
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
		"   (no transaction costs; one year, one path: not evidence of an edge either way)",
	)

	ok := abs(state.cumulative_pnl - pnl_check) < 1e-6 * CAP && bad_weights == 0
	fmt.printf(
		"%s P&L %.0f reconciles with the weights held times the next day's returns (%.0f); weights >= 0 and exposure <= 1 on all %d days\n",
		"OK  " if ok else "FAIL",
		state.cumulative_pnl,
		pnl_check,
		TEST,
	)
	record_failures(int(!ok))
}
