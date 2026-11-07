package tests

import bfin "../wotan/behavioral_finance"
import plot "../wotan/plot"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// George & Hwang (2004) 52-week-high strategy: each 21-day month rank the stocks on
// PTH = P / 52-week high, long the top 30%, short the bottom 30% (equally weighted),
// held 6 months with overlapping cohorts (Jegadeesh & Titman); t-statistics
// Newey-West with 5 lags (the overlapping cohorts make the monthly spread an MA(5)).
//   A. synthetic: 500 stocks, 10 years. Prices follow fundamental value, except that
//      while value is above the 52-week high the price closes only 2% of the gap a
//      day (investors anchored on the high). The strategy earns a significant spread;
//      on the control panel (no anchoring: price = value) it does not.
//   B. the 24 large caps, 10 years of adjusted closes: PTH <= 1 checked, the returns
//      printed (no assertion on them)

@(private = "file")
_nw_mean_t :: proc(x: []f64, lags: int) -> (mean, t: f64, n: int) {
	vals := make([dynamic]f64, context.temp_allocator)
	for v in x {if !math.is_nan(v) {append(&vals, v)}}
	n = len(vals)
	if n < 2 {return}
	for v in vals {mean += v / f64(n)}
	lrv := 0.0
	for l in 0 ..= lags {
		g := 0.0
		for i in l ..< n {g += (vals[i] - mean) * (vals[i - l] - mean)}
		g /= f64(n)
		lrv += g if l == 0 else 2 * (1 - f64(l) / f64(lags + 1)) * g
	}
	t = mean / math.sqrt(lrv / f64(n))
	return
}

// prices [assets][days] -> monthly (21-day) George-Hwang (6, 6) returns
@(private = "file")
_gh_strategy :: proc(prices: [][]f64, first_day: int) -> bfin.Overlapping_Returns {
	pth := bfin.compute_anchoring_signal(prices, 252)
	defer bfin.compute_anchoring_signal_free(pth)
	NA, D := len(prices), len(prices[0])
	days := make([dynamic]int, context.temp_allocator)
	for d := first_day; d < D; d += 21 {append(&days, d)}
	M := len(days)
	W := make([][]f64, M, context.temp_allocator)
	R := make([][]f64, M, context.temp_allocator)
	col := make([]f64, NA, context.temp_allocator)
	for d, m in days {
		for a in 0 ..< NA {col[a] = pth[a][d]}
		W[m] = bfin.anchoring_portfolio(col, 0.3, context.temp_allocator)
		R[m] = make([]f64, NA, context.temp_allocator)
		for a in 0 ..< NA {
			R[m][a] = prices[a][d] / prices[a][days[m - 1]] - 1 if m > 0 else math.nan_f64()
		}
	}
	return bfin.overlapping_strategy_returns(W, R, 6, 0)
}

anchoring_backtest_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Anchoring & 52-Week High (George & Hwang 2004) ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. synthetic panels
	NS, DS :: 500, 2520
	spreads: [2]f64
	tstats: [2]f64
	for anchored, k in ([]bool{true, false}) {
		prices := make([][]f64, NS)
		defer {for p in prices {delete(p)}; delete(prices)}
		lps := make([]f64, DS)
		defer delete(lps)
		q := make([]int, DS) // monotone queue: the running 252-day high of lps
		defer delete(q)
		for a in 0 ..< NS {
			prices[a] = make([]f64, DS)
			v := math.ln(20 + 80 * rand.float64())
			lp := v
			head, tail := 0, 0
			for t in 0 ..< DS {
				v += 0.02 * rand.float64_normal(0, 1)
				for head < tail && q[head] < t - 252 {head += 1}
				high := lps[q[head]] if head < tail else lp // high of days t-252 .. t-1
				if anchored && v > high {lp += 0.02 * (v - lp)} else {lp = v}
				lps[t] = lp
				for tail > head && lps[q[tail - 1]] <= lp {tail -= 1}
				q[tail] = t
				tail += 1
				prices[a][t] = math.exp(lp)
			}
		}
		res := _gh_strategy(prices, 251)
		defer bfin.overlapping_returns_free(&res)
		spreads[k], tstats[k], _ = _nw_mean_t(res.spread, 5)
	}
	expect(
		&passed,
		&total,
		tstats[0] > 3 && abs(tstats[1]) < 3,
		fmt.tprintf(
			"anchored prices: winner - loser %.2f%% a month (t %.1f); control without anchoring %.2f%% (t %.1f)",
			100 * spreads[0],
			tstats[0],
			100 * spreads[1],
			tstats[1],
		),
	)

	// B. the 24 large caps
	NA := len(LARGE_CAP_TICKERS)
	P, _, dates, ok := yahoo_adjclose_panel(LARGE_CAP_TICKERS, .TenYears, 1500)
	if !ok {
		fmt.println("Not enough data (offline?): real-data backtest skipped")
	} else {
		D := len(dates)
		prices := make([][]f64, NA)
		defer {for p in prices {delete(p)}; delete(prices)}
		for a in 0 ..< NA {
			prices[a] = make([]f64, D)
			for t in 0 ..< D {prices[a][t] = P[t * NA + a]}
		}
		pth := bfin.compute_anchoring_signal(prices, 252)
		max_pth := 0.0
		for row in pth {for v in row {if !math.is_nan(v) {max_pth = max(max_pth, v)}}}
		bfin.compute_anchoring_signal_free(pth)
		res := _gh_strategy(prices, 251)
		defer bfin.overlapping_returns_free(&res)
		m, t, n := _nw_mean_t(res.spread, 5)
		ml, _, _ := _nw_mean_t(res.long, 5)
		msh, _, _ := _nw_mean_t(res.short, 5)
		expect(
			&passed,
			&total,
			max_pth == 1,
			fmt.tprintf("large caps: PTH <= 1 on %d days (max %.4f)", D, max_pth),
		)
		fmt.printf(
			"Large caps, %d months: winners %.2f%%, losers %.2f%%, winner - loser %.2f%% a month (Newey-West t %.2f)\n",
			n,
			100 * ml,
			100 * msh,
			100 * m,
			t,
		)
		// cumulative winner - loser
		xs := make([]f64, n)
		ys := make([]f64, n)
		defer {delete(xs); delete(ys)}
		acc, i := 1.0, 0
		for v in res.spread {
			if math.is_nan(v) {continue}
			acc *= 1 + v
			xs[i], ys[i] = f64(i), acc
			i += 1
		}
		cfg := plot.DEFAULT_PLOT_CONFIG
		cfg.title = "52-week high: winner - loser (24 large caps)"
		cfg.x_label = "Month"
		cfg.y_label = "Growth of 1"
		cfg.show_grid = true
		line := []plot.LineData {
			{xs = xs, ys = ys, color = plot.RED, style = .Solid, label = "Winner - loser"},
		}
		if plot.multi_line_png(line, "plots/anchoring_backtest.png", cfg, allocator) {
			fmt.println("Saved: plots/anchoring_backtest.png")
		}
	}

	fmt.printf("Anchoring backtest: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
