package tests

import ml "../wotan/analytics/ML"
import bfin "../wotan/behavioral_finance"
import l "../wotan/linalg"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// The disposition effect:
//   A. Odean's PGR / PLR on simulated investors (100 accounts, 10 stocks, 500 days of
//      driftless prices): disposition investors sell a winning position with
//      probability 10% a day and a losing one with 4%; neutral investors sell either
//      with 7%. PGR > PLR significantly for the first, not for the second.
//   B. Grinblatt & Han's (2005) market: P = w F + (1 - w) R, the reference price
//      R updated with turnover (R' = V P + (1 - V) R), F a random walk; 200 stocks,
//      5 years of weekly warm-up then 10 years of weekly Fama-MacBeth regressions of
//      the next week's return. With disposition investors (w = 0.7) the overhang
//      predicts returns, and momentum (weeks -52 .. -5) does on its own but loses most
//      of its t-statistic next to the overhang (Grinblatt & Han: it becomes
//      insignificant; here it falls to under half, as the estimated overhang is a
//      noisy proxy: on 10 seeds from t 9 - 11 to 0.8 - 3.1); without them (w = 1)
//      neither predicts (|t| < 3.5).
//   C. The 24 large caps, weekly (5 trading days), 10 years: the same regressions
//      (printed). Turnover is a proxy: Yahoo has no shares outstanding, so each stock's
//      weekly volume is scaled to an average turnover of 1% a week (Grinblatt & Han's
//      sample mean).

@(private = "file")
_simulate_investors :: proc(prices: [][]f64, p_gain, p_loss: f64) -> bfin.Disposition_Counts {
	NA, ND := len(prices), len(prices[0])
	trades := make([dynamic]bfin.Trade, context.temp_allocator)
	defer delete(trades)
	held := make([]f64, NA, context.temp_allocator)
	cost := make([]f64, NA, context.temp_allocator)
	defer {delete(held, context.temp_allocator); delete(cost, context.temp_allocator)}
	for acct in 0 ..< 100 {
		for &h in held {h = 0}
		for d in 0 ..< ND {
			for a in 0 ..< NA {
				p := prices[a][d]
				if held[a] > 0 {
					prob := p_gain if p > cost[a] else p_loss
					if p == cost[a] {prob = 0}
					if rand.float64() < prob {
						append(&trades, bfin.Trade{acct, a, d, -held[a], p})
						held[a] = 0
						continue
					}
				} else if rand.float64() < 0.03 {
					append(&trades, bfin.Trade{acct, a, d, 10, p})
					held[a], cost[a] = 10, p
				}
			}
		}
	}
	return bfin.odean_disposition(trades[:], prices)
}

// Fama-MacBeth of r_{s+1} on (1, momentum) and (1, momentum, overhang), weekly prices
// [stock][week] and overhangs; formation weeks first .. last - 1
@(private = "file")
_gh_regressions :: proc(P, G: [][]f64, first: int) -> (alone, both: ml.Fama_MacBeth_Result) {
	N, TW := len(P), len(P[0])
	X1 := make([dynamic]l.Matrix(f64), context.temp_allocator)
	X2 := make([dynamic]l.Matrix(f64), context.temp_allocator)
	Y := make([dynamic][]f64, context.temp_allocator)
	defer {
		for i in 0 ..< len(Y) {l.matrix_free(&X1[i]); l.matrix_free(&X2[i]); delete(Y[i], context.temp_allocator)}
		delete(X1)
		delete(X2)
		delete(Y)
	}
	for s in first ..< TW - 1 {
		x1 := l.matrix_new(f64, N, 2, context.temp_allocator)
		x2 := l.matrix_new(f64, N, 3, context.temp_allocator)
		y := make([]f64, N, context.temp_allocator)
		for i in 0 ..< N {
			mom := P[i][s - 4] / P[i][s - 52] - 1
			x1.data[i * 2], x1.data[i * 2 + 1] = 1, mom
			x2.data[i * 3], x2.data[i * 3 + 1], x2.data[i * 3 + 2] = 1, mom, G[i][s]
			y[i] = P[i][s + 1] / P[i][s] - 1
		}
		append(&X1, x1)
		append(&X2, x2)
		append(&Y, y)
	}
	return ml.fama_macbeth(X1[:], Y[:]), ml.fama_macbeth(X2[:], Y[:])
}

disposition_effect_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== The Disposition Effect ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. investors
	NA, ND :: 10, 500
	prices := make([][]f64, NA)
	defer {for r in prices {delete(r)}; delete(prices)}
	for a in 0 ..< NA {
		prices[a] = make([]f64, ND)
		p := 50.0
		for d in 0 ..< ND {
			p *= math.exp(0.02 * rand.float64_normal(0, 1))
			prices[a][d] = math.round(100 * p) / 100
		}
	}
	disp := _simulate_investors(prices, 0.10, 0.04)
	neutral := _simulate_investors(prices, 0.07, 0.07)
	expect(
		&passed,
		&total,
		disp.t_stat > 5 && abs(neutral.t_stat) < 3,
		fmt.tprintf(
			"Odean: disposition investors PGR %.3f vs PLR %.3f (t %.1f); neutral investors %.3f vs %.3f (t %.1f)",
			disp.pgr,
			disp.plr,
			disp.t_stat,
			neutral.pgr,
			neutral.plr,
			neutral.t_stat,
		),
	)

	// B. Grinblatt-Han market
	N, WARM, TW :: 200, 260, 260 + 520
	res: [2][2]ml.Fama_MacBeth_Result
	for w, k in ([]f64{0.7, 1}) {
		P := make([][]f64, N)
		V := make([][]f64, N)
		G := make([][]f64, N)
		defer {
			for i in 0 ..< N {delete(P[i]); delete(V[i]); delete(G[i])}
			delete(P)
			delete(V)
			delete(G)
		}
		for i in 0 ..< N {
			P[i] = make([]f64, TW)
			V[i] = make([]f64, TW)
			F := 50.0
			R := F
			vi := 0.005 + 0.025 * rand.float64()
			for t in 0 ..< TW {
				F *= math.exp(0.04 * rand.float64_normal(0, 1) - 0.0008)
				p := w * F + (1 - w) * R
				v := clamp(vi * math.exp(0.5 * rand.float64_normal(0, 1)), 0, 1)
				P[i][t], V[i][t] = p, v
				R = v * p + (1 - v) * R
			}
			G[i] = bfin.capital_gains_overhang(P[i], V[i], WARM)
		}
		res[k][0], res[k][1] = _gh_regressions(P, G, WARM + 1)
	}
	defer {for &r in res {ml.fama_macbeth_result_free(&r[0]); ml.fama_macbeth_result_free(&r[1])}}
	d_alone, d_both := res[0][0], res[0][1]
	c_alone, c_both := res[1][0], res[1][1]
	expect(
		&passed,
		&total,
		d_both.t_stat[2] > 3 &&
		d_alone.t_stat[1] > 4 &&
		abs(d_both.t_stat[1]) < 0.5 * d_alone.t_stat[1],
		fmt.tprintf(
			"disposition market (w 0.7), %d weeks: overhang t %.1f; momentum alone t %.1f, next to the overhang t %.1f",
			d_both.n_periods,
			d_both.t_stat[2],
			d_alone.t_stat[1],
			d_both.t_stat[1],
		),
	)
	expect(
		&passed,
		&total,
		abs(c_both.t_stat[2]) < 3.5 && abs(c_alone.t_stat[1]) < 3.5,
		fmt.tprintf(
			"control market (w 1): overhang t %.1f, momentum t %.1f",
			c_both.t_stat[2],
			c_alone.t_stat[1],
		),
	)

	// C. the 24 large caps
	NS := len(LARGE_CAP_TICKERS)
	panel, ok, bad := net.read_yahoo_panel(LARGE_CAP_TICKERS, .Daily, .TenYears)
	defer net.yahoo_panel_free(&panel)
	DP, DV, dates := panel.adj_close, panel.dollar_volume, panel.dates
	ok &&= len(dates) >= 2000
	if !ok {
		fail_missing_data(
			#procedure,
			fmt.tprintf("Yahoo data for %s", LARGE_CAP_TICKERS[bad]) if bad >= 0 else fmt.tprintf("2000 common days of Yahoo data (got %d)", len(dates)),
		)
	} else {
		NW := len(dates) / 5
		P := make([][]f64, NS)
		V := make([][]f64, NS)
		G := make([][]f64, NS)
		defer {
			for i in 0 ..< NS {delete(P[i]); delete(V[i]); delete(G[i])}
			delete(P)
			delete(V)
			delete(G)
		}
		for i in 0 ..< NS {
			P[i] = make([]f64, NW)
			V[i] = make([]f64, NW)
			mean := 0.0
			for wk in 0 ..< NW {
				vol := 0.0 // shares traded in the week: dollar volume / price
				for d in 5 * wk ..< 5 * wk + 5 {vol += DV[d * NS + i] / DP[d * NS + i]}
				P[i][wk] = DP[(5 * wk + 4) * NS + i]
				V[i][wk] = vol
				mean += vol / f64(NW)
			}
			for &v in V[i] {v = min(1, 0.01 * v / mean)}
			G[i] = bfin.capital_gains_overhang(P[i], V[i], 260)
		}
		alone, both := _gh_regressions(P, G, 261)
		defer {ml.fama_macbeth_result_free(&alone); ml.fama_macbeth_result_free(&both)}
		fmt.printf(
			"Large caps, %d weeks: overhang %.4f (t %.2f); momentum alone %.4f (t %.2f), next to the overhang %.4f (t %.2f)\n",
			both.n_periods,
			both.coef[2],
			both.t_stat[2],
			alone.coef[1],
			alone.t_stat[1],
			both.coef[1],
			both.t_stat[1],
		)
	}

	fmt.printf("Disposition effect: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
