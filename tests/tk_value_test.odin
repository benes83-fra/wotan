package tests

import ml "../wotan/analytics/ML"
import bfin "../wotan/behavioral_finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// The prospect theory value TK of a stock (Barberis, Mukherjee & Wang 2016,
// behavioral_finance/tk_value.odin)
//   tk_value_reference_test: TK against a literal transcription of BMW eq. (8) (m
//     negative and n non-negative excess returns, TK92 parameters) on LCG returns in
//     excess of a market, a NaN in a window, raw returns; the shape: TK falls with
//     volatility (loss aversion); at equal mean and variance positive skew > symmetric
//     > negative skew with probability weighting, the order reversed without it
//   tk_value_test: A. a market where expected returns fall with each stock's population
//     TK (20000 draws of its distribution: volatility 4 - 14% a month and up to a 5%
//     chance a month of a +40% jump; 0.5% a month lower expected return per standard
//     deviation of TK), 1000 stocks, 60 months of warm-up then 240 months: TK
//     estimated from 60 months predicts returns, the low-minus-high TK quintile spread
//     and the Fama-MacBeth slope (on 8 seeds t 6.3 - 9.1 and -7.3 to -9.6); not in the
//     control market (|t| <= 1.9). B. the 24 large
//     caps, monthly returns over their equal-weighted average (printed)

@(private = "file")
_tk_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

tk_value_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Prospect Theory Value (TK) Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	T, W :: 100, 60
	x: u64 = 5
	mkt := make([]f64, T)
	defer delete(mkt)
	for &m in mkt {m = 0.01 + 0.08 * (_tk_u(&x) - 0.5)}
	stocks: [3][]f64
	for k in 0 ..< 3 {
		stocks[k] = make([]f64, T)
		for t in 0 ..< T {
			stocks[k][t] = mkt[t] + 0.002 * f64(k) + 0.12 * (_tk_u(&x) - 0.5)
			stocks[k][t] += 0.3 if _tk_u(&x) < 0.03 * f64(k) else 0
		}
	}
	defer {for s in stocks {delete(s)}}
	stocks[2][70] = math.nan_f64()
	want := [3][4]f64 {
		{40, -1.0824299795286867, -0.03564967924269578, -0.023141575090894068},
		{40, -0.8571245994901302, -0.029413117783085196, -0.019375822286443682},
		{11, 0.10740218192234799, 0.010675879813918304, 0},
	}
	err := 0.0
	counts_ok := true
	for k in 0 ..< 3 {
		tk := bfin.tk_value(stocks[k], mkt, W)
		defer delete(tk)
		n, s := 0, 0.0
		for v in tk {if !math.is_nan(v) {n += 1; s += v}}
		counts_ok &&= n == int(want[k][0]) && math.is_nan(tk[W - 1])
		err = max(
			err,
			abs(s - want[k][1]) / abs(want[k][1]),
			abs(tk[W] - want[k][2]) / abs(want[k][2]),
		)
		if k <
		   2 {err = max(err, abs(tk[T - 1] - want[k][3]) / abs(want[k][3]))} else {counts_ok &&= math.is_nan(tk[T - 1])}
	}
	raw := bfin.tk_value(stocks[0], nil, W)
	defer delete(raw)
	err = max(
		err,
		abs(raw[W] + 0.035409851055053815) / 0.035409851055053815,
		abs(raw[T - 1] + 0.015020043806522755) / 0.015020043806522755,
	)
	expect(
		&passed,
		&total,
		counts_ok && err < 1e-13,
		fmt.tprintf(
			"TK = BMW eq. (8) over 60 months in excess of the market and raw (relative error %.1e); NaN in warm-up and while a NaN return is in the window",
			err,
		),
	)

	// shape
	r := make([]f64, 60)
	defer delete(r)
	vol: [3]f64
	for a, k in ([]f64{0.03, 0.06, 0.12}) {
		for i in 0 ..< 60 {r[i] = a if i % 2 == 0 else -a}
		vol[k] = bfin.cpt_value(r, bfin.TK92_PARAMS)
	}
	flat := bfin.TK92_PARAMS
	flat.gamma, flat.gamma_minus = 1, 1
	skew: [2][3]f64 // weighting / none: positive, symmetric, negative skew (mean 0, sd 6%)
	for p, k in ([]bfin.Prospect_Params{bfin.TK92_PARAMS, flat}) {
		for i in 0 ..< 60 {r[i] = 0.18 if i < 6 else -0.02}
		skew[k][0] = bfin.cpt_value(r, p)
		for i in 0 ..< 60 {r[i] = 0.06 if i % 2 == 0 else -0.06}
		skew[k][1] = bfin.cpt_value(r, p)
		for i in 0 ..< 60 {r[i] = -0.18 if i < 6 else 0.02}
		skew[k][2] = bfin.cpt_value(r, p)
	}
	expect(
		&passed,
		&total,
		vol[0] > vol[1] &&
		vol[1] > vol[2] &&
		skew[0][0] > skew[0][1] &&
		skew[0][1] > skew[0][2] &&
		skew[1][0] < skew[1][2],
		fmt.tprintf(
			"TK falls with volatility (±3/6/12%%: %.4f, %.4f, %.4f); positive skew > symmetric > negative skew with weighting (%.4f, %.4f, %.4f), negative skew best without (%.4f vs %.4f)",
			vol[0],
			vol[1],
			vol[2],
			skew[0][0],
			skew[0][1],
			skew[0][2],
			skew[1][2],
			skew[1][0],
		),
	)
	fmt.printf("TK reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

tk_value_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Prospect Theory Value (TK) and Stock Returns ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. synthetic market
	N, WARM, M :: 1000, 60, 300
	J :: 0.4
	sigma := make([]f64, N)
	q := make([]f64, N)
	pop := make([]f64, N)
	defer {delete(sigma); delete(q); delete(pop)}
	draws := make([]f64, 20000)
	defer delete(draws)
	for i in 0 ..< N {
		sigma[i] = 0.04 + 0.10 * rand.float64()
		q[i] = 0.05 * rand.float64()
		for &d in draws {d = sigma[i] * rand.float64_normal(0, 1) + J * ((1 if rand.float64() < q[i] else 0) - q[i])}
		pop[i] = bfin.cpt_value(draws, bfin.TK92_PARAMS)
	}
	pm, ps := 0.0, 0.0
	for v in pop {pm += v / f64(N)}
	for v in pop {ps += (v - pm) * (v - pm) / f64(N)}
	ps = math.sqrt(ps)
	results: [2][4]f64
	n_months := 0
	for b, k in ([]f64{0.005, 0}) {
		R := make([][]f64, N)
		TK := make([][]f64, N)
		defer {
			for i in 0 ..< N {delete(R[i]); delete(TK[i])}
			delete(R)
			delete(TK)
		}
		mkt := make([]f64, M)
		defer delete(mkt)
		for &m in mkt {m = 0.007 + 0.045 * rand.float64_normal(0, 1)}
		for i in 0 ..< N {
			R[i] = make([]f64, M)
			mu := -b * (pop[i] - pm) / ps // prospect theory investors overprice high-TK stocks
			for t in 0 ..< M {
				e :=
					sigma[i] * rand.float64_normal(0, 1) +
					J * ((1 if rand.float64() < q[i] else 0) - q[i])
				R[i][t] = mkt[t] + mu + e
			}
			TK[i] = bfin.tk_value(R[i], mkt, WARM)
		}
		results[k][0], results[k][1], results[k][2], results[k][3], n_months = signal_sort(
			R,
			TK,
			WARM,
		)
	}
	expect(
		&passed,
		&total,
		results[0][1] > 3 && results[0][3] < -3,
		fmt.tprintf(
			"mispriced market, %d months: low - high TK quintile %.2f%% a month (t %.1f), Fama-MacBeth slope on TK %.3f (t %.1f)",
			n_months,
			100 * results[0][0],
			results[0][1],
			results[0][2],
			results[0][3],
		),
	)
	expect(
		&passed,
		&total,
		abs(results[1][1]) < 3.5 && abs(results[1][3]) < 3.5,
		fmt.tprintf(
			"control market: low - high %.2f%% (t %.1f), slope t %.1f",
			100 * results[1][0],
			results[1][1],
			results[1][3],
		),
	)

	// B. the 24 large caps
	NS := len(LARGE_CAP_TICKERS)
	DP, _, dates, ok := yahoo_adjclose_panel(LARGE_CAP_TICKERS, .TenYears, 2000)
	if !ok {
		fmt.println("Not enough data (offline?): large-cap check skipped")
	} else {
		MM := (len(dates) - 1) / 21
		R := make([][]f64, NS)
		TK := make([][]f64, NS)
		mkt := make([]f64, MM)
		defer {
			for i in 0 ..< NS {delete(R[i]); delete(TK[i])}
			delete(R)
			delete(TK)
			delete(mkt)
		}
		for i in 0 ..< NS {
			R[i] = make([]f64, MM)
			for m in 0 ..< MM {
				R[i][m] = DP[(21 * (m + 1)) * NS + i] / DP[(21 * m) * NS + i] - 1
				mkt[m] += R[i][m] / f64(NS)
			}
		}
		for i in 0 ..< NS {TK[i] = bfin.tk_value(R[i], mkt, 60)}
		sm, st, fc, ft, nm := signal_sort(R, TK, 60)
		fmt.printf(
			"Large caps, %d months: low - high TK quintile %.2f%% a month (t %.2f), Fama-MacBeth slope %.3f (t %.2f)\n",
			nm,
			100 * sm,
			st,
			fc,
			ft,
		)
	}

	fmt.printf("TK and returns: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
