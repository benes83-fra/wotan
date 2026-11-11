package tests

import bfin "../wotan/behavioral_finance"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Lottery demand: MAX (Bali, Cakici & Whitelaw 2011) and the salience theory value ST
// (Bordalo, Gennaioli & Shleifer; Cosemans & Frehen 2021), behavioral_finance/lottery.odin
//   lottery_reference_test: Cosemans & Frehen's worked example of diminishing
//     sensitivity; salience weights have mean 1, fall with rank by delta, delta = 1 gives
//     ST = 0; ST, MAX and MAX(5) on an LCG panel (20 assets, 3 months, gaps, an
//     infinite return, a month with too few days) and ST with tied salience = a
//     separate numpy / scipy implementation; a long-shot stock has ST > 0, a
//     crash-prone one ST < 0
//   lottery_test: A. daily markets (1000 stocks, 21-day months, betas 0.5 - 1.5, daily
//     volatility 1 - 3%, chances up to 5% a day each of a +10% and a -10% day): expected
//     returns fall by 1% a month per standard deviation of each stock's population ST
//     (its mean over 60 warm-up months; CF's high-minus-low is -1.9% a month EW), or
//     of its population MAX, or not at all (control); 240 months of last month's
//     measure predicting this month's return (low-minus-high decile, Fama-MacBeth):
//     each measure predicts in its own market, nothing in the control. On 9 seeds:
//     ST spread t 4.8 - 7.7, FM t -5.3 to -8.4; MAX 6.8 - 9.7 and -9.5 to -12.9;
//     control |t| <= 2.5. (With rarer +-15% days, chances up to 2%, a month's ST is
//     mostly noise outside jump months: spread t 4.8 but FM t -2.7.)
//     B. the 24 large caps, 21-day months, 10 years (printed)

@(private = "file")
_lt_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

@(private = "file")
_lt_rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}

lottery_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Lottery Demand (MAX, Salience) Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// 1. CF's example: a 10% return on a 5% market day is less salient than a 5%
	// return on a flat day (sigma 0.05 / 0.25 = 0.2 vs 0.05 / 0.15 = 1/3), so the
	// second day gets rank 1: weights delta / mean and delta^2 / mean
	w2: [2]f64
	bfin.salience_weights({0.10, 0.05}, {0.05, 0.0}, w2[:])
	ok1 :=
		abs(w2[1] - 2 * 0.7 / (0.7 + 0.49)) < 1e-15 && abs(w2[0] - 2 * 0.49 / (0.7 + 0.49)) < 1e-15
	// 21 days: mean 1, consecutive ranks in the ratio delta; delta = 1: ST = 0
	r21, m21, w21: [21]f64
	x: u64 = 3
	for s in 0 ..< 21 {r21[s], m21[s] = 0.04 * (_lt_u(&x) - 0.5), 0.02 * (_lt_u(&x) - 0.5)}
	bfin.salience_weights(r21[:], m21[:], w21[:])
	mean, wmax, n7 := 0.0, 0.0, 0
	for w in w21 {mean += w / 21; wmax = max(wmax, w)}
	for w in w21 {
		for v in w21 {if abs(v / w - 0.7) < 1e-12 {n7 += 1}}
	}
	st1 := bfin.salience_value(r21[:], m21[:], delta = 1)
	expect(
		&passed,
		&total,
		ok1 &&
		abs(mean - 1) < 1e-14 &&
		n7 == 20 &&
		abs(wmax - 21 * 0.7 * 0.3 / (0.7 * (1 - math.pow(0.7, 21)))) < 1e-12 &&
		abs(st1) < 1e-17,
		fmt.tprintf(
			"salience weights: CF's example (%.4f on the flat day vs %.4f), mean 1, 20 consecutive-rank ratios = delta, top weight %.3f; delta = 1: ST %.0e",
			w2[1],
			w2[0],
			wmax,
			st1,
		),
	)

	// 2. LCG panel = numpy / scipy
	x = 5
	N, T :: 20, 63
	months := []int{0, 21, 41, 63}
	R := make([][]f64, N)
	defer {for r in R {delete(r)}; delete(R)}
	for i in 0 ..< N {
		R[i] = make([]f64, T)
		for t in 0 ..< T {
			a := 0.03 * (_lt_u(&x) - 0.5)
			R[i][t] = a + (0.2 if _lt_u(&x) < 0.02 else 0)
		}
	}
	for t in 30 ..< 45 {R[2][t] = math.nan_f64()}
	R[4][50] = math.inf_f64(1)
	ST := bfin.salience_theory_value(R, months)
	M1 := bfin.max_daily_return(R, months)
	M5 := bfin.max_daily_return(R, months, 5)
	defer {bfin.lottery_measure_free(ST); bfin.lottery_measure_free(M1)
		bfin.lottery_measure_free(M5)}
	nst, nmx, s1, s2, x1, x5 := 0, 0, 0.0, 0.0, 0.0, 0.0
	for i in 0 ..< N {
		for k in 0 ..< 3 {
			if math.is_nan(ST[i][k]) {nst += 1} else {s1 += ST[i][k]; s2 += ST[i][k] * ST[i][k]}
			if math.is_nan(M1[i][k]) {nmx += 1} else {x1 += M1[i][k]; x5 += M5[i][k]}
		}
	}
	e2 := max(
		_lt_rel(s1, 1.2552126875389824),
		_lt_rel(s2, 0.09320934634721716),
		_lt_rel(ST[0][0], 0.04979038393760654),
		_lt_rel(ST[2][2], -0.0025726317586941055),
		_lt_rel(x1, 5.13545880188443),
		_lt_rel(x5, 1.8631856997507639),
		_lt_rel(M5[4][2], 0.04926469105834576),
	)
	// tied salience (two days with r = m, two with the same r and m), theta 0.05, delta 0.6
	rt := []f64{0.01, 0.02, 0.02, -0.03, 0.05, 0.0, -0.01, 0.04}
	mt := []f64{0.01, 0.0, 0.0, -0.03, 0.01, 0.02, 0.0, -0.01}
	stt := bfin.salience_value(rt, mt, 0.05, 0.6, 1)
	e2 = max(e2, _lt_rel(stt, 0.019185760706548913))
	expect(
		&passed,
		&total,
		nst == 1 &&
		nmx == 1 &&
		math.is_nan(ST[2][1]) &&
		math.is_nan(bfin.salience_value(rt, mt)) &&
		e2 < 1e-12,
		fmt.tprintf(
			"ST, MAX and MAX(5) on 20 assets x 3 months with gaps, and ST with tied salience = numpy / scipy (relative error %.1e; NaN below 15 valid days)",
			e2,
		),
	)

	// 3. a long shot (small losses, one +30% day) has a salient upside, a crash-prone
	// stock (small gains, one -30% day) a salient downside; the market is flat
	ls, cr, flat: [21]f64
	for s in 0 ..< 21 {ls[s], cr[s] = -0.015, 0.015}
	ls[7], cr[7] = 0.30, -0.30
	st_ls := bfin.salience_value(ls[:], flat[:])
	st_cr := bfin.salience_value(cr[:], flat[:])
	expect(
		&passed,
		&total,
		st_ls > 0 && st_cr < 0 && abs(st_ls + st_cr) < 1e-15,
		fmt.tprintf(
			"salient upside: long shot ST %.4f > 0, crash-prone ST %.4f < 0 (reflection)",
			st_ls,
			st_cr,
		),
	)

	fmt.printf("Lottery reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// signal of month k - 1 aligned with the return of month k
@(private = "file")
_lt_lag :: proc(S: [][]f64) -> [][]f64 {
	out := make([][]f64, len(S))
	for s, i in S {
		out[i] = make([]f64, len(s))
		out[i][0] = math.nan_f64()
		copy(out[i][1:], s[:len(s) - 1])
	}
	return out
}

// standardized (mean 0, sample sd 1)
@(private = "file")
_lt_z :: proc(v: []f64) -> []f64 {
	m, sd := 0.0, 0.0
	for x in v {m += x / f64(len(v))}
	for x in v {sd += (x - m) * (x - m) / f64(len(v) - 1)}
	sd = math.sqrt(sd)
	z := make([]f64, len(v))
	for x, i in v {z[i] = (x - m) / sd}
	return z
}

// simulated market: months (one formation month before the 240 tested), mispricing per
// standard deviation of the population measure (a month), lottery days +-JUMP with
// chances up to LT_P a day
LT_M :: 241
LT_B :: 0.01
JUMP :: 0.1
LT_P :: 0.05

lottery_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Lottery Demand (MAX, Salience) and Stock Returns ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. simulated daily markets
	N, D, WARM, M :: 1000, 21, 60, LT_M
	beta := make([]f64, N)
	vol := make([]f64, N)
	pu := make([]f64, N)
	pd := make([]f64, N)
	pop := make([]f64, N) // population ST
	pmx := make([]f64, N) // population MAX
	defer {delete(beta); delete(vol); delete(pu); delete(pd); delete(pop); delete(pmx)}
	for i in 0 ..< N {
		beta[i], vol[i] = 0.5 + rand.float64(), 0.01 + 0.02 * rand.float64()
		pu[i], pd[i] = LT_P * rand.float64(), LT_P * rand.float64()
	}
	day :: proc(i: int, m, mu: f64, beta, vol, pu, pd: []f64) -> f64 {
		j :=
			JUMP * ((1 if rand.float64() < pu[i] else 0) - pu[i]) -
			JUMP * ((1 if rand.float64() < pd[i] else 0) - pd[i])
		return beta[i] * m + vol[i] * rand.float64_normal(0, 1) + j + mu
	}
	months := make([]int, M + 1)
	defer delete(months)
	for k in 0 ..= M {months[k] = D * k}
	// population ST: mean ST over WARM months
	{
		W := make([][]f64, N)
		for i in 0 ..< N {W[i] = make([]f64, WARM * D)}
		mk := make([]f64, WARM * D)
		for t in 0 ..< WARM * D {
			m := 0.01 * rand.float64_normal(0, 1)
			for i in 0 ..< N {W[i][t] = day(i, m, 0, beta, vol, pu, pd)}
		}
		wm := make([]int, WARM + 1)
		for k in 0 ..= WARM {wm[k] = D * k}
		ws := bfin.salience_theory_value(W, wm)
		wx := bfin.max_daily_return(W, wm)
		for i in 0 ..< N {
			for k in 0 ..< WARM {pop[i] += ws[i][k] / f64(WARM); pmx[i] += wx[i][k] / f64(WARM)}
		}
		bfin.lottery_measure_free(ws)
		bfin.lottery_measure_free(wx)
		for r in W {delete(r)}
		delete(W)
		delete(mk)
		delete(wm)
	}
	zst, zmx := _lt_z(pop), _lt_z(pmx)
	defer {delete(zst); delete(zmx)}
	res: [3][2][4]f64 // [ST-, MAX-mispriced, control][ST, MAX][spread, t, fm, fm t]
	n_months := 0
	for z, k in ([][]f64{zst, zmx, nil}) {
		R := make([][]f64, N)
		defer {for r in R {delete(r)}; delete(R)}
		for i in 0 ..< N {R[i] = make([]f64, M * D)}
		for t in 0 ..< M * D {
			m := 0.01 * rand.float64_normal(0, 1)
			for i in 0 ..< N {
				mu := -LT_B * z[i] / f64(D) if z != nil else 0 // lottery demand overprices
				R[i][t] = day(i, m, mu, beta, vol, pu, pd)
			}
		}
		MR := monthly_compound(R, months)
		ST := bfin.salience_theory_value(R, months)
		MX := bfin.max_daily_return(R, months)
		LST, LMX := _lt_lag(ST), _lt_lag(MX)
		defer {
			bfin.lottery_measure_free(MR)
			bfin.lottery_measure_free(ST)
			bfin.lottery_measure_free(MX)
			bfin.lottery_measure_free(LST)
			bfin.lottery_measure_free(LMX)
		}
		for S, j in ([][][]f64{LST, LMX}) {
			r := &res[k][j]
			r[0], r[1], r[2], r[3], n_months = signal_sort(MR, S, 1, 0.1)
		}
	}
	expect(
		&passed,
		&total,
		res[0][0][1] > 3 && res[0][0][3] < -3,
		fmt.tprintf(
			"market mispriced by population ST, %d months: low - high ST decile %.2f%% a month (t %.1f), FM slope t %.1f (MAX: spread t %.1f, FM t %.1f)",
			n_months,
			100 * res[0][0][0],
			res[0][0][1],
			res[0][0][3],
			res[0][1][1],
			res[0][1][3],
		),
	)
	expect(
		&passed,
		&total,
		res[1][1][1] > 3 && res[1][1][3] < -3,
		fmt.tprintf(
			"market mispriced by population MAX: low - high MAX decile %.2f%% a month (t %.1f), FM slope t %.1f (ST: spread t %.1f, FM t %.1f)",
			100 * res[1][1][0],
			res[1][1][1],
			res[1][1][3],
			res[1][0][1],
			res[1][0][3],
		),
	)
	expect(
		&passed,
		&total,
		abs(res[2][0][1]) < 3.5 &&
		abs(res[2][0][3]) < 3.5 &&
		abs(res[2][1][1]) < 3.5 &&
		abs(res[2][1][3]) < 3.5,
		fmt.tprintf(
			"control market: ST spread t %.1f, FM t %.1f; MAX spread t %.1f, FM t %.1f",
			res[2][0][1],
			res[2][0][3],
			res[2][1][1],
			res[2][1][3],
		),
	)

	// B. the 24 large caps, 21-day months
	NS := len(LARGE_CAP_TICKERS)
	panel, ok, bad := net.read_yahoo_panel(LARGE_CAP_TICKERS, .Daily, .TenYears)
	defer net.yahoo_panel_free(&panel)
	DP, dates := panel.adj_close, panel.dates
	ok &&= len(dates) >= 2000
	if !ok {
		fail_missing_data(
			#procedure,
			fmt.tprintf("Yahoo data for %s", LARGE_CAP_TICKERS[bad]) if bad >= 0 else fmt.tprintf("2000 common days of Yahoo data (got %d)", len(dates)),
		)
	} else {
		ND := len(dates) - 1
		MM := ND / D
		R := make([][]f64, NS)
		for i in 0 ..< NS {
			R[i] = make([]f64, MM * D)
			for t in 0 ..< MM * D {R[i][t] = DP[(t + 1) * NS + i] / DP[t * NS + i] - 1}
		}
		lm := make([]int, MM + 1)
		for k in 0 ..= MM {lm[k] = D * k}
		MR := monthly_compound(R, lm)
		LST := _lt_lag(bfin.salience_theory_value(R, lm))
		LMX := _lt_lag(bfin.max_daily_return(R, lm))
		for S, j in ([][][]f64{LST, LMX}) {
			sm, st, fc, ft, nm := signal_sort(MR, S, 1)
			fmt.printf(
				"Large caps, %d months, %s: low - high quintile %.2f%% a month (t %.2f), FM slope %.3f (t %.2f)\n",
				nm,
				"ST" if j == 0 else "MAX",
				100 * sm,
				st,
				fc,
				ft,
			)
		}
	}

	fmt.printf("Lottery demand: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
