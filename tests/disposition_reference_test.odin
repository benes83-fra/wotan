package tests

import ml "../wotan/analytics/ML"
import bfin "../wotan/behavioral_finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// The disposition effect (behavioral_finance/disposition.odin) and Fama-MacBeth
// regressions (analytics/ML/fama_macbeth.odin):
// - Grinblatt & Han (2005) eq. (25) reference price and capital gains overhang against
//   numpy with the explicit products (LCG prices and turnovers, window 52)
// - Odean (1998) PGR / PLR on a hand-worked log, and on 84 generated trades (3
//   accounts, 4 assets, 40 days) against a separate Python implementation, sale days
//   only (Odean) and every holding day
// - Fama-MacBeth estimates, standard errors, Newey-West(3) standard errors against
//   numpy / statsmodels OLS(cov_type="HAC")

@(private = "file")
_DP_TRADES := [84]bfin.Trade {
	{0, 2, 0, 3.0, 23.8372},
	{0, 1, 4, 2.0, 28.1923},
	{0, 1, 5, 3.0, 28.4594},
	{0, 2, 6, 5.0, 24.7026},
	{0, 1, 8, 4.0, 28.081},
	{0, 0, 11, -2.0, 18.7532},
	{0, 3, 12, 5.0, 25.6225},
	{0, 1, 14, 5.0, 30.0503},
	{0, 3, 18, 3.0, 26.1877},
	{0, 3, 23, 4.0, 28.7986},
	{0, 2, 25, 1.0, 27.5099},
	{0, 2, 26, 4.0, 28.1102},
	{0, 3, 28, -3.0, 30.1696},
	{0, 1, 36, 4.0, 30.0961},
	{0, 2, 37, 5.0, 29.4988},
	{1, 1, 0, 2.0, 27.9474},
	{1, 1, 4, 5.0, 27.7569},
	{1, 3, 5, 3.0, 27.3875},
	{1, 2, 10, 4.0, 24.9504},
	{1, 3, 12, 1.0, 25.5954},
	{1, 1, 13, -4.0, 30.2147},
	{1, 2, 19, 4.0, 25.0387},
	{1, 2, 22, -3.0, 26.5572},
	{1, 0, 26, 4.0, 17.5283},
	{1, 1, 27, 3.0, 29.6859},
	{1, 1, 32, -3.0, 31.0432},
	{1, 2, 34, -3.0, 29.0033},
	{1, 2, 36, 1.0, 29.3205},
	{1, 0, 38, 2.0, 16.1297},
	{1, 3, 39, 1.0, 32.2793},
	{2, 3, 7, 2.0, 26.6198},
	{2, 3, 8, -2.0, 26.8828},
	{2, 2, 11, 4.0, 24.9779},
	{2, 3, 16, 5.0, 26.9242},
	{2, 2, 22, 2.0, 26.4948},
	{2, 2, 24, -1.0, 27.085},
	{2, 2, 25, 2.0, 27.8319},
	{2, 1, 31, 3.0, 32.0179},
	{2, 2, 32, 2.0, 30.9399},
	{2, 1, 33, 5.0, 30.2194},
	{2, 3, 35, 4.0, 31.8133},
	{2, 1, 39, 1.0, 29.9911},
	{0, 3, 1, 2.0, 26.6988},
	{0, 3, 4, -2.0, 28.1059},
	{0, 3, 5, 5.0, 27.5718},
	{0, 1, 7, -4.0, 28.2933},
	{0, 0, 10, 2.0, 18.8426},
	{0, 1, 11, -1.0, 28.1716},
	{0, 1, 13, 5.0, 30.1532},
	{0, 2, 15, -3.0, 24.286},
	{0, 3, 19, 3.0, 26.3937},
	{0, 1, 24, 5.0, 29.888},
	{0, 3, 25, 5.0, 29.7045},
	{0, 2, 28, 3.0, 28.8613},
	{0, 0, 29, 2.0, 17.2789},
	{0, 3, 36, 4.0, 31.7604},
	{0, 2, 39, -2.0, 29.6058},
	{1, 1, 1, -2.0, 28.2637},
	{1, 0, 5, 4.0, 20.3348},
	{1, 3, 9, -3.0, 26.3711},
	{1, 0, 11, 2.0, 18.8912},
	{1, 0, 13, 1.0, 18.8521},
	{1, 2, 16, 1.0, 25.0281},
	{1, 1, 20, -1.0, 27.7791},
	{1, 0, 24, -3.0, 18.7821},
	{1, 2, 26, 2.0, 28.1111},
	{1, 0, 28, -4.0, 17.3628},
	{1, 2, 32, -4.0, 30.56},
	{1, 1, 36, 5.0, 30.2539},
	{1, 1, 37, 5.0, 29.693},
	{1, 3, 38, -1.0, 32.3101},
	{2, 0, 6, 2.0, 20.1658},
	{2, 0, 8, 1.0, 19.2786},
	{2, 0, 11, -2.0, 18.7446},
	{2, 2, 12, 4.0, 24.2867},
	{2, 0, 19, 4.0, 17.6547},
	{2, 1, 24, 5.0, 29.8131},
	{2, 0, 25, -4.0, 18.3289},
	{2, 0, 28, 5.0, 17.4677},
	{2, 3, 31, 4.0, 31.1481},
	{2, 0, 33, 1.0, 16.6102},
	{2, 2, 33, -2.0, 29.5638},
	{2, 2, 38, 2.0, 30.1763},
	{2, 2, 39, 5.0, 29.8109},
}

@(private = "file")
_dp_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

@(private = "file")
_rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}

disposition_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Disposition Effect Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	x: u64 = 11

	// 1. Grinblatt-Han reference price and overhang
	T, W :: 200, 52
	P := make([]f64, T)
	V := make([]f64, T)
	defer {delete(P); delete(V)}
	p := 30.0
	for t in 0 ..< T {
		p *= math.exp(0.05 * (_dp_u(&x) - 0.5))
		P[t] = p
		V[t] = 0.005 + 0.03 * _dp_u(&x)
	}
	V[150] = 1.5
	R := bfin.reference_price(P, V, W)
	g := bfin.capital_gains_overhang(P, V, W)
	defer {delete(R); delete(g)}
	n, r1, r2, g1, g2 := 0, 0.0, 0.0, 0.0, 0.0
	for v, t in R {
		if math.is_nan(v) {continue}
		n += 1
		r1 += v
		r2 += v * v
		g1 += g[t]
		g2 += g[t] * g[t]
	}
	e1 := max(
		_rel(r1, 2778.1909705125954),
		_rel(r2, 78097.9542351845),
		_rel(g1, -3.6988789765668564),
		_rel(g2, 0.25254328806417237),
	)
	e1 = max(
		e1,
		_rel(R[52], 30.591963432496932),
		_rel(g[52], -0.10451455228138248),
		_rel(R[120], 27.496115969117795),
		_rel(g[150], -0.002698475450625687),
	)
	expect(
		&passed,
		&total,
		n == 99 && e1 < 1e-12 && math.is_nan(R[51]) && math.is_nan(R[151]),
		fmt.tprintf(
			"reference price (eq. 25, window 52) and overhang = numpy with explicit products: %d values (NaN in warm-up and while a turnover > 1 is in the window), relative error %.1e",
			n,
			e1,
		),
	)

	// 2. Odean: a hand-worked log. Account 0 buys A 10, B 20; day 2 sells A at 12 (RG),
	// B closes 18 (PL); buys C 5; day 4 sells B at 21 (RG), C closes 4 (PL). Account 1
	// buys A, B, C; day 1 sells C at 4 (RL), A 11 (PG), B 19 (PL); day 3 sells A at 12
	// (RG), B 19 (PL). RG 3, RL 1, PG 1, PL 4: PGR 0.75, PLR 0.2
	hp := [][]f64{{10, 11, 12, 12, 13}, {20, 19, 18, 19, 21}, {5, 5, 5, 5, 4}}
	ht := []bfin.Trade {
		{0, 0, 0, 100, 10},
		{0, 1, 0, 50, 20},
		{0, 0, 2, -100, 12},
		{0, 2, 3, 10, 5},
		{0, 1, 4, -50, 21},
		{1, 0, 0, 10, 10},
		{1, 1, 0, 10, 20},
		{1, 2, 0, 10, 5},
		{1, 2, 1, -10, 4},
		{1, 0, 3, -10, 12},
	}
	hc := bfin.odean_disposition(ht, hp)
	se := math.sqrt(0.75 * 0.25 / 4 + 0.2 * 0.8 / 5)
	expect(
		&passed,
		&total,
		hc.realized_gains == 3 &&
		hc.realized_losses == 1 &&
		hc.paper_gains == 1 &&
		hc.paper_losses == 4 &&
		abs(hc.pgr - 0.75) < 1e-15 &&
		abs(hc.plr - 0.2) < 1e-15 &&
		abs(hc.t_stat - 0.55 / se) < 1e-12,
		fmt.tprintf(
			"PGR / PLR on a hand-worked log: %.2f / %.2f (t %.3f)",
			hc.pgr,
			hc.plr,
			hc.t_stat,
		),
	)

	// 3. Odean on the generated log (3 accounts, 4 assets, 40 days, partial sales,
	// repeated purchases, out-of-order input)
	NA, ND :: 4, 40
	prices := make([][]f64, NA)
	defer {for r in prices {delete(r)}; delete(prices)}
	for a in 0 ..< NA {
		prices[a] = make([]f64, ND)
		q := 20 + 20 * _dp_u(&x)
		for d in 0 ..< ND {
			q *= math.exp(0.08 * (_dp_u(&x) - 0.5))
			prices[a][d] = q
		}
	}
	cs := bfin.odean_disposition(_DP_TRADES[:], prices)
	ca := bfin.odean_disposition(_DP_TRADES[:], prices, sale_days_only = false)
	ok3 :=
		cs.realized_gains == 16 &&
		cs.realized_losses == 7 &&
		cs.paper_gains == 27 &&
		cs.paper_losses == 16 &&
		_rel(cs.t_stat, 0.5599214319414559) < 1e-12 &&
		ca.paper_gains == 192 &&
		ca.paper_losses == 125 &&
		_rel(ca.t_stat, 0.889311111856069) < 1e-12
	expect(
		&passed,
		&total,
		ok3,
		fmt.tprintf(
			"84 generated trades = a separate Python implementation: sale days RG %d RL %d PG %d PL %d (PGR %.3f, PLR %.3f); every holding day PG %d PL %d",
			cs.realized_gains,
			cs.realized_losses,
			cs.paper_gains,
			cs.paper_losses,
			cs.pgr,
			cs.plr,
			ca.paper_gains,
			ca.paper_losses,
		),
	)

	// 4. Fama-MacBeth: 30 periods of 25 assets, constant + 2 regressors
	TP, NS :: 30, 25
	x = 99 // the reference draws the trade log in between: restart the generator
	Xs := make([]l.Matrix(f64), TP)
	ys := make([][]f64, TP)
	defer {
		for t in 0 ..< TP {l.matrix_free(&Xs[t]); delete(ys[t])}
		delete(Xs)
		delete(ys)
	}
	for t in 0 ..< TP {
		Xs[t] = l.matrix_new(f64, NS, 3)
		ys[t] = make([]f64, NS)
		for i in 0 ..< NS {
			x1, x2 := _dp_u(&x) - 0.5, _dp_u(&x) - 0.5
			Xs[t].data[i * 3 + 0], Xs[t].data[i * 3 + 1], Xs[t].data[i * 3 + 2] = 1, x1, x2
			ys[t][i] = 0.01 + 0.05 * x1 - 0.02 * x2 + 0.03 * (_dp_u(&x) - 0.5)
		}
	}
	fm := ml.fama_macbeth(Xs, ys)
	fmn := ml.fama_macbeth(Xs, ys, nw_lags = 3)
	defer {ml.fama_macbeth_result_free(&fm); ml.fama_macbeth_result_free(&fmn)}
	want_c := [3]f64{0.009943849921891488, 0.05063911241278001, -0.019527256062831042}
	want_se := [3]f64{0.0002599149590936295, 0.0010059786836992804, 0.0010745465931300835}
	want_nw := [3]f64{0.00018573978599907134, 0.0007015139962421558, 0.0010913411901622826}
	e4 := 0.0
	for j in 0 ..< 3 {e4 = max(e4, _rel(fm.coef[j], want_c[j]), _rel(fm.std_err[j], want_se[j]), _rel(fmn.std_err[j], want_nw[j]), _rel(fmn.coef[j], want_c[j]))}
	expect(
		&passed,
		&total,
		fm.n_periods == 30 && e4 < 1e-10,
		fmt.tprintf(
			"Fama-MacBeth: coefficients, standard errors (sample sd / sqrt T) and Newey-West(3) standard errors = numpy / statsmodels HAC (relative error %.1e)",
			e4,
		),
	)

	fmt.printf("Disposition reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
