package tests

import bfin "../wotan/behavioral_finance"
import "core:fmt"
import "core:math"
import "core:mem"

// Anchoring (behavioral_finance/anchoring.odin) against pandas / numpy on an LCG panel
// (10 assets, 500 days, lookback 60, a NaN, a zero and a negative price, one asset
// delisted at day 450): PTH = p / p.rolling(60).max() (pandas), the top / bottom 30%
// portfolios at 21-day month ends (numpy lexsort), Jegadeesh-Titman overlapping
// returns, hold 3, skip 1 (an explicit loop). The behavioral Kalman filter with
// alpha = 0 against the local-level steady state.

// _NA marks an expected NaN
@(private = "file")
_NA :: -999.0

@(private = "file")
_AN_SAMPLES := [7][3]f64 {
	{0, 59, 1.0},
	{0, 300, 0.9616803524767987},
	{2, 159, _NA},
	{2, 160, 0.9715938442398759},
	{4, 259, _NA},
	{4, 260, 1.0},
	{9, 499, 0.949859025926044},
}

@(private = "file")
_AN_LEGS := [21][2]f64 {
	{_NA, _NA},
	{_NA, _NA},
	{_NA, _NA},
	{_NA, _NA},
	{0.051329661829012135, -0.0027229491805290107},
	{0.0019096978490456273, -0.06775254777481575},
	{0.01674399487296328, 0.0474445334604375},
	{0.02669899568560112, 0.04609903755712186},
	{-0.002905912195370243, -0.02200354643231378},
	{0.00461513026651348, -0.03947120279856679},
	{0.004582220335238609, -0.021882127795974596},
	{0.12222884046068046, -0.051497960446532855},
	{0.0020411634457650307, -0.0033704852705054023},
	{-0.00570132005211204, 0.046695255783988164},
	{-0.01699309673770392, 0.08675364683508942},
	{0.01897999529884362, -0.015111121791623279},
	{-0.02216491615148683, 0.038863835558587606},
	{0.019485951623035225, 0.04208969693476477},
	{0.016655037438798988, -0.046550679422597434},
	{-0.04069573900183726, -0.023712850270573124},
	{0.03678100445901597, 0.05145456489305018},
}

@(private = "file")
_an_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

@(private = "file")
_same :: proc(a, b: f64, tol: f64) -> bool {
	if b == _NA {return math.is_nan(a)}
	return abs(a - b) <= tol * max(1, abs(b))
}

anchoring_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Anchoring (52-Week High) Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	NA, T, L :: 10, 500, 60
	x: u64 = 77
	prices := make([][]f64, NA)
	defer {for p in prices {delete(p)}; delete(prices)}
	for a in 0 ..< NA {
		prices[a] = make([]f64, T)
		p := 50 + 50 * _an_u(&x)
		for t in 0 ..< T {
			if t > 0 {p *= math.exp(0.06 * (_an_u(&x) - 0.5))}
			prices[a][t] = p
		}
	}
	prices[2][100] = math.nan_f64()
	prices[4][200] = 0
	prices[5][10] = -1
	for t in 450 ..< T {prices[7][t] = math.nan_f64()}

	pth := bfin.compute_anchoring_signal(prices, L)
	defer bfin.compute_anchoring_signal_free(pth)
	n, s1, s2, mx := 0, 0.0, 0.0, 0.0
	for row in pth {
		for v in row {
			if math.is_nan(v) {continue}
			n += 1
			s1 += v
			s2 += v * v
			mx = max(mx, v)
		}
	}
	samples_ok := true
	for sm in _AN_SAMPLES {samples_ok &&= _same(pth[int(sm[0])][int(sm[1])], sm[2], 1e-15)}
	expect(
		&passed,
		&total,
		n == 4229 &&
		_same(s1, 3.837348412869262e+03, 1e-13) &&
		_same(s2, 3.505003845741364e+03, 1e-13) &&
		mx == 1 &&
		samples_ok,
		fmt.tprintf(
			"PTH = pandas p / p.rolling(60).max(): %d finite values, their sum and sum of squares, max %.0f (today in the window), NaN through invalid prices and warm-up",
			n,
			mx,
		),
	)

	// month ends every 21 days from the first full window
	days := make([dynamic]int)
	defer delete(days)
	for d := L - 1; d < T; d += 21 {append(&days, d)}
	Mn := len(days)
	W := make([][]f64, Mn)
	R := make([][]f64, Mn)
	defer {
		for i in 0 ..< Mn {delete(W[i]); delete(R[i])}
		delete(W)
		delete(R)
	}
	col := make([]f64, NA)
	defer delete(col)
	wsum := 0.0
	for d, m in days {
		for a in 0 ..< NA {col[a] = pth[a][d]}
		W[m] = bfin.anchoring_portfolio(col, 0.3)
		for wv, a in W[m] {wsum += wv * f64(a + 1)}
		R[m] = make([]f64, NA)
		for a in 0 ..< NA {
			R[m][a] = prices[a][d] / prices[a][days[m - 1]] - 1 if m > 0 else math.nan_f64()
		}
	}
	legs := bfin.overlapping_strategy_returns(W, R, hold = 3, skip = 1)
	defer bfin.overlapping_returns_free(&legs)
	legs_ok := Mn == 21 && _same(wsum, -21.499999999999996, 1e-14)
	for lg, s in _AN_LEGS {
		legs_ok &&= _same(legs.long[s], lg[0], 1e-14) && _same(legs.short[s], lg[1], 1e-14)
		legs_ok &&= lg[0] == _NA || _same(legs.spread[s], lg[0] - lg[1], 1e-14)
	}
	expect(
		&passed,
		&total,
		legs_ok,
		fmt.tprintf(
			"top / bottom 30%% portfolios on %d month ends and Jegadeesh-Titman (hold 3, skip 1) long / short / spread returns = numpy",
			Mn,
		),
	)

	// Kalman: alpha = 0 is the local-level filter; its gain converges to the steady
	// state K = Pp / (Pp + R), Pp = (q + sqrt(q^2 + 4 q R)) / 2
	q, Rv := 0.01, 0.1
	obs := make([]f64, 200)
	defer delete(obs)
	for &o, i in obs {o = math.sin(f64(i) / 7) + 0.3 * (_an_u(&x) - 0.5)}
	est, gain, vars := bfin.behavioral_kalman_filter(obs, 0, 0, q, Rv)
	defer {delete(est); delete(gain); delete(vars)}
	Pp := (q + math.sqrt(q * q + 4 * q * Rv)) / 2
	Kss := Pp / (Pp + Rv)
	// with alpha = 1 and an anchor above every observation the estimate never rises
	est1, gain1, vars1 := bfin.behavioral_kalman_filter(obs, 10, 1, q, Rv)
	defer {delete(est1); delete(gain1); delete(vars1)}
	never_up := true
	for i in 1 ..< len(est1) {never_up &&= est1[i] <= est1[i - 1] + 1e-15}
	// per-step anchors equal to the constant give the same filter
	anchors := make([]f64, len(obs))
	defer delete(anchors)
	for &a in anchors {a = 0.2}
	ea, _, _ := bfin.behavioral_kalman_filter(obs, 0.2, 0.5, q, Rv)
	eb, _, _ := bfin.behavioral_kalman_filter(obs, 99, 0.5, q, Rv, anchors)
	defer {delete(ea); delete(eb)}
	same_anchor := true
	for v, i in ea {same_anchor &&= v == eb[i]}
	expect(
		&passed,
		&total,
		abs(gain[199] - Kss) < 1e-12 &&
		abs(vars[199] - (1 - Kss) * Pp) < 1e-12 &&
		never_up &&
		same_anchor,
		fmt.tprintf(
			"Kalman: alpha 0 reaches the steady-state gain %.6f and variance (Joseph form = (1 - K) P at the optimal gain); alpha 1 below the anchor never moves up; per-step anchors = the constant",
			Kss,
		),
	)

	fmt.printf("Anchoring reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
