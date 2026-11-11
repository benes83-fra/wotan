package tests

import bfin "../wotan/behavioral_finance"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Herding (behavioral_finance/herding.odin)
//   herding_reference_test: CSSD / CSAD on a hand-worked period and on an LCG panel
//     (30 assets, 300 periods, gaps, an infinite return, a period with one valid
//     return; equally weighted and given market) = numpy; Christie & Huang (tails 5%
//     and 1%) and Chang, Cheng & Khorana (all, up and down markets) regressions with
//     classical and Newey-West(4) standard errors = statsmodels OLS / HAC; no tail
//     periods -> NaN
//   herding_test: A. simulated daily markets, 100 stocks, 2500 days (market volatility
//     1% a day, 3% on 10% of days; betas 0.5 - 1.5; idiosyncratic 1.5%): rational
//     (CAPM): dispersion grows in extreme markets (CH b_L, b_U > 0, CCK g2 > 0); strong
//     herding (deviations from the market scaled by exp(-(R_m / 5%)^2)): CH and CCK
//     both detect it (b < 0, g2 < 0); mild herding (8%): dispersion still rises with
//     |R_m| but at a decreasing rate (CCK g1 > 0, g2 < 0), CH weaker (CCK's argument for
//     their test). On 8 seeds: rational CH t 8.1 - 12.2, CCK g2 t 9.7 - 24; strong CH t
//     -9.9 to -14.6, CCK g2 t -3.1 to -14.5 (dispersion collapses in the tails, which
//     the quadratic fits poorly, hence the bound t < -2); mild CCK g1 t 4.7 - 7.9, g2 t
//     -8.1 to -12.0 (up / down >= 4.0 in absolute value), CH t +1.9 to -3.1.
//     B. the 24 large caps, daily, 10 years (printed)

@(private = "file")
_hd_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

@(private = "file")
_hd_rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}

// max relative error of a regression against (coef, se, r2, n)
@(private = "file")
_hd_err :: proc(r: bfin.Herding_Regression, c, se: [3]f64, r2: f64, n: int) -> f64 {
	if r.n_obs != n {return math.inf_f64(1)}
	e := _hd_rel(r.r2, r2)
	for j in 0 ..< 3 {e = max(e, _hd_rel(r.coef[j], c[j]), _hd_rel(r.std_err[j], se[j]))}
	return e
}

herding_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Herding Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// 1. hand-worked: returns 1, 2, 3, 6% -> R_m 3%, deviations -2, -1, 0, 3%:
	// CSSD = sqrt(14 / 3)%, CSAD = 1.5%
	hand := [][]f64{{0.01}, {0.02}, {0.03}, {0.06}}
	hd := bfin.cross_sectional_dispersion(hand)
	defer bfin.herding_dispersion_free(&hd)
	expect(
		&passed,
		&total,
		abs(hd.market[0] - 0.03) < 1e-16 &&
		abs(hd.cssd[0] - 0.01 * math.sqrt(14.0 / 3)) < 1e-16 &&
		abs(hd.csad[0] - 0.015) < 1e-16,
		fmt.tprintf("hand-worked period: CSSD %.6f, CSAD %.4f", hd.cssd[0], hd.csad[0]),
	)

	// 2. LCG panel = numpy
	x: u64 = 7
	N, T :: 30, 300
	beta := make([]f64, N)
	R := make([][]f64, N)
	vm := make([]f64, T)
	defer {for r in R {delete(r)}; delete(R); delete(beta); delete(vm)}
	for i in 0 ..< N {beta[i] = 0.5 + _hd_u(&x); R[i] = make([]f64, T)}
	for t in 0 ..< T {
		m := 0.06 * (_hd_u(&x) - 0.5)
		vm[t] = m + 0.002 * (_hd_u(&x) - 0.5)
		for i in 0 ..< N {R[i][t] = beta[i] * m + 0.04 * (_hd_u(&x) - 0.5)}
	}
	for t in 10 ..< 20 {R[3][t] = math.nan_f64()}
	for i in 1 ..< N {R[i][50] = math.nan_f64()}
	R[0][50] = 0.01
	R[7][200] = math.inf_f64(1)
	d := bfin.cross_sectional_dispersion(R)
	dv := bfin.cross_sectional_dispersion(R, vm)
	defer {bfin.herding_dispersion_free(&d); bfin.herding_dispersion_free(&dv)}
	nan, sm, ss, sa, svs, sva := 0, 0.0, 0.0, 0.0, 0.0, 0.0
	for t in 0 ..< T {
		if math.is_nan(d.cssd[t]) {nan += 1; continue}
		sm += d.market[t]
		ss += d.cssd[t]
		sa += d.csad[t]
		svs += dv.cssd[t]
		sva += dv.csad[t]
	}
	e2 := max(
		_hd_rel(sm, 0.132493933113835),
		_hd_rel(ss, 3.7929762584321374),
		_hd_rel(sa, 3.1448945960380876),
		_hd_rel(svs, 3.85648759010643),
		_hd_rel(sva, 3.205270650266655),
	)
	e2 = max(
		e2,
		_hd_rel(d.market[15], 0.017551911590240266),
		_hd_rel(d.cssd[15], 0.01152765977318359),
		_hd_rel(d.csad[15], 0.009675393796990938),
		_hd_rel(d.cssd[200], 0.012907066896950948),
		_hd_rel(d.csad[200], 0.011053307694307509),
	)
	expect(
		&passed,
		&total,
		nan == 1 &&
		math.is_nan(d.csad[50]) &&
		d.n_assets[15] == 29 &&
		d.n_assets[200] == 29 &&
		e2 < 1e-12,
		fmt.tprintf(
			"CSSD / CSAD on 30 x 300 with gaps = numpy, equally weighted and given market (relative error %.1e; NaN with one valid return)",
			e2,
		),
	)

	// 3. Christie & Huang = statsmodels
	ch := [4]bfin.Herding_Regression {
		bfin.christie_huang_regression(d.cssd, d.market, 0.05),
		bfin.christie_huang_regression(d.cssd, d.market, 0.05, nw_lags = 4),
		bfin.christie_huang_regression(d.cssd, d.market, 0.01),
		bfin.christie_huang_regression(d.cssd, d.market, 0.01, nw_lags = 4),
	}
	e3 := max(
		_hd_err(
			ch[0],
			{0.012504052262643998, 0.0017985735461174784, 0.0020774715897097238},
			{9.3278131631983e-05, 0.0004208605813105502, 0.0004208605813105502},
			0.12076274640843809,
			299,
		),
		_hd_err(
			ch[1],
			{0.012504052262643998, 0.0017985735461174784, 0.0020774715897097238},
			{9.350164352256855e-05, 0.0004057632012665582, 0.00044207649846925525},
			0.12076274640843809,
			299,
		),
		_hd_err(
			ch[2],
			{0.012678074418532834, 0.0007631742231949238, 0.0003528294222151806},
			{9.526160537078174e-05, 0.0011608634107792577, 0.001160863410779258},
			0.0017601083316300947,
			299,
		),
		_hd_err(
			ch[3],
			{0.012678074418532834, 0.0007631742231949238, 0.0003528294222151806},
			{9.82630257813174e-05, 0.0017815488537342725, 0.00011424512376783161},
			0.0017601083316300947,
			299,
		),
	)
	expect(
		&passed,
		&total,
		e3 < 1e-9,
		fmt.tprintf(
			"Christie-Huang, tails 5%% and 1%%, classical and Newey-West(4) = statsmodels (relative error %.1e; 5%%: b_L %.4f, b_U %.4f)",
			e3,
			ch[0].coef[1],
			ch[0].coef[2],
		),
	)

	// 4. Chang, Cheng & Khorana = statsmodels
	ck := [6]bfin.Herding_Regression {
		bfin.cck_regression(d.csad, d.market),
		bfin.cck_regression(d.csad, d.market, nw_lags = 4),
		bfin.cck_regression(d.csad, d.market, .Up),
		bfin.cck_regression(d.csad, d.market, .Up, nw_lags = 4),
		bfin.cck_regression(d.csad, d.market, .Down),
		bfin.cck_regression(d.csad, d.market, .Down, nw_lags = 4),
	}
	e4 := max(
		_hd_err(
			ck[0],
			{0.009795638312670307, 0.004655778707328965, 1.8869614525732463},
			{0.0002493764852301311, 0.034927733064780205, 1.0382505651607215},
			0.1830280812711832,
			299,
		),
		_hd_err(
			ck[1],
			{0.009795638312670307, 0.004655778707328965, 1.8869614525732463},
			{0.00024956453748192986, 0.03827654779145859, 1.185907679151692},
			0.1830280812711832,
			299,
		),
		_hd_err(
			ck[2],
			{0.010016918134639393, -0.010743640790249885, 2.2153603878369816},
			{0.0003455492223858021, 0.04720896082835729, 1.3917697912206988},
			0.16802900482117966,
			154,
		),
		_hd_err(
			ck[3],
			{0.010016918134639393, -0.010743640790249885, 2.2153603878369816},
			{0.0003809069849128628, 0.05871219358485077, 1.8397555226093303},
			0.16802900482117966,
			154,
		),
		_hd_err(
			ck[4],
			{0.009578537470924908, 0.018626093054541666, 1.602003525746341},
			{0.0003627892895058151, 0.052260819924975555, 1.5668325581158218},
			0.20234275941920254,
			145,
		),
		_hd_err(
			ck[5],
			{0.009578537470924908, 0.018626093054541666, 1.602003525746341},
			{0.0003334912484488391, 0.05301299012821975, 1.6044124685807242},
			0.20234275941920254,
			145,
		),
	)
	expect(
		&passed,
		&total,
		e4 < 1e-9,
		fmt.tprintf(
			"Chang-Cheng-Khorana, all / up / down markets, classical and Newey-West(4) = statsmodels (relative error %.1e; %d up, %d down periods)",
			e4,
			ck[2].n_obs,
			ck[4].n_obs,
		),
	)

	// 5. a tail with no periods (floor(0.001 * 299) = 0): NaN, not a singular fit
	none := bfin.christie_huang_regression(d.cssd, d.market, 0.001)
	expect(
		&passed,
		&total,
		math.is_nan(none.coef[1]) && math.is_nan(none.t_stat[2]) && none.n_obs == 299,
		"Christie-Huang with an empty tail: NaN coefficients",
	)

	fmt.printf("Herding reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// simulated market: herd = 0 rational, else deviations from the market scaled by
// exp(-(m / herd)^2); returns CH (5% tails) and CCK (all, up, down), Newey-West(5)
@(private = "file")
_hd_market :: proc(herd: f64) -> (ch: bfin.Herding_Regression, ck: [3]bfin.Herding_Regression) {
	N, T :: 100, 2500
	beta := make([]f64, N, context.temp_allocator)
	R := make([][]f64, N, context.temp_allocator)
	defer {for r in R {delete(r, context.temp_allocator)}; delete(R, context.temp_allocator)
		delete(beta, context.temp_allocator)}
	for i in 0 ..< N {beta[i] = 0.5 + rand.float64(); R[i] = make([]f64, T, context.temp_allocator)}
	for t in 0 ..< T {
		vol := 0.03 if rand.float64() < 0.1 else 0.01
		m := vol * rand.float64_normal(0, 1)
		s := math.exp(-(m / herd) * (m / herd)) if herd > 0 else 1
		for i in 0 ..< N {R[i][t] = m + s * ((beta[i] - 1) * m + 0.015 * rand.float64_normal(0, 1))}
	}
	d := bfin.cross_sectional_dispersion(R, allocator = context.temp_allocator)
	defer bfin.herding_dispersion_free(&d)
	ch = bfin.christie_huang_regression(d.cssd, d.market, 0.05, nw_lags = 5)
	for st, k in ([]bfin.Herding_Market_State{.All, .Up, .Down}) {
		ck[k] = bfin.cck_regression(d.csad, d.market, st, nw_lags = 5)
	}
	return
}

herding_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Herding ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. simulated markets
	rch, rck := _hd_market(0)
	expect(
		&passed,
		&total,
		rch.t_stat[1] > 3 && rch.t_stat[2] > 3 && rck[0].t_stat[2] > 3,
		fmt.tprintf(
			"rational market: dispersion rises in the tails (CH b_L t %.1f, b_U t %.1f), CCK g2 %.2f (t %.1f) >= 0",
			rch.t_stat[1],
			rch.t_stat[2],
			rck[0].coef[2],
			rck[0].t_stat[2],
		),
	)
	sch, sck := _hd_market(0.05)
	expect(
		&passed,
		&total,
		sch.t_stat[1] < -3 && sch.t_stat[2] < -3 && sck[0].t_stat[2] < -2,
		fmt.tprintf(
			"strong herding: CH b_L t %.1f, b_U t %.1f; CCK g2 %.2f (t %.1f)",
			sch.t_stat[1],
			sch.t_stat[2],
			sck[0].coef[2],
			sck[0].t_stat[2],
		),
	)
	mch, mck := _hd_market(0.08)
	expect(
		&passed,
		&total,
		mck[0].t_stat[1] > 3 &&
		mck[0].t_stat[2] < -3 &&
		mck[1].t_stat[2] < -3 &&
		mck[2].t_stat[2] < -3,
		fmt.tprintf(
			"mild herding: CCK g1 %.3f (t %.1f), g2 %.2f (t %.1f; up t %.1f, down t %.1f); CH b_L t %.1f, b_U t %.1f",
			mck[0].coef[1],
			mck[0].t_stat[1],
			mck[0].coef[2],
			mck[0].t_stat[2],
			mck[1].t_stat[2],
			mck[2].t_stat[2],
			mch.t_stat[1],
			mch.t_stat[2],
		),
	)

	// B. the 24 large caps, daily
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
		R := make([][]f64, NS)
		defer {for r in R {delete(r)}; delete(R)}
		for i in 0 ..< NS {
			R[i] = make([]f64, ND)
			for t in 0 ..< ND {R[i][t] = DP[(t + 1) * NS + i] / DP[t * NS + i] - 1}
		}
		d := bfin.cross_sectional_dispersion(R)
		defer bfin.herding_dispersion_free(&d)
		c5 := bfin.christie_huang_regression(d.cssd, d.market, 0.05, nw_lags = 5)
		c1 := bfin.christie_huang_regression(d.cssd, d.market, 0.01, nw_lags = 5)
		fmt.printf(
			"Large caps, %d days: CH 5%% b_L %.4f (t %.2f), b_U %.4f (t %.2f); 1%% b_L t %.2f, b_U t %.2f\n",
			c5.n_obs,
			c5.coef[1],
			c5.t_stat[1],
			c5.coef[2],
			c5.t_stat[2],
			c1.t_stat[1],
			c1.t_stat[2],
		)
		for st in bfin.Herding_Market_State {
			r := bfin.cck_regression(d.csad, d.market, st, nw_lags = 5)
			fmt.printf(
				"  CCK %v (%d days): g1 %.4f (t %.2f), g2 %.3f (t %.2f)\n",
				st,
				r.n_obs,
				r.coef[1],
				r.t_stat[1],
				r.coef[2],
				r.t_stat[2],
			)
		}
	}

	fmt.printf("Herding: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
