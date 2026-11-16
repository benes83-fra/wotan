package tests

import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Regression tests for wotan/finance issues found while writing the finance
// guides. Options and simulation: the binomial and trinomial trees read the
// level-2 node values for gamma and theta inside the backward induction loop,
// which starts at level n − 1, so with n_steps ≤ 2 (n = 2) gamma came out as 0
// and theta as −price / (2Δt). black_scholes_price multiplied by a 1 × 1
// constant, so a batch of options (n × 1 inputs) panicked in tensor_mul; and
// its intermediates computed from inputs without gradient (e^{−rT}, √T for
// fixed r, T), or all of them without any gradient, were not linked into the
// graph and leaked. Volatility and exotics: the FX normal CDF was
// ½(1 + erf(x/√2)), which cancels in the left tail, so far out-of-the-money
// Garman-Kohlhagen prices came out as 0 or negative; fx_digital_call_price
// divided by σ, so σ = 0 with the forward at the strike gave NaN;
// calibrate_sabr said β could be fixed by equal bounds but hard-coded its
// bounds (now fixed_beta). Fixed income: the convertible lattice ended at
// conversion_end, so when conversion stopped before maturity the redemption
// and later coupons were paid early (a 2030 bond convertible until 2028 was
// worth 101.78 with a worthless share, above its 92.60 straight value; QuantLib
// does the same); cpi_fixing back-cast months before the first published
// fixing from the curve instead of returning NaN. Rates and credit: swaps that
// started before the curve date counted their paid coupons (at a discount
// factor of 1) and projected the floating coupon in progress from the curve;
// the Hull-White and G2++ caplet / floorlet tensors and the calibration losses
// built graphs from constants only, which were never freed (62 allocations per
// strip, 53 000 per calibration). Portfolio construction: the projection onto
// constraints with group limits (Dykstra) stopped when the point did not move
// for a sweep, which happens while the increments still drift after a long
// gradient step, so the constrained optimizers returned weights breaking a
// group limit (min variance) or stopped at the start (max Sharpe, utility).
// Later checks: unshifted Sobol' normals started at the origin (z ≈ −6.4 on
// path 0); compute_greeks, the finite-difference and the Longstaff-Schwartz
// pricers had no dividend yield; context_submit_order dropped limit and stop
// orders; VaR / ES came with two sign conventions and value_at_risk(.GARCH)
// was an unconditional normal VaR; the GARCH backtests were in sample;
// CDS rpv01 was per 1M whatever the notional; the PDPM payoff procedures took
// unused spot / maturity arguments, and four PDPM / market-data types had no
// free procedure; linear-zero curves held the zero rate flat beyond the last
// pillar (QuantLib carries the last instantaneous forward); the callable and
// puttable bond pricers took a curve callback without user data.

// limit / stop order script state (finance_fixes_test)
@(private = "file")
_FF_Order :: struct {
	id:     int,
	status: fin.OrderStatus,
}
@(private = "file")
_ff_ord: [6]_FF_Order
@(private = "file")
_ff_cancel: [2]bool

@(private = "file")
_ff_submit :: proc(ctx: ^fin.StrategyContext, o: fin.Order) -> _FF_Order {
	id, st := fin.context_submit_order(ctx, o)
	return {id, st}
}

@(private = "file")
ff_vec :: proc(v: []f64, grad: bool) -> ^t.Tensor {
	m := l.matrix_new(f64, len(v), 1)
	copy(m.data, v)
	return t.tensor_new(m, grad)
}

finance_fixes_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== finance regression tests ===")
	passed, total := 0, 0
	report :: proc(name: string, ok: bool, passed, total: ^int) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	{ 	// 2-step trees: Greeks from the expiry level
		// S = 100, K = 105, T = 1, r = 5%, σ = 30%, q = 4%; level 2 is the expiry, so
		// the middle node (price S) is worth the payoff 5 and theta = (5 − V)/(2Δt)
		b := fin.american_option_binomial(100, 105, 1, 0.05, 0.3, 0.04, .Put, 2, allocator)
		tr := fin.american_option_trinomial(100, 105, 1, 0.05, 0.3, 0.04, .Put, 2, allocator)
		// QuantLib 1.44 BinomialCRRVanillaEngine, 2 steps: gamma 0.0207131506156
		good :=
			math.abs(b.gamma - 0.0207131506156) < 1e-12 &&
			math.abs(b.theta - (5 - b.price)) < 1e-12 &&
			tr.gamma > 0 &&
			math.abs(tr.theta - (5 - tr.price)) < 1e-12
		report(
			fmt.tprintf(
				"2-step trees: binomial gamma %.10f theta %.6f, trinomial gamma %.6f theta %.6f",
				b.gamma,
				b.theta,
				tr.gamma,
				tr.theta,
			),
			good,
			&passed,
			&total,
		)
	}

	{ 	// FX: far out-of-the-money Garman-Kohlhagen prices keep their precision
		// S = 1.1, T = 0.5, r_d = 3%, r_f = 1%, σ = 10%; exact values (mpmath, 40 digits)
		call := fin.fx_gk_price(1.1, 2.0, 0.5, 0.03, 0.01, 0.1, .Call)
		put := fin.fx_gk_price(1.1, 0.605, 0.5, 0.03, 0.01, 0.1, .Put)
		far := fin.fx_gk_price(1.1, 1.21 / 3.0, 0.5, 0.03, 0.01, 0.1, .Put)
		good :=
			math.abs(call / 5.65366489261909e-19 - 1) < 1e-9 &&
			math.abs(put / 2.66807118060116e-20 - 1) < 1e-9 &&
			math.abs(far / 2.28047185159941e-49 - 1) < 1e-9
		report(
			fmt.tprintf("FX far out of the money: call %.6e, put %.6e, %.6e", call, put, far),
			good,
			&passed,
			&total,
		)
	}
	{ 	// FX digital without volatility: S_T is the forward
		at_fwd := fin.fx_digital_call_price(1.1, 1.1, 0.5, 0.02, 0.02, 0) // forward = strike
		above := fin.fx_digital_call_price(1.1, 1.0, 0.5, 0.03, 0.01, 0)
		good := at_fwd == 0 && math.abs(above - math.exp(-0.015)) < 1e-15
		report(
			fmt.tprintf(
				"FX digital at sigma = 0: %v at the forward, %.12f in the money",
				at_fwd,
				above,
			),
			good,
			&passed,
			&total,
		)
	}
	{ 	// SABR with β held fixed (calibrate_sabr's comment promised it; no way to do it)
		p := fin.SABR_Params {
			alpha = 1.3,
			beta  = 0.6,
			rho   = -0.35,
			nu    = 0.5,
		}
		F := 100 * math.exp(0.03)
		quotes: [7]fin.VolSurfacePoint
		for &q, i in quotes {
			K := 70 + 10 * f64(i)
			q = {
				strike      = K,
				expiry      = 1,
				implied_vol = fin.sabr_implied_vol(F, K, 1, p),
			}
		}
		r := fin.calibrate_sabr(quotes[:], 100, 0.03, allocator, fixed_beta = 0.6)
		good :=
			r.params.beta == 0.6 &&
			math.abs(r.params.alpha - 1.3) < 1e-8 &&
			math.abs(r.params.rho + 0.35) < 1e-8 &&
			math.abs(r.params.nu - 0.5) < 1e-8
		report(
			fmt.tprintf("calibrate_sabr with fixed_beta = 0.6: %v", r.params),
			good,
			&passed,
			&total,
		)
	}
	{ 	// black_scholes_price leaves nothing behind, with and without gradients
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.default_allocator())
		defer mem.tracking_allocator_destroy(&track)
		context.allocator = mem.tracking_allocator(&track)

		S, K, T, r, sigma :=
			ff_vec({100}, false),
			ff_vec({110}, false),
			ff_vec({0.5}, false),
			ff_vec({0.03}, false),
			ff_vec({0.25}, true)
		p := fin.black_scholes_price(S, K, T, r, sigma, .Put, context.allocator)
		t.tensor_backward(p)
		vega := sigma.grad.data[0]
		t.tensor_free_graph(p)
		with_grad := len(track.allocation_map) // the 5 inputs and sigma's gradient
		sigma.requires_grad = false
		q := fin.black_scholes_price(S, K, T, r, sigma, .Call, context.allocator)
		call := q.data.data[0]
		t.tensor_free_graph(q)
		without := len(track.allocation_map)
		for x in ([]^t.Tensor{S, K, T, r, sigma}) {t.tensor_free(x)}
		g := fin.compute_greeks(100, 110, 0.5, 0.03, 0.25, .Put)
		good :=
			with_grad == 11 &&
			without == 11 &&
			len(track.allocation_map) == 0 &&
			math.abs(vega - 100 * g.vega) < 1e-10 &&
			math.abs(call - fin.compute_greeks(100, 110, 0.5, 0.03, 0.25, .Call).price) < 1e-10
		report(
			fmt.tprintf(
				"black_scholes_price with constant K, T, r: live allocations %d, %d, then %d",
				with_grad,
				without,
				len(track.allocation_map),
			),
			good,
			&passed,
			&total,
		)
	}
	{ 	// convertible whose conversion right ends two years before maturity
		today := w.Date{2026, 11, 11}
		sch := fin.make_schedule(
			{2023, 3, 15},
			{2030, 3, 15},
			{6, .Months},
			{.TARGET},
			.Following,
			.Following,
			allocator = allocator,
		)
		bond := fin.DatedBond{100, 0.025, sch, .ACT_ACT_ICMA, .Following, 0}
		mkt := fin.ConvertibleMarket {
			spot          = 1e-6,
			rate          = 0.03,
			dividend      = 0.01,
			volatility    = 0.3,
			credit_spread = 0.02,
		}
		cb := fin.ConvertibleBond{bond, 3.5, {2023, 3, 15}, {2028, 3, 15}, nil}
		worthless := fin.price_convertible_bond(cb, today, mkt, allocator = allocator)
		// the straight bond: no conversion value, the full-length lattice
		straight := cb
		straight.conversion_ratio, straight.conversion_end = 1e-12, {2030, 3, 15}
		floor := fin.price_convertible_bond(straight, today, mkt, allocator = allocator)
		// no credit spread: equal to a 2028 convertible redeeming the 2028 value of
		// the later flows, 98.9926 (QuantLib BinomialCRRConvertibleEngine, 4001 steps)
		mkt.spot, mkt.credit_spread = 25, 0
		short_window := fin.price_convertible_bond(cb, today, mkt, 4001, allocator)
		good := math.abs(worthless - floor) < 1e-9 && math.abs(short_window - 107.027431) < 3e-3
		report(
			fmt.tprintf(
				"convertible, conversion until 2028 of a 2030 bond: S ~ 0 %.6f (straight bond %.6f), no credit spread %.6f (QuantLib equivalent 107.027431)",
				worthless,
				floor,
				short_window,
			),
			good,
			&passed,
			&total,
		)
		fin.free_schedule(&sch, allocator)
	}
	{ 	// CPI before the first published fixing: no back-cast from the curve
		months := [3]w.Date{{2026, 7, 1}, {2026, 8, 1}, {2026, 9, 1}}
		values := [3]f64{126.5, 126.8, 127.3}
		curve := fin.ZeroInflationCurve {
			base_date = {2026, 9, 1},
			day_count = .ACT_365,
			months    = {{2027, 9, 1}},
			rates     = {0.02},
		}
		index := fin.CPIIndex{months[:], values[:], &curve}
		before := fin.cpi_fixing(&index, {2026, 6, 1})
		after := fin.cpi_fixing(&index, {2027, 9, 1})
		good := math.is_nan(before) && math.abs(after - 127.3 * 1.02) < 1e-12
		report(
			fmt.tprintf(
				"CPI fixing before the first published month: %v; a year after the base %.6f",
				before,
				after,
			),
			good,
			&passed,
			&total,
		)
	}

	{ 	// seasoned swaps: paid flows left out, the coupon in progress from its fixing
		today := w.Date{2026, 11, 11}
		ois := fin.YieldCurve {
			tenors         = {1},
			zero_rates     = {0.02},
			reference_date = today,
		}
		e6 := fin.YieldCurve {
			tenors         = {1},
			zero_rates     = {0.025},
			reference_date = today,
		}
		e6_conv := fin.SwapConventions {
			calendar         = {.TARGET},
			settlement_days  = 2,
			fixed_tenor      = {1, .Years},
			fixed_convention = .ModifiedFollowing,
			fixed_day_count  = .THIRTY_360,
			float_tenor      = {6, .Months},
			float_convention = .ModifiedFollowing,
			float_day_count  = .ACT_360,
			fixing_days      = 2,
			indexed_coupons  = true,
		}
		sw := fin.make_dated_swap(
			e6_conv,
			{2024, 11, 13},
			{5, .Years},
			0.025,
			10e6,
			allocator = allocator,
		)
		missing := fin.price_dated_swap(sw, &ois, &e6, allocator).npv
		sw.current_fixing = 0.0215 // fixed on 11 May 2026
		euribor := fin.price_dated_swap(sw, &ois, &e6, allocator).npv
		fin.free_dated_swap(&sw, allocator)
		ois_conv := e6_conv
		ois_conv.fixed_day_count, ois_conv.float_tenor, ois_conv.payment_lag =
			.ACT_360, {1, .Years}, 1
		ois_conv.fixing_days, ois_conv.indexed_coupons = 0, false
		o := fin.make_dated_swap(
			ois_conv,
			{2026, 3, 2},
			{3, .Years},
			0.02,
			10e6,
			allocator = allocator,
		)
		growth := 1.0 // €STR at 2% every business day since 2 Mar 2026
		for d := (w.Date{2026, 3, 2}); w.date_less(d, today); {
			next := fin.calendar_advance({.TARGET}, d, 1, .Days)
			growth *= 1 + 0.02 * fin.year_fraction(.ACT_360, d, next)
			d = next
		}
		o.current_fixing = (growth - 1) / fin.year_fraction(.ACT_360, {2026, 3, 2}, today)
		overnight := fin.price_dated_swap(o, &ois, allocator = allocator).npv
		fin.free_dated_swap(&o, allocator)
		// time axis: started 0.1 years ago
		ts := fin.create_payer_swap(1, 0.03, 4.9)
		ts.effective_date = -0.1
		flat := fin.YieldCurve {
			tenors     = {1},
			zero_rates = {0.03},
		}
		ts_missing := fin.price_swap(ts, &flat, allocator).npv
		ts.current_fixing = 0.031
		ts_npv := fin.price_swap(ts, &flat, allocator).npv
		// current period: (0.031 − 0.03)·dcf paid at 0.15; then a forward-starting
		// swap from 0.15: floating DF(0.15) − DF(4.9) against 3% × its annuity
		dcf := 0.25 * 365.0 / 360.0
		ts_expected :=
			0.001 * dcf * math.exp(-0.03 * 0.15) +
			(math.exp(-0.03 * 0.15) - math.exp(-0.03 * 4.9)) -
			0.03 * fin._compute_swap_annuity(0.25, 0.15, 4.9, .ACT_360, &flat)
		good :=
			math.is_nan(missing) &&
			math.abs(euribor - -131244.220046) < 1e-5 &&
			math.abs(overnight - -278.191618) < 1e-5 &&
			math.is_nan(ts_missing) &&
			math.abs(ts_npv - ts_expected) < 1e-15
		report(
			fmt.tprintf(
				"seasoned swaps: Euribor without the fixing %v, with it %.6f (QuantLib -131244.220046); OIS in progress %.6f (QuantLib -278.191618); time axis %v / %.12f (%.12f)",
				missing,
				euribor,
				overnight,
				ts_missing,
				ts_npv,
				ts_expected,
			),
			good,
			&passed,
			&total,
		)
	}
	{ 	// Hull-White / G2++ caplet tensors and calibration leave nothing behind
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.default_allocator())
		defer mem.tracking_allocator_destroy(&track)
		context.allocator = mem.tracking_allocator(&track)
		Ts, Te := []f64{1, 2, 3, 4}, []f64{2, 3, 4, 5}
		F, P, K: [4]f64
		for i in 0 ..< 4 {F[i], P[i], K[i] = math.exp(0.03) - 1, math.exp(-0.03 * Te[i]), 0.032}
		tS, tE, tF, tK, tP :=
			ff_vec(Ts, false),
			ff_vec(Te, false),
			ff_vec(F[:], true),
			ff_vec(K[:], false),
			ff_vec(P[:], false)
		cap := fin.hull_white_caplet_price_tensor(0.1, 0.01, tS, tE, tF, tK, tP, context.allocator)
		s := t.tensor_sum(cap)
		t.tensor_backward(s)
		dF := tF.grad.data[0]
		caplet := cap.data.data[0]
		t.tensor_free_graph(s)
		tF.requires_grad = false
		flr := fin.hull_white_floorlet_price_tensor(
			0.1,
			0.01,
			tS,
			tE,
			tF,
			tK,
			tP,
			context.allocator,
		)
		t.tensor_free_graph(flr)
		g2 := fin.hull_white_2f_caplet_price_tensor(
			{0.3, 0.05, 0.008, 0.006, -0.6},
			tS,
			tE,
			tF,
			tK,
			tP,
			context.allocator,
		)
		t.tensor_free_graph(g2)
		after_prices := len(track.allocation_map) // the 5 inputs and F's gradient
		strip := fin.hw_price_cap_floor_strip(
			0.03,
			0.032,
			Ts,
			Te,
			[]f64{1, 1, 1, 1},
			{0.1, 0.01},
			true,
		)
		cal := fin.calibrate_hull_white(strip.leg_prices, Ts, Te, F[:], K[:], P[:], 4)
		delete(strip.leg_prices)
		after_cal := len(track.allocation_map)
		for x in ([]^t.Tensor{tS, tE, tF, tK, tP}) {t.tensor_free(x)}
		// QuantLib HullWhite(a 0.1, σ 1%) on a flat 3% curve: (1+K)·ZBP(1/(1+K), 1, 2)
		good :=
			after_prices == 11 &&
			after_cal == 11 &&
			len(track.allocation_map) == 0 &&
			math.abs(caplet - 0.0028302761) < 1e-10 &&
			dF > 0 &&
			math.abs(cal.params.a - 0.1) < 1e-3 &&
			math.abs(cal.params.sigma - 0.01) < 1e-4
		report(
			fmt.tprintf(
				"HW / G2++ caplet tensors and calibrate_hull_white: live allocations %d after pricing, %d after calibrating, then %d; caplet %.10f, dV/dF %.6f",
				after_prices,
				after_cal,
				len(track.allocation_map),
				caplet,
				dF,
			),
			good,
			&passed,
			&total,
		)
	}

	{ 	// constrained portfolios with a group limit (SciPy SLSQP references)
		vol := [5]f64{0.04, 0.06, 0.15, 0.18, 0.25}
		corr := [25]f64 {
			1,
			.7,
			.1,
			0,
			-.1,
			.7,
			1,
			.2,
			.1,
			0,
			.1,
			.2,
			1,
			.7,
			.6,
			0,
			.1,
			.7,
			1,
			.7,
			-.1,
			0,
			.6,
			.7,
			1,
		}
		mu := []f64{0.03, 0.04, 0.07, 0.08, 0.11}
		cov := l.matrix_new(f64, 5, 5, allocator)
		for i in 0 ..< 5 {for j in 0 ..< 5 {cov.data[i * 5 + j] = corr[i * 5 + j] * vol[i] * vol[j]}}
		c := fin.PortfolioConstraints {
			no_short_selling = true,
			max_weight       = 0.4,
			group_limits     = []fin.GroupLimit{{asset_indices = []int{0, 1}, max_weight = 0.5}},
		}
		mv := fin.constrained_min_variance_portfolio(&cov, c, allocator)
		ms := fin.constrained_max_sharpe_portfolio(mu, &cov, 0.02, c, allocator)
		ra := fin.constrained_risk_aversion_portfolio(mu, &cov, 4, c, allocator)
		// a long step: [26.44, 31.77, 14.36, 9.97, 13.51]
		pr := fin.project_constraints([]f64{26.44, 31.77, 14.36, 9.97, 13.51}, c, allocator)
		ref_mv := [5]f64{0.4, 0.1, 0.376491, 0.123509, 0}
		ref_ms := [5]f64{0.1, 0.4, 0.208946, 0.103195, 0.187859}
		ref_ra := [5]f64{0.1, 0.4, 0.164315, 0.092971, 0.242714}
		ref_pr := [5]f64{0.1, 0.4, 0.4, 0, 0.1}
		worst := 0.0
		for i in 0 ..< 5 {
			worst = max(
				worst,
				abs(mv[i] - ref_mv[i]),
				abs(ms[i] - ref_ms[i]),
				abs(ra[i] - ref_ra[i]),
				abs(pr[i] - ref_pr[i]),
			)
		}
		report(
			fmt.tprintf(
				"constrained portfolios with bonds <= 50%%: min variance %.6f, max Sharpe %.6f, utility %.6f, projection of a long step %.6f; max diff to SciPy %.1e",
				mv,
				ms,
				ra,
				pr,
				worst,
			),
			worst < 2e-6,
			&passed,
			&total,
		)
		l.matrix_free(&cov)
		for x in ([][]f64{mv, ms, ra, pr}) {delete(x, allocator)}
	}

	{ 	// unshifted Sobol' normals skip the origin (QuantLib SobolRsg; scipy points 1..3)
		z := fin.mc_generate_sobol_normals(3, 3, brownian_bridge = false, allocator = allocator)
		defer delete(z, allocator)
		// scipy.stats.norm.ppf(Sobol(3, scramble=False).fast_forward(1).random(3) + 2^-33)
		ref := [9]f64 {
			2.918099372916623e-10,
			2.918099372916623e-10,
			2.918099372916623e-10,
			6.744897505624251e-01,
			-6.744897498297384e-01,
			-6.744897498297384e-01,
			-6.744897498297384e-01,
			6.744897505624251e-01,
			6.744897505624251e-01,
		}
		worst := 0.0
		for k in 0 ..< 9 {worst = max(worst, abs(z[k] - ref[k]))}
		zl := fin.mc_generate_sobol_normals(1024, 4, allocator = allocator)
		defer delete(zl, allocator)
		lowest := 0.0
		for v in zl {lowest = min(lowest, v)}
		s := fin.make_sobol(3, 1, 0, allocator)
		defer fin.free_sobol(&s, allocator)
		pt: [3]f64
		fin.sobol_next(&s, pt[:])
		// one-step put, 1024 unshifted points: the origin made it +0.0428 too high
		exact := fin.compute_greeks(100, 100, 1, 0.05, 0.2, .Put).price
		zp := fin.mc_generate_sobol_normals(1024, 1, allocator = allocator)
		defer delete(zp, allocator)
		paths := fin.mc_generate_gbm_paths(100, 0.05, 0.2, 1, 1024, 1, zp, allocator)
		defer delete(paths, allocator)
		put := 0.0
		for p in 0 ..< 1024 {put += max(100 - paths[p * 2 + 1], 0)}
		put = put / 1024 * math.exp_f64(-0.05)
		report(
			fmt.tprintf(
				"unshifted Sobol' normals start at point 1: first 3 points vs scipy max diff %.1e; lowest z over 1024 paths x 4 steps %.3f (was -6.4 at the origin); make_sobol(skip = 1) starts at %v like QuantLib SobolRsg; 1024-point put error %+.5f",
				worst,
				lowest,
				pt,
				put - exact,
			),
			worst < 1e-12 && lowest > -4 && pt == [3]f64{0.5, 0.5, 0.5} && abs(put - exact) < 0.02,
			&passed,
			&total,
		)
	}

	{ 	// compute_greeks / price_and_greeks with a dividend yield vs QuantLib AnalyticEuropeanEngine
		// (BlackScholesMertonProcess with flat dividend curve; vega, rho per 1%, theta per day)
		Case :: struct {
			S, K, T, r, q, sigma: f64,
			opt:                  fin.OptionType,
			ref:                  [6]f64,
		}
		cases := [4]Case {
			{
				100,
				110,
				1,
				0.03,
				0.02,
				0.25,
				.Call,
				{
					6.404075273732313,
					0.4061943865601913,
					0.015280246206286947,
					0.3820061551571738,
					-0.013668901555651772,
					0.3421536338228681,
				},
			},
			{
				100,
				110,
				1,
				0.03,
				0.02,
				0.25,
				.Put,
				{
					15.13321663339267,
					-0.5740042867465641,
					0.015280246206286947,
					0.3820061551571738,
					-0.010265962064976217,
					-0.725336453080491,
				},
			},
			{
				100,
				95,
				2,
				0.01,
				0.05,
				0.3,
				.Call,
				{
					14.139444222623252,
					0.5043882285195465,
					0.008420001695047146,
					0.5052001017028289,
					-0.004465899743744642,
					0.7259875725866278,
				},
			},
			{
				100,
				95,
				2,
				0.01,
				0.05,
				0.3,
				.Put,
				{
					16.774576383169062,
					-0.40044918951641295,
					0.008420001695047146,
					0.5052001017028289,
					-0.014309730841110086,
					-1.1363899066962073,
				},
			},
		}
		worst := 0.0
		for c in cases {
			g := fin.compute_greeks(c.S, c.K, c.T, c.r, c.sigma, c.opt, q = c.q)
			got := [6]f64{g.price, g.delta, g.gamma, g.vega, g.theta, g.rho}
			for k in 0 ..< 6 {worst = max(worst, abs(got[k] - c.ref[k]) / max(abs(c.ref[k]), 1e-3))}
			pr, _ := fin.price_and_greeks(c.S, c.K, c.T, c.r, c.sigma, c.opt, q = c.q)
			worst = max(worst, abs(pr - g.price) / g.price)
		}
		// σ = 0: discounted forward payoff S e^{-qT} - K e^{-rT} = -2.6351321605458082 (put ITM)
		z := fin.compute_greeks(100, 95, 2, 0.01, 0, .Put, q = 0.05)
		dq := math.exp_f64(-0.1)
		zero_ok :=
			abs(z.price - 2.6351321605458082) < 1e-12 &&
			abs(z.delta + dq) < 1e-15 &&
			abs(z.theta + (0.05 * 100 * dq - 0.01 * 95 * math.exp_f64(-0.02)) / 365) < 1e-15
		report(
			fmt.tprintf(
				"compute_greeks with q (4 cases, price and all Greeks) vs QuantLib AnalyticEuropeanEngine: max rel err %.1e; sigma = 0 with q prices the forward (put %.10f)",
				worst,
				z.price,
			),
			worst < 1e-12 && zero_ok,
			&passed,
			&total,
		)
	}

	{ 	// finite differences with a dividend yield vs QuantLib FdBlackScholesVanillaEngine:
		// 2·V(8000 x 8000) − V(4000 x 4000) (its grid error halves with the grid),
		// put 8.882701964, call 8.407663772; deltas from the 2000 x 2000 grid
		ap := fin.finite_difference_american(
			100,
			100,
			1,
			0.05,
			0.25,
			.Put,
			3200,
			3200,
			allocator,
			q = 0.03,
		)
		ac := fin.fd_american_call(100, 100, 1, 0.05, 0.25, 3200, 3200, allocator, q = 0.08)
		ec := fin.crank_nicolson_call(100, 90, 1, 0.02, 0.3, 3200, 3200, allocator, q = 0.04)
		ac0 := fin.fd_american_call(100, 100, 1, 0.05, 0.25, 800, 800, allocator)
		eu0 := fin.compute_greeks(100, 100, 1, 0.05, 0.25, .Call).price
		eq := fin.compute_greeks(100, 100, 1, 0.05, 0.25, .Call, q = 0.08).price
		e_ap, e_ac := abs(ap.price - 8.882701964), abs(ac.price - 8.407663772)
		e_ec := abs(ec.price - 15.356039698298112) // analytic BSM
		e_d := max(abs(ap.delta + 0.424326676484122), abs(ac.delta - 0.49820850916592013))
		report(
			fmt.tprintf(
				"finite differences with q, 3200 x 3200: American put q = 3%% %.6f (QuantLib 8.882702), American call q = 8%% %.6f (8.407664, European %.6f), Crank-Nicolson European call q = 4%% %.6f (analytic 15.356040); errors %.1e / %.1e / %.1e, deltas %.1e; q = 0 call stays European (%.1e)",
				ap.price,
				ac.price,
				eq,
				ec.price,
				e_ap,
				e_ac,
				e_ec,
				e_d,
				abs(ac0.price - eu0),
			),
			e_ap < 5e-5 &&
			e_ac < 5e-5 &&
			e_ec < 1e-5 &&
			e_d < 1e-4 &&
			ac.price > eq + 0.1 &&
			abs(ac0.price - eu0) < 1e-3,
			&passed,
			&total,
		)
	}

	{ 	// Longstaff-Schwartz with a dividend yield vs QuantLib Bermudan (73 dates, FD 2000 x 1000)
		lp := fin.lsm_american_vanilla_reduced(
			100,
			105,
			1,
			0.05,
			0.3,
			.Put,
			100_000,
			73,
			73,
			3,
			allocator,
			q = 0.03,
		)
		lc := fin.lsm_american_vanilla_reduced(
			100,
			95,
			1,
			0.03,
			0.3,
			.Call,
			100_000,
			73,
			73,
			3,
			allocator,
			q = 0.07,
		)
		lw := fin.lsm_american_call(100, 95, 1, 0.03, 0.3, 20_000, 73, 73, 3, allocator, q = 0.07)
		e_p, e_c := lp.price - 13.577545310054507, lc.price - 12.24653647239092
		report(
			fmt.tprintf(
				"Longstaff-Schwartz with q: Bermudan put q = 3%% %.4f (QuantLib 13.5775), Bermudan call q = 7%% %.4f (12.2465, European 11.5863; wrapper %.4f): errors %+.4f / %+.4f",
				lp.price,
				lc.price,
				lw.price,
				e_p,
				e_c,
			),
			abs(e_p) < 0.04 && abs(e_c) < 0.04 && abs(lw.price - 12.24653647239092) < 0.15,
			&passed,
			&total,
		)
	}

	{ 	// limit and stop orders: pending (good till cancelled), filled at the first
		// close meeting the trigger, cancellable; rejected orders are reported
		prices := [8]f64{100, 98, 96, 99, 103, 105, 101, 97}
		df := w.dataframe_new()
		defer w.destroy_dataframe(&df)
		col := w.column_new("A", .Float, len(prices))
		for v in prices {w.append_float(&col, v)}
		w.add_column(&df, col)
		df.rows = len(prices)
		script :: proc(ctx: ^fin.StrategyContext) {
			o :: proc(side: fin.OrderSide, qty: f64, kind: fin.OrderType, px: f64) -> fin.Order {
				return {
					symbol = "A",
					side = side,
					quantity = qty,
					order_type = kind,
					limit_price = px if kind == .Limit else 0,
					stop_price = px if kind == .Stop else 0,
				}
			}
			switch ctx.current_bar {
			case 0:
				_ff_ord[0] = _ff_submit(ctx, o(.Buy, 10, .Limit, 97)) // rests
				_ff_ord[1] = _ff_submit(ctx, o(.Buy, 5, .Stop, 104)) // rests
				_ff_ord[2] = _ff_submit(ctx, o(.Buy, 1, .Limit, 101)) // marketable
				_ff_ord[3] = _ff_submit(ctx, o(.Buy, 1, .Limit, 0)) // no limit
			case 2:
				_ff_ord[4] = _ff_submit(ctx, o(.Sell, 10, .Stop, 95)) // stop loss
			case 5:
				_ff_cancel[0] = fin.context_cancel_order(ctx, _ff_ord[4].id)
				_ff_cancel[1] = fin.context_cancel_order(ctx, _ff_ord[4].id)
				_ff_ord[5] = _ff_submit(ctx, o(.Sell, 5, .Limit, 110)) // never fills
			}
		}
		cfg := fin.BacktestConfig {
			initial_capital = 10_000,
			commission_rate = 0.001,
			slippage_rate   = 0.002,
		}
		res := fin.backtest_run(&df, []string{"A"}, script, cfg, allocator)
		defer {delete(res.equity_curve, allocator); delete(res.trades, allocator)}
		want_id := [3]int{3, 1, 2}
		want_bar := [3]int{0, 2, 5}
		want_px := [3]f64{100 * 1.002, 96 * 1.002, 105 * 1.002}
		fills_ok := len(res.trades) == 3
		cash := 10_000.0
		if fills_ok {
			for tr, k in res.trades {
				fills_ok &&=
					tr.order_id == want_id[k] &&
					tr.fill_time == want_bar[k] &&
					abs(tr.fill_price - want_px[k]) < 1e-12
				cash -= tr.order.quantity * tr.fill_price * 1.001
			}
		}
		st_ok :=
			_ff_ord[0].status == .Pending &&
			_ff_ord[1].status == .Pending &&
			_ff_ord[2].status == .Filled &&
			_ff_ord[3] == _FF_Order{0, .Rejected} &&
			_ff_ord[4].status == .Pending &&
			_ff_ord[5] == _FF_Order{5, .Pending} &&
			_ff_cancel == [2]bool{true, false}
		final := res.equity_curve[len(prices) - 1]
		report(
			fmt.tprintf(
				"limit / stop orders on closes 100 98 96 99 103 105 101 97: buy limit 97 fills at 96 on bar 2, buy stop 104 at 105 on bar 5, marketable limit at once, sell stop cancelled, limit without a price rejected, %d order left open; equity %.4f (expected %.4f)",
				res.open_orders,
				final,
				cash + 16 * 97,
			),
			fills_ok && st_ok && res.open_orders == 1 && abs(final - (cash + 16 * 97)) < 1e-9,
			&passed,
			&total,
		)
	}

	{ 	// VaR / ES are positive loss magnitudes everywhere; value_at_risk(.GARCH) is a
		// GARCH(1,1) one-step forecast (arch 8.0: mean='Zero' on the demeaned returns,
		// normal errors); GARCH backtests are out of sample
		n := 2000
		r := make([]f64, n, allocator)
		defer delete(r, allocator)
		{
			// GARCH(1,1) ω 2e-6, α 0.08, β 0.9, μ 0.0005 driven by splitmix64 normals
			state: u64 = 42
			s2 := 2e-6 / 0.02
			for i in 0 ..< n {
				state += 0x9E3779B97F4A7C15
				x := state
				x = (x ~ (x >> 30)) * 0xBF58476D1CE4E5B9
				x = (x ~ (x >> 27)) * 0x94D049BB133111EB
				x ~= x >> 31
				u := (f64(x >> 11) + 0.5) * 1.1102230246251565e-16 // 2^-53
				e := math.sqrt(s2) * fin.norm_inv(u)
				r[i] = 0.0005 + e
				s2 = 2e-6 + 0.08 * e * e + 0.9 * s2
			}
		}
		h95, h99 := fin.var_historical(r, 0.95), fin.var_historical(r, 0.99)
		p95 := fin.var_parametric(r, 0.95)
		e95 := fin.conditional_var(r, 0.95)
		g95 := fin.value_at_risk(r, 0.95, .GARCH)
		g99 := fin.value_at_risk(r, 0.99, .GARCH)
		dm := fin.compute_distribution_metrics(r, allocator)
		ev := fin.evt_fit(r, 0.90, allocator)
		rm := fin.calculate_risk_metrics(r)
		signs_ok :=
			abs(h95 - 0.015554061056235777) < 1e-15 &&
			abs(h99 - 0.024181792528183203) < 1e-15 &&
			abs(p95 - 0.015519341566472802) < 1e-14 &&
			e95 > h95 &&
			fin.expected_shortfall(r, 0.95) == e95 &&
			fin.portfolio_cvar_historical(r, 0.95) == e95 &&
			dm.value_at_risk_95 == h95 &&
			dm.expected_shortfall_95 > 0 &&
			rm.var_95 == h95 &&
			rm.cvar_95 == e95 &&
			ev.var_99 > 0 &&
			ev.cvar_99 > ev.var_99 &&
			fin.var_garch_single(1e-4, 0.99) > 0 &&
			fin.ged_cvar(0.99, 1.5) > 0
		g_ok :=
			abs(g95 / 0.011344989781115655 - 1) < 1e-3 &&
			abs(g99 / 0.016295728191681808 - 1) < 1e-3

		// Out-of-sample backtests: parameters from the first half only. Changing the
		// test half leaves the fit and σ_t up to the first test day unchanged.
		half := n / 2
		o := fin.garch_out_of_sample(r, half, allocator = allocator)
		defer fin.free_garch_out_of_sample(&o, allocator)
		r2 := make([]f64, n, allocator)
		defer delete(r2, allocator)
		copy(r2, r)
		for &v in r2[half:] {v *= 3}
		o2 := fin.garch_out_of_sample(r2, half, allocator = allocator)
		defer fin.free_garch_out_of_sample(&o2, allocator)
		gm := fin.garch_risk_metrics(r, allocator = allocator)
		v99 := fin.var_garch(o.cond_var[half:], 0.99, allocator = allocator)
		defer delete(v99, allocator)
		bt := fin.backtest_var(r[half:], v99, 0.99)
		oos_ok :=
			o2.fit.params.omega == o.fit.params.omega &&
			o2.cond_var[half] == o.cond_var[half] &&
			o2.cond_var[half + 1] != o.cond_var[half + 1] &&
			gm.backtest_start == half &&
			gm.backtest_99 == bt &&
			gm.backtest_99.n_obs == half
		report(
			fmt.tprintf(
				"VaR as positive losses: historical 95%% %.6f, parametric %.6f, ES %.6f, EVT 99%% %.6f / %.6f; value_at_risk(.GARCH) 95%% %.6f / 99%% %.6f vs arch GARCH(1,1) %.6f / %.6f; out-of-sample GARCH backtest on the second half: %d breaches of 99%% VaR in %d (fit unchanged when the test half is scaled x3)",
				h95,
				p95,
				e95,
				ev.var_99,
				ev.cvar_99,
				g95,
				g99,
				0.011344989781115655,
				0.016295728191681808,
				gm.backtest_99.n_breaches,
				gm.backtest_99.n_obs,
			),
			signs_ok && g_ok && oos_ok,
			&passed,
			&total,
		)

		// Periodic refits (refit_every = 250 on the 1000 test returns: 4 fits). Each
		// segment is exactly the single-split fit on the returns known by its start:
		// garch_out_of_sample(r[:end], start), and for GARCH-EVT garch_evt_fit(r[:end],
		// train_fraction = start / end); μ is the mean of r[:start]; scaling the returns
		// from the last refit on changes nothing before it; refit_every beyond the test
		// period is the single fit; garch_risk_metrics backtests the refitted VaR.
		{
			RF :: 250
			orf := fin.garch_out_of_sample(r, half, allocator = allocator, refit_every = RF)
			defer fin.free_garch_out_of_sample(&orf, allocator)
			ev_rf := fin.garch_evt_fit(r, 0.90, allocator, 0.5, refit_every = RF)
			defer {delete(ev_rf.var_95_series, allocator); delete(ev_rf.var_99_series, allocator)}
			seg_ok := len(orf.segments) == (n - half) / RF
			worst_mu := 0.0
			for &seg, k in orf.segments {
				e := n if k == len(orf.segments) - 1 else orf.segments[k + 1].start
				seg_ok &&= seg.start == half + k * RF
				ref := fin.garch_out_of_sample(r[:e], seg.start, allocator = allocator)
				for t in seg.start ..< e {seg_ok &&= orf.cond_var[t] == ref.cond_var[t] && orf.resid[t] == ref.resid[t]}
				seg_ok &&=
					seg.fit.params.omega == ref.fit.params.omega &&
					fin.garch_out_of_sample_segment(&orf, e - 1) == &seg
				fin.free_garch_out_of_sample(&ref, allocator)
				m := 0.0
				for v in r[:seg.start] {m += v}
				worst_mu = max(worst_mu, abs(seg.mu - m / f64(seg.start)))
				eref := fin.garch_evt_fit(r[:e], 0.90, allocator, f64(seg.start) / f64(e))
				for t in seg.start ..< e {seg_ok &&= ev_rf.var_99_series[t] == eref.var_99_series[t]}
				delete(eref.var_95_series, allocator); delete(eref.var_99_series, allocator)
			}
			for t in 0 ..< half {seg_ok &&= orf.cond_var[t] == o.cond_var[t]}

			last := orf.segments[len(orf.segments) - 1].start
			r3 := make([]f64, n, allocator)
			defer delete(r3, allocator)
			copy(r3, r)
			for &v in r3[last:] {v *= 3}
			o3 := fin.garch_out_of_sample(r3, half, allocator = allocator, refit_every = RF)
			defer fin.free_garch_out_of_sample(&o3, allocator)
			causal := true
			for t in 0 ..= last {causal &&= o3.cond_var[t] == orf.cond_var[t]}
			causal &&= o3.cond_var[last + 1] != orf.cond_var[last + 1]

			obig := fin.garch_out_of_sample(r, half, allocator = allocator, refit_every = n)
			defer fin.free_garch_out_of_sample(&obig, allocator)
			same := len(obig.segments) == 1
			for t in 0 ..< n {same &&= obig.cond_var[t] == o.cond_var[t]}

			gr := fin.garch_risk_metrics(r, allocator = allocator, refit_every = RF)
			v99r := fin.var_garch(orf.cond_var[half:], 0.99, allocator = allocator)
			defer delete(v99r, allocator)
			btr := fin.backtest_var(r[half:], v99r, 0.99)
			report(
				fmt.tprintf(
					"GARCH refits every %d test returns: %d segments, each = the single-split fit on the returns before it (GARCH and GARCH-EVT), causal %v, refit_every > test period = single fit %v; 99%% breaches %d (single fit %d) of %d",
					RF,
					len(orf.segments),
					causal,
					same,
					gr.backtest_99.n_breaches,
					gm.backtest_99.n_breaches,
					gr.backtest_99.n_obs,
				),
				seg_ok && worst_mu < 1e-15 && causal && same && gr.backtest_99 == btr,
				&passed,
				&total,
			)
		}
	}

	{ 	// CDS rpv01 on the notional; PDPM payoffs without dead arguments; free procs
		tenors := []f64{1, 3, 5, 7, 10}
		spreads := []f64{0.0060, 0.0085, 0.0110, 0.0125, 0.0135}
		curve := fin.bootstrap_cds_curve(tenors, spreads, 0.4, 0.03, 0.25, allocator)
		defer {delete(curve.tenors, allocator); delete(curve.hazards, allocator)}
		a := fin.price_cds_full(1e6, 0.01, 5, curve, 0.25)
		b := fin.price_cds_full(1e7, 0.01, 5, curve, 0.25)
		cds_ok :=
			abs(b.rpv01 / a.rpv01 - 10) < 1e-12 &&
			abs(b.rpv01 - b.premium_leg_pv / 0.01 * 1e-4) < 1e-6 &&
			abs(a.rpv01 - 445.7024158) < 1e-6

		R := []f64{-0.08, -0.03, 0.0, 0.02, 0.05, 0.09}
		ST := make([]f64, len(R), allocator)
		defer delete(ST, allocator)
		for x, i in R {ST[i] = 100 * (1 + x)}
		sdf := fin.sdf_linear_factor(R, 0.01, allocator)
		c := fin.price_european_call(100, &sdf, ST, allocator)
		pay := make([]f64, len(R), allocator)
		defer delete(pay, allocator)
		for s, i in ST {pay[i] = max(s - 100, 0)}
		ref := fin.price_payoff(pay, &sdf, nil, allocator)
		pdpm_ok := c.price == ref.price && c.risk_premium == ref.risk_premium
		fin.free_pdpm_result(&c, allocator)
		fin.free_pdpm_result(&ref, allocator)
		fin.free_stochastic_discount_factor(&sdf, allocator)
		md := fin.load_market_data("data/spy_prices.csv", allocator)
		oc := fin.load_option_chain("data/spy_options.csv", md.dates[md.n_obs - 1], allocator)
		types_ok :=
			oc.n_options > 0 && (oc.option_types[0] == "call" || oc.option_types[0] == "put")
		fin.free_option_chain(&oc, allocator)
		fin.free_market_data(&md, allocator)
		free_ok :=
			c.state_prices == nil &&
			sdf.values == nil &&
			sdf.parameters == nil &&
			oc.strikes == nil &&
			oc.n_options == 0 &&
			md.prices == nil &&
			md.n_obs == 0
		report(
			fmt.tprintf(
				"CDS rpv01 scales with the notional (1M %.4f, 10M %.4f); price_european_call(K, sdf, S_T) = price_payoff; free_pdpm_result / free_stochastic_discount_factor / free_option_chain / free_market_data release and zero",
				a.rpv01,
				b.rpv01,
			),
			cds_ok && pdpm_ok && types_ok && free_ok,
			&passed,
			&total,
		)
	}

	{ 	// a batch of options (last: it used to panic)
		S, K := ff_vec({100, 100, 100}, true), ff_vec({90, 100, 110}, false)
		T, r, sigma :=
			ff_vec({0.5, 1, 2}, false),
			ff_vec({0.03, 0.03, 0.03}, false),
			ff_vec({0.25, 0.25, 0.25}, false)
		p := fin.black_scholes_price(S, K, T, r, sigma, .Call, allocator)
		s := t.tensor_sum(p)
		t.tensor_backward(s)
		good := true
		for k in 0 ..< 3 {
			g := fin.compute_greeks(100, K.data.data[k], T.data.data[k], 0.03, 0.25, .Call)
			good =
				good &&
				math.abs(p.data.data[k] - g.price) < 1e-10 &&
				math.abs(S.grad.data[k] - g.delta) < 1e-12
		}
		report(
			fmt.tprintf("black_scholes_price on 3 options: %.6f", p.data.data),
			good,
			&passed,
			&total,
		)
		t.tensor_free_graph(s)
		for x in ([]^t.Tensor{S, K, T, r, sigma}) {t.tensor_free(x)}
	}

	{ 	// linear-zero curve beyond the last pillar: the instantaneous forward at the
		// last pillar is carried on, as QuantLib's InterpolatedZeroCurve<Linear>
		// (QuantLib 1.44 ZeroCurve, nodes 1/2/5/10 y at 3.0/3.4/3.9/3.75 %)
		curve := fin.YieldCurve {
			tenors        = []f64{1, 2, 5, 10},
			zero_rates    = []f64{0.030, 0.034, 0.039, 0.0375},
			interpolation = .LinearZero,
		}
		ts := []f64{0.5, 7, 10, 12, 15, 30}
		ql := []f64{0.03, 0.0384, 0.0375, 0.037, 0.0365, 0.0355}
		worst := 0.0
		for tt, i in ts {worst = max(worst, abs(fin.yield_curve_zero_rate(&curve, tt) - ql[i]))}
		df30 := fin.yield_curve_discount_factor(&curve, 30)
		worst = max(worst, abs(df30 - 0.3447278547672202))
		report(
			fmt.tprintf(
				"linear-zero extrapolation = QuantLib (z(30) %.10f, DF(30) %.12f): max err %.1e",
				fin.yield_curve_zero_rate(&curve, 30),
				df30,
				worst,
			),
			worst < 1e-14,
			&passed,
			&total,
		)
	}
	{ 	// callable / puttable bonds take a bootstrapped curve through user_data
		// (discount_fn_yield_curve), no global needed: same price as a closure
		flat := fin.YieldCurve {
			tenors        = []f64{1, 30},
			zero_rates    = []f64{0.03, 0.03},
			interpolation = .LinearZero,
		}
		p0 :: proc(tt: f64, _: rawptr) -> f64 {return math.exp(-0.03 * tt)}
		bond := fin.CallableBond {
			face_value       = 100,
			coupon_rate      = 0.05,
			coupon_frequency = 2,
			maturity         = 10,
			call_schedule    = []fin.CallDate{{5, 102}, {7, 101}, {9, 100}},
		}
		rand.reset(11)
		a := fin.callable_bond_lsm_hw1f(bond, 0.1, 0.01, 0.03, p0, 4000, allocator = allocator)
		rand.reset(11)
		b := fin.callable_bond_lsm_hw1f(
			bond,
			0.1,
			0.01,
			0.03,
			fin.discount_fn_yield_curve,
			4000,
			user_data = &flat,
			allocator = allocator,
		)
		report(
			fmt.tprintf(
				"callable bond on a YieldCurve via user_data: %.10f = %.10f (closure)",
				b.price,
				a.price,
			),
			abs(a.price - b.price) < 1e-10 * a.price && a.price > 90,
			&passed,
			&total,
		)
	}

	fmt.printf("finance regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
