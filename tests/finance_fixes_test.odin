package tests

import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
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

	fmt.printf("finance regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
