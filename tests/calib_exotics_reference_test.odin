package tests

import fin "../wotan/finance"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Heston / Merton jump diffusion / local volatility Monte Carlo exotics against
//   numpy Monte Carlo with 1M paths on the same 50-step discretisation (full
//   truncation Heston, Poisson jumps, bilinear local vol) with plain knock-out
//   indicators and central differences on common random numbers, and
//   closed forms: Heston via scipy quad, the Merton series, Black-Scholes.
// Monte Carlo results must agree within 4 standard errors of the difference.
calib_exotics_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Heston / MJD / local-vol exotics (numpy / scipy reference) ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	r := 0.03
	hp := fin.Heston_Params {
		v0    = 0.04,
		kappa = 1.5,
		theta = 0.06,
		sigma = 0.6,
		rho   = -0.7,
	}
	hp2 := fin.Heston_Params {
		v0    = 0.09,
		kappa = 2.0,
		theta = 0.09,
		sigma = 0.4,
		rho   = -0.3,
	}
	mj := fin.MJD_Params {
		sigma   = 0.2,
		lambda  = 0.8,
		mu_j    = -0.1,
		sigma_j = 0.15,
	}
	LV_K := [?]f64{60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0}
	LV_T := [?]f64{0.1, 0.4, 0.7, 1.0}
	LV_D := [?]f64 {
		0.24823570447397836,
		0.21480425390842448,
		0.19544826112327934,
		0.18577520956492077,
		0.183,
		0.18527100759358317,
		0.1913102875179428,
		0.2002087518173919,
		0.21130339150422037,
		0.25723570447397837,
		0.22380425390842448,
		0.20444826112327935,
		0.19477520956492078,
		0.192,
		0.19427100759358318,
		0.2003102875179428,
		0.2092087518173919,
		0.22030339150422037,
		0.2662357044739784,
		0.23280425390842446,
		0.21344826112327933,
		0.20377520956492076,
		0.20099999999999998,
		0.20327100759358316,
		0.20931028751794278,
		0.21820875181739188,
		0.22930339150422036,
		0.2752357044739784,
		0.24180425390842447,
		0.22244826112327934,
		0.21277520956492077,
		0.21,
		0.21227100759358317,
		0.2183102875179428,
		0.2272087518173919,
		0.23830339150422036,
	}
	lv := fin.LocalVolSurface {
		strikes      = LV_K[:],
		expiries     = LV_T[:],
		lv_data      = LV_D[:],
		num_strikes  = len(LV_K),
		num_expiries = len(LV_T),
	}

	// --- Monte Carlo against numpy: price, delta, vega (per-path std of the reference) ---
	N_MC :: 40000
	se_scale := math.sqrt_f64(1.0 / N_MC + 1.0 / 1e6)
	check :: proc(
		passed, total: ^int,
		label: string,
		got, ref, std: []f64,
		names: []string,
		se_scale: f64,
	) {
		ok := true
		desc := ""
		for v, i in got {
			z := (v - ref[i]) / (std[i] * se_scale)
			ok &&= abs(z) < 4
			desc = fmt.tprintf("%s %s %.4f (ref %.4f, z %+.1f)", desc, names[i], v, ref[i], z)
		}
		expect(passed, total, ok, fmt.tprintf("%s:%s", label, desc))
	}
	pdv := []string{"price", "delta", "vega"}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.heston_mc_barrier_option(
			100,
			100,
			1,
			r,
			125,
			true,
			.Call,
			hp,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"Heston up-and-out call (B=125)",
			got[:],
			[]f64{5.2970654699456174, 0.08279331454596067, -32.65326460948728},
			[]f64{7.0808792663832225, 2.177079359593748, 228.98214701366157},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.heston_mc_barrier_option(
			100,
			100,
			1,
			r,
			80,
			false,
			.Put,
			hp,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"Heston down-and-out put (B=80)",
			got[:],
			[]f64{0.9568950457460343, -0.03990041634497121, -2.290627373379296},
			[]f64{2.9461188000291316, 0.8261322418808519, 102.91083521499934},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.heston_mc_asian_option(100, 100, 1, r, .Call, hp, N_MC, 50)
		check(
			&passed,
			&total,
			"Heston Asian call",
			got[:],
			[]f64{5.392459874410114, 0.6526146375509149, 22.66765878953872},
			[]f64{6.245639075730503, 0.5066153957639145, 52.636586322389526},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.heston_mc_lookback_option(100, 100, 1, r, .Call, hp, N_MC, 50)
		check(
			&passed,
			&total,
			"Heston lookback call",
			got[:],
			[]f64{13.904732609191148, 1.0704564466918076, 70.39808637606713},
			[]f64{10.65856792922007, 0.1963654814251425, 89.15893754449729},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.heston_mc_lookback_option(100, 100, 1, r, .Put, hp, N_MC, 50)
		check(
			&passed,
			&total,
			"Heston lookback put",
			got[:],
			[]f64{13.433336990604205, -0.7667122868193044, 64.46132764145474},
			[]f64{14.054228639783236, 0.1683200365020139, 77.68851434570624},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.heston_mc_barrier_option(
			100,
			100,
			1,
			r,
			1e12,
			true,
			.Call,
			hp,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"Heston European call (barrier out of reach)",
			got[:],
			[]f64{9.538453103572586, 0.701702891520814, 37.938512910288054},
			[]f64{11.108987734453764, 0.5448654084258888, 98.38264328545971},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.mjd_mc_barrier_option(
			100,
			100,
			1,
			r,
			130,
			true,
			.Call,
			mj,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"MJD up-and-out call (B=130)",
			got[:],
			[]f64{3.2499502916634313, 0.02119662487287741, -21.845274175983622},
			[]f64{6.429952097493679, 1.846715692640792, 371.902863224475},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.mjd_mc_barrier_option(
			100,
			100,
			1,
			r,
			75,
			false,
			.Put,
			mj,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"MJD down-and-out put (B=75)",
			got[:],
			[]f64{2.644150215049183, -0.05508120215233778, -5.736756190022447},
			[]f64{5.37101244599261, 1.363761782159751, 342.498548338821},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.mjd_mc_asian_option(100, 100, 1, r, .Call, mj, N_MC, 50)
		check(
			&passed,
			&total,
			"MJD Asian call",
			got[:],
			[]f64{6.419070756985787, 0.5801415534083554, 19.087104036791494},
			[]f64{9.319568205604282, 0.5408823946831468, 42.66998042092279},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.mjd_mc_barrier_option(
			100,
			100,
			1,
			r,
			1e12,
			true,
			.Call,
			mj,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"MJD European call (barrier out of reach)",
			got[:],
			[]f64{11.298447399012883, 0.6175050677976591, 31.51890038736216},
			[]f64{16.96582257186463, 0.6033596049462868, 79.93793227726105},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.lv_mc_barrier_option(
			100,
			100,
			1,
			r,
			125,
			true,
			.Call,
			lv,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"LV (smile surface) up-and-out call (B=125)",
			got[:],
			[]f64{2.5446763351766024, 0.02358984179613725, -18.936645112297537},
			[]f64{5.150191374057934, 1.5799841612475267, 144.7851622395406},
			pdv,
			se_scale,
		)
	}
	{
		got: [3]f64
		got[0], got[1], got[2] = fin.lv_mc_barrier_option(
			100,
			100,
			1,
			r,
			80,
			false,
			.Put,
			lv,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"LV (smile surface) down-and-out put (B=80)",
			got[:],
			[]f64{2.1201879316932453, -0.027582155466467667, -11.795850513436646},
			[]f64{4.310032417927172, 1.2763270945097367, 140.04899118600417},
			pdv,
			se_scale,
		)
	}
	{
		got: [5]f64
		got[0], got[1], got[2], got[3], got[4] = fin.heston_mc_basket_option(
			100,
			90,
			95,
			1,
			r,
			0.6,
			0.4,
			.Call,
			hp,
			hp2,
			0.4,
			N_MC,
			50,
		)
		check(
			&passed,
			&total,
			"Heston basket call (corr 0.4)",
			got[:],
			[]f64 {
				9.292794740773898,
				0.3912838680561725,
				0.2690995551474491,
				19.930504935357913,
				10.213269996109766,
			},
			[]f64 {
				12.311254891553434,
				0.33129754420280805,
				0.23945784322935645,
				59.54167128158931,
				33.11427828112405,
			},
			[]string{"price", "delta1", "delta2", "vega1", "vega2"},
			se_scale,
		)
	}

	// --- closed forms ---
	{
		hq := fin.heston_price(100, 100, 1, r, hp, .Call, 2000)
		expect(
			&passed,
			&total,
			abs(hq - 9.508931169472511) < 1e-6,
			fmt.tprintf("heston_price matches scipy quad (%.8f vs 9.50893117)", hq),
		)
		// Discretisation bias of the 100-step scheme is far below the Monte Carlo error
		rand.reset(21)
		p, _, _ := fin.heston_mc_barrier_option(100, 100, 1, r, 1e12, true, .Call, hp, 100000, 100)
		z := (p - hq) / (11.108987734453764 * math.sqrt_f64(1.0 / 100000))
		expect(
			&passed,
			&total,
			abs(z) < 4,
			fmt.tprintf(
				"Heston Monte Carlo European call vs semi-analytic: %.4f vs %.4f (z %+.1f; the old scheme was biased by +0.07, 7 sigma)",
				p,
				hq,
				z,
			),
		)
		rand.reset(22)
		m, md, _ := fin.heston_mc_barrier_option(100, 0, 1, r, 1e12, true, .Call, hp, 100000, 100)
		expect(
			&passed,
			&total,
			abs(m - 100) < 0.2 && abs(md - m / 100) < 1e-12,
			fmt.tprintf(
				"Heston discounted stock is a martingale (%.4f) and its delta is V/S_0",
				m,
			),
		)
		mq := 11.304326205549225
		rand.reset(23)
		pm, _, _ := fin.mjd_mc_barrier_option(100, 100, 1, r, 1e12, true, .Call, mj, 100000, 50)
		zm := (pm - mq) / (16.96582257186463 * math.sqrt_f64(1.0 / 100000))
		expect(
			&passed,
			&total,
			abs(zm) < 4,
			fmt.tprintf(
				"MJD Monte Carlo European call vs Merton series: %.4f vs %.4f (z %+.1f)",
				pm,
				mq,
				zm,
			),
		)
	}

	// --- vega is per unit of volatility in every model: Black-Scholes limits ---
	{
		bs := fin.compute_greeks(100, 100, 1, r, 0.2, .Call)
		flat_h := fin.Heston_Params {
			v0    = 0.04,
			kappa = 1,
			theta = 0.04,
			sigma = 0,
			rho   = 0,
		}
		rand.reset(31)
		hpz, hd, hv := fin.heston_mc_barrier_option(
			100,
			100,
			1,
			r,
			1e12,
			true,
			.Call,
			flat_h,
			40000,
			50,
		)
		flat_m := fin.MJD_Params {
			sigma = 0.2,
		}
		rand.reset(32)
		mp, md, mv := fin.mjd_mc_barrier_option(
			100,
			100,
			1,
			r,
			1e12,
			true,
			.Call,
			flat_m,
			40000,
			50,
		)
		flat_d := [?]f64{0.2, 0.2, 0.2, 0.2}
		flat_lv := fin.LocalVolSurface {
			strikes      = []f64{50, 150},
			expiries     = []f64{0, 2},
			lv_data      = flat_d[:],
			num_strikes  = 2,
			num_expiries = 2,
		}
		rand.reset(33)
		lp, ld, lvv := fin.lv_mc_barrier_option(
			100,
			100,
			1,
			r,
			1e12,
			true,
			.Call,
			flat_lv,
			40000,
			50,
		)
		// per-path std of the same estimators in a 1M-path Black-Scholes run
		se :=
			[3]f64{14.11140059226685, 0.5799295597778185, 74.94017397372534} / math.sqrt_f64(40000)
		ok := true
		for g in ([][3]f64{{hpz, hd, hv}, {mp, md, mv}, {lp, ld, lvv}}) {
			ok &&=
				abs(g[0] - bs.price) < 4 * se[0] &&
				abs(g[1] - bs.delta) < 4 * se[1] &&
				abs(g[2] - 100 * bs.vega) < 4 * se[2]
		}
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"Heston(sigma=0) / MJD(lambda=0) / flat LV give Black-Scholes price, delta, vega %.2f: vegas %.2f %.2f %.2f (were vol-of-vol, -, 0.2 x vega)",
				100 * bs.vega,
				hv,
				mv,
				lvv,
			),
		)
	}

	// --- Dupire surface ---
	{
		flat := fin.Heston_Params {
			v0    = 0.04,
			kappa = 1,
			theta = 0.04,
			sigma = 1e-4,
			rho   = 0,
		}
		surf := fin.build_lv_surface_from_heston(100, 0.05, flat)
		worst := 0.0
		for x in surf.lv_data {worst = max(worst, abs(x - 0.2))}
		fin.lv_surface_free(&surf)
		expect(
			&passed,
			&total,
			worst < 2e-3,
			fmt.tprintf(
				"Dupire surface of a flat model is flat at 0.2 for r = 5%% (max deviation %.1e; the +r*C term gave up to 67)",
				worst,
			),
		)
	}

	// --- memory ---
	{
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.heap_allocator())
		defer mem.tracking_allocator_destroy(&track)
		tal := mem.tracking_allocator(&track)
		context.allocator = tal
		_, _, _ = fin.heston_mc_barrier_option(100, 100, 1, r, 120, true, .Call, hp, 20000, 252)
		_, _, _, _, _ = fin.heston_mc_basket_option(
			100,
			90,
			95,
			1,
			r,
			0.6,
			0.4,
			.Call,
			hp,
			hp2,
			0.4,
			20000,
			252,
		)
		_, _, _ = fin.mjd_mc_asian_option(100, 100, 1, r, .Call, mj, 20000, 252)
		_, _, _ = fin.lv_mc_barrier_option(100, 100, 1, r, 125, true, .Call, lv, 20000, 252)
		peak := track.peak_memory_allocated
		context.allocator = allocator
		expect(
			&passed,
			&total,
			len(track.allocation_map) == 0 && peak < 16 * 1024,
			fmt.tprintf(
				"pricers keep one path of normals (peak %d bytes for 20000 x 252; was 80 MB of stored normals)",
				peak,
			),
		)
	}

	fmt.printf("Calibrated exotics summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
