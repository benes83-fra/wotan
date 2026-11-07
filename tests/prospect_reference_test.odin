package tests

import bfin "../wotan/behavioral_finance"
import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

// CPT option prices (behavioral_finance/prospect.odin) against Nardon & Pianca (2012,
// "Prospect theory: an application to European option pricing", Ca' Foscari WP 34/2012),
// who price as Versluis, Lehnert & Wolff (2010). S0 100, mu = r = 0.05, sigma 0.2, T 1;
// TK92 (a = b = 0.88, lambda 2.25, gamma+ 0.61, gamma- 0.69), moderate sentiment (0.988,
// 1.125, 0.961, 0.969) with TK and with Prelec weighting.
// _CPT_REF: writer call / writer put / holder call / holder put, segregated and
// aggregated, from an independent scipy implementation (quad in the normal variable,
// brentq) that reproduces the paper's Tables 3 - 5; where they differ (Table 5 TK
// aggregated K 100: paper 8.3010) a 30-digit mpmath check confirms the reference
// (8.30427). _CPT_TABLE6: the paper's Table 6 (mu = r = 0.1), holder - writer intervals
// (three printed values, 19.40, 9.40 and 11.95, are 0.009 / 0.006 / 0.008 below the
// computed 19.4086, 9.4060, 11.9582: truncated rather than rounded).
// Accuracy of the grid (4000 cells): ~1e-5 relative.

@(private = "file")
_CPT_REF := [36][5]f64 {
	{0, 0, 80, 59.1367036208, 37.0964389214},
	{0, 0, 100, 29.2077286243, 22.3340986520},
	{0, 0, 120, 12.3974804866, 11.2360947308},
	{0, 1, 80, 27.7654615752, 25.6390607508},
	{0, 1, 100, 11.9500858494, 11.3814197119},
	{0, 1, 120, 3.8139957943, 3.7446431249},
	{0, 2, 80, 27.6831865175, 25.7037822014},
	{0, 2, 100, 11.9028333624, 11.4246766706},
	{0, 2, 120, 3.8393975371, 3.7830134889},
	{1, 0, 80, 3.1571907897, 3.0876070502},
	{1, 0, 100, 15.6905442408, 12.5269351885},
	{1, 0, 120, 40.2731558891, 26.3766490960},
	{1, 1, 80, 0.8204477726, 0.8144986790},
	{1, 1, 100, 6.3848034712, 6.1384589967},
	{1, 1, 120, 19.5999297288, 18.2390081866},
	{1, 2, 80, 0.8378693893, 0.8321133253},
	{1, 2, 100, 6.3487892548, 6.1483449036},
	{1, 2, 120, 19.5061874808, 18.2748067695},
	{2, 0, 80, 9.0923564949, 18.9272731247},
	{2, 0, 100, 4.8155502856, 8.3042701851},
	{2, 0, 120, 2.3073382939, 3.1864306550},
	{2, 1, 80, 21.9087950262, 23.8714454812},
	{2, 1, 100, 9.4774358918, 10.0519343737},
	{2, 1, 120, 3.0551587625, 3.1402585388},
	{2, 2, 80, 21.8318112973, 23.9136648114},
	{2, 2, 100, 9.4331939067, 10.0827697184},
	{2, 2, 120, 3.0815879426, 3.1787186681},
	{3, 0, 80, 0.6285188088, 0.7733764067},
	{3, 0, 100, 2.5780356480, 4.2139058339},
	{3, 0, 120, 6.0935751597, 12.2989455311},
	{3, 1, 80, 0.6615060262, 0.6708235906},
	{3, 1, 100, 5.0687321946, 5.3314915758},
	{3, 1, 120, 15.4617065069, 16.7676223733},
	{3, 2, 80, 0.6791757891, 0.6890391643},
	{3, 2, 100, 5.0340871959, 5.3338422472},
	{3, 2, 120, 15.3718930127, 16.7817337608},
}

// K, TK segregated holder / writer, TK aggregated holder / writer, moderate seg., agg.
@(private = "file")
_CPT_TABLE6 := [5][9]f64 {
	{80, 10.20, 66.76, 21.78, 40.35, 24.91, 31.58, 27.24, 29.04},
	{90, 7.67, 49.20, 15.27, 32.62, 17.86, 22.61, 19.34, 21.04},
	{100, 5.66, 35.14, 10.30, 25.62, 11.95, 15.10, 12.78, 14.27},
	{110, 4.11, 24.30, 6.72, 19.40, 7.47, 9.40, 7.87, 9.03},
	{120, 2.90, 16.22, 4.27, 14.06, 4.38, 5.48, 4.54, 5.34},
}

prospect_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Prospect Theory Option Pricing Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	S, r, sigma, T := 100.0, 0.05, 0.2, 1.0
	MODERATE :: bfin.Prospect_Params{0.988, 0.988, 1.125, 0.961, 0.969, .Tversky_Kahneman}
	sets := [3]bfin.Prospect_Params{bfin.TK92_PARAMS, MODERATE, MODERATE}
	sets[2].weighting = .Prelec
	types := [4]fin.OptionType{.Call, .Put, .Call, .Put}
	sides := [4]bfin.CPT_Side{.Writer, .Writer, .Holder, .Holder}
	frames := [2]bfin.CPT_Frame{.Segregated, .Aggregated}

	// 1. the 72 reference prices
	err := 0.0
	err_coarse := 0.0
	for row in _CPT_REF {
		k, ps, K := int(row[0]), int(row[1]), row[2]
		for fr, j in frames {
			got := bfin.prospect_theory_price(
				S,
				K,
				T,
				r,
				r,
				sigma,
				sets[ps],
				types[k],
				sides[k],
				fr,
				4000,
			)
			coarse := bfin.prospect_theory_price(
				S,
				K,
				T,
				r,
				r,
				sigma,
				sets[ps],
				types[k],
				sides[k],
				fr,
				1000,
			)
			err = max(err, abs(got - row[3 + j]) / row[3 + j])
			err_coarse = max(err_coarse, abs(coarse - row[3 + j]) / row[3 + j])
		}
	}
	expect(
		&passed,
		&total,
		err < 2e-5,
		fmt.tprintf(
			"72 prices (writer / holder, call / put, segregated / aggregated, TK92 / moderate / Prelec) = reference (max relative error %.1e at 4000 cells)",
			err,
		),
	)
	expect(
		&passed,
		&total,
		err_coarse / err > 10,
		fmt.tprintf(
			"second-order grid: 1000 cells %.1e, 4000 cells %.1e (ratio %.0f, 16 for h^2)",
			err_coarse,
			err,
			err_coarse / err,
		),
	)

	// 2. Table 6: mu = r = 0.1, holder (lower) and writer (upper) prices
	e6 := 0.0
	n_rounded := 0
	inside := true
	for row in _CPT_TABLE6 {
		K := row[0]
		bs := fin.black_scholes_call(S, K, T, 0.1, sigma)
		for ps in 0 ..< 2 {
			for fr, j in frames {
				lo := bfin.prospect_theory_price(
					S,
					K,
					T,
					0.1,
					0.1,
					sigma,
					sets[ps],
					.Call,
					.Holder,
					fr,
					4000,
				)
				hi := bfin.prospect_theory_price(
					S,
					K,
					T,
					0.1,
					0.1,
					sigma,
					sets[ps],
					.Call,
					.Writer,
					fr,
					4000,
				)
				c := 1 + 4 * ps + 2 * j
				e6 = max(e6, abs(lo - row[c]), abs(hi - row[c + 1]))
				n_rounded += int(abs(lo - row[c]) <= 0.005) + int(abs(hi - row[c + 1]) <= 0.005)
				inside &&= lo < bs && bs < hi
			}
		}
	}
	expect(
		&passed,
		&total,
		n_rounded >= 37 && e6 < 0.01 && inside,
		fmt.tprintf(
			"Table 6 (mu = r = 0.1): %d of the 40 holder / writer bounds round to the printed 2 decimals, all within %.4f; Black-Scholes inside every interval",
			n_rounded,
			e6,
		),
	)

	// 3. rational investor = Black-Scholes, every side / frame / type / weighting
	eb := 0.0
	for K in ([]f64{80, 100, 120}) {
		call := fin.black_scholes_call(S, K, T, r, sigma)
		for k in 0 ..< 4 {
			bs := call if types[k] == .Call else call - S + K * math.exp(-r * T)
			for fr in frames {
				for wt in bfin.CPT_Weighting {
					p := bfin.RATIONAL_PARAMS
					p.weighting = wt
					got := bfin.prospect_theory_price(
						S,
						K,
						T,
						r,
						r,
						sigma,
						p,
						types[k],
						sides[k],
						fr,
						4000,
					)
					eb = max(eb, abs(got - bs) / bs)
				}
			}
		}
	}
	expect(
		&passed,
		&total,
		eb < 1e-5,
		fmt.tprintf(
			"alpha = beta = lambda = gamma = 1, mu = r: Black-Scholes calls and puts (max relative error %.1e)",
			eb,
		),
	)

	// 4. the price is the indifference premium: the prospect value is zero there, the
	// writer gains from a higher premium, the holder loses
	pre := bfin.prospect_lognormal_grid(S, 100, T, r, sigma, 2000)
	defer bfin.prospect_precomputed_free(&pre)
	vz, sgn_ok := 0.0, true
	for k in 0 ..< 4 {
		for fr in frames {
			c := bfin.prospect_price_precomputed(
				&pre,
				100,
				T,
				r,
				bfin.TK92_PARAMS,
				types[k],
				sides[k],
				fr,
			)
			v0 := bfin.prospect_option_value(
				&pre,
				100,
				T,
				r,
				c,
				bfin.TK92_PARAMS,
				types[k],
				sides[k],
				fr,
			)
			v1 := bfin.prospect_option_value(
				&pre,
				100,
				T,
				r,
				c + 0.01,
				bfin.TK92_PARAMS,
				types[k],
				sides[k],
				fr,
			)
			vz = max(vz, abs(v0))
			sgn_ok &&= (v1 > 0) == (sides[k] == .Writer)
		}
	}
	expect(
		&passed,
		&total,
		vz < 1e-9 && sgn_ok,
		fmt.tprintf(
			"prospect value at the price = 0 (max |V| %.1e); a higher premium helps the writer, hurts the holder",
			vz,
		),
	)

	// 5. behavioral Heston: rational = the Heston price; near-zero vol-of-vol = the
	// log-normal CPT price
	hp := fin.Heston_Params {
		v0    = 0.04,
		kappa = 2,
		theta = 0.05,
		sigma = 0.5,
		rho   = -0.7,
	}
	eh := 0.0
	for K in ([]f64{80, 100, 120}) {
		for Tm in ([]f64{0.1, 1}) {
			want := fin.heston_price(S, K, Tm, r, hp, .Call, 1000)
			got := bfin.behavioral_heston_price(S, K, Tm, r, hp, bfin.RATIONAL_PARAMS, .Call)
			eh = max(eh, abs(got - want) / max(want, 0.01))
		}
	}
	flat := fin.Heston_Params {
		v0    = 0.04,
		kappa = 1,
		theta = 0.04,
		sigma = 1e-4,
		rho   = 0,
	}
	el := 0.0
	for K in ([]f64{90, 110}) {
		for k in 0 ..< 4 {
			want := bfin.prospect_theory_price(
				S,
				K,
				T,
				r,
				r,
				sigma,
				bfin.TK92_PARAMS,
				types[k],
				sides[k],
				.Aggregated,
				4000,
			)
			got := bfin.behavioral_heston_price(
				S,
				K,
				T,
				r,
				flat,
				bfin.TK92_PARAMS,
				types[k],
				sides[k],
				.Aggregated,
			)
			el = max(el, abs(got - want) / want)
		}
	}
	expect(
		&passed,
		&total,
		eh < 5e-5 && el < 2e-5,
		fmt.tprintf(
			"behavioral Heston: rational = heston_price (relative error %.1e); vol-of-vol -> 0 = the log-normal TK92 prices (%.1e)",
			eh,
			el,
		),
	)

	fmt.printf("Prospect theory reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
