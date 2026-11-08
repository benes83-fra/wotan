package tests

import ml "../wotan/analytics/ML"
import bfin "../wotan/behavioral_finance"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Momentum (Jegadeesh & Titman 1993), long-term reversal (De Bondt & Thaler 1985)
// and the frog in the pan (Da, Gurun & Warachka 2014), behavioral_finance/momentum.odin
//   momentum_reference_test: past returns (compounded with a skip, summed market-
//     adjusted), ID / ID_Z / PRET on uneven months with zero-return days and a gap, and
//     quantile / sequential / independent sorts with ties = separate pandas / numpy
//     implementations; DGW's examples; DGW Appendix A: the model's momentum covariance
//     m (1 - m) N k^3 / (3 L) (eq. A4) and its simulation table (10,000 paths,
//     PRET x ID quintiles: PRET, FRET, momentum and the ID-continuation correlations,
//     against our mean over 20 such simulations; one row of the paper is off, see
//     there). On 10 seeds: A4 within 2.7 se; FRET cells within 3.6 sd, PRET within
//     3.4 sd (the left-out cell 3.6 - 4.9), momentum within 2.4, correlations within 2.6
//   momentum_test: A. DGW's model day by day (1000 stocks, 21-day months, 235 tested
//     months): the FIP investors' share m = 0.5 of the signals below k is released
//     later (2-month half-life); k = 0.03 rather than the paper's 0.02 (momentum grows
//     with k^3; with 0.02 and a 3-month half-life JT's t was -0.1 to 2.4 on 9 seeds).
//     JT 12-1 deciles held 6 months, continuous - discrete momentum (DGW Table 2A) and
//     the Fama-MacBeth PRET x ID slope (eq. 10), against a control with m = 0. B. an
//     overreaction market (sentiment overshoots news and decays) where DBT's losers
//     beat the winners, and its control. On 9 seeds: JT t 2.6 - 4.9, continuous -
//     discrete t 3.4 - 6.4, FM t -2.3 to -6.4; DBT t 17.5 - 20.7; controls |t| <= 2.2.
//     The single overreaction mechanism reverses at every horizon (JT t -16 to -19
//     there): momentum and long-term reversal together need both mechanisms.
//     C. the 24 large caps (printed)

@(private = "file")
_mo_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

@(private = "file")
_mo_close :: proc(a, b: f64, tol: f64) -> bool {
	if math.is_nan(b) {return math.is_nan(a)}
	return abs(a - b) <= tol
}

// DGW Appendix A table: rows ID discrete .. continuous, columns PRET winner .. loser
@(private = "file")
DGW_PRET := [5][5]f64 {
	{51.19, 22.20, 0.84, -21.57, -50.74},
	{56.08, 24.06, -0.69, -22.39, -55.15},
	{59.67, 23.99, -0.14, -24.71, -58.24},
	{70.71, 23.97, -1.52, -26.10, -68.73},
	{80.50, 26.06, -0.75, -27.44, -82.20},
}
@(private = "file")
DGW_FRET := [5][5]f64 {
	{-5.01, -5.70, -0.62, 5.73, 4.08},
	{-1.10, -2.14, 0.57, 1.92, 0.62},
	{1.41, 0.46, 0.35, -0.44, -1.45},
	{2.85, 3.11, -0.06, -2.76, -2.89},
	{6.88, 6.28, -0.44, -6.45, -6.27},
}
@(private = "file")
DGW_CORR := [5]f64{-0.65, -0.75, -0.02, -0.75, -0.67}
@(private = "file")
DGW_MOM := [5]f64{-9.09, -1.72, 2.86, 5.74, 13.15}

momentum_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Momentum, Reversal, Frog in the Pan Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	// reference values (scratch mom_ref.py: numpy / pandas, same LCG draws)
	PR_COMP := [5][12]f64 {
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			0.0022154205087068224,
			-0.06289188144778901,
			-0.0324951476856955,
			0.05015855057431362,
			0.061294265041660356,
			-0.015533196841081454,
			-0.0617383531104152,
			-0.022026305746872032,
			0.04991366144202347,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.0304634369391914,
			-0.016577154435799102,
			-0.014301212048033918,
			-0.014543882844544442,
			0.03062095245887342,
			-0.011122947847986286,
			0.002954990598145857,
			-0.039783576387329656,
			-0.018407410627010123,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.023374694540494567,
			-0.03862227713289046,
			0.032842575318994705,
			0.035529122636100086,
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			0.01572745465327352,
			-0.01631988402851936,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			0.004043798924991515,
			-0.0706986599918451,
			-0.06117754001464237,
			-0.017068454174631986,
			0.05103885886561965,
			0.05355383075271414,
			0.02172132454123865,
			0.036418184387328356,
			-0.0200784782839577,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.041197085640143016,
			-0.07170491598944717,
			-0.08222307629348602,
			-0.05178198999886652,
			-0.0437400675195736,
			-0.00045411134810346976,
			0.04892210204379466,
			0.08723433223533994,
			0.08947540329861625,
		},
	}
	PR_SUM := [5][12]f64 {
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			0.0012755699447657455,
			-0.041507967683263564,
			-0.01635714584681105,
			-0.034831747885729715,
			-0.023769254647882453,
			-0.05389710262090151,
			-0.06499766513216257,
			0.021316443349404343,
			0.021050283400899542,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.02552827293516177,
			-0.0320033162519226,
			-0.03469922643413904,
			-0.0584250240227922,
			-0.0575117705346259,
			-0.02198654260546927,
			-0.041298029013021924,
			-0.049252336549606535,
			0.023642781223409282,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.019473035344107673,
			-0.021988584234995927,
			0.03312849459340715,
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.007964395086634959,
			-0.02107630331499765,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.007371326310544997,
			-0.07399472367443989,
			-0.0795626670891102,
			-0.05610626446252548,
			0.018731422456201893,
			-0.0013428914330244138,
			0.03939868519752189,
			0.006646430725705452,
			-0.05523977500910636,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.03803157423483383,
			-0.10763961652247366,
			-0.12248500854229893,
			-0.1349658722711433,
			-0.06609649297473366,
			0.0053415711446635275,
			0.04482013539320615,
			0.12201691177463891,
			0.05602296857873683,
		},
	}
	ID_BOUNDS := []int{0, 19, 40, 63, 83, 105, 126, 145, 168}
	ID_0 := [4][8]f64 {
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.1746031746031746,
			-0.140625,
			-0.046153846153846156,
			-0.1111111111111111,
			-0.14516129032258066,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.06349206349206349,
			-0.09375,
			-0.1076923076923077,
			0.015873015873015872,
			0.04838709677419355,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.4603174603174603,
			-0.46875,
			-0.4153846153846154,
			-0.4603174603174603,
			-0.45161290322580644,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.16923076923076924,
			-0.19047619047619047,
			-0.12903225806451613,
		},
	}
	PRET_0 := [4][8]f64 {
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.13936719247899865,
			-0.10486229972734695,
			-0.03617139674763914,
			-0.05027314901147861,
			-0.06970949913879554,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			0.15296845192175645,
			0.21591534255914646,
			0.23840941849042396,
			0.1226104984377705,
			0.04077270427657309,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			0.4631784870480098,
			0.4851815980502534,
			0.4177225726789502,
			0.4636235361503216,
			0.5164673501669568,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			0.4569340402583639,
			0.4271000348972067,
			0.3603099318145404,
		},
	}
	ID_1 := [4][8]f64 {
		{
			math.nan_f64(),
			-0.25,
			-0.14285714285714285,
			-0.1,
			0.05263157894736842,
			-0.1,
			-0.25,
			-0.024390243902439025,
		},
		{
			math.nan_f64(),
			-0.07692307692307693,
			0.0,
			-0.17073170731707318,
			-0.14285714285714285,
			0.17073170731707318,
			-0.08108108108108109,
			-0.024390243902439025,
		},
		{
			math.nan_f64(),
			-0.47368421052631576,
			-0.5454545454545454,
			-0.3953488372093023,
			-0.38095238095238093,
			-0.5348837209302325,
			-0.45,
			-0.07317073170731707,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			-0.21951219512195122,
			-0.2,
			-0.14285714285714285,
			-0.15789473684210525,
			-0.36585365853658536,
		},
	}
	PRET_1 := [4][8]f64 {
		{
			math.nan_f64(),
			-0.10540428808380364,
			-0.1119606304040055,
			-0.03027472764317851,
			0.0018638935870436413,
			-0.05780436486660845,
			-0.06401801495157478,
			-0.011199121644590582,
		},
		{
			math.nan_f64(),
			0.0889307382334048,
			0.09516428796836918,
			0.1755503410825201,
			0.1696261767303704,
			0.011125432959666837,
			-0.012054746003885564,
			0.004980158794919642,
		},
		{
			math.nan_f64(),
			0.2794435385256231,
			0.3629944641387257,
			0.24612511579063856,
			0.23969563596822363,
			0.3432100020461828,
			0.33291808195499106,
			0.17924421749345454,
		},
		{
			math.nan_f64(),
			math.nan_f64(),
			math.nan_f64(),
			0.31669018989068887,
			0.26552530474776814,
			0.24778532554104227,
			0.2293670769841114,
			0.304225979372863,
		},
	}
	SA := [40]f64 {
		0.2,
		0.8,
		1.0,
		math.nan_f64(),
		0.1,
		0.0,
		0.4,
		0.7,
		0.7,
		0.2,
		0.6,
		0.9,
		0.7,
		0.1,
		0.9,
		0.2,
		0.7,
		math.nan_f64(),
		0.9,
		0.5,
		0.3,
		0.8,
		1.0,
		0.1,
		0.6,
		0.5,
		0.9,
		0.0,
		0.1,
		0.2,
		0.2,
		0.3,
		0.6,
		0.5,
		0.5,
		0.4,
		0.7,
		0.3,
		0.1,
		0.7,
	}
	SB := [40]f64 {
		0.38,
		0.38,
		0.21,
		0.23,
		0.13,
		0.78,
		0.35,
		0.04,
		0.66,
		0.8,
		0.74,
		0.66,
		0.01,
		0.3,
		0.39,
		0.43,
		0.96,
		0.14,
		0.22,
		0.99,
		0.46,
		0.15,
		0.38,
		0.97,
		0.69,
		math.nan_f64(),
		0.69,
		0.61,
		0.81,
		0.63,
		0.82,
		0.34,
		0.65,
		0.84,
		0.03,
		0.06,
		0.49,
		0.25,
		0.88,
		0.6,
	}
	G1 := [40]int {
		0,
		3,
		4,
		-1,
		0,
		0,
		2,
		3,
		3,
		1,
		2,
		4,
		3,
		0,
		4,
		1,
		3,
		-1,
		4,
		2,
		1,
		4,
		4,
		0,
		2,
		-1,
		4,
		0,
		0,
		1,
		1,
		1,
		2,
		2,
		2,
		2,
		3,
		1,
		0,
		3,
	}
	GS := [40]int {
		1,
		1,
		0,
		-1,
		0,
		2,
		1,
		0,
		2,
		2,
		2,
		2,
		0,
		0,
		2,
		1,
		3,
		-1,
		1,
		3,
		1,
		0,
		1,
		3,
		2,
		-1,
		3,
		1,
		2,
		2,
		3,
		0,
		1,
		3,
		0,
		0,
		1,
		0,
		3,
		2,
	}
	GI := [40]int {
		1,
		1,
		0,
		-1,
		0,
		3,
		1,
		0,
		2,
		3,
		2,
		2,
		0,
		0,
		1,
		1,
		3,
		-1,
		0,
		3,
		1,
		0,
		1,
		3,
		2,
		-1,
		2,
		2,
		3,
		2,
		3,
		1,
		2,
		3,
		0,
		0,
		1,
		0,
		3,
		2,
	}

	// 1. past returns: 5 assets, 12 periods (asset 2 has a NaN at 6), compounded over 3
	// periods skipping 1 (pandas (1 + r).rolling(3).prod().shift(1) - 1), summed over 4
	// net of a market (rolling(4).sum() of r - m)
	{
		x: u64 = 11
		R := make([][]f64, 5)
		mkt := make([]f64, 12)
		defer {for r in R {delete(r)}; delete(R); delete(mkt)}
		for i in 0 ..< 5 {
			R[i] = make([]f64, 12)
			for t in 0 ..< 12 {R[i][t] = 0.1 * (_mo_u(&x) - 0.5)}
		}
		for t in 0 ..< 12 {mkt[t] = 0.05 * (_mo_u(&x) - 0.5)}
		R[2][6] = math.nan_f64()
		pc := bfin.past_return(R, 3, 1)
		ps := bfin.past_return(R, 4, market = mkt, kind = .Sum)
		defer {bfin.signal_panel_free(pc); bfin.signal_panel_free(ps)}
		ok, err := true, 0.0
		for i in 0 ..< 5 {
			for t in 0 ..< 12 {
				ok &&=
					_mo_close(pc[i][t], PR_COMP[i][t], 1e-15) &&
					_mo_close(ps[i][t], PR_SUM[i][t], 1e-15)
				if !math.is_nan(PR_COMP[i][t]) {err = max(err, abs(pc[i][t] - PR_COMP[i][t]))}
				if !math.is_nan(PR_SUM[i][t]) {err = max(err, abs(ps[i][t] - PR_SUM[i][t]))}
			}
		}
		// by hand: (1.1 * 0.95 - 1) over periods 0 - 1 seen from 2
		h := bfin.past_return([][]f64{{0.1, -0.05, 0.2}}, 2, 1)
		defer bfin.signal_panel_free(h)
		ok &&= abs(h[0][2] - (1.1 * 0.95 - 1)) < 1e-15 && math.is_nan(h[0][1])
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"past returns (compounded, skip 1; summed net of the market) = pandas rolling, max diff %.1e, NaN windows NaN",
				err,
			),
		)
	}

	// 2. ID: DGW's sequences {2,2,2,2,2,2} and {1,1,1,1,1,7} (percent) both have ID -1;
	// a loser with mostly small gains is discrete; zero days count for ID, not ID_Z
	{
		a := bfin.information_discreteness_value({.02, .02, .02, .02, .02, .02}, 0.12)
		b := bfin.information_discreteness_value({.01, .01, .01, .01, .01, .07}, 0.12)
		c := bfin.information_discreteness_value({.01, .01, .01, -.2}, -0.17) // -1 * (1 - 3) / 4
		z := []f64{.01, 0, 0, -.01, .02}
		d := bfin.information_discreteness_value(z, 0.02) // (1 - 2) / 5
		e := bfin.information_discreteness_value(z, 0.02, .ID_Z) // (1 - 2) / 3
		f := bfin.information_discreteness_value({0, 0}, 0.0, .ID_Z)
		g := bfin.information_discreteness_value({.01, math.nan_f64()}, 0.01)
		expect(
			&passed,
			&total,
			a == -1 &&
			b == -1 &&
			c == 0.5 &&
			abs(d + 0.2) < 1e-15 &&
			abs(e + 1.0 / 3) < 1e-15 &&
			f == 0 &&
			math.is_nan(g),
			"ID: DGW's examples A and B -1, loser with small gains 0.5, zero days in ID and not in ID_Z, NaN day -> NaN",
		)
	}

	// 3. ID / ID_Z panel: 4 assets, 8 months of 19 - 23 days, returns on a 0.1% grid
	// (zero days), asset 3 has a NaN day in month 1; ID over 3 months skipping 1, ID_Z
	// over 2 skipping 0 = a numpy implementation
	{
		x: u64 = 5
		D := ID_BOUNDS[len(ID_BOUNDS) - 1]
		R := make([][]f64, 4)
		defer {for r in R {delete(r)}; delete(R)}
		for i in 0 ..< 4 {
			R[i] = make([]f64, D)
			for t in 0 ..< D {R[i][t] = math.round((0.04 * (_mo_u(&x) - 0.5) + 0.002 * f64(i)) * 1000) / 1000}
		}
		R[3][30] = math.nan_f64()
		zeros := 0
		for r in R {for v in r {if v == 0 {zeros += 1}}}
		id0, pr0 := bfin.information_discreteness(R, ID_BOUNDS, 3, 1)
		id1, pr1 := bfin.information_discreteness(R, ID_BOUNDS, 2, 0, .ID_Z)
		defer {
			bfin.signal_panel_free(id0); bfin.signal_panel_free(pr0)
			bfin.signal_panel_free(id1); bfin.signal_panel_free(pr1)
		}
		ok, err := true, 0.0
		for i in 0 ..< 4 {
			for k in 0 ..< 8 {
				ok &&=
					_mo_close(id0[i][k], ID_0[i][k], 1e-15) &&
					_mo_close(id1[i][k], ID_1[i][k], 1e-15)
				ok &&=
					_mo_close(pr0[i][k], PRET_0[i][k], 1e-14) &&
					_mo_close(pr1[i][k], PRET_1[i][k], 1e-14)
				if !math.is_nan(PRET_0[i][k]) {err = max(err, abs(pr0[i][k] - PRET_0[i][k]))}
				if !math.is_nan(PRET_1[i][k]) {err = max(err, abs(pr1[i][k] - PRET_1[i][k]))}
			}
		}
		expect(
			&passed,
			&total,
			ok && zeros > 0,
			fmt.tprintf(
				"ID (3 months skip 1) and ID_Z (2 months) panels with %d zero days = numpy, PRET max diff %.1e",
				zeros,
				err,
			),
		)
	}

	// 4. sorts of 40 assets with ties and NaNs: quintiles of A, then quartiles of B
	// within them (sequential) or on all (independent) = numpy (rank r -> r n / count)
	{
		g1, gs, gi, h := make([]int, 40), make([]int, 40), make([]int, 40), make([]int, 40)
		defer {delete(g1); delete(gs); delete(gi); delete(h)}
		bfin.double_sort(SA[:], SB[:], 5, 4, g1, gs)
		bfin.double_sort(SA[:], SB[:], 5, 4, h, gi, .Independent)
		ok := true
		for i in 0 ..< 40 {ok &&= g1[i] == G1[i] && gs[i] == GS[i] && gi[i] == GI[i] && h[i] == G1[i]}
		// the double-sort portfolio: +1 on the top A quintile, -1 on the bottom, within B group 2
		w := bfin.double_sort_portfolio(SA[:], SB[:], 2, 5, 4)
		defer delete(w)
		nl, ns := 0, 0
		for i in 0 ..< 40 {
			if GS[i] == 2 && G1[i] == 4 {nl += 1}
			if GS[i] == 2 && G1[i] == 0 {ns += 1}
		}
		sl, ss := 0.0, 0.0
		for i in 0 ..< 40 {
			want := 0.0
			if GS[i] == 2 && G1[i] == 4 {want = 1 / f64(nl)}
			if GS[i] == 2 && G1[i] == 0 {want = -1 / f64(ns)}
			ok &&= w[i] == want
			if w[i] > 0 {sl += w[i]} else {ss += w[i]}
		}
		expect(
			&passed,
			&total,
			ok && abs(sl - 1) < 1e-15 && abs(ss + 1) < 1e-15,
			"quantile, sequential and independent double sorts (ties, NaNs) = numpy; double-sort portfolio weights",
		)
	}

	// 5. DGW Appendix A: N = 250 signals uniform on [-L, L]; FIP investors (share m)
	// ignore those below k until period 2: P1 = s1 - m sum s_i 1{|s_i| < k}, FRET =
	// s1 - P1. Cov(P1, FRET) = m (1 - m) N k^3 / (3 L) (eq. A4), 100,000 paths.
	// Their table: 10,000 paths sorted on P1 quintiles, then ID quintiles (ID from the
	// signs of the draws, sgn(P1)); the paper's numbers are one simulation, so each
	// cell has to agree within 4 sqrt(2) standard errors (ours, both draws)
	{
		Mm, K, Lw, N :: 0.5, 0.02, 0.05, 250
		path :: proc(s: []f64) -> (p1, fret: f64) {
			s1, tr := 0.0, 0.0
			for &v in s {
				v = Lw * (2 * rand.float64() - 1)
				s1 += v
				if abs(v) < K {tr += v}
			}
			return s1 - Mm * tr, Mm * tr
		}
		s := make([]f64, N)
		defer delete(s)
		NC :: 100_000
		mp, mf, cpf := 0.0, 0.0, 0.0
		prods := make([]f64, NC)
		defer delete(prods)
		pp := make([]f64, NC)
		ff := make([]f64, NC)
		defer {delete(pp); delete(ff)}
		for j in 0 ..< NC {
			pp[j], ff[j] = path(s)
			mp += pp[j] / NC
			mf += ff[j] / NC
		}
		for j in 0 ..< NC {prods[j] = (pp[j] - mp) * (ff[j] - mf); cpf += prods[j] / NC}
		vp := 0.0
		for v in prods {vp += (v - cpf) * (v - cpf) / (NC - 1)}
		se := math.sqrt(vp / NC)
		want := Mm * (1 - Mm) * N * K * K * K / (3 * Lw)
		expect(
			&passed,
			&total,
			abs(cpf - want) < 4 * se,
			fmt.tprintf(
				"DGW eq. A4: Cov(PRET, FRET) %.6f vs m(1-m)N k^3/(3L) = %.6f (se %.6f)",
				cpf,
				want,
				se,
			),
		)

		// the table: REPS simulations of P paths each; the paper's single draw is compared
		// with our mean over the draws, within 4 sd sqrt(1 + 1 / REPS) (sd over draws:
		// with ID on a grid of 2 / N, the cells' composition moves between draws and a
		// cell's PRET varies by a few points, far more than its within-cell se)
		P, REPS :: 10_000, 20
		p1 := make([]f64, P)
		fr := make([]f64, P)
		id := make([]f64, P)
		g1 := make([]int, P)
		g2 := make([]int, P)
		defer {delete(p1); delete(fr); delete(id); delete(g1); delete(g2)}
		// per draw: 25 PRET cells, 25 FRET cells, 5 correlations, 5 momentum values
		Stat :: struct {
			sum, sq: f64,
		}
		cp, cf: [5][5]Stat
		cc, cm: [5]Stat
		add :: proc(s: ^Stat, v: f64) {s.sum += v; s.sq += v * v}
		for _ in 0 ..< REPS {
			for j in 0 ..< P {
				p1[j], fr[j] = path(s)
				id[j] = bfin.information_discreteness_value(s, p1[j])
			}
			bfin.double_sort(p1, id, 5, 5, g1, g2)
			// cell [row = ID discrete (4) .. continuous (0)][column = winner (4) .. loser (0)]
			n, sp, sf: [5][5]f64
			for j in 0 ..< P {
				r, c := 4 - g2[j], 4 - g1[j]
				n[r][c] += 1
				sp[r][c] += 100 * p1[j]
				sf[r][c] += 100 * fr[j]
			}
			for r in 0 ..< 5 {
				for c in 0 ..< 5 {add(&cp[r][c], sp[r][c] / n[r][c]); add(&cf[r][c], sf[r][c] / n[r][c])}
				add(&cm[r], sf[r][0] / n[r][0] - sf[r][4] / n[r][4])
			}
			// correlation of ID and the return continuation within each PRET quintile:
			// FRET for the top three, -FRET for the bottom two (the paper's signs: ID is
			// negatively correlated with momentum for past winners and past losers)
			for c in 0 ..< 5 {
				q := 4 - c
				mx, my, sxx, syy, sxy, cnt := 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
				for j in 0 ..< P {if g1[j] == q {mx += id[j]; my += fr[j]; cnt += 1}}
				mx /= cnt
				my /= cnt
				for j in 0 ..< P {
					if g1[j] != q {continue}
					sxx += (id[j] - mx) * (id[j] - mx)
					syy += (fr[j] - my) * (fr[j] - my)
					sxy += (id[j] - mx) * (fr[j] - my)
				}
				add(&cc[c], (sxy if c < 3 else -sxy) / math.sqrt(sxx * syy))
			}
		}
		// distance from the paper's value in units of sd sqrt(1 + 1 / REPS)
		dist :: proc(s: Stat, paper: f64) -> (mean, z: f64) {
			mean = s.sum / REPS
			sd := math.sqrt(max(s.sq - REPS * mean * mean, 0) / (REPS - 1))
			return mean, abs(mean - paper) / (sd * math.sqrt(1 + 1.0 / REPS))
		}
		// the paper's row for the second most continuous ID quintile is left out of the
		// PRET and momentum checks (printed, and kept in the monotonicity check): its
		// winners' PRET is 70.71 against 65.5 (sd 0.9 over draws) here and in a separate
		// numpy simulation, 3.6 - 4.9 sd off on every seed tried (its losers, -68.73
		// against -65.3, are also more extreme), and its momentum 5.74 against 6.6 (2.4 -
		// 4.2 sd); its FRET cells agree
		zf, zp, zc, zm := 0.0, 0.0, 0.0, 0.0
		odd_mean, odd_z := dist(cp[3][0], DGW_PRET[3][0])
		mom, cs: [5]f64
		for r in 0 ..< 5 {
			for c in 0 ..< 5 {
				_, z1 := dist(cf[r][c], DGW_FRET[r][c])
				_, z2 := dist(cp[r][c], DGW_PRET[r][c])
				zf = max(zf, z1)
				if r != 3 || c != 0 {zp = max(zp, z2)}
			}
			z: f64
			mom[r], z = dist(cm[r], DGW_MOM[r])
			if r != 3 {zm = max(zm, z)}
			cs[r], z = dist(cc[r], DGW_CORR[r])
			zc = max(zc, z)
		}
		expect(
			&passed,
			&total,
			zf < 4 &&
			zp < 4 &&
			zm < 4 &&
			mom[4] > mom[3] &&
			mom[3] > mom[2] &&
			mom[2] > mom[1] &&
			mom[1] > mom[0],
			fmt.tprintf(
				"DGW Appendix A, %d x %d paths, 25 PRET x ID cells: FRET, PRET within %.1f, %.1f sd of the paper (ID quintile 4 winners' PRET %.2f, paper 70.71, %.1f sd, left out); momentum discrete .. continuous %.2f %.2f %.2f %.2f %.2f%% (paper -9.09 -1.72 2.86 5.74 13.15, within %.1f sd but quintile 4)",
				REPS,
				P,
				zf,
				zp,
				odd_mean,
				odd_z,
				mom[0],
				mom[1],
				mom[2],
				mom[3],
				mom[4],
				zm,
			),
		)
		expect(
			&passed,
			&total,
			zc < 4,
			fmt.tprintf(
				"DGW Appendix A: corr(ID, continuation) winner .. loser %.2f %.2f %.2f %.2f %.2f (paper -0.65 -0.75 -0.02 -0.75 -0.67, within %.1f sd)",
				cs[0],
				cs[1],
				cs[2],
				cs[3],
				cs[4],
				zc,
			),
		)
	}

	fmt.printf("Momentum reference: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// [period][asset] from [asset][period]
@(private = "file")
_mo_transpose :: proc(X: [][]f64) -> [][]f64 {
	out := make([][]f64, len(X[0]))
	for t in 0 ..< len(X[0]) {
		out[t] = make([]f64, len(X))
		for i in 0 ..< len(X) {out[t][i] = X[i][t]}
	}
	return out
}

// mean monthly spread of the overlapping strategy with weights W [month][asset] on
// returns Rt [month][asset] from month `first` on, Newey-West t (6 lags, as DGW)
@(private = "file")
_mo_strategy :: proc(W, Rt: [][]f64, hold, first: int) -> (mean, t: f64) {
	o := bfin.overlapping_strategy_returns(W, Rt, hold, allocator = context.temp_allocator)
	defer bfin.overlapping_returns_free(&o)
	return mean_t(o.spread[first:], 6)
}

// JT / DBT weights per month from a signal panel S [asset][month]: long the top
// `fraction`, short the bottom one
@(private = "file")
_mo_weights :: proc(S: [][]f64, fraction := 0.1) -> [][]f64 {
	W := make([][]f64, len(S[0]))
	col := make([]f64, len(S))
	defer delete(col)
	for t in 0 ..< len(S[0]) {
		for i in 0 ..< len(S) {col[i] = S[i][t]}
		W[t] = bfin.long_short_portfolio(col, fraction)
	}
	return W
}

// Fama-MacBeth (DGW eq. 10) of the six-month return after month t on PRET, ID and
// PRET x ID for months first .. M - 7, Newey-West t (6 lags): the PRET x ID slope
@(private = "file")
_mo_fm :: proc(PRET, ID, MR: [][]f64, first: int) -> (b3, t3: f64) {
	N, M := len(PRET), len(PRET[0])
	Xs := make([dynamic]l.Matrix(f64))
	Ys := make([dynamic][]f64)
	defer {
		for i in 0 ..< len(Xs) {l.matrix_free(&Xs[i]); delete(Ys[i])}
		delete(Xs)
		delete(Ys)
	}
	for t in first ..< M - 6 {
		n := 0
		for i in 0 ..< N {if !math.is_nan(PRET[i][t]) {n += 1}}
		X := l.matrix_new(f64, n, 4)
		y := make([]f64, n)
		j := 0
		for i in 0 ..< N {
			p := PRET[i][t]
			if math.is_nan(p) {continue}
			g := 1.0
			for s in t + 1 ..= t + 6 {g *= 1 + MR[i][s]}
			X.data[4 * j], X.data[4 * j + 1], X.data[4 * j + 2], X.data[4 * j + 3] =
				1, p, ID[i][t], p * ID[i][t]
			y[j] = g - 1
			j += 1
		}
		append(&Xs, X)
		append(&Ys, y)
	}
	fm := ml.fama_macbeth(Xs[:], Ys[:], 6)
	defer ml.fama_macbeth_result_free(&fm)
	return fm.coef[3], fm.t_stat[3]
}

@(private = "file")
_mo_free :: proc(X: [][]f64) {for r in X {delete(r)}; delete(X)}

// FIP market (DGW's illustrative model, day by day): signals uniform on [-FIP_L,
// FIP_L]; the FIP investors (share m) leave the part m s of each signal below FIP_K
// out of the price, a pool released at a constant rate a day (half-life FIP_HL months)
FIP_K :: 0.03
FIP_L :: 0.05
FIP_HL :: 2.0

@(private = "file")
_mo_fip_market :: proc(N, days: int, m: f64) -> [][]f64 {
	rel := 1 - math.pow(0.5, 1 / (21 * FIP_HL))
	R := make([][]f64, N)
	for i in 0 ..< N {
		R[i] = make([]f64, days)
		pool := 0.0
		for d in 0 ..< days {
			s := FIP_L * (2 * rand.float64() - 1)
			late := m * s if abs(s) < FIP_K else 0
			out := rel * pool
			pool += late - out
			R[i][d] = s - late + out
		}
	}
	return R
}

momentum_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Momentum, Long-Term Reversal and the Frog in the Pan ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. FIP markets: m = 0.5 (DGW) and m = 0 (control: the signals are the returns)
	N, D, BURN, M :: 1000, 21, 12, 264
	months := make([]int, M + 1)
	defer delete(months)
	for k in 0 ..= M {months[k] = D * k}
	FIRST :: BURN + 11 + 6 // first month whose six cohorts all exist
	fip: [2][6]f64 // [market][JT mean, t, cont - disc mean, t, FM b3, t]
	mom_q: [2][5]f64
	for mm, k in ([2]f64{0.5, 0}) {
		R := _mo_fip_market(N, M * D, mm)
		defer _mo_free(R)
		MR := monthly_compound(R, months)
		ID, PRET := bfin.information_discreteness(R, months)
		MRt := _mo_transpose(MR)
		defer {_mo_free(MR); _mo_free(MRt); bfin.signal_panel_free(ID)
			bfin.signal_panel_free(PRET)}
		for t in 0 ..< BURN {for i in 0 ..< N {PRET[i][t + 11] = math.nan_f64()}} 	// burn-in
		// JT: deciles of the 12-1 return, held 6 months
		W := _mo_weights(PRET)
		fip[k][0], fip[k][1] = _mo_strategy(W, MRt, 6, FIRST)
		_mo_free(W)
		// DGW: PRET quintiles, then ID quintiles; winner - loser within each ID quintile
		sp: [5][]f64
		col, cid := make([]f64, N), make([]f64, N)
		for g in 0 ..< 5 {
			Wg := make([][]f64, M)
			for t in 0 ..< M {
				for i in 0 ..< N {col[i], cid[i] = PRET[i][t], ID[i][t]}
				Wg[t] = bfin.double_sort_portfolio(col, cid, g)
			}
			o := bfin.overlapping_strategy_returns(Wg, MRt, 6)
			sp[g] = o.spread
			delete(o.long)
			delete(o.short)
			_mo_free(Wg)
			mom_q[k][g], _ = mean_t(sp[g][FIRST:])
		}
		delete(col)
		delete(cid)
		diff := make([]f64, M - FIRST)
		for t in FIRST ..< M {diff[t - FIRST] = sp[0][t] - sp[4][t]}
		fip[k][2], fip[k][3] = mean_t(diff, 6)
		delete(diff)
		for g in 0 ..< 5 {delete(sp[g])}
		fip[k][4], fip[k][5] = _mo_fm(PRET, ID, MR, BURN + 11)
	}
	fmt.printf(
		"FIP market (m 0.5, k %.2f, L %.2f, delayed part released with a %.0f-month half-life): winner - loser by ID quintile, continuous .. discrete, %% a month: %.2f %.2f %.2f %.2f %.2f\n",
		FIP_K,
		FIP_L,
		FIP_HL,
		100 * mom_q[0][0],
		100 * mom_q[0][1],
		100 * mom_q[0][2],
		100 * mom_q[0][3],
		100 * mom_q[0][4],
	)
	expect(
		&passed,
		&total,
		fip[0][1] > 2 && fip[0][3] > 2 && fip[0][5] < -2,
		fmt.tprintf(
			"FIP market, %d months: JT 12-1 decile momentum %.2f%% a month (t %.1f); continuous - discrete momentum %.2f%% (t %.1f); FM PRET x ID %.3f (t %.1f)",
			M - FIRST,
			100 * fip[0][0],
			fip[0][1],
			100 * fip[0][2],
			fip[0][3],
			fip[0][4],
			fip[0][5],
		),
	)
	expect(
		&passed,
		&total,
		abs(fip[1][1]) < 3 && abs(fip[1][3]) < 3 && abs(fip[1][5]) < 3,
		fmt.tprintf(
			"control (m = 0): JT %.2f%% (t %.1f); continuous - discrete %.2f%% (t %.1f); FM PRET x ID t %.1f",
			100 * fip[1][0],
			fip[1][1],
			100 * fip[1][2],
			fip[1][3],
			fip[1][5],
		),
	)

	// B. overreaction markets (monthly): r = f + e + X_t - X_{t-1}, news e ~ N(0, OV_S),
	// sentiment X_t = OV_PHI X_{t-1} + b e_t overshoots and decays; market f ~ N(0.5%,
	// 4%). b = OV_B and b = 0 (control). De Bondt & Thaler: 36-month summed
	// market-adjusted returns, losers - winners (deciles), held 36 months
	OV_S, OV_PHI, OV_B :: 0.08, 0.97, 0.5
	NB, MB :: 1000, 600
	ov: [2][4]f64 // [market][DBT mean, t, JT mean, t]
	for b, k in ([2]f64{OV_B, 0}) {
		R := make([][]f64, NB)
		X := make([]f64, NB)
		mkt := make([]f64, MB)
		defer {_mo_free(R); delete(X); delete(mkt)}
		for i in 0 ..< NB {
			R[i] = make([]f64, MB)
			X[i] = rand.float64_normal(0, b * OV_S / math.sqrt(1 - OV_PHI * OV_PHI))
		}
		for t in 0 ..< MB {
			f := rand.float64_normal(0.005, 0.04)
			for i in 0 ..< NB {
				e := rand.float64_normal(0, OV_S)
				x := OV_PHI * X[i] + b * e
				R[i][t] = f + e + x - X[i]
				X[i] = x
				mkt[t] += R[i][t] / NB
			}
		}
		Rt := _mo_transpose(R)
		defer _mo_free(Rt)
		CU := bfin.past_return(R, 36, market = mkt, kind = .Sum)
		W := _mo_weights(CU)
		m, t := _mo_strategy(W, Rt, 36, 36 + 36)
		ov[k][0], ov[k][1] = -m, -t
		_mo_free(W)
		bfin.signal_panel_free(CU)
		PR := bfin.past_return(R, 11, 1)
		W = _mo_weights(PR)
		ov[k][2], ov[k][3] = _mo_strategy(W, Rt, 6, 12 + 6)
		_mo_free(W)
		bfin.signal_panel_free(PR)
	}
	expect(
		&passed,
		&total,
		ov[0][1] > 2 && abs(ov[1][1]) < 3,
		fmt.tprintf(
			"overreaction market (b %.1f, phi %.2f), %d months: DBT 36-month losers - winners %.2f%% a month (t %.1f), JT 12-1 %.2f%% (t %.1f); control: DBT %.2f%% (t %.1f), JT t %.1f",
			OV_B,
			OV_PHI,
			MB - 72,
			100 * ov[0][0],
			ov[0][1],
			100 * ov[0][2],
			ov[0][3],
			100 * ov[1][0],
			ov[1][1],
			ov[1][3],
		),
	)

	// C. the 24 large caps, 21-day months, 10 years (printed): JT 12-1 quintiles held 6
	// months, DGW's Fama-MacBeth, DBT 36 / 36 quintiles (few months)
	NS := len(LARGE_CAP_TICKERS)
	DP, _, dates, ok := yahoo_adjclose_panel(LARGE_CAP_TICKERS, .TenYears, 2000)
	if !ok {
		fmt.println("Not enough data (offline?): large-cap check skipped")
	} else {
		MM := (len(dates) - 1) / D
		R := make([][]f64, NS)
		defer _mo_free(R)
		for i in 0 ..< NS {
			R[i] = make([]f64, MM * D)
			for t in 0 ..< MM * D {R[i][t] = DP[(t + 1) * NS + i] / DP[t * NS + i] - 1}
		}
		lm := make([]int, MM + 1)
		defer delete(lm)
		for k in 0 ..= MM {lm[k] = D * k}
		MR := monthly_compound(R, lm)
		MRt := _mo_transpose(MR)
		ID, PRET := bfin.information_discreteness(R, lm)
		defer {_mo_free(MR); _mo_free(MRt); bfin.signal_panel_free(ID)
			bfin.signal_panel_free(PRET)}
		W := _mo_weights(PRET, 0.2)
		jm, jt := _mo_strategy(W, MRt, 6, 11 + 6)
		_mo_free(W)
		b3, t3 := _mo_fm(PRET, ID, MR, 11)
		mk := make([]f64, MM)
		defer delete(mk)
		for t in 0 ..< MM {for i in 0 ..< NS {mk[t] += MR[i][t] / f64(NS)}}
		CU := bfin.past_return(MR, 36, market = mk, kind = .Sum)
		W = _mo_weights(CU, 0.2)
		dm, dt := _mo_strategy(W, MRt, 36, 72)
		_mo_free(W)
		bfin.signal_panel_free(CU)
		fmt.printf(
			"Large caps, %d months: JT 12-1 quintiles %.2f%% a month (t %.2f, %d months); FM PRET x ID %.3f (t %.2f); DBT losers - winners %.2f%% (t %.2f, %d months)\n",
			MM,
			100 * jm,
			jt,
			MM - 17,
			b3,
			t3,
			-100 * dm,
			-dt,
			MM - 72,
		)
	}

	fmt.printf("Momentum: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
