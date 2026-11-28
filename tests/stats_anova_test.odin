package tests

import w "../wotan/core"
import st "../wotan/stats"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"

// Analysis of variance in wotan/stats.
//
// - R's PlantGrowth (one-way, Tukey) and ToothGrowth (two-way) data; the
//   unbalanced design drops every 7th observation of ToothGrowth. Sums of
//   squares, F and degrees of freedom are exact rational computations, the
//   p-values 40-digit (mpmath) F tails of the exact statistics; tolerance
//   1e-12. They reproduce R's printed aov, oneway.test and TukeyHSD output.
// - The studentized range distribution against Student's t (k = 2, where
//   q / sqrt(2) is |t|) and against 30-digit double integrals (mpmath,
//   Gauss-Legendre panels in s and z, an integration independent of the
//   library's). Tukey's p-values against the same integrals.
// - Invalid input gives ok = false; NaN where a statistic is undefined.
@(private = "file")
PG_CTRL := []f64{4.17, 5.58, 5.18, 6.11, 4.5, 4.61, 5.17, 4.53, 5.33, 5.14}
@(private = "file")
PG_TRT1 := []f64{4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69}
@(private = "file")
PG_TRT2 := []f64{6.31, 5.12, 5.54, 5.5, 5.37, 5.29, 4.92, 6.15, 5.8, 5.26}
// ToothGrowth len by cell: VC 0.5, 1, 2 mg, then OJ 0.5, 1, 2 mg
@(private = "file")
TOOTH := [6][]f64 {
	{4.2, 11.5, 7.3, 5.8, 6.4, 10.0, 11.2, 11.2, 5.2, 7.0},
	{16.5, 16.5, 15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5},
	{23.6, 18.5, 33.9, 25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5},
	{15.2, 21.5, 17.6, 9.7, 14.5, 10.0, 8.2, 9.4, 16.5, 9.7},
	{19.7, 23.3, 23.6, 26.4, 20.0, 25.2, 25.8, 21.2, 14.5, 27.3},
	{25.5, 26.4, 22.4, 24.5, 24.8, 30.9, 26.4, 27.3, 29.4, 23.0},
}
@(private = "file")
W_PG_EQUAL := []f64 {
	4.846087862380135,
	0.015909958325622926,
	2.0,
	27.0,
	3.7663399999999996,
	10.492090000000001,
	0.2641482968321196,
	0.20407884598997086,
}
@(private = "file")
W_PG_WELCH := []f64 {
	5.180972408113187,
	0.017392821490170005,
	2.0,
	17.128418616644133,
	3.7663399999999996,
	10.492090000000001,
	0.2641482968321196,
	0.20407884598997086,
}
@(private = "file")
W_PG_BF := []f64 {
	4.846087862380135,
	0.0179274252722879,
	2.0,
	22.208370194282104,
	3.7663399999999996,
	10.492090000000001,
	0.2641482968321196,
	0.20407884598997086,
}
@(private = "file")
W_PG_OFFSET := []f64 {
	4.846087862616175,
	0.015909958322859522,
	2.0,
	27.0,
	3.7663400002255476,
	10.492090000117281,
	0.264148296841587,
	0.20407884599993945,
}
@(private = "file")
W_TG_T1 := []f64 {
	1.0,
	205.35,
	15.57197945249724,
	0.0002311828097734249,
	2.0,
	2426.434333333333,
	91.99996489286707,
	4.046291195992231e-18,
	2.0,
	108.31900000000002,
	4.106991094022519,
	0.021860268964790992,
	54.0,
	712.106,
	math.QNAN_F64,
	math.QNAN_F64,
}
@(private = "file")
W_TG_T2 := []f64 {
	1.0,
	205.35,
	15.57197945249724,
	0.0002311828097734249,
	2.0,
	2426.434333333333,
	91.99996489286707,
	4.046291195992231e-18,
	2.0,
	108.31900000000002,
	4.106991094022519,
	0.021860268964790992,
	54.0,
	712.106,
	math.QNAN_F64,
	math.QNAN_F64,
}
@(private = "file")
W_TG_T3 := []f64 {
	1.0,
	205.35,
	15.57197945249724,
	0.0002311828097734249,
	2.0,
	2426.434333333333,
	91.99996489286707,
	4.046291195992231e-18,
	2.0,
	108.31900000000002,
	4.106991094022519,
	0.021860268964790992,
	54.0,
	712.106,
	math.QNAN_F64,
	math.QNAN_F64,
}
// unbalanced rows: 51
@(private = "file")
W_TGU_T1 := []f64 {
	1.0,
	139.61060904977373,
	11.028904184222103,
	0.0017861710171803073,
	2.0,
	2363.7523103402045,
	93.36538936192684,
	9.663797833287781e-17,
	2.0,
	44.47605119825709,
	1.7567508318198584,
	0.18423517350526475,
	45.0,
	569.6374999999999,
	math.QNAN_F64,
	math.QNAN_F64,
}
@(private = "file")
W_TGU_T2 := []f64 {
	1.0,
	132.52880174291937,
	10.469458345757385,
	0.0022784710924732203,
	2.0,
	2363.7523103402045,
	93.36538936192684,
	9.663797833287781e-17,
	2.0,
	44.47605119825709,
	1.7567508318198584,
	0.18423517350526475,
	45.0,
	569.6374999999999,
	math.QNAN_F64,
	math.QNAN_F64,
}
@(private = "file")
W_TGU_T3 := []f64 {
	1.0,
	132.52880174291937,
	10.469458345757385,
	0.0022784710924732203,
	2.0,
	2331.7117701525053,
	92.09982634294859,
	1.2372787898687994e-16,
	2.0,
	44.47605119825709,
	1.7567508318198584,
	0.18423517350526475,
	45.0,
	569.6374999999999,
	math.QNAN_F64,
	math.QNAN_F64,
}
@(private = "file")
W_TGU_MAIN := []f64 {
	1.0,
	132.52880174291937,
	10.142837053120688,
	0.0025721969682660534,
	2.0,
	2363.7523103402045,
	90.45261936430049,
	7.710338046208223e-17,
	47.0,
	614.113551198257,
	math.QNAN_F64,
	math.QNAN_F64,
}
// k, df, q, cdf, sf from 30-digit integrals
@(private = "file")
SR_REF := [][5]f64 {
	{3.0, 10.0, 3.5, 0.92289668916158826546, 0.077103310838411829548},
	{3.0, 5.0, 20.0, 0.99992462975370589255, 0.000075370246294204242644},
	{4.0, math.INF_F64, 3.6, 0.94680542456049425188, 0.053194575439505789387},
	{4.0, math.INF_F64, 0.1, 0.00012674944167546378842, 0.99987325055832457751},
	{5.0, 20.0, 0.5, 0.0036329946071928338801, 0.99636700539280725153},
	{10.0, 30.0, 8.0, 0.99986020082448648795, 0.00013979917551358936584},
	{10.0, 30.0, 4.5, 0.91653775754058786796, 0.083462242459412212014},
	{6.0, 2.5, 7.0, 0.90385560262101548999, 0.096144397378984594766},
	{20.0, 100.0, 15.0, 0.99999999999999922939, 8.8247895783288058145e-16},
	{20.0, 100.0, 5.5, 0.97658639132024130123, 0.023413608679758790125},
	{3.0, 1000.0, 3.3, 0.94824251810979888583, 0.051757481890201206174},
	{50.0, 60.0, 6.0, 0.95406279264678808458, 0.04593720735321203381},
	{8.0, 12.0, 12.0, 0.99995992962205893189, 0.000040070377941145300961},
	{3.0, 40.0, 0.05, 0.000688909927095295864, 0.99931109007290479513},
	{5.0, 1000000.0, 4.0, 0.96230332510818565624, 0.0376966748918144276},
	{3.0, 1.0, 2.0, 0.44071150396798812275, 0.55928849603201197168},
	{4.0, 2.0, 0.3, 0.0043722368207030407913, 0.99562776317929705662},
	{7.0, 15.0, 1.2, 0.024905652538356205757, 0.97509434746164387353},
	{12.0, math.INF_F64, 9.0, 0.99999998704580942156, 1.2954190587707162074e-8},
}
// studentized range statistic and p-value of the PlantGrowth pairs (ctrl-trt1, ctrl-trt2, trt1-trt2)
@(private = "file")
W_TUKEY := []f64 {
	1.882022399688450286793714,
	0.39087114420210662264,
	2.505981308480038698035349,
	0.19799599129957238766,
	4.388003708168488984829063,
	0.012006423979493634713,
}

@(private = "file")
_aworst :: proc(got, want: []f64) -> f64 {
	if len(got) != len(want) {return math.inf_f64(1)}
	worst := 0.0
	for g, i in got {
		e := want[i]
		err: f64
		switch {
		case math.is_nan(e) || math.is_nan(g):
			err = 0 if math.is_nan(e) && math.is_nan(g) else math.inf_f64(1)
		case:
			err = abs(g - e) / max(abs(e), 1e-300)
		}
		worst = max(worst, err)
	}
	return worst
}

@(private = "file")
_oneway_err :: proc(r: st.Oneway_Result, ok: bool, want: []f64) -> f64 {
	if !ok {return math.inf_f64(1)}
	got := [8]f64 {
		r.statistic,
		r.p_value,
		r.df,
		r.df2,
		r.ss_between,
		r.ss_within,
		r.eta_squared,
		r.omega_squared,
	}
	return _aworst(got[:], want)
}

// df, sum of squares, F, p of every line of the table
@(private = "file")
_table_err :: proc(t: st.Anova_Table, ok: bool, want: []f64) -> f64 {
	if !ok {return math.inf_f64(1)}
	got := make([dynamic]f64, 0, context.temp_allocator)
	for e in t.terms {append(&got, e.df, e.sum_sq, e.f, e.p_value)}
	return _aworst(got[:], want)
}

// ToothGrowth as y, supp (0 = OJ, 1 = VC) and dose (0, 1, 2 for 0.5, 1, 2 mg)
@(private = "file")
_tooth :: proc(drop_every_7th: bool) -> (y: []f64, supp, dose: []int) {
	ys := make([dynamic]f64, 0, context.temp_allocator)
	ss := make([dynamic]int, 0, context.temp_allocator)
	ds := make([dynamic]int, 0, context.temp_allocator)
	i := 0
	for cell, c in TOOTH {
		for v in cell {
			if !(drop_every_7th && i % 7 == 3) {
				append(&ys, v)
				append(&ss, 1 if c < 3 else 0)
				append(&ds, c % 3)
			}
			i += 1
		}
	}
	return ys[:], ss[:], ds[:]
}

@(private = "file")
_Tally :: struct {
	worst: f64,
	name:  string,
}

@(private = "file")
_anote :: proc(t: ^_Tally, err: f64, name: string) {
	if !(err <= t.worst) {
		t.worst = err
		t.name = name
	}
}

stats_anova_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ANOVA ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	report :: proc(passed, total: ^int, t: _Tally, what: string, tol := 1e-12) {
		bad := !(t.worst <= tol)
		expect(
			passed,
			total,
			!bad,
			fmt.tprintf(
				"%s (worst %.1e%s%s)",
				what,
				t.worst,
				", " if bad else "",
				t.name if bad else "",
			),
		)
	}
	plant := [][]f64{PG_CTRL, PG_TRT1, PG_TRT2}

	// 1. one-way ANOVA
	{
		t: _Tally
		_anote(&t, _oneway_err(st.anova_oneway(plant), W_PG_EQUAL), "equal")
		_anote(&t, _oneway_err(st.anova_oneway(plant, .Welch), W_PG_WELCH), "Welch")
		_anote(&t, _oneway_err(st.anova_oneway(plant, .Brown_Forsythe), W_PG_BF), "Brown-Forsythe")
		// values near 1e6 that differ in the decimals
		shifted := make([][]f64, 3, context.temp_allocator)
		for s, i in plant {
			shifted[i] = make([]f64, len(s), context.temp_allocator)
			for v, j in s {shifted[i][j] = v + 1e6}
		}
		_anote(&t, _oneway_err(st.anova_oneway(shifted), W_PG_OFFSET), "offset 1e6")
		report(
			&passed,
			&total,
			t,
			"one-way ANOVA on PlantGrowth: classical, Welch, Brown-Forsythe (F, p, df, sums of squares, eta², omega²)",
		)
	}

	// 2. factorial ANOVA
	{
		t: _Tally
		names := []string{"supp", "dose"}
		y, supp, dose := _tooth(false)
		fs := [][]int{supp, dose}
		wants := [3][]f64{W_TG_T1, W_TG_T2, W_TG_T3}
		wantu := [3][]f64{W_TGU_T1, W_TGU_T2, W_TGU_T3}
		for tp, i in st.Anova_Type {
			tb, ok := st.anova(y, fs, names, type = tp, allocator = context.temp_allocator)
			_anote(&t, _table_err(tb, ok, wants[i]), fmt.tprintf("balanced %v", tp))
		}
		yu, su, du := _tooth(true)
		fu := [][]int{su, du}
		for tp, i in st.Anova_Type {
			tb, ok := st.anova(yu, fu, names, type = tp, allocator = context.temp_allocator)
			_anote(&t, _table_err(tb, ok, wantu[i]), fmt.tprintf("unbalanced %v", tp))
		}
		tb, ok := st.anova(yu, fu, names, interactions = false, allocator = context.temp_allocator)
		_anote(&t, _table_err(tb, ok, W_TGU_MAIN), "main effects")
		report(
			&passed,
			&total,
			t,
			"factorial ANOVA on ToothGrowth, balanced and unbalanced, Types I / II / III and main effects only (df, SS, F, p of every line)",
		)

		// Type I lines add up to the total; names; effect sizes
		t1, _ := st.anova(yu, fu, names, type = .I)
		defer st.anova_table_destroy(&t1)
		sum, eta := 0.0, 0.0
		for e in t1.terms {sum += e.sum_sq}
		for e in t1.terms[:3] {eta += e.eta_squared}
		r := t1.terms[3]
		e := t1.terms[0]
		pe := e.sum_sq / (e.sum_sq + r.sum_sq)
		om := (e.sum_sq - e.df * r.mean_sq) / (e.sum_sq + (f64(t1.n) - e.df) * r.mean_sq)
		names_ok :=
			len(t1.terms) == 4 &&
			t1.terms[0].name == "supp" &&
			t1.terms[1].name == "dose" &&
			t1.terms[2].name == "supp:dose" &&
			r.name == "Residual" &&
			math.is_nan(r.f) &&
			math.is_nan(r.p_value)
		expect(
			&passed,
			&total,
			names_ok &&
			abs(sum - t1.ss_tot) <= 1e-12 * t1.ss_tot &&
			abs(eta + r.sum_sq / t1.ss_tot - 1) <= 1e-14 &&
			abs(e.partial_eta_squared - pe) <= 1e-15 &&
			abs(e.omega_squared - om) <= 1e-15,
			"ANOVA table: term names, Type I sums of squares add up to the total, eta², partial eta², partial omega²",
		)
	}

	// 3. anova_from_df
	{
		y, supp, dose := _tooth(false)
		df := w.dataframe_new()
		defer w.destroy_dataframe(&df)
		len_col := w.column_new("len", .Float, len(y) + 1)
		sup := make([]string, len(y) + 1, context.temp_allocator)
		mg := make([]int, len(y) + 1, context.temp_allocator)
		doses := [3]int{500, 1000, 2000}
		for v, i in y {
			w.append_float(&len_col, v)
			sup[i] = "VC" if supp[i] == 1 else "OJ"
			mg[i] = doses[dose[i]]
		}
		w.append_null(&len_col) // a row with a null response is left out
		sup[len(y)], mg[len(y)] = "OJ", 500
		w.add_column(&df, len_col)
		w.add_column(&df, w.column_from_strings("supp", sup))
		w.add_column(&df, w.column_from_ints("dose", mg))
		tb, ok := st.anova_from_df(&df, "len", {"supp", "dose"})
		defer st.anova_table_destroy(&tb)
		_, bad1 := st.anova_from_df(&df, "supp", {"dose"}, allocator = context.temp_allocator)
		_, bad2 := st.anova_from_df(&df, "len", {"nope"}, allocator = context.temp_allocator)
		expect(
			&passed,
			&total,
			ok &&
			tb.n == 60 &&
			_table_err(tb, ok, W_TG_T2) <= 1e-12 &&
			tb.terms[2].name == "supp:dose" &&
			!bad1 &&
			!bad2,
			"anova_from_df: String and Int factors, null rows left out, same table as anova; a non-numeric response or missing column gives ok = false",
		)
	}

	// 4. studentized range distribution
	{
		t: _Tally
		// k = 2: q / sqrt(2) is |t| with df degrees of freedom
		for df in ([]f64{0.7, 3.5, 57, 1e6}) {
			for q in ([]f64{0.01, 1, 5, 25}) {
				d := st.Studentized_Range{2, df}
				x := q / math.SQRT_TWO
				ts := 2 * st.student_t_sf(st.Student_T{df}, x)
				tc := 1 - ts if ts > 0.5 else (st.student_t_cdf(st.Student_T{df}, x) - 0.5) * 2
				es := abs(st.studentized_range_sf(d, q) - ts) / ts
				ec := abs(st.studentized_range_cdf(d, q) - tc) / tc
				_anote(&t, max(es, ec), fmt.tprintf("k = 2, df %v, q %v", df, q))
			}
		}
		for row in SR_REF {
			d := st.Studentized_Range{row[0], row[1]}
			ec := abs(st.studentized_range_cdf(d, row[2]) - row[3]) / row[3]
			es := abs(st.studentized_range_sf(d, row[2]) - row[4]) / row[4]
			_anote(&t, max(ec, es), fmt.tprintf("k %v df %v q %v", row[0], row[1], row[2]))
		}
		// quantiles invert the smaller tail (the larger one is ill-conditioned)
		for row in SR_REF[:6] {
			d := st.Studentized_Range{row[0], row[1]}
			q :=
				st.studentized_range_ppf(d, row[3]) if row[3] <= 0.5 else st.studentized_range_isf(d, row[4])
			_anote(
				&t,
				abs(q - row[2]) / row[2],
				fmt.tprintf("quantile k %v df %v", row[0], row[1]),
			)
		}
		nan_ok :=
			math.is_nan(st.studentized_range_sf(st.Studentized_Range{1, 10}, 2)) &&
			math.is_nan(st.studentized_range_cdf(st.Studentized_Range{3, 0}, 2)) &&
			math.is_nan(st.studentized_range_ppf(st.Studentized_Range{3, 10}, 1.5)) &&
			st.sf(st.Studentized_Range{3, 10}, 0) == 1 &&
			st.cdf(st.Studentized_Range{3, math.inf_f64(1)}, math.inf_f64(1)) == 1
		report(
			&passed,
			&total,
			t,
			"studentized range: k = 2 against Student's t (df 0.7 to 1e6, both tails), 19 points against 30-digit integrals, quantiles",
		)
		expect(
			&passed,
			&total,
			nan_ok,
			"studentized range: invalid k, df or probability give NaN; limits at 0 and +Inf",
		)
	}

	// 5. Tukey HSD
	{
		r, ok := st.tukey_hsd(plant)
		defer st.tukey_destroy(&r)
		t: _Tally
		if !ok || len(r.pairs) != 3 {
			_anote(&t, math.inf_f64(1), "not ok")
		} else {
			for e, i in r.pairs {
				_anote(&t, abs(e.statistic - W_TUKEY[2 * i]) / W_TUKEY[2 * i], "q")
				_anote(&t, abs(e.p_value - W_TUKEY[2 * i + 1]) / W_TUKEY[2 * i + 1], "p")
			}
			d := st.Studentized_Range{3, 27}
			_anote(&t, abs(st.studentized_range_sf(d, r.q_crit) - 0.05) / 0.05, "q_crit")
		}
		report(
			&passed,
			&total,
			t,
			"Tukey HSD on PlantGrowth: studentized range statistics, p-values against 30-digit integrals, critical value",
		)
		// R's TukeyHSD(aov(weight ~ group, PlantGrowth)), printed to 7 digits
		r_ok := ok && len(r.pairs) == 3
		if r_ok {
			R := [3][4]f64 {
				{0.371, -0.3202161, 1.0622161, 0.3908711},
				{-0.494, -1.1852161, 0.1972161, 0.1979960},
				{-0.865, -1.5562161, -0.1737839, 0.0120064},
			}
			for e, i in r.pairs {
				r_ok =
					r_ok &&
					abs(e.diff - R[i][0]) < 1e-12 &&
					abs(e.ci_low - R[i][1]) < 1e-7 &&
					abs(e.ci_high - R[i][2]) < 1e-7 &&
					abs(e.p_value - R[i][3]) < 1e-7
			}
		}
		_, b1 := st.tukey_hsd({PG_CTRL}, allocator = context.temp_allocator)
		_, b2 := st.tukey_hsd(plant, confidence = 1, allocator = context.temp_allocator)
		_, b3 := st.tukey_hsd({{1}, {2}}, allocator = context.temp_allocator)
		expect(
			&passed,
			&total,
			r_ok && !b1 && !b2 && !b3,
			"Tukey HSD matches R's TukeyHSD (differences, intervals, adjusted p); one group, confidence 1 or no residual df give ok = false",
		)
	}

	// 6. conventions
	{
		_, a1 := st.anova_oneway({PG_CTRL})
		_, a2 := st.anova_oneway({{1}, {2}})
		_, a3 := st.anova_oneway({PG_CTRL, {1}}, .Welch)
		_, a4 := st.anova_oneway({PG_CTRL, {1, math.nan_f64()}})
		zero, zok := st.anova_oneway({{1, 1}, {2, 2}}, .Welch)
		y, supp, dose := _tooth(false)
		_, b1 := st.anova(y, {supp}, {"a", "b"}, allocator = context.temp_allocator)
		_, b2 := st.anova(y[:59], {supp}, {"a"}, allocator = context.temp_allocator)
		one := make([]int, 60, context.temp_allocator)
		_, b3 := st.anova(y, {supp, one}, {"a", "b"}, allocator = context.temp_allocator)
		neg := slice.clone(dose, context.temp_allocator)
		neg[0] = -1
		_, b4 := st.anova(y, {neg}, {"a"}, allocator = context.temp_allocator)
		// a saturated model leaves no residual degrees of freedom
		_, b5 := st.anova(
			y[:6],
			{{0, 0, 0, 1, 1, 1}, {0, 1, 2, 0, 1, 2}},
			{"a", "b"},
			allocator = context.temp_allocator,
		)
		// an empty cell loses the aliased interaction degree of freedom
		ye := []f64{1, 2, 3, 4, 5, 6.5, 7, 8.25, 9}
		fa := []int{0, 0, 0, 1, 1, 1, 1, 1, 1}
		fb := []int{0, 0, 1, 0, 1, 2, 0, 1, 2}
		te, eok := st.anova(ye, {fa, fb}, {"a", "b"}, allocator = context.temp_allocator)
		expect(
			&passed,
			&total,
			!a1 &&
			!a2 &&
			!a3 &&
			!a4 &&
			!b1 &&
			!b2 &&
			!b3 &&
			!b4 &&
			!b5 &&
			zok &&
			math.is_nan(zero.statistic) &&
			math.is_nan(zero.p_value) &&
			eok &&
			te.terms[2].df == 1 &&
			te.terms[3].df == 4,
			"invalid input gives ok = false (9 cases); a group without spread gives NaN in Welch; an empty cell drops the aliased interaction df",
		)
	}

	fmt.printf("ANOVA: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
