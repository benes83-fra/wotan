package tests

import w "../wotan/core"
import st "../wotan/stats"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"

// Hypothesis tests of wotan/stats.
//
// - Published data with known answers (Student's sleep data, Mendel's peas,
//   the tea-tasting and twin-conviction tables, R's Hollander & Wolfe
//   examples): every statistic, p-value, degrees of freedom and interval
//   against references to 1e-12. The references are 40-digit (mpmath) or
//   exact rational computations, or scipy / statsmodels where their result
//   is exact. scipy 1.10's t quantile (2e-11 at df = 1), single-precision
//   Shapiro-Wilk and one-sided Pearson p-values for negative r are not used.
// - Exact null distributions beyond the reach of scipy's exact methods
//   (Mann-Whitney 60 x 70, Wilcoxon n = 80, Kendall n = 50, one-sided
//   two-sample KS 1000 x 1500) against integer path / permutation counts, on
//   integer-generated samples ((i * a) mod m) / m that are the same doubles
//   in Python and Odin.
// - Invalid input gives ok = false; undefined statistics give NaN, as scipy.
// - The DataFrame helpers: nulls and NaN dropped, sorted group labels,
//   crosstab counts, and the tests on their output.

@(private = "file")
SLEEP1 := []f64{0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0}
@(private = "file")
SLEEP2 := []f64{1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4}
@(private = "file")
MWX := []f64{0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46}
@(private = "file")
MWY := []f64{1.15, 0.88, 0.90, 0.74, 1.21}
@(private = "file")
DEPX := []f64{1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30}
@(private = "file")
DEPY := []f64{0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29}
@(private = "file")
HW1 := []f64{2.9, 3.0, 2.5, 2.6, 3.2}
@(private = "file")
HW2 := []f64{3.8, 2.7, 4.0, 2.4}
@(private = "file")
HW3 := []f64{2.8, 3.4, 3.7, 2.2, 2.0}
@(private = "file")
CORX := []f64{44.4, 45.9, 41.9, 53.3, 44.7, 44.1, 50.7, 45.2, 60.1}
@(private = "file")
CORY := []f64{2.6, 3.1, 2.5, 5.0, 3.6, 4.0, 5.2, 2.8, 3.8}
@(private = "file")
PVALS := []f64{0.01, 0.04, 0.03, 0.005, 0.2, 0.04, 0.5, 0.0001}

// expected values (see the header)
@(private = "file")
W_PAIRED := []f64 {
	-4.062127683382037,
	0.002832890197384271,
	9.0,
	-1.58,
	0.3889587238883952,
	-2.4598857632769824,
	-0.7001142367230175,
}
@(private = "file")
W_WELCH := []f64 {
	-1.860813467486853,
	0.07939414018735814,
	17.77647351617849,
	-1.58,
	0.849091017238762,
	-3.3654832307117104,
	0.2054832307117106,
}
@(private = "file")
W_STUDENT_LESS := []f64 {
	-1.860813467486853,
	0.03959335710796905,
	18.0,
	-1.58,
	0.849091017238762,
	math.inf_f64(-1),
	-0.10762216830039809,
}
@(private = "file")
W_ONE_GREATER := []f64 {
	2.100552849818713,
	0.032529942834047754,
	9.0,
	2.33,
	0.6331666447310692,
	1.4543123341427606,
	math.inf_f64(1),
}
@(private = "file")
W_Z2 := []f64 {
	-1.860813467486853,
	0.06277052297856904,
	0.0,
	-1.58,
	0.849091017238762,
	-3.2441878133844515,
	0.08418781338445144,
}
@(private = "file")
W_BINOM_7_20 := []f64{0.35, 0.6294979666766777, 0.15390920478454118, 0.592188534532828}
@(private = "file")
W_BINOM_7_20_G := []f64{0.35, 0.391990187799076, 0.17731091757444914, 1.0}
@(private = "file")
W_MENDEL_BINOM := []f64 {
	0.7372972972972973,
	0.3824915595748517,
	0.707668264079039,
	0.7654065582415253,
}
@(private = "file")
W_MENDEL_CI := []f64 {
	0.7079959876196205,
	0.7646358009714889,
	0.7074390554781229,
	0.7651554240456733,
	0.707668264079039,
	0.7654065582415253,
	0.7082258165034296,
	0.7648864606853041,
	0.7079790581519885,
	0.7646527304391209,
	0.7089356915233714,
	0.7656589030712232,
}
@(private = "file")
W_PZ1 := []f64{-0.8922085160073477, 0.3722811784587139}
@(private = "file")
W_PZ2 := []f64{-0.09751163391198005, 0.9223200957342691}
@(private = "file")
W_MENDEL_CHI := []f64{0.47002398081534774, 0.925425895103616, 3.0}
@(private = "file")
W_MENDEL_G := []f64{0.47544523899826313, 0.9242519039745261, 3.0}
@(private = "file")
W_PARTY := []f64 {
	30.070149095754672,
	2.953589183211758e-07,
	2.0,
	703.6713819368879,
	319.64526659412405,
	533.683351468988,
	542.3286180631121,
	246.35473340587595,
	411.316648531012,
}
@(private = "file")
W_YATES := []f64{4.1714577497665735, 0.04111041419430708, 1.0}
@(private = "file")
W_TEA := []f64{0.4857142857142857, 0.9857142857142858, 0.24285714285714285}
@(private = "file")
W_TWINS := []f64{0.0005367241191434358, 0.00046518094336290503, 0.9999845190186861}
@(private = "file")
W_MWU_EXACT := []f64{35.0, 0.1272061272061272}
@(private = "file")
W_MWU_SLEEP := []f64{25.5, 0.06932757543362658}
@(private = "file")
W_WIL_EXACT := []f64{40.0, 0.01953125}
@(private = "file")
W_WIL_SLEEP := []f64{0.0, 0.007632441648205508}
@(private = "file")
W_WIL_SLEEP_PRATT := []f64{0.0, 0.006801553132897053}
@(private = "file")
W_KRUSKAL := []f64{0.7714285714285722, 0.6799647735788936, 2.0}
@(private = "file")
W_MWU_60_70 := []f64{1869.0, 0.28275906058760725}
@(private = "file")
W_WIL_80 := []f64{2107.0, 0.9905976090322428}
@(private = "file")
W_KENDALL_50 := []f64{0.05795918367346939, 0.5600703044243674}
@(private = "file")
W_KS2_1000_1500 := []f64{0.032333333333333325, 0.28099078870201893}
@(private = "file")
W_KS1_SLEEP := []f64{0.18511634072019578, 0.8233303885065042}
@(private = "file")
W_KS1_U50 := []f64{0.030757806066847415, 0.9999999975826077}
@(private = "file")
W_KS1_U1000 := []f64{0.0356277697260321, 0.15413995230004174}
@(private = "file")
W_KS2_SLEEP := []f64{0.4, 0.41752365281777043}
@(private = "file")
W_KS2_SLEEP_G := []f64{0.4, 0.2097902097902098}
@(private = "file")
W_SW_SLEEP := []f64{0.9258060289366404, 0.40792879642998797}
@(private = "file")
W_SW_3 := []f64{0.9642857142857143, 0.6368868450289699}
@(private = "file")
W_SW_U500 := []f64{0.9544718481602923, 2.695231241742814e-11}
@(private = "file")
W_AD_SLEEP := []f64{0.35715708336232754, 0.3784707224362548, 0.501, 0.57, 0.684, 0.798, 0.95}
@(private = "file")
W_N200 := []f64 {
	0.43013955876491816,
	0.6670941253955662,
	-1.3635895240535179,
	0.08634839551812144,
	2.044396430122978,
	0.3598031455307864,
	1.6867506618732107,
	0.43025581398769014,
}
@(private = "file")
W_VAR_HW := []f64 {
	4.476854343847557,
	0.03780330908861112,
	4.781469208069268,
	0.032039911416238256,
	4.781469208069277,
	0.03203991141623811,
	3.2794144046012064,
	0.1940368475168175,
}
@(private = "file")
W_PEARSON := []f64 {
	0.5711815575638287,
	0.10817305434949387,
	-0.14974263303675728,
	0.8955794618387393,
}
@(private = "file")
W_SPEARMAN := []f64{0.6, 0.0876228290414024}
@(private = "file")
W_KENDALL := []f64{0.4444444444444444, 0.11943893298059964}
@(private = "file")
W_PEARSON_NEG_LESS := []f64{-0.5711815575638287, 0.054086527174746955}
@(private = "file")
W_PADJ := []f64 {
	0.08,
	0.32,
	0.24,
	0.04,
	1.0,
	0.32,
	1.0,
	0.0008,
	0.0772553055720799,
	0.27861042101616634,
	0.21625664056230393,
	0.039306956424563125,
	0.83222784,
	0.27861042101616634,
	0.99609375,
	0.0007997200559930006,
	0.06,
	0.16,
	0.15,
	0.035,
	0.4,
	0.16,
	0.5,
	0.0008,
	0.058519850599,
	0.15065344,
	0.14126597429999999,
	0.034479353190515705,
	0.36,
	0.15065344,
	0.5,
	0.0007997200559930006,
	0.06,
	0.12,
	0.12,
	0.035,
	0.4,
	0.12,
	0.5,
	0.0008,
	0.06,
	0.12,
	0.09,
	0.035,
	0.4,
	0.12,
	0.5,
	0.0008,
	0.02666666666666667,
	0.05333333333333334,
	0.05333333333333334,
	0.02,
	0.2285714285714286,
	0.05333333333333334,
	0.5,
	0.0008,
	0.07247619047619047,
	0.14495238095238094,
	0.14495238095238094,
	0.054357142857142854,
	0.6212244897959183,
	0.14495238095238094,
	1.0,
	0.002174285714285714,
}

@(private = "file")
_worst :: proc(got, want: []f64) -> f64 {
	if len(got) != len(want) {return math.inf_f64(1)}
	worst := 0.0
	for g, i in got {
		e := want[i]
		err: f64
		switch {
		case math.is_nan(e) || math.is_nan(g):
			err = 0 if math.is_nan(e) && math.is_nan(g) else math.inf_f64(1)
		case math.is_inf(e) || math.is_inf(g):
			err = 0 if g == e else math.inf_f64(1)
		case:
			err = abs(g - e) / max(abs(e), 1e-300)
		}
		worst = max(worst, err)
	}
	return worst
}

@(private = "file")
_mean_err :: proc(r: st.Mean_Test_Result, ok: bool, want: []f64) -> f64 {
	if !ok {return math.inf_f64(1)}
	got := [7]f64{r.statistic, r.p_value, r.df, r.estimate, r.std_error, r.ci_low, r.ci_high}
	return _worst(got[:], want)
}

@(private = "file")
_test_err :: proc(r: st.Test_Result, ok: bool, want: []f64) -> f64 {
	if !ok {return math.inf_f64(1)}
	got := [3]f64{r.statistic, r.p_value, r.df}
	return _worst(got[:len(want)], want)
}

// ((i * mul) mod m) / m for i = 1 .. n: the same doubles in any language
@(private = "file")
_ilist :: proc(n, mul, m: int) -> []f64 {
	r := make([]f64, n)
	for i in 1 ..= n {r[i - 1] = f64((i * mul) % m) / f64(m)}
	return r
}

// worst error so far, with the name of the check that gave it
@(private = "file")
Tally :: struct {
	worst: f64,
	name:  string,
}

@(private = "file")
_note :: proc(t: ^Tally, err: f64, name: string) {
	if !(err <= t.worst) {
		t.worst = err
		t.name = name
	}
}

stats_tests_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Statistical tests ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	report :: proc(passed, total: ^int, t: Tally, what: string) {
		expect(
			passed,
			total,
			t.worst <= 1e-12,
			fmt.tprintf(
				"%s (worst %.1e%s%s)",
				what,
				t.worst,
				", " if t.worst > 1e-12 else "",
				t.name if t.worst > 1e-12 else "",
			),
		)
	}

	// 1. t- and z-tests
	{
		t: Tally
		_note(&t, _mean_err(st.t_test_paired(SLEEP1, SLEEP2), W_PAIRED), "paired")
		_note(&t, _mean_err(st.t_test_two_sample(SLEEP1, SLEEP2), W_WELCH), "Welch")
		_note(
			&t,
			_mean_err(
				st.t_test_two_sample(SLEEP1, SLEEP2, equal_var = true, alternative = .Less),
				W_STUDENT_LESS,
			),
			"Student, less",
		)
		_note(
			&t,
			_mean_err(
				st.t_test_one_sample(SLEEP2, mu = 1, alternative = .Greater, confidence = 0.9),
				W_ONE_GREATER,
			),
			"one sample",
		)
		_note(&t, _mean_err(st.z_test_two_sample(SLEEP1, SLEEP2), W_Z2), "z")
		report(
			&passed,
			&total,
			t,
			"t-tests (paired, Welch, Student, one-sample) and z-test on Student's sleep data: statistic, p, df, estimate, standard error, interval",
		)
	}

	// 2. proportions
	{
		t: Tally
		b, ok := st.binomial_test(7, 20, 0.3)
		g := [4]f64{b.estimate, b.p_value, b.ci_low, b.ci_high}
		_note(&t, _worst(g[:], W_BINOM_7_20) if ok else math.inf_f64(1), "binomial 7/20")
		b, ok = st.binomial_test(7, 20, 0.3, .Greater)
		g = {b.estimate, b.p_value, b.ci_low, b.ci_high}
		_note(
			&t,
			_worst(g[:], W_BINOM_7_20_G) if ok else math.inf_f64(1),
			"binomial 7/20, greater",
		)
		b, ok = st.binomial_test(682, 925, 0.75)
		g = {b.estimate, b.p_value, b.ci_low, b.ci_high}
		_note(&t, _worst(g[:], W_MENDEL_BINOM) if ok else math.inf_f64(1), "binomial, Mendel")
		cis: [12]f64
		for m, i in ([6]st.CI_Method{.Wilson, .Wilson_CC, .Clopper_Pearson, .Jeffreys, .Agresti_Coull, .Normal}) {
			cis[2 * i], cis[2 * i + 1], _ = st.proportion_ci(682, 925, 0.95, m)
		}
		_note(&t, _worst(cis[:], W_MENDEL_CI), "proportion intervals")
		_note(
			&t,
			_test_err(
				st.proportion_z_test_one_sample(682, 925, 0.75, null_variance = true),
				W_PZ1,
			),
			"z, one proportion",
		)
		_note(
			&t,
			_test_err(st.proportion_z_test_two_sample(83, 86, 90, 93), W_PZ2),
			"z, two proportions",
		)
		report(
			&passed,
			&total,
			t,
			"binomial test with Clopper-Pearson interval, six proportion intervals (Mendel's peas) and z-tests for proportions",
		)
	}

	// 3. counts
	{
		t: Tally
		obs := []f64{315, 108, 101, 32}
		exp := []f64{556.0 * 9 / 16, 556.0 * 3 / 16, 556.0 * 3 / 16, 556.0 / 16}
		_note(&t, _test_err(st.chi_square_test(obs, exp), W_MENDEL_CHI), "Mendel, Pearson")
		_note(
			&t,
			_test_err(st.chi_square_test(obs, exp, divergence = .Log_Likelihood), W_MENDEL_G),
			"Mendel, G",
		)
		party := []f64{762, 327, 468, 484, 239, 477}
		e: [6]f64
		r, ok := st.chi_square_contingency(party, 3, expected = e[:])
		got := [9]f64{r.statistic, r.p_value, r.df, e[0], e[1], e[2], e[3], e[4], e[5]}
		_note(&t, _worst(got[:], W_PARTY) if ok else math.inf_f64(1), "party table")
		_note(
			&t,
			_test_err(st.chi_square_contingency([]f64{12, 5, 3, 9}, 2), W_YATES),
			"2 x 2, Yates",
		)
		fisher_err :: proc(table: [2][2]int, want: []f64) -> f64 {
			got: [3]f64
			for alt, i in ([3]st.Alternative{.Two_Sided, .Less, .Greater}) {
				f, ok := st.fisher_exact(table, alt)
				got[i] = f.p_value if ok else math.nan_f64()
			}
			return _worst(got[:], want)
		}
		_note(&t, fisher_err({{3, 1}, {1, 3}}, W_TEA), "Fisher, tea")
		_note(&t, fisher_err({{2, 15}, {10, 3}}, W_TWINS), "Fisher, twins")
		f, _ := st.fisher_exact({{3, 1}, {1, 3}})
		_note(&t, abs(f.odds_ratio - 9) / 9, "odds ratio")
		report(
			&passed,
			&total,
			t,
			"chi-squared goodness of fit (Pearson, G), independence with expected counts and Yates' correction, Fisher's exact test (all alternatives)",
		)
	}

	// 4. rank tests
	{
		t: Tally
		_note(
			&t,
			_test_err(st.mann_whitney_u(MWX, MWY, .Greater), W_MWU_EXACT),
			"Mann-Whitney exact",
		)
		_note(&t, _test_err(st.mann_whitney_u(SLEEP1, SLEEP2), W_MWU_SLEEP), "Mann-Whitney, ties")
		_note(
			&t,
			_test_err(st.wilcoxon_signed_rank(DEPX, DEPY, .Greater), W_WIL_EXACT),
			"Wilcoxon exact",
		)
		_note(
			&t,
			_test_err(st.wilcoxon_signed_rank(SLEEP1, SLEEP2), W_WIL_SLEEP),
			"Wilcoxon, zero and ties",
		)
		_note(
			&t,
			_test_err(
				st.wilcoxon_signed_rank(SLEEP1, SLEEP2, zero_method = .Pratt, correction = true),
				W_WIL_SLEEP_PRATT,
			),
			"Wilcoxon, Pratt",
		)
		_note(&t, _test_err(st.kruskal_wallis({HW1, HW2, HW3}), W_KRUSKAL), "Kruskal-Wallis")
		report(
			&passed,
			&total,
			t,
			"Mann-Whitney U, Wilcoxon signed-rank (exact, ties and zeros, Pratt) and Kruskal-Wallis",
		)
	}

	// 5. exact null distributions against integer counts
	{
		t: Tally
		x := _ilist(60, 7919, 10007)
		defer delete(x)
		y := _ilist(70, 104729, 10009)
		defer delete(y)
		for &v in y {v += 0.08}
		_note(
			&t,
			_test_err(st.mann_whitney_u(x, y, method = .Exact), W_MWU_60_70),
			"Mann-Whitney 60 x 70",
		)
		d := _ilist(80, 7919, 10007)
		defer delete(d)
		for &v in d {v -= 0.43}
		_note(
			&t,
			_test_err(st.wilcoxon_signed_rank(d, alternative = .Less, method = .Exact), W_WIL_80),
			"Wilcoxon n = 80",
		)
		kx := _ilist(50, 7919, 10007)
		defer delete(kx)
		ky := _ilist(50, 104729, 10009)
		defer delete(ky)
		for &v, i in ky {v += 0.3 * kx[i]}
		k, kok := st.kendall(kx, ky, method = .Exact)
		kg := [2]f64{k.coefficient, k.p_value}
		_note(&t, _worst(kg[:], W_KENDALL_50) if kok else math.inf_f64(1), "Kendall n = 50")
		a := _ilist(1000, 7919, 10007)
		defer delete(a)
		b := _ilist(1500, 104729, 10009)
		defer delete(b)
		for &v in b {v = v * 1.02 + 0.01}
		_note(
			&t,
			_test_err(st.ks_test_two_sample(a, b, .Greater), W_KS2_1000_1500),
			"KS 1000 x 1500, one-sided",
		)
		report(
			&passed,
			&total,
			t,
			"exact null distributions match integer counts: Mann-Whitney 60 x 70, Wilcoxon n = 80, Kendall n = 50, one-sided KS 1000 x 1500",
		)
	}

	// 6. Kolmogorov-Smirnov
	{
		t: Tally
		_note(
			&t,
			_test_err(st.ks_test_one_sample(SLEEP2, st.Normal{2.33, 2}), W_KS1_SLEEP),
			"one sample, normal",
		)
		for n in ([2]int{50, 1000}) {
			u := _ilist(n, 7919, 10007)
			defer delete(u)
			for &v in u {v = math.pow(v, 1.1)}
			want := W_KS1_U50 if n == 50 else W_KS1_U1000
			_note(
				&t,
				_test_err(
					st.ks_test_one_sample_fn(u, proc(x: f64) -> f64 {return clamp(x, 0, 1)}),
					want,
				),
				fmt.tprintf("one sample, uniform, n = %d", n),
			)
		}
		_note(&t, _test_err(st.ks_test_two_sample(SLEEP1, SLEEP2), W_KS2_SLEEP), "two samples")
		_note(
			&t,
			_test_err(st.ks_test_two_sample(SLEEP1, SLEEP2, .Greater), W_KS2_SLEEP_G),
			"two samples, greater",
		)
		report(
			&passed,
			&total,
			t,
			"Kolmogorov-Smirnov: exact one-sample (n = 10, 50, 1000; a distribution or a cdf of your own) and two-sample (two- and one-sided)",
		)
	}

	// 7. normality
	{
		t: Tally
		_note(&t, _test_err(st.shapiro_wilk(SLEEP1), W_SW_SLEEP), "Shapiro-Wilk n = 10")
		_note(&t, _test_err(st.shapiro_wilk([]f64{1, 2, 4}), W_SW_3), "Shapiro-Wilk n = 3")
		u := _ilist(500, 7919, 10007)
		defer delete(u)
		_note(&t, _test_err(st.shapiro_wilk(u), W_SW_U500), "Shapiro-Wilk n = 500")
		ad, aok := st.anderson_darling(SLEEP2)
		ag := [7]f64 {
			ad.statistic,
			ad.p_value,
			ad.critical_values[0],
			ad.critical_values[1],
			ad.critical_values[2],
			ad.critical_values[3],
			ad.critical_values[4],
		}
		_note(&t, _worst(ag[:], W_AD_SLEEP) if aok else math.inf_f64(1), "Anderson-Darling")
		a := _ilist(200, 7919, 10007)
		defer delete(a)
		b := _ilist(200, 104729, 10009)
		defer delete(b)
		c := _ilist(200, 1299709, 10037)
		defer delete(c)
		s := make([]f64, 200)
		defer delete(s)
		for i in 0 ..< 200 {s[i] = a[i] + b[i] + c[i]}
		sk, _ := st.skew_test(s)
		ku, _ := st.kurtosis_test(s, .Less)
		dp, _ := st.dagostino_pearson(s)
		jb, _ := st.jarque_bera(s)
		ng := [8]f64 {
			sk.statistic,
			sk.p_value,
			ku.statistic,
			ku.p_value,
			dp.statistic,
			dp.p_value,
			jb.statistic,
			jb.p_value,
		}
		_note(&t, _worst(ng[:], W_N200), "moment tests")
		report(
			&passed,
			&total,
			t,
			"normality: Shapiro-Wilk (n = 3, 10, 500), Anderson-Darling with critical values, skewness, kurtosis, D'Agostino-Pearson, Jarque-Bera",
		)
	}

	// 8. equal variances
	{
		got: [8]f64
		for c, i in ([3]st.Levene_Center{.Median, .Mean, .Trimmed}) {
			r, _ := st.levene({HW1, HW2, HW3}, c)
			got[2 * i], got[2 * i + 1] = r.statistic, r.p_value
		}
		r, _ := st.bartlett({HW1, HW2, HW3})
		got[6], got[7] = r.statistic, r.p_value
		t: Tally
		_note(&t, _worst(got[:], W_VAR_HW), "Levene / Bartlett")
		report(&passed, &total, t, "equal variances: Levene (median, mean, trimmed) and Bartlett")
	}

	// 9. correlation
	{
		t: Tally
		p, _ := st.pearson(CORX, CORY)
		pg := [4]f64{p.coefficient, p.p_value, p.ci_low, p.ci_high}
		_note(&t, _worst(pg[:], W_PEARSON), "Pearson")
		s, _ := st.spearman(CORX, CORY)
		sg := [2]f64{s.coefficient, s.p_value}
		_note(&t, _worst(sg[:], W_SPEARMAN), "Spearman")
		k, _ := st.kendall(CORX, CORY)
		kg := [2]f64{k.coefficient, k.p_value}
		_note(&t, _worst(kg[:], W_KENDALL), "Kendall")
		neg := make([]f64, len(CORY))
		defer delete(neg)
		for v, i in CORY {neg[i] = -v}
		pn, _ := st.pearson(CORX, neg, .Less)
		ng := [2]f64{pn.coefficient, pn.p_value}
		_note(&t, _worst(ng[:], W_PEARSON_NEG_LESS), "Pearson, negative r, less")
		report(
			&passed,
			&total,
			t,
			"correlation: Pearson with interval (also one-sided for negative r), Spearman, Kendall exact",
		)
	}

	// 10. multiple testing
	{
		got := make([dynamic]f64)
		defer delete(got)
		for m in st.P_Adjust {
			adj, _ := st.p_adjust(PVALS, m)
			append(&got, ..adj)
			delete(adj)
		}
		t: Tally
		_note(&t, _worst(got[:], W_PADJ), "p_adjust")
		report(
			&passed,
			&total,
			t,
			"p_adjust: Bonferroni, Sidak, Holm, Holm-Sidak, Hochberg, Hommel, BH, BY (a tie included)",
		)
	}

	// 11. conventions
	{
		nan := math.nan_f64()
		_, ok1 := st.t_test_one_sample([]f64{1})
		_, ok2 := st.t_test_paired(SLEEP1, SLEEP2[:9])
		_, ok3 := st.t_test_two_sample(SLEEP1, []f64{1, nan})
		_, ok4 := st.t_test_one_sample(SLEEP1, confidence = 1)
		_, ok5 := st.binomial_test(21, 20)
		_, ok6 := st.chi_square_test([]f64{1, 2}, ddof = 1)
		_, ok7 := st.chi_square_contingency([]f64{1, 0, 2, 0}, 2)
		_, ok8 := st.kruskal_wallis({[]f64{1, 1}, []f64{1}})
		_, ok9 := st.shapiro_wilk([]f64{2, 2, 2})
		_, ok10 := st.p_adjust([]f64{0.5, 1.5})
		_, ok11 := st.kendall([]f64{1, 1, 2}, []f64{1, 2, 3}, method = .Exact)
		_, ok12 := st.mann_whitney_u([]f64{}, []f64{1})
		invalid_ok := !(ok1 ||
			ok2 ||
			ok3 ||
			ok4 ||
			ok5 ||
			ok6 ||
			ok7 ||
			ok8 ||
			ok9 ||
			ok10 ||
			ok11 ||
			ok12)
		pc, pok := st.pearson([]f64{1, 1, 1}, []f64{1, 2, 3})
		fz, fok := st.fisher_exact({{0, 0}, {3, 4}})
		cz, cok := st.chi_square_contingency([]f64{5, 7}, 1) // one column: df 0
		undefined_ok :=
			pok &&
			math.is_nan(pc.coefficient) &&
			math.is_nan(pc.p_value) &&
			fok &&
			math.is_nan(fz.odds_ratio) &&
			fz.p_value == 1 &&
			cok &&
			cz.statistic == 0 &&
			cz.p_value == 1
		expect(
			&passed,
			&total,
			invalid_ok && undefined_ok,
			"invalid input gives ok = false (12 cases); constant data give NaN, a zero margin p = 1, df 0 p = 1",
		)
	}

	// 12. DataFrame helpers
	{
		df := w.dataframe_new()
		defer w.destroy_dataframe(&df)
		vals := w.column_new("value", .Float, 8)
		for v, i in ([8]f64{2.9, 3.8, 2.8, 3.0, 0, 2.7, math.nan_f64(), 3.4}) {
			if i == 4 {w.append_null(&vals)} else {w.append_float(&vals, v)}
		}
		w.add_column(&df, vals)
		w.add_column(
			&df,
			w.column_from_strings("group", []string{"b", "a", "c", "b", "a", "a", "c", "c"}),
		)
		w.add_column(
			&df,
			w.column_from_bools("flag", []bool{true, false, true, true, false, true, false, true}),
		)
		w.add_column(&df, w.column_from_ints("count", []int{3, 1, 4, 1, 5, 9, 2, 6}))
		v, vok := st.column_values(&df, "value")
		defer delete(v)
		col_ok := vok && slice.equal(v, []f64{2.9, 3.8, 2.8, 3.0, 2.7, 3.4})
		g, gok := st.grouped_values(&df, "value", "group")
		defer st.groups_destroy(&g)
		grp_ok :=
			gok &&
			len(g.labels) == 3 &&
			g.labels[0] == "a" &&
			g.labels[1] == "b" &&
			g.labels[2] == "c"
		grp_ok =
			grp_ok &&
			slice.equal(g.samples[0], []f64{3.8, 2.7}) &&
			slice.equal(g.samples[1], []f64{2.9, 3.0}) &&
			slice.equal(g.samples[2], []f64{2.8, 3.4})
		kg, _ := st.kruskal_wallis(g.samples)
		kd, _ := st.kruskal_wallis({{3.8, 2.7}, {2.9, 3.0}, {2.8, 3.4}})
		grp_ok = grp_ok && kg == kd
		tb, tok := st.crosstab(&df, "group", "flag")
		defer st.table_destroy(&tb)
		tab_ok :=
			tok &&
			tb.rows == 3 &&
			tb.cols == 2 &&
			tb.col_labels[0] == "false" &&
			tb.col_labels[1] == "true"
		tab_ok = tab_ok && slice.equal(tb.counts, []f64{2, 1, 0, 2, 1, 2})
		x, y, pok := st.paired_values(&df, "value", "count")
		defer delete(x)
		defer delete(y)
		pair_ok := pok && slice.equal(y, []f64{3, 1, 4, 1, 9, 6})
		ig, iok := st.grouped_values(&df, "count", "count")
		defer st.groups_destroy(&ig)
		int_ok :=
			iok &&
			len(ig.labels) == 7 &&
			ig.labels[0] == "1" &&
			ig.labels[6] == "9" &&
			len(ig.samples[0]) == 2
		_, m1 := st.column_values(&df, "missing")
		_, m2 := st.column_values(&df, "group")
		_, m3 := st.grouped_values(&df, "value", "nope")
		expect(
			&passed,
			&total,
			col_ok && grp_ok && tab_ok && pair_ok && int_ok && !m1 && !m2 && !m3,
			"DataFrame columns: nulls and NaN dropped, paired rows, sorted String / Int / Bool labels, crosstab counts; missing or non-numeric columns give ok = false",
		)
	}

	fmt.printf("Statistical tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
