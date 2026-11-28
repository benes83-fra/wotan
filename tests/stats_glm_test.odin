package tests

import w "../wotan/core"
import st "../wotan/stats"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"

// Generalized linear models of wotan/stats.
//
// - Published data of R's documentation and Venables & Ripley: Dobson's
//   Poisson counts and McCullagh & Nelder's clotting times (?glm), the
//   budworm dose-response (binomial with 20 trials), warpbreaks (Poisson and
//   negative binomial; glm.nb's theta = 9.944). They reproduce R's printed
//   coefficients.
// - Expected values from IRLS run to convergence in 40-digit mpmath, an
//   implementation independent of the library: coefficients, standard errors,
//   deviances, Pearson chi2, scale, llf, AIC, BIC, alpha (relative 1e-12).
// - The DataFrame entry with String factors, the likelihood-ratio test,
//   prediction, the summary, and the conventions.
@(private = "file")
DOB_COUNTS := []f64{18.0, 17.0, 15.0, 20.0, 10.0, 20.0, 25.0, 13.0, 12.0}
@(private = "file")
CL_LOT1 := []f64{118.0, 58.0, 42.0, 35.0, 27.0, 25.0, 21.0, 19.0, 18.0}
@(private = "file")
CL_LOT2 := []f64{69.0, 35.0, 26.0, 21.0, 18.0, 16.0, 13.0, 12.0, 12.0}
@(private = "file")
BW_DEAD := []f64{1.0, 4.0, 9.0, 13.0, 18.0, 20.0, 0.0, 2.0, 6.0, 10.0, 12.0, 16.0}
@(private = "file")
WB_Y := []f64 {
	26.0,
	30.0,
	54.0,
	25.0,
	70.0,
	52.0,
	51.0,
	26.0,
	67.0,
	18.0,
	21.0,
	29.0,
	17.0,
	12.0,
	18.0,
	35.0,
	30.0,
	36.0,
	36.0,
	21.0,
	24.0,
	18.0,
	10.0,
	43.0,
	28.0,
	15.0,
	26.0,
	27.0,
	14.0,
	29.0,
	19.0,
	29.0,
	31.0,
	41.0,
	20.0,
	44.0,
	42.0,
	26.0,
	19.0,
	16.0,
	39.0,
	28.0,
	21.0,
	39.0,
	29.0,
	20.0,
	21.0,
	24.0,
	17.0,
	13.0,
	15.0,
	15.0,
	16.0,
	28.0,
}
// ln u of the clotting data, as doubles
@(private = "file")
LOG_U := []f64 {
	1.6094379124341003,
	2.302585092994046,
	2.70805020110221,
	2.995732273553991,
	3.4011973816621555,
	3.6888794541139363,
	4.0943445622221,
	4.382026634673881,
	4.605170185988092,
}
@(private = "file")
W_DOBSON := []f64 {
	3.044522437723423,
	-0.4542552722775964,
	-0.2929871246814741,
	-2.9704477620408985e-41,
	-1.4848632473103216e-42,
	0.17089865185644154,
	0.20217075919384553,
	0.19274234515979285,
	0.2,
	0.2,
	5.129141077001148,
	10.581445863750861,
	5.173201621073962,
	1.0,
	-23.380659200978837,
	56.76131840195767,
	57.74744128863877,
	0.0,
}
@(private = "file")
W_CLOT := []f64 {
	-0.016554381726200294,
	0.015343114910324671,
	0.0009275491386581969,
	0.0004149596426663356,
	0.01672971517848384,
	3.512826263828517,
	0.01712225369465307,
	0.002446036242093296,
	-16.15044387593012,
	38.30088775186024,
	38.892561483868896,
	0.0,
}
@(private = "file")
W_CLOT_GAUSS := []f64 {
	1000078.0759701392,
	-16.137136803594352,
	11.201992604687332,
	3.2556730915632768,
	590.723114151373,
	2664.0,
	590.723114151373,
	84.389016307339,
	-31.598999517253315,
	69.19799903450664,
	69.7896727665153,
	0.0,
}
@(private = "file")
W_CLOT_IG := []f64 {
	4.723411084742536,
	-0.5119391264533646,
	0.20188504104503177,
	0.053029279164318405,
	0.005656249329743141,
	0.12592520255563733,
	0.006531598051159668,
	0.0009330854358799525,
	-20.83951836179533,
	47.67903672359066,
	48.27071045559932,
	0.0,
}
@(private = "file")
W_BUDWORM := []f64 {
	-2.993541755171029,
	0.1749867878555062,
	0.9060364354674575,
	0.35291298873649757,
	0.5526997610546099,
	0.7783100816083164,
	0.16710166040079874,
	0.26999025879703886,
	4.993727307620334,
	124.8755926044078,
	3.5046938293379624,
	1.0,
	-17.552064154518405,
	43.10412830903681,
	45.043754908188816,
	0.0,
}
@(private = "file")
W_BUDWORM_CLOGLOG := []f64 {
	-2.9349737107859433,
	0.8277210910850131,
	0.729328107229317,
	0.3467927073075783,
	0.23038731331749238,
	0.08414501893210126,
	6.827619439241268,
	124.8755926044078,
	5.657450337176674,
	1.0,
	-18.469010220328872,
	42.938020440657745,
	44.39274039002175,
	0.0,
}
@(private = "file")
W_WARP_NB := []f64 {
	3.673354566579144,
	-0.1862110523510148,
	-0.29922723862867945,
	-0.5113955151913886,
	0.09790304729044666,
	0.10096139942882393,
	0.12172849680223351,
	0.12373986446503527,
	53.72257070233539,
	75.46397910523442,
	53.66422726914542,
	1.0,
	-199.38190388900347,
	408.76380777800694,
	418.7087280108283,
	0.1005592559184074,
}
@(private = "file")
W_WARP_POIS := []f64 {
	3.6919631449407935,
	-0.20598844263862093,
	-0.3213204316006107,
	-0.5184884965115598,
	0.045410794342801035,
	0.05157124278372991,
	0.06026591669548091,
	0.06395951939589972,
	210.39188876245385,
	297.37221180460534,
	213.07609419822185,
	1.0,
	-242.5279832089788,
	493.0559664179576,
	501.01190260421475,
	0.0,
}
// likelihood ratio of tension in the Poisson model
@(private = "file")
W_WARP_LR := []f64{70.94157050804336}

// worst relative error of a fit against an expected layout: params, std
// errors, deviance, null deviance, Pearson chi2, scale, llf, aic, bic, alpha.
// Parameters are measured against max(|value|, standard error), so a
// coefficient that is 0 up to rounding (Dobson's treatments) counts as one.
@(private = "file")
_glm_err :: proc(r: st.Glm_Result, ok: bool, want: []f64) -> f64 {
	if !ok {return math.inf_f64(1)}
	p := len(r.params)
	if len(want) != 2 * p + 8 {return math.inf_f64(1)}
	worst := 0.0
	for j in 0 ..< p {
		worst = max(worst, abs(r.params[j] - want[j]) / max(abs(want[j]), want[p + j]))
		worst = max(worst, abs(r.std_errors[j] - want[p + j]) / want[p + j])
	}
	got := [8]f64 {
		r.deviance,
		r.null_deviance,
		r.pearson_chi2,
		r.scale,
		r.log_likelihood,
		r.aic,
		r.bic,
		r.alpha,
	}
	for g, i in got {
		e := want[2 * p + i]
		worst = max(worst, abs(g - e) / max(abs(e), 1e-300))
	}
	return worst
}

@(private = "file")
_GTally :: struct {
	worst: f64,
	name:  string,
}

@(private = "file")
_gnote :: proc(t: ^_GTally, err: f64, name: string) {
	if !(err <= t.worst) {
		t.worst = err
		t.name = name
	}
}

stats_glm_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Generalized linear models ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	report :: proc(passed, total: ^int, t: _GTally, what: string) {
		bad := !(t.worst <= 1e-12)
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
	ta := context.temp_allocator
	male := make([]f64, 12, ta)
	ld := make([]f64, 12, ta)
	mld := make([]f64, 12, ta)
	trials := make([]f64, 12, ta)
	for i in 0 ..< 12 {
		male[i] = 1 if i < 6 else 0
		ld[i] = f64(i % 6)
		mld[i] = male[i] * ld[i]
		trials[i] = 20
	}
	wool_b := make([]f64, 54, ta)
	ten_m := make([]f64, 54, ta)
	ten_h := make([]f64, 54, ta)
	for i in 0 ..< 54 {
		cell := i / 9
		wool_b[i] = 1 if cell >= 3 else 0
		ten_m[i] = 1 if cell % 3 == 1 else 0
		ten_h[i] = 1 if cell % 3 == 2 else 0
	}

	// 1. the families on published data
	{
		t: _GTally
		r, ok := st.glm(CL_LOT1, {LOG_U}, {"log_u"}, .Gamma, allocator = ta)
		_gnote(&t, _glm_err(r, ok, W_CLOT), "Gamma")
		lot2 := make([]f64, 9, ta)
		for v, i in CL_LOT2 {lot2[i] = v + 1e6}
		r, ok = st.glm(lot2, {LOG_U}, {"log_u"}, .Gaussian, allocator = ta)
		_gnote(&t, _glm_err(r, ok, W_CLOT_GAUSS), "Gaussian, values near 1e6")
		r, ok = st.glm(
			CL_LOT2,
			{LOG_U},
			{"log_u"},
			.Inverse_Gaussian,
			{link = .Log},
			allocator = ta,
		)
		_gnote(&t, _glm_err(r, ok, W_CLOT_IG), "inverse Gaussian, log link")
		r, ok = st.glm(
			BW_DEAD,
			{male, ld, mld},
			{"male", "ldose", "male:ldose"},
			.Binomial,
			{trials = trials},
			allocator = ta,
		)
		_gnote(&t, _glm_err(r, ok, W_BUDWORM), "binomial")
		r, ok = st.glm(
			BW_DEAD,
			{male, ld},
			{"male", "ldose"},
			.Binomial,
			{trials = trials, link = .Cloglog},
			allocator = ta,
		)
		_gnote(&t, _glm_err(r, ok, W_BUDWORM_CLOGLOG), "binomial, cloglog")
		r, ok = st.glm(
			WB_Y,
			{wool_b, ten_m, ten_h},
			{"woolB", "tensionM", "tensionH"},
			.Poisson,
			allocator = ta,
		)
		_gnote(&t, _glm_err(r, ok, W_WARP_POIS), "Poisson")
		r, ok = st.glm(
			WB_Y,
			{wool_b, ten_m, ten_h},
			{"woolB", "tensionM", "tensionH"},
			.Negative_Binomial,
			allocator = ta,
		)
		_gnote(&t, _glm_err(r, ok, W_WARP_NB), "negative binomial, alpha estimated")
		report(
			&passed,
			&total,
			t,
			"GLMs on published data: Gamma (clotting), Gaussian near 1e6, inverse Gaussian, binomial with trials (budworm, logit and cloglog), Poisson and negative binomial with estimated alpha (warpbreaks): coefficients, standard errors, deviances, Pearson chi2, scale, llf, AIC, BIC, alpha",
		)
	}

	// 2. DataFrame input with String factors (Dobson's example of R's ?glm)
	{
		df := w.dataframe_new()
		defer w.destroy_dataframe(&df)
		counts := w.column_new("counts", .Float, 10)
		for v in DOB_COUNTS {w.append_float(&counts, v)}
		w.append_null(&counts)
		w.add_column(&df, counts)
		w.add_column(
			&df,
			w.column_from_strings(
				"outcome",
				[]string{"1", "2", "3", "1", "2", "3", "1", "2", "3", "1"},
			),
		)
		w.add_column(
			&df,
			w.column_from_strings(
				"treatment",
				[]string{"1", "1", "1", "2", "2", "2", "3", "3", "3", "1"},
			),
		)
		r, ok := st.glm_from_df(&df, "counts", {"outcome", "treatment"}, .Poisson)
		defer st.glm_destroy(&r)
		names_ok :=
			ok &&
			len(r.names) == 5 &&
			r.names[0] == "Intercept" &&
			r.names[1] == "outcome[T.2]" &&
			r.names[4] == "treatment[T.3]" &&
			r.n == 9
		_, bad := st.glm_from_df(&df, "outcome", {"treatment"}, .Poisson, allocator = ta)
		expect(
			&passed,
			&total,
			names_ok && _glm_err(r, ok, W_DOBSON) <= 1e-12 && !bad,
			"glm_from_df: Dobson's Poisson example with String factors in treatment coding, a null row left out; a String response gives ok = false",
		)
	}

	// 3. likelihood ratio, prediction, summary
	{
		full, _ := st.glm(
			WB_Y,
			{wool_b, ten_m, ten_h},
			{"woolB", "tensionM", "tensionH"},
			.Poisson,
		)
		defer st.glm_destroy(&full)
		restr, _ := st.glm(WB_Y, {wool_b}, {"woolB"}, .Poisson)
		defer st.glm_destroy(&restr)
		lr, lok := st.glm_lr_test(&restr, &full)
		_, back := st.glm_lr_test(&full, &restr)
		lr_ok :=
			lok &&
			!back &&
			lr.df == 2 &&
			abs(lr.statistic - W_WARP_LR[0]) <= 1e-12 * W_WARP_LR[0] &&
			lr.p_value == st.chi_squared_sf(st.Chi_Squared{2}, lr.statistic)
		mu, pok := st.glm_predict(&full, {wool_b[:3], ten_m[:3], ten_h[:3]})
		defer delete(mu)
		pred_ok := pok && len(mu) == 3 && abs(mu[0] - full.fitted[0]) <= 1e-15 * mu[0]
		s := st.glm_summary(&full, ta)
		sum_ok :=
			strings.contains(s, "Poisson family, log link") &&
			strings.contains(s, "tensionH") &&
			strings.contains(s, "P>|z|")
		expect(
			&passed,
			&total,
			lr_ok && pred_ok && sum_ok,
			"glm_lr_test (tension in warpbreaks: 2 df, statistic to 1e-12; reversed models give ok = false), glm_predict reproduces the fitted means, glm_summary",
		)
	}

	// 4. conventions
	{
		bin := []f64{0, 0, 0, 1, 1, 1}
		xs := []f64{1, 2, 3, 4, 5, 6}
		sep, sok := st.glm(bin, {xs}, {"x"}, .Binomial, allocator = ta) // complete separation
		_, b1 := st.glm(xs, {xs[:5]}, {"x"}, .Poisson, allocator = ta)
		_, b2 := st.glm([]f64{1, -1, 2, 3}, {}, {}, .Poisson, allocator = ta)
		_, b3 := st.glm([]f64{1, 0, 2, 3}, {}, {}, .Gamma, allocator = ta)
		_, b4 := st.glm([]f64{0, 1, 2}, {}, {}, .Binomial, allocator = ta)
		_, b5 := st.glm(xs, {xs, xs}, {"a", "b"}, .Poisson, allocator = ta) // collinear
		_, b6 := st.glm(xs, {xs}, {"a", "b"}, .Poisson, allocator = ta)
		_, b7 := st.glm(xs[:2], {xs[:2]}, {"a"}, .Poisson, allocator = ta) // no residual df
		_, b9 := st.glm(xs, {xs}, {"x"}, .Poisson, {trials = xs}, allocator = ta)
		ut, uok := st.glm(CL_LOT1, {LOG_U}, {"log_u"}, .Gamma, {use_t = true}, allocator = ta)
		t_ok :=
			uok &&
			abs(ut.p_values[1] - 2 * st.student_t_sf(st.Student_T{7}, abs(ut.statistics[1]))) <=
				1e-15
		expect(
			&passed,
			&total,
			sok && !sep.converged && !b1 && !b2 && !b3 && !b4 && !b5 && !b6 && !b7 && !b9 && t_ok,
			"complete separation returns converged = false; invalid input gives ok = false (8 cases); use_t gives t p-values on df_resid",
		)
	}

	fmt.printf("Generalized linear models: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
