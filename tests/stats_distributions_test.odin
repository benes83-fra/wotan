package tests

import a "../wotan/analytics"
import ml "../wotan/analytics/ML"
import fin "../wotan/finance"
import st "../wotan/stats"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

// Distributions and special functions of wotan/stats.
//
// - 122 values against 60-digit references (mpmath, with each tail of the
//   incomplete gamma and beta functions computed directly): pdf / pmf, log
//   densities, cdf, sf, ppf and isf of every distribution, in the centre and
//   far in the tails (down to 1e-300), with extreme parameters (t with
//   df = 1e6, Beta(1e5, 1e5), Binomial(1e7, 0.3), Poisson(1e6)); the special
//   functions directly. Relative error 1e-12 (log densities: absolute, per
//   unit of the value above 1). The discrete quantiles must be exact.
// - The conventions shared with scipy: NaN for invalid parameters and
//   probabilities, quantiles at 0 and 1, discrete ppf(0) = -1, pmf of a
//   non-integer 0, infinite moments.
// - Random draws (fixed seed): a Kolmogorov-Smirnov test against the cdf for
//   the continuous distributions, a chi-squared test for the discrete ones,
//   on both sides of the samplers' method switches; the same seed gives the
//   same draws.
// - The helpers elsewhere that now use stats (analytics chi2_sf, ML's t and F
//   tails, finance norm_inv) agree with it.

@(private = "file")
Dist_Ref :: struct {
	dist, fn: string,
	a, b, x:  f64,
	want:     f64,
}

@(private = "file")
DIST_REFS := []Dist_Ref {
	{"normal", "cdf", 0.0, 1.0, -38.0, 2.8854283600687843e-316},
	{"normal", "cdf", 0.0, 1.0, -5.0, 2.8665157187919391e-7},
	{"normal", "cdf", 0.0, 1.0, 0.3, 0.61791142218895263},
	{"normal", "sf", 0.0, 1.0, 8.0, 6.2209605742717841e-16},
	{"normal", "sf", 0.0, 1.0, 30.0, 4.9067139271481871e-198},
	{"normal", "pdf", 0.0, 1.0, 37.0, 2.1200065515246056e-298},
	{"normal", "logpdf", 0.0, 1.0, 3.0, -5.4189385332046727},
	{"normal", "ppf", 0.0, 1.0, 1e-300, -37.047096299361199},
	{"normal", "ppf", 0.0, 1.0, 0.025, -1.9599639845400542},
	{"normal", "ppf", 0.0, 1.0, 0.7, 0.52440051270804066},
	{"normal", "isf", 0.0, 1.0, 1e-20, 9.2623400897984076},
	{"normal", "pdf", 3.0, 0.5, 2.2, 0.22184166935891124},
	{"normal", "cdf", 3.0, 0.5, 1.0, 3.1671241833119921e-5},
	{"normal", "isf", 3.0, 0.5, 0.001, 4.5451161530839068},
	{"lognorm", "pdf", 0.5, 0.8, 2.0, 0.24217677488483339},
	{"lognorm", "cdf", 0.5, 0.8, 0.1, 0.00022982501310415205},
	{"lognorm", "sf", 0.5, 0.8, 50.0, 9.9938228292925866e-6},
	{"lognorm", "ppf", 0.5, 0.8, 0.99, 10.602377826676355},
	{"lognorm", "isf", 0.5, 0.8, 1e-12, 458.32890530854765},
	{"t", "cdf", 3.0, 0.0, -40.0, 1.7190340394579264e-5},
	{"t", "sf", 3.0, 0.0, 2.1, 0.063282601274842298},
	{"t", "pdf", 3.0, 0.0, 0.5, 0.31318091100882862},
	{"t", "ppf", 3.0, 0.0, 1e-10, -2225.7692846830932},
	{"t", "isf", 3.0, 0.0, 0.025, 3.1824463052837095},
	{"t", "cdf", 0.5, 0.0, -1000000.0, 0.00032070097541419885},
	{"t", "isf", 0.5, 0.0, 0.01, 1028.4910104716219},
	{"t", "cdf", 1000000.0, 0.0, -2.0, 0.022750266925659604},
	{"t", "ppf", 1000000.0, 0.0, 1e-100, -21.275865985490619},
	{"t", "pdf", 1000000.0, 0.0, 0.1, 0.39695244626404889},
	{"t", "isf", 30.0, 0.0, 1e-300, 50178575360.505081},
	{"t", "sf", 30.0, 0.0, 12.0, 2.790092707599628e-13},
	{"chi2", "cdf", 5.0, 0.0, 0.01, 5.3002700426865172e-7},
	{"chi2", "sf", 5.0, 0.0, 100.0, 5.2851483609432401e-20},
	{"chi2", "pdf", 5.0, 0.0, 3.0, 0.15418032980376928},
	{"chi2", "ppf", 5.0, 0.0, 1e-20, 3.2334077955187368e-8},
	{"chi2", "isf", 5.0, 0.0, 0.05, 11.070497693516354},
	{"chi2", "sf", 10000.0, 0.0, 11000.0, 3.6183295580964793e-12},
	{"chi2", "ppf", 10000.0, 0.0, 0.5, 9999.3333412351448},
	{"chi2", "pdf", 10000.0, 0.0, 9900.0, 0.0022153875873443173},
	{"chi2", "cdf", 0.1, 0.0, 1e-30, 0.031376943914449888},
	{"chi2", "isf", 0.1, 0.0, 0.3, 0.00093312736979051819},
	{"f", "cdf", 5.0, 12.0, 0.2, 0.043670032648045827},
	{"f", "sf", 5.0, 12.0, 4.0, 0.02283267494113181},
	{"f", "pdf", 5.0, 12.0, 1.0, 0.51059115319443894},
	{"f", "ppf", 5.0, 12.0, 0.95, 3.1058752390841235},
	{"f", "isf", 5.0, 12.0, 1e-15, 1184.4444068268658},
	{"f", "isf", 1.0, 1.0, 1e-100, 4.0528473456935107e+199},
	{"f", "cdf", 1.0, 1.0, 1e-08, 6.3661977024551545e-5},
	{"f", "cdf", 1000.0, 10000.0, 1.0, 0.50510346743022185},
	{"f", "sf", 1000.0, 10000.0, 2.0, 9.6795820502940513e-60},
	{"f", "ppf", 1000.0, 10000.0, 0.999, 1.1521051646629074},
	{"gamma", "cdf", 0.01, 1.0, 1e-200, 0.01005706528500385},
	{"gamma", "sf", 0.01, 1.0, 3.0, 0.00013297135657015548},
	{"gamma", "ppf", 0.01, 1.0, 0.5, 4.4655350189103551e-31},
	{"gamma", "pdf", 3.0, 2.5, 7.0, 0.095350178196341769},
	{"gamma", "cdf", 3.0, 2.5, 1.0, 0.0079263318672538349},
	{"gamma", "sf", 3.0, 2.5, 80.0, 6.9019702242563258e-12},
	{"gamma", "ppf", 3.0, 2.5, 1e-12, 0.0004543007865659361},
	{"gamma", "isf", 3.0, 2.5, 1e-05, 41.38382102103625},
	{"gamma", "logpdf", 3.0, 2.5, 0.001, -17.257929934146685},
	{"gamma", "cdf", 100000.0, 1.0, 90000.0, 1.9782570322356405e-235},
	{"gamma", "sf", 100000.0, 1.0, 100500.0, 0.057103269976028711},
	{"gamma", "isf", 100000.0, 1.0, 1e-300, 112176.85724295326},
	{"beta", "cdf", 2.0, 5.0, 0.1, 0.11426500000000001},
	{"beta", "sf", 2.0, 5.0, 0.8, 0.0015999999999999983},
	{"beta", "pdf", 2.0, 5.0, 0.3, 2.1609000000000001},
	{"beta", "ppf", 2.0, 5.0, 1e-300, 2.5819888974716113e-151},
	{"beta", "isf", 2.0, 5.0, 0.001, 0.81861386691913397},
	{"beta", "cdf", 100000.0, 100000.0, 0.498882, 0.15866261008274089},
	{"beta", "sf", 100000.0, 100000.0, 0.506708, 9.8610109021523736e-10},
	{"beta", "pdf", 100000.0, 100000.0, 0.5, 356.82437719980396},
	{"beta", "cdf", 0.001, 2.0, 0.01, 0.99639086871470497},
	{"beta", "sf", 0.001, 2.0, 0.01, 0.0036091312852950259},
	{"beta", "ppf", 10000.0, 0.5, 1e-20, 0.99565128756759698},
	{"beta", "sf", 10000.0, 0.5, 0.9999, 0.8427059822366485},
	{"beta", "cdf", 30.0, 40.0, 1e-10, 3.162727991385742e-281},
	{"beta", "pdf", 30.0, 40.0, 1e-10, 9.4881839729635508e-270},
	{"expon", "cdf", 2.5, 0.0, 1e-12, 2.4999999999968749e-12},
	{"expon", "sf", 2.5, 0.0, 300.0, 1.9016849634750064e-326},
	{"expon", "ppf", 2.5, 0.0, 0.5, 0.27725887222397812},
	{"expon", "isf", 2.5, 0.0, 1e-300, 276.31021115928548},
	{"binom", "pmf", 20.0, 0.3, 6.0, 0.19163898275344258},
	{"binom", "cdf", 20.0, 0.3, 3.0, 0.10708680450373102},
	{"binom", "sf", 20.0, 0.3, 15.0, 5.550253078298757e-6},
	{"binom", "ppf", 20.0, 0.3, 0.5, 6.0},
	{"binom", "isf", 20.0, 0.3, 1e-08, 18.0},
	{"binom", "logpmf", 20.0, 0.3, 19.0, -20.236425952577526},
	{"binom", "pmf", 10000000.0, 0.3, 2988410.0, 3.4677324270822903e-18},
	{"binom", "cdf", 10000000.0, 0.3, 2988410.0, 6.1998700558793347e-16},
	{"binom", "sf", 10000000.0, 0.3, 3008694.0, 9.9778096182555549e-10},
	{"binom", "logpmf", 10000000.0, 0.3, 3000000.0, -8.1976625159007048},
	{"binom", "ppf", 1000.0, 0.5, 1e-300, 1.0},
	{"binom", "cdf", 1000.0, 0.5, 0.0, 9.3326361850321888e-302},
	{"poisson", "pmf", 4.0, 0.0, 0.0, 0.01831563888873418},
	{"poisson", "pmf", 4.0, 0.0, 10.0, 0.0052924766764201201},
	{"poisson", "cdf", 4.0, 0.0, 1.0, 0.091578194443670901},
	{"poisson", "sf", 4.0, 0.0, 20.0, 1.9230584594146954e-9},
	{"poisson", "ppf", 4.0, 0.0, 0.9, 7.0},
	{"poisson", "isf", 4.0, 0.0, 1e-15, 28.0},
	{"poisson", "pmf", 1000000.0, 0.0, 992000.0, 4.6560970286292952e-18},
	{"poisson", "sf", 1000000.0, 0.0, 1006000.0, 1.0194297537713864e-9},
	{"poisson", "ppf", 1000000.0, 0.0, 1e-300, 963182.0},
	{"poisson", "logpmf", 1000000.0, 0.0, 1000000.0, -7.8266938955201431},
	{"nbinom", "pmf", 3.5, 0.2, 14.0, 0.046508401696868977},
	{"nbinom", "cdf", 3.5, 0.2, 2.0, 0.031626945473757031},
	{"nbinom", "sf", 3.5, 0.2, 60.0, 0.00024226010207504046},
	{"nbinom", "ppf", 3.5, 0.2, 0.5, 13.0},
	{"nbinom", "isf", 3.5, 0.2, 1e-06, 89.0},
	{"nbinom", "cdf", 10000.0, 0.5, 9858.0, 0.15851520584872768},
	{"nbinom", "logpmf", 10000.0, 0.5, 10848.0, -23.183268376438817},
	{"special", "gammainc", 30.0, 0.0, 3e-07, 7.7620679552933447e-229},
	{"special", "gammaincc", 1000.0, 0.0, 2000.0, 6.8473494596147532e-136},
	{"special", "gammaincinv", 4.5, 0.0, 1e-100, 1.4445944703286264e-22},
	{"special", "gammainccinv", 0.1, 0.0, 1e-20, 40.447481384592756},
	{"special", "betainc", 1000000.0, 1000000.0, 0.5, 0.5},
	{"special", "betaln", 1000000.0, 1000000.0, 0.0, -1386300.0033629211},
	{"special", "betaincc", 50.0, 0.5, 0.99, 0.68269560212580263},
	{"special", "betaincinv", 3.0, 7.0, 1e-300, 2.2833557019814711e-101},
	{"special", "expm1", 0.0, 0.0, 1e-06, 1.0000005000001666e-6},
	{"special", "erfinv", 0.0, 0.0, -1e-09, -8.8622692545275807e-10},
	{"special", "erfcinv", 0.0, 0.0, 1e-300, 26.209469960516124},
	{"special", "erfinv", 0.0, 0.0, 0.999, 2.3267537655135245},
}

@(private = "file")
_dist_cont :: proc(d: $D, fn: string, x: f64) -> f64 {
	switch fn {
	case "pdf":
		return st.pdf(d, x)
	case "logpdf":
		return st.logpdf(d, x)
	case "cdf":
		return st.cdf(d, x)
	case "sf":
		return st.sf(d, x)
	case "ppf":
		return st.ppf(d, x)
	case "isf":
		return st.isf(d, x)
	}
	return math.nan_f64()
}

@(private = "file")
_dist_disc :: proc(d: $D, fn: string, x: f64) -> f64 {
	switch fn {
	case "pmf":
		return st.pmf(d, x)
	case "logpmf":
		return st.logpmf(d, x)
	case "cdf":
		return st.cdf(d, x)
	case "sf":
		return st.sf(d, x)
	case "ppf":
		return st.ppf(d, x)
	case "isf":
		return st.isf(d, x)
	}
	return math.nan_f64()
}

@(private = "file")
_dist_eval :: proc(r: Dist_Ref) -> f64 {
	switch r.dist {
	case "normal":
		return _dist_cont(st.Normal{r.a, r.b}, r.fn, r.x)
	case "lognorm":
		return _dist_cont(st.Log_Normal{r.a, r.b}, r.fn, r.x)
	case "t":
		return _dist_cont(st.Student_T{r.a}, r.fn, r.x)
	case "chi2":
		return _dist_cont(st.Chi_Squared{r.a}, r.fn, r.x)
	case "f":
		return _dist_cont(st.Fisher_F{r.a, r.b}, r.fn, r.x)
	case "gamma":
		return _dist_cont(st.Gamma{r.a, r.b}, r.fn, r.x)
	case "beta":
		return _dist_cont(st.Beta{r.a, r.b}, r.fn, r.x)
	case "expon":
		return _dist_cont(st.Exponential{r.a}, r.fn, r.x)
	case "binom":
		return _dist_disc(st.Binomial{int(r.a), r.b}, r.fn, r.x)
	case "poisson":
		return _dist_disc(st.Poisson{r.a}, r.fn, r.x)
	case "nbinom":
		return _dist_disc(st.Negative_Binomial{r.a, r.b}, r.fn, r.x)
	case "special":
		switch r.fn {
		case "gammainc":
			return st.gamma_p(r.a, r.x)
		case "gammaincc":
			return st.gamma_q(r.a, r.x)
		case "gammaincinv":
			return st.gamma_p_inv(r.a, r.x)
		case "gammainccinv":
			return st.gamma_q_inv(r.a, r.x)
		case "betainc":
			return st.beta_inc(r.a, r.b, r.x)
		case "betaincc":
			return st.beta_inc_c(r.a, r.b, r.x)
		case "betaincinv":
			return st.beta_inc_inv(r.a, r.b, r.x)
		case "betaln":
			return st.log_beta(r.a, r.b)
		case "expm1":
			return st.expm1(r.x)
		case "erfinv":
			return st.erf_inv(r.x)
		case "erfcinv":
			return st.erfc_inv(r.x)
		}
	}
	return math.nan_f64()
}

// sqrt(n) times the Kolmogorov-Smirnov distance of the draws to the cdf
// (above 1.95 rejects at the 0.1% level)
@(private = "file")
_dist_ks :: proc(xs: []f64, d: $D) -> f64 {
	slice.sort(xs)
	n := f64(len(xs))
	worst := 0.0
	for x, i in xs {
		f := st.cdf(d, x)
		worst = max(worst, abs(f - f64(i) / n), abs(f - f64(i + 1) / n))
	}
	return worst * math.sqrt(n)
}

// p-value of the chi-squared goodness of fit of the draws, one bin per value
// between the 0.01% and 99.99% quantiles and one for each tail
@(private = "file")
_dist_gof :: proc(ks: []int, d: $D) -> f64 {
	lo := int(st.ppf(d, 1e-4))
	hi := int(st.ppf(d, 1 - 1e-4))
	counts := make([]f64, hi - lo + 3)
	defer delete(counts)
	for k in ks {counts[clamp(k - lo + 1, 0, hi - lo + 2)] += 1}
	n := f64(len(ks))
	chi := 0.0
	bins := 0
	for c, i in counts {
		e: f64
		switch i {
		case 0:
			e = n * st.cdf(d, f64(lo - 1))
		case len(counts) - 1:
			e = n * st.sf(d, f64(hi))
		case:
			e = n * st.pmf(d, f64(lo + i - 1))
		}
		if e <= 0 {continue}
		chi += (c - e) * (c - e) / e
		bins += 1
	}
	return st.sf(st.Chi_Squared{f64(bins - 1)}, chi)
}

stats_distributions_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Statistical distributions ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// 1. reference values, grouped by distribution
	{
		groups := []string {
			"normal",
			"lognorm",
			"t",
			"chi2",
			"f",
			"gamma",
			"beta",
			"expon",
			"binom",
			"poisson",
			"nbinom",
			"special",
		}
		for g in groups {
			n := 0
			worst := 0.0
			bad := ""
			for r in DIST_REFS {
				if r.dist != g {continue}
				n += 1
				v := _dist_eval(r)
				err: f64
				switch {
				case math.is_inf(r.want) || math.is_inf(v):
					err = 0 if v == r.want else math.inf_f64(1)
				case r.fn == "logpdf" || r.fn == "logpmf" || r.fn == "betaln":
					err = abs(v - r.want) / max(1, abs(r.want))
				case:
					err = abs(v - r.want) / max(abs(r.want), 1e-300)
				}
				if !(err <= worst) {
					worst = err
					bad = fmt.tprintf(
						"%s(%v, %v) at %v: %.17g, expected %.17g",
						r.fn,
						r.a,
						r.b,
						r.x,
						v,
						r.want,
					)
				}
			}
			expect(
				&passed,
				&total,
				n > 0 && worst <= 1e-12,
				fmt.tprintf(
					"%s: %d values, largest relative error %.1e%s",
					g,
					n,
					worst,
					"" if worst <= 1e-12 else fmt.tprintf(" (%s)", bad),
				),
			)
		}
	}

	// 2. conventions shared with scipy
	{
		nan := math.is_nan_f64
		ok :=
			nan(st.cdf(st.Normal{0, 0}, 1)) &&
			nan(st.cdf(st.Student_T{-1}, 1)) &&
			nan(st.pdf(st.Beta{0, 1}, 0.5)) &&
			nan(st.pmf(st.Binomial{10, 1.5}, 2)) &&
			nan(st.cdf(st.Poisson{-1}, 2)) &&
			nan(st.ppf(st.Normal{0, 1}, -0.1)) &&
			nan(st.isf(st.Gamma{2, 1}, 1.1)) &&
			nan(st.ppf(st.Binomial{10, 0.5}, 2))
		expect(&passed, &total, ok, "invalid parameters and probabilities give NaN")

		ok =
			st.ppf(st.Normal{0, 1}, 0) == math.inf_f64(-1) &&
			st.ppf(st.Normal{0, 1}, 1) == math.inf_f64(1) &&
			st.ppf(st.Gamma{2, 1}, 0) == 0 &&
			st.ppf(st.Gamma{2, 1}, 1) == math.inf_f64(1) &&
			st.ppf(st.Beta{2, 3}, 0) == 0 &&
			st.ppf(st.Beta{2, 3}, 1) == 1 &&
			st.isf(st.Student_T{4}, 0) == math.inf_f64(1) &&
			st.ppf(st.Student_T{0.3}, 1e-300) == math.inf_f64(-1)
		expect(
			&passed,
			&total,
			ok,
			"quantiles at 0 and 1 are the ends of the support; beyond the doubles, ±Inf",
		)

		b := st.Binomial{10, 0.4}
		ok =
			st.ppf(b, 0) == -1 &&
			st.ppf(b, 1) == 10 &&
			st.ppf(st.Poisson{3}, 1) == math.inf_f64(1) &&
			st.isf(b, 1) == -1 &&
			st.pmf(b, 2.5) == 0 &&
			st.cdf(b, 2.5) == st.cdf(b, 2) &&
			st.logpmf(b, 11) == math.inf_f64(-1)
		expect(
			&passed,
			&total,
			ok,
			"discrete: ppf(0) = -1, ppf(1) = the largest value, pmf of a non-integer 0, cdf rounds down",
		)

		ok =
			st.mean(st.Student_T{1}) == math.inf_f64(1) &&
			st.variance(st.Student_T{1.5}) == math.inf_f64(1) &&
			math.is_nan(st.variance(st.Student_T{1})) &&
			st.variance(st.Fisher_F{3, 4}) == math.inf_f64(1) &&
			st.mean(st.Fisher_F{3, 2}) == math.inf_f64(1) &&
			st.variance(st.Negative_Binomial{3.5, 0.2}) == 70
		expect(&passed, &total, ok, "moments: infinite where they diverge, NaN where undefined")

		// cdf + sf = 1 and ppf(cdf(x)) = x in the centre
		worst := 0.0
		for x in ([]f64{0.3, 1.7, 4.2}) {
			for d in ([]st.Gamma{{0.7, 1}, {5, 0.5}}) {
				worst = max(worst, abs(st.cdf(d, x) + st.sf(d, x) - 1))
				worst = max(worst, abs(st.ppf(d, st.cdf(d, x)) - x) / x)
			}
			f := st.Fisher_F{4, 9}
			worst = max(worst, abs(st.isf(f, st.sf(f, x)) - x) / x)
		}
		expect(
			&passed,
			&total,
			worst < 1e-13,
			fmt.tprintf("round trips: cdf + sf = 1, ppf(cdf(x)) = x (largest error %.1e)", worst),
		)
	}

	// 3. random draws
	{
		N :: 50_000
		worst_ks := 0.0
		worst_name := ""
		ks_of :: proc(d: $D, worst: ^f64, name: ^string, label: string) {
			xs := st.sample(d, N)
			defer delete(xs)
			k := _dist_ks(xs, d)
			if k > worst^ {
				worst^ = k
				name^ = label
			}
		}
		ks_of(st.Normal{1, 2}, &worst_ks, &worst_name, "normal")
		ks_of(st.Log_Normal{0.5, 0.8}, &worst_ks, &worst_name, "log-normal")
		ks_of(st.Student_T{3}, &worst_ks, &worst_name, "t(3)")
		ks_of(st.Chi_Squared{7}, &worst_ks, &worst_name, "chi-squared(7)")
		ks_of(st.Gamma{0.05, 2}, &worst_ks, &worst_name, "gamma(0.05)")
		ks_of(st.Gamma{4.5, 2}, &worst_ks, &worst_name, "gamma(4.5)")
		ks_of(st.Beta{0.2, 0.3}, &worst_ks, &worst_name, "beta(0.2, 0.3)")
		ks_of(st.Beta{2, 5}, &worst_ks, &worst_name, "beta(2, 5)")
		ks_of(st.Fisher_F{5, 12}, &worst_ks, &worst_name, "F(5, 12)")
		ks_of(st.Exponential{2.5}, &worst_ks, &worst_name, "exponential")
		expect(
			&passed,
			&total,
			worst_ks < 1.95,
			fmt.tprintf(
				"continuous draws: Kolmogorov-Smirnov √n·D below 1.95 for 10 distributions (largest %.2f, %s)",
				worst_ks,
				worst_name,
			),
		)

		worst_p := 1.0
		p_name := ""
		gof_of :: proc(d: $D, worst: ^f64, name: ^string, label: string) {
			ks := st.sample(d, N)
			defer delete(ks)
			p := _dist_gof(ks, d)
			if p < worst^ {
				worst^ = p
				name^ = label
			}
		}
		gof_of(st.Binomial{20, 0.3}, &worst_p, &p_name, "binomial(20, 0.3), inversion")
		gof_of(st.Binomial{1000, 0.4}, &worst_p, &p_name, "binomial(1000, 0.4), BTRS")
		gof_of(st.Binomial{200, 0.93}, &worst_p, &p_name, "binomial(200, 0.93)")
		gof_of(st.Poisson{4}, &worst_p, &p_name, "Poisson(4), multiplication")
		gof_of(st.Poisson{57}, &worst_p, &p_name, "Poisson(57), PTRS")
		gof_of(st.Negative_Binomial{3.5, 0.2}, &worst_p, &p_name, "negative binomial(3.5, 0.2)")
		expect(
			&passed,
			&total,
			worst_p > 1e-3,
			fmt.tprintf(
				"discrete draws: chi-squared p-values above 0.001 for 6 distributions (smallest %.3f, %s)",
				worst_p,
				p_name,
			),
		)

		r1 := rand.create(77)
		context.random_generator = rand.default_random_generator(&r1)
		one := st.sample(st.Gamma{2, 1}, 5)
		defer delete(one)
		r2 := rand.create(77)
		context.random_generator = rand.default_random_generator(&r2)
		two := st.sample(st.Gamma{2, 1}, 5)
		defer delete(two)
		expect(&passed, &total, slice.equal(one, two), "the same seed gives the same draws")
	}

	// 4. the helpers elsewhere agree with stats
	{
		t3 := st.Student_T{3}
		ok :=
			ml.student_t_sf2(2.1, 3) == 2 * st.sf(t3, 2.1) &&
			ml.student_t_cdf(-2.1, 3) == st.cdf(t3, -2.1) &&
			ml.student_t_ppf(0.975, 3) == st.ppf(t3, 0.975) &&
			ml.f_dist_sf(4, 5, 12) == st.sf(st.Fisher_F{5, 12}, 4) &&
			a.chi2_sf(100, 5) == st.sf(st.Chi_Squared{5}, 100) &&
			fin.norm_inv(0.025) == st.ppf(st.Normal{0, 1}, 0.025)
		expect(
			&passed,
			&total,
			ok,
			"analytics chi2_sf, ML's t and F helpers and finance norm_inv give the stats values",
		)
	}

	fmt.printf("Distributions: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
