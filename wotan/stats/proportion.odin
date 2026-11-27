package stats

import "core:math"

// Tests and intervals for proportions: the exact binomial test, confidence
// intervals for a proportion, and z-tests for one or two proportions.

// Result of binomial_test.
Binomial_Test_Result :: struct {
	estimate: f64, // k / n
	p_value:  f64,
	ci_low:   f64, // Clopper-Pearson interval at the requested confidence,
	ci_high:  f64, // one-sided for a one-sided alternative (as R's binom.test)
}

// Exact binomial test: is the success probability of k successes in n
// trials equal to p? The two-sided p-value adds the probabilities of all
// outcomes no more likely than k (scipy's binomtest, R's binom.test).
binomial_test :: proc(
	k, n: int,
	p := 0.5,
	alternative := Alternative.Two_Sided,
	confidence := 0.95,
) -> (
	res: Binomial_Test_Result,
	ok: bool,
) {
	if n < 1 ||
	   k < 0 ||
	   k > n ||
	   !(p >= 0 && p <= 1) ||
	   !_valid_confidence(confidence) {return {}, false}
	d := Binomial{n, p}
	kf, nf := f64(k), f64(n)
	switch alternative {
	case .Less:
		res.p_value = binomial_cdf(d, kf)
	case .Greater:
		res.p_value = binomial_sf(d, kf - 1)
	case .Two_Sided:
		dk := binomial_pmf(d, kf)
		rerr := 1 + 1e-7
		if kf == p * nf {
			res.p_value = 1
		} else if kf < p * nf {
			// outcomes above the mean with probability at most that of k
			ix := _search_increasing(_neg_binom_pmf, &d, -dk * rerr, math.ceil(p * nf), nf)
			y := nf - ix + (1 if dk * rerr == binomial_pmf(d, ix) else 0)
			res.p_value = binomial_cdf(d, kf) + binomial_sf(d, nf - y)
		} else {
			ix := _search_increasing(_binom_pmf, &d, dk * rerr, 0, math.floor(p * nf))
			res.p_value = binomial_cdf(d, ix) + binomial_sf(d, kf - 1)
		}
		res.p_value = min(1, res.p_value)
	}
	res.estimate = kf / nf
	res.ci_low, res.ci_high, _ = proportion_ci(k, n, confidence, .Clopper_Pearson, alternative)
	return res, true
}

@(private)
_binom_pmf :: proc(data: rawptr, x: f64) -> f64 {return binomial_pmf((^Binomial)(data)^, x)}

@(private)
_neg_binom_pmf :: proc(data: rawptr, x: f64) -> f64 {return -binomial_pmf((^Binomial)(data)^, x)}

// largest integer x in [lo, hi] with f(x) <= target, f increasing there;
// lo - 1 if there is none (scipy's _binary_search_for_binom_tst)
@(private)
_search_increasing :: proc(
	f: proc(data: rawptr, x: f64) -> f64,
	data: rawptr,
	target, lo_in, hi_in: f64,
) -> f64 {
	lo, hi := lo_in, hi_in
	for lo < hi {
		mid := lo + math.floor((hi - lo) / 2)
		v := f(data, mid)
		if v < target {
			lo = mid + 1
		} else if v > target {
			hi = mid - 1
		} else {
			return mid
		}
	}
	return lo if f(data, lo) <= target else lo - 1
}

// Confidence interval for a proportion.
CI_Method :: enum {
	Wilson, // score interval; good coverage, the default
	Wilson_CC, // Wilson with continuity correction
	Clopper_Pearson, // exact (beta quantiles); conservative
	Jeffreys, // beta quantiles of the Jeffreys posterior
	Agresti_Coull, // Wald interval on k + z²/2 successes in n + z² trials
	Normal, // Wald interval p ± z √(p (1 - p) / n), clipped to [0, 1]
}

// Confidence interval for the success probability from k successes in n
// trials. A one-sided alternative gives a one-sided interval: Less bounds
// the probability from above (lo = 0), Greater from below (hi = 1). The
// formulas follow scipy's binomtest(...).proportion_ci (Wilson, Wilson_CC,
// Clopper_Pearson) and statsmodels' proportion_confint (the others).
proportion_ci :: proc(
	k, n: int,
	confidence := 0.95,
	method := CI_Method.Wilson,
	alternative := Alternative.Two_Sided,
) -> (
	lo, hi: f64,
	ok: bool,
) {
	if n < 1 || k < 0 || k > n || !_valid_confidence(confidence) {return 0, 0, false}
	kf, nf := f64(k), f64(n)
	alpha := (1 - confidence) / 2 if alternative == .Two_Sided else 1 - confidence
	z := normal_isf(STANDARD_NORMAL, alpha)
	p := kf / nf
	q := 1 - p
	switch method {
	case .Clopper_Pearson:
		lo = 0 if k == 0 else beta_ppf(Beta{kf, nf - kf + 1}, alpha)
		hi = 1 if k == n else beta_isf(Beta{kf + 1, nf - kf}, alpha)
	case .Jeffreys:
		lo = beta_ppf(Beta{kf + 0.5, nf - kf + 0.5}, alpha)
		hi = beta_isf(Beta{kf + 0.5, nf - kf + 0.5}, alpha)
	case .Wilson:
		denom := 2 * (nf + z * z)
		center := (2 * nf * p + z * z) / denom
		delta := z / denom * math.sqrt(4 * nf * p * q + z * z)
		lo = 0 if k == 0 else center - delta
		hi = 1 if k == n else center + delta
	case .Wilson_CC:
		denom := 2 * (nf + z * z)
		center := (2 * nf * p + z * z) / denom
		lo =
			0 if k == 0 else center - (1 + z * math.sqrt(z * z - 2 - 1 / nf + 4 * p * (nf * q + 1))) / denom
		hi =
			1 if k == n else center + (1 + z * math.sqrt(z * z + 2 - 1 / nf + 4 * p * (nf * q - 1))) / denom
	case .Agresti_Coull:
		nc := nf + z * z
		pc := (kf + z * z / 2) / nc
		d := z * math.sqrt(pc * (1 - pc) / nc)
		lo, hi = clamp(pc - d, 0, 1), clamp(pc + d, 0, 1)
	case .Normal:
		d := z * math.sqrt(p * q / nf)
		lo, hi = clamp(p - d, 0, 1), clamp(p + d, 0, 1)
	}
	if alternative == .Less {lo = 0}
	if alternative == .Greater {hi = 1}
	return lo, hi, true
}

// z-test of one proportion: is the success probability of k successes in n
// trials equal to p0? The standard error uses the observed proportion, or p0
// with null_variance (statsmodels' proportions_ztest, prop_var = p0).
proportion_z_test_one_sample :: proc(
	k, n: int,
	p0: f64,
	alternative := Alternative.Two_Sided,
	null_variance := false,
) -> (
	res: Test_Result,
	ok: bool,
) {
	if n < 1 || k < 0 || k > n || !(p0 >= 0 && p0 <= 1) {return {}, false}
	p := f64(k) / f64(n)
	pv := p0 if null_variance else p
	res.statistic = (p - p0) / math.sqrt(pv * (1 - pv) / f64(n))
	res.p_value = _p_normal(res.statistic, alternative)
	return res, true
}

// z-test of two proportions k1 / n1 and k2 / n2: is their difference equal
// to diff? Uses the pooled proportion for the standard error
// (statsmodels' proportions_ztest).
proportion_z_test_two_sample :: proc(
	k1, n1, k2, n2: int,
	diff := 0.0,
	alternative := Alternative.Two_Sided,
) -> (
	res: Test_Result,
	ok: bool,
) {
	if n1 < 1 || n2 < 1 || k1 < 0 || k2 < 0 || k1 > n1 || k2 > n2 {return {}, false}
	a, b := f64(n1), f64(n2)
	pp := f64(k1 + k2) / (a + b)
	res.statistic = (f64(k1) / a - f64(k2) / b - diff) / math.sqrt(pp * (1 - pp) * (1 / a + 1 / b))
	res.p_value = _p_normal(res.statistic, alternative)
	return res, true
}
