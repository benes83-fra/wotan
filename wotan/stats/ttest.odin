package stats

import "core:math"

// t-tests and z-tests on means.

// Result of a test on a mean or a difference of means.
Mean_Test_Result :: struct {
	statistic: f64, // t (or z): (estimate - mu) / std_error
	p_value:   f64,
	df:        f64, // degrees of freedom of the t reference; 0 for a z-test
	estimate:  f64, // the mean, or the difference of means x - y
	std_error: f64,
	ci_low:    f64, // confidence interval for the estimate at the requested level;
	ci_high:   f64, // one-sided alternatives leave one end infinite
}

// One-sample t-test: is the mean of x equal to mu? Needs 2 values.
t_test_one_sample :: proc(
	x: []f64,
	mu := 0.0,
	alternative := Alternative.Two_Sided,
	confidence := 0.95,
) -> (
	res: Mean_Test_Result,
	ok: bool,
) {
	if len(x) < 2 || !_all_finite(x) || !_valid_confidence(confidence) {return {}, false}
	n := f64(len(x))
	m, v := _mean_var(x)
	return _mean_test(m, _shifted_mean(x, mu), math.sqrt(v / n), n - 1, alternative, confidence),
		true
}

// Paired t-test: the one-sample test on the differences x[i] - y[i].
t_test_paired :: proc(
	x, y: []f64,
	mu := 0.0,
	alternative := Alternative.Two_Sided,
	confidence := 0.95,
) -> (
	res: Mean_Test_Result,
	ok: bool,
) {
	if len(x) != len(y) || len(x) < 2 {return {}, false}
	d := make([]f64, len(x))
	defer delete(d)
	for i in 0 ..< len(x) {d[i] = x[i] - y[i]}
	return t_test_one_sample(d, mu, alternative, confidence)
}

// Two-sample t-test: is mean(x) - mean(y) equal to mu? By default Welch's
// test, which does not assume equal variances (as R's t.test); equal_var
// gives Student's test with the pooled variance (scipy's default). Needs 2
// values in each sample.
t_test_two_sample :: proc(
	x, y: []f64,
	mu := 0.0,
	equal_var := false,
	alternative := Alternative.Two_Sided,
	confidence := 0.95,
) -> (
	res: Mean_Test_Result,
	ok: bool,
) {
	if len(x) < 2 ||
	   len(y) < 2 ||
	   !_all_finite(x) ||
	   !_all_finite(y) ||
	   !_valid_confidence(confidence) {return {}, false}
	n1, n2 := f64(len(x)), f64(len(y))
	_, v1 := _mean_var(x)
	_, v2 := _mean_var(y)
	diff := _mean_diff(x, y)
	se, df: f64
	if equal_var {
		df = n1 + n2 - 2
		sp := ((n1 - 1) * v1 + (n2 - 1) * v2) / df
		se = math.sqrt(sp * (1 / n1 + 1 / n2))
	} else {
		a, b := v1 / n1, v2 / n2
		se = math.sqrt(a + b)
		df = (a + b) * (a + b) / (a * a / (n1 - 1) + b * b / (n2 - 1))
	}
	return _mean_test(diff, diff - mu, se, df, alternative, confidence), true
}

// One-sample z-test: as the t-test, with the normal distribution as
// reference (statsmodels' ztest). Suited to large samples.
z_test_one_sample :: proc(
	x: []f64,
	mu := 0.0,
	alternative := Alternative.Two_Sided,
	confidence := 0.95,
) -> (
	res: Mean_Test_Result,
	ok: bool,
) {
	if len(x) < 2 || !_all_finite(x) || !_valid_confidence(confidence) {return {}, false}
	n := f64(len(x))
	m, v := _mean_var(x)
	return _mean_test(m, _shifted_mean(x, mu), math.sqrt(v / n), 0, alternative, confidence), true
}

// Two-sample z-test on mean(x) - mean(y) with the pooled variance
// (statsmodels' ztest).
z_test_two_sample :: proc(
	x, y: []f64,
	mu := 0.0,
	alternative := Alternative.Two_Sided,
	confidence := 0.95,
) -> (
	res: Mean_Test_Result,
	ok: bool,
) {
	if len(x) < 2 ||
	   len(y) < 2 ||
	   !_all_finite(x) ||
	   !_all_finite(y) ||
	   !_valid_confidence(confidence) {return {}, false}
	n1, n2 := f64(len(x)), f64(len(y))
	_, v1 := _mean_var(x)
	_, v2 := _mean_var(y)
	diff := _mean_diff(x, y)
	sp := ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
	return _mean_test(
			diff,
			diff - mu,
			math.sqrt(sp * (1 / n1 + 1 / n2)),
			0,
			alternative,
			confidence,
		),
		true
}

// mean(x) - mean(y), both taken relative to x[0] so that a large common
// offset does not round the difference
@(private)
_mean_diff :: proc(x, y: []f64) -> f64 {
	return _shifted_mean(x, x[0]) - _shifted_mean(y, x[0])
}

// statistic, p-value and interval from an estimate, its excess over the
// hypothesised value (computed without cancellation by the caller), its
// standard error and the t degrees of freedom (0: normal reference)
@(private)
_mean_test :: proc(
	estimate, excess, se, df: f64,
	alternative: Alternative,
	confidence: f64,
) -> (
	r: Mean_Test_Result,
) {
	r.estimate = estimate
	r.std_error = se
	r.df = df
	r.statistic = excess / se
	if df == 0 {
		r.p_value = _p_normal(r.statistic, alternative)
		r.ci_low, r.ci_high = _ci(estimate, se, confidence, alternative, _upper_q_normal, nil)
	} else {
		r.p_value = _p_t(r.statistic, df, alternative)
		d := df
		r.ci_low, r.ci_high = _ci(estimate, se, confidence, alternative, _upper_q_t, &d)
	}
	return
}
