package stats

import "core:math"
import "core:slice"

// Hypothesis tests: shared types and helpers.
//
// Every test returns (result, ok). ok is false when the input cannot be
// tested at all: too few observations, slices of different lengths where
// pairs are needed, a non-finite value, a confidence level outside (0, 1).
// Data that can be tested but make the statistic undefined (all values
// equal, say) give ok = true and a NaN or infinite statistic, as in scipy.

// The alternative hypothesis. For two samples, Less means the first tends
// to be smaller than the second (mean, median or distribution); for one
// sample, smaller than the hypothesised value.
Alternative :: enum {
	Two_Sided,
	Less,
	Greater,
}

// How a p-value from a discrete null distribution is computed. Auto picks
// the exact distribution for small samples without ties (the rule of each
// test says when), otherwise the normal (or limiting) approximation.
Method :: enum {
	Auto,
	Exact,
	Asymptotic,
}

// Result of a test with a single statistic.
Test_Result :: struct {
	statistic: f64,
	p_value:   f64,
	df:        f64, // degrees of freedom of a chi-squared, t or F reference (numerator for F); 0 if none
	df2:       f64, // denominator degrees of freedom of an F reference; 0 otherwise
}

// ---------------------------------------------------------------------------
// p-values from a statistic

// p-value of a statistic z that is standard normal under the null
@(private)
_p_normal :: proc(z: f64, alternative: Alternative) -> f64 {
	switch alternative {
	case .Less:
		return normal_cdf(STANDARD_NORMAL, z)
	case .Greater:
		return normal_sf(STANDARD_NORMAL, z)
	case .Two_Sided:
		return min(1, 2 * normal_sf(STANDARD_NORMAL, abs(z)))
	}
	return math.nan_f64()
}

// p-value of a t statistic with df degrees of freedom
@(private)
_p_t :: proc(t, df: f64, alternative: Alternative) -> f64 {
	d := Student_T{df}
	switch alternative {
	case .Less:
		return student_t_cdf(d, t)
	case .Greater:
		return student_t_sf(d, t)
	case .Two_Sided:
		return min(1, 2 * student_t_sf(d, abs(t)))
	}
	return math.nan_f64()
}

// confidence interval estimate ± crit * se, one-sided ones open on one side;
// crit_fn(p) is the upper p quantile of the reference distribution
@(private)
_ci :: proc(
	estimate, se, confidence: f64,
	alternative: Alternative,
	upper_q: proc(data: rawptr, q: f64) -> f64,
	data: rawptr,
) -> (
	lo, hi: f64,
) {
	switch alternative {
	case .Two_Sided:
		c := upper_q(data, (1 - confidence) / 2)
		return estimate - c * se, estimate + c * se
	case .Less:
		return math.inf_f64(-1), estimate + upper_q(data, 1 - confidence) * se
	case .Greater:
		return estimate - upper_q(data, 1 - confidence) * se, math.inf_f64(1)
	}
	return math.nan_f64(), math.nan_f64()
}

@(private)
_upper_q_normal :: proc(data: rawptr, q: f64) -> f64 {return normal_isf(STANDARD_NORMAL, q)}

@(private)
_upper_q_t :: proc(data: rawptr, q: f64) -> f64 {return student_t_isf(Student_T{(^f64)(data)^}, q)}

// ---------------------------------------------------------------------------
// input checks and moments

@(private)
_all_finite :: proc(x: []f64) -> bool {
	for v in x {if math.is_nan(v) || math.is_inf(v) {return false}}
	return true
}

@(private)
_valid_confidence :: proc(c: f64) -> bool {return c > 0 && c < 1}

// The centre of a sample as c + m, c a value of the sample and m the mean of
// x - c (refined by the mean of the residuals). Deviations from the mean
// are then (x - c) - m: values sharing a large offset (prices near 1e6, say)
// lose nothing to the rounding of the mean itself.
@(private)
_Center :: struct {
	c, m: f64,
}

@(private)
_center :: proc(x: []f64) -> _Center {
	if len(x) == 0 {return {}}
	c := x[0]
	return {c, _shifted_mean(x, c)}
}

// mean of x - c, refined by the mean of the residuals
@(private)
_shifted_mean :: proc(x: []f64, c: f64) -> f64 {
	n := f64(len(x))
	m := 0.0
	for v in x {m += v - c}
	m /= n
	r := 0.0
	for v in x {r += (v - c) - m}
	return m + r / n
}

@(private)
_dev :: #force_inline proc(cn: _Center, v: f64) -> f64 {return (v - cn.c) - cn.m}

// mean and variance with divisor n - ddof
@(private)
_mean_var :: proc(x: []f64, ddof := 1.0) -> (mean, variance: f64) {
	n := f64(len(x))
	cn := _center(x)
	ss: _Sum
	for v in x {
		d := _dev(cn, v)
		_sum_add(&ss, d * d)
	}
	return cn.c + cn.m, _sum(ss) / (n - ddof)
}

// Neumaier's compensated sum
@(private)
_Sum :: struct {
	s, c: f64,
}

@(private)
_sum_add :: #force_inline proc(a: ^_Sum, v: f64) {
	t := a.s + v
	if abs(a.s) >= abs(v) {a.c += (a.s - t) + v} else {a.c += (v - t) + a.s}
	a.s = t
}

@(private)
_sum :: #force_inline proc(a: _Sum) -> f64 {return a.s + a.c}

// biased central moments m2, m3, m4 (divisor n)
@(private)
_central_moments :: proc(x: []f64) -> (mean, m2, m3, m4: f64) {
	n := f64(len(x))
	cn := _center(x)
	mean = cn.c + cn.m
	s2, s3, s4: _Sum
	for v in x {
		d := _dev(cn, v)
		d2 := d * d
		_sum_add(&s2, d2)
		_sum_add(&s3, d2 * d)
		_sum_add(&s4, d2 * d2)
	}
	return mean, _sum(s2) / n, _sum(s3) / n, _sum(s4) / n
}

// ---------------------------------------------------------------------------
// ranks

// Average ranks (1-based, ties share the mean of their positions) of x,
// written to ranks. Returns the tie term Σ (t³ - t) over groups of t equal
// values. ranks must have the length of x; scratch from the temp allocator.
@(private)
_rank :: proc(x: []f64, ranks: []f64) -> (tie_term: f64) {
	n := len(x)
	idx := _order(x)
	i := 0
	for i < n {
		j := i + 1
		for j < n && x[idx[j]] == x[idx[i]] {j += 1}
		r := 0.5 * f64(i + j + 1) // mean of positions i+1 .. j
		for k in i ..< j {ranks[idx[k]] = r}
		t := f64(j - i)
		tie_term += t * t * t - t
		i = j
	}
	return
}

@(private)
_Keyed :: struct {
	v: f64,
	i: int,
}

// indices that sort x ascending (temp allocator)
@(private)
_order :: proc(x: []f64) -> []int {
	n := len(x)
	k := make([]_Keyed, n, context.temp_allocator)
	for i in 0 ..< n {k[i] = {x[i], i}}
	slice.sort_by(k, proc(a, b: _Keyed) -> bool {return a.v < b.v || (a.v == b.v && a.i < b.i)})
	idx := make([]int, n, context.temp_allocator)
	for i in 0 ..< n {idx[i] = k[i].i}
	return idx
}

// sorted copy of x in the temp allocator
@(private)
_sorted :: proc(x: []f64) -> []f64 {
	s := make([]f64, len(x), context.temp_allocator)
	copy(s, x)
	slice.sort(s)
	return s
}

// median of a sorted slice
@(private)
_median_sorted :: proc(s: []f64) -> f64 {
	n := len(s)
	if n % 2 == 1 {return s[n / 2]}
	return 0.5 * (s[n / 2 - 1] + s[n / 2])
}
