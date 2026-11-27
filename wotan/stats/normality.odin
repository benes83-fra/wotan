package stats

import w "../core"
import "core:math"

// Tests of normality: Shapiro-Wilk, Anderson-Darling, D'Agostino-Pearson
// (with its skewness and kurtosis parts) and Jarque-Bera.

// Shapiro-Wilk test of normality, by Royston's algorithm AS R94 for 3 to
// 5000 values (larger samples are accepted, their p-values less accurate).
// The statistic W is near 1 for normal data; small W rejects normality.
// Computed in double precision throughout, as R's shapiro.test (scipy's
// single-precision Fortran agrees to about 1e-6). All values equal gives
// ok = false.
shapiro_wilk :: proc(x: []f64) -> (res: Test_Result, ok: bool) {
	n := len(x)
	if n < 3 || !_all_finite(x) {return {}, false}
	w.TEMP_GUARD()
	s := _sorted(x)
	rng := s[n - 1] - s[0]
	if rng <= 0 {return {}, false}
	nn2 := n / 2
	a := make([]f64, nn2 + 1, context.temp_allocator) // 1-based
	an := f64(n)
	SMALL :: 1e-19
	c1 := [6]f64{0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056}
	c2 := [6]f64{0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633}
	if n == 3 {
		a[1] = math.SQRT_TWO / 2
	} else {
		an25 := an + 0.25
		summ2 := 0.0
		for i in 1 ..= nn2 {
			a[i] = normal_ppf(STANDARD_NORMAL, (f64(i) - 0.375) / an25)
			summ2 += a[i] * a[i]
		}
		summ2 *= 2
		ssumm2 := math.sqrt(summ2)
		rsn := 1 / math.sqrt(an)
		a1 := _sw_poly(c1[:], rsn) - a[1] / ssumm2
		i1: int
		fac: f64
		if n > 5 {
			i1 = 3
			a2 := -a[2] / ssumm2 + _sw_poly(c2[:], rsn)
			fac = math.sqrt(
				(summ2 - 2 * (a[1] * a[1]) - 2 * (a[2] * a[2])) /
				(1 - 2 * (a1 * a1) - 2 * (a2 * a2)),
			)
			a[2] = a2
		} else {
			i1 = 2
			fac = math.sqrt((summ2 - 2 * (a[1] * a[1])) / (1 - 2 * (a1 * a1)))
		}
		a[1] = a1
		for i in i1 ..= nn2 {a[i] /= -fac}
	}
	// W as the squared correlation of the data and the coefficients; the
	// data relative to the smallest value (W does not change, and offset
	// data keep their precision)
	sx := 0.0
	sa := -a[1]
	for i, j := 1, n - 1; i < n; j -= 1 {
		sx += (s[i] - s[0]) / rng
		i += 1
		if i != j {sa += f64(_sign(i - j)) * a[min(i, j)]}
	}
	sa /= an
	sx /= an
	ssa, ssx, sax: f64
	for i in 0 ..< n {
		asa := _sw_coef(a, i, n - 1 - i) - sa
		xsx := (s[i] - s[0]) / rng - sx
		ssa += asa * asa
		ssx += xsx * xsx
		sax += asa * xsx
	}
	// w1 = 1 - W = 1 - r² as the residual sum of squares of the data on the
	// coefficients over ssx: no cancellation when W is near 1
	beta := sax / ssa
	rss := 0.0
	for i in 0 ..< n {
		r := ((s[i] - s[0]) / rng - sx) - beta * (_sw_coef(a, i, n - 1 - i) - sa)
		rss += r * r
	}
	w1 := rss / ssx
	res.statistic = 1 - w1
	if n == 3 {
		// exact
		res.p_value = max(0, 6 / math.PI * (math.asin(math.sqrt(res.statistic)) - math.PI / 3))
		return res, true
	}
	y := math.ln(w1)
	lxx := math.ln(an)
	m, sd: f64
	if n <= 11 {
		gamma := -2.273 + 0.459 * an
		if y >= gamma {
			res.p_value = 1e-99
			return res, true
		}
		y = -math.ln(gamma - y)
		m = _sw_poly([]f64{0.544, -0.39978, 0.025054, -6.714e-4}, an)
		sd = math.exp(_sw_poly([]f64{1.3822, -0.77857, 0.062767, -0.0020322}, an))
	} else {
		m = _sw_poly([]f64{-1.5861, -0.31082, -0.083751, 0.0038915}, lxx)
		sd = math.exp(_sw_poly([]f64{-0.4803, -0.082676, 0.0030302}, lxx))
	}
	res.p_value = normal_sf(Normal{m, sd}, y)
	return res, true
}

// coefficient of the i-th smallest value (j = n - 1 - i): -a[k] in the
// lower half, a[k] in the upper, 0 in the middle
@(private)
_sw_coef :: proc(a: []f64, i, j: int) -> f64 {
	if i == j {return 0}
	return f64(_sign(i - j)) * a[1 + min(i, j)]
}

// c[0] + c[1] x + ... (AS 181's poly)
@(private)
_sw_poly :: proc(c: []f64, x: f64) -> f64 {
	r := c[0]
	if len(c) > 1 {
		p := x * c[len(c) - 1]
		for j := len(c) - 2; j > 0; j -= 1 {p = (p + c[j]) * x}
		r += p
	}
	return r
}

@(private)
_sign :: proc(v: int) -> int {return 1 if v > 0 else (-1 if v < 0 else 0)}

// Result of anderson_darling.
Anderson_Result :: struct {
	statistic:       f64, // A²
	p_value:         f64, // from D'Agostino and Stephens' approximation (statsmodels' normal_ad)
	critical_values: [5]f64, // A² beyond critical_values[i] rejects at level ANDERSON_LEVELS[i]
}

// Significance levels (%) of Anderson_Result.critical_values.
ANDERSON_LEVELS :: [5]f64{15, 10, 5, 2.5, 1}

// Anderson-Darling test of normality, mean and standard deviation estimated
// from the data. Gives the critical values of scipy's anderson and the
// p-value of statsmodels' normal_ad. Needs 3 values; all equal gives NaN.
anderson_darling :: proc(x: []f64) -> (res: Anderson_Result, ok: bool) {
	n := len(x)
	if n < 3 || !_all_finite(x) {return {}, false}
	w.TEMP_GUARD()
	y := _sorted(x)
	_, v := _mean_var(x)
	cn := _center(x)
	sd := math.sqrt(v)
	nf := f64(n)
	// A² = -n - S with S near -n: S summed with Neumaier's compensation
	s, comp := nf, 0.0 // n + S
	for i in 0 ..< n {
		lc := _log_ndtr(_dev(cn, y[i]) / sd)
		ls := _log_ndtr(-_dev(cn, y[n - 1 - i]) / sd)
		t := f64(2 * i + 1) / nf * (lc + ls)
		u := s + t
		comp += (s - u) + t if abs(s) >= abs(t) else (t - u) + s
		s = u
	}
	a2 := -(s + comp)
	res.statistic = a2
	avals := [5]f64{0.576, 0.656, 0.787, 0.918, 1.092}
	for i in 0 ..< 5 {res.critical_values[i] = _round_half_even(avals[i] / (1 + 4 / nf - 25 / (nf * nf)) * 1000) / 1000}
	aa := a2 * (1 + 0.75 / nf + 2.25 / (nf * nf))
	switch {
	case math.is_nan(aa):
		res.p_value = aa
	case aa < 0.2:
		res.p_value = 1 - math.exp(-13.436 + 101.14 * aa - 223.73 * aa * aa)
	case aa < 0.34:
		res.p_value = 1 - math.exp(-8.318 + 42.796 * aa - 59.938 * aa * aa)
	case aa < 0.6:
		res.p_value = math.exp(0.9177 - 4.279 * aa - 1.38 * aa * aa)
	case aa <= 13:
		res.p_value = math.exp(1.2937 - 5.709 * aa + 0.0186 * aa * aa)
	case:
		res.p_value = 0
	}
	return res, true
}

// log Φ(z), also far in the lower tail
@(private)
_log_ndtr :: proc(z: f64) -> f64 {
	if z > -30 {return math.ln(normal_cdf(STANDARD_NORMAL, z))}
	// asymptotic series of the Mills ratio
	z2 := z * z
	s, t := 1.0, 1.0
	for k in 1 ..= 12 {
		t *= -f64(2 * k - 1) / z2
		s += t
	}
	return -0.5 * z2 - math.ln(-z) - 0.5 * math.ln(2 * math.PI) + math.ln(s)
}

@(private)
_round_half_even :: proc(x: f64) -> f64 {
	r := math.round(x)
	if abs(x - math.trunc(x)) == 0.5 {r = 2 * math.round(x / 2)}
	return r
}

// D'Agostino's test that the skewness is that of a normal distribution
// (scipy's skewtest). Needs 8 values.
skew_test :: proc(x: []f64, alternative := Alternative.Two_Sided) -> (res: Test_Result, ok: bool) {
	if len(x) < 8 || !_all_finite(x) {return {}, false}
	res.statistic = _skew_z(x)
	res.p_value = _p_normal(res.statistic, alternative)
	return res, true
}

// Anscombe and Glynn's test that the kurtosis is that of a normal
// distribution (scipy's kurtosistest). Needs 5 values; meant for 20 or more.
kurtosis_test :: proc(
	x: []f64,
	alternative := Alternative.Two_Sided,
) -> (
	res: Test_Result,
	ok: bool,
) {
	if len(x) < 5 || !_all_finite(x) {return {}, false}
	res.statistic = _kurtosis_z(x)
	res.p_value = _p_normal(res.statistic, alternative)
	return res, true
}

// D'Agostino and Pearson's omnibus test of normality: the sum of the squared
// z-scores of skew_test and kurtosis_test, chi-squared with 2 degrees of
// freedom (scipy's normaltest). Needs 8 values.
dagostino_pearson :: proc(x: []f64) -> (res: Test_Result, ok: bool) {
	if len(x) < 8 || !_all_finite(x) {return {}, false}
	zs := _skew_z(x)
	zk := _kurtosis_z(x)
	res.statistic = zs * zs + zk * zk
	res.df = 2
	res.p_value = chi_squared_sf(Chi_Squared{2}, res.statistic)
	return res, true
}

// Jarque-Bera test of normality: n/6 (S² + (K - 3)²/4) from the sample
// skewness S and kurtosis K, chi-squared with 2 degrees of freedom (scipy's
// jarque_bera). Needs 2 values; all equal gives NaN.
jarque_bera :: proc(x: []f64) -> (res: Test_Result, ok: bool) {
	if len(x) < 2 || !_all_finite(x) {return {}, false}
	nf := f64(len(x))
	s, k := _skew_kurt(x)
	res.statistic = nf / 6 * (s * s + (k - 3) * (k - 3) / 4)
	res.df = 2
	res.p_value = chi_squared_sf(Chi_Squared{2}, res.statistic)
	return res, true
}

// sample skewness m3 / m2^1.5 and kurtosis m4 / m2² (biased; NaN for
// constant data)
@(private)
_skew_kurt :: proc(x: []f64) -> (skew, kurt: f64) {
	mean, m2, m3, m4 := _central_moments(x)
	eps := 2.220446049250313e-16 * abs(mean)
	if m2 <= eps * eps {return math.nan_f64(), math.nan_f64()}
	return m3 / math.pow(m2, 1.5), m4 / (m2 * m2)
}

@(private)
_skew_z :: proc(x: []f64) -> f64 {
	n := f64(len(x))
	b2, _ := _skew_kurt(x)
	y := b2 * math.sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
	beta2 :=
		3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
	w2 := -1 + math.sqrt(2 * (beta2 - 1))
	delta := 1 / math.sqrt(0.5 * math.ln(w2))
	alpha := math.sqrt(2 / (w2 - 1))
	t := y / alpha
	return delta * math.asinh(t)
}

@(private)
_kurtosis_z :: proc(x: []f64) -> f64 {
	n := f64(len(x))
	_, b2 := _skew_kurt(x)
	e := 3 * (n - 1) / (n + 1)
	varb2 := 24 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5))
	xs := (b2 - e) / math.sqrt(varb2)
	sb1 :=
		6 *
		(n * n - 5 * n + 2) /
		((n + 7) * (n + 9)) *
		math.sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
	a := 6 + 8 / sb1 * (2 / sb1 + math.sqrt(1 + 4 / (sb1 * sb1)))
	term1 := 1 - 2 / (9 * a)
	denom := 1 + xs * math.sqrt(2 / (a - 4))
	if denom == 0 {return math.nan_f64()}
	term2 := math.sign(denom) * math.cbrt((1 - 2 / a) / abs(denom))
	return (term1 - term2) / math.sqrt(2 / (9 * a))
}
