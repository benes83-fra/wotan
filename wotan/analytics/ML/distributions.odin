package ML

import "core:math"

// Student-t distribution helpers for regression inference (p-values, CIs).

// Regularized incomplete beta I_x(a, b) (continued fraction, modified Lentz)
_reg_inc_beta :: proc(a, b, x: f64) -> f64 {
	if x <= 0.0 do return 0.0
	if x >= 1.0 do return 1.0
	// Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) where the fraction converges fast
	if x > (a + 1.0) / (a + b + 2.0) {
		return 1.0 - _reg_inc_beta(b, a, 1.0 - x)
	}
	lab, _ := math.lgamma(a + b)
	la, _ := math.lgamma(a)
	lb, _ := math.lgamma(b)
	front := math.exp(lab - la - lb + a * math.ln(x) + b * math.ln(1.0 - x)) / a

	tiny :: 1e-300
	f, c, d := 1.0, 1.0, 0.0
	for i in 0 ..= 400 {
		m := f64(i / 2)
		num: f64
		if i == 0 {
			num = 1.0
		} else if i % 2 == 0 {
			num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
		} else {
			num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
		}
		d = 1.0 + num * d
		if math.abs(d) < tiny do d = tiny
		d = 1.0 / d
		c = 1.0 + num / c
		if math.abs(c) < tiny do c = tiny
		cd := c * d
		f *= cd
		if math.abs(1.0 - cd) < 1e-15 do break
	}
	return front * (f - 1.0)
}

// Two-sided tail probability P(|T| > |t|) for T ~ t(df)
student_t_sf2 :: proc(t, df: f64) -> f64 {
	if math.is_nan(t) || df <= 0.0 do return math.nan_f64()
	if math.is_inf(t) do return 0.0
	return _reg_inc_beta(0.5 * df, 0.5, df / (df + t * t))
}

// CDF of t(df)
student_t_cdf :: proc(t, df: f64) -> f64 {
	tail := 0.5 * student_t_sf2(t, df)
	return 1.0 - tail if t >= 0.0 else tail
}

// Quantile of t(df) (bisection on the CDF; |error| ~ 1e-14 relative)
student_t_ppf :: proc(p, df: f64) -> f64 {
	if p <= 0.0 do return -math.INF_F64
	if p >= 1.0 do return math.INF_F64
	if df <= 0.0 || math.is_nan(p) do return math.nan_f64()
	lo, hi := -1.0, 1.0
	for student_t_cdf(lo, df) > p && lo > -1e300 {lo *= 2.0}
	for student_t_cdf(hi, df) < p && hi < 1e300 {hi *= 2.0}
	for _ in 0 ..< 300 {
		mid := 0.5 * (lo + hi)
		if mid == lo || mid == hi do break
		if student_t_cdf(mid, df) < p {lo = mid} else {hi = mid}
	}
	return 0.5 * (lo + hi)
}
