package ML

import st "../../stats"
import "core:math"

// Student-t and F helpers for regression inference (p-values, CIs), from the
// stats package.

// Regularized incomplete beta I_x(a, b)
_reg_inc_beta :: proc(a, b, x: f64) -> f64 {
	if x <= 0.0 do return 0.0
	if x >= 1.0 do return 1.0
	return st.beta_inc(a, b, x)
}

// Two-sided tail probability P(|T| > |t|) for T ~ t(df)
student_t_sf2 :: proc(t, df: f64) -> f64 {
	if math.is_nan(t) || df <= 0.0 do return math.nan_f64()
	if math.is_inf(t) do return 0.0
	return 2 * st.student_t_sf(st.Student_T{df}, abs(t))
}

// CDF of t(df)
student_t_cdf :: proc(t, df: f64) -> f64 {
	if math.is_nan(t) || df <= 0.0 do return math.nan_f64()
	return st.student_t_cdf(st.Student_T{df}, t)
}

// Quantile of t(df)
student_t_ppf :: proc(p, df: f64) -> f64 {
	if p <= 0.0 do return -math.INF_F64
	if p >= 1.0 do return math.INF_F64
	if df <= 0.0 || math.is_nan(p) do return math.nan_f64()
	return st.student_t_ppf(st.Student_T{df}, p)
}

// F distribution survival function P(F > x), F ~ F(d1, d2)
f_dist_sf :: proc(x, d1, d2: f64) -> f64 {
	if math.is_nan(x) || d1 <= 0 || d2 <= 0 do return math.nan_f64()
	if x <= 0 do return 1.0
	if math.is_inf(x) do return 0.0
	return st.fisher_f_sf(st.Fisher_F{d1, d2}, x)
}
