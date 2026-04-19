package core

import "core:math"
import "core:mem"

// ------------------------------------------------------------
// Ljung–Box test for residual autocorrelation
// ------------------------------------------------------------
//
// Inputs:
//   v        : residuals (innovations)
//   max_lag  : number of lags to test (e.g. 10, 20)
//   dof_adj  : number of model parameters (p+q) to subtract from df
//
// Returns:
//   Q        : Ljung–Box Q statistic
//   df       : degrees of freedom
//   p_value  : chi-square tail probability
// ------------------------------------------------------------
ljung_box :: proc(
	v: []f64,
	max_lag: int,
	dof_adj: int, // usually p+q
	allocator: mem.Allocator = context.allocator,
) -> (
	Q: f64,
	df: int,
	p_value: f64,
) {

	n := len(v)
	if n <= 1 {
		return 0.0, 0, 1.0
	}

	// Compute ACF of residuals
	ac := acf(v, max_lag, allocator)

	Q = 0.0
	for k in 1 ..= max_lag {
		rk := ac[k]
		Q += (rk * rk) / f64(n - k)
	}
	Q *= f64(n) * (f64(n) + 2.0)

	// degrees of freedom
	df = max_lag - dof_adj
	if df < 1 {
		df = 1
	}

	// p-value = 1 - CDF_chi2(Q; df)
	// chi-square CDF via regularized gamma
	// CDF = gammainc(df/2, Q/2)
	x := Q / 2.0
	a := f64(df) / 2.0
	cdf := gamma_inc_lower_regularized(a, x) // lower incomplete gamma normalized
	p_value = 1.0 - cdf

	return
}

df_ljung_box :: proc(
	v: []f64,
	max_lag: int,
	dof_adj: int,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	Q, df, p := ljung_box(v, max_lag, dof_adj, allocator)

	out := dataframe_new()
	add_column(&out, column_from_floats("Q", []f64{Q}))
	add_column(&out, column_from_ints("df", []int{df}))
	add_column(&out, column_from_floats("p_value", []f64{p}))
	out.rows = 1
	return out
}


// ------------------------------------------------------------
// Regularized lower incomplete gamma P(a, x)
// Needed for chi-square CDF
// ------------------------------------------------------------
gamma_inc_lower_regularized :: proc(a, x: f64) -> f64 {
	if x < 0 || a <= 0 {
		return 0.0
	}

	// Series expansion for x < a+1
	if x < a + 1.0 {
		sum := 1.0 / a
		term := sum
		ap := a

		for i in 1 ..= 100 {
			ap += 1.0
			term *= x / ap
			sum += term
			if abs(term) < 1e-14 {
				break
			}
		}

		return sum * math.exp(-x + a * math.ln(x) - math.ln(math.gamma_f64(a)))
	}

	// Continued fraction for x >= a+1
	// Lentz’s algorithm
	eps := 1e-14
	FPMIN := 1e-300

	b := x + 1.0 - a
	c := 1.0 / FPMIN
	d := 1.0 / b
	h := d

	for i in 1 ..= 100 {
		an := -f64(i) * (f64(i) - a)
		b += 2.0
		d = an * d + b
		if abs(d) < FPMIN {d = FPMIN}

		c = b + an / c
		if abs(c) < FPMIN {c = FPMIN}

		d = 1.0 / d
		delta := d * c
		h *= delta

		if abs(delta - 1.0) < eps {
			break
		}
	}

	cf := h * math.exp(-x + a * math.ln(x) - math.ln(math.gamma_f64(a)))
	return 1.0 - cf
}
