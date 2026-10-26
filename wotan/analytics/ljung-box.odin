package analytics

import w "../core"
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

	// r_k needs n-k > 0 terms (k = n would divide 0 by 0)
	max_lag := min(max_lag, n - 1)

	// Compute ACF of residuals
	ac := acf(v, max_lag, allocator)

	Q = 0.0
	for k in 1 ..= max_lag {
		rk := ac[k]
		Q += (rk * rk) / f64(n - k)
	}
	Q *= f64(n) * (f64(n) + 2.0)

	// degrees of freedom (statsmodels acorr_ljungbox: lags - model_df, with a
	// NaN p-value when that is <= 0; we clamp to 1 instead)
	df = max_lag - dof_adj
	if df < 1 {
		df = 1
	}

	// p-value = P(chi2(df) > Q)
	p_value = chi2_sf(Q, f64(df))

	return
}

df_ljung_box :: proc(
	v: []f64,
	max_lag: int,
	dof_adj: int,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {

	Q, df, p := ljung_box(v, max_lag, dof_adj, allocator)

	out := w.dataframe_new()
	w.add_column(&out, w.column_from_floats("Q", []f64{Q}))
	w.add_column(&out, w.column_from_ints("df", []int{df}))
	w.add_column(&out, w.column_from_floats("p_value", []f64{p}))
	out.rows = 1
	return out
}


// ------------------------------------------------------------
// Regularized lower incomplete gamma P(a, x)
// Needed for chi-square CDF
// ------------------------------------------------------------
gamma_inc_lower_regularized :: proc(a, x: f64) -> f64 {
	P, _ := _gamma_inc_regularized(a, x)
	return P
}

// Chi-square survival function P(X > x), X ~ chi2(df). Returned from Q(a, x)
// directly (not 1 - P) so tiny p-values keep their relative precision.
chi2_sf :: proc(x, df: f64) -> f64 {
	_, Q := _gamma_inc_regularized(df / 2.0, x / 2.0)
	return Q
}

// P(a, x) and Q(a, x) = 1 - P(a, x): series for x < a+1, Lentz continued
// fraction otherwise (Numerical Recipes gser / gcf). The prefactor uses
// lgamma (gamma(a) overflows for a > 171, i.e. df > 342) and the iteration
// cap grows with a (both expansions need O(sqrt(a)) terms).
_gamma_inc_regularized :: proc(a, x: f64) -> (P: f64, Q: f64) {
	if x <= 0 || a <= 0 {
		return 0.0, 1.0
	}

	eps := 1e-15
	max_iter := 200 + int(10.0 * math.sqrt(a))
	lg, _ := math.lgamma(a)
	prefactor := math.exp(-x + a * math.ln(x) - lg)

	// Series expansion for x < a+1
	if x < a + 1.0 {
		sum := 1.0 / a
		term := sum
		ap := a

		for _ in 1 ..= max_iter {
			ap += 1.0
			term *= x / ap
			sum += term
			if abs(term) < abs(sum) * eps {
				break
			}
		}

		P = sum * prefactor
		return P, 1.0 - P
	}

	// Continued fraction for x >= a+1
	// Lentz’s algorithm
	FPMIN := 1e-300

	b := x + 1.0 - a
	c := 1.0 / FPMIN
	d := 1.0 / b
	h := d

	for i in 1 ..= max_iter {
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

	Q = h * prefactor
	return 1.0 - Q, Q
}
