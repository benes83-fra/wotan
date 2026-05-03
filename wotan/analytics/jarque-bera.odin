package analytics

import w "../core"
import "core:math"
import "core:mem"

// ------------------------------------------------------------
// Jarque–Bera normality test
// ------------------------------------------------------------
//
// Inputs:
//   v : residuals
//
// Returns:
//   JB       : Jarque–Bera statistic
//   p_value  : chi-square tail probability with df=2
// ------------------------------------------------------------
jarque_bera :: proc(
	v: []f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	JB: f64,
	p_value: f64,
) {
	n := len(v)
	if n < 3 {
		return 0.0, 1.0
	}

	// mean
	mean := 0.0
	for x in v {
		mean += x
	}
	mean /= f64(n)

	// compute central moments
	m2 := 0.0
	m3 := 0.0
	m4 := 0.0

	for x in v {
		d := x - mean
		d2 := d * d
		m2 += d2
		m3 += d2 * d
		m4 += d2 * d2
	}

	m2 /= f64(n)
	m3 /= f64(n)
	m4 /= f64(n)

	if m2 == 0 {
		return 0.0, 1.0
	}

	// sample skewness and kurtosis
	S := m3 / math.pow(m2, 1.5)
	K := m4 / (m2 * m2)

	// Jarque–Bera statistic
	JB = (f64(n) / 6.0) * (S * S + 0.25 * (K - 3.0) * (K - 3.0))

	// p-value = 1 - CDF_chi2(JB; df=2)
	a := 1.0 // df/2 = 2/2
	x := JB / 2.0
	cdf := gamma_inc_lower_regularized(a, x)
	p_value = 1.0 - cdf

	return
}
df_jarque_bera :: proc(v: []f64, allocator: mem.Allocator = context.allocator) -> w.DataFrame {

	JB, p := jarque_bera(v, allocator)

	out := w.dataframe_new()
	w.add_column(&out, w.column_from_floats("JB", []f64{JB}))
	w.add_column(&out, w.column_from_floats("p_value", []f64{p}))
	out.rows = 1
	return out
}
