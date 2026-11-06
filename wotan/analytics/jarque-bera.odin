package analytics

import w "../core"
import "core:math"
import "core:mem"

// Mean, population variance (divisor n), skewness m3 / m2^1.5 and kurtosis m4 / m2^2
// (not excess: 3 for a normal), as scipy.stats.skew / kurtosis(fisher=False) with
// bias=True. Skewness and kurtosis are 0 and 3 when the variance is 0.
sample_moments :: proc(v: []f64) -> (mean, variance, skewness, kurtosis: f64) {
	n := f64(len(v))
	if len(v) == 0 {return 0, 0, 0, 3}
	for x in v {mean += x}
	mean /= n
	m3, m4: f64
	for x in v {
		d := x - mean
		d2 := d * d
		variance += d2
		m3 += d2 * d
		m4 += d2 * d2
	}
	variance /= n
	m3 /= n
	m4 /= n
	if variance == 0 {return mean, 0, 0, 3}
	return mean, variance, m3 / math.pow(variance, 1.5), m4 / (variance * variance)
}

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

	_, m2, S, K := sample_moments(v)
	if m2 == 0 {
		return 0.0, 1.0
	}

	// Jarque–Bera statistic
	JB = (f64(n) / 6.0) * (S * S + 0.25 * (K - 3.0) * (K - 3.0))

	// p-value = P(chi2(2) > JB)
	p_value = chi2_sf(JB, 2.0)

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
