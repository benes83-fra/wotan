package stats

import w "../core"
import "core:math"

// Tests for equal variances across samples: Levene (Brown-Forsythe) and
// Bartlett.

// The centre Levene's test measures deviations from.
Levene_Center :: enum {
	Median, // Brown-Forsythe; robust to non-normal data (the default)
	Mean, // Levene's original test
	Trimmed, // trimmed mean; the samples are trimmed as well
}

// Levene's test: do the samples have equal variances? An F statistic on the
// absolute deviations from each sample's centre, with k - 1 and N - k
// degrees of freedom (scipy's levene). Trimmed cuts `proportion_to_cut` of
// the values from each end of every sample. Needs two non-empty samples.
levene :: proc(
	samples: [][]f64,
	center := Levene_Center.Median,
	proportion_to_cut := 0.05,
) -> (
	res: Test_Result,
	ok: bool,
) {
	k := len(samples)
	if k < 2 || !(proportion_to_cut >= 0 && proportion_to_cut < 0.5) {return {}, false}
	w.TEMP_GUARD()
	sm := make([][]f64, k, context.temp_allocator)
	for s, i in samples {
		if len(s) == 0 || !_all_finite(s) {return {}, false}
		sm[i] = s
		if center == .Trimmed {
			srt := _sorted(s)
			cut := int(proportion_to_cut * f64(len(s)))
			sm[i] = srt[cut:len(s) - cut]
			if len(sm[i]) == 0 {return {}, false}
		}
	}
	ntot := 0
	zbari := make([]f64, k, context.temp_allocator)
	z := make([][]f64, k, context.temp_allocator)
	zbar := 0.0
	for s, i in sm {
		c: f64
		switch center {
		case .Median:
			c = _median_sorted(_sorted(s))
		case .Mean, .Trimmed:
			c, _ = _mean_var(s, 0)
		}
		z[i] = make([]f64, len(s), context.temp_allocator)
		for v, j in s {z[i][j] = abs(v - c)}
		zbari[i], _ = _mean_var(z[i], 0)
		zbar += zbari[i] * f64(len(s))
		ntot += len(s)
	}
	nt := f64(ntot)
	kf := f64(k)
	zbar /= nt
	numer, dvar: f64
	for s, i in sm {
		d := zbari[i] - zbar
		numer += f64(len(s)) * d * d
		for v in z[i] {dvar += (v - zbari[i]) * (v - zbari[i])}
	}
	res.statistic = (nt - kf) * numer / ((kf - 1) * dvar)
	res.df, res.df2 = kf - 1, nt - kf
	res.p_value = fisher_f_sf(Fisher_F{res.df, res.df2}, res.statistic)
	return res, true
}

// Bartlett's test: do the samples have equal variances? Sensitive to
// non-normality, where levene is the safer choice. Chi-squared with k - 1
// degrees of freedom (scipy's bartlett). Needs two samples of 2 or more.
bartlett :: proc(samples: [][]f64) -> (res: Test_Result, ok: bool) {
	k := len(samples)
	if k < 2 {return {}, false}
	nt, sp, slog, sinv: f64
	for s in samples {
		if len(s) < 2 || !_all_finite(s) {return {}, false}
		ni := f64(len(s))
		_, v := _mean_var(s)
		nt += ni
		sp += (ni - 1) * v
		slog += (ni - 1) * math.ln(v)
		sinv += 1 / (ni - 1)
	}
	kf := f64(k)
	sp /= nt - kf
	numer := (nt - kf) * math.ln(sp) - slog
	denom := 1 + 1 / (3 * (kf - 1)) * (sinv - 1 / (nt - kf))
	res.statistic = numer / denom
	res.df = kf - 1
	res.p_value = chi_squared_sf(Chi_Squared{res.df}, res.statistic)
	return res, true
}
