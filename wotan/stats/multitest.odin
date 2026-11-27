package stats

import w "../core"
import "core:math"

// Adjusting p-values for multiple testing.

// Adjustment methods of p_adjust. The first six control the family-wise
// error rate (the chance of any false rejection), the last two the false
// discovery rate (the expected share of false rejections).
P_Adjust :: enum {
	Bonferroni, // min(1, m p)
	Sidak, // 1 - (1 - p)^m; exact for independent tests
	Holm, // step-down Bonferroni; never worse than Bonferroni
	Holm_Sidak, // step-down Sidak
	Hochberg, // step-up; needs independent or positively dependent tests
	Hommel, // more powerful than Hochberg under the same conditions
	BH, // Benjamini-Hochberg false discovery rate
	BY, // Benjamini-Yekutieli; any dependence
}

// Adjusted p-values for the m p-values p, in their original order: reject
// hypothesis i at level alpha when the adjusted value is at most alpha. As
// statsmodels' multipletests (R's p.adjust). p-values must lie in [0, 1].
p_adjust :: proc(
	p: []f64,
	method := P_Adjust.Holm,
	allocator := context.allocator,
) -> (
	adjusted: []f64,
	ok: bool,
) {
	m := len(p)
	for v in p {if !(v >= 0 && v <= 1) {return nil, false}}
	w.TEMP_GUARD(ignore = _is_temp(allocator))
	idx := _order(p)
	s := make([]f64, m, context.temp_allocator) // sorted p
	for i in 0 ..< m {s[i] = p[idx[i]]}
	a := make([]f64, m, context.temp_allocator) // adjusted, sorted order
	mf := f64(m)
	switch method {
	case .Bonferroni:
		for v, i in s {a[i] = v * mf}
	case .Sidak:
		for v, i in s {a[i] = -expm1(mf * math.log1p(-v))}
	case .Holm:
		for v, i in s {a[i] = v * f64(m - i)}
		_cummax(a)
	case .Holm_Sidak:
		for v, i in s {a[i] = -expm1(f64(m - i) * math.log1p(-v))}
		_cummax(a)
	case .Hochberg:
		for v, i in s {a[i] = v * f64(m - i)}
		_rev_cummin(a)
	case .Hommel:
		copy(a, s)
		for k := m; k > 1; k -= 1 {
			// cim = min over the k largest of k p_(m-k+i) / i
			cim := math.inf_f64(1)
			for i in 1 ..= k {cim = min(cim, f64(k) * s[m - k + i - 1] / f64(i))}
			for j in m - k ..< m {a[j] = max(a[j], cim)}
			for j in 0 ..< m - k {a[j] = max(a[j], min(f64(k) * s[j], cim))}
		}
	case .BH, .BY:
		c := 1.0
		if method == .BY {
			c = 0
			for i in 1 ..= m {c += 1 / f64(i)}
		}
		for v, i in s {a[i] = v * mf * c / f64(i + 1)}
		_rev_cummin(a)
	}
	adjusted = make([]f64, m, allocator)
	for i in 0 ..< m {adjusted[idx[i]] = min(a[i], 1)}
	return adjusted, true
}

@(private)
_cummax :: proc(a: []f64) {
	for i in 1 ..< len(a) {a[i] = max(a[i], a[i - 1])}
}

@(private)
_rev_cummin :: proc(a: []f64) {
	for i := len(a) - 2; i >= 0; i -= 1 {a[i] = min(a[i], a[i + 1])}
}
