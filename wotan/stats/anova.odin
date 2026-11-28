package stats

import w "../core"
import "core:math"
import "core:mem"
import "core:slice"
import "core:strings"

// Analysis of variance: one-way ANOVA (classical, Welch, Brown-Forsythe),
// factorial ANOVA with Type I, II or III sums of squares, and Tukey's HSD
// comparisons of all pairs of groups.

// ---------------------------------------------------------------------------
// one-way ANOVA

// How anova_oneway treats the group variances.
Oneway_Variance :: enum {
	Equal, // classical F test, equal variances assumed
	Welch, // Welch's F test for unequal variances (R's oneway.test)
	Brown_Forsythe, // Brown-Forsythe F* test for means with unequal variances
}

Oneway_Result :: struct {
	statistic:     f64, // F
	p_value:       f64,
	df, df2:       f64, // numerator and denominator degrees of freedom
	ss_between:    f64, // sum of squares between groups
	ss_within:     f64, // sum of squares within groups
	eta_squared:   f64, // ss_between / ss_total
	omega_squared: f64, // (ss_between - (k - 1) MS_within) / (ss_total + MS_within)
}

// Do k groups have the same mean? Equal is scipy's f_oneway, Welch is
// R's oneway.test and statsmodels' anova_oneway(use_var = "unequal"),
// Brown_Forsythe the F* test with k - 1 and Satterthwaite degrees of freedom
// (statsmodels' df2 / pvalue2 for use_var = "bf"). The sums of squares and
// effect sizes are those of the classical ANOVA whatever the variant.
//
// Needs two groups. Equal needs more values than groups; Welch and
// Brown_Forsythe need two values in every group, and give NaN when a group
// has no spread.
anova_oneway :: proc(
	samples: [][]f64,
	variance := Oneway_Variance.Equal,
) -> (
	res: Oneway_Result,
	ok: bool,
) {
	k := len(samples)
	if k < 2 {return {}, false}
	ntot := 0
	for s in samples {
		if len(s) == 0 || !_all_finite(s) {return {}, false}
		if variance != .Equal && len(s) < 2 {return {}, false}
		ntot += len(s)
	}
	if ntot <= k {return {}, false}
	w.TEMP_GUARD()
	// all means relative to one value of the data, so a shared offset costs nothing
	c := samples[0][0]
	means := make([]f64, k, context.temp_allocator)
	vars := make([]f64, k, context.temp_allocator)
	nf := make([]f64, k, context.temp_allocator)
	ssw: _Sum
	for s, i in samples {
		m := _shifted_mean(s, c)
		ss: _Sum
		for v in s {
			d := (v - c) - m
			_sum_add(&ss, d * d)
		}
		means[i], nf[i] = m, f64(len(s))
		_sum_add(&ssw, _sum(ss))
		vars[i] = _sum(ss) / (nf[i] - 1) if len(s) > 1 else 0
	}
	nt, kf := f64(ntot), f64(k)
	grand := 0.0
	for m, i in means {grand += nf[i] * m}
	grand /= nt
	r := 0.0
	for m, i in means {r += nf[i] * (m - grand)}
	grand += r / nt
	ssb: _Sum
	for m, i in means {_sum_add(&ssb, nf[i] * (m - grand) * (m - grand))}
	res.ss_between, res.ss_within = _sum(ssb), _sum(ssw)
	msw := res.ss_within / (nt - kf)
	sst := res.ss_between + res.ss_within
	res.eta_squared = res.ss_between / sst
	res.omega_squared = (res.ss_between - (kf - 1) * msw) / (sst + msw)

	switch variance {
	case .Equal:
		res.statistic = res.ss_between / (kf - 1) / msw
		res.df, res.df2 = kf - 1, nt - kf
	case .Welch:
		wts := make([]f64, k, context.temp_allocator)
		wsum := 0.0
		for i in 0 ..< k {
			wts[i] = nf[i] / vars[i]
			wsum += wts[i]
		}
		mw := 0.0
		for i in 0 ..< k {mw += wts[i] * means[i]}
		mw /= wsum
		a, tmp := 0.0, 0.0
		for i in 0 ..< k {
			a += wts[i] * (means[i] - mw) * (means[i] - mw)
			u := 1 - wts[i] / wsum
			tmp += u * u / (nf[i] - 1)
		}
		tmp /= kf * kf - 1
		res.statistic = a / (kf - 1) / (1 + 2 * (kf - 2) * tmp)
		res.df, res.df2 = kf - 1, 1 / (3 * tmp)
	case .Brown_Forsythe:
		den, dd := 0.0, 0.0
		for i in 0 ..< k {
			u := (1 - nf[i] / nt) * vars[i]
			den += u
			dd += u * u / (nf[i] - 1)
		}
		res.statistic = res.ss_between / den
		res.df, res.df2 = kf - 1, den * den / dd
	}
	res.p_value = fisher_f_sf(Fisher_F{res.df, res.df2}, res.statistic)
	if math.is_nan(res.statistic) {res.p_value = math.nan_f64()}
	return res, true
}

// ---------------------------------------------------------------------------
// factorial ANOVA

// Which sums of squares anova reports for each term.
Anova_Type :: enum {
	I, // sequential: each term after the ones listed before it
	II, // each term after all terms that do not contain it (R's car::Anova default)
	III, // each term after all other terms, with sum-to-zero coding
}

// One line of an ANOVA table. The last line of a table is the residual,
// whose f, p_value and effect sizes are NaN.
Anova_Term :: struct {
	name:                string, // factor name, "a:b" for an interaction, "Residual"
	df:                  f64,
	sum_sq:              f64,
	mean_sq:             f64,
	f:                   f64,
	p_value:             f64,
	eta_squared:         f64, // sum_sq / total sum of squares
	partial_eta_squared: f64, // sum_sq / (sum_sq + residual sum_sq)
	omega_squared:       f64, // partial ω²: (sum_sq - df MSE) / (sum_sq + (n - df) MSE)
}

Anova_Table :: struct {
	terms:  []Anova_Term,
	n:      int, // observations used
	ss_tot: f64, // total sum of squares about the mean
}

// Factorial ANOVA of y on one or more factors. factors[f][i] is the level
// of observation i in factor f: any non-negative integers, one per level
// (data.odin's helpers and anova_from_df produce them). With interactions,
// the model holds every interaction of the factors (a, b, a:b for two),
// otherwise main effects only. Terms come in order of size, then in the
// order of the factors.
//
// The sums of squares come from a QR decomposition of the design in
// sum-to-zero coding, not from differences of residual sums, so a small
// effect beside a large residual keeps its digits. Type III uses the sum
// coding statsmodels needs `C(a, Sum)` for; Types I and II do not depend on
// the coding. Unbalanced designs and empty cells are fine; terms whose
// columns are aliased lose those degrees of freedom, as in R.
//
// Matches statsmodels' anova_lm(typ = 1, 2, 3) and R's anova / car::Anova.
// Needs a finite y, every factor with two or more levels, and more
// observations than model parameters.
anova :: proc(
	y: []f64,
	factors: [][]int,
	names: []string,
	interactions := true,
	type := Anova_Type.II,
	allocator := context.allocator,
) -> (
	table: Anova_Table,
	ok: bool,
) {
	n := len(y)
	nf := len(factors)
	if nf == 0 || nf > 16 || len(names) != nf || !_all_finite(y) {return {}, false}
	for f in factors {
		if len(f) != n {return {}, false}
		for v in f {if v < 0 {return {}, false}}
	}
	w.TEMP_GUARD(ignore = _is_temp(allocator))

	// dense levels and sum-coded columns of every main effect
	main_cols := make([][][]f64, nf, context.temp_allocator)
	for f, j in factors {
		u := slice.clone(f, context.temp_allocator)
		slice.sort(u)
		u = slice.unique(u)
		if len(u) < 2 {return {}, false}
		l := len(u)
		cols := make([][]f64, l - 1, context.temp_allocator)
		for &col in cols {col = make([]f64, n, context.temp_allocator)}
		for v, i in f {
			lv, _ := slice.binary_search(u, v)
			if lv == l - 1 {
				for col in cols {col[i] = -1}
			} else {
				cols[lv][i] = 1
			}
		}
		main_cols[j] = cols
	}

	// terms as bit masks over the factors, ordered by size then factor order
	terms := make([dynamic]u32, 0, context.temp_allocator)
	if interactions {
		for size in 1 ..= nf {
			_anova_masks(&terms, nf, size, 0, 0, 0)
		}
	} else {
		for j in 0 ..< nf {append(&terms, u32(1) << uint(j))}
	}
	nt := len(terms)
	term_cols := make([][][]f64, nt, context.temp_allocator)
	for t, ti in terms {term_cols[ti] = _anova_term_cols(t, main_cols, n)}

	// y relative to its mean (the intercept absorbs the shift exactly)
	cn := _center(y)
	yc := make([]f64, n, context.temp_allocator)
	sst: _Sum
	for v, i in y {
		yc[i] = _dev(cn, v)
		_sum_add(&sst, yc[i] * yc[i])
	}
	ones := make([]f64, n, context.temp_allocator)
	for &v in ones {v = 1}

	// full model: residual, and the Type I sums along the way
	order := make([]int, nt, context.temp_allocator)
	for i in 0 ..< nt {order[i] = i}
	seq_ss := make([]f64, nt, context.temp_allocator)
	seq_df := make([]int, nt, context.temp_allocator)
	rss, rank := _anova_qr(ones, term_cols, order, -1, yc, seq_ss, seq_df)
	if rank >= n {return {}, false}

	ss := make([]f64, nt, context.temp_allocator)
	dfs := make([]int, nt, context.temp_allocator)
	switch type {
	case .I:
		copy(ss, seq_ss)
		copy(dfs, seq_df)
	case .II, .III:
		for t, ti in terms {
			sel := make([dynamic]int, 0, context.temp_allocator)
			for u, ui in terms {
				if ui == ti {continue}
				if type == .II && u & t == t {continue} 	// u contains t
				append(&sel, ui)
			}
			append(&sel, ti)
			_, _ = _anova_qr(ones, term_cols, sel[:], ti, yc, ss, dfs)
		}
	}

	dfr := f64(n - rank)
	mse := rss / dfr
	table.n = n
	table.ss_tot = _sum(sst)
	table.terms = make([]Anova_Term, nt + 1, allocator)
	for t, ti in terms {
		e := &table.terms[ti]
		e.name = _anova_term_name(t, names, allocator)
		e.df = f64(dfs[ti])
		e.sum_sq = ss[ti]
		e.mean_sq = ss[ti] / e.df
		e.f = e.mean_sq / mse
		e.p_value = fisher_f_sf(Fisher_F{e.df, dfr}, e.f)
		e.eta_squared = ss[ti] / table.ss_tot
		e.partial_eta_squared = ss[ti] / (ss[ti] + rss)
		e.omega_squared = (ss[ti] - e.df * mse) / (ss[ti] + (f64(n) - e.df) * mse)
		if dfs[ti] == 0 {
			e.mean_sq, e.f, e.p_value = math.nan_f64(), math.nan_f64(), math.nan_f64()
		}
	}
	nan := math.nan_f64()
	table.terms[nt] = Anova_Term {
		name                = strings.clone("Residual", allocator),
		df                  = dfr,
		sum_sq              = rss,
		mean_sq             = mse,
		f                   = nan,
		p_value             = nan,
		eta_squared         = nan,
		partial_eta_squared = nan,
		omega_squared       = nan,
	}
	return table, true
}

// anova on columns of a DataFrame: `response` numeric, each factor a String,
// Int or Bool column. Rows with a null in any of them are left out.
//
//     t, _ := stats.anova_from_df(&df, "yield", {"fertilizer", "field"})
//     defer stats.anova_table_destroy(&t)
anova_from_df :: proc(
	df: ^w.DataFrame,
	response: string,
	factors: []string,
	interactions := true,
	type := Anova_Type.II,
	allocator := context.allocator,
) -> (
	table: Anova_Table,
	ok: bool,
) {
	w.TEMP_GUARD(ignore = _is_temp(allocator))
	cv := _numeric_column(df, response) or_return
	nf := len(factors)
	codes := make([][]int, nf, context.temp_allocator)
	for name, j in factors {
		cj, labels := _group_codes(df, name, context.temp_allocator) or_return
		codes[j] = cj
		_ = labels
	}
	y := make([dynamic]f64, 0, context.temp_allocator)
	fl := make([][dynamic]int, nf, context.temp_allocator)
	for &l in fl {l = make([dynamic]int, 0, context.temp_allocator)}
	rows: for i in 0 ..< cv.len {
		v, null := w.column_at_numeric(cv, i)
		if null {continue}
		for j in 0 ..< nf {if codes[j][i] < 0 {continue rows}}
		append(&y, v)
		for j in 0 ..< nf {append(&fl[j], codes[j][i])}
	}
	fs := make([][]int, nf, context.temp_allocator)
	for l, j in fl {fs[j] = l[:]}
	return anova(y[:], fs, factors, interactions, type, allocator)
}

anova_table_destroy :: proc(t: ^Anova_Table, allocator := context.allocator) {
	for e in t.terms {delete(e.name, allocator)}
	delete(t.terms, allocator)
	t^ = {}
}

// all masks with `size` bits among the first nf, in lexicographic order
@(private)
_anova_masks :: proc(out: ^[dynamic]u32, nf, size, start: int, mask: u32, have: int) {
	if have == size {
		append(out, mask)
		return
	}
	for j in start ..< nf {
		_anova_masks(out, nf, size, j + 1, mask | u32(1) << uint(j), have + 1)
	}
}

// columns of a term: products of one column of each of its factors
@(private)
_anova_term_cols :: proc(t: u32, main_cols: [][][]f64, n: int) -> [][]f64 {
	cols := make([dynamic][]f64, 0, context.temp_allocator)
	first := true
	for j in 0 ..< len(main_cols) {
		if t & (u32(1) << uint(j)) == 0 {continue}
		if first {
			for c in main_cols[j] {append(&cols, c)}
			first = false
			continue
		}
		next := make([dynamic][]f64, 0, context.temp_allocator)
		for a in cols {
			for b in main_cols[j] {
				p := make([]f64, n, context.temp_allocator)
				for i in 0 ..< n {p[i] = a[i] * b[i]}
				append(&next, p)
			}
		}
		cols = next
	}
	return cols[:]
}

@(private)
_anova_term_name :: proc(t: u32, names: []string, allocator: mem.Allocator) -> string {
	b := strings.builder_make(context.temp_allocator)
	for j in 0 ..< len(names) {
		if t & (u32(1) << uint(j)) == 0 {continue}
		if strings.builder_len(b) > 0 {strings.write_byte(&b, ':')}
		strings.write_string(&b, names[j])
	}
	return strings.clone(strings.to_string(b), allocator)
}

// Householder QR of [1, columns of terms in `sel` order] applied to y.
// Columns that depend on earlier ones are skipped. The sum of squares of
// the rotated y at the pivots of each term goes to ss[term], their number to
// df[term] (only for `target` when target >= 0); returns the residual sum
// of squares and the rank.
@(private)
_anova_qr :: proc(
	ones: []f64,
	term_cols: [][][]f64,
	sel: []int,
	target: int,
	y: []f64,
	ss: []f64,
	df: []int,
) -> (
	rss: f64,
	rank: int,
) {
	n := len(y)
	cols := make([dynamic][]f64, 0, context.temp_allocator)
	owner := make([dynamic]int, 0, context.temp_allocator)
	append(&cols, slice.clone(ones, context.temp_allocator))
	append(&owner, -1)
	for ti in sel {
		for c in term_cols[ti] {
			append(&cols, slice.clone(c, context.temp_allocator))
			append(&owner, ti)
		}
	}
	z := slice.clone(y, context.temp_allocator)
	for ti in sel {
		if target < 0 || ti == target {ss[ti], df[ti] = 0, 0}
	}
	acc := make([]_Sum, len(term_cols), context.temp_allocator)
	r := 0
	for j in 0 ..< len(cols) {
		if r >= n {break}
		a := cols[j]
		orig := 0.0
		for v in a {orig += v * v}
		nrm := 0.0
		for i in r ..< n {nrm += a[i] * a[i]}
		if nrm <= 1e-20 * orig || nrm == 0 {continue} 	// aliased with earlier columns
		// reflector v = a[r:] + sign(a[r]) |a[r:]| e1, H = I - 2 v vᵀ / vᵀv
		alpha := -math.copy_sign(math.sqrt(nrm), a[r])
		a[r] -= alpha
		vtv := 0.0
		for i in r ..< n {vtv += a[i] * a[i]}
		for jj in j + 1 ..< len(cols) {
			b := cols[jj]
			d := 0.0
			for i in r ..< n {d += a[i] * b[i]}
			f := 2 * d / vtv
			for i in r ..< n {b[i] -= f * a[i]}
		}
		d := 0.0
		for i in r ..< n {d += a[i] * z[i]}
		f := 2 * d / vtv
		for i in r ..< n {z[i] -= f * a[i]}
		o := owner[j]
		if o >= 0 && (target < 0 || o == target) {
			_sum_add(&acc[o], z[r] * z[r])
			df[o] += 1
		}
		r += 1
	}
	for ti in sel {
		if target < 0 || ti == target {ss[ti] = _sum(acc[ti])}
	}
	res: _Sum
	for i in r ..< n {_sum_add(&res, z[i] * z[i])}
	return _sum(res), r
}

// ---------------------------------------------------------------------------
// Tukey's HSD

// One comparison of Tukey's HSD: group i against group j (i < j).
Tukey_Pair :: struct {
	i, j:      int,
	diff:      f64, // mean_i - mean_j
	std_error: f64, // sqrt(MSE / 2 (1/n_i + 1/n_j)), the scale of the studentized range
	statistic: f64, // |diff| / std_error, studentized range
	p_value:   f64, // adjusted for all k (k - 1) / 2 comparisons
	ci_low:    f64,
	ci_high:   f64, // simultaneous confidence interval of the difference
}

Tukey_Result :: struct {
	pairs:  []Tukey_Pair, // (0,1), (0,2), ..., (1,2), ...
	df:     f64, // N - k
	mse:    f64, // pooled within-group variance
	q_crit: f64, // studentized range quantile used for the intervals
}

// Tukey's honestly significant difference test (Tukey-Kramer for unequal
// group sizes): compares the means of every pair of groups with p-values
// and simultaneous confidence intervals that hold for all pairs together
// (scipy's tukey_hsd, R's TukeyHSD). Needs two groups, more values than
// groups, and a confidence level in (0, 1).
tukey_hsd :: proc(
	samples: [][]f64,
	confidence := 0.95,
	allocator := context.allocator,
) -> (
	res: Tukey_Result,
	ok: bool,
) {
	k := len(samples)
	if k < 2 || !_valid_confidence(confidence) {return {}, false}
	ntot := 0
	for s in samples {
		if len(s) == 0 || !_all_finite(s) {return {}, false}
		ntot += len(s)
	}
	if ntot <= k {return {}, false}
	w.TEMP_GUARD(ignore = _is_temp(allocator))
	c := samples[0][0]
	means := make([]f64, k, context.temp_allocator)
	ssw: _Sum
	for s, i in samples {
		means[i] = _shifted_mean(s, c)
		for v in s {
			d := (v - c) - means[i]
			_sum_add(&ssw, d * d)
		}
	}
	res.df = f64(ntot - k)
	res.mse = _sum(ssw) / res.df
	dist := Studentized_Range{f64(k), res.df}
	res.q_crit = studentized_range_isf(dist, 1 - confidence)
	res.pairs = make([]Tukey_Pair, k * (k - 1) / 2, allocator)
	p := 0
	for i in 0 ..< k {
		for j in i + 1 ..< k {
			e := &res.pairs[p]
			p += 1
			e.i, e.j = i, j
			e.diff = means[i] - means[j]
			e.std_error = math.sqrt(
				0.5 * res.mse * (1 / f64(len(samples[i])) + 1 / f64(len(samples[j]))),
			)
			e.statistic = abs(e.diff) / e.std_error
			e.p_value = studentized_range_sf(dist, e.statistic)
			e.ci_low = e.diff - res.q_crit * e.std_error
			e.ci_high = e.diff + res.q_crit * e.std_error
		}
	}
	return res, true
}

tukey_destroy :: proc(r: ^Tukey_Result, allocator := context.allocator) {
	delete(r.pairs, allocator)
	r^ = {}
}
