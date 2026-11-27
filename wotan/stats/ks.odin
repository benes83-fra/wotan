package stats

import w "../core"
import "core:math"

// Kolmogorov-Smirnov tests, with the exact distributions of the statistics
// (scipy's kstwo / ksone / kstwobign and ks_2samp methods).

// One-sample Kolmogorov-Smirnov test: does x come from distribution `dist`
// (any distribution of this package, its parameters fixed in advance)?
// The statistic is D = sup |F_n - F|; for Less / Greater the one-sided
// D- = sup (F - F_n) and D+ = sup (F_n - F), whose large values support
// x tending to be larger, resp. smaller, than the distribution. Auto and
// Exact use the exact distribution of D, Asymptotic Kolmogorov's limit.
ks_test_one_sample :: proc(
	x: []f64,
	dist: $D,
	alternative := Alternative.Two_Sided,
	method := Method.Auto,
) -> (
	res: Test_Result,
	ok: bool,
) {
	if len(x) == 0 || !_all_finite(x) {return {}, false}
	w.TEMP_GUARD()
	s := _sorted(x)
	f := make([]f64, len(s), context.temp_allocator)
	for v, i in s {f[i] = cdf(dist, v)}
	return _ks_from_cdf(f, alternative, method), true
}

// ks_test_one_sample against a cumulative distribution function of your own.
ks_test_one_sample_fn :: proc(
	x: []f64,
	cdf_fn: proc(x: f64) -> f64,
	alternative := Alternative.Two_Sided,
	method := Method.Auto,
) -> (
	res: Test_Result,
	ok: bool,
) {
	if len(x) == 0 || !_all_finite(x) {return {}, false}
	w.TEMP_GUARD()
	s := _sorted(x)
	f := make([]f64, len(s), context.temp_allocator)
	for v, i in s {f[i] = cdf_fn(v)}
	return _ks_from_cdf(f, alternative, method), true
}

// statistic and p-value from the distribution function at the sorted sample
@(private)
_ks_from_cdf :: proc(f: []f64, alternative: Alternative, method: Method) -> (res: Test_Result) {
	n := len(f)
	nf := f64(n)
	dplus, dminus := math.inf_f64(-1), math.inf_f64(-1)
	for v, i in f {
		dplus = max(dplus, f64(i + 1) / nf - v)
		dminus = max(dminus, v - f64(i) / nf)
	}
	switch alternative {
	case .Greater:
		res.statistic = dplus
		res.p_value = clamp(_smirnov_sf(n, dplus), 0, 1)
	case .Less:
		res.statistic = dminus
		res.p_value = clamp(_smirnov_sf(n, dminus), 0, 1)
	case .Two_Sided:
		d := max(dplus, dminus)
		res.statistic = d
		if method == .Asymptotic {
			res.p_value = _kolmogorov_sf(d * math.sqrt(nf))
		} else {
			res.p_value = _kolmogn(n, d, false)
		}
		res.p_value = clamp(res.p_value, 0, 1)
	}
	return
}

// Two-sample Kolmogorov-Smirnov test: do x and y come from the same
// continuous distribution? D = sup |F_x - F_y|; for Greater D+ = sup
// (F_x - F_y), large when x tends to be smaller than y, for Less
// D- = sup (F_y - F_x). Auto computes the exact p-value when both samples
// have at most 10000 values, otherwise the asymptotic one (scipy's
// ks_2samp); exact p-values report D rounded to the lattice of the sample
// sizes.
ks_test_two_sample :: proc(
	x, y: []f64,
	alternative := Alternative.Two_Sided,
	method := Method.Auto,
) -> (
	res: Test_Result,
	ok: bool,
) {
	n1, n2 := len(x), len(y)
	if n1 == 0 || n2 == 0 || !_all_finite(x) || !_all_finite(y) {return {}, false}
	w.TEMP_GUARD()
	a := _sorted(x)
	b := _sorted(y)
	maxs, mins := math.inf_f64(-1), math.inf_f64(1)
	for data in ([2][]f64{a, b}) {
		for v in data {
			c1 := f64(_count_le(a, v)) / f64(n1)
			c2 := f64(_count_le(b, v)) / f64(n2)
			maxs = max(maxs, c1 - c2)
			mins = min(mins, c1 - c2)
		}
	}
	mins = clamp(-mins, 0, 1)
	d := maxs
	if alternative == .Less || (alternative == .Two_Sided && mins > maxs) {d = mins}
	m := method
	if m == .Auto {m = .Exact if max(n1, n2) <= 10000 else .Asymptotic}
	p := math.nan_f64()
	if m == .Exact {
		g := _gcd(n1, n2)
		lcm := f64(n1 / g) * f64(n2)
		h := math.round(d * lcm)
		d = h / lcm
		if h == 0 {
			p = 1
		} else if alternative == .Two_Sided {
			p =
				_ks2_outside_square(n1, h) if n1 == n2 else _ks2_outer_inside(n1, n2, g, int(h), true)
		} else if n1 == n2 {
			p = 1
			for j in 0 ..< int(h) {p *= f64(n1 - j) / f64(n1 + j + 1)}
		} else {
			p = _ks2_outer_inside(n1, n2, g, int(h), false)
		}
		if !(p >= 0 && p <= 1) {m = .Asymptotic}
	}
	if m == .Asymptotic {
		mm, nn := f64(max(n1, n2)), f64(min(n1, n2))
		en := mm * nn / (mm + nn)
		if alternative == .Two_Sided {
			p = _kolmogn(int(_round_half_even(en)), d, false) // numpy's round
		} else {
			z := math.sqrt(en) * d
			p = math.exp(-2 * z * z - 2 * z * (mm + 2 * nn) / math.sqrt(mm * nn * (mm + nn)) / 3)
		}
	}
	res.statistic = d
	res.p_value = clamp(p, 0, 1)
	return res, true
}

// number of values <= v in the sorted slice s
@(private)
_count_le :: proc(s: []f64, v: f64) -> int {
	lo, hi := 0, len(s)
	for lo < hi {
		mid := (lo + hi) / 2
		if s[mid] <= v {lo = mid + 1} else {hi = mid}
	}
	return lo
}

@(private)
_gcd :: proc(a, b: int) -> int {
	a, b := a, b
	for b != 0 {a, b = b, a % b}
	return a
}

// P(D >= h / n) for two samples of size n (reflection formula)
@(private)
_ks2_outside_square :: proc(n: int, h: f64) -> f64 {
	hi := int(h)
	p := 0.0
	for k := n / hi; k >= 0; k -= 1 {
		p1 := 1.0
		for j in 0 ..< hi {
			p1 = f64(n - k * hi - j) * p1 / f64(n + k * hi + j + 1)
		}
		p = p1 * (1 - p)
	}
	return 2 * p
}

// Probability that a random lattice path from (0, 0) to (m, n) leaves the
// band |i / m - j / n| < h / lcm (two_sided), or crosses i / m - j / n >= h /
// lcm (one-sided), computed directly as in Viehmann (2021): A holds, along a
// column, the chance that a path to (i, j) has left already. scipy's
// _compute_outer_prob_inside_method, with the one-sided band added.
@(private)
_ks2_outer_inside :: proc(m_in, n_in, g, h: int, two_sided: bool) -> f64 {
	m, n := max(m_in, n_in), min(m_in, n_in)
	mg, ng := m / g, n / g
	minj, maxj := 0, min(_ceil_div(h, mg), n + 1)
	if !two_sided {maxj = n + 1}
	curlen := maxj - minj
	a := make([]f64, n + 2, context.temp_allocator)
	for &v in a {v = 1}
	for j in minj ..< maxj {a[j] = 0}
	for i in 1 ..= m {
		lastminj, lastlen := minj, curlen
		minj = max(_floor_div(ng * i - h, mg) + 1, 0)
		minj = min(minj, n)
		maxj = min(_ceil_div(ng * i + h, mg), n + 1) if two_sided else n + 1
		if maxj <= minj {return 1}
		val := 0.0 if minj == 0 else 1.0
		for jj in 0 ..< maxj - minj {
			j := jj + minj
			val = (a[jj + minj - lastminj] * f64(i) + val * f64(j)) / f64(i + j)
			a[jj] = val
		}
		curlen = maxj - minj
		if lastlen > curlen {
			for t in maxj - minj ..< maxj - minj + (lastlen - curlen) {a[t] = 1}
		}
	}
	return a[maxj - minj - 1]
}

@(private)
_floor_div :: proc(a, b: int) -> int {
	q := a / b
	if (a % b != 0) && ((a < 0) != (b < 0)) {q -= 1}
	return q
}

@(private)
_ceil_div :: proc(a, b: int) -> int {return -_floor_div(-a, b)}

// ---------------------------------------------------------------------------
// Distributions of the statistics

// Kolmogorov's limiting distribution: P(√n D > y) for large n
@(private)
_kolmogorov_sf :: proc(y: f64) -> f64 {
	if y <= 0 {return 1}
	if y <= 0.82 {
		// 1 - √(2π) / y Σ exp(-(2k - 1)² π² / (8 y²))
		s := 0.0
		for k in 1 ..= 20 {
			t := math.exp(-f64((2 * k - 1) * (2 * k - 1)) * math.PI * math.PI / (8 * y * y))
			s += t
			if t < 1e-17 * s {break}
		}
		return 1 - math.sqrt(2 * math.PI) / y * s
	}
	// 2 Σ (-1)^(k-1) exp(-2 k² y²)
	s := 0.0
	sign := 1.0
	for k in 1 ..= 100 {
		t := math.exp(-2 * f64(k * k) * y * y)
		s += sign * t
		sign = -sign
		if t < 1e-17 * abs(s) {break}
	}
	return 2 * s
}

// One-sided exact: P(D+_n >= d) by the Birnbaum-Tingey formula
// d Σ_{j <= n (1 - d)} C(n, j) (d + j/n)^(j-1) (1 - d - j/n)^(n-j), each term
// a binomial probability times d / (d + j/n), all positive. Large n sum
// only the terms around the largest one.
@(private)
_smirnov_sf :: proc(n: int, d: f64) -> f64 {
	if math.is_nan(d) {return d}
	if d <= 0 {return 1}
	if d >= 1 {return 0}
	if n == 1 {return 1 - d}
	nf := f64(n)
	jmax := int(math.floor(nf * (1 - d)))
	jmax = min(jmax, n - 1)
	term :: proc(n: f64, j: int, d: f64) -> f64 {
		jf := f64(j)
		p := d + jf / n
		q := (n - jf) / n - d
		if q <= 0 {return 0}
		return math.exp(_log_dbinom(jf, n, p, q)) * d / p
	}
	if n <= 2000 {
		s := 0.0
		for j in 0 ..= jmax {s += term(nf, j, d)}
		return s
	}
	// the terms rise to one maximum and fall again: find it, then add
	// outwards until the terms are negligible
	lo, hi := 0, jmax
	for hi - lo > 2 {
		m1 := lo + (hi - lo) / 3
		m2 := hi - (hi - lo) / 3
		if term(nf, m1, d) < term(nf, m2, d) {lo = m1 + 1} else {hi = m2 - 1}
	}
	top := lo
	for j in lo + 1 ..= hi {if term(nf, j, d) > term(nf, top, d) {top = j}}
	s := term(nf, top, d)
	for j := top - 1; j >= 0; j -= 1 {
		t := term(nf, j, d)
		s += t
		if t < 1e-18 * s {break}
	}
	for j := top + 1; j <= jmax; j += 1 {
		t := term(nf, j, d)
		s += t
		if t < 1e-18 * s {break}
	}
	return s
}

// Two-sided exact: P(D_n <= x) (cdf) or P(D_n > x), by the algorithm choice
// of Simard and L'Ecuyer (2011) as in scipy's _kolmogn
@(private)
_kolmogn :: proc(n: int, x: f64, want_cdf: bool) -> f64 {
	pick :: proc(c: f64, want_cdf: bool) -> f64 {return clamp(c if want_cdf else 1 - c, 0, 1)}
	pick_sf :: proc(s: f64, want_cdf: bool) -> f64 {return clamp(1 - s if want_cdf else s, 0, 1)}
	if math.is_nan(x) {return x}
	if x >= 1 {return pick(1, want_cdf)}
	if x <= 0 {return pick(0, want_cdf)}
	nf := f64(n)
	t := nf * x
	if t <= 1 {
		// Ruben-Gambino: 1/(2n) <= x <= 1/n
		if t <= 0.5 {return pick(0, want_cdf)}
		prob: f64
		if n <= 140 {
			prob = 1
			for i in 1 ..= n {prob *= f64(i) / nf * (2 * t - 1)}
		} else {
			prob = math.exp(_log_nfact_div_npow(nf) + nf * math.ln(2 * t - 1))
		}
		return pick(prob, want_cdf)
	}
	if t >= nf - 1 {return pick_sf(2 * math.pow(1 - x, nf), want_cdf)}
	if x >= 0.5 {return pick_sf(2 * _smirnov_sf(n, x), want_cdf)}
	nxsq := t * x
	if n <= 140 {
		if nxsq <= 0.754693 {return pick(_kolmogn_dmtw(n, x), want_cdf)}
		if nxsq <= 4 {return pick(_kolmogn_pomeranz(n, x), want_cdf)}
		return pick_sf(2 * _smirnov_sf(n, x), want_cdf)
	}
	if !want_cdf {
		if nxsq >= 370 {return 0}
		if nxsq >= 2.2 {return clamp(2 * _smirnov_sf(n, x), 0, 1)}
	}
	c: f64
	if nxsq >= 18 {
		c = 1
	} else if n <= 100000 && nf * math.pow(x, 1.5) <= 1.4 {
		c = _kolmogn_dmtw(n, x)
	} else {
		c = _kolmogn_pelz_good(n, x)
	}
	return pick(c, want_cdf)
}

// log(n! / n^n) by Stirling's series
@(private)
_log_nfact_div_npow :: proc(n: f64) -> f64 {
	c := [8]f64 {
		-2.955065359477124183e-2,
		6.4102564102564102564e-3,
		-1.9175269175269175269e-3,
		8.4175084175084175084e-4,
		-5.952380952380952381e-4,
		7.9365079365079365079e-4,
		-2.7777777777777777778e-3,
		8.3333333333333333333e-2,
	}
	rn := 1 / n
	z := rn / n
	p := 0.0
	for v in c {p = p * z + v}
	return math.ln(n) / 2 - n + math.ln(2 * math.PI) / 2 + rn * p
}

@(private)
KS_EP128 :: 3.402823669209385e38 // 2^128

@(private)
KS_EM128 :: 2.938735877055719e-39 // 2^-128

// P(D_n <= d) by Durbin's matrix, as Marsaglia, Tsang and Wang (2003)
@(private)
_kolmogn_dmtw :: proc(n: int, d: f64) -> f64 {
	if d >= 1 {return 1}
	nd := f64(n) * d
	if nd <= 0.5 {return 0}
	k := int(math.ceil(nd))
	h := f64(k) - nd
	m := 2 * k - 1
	H := make([]f64, m * m, context.temp_allocator)
	v := make([]f64, m, context.temp_allocator)
	wv := make([]f64, m, context.temp_allocator)
	fac := 1.0
	for j in 1 ..= m {
		wv[j - 1] = fac
		fac /= f64(j)
		v[j - 1] = (1 - math.pow(h, f64(j))) * fac
	}
	tt := math.pow(max(2 * h - 1, 0), f64(m)) - 2 * math.pow(h, f64(m))
	v[m - 1] = (1 + tt) * fac
	for i in 1 ..< m {
		for r in i - 1 ..< m {H[r * m + i] = wv[r - (i - 1)]}
	}
	for r in 0 ..< m {H[r * m] = v[r]}
	for c in 0 ..< m {H[(m - 1) * m + c] = v[m - 1 - c]}
	P := make([]f64, m * m, context.temp_allocator)
	for i in 0 ..< m {P[i * m + i] = 1}
	T := make([]f64, m * m, context.temp_allocator)
	expnt, hexpnt := 0, 0
	nn := n
	for nn > 0 {
		if nn % 2 == 1 {
			_matmul(P, H, T, m)
			copy(P, T)
			expnt += hexpnt
		}
		_matmul(H, H, T, m)
		copy(H, T)
		hexpnt *= 2
		if abs(H[(k - 1) * m + k - 1]) > KS_EP128 {
			for &e in H {e /= KS_EP128}
			hexpnt += 128
		}
		nn /= 2
	}
	p := P[(k - 1) * m + k - 1]
	for i in 1 ..= n {
		p = f64(i) * p / f64(n)
		if abs(p) < KS_EM128 {
			p *= KS_EP128
			expnt -= 128
		}
	}
	if expnt != 0 {p = math.ldexp(p, expnt)}
	return clamp(p, 0, 1)
}

// C = A B for m x m row-major matrices
@(private)
_matmul :: proc(A, B, C: []f64, m: int) {
	for i in 0 ..< m {
		for j in 0 ..< m {C[i * m + j] = 0}
		for t in 0 ..< m {
			a := A[i * m + t]
			if a == 0 {continue}
			for j in 0 ..< m {C[i * m + j] += a * B[t * m + j]}
		}
	}
}

// P(D_n <= x) by Pomeranz's recursion (1974)
@(private)
_kolmogn_pomeranz :: proc(n: int, x: f64) -> f64 {
	t := f64(n) * x
	ll := int(math.floor(t))
	f := t - f64(ll)
	g := min(f, 1 - f)
	ceilf := 1 if f > 0 else 0
	roundf := 1 if f > 0.5 else 0
	npwrs := 2 * (ll + 1)
	gp := make([]f64, npwrs, context.temp_allocator)
	tgp := make([]f64, npwrs, context.temp_allocator)
	omp := make([]f64, npwrs, context.temp_allocator)
	gp[0], tgp[0], omp[0] = 1, 1, 1
	nf := f64(n)
	for m in 1 ..< npwrs {
		gp[m] = gp[m - 1] * (g / nf) / f64(m)
		tgp[m] = tgp[m - 1] * (2 * g / nf) / f64(m)
		omp[m] = omp[m - 1] * ((1 - 2 * g) / nf) / f64(m)
	}
	V0 := make([]f64, npwrs, context.temp_allocator)
	V1 := make([]f64, npwrs, context.temp_allocator)
	V1[0] = 1
	v0s, v1s := 0, 0
	expnt := 0
	j1, j2 := _pomeranz_j1j2(0, n, ll, ceilf, roundf)
	for i in 1 ..< 2 * n + 2 {
		k1 := j1
		V0, V1 = V1, V0
		v0s, v1s = v1s, v0s
		for &e in V1 {e = 0}
		j1, j2 = _pomeranz_j1j2(i, n, ll, ceilf, roundf)
		pw := gp
		if !(i == 1 || i == 2 * n + 1) {pw = tgp if i % 2 == 1 else omp}
		ln2 := j2 - k1 + 1
		if ln2 > 0 {
			// the full convolution of V0[k1 - v0s : +ln2] (cut at the end of
			// V0, as numpy slices) with pw[:ln2], entries j1 - k1 .. j2 - k1
			s0 := k1 - v0s
			alen := min(ln2, npwrs - s0)
			cs := j1 - k1
			mx := 0.0
			for c in 0 ..< j2 - j1 + 1 {
				idx := cs + c
				acc := 0.0
				for p in max(0, idx - (ln2 - 1)) ..= min(alen - 1, idx) {
					acc += V0[s0 + p] * pw[idx - p]
				}
				V1[c] = acc
				mx = max(mx, acc)
			}
			if 0 < mx && mx < KS_EM128 {
				for &e in V1 {e *= KS_EP128}
				expnt -= 128
			}
			v1s = v0s + j1 - k1
		}
	}
	ans := V1[n - v1s]
	for m in 1 ..= n {
		if abs(ans) > KS_EP128 {
			ans *= KS_EM128
			expnt += 128
		}
		ans *= f64(m)
	}
	if expnt != 0 {ans = math.ldexp(ans, expnt)}
	return clamp(ans, 0, 1)
}

@(private)
_pomeranz_j1j2 :: proc(i, n, ll, ceilf, roundf: int) -> (int, int) {
	j1, j2: int
	if i == 0 {
		j1, j2 = -ll - ceilf - 1, ll + ceilf - 1
	} else {
		h, odd := (i + 1) / 2, (i + 1) % 2
		if odd == 0 {
			// i is odd
			if h == n + 1 {
				j1, j2 = n - ll - ceilf - 1, n + ll + ceilf - 1
			} else {
				j1, j2 = h - 1 - ll - roundf - 1, h + ll - 1 + ceilf - 1
			}
		} else {
			j1, j2 = h - 1 - ll - 1, h + ll + roundf - 1
		}
	}
	return max(j1 + 2, 0), min(j2, n)
}

// P(D_n <= x) by the Pelz-Good (1976) asymptotic series
@(private)
_kolmogn_pelz_good :: proc(n: int, x: f64) -> f64 {
	if x <= 0 {return 0}
	if x >= 1 {return 1}
	PI2 :: math.PI * math.PI
	PI4 :: PI2 * PI2
	PI6 :: PI4 * PI2
	SQRT2PI :: 2.5066282746310002
	z := math.sqrt(f64(n)) * x
	z2 := z * z
	z3, z4, z6 := z2 * z, z2 * z2, z2 * z2 * z2
	qlog := -PI2 / 8 / z2
	if qlog < -708 {return 0}
	q := math.exp(qlog)
	k1a := -z2
	k1b := PI2 / 4
	k2a := 6 * z6 + 2 * z4
	k2b := (2 * z4 - 5 * z2) * PI2 / 4
	k2c := PI4 * (1 - 2 * z2) / 16
	k3d := PI6 * (5 - 30 * z2) / 64
	k3c := PI4 * (-60 * z2 + 212 * z4) / 16
	k3b := PI2 * (135 * z4 - 96 * z6) / 4
	k3a := -30 * z6 - 90 * z6 * z2
	K: [4]f64
	maxk := int(math.ceil(16 * z / math.PI))
	for k := maxk; k > 0; k -= 1 {
		m := f64(2 * k - 1)
		m2 := m * m
		m4, m6 := m2 * m2, m2 * m2 * m2
		qp := math.pow(q, f64(8 * k))
		c := [4]f64 {
			1,
			k1a + k1b * m2,
			k2a + k2b * m2 + k2c * m4,
			k3a + k3b * m2 + k3c * m4 + k3d * m6,
		}
		for i in 0 ..< 4 {K[i] = K[i] * qp + c[i]}
	}
	div := [4]f64{z, 6 * z4, 72 * z6 * z, 6480 * z6 * z4}
	for i in 0 ..< 4 {K[i] = K[i] * q * SQRT2PI / div[i]}
	q = math.exp(-PI2 / 2 / z2)
	sqrt3z := math.sqrt(f64(3)) * z
	k2extra, k3extra: f64
	for k := maxk; k > 0; k -= 1 {
		kf := f64(k)
		qp := math.pow(q, kf * kf)
		k2extra += kf * kf * qp
		kspi := math.PI * kf
		k3extra += (sqrt3z + kspi) * (sqrt3z - kspi) * kf * kf * qp
	}
	K[2] += k2extra * PI2 * SQRT2PI / (-36 * z3)
	K[3] += k3extra * PI2 * SQRT2PI / (216 * z6)
	sn := math.sqrt(f64(n))
	return K[0] + K[1] / sn + K[2] / f64(n) + K[3] / (f64(n) * sn)
}
