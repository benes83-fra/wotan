package analytics

import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Trend / seasonal decomposition: Hodrick-Prescott filter, classical moving-
// average decomposition and STL (statsmodels hpfilter, seasonal_decompose, STL)
// ============================================================================

// Hodrick-Prescott filter: trend τ = argmin Σ (x_t - τ_t)² + λ Σ (Δ²τ_t)², i.e.
// (I + λ KᵀK) τ = x with K the second-difference matrix (statsmodels hpfilter).
// λ = 1600 for quarterly, 129600 (Ravn-Uhlig) / 14400 for monthly, 6.25 / 100 annual data.
// Returns cycle = x - trend and trend.
hp_filter :: proc(x: []f64, lambda: f64 = 1600, allocator := context.allocator) -> (cycle, trend: []f64) {
	n := len(x)
	if n < 3 do panic("hp_filter: need at least 3 observations")
	if lambda < 0 do panic("hp_filter: lambda must be >= 0")
	// A = I + λ KᵀK is symmetric pentadiagonal: store bands d0 (diag), d1, d2
	d0 := make([]f64, n, context.temp_allocator)
	d1 := make([]f64, n, context.temp_allocator)
	d2 := make([]f64, n, context.temp_allocator)
	defer {
		delete(d0, context.temp_allocator)
		delete(d1, context.temp_allocator)
		delete(d2, context.temp_allocator)
	}
	// KᵀK = Σ_r k_r k_rᵀ with k_r = e_r - 2 e_{r+1} + e_{r+2}
	for r in 0 ..< n - 2 {
		c := [3]f64{1, -2, 1}
		for a in 0 ..< 3 {
			d0[r + a] += lambda * c[a] * c[a]
			if a < 2 do d1[r + a] += lambda * c[a] * c[a + 1]
			if a < 1 do d2[r + a] += lambda * c[a] * c[a + 2]
		}
	}
	for i in 0 ..< n {d0[i] += 1}
	// banded Cholesky A = L Lᵀ (bandwidth 2): L stored as l0 (diag), l1, l2 (sub-diagonals)
	l0 := make([]f64, n, context.temp_allocator)
	l1 := make([]f64, n, context.temp_allocator)
	l2 := make([]f64, n, context.temp_allocator)
	z := make([]f64, n, context.temp_allocator)
	defer {
		delete(l0, context.temp_allocator)
		delete(l1, context.temp_allocator)
		delete(l2, context.temp_allocator)
		delete(z, context.temp_allocator)
	}
	for i in 0 ..< n {
		// L[i, i-2] = A[i, i-2] / L[i-2, i-2]
		if i >= 2 do l2[i] = d2[i - 2] / l0[i - 2]
		// L[i, i-1] = (A[i, i-1] - L[i, i-2] L[i-1, i-2]) / L[i-1, i-1]
		if i >= 1 {
			v := d1[i - 1]
			if i >= 2 do v -= l2[i] * l1[i - 1]
			l1[i] = v / l0[i - 1]
		}
		v := d0[i] - l1[i] * l1[i] - l2[i] * l2[i]
		l0[i] = math.sqrt(v)
	}
	// L z = x, Lᵀ τ = z
	trend = make([]f64, n, allocator)
	cycle = make([]f64, n, allocator)
	for i in 0 ..< n {
		v := x[i]
		if i >= 1 do v -= l1[i] * z[i - 1]
		if i >= 2 do v -= l2[i] * z[i - 2]
		z[i] = v / l0[i]
	}
	for i := n - 1; i >= 0; i -= 1 {
		v := z[i]
		if i + 1 < n do v -= l1[i + 1] * trend[i + 1]
		if i + 2 < n do v -= l2[i + 2] * trend[i + 2]
		trend[i] = v / l0[i]
	}
	for i in 0 ..< n {cycle[i] = x[i] - trend[i]}
	return
}

DecomposeModel :: enum {
	Additive, // x = trend + seasonal + resid
	Multiplicative, // x = trend · seasonal · resid
}

DecomposeResult :: struct {
	trend:     []f64, // NaN where the moving average is not defined (classical)
	seasonal:  []f64,
	resid:     []f64,
	weights:   []f64, // STL robustness weights (1 if not robust; nil for classical)
	allocator: mem.Allocator,
}

decompose_free :: proc(r: ^DecomposeResult) {
	delete(r.trend, r.allocator)
	delete(r.seasonal, r.allocator)
	delete(r.resid, r.allocator)
	delete(r.weights, r.allocator)
}

// Classical decomposition (statsmodels seasonal_decompose, two-sided filter,
// no trend extrapolation): trend = centred moving average over one period
// (weights ½, 1, …, 1, ½ / period for even periods), seasonal = mean of the
// detrended values of each phase, centred to mean 0 (additive) / 1 (multiplicative).
seasonal_decompose :: proc(
	x: []f64,
	period: int,
	model: DecomposeModel = .Additive,
	allocator := context.allocator,
) -> DecomposeResult {
	n := len(x)
	if period < 2 do panic("seasonal_decompose: period must be >= 2")
	if n < 2 * period do panic("seasonal_decompose: needs two full periods")
	for v in x {
		if math.is_nan(v) || math.is_inf(v) do panic("seasonal_decompose: missing / non-finite values")
		if model == .Multiplicative && !(v > 0) do panic("seasonal_decompose: multiplicative model needs positive data")
	}
	filt := make([]f64, period + 1 if period % 2 == 0 else period, context.temp_allocator)
	defer delete(filt, context.temp_allocator)
	if period % 2 == 0 {
		for &w in filt {w = 1 / f64(period)}
		filt[0], filt[period] = 0.5 / f64(period), 0.5 / f64(period)
	} else {
		for &w in filt {w = 1 / f64(period)}
	}
	L := len(filt)
	half := L / 2
	res := DecomposeResult {
		trend     = make([]f64, n, allocator),
		seasonal  = make([]f64, n, allocator),
		resid     = make([]f64, n, allocator),
		allocator = allocator,
	}
	for t in 0 ..< n {
		if t < half || t + half >= n {
			res.trend[t] = math.nan_f64()
			continue
		}
		s := 0.0
		for k in 0 ..< L {s += filt[k] * x[t - half + k]}
		res.trend[t] = s
	}
	avg := make([]f64, period, context.temp_allocator)
	defer delete(avg, context.temp_allocator)
	for ph in 0 ..< period {
		s, c := 0.0, 0
		for t := ph; t < n; t += period {
			if math.is_nan(res.trend[t]) do continue
			s += x[t] / res.trend[t] if model == .Multiplicative else x[t] - res.trend[t]
			c += 1
		}
		avg[ph] = s / f64(c)
	}
	mean := 0.0
	for v in avg {mean += v}
	mean /= f64(period)
	for &v in avg {v = v / mean if model == .Multiplicative else v - mean}
	for t in 0 ..< n {
		res.seasonal[t] = avg[t % period]
		if model == .Multiplicative {
			res.resid[t] = x[t] / res.seasonal[t] / res.trend[t]
		} else {
			res.resid[t] = x[t] - res.trend[t] - res.seasonal[t]
		}
	}
	return res
}

// ----------------------------------------------------------------------------
// STL (Cleveland, Cleveland, McRae & Terpenning 1990), a port of the netlib
// stl.f used by statsmodels STL
// ----------------------------------------------------------------------------

STLParams :: struct {
	period:       int, // season length (>= 2)
	seasonal:     int, // seasonal smoother length (odd, >= 3; 0 → 7)
	trend:        int, // trend smoother length (odd; 0 → smallest odd ≥ 1.5 period / (1 - 1.5 / seasonal))
	low_pass:     int, // low-pass length (odd; 0 → smallest odd > period)
	seasonal_deg: int, // 0 or 1 (loess degree); statsmodels defaults: 1, 1, 1
	trend_deg:    int,
	low_pass_deg: int,
	robust:       bool, // robustness iterations (inner 2, outer 15; else inner 5, outer 0)
	seasonal_jump, trend_jump, low_pass_jump: int, // loess evaluated every k points (0 → 1)
	inner_iter, outer_iter: int, // 0 → the defaults above
	degrees_set:  bool, // false: use the default degrees 1 (set true to use 0 degrees)
}

// STL decomposition x = trend + seasonal + resid (statsmodels STL(x, period, …).fit())
stl_decompose :: proc(x: []f64, params: STLParams, allocator := context.allocator) -> DecomposeResult {
	n := len(x)
	np := params.period
	if np < 2 do panic("stl_decompose: period must be >= 2")
	if n < 2 * np do panic("stl_decompose: needs at least two full periods")
	ns := params.seasonal if params.seasonal > 0 else 7
	nt := params.trend
	if nt <= 0 {
		nt = int(math.ceil(1.5 * f64(np) / (1 - 1.5 / f64(ns))))
		if nt % 2 == 0 do nt += 1
	}
	nl := params.low_pass
	if nl <= 0 {
		nl = np + 1
		if nl % 2 == 0 do nl += 1
	}
	if ns < 3 || ns % 2 == 0 do panic("stl_decompose: seasonal must be odd and >= 3")
	if nt < 3 || nt % 2 == 0 do panic("stl_decompose: trend must be odd and >= 3")
	if nl < 3 || nl % 2 == 0 do panic("stl_decompose: low_pass must be odd and >= 3")
	isdeg, itdeg, ildeg := 1, 1, 1
	if params.degrees_set {isdeg, itdeg, ildeg = params.seasonal_deg, params.trend_deg, params.low_pass_deg}
	nsjump := max(params.seasonal_jump, 1)
	ntjump := max(params.trend_jump, 1)
	nljump := max(params.low_pass_jump, 1)
	ni := params.inner_iter if params.inner_iter > 0 else (2 if params.robust else 5)
	no := params.outer_iter if params.outer_iter > 0 else (15 if params.robust else 0)

	res := DecomposeResult {
		trend     = make([]f64, n, allocator),
		seasonal  = make([]f64, n, allocator),
		resid     = make([]f64, n, allocator),
		weights   = make([]f64, n, allocator),
		allocator = allocator,
	}
	rw := res.weights
	work := make([]f64, 5 * (n + 2 * np), context.temp_allocator)
	defer delete(work, context.temp_allocator)
	w1 := work[0 * (n + 2 * np):1 * (n + 2 * np)]
	w2 := work[1 * (n + 2 * np):2 * (n + 2 * np)]
	w3 := work[2 * (n + 2 * np):3 * (n + 2 * np)]
	w4 := work[3 * (n + 2 * np):4 * (n + 2 * np)]
	w5 := work[4 * (n + 2 * np):5 * (n + 2 * np)]

	userw := false
	k := 0
	for {
		_stl_stp(x, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, userw, rw, res.seasonal, res.trend, w1, w2, w3, w4, w5)
		k += 1
		if k > no do break
		for i in 0 ..< n {w1[i] = res.trend[i] + res.seasonal[i]}
		_stl_rwts(x, w1[:n], rw)
		userw = true
	}
	if no <= 0 {for &v in rw {v = 1}}
	for i in 0 ..< n {res.resid[i] = x[i] - res.trend[i] - res.seasonal[i]}
	return res
}

// Inner loop: ni passes of seasonal smoothing + low-pass + trend smoothing
@(private = "file")
_stl_stp :: proc(
	y: []f64,
	np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni: int,
	userw: bool,
	rw, season, trend, w1, w2, w3, w4, w5: []f64,
) {
	n := len(y)
	for _ in 0 ..< ni {
		for i in 0 ..< n {w1[i] = y[i] - trend[i]}
		_stl_ss(w1[:n], np, ns, isdeg, nsjump, userw, rw, w2, w3, w4, w5, season)
		_stl_fts(w2[:n + 2 * np], np, w3, w1)
		// stless(work3, n, nl, ildeg, nljump, false, work4, work1, work5): low-pass L → work1
		_stl_ess(w3[:n], nl, ildeg, nljump, false, w4, w1[:n], w5, nil)
		for i in 0 ..< n {season[i] = w2[np + i] - w1[i]}
		for i in 0 ..< n {w1[i] = y[i] - season[i]}
		_stl_ess(w1[:n], nt, itdeg, ntjump, userw, rw, trend, w3, w5)
	}
}

// Robustness weights: bisquare of |r| / (6 · median |r|)
@(private = "file")
_stl_rwts :: proc(y, fit, rw: []f64) {
	n := len(y)
	r := make([]f64, n, context.temp_allocator)
	defer delete(r, context.temp_allocator)
	for i in 0 ..< n {r[i] = math.abs(y[i] - fit[i])}
	srt := slice.clone(r, context.temp_allocator)
	defer delete(srt, context.temp_allocator)
	slice.sort(srt)
	mid0 := n / 2 + 1 // Fortran 1-based: mid(1) = n/2 + 1, mid(2) = n - mid(1) + 1
	mid1 := n - mid0 + 1
	cmad := 3 * (srt[mid0 - 1] + srt[mid1 - 1])
	c9, c1 := 0.999 * cmad, 0.001 * cmad
	for i in 0 ..< n {
		if r[i] <= c1 {
			rw[i] = 1
		} else if r[i] <= c9 {
			u := r[i] / cmad
			rw[i] = (1 - u * u) * (1 - u * u)
		} else {
			rw[i] = 0
		}
	}
}

// Cycle-subseries smoothing: every phase is loess-smoothed and extended by one
// point at both ends; season (length n + 2 np) holds the result
@(private = "file")
_stl_ss :: proc(y: []f64, np, ns, isdeg, nsjump: int, userw: bool, rw, season, work1, work2, work3: []f64, work4: []f64) {
	n := len(y)
	for j in 1 ..= np {
		k := (n - j) / np + 1
		for i in 1 ..= k {work1[i - 1] = y[(i - 1) * np + j - 1]}
		if userw {for i in 1 ..= k {work3[i - 1] = rw[(i - 1) * np + j - 1]}}
		// work2[1 …] = smoothed subseries (work2[0] and work2[k+1] are the extensions)
		_stl_ess(work1[:k], ns, isdeg, nsjump, userw, work3, work2[1:k + 1], work4, nil)
		nright := min(ns, k)
		v, ok := _stl_est(work1[:k], ns, isdeg, 0, 1, nright, work4, userw, work3)
		work2[0] = v if ok else work2[1]
		nleft := max(1, k - ns + 1)
		v, ok = _stl_est(work1[:k], ns, isdeg, f64(k + 1), nleft, k, work4, userw, work3)
		work2[k + 1] = v if ok else work2[k]
		for m in 1 ..= k + 2 {season[(m - 1) * np + j - 1] = work2[m - 1]}
	}
}

// Low-pass filter of the cycle-subseries: moving averages of length np, np, 3
@(private = "file")
_stl_fts :: proc(x: []f64, np: int, trend, work: []f64) {
	n := len(x)
	_stl_ma(x, np, trend)
	_stl_ma(trend[:n - np + 1], np, work)
	_stl_ma(work[:n - 2 * np + 2], 3, trend)
}

@(private = "file")
_stl_ma :: proc(x: []f64, length: int, ave: []f64) {
	n := len(x)
	newn := n - length + 1
	flen := f64(length)
	v := 0.0
	for i in 0 ..< length {v += x[i]}
	ave[0] = v / flen
	if newn > 1 {
		k := length
		m := 0
		for j in 1 ..< newn {
			v = v - x[m] + x[k]
			m += 1
			k += 1
			ave[j] = v / flen
		}
	}
}

// Loess smoothing of y (length n) into ys, evaluated every njump points and
// linearly interpolated in between (stless)
@(private = "file")
_stl_ess :: proc(y: []f64, length, ideg, njump: int, userw: bool, rw: []f64, ys: []f64, res: []f64, _unused: []f64) {
	n := len(y)
	if n < 2 {
		ys[0] = y[0]
		return
	}
	newnj := min(njump, n - 1)
	nleft, nright := 0, 0
	if length >= n {
		nleft, nright = 1, n
		for i := 1; i <= n; i += newnj {
			v, ok := _stl_est(y, length, ideg, f64(i), nleft, nright, res, userw, rw)
			ys[i - 1] = v if ok else y[i - 1]
		}
	} else if newnj == 1 {
		nsh := (length + 1) / 2
		nleft, nright = 1, length
		for i in 1 ..= n {
			if i > nsh && nright != n {
				nleft += 1
				nright += 1
			}
			v, ok := _stl_est(y, length, ideg, f64(i), nleft, nright, res, userw, rw)
			ys[i - 1] = v if ok else y[i - 1]
		}
	} else {
		nsh := (length + 1) / 2
		for i := 1; i <= n; i += newnj {
			if i < nsh {
				nleft, nright = 1, length
			} else if i >= n - nsh + 1 {
				nleft, nright = n - length + 1, n
			} else {
				nleft, nright = i - nsh + 1, length + i - nsh
			}
			v, ok := _stl_est(y, length, ideg, f64(i), nleft, nright, res, userw, rw)
			ys[i - 1] = v if ok else y[i - 1]
		}
	}
	if newnj != 1 {
		for i := 1; i <= n - newnj; i += newnj {
			delta := (ys[i + newnj - 1] - ys[i - 1]) / f64(newnj)
			for j in i + 1 ..= i + newnj - 1 {ys[j - 1] = ys[i - 1] + delta * f64(j - i)}
		}
		k := ((n - 1) / newnj) * newnj + 1
		if k != n {
			v, ok := _stl_est(y, length, ideg, f64(n), nleft, nright, res, userw, rw)
			ys[n - 1] = v if ok else y[n - 1]
			if k != n - 1 {
				delta := (ys[n - 1] - ys[k - 1]) / f64(n - k)
				for j in k + 1 ..= n - 1 {ys[j - 1] = ys[k - 1] + delta * f64(j - k)}
			}
		}
	}
}

// Local (degree 0 / 1) tricube-weighted fit at xs using points nleft … nright (stlest)
@(private = "file")
_stl_est :: proc(y: []f64, length, ideg: int, xs: f64, nleft, nright: int, w: []f64, userw: bool, rw: []f64) -> (ys: f64, ok: bool) {
	n := len(y)
	rng := f64(n) - 1
	h := max(xs - f64(nleft), f64(nright) - xs)
	if length > n do h += f64((length - n) / 2)
	h9, h1 := 0.999 * h, 0.001 * h
	a := 0.0
	for j in nleft ..= nright {
		w[j - 1] = 0
		r := math.abs(f64(j) - xs)
		if r <= h9 {
			if r <= h1 {
				w[j - 1] = 1
			} else {
				q := r / h
				q = 1 - q * q * q
				w[j - 1] = q * q * q
			}
			if userw do w[j - 1] *= rw[j - 1]
			a += w[j - 1]
		}
	}
	if a <= 0 do return 0, false
	for j in nleft ..= nright {w[j - 1] /= a}
	if h > 0 && ideg > 0 {
		a = 0
		for j in nleft ..= nright {a += w[j - 1] * f64(j)}
		b := xs - a
		c := 0.0
		for j in nleft ..= nright {c += w[j - 1] * (f64(j) - a) * (f64(j) - a)}
		if math.sqrt(c) > 0.001 * rng {
			b /= c
			for j in nleft ..= nright {w[j - 1] *= b * (f64(j) - a) + 1}
		}
	}
	ys = 0
	for j in nleft ..= nright {ys += w[j - 1] * y[j - 1]}
	return ys, true
}
