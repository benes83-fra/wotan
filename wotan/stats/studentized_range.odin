package stats

import "core:math"

// The studentized range distribution: the range of k independent standard
// normal values divided by an independent sqrt(χ²_df / df) (scipy's
// studentized_range). It is the reference distribution of Tukey's HSD test.
// k > 1 and df > 0; df = +Inf gives the range of k normal values itself.
// cdf and sf are each one integral of a positive integrand, so both tails
// keep about 13 significant digits.
Studentized_Range :: struct {
	k, df: f64,
}

@(private)
_sr_ok :: proc(d: Studentized_Range) -> bool {
	return d.k > 1 && !math.is_inf(d.k) && d.df > 0 && !math.is_nan(d.df)
}

studentized_range_cdf :: proc(d: Studentized_Range, q: f64) -> f64 {
	if !_sr_ok(d) || math.is_nan(q) {return math.nan_f64()}
	if q <= 0 {return 0}
	if math.is_inf(q) {return 1}
	return _sr_tail(d, q, false)
}

studentized_range_sf :: proc(d: Studentized_Range, q: f64) -> f64 {
	if !_sr_ok(d) || math.is_nan(q) {return math.nan_f64()}
	if q <= 0 {return 1}
	if math.is_inf(q) {return 0}
	return _sr_tail(d, q, true)
}

studentized_range_ppf :: proc(d: Studentized_Range, p: f64) -> f64 {
	if !_sr_ok(d) || !(p >= 0 && p <= 1) {return math.nan_f64()}
	if p == 0 {return 0}
	if p == 1 {return math.inf_f64(1)}
	if p <= 0.5 {return _sr_solve(d, p, false)}
	return _sr_solve(d, 1 - p, true)
}

studentized_range_isf :: proc(d: Studentized_Range, q: f64) -> f64 {
	if !_sr_ok(d) || !(q >= 0 && q <= 1) {return math.nan_f64()}
	if q == 0 {return math.inf_f64(1)}
	if q == 1 {return 0}
	if q <= 0.5 {return _sr_solve(d, q, true)}
	return _sr_solve(d, 1 - q, false)
}

// ---------------------------------------------------------------------------

// log Φ(z), accurate on both sides
@(private)
_log_phi :: proc(z: f64) -> f64 {
	if z > 0 {return math.log1p(-normal_sf(STANDARD_NORMAL, z))}
	return _log_ndtr(z)
}

@(private)
_Range_Data :: struct {
	w, k:  f64,
	upper: bool,
}

// k φ(z) [Φ(z)^(k-1) - (Φ(z) - Φ(z-w))^(k-1)] (upper) or
// k φ(z) (Φ(z) - Φ(z-w))^(k-1) (lower), from logarithms so that neither
// tail cancels
@(private)
_range_integrand :: proc(data: rawptr, z: f64) -> f64 {
	r := (^_Range_Data)(data)
	la := _log_phi(z)
	lb := _log_phi(z - r.w)
	t := lb - la
	ld := math.log1p(-math.exp(t)) if t < -0.6931471805599453 else math.ln(-expm1(t)) // ln(Φ(z)-Φ(z-w)) - ln Φ(z)
	lphi := -0.5 * z * z - 0.9189385332046728
	km := r.k - 1
	if r.upper {
		return -r.k * math.exp(lphi + km * la) * expm1(km * ld)
	}
	return r.k * math.exp(lphi + km * (la + ld))
}

// P(range of k standard normals > w) (upper) or <= w
@(private)
_range_tail :: proc(w, k: f64, upper: bool) -> f64 {
	if w <= 0 {return 1 if upper else 0}
	data := _Range_Data{w, k, upper}
	lo := min(-9.0, 0.5 * w - 9)
	hi := max(9.0, 0.5 * w + 9)
	return _gk_adapt(_range_integrand, &data, lo, hi, int(math.ceil((hi - lo) / 3)))
}

@(private)
_SR_Data :: struct {
	q, k, df, lc: f64,
	upper:        bool,
}

// integrand of the outer integral over u = ln s, s = sqrt(χ²/df)
@(private)
_sr_log_integrand :: proc(d: ^_SR_Data, u: f64) -> f64 {
	s := math.exp(u)
	r := _range_tail(d.q * s, d.k, d.upper)
	if r <= 0 {return math.inf_f64(-1)}
	// ln f(s) + ln s with f the density of s; a(2u + 1 - s^2) = a log1pmx(s^2 - 1)
	x := expm1(2 * u)
	v := log1pmx(x) if abs(x) <= 0.5 else 2 * u - x // log1p(x) would round to ln(0) far left
	return d.lc + 0.5 * d.df * v + math.ln(r)
}

@(private)
_sr_integrand :: proc(data: rawptr, u: f64) -> f64 {
	return math.exp(_sr_log_integrand((^_SR_Data)(data), u))
}

@(private)
_sr_tail :: proc(d: Studentized_Range, q: f64, upper: bool) -> f64 {
	if math.is_inf(d.df) || d.df > 1e15 {return _range_tail(q, d.k, upper)}
	nu := d.df
	data := _SR_Data {
		q     = q,
		k     = d.k,
		df    = nu,
		lc    = math.LN2 + 0.5 * math.ln(0.5 * nu) - LOG_SQRT_2PI - _stirlerr(0.5 * nu), // Stirling form, no cancellation for large df
		upper = upper,
	}
	// climb to the peak of the integrand in steps of about its width, then
	// walk out until it has fallen by e^-40 on each side
	h := 1 / math.sqrt(2 * nu)
	u0 := 0.5 * math.ln(nu / (nu + 0.5 * q * q)) if upper else 0.0
	f0 := _sr_log_integrand(&data, u0)
	for dir in ([2]f64{1, -1}) {
		for _ in 0 ..< 400 {
			f1 := _sr_log_integrand(&data, u0 + dir * h)
			if !(f1 > f0) {break}
			u0, f0 = u0 + dir * h, f1
		}
	}
	if math.is_inf(f0) {return 0}
	lo, hi := u0, u0
	for _ in 0 ..< 2000 {
		lo -= h
		if _sr_log_integrand(&data, lo) < f0 - 40 {break}
	}
	for _ in 0 ..< 2000 {
		hi += h
		if _sr_log_integrand(&data, hi) < f0 - 40 {break}
	}
	return min(1, _gk_adapt(_sr_integrand, &data, lo, hi, int(math.ceil((hi - lo) / (2 * h)))))
}

// q with sf(q) = p (upper) or cdf(q) = p, p <= 0.5: Illinois steps on the
// logarithm of the tail inside a bracket
@(private)
_sr_solve :: proc(d: Studentized_Range, p: f64, upper: bool) -> f64 {
	lp := math.ln(p)
	g :: proc(d: Studentized_Range, q, lp: f64, upper: bool) -> f64 {
		v := _sr_tail(d, q, upper)
		r := math.ln(v) - lp
		return r if upper else -r // decreasing in q either way
	}
	a, b := 0.0, 4.0
	ga := -lp if upper else math.inf_f64(1)
	gb := g(d, b, lp, upper)
	for gb > 0 {
		a, ga = b, gb
		b *= 2
		gb = g(d, b, lp, upper)
		if b > 1e300 {return math.inf_f64(1)}
	}
	if math.is_inf(ga) {
		// bracket the lower end away from 0, where the tail logarithm is infinite
		for {
			m := 0.5 * b
			gm := g(d, m, lp, upper)
			if gm > 0 {a, ga = m, gm; break}
			b, gb = m, gm
			if b < 1e-300 {return 0}
		}
	}
	side := 0
	for _ in 0 ..< 200 {
		x := b - gb * (b - a) / (gb - ga)
		if !(x > min(a, b) && x < max(a, b)) {x = 0.5 * (a + b)}
		gx := g(d, x, lp, upper)
		if gx == 0 {return x}
		if (gx > 0) == (ga > 0) {
			a, ga = x, gx
			if side == -1 {gb *= 0.5}
			side = -1
		} else {
			b, gb = x, gx
			if side == 1 {ga *= 0.5}
			side = 1
		}
		if abs(b - a) <= 4e-16 * abs(x) {return x}
	}
	return 0.5 * (a + b)
}

// ---------------------------------------------------------------------------
// adaptive Gauss-Kronrod (10/21 point) quadrature of a smooth integrand,
// starting from `panels` equal pieces and splitting the piece with the
// largest error estimate until the estimate falls below 1e-13 of the total

@(private)
_GK_X := [11]f64 {
	0.995657163025808080735527280689003,
	0.973906528517171720077964012084452,
	0.930157491355708226001207180059508,
	0.865063366688984510732096688423493,
	0.780817726586416897063717578345042,
	0.679409568299024406234327365114874,
	0.562757134668604683339000099272694,
	0.433395394129247190799265943165784,
	0.294392862701460198131126603103866,
	0.148874338981631210884826001129720,
	0,
}

@(private)
_GK_WK := [11]f64 {
	0.011694638867371874278064396062192,
	0.032558162307964727478818972459390,
	0.054755896574351996031381300244580,
	0.075039674810919952767043140916190,
	0.093125454583697605535065465083366,
	0.109387158802297641899210590325805,
	0.123491976262065851077208067330237,
	0.134709217311473325928054001771707,
	0.142775938577060080797094273138717,
	0.147739104901338491374841515972068,
	0.149445554002916905664936468389821,
}

@(private)
_GK_WG := [5]f64 {
	0.066671344308688137593568809893332,
	0.149451349150580593145776339657697,
	0.219086362515982043995534934228163,
	0.269266719309996355091226921569469,
	0.295524224714752870173892994651338,
}

@(private)
_gk21 :: proc(f: proc(_: rawptr, _: f64) -> f64, data: rawptr, a, b: f64) -> (k, err: f64) {
	c := 0.5 * (a + b)
	h := 0.5 * (b - a)
	fv: [21]f64
	fv[20] = f(data, c)
	rk := fv[20] * _GK_WK[10]
	rg := 0.0
	for j in 0 ..< 10 {
		x := h * _GK_X[j]
		fv[2 * j], fv[2 * j + 1] = f(data, c - x), f(data, c + x)
		s := fv[2 * j] + fv[2 * j + 1]
		rk += _GK_WK[j] * s
		if j % 2 == 1 {rg += _GK_WG[j / 2] * s}
	}
	// QUADPACK's error estimate: |K21 - G10| scaled down for smooth integrands
	mean := 0.5 * rk
	asc := _GK_WK[10] * abs(fv[20] - mean)
	for j in 0 ..< 10 {asc += _GK_WK[j] * (abs(fv[2 * j] - mean) + abs(fv[2 * j + 1] - mean))}
	err = abs((rk - rg) * h)
	asc *= abs(h)
	if asc != 0 && err != 0 {err = asc * min(1, math.pow(200 * err / asc, 1.5))}
	return rk * h, err
}

@(private)
_gk_adapt :: proc(f: proc(_: rawptr, _: f64) -> f64, data: rawptr, a, b: f64, panels: int) -> f64 {
	Piece :: struct {
		a, b, k, err: f64,
	}
	MAX :: 400
	pieces: [MAX]Piece
	n := clamp(panels, 1, MAX / 2)
	h := (b - a) / f64(n)
	for i in 0 ..< n {
		x0 := a + f64(i) * h
		x1 := b if i == n - 1 else x0 + h
		k, e := _gk21(f, data, x0, x1)
		pieces[i] = {x0, x1, k, e}
	}
	for {
		total, err: _Sum
		worst := 0
		for i in 0 ..< n {
			_sum_add(&total, pieces[i].k)
			_sum_add(&err, pieces[i].err)
			if pieces[i].err > pieces[worst].err {worst = i}
		}
		if _sum(err) <= 1e-13 * abs(_sum(total)) || n == MAX {return _sum(total)}
		p := pieces[worst]
		m := 0.5 * (p.a + p.b)
		k1, e1 := _gk21(f, data, p.a, m)
		k2, e2 := _gk21(f, data, m, p.b)
		pieces[worst] = {p.a, m, k1, e1}
		pieces[n] = {m, p.b, k2, e2}
		n += 1
	}
}
