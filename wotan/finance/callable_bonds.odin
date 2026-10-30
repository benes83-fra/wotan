package finance

import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// CALLABLE BOND STRUCTURES
// ============================================================================

// Coupons are paid at maturity − k/coupon_frequency (a short first period pays a
// pro-rata coupon). Call prices are clean: on a call date that is also a coupon
// date the coupon is paid as well.
CallableBond :: struct {
	face_value:       f64,
	coupon_rate:      f64, // Annual coupon rate
	coupon_frequency: int, // Payments per year (e.g., 2 for semi-annual)
	maturity:         f64, // Bond maturity in years
	call_schedule:    []CallDate, // Call dates and prices (dates outside (0, maturity) are ignored)
	settlement:       f64, // Settlement date (usually 0; not used by the pricer)
}

CallDate :: struct {
	time:  f64, // Time from today (in years)
	price: f64, // Call price (usually par or par + premium)
}

CallableBondResult :: struct {
	price:               f64,
	straight_bond_price: f64,
	embedded_call_value: f64,
	oas:                 f64, // Always 0 here (the price is at zero spread); see callable_bond_oas
	effective_duration:  f64, // −∂P/∂s / P for a parallel shift s of the initial curve
	effective_convexity: f64, // ∂²P/∂s² / P
	call_probability:    f64, // Probability of being called
}

// ============================================================================
// HULL-WHITE 1F BUILDING BLOCKS
// ============================================================================
// r(t) = α(t) + x(t), dx = −a x dt + σ dW, x(0) = r0 − f(0,0), with
// α(t) = f(0,t) + σ²/2 B(a,t)² so the model reprices P(0,·) when r0 = f(0,0).
// A constant spread s added to r is exactly a parallel shift of the initial
// curve, so it serves as both the OAS and the effective-duration bump.

// (1 − e^{−k t}) / k, → t as k → 0
@(private = "file")
_hw_b :: proc(k, t: f64) -> f64 {
	x := k * t
	if abs(x) < 1e-5 {return t * (1.0 - x * (0.5 - x / 6.0))}
	return (1.0 - math.exp_f64(-x)) / k
}

// ∫_0^t B(a,u)² du = (t − 2 B(a,t) + B(2a,t)) / a²
@(private = "file")
_hw_k :: proc(a, t: f64) -> f64 {
	x := a * t
	if abs(x) < 1e-3 {return t * t * t * (1.0 / 3.0 - x / 4.0 + 7.0 * x * x / 60.0)}
	return (t - 2.0 * _hw_b(a, t) + _hw_b(2.0 * a, t)) / (a * a)
}

// Instantaneous forward f(0,t) = −∂ ln P(0,t)/∂t (one-sided near 0)
@(private = "file")
_hw_fwd :: proc(P0T: proc(_: f64) -> f64, t: f64) -> f64 {
	h := 1e-5
	if t > h {
		return (math.ln(P0T(t - h)) - math.ln(P0T(t + h))) / (2.0 * h)
	}
	return (3.0 * math.ln(P0T(t)) - 4.0 * math.ln(P0T(t + h)) + math.ln(P0T(t + 2.0 * h))) / (2.0 * h)
}

// ============================================================================
// ZERO-COUPON BOND PRICE UNDER HULL-WHITE 1F
// ============================================================================
// P(t,T) = P(0,T)/P(0,t) · exp(B f(0,t) − σ²/(4a) (1 − e^{−2at}) B² − B r_t),  B = B(a, T − t)
_hw1f_zero_coupon_bond :: proc(
	r_t: f64, // Current short rate
	t: f64, // Current time
	T: f64, // Maturity
	a: f64, // Mean reversion
	sigma: f64, // Volatility
	P0T_func: proc(_: f64) -> f64, // Market discount curve P(0,.)
) -> f64 {
	if t >= T {return 1.0}
	B := _hw_b(a, T - t)
	v := 0.5 * sigma * sigma * _hw_b(2.0 * a, t)
	return P0T_func(T) / P0T_func(t) * math.exp_f64(B * _hw_fwd(P0T_func, t) - v * B * B - B * r_t)
}

// ============================================================================
// BERMUDAN BOND ENGINE (shared with puttable_bonds.odin)
// ============================================================================
// Paths of (x, ∫x) are sampled exactly between exercise dates (they are jointly
// Gaussian), so there is no time-stepping error. The straight bond at each
// exercise date is analytic; Longstaff–Schwartz is applied to the embedded
// option only: issuer call value max(S − K, 0), investor put value max(K − S, 0),
// regressed on powers of the standardised state over in-the-money paths.

@(private)
_HW_Bermudan :: struct {
	is_call:    bool,
	a, sigma:   f64,
	degree:     int,
	n_paths:    int,
	cf_t:       []f64, // flow times
	ex_t, ex_k: []f64, // exercise dates (sorted, inside (0, maturity)) and prices
	s0_lc:      []f64, // ln(amount · P(0,t_i | x0)) per flow
	lc, bb:     []f64, // per (date j, flow i): straight bond = Σ_{i ≥ first[j]} exp(lc − bb x − s τ)
	first:      []int,
	x, ld:      []f64, // per (path, date): state, ln of the unshifted discount from the previous date
	allocator:  mem.Allocator,
}

@(private)
_hw_bermudan_build :: proc(
	face, coupon_rate: f64,
	freq: int,
	maturity: f64,
	ex_t_in, ex_k_in: []f64,
	is_call: bool,
	a, sigma, r0: f64,
	P0T: proc(_: f64) -> f64,
	n_paths, degree: int,
	allocator: mem.Allocator,
) -> _HW_Bermudan {
	e := _HW_Bermudan {
		is_call   = is_call,
		a         = a,
		sigma     = sigma,
		degree    = max(degree, 0),
		n_paths   = max(2, n_paths + n_paths % 2), // antithetic pairs
		allocator = allocator,
	}

	// Coupon schedule backwards from maturity
	f := f64(max(freq, 1))
	mf := maturity * f
	n_cf := int(math.round(mf))
	if abs(mf - f64(n_cf)) > 1e-9 * max(1.0, mf) {n_cf = int(math.ceil(mf))}
	n_cf = max(n_cf, 1)
	e.cf_t = make([]f64, n_cf, allocator)
	amount := make([]f64, n_cf, allocator)
	defer delete(amount, allocator)
	coupon := face * coupon_rate / f
	for i in 0 ..< n_cf {
		e.cf_t[i] = maturity - f64(n_cf - 1 - i) / f
		amount[i] = coupon
	}
	stub := e.cf_t[0] * f
	if abs(stub - 1.0) > 1e-9 {amount[0] = coupon * stub}
	amount[n_cf - 1] += face

	// Exercise dates inside (0, maturity), sorted
	idx := make([dynamic]int, 0, len(ex_t_in), allocator)
	defer delete(idx)
	for t, i in ex_t_in {
		if t > 1e-12 && t < maturity - 1e-12 {append(&idx, i)}
	}
	m := len(idx)
	e.ex_t = make([]f64, m, allocator)
	e.ex_k = make([]f64, m, allocator)
	for i, j in idx {
		e.ex_t[j] = ex_t_in[i]
		e.ex_k[j] = ex_k_in[i]
	}
	// insertion sort by time (schedules are short)
	for j in 1 ..< m {
		for q := j; q > 0 && e.ex_t[q] < e.ex_t[q - 1]; q -= 1 {
			e.ex_t[q], e.ex_t[q - 1] = e.ex_t[q - 1], e.ex_t[q]
			e.ex_k[q], e.ex_k[q - 1] = e.ex_k[q - 1], e.ex_k[q]
		}
	}

	s2 := 0.5 * sigma * sigma
	x0 := r0 - _hw_fwd(P0T, 0.0)
	e.s0_lc = make([]f64, n_cf, allocator)
	for t, i in e.cf_t {
		e.s0_lc[i] = math.ln(amount[i] * P0T(t)) - _hw_b(a, t) * x0
	}

	// Straight bond coefficients at each exercise date (flows strictly after it)
	e.lc = make([]f64, m * n_cf, allocator)
	e.bb = make([]f64, m * n_cf, allocator)
	e.first = make([]int, m, allocator)
	for j in 0 ..< m {
		tj := e.ex_t[j]
		fj := _hw_fwd(P0T, tj)
		bj := _hw_b(a, tj)
		alpha := fj + s2 * bj * bj
		v := s2 * _hw_b(2.0 * a, tj)
		lp := math.ln(P0T(tj))
		e.first[j] = n_cf
		for i := n_cf - 1; i >= 0 && e.cf_t[i] > tj + 1e-12; i -= 1 {e.first[j] = i}
		for i in e.first[j] ..< n_cf {
			B := _hw_b(a, e.cf_t[i] - tj)
			e.lc[j * n_cf + i] = math.ln(amount[i] * P0T(e.cf_t[i])) - lp + B * fj - v * B * B - B * alpha
			e.bb[j * n_cf + i] = B
		}
	}

	// Exact joint sampling of (x(t_j), ∫ x) between consecutive exercise dates
	n := e.n_paths
	half := n / 2
	e.x = make([]f64, n * m, allocator)
	e.ld = make([]f64, n * m, allocator)
	z := make([]f64, 2 * m, allocator)
	defer delete(z, allocator)
	for p in 0 ..< half {
		for k in 0 ..< 2 * m {z[k] = rand.float64_normal(0.0, 1.0)}
		for sgn, side in ([]f64{1.0, -1.0}) {
			row := (p + side * half) * m
			xs := x0
			tp := 0.0
			for j in 0 ..< m {
				tj := e.ex_t[j]
				dt := tj - tp
				b := _hw_b(a, dt)
				sx := sigma * math.sqrt_f64(_hw_b(2.0 * a, dt))
				c := s2 * b * b
				beta := c / sx if sx > 0.0 else 0.0
				sr := math.sqrt_f64(max(sigma * sigma * _hw_k(a, dt) - beta * beta, 0.0))
				det := math.ln(P0T(tp) / P0T(tj)) + s2 * (_hw_k(a, tj) - _hw_k(a, tp))
				z1, z2 := sgn * z[2 * j], sgn * z[2 * j + 1]
				integral := xs * b + beta * z1 + sr * z2
				xs = xs * math.exp_f64(-a * dt) + sx * z1
				e.x[row + j] = xs
				e.ld[row + j] = -det - integral
				tp = tj
			}
		}
	}
	return e
}

@(private)
_hw_bermudan_free :: proc(e: ^_HW_Bermudan) {
	delete(e.cf_t, e.allocator)
	delete(e.ex_t, e.allocator)
	delete(e.ex_k, e.allocator)
	delete(e.s0_lc, e.allocator)
	delete(e.lc, e.allocator)
	delete(e.bb, e.allocator)
	delete(e.first, e.allocator)
	delete(e.x, e.allocator)
	delete(e.ld, e.allocator)
}

// Straight bond value at time 0 under a parallel shift s
@(private)
_hw_bermudan_straight :: proc(e: ^_HW_Bermudan, s: f64) -> f64 {
	v := 0.0
	for t, i in e.cf_t {v += math.exp_f64(e.s0_lc[i] - s * t)}
	return v
}

// Straight bond at exercise date j and state x minus the strike (call), or the
// strike minus the straight bond (put); coef holds the shifted coefficients
@(private = "file")
_hw_ex_value :: proc(e: ^_HW_Bermudan, coef: []f64, j: int, x: f64) -> f64 {
	n_cf := len(e.cf_t)
	v := 0.0
	for i in e.first[j] ..< n_cf {v += coef[j * n_cf + i] * math.exp_f64(-e.bb[j * n_cf + i] * x)}
	return v - e.ex_k[j] if e.is_call else e.ex_k[j] - v
}

@(private = "file")
_hw_shift_coef :: proc(e: ^_HW_Bermudan, s: f64, allocator: mem.Allocator) -> []f64 {
	m, n_cf := len(e.ex_t), len(e.cf_t)
	coef := make([]f64, m * n_cf, allocator)
	for j in 0 ..< m {
		for i in e.first[j] ..< n_cf {
			coef[j * n_cf + i] = math.exp_f64(e.lc[j * n_cf + i] - s * (e.cf_t[i] - e.ex_t[j]))
		}
	}
	return coef
}

// Price of the callable (puttable) bond and the probability of exercise, with
// a parallel shift s of the curve (the OAS). If `stop` is given it receives
// each path's exercise date index (len(ex_t) = never exercised).
@(private)
_hw_bermudan_price :: proc(e: ^_HW_Bermudan, s: f64, stop: []int = nil) -> (price: f64, ex_prob: f64) {
	S0 := _hw_bermudan_straight(e, s)
	m := len(e.ex_t)
	n, nb := e.n_paths, e.degree + 1
	alloc := e.allocator
	if m == 0 {
		for p in 0 ..< len(stop) {stop[p] = 0}
		return S0, 0.0
	}

	coef := _hw_shift_coef(e, s, alloc)
	U := make([]f64, n, alloc)
	ev := make([]f64, n, alloc)
	X := make([]f64, n * nb, alloc)
	y := make([]f64, n, alloc)
	itm := make([]int, n, alloc)
	st := stop if stop != nil else make([]int, n, alloc)
	defer {
		delete(coef, alloc); delete(U, alloc); delete(ev, alloc)
		delete(X, alloc); delete(y, alloc); delete(itm, alloc)
		if stop == nil {delete(st, alloc)}
	}

	for p in 0 ..< n {
		g := _hw_ex_value(e, coef, m - 1, e.x[p * m + m - 1])
		U[p] = max(g, 0.0)
		st[p] = m - 1 if g > 0.0 else m
	}
	for j := m - 2; j >= 0; j -= 1 {
		dt := e.ex_t[j + 1] - e.ex_t[j]
		n_itm := 0
		mean, sq := 0.0, 0.0
		for p in 0 ..< n {
			U[p] *= math.exp_f64(e.ld[p * m + j + 1] - s * dt)
			ev[p] = _hw_ex_value(e, coef, j, e.x[p * m + j])
			if ev[p] > 0.0 {
				xv := e.x[p * m + j]
				itm[n_itm] = p
				n_itm += 1
				mean += xv
				sq += xv * xv
			}
		}
		if n_itm <= nb {continue}
		mean /= f64(n_itm)
		sd := math.sqrt_f64(max(sq / f64(n_itm) - mean * mean, 0.0))
		if !(sd > 0.0) {sd = 1.0}
		for k in 0 ..< n_itm {
			p := itm[k]
			zv := (e.x[p * m + j] - mean) / sd
			pw := 1.0
			for q in 0 ..< nb {
				X[k * nb + q] = pw
				pw *= zv
			}
			y[k] = U[p]
		}
		beta := _least_squares_regression(X, y, n_itm, nb, alloc)
		defer delete(beta, alloc)
		for k in 0 ..< n_itm {
			p := itm[k]
			zv := (e.x[p * m + j] - mean) / sd
			cont := 0.0
			for q := nb - 1; q >= 0; q -= 1 {cont = cont * zv + beta[q]}
			if ev[p] > cont {
				U[p] = ev[p]
				st[p] = j
			}
		}
	}

	opt, n_hit := 0.0, 0
	for p in 0 ..< n {
		opt += U[p] * math.exp_f64(e.ld[p * m] - s * e.ex_t[0])
		if st[p] < m {n_hit += 1}
	}
	opt /= f64(n)
	price = S0 - opt if e.is_call else S0 + opt
	return price, f64(n_hit) / f64(n)
}

// Price under shift s with the exercise dates frozen at `stop`
@(private)
_hw_bermudan_price_fixed :: proc(e: ^_HW_Bermudan, s: f64, stop: []int) -> f64 {
	S0 := _hw_bermudan_straight(e, s)
	m := len(e.ex_t)
	if m == 0 {return S0}
	coef := _hw_shift_coef(e, s, e.allocator)
	defer delete(coef, e.allocator)
	opt := 0.0
	for p in 0 ..< e.n_paths {
		j := stop[p]
		if j >= m {continue}
		ld := 0.0
		for k in 0 ..= j {ld += e.ld[p * m + k]}
		opt += _hw_ex_value(e, coef, j, e.x[p * m + j]) * math.exp_f64(ld - s * e.ex_t[j])
	}
	opt /= f64(e.n_paths)
	return S0 - opt if e.is_call else S0 + opt
}

// Parallel shift s with model price = market_price, by Illinois regula falsi on
// common random numbers; NaN if no s in [−1, 1] brackets the price
@(private)
_hw_bermudan_oas :: proc(e: ^_HW_Bermudan, market_price: f64) -> f64 {
	lo, hi := -0.05, 0.05
	p_lo, _ := _hw_bermudan_price(e, lo)
	p_hi, _ := _hw_bermudan_price(e, hi)
	for p_lo < market_price && lo > -1.0 {
		lo = max(2.0 * lo, -1.0)
		p_lo, _ = _hw_bermudan_price(e, lo)
	}
	for p_hi > market_price && hi < 1.0 {
		hi = min(2.0 * hi, 1.0)
		p_hi, _ = _hw_bermudan_price(e, hi)
	}
	if !(p_lo >= market_price && p_hi <= market_price) {return math.QNAN_F64}
	// Illinois regula falsi on f(s) = P(s) − market_price (f(lo) ≥ 0 ≥ f(hi))
	f_lo, f_hi := p_lo - market_price, p_hi - market_price
	side := 0
	for _ in 0 ..< 100 {
		if f_lo == f_hi {break}
		mid := hi - f_hi * (hi - lo) / (f_hi - f_lo)
		if !(mid > lo && mid < hi) {mid = 0.5 * (lo + hi)}
		p_mid, _ := _hw_bermudan_price(e, mid)
		f_mid := p_mid - market_price
		if abs(f_mid) <= 1e-12 * market_price || hi - lo <= 1e-12 {return mid}
		if f_mid > 0.0 {
			lo, f_lo = mid, f_mid
			if side == -1 {f_hi *= 0.5}
			side = -1
		} else {
			hi, f_hi = mid, f_mid
			if side == 1 {f_lo *= 0.5}
			side = 1
		}
	}
	return 0.5 * (lo + hi)
}

// ============================================================================
// OAS SOLVER
// ============================================================================
// Constant spread over the Hull-White short rate (a parallel shift of the
// initial curve) at which the model price equals market_price; NaN if no
// spread in [−100%, 100%] matches. n_steps is unused: the paths are sampled
// exactly at the call dates.
callable_bond_oas :: proc(
	bond: CallableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	market_price: f64,
	n_paths: int = 5000,
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	e := _hw_callable_build(bond, a, sigma, r0, P0T_func, n_paths, poly_degree, allocator)
	defer _hw_bermudan_free(&e)
	return _hw_bermudan_oas(&e, market_price)
}

@(private = "file")
_hw_callable_build :: proc(
	bond: CallableBond,
	a, sigma, r0: f64,
	P0T: proc(_: f64) -> f64,
	n_paths, degree: int,
	allocator: mem.Allocator,
) -> _HW_Bermudan {
	m := len(bond.call_schedule)
	t := make([]f64, m, allocator)
	k := make([]f64, m, allocator)
	defer {delete(t, allocator); delete(k, allocator)}
	for c, i in bond.call_schedule {
		t[i] = c.time
		k[i] = c.price
	}
	return _hw_bermudan_build(bond.face_value, bond.coupon_rate, bond.coupon_frequency, bond.maturity, t, k, true, a, sigma, r0, P0T, n_paths, degree, allocator)
}

// ============================================================================
// MAIN API: LSM FOR CALLABLE BONDS (HULL-WHITE 1F)
// ============================================================================
// Price at zero spread. r0 = f(0,0) makes the model fit P0T_func exactly; another
// r0 starts the short rate off the curve. Effective duration and convexity use
// ±50 bp parallel shifts on the same paths (the exercise policy is re-estimated
// for each shift; smaller shifts amplify the regression noise). n_steps is unused (exact sampling).
callable_bond_lsm_hw1f :: proc(
	bond: CallableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	n_paths: int = 10000,
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> CallableBondResult {
	e := _hw_callable_build(bond, a, sigma, r0, P0T_func, n_paths, poly_degree, allocator)
	defer _hw_bermudan_free(&e)

	price, call_prob := _hw_bermudan_price(&e, 0.0)
	straight_price := _hw_bermudan_straight(&e, 0.0)
	h := 0.005
	price_up, _ := _hw_bermudan_price(&e, h)
	price_dn, _ := _hw_bermudan_price(&e, -h)

	return CallableBondResult {
		price               = price,
		straight_bond_price = straight_price,
		embedded_call_value = straight_price - price,
		oas                 = 0.0,
		effective_duration  = -(price_up - price_dn) / (2.0 * h * price),
		effective_convexity = (price_up + price_dn - 2.0 * price) / (h * h * price),
		call_probability    = call_prob,
	}
}
