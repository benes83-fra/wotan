package finance

import l "../linalg"
import opt "../optimize"
import t "../tensor"
import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// HULL-WHITE (1-FACTOR) INTEREST RATE MODEL
// ============================================================================

HW_Params :: struct {
	a:     f64, // Mean reversion speed
	sigma: f64, // Short rate volatility
}

// ============================================================================
// Tensorized Analytical Caplet / Floorlet Pricing
// ============================================================================
// A caplet on [T_s, T_e] is (1 + δK) puts on the zero-coupon bond P(T_s, T_e)
// with strike 1/(1 + δK), δ = T_e − T_s. In a Gaussian short-rate model ln P(T_s,T_e)
// is normal, which gives, with F the simple forward rate and P = P(0, T_e):
//   caplet   = P [(1+δF) N(d1) − (1+δK) N(d2)]
//   floorlet = P [(1+δK) N(−d2) − (1+δF) N(−d1)]
//   d1,2 = (ln((1+δF)/(1+δK)) ± v/2) / √v
// where v is the variance of ln P(T_s, T_e) (Brigo–Mercurio §3.3.2 / §4.2.3).

// Constant N×1 tensor of ones, freed with the graph that uses it
@(private = "file")
_ones_tensor :: proc(n: int, allocator: mem.Allocator) -> ^t.Tensor {
	m := l.matrix_new(f64, n, 1, allocator)
	for i in 0 ..< n {m.data[i] = 1.0}
	one := t.tensor_new(m, false, allocator)
	one.owned_by_graph = true
	return one
}

@(private = "file")
_hw_capfloor_tensor :: proc(
	v, T_start, T_end, F, K, P, one: ^t.Tensor,
	is_cap: bool,
) -> ^t.Tensor {
	delta_T := t.tensor_sub(T_end, T_start)
	gF := t.tensor_add(one, t.tensor_mul(delta_T, F))
	gK := t.tensor_add(one, t.tensor_mul(delta_T, K))
	sd := t.tensor_sqrt(v)
	d1 := t.tensor_div(t.tensor_add(t.tensor_log(t.tensor_div(gF, gK)), t.tensor_scale(v, 0.5)), sd)
	d2 := t.tensor_sub(d1, sd)
	value: ^t.Tensor
	if is_cap {
		value = t.tensor_sub(t.tensor_mul(gF, t.tensor_norm_cdf(d1)), t.tensor_mul(gK, t.tensor_norm_cdf(d2)))
	} else {
		value = t.tensor_sub(
			t.tensor_mul(gK, t.tensor_norm_cdf(t.tensor_neg(d2))),
			t.tensor_mul(gF, t.tensor_norm_cdf(t.tensor_neg(d1))),
		)
	}
	return t.tensor_mul(P, value)
}

// Hull-White 1F: v = σ² B(a, δ)² (1 − e^{−2a T_s}) / (2a)
@(private = "file")
_hw1f_bond_var_tensor :: proc(a, sigma: f64, T_start, T_end, one: ^t.Tensor) -> ^t.Tensor {
	B := t.tensor_scale(t.tensor_sub(one, t.tensor_exp(t.tensor_scale(t.tensor_sub(T_end, T_start), -a))), 1.0 / a)
	vf := t.tensor_scale(t.tensor_sub(one, t.tensor_exp(t.tensor_scale(T_start, -2.0 * a))), 1.0 / (2.0 * a))
	return t.tensor_scale(t.tensor_mul(t.tensor_mul(B, B), vf), sigma * sigma)
}

// Caplet prices (notional 1) under Hull-White 1F. F: simple forward rates over
// [T_start, T_end], K: strikes, P: discount factors P(0, T_end). Free the result
// with tensor_free_graph.
hull_white_caplet_price_tensor :: proc(
	a: f64,
	sigma: f64,
	T_start: ^t.Tensor,
	T_end: ^t.Tensor,
	F: ^t.Tensor,
	K: ^t.Tensor,
	P: ^t.Tensor,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	one := _ones_tensor(T_start.data.rows, allocator)
	v := _hw1f_bond_var_tensor(a, sigma, T_start, T_end, one)
	return _hw_capfloor_tensor(v, T_start, T_end, F, K, P, one, true)
}

// ============================================================================
// Hull-White Calibration to Caplet Strip (Finite Difference Optimizer)
// ============================================================================

HW_CalibrationResult :: struct {
	params:     HW_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}

// Standalone helper to avoid illegal closure captures in Odin
_hw_eval_loss :: proc(
	a_val: f64,
	sigma_val: f64,
	T_start_t: ^t.Tensor,
	T_end_t: ^t.Tensor,
	F_t: ^t.Tensor,
	K_t: ^t.Tensor,
	P_t: ^t.Tensor,
	market_t: ^t.Tensor,
	allocator: mem.Allocator,
) -> f64 {
	loss_t := hull_white_caplet_price_tensor(
		a_val,
		sigma_val,
		T_start_t,
		T_end_t,
		F_t,
		K_t,
		P_t,
		allocator,
	)

	d := t.tensor_sub(loss_t, market_t)
	d_sq := t.tensor_mul(d, d)
	mse := t.tensor_mean(d_sq)
	r := t.tensor_sqrt(mse)

	res := r.data.data[0]

	t.tensor_free_graph(r)
	return res
}

// ----------------------------------------------------------------------------
// Objective adapters for the unified calibrator (optimize.calibrate)
// ----------------------------------------------------------------------------

@(private = "file")
_HW_Caplet_Ctx :: struct {
	T_start, T_end, F, K, P, market: ^t.Tensor,
	allocator:                       mem.Allocator,
}

@(private = "file")
_hw1f_objective :: proc(x: []f64, user_data: rawptr) -> f64 {
	c := (^_HW_Caplet_Ctx)(user_data)
	return _hw_eval_loss(x[0], x[1], c.T_start, c.T_end, c.F, c.K, c.P, c.market, c.allocator)
}

@(private = "file")
_hw2f_objective :: proc(x: []f64, user_data: rawptr) -> f64 {
	c := (^_HW_Caplet_Ctx)(user_data)
	p := HW2_Params{a = x[0], b = x[1], sigma = x[2], eta = x[3], rho = x[4]}
	return _hw2_eval_loss(p, c.T_start, c.T_end, c.F, c.K, c.P, c.market, c.allocator)
}

@(private = "file")
_HW2F_Joint_Ctx :: struct {
	r0:              f64,
	caps:            _HW_Caplet_Ctx,
	swaption_specs:  []SwaptionSpec,
	swaption_market: []f64,
	n_swaptions:     int,
}

@(private = "file")
_hw2f_joint_objective :: proc(x: []f64, user_data: rawptr) -> f64 {
	c := (^_HW2F_Joint_Ctx)(user_data)
	p := HW2_Params{a = x[0], b = x[1], sigma = x[2], eta = x[3], rho = x[4]}
	return _hw2f_joint_loss(
		p,
		c.r0,
		c.caps.T_start,
		c.caps.T_end,
		c.caps.F,
		c.caps.K,
		c.caps.P,
		c.caps.market,
		c.swaption_specs,
		c.swaption_market,
		c.n_swaptions,
		c.caps.allocator,
	)
}

calibrate_hull_white :: proc(
	market_prices: []f64,
	T_start: []f64,
	T_end: []f64,
	F: []f64,
	K: []f64,
	P: []f64,
	n_caplets: int,
	allocator: mem.Allocator = context.allocator,
) -> HW_CalibrationResult {
	T_start_t := _make_const_tensor(T_start, n_caplets, allocator)
	T_end_t := _make_const_tensor(T_end, n_caplets, allocator)
	F_t := _make_const_tensor(F, n_caplets, allocator)
	K_t := _make_const_tensor(K, n_caplets, allocator)
	P_t := _make_const_tensor(P, n_caplets, allocator)
	market_t := _make_const_tensor(market_prices, n_caplets, allocator)

	ctx := _HW_Caplet_Ctx{T_start_t, T_end_t, F_t, K_t, P_t, market_t, allocator}
	params := []opt.Calibration_Param {
		{"a", 0.05, 0.01, 0.5, .Log},
		{"sigma", 0.01, 0.001, 0.1, .Log},
	}
	config := opt.Calibration_Config {
		optimizer = {type = .Adam, learning_rate = 0.05, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8},
		max_iter = 300,
		loss_threshold = 1e-6,
		fd_step = 1e-5,
	}
	res := opt.calibrate(params, _hw1f_objective, &ctx, config, allocator)
	defer opt.calibration_result_free(&res)

	t.tensor_free(T_start_t)
	t.tensor_free(T_end_t)
	t.tensor_free(F_t)
	t.tensor_free(K_t)
	t.tensor_free(P_t)
	t.tensor_free(market_t)

	return HW_CalibrationResult {
		params = HW_Params{a = res.x[0], sigma = res.x[1]},
		rmse = res.loss,
		converged = res.converged,
		iterations = res.iterations,
	}
}

_make_const_tensor :: proc(data: []f64, n: int, allocator: mem.Allocator) -> ^t.Tensor {
	mat := l.matrix_new(f64, n, 1, allocator)
	for i in 0 ..< n {
		mat.data[i] = data[i]
	}
	return t.tensor_new(mat, false, allocator)
}
// ============================================================================
// G2++ / HULL-WHITE MONTE CARLO ENGINE (flat initial curve r0)
// ============================================================================
// r(t) = x(t) + y(t) + φ(t),  dx = −a x dt + σ dW1,  dy = −b y dt + η dW2,
// d⟨W1, W2⟩ = ρ dt, with φ fitted to P(0,t) = e^{−r0 t} (Brigo–Mercurio §4.2).
// Hull-White 1F is the case η = 0. (x, y, ∫(x + y)) is sampled exactly between
// event dates (it is jointly Gaussian), so there is no time-stepping error and
// the n_steps arguments of the Monte Carlo pricers are unused. Bonds are priced
// analytically on each path and payoffs are discounted along the path.

@(private = "file")
_g2_from_hw :: proc(p: HW_Params) -> HW2_Params {
	return HW2_Params{a = p.a, b = p.a, sigma = p.sigma, eta = 0.0, rho = 0.0}
}

// ∫_0^τ B(a,u) B(b,u) du
@(private = "file")
_g2_j :: proc(a, b, tau: f64) -> f64 {
	if abs(a * tau) < 1e-3 || abs(b * tau) < 1e-3 {
		// Simpson on the smooth integrand (small mean reversion)
		n := 64
		h := tau / f64(n)
		s := 0.0
		for i in 0 ..= n {
			u := f64(i) * h
			w := 1.0 if i == 0 || i == n else (4.0 if i % 2 == 1 else 2.0)
			s += w * _hw_b(a, u) * _hw_b(b, u)
		}
		return s * h / 3.0
	}
	return (tau - _hw_b(a, tau) - _hw_b(b, tau) + _hw_b(a + b, tau)) / (a * b)
}

// V(τ): variance of ∫_t^{t+τ} (x + y) du given F_t
@(private = "file")
_g2_v :: proc(p: HW2_Params, tau: f64) -> f64 {
	if tau <= 0.0 {return 0.0}
	v := p.sigma * p.sigma * _hw_k(p.a, tau)
	if p.eta != 0.0 {
		v += p.eta * p.eta * _hw_k(p.b, tau) + 2.0 * p.rho * p.sigma * p.eta * _g2_j(p.a, p.b, tau)
	}
	return v
}

// P(t, T) = P(0,T)/P(0,t) · exp(½[V(T−t) − V(T) + V(t)] − B(a,T−t) x − B(b,T−t) y),
// as ln-coefficients so that ln P = lc − ba x − bb y
@(private = "file")
_G2_Bond :: struct {
	lc, ba, bb: f64,
}

@(private = "file")
_g2_bond :: proc(p: HW2_Params, r0, t_, T: f64) -> _G2_Bond {
	tau := T - t_
	if tau <= 0.0 {return {}}
	return _G2_Bond {
		lc = -r0 * tau + 0.5 * (_g2_v(p, tau) - _g2_v(p, T) + _g2_v(p, t_)),
		ba = _hw_b(p.a, tau),
		bb = _hw_b(p.b, tau),
	}
}

@(private = "file")
_g2_bond_at :: #force_inline proc(c: _G2_Bond, x, y: f64) -> f64 {
	return math.exp_f64(c.lc - c.ba * x - c.bb * y)
}

// Paths at the event dates: factors and the log discount from the previous
// date (date −1 = time 0), path-major
@(private = "file")
_G2_Paths :: struct {
	n, m:     int,
	x, y, ld: []f64,
}

@(private = "file")
_g2_paths_free :: proc(ps: ^_G2_Paths, allocator: mem.Allocator) {
	delete(ps.x, allocator)
	delete(ps.y, allocator)
	delete(ps.ld, allocator)
}

// Standard normals for n_paths (antithetic pairs) and m dates: 3 per date and base path
@(private = "file")
_g2_normals :: proc(n_paths, m: int, allocator: mem.Allocator) -> []f64 {
	half := max(1, (n_paths + 1) / 2)
	z := make([]f64, half * m * 3, allocator)
	for i in 0 ..< len(z) {z[i] = rand.float64_normal(0.0, 1.0)}
	return z
}

@(private = "file")
_g2_simulate :: proc(
	p: HW2_Params,
	r0: f64,
	dates: []f64,
	z: []f64,
	allocator: mem.Allocator,
) -> _G2_Paths {
	m := len(dates)
	half := len(z) / max(3 * m, 1)
	ps := _G2_Paths{n = 2 * half, m = m}
	ps.x = make([]f64, ps.n * m, allocator)
	ps.y = make([]f64, ps.n * m, allocator)
	ps.ld = make([]f64, ps.n * m, allocator)
	a, b, s, e, rho := p.a, p.b, p.sigma, p.eta, p.rho

	// Per-interval transition: means, Cholesky factor of Cov(x', y', I), deterministic drift
	Step :: struct {
		ea, eb, ba, bb, det: f64,
		L:                   [6]f64, // lower triangle: L00, L10, L11, L20, L21, L22
	}
	steps := make([]Step, m, allocator)
	defer delete(steps, allocator)
	tp := 0.0
	for j in 0 ..< m {
		dt := max(dates[j] - tp, 0.0)
		ba, bb, bab := _hw_b(a, dt), _hw_b(b, dt), _hw_b(a + b, dt)
		cxx := s * s * _hw_b(2.0 * a, dt)
		cyy := e * e * _hw_b(2.0 * b, dt)
		cxy := rho * s * e * bab
		cII := _g2_v(p, dt)
		cxI := 0.5 * s * s * ba * ba
		cyI := 0.5 * e * e * bb * bb
		if e != 0.0 {
			cxI += rho * s * e * (ba - bab) / b
			cyI += rho * s * e * (bb - bab) / a
		}
		// PSD Cholesky (a zero pivot zeroes its column)
		C := [3][3]f64{{cxx, cxy, cxI}, {cxy, cyy, cyI}, {cxI, cyI, cII}}
		Lm: [3][3]f64
		for i in 0 ..< 3 {
			for k in 0 ..= i {
				sum := C[i][k]
				for q in 0 ..< k {sum -= Lm[i][q] * Lm[k][q]}
				if i == k {
					Lm[i][i] = math.sqrt_f64(sum) if sum > 1e-14 * max(C[i][i], 1e-300) else 0.0
				} else {
					Lm[i][k] = sum / Lm[k][k] if Lm[k][k] > 0.0 else 0.0
				}
			}
		}
		steps[j] = Step {
			ea  = math.exp_f64(-a * dt),
			eb  = math.exp_f64(-b * dt),
			ba  = ba,
			bb  = bb,
			det = -r0 * dt - 0.5 * (_g2_v(p, dates[j]) - _g2_v(p, tp)),
			L   = {Lm[0][0], Lm[1][0], Lm[1][1], Lm[2][0], Lm[2][1], Lm[2][2]},
		}
		tp = dates[j]
	}

	for q in 0 ..< half {
		for sgn, side in ([]f64{1.0, -1.0}) {
			row := (q + side * half) * m
			x, y := 0.0, 0.0
			for j in 0 ..< m {
				st := &steps[j]
				z1 := sgn * z[(q * m + j) * 3]
				z2 := sgn * z[(q * m + j) * 3 + 1]
				z3 := sgn * z[(q * m + j) * 3 + 2]
				integral := x * st.ba + y * st.bb + st.L[3] * z1 + st.L[4] * z2 + st.L[5] * z3
				x = x * st.ea + st.L[0] * z1
				y = y * st.eb + st.L[1] * z1 + st.L[2] * z2
				ps.x[row + j] = x
				ps.y[row + j] = y
				ps.ld[row + j] = st.det - integral
			}
		}
	}
	return ps
}

// Cap (is_cap) or floor: Σ_i δ_i (±(L_i − K))⁺ paid at T_end[i], valued at the
// fixing T_start[i] as δ (±(L − K))⁺ P(T_s, T_e), L = (1/P(T_s,T_e) − 1)/δ.
// Periods with T_start ≤ 0 are valued on today's curve.
@(private = "file")
_g2_capfloor_mc :: proc(
	p: HW2_Params,
	r0, K: f64,
	T_start, T_end, delta: []f64,
	n: int,
	is_cap: bool,
	z: []f64,
	allocator: mem.Allocator,
) -> f64 {
	payoff :: proc(P, d, K: f64, is_cap: bool) -> f64 {
		L := (1.0 / P - 1.0) / d
		return d * max(L - K if is_cap else K - L, 0.0) * P
	}
	dates := make([dynamic]f64, 0, n, allocator)
	defer delete(dates)
	idx := make([]int, n, allocator) // date index of each period (−1: already fixed)
	defer delete(idx, allocator)
	bonds := make([]_G2_Bond, n, allocator)
	defer delete(bonds, allocator)
	fixed := 0.0
	for i in 0 ..< n {
		bonds[i] = _g2_bond(p, r0, max(T_start[i], 0.0), T_end[i])
		idx[i] = -1
		if T_start[i] <= 0.0 {
			fixed += payoff(math.exp_f64(-r0 * T_end[i]) / math.exp_f64(-r0 * max(T_start[i], 0.0)), delta[i], K, is_cap)
			continue
		}
		if len(dates) == 0 || T_start[i] > dates[len(dates) - 1] + 1e-12 {append(&dates, T_start[i])}
		idx[i] = len(dates) - 1
	}
	if len(dates) == 0 {return fixed}
	m := len(dates)
	ps := _g2_simulate(p, r0, dates[:], z, allocator)
	defer _g2_paths_free(&ps, allocator)

	total := 0.0
	for q in 0 ..< ps.n {
		ld := 0.0
		j := 0
		for i in 0 ..< n {
			if idx[i] < 0 {continue}
			for j <= idx[i] {
				ld += ps.ld[q * m + j]
				j += 1
			}
			k := idx[i]
			P := _g2_bond_at(bonds[i], ps.x[q * m + k], ps.y[q * m + k])
			total += payoff(P, delta[i], K, is_cap) * math.exp_f64(ld)
		}
	}
	return fixed + total / f64(ps.n)
}

// Payer (receiver) swaption value at T_exp on a swap paying K·δ at
// T_exp + jδ, j = 1..n, with n = round((T_maturity − T_exp)/δ)
@(private = "file")
_g2_swaption_mc :: proc(
	p: HW2_Params,
	r0, K, T_exp, T_maturity, delta: f64,
	is_payer: bool,
	z: []f64,
	allocator: mem.Allocator,
) -> f64 {
	n_pay := int(math.round((T_maturity - T_exp) / delta))
	if n_pay <= 0 || T_exp <= 0.0 {return 0.0}
	bonds := make([]_G2_Bond, n_pay, allocator)
	defer delete(bonds, allocator)
	for j in 0 ..< n_pay {bonds[j] = _g2_bond(p, r0, T_exp, T_exp + f64(j + 1) * delta)}
	ps := _g2_simulate(p, r0, []f64{T_exp}, z, allocator)
	defer _g2_paths_free(&ps, allocator)

	total := 0.0
	for q in 0 ..< ps.n {
		x, y := ps.x[q], ps.y[q]
		annuity := 0.0
		for j in 0 ..< n_pay {annuity += delta * _g2_bond_at(bonds[j], x, y)}
		swap := 1.0 - _g2_bond_at(bonds[n_pay - 1], x, y) - K * annuity
		if !is_payer {swap = -swap}
		total += max(swap, 0.0) * math.exp_f64(ps.ld[q])
	}
	return total / f64(ps.n)
}

// Price and central-difference Greeks on common random numbers: delta_r is the
// derivative with respect to the flat rate r0 (a parallel curve shift), vega
// with respect to σ
@(private = "file")
_G2_Pricer :: struct {
	kind:                  enum {
		Cap,
		Floor,
		Swaption,
	},
	K:                     f64,
	T_start, T_end, delta: []f64,
	n:                     int,
	T_exp, T_mat, dlt:     f64,
	is_payer:              bool,
}

@(private = "file")
_g2_price :: proc(c: ^_G2_Pricer, p: HW2_Params, r0: f64, z: []f64, allocator: mem.Allocator) -> f64 {
	switch c.kind {
	case .Cap, .Floor:
		return _g2_capfloor_mc(p, r0, c.K, c.T_start, c.T_end, c.delta, c.n, c.kind == .Cap, z, allocator)
	case .Swaption:
		return _g2_swaption_mc(p, r0, c.K, c.T_exp, c.T_mat, c.dlt, c.is_payer, z, allocator)
	}
	return 0.0
}

@(private = "file")
_g2_price_greeks :: proc(
	c: ^_G2_Pricer,
	p: HW2_Params,
	r0: f64,
	n_paths, n_dates: int,
	allocator: mem.Allocator,
) -> (
	price, delta_r, vega: f64,
) {
	z := _g2_normals(n_paths, max(n_dates, 1), allocator)
	defer delete(z, allocator)
	price = _g2_price(c, p, r0, z, allocator)
	h_r := max(0.01 * abs(r0), 1e-4)
	delta_r = (_g2_price(c, p, r0 + h_r, z, allocator) - _g2_price(c, p, r0 - h_r, z, allocator)) / (2.0 * h_r)
	h_s := max(0.01 * p.sigma, 1e-6)
	up, dn := p, p
	up.sigma += h_s
	dn.sigma -= h_s
	vega = (_g2_price(c, up, r0, z, allocator) - _g2_price(c, dn, r0, z, allocator)) / (2.0 * h_s)
	return
}

// Number of distinct positive fixing dates (the simulation grid of a strip)
@(private = "file")
_g2_count_dates :: proc(T_start: []f64, n: int) -> int {
	m := 0
	last := 0.0
	for i in 0 ..< n {
		if T_start[i] > 0.0 && (m == 0 || T_start[i] > last + 1e-12) {
			m += 1
			last = T_start[i]
		}
	}
	return m
}

// ============================================================================
// Hull-White Monte Carlo: Cap Option (with CRN Greeks)
// ============================================================================
// Cap Σ δ_i (L_i − K)⁺ under Hull-White 1F fitted to the flat curve e^{−r0 t}.
// T_start must be non-decreasing. Matches hw_price_cap_floor_strip up to Monte
// Carlo error. n_steps is unused (exact sampling at the fixing dates).
hw_mc_cap_option :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_caplets: int,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta_r: f64,
	vega: f64,
) {
	c := _G2_Pricer{kind = .Cap, K = K, T_start = T_start, T_end = T_end, delta = delta, n = n_caplets}
	return _g2_price_greeks(&c, _g2_from_hw(params), r0, n_paths, _g2_count_dates(T_start, n_caplets), allocator)
}

// ============================================================================
// HULL-WHITE 2-FACTOR INTEREST RATE MODEL (Brigo-Mercurio formulation)
// ============================================================================

HW2_Params :: struct {
	a:     f64, // Mean reversion of short rate (fast)
	b:     f64, // Mean reversion of second factor (slow)
	sigma: f64, // Volatility of short rate
	eta:   f64, // Volatility of second factor
	rho:   f64, // Correlation between Brownian motions
}

// ============================================================================
// Tensorized Analytical Caplet Pricing - 2 Factor (G2++)
// ============================================================================
// v = σ²/(2a) B_a² (1 − e^{−2aT}) + η²/(2b) B_b² (1 − e^{−2bT})
//     + 2ρση/(a+b) B_a B_b (1 − e^{−(a+b)T}),  B_k = B(k, T_e − T_s), T = T_s
// (Brigo–Mercurio eq. 4.31), in the caplet formula above.
hull_white_2f_caplet_price_tensor :: proc(
	params: HW2_Params,
	T_start: ^t.Tensor,
	T_end: ^t.Tensor,
	F: ^t.Tensor,
	K: ^t.Tensor,
	P: ^t.Tensor,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	one := _ones_tensor(T_start.data.rows, allocator)
	delta_T := t.tensor_sub(T_end, T_start)
	bk :: proc(one, delta_T: ^t.Tensor, k: f64) -> ^t.Tensor {
		return t.tensor_scale(t.tensor_sub(one, t.tensor_exp(t.tensor_scale(delta_T, -k))), 1.0 / k)
	}
	vk :: proc(one, T_start: ^t.Tensor, k: f64) -> ^t.Tensor {
		return t.tensor_scale(t.tensor_sub(one, t.tensor_exp(t.tensor_scale(T_start, -k))), 1.0 / k)
	}
	a, b := params.a, params.b
	B_a, B_b := bk(one, delta_T, a), bk(one, delta_T, b)
	term1 := t.tensor_scale(t.tensor_mul(t.tensor_mul(B_a, B_a), vk(one, T_start, 2.0 * a)), params.sigma * params.sigma)
	term2 := t.tensor_scale(t.tensor_mul(t.tensor_mul(B_b, B_b), vk(one, T_start, 2.0 * b)), params.eta * params.eta)
	term3 := t.tensor_scale(t.tensor_mul(t.tensor_mul(B_a, B_b), vk(one, T_start, a + b)), 2.0 * params.rho * params.sigma * params.eta)
	v := t.tensor_add(t.tensor_add(term1, term2), term3)
	return _hw_capfloor_tensor(v, T_start, T_end, F, K, P, one, true)
}

// ============================================================================
// Hull-White 2-Factor Calibration (Finite Difference Optimizer)
// ============================================================================

HW2_CalibrationResult :: struct {
	params:     HW2_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}

// Standalone helper to avoid illegal closure captures in Odin
_hw2_eval_loss :: proc(
	p: HW2_Params,
	T_start_t: ^t.Tensor,
	T_end_t: ^t.Tensor,
	F_t: ^t.Tensor,
	K_t: ^t.Tensor,
	P_t: ^t.Tensor,
	market_t: ^t.Tensor,
	allocator: mem.Allocator,
) -> f64 {
	loss_t := hull_white_2f_caplet_price_tensor(p, T_start_t, T_end_t, F_t, K_t, P_t, allocator)

	d := t.tensor_sub(loss_t, market_t)
	d_sq := t.tensor_mul(d, d)
	mse := t.tensor_mean(d_sq)
	r := t.tensor_sqrt(mse)

	res := r.data.data[0]

	t.tensor_free_graph(r)
	return res
}

calibrate_hull_white_2f :: proc(
	market_prices: []f64,
	T_start: []f64,
	T_end: []f64,
	F: []f64,
	K: []f64,
	P: []f64,
	n_caplets: int,
	allocator: mem.Allocator = context.allocator,
) -> HW2_CalibrationResult {
	T_start_t := _make_const_tensor(T_start, n_caplets, allocator)
	T_end_t := _make_const_tensor(T_end, n_caplets, allocator)
	F_t := _make_const_tensor(F, n_caplets, allocator)
	K_t := _make_const_tensor(K, n_caplets, allocator)
	P_t := _make_const_tensor(P, n_caplets, allocator)
	market_t := _make_const_tensor(market_prices, n_caplets, allocator)

	ctx := _HW_Caplet_Ctx{T_start_t, T_end_t, F_t, K_t, P_t, market_t, allocator}
	params := []opt.Calibration_Param {
		{"a", 0.1, 0.01, 1.0, .Log},
		{"b", 0.01, 0.01, 1.0, .Log},
		{"sigma", 0.01, 0.001, 0.05, .Log},
		{"eta", 0.005, 0.001, 0.05, .Log},
		{"rho", -0.3, -0.99, 0.99, .Linear},
	}
	config := opt.Calibration_Config {
		optimizer = {type = .Adam, learning_rate = 0.02, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8},
		max_iter = 400,
		loss_threshold = 1e-6,
		fd_step = 1e-4,
	}
	res := opt.calibrate(params, _hw2f_objective, &ctx, config, allocator)
	defer opt.calibration_result_free(&res)

	t.tensor_free(T_start_t)
	t.tensor_free(T_end_t)
	t.tensor_free(F_t)
	t.tensor_free(K_t)
	t.tensor_free(P_t)
	t.tensor_free(market_t)

	return HW2_CalibrationResult {
		params = HW2_Params{a = res.x[0], b = res.x[1], sigma = res.x[2], eta = res.x[3], rho = res.x[4]},
		rmse = res.loss,
		converged = res.converged,
		iterations = res.iterations,
	}
}

// ============================================================================
// Hull-White 2-Factor Monte Carlo: Cap Option (with CRN Greeks)
// ============================================================================
// Cap under G2++ fitted to the flat curve e^{−r0 t}; vega is ∂/∂σ (first factor).
// Matches hull_white_2f_caplet_price_tensor up to Monte Carlo error. n_steps is
// unused (exact sampling at the fixing dates).
hw2_mc_cap_option :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_caplets: int,
	params: HW2_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta_r: f64,
	vega: f64,
) {
	c := _G2_Pricer{kind = .Cap, K = K, T_start = T_start, T_end = T_end, delta = delta, n = n_caplets}
	return _g2_price_greeks(&c, params, r0, n_paths, _g2_count_dates(T_start, n_caplets), allocator)
}
// ============================================================================
// SWAPTION PRICING & JOINT CALIBRATION (Caps + Swaptions)
// ============================================================================

SwaptionSpec :: struct {
	T_exp:       f64,
	swap_length: int, // Number of annual payments
	F:           f64, // Forward swap rate
	K:           f64, // Strike
	annuity:     f64, // PV01 / Annuity factor
}
// Total normal (Bachelier) variance of the forward swap rate at expiry under
// G2++, frozen-weights approximation (Schrager-Pelsser style).
//
// With bond sensitivities dP(t,T_i)/dx = -B_a(T_i - t) P frozen on today's
// curve P(0,t) = exp(-r0 t), the swap rate responds to the factors with
//   C_x(t) = [-B_a(T0-t) P(T0) + B_a(Tn-t) P(Tn) + F sum_i B_a(Ti-t) P(Ti)] / A
//          = alpha_x + kappa_x e^{a t}     (alpha_x = 0 when F is the par rate)
// and the variance integrates in closed form over [0, T]. Matches the exact
// G2++ swaption price to ~0.03% (see tests/hull_white_test.odin).
compute_hw2f_swap_variance :: proc(
	spec: SwaptionSpec,
	params: HW2_Params,
	r0: f64, // flat continuously compounded rate for discount factors
) -> f64 {
	a, b := params.a, params.b
	sigma, eta, rho := params.sigma, params.eta, params.rho
	T := spec.T_exp
	S0, A0 := spec.F, spec.annuity

	// Payment schedule: annual, T_i = T + i, i = 1..swap_length
	P_T0 := math.exp_f64(-r0 * T)
	P_Tn := math.exp_f64(-r0 * (T + f64(spec.swap_length)))
	G_a := -P_T0 * math.exp_f64(-a * T) + P_Tn * math.exp_f64(-a * (T + f64(spec.swap_length)))
	G_b := -P_T0 * math.exp_f64(-b * T) + P_Tn * math.exp_f64(-b * (T + f64(spec.swap_length)))
	H := -P_T0 + P_Tn // level part of the numerator
	for k in 1 ..= spec.swap_length {
		T_i := T + f64(k)
		P_Ti := math.exp_f64(-r0 * T_i)
		G_a += S0 * P_Ti * math.exp_f64(-a * T_i)
		G_b += S0 * P_Ti * math.exp_f64(-b * T_i)
		H += S0 * P_Ti
	}
	// C_x(t) = alpha_x + kappa_x e^{a t}, C_y(t) = alpha_y + kappa_y e^{b t}
	alpha_x, kappa_x := H / (a * A0), -G_a / (a * A0)
	alpha_y, kappa_y := H / (b * A0), -G_b / (b * A0)

	// integral_0^T (alpha + kappa e^{k t})(alpha' + kappa' e^{k' t}) dt
	cross :: proc(al, ka, k, al2, ka2, k2, T: f64) -> f64 {
		return al * al2 * T + al * ka2 * (math.exp_f64(k2 * T) - 1.0) / k2 + al2 * ka * (math.exp_f64(k * T) - 1.0) / k + ka * ka2 * (math.exp_f64((k + k2) * T) - 1.0) / (k + k2)
	}
	return sigma * sigma * cross(alpha_x, kappa_x, a, alpha_x, kappa_x, a, T) +
		eta * eta * cross(alpha_y, kappa_y, b, alpha_y, kappa_y, b, T) +
		2.0 * rho * sigma * eta * cross(alpha_x, kappa_x, a, alpha_y, kappa_y, b, T)
}

// Price a strip of payer swaptions under G2++ with the Bachelier (normal)
// formula on the frozen-weights swap-rate variance.
hw2f_swaption_price_strip :: proc(
	params: HW2_Params,
	swaptions: []SwaptionSpec,
	n_swaptions: int,
	r0: f64,
	allocator: mem.Allocator,
) -> []f64 {
	prices := make([]f64, n_swaptions, allocator)

	for i in 0 ..< n_swaptions {
		spec := swaptions[i]
		swap_var := compute_hw2f_swap_variance(spec, params, r0)

		if swap_var < 1e-18 {
			prices[i] = spec.annuity * math.max(spec.F - spec.K, 0.0)
			continue
		}

		sd := math.sqrt_f64(swap_var)
		d := (spec.F - spec.K) / sd
		N_d := 0.5 * (1.0 + math.erf(d / math.sqrt_f64(2.0)))
		phi_d := math.exp_f64(-0.5 * d * d) / math.sqrt_f64(2.0 * math.PI)
		prices[i] = spec.annuity * ((spec.F - spec.K) * N_d + sd * phi_d)
	}
	return prices
}

// Joint loss function: Caps RMSE + Swaptions RMSE
_hw2f_joint_loss :: proc(
	params: HW2_Params,
	r0: f64,
	// Caplet data
	cap_T_start_t: ^t.Tensor,
	cap_T_end_t: ^t.Tensor,
	cap_F_t: ^t.Tensor,
	cap_K_t: ^t.Tensor,
	cap_P_t: ^t.Tensor,
	cap_market_t: ^t.Tensor,
	// Swaption data
	swaption_specs: []SwaptionSpec,
	swaption_market: []f64,
	n_swaptions: int,
	allocator: mem.Allocator,
) -> f64 {
	// 1. Caplet loss (SIMD tensor)
	cap_model_t := hull_white_2f_caplet_price_tensor(
		params,
		cap_T_start_t,
		cap_T_end_t,
		cap_F_t,
		cap_K_t,
		cap_P_t,
		allocator,
	)
	// Relative errors: caps (~1e-4) and swaptions (~1e-2) are priced on very
	// different scales; relative RMSE lets both instruments inform the fit.
	cap_diff := t.tensor_div(t.tensor_sub(cap_model_t, cap_market_t), cap_market_t)
	cap_diff_sq := t.tensor_mul(cap_diff, cap_diff)
	cap_mse := t.tensor_mean(cap_diff_sq)
	cap_rmse := t.tensor_sqrt(cap_mse)
	cap_loss := cap_rmse.data.data[0]
	t.tensor_free_graph(cap_rmse)

	// 2. Swaption loss (analytical)
	swaption_model := hw2f_swaption_price_strip(params, swaption_specs, n_swaptions, r0, allocator)
	defer delete(swaption_model, allocator)

	swaption_sse := 0.0
	for i in 0 ..< n_swaptions {
		err := (swaption_model[i] - swaption_market[i]) / swaption_market[i]
		swaption_sse += err * err
	}
	swaption_rmse := math.sqrt_f64(swaption_sse / f64(n_swaptions))

	// 3. Equal weight between caps and swaptions. No prior: a fixed-centre prior
	// dominated the (small) data loss and pulled the fit toward its centres.
	// a > b is enforced by the projection in calibrate_hull_white_2f_joint.
	return cap_loss + swaption_rmse
}

// ============================================================================
// JOINT CALIBRATION: Caps + Swaptions
// ============================================================================

HW2F_JointResult :: struct {
	params:     HW2_Params,
	rmse:       f64, // relative RMSE of caplets + relative RMSE of swaptions (dimensionless)
	converged:  bool,
	iterations: int,
}
calibrate_hull_white_2f_joint :: proc(
	cap_market: []f64, // Caplet data
	cap_T_start: []f64,
	cap_T_end: []f64,
	cap_F: []f64,
	cap_K: []f64,
	cap_P: []f64,
	n_caplets: int,
	// Swaption data
	swaption_specs: []SwaptionSpec,
	swaption_market: []f64,
	n_swaptions: int,
	r0: f64, // <-- NEW: pass r0 for proper annuity weighting
	allocator: mem.Allocator = context.allocator,
) -> HW2F_JointResult {

	// Load caplet tensors
	cap_T_start_t := _make_const_tensor(cap_T_start, n_caplets, allocator)
	cap_T_end_t := _make_const_tensor(cap_T_end, n_caplets, allocator)
	cap_F_t := _make_const_tensor(cap_F, n_caplets, allocator)
	cap_K_t := _make_const_tensor(cap_K, n_caplets, allocator)
	cap_P_t := _make_const_tensor(cap_P, n_caplets, allocator)
	cap_market_t := _make_const_tensor(cap_market, n_caplets, allocator)

	ctx := _HW2F_Joint_Ctx{
		r0 = r0,
		caps = {cap_T_start_t, cap_T_end_t, cap_F_t, cap_K_t, cap_P_t, cap_market_t, allocator},
		swaption_specs = swaption_specs,
		swaption_market = swaption_market,
		n_swaptions = n_swaptions,
	}
	// Starting point close to realistic IR parameters, with a > b enforced
	params := []opt.Calibration_Param {
		{"a", 0.10, 0.02, 1.0, .Log},
		{"b", 0.03, 0.005, 1.0, .Log}, // tighter bound b <= a - 0.01 applied by `project`
		{"sigma", 0.008, 0.001, 0.05, .Log},
		{"eta", 0.005, 0.001, 0.05, .Log},
		{"rho", -0.4, -0.99, 0.99, .Linear},
	}
	config := opt.Calibration_Config {
		optimizer = {type = .Adam, learning_rate = 0.05, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8},
		max_iter = 600, // more iterations for the 5D landscape
		loss_threshold = 1e-6,
		fd_step = 1e-4,
		project = proc(x: []f64) {x[1] = math.max(0.005, math.min(x[0] - 0.01, x[1]))}, // b < a
	}
	res := opt.calibrate(params, _hw2f_joint_objective, &ctx, config, allocator)
	defer opt.calibration_result_free(&res)

	t.tensor_free(cap_T_start_t)
	t.tensor_free(cap_T_end_t)
	t.tensor_free(cap_F_t)
	t.tensor_free(cap_K_t)
	t.tensor_free(cap_P_t)
	t.tensor_free(cap_market_t)

	return HW2F_JointResult {
		params = HW2_Params{a = res.x[0], b = res.x[1], sigma = res.x[2], eta = res.x[3], rho = res.x[4]},
		rmse = res.loss,
		converged = res.converged,
		iterations = res.iterations,
	}
}
// ============================================================================
// BERMUDAN SWAPTION PRICING (HW 2-FACTOR)
// ============================================================================

// G2++ zero-coupon bond P(t, T) on the flat curve e^{−r0 t}, given the short rate
// r_t and the second factor x_t (the y factor of Brigo–Mercurio); the first factor
// is r_t − φ(t) − x_t with φ(t) = r0 + σ²/2 B_a(t)² + η²/2 B_b(t)² + ρση B_a(t) B_b(t).
hw2f_zero_coupon_bond :: proc(
	t: f64,
	T: f64,
	r_t: f64,
	x_t: f64,
	params: HW2_Params,
	r0: f64,
) -> f64 {
	if T - t <= 0.0 {return 1.0}
	p := params
	ba, bb := _hw_b(p.a, t), _hw_b(p.b, t)
	phi := r0 + 0.5 * p.sigma * p.sigma * ba * ba + 0.5 * p.eta * p.eta * bb * bb + p.rho * p.sigma * p.eta * ba * bb
	return _g2_bond_at(_g2_bond(p, r0, t, T), r_t - phi - x_t, x_t)
}

// Hull-White 2-Factor Monte Carlo: Bermudan payer swaption. Exercising at
// exercise_dates[i] enters a swap paying K·δ at t + δ, t + 2δ, …, maturity and
// receiving floating (worth 1 − P(t, maturity)). Longstaff–Schwartz on exact
// G2++ paths (flat curve e^{−r0 t}), regressing on 1, x, y, x², xy, y² and the
// exercise value over in-the-money paths (1, x, x² and the exercise value when
// η = 0, the Hull-White 1F case). n_steps is unused.
hw2f_mc_bermudan_option :: proc(
	r0: f64,
	K: f64,
	exercise_dates: []f64, // e.g., [1.0, 2.0, 3.0, 4.0, 5.0]
	maturity: f64, // e.g., 10.0
	delta: f64, // e.g., 1.0
	params: HW2_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	dates := make([dynamic]f64, 0, len(exercise_dates), allocator)
	defer delete(dates)
	for d in exercise_dates {
		if d > 0.0 && d < maturity - 1e-12 {append(&dates, d)}
	}
	for j in 1 ..< len(dates) {
		for q := j; q > 0 && dates[q] < dates[q - 1]; q -= 1 {dates[q], dates[q - 1] = dates[q - 1], dates[q]}
	}
	m := len(dates)
	if m == 0 {return 0.0}

	// Bonds from each exercise date to its fixed-leg payment dates
	Leg :: struct {
		bonds: []_G2_Bond,
	}
	legs := make([]Leg, m, allocator)
	defer {
		for lg in legs {delete(lg.bonds, allocator)}
		delete(legs, allocator)
	}
	for j in 0 ..< m {
		n_pay := max(int(math.round((maturity - dates[j]) / delta)), 1)
		legs[j].bonds = make([]_G2_Bond, n_pay, allocator)
		for k in 0 ..< n_pay {
			Tk := min(dates[j] + f64(k + 1) * delta, maturity)
			legs[j].bonds[k] = _g2_bond(params, r0, dates[j], Tk)
		}
	}
	exercise :: proc(lg: Leg, delta, K, x, y: f64) -> f64 {
		annuity := 0.0
		for c in lg.bonds {annuity += delta * _g2_bond_at(c, x, y)}
		return 1.0 - _g2_bond_at(lg.bonds[len(lg.bonds) - 1], x, y) - K * annuity
	}

	z := _g2_normals(n_paths, m, allocator)
	defer delete(z, allocator)
	ps := _g2_simulate(params, r0, dates[:], z, allocator)
	defer _g2_paths_free(&ps, allocator)
	n := ps.n

	NB :: 7
	U := make([]f64, n, allocator)
	ev := make([]f64, n, allocator)
	itm := make([]int, n, allocator)
	X := make([]f64, n * NB, allocator)
	yv := make([]f64, n, allocator)
	defer {
		delete(U, allocator); delete(ev, allocator); delete(itm, allocator)
		delete(X, allocator); delete(yv, allocator)
	}
	for q in 0 ..< n {
		U[q] = max(exercise(legs[m - 1], delta, K, ps.x[q * m + m - 1], ps.y[q * m + m - 1]), 0.0)
	}
	for j := m - 2; j >= 0; j -= 1 {
		n_itm := 0
		sx, sy := 0.0, 0.0
		for q in 0 ..< n {
			U[q] *= math.exp_f64(ps.ld[q * m + j + 1])
			ev[q] = exercise(legs[j], delta, K, ps.x[q * m + j], ps.y[q * m + j])
			if ev[q] > 0.0 {
				itm[n_itm] = q
				n_itm += 1
			}
			sx += ps.x[q * m + j] * ps.x[q * m + j]
			sy += ps.y[q * m + j] * ps.y[q * m + j]
		}
		// Factors scaled to O(1); without a second factor (η = 0) y ≡ 0 and its
		// columns are dropped (they would make the regression singular)
		sx = math.sqrt_f64(sx / f64(n))
		sy = math.sqrt_f64(sy / f64(n))
		has_y := sy > 1e-300
		nb := NB if has_y else 4
		if n_itm <= nb {continue}
		if !(sx > 0.0) {sx = 1.0}
		if !has_y {sy = 1.0}
		basis :: proc(row: []f64, x, y, e: f64, has_y: bool) {
			if has_y {
				row[0], row[1], row[2], row[3] = 1.0, x, y, x * x
				row[4], row[5], row[6] = x * y, y * y, e
			} else {
				row[0], row[1], row[2], row[3] = 1.0, x, x * x, e
			}
		}
		for k in 0 ..< n_itm {
			q := itm[k]
			basis(X[k * nb:(k + 1) * nb], ps.x[q * m + j] / sx, ps.y[q * m + j] / sy, ev[q] * 100.0, has_y)
			yv[k] = U[q]
		}
		beta := _least_squares_regression(X, yv, n_itm, nb, allocator)
		defer delete(beta, allocator)
		row: [NB]f64
		for k in 0 ..< n_itm {
			q := itm[k]
			basis(row[:], ps.x[q * m + j] / sx, ps.y[q * m + j] / sy, ev[q] * 100.0, has_y)
			cont := 0.0
			for c in 0 ..< nb {cont += beta[c] * row[c]}
			if ev[q] > cont {U[q] = ev[q]}
		}
	}
	total := 0.0
	for q in 0 ..< n {total += U[q] * math.exp_f64(ps.ld[q * m])}
	return total / f64(n)
}
// ============================================================================
// 1. FLOORLET & FLOOR PRICING (HW 1F)
// ============================================================================

// Floorlet prices (notional 1) under Hull-White 1F; same inputs as
// hull_white_caplet_price_tensor (cap − floor = P(0,T_s) − (1 + δK) P(0,T_e)).
hull_white_floorlet_price_tensor :: proc(
	a: f64,
	sigma: f64,
	T_start: ^t.Tensor,
	T_end: ^t.Tensor,
	F: ^t.Tensor,
	K: ^t.Tensor,
	P: ^t.Tensor,
	allocator: mem.Allocator,
) -> ^t.Tensor {
	one := _ones_tensor(T_start.data.rows, allocator)
	v := _hw1f_bond_var_tensor(a, sigma, T_start, T_end, one)
	return _hw_capfloor_tensor(v, T_start, T_end, F, K, P, one, false)
}

// Hull-White Monte Carlo: Floor Option (with CRN Greeks); see hw_mc_cap_option.
hw_mc_floor_option :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	n_floorlets: int,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta_r: f64,
	vega: f64,
) {
	c := _G2_Pricer{kind = .Floor, K = K, T_start = T_start, T_end = T_end, delta = delta, n = n_floorlets}
	return _g2_price_greeks(&c, _g2_from_hw(params), r0, n_paths, _g2_count_dates(T_start, n_floorlets), allocator)
}

// ============================================================================
// 2. HULL-WHITE 1F EUROPEAN SWAPTION PRICING (Monte Carlo)
// ============================================================================
// Payer (receiver) swaption expiring at T_exp on a swap paying K·δ at
// T_exp + jδ up to T_maturity, under Hull-White 1F fitted to e^{−r0 t}. Matches
// Jamshidian's formula up to Monte Carlo error. n_steps is unused.
hw_mc_swaption_option :: proc(
	r0: f64,
	K: f64,
	T_exp: f64,
	T_maturity: f64,
	delta: f64,
	is_payer: bool,
	params: HW_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta_r: f64,
	vega: f64,
) {
	c := _G2_Pricer{kind = .Swaption, K = K, T_exp = T_exp, T_mat = T_maturity, dlt = delta, is_payer = is_payer}
	return _g2_price_greeks(&c, _g2_from_hw(params), r0, n_paths, 1, allocator)
}

// ============================================================================
// 3. CAP / FLOOR STRIP PRICING (Convenience)
// ============================================================================

CapFloorStripResult :: struct {
	total_price: f64,
	strike:      f64,
	is_cap:      bool,
	leg_prices:  []f64, // Price of each individual caplet/floorlet
}

// Price a whole strip of caplets or floorlets analytically using HW 1F
hw_price_cap_floor_strip :: proc(
	r0: f64,
	K: f64,
	T_start: []f64,
	T_end: []f64,
	delta: []f64,
	params: HW_Params,
	is_cap: bool,
	allocator: mem.Allocator = context.allocator,
) -> CapFloorStripResult {
	n := len(T_start)

	T_start_t := _make_const_tensor(T_start, n, allocator)
	T_end_t := _make_const_tensor(T_end, n, allocator)

	// Forward rates F_i = (P(0, T_start) / P(0, T_end) - 1) / delta
	// For flat curve r0: P(0, t) = exp(-r0 * t)
	F_data := make([]f64, n, allocator)
	P_data := make([]f64, n, allocator)
	K_data := make([]f64, n, allocator)

	for i in 0 ..< n {
		P_start := math.exp_f64(-r0 * T_start[i])
		P_end := math.exp_f64(-r0 * T_end[i])
		F_data[i] = (P_start / P_end - 1.0) / delta[i]
		P_data[i] = P_end
		K_data[i] = K
	}

	F_t := _make_const_tensor(F_data, n, allocator)
	P_t := _make_const_tensor(P_data, n, allocator)
	K_t := _make_const_tensor(K_data, n, allocator)

	price_tensor: ^t.Tensor
	if is_cap {
		price_tensor = hull_white_caplet_price_tensor(
			params.a,
			params.sigma,
			T_start_t,
			T_end_t,
			F_t,
			K_t,
			P_t,
			allocator,
		)
	} else {
		price_tensor = hull_white_floorlet_price_tensor(
			params.a,
			params.sigma,
			T_start_t,
			T_end_t,
			F_t,
			K_t,
			P_t,
			allocator,
		)
	}

	leg_prices := make([]f64, n, allocator)
	total_price := 0.0
	for i in 0 ..< n {
		leg_prices[i] = price_tensor.data.data[i]
		total_price += leg_prices[i]
	}

	t.tensor_free_graph(price_tensor)
	t.tensor_free(T_start_t)
	t.tensor_free(T_end_t)
	t.tensor_free(F_t)
	t.tensor_free(P_t)
	t.tensor_free(K_t)
	delete(F_data, allocator)
	delete(P_data, allocator)
	delete(K_data, allocator)

	return CapFloorStripResult {
		total_price = total_price,
		strike = K,
		is_cap = is_cap,
		leg_prices = leg_prices,
	}
}
