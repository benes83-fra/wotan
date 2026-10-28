package finance

import "core:math"
import rand "core:math/rand"
import "core:mem"

// ============================================================================
// Monte Carlo Exotics under Heston, Merton Jump Diffusion and Local Volatility
// ============================================================================
// Conventions shared by every pricer in this file:
// - Equidistant grid t_i = i·T/n_steps, i = 1..n_steps. Barriers are monitored
//   on the grid (not at t_0), Asian averages use S_1..S_n, lookback extremes
//   include S_0.
// - Greeks are central differences on common random numbers: every path draws
//   its normals once and is re-simulated for the base and each bumped input, so
//   all runs see exactly the same randomness. Delta bumps S_0 by ±1%.
// - Barrier options use one-step survival (Glasserman & Staum 2001): each step
//   is drawn conditional on ending on the live side of the barrier and the path
//   is weighted by the product of the survival probabilities. This is unbiased
//   for the discretely monitored barrier, has lower variance, and is smooth in
//   the inputs, so the differences are not dominated by paths that knock out in
//   one run and not in the other.
// - "vega" is the derivative per unit of volatility, comparable to the
//   Black-Scholes vega of the monte_carlo_* pricers in european_options.odin:
//     Heston: parallel shift of √v0 and √θ (instantaneous and long-run vol)
//     MJD:    the diffusion volatility σ
//     LV:     parallel (additive) shift of the local-volatility surface
//   In the limits that reduce to Black-Scholes (Heston with σ = 0 and v0 = θ,
//   MJD with λ = 0, a flat LV surface) they agree with it.
// ============================================================================

// Up to 9 scenarios (base + bumps) per path
@(private = "file")
_MAX_SCN :: 9

// Runs n_paths paths with n_z standard normals each; path(ctx, k, z) returns
// the undiscounted payoff of scenario k on the path's normals z.
@(private = "file")
_crn_mc :: proc(
	ctx: ^$C,
	path: proc(ctx: ^C, k: int, z: []f64) -> f64,
	n_scn, n_paths, n_z: int,
	allocator: mem.Allocator,
) -> (
	sum: [_MAX_SCN]f64,
) {
	z := make([]f64, n_z, allocator)
	defer delete(z, allocator)
	for _ in 0 ..< n_paths {
		for &x in z {x = rand.float64_normal(0.0, 1.0)}
		for k in 0 ..< n_scn {sum[k] += path(ctx, k, z)}
	}
	return
}

// Price, delta and vega from the sums of scenarios (base, S_0 ± h_S, vol ± h_v)
@(private = "file")
_crn_greeks :: proc(
	sum: [_MAX_SCN]f64,
	n_paths: int,
	r, T, h_S, h_v: f64,
) -> (
	price, delta, vega: f64,
) {
	scale := math.exp(-r * T) / f64(n_paths)
	price = sum[0] * scale
	delta = (sum[1] - sum[2]) * scale / (2.0 * h_S)
	vega = (sum[3] - sum[4]) * scale / (2.0 * h_v)
	return
}

@(private = "file")
_payoff :: #force_inline proc(S, K: f64, opt: OptionType) -> f64 {
	return max(S - K, 0) if opt == .Call else max(K - S, 0)
}

// Log of the barrier; −∞ for barrier ≤ 0 (an up barrier is then always hit, a
// down barrier never)
@(private = "file")
_log_barrier :: proc(barrier: f64) -> f64 {
	return math.ln(barrier) if barrier > 0 else math.inf_f64(-1)
}

// One step x → x + m + s·z of the log-price, z ~ N(0,1), conditioned on ending
// on the live side of the log-barrier lb (below for an up barrier, above for a
// down barrier). z0 is the step's own standard normal; the conditional draw
// Φ⁻¹(Φ(z0)·p) (mirrored for down barriers) is z0 itself when the survival
// probability p is 1, so scenarios far from the barrier use identical steps.
// Returns the new log-price and p.
@(private = "file")
_survive_step :: proc(x, m, s, lb: f64, is_up: bool, z0: f64) -> (x_new, surv: f64) {
	if s <= 0 {
		x_new = x + m
		alive := x_new < lb if is_up else x_new > lb
		return x_new, 1 if alive else 0
	}
	sg := 1.0 if is_up else -1.0
	a := sg * (lb - x - m) / s
	if a > 8.3 {return x + m + s * z0, 1} 	// Φ(a) rounds to 1
	surv = _bs_cdf(a)
	if surv == 1 {return x + m + s * z0, 1}
	if surv < 1e-280 {return x, 0}
	return x + m + s * sg * _bs_ppf(_bs_cdf(sg * z0) * surv), surv
}

// ============================================================================
// HESTON
// ============================================================================

// One full-truncation Euler step (Lord, Koekkoek & van Dijk 2010): the variance
// may become negative, and v⁺ = max(v, 0) drives both the variance and the
// log-price. Returns the mean m and volatility s of the log-price step given the
// variance shock z_v (the stock's own shock z then enters as x += m + s·z),
// and advances v.
@(private = "file")
_heston_step :: #force_inline proc(
	v: ^f64,
	p: Heston_Params,
	r, dt, sqrt_dt, rho_perp, z_v: f64,
) -> (
	m, s: f64,
) {
	vp := max(v^, 0)
	sv := math.sqrt(vp) * sqrt_dt
	m = (r - 0.5 * vp) * dt + p.rho * sv * z_v
	s = rho_perp * sv
	v^ += p.kappa * (p.theta - vp) * dt + p.sigma * sv * z_v
	return
}

// Bump for the Heston vega: 0.01 in volatility, smaller if √v0 or √θ is below 0.02
@(private = "file")
_heston_vol_bump :: proc(p: Heston_Params) -> f64 {
	lo := math.sqrt(max(min(p.v0, p.theta), 0))
	return 0.5 * lo if lo > 0 && lo < 0.02 else 0.01
}

// Shifts √v0 and √θ by h
@(private = "file")
_heston_vol_shift :: proc(p: Heston_Params, h: f64) -> Heston_Params {
	q := p
	a := max(math.sqrt(max(p.v0, 0)) + h, 0)
	b := max(math.sqrt(max(p.theta, 0)) + h, 0)
	q.v0, q.theta = a * a, b * b
	return q
}

@(private = "file")
_Heston_MC :: struct {
	S_0:     [5]f64,
	p:       [5]Heston_Params,
	K, T, r: f64,
	lb:      f64,
	is_up:   bool,
	opt:     OptionType,
}

@(private = "file")
_heston_mc_init :: proc(
	c: ^_Heston_MC,
	S_0, K, T, r: f64,
	p: Heston_Params,
	opt: OptionType,
) -> (
	h_S, h_v: f64,
) {
	h_S = 0.01 * S_0
	h_v = _heston_vol_bump(p)
	c.S_0 = {S_0, S_0 + h_S, S_0 - h_S, S_0, S_0}
	c.p = {p, p, p, _heston_vol_shift(p, h_v), _heston_vol_shift(p, -h_v)}
	c.K, c.T, c.r, c.opt = K, T, r, opt
	return
}

// ============================================================================
// 1. BARRIER OPTIONS
// ============================================================================

@(private = "file")
_heston_barrier_path :: proc(c: ^_Heston_MC, k: int, z: []f64) -> f64 {
	p := c.p[k]
	n := len(z) / 2
	dt := c.T / f64(n)
	sqrt_dt := math.sqrt(dt)
	rho_perp := math.sqrt(max(1.0 - p.rho * p.rho, 0))
	x, v, w := math.ln(c.S_0[k]), p.v0, 1.0
	for i in 0 ..< n {
		m, s := _heston_step(&v, p, c.r, dt, sqrt_dt, rho_perp, z[2 * i])
		surv: f64
		x, surv = _survive_step(x, m, s, c.lb, c.is_up, z[2 * i + 1])
		w *= surv
		if w == 0 {return 0}
	}
	return w * _payoff(math.exp(x), c.K, c.opt)
}

// Generalized Barrier Option (Supports Call/Put and Up/Down): knocked out when
// S_i ≥ barrier (up) or S_i ≤ barrier (down) on the grid
heston_mc_barrier_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	barrier: f64,
	is_up: bool,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(T > 0) {return}
	c: _Heston_MC
	h_S, h_v := _heston_mc_init(&c, S_0, K, T, r, params, opt)
	c.lb, c.is_up = _log_barrier(barrier), is_up
	sum := _crn_mc(&c, _heston_barrier_path, 5, n_paths, 2 * n_steps, allocator)
	return _crn_greeks(sum, n_paths, r, T, h_S, h_v)
}

// Legacy Wrapper: Keeps your existing code working perfectly
heston_mc_barrier_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	barrier: f64,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	return heston_mc_barrier_option(
		S_0,
		K,
		T,
		r,
		barrier,
		true,
		.Call,
		params,
		n_paths,
		n_steps,
		allocator,
	)
}


// ============================================================================
// 2. ASIAN OPTIONS
// ============================================================================

@(private = "file")
_heston_asian_path :: proc(c: ^_Heston_MC, k: int, z: []f64) -> f64 {
	p := c.p[k]
	n := len(z) / 2
	dt := c.T / f64(n)
	sqrt_dt := math.sqrt(dt)
	rho_perp := math.sqrt(max(1.0 - p.rho * p.rho, 0))
	x, v, sum := math.ln(c.S_0[k]), p.v0, 0.0
	for i in 0 ..< n {
		m, s := _heston_step(&v, p, c.r, dt, sqrt_dt, rho_perp, z[2 * i])
		x += m + s * z[2 * i + 1]
		sum += math.exp(x)
	}
	return _payoff(sum / f64(n), c.K, c.opt)
}

// Arithmetic-average Asian option on mean(S_1..S_n)
heston_mc_asian_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(T > 0) {return}
	c: _Heston_MC
	h_S, h_v := _heston_mc_init(&c, S_0, K, T, r, params, opt)
	sum := _crn_mc(&c, _heston_asian_path, 5, n_paths, 2 * n_steps, allocator)
	return _crn_greeks(sum, n_paths, r, T, h_S, h_v)
}

heston_mc_asian_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	return heston_mc_asian_option(S_0, K, T, r, .Call, params, n_paths, n_steps, allocator)
}


// ============================================================================
// 3. LOOKBACK OPTIONS
// ============================================================================

@(private = "file")
_heston_lookback_path :: proc(c: ^_Heston_MC, k: int, z: []f64) -> f64 {
	p := c.p[k]
	n := len(z) / 2
	dt := c.T / f64(n)
	sqrt_dt := math.sqrt(dt)
	rho_perp := math.sqrt(max(1.0 - p.rho * p.rho, 0))
	x, v := math.ln(c.S_0[k]), p.v0
	x_max, x_min := x, x
	for i in 0 ..< n {
		m, s := _heston_step(&v, p, c.r, dt, sqrt_dt, rho_perp, z[2 * i])
		x += m + s * z[2 * i + 1]
		x_max, x_min = max(x_max, x), min(x_min, x)
	}
	if c.opt == .Call {return max(math.exp(x_max) - c.K, 0)}
	return max(c.K - math.exp(x_min), 0)
}

// Fixed-strike lookback: call pays max(max S − K, 0), put max(K − min S, 0)
heston_mc_lookback_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(T > 0) {return}
	c: _Heston_MC
	h_S, h_v := _heston_mc_init(&c, S_0, K, T, r, params, opt)
	sum := _crn_mc(&c, _heston_lookback_path, 5, n_paths, 2 * n_steps, allocator)
	return _crn_greeks(sum, n_paths, r, T, h_S, h_v)
}

heston_mc_lookback_call :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	return heston_mc_lookback_option(S_0, K, T, r, .Call, params, n_paths, n_steps, allocator)
}


// ============================================================================
// 4. 2-ASSET BASKET OPTIONS
// ============================================================================

@(private = "file")
_Heston_Basket_MC :: struct {
	S1, S2:      [9]f64,
	p1, p2:      [9]Heston_Params,
	K, T, r:     f64,
	w1, w2:      f64,
	opt:         OptionType,
	rho_12_star: f64, // correlation of the stocks' own shocks
	rho_12_perp: f64,
}

// Four normals per step: variance shocks U1, U2 and stock shocks U3, U4
@(private = "file")
_heston_basket_path :: proc(c: ^_Heston_Basket_MC, k: int, z: []f64) -> f64 {
	p1, p2 := c.p1[k], c.p2[k]
	n := len(z) / 4
	dt := c.T / f64(n)
	sqrt_dt := math.sqrt(dt)
	perp1 := math.sqrt(max(1.0 - p1.rho * p1.rho, 0))
	perp2 := math.sqrt(max(1.0 - p2.rho * p2.rho, 0))
	x1, x2 := math.ln(c.S1[k]), math.ln(c.S2[k])
	v1, v2 := p1.v0, p2.v0
	for i in 0 ..< n {
		U := z[4 * i:][:4]
		m1, s1 := _heston_step(&v1, p1, c.r, dt, sqrt_dt, perp1, U[0])
		m2, s2 := _heston_step(&v2, p2, c.r, dt, sqrt_dt, perp2, U[1])
		x1 += m1 + s1 * U[2]
		x2 += m2 + s2 * (c.rho_12_star * U[2] + c.rho_12_perp * U[3])
	}
	return _payoff(c.w1 * math.exp(x1) + c.w2 * math.exp(x2), c.K, c.opt)
}

// European basket on the terminal prices: pays max(±(w1·S1_T + w2·S2_T − K), 0).
// corr_12 is the correlation of the two stocks' Brownian motions; each variance
// is driven by its own shock, correlated with its stock only. The stock shocks
// are correlated with ρ* = corr_12 / (√(1−ρ1²)·√(1−ρ2²)), clamped to [−1, 1]
// when the requested correlation is not attainable.
// (monte_carlo_basket_option in european_options.odin is an Asian basket on
// the average prices.)
heston_mc_basket_option :: proc(
	S1_0: f64,
	S2_0: f64,
	K: f64,
	T: f64,
	r: f64,
	w1: f64,
	w2: f64,
	opt: OptionType,
	params1: Heston_Params,
	params2: Heston_Params,
	corr_12: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta1: f64,
	delta2: f64,
	vega1: f64,
	vega2: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(T > 0) {return}
	denom :=
		math.sqrt(max(1.0 - params1.rho * params1.rho, 0)) *
		math.sqrt(max(1.0 - params2.rho * params2.rho, 0))
	rho_star := clamp(corr_12 / denom, -1, 1) if denom > 0 else 0

	h1, h2 := 0.01 * S1_0, 0.01 * S2_0
	hv1, hv2 := _heston_vol_bump(params1), _heston_vol_bump(params2)
	up1, dn1 := _heston_vol_shift(params1, hv1), _heston_vol_shift(params1, -hv1)
	up2, dn2 := _heston_vol_shift(params2, hv2), _heston_vol_shift(params2, -hv2)
	c := _Heston_Basket_MC {
		S1          = {S1_0, S1_0 + h1, S1_0 - h1, S1_0, S1_0, S1_0, S1_0, S1_0, S1_0},
		S2          = {S2_0, S2_0, S2_0, S2_0 + h2, S2_0 - h2, S2_0, S2_0, S2_0, S2_0},
		p1          = {params1, params1, params1, params1, params1, up1, dn1, params1, params1},
		p2          = {params2, params2, params2, params2, params2, params2, params2, up2, dn2},
		K           = K,
		T           = T,
		r           = r,
		w1          = w1,
		w2          = w2,
		opt         = opt,
		rho_12_star = rho_star,
		rho_12_perp = math.sqrt(1.0 - rho_star * rho_star),
	}
	sum := _crn_mc(&c, _heston_basket_path, 9, n_paths, 4 * n_steps, allocator)
	scale := math.exp(-r * T) / f64(n_paths)
	price = sum[0] * scale
	delta1 = (sum[1] - sum[2]) * scale / (2.0 * h1)
	delta2 = (sum[3] - sum[4]) * scale / (2.0 * h2)
	vega1 = (sum[5] - sum[6]) * scale / (2.0 * hv1)
	vega2 = (sum[7] - sum[8]) * scale / (2.0 * hv2)
	return
}

heston_mc_basket_call :: proc(
	S1_0: f64,
	S2_0: f64,
	K: f64,
	T: f64,
	r: f64,
	w1: f64,
	w2: f64,
	params1: Heston_Params,
	params2: Heston_Params,
	corr_12: f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta1: f64,
	delta2: f64,
	vega1: f64,
	vega2: f64,
) {
	return heston_mc_basket_option(
		S1_0,
		S2_0,
		K,
		T,
		r,
		w1,
		w2,
		.Call,
		params1,
		params2,
		corr_12,
		n_paths,
		n_steps,
		allocator,
	)
}
// ============================================================================
// MERTON JUMP DIFFUSION (MJD) MODEL
// ============================================================================

MJD_Params :: struct {
	sigma:   f64, // Diffusion volatility
	lambda:  f64, // Jump intensity (expected jumps per year)
	mu_j:    f64, // Mean of the log-normal jump size
	sigma_j: f64, // Volatility of the log-normal jump size
}

@(private = "file")
_MJD_MC :: struct {
	S_0:     [5]f64,
	p:       [5]MJD_Params,
	K, T, r: f64,
	lb:      f64,
	is_up:   bool,
	opt:     OptionType,
}

@(private = "file")
_mjd_mc_init :: proc(
	c: ^_MJD_MC,
	S_0, K, T, r: f64,
	p: MJD_Params,
	opt: OptionType,
) -> (
	h_S, h_v: f64,
) {
	h_S = 0.01 * S_0
	h_v = max(0.01 * p.sigma, 0.001)
	up, dn := p, p
	up.sigma += h_v
	dn.sigma -= h_v
	c.S_0 = {S_0, S_0 + h_S, S_0 - h_S, S_0, S_0}
	c.p = {p, p, p, up, dn}
	c.K, c.T, c.r, c.opt = K, T, r, opt
	return
}

// Number of jumps in a step: Poisson(mean) by inversion of u
@(private = "file")
_poisson_inv :: proc(u, mean: f64) -> int {
	pk := math.exp(-mean)
	cdf := pk
	k := 0
	for u > cdf && k < 200 {
		k += 1
		pk *= mean / f64(k)
		cdf += pk
	}
	return k
}

// Three normals per step: diffusion Z, jump size Z_J and Z_U, whose Φ(Z_U) is
// the uniform for the jump count. The n jumps of a step have log-size
// n·μ_J + √n·σ_J·Z_J; the drift is compensated, r − λ·(e^{μ_J + σ_J²/2} − 1).
// Barrier steps condition the diffusion on the jump, which is drawn first.
@(private = "file")
_mjd_path :: proc(c: ^_MJD_MC, k: int, z: []f64, barrier: bool) -> f64 {
	p := c.p[k]
	n := len(z) / 3
	dt := c.T / f64(n)
	s := p.sigma * math.sqrt(dt)
	kappa := math.exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0
	mu := (c.r - p.lambda * kappa - 0.5 * p.sigma * p.sigma) * dt
	x, w, sum := math.ln(c.S_0[k]), 1.0, 0.0
	for i in 0 ..< n {
		Z, Z_J, Z_U := z[3 * i], z[3 * i + 1], z[3 * i + 2]
		jump := 0.0
		if n_j := _poisson_inv(_bs_cdf(Z_U), p.lambda * dt); n_j > 0 {
			jump = f64(n_j) * p.mu_j + math.sqrt(f64(n_j)) * p.sigma_j * Z_J
		}
		if barrier {
			surv: f64
			x, surv = _survive_step(x, mu + jump, s, c.lb, c.is_up, Z)
			w *= surv
			if w == 0 {return 0}
		} else {
			x += mu + jump + s * Z
			sum += math.exp(x)
		}
	}
	if barrier {return w * _payoff(math.exp(x), c.K, c.opt)}
	return _payoff(sum / f64(n), c.K, c.opt)
}

@(private = "file")
_mjd_barrier_path :: proc(c: ^_MJD_MC, k: int, z: []f64) -> f64 {return _mjd_path(c, k, z, true)}

@(private = "file")
_mjd_asian_path :: proc(c: ^_MJD_MC, k: int, z: []f64) -> f64 {return _mjd_path(c, k, z, false)}

// Barrier option (call/put, up/down) under Merton jump diffusion
mjd_mc_barrier_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	barrier: f64,
	is_up: bool,
	opt: OptionType,
	params: MJD_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(T > 0) {return}
	c: _MJD_MC
	h_S, h_v := _mjd_mc_init(&c, S_0, K, T, r, params, opt)
	c.lb, c.is_up = _log_barrier(barrier), is_up
	sum := _crn_mc(&c, _mjd_barrier_path, 5, n_paths, 3 * n_steps, allocator)
	return _crn_greeks(sum, n_paths, r, T, h_S, h_v)
}

// Arithmetic-average Asian option under Merton jump diffusion
mjd_mc_asian_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	opt: OptionType,
	params: MJD_Params,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(T > 0) {return}
	c: _MJD_MC
	h_S, h_v := _mjd_mc_init(&c, S_0, K, T, r, params, opt)
	sum := _crn_mc(&c, _mjd_asian_path, 5, n_paths, 3 * n_steps, allocator)
	return _crn_greeks(sum, n_paths, r, T, h_S, h_v)
}
// ============================================================================
// LOCAL VOLATILITY (DUPIRE) MODEL
// ============================================================================

LocalVolSurface :: struct {
	strikes:      []f64,
	expiries:     []f64,
	lv_data:      []f64, // Flattened 2D grid: lv_data[expiry_idx * num_strikes + strike_idx]
	num_strikes:  int,
	num_expiries: int,
}

// Frees a surface made by build_lv_surface_from_heston
lv_surface_free :: proc(lv: ^LocalVolSurface, allocator: mem.Allocator = context.allocator) {
	delete(lv.strikes, allocator)
	delete(lv.expiries, allocator)
	delete(lv.lv_data, allocator)
	lv^ = {}
}

// Helper: Bilinear interpolation for Local Volatility
get_local_vol :: proc(lv: LocalVolSurface, S: f64, t: f64) -> f64 {
	// Clamp S and t to grid bounds
	S := S
	t := t
	if S <= lv.strikes[0] {S = lv.strikes[0] + 1e-6}
	if S >= lv.strikes[lv.num_strikes - 1] {S = lv.strikes[lv.num_strikes - 1] - 1e-6}
	if t <= lv.expiries[0] {t = lv.expiries[0] + 1e-6}
	if t >= lv.expiries[lv.num_expiries - 1] {t = lv.expiries[lv.num_expiries - 1] - 1e-6}

	// Find indices
	i_low := 0
	for i in 0 ..< lv.num_strikes - 1 {
		if lv.strikes[i] <= S && S < lv.strikes[i + 1] {
			i_low = i
			break
		}
	}
	i_high := i_low + 1

	j_low := 0
	for j in 0 ..< lv.num_expiries - 1 {
		if lv.expiries[j] <= t && t < lv.expiries[j + 1] {
			j_low = j
			break
		}
	}
	j_high := j_low + 1

	// Weights
	w_k := (S - lv.strikes[i_low]) / (lv.strikes[i_high] - lv.strikes[i_low])
	w_t := (t - lv.expiries[j_low]) / (lv.expiries[j_high] - lv.expiries[j_low])

	// Bilinear interpolation
	v00 := lv.lv_data[j_low * lv.num_strikes + i_low]
	v10 := lv.lv_data[j_low * lv.num_strikes + i_high]
	v01 := lv.lv_data[j_high * lv.num_strikes + i_low]
	v11 := lv.lv_data[j_high * lv.num_strikes + i_high]

	v0 := v00 * (1.0 - w_k) + v10 * w_k
	v1 := v01 * (1.0 - w_k) + v11 * w_k

	return v0 * (1.0 - w_t) + v1 * w_t
}

// Build LV Surface from Calibrated Heston Model using Dupire's Formula.
// Without dividends: σ²_loc(K, T) = (∂C/∂T + r·K·∂C/∂K) / (½·K²·∂²C/∂K²).
// Put-call parity leaves the formula unchanged for puts, which are used below
// the forward so that deep in-the-money calls do not lose the derivatives to
// cancellation. Free the result with lv_surface_free.
build_lv_surface_from_heston :: proc(
	S_0: f64,
	r: f64,
	params: Heston_Params,
	allocator: mem.Allocator = context.allocator,
) -> LocalVolSurface {
	num_strikes := 21
	num_expiries := 10

	strikes := make([]f64, num_strikes, allocator)
	expiries := make([]f64, num_expiries, allocator)
	lv_data := make([]f64, num_strikes * num_expiries, allocator)

	// Define grid
	for i in 0 ..< num_strikes {
		strikes[i] = S_0 * (0.6 + f64(i) * 0.04) // 0.6 S_0 to 1.4 S_0
	}
	for j in 0 ..< num_expiries {
		expiries[j] = 0.1 + f64(j) * 0.1 // 0.1 to 1.0 years
	}

	// Finite difference steps
	dK := S_0 * 0.005
	dT := 1.0 / 365.0

	// Compute the grid (skip boundaries for finite differences)
	for j in 1 ..< num_expiries - 1 {
		T := expiries[j]
		for i in 1 ..< num_strikes - 1 {
			K := strikes[i]
			opt := OptionType.Put if K < S_0 * math.exp(r * T) else OptionType.Call

			// 1. Heston prices at (K, T), (K, T ± dT) and (K ± dK, T)
			C := heston_price(S_0, K, T, r, params, opt, 2000)
			C_T_plus := heston_price(S_0, K, T + dT, r, params, opt, 2000)
			C_T_minus := heston_price(S_0, K, T - dT, r, params, opt, 2000)
			C_K_plus := heston_price(S_0, K + dK, T, r, params, opt, 2000)
			C_K_minus := heston_price(S_0, K - dK, T, r, params, opt, 2000)

			// 2. Dupire derivatives (central differences)
			dC_dT := (C_T_plus - C_T_minus) / (2.0 * dT)
			dC_dK := (C_K_plus - C_K_minus) / (2.0 * dK)
			d2C_dK2 := (C_K_plus - 2.0 * C + C_K_minus) / (dK * dK)

			// 3. Prevent numerical instability
			if d2C_dK2 < 1e-8 {d2C_dK2 = 1e-8}

			// 4. Dupire's formula
			sigma_loc_sq := (dC_dT + r * K * dC_dK) / (0.5 * K * K * d2C_dK2)
			if sigma_loc_sq < 0.0 {sigma_loc_sq = 0.0} 	// Safety clamp

			lv_data[j * num_strikes + i] = math.sqrt_f64(sigma_loc_sq)
		}
	}

	// 5. Fill boundaries with nearest valid value
	for j in 0 ..< num_expiries {
		lv_data[j * num_strikes + 0] = lv_data[j * num_strikes + 1]
		lv_data[j * num_strikes + num_strikes - 1] = lv_data[j * num_strikes + num_strikes - 2]
	}
	for i in 0 ..< num_strikes {
		lv_data[0 * num_strikes + i] = lv_data[1 * num_strikes + i]
		lv_data[(num_expiries - 1) * num_strikes + i] =
			lv_data[(num_expiries - 2) * num_strikes + i]
	}

	return LocalVolSurface {
		strikes = strikes,
		expiries = expiries,
		lv_data = lv_data,
		num_strikes = num_strikes,
		num_expiries = num_expiries,
	}
}

// ============================================================================
// Local Volatility Monte Carlo: Barrier Option (with CRN Greeks)
// ============================================================================
@(private = "file")
_LV_MC :: struct {
	S_0:     [5]f64,
	shift:   [5]f64, // parallel shift of the surface
	lv:      LocalVolSurface,
	K, T, r: f64,
	lb:      f64,
	is_up:   bool,
	opt:     OptionType,
}

@(private = "file")
_lv_barrier_path :: proc(c: ^_LV_MC, k: int, z: []f64) -> f64 {
	n := len(z)
	dt := c.T / f64(n)
	sqrt_dt := math.sqrt(dt)
	x, w := math.ln(c.S_0[k]), 1.0
	for i in 0 ..< n {
		sigma := max(get_local_vol(c.lv, math.exp(x), f64(i) * dt) + c.shift[k], 0)
		surv: f64
		x, surv = _survive_step(
			x,
			(c.r - 0.5 * sigma * sigma) * dt,
			sigma * sqrt_dt,
			c.lb,
			c.is_up,
			z[i],
		)
		w *= surv
		if w == 0 {return 0}
	}
	return w * _payoff(math.exp(x), c.K, c.opt)
}

// Barrier option (call/put, up/down) under local volatility. The local vol of a
// step is taken at its start. Vega is the derivative for a parallel shift of
// the whole surface (σ_loc + h), so a flat surface σ gives the Black-Scholes vega.
lv_mc_barrier_option :: proc(
	S_0: f64,
	K: f64,
	T: f64,
	r: f64,
	barrier: f64,
	is_up: bool,
	opt: OptionType,
	lv: LocalVolSurface,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	price: f64,
	delta: f64,
	vega: f64,
) {
	if n_paths <= 0 || n_steps <= 0 || !(T > 0) {return}
	h_S, h_v := 0.01 * S_0, 0.01
	c := _LV_MC {
		S_0   = {S_0, S_0 + h_S, S_0 - h_S, S_0, S_0},
		shift = {0, 0, 0, h_v, -h_v},
		lv    = lv,
		K     = K,
		T     = T,
		r     = r,
		lb    = _log_barrier(barrier),
		is_up = is_up,
		opt   = opt,
	}
	sum := _crn_mc(&c, _lv_barrier_path, 5, n_paths, n_steps, allocator)
	return _crn_greeks(sum, n_paths, r, T, h_S, h_v)
}
