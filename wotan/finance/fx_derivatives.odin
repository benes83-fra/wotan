package finance

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// FX GARMAN-KOHLHAGEN MODEL
// ============================================================================

FXOptionType :: enum {
	Call,
	Put,
}

FXGKResult :: struct {
	price:   f64,
	delta:   f64, // dP/dS (spot delta)
	delta_f: f64, // forward delta e^{r_f T} · dP/dS (not premium-adjusted)
	gamma:   f64, // d²P/dS²
	vega:    f64, // dP/dσ
	theta:   f64, // -dP/dT
	rho_d:   f64, // dP/dr_d
	rho_f:   f64, // dP/dr_f
}

_N :: proc(x: f64) -> f64 {
	return 0.5 * (1.0 + math.erf(x / math.sqrt_f64(2.0)))
}

_n :: proc(x: f64) -> f64 {
	return math.exp_f64(-0.5 * x * x) / math.sqrt_f64(2.0 * math.PI)
}

// Standard normal quantile: Acklam's rational approximation refined by one
// Halley step (full double precision); ±Inf outside (0, 1)
_N_inv :: proc(p: f64) -> f64 {
	if p <= 0.0 {return math.inf_f64(-1)}
	if p >= 1.0 {return math.inf_f64(1)}
	x := norm_inv(p)
	e := _N(x) - p
	u := e * math.sqrt_f64(2.0 * math.PI) * math.exp_f64(0.5 * x * x)
	return x - u / (1.0 + 0.5 * x * u)
}

// Garman-Kohlhagen price; σ = 0 gives the discounted forward intrinsic value
fx_gk_price :: proc(
	S: f64,
	K: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
	opt: FXOptionType,
) -> f64 {
	if T <= 0.0 {
		if opt == .Call {return math.max(S - K, 0.0)}
		return math.max(K - S, 0.0)
	}
	df_d := math.exp_f64(-r_d * T)
	df_f := math.exp_f64(-r_f * T)
	if sigma <= 0.0 {
		fwd := S * df_f - K * df_d
		return max(fwd, 0.0) if opt == .Call else max(-fwd, 0.0)
	}
	sqrt_T := math.sqrt_f64(T)
	d1 := (math.ln(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	d2 := d1 - sigma * sqrt_T

	if opt == .Call {
		return S * df_f * _N(d1) - K * df_d * _N(d2)
	}
	return K * df_d * _N(-d2) - S * df_f * _N(-d1)
}

fx_gk_greeks :: proc(
	S: f64,
	K: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
	opt: FXOptionType,
) -> FXGKResult {
	price := fx_gk_price(S, K, T, r_d, r_f, sigma, opt)
	if T <= 0.0 || sigma <= 0.0 {
		return FXGKResult{price = price}
	}
	sqrt_T := math.sqrt_f64(T)
	d1 := (math.ln(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	d2 := d1 - sigma * sqrt_T
	df_d := math.exp_f64(-r_d * T)
	df_f := math.exp_f64(-r_f * T)

	gamma := df_f * _n(d1) / (S * sigma * sqrt_T)
	vega := S * df_f * _n(d1) * sqrt_T

	delta, rho_d, rho_f, theta: f64
	if opt == .Call {
		delta = df_f * _N(d1)
		rho_d = K * T * df_d * _N(d2)
		rho_f = -S * T * df_f * _N(d1)
		theta =
			-S * df_f * _n(d1) * sigma / (2.0 * sqrt_T) +
			r_f * S * df_f * _N(d1) -
			r_d * K * df_d * _N(d2)
	} else {
		delta = -df_f * _N(-d1)
		rho_d = -K * T * df_d * _N(-d2)
		rho_f = S * T * df_f * _N(-d1)
		theta =
			-S * df_f * _n(d1) * sigma / (2.0 * sqrt_T) -
			r_f * S * df_f * _N(-d1) +
			r_d * K * df_d * _N(-d2)
	}

	return FXGKResult {
		price = price,
		delta = delta,
		delta_f = delta * math.exp_f64(r_f * T),
		gamma = gamma,
		vega = vega,
		theta = theta,
		rho_d = rho_d,
		rho_f = rho_f,
	}
}

fx_gk_call :: proc(S: f64, K: f64, T: f64, r_d: f64, r_f: f64, sigma: f64) -> FXGKResult {
	return fx_gk_greeks(S, K, T, r_d, r_f, sigma, .Call)
}

fx_gk_put :: proc(S: f64, K: f64, T: f64, r_d: f64, r_f: f64, sigma: f64) -> FXGKResult {
	return fx_gk_greeks(S, K, T, r_d, r_f, sigma, .Put)
}

// Strike with the given spot delta (call Δ ∈ (0, e^{−r_f T}), put Δ ∈ (−e^{−r_f T}, 0)),
// not premium-adjusted
fx_strike_from_delta :: proc(
	S: f64,
	delta_target: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
	opt: FXOptionType,
) -> f64 {
	df_f := math.exp_f64(-r_f * T)
	d1: f64
	if opt == .Call {
		d1 = _N_inv(delta_target / df_f)
	} else {
		d1 = -_N_inv(-delta_target / df_f)
	}
	sqrt_T := math.sqrt_f64(T)
	return S * math.exp_f64(-d1 * sigma * sqrt_T + (r_d - r_f + 0.5 * sigma * sigma) * T)
}

// ============================================================================
// FX VOLATILITY SMILE & VANNA-VOLGA
// ============================================================================

// 25-delta smile quotes (smile-strangle butterfly, spot deltas)
FXSmile :: struct {
	atm_vol:           f64,
	risk_reversal_25d: f64,
	butterfly_25d:     f64,
}

FXSmileVols :: struct {
	vol_25d_put:  f64,
	vol_atm:      f64,
	vol_25d_call: f64,
}

fx_smile_to_vols :: proc(smile: FXSmile) -> FXSmileVols {
	return FXSmileVols {
		vol_25d_put = smile.atm_vol - smile.risk_reversal_25d / 2.0 + smile.butterfly_25d,
		vol_atm = smile.atm_vol,
		vol_25d_call = smile.atm_vol + smile.risk_reversal_25d / 2.0 + smile.butterfly_25d,
	}
}

_VannaVolgaSensitivities :: struct {
	vega:  f64,
	vanna: f64,
	volga: f64,
}

_fx_vanna_volga_sensitivities :: proc(
	S: f64,
	K: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
) -> _VannaVolgaSensitivities {
	sqrt_T := math.sqrt_f64(T)
	d1 := (math.ln(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	d2 := d1 - sigma * sqrt_T
	df_f := math.exp_f64(-r_f * T)
	vega := S * df_f * _n(d1) * sqrt_T
	return _VannaVolgaSensitivities {
		vega  = vega,
		vanna = -vega * d2 / (S * sigma * sqrt_T), // ∂vega/∂S = −e^{−r_f T} n(d1) d2 / σ
		volga = vega * d1 * d2 / sigma,
	}
}

VannaVolgaResult :: struct {
	price_bs:        f64,
	price_vv:        f64,
	adjustment:      f64,
	weight_25d_put:  f64,
	weight_atm:      f64,
	weight_25d_call: f64,
}

// Vanna-volga smile adjustment (Castagna & Mercurio 2007). The three pillars are
// the 25Δ put, the ATM-forward call and the 25Δ call, with strikes from their
// smile vols. Weights w solve Σ w_i (vega, vanna, volga)_i = the exotic's
// (vega, vanna, volga), all Greeks at the ATM vol, and
//   price_vv = exotic_bs_price + Σ w_i (market_price_i − GK_i(σ_ATM)).
// exotic_bs_price and the exotic Greeks must also be at the ATM vol. With the
// exotic's vega the price of a vanilla at a pillar strike is that pillar's
// market price; the default exotic_vega = 0 hedges vanna and volga only.
fx_vanna_volga_adjust :: proc(
	S: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	smile: FXSmile,
	exotic_bs_price: f64,
	exotic_vanna: f64,
	exotic_volga: f64,
	market_price_25d_put: f64,
	market_price_atm: f64,
	market_price_25d_call: f64,
	allocator: mem.Allocator = context.allocator,
	exotic_vega: f64 = 0.0,
) -> VannaVolgaResult {
	vols := fx_smile_to_vols(smile)
	sa := vols.vol_atm

	K := [3]f64 {
		fx_strike_from_delta(S, -0.25, T, r_d, r_f, vols.vol_25d_put, .Put),
		S * math.exp_f64((r_d - r_f) * T),
		fx_strike_from_delta(S, 0.25, T, r_d, r_f, vols.vol_25d_call, .Call),
	}
	market := [3]f64{market_price_25d_put, market_price_atm, market_price_25d_call}
	kinds := [3]FXOptionType{.Put, .Call, .Call}

	A := l.matrix_new(f64, 3, 3, allocator)
	defer l.matrix_free(&A)
	cost: [3]f64
	for i in 0 ..< 3 {
		g := _fx_vanna_volga_sensitivities(S, K[i], T, r_d, r_f, sa)
		A.data[0 * 3 + i] = g.vega
		A.data[1 * 3 + i] = g.vanna
		A.data[2 * 3 + i] = g.volga
		cost[i] = market[i] - fx_gk_price(S, K[i], T, r_d, r_f, sa, kinds[i])
	}
	b := []f64{exotic_vega, exotic_vanna, exotic_volga}

	LU, piv, _, ok := l.lu_decompose(&A, allocator)
	defer l.matrix_free(&LU)
	defer delete(piv, allocator)
	if !ok {
		return VannaVolgaResult{price_bs = exotic_bs_price, price_vv = exotic_bs_price}
	}
	x := l.lu_solve(&LU, piv, b, allocator)
	defer delete(x, allocator)

	adjustment := x[0] * cost[0] + x[1] * cost[1] + x[2] * cost[2]
	return VannaVolgaResult {
		price_bs = exotic_bs_price,
		price_vv = exotic_bs_price + adjustment,
		adjustment = adjustment,
		weight_25d_put = x[0],
		weight_atm = x[1],
		weight_25d_call = x[2],
	}
}

// ============================================================================
// FX EXOTIC OPTIONS
// ============================================================================

// Central finite-difference vanna ∂²P/∂S∂σ, volga ∂²P/∂σ² and vega ∂P/∂σ of
// price(S, σ, user)
@(private = "file")
_fx_fd_greeks :: proc(
	price: proc(S, sigma: f64, user: rawptr) -> f64,
	user: rawptr,
	S, sigma, rel_h: f64,
) -> (
	vanna, volga, vega: f64,
) {
	hS, hs := rel_h * S, rel_h * sigma
	pp := price(S + hS, sigma + hs, user)
	mp := price(S - hS, sigma + hs, user)
	pm := price(S + hS, sigma - hs, user)
	mm := price(S - hS, sigma - hs, user)
	vanna = (pp - mp - pm + mm) / (4.0 * hS * hs)
	up, mid, dn := price(S, sigma + hs, user), price(S, sigma, user), price(S, sigma - hs, user)
	volga = (up - 2.0 * mid + dn) / (hs * hs)
	vega = (up - dn) / (2.0 * hs)
	return
}

// Cash-or-nothing call: pays 1 (domestic) at T if S_T > K
fx_digital_call_price :: proc(S: f64, K: f64, T: f64, r_d: f64, r_f: f64, sigma: f64) -> f64 {
	if T <= 0.0 {return S > K ? 1.0 : 0.0}
	sqrt_T := math.sqrt_f64(T)
	d2 := (math.ln(S / K) + (r_d - r_f - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	return math.exp_f64(-r_d * T) * _N(d2)
}

// Vanna, volga and vega of the digital call by central differences
fx_digital_call_sensitivities :: proc(
	S: f64,
	K: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
) -> (
	vanna: f64,
	volga: f64,
	vega: f64,
) {
	q := [4]f64{K, T, r_d, r_f}
	f :: proc(S, sigma: f64, user: rawptr) -> f64 {
		q := (^[4]f64)(user)
		return fx_digital_call_price(S, q[0], q[1], q[2], q[3], sigma)
	}
	return _fx_fd_greeks(f, &q, S, sigma, 1e-3)
}

// ============================================================================
// DOWN-AND-OUT DIGITAL CALL (Analytical)
// ============================================================================
// Pays 1 unit of domestic currency at T if S_T > K and the spot stays above L
// (continuously monitored). Method of images (Reiner & Rubinstein 1991); for
// K ≤ L the strike condition is implied by survival and L takes K's place.

fx_down_and_out_digital_call_price :: proc(
	S: f64,
	K: f64,
	L: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
) -> f64 {
	if S <= L {return 0.0}
	if T <= 0.0 {return S > K ? 1.0 : 0.0}

	Ke := max(K, L)
	sqrt_T := math.sqrt_f64(T)
	mu := (r_d - r_f - 0.5 * sigma * sigma) / (sigma * sigma)
	d2 := (math.ln(S / Ke) + (r_d - r_f - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	d2_image :=
		(math.ln((L * L) / (S * Ke)) + (r_d - r_f - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	return math.exp_f64(-r_d * T) * (_N(d2) - math.pow_f64(L / S, 2.0 * mu) * _N(d2_image))
}

// Vanna, volga and vega by central differences on the analytical price
fx_down_and_out_digital_call_sensitivities :: proc(
	S: f64,
	K: f64,
	L: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
) -> (
	vanna: f64,
	volga: f64,
	vega: f64,
) {
	q := [5]f64{K, L, T, r_d, r_f}
	f :: proc(S, sigma: f64, user: rawptr) -> f64 {
		q := (^[5]f64)(user)
		return fx_down_and_out_digital_call_price(S, q[0], q[1], q[2], q[3], q[4], sigma)
	}
	return _fx_fd_greeks(f, &q, S, sigma, 1e-4)
}

// ============================================================================
// DOUBLE-NO-TOUCH (DNT) BARRIER OPTION - PDE IMPLEMENTATION
// ============================================================================
// Pays 1 (domestic) at T if L < S_t < U for all t ≤ T. Solves
//   V_τ = ½σ² V_xx + (r_d − r_f − ½σ²) V_x − r_d V,  x = ln S ∈ [ln L, ln U],
// V = 0 on the barriers, V(τ = 0) = 1 inside, with Crank–Nicolson steps; the
// first step is replaced by four fully implicit quarter steps (Rannacher) to damp
// the oscillations CN produces from the jump at the barriers. The price at S is
// interpolated quadratically from the three nearest nodes.

fx_dnt_pde_price :: proc(
	S: f64,
	L: f64,
	U: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
	n_space: int = 200,
	n_time: int = 1000,
) -> f64 {
	if S <= L || S >= U {return 0.0}
	if T <= 0.0 {return 1.0}

	ns := max(n_space, 4)
	nt := max(n_time, 1)
	x_min := math.ln(L)
	dx := (math.ln(U) - x_min) / f64(ns - 1)
	dt := T / f64(nt)
	alpha := 0.5 * sigma * sigma
	beta := r_d - r_f - 0.5 * sigma * sigma

	// L_h V_i = lo V_{i−1} + di V_i + up V_{i+1}
	lo := alpha / (dx * dx) - beta / (2.0 * dx)
	di := -2.0 * alpha / (dx * dx) - r_d
	up := alpha / (dx * dx) + beta / (2.0 * dx)

	alloc := context.temp_allocator
	V := make([]f64, ns, alloc)
	nu := ns - 2
	a := make([]f64, nu - 1, alloc)
	b := make([]f64, nu, alloc)
	c := make([]f64, nu - 1, alloc)
	d := make([]f64, nu, alloc)
	defer {delete(V, alloc); delete(a, alloc); delete(b, alloc); delete(c, alloc)
		delete(d, alloc)}
	for i in 1 ..< ns - 1 {V[i] = 1.0}

	// One step of size h with weight θ on the new time level (θ = 1 implicit, ½ CN)
	step :: proc(V, a, b, c, d: []f64, lo, di, up, h, theta: f64) {
		nu := len(d)
		e := (1.0 - theta) * h
		for i in 0 ..< nu {
			k := i + 1
			d[i] = V[k] + e * (lo * V[k - 1] + di * V[k] + up * V[k + 1])
			b[i] = 1.0 - theta * h * di
			if i > 0 {a[i - 1] = -theta * h * lo}
			if i < nu - 1 {c[i] = -theta * h * up}
		}
		_thomas_algorithm(a, b, c, d, nu)
		for i in 0 ..< nu {V[i + 1] = d[i]}
	}
	for _ in 0 ..< 4 {step(V, a, b, c, d, lo, di, up, dt / 4.0, 1.0)}
	for _ in 1 ..< nt {step(V, a, b, c, d, lo, di, up, dt, 0.5)}

	// Quadratic interpolation on the nearest three nodes
	u := (math.ln(S) - x_min) / dx
	j := clamp(int(math.round(u)), 1, ns - 2)
	w := u - f64(j)
	return V[j - 1] * 0.5 * w * (w - 1.0) + V[j] * (1.0 - w * w) + V[j + 1] * 0.5 * w * (w + 1.0)
}

// Vanna, volga and vega by central differences on the PDE price (default grid)
fx_dnt_pde_sensitivities :: proc(
	S: f64,
	L: f64,
	U: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
) -> (
	vanna: f64,
	volga: f64,
	vega: f64,
) {
	q := [5]f64{L, U, T, r_d, r_f}
	f :: proc(S, sigma: f64, user: rawptr) -> f64 {
		q := (^[5]f64)(user)
		return fx_dnt_pde_price(S, q[0], q[1], q[2], q[3], q[4], sigma)
	}
	return _fx_fd_greeks(f, &q, S, sigma, 1e-3)
}
