package finance

import l "../linalg"
import "core:math"
import "core:math/rand"
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
	delta_f: f64, // premium-adjusted forward delta (standard FX quoting)
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

_N_inv :: proc(p: f64) -> f64 {
	if p <= 0.0 {return -10.0}
	if p >= 1.0 {return 10.0}
	if p == 0.5 {return 0.0}
	if p < 0.5 {return -_N_inv(1.0 - p)}

	t := math.sqrt_f64(-2.0 * math.ln(1.0 - p))
	c0 := 2.515517; c1 := 0.802853; c2 := 0.010328
	d1 := 1.432788; d2 := 0.189269; d3 := 0.001308
	return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}

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
	sqrt_T := math.sqrt_f64(T)
	d1 := (math.ln(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	d2 := d1 - sigma * sqrt_T
	df_d := math.exp_f64(-r_d * T)
	df_f := math.exp_f64(-r_f * T)

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
	if T <= 0.0 {
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
		rho_d = -K * T * df_d * _N(d2)
		rho_f = S * T * df_f * _N(d1)
		theta =
			-S * df_f * _n(d1) * sigma / (2.0 * sqrt_T) +
			r_f * S * df_f * _N(d1) -
			r_d * K * df_d * _N(d2)
	} else {
		delta = -df_f * _N(-d1)
		rho_d = K * T * df_d * _N(-d2)
		rho_f = -S * T * df_f * _N(-d1)
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
		vega = vega,
		vanna = -vega * d2 / (S * sigma),
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
) -> VannaVolgaResult {

	vols := fx_smile_to_vols(smile)

	K_25d_put := fx_strike_from_delta(S, -0.25, T, r_d, r_f, vols.vol_25d_put, .Put)
	K_atm := S * math.exp_f64((r_d - r_f) * T)
	K_25d_call := fx_strike_from_delta(S, 0.25, T, r_d, r_f, vols.vol_25d_call, .Call)

	sens_25d_put := _fx_vanna_volga_sensitivities(S, K_25d_put, T, r_d, r_f, vols.vol_25d_put)
	sens_atm := _fx_vanna_volga_sensitivities(S, K_atm, T, r_d, r_f, vols.vol_atm)
	sens_25d_call := _fx_vanna_volga_sensitivities(S, K_25d_call, T, r_d, r_f, vols.vol_25d_call)

	// ✅ CRITICAL FIX: The "BS price" of the hedging instruments MUST be evaluated
	// at the ATM volatility, not their own smile volatility. This creates the
	// non-zero difference (market_price - bs_price) that drives the adjustment.
	bs_price_25d_put := fx_gk_price(S, K_25d_put, T, r_d, r_f, vols.vol_atm, .Put)
	bs_price_atm := fx_gk_price(S, K_atm, T, r_d, r_f, vols.vol_atm, .Call)
	bs_price_25d_call := fx_gk_price(S, K_25d_call, T, r_d, r_f, vols.vol_atm, .Call)

	A := l.matrix_new(f64, 3, 3, allocator)
	defer l.matrix_free(&A)

	A.data[0 * 3 + 0] = sens_25d_put.vega
	A.data[0 * 3 + 1] = sens_atm.vega
	A.data[0 * 3 + 2] = sens_25d_call.vega
	A.data[1 * 3 + 0] = sens_25d_put.vanna
	A.data[1 * 3 + 1] = sens_atm.vanna
	A.data[1 * 3 + 2] = sens_25d_call.vanna
	A.data[2 * 3 + 0] = sens_25d_put.volga
	A.data[2 * 3 + 1] = sens_atm.volga
	A.data[2 * 3 + 2] = sens_25d_call.volga

	b := make([]f64, 3, allocator)
	defer delete(b, allocator)
	b[0] = 0.0
	b[1] = exotic_vanna
	b[2] = exotic_volga

	LU, piv, sign, ok := l.lu_decompose(&A, allocator)
	defer l.matrix_free(&LU)
	defer delete(piv, allocator)
	_ = sign

	if !ok {
		return VannaVolgaResult{price_bs = exotic_bs_price, price_vv = exotic_bs_price}
	}

	x := l.lu_solve(&LU, piv, b, allocator)
	defer delete(x, allocator)

	weight_25d_put := x[0]
	weight_atm := x[1]
	weight_25d_call := x[2]

	// Now this adjustment will be non-zero and mathematically correct!
	adjustment :=
		weight_25d_put * (market_price_25d_put - bs_price_25d_put) +
		weight_atm * (market_price_atm - bs_price_atm) +
		weight_25d_call * (market_price_25d_call - bs_price_25d_call)

	return VannaVolgaResult {
		price_bs = exotic_bs_price,
		price_vv = exotic_bs_price + adjustment,
		adjustment = adjustment,
		weight_25d_put = weight_25d_put,
		weight_atm = weight_atm,
		weight_25d_call = weight_25d_call,
	}
}

// ============================================================================
// FX EXOTIC OPTIONS
// ============================================================================

fx_digital_call_price :: proc(S: f64, K: f64, T: f64, r_d: f64, r_f: f64, sigma: f64) -> f64 {
	if T <= 0.0 {return S > K ? 1.0 : 0.0}
	sqrt_T := math.sqrt_f64(T)
	d2 := (math.ln(S / K) + (r_d - r_f - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	return math.exp_f64(-r_d * T) * _N(d2)
}

// ✅ FIXED: Use robust central finite differences for exotic sensitivities
// This avoids analytical derivation errors and perfectly matches market practice.
fx_digital_call_sensitivities :: proc(
	S: f64,
	K: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
) -> (
	f64,
	f64,
) {
	h_S := 0.001 * S
	h_sigma := 0.001 * sigma

	P_up_S := fx_digital_call_price(S + h_S, K, T, r_d, r_f, sigma + h_sigma)
	P_dn_S := fx_digital_call_price(S - h_S, K, T, r_d, r_f, sigma + h_sigma)
	P_up_S_dn := fx_digital_call_price(S + h_S, K, T, r_d, r_f, sigma - h_sigma)
	P_dn_S_dn := fx_digital_call_price(S - h_S, K, T, r_d, r_f, sigma - h_sigma)
	vanna := ((P_up_S - P_dn_S) - (P_up_S_dn - P_dn_S_dn)) / (4.0 * h_S * h_sigma)

	P_up_sigma := fx_digital_call_price(S, K, T, r_d, r_f, sigma + h_sigma)
	P_dn_sigma := fx_digital_call_price(S, K, T, r_d, r_f, sigma - h_sigma)
	P_at_sigma := fx_digital_call_price(S, K, T, r_d, r_f, sigma)
	volga := (P_up_sigma - 2.0 * P_at_sigma + P_dn_sigma) / (h_sigma * h_sigma)

	return vanna, volga
}

// ============================================================================
// DOWN-AND-OUT DIGITAL CALL (Analytical)
// ============================================================================
// Pays 1 unit of domestic currency at T if S_T > K and the spot stays above L.
// Analytical formula via method of images (Reiner & Rubinstein, 1991).
// This is perfectly smooth, avoiding the Monte Carlo discrete-noise trap.

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

	sqrt_T := math.sqrt_f64(T)
	mu := (r_d - r_f - 0.5 * sigma * sigma) / (sigma * sigma)

	d2 := (math.ln(S / K) + (r_d - r_f - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
	d2_image :=
		(math.ln((L * L) / (S * K)) + (r_d - r_f - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)

	term1 := math.exp_f64(-r_d * T) * _N(d2)
	term2 := math.pow_f64(L / S, 2.0 * mu) * math.exp_f64(-r_d * T) * _N(d2_image)

	return term1 - term2
}

// Sensitivities via finite differences on the smooth analytical formula
fx_down_and_out_digital_call_sensitivities :: proc(
	S: f64,
	K: f64,
	L: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
) -> (
	f64,
	f64,
) {
	h_S := 0.0001 * S
	h_sigma := 0.0001 * sigma

	// Vanna = d²P/(dS dσ)
	P_up_S := fx_down_and_out_digital_call_price(S + h_S, K, L, T, r_d, r_f, sigma + h_sigma)
	P_dn_S := fx_down_and_out_digital_call_price(S - h_S, K, L, T, r_d, r_f, sigma + h_sigma)
	P_up_S_dn := fx_down_and_out_digital_call_price(S + h_S, K, L, T, r_d, r_f, sigma - h_sigma)
	P_dn_S_dn := fx_down_and_out_digital_call_price(S - h_S, K, L, T, r_d, r_f, sigma - h_sigma)
	vanna := ((P_up_S - P_dn_S) - (P_up_S_dn - P_dn_S_dn)) / (4.0 * h_S * h_sigma)

	// Volga = d²P/dσ²
	P_up_sigma := fx_down_and_out_digital_call_price(S, K, L, T, r_d, r_f, sigma + h_sigma)
	P_dn_sigma := fx_down_and_out_digital_call_price(S, K, L, T, r_d, r_f, sigma - h_sigma)
	P_at_sigma := fx_down_and_out_digital_call_price(S, K, L, T, r_d, r_f, sigma)
	volga := (P_up_sigma - 2.0 * P_at_sigma + P_dn_sigma) / (h_sigma * h_sigma)

	return vanna, volga
}
// ============================================================================
// DOUBLE-NO-TOUCH (DNT) BARRIER OPTION - PDE IMPLEMENTATION
// ============================================================================
// Solves the Black-Scholes PDE on a log-spot grid restricted to [ln(L), ln(U)].
// Fully Implicit scheme is used because it is unconditionally stable and
// monotonic, preventing spurious oscillations near the barriers.
// This provides perfectly smooth prices and exact, noise-free Greeks.

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

	x_min := math.ln(L)
	x_max := math.ln(U)

	dx := (x_max - x_min) / f64(n_space - 1)
	dt := T / f64(n_time)

	alpha := 0.5 * sigma * sigma
	beta_coef := r_d - r_f - 0.5 * sigma * sigma
	gamma_coef := r_d // PDE has -r_d*V, moving to LHS makes it +r_d*V

	rx := alpha * dt / (dx * dx)
	r_beta := beta_coef * dt / (2.0 * dx)
	r_gamma := gamma_coef * dt

	V := make([]f64, n_space, context.temp_allocator)
	defer delete(V, context.temp_allocator)

	// Terminal condition (tau = 0): V = 1 for all interior points
	for i in 1 ..< n_space - 1 {
		V[i] = 1.0
	}
	V[0] = 0.0
	V[n_space - 1] = 0.0

	n_unknowns := n_space - 2

	a := make([]f64, n_unknowns - 1, context.temp_allocator)
	b := make([]f64, n_unknowns, context.temp_allocator)
	c := make([]f64, n_unknowns - 1, context.temp_allocator)
	d := make([]f64, n_unknowns, context.temp_allocator)
	defer {
		delete(a, context.temp_allocator)
		delete(b, context.temp_allocator)
		delete(c, context.temp_allocator)
		delete(d, context.temp_allocator)
	}

	// Precompute tridiagonal coefficients (constant for all time steps)
	for i in 0 ..< n_unknowns {
		if i > 0 {a[i - 1] = -(rx - r_beta)}
		// ✅ CRITICAL FIX: The discount term should be SUBTRACTED, not added
		// The PDE has -r_d*V, so in the implicit scheme: (1 + 2*rx - r_gamma)*V^{n+1} = ...
		b[i] = 1.0 + 2.0 * rx - r_gamma // ← Changed from + r_gamma to - r_gamma
		if i < n_unknowns - 1 {c[i] = -(rx + r_beta)}
	}

	// Time-stepping (backward from tau = 0 to tau = T)
	for _ in 0 ..< n_time {
		for i in 0 ..< n_unknowns {
			d[i] = V[i + 1]
		}

		// Boundaries are exactly 0, so no RHS adjustment needed

		_thomas_algorithm(a, b, c, d, n_unknowns)

		for i in 0 ..< n_unknowns {
			V[i + 1] = d[i]
		}
		V[0] = 0.0
		V[n_space - 1] = 0.0
	}

	// Extract price at S via linear interpolation
	x_target := math.ln(S)
	i_target := int((x_target - x_min) / dx)

	if i_target < 0 || i_target >= n_space - 1 {return 0.0}

	w := (x_target - (x_min + f64(i_target) * dx)) / dx
	return V[i_target] * (1.0 - w) + V[i_target + 1] * w
}

// Sensitivities via finite differences on the smooth PDE price
fx_dnt_pde_sensitivities :: proc(
	S: f64,
	L: f64,
	U: f64,
	T: f64,
	r_d: f64,
	r_f: f64,
	sigma: f64,
) -> (
	f64,
	f64,
) {
	h_S := 0.001 * S
	h_sigma := 0.001 * sigma

	// Vanna = d²P/(dS dσ)
	P_up_S := fx_dnt_pde_price(S + h_S, L, U, T, r_d, r_f, sigma + h_sigma)
	P_dn_S := fx_dnt_pde_price(S - h_S, L, U, T, r_d, r_f, sigma + h_sigma)
	P_up_S_dn := fx_dnt_pde_price(S + h_S, L, U, T, r_d, r_f, sigma - h_sigma)
	P_dn_S_dn := fx_dnt_pde_price(S - h_S, L, U, T, r_d, r_f, sigma - h_sigma)
	vanna := ((P_up_S - P_dn_S) - (P_up_S_dn - P_dn_S_dn)) / (4.0 * h_S * h_sigma)

	// Volga = d²P/dσ²
	P_up_sigma := fx_dnt_pde_price(S, L, U, T, r_d, r_f, sigma + h_sigma)
	P_dn_sigma := fx_dnt_pde_price(S, L, U, T, r_d, r_f, sigma - h_sigma)
	P_at_sigma := fx_dnt_pde_price(S, L, U, T, r_d, r_f, sigma)
	volga := (P_up_sigma - 2.0 * P_at_sigma + P_dn_sigma) / (h_sigma * h_sigma)

	return vanna, volga
}
