package finance

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Market Data Structures
// ============================================================================

OptionQuote :: struct {
	strike:   f64,
	expiry:   f64, // years
	call_bid: f64,
	call_ask: f64,
	put_bid:  f64,
	put_ask:  f64,
}

VolSurfacePoint :: struct {
	strike:       f64,
	expiry:       f64,
	implied_vol:  f64,
	market_price: f64,
}

// ============================================================================
// SABR Model (Hagan et al. 2002)
// ============================================================================

SABR_Params :: struct {
	alpha: f64, // initial volatility
	beta:  f64, // CEV exponent (typically 0.5-1.0)
	rho:   f64, // correlation between spot and vol
	nu:    f64, // vol of vol
}

sabr_implied_vol :: proc(F: f64, K: f64, T: f64, params: SABR_Params) -> f64 {
	alpha := params.alpha
	beta := params.beta
	rho := params.rho
	nu := params.nu

	if alpha <= 0.0 || T <= 0.0 {return 0.0}

	if math.abs(F - K) < 1e-10 {
		F_beta := math.pow(F, 1.0 - beta)
		F_2beta := math.pow(F, 2.0 - 2.0 * beta)
		term1 := (1.0 - beta) * (1.0 - beta) / (24.0 * F_2beta)
		term2 := rho * beta * nu * alpha / (4.0 * F_beta)
		term3 := (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
		return alpha * (1.0 + (term1 + term2 + term3) * T) / F_beta
	}

	FK := F * K
	FK_mid := math.pow(FK, (1.0 - beta) / 2.0)
	log_FK := math.ln(F / K)

	z := (nu / alpha) * FK_mid * log_FK
	x_z := 0.0

	if math.abs(z) < 1e-10 {
		x_z = 1.0
	} else {
		sqrt_term := math.sqrt_f64(1.0 - 2.0 * rho * z + z * z)
		x_z = math.ln((sqrt_term + z - rho) / (1.0 - rho))
	}

	if math.abs(x_z) < 1e-10 {return 0.0}

	z_over_x := z / x_z
	term1 := (1.0 - beta) * (1.0 - beta) / 24.0 * log_FK * log_FK
	term2 :=
		(1.0 - beta) * (1.0 - beta) * (1.0 - beta) * (1.0 - beta) / 1920.0 * math.pow(log_FK, 4)
	denom := FK_mid * (1.0 + term1 + term2)

	sigma := alpha * z_over_x / denom
	corr1 := (1.0 - beta) * (1.0 - beta) / 24.0 * alpha * alpha / (FK_mid * FK_mid)
	corr2 := 0.25 * rho * beta * nu * alpha / FK_mid
	corr3 := (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu

	return sigma * (1.0 + (corr1 + corr2 + corr3) * T)
}

// ============================================================================
// Heston Model
// ============================================================================

Heston_Params :: struct {
	v0:    f64, // initial variance
	kappa: f64, // mean reversion speed
	theta: f64, // long-run variance
	sigma: f64, // vol of vol
	rho:   f64, // correlation
}

heston_characteristic_function :: proc(
	u: f64,
	S: f64,
	K: f64,
	r: f64,
	T: f64,
	params: Heston_Params,
	phi_type: int,
) -> complex128 {
	v0 := params.v0
	kappa := params.kappa
	theta := params.theta
	sigma := params.sigma
	rho := params.rho

	iu := complex(0.0, u)

	// d = sqrt((rho*sigma*i*u - kappa)^2 + sigma^2 * (i*u + u^2))
	a := complex(rho * sigma, 0.0) * iu - complex(kappa, 0.0)
	b := complex(sigma * sigma, 0.0)
	c := iu + complex(u * u, 0.0)

	d_squared := a * a + b * c
	d := cmath_sqrt(d_squared)

	// g = (kappa - rho*sigma*i*u + d) / (kappa - rho*sigma*i*u - d)
	num := complex(kappa, 0.0) - complex(rho * sigma, 0.0) * iu + d
	den := complex(kappa, 0.0) - complex(rho * sigma, 0.0) * iu - d
	g := num / den

	exp_dT := cmath_exp(-d * complex(T, 0.0))

	D :=
		(complex(kappa, 0.0) - complex(rho * sigma, 0.0) * iu + d) /
		complex(sigma * sigma, 0.0) *
		(complex(1.0, 0.0) - exp_dT) /
		(complex(1.0, 0.0) - g * exp_dT)

	C_part1 := complex(r, 0.0) * iu * complex(T, 0.0)
	C_part2 :=
		(complex(kappa, 0.0) - complex(rho * sigma, 0.0) * iu + d) *
		complex(T, 0.0) /
		complex(sigma * sigma, 0.0)
	C_part3 :=
		complex(2.0, 0.0) *
		cmath_log(
			(complex(1.0, 0.0) - g * cmath_exp(-d * complex(T, 0.0))) / (complex(1.0, 0.0) - g),
		)

	C := C_part1 + (C_part2 - C_part3) * complex(kappa * theta / (sigma * sigma), 0.0)

	return cmath_exp(C + D * complex(v0, 0.0) + iu * cmath_log(complex(S, 0.0)))
}

heston_price :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	opt: OptionType,
	n_points: int = 100,
) -> f64 {
	if T <= 0.0 {
		if opt == .Call {return math.max(S - K, 0.0)}
		return math.max(K - S, 0.0)
	}

	u_max := 100.0
	du := u_max / f64(n_points)
	P1 := 0.0
	P2 := 0.0

	for i in 0 ..< n_points {
		u := (f64(i) + 0.5) * du

		phi1 := heston_characteristic_function(u, S, K, r, T, params, 1)
		integrand1 := real(cmath_exp(-complex(0.0, u * math.ln(K))) * phi1 / complex(0.0, u))
		P1 += integrand1 * du

		phi2 := heston_characteristic_function(u, S, K, r, T, params, 2)
		integrand2 := real(cmath_exp(-complex(0.0, u * math.ln(K))) * phi2 / complex(0.0, u))
		P2 += integrand2 * du
	}

	P1 = 0.5 + P1 / math.PI
	P2 = 0.5 + P2 / math.PI
	call_price := S * P1 - K * math.exp(-r * T) * P2

	if opt == .Call {return call_price}
	return call_price - S + K * math.exp(-r * T)
}

// ============================================================================
// Calibration
// ============================================================================

SABR_CalibrationResult :: struct {
	params:     SABR_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}

Heston_CalibrationResult :: struct {
	params:     Heston_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}

calibrate_sabr :: proc(
	market_data: []VolSurfacePoint,
	S: f64,
	r: f64,
	allocator: mem.Allocator = context.allocator,
) -> SABR_CalibrationResult {
	params := SABR_Params {
		alpha = 0.2,
		beta  = 0.7,
		rho   = -0.3,
		nu    = 0.4,
	}
	lr := 0.01
	best_rmse: f64 = 1e10
	best_params := params

	for iter in 0 ..< 200 {
		rmse := compute_sabr_rmse(market_data, S, r, params)
		if rmse < best_rmse {
			best_rmse = rmse
			best_params = params
		}
		if rmse < 1e-6 {
			return SABR_CalibrationResult {
				params = best_params,
				rmse = best_rmse,
				converged = true,
				iterations = iter + 1,
			}
		}

		eps := 1e-5
		params_plus, params_minus := params, params

		params_plus.alpha += eps; params_minus.alpha -= eps
		params.alpha -=
			lr *
			(compute_sabr_rmse(market_data, S, r, params_plus) -
					compute_sabr_rmse(market_data, S, r, params_minus)) /
			(2.0 * eps)

		params_plus.rho += eps; params_minus.rho -= eps
		params.rho -=
			lr *
			(compute_sabr_rmse(market_data, S, r, params_plus) -
					compute_sabr_rmse(market_data, S, r, params_minus)) /
			(2.0 * eps)
		params.rho = math.max(-0.99, math.min(0.99, params.rho))

		params_plus.nu += eps; params_minus.nu -= eps
		params.nu -=
			lr *
			(compute_sabr_rmse(market_data, S, r, params_plus) -
					compute_sabr_rmse(market_data, S, r, params_minus)) /
			(2.0 * eps)
		params.nu = math.max(0.01, params.nu)
	}
	return SABR_CalibrationResult {
		params = best_params,
		rmse = best_rmse,
		converged = false,
		iterations = 200,
	}
}

compute_sabr_rmse :: proc(
	market_data: []VolSurfacePoint,
	S: f64,
	r: f64,
	params: SABR_Params,
) -> f64 {
	rmse := 0.0
	for point in market_data {
		F := S * math.exp(r * point.expiry)
		err := sabr_implied_vol(F, point.strike, point.expiry, params) - point.implied_vol
		rmse += err * err
	}
	return math.sqrt_f64(rmse / f64(len(market_data)))
}

calibrate_heston :: proc(
	market_data: []VolSurfacePoint,
	S: f64,
	r: f64,
	allocator: mem.Allocator = context.allocator,
) -> Heston_CalibrationResult {
	params := Heston_Params {
		v0    = 0.04,
		kappa = 2.0,
		theta = 0.04,
		sigma = 0.3,
		rho   = -0.7,
	}
	lr := 0.001
	best_rmse: f64 = 1e10
	best_params := params

	for iter in 0 ..< 100 {
		rmse := compute_heston_rmse(market_data, S, r, params)
		if rmse < best_rmse {
			best_rmse = rmse
			best_params = params
		}
		if rmse < 1e-4 {
			return Heston_CalibrationResult {
				params = best_params,
				rmse = best_rmse,
				converged = true,
				iterations = iter + 1,
			}
		}

		eps := 1e-4
		params_plus, params_minus := params, params

		params_plus.v0 += eps; params_minus.v0 -= eps
		params.v0 -=
			lr *
			(compute_heston_rmse(market_data, S, r, params_plus) -
					compute_heston_rmse(market_data, S, r, params_minus)) /
			(2.0 * eps)
		params.v0 = math.max(0.001, params.v0)

		params_plus.rho += eps; params_minus.rho -= eps
		params.rho -=
			lr *
			(compute_heston_rmse(market_data, S, r, params_plus) -
					compute_heston_rmse(market_data, S, r, params_minus)) /
			(2.0 * eps)
		params.rho = math.max(-0.99, math.min(0.99, params.rho))
	}
	return Heston_CalibrationResult {
		params = best_params,
		rmse = best_rmse,
		converged = false,
		iterations = 100,
	}
}

compute_heston_rmse :: proc(
	market_data: []VolSurfacePoint,
	S: f64,
	r: f64,
	params: Heston_Params,
) -> f64 {
	rmse := 0.0
	for point in market_data {
		model_price := heston_price(S, point.strike, point.expiry, r, params, .Call, 50)
		bs_price := black_scholes_call(S, point.strike, point.expiry, r, point.implied_vol)
		err := model_price - bs_price
		rmse += err * err
	}
	return math.sqrt_f64(rmse / f64(len(market_data)))
}

black_scholes_call :: proc(S: f64, K: f64, T: f64, r: f64, sigma: f64) -> f64 {
	if T <= 0.0 {return math.max(S - K, 0.0)}
	d1 := (math.ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt_f64(T))
	d2 := d1 - sigma * math.sqrt_f64(T)
	N_d1 := 0.5 * (1.0 + math.erf(d1 / math.sqrt_f64(2.0)))
	N_d2 := 0.5 * (1.0 + math.erf(d2 / math.sqrt_f64(2.0)))
	return S * N_d1 - K * math.exp(-r * T) * N_d2
}

// Complex helpers (Odin uses .re and .im)
cmath_sqrt :: proc(z: complex128) -> complex128 {
	r := cmath_abs(z)
	theta := cmath_arg(z)
	return cmath_polar(math.sqrt_f64(r), theta / 2.0)
}

cmath_exp :: proc(z: complex128) -> complex128 {
	r := cmath_abs(z)
	theta := cmath_arg(z)
	return complex(r * math.cos(theta), r * math.sin(theta))
}

cmath_log :: proc(z: complex128) -> complex128 {
	r := cmath_abs(z)
	theta := cmath_arg(z)
	return complex(math.ln(r), theta)
}

cmath_abs :: proc(z: complex128) -> f64 {
	return math.sqrt_f64(real(z) * real(z) + imag(z) * imag(z))
}

cmath_arg :: proc(z: complex128) -> f64 {
	return math.atan2_f64(imag(z), real(z))
}

cmath_polar :: proc(r: f64, theta: f64) -> complex128 {
	return complex(r * math.cos(theta), r * math.sin(theta))
}
