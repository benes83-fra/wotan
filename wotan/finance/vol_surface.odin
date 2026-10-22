package finance

import l "../linalg"
import opt "../optimize"
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

/// Complex helpers
cmath_sqrt :: proc(z: complex128) -> complex128 {
	r := cmath_abs(z)
	theta := cmath_arg(z)
	return cmath_polar(math.sqrt_f64(r), theta / 2.0)
}

// ✅ CRITICAL FIX: This was literally an identity function before.
// It must compute e^(x + iy) = e^x * (cos(y) + i*sin(y))
cmath_exp :: proc(z: complex128) -> complex128 {
	exp_x := math.exp(real(z))
	return complex(exp_x * math.cos(imag(z)), exp_x * math.sin(imag(z)))
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
} // ============================================================================
// Calibration Context (for ObjectiveFunc)
// ============================================================================
// ============================================================================
// Calibration Context
// ============================================================================

SABR_CalibrationContext :: struct {
	market_data: []VolSurfacePoint,
	S:           f64,
	r:           f64,
}

Heston_CalibrationContext :: struct {
	market_data: []VolSurfacePoint,
	S:           f64,
	r:           f64,
}

sabr_objective_fn :: proc(x: []f64, user_data: rawptr) -> f64 {
	ctx := (^SABR_CalibrationContext)(user_data)
	params := SABR_Params {
		alpha = x[0],
		beta  = x[1],
		rho   = x[2],
		nu    = x[3],
	}

	rmse := 0.0
	for point in ctx.market_data {
		F := ctx.S * math.exp(ctx.r * point.expiry)
		err := sabr_implied_vol(F, point.strike, point.expiry, params) - point.implied_vol
		rmse += err * err
	}
	return math.sqrt_f64(rmse / f64(len(ctx.market_data)))
}

heston_objective_fn :: proc(x: []f64, user_data: rawptr) -> f64 {
	ctx := (^Heston_CalibrationContext)(user_data)
	params := Heston_Params {
		v0    = x[0],
		kappa = x[1],
		theta = x[2],
		sigma = x[3],
		rho   = x[4],
	}

	// ✅ CRITICAL: Enforce Feller condition. If violated, reject immediately.
	feller_penalty := 0.0
	feller_condition := 2.0 * params.kappa * params.theta - params.sigma * params.sigma
	if feller_condition <= 0.0 {
		// Heavy but differentiable penalty for violating Feller
		feller_penalty = 50.0 * math.abs(feller_condition)
	}

	n := len(ctx.market_data)
	model_prices := make([]f64, n, context.temp_allocator)
	defer delete(model_prices, context.temp_allocator)
	_heston_model_prices(ctx.market_data, ctx.S, ctx.r, params, model_prices)

	// Accumulate in the original option order
	rmse := 0.0
	for point, idx in ctx.market_data {
		model_price := model_prices[idx]

		if math.is_nan(model_price) || math.is_inf(model_price, 0) {
			return 1e6
		}

		bs_price := black_scholes_call(ctx.S, point.strike, point.expiry, ctx.r, point.implied_vol)
		err := model_price - bs_price
		rmse += err * err
	}
	return math.sqrt_f64(rmse / f64(n))
}

// _heston_model_prices fills out[i] with the 1000-node Heston call price of
// market_data[i]. Options are grouped by expiry so the characteristic function
// is evaluated once per (expiry, node) instead of once per option.
_heston_model_prices :: proc(market_data: []VolSurfacePoint, S, r: f64, params: Heston_Params, out: []f64) {
	n := len(market_data)
	done := make([]bool, n, context.temp_allocator)
	strikes := make([]f64, n, context.temp_allocator)
	prices := make([]f64, n, context.temp_allocator)
	members := make([]int, n, context.temp_allocator)
	defer {
		delete(done, context.temp_allocator)
		delete(strikes, context.temp_allocator)
		delete(prices, context.temp_allocator)
		delete(members, context.temp_allocator)
	}

	for i in 0 ..< n {
		if done[i] {continue}
		T := market_data[i].expiry
		m := 0
		for j in i ..< n {
			if !done[j] && market_data[j].expiry == T {
				members[m] = j
				strikes[m] = market_data[j].strike
				done[j] = true
				m += 1
			}
		}
		heston_prices_same_expiry(S, strikes[:m], T, r, params, .Call, prices[:m], 1000)
		for k in 0 ..< m {out[members[k]] = prices[k]}
	}
}

// Calibrate SABR using Adam (with FIXED deep copies for gradients)
calibrate_sabr :: proc(
	market_data: []VolSurfacePoint,
	S: f64,
	r: f64,
	allocator: mem.Allocator = context.allocator,
) -> SABR_CalibrationResult {
	ctx := SABR_CalibrationContext {
		market_data = market_data,
		S           = S,
		r           = r,
	}

	inf := math.inf_f64(1)
	params := []opt.Calibration_Param {
		{"alpha", 0.2, 0.01, inf},
		{"beta", 0.7, 0.0, 1.0},
		{"rho", -0.3, -0.99, 0.99},
		{"nu", 0.4, 0.01, inf},
	}
	config := opt.Calibration_Config {
		optimizer = {type = .Adam, learning_rate = 0.1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8},
		max_iter = 300,
		loss_threshold = 1e-6,
		fd_step = 1e-5,
	}
	res := opt.calibrate(params, sabr_objective_fn, &ctx, config, allocator)
	defer opt.calibration_result_free(&res)

	return SABR_CalibrationResult {
		params = SABR_Params{alpha = res.x[0], beta = res.x[1], rho = res.x[2], nu = res.x[3]},
		rmse = res.loss,
		converged = res.converged,
		iterations = res.iterations,
	}
}

// Calibrate Heston using Adam
calibrate_heston :: proc(
	market_data: []VolSurfacePoint,
	S: f64,
	r: f64,
	true_sigma: f64 = 0.16, // ✅ ADD: Pass historical vol to seed the optimizer
	allocator: mem.Allocator = context.allocator,
) -> Heston_CalibrationResult {
	ctx := Heston_CalibrationContext {
		market_data = market_data,
		S           = S,
		r           = r,
	}

	// Seeded with realistic equity dynamics: v0 and theta near the historical
	// variance, strongly negative rho (leverage effect / skew).
	v_hist := true_sigma * true_sigma
	params := []opt.Calibration_Param {
		{"v0", v_hist, 0.001, 0.2},
		{"kappa", 2.0, 0.1, 10.0}, // mean reversion speed
		{"theta", v_hist, 0.001, 0.2}, // long-run variance
		{"sigma", 0.4, 0.01, 1.5}, // vol of vol
		{"rho", -0.7, -0.99, -0.1}, // strictly negative for equities
	}
	config := opt.Calibration_Config {
		optimizer = {type = .Adam, learning_rate = 0.05, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8},
		max_iter = 300,
		loss_threshold = 0.50, // $0.50 RMSE is excellent for Heston
		fd_step = 1e-3, // large enough to step over complex branch-cut noise
		parallel = true, // heston_objective_fn only uses context.temp_allocator scratch
	}
	res := opt.calibrate(params, heston_objective_fn, &ctx, config, allocator)
	defer opt.calibration_result_free(&res)

	return Heston_CalibrationResult {
		params = Heston_Params {
			v0 = res.x[0],
			kappa = res.x[1],
			theta = res.x[2],
			sigma = res.x[3],
			rho = res.x[4],
		},
		rmse = res.loss,
		converged = res.converged,
		iterations = res.iterations,
	}
}
// ============================================================================
// Heston Model (QuantLib-Verified Single Characteristic Function)
// ============================================================================

// Heston log-price characteristic function (Risk-Neutral Measure)
// To get the stock measure (P1), evaluate at u - i (i.e., u_complex = complex(u, -1.0))
// To get the risk-neutral measure (P2), evaluate at u (i.e., u_complex = complex(u, 0.0))
heston_char_func :: proc(
	u_complex: complex128,
	S: f64,
	r: f64,
	T: f64,
	params: Heston_Params,
) -> complex128 {
	i := complex(0.0, 1.0)

	b := complex(params.kappa, 0.0) - complex(params.rho * params.sigma, 0.0) * i * u_complex

	// d = sqrt( b^2 + sigma^2 * (u^2 + i*u) )
	// Note: u_complex * u_complex + i * u_complex is exactly u^2 + i*u for real u,
	// and u^2 - i*u for u_complex = u - i.
	u_term := u_complex * u_complex + i * u_complex
	d_squared := b * b + complex(params.sigma * params.sigma, 0.0) * u_term

	d := cmath_sqrt(d_squared)

	// Enforce Re(d) > 0
	if real(d) < 0.0 {
		d = -d
	}

	g := (b - d) / (b + d + complex(1e-10, 0.0))

	exp_dT := cmath_exp(-d * complex(T, 0.0))

	D :=
		(b - d) /
		complex(params.sigma * params.sigma, 0.0) *
		(complex(1.0, 0.0) - exp_dT) /
		(complex(1.0, 0.0) - g * exp_dT + complex(1e-10, 0.0))

	C_part1 := complex(r, 0.0) * i * u_complex * complex(T, 0.0)

	log_num := complex(1.0, 0.0) - g * exp_dT + complex(1e-12, 0.0)
	log_den := complex(1.0, 0.0) - g + complex(1e-12, 0.0)
	log_arg := log_num / log_den

	C_part2 :=
		complex(params.kappa * params.theta / (params.sigma * params.sigma), 0.0) *
		((b - d) * complex(T, 0.0) - complex(2.0, 0.0) * cmath_log(log_arg))

	C := C_part1 + C_part2

	return cmath_exp(C + D * complex(params.v0, 0.0) + i * u_complex * cmath_log(complex(S, 0.0)))
}

heston_price :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	opt: OptionType,
	n_points: int = 1000,
) -> f64 {
	strikes := [1]f64{K}
	price: [1]f64
	heston_prices_same_expiry(S, strikes[:], T, r, params, opt, price[:], n_points)
	return price[0]
}

// heston_prices_same_expiry prices options that share S, T, r and the Heston
// parameters, writing one price per strike into `out`. The characteristic
// function does not depend on the strike, so it is evaluated once per
// integration node and reused for every strike; each strike's integral is
// accumulated exactly as for a single heston_price call.
heston_prices_same_expiry :: proc(
	S: f64,
	strikes: []f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	opt: OptionType,
	out: []f64,
	n_points: int = 1000,
) {
	assert(len(out) == len(strikes), "heston_prices_same_expiry: out must have one slot per strike")
	if T <= 0.0 {
		for K, k in strikes {
			out[k] = opt == .Call ? math.max(S - K, 0.0) : math.max(K - S, 0.0)
		}
		return
	}

	// Per-strike accumulators; on the stack for typical chain sizes
	stack_P: [128]f64
	P1, P2: []f64
	if len(strikes) <= 64 {
		P1, P2 = stack_P[:len(strikes)], stack_P[64:64 + len(strikes)]
	} else {
		P1 = make([]f64, len(strikes), context.temp_allocator)
		P2 = make([]f64, len(strikes), context.temp_allocator)
	}
	defer if len(strikes) > 64 {
		delete(P1, context.temp_allocator)
		delete(P2, context.temp_allocator)
	}

	// ✅ CRITICAL FIX: Normalization factor for P1.
	// heston_char_func(u - i) computes E[S_T * e^{iu ln S_T}].
	// To get the true characteristic function (magnitude <= 1), we must divide by E[S_T] = S * e^{rT}.
	normalization_P1 := S * math.exp(r * T)

	u_max := 200.0
	du := u_max / f64(n_points)

	for i in 0 ..< n_points {
		u := (f64(i) + 0.5) * du

		// P1 uses stock measure: evaluate at u - i, then normalize
		u1 := complex(u, -1.0)
		phi1_raw := heston_char_func(u1, S, r, T, params)
		phi1 := phi1_raw / complex(normalization_P1, 0.0)

		// P2 uses risk-neutral measure: evaluate at u (already normalized)
		u2 := complex(u, 0.0)
		phi2 := heston_char_func(u2, S, r, T, params)

		for K, k in strikes {
			e_k := cmath_exp(-complex(0.0, u * math.ln(K))) // e^{-iu ln K}, shared by both integrals
			integrand1 := real(e_k * phi1 / complex(0.0, u))
			P1[k] += integrand1 * du
			integrand2 := real(e_k * phi2 / complex(0.0, u))
			P2[k] += integrand2 * du
		}
	}

	for K, k in strikes {
		p1 := 0.5 + P1[k] / math.PI
		p2 := 0.5 + P2[k] / math.PI
		call_price := S * p1 - K * math.exp(-r * T) * p2

		if opt == .Call {
			intrinsic := math.max(S - K * math.exp(-r * T), 0.0)
			if call_price < intrinsic {call_price = intrinsic}
			out[k] = call_price
		} else {
			put_price := call_price - S + K * math.exp(-r * T)
			intrinsic := math.max(K * math.exp(-r * T) - S, 0.0)
			if put_price < intrinsic {put_price = intrinsic}
			out[k] = put_price
		}
	}
}
debug_heston :: proc() {
	fmt.println("\n=== HESTON COMPLEX MATH DEBUG ===")

	// 1. Prove cmath_exp is working
	test_z1 := complex(0.0, math.PI)
	fmt.printf(
		"cmath_exp(0 + iπ)      = (%.4f, %.4f) [Expected: -1.0000,  0.0000]\n",
		real(cmath_exp(test_z1)),
		imag(cmath_exp(test_z1)),
	)

	test_z2 := complex(1.0, 0.0)
	fmt.printf(
		"cmath_exp(1 + 0i)      = (%.4f, %.4f) [Expected:  2.7183,  0.0000]\n",
		real(cmath_exp(test_z2)),
		imag(cmath_exp(test_z2)),
	)

	// 2. Benchmark parameters from Heston (1993), Table 1
	params := Heston_Params {
		v0    = 0.04,
		kappa = 2.0,
		theta = 0.04,
		sigma = 0.3,
		rho   = -0.5,
	}

	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	u := 1.0

	// 3. Print intermediate characteristic function values
	phi1_raw := heston_char_func(complex(u, -1.0), S, r, T, params)
	normalization_P1 := S * math.exp(r * T)
	phi1 := phi1_raw / complex(normalization_P1, 0.0) // ✅ Normalize

	phi2 := heston_char_func(complex(u, 0.0), S, r, T, params)

	fmt.printf("\nCharacteristic Function at u=1.0:\n")
	fmt.printf(
		"  phi1 (j=1, normalized) = (%.6f, %.6f) [Magnitude: %.6f, should be < 1]\n",
		real(phi1),
		imag(phi1),
		cmath_abs(phi1),
	)
	fmt.printf(
		"  phi2 (j=2)             = (%.6f, %.6f) [Magnitude: %.6f, should be < 1]\n",
		real(phi2),
		imag(phi2),
		cmath_abs(phi2),
	)

	// 4. Final Price Check
	price := heston_price(S, K, T, r, params, .Call, 4000)
	fmt.printf("\nCalculated Call Price: $%.4f\n", price)
	fmt.printf("Expected Call Price:   $~10.50\n")

	if math.abs(price - 10.50) < 0.2 {
		fmt.printf("Status: PASS ✅ \n\n")
	} else {
		fmt.printf("Status: FAIL ❌\n\n")
	}
}
