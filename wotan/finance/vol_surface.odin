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

// Hagan's lognormal implied volatility:
//   σ_B = α / [(FK)^{(1−β)/2}·(1 + (1−β)²/24·ℓ² + (1−β)⁴/1920·ℓ⁴)] · z/x(z)
//         · (1 + [(1−β)²/24·α²/(FK)^{1−β} + ρβνα/(4(FK)^{(1−β)/2}) + (2−3ρ²)/24·ν²]·T)
// with ℓ = ln(F/K), z = ν/α·(FK)^{(1−β)/2}·ℓ, x(z) = ln((√(1−2ρz+z²) + z − ρ)/(1−ρ)).
// One expression covers the money too: z/x(z) → 1 as z → 0.
sabr_implied_vol :: proc(F: f64, K: f64, T: f64, params: SABR_Params) -> f64 {
	alpha := params.alpha
	beta := params.beta
	rho := params.rho
	nu := params.nu

	if alpha <= 0.0 || T <= 0.0 || F <= 0.0 || K <= 0.0 {return 0.0}

	omb := 1.0 - beta
	log_FK := math.ln(F / K)
	FK_mid := math.pow(F * K, 0.5 * omb)

	// x(z) = log1p((√(1−2ρz+z²) − 1 + z)/(1−ρ)), with √(1−2ρz+z²) − 1 written as
	// (z² − 2ρz)/(√(1−2ρz+z²) + 1): accurate for small z, where z/x(z) → 1
	z := (nu / alpha) * FK_mid * log_FK
	z_over_x := 1.0
	if z != 0.0 {
		q := math.sqrt_f64(1.0 - 2.0 * rho * z + z * z)
		z_over_x = z / math.log1p(((z * z - 2.0 * rho * z) / (q + 1.0) + z) / (1.0 - rho))
	}

	l2 := log_FK * log_FK
	omb2 := omb * omb
	denom := FK_mid * (1.0 + omb2 / 24.0 * l2 + omb2 * omb2 / 1920.0 * l2 * l2)
	corr1 := omb2 / 24.0 * alpha * alpha / (FK_mid * FK_mid)
	corr2 := 0.25 * rho * beta * nu * alpha / FK_mid
	corr3 := (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu

	return alpha / denom * z_over_x * (1.0 + (corr1 + corr2 + corr3) * T)
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
	params:        Heston_Params,
	rmse:          f64, // pricing RMSE (excludes any Feller penalty)
	iv_rmse:       f64, // implied-volatility RMSE (what is minimized)
	converged:     bool,
	iterations:    int,
	feller_margin: f64, // 2*kappa*theta - sigma^2; > 0 means the Feller condition holds
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
	n := len(market_data)
	if n == 0 {return 0}
	model := make([]f64, n, context.temp_allocator)
	defer delete(model, context.temp_allocator)
	_heston_model_prices(market_data, S, r, params, model)
	rmse := 0.0
	for point, i in market_data {
		err := model[i] - black_scholes_call(S, point.strike, point.expiry, r, point.implied_vol)
		rmse += err * err
	}
	return math.sqrt_f64(rmse / f64(n))
}

// Black-Scholes call at full precision (also deep out of the money)
black_scholes_call :: proc(S: f64, K: f64, T: f64, r: f64, sigma: f64) -> f64 {
	return compute_greeks(S, K, T, r, sigma, .Call).price
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
}

// ln(1 + z), accurate for small |z| (Kahan: the rounding of 1 + z is undone)
cmath_log1p :: proc(z: complex128) -> complex128 {
	w := complex(1.0, 0.0) + z
	if w == complex(1.0, 0.0) {return z}
	return cmath_log(w) * z / (w - complex(1.0, 0.0))
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
	market_data:   []VolSurfacePoint,
	S:             f64,
	r:             f64,
	feller_weight: f64, // heston_objective_fn: penalty per unit of Feller violation (0 = none)
	// Used by calibrate_heston's residuals: market prices and vegas of the quotes
	market_prices: []f64,
	vegas:         []f64,
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

	// Optional Feller condition (2*kappa*theta > sigma^2): a penalty proportional
	// to the violation. Fourier pricing does not need it; Monte Carlo simulation
	// of the variance process does (variance can otherwise hit zero).
	feller_penalty := 0.0
	feller_condition := 2.0 * params.kappa * params.theta - params.sigma * params.sigma
	if ctx.feller_weight > 0.0 && feller_condition <= 0.0 {
		feller_penalty = ctx.feller_weight * math.abs(feller_condition)
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
	return math.sqrt_f64(rmse / f64(n)) + feller_penalty
}

// _heston_model_prices fills out[i] with the Heston call price (COS) of
// market_data[i]. Options are grouped by expiry so the characteristic function
// is evaluated once per (expiry, node) instead of once per option.
_heston_model_prices :: proc(
	market_data: []VolSurfacePoint,
	S, r: f64,
	params: Heston_Params,
	out: []f64,
) {
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
		heston_prices_cos(S, strikes[:m], T, r, params, .Call, prices[:m])
		for k in 0 ..< m {out[members[k]] = prices[k]}
	}
}

// Residuals of the SABR fit: model minus market implied volatility
_sabr_residuals :: proc(x: []f64, res: []f64, user_data: rawptr) {
	ctx := (^SABR_CalibrationContext)(user_data)
	params := SABR_Params {
		alpha = x[0],
		beta  = x[1],
		rho   = x[2],
		nu    = x[3],
	}
	for point, i in ctx.market_data {
		F := ctx.S * math.exp(ctx.r * point.expiry)
		res[i] = sabr_implied_vol(F, point.strike, point.expiry, params) - point.implied_vol
	}
}

// Calibrate one SABR parameter set to all quotes by Levenberg-Marquardt on the
// implied-volatility errors. Starts from β = 0.7, ρ = −0.3, ν = 0.4 and the α
// that reproduces the at-the-money volatility of the shortest expiry. (β and ρ
// both tilt the smile and are only weakly identified jointly; fix β by setting
// equal bounds if it is known.)
calibrate_sabr :: proc(
	market_data: []VolSurfacePoint,
	S: f64,
	r: f64,
	allocator: mem.Allocator = context.allocator,
) -> SABR_CalibrationResult {
	if len(market_data) == 0 {return {}}
	ctx := SABR_CalibrationContext {
		market_data = market_data,
		S           = S,
		r           = r,
	}

	beta0 := 0.7
	atm, T_atm := _atm_vol(market_data, S, r, true)
	F_atm := S * math.exp(r * T_atm)
	alpha0 := atm * math.pow(F_atm, 1.0 - beta0)

	inf := math.inf_f64(1)
	params := []opt.Calibration_Param {
		{"alpha", alpha0, 1e-6, inf, .Linear},
		{"beta", beta0, 0.0, 1.0, .Linear},
		{"rho", -0.3, -0.999, 0.999, .Linear},
		{"nu", 0.4, 1e-6, inf, .Linear},
	}
	res := opt.least_squares(
		params,
		_sabr_residuals,
		len(market_data),
		&ctx,
		{max_iter = 200},
		allocator,
	)
	defer opt.least_squares_result_free(&res)

	return SABR_CalibrationResult {
		params = SABR_Params{alpha = res.x[0], beta = res.x[1], rho = res.x[2], nu = res.x[3]},
		rmse = res.rms,
		converged = res.converged,
		iterations = res.iterations,
	}
}

// At-the-money implied volatility (strike closest to the forward) of the
// shortest (or longest) expiry, and that expiry
@(private = "file")
_atm_vol :: proc(market_data: []VolSurfacePoint, S, r: f64, shortest: bool) -> (vol, T: f64) {
	T = market_data[0].expiry
	for p in market_data {
		if shortest ? p.expiry < T : p.expiry > T {T = p.expiry}
	}
	best := math.inf_f64(1)
	F := S * math.exp(r * T)
	for p in market_data {
		if p.expiry != T || !(p.implied_vol > 0) {continue}
		if d := abs(math.ln(p.strike / F)); d < best {best, vol = d, p.implied_vol}
	}
	return
}

// Heston parameters from the optimizer variables. With the Feller condition
// enforced, x[3] is σ/√(2κθ) ∈ (0, 1) instead of σ.
@(private = "file")
_heston_from_x :: proc(x: []f64, enforce_feller: bool) -> Heston_Params {
	p := Heston_Params {
		v0    = x[0],
		kappa = x[1],
		theta = x[2],
		sigma = x[3],
		rho   = x[4],
	}
	if enforce_feller {p.sigma = x[3] * math.sqrt(2.0 * p.kappa * p.theta)}
	return p
}

// Implied-volatility errors of the model prices (QuantLib's ImpliedVolError).
// A model price outside the no-arbitrage bounds has no implied volatility; it
// contributes its vega-weighted price error (model − market)/vega instead.
_heston_residuals :: proc(x: []f64, res: []f64, user_data: rawptr) {
	ctx := (^Heston_CalibrationContext)(user_data)
	params := _heston_from_x(x, ctx.feller_weight > 0)
	model := make([]f64, len(ctx.market_data), context.temp_allocator)
	defer delete(model, context.temp_allocator)
	_heston_model_prices(ctx.market_data, ctx.S, ctx.r, params, model)
	for p, i in ctx.market_data {
		iv, ok, _ := implied_volatility(
			model[i],
			ctx.S,
			p.strike,
			p.expiry,
			ctx.r,
			.Call,
			tol = 1e-12,
		)
		res[i] = iv - p.implied_vol if ok else (model[i] - ctx.market_prices[i]) / ctx.vegas[i]
	}
}

// Calibrate Heston by Levenberg-Marquardt on implied-volatility errors (so
// cheap wings and short expiries count as much as expensive options). v0 and θ
// start from the at-the-money implied variance of the shortest and longest
// expiry (true_sigma² if there is none), κ = 2, σ = 0.5, ρ = −0.5.
calibrate_heston :: proc(
	market_data: []VolSurfacePoint,
	S: f64,
	r: f64,
	true_sigma: f64 = 0.16, // fallback seed for v0 and θ
	allocator: mem.Allocator = context.allocator,
	// > 0 enforces the Feller condition 2κθ > σ² exactly (σ is fitted as a
	// fraction of √(2κθ)); use it when the parameters feed a Monte Carlo
	// simulation of the variance. 0 = unconstrained.
	feller_weight: f64 = 0.0,
) -> Heston_CalibrationResult {
	n := len(market_data)
	if n == 0 {return {}}
	ctx := Heston_CalibrationContext {
		market_data   = market_data,
		S             = S,
		r             = r,
		feller_weight = feller_weight,
		market_prices = make([]f64, n, allocator),
		vegas         = make([]f64, n, allocator),
	}
	defer {
		delete(ctx.market_prices, allocator)
		delete(ctx.vegas, allocator)
	}
	for p, i in market_data {
		g := compute_greeks(S, p.strike, p.expiry, r, p.implied_vol, .Call)
		ctx.market_prices[i] = g.price
		// Floor at 0.1% of the at-the-money vega so far wings cannot dominate
		ctx.vegas[i] = max(100.0 * g.vega, 4e-4 * S * math.sqrt(max(p.expiry, 1e-8)))
	}

	short_vol, _ := _atm_vol(market_data, S, r, true)
	long_vol, _ := _atm_vol(market_data, S, r, false)
	v0 := short_vol * short_vol if short_vol > 0 else true_sigma * true_sigma
	theta := long_vol * long_vol if long_vol > 0 else true_sigma * true_sigma

	enforce := feller_weight > 0.0
	params := []opt.Calibration_Param {
		{"v0", v0, 1e-4, 1.0, .Linear},
		{"kappa", 2.0, 0.01, 20.0, .Linear}, // mean reversion speed
		{"theta", theta, 1e-4, 1.0, .Linear}, // long-run variance
		{"sigma", 0.5, 0.01, 3.0, .Linear}, // vol of vol
		{"rho", -0.5, -0.999, 0.999, .Linear},
	}
	if enforce {
		params[3] = {"sigma/sqrt(2*kappa*theta)", 0.5, 0.01, 0.999, .Linear}
	}
	res := opt.least_squares(
		params,
		_heston_residuals,
		n,
		&ctx,
		{max_iter = 200, parallel = true},
		allocator,
	)
	defer opt.least_squares_result_free(&res)

	p := _heston_from_x(res.x, enforce)
	// Report the pure pricing RMSE (no penalty)
	ctx.feller_weight = 0.0
	rmse := heston_objective_fn([]f64{p.v0, p.kappa, p.theta, p.sigma, p.rho}, &ctx)
	return Heston_CalibrationResult {
		params = p,
		rmse = rmse,
		iv_rmse = res.rms,
		converged = res.converged,
		iterations = res.iterations,
		feller_margin = 2.0 * p.kappa * p.theta - p.sigma * p.sigma,
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
	u := u_complex
	s2 := complex(params.sigma * params.sigma, 0.0)
	one := complex(1.0, 0.0)

	b := complex(params.kappa, 0.0) - complex(params.rho * params.sigma, 0.0) * i * u
	q := u * u + i * u // u² + iu (u² − iu at u − i)
	d := cmath_sqrt(b * b + s2 * q)
	if real(d) < 0.0 {d = -d}

	// "Little Heston trap" form (Albrecher et al. 2007) with Re(d) ≥ 0, written
	// without the cancellation of b − d for small σ: b − d = −σ² q / (b + d), so
	//   D = −q/(b + d) · (1 − e^{−dT}) / (1 − g e^{−dT}),  g = (b − d)/(b + d),
	//   C = iurT + κθ [−qT/(b + d) − 2 ln(1 + y)/σ²],  y = g(1 − e^{−dT})/(1 − g),
	// and y = σ²·Y, so ln(1 + y)/σ² = Y·ln(1 + y)/y stays exact as σ → 0 (the
	// Black-Scholes limit)
	bd := b + d
	g := -s2 * q / (bd * bd)
	e := cmath_exp(-d * complex(T, 0.0))
	D := -q / bd * (one - e) / (one - g * e)
	Y := -q / (bd * bd) * (one - e) / (one - g)
	y := s2 * Y
	log_term := Y if y == complex(0.0, 0.0) else Y * cmath_log1p(y) / y
	C :=
		complex(r * T, 0.0) * i * u +
		complex(params.kappa * params.theta, 0.0) *
			(-q * complex(T, 0.0) / bd - complex(2.0, 0.0) * log_term)

	return cmath_exp(C + D * complex(params.v0, 0.0) + i * u * cmath_log(complex(S, 0.0)))
}

heston_price :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	opt: OptionType,
	n_points: int = 64,
) -> f64 {
	strikes := [1]f64{K}
	price: [1]f64
	heston_prices_same_expiry(S, strikes[:], T, r, params, opt, price[:], n_points)
	return price[0]
}

// heston_prices_same_expiry prices options that share S, T, r and the Heston
// parameters, writing one price per strike into `out`, by the COS method
// (fourier.odin): the characteristic function is evaluated once per cosine
// term for all strikes. n_points is the minimum number of terms; the count
// grows until the characteristic function has decayed, so any value gives
// full accuracy (~1e-12 relative to the price scale).
heston_prices_same_expiry :: proc(
	S: f64,
	strikes: []f64,
	T: f64,
	r: f64,
	params: Heston_Params,
	opt: OptionType,
	out: []f64,
	n_points: int = 64,
) {
	assert(
		len(out) == len(strikes),
		"heston_prices_same_expiry: out must have one slot per strike",
	)
	heston_prices_cos(S, strikes, T, r, params, opt, out, n_points)
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
	fmt.printf("Expected Call Price:   $10.3687 (QuantLib AnalyticHestonEngine)\n")

	if math.abs(price - 10.368686) < 1e-4 {
		fmt.printf("Status: PASS ✅ \n\n")
	} else {
		fmt.printf("Status: FAIL ❌\n\n")
	}
}
