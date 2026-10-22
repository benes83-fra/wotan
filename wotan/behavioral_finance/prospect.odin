package behavioral_finance

import fin "../finance"
import l "../linalg"
import opt "../optimize"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Behavioral Finance: Prospect Theory Calibration
// ============================================================================

Prospect_Params :: struct {
	alpha:  f64, // Value function: gain curvature (TK92: 0.88)
	beta:   f64, // Value function: loss curvature (TK92: 0.88)
	lambda: f64, // Value function: loss aversion (TK92: 2.25)
	gamma:  f64, // Probability weighting: curvature (TK92: 0.61)
}

Prospect_CalibrationResult :: struct {
	params:     Prospect_Params,
	rmse:       f64,
	converged:  bool,
	iterations: int,
}

Prospect_CalibrationContext :: struct {
	market_data: []fin.VolSurfacePoint,
	S:           f64,
	r:           f64,
	true_sigma:  f64, // The physical/historical volatility
	n_points:    int,
	allocator:   mem.Allocator,
	// Optional per-option cache (same order as market_data); see calibrate_prospect_theory
	precomputed: []Prospect_Precomputed,
}

// Prices an option using Prospect Theory distorted probabilities.
// This is the Behavioral Asset Pricing Model (BAPM) equivalent of Black-Scholes.
prospect_theory_price :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	true_sigma: f64,
	params: Prospect_Params,
	opt_type: fin.OptionType,
	n_points: int = 500,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	if T <= 0.0 {
		if opt_type == .Call {return math.max(S - K, 0.0)}
		return math.max(K - S, 0.0)
	}
	pre := prospect_precompute(S, K, T, r, true_sigma, opt_type, n_points, allocator)
	defer delete(pre.cdf, allocator)
	return prospect_theory_price_precomputed(&pre, K, T, r, params, opt_type, allocator)
}

// The part of a prospect-theory price that does not depend on the prospect
// parameters: the CDF of S_T on the terminal-price grid and the reference
// premium. Calibration computes it once per option. Also used by the
// behavioral Heston model, which supplies a Heston CDF instead.
Prospect_Precomputed :: struct {
	cdf:         []f64, // CDF of S_T at ST_min + i*dS
	premium_ref: f64, // price the investor pays (reference point)
	ST_min:      f64,
	dS:          f64,
	n_points:    int,
}

// prospect_precompute: true log-normal CDF and Black-Scholes reference premium.
prospect_precompute :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	true_sigma: f64,
	opt_type: fin.OptionType,
	n_points: int = 500,
	allocator: mem.Allocator = context.allocator,
) -> Prospect_Precomputed {
	// 1. Grid of terminal prices
	ST_min := S * 0.1
	ST_max := S * 3.0
	dS := (ST_max - ST_min) / f64(n_points)

	// 2. True Log-Normal CDF
	mu := (r - 0.5 * true_sigma * true_sigma) * T
	variance := true_sigma * true_sigma * T
	std_dev := math.sqrt_f64(variance)

	true_cdf := make([]f64, n_points, allocator)
	for i in 0 ..< n_points {
		ST := ST_min + f64(i) * dS
		if ST <= 0 {
			true_cdf[i] = 0.0
		} else {
			z := (math.ln(ST / S) - mu) / std_dev
			true_cdf[i] = 0.5 * (1.0 + math.erf(z / math.sqrt_f64(2.0)))
		}
	}

	// Reference premium (the price the investor pays): the standard
	// Black-Scholes price with the true physical volatility as the anchor.
	premium_ref := 0.0
	if opt_type == .Call {
		premium_ref = fin.black_scholes_call(S, K, T, r, true_sigma)
	} else {
		// Put-Call Parity: P = C - S + K*exp(-rT)
		bs_call := fin.black_scholes_call(S, K, T, r, true_sigma)
		premium_ref = bs_call - S + K * math.exp(-r * T)
	}

	return Prospect_Precomputed{cdf = true_cdf, premium_ref = premium_ref, ST_min = ST_min, dS = dS, n_points = n_points}
}

// prospect_theory_price_precomputed values an option under cumulative prospect
// theory from a precomputed CDF and reference premium: probability weighting,
// distorted PDF, value function, certainty equivalent.
prospect_theory_price_precomputed :: proc(
	pre: ^Prospect_Precomputed,
	K: f64,
	T: f64,
	r: f64,
	params: Prospect_Params,
	opt_type: fin.OptionType,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	n_points := pre.n_points
	true_cdf := pre.cdf
	ST_min := pre.ST_min
	dS := pre.dS

	// 3. Apply Probability Weighting w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
	distorted_cdf := make([]f64, n_points, allocator)
	defer delete(distorted_cdf, allocator)

	eps := 1e-7
	gamma := math.max(0.1, math.min(0.99, params.gamma)) // Clamp for numerical stability
	for i in 0 ..< n_points {
		p := math.max(eps, math.min(1.0 - eps, true_cdf[i]))
		p_gamma := math.pow(p, gamma)
		q_gamma := math.pow(1.0 - p, gamma)
		denom := math.pow(p_gamma + q_gamma, 1.0 / gamma)
		distorted_cdf[i] = p_gamma / denom
	}

	// 4. Extract Distorted PDF via finite differences
	dist_pdf := make([]f64, n_points, allocator)
	defer delete(dist_pdf, allocator)

	dist_pdf[0] = distorted_cdf[0] / dS
	for i in 1 ..< n_points {
		diff := distorted_cdf[i] - distorted_cdf[i - 1]
		if diff < 0 {diff = 0} 	// Clamp numerical noise
		dist_pdf[i] = diff / dS
	}

	// Normalize to ensure it integrates to 1
	sum_pdf := 0.0
	for i in 0 ..< n_points {sum_pdf += dist_pdf[i] * dS}
	if sum_pdf > 0 {
		for i in 0 ..< n_points {dist_pdf[i] /= sum_pdf}
	}

	premium_ref := pre.premium_ref

	// 5. Integrate CPT Value * Distorted PDF
	expected_value := 0.0
	for i in 0 ..< n_points {
		ST := ST_min + f64(i) * dS

		// Raw payoff
		raw_payoff := 0.0
		if opt_type == .Call && ST > K {
			raw_payoff = ST - K
		} else if opt_type == .Put && ST < K {
			raw_payoff = K - ST
		}

		// Net of premium (reference point = premium paid)
		// ✅ FIX: Explicitly type as f64 to satisfy Odin's type inference
		net: f64 = raw_payoff - premium_ref

		// Apply Prospect Value function
		v_net := 0.0
		if net >= 0 {
			v_net = math.pow(net, params.alpha)
		} else {
			v_net = -params.lambda * math.pow(-net, params.beta)
		}

		expected_value += v_net * dist_pdf[i] * dS
	}

	// To convert the expected CPT utility back to a dollar price, we find the certainty equivalent.
	// Since the utility is dominated by the gain domain (net > 0), we invert the gain function: v^-1(x) = x^(1/alpha)
	// If the expected value is negative (deep OTM option where premium is always lost), we invert the loss function.
	certainty_equivalent := 0.0
	if expected_value >= 0 {
		certainty_equivalent = math.pow(expected_value, 1.0 / params.alpha)
	} else {
		certainty_equivalent = -math.pow(-expected_value / params.lambda, 1.0 / params.beta)
	}

	return certainty_equivalent * math.exp(-r * T)
}

prospect_objective_fn :: proc(x: []f64, user_data: rawptr) -> f64 {
	ctx := (^Prospect_CalibrationContext)(user_data)
	params := Prospect_Params {
		alpha  = x[0],
		beta   = x[1],
		lambda = x[2],
		gamma  = x[3],
	}

	rmse := 0.0
	for point, idx in ctx.market_data {
		model_price: f64
		if ctx.precomputed != nil && point.expiry > 0.0 {
			// Scratch from the (thread-local) temp allocator: safe for parallel probes
			model_price = prospect_theory_price_precomputed(
				&ctx.precomputed[idx],
				point.strike,
				point.expiry,
				ctx.r,
				params,
				.Call,
				context.temp_allocator,
			)
		} else {
			model_price = prospect_theory_price(
				ctx.S,
				point.strike,
				point.expiry,
				ctx.r,
				ctx.true_sigma,
				params,
				.Call,
				ctx.n_points,
				ctx.allocator,
			)
		}

		// Compare against market price (or convert market IV to price)
		market_price := point.market_price
		if market_price <= 0 {
			market_price = fin.black_scholes_call(
				ctx.S,
				point.strike,
				point.expiry,
				ctx.r,
				point.implied_vol,
			)
		}

		err := model_price - market_price
		rmse += err * err
	}
	return math.sqrt_f64(rmse / f64(len(ctx.market_data)))
}

// Calibrates Prospect Theory parameters to an observed option chain.
// Extracts the "Market-Implied Psychology" (fear, loss aversion, etc.)
calibrate_prospect_theory :: proc(
	market_data: []fin.VolSurfacePoint,
	S: f64,
	r: f64,
	true_sigma: f64, // Historical/Realized Volatility
	allocator: mem.Allocator = context.allocator,
) -> Prospect_CalibrationResult {
	ctx := Prospect_CalibrationContext {
		market_data = market_data,
		S           = S,
		r           = r,
		true_sigma  = true_sigma,
		n_points    = 500,
		allocator   = allocator,
	}

	// The true CDF grid and reference premium do not depend on the prospect
	// parameters being calibrated: compute them once per option.
	ctx.precomputed = make([]Prospect_Precomputed, len(market_data), allocator)
	for point, idx in market_data {
		if point.expiry <= 0.0 {continue}
		ctx.precomputed[idx] = prospect_precompute(S, point.strike, point.expiry, r, true_sigma, .Call, ctx.n_points, allocator)
	}
	defer {
		for pre in ctx.precomputed {delete(pre.cdf, allocator)}
		delete(ctx.precomputed, allocator)
	}

	return _calibrate_prospect_params(&ctx, prospect_objective_fn, 200, true, allocator)
}

// _calibrate_prospect_params fits (alpha, beta, lambda, gamma) from the
// Tversky & Kahneman (1992) starting point; shared by the prospect-theory and
// behavioral Heston calibrations.
_calibrate_prospect_params :: proc(
	ctx: rawptr,
	objective: opt.ObjectiveFunc,
	max_iter: int,
	parallel: bool,
	allocator: mem.Allocator,
) -> Prospect_CalibrationResult {
	params := []opt.Calibration_Param {
		{"alpha", 0.88, 0.1, 1.0, .Linear}, // gain curvature
		{"beta", 0.88, 0.1, 1.0, .Linear}, // loss curvature
		{"lambda", 2.25, 1.0, 5.0, .Linear}, // loss aversion, usually 1 to 5
		{"gamma", 0.61, 0.1, 0.99, .Linear}, // probability weighting
	}
	config := opt.Calibration_Config {
		optimizer = {type = .Adam, learning_rate = 0.05, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8},
		max_iter = max_iter,
		loss_threshold = 1e-4,
		fd_step = 1e-4,
		parallel = parallel,
	}
	res := opt.calibrate(params, objective, ctx, config, allocator)
	defer opt.calibration_result_free(&res)

	return Prospect_CalibrationResult {
		params = Prospect_Params{alpha = res.x[0], beta = res.x[1], lambda = res.x[2], gamma = res.x[3]},
		rmse = res.loss,
		converged = res.converged,
		iterations = res.iterations,
	}
}
BehavioralSignal :: struct {
	regime:     string, // "Complacency", "Panic", "Euphoria", "Rational"
	action:     string, // "Buy Tail Protection", "Short Volatility", "Hold"
	conviction: f64, // 0.0 to 1.0
	bvrp:       f64, // Behavioral Volatility Risk Premium (IV - HV)
}
generate_behavioral_signal :: proc(
	market_params: Prospect_Params,
	hist_vol: f64,
	avg_implied_vol: f64,
) -> BehavioralSignal {
	sig: BehavioralSignal
	sig.bvrp = avg_implied_vol - hist_vol

	// 1. Detect Extreme Put Skew (The "Reflection Effect" Trade)
	// If lambda is high (> 2.5) and beta is low (< 0.5), the Put Skew is irrationally steep.
	// Puts are vastly more expensive than equidistant Calls.
	if market_params.lambda > 2.5 && market_params.beta < 0.5 {
		sig.regime = "Extreme Put Skew (Reflection Effect)"
		sig.action = "RISK REVERSAL (Sell OTM Puts / Buy OTM Calls)"
		sig.conviction = math.min(1.0, (market_params.lambda - 2.0) / 2.0)
		return sig
	}

	// 2. Detect Tail Panic (Short Volatility)
	if market_params.gamma < 0.55 {
		sig.regime = "Panic (Crashophobia)"
		if sig.bvrp > 0.05 {
			sig.action = "SHORT VOLATILITY (Sell OTM Puts / Short Strangle)"
			sig.conviction = math.min(1.0, sig.bvrp / 0.15)
		} else {
			sig.action = "Hold (Fear is high, but IV hasn't decoupled from HV yet)"
			sig.conviction = 0.2
		}
		return sig
	}

	// 3. Detect Complacency (Long Volatility)
	if market_params.gamma > 0.75 {
		sig.regime = "Complacency / Euphoria"
		if sig.bvrp < 0.02 {
			sig.action = "BUY TAIL PROTECTION (Buy cheap OTM Puts)"
			sig.conviction = (market_params.gamma - 0.75) / 0.25
		} else {
			sig.action = "Hold (Market is rational and fairly priced)"
			sig.conviction = 0.1
		}
		return sig
	}

	// 4. Standard Regime
	sig.regime = "Standard Behavioral Bias"
	sig.action = "Delta Neutral / Market Making"
	sig.conviction = 0.3
	return sig
}
