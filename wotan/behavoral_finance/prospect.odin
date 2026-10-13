package behavorial_finance

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

	// 1. Grid of terminal prices
	ST_min := S * 0.1
	ST_max := S * 3.0
	dS := (ST_max - ST_min) / f64(n_points)

	// 2. True Log-Normal CDF
	mu := (r - 0.5 * true_sigma * true_sigma) * T
	variance := true_sigma * true_sigma * T
	std_dev := math.sqrt_f64(variance)

	true_cdf := make([]f64, n_points, allocator)
	defer delete(true_cdf, allocator)

	for i in 0 ..< n_points {
		ST := ST_min + f64(i) * dS
		if ST <= 0 {
			true_cdf[i] = 0.0
		} else {
			z := (math.ln(ST / S) - mu) / std_dev
			true_cdf[i] = 0.5 * (1.0 + math.erf(z / math.sqrt_f64(2.0)))
		}
	}

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

	// 5. Integrate Payoff * Distorted PDF
	expected_payoff := 0.0
	for i in 0 ..< n_points {
		ST := ST_min + f64(i) * dS
		payoff := 0.0
		if opt_type == .Call && ST > K {
			payoff = ST - K
		} else if opt_type == .Put && ST < K {
			payoff = K - ST
		}
		expected_payoff += payoff * dist_pdf[i] * dS
	}

	return expected_payoff * math.exp(-r * T)
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
	for point in ctx.market_data {
		model_price := prospect_theory_price(
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
	x := make([]f64, 4, allocator)
	defer delete(x, allocator)

	// Initial guess: Tversky & Kahneman (1992)
	x[0] = 0.88 // alpha
	x[1] = 0.88 // beta
	x[2] = 2.25 // lambda
	x[3] = 0.61 // gamma

	ctx := Prospect_CalibrationContext {
		market_data = market_data,
		S           = S,
		r           = r,
		true_sigma  = true_sigma,
		n_points    = 500,
		allocator   = allocator,
	}

	opt_config := opt.OptimizerConfig {
		type          = .Adam,
		learning_rate = 0.05,
		beta1         = 0.9,
		beta2         = 0.999,
		epsilon       = 1e-8,
	}
	optimizer := opt.optimizer_init(opt_config, 4, allocator)
	defer opt.optimizer_free(&optimizer)

	gradient := make([]f64, 4, allocator)
	defer delete(gradient, allocator)

	eps := 1e-4
	best_loss := prospect_objective_fn(x, rawptr(&ctx))
	best_x := make([]f64, 4, allocator)
	defer delete(best_x, allocator)
	copy(best_x, x)

	max_iter := 200
	converged := false

	for iter in 0 ..< max_iter {
		current_loss := prospect_objective_fn(x, rawptr(&ctx))
		if current_loss < best_loss {
			best_loss = current_loss
			copy(best_x, x)
		}
		if current_loss < 1e-4 {
			converged = true
			break
		}

		for i in 0 ..< 4 {
			x_plus := make([]f64, 4, allocator)
			x_minus := make([]f64, 4, allocator)
			copy(x_plus, x)
			copy(x_minus, x)

			x_plus[i] += eps
			x_minus[i] -= eps

			loss_plus := prospect_objective_fn(x_plus, rawptr(&ctx))
			loss_minus := prospect_objective_fn(x_minus, rawptr(&ctx))

			gradient[i] = (loss_plus - loss_minus) / (2.0 * eps)

			delete(x_plus, allocator)
			delete(x_minus, allocator)
		}

		opt.optimizer_step(&optimizer, x, gradient)

		// Enforce behavioral bounds
		x[0] = math.max(0.1, math.min(1.0, x[0])) // alpha
		x[1] = math.max(0.1, math.min(1.0, x[1])) // beta
		x[2] = math.max(1.0, math.min(5.0, x[2])) // lambda (loss aversion usually 1 to 5)
		x[3] = math.max(0.1, math.min(0.99, x[3])) // gamma
	}

	return Prospect_CalibrationResult {
		params = Prospect_Params {
			alpha = best_x[0],
			beta = best_x[1],
			lambda = best_x[2],
			gamma = best_x[3],
		},
		rmse = best_loss,
		converged = converged,
		iterations = max_iter,
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

	// 1. Calculate the Behavioral Volatility Risk Premium (BVRP)
	// This is the exact dollar amount investors are over/under-paying for fear.
	sig.bvrp = avg_implied_vol - hist_vol

	// 2. Determine the Psychological Regime
	if market_params.gamma < 0.55 {
		sig.regime = "Panic (Crashophobia)"
		if sig.bvrp > 0.05 { 	// IV is 5%+ higher than HV
			sig.action = "SHORT VOLATILITY (Sell OTM Puts / Short Strangle)"
			sig.conviction = math.min(1.0, sig.bvrp / 0.15) // Max conviction at 15% spread
		} else {
			sig.action = "Hold (Fear is high, but not mispriced)"
			sig.conviction = 0.2
		}
	} else if market_params.gamma > 0.75 {
		sig.regime = "Complacency / Euphoria"
		if sig.bvrp < 0.02 { 	// IV is barely above HV
			sig.action = "BUY TAIL PROTECTION (Buy cheap OTM Puts)"
			// Conviction increases the closer gamma is to 1.0 (pure rationality/blindness)
			sig.conviction = (market_params.gamma - 0.75) / 0.25
		} else {
			sig.action = "Hold (Market is rational and fairly priced)"
			sig.conviction = 0.1
		}
	} else {
		sig.regime = "Standard Behavioral Bias"
		sig.action = "Delta Neutral / Market Making"
		sig.conviction = 0.3
	}

	return sig
}
