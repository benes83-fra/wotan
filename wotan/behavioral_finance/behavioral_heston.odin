package behavioral_finance

import fin "../finance"
import opt "../optimize"
import "core:math"
import "core:mem"

// ============================================================================
// Behavioral Heston Model (Physics + Psychology)
// ============================================================================

// Fast extraction of Heston CDF on a grid using Gil-Pelaez inversion
// Uses pure real arithmetic in the inner loop for massive speedups.
extract_heston_cdf_grid :: proc(
	S: f64,
	r: f64,
	T: f64,
	params: fin.Heston_Params,
	ST_min, ST_max: f64,
	n_grid: int,
	n_int: int,
	allocator: mem.Allocator,
) -> []f64 {
	cdf := make([]f64, n_grid, allocator)
	u_max := 100.0
	du := u_max / f64(n_int)

	for i in 0 ..< n_grid {
		K := ST_min + (ST_max - ST_min) * f64(i) / f64(n_grid - 1)
		if K <= 0 {
			cdf[i] = 0.0
			continue
		}

		P2 := 0.0
		lnK := math.ln(K)
		for j in 0 ..< n_int {
			u := (f64(j) + 0.5) * du
			u2 := complex(u, 0.0)
			phi2 := fin.heston_char_func(u2, S, r, T, params)

			cos_u_lnK := math.cos(u * lnK)
			sin_u_lnK := math.sin(u * lnK)
			phi_R := real(phi2)
			phi_I := imag(phi2)

			// Re[ e^{-iu ln K} phi(u) / (iu) ] = (phi_I * cos - phi_R * sin) / u
			integrand := (phi_I * cos_u_lnK - phi_R * sin_u_lnK) / u
			P2 += integrand * du
		}
		P_gt_K := 0.5 + P2 / math.PI
		cdf[i] = 1.0 - P_gt_K // F(K) = P(S_T <= K)
	}
	return cdf
}

behavioral_heston_price :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	heston_params: fin.Heston_Params,
	prospect_params: Prospect_Params,
	opt_type: fin.OptionType,
	n_grid: int = 100,
	n_int: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	if T <= 0.0 {
		if opt_type == .Call {return math.max(S - K, 0.0)}
		return math.max(K - S, 0.0)
	}
	pre := behavioral_heston_precompute(S, K, T, r, heston_params, opt_type, n_grid, n_int, allocator)
	defer delete(pre.heston_cdf, allocator)
	return behavioral_heston_price_precomputed(&pre, K, T, r, prospect_params, opt_type, allocator)
}

// The part of a behavioral Heston price that depends only on the Heston
// parameters (not on the prospect parameters): the Heston CDF of S_T on the
// price grid and the Heston reference premium. Calibrating the prospect
// parameters with Heston fixed computes this once per option.
BehavioralHeston_Precomputed :: struct {
	heston_cdf:  []f64,
	premium_ref: f64,
	ST_min:      f64,
	dS:          f64,
	n_grid:      int,
}

behavioral_heston_precompute :: proc(
	S: f64,
	K: f64,
	T: f64,
	r: f64,
	heston_params: fin.Heston_Params,
	opt_type: fin.OptionType,
	n_grid: int = 100,
	n_int: int = 200,
	allocator: mem.Allocator = context.allocator,
) -> BehavioralHeston_Precomputed {
	ST_min := S * 0.2
	ST_max := S * 3.0
	return BehavioralHeston_Precomputed {
		// 1. Extract Heston CDF (The Physics)
		heston_cdf = extract_heston_cdf_grid(S, r, T, heston_params, ST_min, ST_max, n_grid, n_int, allocator),
		// 4. Reference Premium (Anchor for Value Function)
		premium_ref = fin.heston_price(S, K, T, r, heston_params, opt_type, 200),
		ST_min = ST_min,
		dS = (ST_max - ST_min) / f64(n_grid),
		n_grid = n_grid,
	}
}

// behavioral_heston_price_precomputed applies the prospect-theory valuation to a
// precomputed Heston CDF (steps 2, 3 and 5 of behavioral_heston_price).
behavioral_heston_price_precomputed :: proc(
	pre: ^BehavioralHeston_Precomputed,
	K: f64,
	T: f64,
	r: f64,
	prospect_params: Prospect_Params,
	opt_type: fin.OptionType,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	n_grid := pre.n_grid
	heston_cdf := pre.heston_cdf
	ST_min := pre.ST_min
	dS := pre.dS

	// 2. Apply Probability Weighting (The Psychology)
	distorted_cdf := make([]f64, n_grid, allocator)
	defer delete(distorted_cdf, allocator)

	eps := 1e-7
	gamma := math.max(0.1, math.min(0.99, prospect_params.gamma))
	for i in 0 ..< n_grid {
		p := math.max(eps, math.min(1.0 - eps, heston_cdf[i]))
		p_gamma := math.pow(p, gamma)
		q_gamma := math.pow(1.0 - p, gamma)
		denom := math.pow(p_gamma + q_gamma, 1.0 / gamma)
		distorted_cdf[i] = p_gamma / denom
	}

	// 3. Extract Distorted PDF
	dist_pdf := make([]f64, n_grid, allocator)
	defer delete(dist_pdf, allocator)

	dist_pdf[0] = distorted_cdf[0] / dS
	for i in 1 ..< n_grid {
		diff := distorted_cdf[i] - distorted_cdf[i - 1]
		if diff < 0 {diff = 0}
		dist_pdf[i] = diff / dS
	}

	// Normalize
	sum_pdf := 0.0
	for i in 0 ..< n_grid {sum_pdf += dist_pdf[i] * dS}
	if sum_pdf > 0 {
		for i in 0 ..< n_grid {dist_pdf[i] /= sum_pdf}
	}

	premium_ref := pre.premium_ref

	// 5. Integrate CPT Value
	expected_value := 0.0
	for i in 0 ..< n_grid {
		ST := ST_min + f64(i) * dS
		raw_payoff := 0.0
		if opt_type == .Call && ST > K {
			raw_payoff = ST - K
		} else if opt_type == .Put && ST < K {
			raw_payoff = K - ST
		}

		net: f64 = raw_payoff - premium_ref
		v_net := 0.0
		if net >= 0 {
			v_net = math.pow(net, prospect_params.alpha)
		} else {
			v_net = -prospect_params.lambda * math.pow(-net, prospect_params.beta)
		}
		expected_value += v_net * dist_pdf[i] * dS
	}

	// Certainty Equivalent
	certainty_equivalent := 0.0
	if expected_value >= 0 {
		certainty_equivalent = math.pow(expected_value, 1.0 / prospect_params.alpha)
	} else {
		certainty_equivalent = -math.pow(
			-expected_value / prospect_params.lambda,
			1.0 / prospect_params.beta,
		)
	}

	return certainty_equivalent * math.exp(-r * T)
}

// ============================================================================
// Behavioral Heston Calibration
// ============================================================================

BehavioralHeston_CalibrationContext :: struct {
	market_data:   []fin.VolSurfacePoint,
	S:             f64,
	r:             f64,
	heston_params: fin.Heston_Params,
	n_grid:        int,
	n_int:         int,
	allocator:     mem.Allocator,
	// Optional per-option cache (same order as market_data); see calibrate_behavioral_heston
	precomputed:   []BehavioralHeston_Precomputed,
}

behavioral_heston_objective_fn :: proc(x: []f64, user_data: rawptr) -> f64 {
	ctx := (^BehavioralHeston_CalibrationContext)(user_data)
	prospect_params := Prospect_Params {
		alpha  = x[0],
		beta   = x[1],
		lambda = x[2],
		gamma  = x[3],
	}

	rmse := 0.0
	for point, idx in ctx.market_data {
		model_price: f64
		if ctx.precomputed != nil && point.expiry > 0.0 {
			model_price = behavioral_heston_price_precomputed(
				&ctx.precomputed[idx],
				point.strike,
				point.expiry,
				ctx.r,
				prospect_params,
				.Call,
				ctx.allocator,
			)
		} else {
			model_price = behavioral_heston_price(
				ctx.S,
				point.strike,
				point.expiry,
				ctx.r,
				ctx.heston_params,
				prospect_params,
				.Call,
				ctx.n_grid,
				ctx.n_int,
				ctx.allocator,
			)
		}

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

calibrate_behavioral_heston :: proc(
	market_data: []fin.VolSurfacePoint,
	S: f64,
	r: f64,
	heston_params: fin.Heston_Params,
	allocator: mem.Allocator = context.allocator,
) -> Prospect_CalibrationResult {
	x := make([]f64, 4, allocator)
	defer delete(x, allocator)

	x[0] = 0.88 // alpha
	x[1] = 0.88 // beta
	x[2] = 2.25 // lambda
	x[3] = 0.61 // gamma

	ctx := BehavioralHeston_CalibrationContext {
		market_data   = market_data,
		S             = S,
		r             = r,
		heston_params = heston_params,
		n_grid        = 100,
		n_int         = 200,
		allocator     = allocator,
	}

	// The Heston CDF grid and reference premium do not depend on the prospect
	// parameters being calibrated: compute them once per option.
	ctx.precomputed = make([]BehavioralHeston_Precomputed, len(market_data), allocator)
	for point, idx in market_data {
		if point.expiry <= 0.0 {continue}
		ctx.precomputed[idx] = behavioral_heston_precompute(
			S,
			point.strike,
			point.expiry,
			r,
			heston_params,
			.Call,
			ctx.n_grid,
			ctx.n_int,
			allocator,
		)
	}
	defer {
		for pre in ctx.precomputed {delete(pre.heston_cdf, allocator)}
		delete(ctx.precomputed, allocator)
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
	best_loss := behavioral_heston_objective_fn(x, rawptr(&ctx))
	best_x := make([]f64, 4, allocator)
	defer delete(best_x, allocator)
	copy(best_x, x)

	max_iter := 100 // Kept reasonable for speed
	converged := false

	for iter in 0 ..< max_iter {
		current_loss := behavioral_heston_objective_fn(x, rawptr(&ctx))
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

			loss_plus := behavioral_heston_objective_fn(x_plus, rawptr(&ctx))
			loss_minus := behavioral_heston_objective_fn(x_minus, rawptr(&ctx))

			gradient[i] = (loss_plus - loss_minus) / (2.0 * eps)

			delete(x_plus, allocator)
			delete(x_minus, allocator)
		}

		opt.optimizer_step(&optimizer, x, gradient)

		x[0] = math.max(0.1, math.min(1.0, x[0]))
		x[1] = math.max(0.1, math.min(1.0, x[1]))
		x[2] = math.max(1.0, math.min(5.0, x[2]))
		x[3] = math.max(0.1, math.min(0.99, x[3]))
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
