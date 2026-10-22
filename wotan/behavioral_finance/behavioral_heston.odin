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
	defer delete(pre.cdf, allocator)
	return behavioral_heston_price_precomputed(&pre, K, T, r, prospect_params, opt_type, allocator)
}

// The part of a behavioral Heston price that depends only on the Heston
// parameters (not on the prospect parameters): the Heston CDF of S_T on the
// price grid and the Heston reference premium. Calibrating the prospect
// parameters with Heston fixed computes this once per option. The valuation
// itself is the shared prospect-theory step (prospect_theory_price_precomputed).
BehavioralHeston_Precomputed :: Prospect_Precomputed

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
		cdf = extract_heston_cdf_grid(S, r, T, heston_params, ST_min, ST_max, n_grid, n_int, allocator),
		// 4. Reference Premium (Anchor for Value Function)
		premium_ref = fin.heston_price(S, K, T, r, heston_params, opt_type, 200),
		ST_min = ST_min,
		dS = (ST_max - ST_min) / f64(n_grid),
		n_points = n_grid,
	}
}

// behavioral_heston_price_precomputed applies the prospect-theory valuation
// (probability weighting, value function, certainty equivalent) to a
// precomputed Heston CDF.
behavioral_heston_price_precomputed :: prospect_theory_price_precomputed

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
			// Scratch from the (thread-local) temp allocator: safe for parallel probes
			model_price = behavioral_heston_price_precomputed(
				&ctx.precomputed[idx],
				point.strike,
				point.expiry,
				ctx.r,
				prospect_params,
				.Call,
				context.temp_allocator,
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
		for pre in ctx.precomputed {delete(pre.cdf, allocator)}
		delete(ctx.precomputed, allocator)
	}

	return _calibrate_prospect_params(&ctx, behavioral_heston_objective_fn, 100, true, allocator)
}
