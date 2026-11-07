package behavioral_finance

import fin "../finance"
import "core:math"
import "core:mem"

// ============================================================================
// Behavioral Heston: CPT option prices on a Heston distribution of S_T
// ============================================================================
// The CPT valuation of prospect.odin with S_T distributed as in the Heston model
// (drift r: Heston parameters calibrated to option prices describe the risk-neutral
// distribution). With RATIONAL_PARAMS the price is the Heston price, so the CPT
// parameters measure what remains of the market's prices after stochastic volatility.

// Heston CDF F(x) = P(S_T <= x) at each node, by Gil-Pelaez inversion of the
// characteristic function (drift r): P(S_T > x) = 1/2 + 1/pi int_0^inf
// Re[e^{-iu ln x} phi(u) / (iu)] du, midpoint rule with n_int points on (0, u_max].
// phi does not depend on x: evaluated once per u.
extract_heston_cdf_grid :: proc(
	S, r, T: f64,
	params: fin.Heston_Params,
	nodes: []f64,
	u_max: f64,
	n_int: int,
	allocator := context.allocator,
) -> []f64 {
	du := u_max / f64(n_int)
	phi_r := make([]f64, n_int, context.temp_allocator)
	phi_i := make([]f64, n_int, context.temp_allocator)
	defer delete(phi_r, context.temp_allocator)
	defer delete(phi_i, context.temp_allocator)
	for j in 0 ..< n_int {
		u := (f64(j) + 0.5) * du
		p := fin.heston_char_func(complex(u, 0), S, r, T, params)
		phi_r[j], phi_i[j] = real(p) / u, imag(p) / u
	}
	cdf := make([]f64, len(nodes), allocator)
	for x, k in nodes {
		if x <= 0 {continue}
		lx := math.ln(x)
		acc := 0.0
		// Re[e^{-ia} (pr + i pi) / i] = pi cos a - pr sin a at a = (j + 1/2) du ln x;
		// cos a, sin a by rotating with step du ln x
		c, sn := math.cos(0.5 * du * lx), math.sin(0.5 * du * lx)
		dc, ds := math.cos(du * lx), math.sin(du * lx)
		for j in 0 ..< n_int {
			acc += phi_i[j] * c - phi_r[j] * sn
			c, sn = c * dc - sn * ds, sn * dc + c * ds
		}
		cdf[k] = 1 - (0.5 + acc * du / math.PI)
	}
	return cdf
}

// The Heston distribution of S_T on a log grid of n_grid cells (the price error
// falls as 1/n_grid^2: ~5e-6 relative at 4000), the strike a node,
// over +-12 standard deviations of ln S_T (variance from max(v0, theta)); the CDF made
// monotone (clipped into [0, 1], running maximum) and pinned to 0 and 1 at the ends
behavioral_heston_precompute :: proc(
	S, K, T, r: f64,
	heston_params: fin.Heston_Params,
	n_grid := 4000,
	n_int := 2000,
	allocator := context.allocator,
) -> Prospect_Precomputed {
	assert(S > 0 && K > 0 && T > 0 && n_grid >= 2, "behavioral_heston_precompute: bad inputs")
	v := max(heston_params.v0, heston_params.theta)
	s := math.sqrt(v * T)
	m := math.ln(S) + (r - 0.5 * v) * T
	HALF :: 12.0
	h := 2 * HALF * s / f64(n_grid)
	lK := math.ln(K)
	j0 := int(math.ceil((m - HALF * s - lK) / h))
	j1 := int(math.floor((m + HALF * s - lK) / h))
	n := j1 - j0 + 1
	nodes := make([]f64, n, allocator)
	for j in 0 ..< n {nodes[j] = math.exp(lK + f64(j0 + j) * h)}
	// phi decays at least like a Gaussian of variance ~ s^2 near the centre; u_max
	// 40/s, step du = u_max / n_int (aliasing period 2 pi / du >> the grid's width)
	cdf := extract_heston_cdf_grid(S, r, T, heston_params, nodes, 40 / s, n_int, allocator)
	run := 0.0
	for &F in cdf {
		F = clamp(F, 0, 1)
		run = max(run, F)
		F = run
	}
	cdf[0], cdf[n - 1] = 0, 1
	return {nodes = nodes, cdf = cdf, allocator = allocator}
}

behavioral_heston_price :: proc(
	S, K, T, r: f64,
	heston_params: fin.Heston_Params,
	prospect_params: Prospect_Params,
	opt_type: fin.OptionType,
	side: CPT_Side = .Writer,
	frame: CPT_Frame = .Aggregated,
	n_grid := 4000,
	n_int := 2000,
) -> f64 {
	if T <= 0 {
		return max(S - K, 0) if opt_type == .Call else max(K - S, 0)
	}
	pre := behavioral_heston_precompute(
		S,
		K,
		T,
		r,
		heston_params,
		n_grid,
		n_int,
		context.temp_allocator,
	)
	defer prospect_precomputed_free(&pre)
	return prospect_price_precomputed(&pre, K, T, r, prospect_params, opt_type, side, frame)
}

// Calibrates the CPT parameters on top of fixed (already calibrated) Heston
// parameters
calibrate_behavioral_heston :: proc(
	market_data: []fin.VolSurfacePoint,
	S, r: f64,
	heston_params: fin.Heston_Params,
	setup: Prospect_Calibration_Setup = {},
	opt_types: []fin.OptionType = nil,
	allocator := context.allocator,
) -> Prospect_CalibrationResult {
	n_grid := setup.n_points if setup.n_points > 0 else 4000
	pre := make([]Prospect_Precomputed, len(market_data), allocator)
	defer {
		for &p in pre {prospect_precomputed_free(&p)}
		delete(pre, allocator)
	}
	for point, i in market_data {
		pre[i] = behavioral_heston_precompute(
			S,
			point.strike,
			point.expiry,
			r,
			heston_params,
			n_grid,
			2000,
			allocator,
		)
	}
	return _calibrate_prospect(market_data, opt_types, S, r, setup, pre, allocator)
}
