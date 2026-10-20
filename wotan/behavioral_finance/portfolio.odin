package behavioral_finance

import fin "../finance"
import l "../linalg"
import "core:math"
import "core:mem"
// ============================================================================
// Prospect Theory Portfolio Optimization
// ============================================================================

ProspectPortfolioParams :: struct {
	alpha:     f64, // Gain curvature (TK92 default: 0.88)
	beta:      f64, // Loss curvature (TK92 default: 0.88)
	lambda:    f64, // Loss aversion (TK92 default: 2.25)
	ref_point: f64, // Reference point (e.g. 0.0 or risk-free rate)
}

ProspectMetrics :: struct {
	weights:          []f64,
	prospect_utility: f64,
	expected_return:  f64,
	volatility:       f64,
	loss_probability: f64, // Fraction of scenarios below ref_point
}

// Evaluate Prospect Metrics for a given portfolio
prospect_metrics :: proc(
	weights: []f64,
	returns_data: ^l.Matrix(f64), // rows = scenarios (time), cols = assets
	params: ProspectPortfolioParams,
) -> ProspectMetrics {
	T := returns_data.rows
	N := returns_data.cols

	utility := 0.0
	sum_ret := 0.0
	sum_sq := 0.0
	losses := 0

	for t in 0 ..< T {
		R_t := 0.0
		for i in 0 ..< N {
			R_t += weights[i] * returns_data.data[t * N + i]
		}

		sum_ret += R_t
		sum_sq += R_t * R_t

		x := R_t - params.ref_point
		if x < 0 {
			losses += 1
			utility += -params.lambda * math.pow(-x, params.beta)
		} else {
			utility += math.pow(x, params.alpha)
		}
	}

	mean_ret := sum_ret / f64(T)
	variance := (sum_sq / f64(T)) - (mean_ret * mean_ret)
	vol := math.sqrt_f64(math.max(0.0, variance))

	return ProspectMetrics {
		weights = weights,
		prospect_utility = utility / f64(T),
		expected_return = mean_ret,
		volatility = vol,
		loss_probability = f64(losses) / f64(T),
	}
}

// Maximize Prospect Utility using Projected Gradient Ascent.
// Naturally tilts away from high-downside assets.
prospect_portfolio_optimize :: proc(
	returns_data: ^l.Matrix(f64),
	params: ProspectPortfolioParams,
	learning_rate: f64 = 0.1, // Lowered for stability
	max_iter: int = 3000,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	T := returns_data.rows
	N := returns_data.cols

	// Initialize with equal weights
	w := make([]f64, N, allocator)
	for i in 0 ..< N {w[i] = 1.0 / f64(N)}

	grad := make([]f64, N, allocator)
	defer delete(grad, allocator)

	port_rets := make([]f64, T, allocator)
	defer delete(port_rets, allocator)

	eps := 1e-4 // ✅ FIX: Epsilon to prevent gradient singularity at x=0

	for iter in 0 ..< max_iter {
		// 1. Compute portfolio returns for all scenarios
		for t in 0 ..< T {
			sum := 0.0
			for i in 0 ..< N {
				sum += w[i] * returns_data.data[t * N + i]
			}
			port_rets[t] = sum
		}

		// 2. Compute gradient
		for i in 0 ..< N {grad[i] = 0.0}

		for t in 0 ..< T {
			x := port_rets[t] - params.ref_point

			// ✅ FIX: Smoothed Derivative of Prospect Value function v(x)
			dv_dx := 0.0
			if x > eps {
				dv_dx = params.alpha * math.pow(x, params.alpha - 1.0)
			} else if x < -eps {
				dv_dx = params.lambda * params.beta * math.pow(-x, params.beta - 1.0)
			} else {
				// In the epsilon neighborhood, cap the gradient to prevent explosion
				dv_dx = params.alpha * math.pow(eps, params.alpha - 1.0)
			}

			// Chain rule: dU/dw_i = (1/T) * sum_t dv/dx * r_{t,i}
			for i in 0 ..< N {
				grad[i] += dv_dx * returns_data.data[t * N + i]
			}
		}

		for i in 0 ..< N {grad[i] /= f64(T)}

		// 3. Gradient Ascent step
		for i in 0 ..< N {w[i] += learning_rate * grad[i]}

		// 4. Project onto simplex (Long-only: w_i >= 0, sum(w) = 1)
		sum_w := 0.0
		for i in 0 ..< N {
			if w[i] < 0.0 {w[i] = 0.0}
			sum_w += w[i]
		}
		if sum_w > 1e-12 {
			for i in 0 ..< N {w[i] /= sum_w}
		} else {
			for i in 0 ..< N {w[i] = 1.0 / f64(N)}
		}
	}

	return w
}
// ============================================================================
// Behavioral Portfolio Theory (Shefrin & Statman, 2000)
// ============================================================================

// Optimizes a 3-layer Mental Accounting Pyramid.
// Layer 1 (Safety): Min Variance (Capital Preservation)
// Layer 2 (Aspiration): Max Sharpe (Efficient Growth)
// Layer 3 (Speculative): Risk-Seeking Prospect (Lottery Tickets)
// Maximize Upside Variance (The "Lottery Ticket" Objective)
// Hunts for assets with explosive positive spikes, ignoring downside risk.
upside_variance_optimize :: proc(
	returns_data: ^l.Matrix(f64),
	learning_rate: f64 = 0.1,
	max_iter: int = 3000,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	T := returns_data.rows
	N := returns_data.cols

	w := make([]f64, N, allocator)
	for i in 0 ..< N {w[i] = 1.0 / f64(N)}

	grad := make([]f64, N, allocator)
	defer delete(grad, allocator)
	port_rets := make([]f64, T, allocator)
	defer delete(port_rets, allocator)

	for iter in 0 ..< max_iter {
		// 1. Compute portfolio returns and mean
		sum := 0.0
		for t in 0 ..< T {
			r := 0.0
			for i in 0 ..< N {r += w[i] * returns_data.data[t * N + i]}
			port_rets[t] = r
			sum += r
		}
		mean_ret := sum / f64(T)

		// 2. Compute gradient of Upside Variance: E[ max(R - mean, 0)^2 ]
		for i in 0 ..< N {grad[i] = 0.0}

		for t in 0 ..< T {
			diff := port_rets[t] - mean_ret
			if diff > 0 {
				// Gradient of upside semi-variance
				for i in 0 ..< N {
					grad[i] += 2.0 * diff * (returns_data.data[t * N + i] - mean_ret)
				}
			}
		}
		for i in 0 ..< N {grad[i] /= f64(T)}

		// 3. Gradient Ascent
		for i in 0 ..< N {w[i] += learning_rate * grad[i]}

		// 4. Project onto simplex
		sum_w := 0.0
		for i in 0 ..< N {
			if w[i] < 0.0 {w[i] = 0.0}
			sum_w += w[i]
		}
		if sum_w > 1e-12 {
			for i in 0 ..< N {w[i] /= sum_w}
		} else {
			for i in 0 ..< N {w[i] = 1.0 / f64(N)}
		}
	}
	return w
}

// ============================================================================
// Behavioral Portfolio Theory (V1 & V2) - Fixed
// ============================================================================

bpt_optimize :: proc(
	returns_data: ^l.Matrix(f64),
	safety_alloc: f64,
	aspiration_alloc: f64,
	speculative_alloc: f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	combined: []f64,
	w_safety: []f64,
	w_asp: []f64,
	w_spec: []f64,
) {
	N := returns_data.cols
	mu := fin.expected_returns_from_history(returns_data, allocator)
	defer delete(mu, allocator)
	cov := fin.covariance_matrix(returns_data, allocator)
	defer l.matrix_free(&cov)

	w_safety = fin.min_variance_portfolio(&cov, allocator)
	w_asp = fin.max_sharpe_portfolio(mu, &cov, 0.0, allocator)

	// ✅ FIX: Use proper Upside Variance for Speculative layer
	w_spec = upside_variance_optimize(returns_data, 0.1, 3000, allocator)

	combined = make([]f64, N, allocator)
	for i in 0 ..< N {
		combined[i] =
			(safety_alloc * w_safety[i]) +
			(aspiration_alloc * w_asp[i]) +
			(speculative_alloc * w_spec[i])
	}
	return combined, w_safety, w_asp, w_spec
}

// Helper: Extract a sub-matrix of returns based on a boolean universe mask
extract_sub_returns :: proc(
	returns_data: ^l.Matrix(f64),
	mask: []bool,
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	T := returns_data.rows
	N := returns_data.cols
	sub_cols := 0
	for b in mask {if b {sub_cols += 1}}

	sub_mat := l.matrix_new(f64, T, sub_cols, allocator)
	col_idx := 0
	for j in 0 ..< N {
		if mask[j] {
			for i in 0 ..< T {
				sub_mat.data[i * sub_cols + col_idx] = returns_data.data[i * N + j]
			}
			col_idx += 1
		}
	}
	return sub_mat
}

bpt_optimize_v2 :: proc(
	returns_data: ^l.Matrix(f64),
	safety_universe: []bool,
	aspiration_universe: []bool,
	speculative_universe: []bool,
	safety_alloc: f64,
	aspiration_alloc: f64,
	speculative_alloc: f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	combined: []f64,
	w_safety: []f64,
	w_asp: []f64,
	w_spec: []f64,
) {
	N := returns_data.cols

	safe_mat := extract_sub_returns(returns_data, safety_universe, allocator)
	defer l.matrix_free(&safe_mat)
	asp_mat := extract_sub_returns(returns_data, aspiration_universe, allocator)
	defer l.matrix_free(&asp_mat)
	spec_mat := extract_sub_returns(returns_data, speculative_universe, allocator)
	defer l.matrix_free(&spec_mat)

	safe_cov := fin.covariance_matrix(&safe_mat, allocator)
	defer l.matrix_free(&safe_cov)
	sub_w_safety := fin.min_variance_portfolio(&safe_cov, allocator)
	defer delete(sub_w_safety, allocator)

	asp_mu := fin.expected_returns_from_history(&asp_mat, allocator)
	defer delete(asp_mu, allocator)
	asp_cov := fin.covariance_matrix(&asp_mat, allocator)
	defer l.matrix_free(&asp_cov)
	sub_w_asp := fin.max_sharpe_portfolio(asp_mu, &asp_cov, 0.0, allocator)
	defer delete(sub_w_asp, allocator)

	// ✅ FIX: Use proper Upside Variance for Speculative layer
	sub_w_spec := upside_variance_optimize(&spec_mat, 0.1, 3000, allocator)
	defer delete(sub_w_spec, allocator)

	w_safety = make([]f64, N, allocator)
	w_asp = make([]f64, N, allocator)
	w_spec = make([]f64, N, allocator)

	safe_idx := 0
	asp_idx := 0
	spec_idx := 0

	for j in 0 ..< N {
		if safety_universe[j] {w_safety[j] = sub_w_safety[safe_idx]; safe_idx += 1}
		if aspiration_universe[j] {w_asp[j] = sub_w_asp[asp_idx]; asp_idx += 1}
		if speculative_universe[j] {w_spec[j] = sub_w_spec[spec_idx]; spec_idx += 1}
	}

	combined = make([]f64, N, allocator)
	for i in 0 ..< N {
		combined[i] =
			(safety_alloc * w_safety[i]) +
			(aspiration_alloc * w_asp[i]) +
			(speculative_alloc * w_spec[i])
	}

	return combined, w_safety, w_asp, w_spec
}
