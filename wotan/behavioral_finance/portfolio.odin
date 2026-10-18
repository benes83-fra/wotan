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

	// Compute moments for classical layers
	mu := fin.expected_returns_from_history(returns_data, allocator)
	defer delete(mu, allocator)

	cov := fin.covariance_matrix(returns_data, allocator)
	defer l.matrix_free(&cov)

	// 1. Safety Layer: Min Variance (Protect the downside)
	w_safety = fin.min_variance_portfolio(&cov, allocator)

	// 2. Aspiration Layer: Max Sharpe (Efficient market growth)
	w_asp = fin.max_sharpe_portfolio(mu, &cov, 0.0, allocator)

	// 3. Speculative Layer: Risk-Seeking Prospect (Chase the upside)
	// alpha > 1.0 makes the utility function convex for gains (risk-seeking).
	// lambda = 1.0 removes loss aversion (they don't care about the downside here).
	spec_params := ProspectPortfolioParams {
		alpha     = 1.2,
		beta      = 0.88,
		lambda    = 1.0,
		ref_point = 0.0,
	}
	w_spec = prospect_portfolio_optimize(returns_data, spec_params, 0.1, 2000, allocator)

	// 4. Combine Weights (The Mental Accounting Pyramid)
	combined = make([]f64, N, allocator)
	for i in 0 ..< N {
		combined[i] =
			(safety_alloc * w_safety[i]) +
			(aspiration_alloc * w_asp[i]) +
			(speculative_alloc * w_spec[i])
	}

	return combined, w_safety, w_asp, w_spec
}
