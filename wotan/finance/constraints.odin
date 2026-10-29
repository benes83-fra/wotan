// wotan/finance/constraints.odin
package finance

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Portfolio Constraints
// ============================================================================
PortfolioConstraints :: struct {
	no_short_selling: bool, // weights >= 0
	max_weight:       f64, // weights <= max_weight (0 = no limit)
	min_weight:       f64, // weights >= min_weight if included (0 = no limit)
	group_limits:     []GroupLimit, // sector/asset class limits
}

GroupLimit :: struct {
	asset_indices: []int, // indices of assets in this group
	max_weight:    f64, // sum of weights in group <= max_weight
}

// ============================================================================
// Projection Operators
// ============================================================================

// Project onto simplex: weights sum to 1, weights >= 0
// Uses Duchi et al. algorithm (O(n log n))
project_simplex :: proc(weights: []f64, allocator: mem.Allocator) -> []f64 {
	n := len(weights)
	result := make([]f64, n, allocator)
	copy(result, weights)

	// Sort in descending order
	sorted := make([]f64, n, allocator)
	defer delete(sorted, allocator) // FIX: Add defer to ensure cleanup
	copy(sorted, result)

	// Simple insertion sort for small n
	for i in 1 ..< n {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] < key {
			sorted[j + 1] = sorted[j]
			j -= 1
		}
		sorted[j + 1] = key
	}

	// Find rho
	cumsum := 0.0
	rho := -1
	for j in 0 ..< n {
		cumsum += sorted[j]
		if sorted[j] + (1.0 - cumsum) / f64(j + 1) > 0.0 {
			rho = j
		} else {
			break
		}
	}

	if rho < 0 {
		// All weights should be zero (shouldn't happen with sum=1)
		for i in 0 ..< n {result[i] = 1.0 / f64(n)}
		return result // sorted will be deleted by defer
	}

	// Compute theta
	cumsum = 0.0
	for j in 0 ..= rho {
		cumsum += sorted[j]
	}
	theta := (cumsum - 1.0) / f64(rho + 1)

	// Project
	for i in 0 ..< n {
		result[i] = math.max(weights[i] - theta, 0.0)
	}

	return result // sorted will be deleted by defer
}

// Project onto box constraints: min_weight <= weights <= max_weight
project_box :: proc(weights: []f64, min_w: f64, max_w: f64) {
	for i in 0 ..< len(weights) {
		if weights[i] < min_w {
			weights[i] = min_w
		}
		if max_w > 0.0 && weights[i] > max_w {
			weights[i] = max_w
		}
	}
}

// Project onto group constraints
project_groups :: proc(weights: []f64, groups: []GroupLimit) {
	for group in groups {
		// Sum current weights in group
		sum := 0.0
		for idx in group.asset_indices {
			if idx >= 0 && idx < len(weights) {
				sum += weights[idx]
			}
		}

		// If sum exceeds limit, scale down proportionally
		if sum > group.max_weight && sum > 1e-10 {
			scale := group.max_weight / sum
			for idx in group.asset_indices {
				if idx >= 0 && idx < len(weights) {
					weights[idx] *= scale
				}
			}
		}
	}
}

// Combined projection: apply all constraints iteratively
project_constraints :: proc(
	weights: []f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
) -> []f64 {
	result := weights

	// Project onto simplex first (ensures sum=1, >=0 if no_short_selling)
	if constraints.no_short_selling {
		result = project_simplex(result, allocator)
	} else {
		// Just ensure sum=1
		sum := 0.0
		for w in result {sum += w}
		if math.abs(sum) > 1e-10 {
			for i in 0 ..< len(result) {
				result[i] /= sum
			}
		}
	}

	// Apply box constraints
	if constraints.max_weight > 0.0 || constraints.min_weight > 0.0 {
		project_box(result, constraints.min_weight, constraints.max_weight)
	}

	// Apply group constraints
	if len(constraints.group_limits) > 0 {
		project_groups(result, constraints.group_limits)
	}

	// Re-normalize if needed
	sum := 0.0
	for w in result {sum += w}
	if math.abs(sum - 1.0) > 1e-6 {
		for i in 0 ..< len(result) {
			result[i] /= sum
		}
	}

	return result
}

// ============================================================================
// Constrained Portfolio Optimization
// ============================================================================

// Compute portfolio variance: w' Σ w
portfolio_variance :: proc(weights: []f64, cov: ^l.Matrix(f64)) -> f64 {
	n := len(weights)
	var := 0.0

	for i in 0 ..< n {
		for j in 0 ..< n {
			var += weights[i] * weights[j] * cov.data[i * cov.cols + j]
		}
	}

	return var
}

// Compute portfolio variance gradient: 2 Σ w
portfolio_variance_gradient :: proc(
	weights: []f64,
	cov: ^l.Matrix(f64),
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)
	grad := make([]f64, n, allocator)

	for i in 0 ..< n {
		for j in 0 ..< n {
			grad[i] += 2.0 * cov.data[i * cov.cols + j] * weights[j]
		}
	}

	return grad
}

// Compute negative Sharpe ratio
negative_sharpe :: proc(weights: []f64, returns: []f64, cov: ^l.Matrix(f64), rf: f64) -> f64 {
	port_return := 0.0
	for i in 0 ..< len(weights) {
		port_return += weights[i] * returns[i]
	}

	port_var := portfolio_variance(weights, cov)
	port_vol := math.sqrt(port_var)

	if port_vol < 1e-10 {
		return 1e10 // Avoid division by zero
	}

	return -(port_return - rf) / port_vol
}

// Compute negative Sharpe gradient
negative_sharpe_gradient :: proc(
	weights: []f64,
	returns: []f64,
	cov: ^l.Matrix(f64),
	rf: f64,
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)

	port_return := 0.0
	for i in 0 ..< n {
		port_return += weights[i] * returns[i]
	}

	port_var := portfolio_variance(weights, cov)
	port_vol := math.sqrt(port_var)

	if port_vol < 1e-10 {
		grad := make([]f64, n, allocator)
		return grad
	}

	// Gradient of -(μ - rf) / σ
	// = -[σ * μ - (μ - rf) * (Σw)/σ] / σ²
	grad := make([]f64, n, allocator)

	var_grad := portfolio_variance_gradient(weights, cov, allocator)

	for i in 0 ..< n {
		grad[i] =
			-(returns[i] * port_vol - (port_return - rf) * var_grad[i] / port_vol) /
			(port_vol * port_vol)
	}

	delete(var_grad, allocator)
	return grad
}

// Constrained minimum variance portfolio
constrained_min_variance_portfolio :: proc(
	cov: ^l.Matrix(f64),
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 1000,
	tol: f64 = 1e-6,
) -> []f64 {
	n := cov.rows

	// Initialize with equal weights
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = 1.0 / f64(n)
	}

	// Project initial weights
	old_weights := weights
	weights = project_constraints(weights, constraints, allocator)
	delete(old_weights, allocator)
	// Projected gradient descent
	lr := 0.01
	prev_var := portfolio_variance(weights, cov)

	for iter in 0 ..< max_iter {
		// Compute gradient
		grad := portfolio_variance_gradient(weights, cov, allocator)

		// Gradient step
		for i in 0 ..< n {
			weights[i] -= lr * grad[i]
		}

		// Project onto constraints
		old_weights := weights
		weights = project_constraints(weights, constraints, allocator)
		delete(old_weights, allocator)
		// Check convergence
		var := portfolio_variance(weights, cov)
		if math.abs(var - prev_var) < tol {
			delete(grad, allocator)
			break
		}
		prev_var = var

		delete(grad, allocator)
	}

	return weights
}

// Constrained maximum Sharpe ratio portfolio
constrained_max_sharpe_portfolio :: proc(
	returns: []f64,
	cov: ^l.Matrix(f64),
	rf: f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 1000,
	tol: f64 = 1e-6,
) -> []f64 {
	n := len(returns)

	// Initialize with equal weights
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = 1.0 / f64(n)
	}

	// Project initial weights
	old_weights := weights
	weights = project_constraints(weights, constraints, allocator)
	delete(old_weights, allocator)
	// Projected gradient descent
	lr := 0.01
	prev_sharpe := negative_sharpe(weights, returns, cov, rf)

	for iter in 0 ..< max_iter {
		// Compute gradient
		grad := negative_sharpe_gradient(weights, returns, cov, rf, allocator)

		// Gradient step
		for i in 0 ..< n {
			weights[i] -= lr * grad[i]
		}

		// Project onto constraints
		old_weights := weights
		weights = project_constraints(weights, constraints, allocator)
		delete(old_weights, allocator)
		// Check convergence
		sharpe := negative_sharpe(weights, returns, cov, rf)
		if math.abs(sharpe - prev_sharpe) < tol {
			delete(grad, allocator)
			break
		}
		prev_sharpe = sharpe

		delete(grad, allocator)
	}

	return weights
}

// Constrained mean-variance portfolio (target return)
constrained_mean_variance_portfolio :: proc(
	returns: []f64,
	cov: ^l.Matrix(f64),
	target_return: f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 1000,
	tol: f64 = 1e-6,
) -> []f64 {
	n := len(returns)

	// Initialize with equal weights
	weights := make([]f64, n, allocator)
	for i in 0 ..< n {
		weights[i] = 1.0 / f64(n)
	}

	// Project initial weights
	old_weights := weights
	weights = project_constraints(weights, constraints, allocator)
	delete(old_weights, allocator)
	// Projected gradient descent with penalty for target return
	lr := 0.01
	penalty := 10.0
	prev_obj := 0.0

	for iter in 0 ..< max_iter {
		// Compute variance gradient
		var_grad := portfolio_variance_gradient(weights, cov, allocator)

		// Compute return gradient
		ret_grad := make([]f64, n, allocator)
		for i in 0 ..< n {
			ret_grad[i] = returns[i]
		}

		// Combined gradient with penalty
		current_return := 0.0
		for i in 0 ..< n {
			current_return += weights[i] * returns[i]
		}

		return_error := current_return - target_return

		for i in 0 ..< n {
			weights[i] -= lr * (var_grad[i] + penalty * return_error * ret_grad[i])
		}

		// Project onto constraints
		old_weights := weights
		weights = project_constraints(weights, constraints, allocator)
		delete(old_weights, allocator)

		// Check convergence
		obj := portfolio_variance(weights, cov) + penalty * return_error * return_error
		if math.abs(obj - prev_obj) < tol {
			delete(var_grad, allocator)
			delete(ret_grad, allocator)
			break
		}
		prev_obj = obj

		delete(var_grad, allocator)
		delete(ret_grad, allocator)
	}

	return weights
}
// ============================================================================
// Strict Constraint Enforcement
// ============================================================================

// enforce_constraints strictly applies all constraints to the weights
// Call this AFTER the optimization solver returns
// Replace the enforce_constraints function with this corrected version:
enforce_constraints :: proc(
	weights: []f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
) -> []f64 {
	out := make([]f64, len(weights), allocator)
	copy(out, weights)

	// 1. Hard clip individual bounds
	if constraints.no_short_selling {
		for i in 0 ..< len(out) {
			if out[i] < 0.0 {out[i] = 0.0}
		}
	}
	if constraints.max_weight > 0.0 {
		for i in 0 ..< len(out) {
			if out[i] > constraints.max_weight {out[i] = constraints.max_weight}
		}
	}

	// 2. Hard clip group/sector limits (Proportional Scaling)
	for group in constraints.group_limits {
		group_sum := 0.0
		for idx in group.asset_indices {
			group_sum += out[idx]
		}
		// If the group exceeds the limit, scale them down proportionally
		if group_sum > group.max_weight + 1e-8 {
			scale := group.max_weight / group_sum
			for idx in group.asset_indices {
				out[idx] *= scale
			}
		}
	}

	// 3. Re-normalize to sum to 1.0 (Iterative projection)
	// Clipping can cause the sum to drop below 1.0. We distribute the remainder
	// to assets that haven't hit their max constraints.
	for _ in 0 ..< 5 { 	// Max 5 iterations to converge
		sum := 0.0
		for w in out {sum += w}
		if math.abs(sum - 1.0) < 1e-6 {break}

		// Find how many assets can still accept more weight
		adjustable_count := 0
		for i in 0 ..< len(out) {
			can_adjust := true
			if constraints.no_short_selling && out[i] <= 0.0 {can_adjust = false}
			if constraints.max_weight > 0.0 &&
			   out[i] >= constraints.max_weight {can_adjust = false}
			if can_adjust {adjustable_count += 1}
		}

		if adjustable_count == 0 {break} 	// Prevent division by zero

		diff := (1.0 - sum) / f64(adjustable_count)
		for i in 0 ..< len(out) {
			can_adjust := true
			if constraints.no_short_selling && out[i] <= 0.0 {can_adjust = false}
			if constraints.max_weight > 0.0 &&
			   out[i] >= constraints.max_weight {can_adjust = false}

			if can_adjust {
				out[i] += diff
				// Re-apply hard limits just in case
				if constraints.max_weight > 0.0 && out[i] > constraints.max_weight {
					out[i] = constraints.max_weight
				}
				if constraints.no_short_selling && out[i] < 0.0 {
					out[i] = 0.0
				}
			}
		}
	}

	return out
}

// ============================================================================
// Exact Euclidean projection onto the constraint set
// ============================================================================
// C = {Σw = 1, lo ≤ w_i ≤ hi} ∩ {Σ_{i∈g} w_i ≤ max_g for each group}, with
// lo = 0 for no_short_selling (raised to min_weight when that is set, as
// project_box does) and hi = max_weight when > 0. The box-and-budget part is
// w_i = clamp(v_i - θ, lo, hi) with θ found by bisection; group limits are added
// with Dykstra's alternating projections, so the result is the nearest feasible
// point (unlike project_constraints, which applies the pieces one after another).
// If the box cannot hold a budget of 1 the weights end at the nearest bound.
@(private)
_project_constraints_exact :: proc(v: []f64, c: PortfolioConstraints) {
	n := len(v)
	if n == 0 {return}
	lo := -math.F64_MAX
	if c.no_short_selling {lo = 0.0}
	if c.min_weight > 0.0 {lo = math.max(lo, c.min_weight)}
	hi := c.max_weight if c.max_weight > 0.0 else math.F64_MAX

	if len(c.group_limits) == 0 {
		_project_box_budget(v, lo, hi)
		return
	}

	// Dykstra: halfspaces first, the box-and-budget set last, so the result
	// satisfies the budget and bounds exactly
	m := len(c.group_limits)
	inc := make([]f64, (m + 1) * n, context.temp_allocator)
	defer delete(inc, context.temp_allocator)
	prev := make([]f64, n, context.temp_allocator)
	defer delete(prev, context.temp_allocator)
	for _ in 0 ..< 10_000 {
		copy(prev, v)
		for g, k in c.group_limits {
			p := inc[k * n:(k + 1) * n]
			for i in 0 ..< n {v[i] += p[i]}
			s, cnt := 0.0, 0
			for idx in g.asset_indices {
				if idx >= 0 && idx < n {s += v[idx]; cnt += 1}
			}
			for i in 0 ..< n {p[i] = v[i]}
			if s > g.max_weight && cnt > 0 {
				shift := (s - g.max_weight) / f64(cnt)
				for idx in g.asset_indices {
					if idx >= 0 && idx < n {v[idx] -= shift}
				}
			}
			for i in 0 ..< n {p[i] -= v[i]}
		}
		p := inc[m * n:]
		for i in 0 ..< n {v[i] += p[i]; p[i] = v[i]}
		_project_box_budget(v, lo, hi)
		for i in 0 ..< n {p[i] -= v[i]}

		change := 0.0
		for i in 0 ..< n {change = math.max(change, math.abs(v[i] - prev[i]))}
		if change < 1e-15 {break}
	}
}

// w_i = clamp(v_i - θ, lo, hi) with Σw = 1
@(private = "file")
_project_box_budget :: proc(v: []f64, lo, hi: f64) {
	n := len(v)
	excess :: proc(v: []f64, theta, lo, hi: f64) -> f64 {
		s := 0.0
		for x in v {s += math.clamp(x - theta, lo, hi)}
		return s - 1.0
	}
	// bracket the root of the decreasing excess(θ), starting from the unclamped θ
	mean := 0.0
	for x in v {mean += x}
	a := (mean - 1.0) / f64(n)
	b := a
	step := 1.0
	for excess(v, a, lo, hi) < 0 && step < 1e300 {a -= step; step *= 2}
	step = 1.0
	for excess(v, b, lo, hi) > 0 && step < 1e300 {b += step; step *= 2}
	for _ in 0 ..< 200 {
		mid := 0.5 * (a + b)
		if mid <= a || mid >= b {break}
		if excess(v, mid, lo, hi) > 0 {a = mid} else {b = mid}
	}
	theta := 0.5 * (a + b)
	for &x in v {x = math.clamp(x - theta, lo, hi)}
}
