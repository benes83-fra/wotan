// wotan/finance/constraints.odin
package finance

import l "../linalg"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Portfolio Constraints
// ============================================================================
// The feasible set is C = {Σw = 1, lo ≤ w_i ≤ hi} ∩ {Σ_{i∈g} w_i ≤ max_g}, with
// lo = 0 for no_short_selling (raised to min_weight when that is > 0) and
// hi = max_weight when > 0. min_weight applies to every asset; there is no
// "only if included" (semi-continuous) handling, which would make C non-convex.
PortfolioConstraints :: struct {
	no_short_selling: bool, // weights >= 0
	max_weight:       f64, // weights <= max_weight (0 = no limit)
	min_weight:       f64, // weights >= min_weight for every asset (0 = no limit)
	group_limits:     []GroupLimit, // sector/asset class limits
}

GroupLimit :: struct {
	asset_indices: []int, // indices of assets in this group (no duplicates)
	max_weight:    f64, // sum of weights in group <= max_weight
}

// ============================================================================
// Projection Operators
// ============================================================================

// Euclidean projection onto the simplex {w ≥ 0, Σw = 1} (Duchi et al. 2008)
project_simplex :: proc(weights: []f64, allocator: mem.Allocator) -> []f64 {
	n := len(weights)
	result := make([]f64, n, allocator)
	if n == 0 {return result}
	sorted := slice.clone(weights, context.temp_allocator)
	defer delete(sorted, context.temp_allocator)
	slice.sort(sorted)
	slice.reverse(sorted)

	cumsum, theta := 0.0, 0.0
	for j in 0 ..< n {
		cumsum += sorted[j]
		t := (cumsum - 1.0) / f64(j + 1)
		if sorted[j] - t > 0.0 {theta = t}
	}
	for i in 0 ..< n {result[i] = math.max(weights[i] - theta, 0.0)}
	return result
}

// Clip to min_weight <= w <= max_weight in place (max_w = 0: no upper limit).
// A clip, not a projection onto the budget: the sum changes
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

// Scale each over-limit group down to its limit in place (the sum changes)
project_groups :: proc(weights: []f64, groups: []GroupLimit) {
	for group in groups {
		sum := 0.0
		for idx in group.asset_indices {
			if idx >= 0 && idx < len(weights) {
				sum += weights[idx]
			}
		}
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

// Nearest point of the feasible set (see PortfolioConstraints), as a new slice;
// the input is not modified
project_constraints :: proc(
	weights: []f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
) -> []f64 {
	result := slice.clone(weights, allocator)
	_project_constraints_exact(result, constraints)
	return result
}

// Exact Euclidean projection onto C in place. If the box cannot hold a budget of
// 1 the weights end at the nearest bound
@(private)
_project_constraints_exact :: proc(v: []f64, c: PortfolioConstraints) {
	_project_feasible(v, c, nil, 0)
}

// Projection onto C, intersected with {μᵀw = target} when mu is given.
// Box, budget and target: w_i = clamp(v_i - θ - φ μ_i, lo, hi), θ and φ by nested
// bisection (μᵀw is monotone in φ). Group limits are added with Dykstra's
// alternating projections, that set last so the result meets the budget, bounds
// and target exactly
@(private = "file")
_project_feasible :: proc(v: []f64, c: PortfolioConstraints, mu: []f64, target: f64) {
	n := len(v)
	if n == 0 {return}
	lo := -math.F64_MAX
	if c.no_short_selling {lo = 0.0}
	if c.min_weight > 0.0 {lo = math.max(lo, c.min_weight)}
	hi := c.max_weight if c.max_weight > 0.0 else math.F64_MAX

	m := len(c.group_limits)
	if m == 0 {
		_project_box_budget_target(v, lo, hi, mu, target)
		return
	}
	n_sets := m

	inc := make([]f64, (n_sets + 1) * n, context.temp_allocator)
	defer delete(inc, context.temp_allocator)
	prev := make([]f64, n, context.temp_allocator)
	defer delete(prev, context.temp_allocator)
	for _ in 0 ..< 20_000 {
		copy(prev, v)
		for k in 0 ..< n_sets {
			p := inc[k * n:(k + 1) * n]
			for i in 0 ..< n {v[i] += p[i]; p[i] = v[i]}
			g := c.group_limits[k]
			s, cnt := 0.0, 0
			for idx in g.asset_indices {
				if idx >= 0 && idx < n {s += v[idx]; cnt += 1}
			}
			if s > g.max_weight && cnt > 0 {
				shift := (s - g.max_weight) / f64(cnt)
				for idx in g.asset_indices {
					if idx >= 0 && idx < n {v[idx] -= shift}
				}
			}
			for i in 0 ..< n {p[i] -= v[i]}
		}
		p := inc[n_sets * n:]
		for i in 0 ..< n {v[i] += p[i]; p[i] = v[i]}
		_project_box_budget_target(v, lo, hi, mu, target)
		for i in 0 ..< n {p[i] -= v[i]}

		change := 0.0
		for i in 0 ..< n {change = math.max(change, math.abs(v[i] - prev[i]))}
		if change < 1e-16 {break}
	}
}

// w_i = clamp(v_i - θ - φ μ_i, lo, hi) with Σw = 1 and μᵀw = target (φ = 0
// without mu). μᵀw(φ) is non-increasing in φ, so φ is found by bisection. A
// target outside the attainable range ends at the nearest attainable return
@(private = "file")
_project_box_budget_target :: proc(v: []f64, lo, hi: f64, mu: []f64, target: f64) {
	if mu == nil {
		_project_box_budget(v, lo, hi)
		return
	}
	orig := slice.clone(v, context.temp_allocator)
	defer delete(orig, context.temp_allocator)
	at :: proc(v, orig, mu: []f64, phi, lo, hi: f64) -> f64 {
		for i in 0 ..< len(v) {v[i] = orig[i] - phi * mu[i]}
		_project_box_budget(v, lo, hi)
		return l.dot_simd(mu, v)
	}
	// scale φ to the data: one unit of φ moves the weights by about |μ|
	mu_max := 0.0
	for x in mu {mu_max = math.max(mu_max, math.abs(x))}
	if mu_max == 0 {
		_project_box_budget(v, lo, hi)
		return
	}
	step := 1.0 / mu_max
	a, b := 0.0, 0.0
	for at(v, orig, mu, a, lo, hi) < target && step < 1e300 {a -= step; step *= 2}
	step = 1.0 / mu_max
	for at(v, orig, mu, b, lo, hi) > target && step < 1e300 {b += step; step *= 2}
	for _ in 0 ..< 200 {
		mid := 0.5 * (a + b)
		if mid <= a || mid >= b {break}
		if at(v, orig, mu, mid, lo, hi) > target {a = mid} else {b = mid}
	}
	at(v, orig, mu, 0.5 * (a + b), lo, hi)
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

// ============================================================================
// Objectives
// ============================================================================

// Compute portfolio variance: w' Σ w
portfolio_variance :: proc(weights: []f64, cov: ^l.Matrix(f64)) -> f64 {
	n := len(weights)
	var := 0.0
	for i in 0 ..< n {
		var += weights[i] * l.dot_simd(cov.data[i * cov.cols:i * cov.cols + n], weights)
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
		grad[i] = 2.0 * l.dot_simd(cov.data[i * cov.cols:i * cov.cols + n], weights)
	}
	return grad
}

// Negative Sharpe ratio -(μᵀw - rf) / σ(w); 1e10 for a zero-volatility portfolio
negative_sharpe :: proc(weights: []f64, returns: []f64, cov: ^l.Matrix(f64), rf: f64) -> f64 {
	port_return := l.dot_simd(weights, returns)
	port_vol := math.sqrt(math.max(portfolio_variance(weights, cov), 0.0))
	if port_vol < 1e-10 {
		return 1e10 // Avoid division by zero
	}
	return -(port_return - rf) / port_vol
}

// Gradient of the negative Sharpe ratio: -(μ σ - (μᵀw - rf) Σw / σ) / σ²
negative_sharpe_gradient :: proc(
	weights: []f64,
	returns: []f64,
	cov: ^l.Matrix(f64),
	rf: f64,
	allocator: mem.Allocator,
) -> []f64 {
	n := len(weights)
	grad := make([]f64, n, allocator)
	port_return := l.dot_simd(weights, returns)
	port_vol := math.sqrt(math.max(portfolio_variance(weights, cov), 0.0))
	if port_vol < 1e-10 {
		return grad
	}
	excess := port_return - rf
	for i in 0 ..< n {
		sigma_w := l.dot_simd(cov.data[i * cov.cols:i * cov.cols + n], weights)
		grad[i] = -(returns[i] * port_vol - excess * sigma_w / port_vol) / (port_vol * port_vol)
	}
	return grad
}

// ============================================================================
// Constrained Portfolio Optimization
// ============================================================================
// All three solve min f(w) over the feasible set by spectral projected gradient
// (Birgin, Martínez & Raydan 2000): Barzilai-Borwein steps, a non-monotone
// Armijo line search and the exact projection. Variance is convex and the
// negative Sharpe ratio is pseudoconvex where the excess return is positive, so
// a stationary point is the global optimum. Iteration stops when the projected
// gradient step ‖P(w - ∇f) - w‖∞ falls below tol (f scaled to O(1)).

@(private = "file")
_Objective :: struct {
	cov:    ^l.Matrix(f64),
	mu:     []f64,
	rf:     f64,
	scale:  f64, // the objective is multiplied by this
	sharpe: bool,
}

@(private = "file")
_obj_value :: proc(o: ^_Objective, w: []f64) -> f64 {
	if o.sharpe {return o.scale * negative_sharpe(w, o.mu, o.cov, o.rf)}
	return o.scale * portfolio_variance(w, o.cov)
}

@(private = "file")
_obj_grad :: proc(o: ^_Objective, w, g: []f64) {
	n := len(w)
	if o.sharpe {
		gs := negative_sharpe_gradient(w, o.mu, o.cov, o.rf, context.temp_allocator)
		for i in 0 ..< n {g[i] = o.scale * gs[i]}
		delete(gs, context.temp_allocator)
		return
	}
	for i in 0 ..< n {
		g[i] = 2.0 * o.scale * l.dot_simd(o.cov.data[i * o.cov.cols:i * o.cov.cols + n], w)
	}
}

@(private = "file")
_spg_minimize :: proc(
	o: ^_Objective,
	c: PortfolioConstraints,
	target_mu: []f64,
	target: f64,
	n: int,
	max_iter: int,
	tol: f64,
	allocator: mem.Allocator,
) -> []f64 {
	MEMORY :: 10
	ALPHA_MIN, ALPHA_MAX :: 1e-30, 1e30
	w := make([]f64, n, allocator)
	if n == 0 {return w}
	buf := make([]f64, 5 * n, context.temp_allocator)
	defer delete(buf, context.temp_allocator)
	g, d, wn, gn, pg :=
		buf[0:n], buf[n:2 * n], buf[2 * n:3 * n], buf[3 * n:4 * n], buf[4 * n:5 * n]

	for i in 0 ..< n {w[i] = 1.0 / f64(n)}
	_project_feasible(w, c, target_mu, target)
	f := _obj_value(o, w)
	_obj_grad(o, w, g)
	history: [MEMORY]f64
	for &h in history {h = f}

	pg_norm :: proc(w, g, pg: []f64, c: PortfolioConstraints, mu: []f64, target: f64) -> f64 {
		for i in 0 ..< len(w) {pg[i] = w[i] - g[i]}
		_project_feasible(pg, c, mu, target)
		r := 0.0
		for i in 0 ..< len(w) {r = math.max(r, math.abs(pg[i] - w[i]))}
		return r
	}
	alpha := 1.0
	if r := pg_norm(w, g, pg, c, target_mu, target);
	   r > 0 {alpha = math.clamp(1.0 / r, ALPHA_MIN, ALPHA_MAX)}

	for k in 0 ..< max_iter {
		if pg_norm(w, g, pg, c, target_mu, target) < tol {break}
		for i in 0 ..< n {d[i] = w[i] - alpha * g[i]}
		_project_feasible(d, c, target_mu, target)
		gd := 0.0
		for i in 0 ..< n {
			d[i] -= w[i]
			gd += g[i] * d[i]
		}
		if gd >= 0 {break} 	// no descent left at round-off level

		f_max := history[0]
		for h in history {f_max = math.max(f_max, h)}
		lambda := 1.0
		fn: f64
		for _ in 0 ..< 60 {
			for i in 0 ..< n {wn[i] = w[i] + lambda * d[i]}
			fn = _obj_value(o, wn)
			if fn <= f_max + 1e-4 * lambda * gd {break}
			// safeguarded quadratic interpolation
			lt := -0.5 * lambda * lambda * gd / (fn - f - lambda * gd)
			lambda = math.clamp(lt, 0.1 * lambda, 0.5 * lambda)
		}
		_obj_grad(o, wn, gn)
		sy, ss := 0.0, 0.0
		for i in 0 ..< n {
			s := wn[i] - w[i]
			sy += s * (gn[i] - g[i])
			ss += s * s
		}
		alpha = ALPHA_MAX if sy <= 0 else math.clamp(ss / sy, ALPHA_MIN, ALPHA_MAX)
		copy(w, wn)
		copy(g, gn)
		f = fn
		history[k % MEMORY] = f
	}
	return w
}

// Mean of the covariance diagonal, to scale the variance objective to O(1)
@(private = "file")
_variance_scale :: proc(cov: ^l.Matrix(f64)) -> f64 {
	n := cov.rows
	s := 0.0
	for i in 0 ..< n {s += cov.data[i * cov.cols + i]}
	return f64(n) / s if s > 0 else 1.0
}

// Constrained minimum variance portfolio: min wᵀΣw over the feasible set
constrained_min_variance_portfolio :: proc(
	cov: ^l.Matrix(f64),
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 10_000,
	tol: f64 = 1e-12,
) -> []f64 {
	o := _Objective {
		cov   = cov,
		scale = _variance_scale(cov),
	}
	return _spg_minimize(&o, constraints, nil, 0, cov.rows, max_iter, tol, allocator)
}

// Constrained maximum Sharpe ratio portfolio: max (μᵀw - rf)/σ(w) over the
// feasible set. Global when some feasible portfolio beats rf
constrained_max_sharpe_portfolio :: proc(
	returns: []f64,
	cov: ^l.Matrix(f64),
	rf: f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 10_000,
	tol: f64 = 1e-12,
) -> []f64 {
	o := _Objective {
		cov    = cov,
		mu     = returns,
		rf     = rf,
		scale  = 1.0,
		sharpe = true,
	}
	return _spg_minimize(&o, constraints, nil, 0, len(returns), max_iter, tol, allocator)
}

// Constrained mean-variance portfolio: min wᵀΣw over the feasible set with
// μᵀw = target_return exactly (the target is part of the projection). The
// target must be attainable under the constraints
constrained_mean_variance_portfolio :: proc(
	returns: []f64,
	cov: ^l.Matrix(f64),
	target_return: f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
	max_iter: int = 10_000,
	tol: f64 = 1e-12,
) -> []f64 {
	o := _Objective {
		cov   = cov,
		scale = _variance_scale(cov),
	}
	return _spg_minimize(
		&o,
		constraints,
		returns,
		target_return,
		len(returns),
		max_iter,
		tol,
		allocator,
	)
}

// ============================================================================
// Strict Constraint Enforcement
// ============================================================================

// The nearest feasible weights (exact projection, new slice). The optimizers
// above already return feasible weights; this is for weights from elsewhere
enforce_constraints :: proc(
	weights: []f64,
	constraints: PortfolioConstraints,
	allocator: mem.Allocator,
) -> []f64 {
	return project_constraints(weights, constraints, allocator)
}
