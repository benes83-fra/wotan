package optimize

import l "../linalg"
import "../util"
import "base:runtime"
import "core:math"
import "core:mem"

// ============================================================================
// Bounded nonlinear least squares (Levenberg-Marquardt)
// ============================================================================
// Minimizes ½·Σ r_i(x)² over box-bounded parameters, the standard method for
// fitting model prices or volatilities to quotes.
//
// Projected Levenberg-Marquardt with an active set: every iterate stays in the
// box. A parameter on a bound whose gradient points out of the box is held
// there for the iteration; the others take the step solving
// (JᵀJ + μ·diag(JᵀJ))·δ = −Jᵀr, computed by QR (linalg.lstsq) on the stacked
// system [J; √(μ·D)]·δ ≈ [−r; 0] so that the conditioning of J is not squared,
// and clipped to the box. The damping μ follows the
// ratio of actual to predicted decrease of the step actually taken (Nielsen
// 1999). The Jacobian comes from central differences (one-sided at a bound),
// its 2n probes optionally in parallel. Solutions on a bound are found exactly.
//
// Stops (converged) when the relative decrease of the cost falls below ftol,
// the step below xtol·(|x| + xtol), or the projected gradient below gtol.
// The .scale field of the parameters is not used.

// Fills residuals (length n_residuals) at the parameter values x
Residual_Func :: proc(x: []f64, residuals: []f64, user_data: rawptr)

Least_Squares_Config :: struct {
	max_iter: int, // 0 = 100
	ftol:     f64, // 0 = 1e-12
	xtol:     f64, // 0 = 1e-10
	gtol:     f64, // 0 = 1e-14
	fd_step:  f64, // finite-difference step, scaled by 1 + |x| (0 = 1e-6)
	// Evaluate the 2n Jacobian probes in parallel. The residual function must
	// then be safe to call from several threads (scratch memory from
	// context.temp_allocator only). Results are identical to a serial run.
	parallel: bool,
}

Least_Squares_Result :: struct {
	x:           []f64, // parameters at the minimum
	cost:        f64, // ½·Σ r²
	rms:         f64, // √(Σ r² / n_residuals)
	converged:   bool, // a tolerance was met (false: max_iter or no further progress)
	iterations:  int,
	evaluations: int, // residual evaluations
	allocator:   mem.Allocator,
}

least_squares :: proc(
	params: []Calibration_Param,
	residual: Residual_Func,
	n_residuals: int,
	user_data: rawptr,
	config: Least_Squares_Config = {},
	allocator := context.allocator,
) -> Least_Squares_Result {
	n, m := len(params), n_residuals
	max_iter := config.max_iter if config.max_iter > 0 else 100
	ftol := config.ftol if config.ftol > 0 else 1e-12
	xtol := config.xtol if config.xtol > 0 else 1e-10
	gtol := config.gtol if config.gtol > 0 else 1e-14
	fd_step := config.fd_step if config.fd_step > 0 else 1e-6

	result := Least_Squares_Result {
		x         = make([]f64, n, allocator),
		allocator = allocator,
	}
	x := make([]f64, n, allocator)
	x_new := make([]f64, n, allocator)
	lo := make([]f64, n, allocator)
	hi := make([]f64, n, allocator)
	r := make([]f64, m, allocator)
	r_new := make([]f64, m, allocator)
	J := l.matrix_new(f64, m, n, allocator)
	free := make([]bool, n, allocator)
	idx := make([]int, n, allocator)
	steps := make([]f64, 2 * n, allocator) // signed offsets of the probes (0: base point)
	probes := make([]f64, 2 * n * n, allocator)
	probe_res := make([]f64, 2 * n * m, allocator)
	defer {
		delete(x, allocator)
		delete(x_new, allocator)
		delete(lo, allocator)
		delete(hi, allocator)
		delete(r, allocator)
		delete(r_new, allocator)
		l.matrix_free(&J)
		delete(free, allocator)
		delete(idx, allocator)
		delete(steps, allocator)
		delete(probes, allocator)
		delete(probe_res, allocator)
	}

	for p, i in params {
		lo[i], hi[i] = p.lower, p.upper
		x[i] = clamp(p.init, p.lower, p.upper)
	}
	residual(x, r, user_data)
	result.evaluations += 1
	cost := _ls_cost(r)

	job := _LS_Job {
		residual  = residual,
		user_data = user_data,
		x         = x,
		steps     = steps,
		probes    = probes,
		probe_res = probe_res,
		n         = n,
		m         = m,
	}

	mu, nu := 0.0, 2.0
	outer: for iter in 0 ..< max_iter {
		result.iterations = iter + 1
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD() // per-iteration scratch

		// Jacobian: central differences, one-sided where a bound is within h
		for j in 0 ..< n {
			h := fd_step * (1.0 + abs(x[j]))
			steps[2 * j] = h if x[j] + h <= hi[j] else 0
			steps[2 * j + 1] = -h if x[j] - h >= lo[j] else 0
		}
		if config.parallel {
			util.parallel_for(2 * n, &job, _ls_probe_range)
		} else {
			_ls_probe_range(&job, 0, 2 * n)
		}
		result.evaluations += 2 * n
		for j in 0 ..< n {
			hp, hm := steps[2 * j], steps[2 * j + 1]
			plus := probe_res[(2 * j) * m:][:m] if hp != 0 else r
			minus := probe_res[(2 * j + 1) * m:][:m] if hm != 0 else r
			span := hp - hm
			for i in 0 ..< m {J.data[i * n + j] = (plus[i] - minus[i]) / span if span > 0 else 0}
		}

		// Gradient g = Jᵀr and Gauss-Newton matrix A = JᵀJ (gain ratio, scaling)
		A := l.xtx_simd(&J, context.temp_allocator)
		g := l.xty(&J, r, context.temp_allocator)
		max_diag := 0.0
		for a in 0 ..< n {max_diag = max(max_diag, A.data[a * n + a])}

		// Active set: parameters on a bound that the descent direction −g leaves
		n_free := 0
		pg := 0.0
		for a in 0 ..< n {
			free[a] = !(x[a] <= lo[a] && g[a] > 0) && !(x[a] >= hi[a] && g[a] < 0)
			if free[a] {
				idx[n_free] = a
				n_free += 1
				pg = max(pg, abs(g[a]))
			}
		}
		if !_ls_finite(pg) {break}
		if pg <= gtol {
			result.converged = true
			break
		}
		if iter == 0 {mu = 1e-3 * max_diag}

		// Stacked system [J_free; √(μ·D)]·δ ≈ [−r; 0] over the free parameters,
		// D = diag(JᵀJ) floored for parameters with no effect. Its least-squares
		// solution is the step (JᵀJ + μ·D)δ = −Jᵀr; QR avoids squaring cond(J).
		stacked := l.matrix_new(f64, m + n_free, n_free, context.temp_allocator)
		rhs := make([]f64, m + n_free, context.temp_allocator)
		for i in 0 ..< m {
			for a in 0 ..< n_free {stacked.data[i * n_free + a] = J.data[i * n + idx[a]]}
			rhs[i] = -r[i]
		}

		for {
			for a in 0 ..< n_free {
				ia := idx[a]
				row := stacked.data[(m + a) * n_free:][:n_free]
				for &v in row {v = 0}
				row[a] = math.sqrt(mu * max(A.data[ia * n + ia], 1e-12 * max_diag))
			}
			sol := l.lstsq(&stacked, rhs, .QR, context.temp_allocator)
			delta := sol.beta
			ok := true
			for d in delta {ok &&= _ls_finite(d)}
			if !ok {
				mu, nu = mu * nu, nu * 2
				if nu > 1e30 {break outer}
				continue
			}

			// Step clipped to the box
			copy(x_new, x)
			for a in 0 ..< n_free {
				ia := idx[a]
				x_new[ia] = clamp(x[ia] + delta[a], lo[ia], hi[ia])
			}
			step_norm, x_norm := 0.0, 0.0
			for a in 0 ..< n {
				d := x_new[a] - x[a]
				step_norm += d * d
				x_norm += x[a] * x[a]
			}
			if math.sqrt(step_norm) <= xtol * (math.sqrt(x_norm) + xtol) {
				result.converged = true
				break outer
			}

			residual(x_new, r_new, user_data)
			result.evaluations += 1
			cost_new := _ls_cost(r_new)

			// Decrease predicted by the Gauss-Newton model for the step taken:
			// −gᵀs − ½·sᵀAs
			pred := 0.0
			for a in 0 ..< n {
				sa := x_new[a] - x[a]
				As := 0.0
				for b in 0 ..< n {As += A.data[a * n + b] * (x_new[b] - x[b])}
				pred -= g[a] * sa + 0.5 * sa * As
			}
			rho := (cost - cost_new) / pred if pred > 0 else -1
			if _ls_finite(cost_new) && rho > 0 {
				decrease := cost - cost_new
				copy(x, x_new)
				copy(r, r_new)
				cost = cost_new
				mu *= max(1.0 / 3.0, 1.0 - math.pow(2.0 * rho - 1.0, 3))
				nu = 2
				if decrease <= ftol * cost {
					result.converged = true
					break outer
				}
				break
			}
			mu, nu = mu * nu, nu * 2
			if nu > 1e30 {break outer} 	// no further progress possible
		}
	}

	copy(result.x, x)
	result.cost = cost
	result.rms = math.sqrt(2.0 * cost / f64(max(m, 1)))
	return result
}

least_squares_result_free :: proc(result: ^Least_Squares_Result) {
	delete(result.x, result.allocator)
	result.x = nil
}

@(private = "file")
_ls_finite :: #force_inline proc(v: f64) -> bool {return !math.is_nan(v) && !math.is_inf(v)}

@(private = "file")
_ls_cost :: proc(r: []f64) -> f64 {
	s := 0.0
	for v in r {s += v * v}
	return 0.5 * s if _ls_finite(s) else math.inf_f64(1)
}

@(private = "file")
_LS_Job :: struct {
	residual:  Residual_Func,
	user_data: rawptr,
	x:         []f64,
	steps:     []f64,
	probes:    []f64,
	probe_res: []f64,
	n, m:      int,
}

// Probe k moves parameter k/2 by steps[k]; zero steps (bounds) are skipped
@(private = "file")
_ls_probe_range :: proc(data: rawptr, begin, end: int) {
	job := (^_LS_Job)(data)
	n, m := job.n, job.m
	for k in begin ..< end {
		if job.steps[k] == 0 {continue}
		probe := job.probes[k * n:][:n]
		copy(probe, job.x)
		probe[k / 2] += job.steps[k]
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		job.residual(probe, job.probe_res[k * m:][:m], job.user_data)
	}
}
