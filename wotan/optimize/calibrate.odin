package optimize

import w "../core"
import "../util"
import "core:math"
import "core:mem"

// ============================================================================
// Unified model calibration
// ============================================================================
// One procedure shared by the model calibrators (prospect theory, Hull-White,
// ...): minimize objective(x) over box-bounded parameters with central
// finite-difference gradients, by one of two methods (config.method).
//
// .Optimizer (the zero value): a first-order optimizer (config.optimizer: SGD,
// Adam, RMSProp) steps on z, per iteration:
//   1. loss = objective(x); keep the best point seen so far
//   2. stop (converged) once loss < config.loss_threshold
//   3. g_i = (f(x + h e_i) - f(x - h e_i)) / 2h  for every parameter
//   4. optimizer step, then clamp every parameter into [lower, upper]
// The result is the best point seen, not the last one.
//
// .LBFGS: lbfgs_minimize on unconstrained variables u, each mapped smoothly
// into its box in z: z = lower + (upper - lower) sigmoid(u) with two finite
// bounds, lower + exp(u) or upper - exp(u) with one, z = u with none. A
// parameter can approach a bound but not reach it, and the map has no flat
// region in which the search could stall. Gradients are central differences
// in u (step fd_step). The search runs to the optimum (no stop at
// loss_threshold) and stops when no gradient component exceeds gtol, after
// max_iter iterations, or when the objective stops decreasing. With
// square_loss, it minimises loss² (same minimiser for a loss >= 0): an RMSE
// has a kink at an exact fit, its square is smooth.

// How the optimizer moves a parameter.
//   .Linear: steps act on x directly.
//   .Log:    steps act on ln(x), i.e. are relative changes. Use for positive
//            parameters whose scale differs from the learning rate (e.g. a
//            volatility of 0.01 under an Adam step of ~0.05). Needs lower > 0.
Param_Scale :: enum {
	Linear,
	Log,
}

// A calibrated parameter: starting value and box bounds (use math.inf_f64(±1)
// for an open side). The objective always receives natural values.
Calibration_Param :: struct {
	name:  string,
	init:  f64,
	lower: f64,
	upper: f64,
	scale: Param_Scale,
}

Calibration_Method :: enum {
	Optimizer, // first-order steps with config.optimizer
	LBFGS, // lbfgs_minimize on smooth maps of the box
}

Calibration_Config :: struct {
	method:         Calibration_Method,
	optimizer:      OptimizerConfig, // .Optimizer only
	max_iter:       int,
	loss_threshold: f64, // converged once the loss falls below this (.Optimizer: also stops)
	fd_step:        f64, // central finite-difference step h
	// Evaluate the 2n finite-difference probes of each iteration in parallel.
	// The objective must then be safe to call from several threads at once
	// (no shared mutable state; scratch memory from context.temp_allocator).
	// Results are identical to a serial run.
	parallel:       bool,
	// Optional constraint linking parameters (e.g. b < a), applied to the
	// natural values after the box clamp on every step.
	project:        proc(x: []f64),
	// Optional stall stop (0 = off), for objectives without a natural target
	// value such as negative log-likelihoods: converged once the best loss has
	// not improved by more than stall_rel_tol * (1 + |best|) for stall_iters
	// consecutive iterations.
	stall_iters:    int,
	stall_rel_tol:  f64,
	// .LBFGS only: converged also when no gradient component (in u) of the
	// minimised function exceeds gtol (0: only loss < loss_threshold counts)
	gtol:           f64,
	// .LBFGS only: minimise loss² instead of loss (for a loss >= 0 such as an RMSE)
	square_loss:    bool,
}

Calibration_Result :: struct {
	x:           []f64, // best parameters found
	loss:        f64, // objective at x
	converged:   bool, // loss < loss_threshold, or (.LBFGS) the gradient test of gtol
	iterations:  int, // optimizer steps taken
	evaluations: int, // objective evaluations
	allocator:   mem.Allocator,
}

calibrate :: proc(
	params: []Calibration_Param,
	objective: ObjectiveFunc,
	user_data: rawptr,
	config: Calibration_Config,
	allocator := context.allocator,
) -> Calibration_Result {
	if config.method == .LBFGS {
		return _calibrate_lbfgs(params, objective, user_data, config, allocator)
	}
	n := len(params)
	result := Calibration_Result {
		x         = make([]f64, n, allocator),
		loss      = math.inf_f64(1),
		allocator = allocator,
	}

	// z: the optimizer's variables (ln x for .Log parameters, x otherwise);
	// x: natural values passed to the objective.
	z := make([]f64, n, allocator)
	x := make([]f64, n, allocator)
	lower := make([]f64, n, allocator)
	upper := make([]f64, n, allocator)
	gradient := make([]f64, n, allocator)
	probes := make([]f64, 2 * n * n, allocator) // one perturbed copy of z per probe
	probe_loss := make([]f64, 2 * n, allocator)
	defer {
		delete(z, allocator)
		delete(x, allocator)
		delete(lower, allocator)
		delete(upper, allocator)
		delete(gradient, allocator)
		delete(probes, allocator)
		delete(probe_loss, allocator)
	}
	any_log := false
	for p, i in params {
		switch p.scale {
		case .Linear:
			z[i], lower[i], upper[i] = p.init, p.lower, p.upper
		case .Log:
			assert(
				p.lower > 0.0 && p.init > 0.0,
				"calibrate: .Log parameters need positive init and lower bound",
			)
			z[i], lower[i], upper[i] =
				math.ln_f64(p.init), math.ln_f64(p.lower), math.ln_f64(p.upper)
			any_log = true
		}
	}
	_to_natural(params, z, x)
	copy(result.x, x)

	optimizer := optimizer_init(config.optimizer, n, allocator)
	defer optimizer_free(&optimizer)

	job := _Probe_Job {
		objective  = objective,
		user_data  = user_data,
		params     = params,
		z          = z,
		probes     = probes,
		probe_loss = probe_loss,
		n          = n,
		h          = config.fd_step,
	}

	stall_ref := math.inf_f64(1) // best loss at the last significant improvement
	stalled := 0
	for _ in 0 ..< config.max_iter {
		loss := objective(x, user_data)
		result.evaluations += 1
		if loss < result.loss {
			result.loss = loss
			copy(result.x, x)
		}
		if loss < config.loss_threshold {
			result.converged = true
			break
		}
		if config.stall_iters > 0 {
			if result.loss < stall_ref - config.stall_rel_tol * (1.0 + abs(result.loss)) {
				stall_ref = result.loss
				stalled = 0
			} else {
				stalled += 1
				if stalled >= config.stall_iters {
					result.converged = true
					break
				}
			}
		}

		// Probes 2i and 2i+1 are x + h e_i and x - h e_i
		if config.parallel {
			util.parallel_for(2 * n, &job, _probe_range)
		} else {
			_probe_range(&job, 0, 2 * n)
		}
		result.evaluations += 2 * n
		for i in 0 ..< n {
			gradient[i] = (probe_loss[2 * i] - probe_loss[2 * i + 1]) / (2.0 * config.fd_step)
		}

		optimizer_step(&optimizer, z, gradient)
		for i in 0 ..< n {
			z[i] = math.max(lower[i], math.min(upper[i], z[i]))
		}
		if config.project != nil {
			if any_log {
				_to_natural(params, z, x)
				config.project(x)
				for p, i in params {z[i] = math.ln_f64(x[i]) if p.scale == .Log else x[i]}
			} else {
				config.project(z) // z == x: no conversion, so linear problems are unaffected
			}
		}
		_to_natural(params, z, x)
		result.iterations += 1
	}

	return result
}

calibration_result_free :: proc(result: ^Calibration_Result) {
	delete(result.x, result.allocator)
	result.x = nil
}

@(private = "file")
_Probe_Job :: struct {
	objective:  ObjectiveFunc,
	user_data:  rawptr,
	params:     []Calibration_Param,
	z:          []f64,
	probes:     []f64,
	probe_loss: []f64,
	n:          int,
	h:          f64,
}

// _probe_range evaluates finite-difference probes [begin, end).
@(private = "file")
_probe_range :: proc(data: rawptr, begin, end: int) {
	job := (^_Probe_Job)(data)
	n := job.n
	for k in begin ..< end {
		i := k / 2
		probe := job.probes[k * n:(k + 1) * n]
		copy(probe, job.z)
		if k % 2 == 0 {
			probe[i] += job.h
		} else {
			probe[i] -= job.h
		}
		_to_natural(job.params, probe, probe)
		// Release the objective's scratch memory after each probe; this matters
		// on pool threads, whose temp allocators are never reset otherwise.
		w.TEMP_GUARD()
		job.probe_loss[k] = job.objective(probe, job.user_data)
	}
}

// _to_natural maps optimizer variables to natural parameter values (exp for
// .Log parameters, unchanged otherwise). z and x may alias. A .Log parameter
// sitting exactly on a bound maps to the bound itself, not exp(ln(bound)).
@(private = "file")
_to_natural :: proc(params: []Calibration_Param, z, x: []f64) {
	for p, i in params {
		if p.scale != .Log {
			x[i] = z[i]
		} else if z[i] == math.ln_f64(p.lower) {
			x[i] = p.lower
		} else if z[i] == math.ln_f64(p.upper) {
			x[i] = p.upper
		} else {
			x[i] = math.exp_f64(z[i])
		}
	}
}

// ============================================================================
// .LBFGS
// ============================================================================

@(private = "file")
_Calib_LBFGS :: struct {
	objective:   ObjectiveFunc,
	user_data:   rawptr,
	params:      []Calibration_Param,
	config:      ^Calibration_Config,
	lo, hi:      []f64, // the box in z (ln for .Log parameters)
	u:           []f64, // the point whose gradient is being probed
	probes:      []f64, // 2n points in u, then mapped to x in place
	probe_loss:  []f64,
	n:           int,
	evaluations: int,
}

// u -> natural x (z in the box, exp for .Log), then the optional projection.
// u and x may alias.
@(private = "file")
_calib_u_to_x :: proc(c: ^_Calib_LBFGS, u, x: []f64) {
	for p, i in c.params {
		lo, hi := c.lo[i], c.hi[i]
		z: f64
		switch {
		case !math.is_inf(lo) && !math.is_inf(hi):
			z = lo + (hi - lo) / (1.0 + math.exp(-u[i]))
		case !math.is_inf(lo):
			z = lo + math.exp(u[i])
		case !math.is_inf(hi):
			z = hi - math.exp(u[i])
		case:
			z = u[i]
		}
		x[i] = math.exp(z) if p.scale == .Log else z
	}
	if c.config.project != nil {c.config.project(x)}
}

// The function L-BFGS minimises at x: loss, or loss² with square_loss
@(private = "file")
_calib_min_value :: proc(c: ^_Calib_LBFGS, x: []f64) -> f64 {
	f := c.objective(x, c.user_data)
	return f * f if c.config.square_loss else f
}

@(private = "file")
_calib_probe_range :: proc(data: rawptr, begin, end: int) {
	c := (^_Calib_LBFGS)(data)
	n := c.n
	for k in begin ..< end {
		probe := c.probes[k * n:(k + 1) * n]
		copy(probe, c.u)
		probe[k / 2] += c.config.fd_step if k % 2 == 0 else -c.config.fd_step
		_calib_u_to_x(c, probe, probe)
		w.TEMP_GUARD() // pool threads never reset their temp allocators otherwise
		c.probe_loss[k] = _calib_min_value(c, probe)
	}
}

// ValueGradFunc for lbfgs_minimize: the value at u and its central-difference gradient
@(private = "file")
_calib_value_grad :: proc(u, grad: []f64, data: rawptr) -> f64 {
	c := (^_Calib_LBFGS)(data)
	n := c.n
	x := make([]f64, n, context.temp_allocator)
	_calib_u_to_x(c, u, x)
	f := _calib_min_value(c, x)
	copy(c.u, u)
	if c.config.parallel {
		util.parallel_for(2 * n, c, _calib_probe_range)
	} else {
		_calib_probe_range(c, 0, 2 * n)
	}
	c.evaluations += 1 + 2 * n
	for i in 0 ..< n {
		grad[i] = (c.probe_loss[2 * i] - c.probe_loss[2 * i + 1]) / (2.0 * c.config.fd_step)
	}
	return f
}

@(private = "file")
_calibrate_lbfgs :: proc(
	params: []Calibration_Param,
	objective: ObjectiveFunc,
	user_data: rawptr,
	config: Calibration_Config,
	allocator: mem.Allocator,
) -> Calibration_Result {
	n := len(params)
	cfg := config
	result := Calibration_Result {
		x         = make([]f64, n, allocator),
		allocator = allocator,
	}
	c := _Calib_LBFGS {
		objective  = objective,
		user_data  = user_data,
		params     = params,
		config     = &cfg,
		lo         = make([]f64, n, allocator),
		hi         = make([]f64, n, allocator),
		u          = make([]f64, n, allocator),
		probes     = make([]f64, 2 * n * n, allocator),
		probe_loss = make([]f64, 2 * n, allocator),
		n          = n,
	}
	u0 := make([]f64, n, allocator)
	defer {
		delete(c.lo, allocator)
		delete(c.hi, allocator)
		delete(c.u, allocator)
		delete(c.probes, allocator)
		delete(c.probe_loss, allocator)
		delete(u0, allocator)
	}
	// the starting point in u: the inverse of the map, a start on a bound moved
	// just inside it
	for p, i in params {
		z0 := p.init
		lo, hi := p.lower, p.upper
		if p.scale == .Log {
			assert(p.lower > 0.0 && p.init > 0.0, "calibrate: .Log parameters need positive init and lower bound")
			z0, lo, hi = math.ln(p.init), math.ln(p.lower), math.ln(p.upper)
		}
		c.lo[i], c.hi[i] = lo, hi
		switch {
		case !math.is_inf(lo) && !math.is_inf(hi):
			q := clamp((z0 - lo) / (hi - lo), 1e-9, 1 - 1e-9)
			u0[i] = math.ln(q / (1 - q))
		case !math.is_inf(lo):
			u0[i] = math.ln(max(z0 - lo, 1e-9 * (1 + abs(lo))))
		case !math.is_inf(hi):
			u0[i] = math.ln(max(hi - z0, 1e-9 * (1 + abs(hi))))
		case:
			u0[i] = z0
		}
	}
	gtol := cfg.gtol if cfg.gtol > 0 else 0
	r := lbfgs_minimize(_calib_value_grad, u0, &c, cfg.max_iter, gtol, allocator = allocator)
	_calib_u_to_x(&c, u0, result.x)
	result.loss = objective(result.x, user_data)
	c.evaluations += 1
	result.iterations = r.n_iter
	result.evaluations = c.evaluations
	result.converged = result.loss < cfg.loss_threshold || (cfg.gtol > 0 && r.converged)
	return result
}
