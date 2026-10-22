package optimize

import "../util"
import "base:runtime"
import "core:math"
import "core:mem"

// ============================================================================
// Unified model calibration
// ============================================================================
// One loop shared by the model calibrators (SABR, Heston, prospect theory,
// Hull-White, ...): minimize objective(x) over box-bounded parameters with a
// gradient-based optimizer and central finite-difference gradients.
//
// Per iteration:
//   1. loss = objective(x); keep the best point seen so far
//   2. stop (converged) once loss < config.loss_threshold
//   3. g_i = (f(x + h e_i) - f(x - h e_i)) / 2h  for every parameter
//   4. optimizer step, then clamp every parameter into [lower, upper]
// The result is the best point seen, not the last one.

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

Calibration_Config :: struct {
	optimizer:      OptimizerConfig,
	max_iter:       int,
	loss_threshold: f64, // converged once the loss falls below this
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
}

Calibration_Result :: struct {
	x:           []f64, // best parameters found
	loss:        f64, // objective at x
	converged:   bool, // stopped because loss < loss_threshold
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
			assert(p.lower > 0.0 && p.init > 0.0, "calibrate: .Log parameters need positive init and lower bound")
			z[i], lower[i], upper[i] = math.ln_f64(p.init), math.ln_f64(p.lower), math.ln_f64(p.upper)
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
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
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
