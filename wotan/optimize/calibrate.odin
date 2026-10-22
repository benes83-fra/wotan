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

// A calibrated parameter: starting value and box bounds (use math.inf_f64(±1)
// for an open side).
Calibration_Param :: struct {
	name:  string,
	init:  f64,
	lower: f64,
	upper: f64,
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
	// Optional constraint linking parameters (e.g. b < a), applied after the
	// box clamp on every step.
	project:        proc(x: []f64),
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

	x := make([]f64, n, allocator)
	gradient := make([]f64, n, allocator)
	probes := make([]f64, 2 * n * n, allocator) // one perturbed copy of x per probe
	probe_loss := make([]f64, 2 * n, allocator)
	defer {
		delete(x, allocator)
		delete(gradient, allocator)
		delete(probes, allocator)
		delete(probe_loss, allocator)
	}
	for p, i in params {x[i] = p.init}
	copy(result.x, x)

	optimizer := optimizer_init(config.optimizer, n, allocator)
	defer optimizer_free(&optimizer)

	job := _Probe_Job {
		objective  = objective,
		user_data  = user_data,
		x          = x,
		probes     = probes,
		probe_loss = probe_loss,
		n          = n,
		h          = config.fd_step,
	}

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

		optimizer_step(&optimizer, x, gradient)
		for p, i in params {
			x[i] = math.max(p.lower, math.min(p.upper, x[i]))
		}
		if config.project != nil {config.project(x)}
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
	x:          []f64,
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
		copy(probe, job.x)
		if k % 2 == 0 {
			probe[i] += job.h
		} else {
			probe[i] -= job.h
		}
		// Release the objective's scratch memory after each probe; this matters
		// on pool threads, whose temp allocators are never reset otherwise.
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		job.probe_loss[k] = job.objective(probe, job.user_data)
	}
}
