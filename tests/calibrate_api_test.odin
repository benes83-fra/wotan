package tests

import opt "../wotan/optimize"
import "core:fmt"
import "core:math"
import "core:mem"

// Contracts of the unified calibration loop (optimize.calibrate): convergence,
// box bounds, linked-parameter projection, correct iteration/evaluation counts,
// and bit-identical results with parallel finite-difference probes; for
// method = .LBFGS the minimum to rounding, a bound approached from inside, the
// squared loss of an RMSE-like objective, log-scale parameters and parallel probes.

cal_bowl :: proc(x: []f64, _: rawptr) -> f64 {
	// Scratch from the temp allocator, as a real (thread-safe) objective would use
	tmp := make([]f64, len(x), context.temp_allocator)
	defer delete(tmp, context.temp_allocator)
	tmp[0], tmp[1] = x[0] - 1.0, x[1] + 2.0
	return tmp[0] * tmp[0] + 3.0 * tmp[1] * tmp[1] + 0.5 * tmp[0] * tmp[1]
}

calibrate_api_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Unified Calibration API ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	inf := math.inf_f64(1)
	adam := opt.OptimizerConfig {
		type          = .Adam,
		learning_rate = 0.05,
		beta1         = 0.9,
		beta2         = 0.999,
		epsilon       = 1e-8,
	}

	// 1. Unconstrained bowl: minimum at (1, -2)
	free := []opt.Calibration_Param {
		{"x0", 0.0, -inf, inf, .Linear},
		{"x1", 0.0, -inf, inf, .Linear},
	}
	cfg := opt.Calibration_Config {
		optimizer      = adam,
		max_iter       = 2000,
		loss_threshold = 1e-10,
		fd_step        = 1e-6,
	}
	r1 := opt.calibrate(free, cal_bowl, nil, cfg, allocator)
	defer opt.calibration_result_free(&r1)
	expect(
		&passed,
		&total,
		r1.converged && abs(r1.x[0] - 1.0) < 1e-3 && abs(r1.x[1] + 2.0) < 1e-3,
		fmt.tprintf(
			"bowl converges to (1, -2): (%.5f, %.5f) in %d steps",
			r1.x[0],
			r1.x[1],
			r1.iterations,
		),
	)
	// every step evaluates x once plus 2n probes; the final converged check is one more
	expect(
		&passed,
		&total,
		r1.iterations < cfg.max_iter && r1.evaluations == r1.iterations * (1 + 2 * 2) + 1,
		fmt.tprintf(
			"iteration/evaluation counts consistent (%d steps, %d evals)",
			r1.iterations,
			r1.evaluations,
		),
	)

	// 2. Box bounds: the minimum (x0 = 1) lies outside [2, 3]
	boxed := []opt.Calibration_Param {
		{"x0", 2.5, 2.0, 3.0, .Linear},
		{"x1", 0.0, -inf, inf, .Linear},
	}
	cfg2 := cfg
	cfg2.max_iter = 300
	r2 := opt.calibrate(boxed, cal_bowl, nil, cfg2, allocator)
	defer opt.calibration_result_free(&r2)
	expect(
		&passed,
		&total,
		r2.x[0] == 2.0 && !r2.converged && r2.iterations == 300,
		fmt.tprintf("box bound holds at the edge (x0 = %.17g, %d steps)", r2.x[0], r2.iterations),
	)

	// 3. Projection: x1 >= x0 + 4 links the parameters (unconstrained optimum violates it)
	cfg3 := cfg2
	cfg3.project = proc(x: []f64) {x[1] = math.max(x[1], x[0] + 4.0)}
	linked := []opt.Calibration_Param {
		{"x0", -5.0, -inf, inf, .Linear},
		{"x1", 0.0, -inf, inf, .Linear},
	}
	r3 := opt.calibrate(linked, cal_bowl, nil, cfg3, allocator)
	defer opt.calibration_result_free(&r3)
	expect(
		&passed,
		&total,
		r3.x[1] >= r3.x[0] + 4.0 - 1e-12,
		fmt.tprintf("projection keeps x1 >= x0 + 4 (%.4f, %.4f)", r3.x[0], r3.x[1]),
	)

	// 4. Parallel probes give bit-identical results
	cfg4 := cfg
	cfg4.max_iter = 150
	serial := opt.calibrate(free, cal_bowl, nil, cfg4, allocator)
	defer opt.calibration_result_free(&serial)
	cfg4.parallel = true
	par := opt.calibrate(free, cal_bowl, nil, cfg4, allocator)
	defer opt.calibration_result_free(&par)
	expect(
		&passed,
		&total,
		serial.x[0] == par.x[0] &&
		serial.x[1] == par.x[1] &&
		serial.loss == par.loss &&
		serial.evaluations == par.evaluations,
		"parallel probes == serial (bit-identical)",
	)

	// 5. Log-scale parameters: small-scale minimum at (0.012, 0.08) from (0.01, 0.05),
	//    where a linear-space Adam step (~0.05) would dwarf the parameters
	small_bowl :: proc(x: []f64, _: rawptr) -> f64 {
		assert(x[0] > 0 && x[1] > 0) // the objective receives natural values
		u, v := (x[0] - 0.012) / 0.012, (x[1] - 0.08) / 0.08
		return u * u + v * v
	}
	logp := []opt.Calibration_Param {
		{"sigma", 0.01, 0.001, 0.1, .Log},
		{"a", 0.05, 0.01, 0.5, .Log},
	}
	cfg5 := opt.Calibration_Config {
		optimizer      = adam,
		max_iter       = 500,
		loss_threshold = 1e-10,
		fd_step        = 1e-6,
	}
	r5 := opt.calibrate(logp, small_bowl, nil, cfg5, allocator)
	defer opt.calibration_result_free(&r5)
	expect(
		&passed,
		&total,
		abs(r5.x[0] - 0.012) < 1e-4 && abs(r5.x[1] - 0.08) < 1e-3,
		fmt.tprintf(
			"log scale reaches (0.012, 0.08): (%.5f, %.5f) in %d steps",
			r5.x[0],
			r5.x[1],
			r5.iterations,
		),
	)

	// 6. A log-scale parameter pinned at its bound reports the exact bound value
	pinned := []opt.Calibration_Param {
		{"sigma", 0.03, 0.02, 0.1, .Log},
		{"a", 0.05, 0.01, 0.5, .Log},
	}
	r6 := opt.calibrate(pinned, small_bowl, nil, cfg5, allocator)
	defer opt.calibration_result_free(&r6)
	expect(
		&passed,
		&total,
		r6.x[0] == 0.02,
		fmt.tprintf("log-scale bound is exact (sigma = %.17g)", r6.x[0]),
	)

	// 7. Stall stop: an objective with no known target value (like an NLL) has no
	//    usable loss_threshold; without the stall stop it burns the full budget
	shifted_bowl :: proc(x: []f64, ud: rawptr) -> f64 {return cal_bowl(x, ud) - 5.0}
	cfg7 := opt.Calibration_Config {
		optimizer      = adam,
		max_iter       = 2000,
		loss_threshold = -inf,
		fd_step        = 1e-6,
	}
	r7_off := opt.calibrate(free, shifted_bowl, nil, cfg7, allocator)
	defer opt.calibration_result_free(&r7_off)
	cfg7.stall_iters, cfg7.stall_rel_tol = 50, 1e-12
	r7 := opt.calibrate(free, shifted_bowl, nil, cfg7, allocator)
	defer opt.calibration_result_free(&r7)
	expect(
		&passed,
		&total,
		!r7_off.converged &&
		r7_off.iterations == 2000 &&
		r7.converged &&
		r7.iterations < 2000 &&
		abs(r7.x[0] - 1.0) < 1e-4 &&
		abs(r7.x[1] + 2.0) < 1e-4,
		fmt.tprintf(
			"stall stop converges at (%.5f, %.5f) after %d steps (off: %d steps)",
			r7.x[0],
			r7.x[1],
			r7.iterations,
			r7_off.iterations,
		),
	)

	// 8. method = .LBFGS: the bowl's minimum to rounding, with the evaluations
	//    counted per value-and-gradient call (1 + 2n) plus the final one
	lb := opt.Calibration_Config {
		method         = .LBFGS,
		max_iter       = 200,
		loss_threshold = 1e-12,
		fd_step        = 1e-6,
	}
	r8 := opt.calibrate(free, cal_bowl, nil, lb, allocator)
	defer opt.calibration_result_free(&r8)
	expect(
		&passed,
		&total,
		r8.converged &&
		abs(r8.x[0] - 1.0) < 1e-8 &&
		abs(r8.x[1] + 2.0) < 1e-8 &&
		(r8.evaluations - 1) % (1 + 2 * 2) == 0,
		fmt.tprintf(
			"L-BFGS: bowl minimum (%.10f, %.10f) in %d iterations, %d evaluations",
			r8.x[0],
			r8.x[1],
			r8.iterations,
			r8.evaluations,
		),
	)

	// 9. L-BFGS keeps a bounded parameter in its box and approaches the bound when
	//    the minimum lies beyond it (reaching it once the sigmoid rounds to its
	//    limit): x0 -> 2, x1 -> -2 - 1/12
	r9 := opt.calibrate(boxed, cal_bowl, nil, lb, allocator)
	defer opt.calibration_result_free(&r9)
	expect(
		&passed,
		&total,
		r9.x[0] >= 2.0 && r9.x[0] - 2.0 < 1e-4 && abs(r9.x[1] + 2.0 + 1.0 / 12.0) < 1e-4,
		fmt.tprintf("L-BFGS: bounded x0 approaches the bound 2 (%.10f, %.6f)", r9.x[0], r9.x[1]),
	)

	// 10. square_loss: an RMSE-like objective sqrt(bowl) has a kink at its exact
	//     zero; L-BFGS on its square reaches the minimum
	root_bowl :: proc(x: []f64, ud: rawptr) -> f64 {return math.sqrt(max(cal_bowl(x, ud), 0))}
	lb_sq := lb
	lb_sq.square_loss, lb_sq.loss_threshold = true, 1e-6
	r10 := opt.calibrate(free, root_bowl, nil, lb_sq, allocator)
	defer opt.calibration_result_free(&r10)
	expect(
		&passed,
		&total,
		r10.converged &&
		r10.loss < 1e-6 &&
		abs(r10.x[0] - 1.0) < 1e-6 &&
		abs(r10.x[1] + 2.0) < 1e-6,
		fmt.tprintf(
			"L-BFGS square_loss: RMSE-like objective to %.1e at (%.8f, %.8f)",
			r10.loss,
			r10.x[0],
			r10.x[1],
		),
	)

	// 11. L-BFGS with .Log parameters, and parallel probes bit-identical to serial
	lb_log := lb
	lb_log.loss_threshold = 1e-14
	r11 := opt.calibrate(logp, small_bowl, nil, lb_log, allocator)
	defer opt.calibration_result_free(&r11)
	lb_log.parallel = true
	r11p := opt.calibrate(logp, small_bowl, nil, lb_log, allocator)
	defer opt.calibration_result_free(&r11p)
	expect(
		&passed,
		&total,
		abs(r11.x[0] - 0.012) < 1e-8 &&
		abs(r11.x[1] - 0.08) < 1e-7 &&
		r11.x[0] == r11p.x[0] &&
		r11.x[1] == r11p.x[1] &&
		r11.evaluations == r11p.evaluations,
		fmt.tprintf(
			"L-BFGS log scale: (%.10f, %.10f) in %d iterations; parallel == serial",
			r11.x[0],
			r11.x[1],
			r11.iterations,
		),
	)

	fmt.printf("Calibration API summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
