package tests

import opt "../wotan/optimize"
import "core:fmt"
import "core:math"
import "core:mem"

// Contracts of the unified calibration loop (optimize.calibrate): convergence,
// box bounds, linked-parameter projection, correct iteration/evaluation counts,
// and bit-identical results with parallel finite-difference probes.

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
	adam := opt.OptimizerConfig{type = .Adam, learning_rate = 0.05, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8}

	// 1. Unconstrained bowl: minimum at (1, -2)
	free := []opt.Calibration_Param{{"x0", 0.0, -inf, inf}, {"x1", 0.0, -inf, inf}}
	cfg := opt.Calibration_Config{optimizer = adam, max_iter = 2000, loss_threshold = 1e-10, fd_step = 1e-6}
	r1 := opt.calibrate(free, cal_bowl, nil, cfg, allocator)
	defer opt.calibration_result_free(&r1)
	expect(&passed, &total, r1.converged && abs(r1.x[0] - 1.0) < 1e-3 && abs(r1.x[1] + 2.0) < 1e-3,
		fmt.tprintf("bowl converges to (1, -2): (%.5f, %.5f) in %d steps", r1.x[0], r1.x[1], r1.iterations))
	// every step evaluates x once plus 2n probes; the final converged check is one more
	expect(&passed, &total, r1.iterations < cfg.max_iter && r1.evaluations == r1.iterations * (1 + 2 * 2) + 1,
		fmt.tprintf("iteration/evaluation counts consistent (%d steps, %d evals)", r1.iterations, r1.evaluations))

	// 2. Box bounds: the minimum (x0 = 1) lies outside [2, 3]
	boxed := []opt.Calibration_Param{{"x0", 2.5, 2.0, 3.0}, {"x1", 0.0, -inf, inf}}
	cfg2 := cfg
	cfg2.max_iter = 300
	r2 := opt.calibrate(boxed, cal_bowl, nil, cfg2, allocator)
	defer opt.calibration_result_free(&r2)
	expect(&passed, &total, r2.x[0] == 2.0 && !r2.converged && r2.iterations == 300,
		fmt.tprintf("box bound holds at the edge (x0 = %.17g, %d steps)", r2.x[0], r2.iterations))

	// 3. Projection: x1 >= x0 + 4 links the parameters (unconstrained optimum violates it)
	cfg3 := cfg2
	cfg3.project = proc(x: []f64) {x[1] = math.max(x[1], x[0] + 4.0)}
	linked := []opt.Calibration_Param{{"x0", -5.0, -inf, inf}, {"x1", 0.0, -inf, inf}}
	r3 := opt.calibrate(linked, cal_bowl, nil, cfg3, allocator)
	defer opt.calibration_result_free(&r3)
	expect(&passed, &total, r3.x[1] >= r3.x[0] + 4.0 - 1e-12,
		fmt.tprintf("projection keeps x1 >= x0 + 4 (%.4f, %.4f)", r3.x[0], r3.x[1]))

	// 4. Parallel probes give bit-identical results
	cfg4 := cfg
	cfg4.max_iter = 150
	serial := opt.calibrate(free, cal_bowl, nil, cfg4, allocator)
	defer opt.calibration_result_free(&serial)
	cfg4.parallel = true
	par := opt.calibrate(free, cal_bowl, nil, cfg4, allocator)
	defer opt.calibration_result_free(&par)
	expect(&passed, &total,
		serial.x[0] == par.x[0] && serial.x[1] == par.x[1] && serial.loss == par.loss && serial.evaluations == par.evaluations,
		"parallel probes == serial (bit-identical)")

	fmt.printf("Calibration API summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
