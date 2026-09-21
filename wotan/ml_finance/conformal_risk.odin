package ml_finance

import util "../util"
import "core:fmt"
import "core:math"
import "core:mem"
// ============================================================================
// Conformal Prediction for Distribution-Free Risk Bounds
// ============================================================================

ConformalPredictor :: struct {
	scores:    []f64,
	alpha:     f64,
	quantile:  f64,
	allocator: mem.Allocator,
}

conformal_new :: proc(allocator: mem.Allocator = context.allocator) -> ConformalPredictor {
	return ConformalPredictor{allocator = allocator, alpha = 0.05}
}

conformal_free :: proc(cp: ^ConformalPredictor) {
	if cp.scores != nil {
		delete(cp.scores, cp.allocator)
	}
}


conformal_calibrate :: proc(
	cp: ^ConformalPredictor,
	actual: []f64,
	predicted: []f64,
	alpha: f64 = 0.05,
) {
	n := len(actual)
	if n == 0 || n != len(predicted) {
		return
	}

	if cp.scores != nil {
		delete(cp.scores, cp.allocator)
	}
	cp.scores = make([]f64, n, cp.allocator)
	cp.alpha = alpha

	// 1. Compute non-conformity scores: E_i = |Y_i - Y_hat_i|
	for i in 0 ..< n {
		cp.scores[i] = math.abs(actual[i] - predicted[i])
	}

	// 2. Sort scores using our custom quicksort
	util.quicksort_f64(cp.scores, 0, n - 1)

	// 3. Compute the conformal quantile Q
	level := math.ceil(f64(n + 1) * (1.0 - alpha)) / f64(n)
	if level > 1.0 {
		level = 1.0
	}

	idx := int(level * f64(n))
	if idx >= n {
		idx = n - 1
	}
	if idx < 0 {
		idx = 0
	}

	cp.quantile = cp.scores[idx]
}

conformal_predict_interval :: proc(
	cp: ^ConformalPredictor,
	y_hat: f64,
) -> (
	lower: f64,
	upper: f64,
) {
	return y_hat - cp.quantile, y_hat + cp.quantile
}

print_conformal_stats :: proc(cp: ^ConformalPredictor) {
	fmt.println("\n--- Conformal Prediction Diagnostics ---")
	fmt.printf("  Calibration Set Size: %d\n", len(cp.scores))
	fmt.printf(
		"  Target Coverage:      %.1f%% (alpha = %.3f)\n",
		(1.0 - cp.alpha) * 100.0,
		cp.alpha,
	)
	fmt.printf("  Conformal Quantile Q: %.4f\n", cp.quantile)
	fmt.println("  Interpretation: For any new forecast Y_hat, the interval")
	fmt.printf(
		"  [Y_hat - %.4f, Y_hat + %.4f] is guaranteed to contain\n",
		cp.quantile,
		cp.quantile,
	)
	fmt.println("  the true value with >= 95% probability (assuming exchangeability).")
}
