package ml_finance

import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Conformal Prediction for Distribution-Free Risk Bounds
// ============================================================================
// Split conformal regression with absolute residuals (Lei et al. 2018, "Distribution-free
// predictive inference for regression"; Angelopoulos & Bates 2021): with n calibration
// scores |y_i - y_hat_i|, Q is the k-th smallest, k = ceil((n + 1)(1 - alpha)). For a new
// point exchangeable with the calibration data, P(|y - y_hat| <= Q) >= 1 - alpha.
// If k > n (n < 1/alpha - 1) no finite Q has the guarantee and Q = +Inf.

ConformalPredictor :: struct {
	scores:    []f64, // sorted calibration scores
	alpha:     f64,
	quantile:  f64, // Q
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

// Pairs with a NaN actual or prediction are left out. Mismatched lengths or no valid pair:
// nothing changes.
conformal_calibrate :: proc(
	cp: ^ConformalPredictor,
	actual: []f64,
	predicted: []f64,
	alpha: f64 = 0.05,
) {
	if len(actual) == 0 || len(actual) != len(predicted) || !(alpha > 0 && alpha < 1) {
		return
	}
	n := 0
	for y, i in actual {if !math.is_nan(y - predicted[i]) {n += 1}}
	if n == 0 {return}
	if cp.scores != nil {
		delete(cp.scores, cp.allocator)
	}
	cp.scores = make([]f64, n, cp.allocator)
	j := 0
	for y, i in actual {
		s := math.abs(y - predicted[i])
		if !math.is_nan(s) {cp.scores[j] = s; j += 1}
	}
	cp.alpha = alpha
	slice.sort(cp.scores)

	// k = ceil((n + 1)(1 - alpha)); the tiny shift keeps an exact integer (e.g. 20 * 0.95)
	// from rounding up past it
	k := int(math.ceil(f64(n + 1) * (1.0 - alpha) - 1e-9))
	k = max(k, 1)
	cp.quantile = k > n ? math.inf_f64(1) : cp.scores[k - 1]
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
		"  [Y_hat - %.4f, Y_hat + %.4f] contains the true value with probability\n",
		cp.quantile,
		cp.quantile,
	)
	fmt.printf("  >= %.1f%% (assuming exchangeability).\n", (1.0 - cp.alpha) * 100.0)
}
