
package ML

import l "../../linalg"
import "core:math"
import "core:mem"
// ============================================================================
// Weighted Least Squares (WLS)
// Solves: min (y - Xβ)ᵀ W (y - Xβ), W = diag(weights)
// ============================================================================
wls_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	weights: []f64,
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	n, p := X.rows, X.cols
	if len(weights) != n do panic("wls_fit: weights length mismatch")

	// Validate weights > 0
	for i in 0 ..< n {
		if weights[i] <= 0.0 do panic("wls_fit: weights must be positive")
	}

	// Transform: X_w = √W * X, y_w = √W * y
	X_w := l.matrix_new(f64, n, p, allocator)
	y_w := make([]f64, n, allocator)

	for i in 0 ..< n {
		sqrt_w := math.sqrt(weights[i])
		for j in 0 ..< p {
			X_w.data[i * X_w.cols + j] = sqrt_w * X.data[i * X.cols + j]
		}
		y_w[i] = sqrt_w * y[i]
	}

	// Solve weighted OLS using existing infrastructure
	result := ols_fit_full(&X_w, y_w, method, allocator)

	// Cleanup transformed inputs
	l.matrix_free(&X_w)
	mem.free(transmute(rawptr)&y_w[0], allocator)

	return result
}
