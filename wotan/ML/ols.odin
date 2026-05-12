package ML

import w "../core"
import l "../linalg"
import "core:fmt"
import "core:mem"
import "core:simd"


OLSResult :: struct {
	beta: []f64, // coefficients
	// later: vcov, stderr, t, r2, etc.
}


// X: n x p (rows = observations, cols = regressors)
ols_xtx :: proc(X: ^l.Matrix(f64), allocator: mem.Allocator = context.allocator) -> l.Matrix(f64) {
	return l.xtx(X, allocator)
}

ols_xty :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	return l.xty(X, y, allocator)
}

ols_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	if X.rows != len(y) do panic("ols_fit: dimension mismatch")

	XtX := ols_xtx(X, allocator)
	Xty := ols_xty(X, y, allocator)

	beta := l.solve_spd_cholesky(&XtX, Xty, allocator)

	res: OLSResult
	res.beta = beta
	return res
}

ols_from_df :: proc(
	df: ^w.DataFrame,
	y_name: string,
	x_names: []string,
	add_intercept: bool = true,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	// Build X
	X := l.matrix_from_df(df, x_names, allocator)
	defer l.matrix_free(&X)

	if add_intercept {
		// extend X with leading column of ones
		n := X.rows
		p := X.cols
		X2 := l.matrix_new(f64, n, p + 1, allocator)
		for i := 0; i < n; i += 1 {
			X2.data[i * X2.cols + 0] = 1.0
			for j := 0; j < p; j += 1 {
				X2.data[i * X2.cols + (j + 1)] = X.data[i * X.cols + j]
			}
		}
		l.matrix_free(&X)
		X = X2
	}

	y := l.vector_from_df(df, y_name, allocator)

	return ols_fit(&X, y, allocator)
}
