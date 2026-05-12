package ML

import l "../linalg"
import "core:fmt"
import "core:mem"
import "core:simd"

// X: n x p (rows = observations, cols = regressors)
ols_xtx :: proc(X: ^l.Matrix(f64), allocator: mem.Allocator = context.allocator) -> l.Matrix(f64) {
	p := X.cols
	n := X.rows

	XtX := l.matrix_new(f64, p, p, allocator)

	for j := 0; j < p; j += 1 {
		col_j := make([]f64, n, context.temp_allocator)
		for i := 0; i < n; i += 1 {
			col_j[i] = X.data[i * X.cols + j]
		}

		for k := j; k < p; k += 1 {
			col_k := make([]f64, n, context.temp_allocator)
			for i := 0; i < n; i += 1 {
				col_k[i] = X.data[i * X.cols + k]
			}

			v := l.dot_simd(col_j, col_k)
			XtX.data[j * XtX.cols + k] = v
			XtX.data[k * XtX.cols + j] = v // symmetry

			delete(col_k, context.temp_allocator)
		}

		delete(col_j, context.temp_allocator)
	}

	return XtX
}

ols_xty :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	if X.rows != len(y) do panic("ols_xty: dimension mismatch")

	p := X.cols
	n := X.rows

	Xty := make([]f64, p, allocator)

	for j := 0; j < p; j += 1 {
		col_j := make([]f64, n, context.temp_allocator)
		for i := 0; i < n; i += 1 {
			col_j[i] = X.data[i * X.cols + j]
		}
		Xty[j] = l.dot_simd(col_j, y)
		delete(col_j, context.temp_allocator)
	}

	return Xty
}


ols_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	XtX := ols_xtx(X, allocator)
	Xty := ols_xty(X, y, allocator)
	beta := ols_solve_cholesky(&XtX, Xty, allocator)
	// optionally: free XtX, Xty if you want
	return beta
}

ols_solve_cholesky :: proc(
	XtX: ^l.Matrix(f64),
	Xty: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	return l.solve_spd_cholesky(XtX, Xty, allocator)
}
