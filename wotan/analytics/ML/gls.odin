package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Generalized Least Squares (GLS)
// Solves: min (y - Xβ)ᵀ Ω⁻¹ (y - Xβ), where Ω (n×n) is known SPD
// ============================================================================
gls_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	Omega: ^l.Matrix(f64), // error covariance, n×n SPD
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	n, p := X.rows, X.cols

	// Validate dimensions
	if len(y) != n do panic("gls_fit: y length mismatch")
	if Omega.rows != n || Omega.cols != n do panic("gls_fit: Omega must be n×n")

	// =====================================================================
	// 1. Cholesky decompose Ω = L Lᵀ (in-place on a copy)
	// =====================================================================
	L := l.matrix_new(f64, n, n, allocator)
	copy(L.data, Omega.data)

	// Check SPD before decomposing
	is_spd := true
	for j in 0 ..< n {
		if L.data[j * L.cols + j] <= 0.0 {
			is_spd = false
			break
		}
	}
	if !is_spd {
		l.matrix_free(&L)
		panic("gls_fit: Omega is not positive definite")
	}

	l.cholesky_decompose(&L) // In-place: L now contains lower triangle

	// =====================================================================
	// 2. Transform y: y_star = L⁻¹ y (forward substitution)
	// =====================================================================
	y_star := l.forward_subst_unit_lower_simd(&L, y, allocator)
	defer delete(y_star, allocator)
	// =====================================================================
	// 3. Transform X: X_star = L⁻¹ X (solve L * X_star = X for each column)
	// =====================================================================
	X_star := l.matrix_new(f64, n, p, allocator)

	for j in 0 ..< p {
		// Extract column j of X
		x_col := make([]f64, n, context.temp_allocator)
		defer delete(x_col, context.temp_allocator)
		for i in 0 ..< n {
			x_col[i] = X.data[i * X.cols + j]
		}

		// Solve L * x_star_col = x_col
		x_star_col := l.forward_subst_unit_lower_simd(&L, x_col, context.temp_allocator)
		defer delete(x_star_col, context.temp_allocator)

		// Store in X_star
		for i in 0 ..< n {
			X_star.data[i * X_star.cols + j] = x_star_col[i]
		}

		// mem.free(transmute(rawptr)&x_col[0], context.temp_allocator)
		// mem.free(transmute(rawptr)&x_star_col[0], context.temp_allocator)
	}

	// =====================================================================
	// 4. Solve standard OLS on transformed data: (X_star, y_star)
	// =====================================================================
	result := ols_fit_full(&X_star, y_star, method, allocator)

	// =====================================================================
	// 5. Cleanup temporary matrices
	// =====================================================================
	l.matrix_free(&L)
	l.matrix_free(&X_star)
	// mem.free(transmute(rawptr)&y_star[0], allocator)

	return result
}
