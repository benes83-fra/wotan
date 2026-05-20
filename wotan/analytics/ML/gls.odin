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
// ============================================================================
// GLS with Kronecker-structured Omega: Omega = A ⊗ B
// ============================================================================
gls_fit_kron :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	Omega_A: ^l.Matrix(f64), // First factor (m×m SPD)
	Omega_B: ^l.Matrix(f64), // Second factor (n×n SPD)
	method: OLSMethod = .Cholesky,
	allocator: mem.Allocator = context.allocator,
) -> OLSResult {
	n, p := X.rows, X.cols
	m := Omega_A.rows
	k := Omega_B.rows

	if m * k != n do panic("gls_fit_kron: dimensions mismatch (A.rows * B.rows must equal X.rows)")
	if len(y) != n do panic("gls_fit_kron: y length mismatch")

	// Cholesky decompose factors
	L_A := l.matrix_new(f64, m, m, allocator)
	copy(L_A.data, Omega_A.data)
	l.cholesky_decompose(&L_A)
	defer l.matrix_free(&L_A)

	L_B := l.matrix_new(f64, k, k, allocator)
	copy(L_B.data, Omega_B.data)
	l.cholesky_decompose(&L_B)
	defer l.matrix_free(&L_B)

	// Transform y: y_star = (L_A ⊗ L_B)⁻¹ y
	y_star := _kron_solve_lower(&L_A, &L_B, y, allocator)
	defer delete(y_star, allocator)

	// Transform X: X_star = (L_A ⊗ L_B)⁻¹ X
	X_star := l.matrix_new(f64, n, p, allocator)
	for j in 0 ..< p {
		x_col := make([]f64, n, context.temp_allocator)
		for i in 0 ..< n {x_col[i] = X.data[i * X.cols + j]}

		x_star_col := _kron_solve_lower(&L_A, &L_B, x_col, context.temp_allocator)

		for i in 0 ..< n {X_star.data[i * X_star.cols + j] = x_star_col[i]}
		delete(x_col, context.temp_allocator)
		delete(x_star_col, context.temp_allocator)
	}

	// Solve OLS on transformed data
	result := ols_fit_full(&X_star, y_star, method, allocator)

	// Cleanup
	l.matrix_free(&X_star)
	return result
}
// ============================================================================
// Solve (L_A ⊗ L_B) x = y for x, where L_A (m×m), L_B (n×n) are lower triangular
// Identity: vec(L_B⁻¹ * Y * L_A⁻ᵀ) = (L_A ⊗ L_B)⁻¹ vec(Y)
// y is column-major vectorization of Y (n×m): y[j*n + i] = Y[i,j]
// ============================================================================
_kron_solve_lower :: proc(
	L_A: ^l.Matrix(f64), // m×m lower triangular
	L_B: ^l.Matrix(f64), // n×n lower triangular
	y: []f64, // length m*n
	allocator: mem.Allocator,
) -> []f64 {
	m := L_A.rows
	n := L_B.rows

	// Step 1: Reshape y into Y (n×m), column-major: Y[i,j] = y[j*n + i]
	Y := l.matrix_new(f64, n, m, context.temp_allocator)
	for j in 0 ..< m {
		for i in 0 ..< n {
			Y.data[i * m + j] = y[j * n + i] // Y.cols = m
		}
	}

	// Step 2: Solve L_B * Z = Y (forward substitution, column by column)
	Z := l.matrix_new(f64, n, m, context.temp_allocator)
	for col in 0 ..< m {
		y_col := make([]f64, n, context.temp_allocator)
		for i in 0 ..< n {y_col[i] = Y.data[i * m + col]}

		z_col := l.forward_subst_unit_lower_simd(L_B, y_col, context.temp_allocator)

		for i in 0 ..< n {Z.data[i * m + col] = z_col[i]}
		delete(y_col, context.temp_allocator)
		delete(z_col, context.temp_allocator)
	}

	// Step 3: Solve Z * L_Aᵀ = X  ⇔  L_A * Xᵀ = Zᵀ
	// Transpose Z (n×m) → ZT (m×n)
	ZT := l.matrix_new(f64, m, n, context.temp_allocator)
	for i in 0 ..< n {
		for j in 0 ..< m {
			ZT.data[j * n + i] = Z.data[i * m + j] // ZT.cols = n
		}
	}

	// Solve L_A * W = ZT for W (m×n), column by column
	W := l.matrix_new(f64, m, n, context.temp_allocator)
	for col in 0 ..< n {
		zt_col := make([]f64, m, context.temp_allocator)
		for i in 0 ..< m {zt_col[i] = ZT.data[i * n + col]}

		w_col := l.forward_subst_unit_lower_simd(L_A, zt_col, context.temp_allocator)

		for i in 0 ..< m {W.data[i * n + col] = w_col[i]}
		delete(zt_col, context.temp_allocator)
		delete(w_col, context.temp_allocator)
	}

	// Transpose W (m×n) → X (n×m)
	X := l.matrix_new(f64, n, m, context.temp_allocator)
	for i in 0 ..< m {
		for j in 0 ..< n {
			X.data[j * m + i] = W.data[i * n + j] // X.cols = m
		}
	}

	// Step 4: Vectorize X (n×m) column-major: x[j*n + i] = X[i,j]
	x := make([]f64, m * n, allocator)
	for j in 0 ..< m {
		for i in 0 ..< n {
			x[j * n + i] = X.data[i * m + j]
		}
	}

	// Cleanup
	l.matrix_free(&Y)
	l.matrix_free(&Z)
	l.matrix_free(&ZT)
	l.matrix_free(&W)
	l.matrix_free(&X)

	return x
}
