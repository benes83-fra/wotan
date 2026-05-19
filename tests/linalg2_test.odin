package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
// Utility


// ------------------------------------------------------------
// 1. Basic 3×3 LU test
// ------------------------------------------------------------
lu_basic_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU BASIC TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	A.data = {2, 1, 1, 4, -6, 0, -2, 7, 2}

	LU, piv, sign, ok := l.lu_decompose(&A, allocator)
	if !ok do panic("LU failed unexpectedly")

	L, U := l.lu_extract_LU(&LU, allocator)

	// Reconstruction: P*A = L*U
	PA := l.matrix_new(f64, 3, 3, allocator)
	for i := 0; i < 3; i += 1 {
		for j := 0; j < 3; j += 1 {
			PA.data[i * 3 + j] = A.data[piv[i] * 3 + j]
		}
	}

	LU_recon := l.matmul_dyn_simd(&L, &U, allocator)

	for i := 0; i < 9; i += 1 {
		assert_close(LU_recon.data[i], PA.data[i], 1e-9, "LU reconstruction")
	}

	fmt.println("LU BASIC TEST OK")
}

// ------------------------------------------------------------
// 2. Solve test: A x = b
// ------------------------------------------------------------
lu_solve_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU SOLVE TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	A.data = {3, 2, -1, 2, -2, 4, -1, 0.5, -1}

	b := []f64{1, -2, 0}

	LU, piv, _, ok := l.lu_decompose(&A, allocator)
	if !ok do panic("LU failed unexpectedly")

	x := l.lu_solve(&LU, piv, b, allocator)

	// Expected solution: x = {1, -2, -2}
	assert_close(x[0], 1.0, 1e-9, "x0")
	assert_close(x[1], -2.0, 1e-9, "x1")
	assert_close(x[2], -2.0, 1e-9, "x2")

	fmt.println("LU SOLVE TEST OK")
}

// ------------------------------------------------------------
// 3. Determinant test
// ------------------------------------------------------------
lu_det_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU DET TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	A.data = {6, 1, 1, 4, -2, 5, 2, 8, 7}

	LU, _, sign, ok := l.lu_decompose(&A, allocator)
	if !ok do panic("LU failed unexpectedly")

	det := l.lu_det(&LU, sign)

	// True determinant = -306
	assert_close(det, -306.0, 1e-9, "det(A)")

	fmt.println("LU DET TEST OK")
}

// ------------------------------------------------------------
// 4. Identity matrix test
// ------------------------------------------------------------
lu_identity_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU IDENTITY TEST ===")

	A := l.matrix_new(f64, 4, 4, allocator)
	for i := 0; i < 4; i += 1 {
		A.data[i * 4 + i] = 1.0
	}

	LU, piv, sign, ok := l.lu_decompose(&A, allocator)
	if !ok do panic("LU failed unexpectedly")

	// piv should be identity
	for i := 0; i < 4; i += 1 {
		if piv[i] != i {
			panic("Identity pivot mismatch")
		}
	}

	// det = 1
	det := l.lu_det(&LU, sign)
	assert_close(det, 1.0, 1e-12, "det(I)")

	fmt.println("LU IDENTITY TEST OK")
}

// ------------------------------------------------------------
// 5. Singular matrix test
// ------------------------------------------------------------
lu_singular_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU SINGULAR TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	A.data = {
		1,
		2,
		3,
		2,
		4,
		6,
		3,
		6,
		9, // rank 1
	}

	_, _, _, ok := l.lu_decompose(&A, allocator)
	if ok {
		panic("LU should have failed on singular matrix")
	}

	fmt.println("LU SINGULAR TEST OK")
}

// ------------------------------------------------------------
// 6. Random matrix stress test
// ------------------------------------------------------------
lu_random_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU RANDOM STRESS TEST ===")

	for trial := 0; trial < 20; trial += 1 {
		n := 6
		A := l.matrix_new(f64, n, n, allocator)

		// Fill with random-ish deterministic values
		for i := 0; i < n * n; i += 1 {
			A.data[i] = math.sin(f64(i * trial + 1)) * 10.0
		}

		LU, piv, _, ok := l.lu_decompose(&A, allocator)
		if !ok do panic("LU failed unexpectedly")

		L, U := l.lu_extract_LU(&LU, allocator)

		// Check P*A = L*U
		PA := l.matrix_new(f64, n, n, allocator)
		for i := 0; i < n; i += 1 {
			for j := 0; j < n; j += 1 {
				PA.data[i * n + j] = A.data[piv[i] * n + j]
			}
		}

		LU_recon := l.matmul_dyn_simd(&L, &U, allocator)

		for i := 0; i < n * n; i += 1 {
			assert_close(LU_recon.data[i], PA.data[i], 1e-6, "random LU recon")
		}
	}

	fmt.println("LU RANDOM STRESS TEST OK")
}

// ------------------------------------------------------------
// 7. Numeric dump test (debugging)
// ------------------------------------------------------------
lu_numeric_dump_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU NUMERIC DUMP TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	A.data = {1, 2, 3, 0, 1, 4, 5, 6, 0}

	LU, piv, sign, ok := l.lu_decompose(&A, allocator)
	if !ok do panic("LU failed unexpectedly")

	fmt.printf("A = %v\n", A.data)
	fmt.printf("piv = %v, sign = %d\n", piv, sign)

	L, U := l.lu_extract_LU(&LU, allocator)

	fmt.println("L:")
	for i := 0; i < 3; i += 1 {
		for j := 0; j < 3; j += 1 {
			fmt.printf("% .12f ", L.data[i * 3 + j])
		}
		fmt.println()
	}

	fmt.println("U:")
	for i := 0; i < 3; i += 1 {
		for j := 0; j < 3; j += 1 {
			fmt.printf("% .12f ", U.data[i * 3 + j])
		}
		fmt.println()
	}

	fmt.println("=== LU NUMERIC DUMP DONE ===")
}

lu_inverse_basic_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU INVERSE BASIC TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	A.data = {1, 2, 3, 0, 1, 4, 5, 6, 0}

	Ainv, ok := l.mat_inverse_lu(&A, allocator)
	if !ok {
		panic("lu_inverse_basic_test: inversion failed")
	}

	// Check A * Ainv ≈ I and Ainv * A ≈ I
	I_left := l.matmul_dyn_simd(&A, &Ainv, allocator)
	I_right := l.matmul_dyn_simd(&Ainv, &A, allocator)

	n := A.rows
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			expected: f64 = (i == j) ? 1.0 : 0.0
			assert_close(I_left.data[i * I_left.cols + j], expected, 1e-9, "A*Ainv")
			assert_close(I_right.data[i * I_right.cols + j], expected, 1e-9, "Ainv*A")
		}
	}

	fmt.println("LU INVERSE BASIC TEST OK")
}

lu_inverse_identity_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU INVERSE IDENTITY TEST ===")

	n := 4
	A := l.matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			A.data[i * A.cols + j] = (i == j) ? 1.0 : 0.0
		}
	}

	Ainv, ok := l.mat_inverse_lu(&A, allocator)
	if !ok {
		panic("lu_inverse_identity_test: inversion failed")
	}

	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			expected: f64 = (i == j) ? 1.0 : 0.0
			assert_close(Ainv.data[i * Ainv.cols + j], expected, 1e-12, "I inverse")
		}
	}

	fmt.println("LU INVERSE IDENTITY TEST OK")
}

lu_inverse_singular_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LU INVERSE SINGULAR TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	// rank-deficient: row2 = row1
	A.data = {1, 2, 3, 1, 2, 3, 0, 0, 1}

	_, ok := l.mat_inverse_lu(&A, allocator)
	if ok {
		panic("lu_inverse_singular_test: expected failure for singular matrix")
	}

	fmt.println("LU INVERSE SINGULAR TEST OK")
}

// -------------------- EIGH TESTS --------------------

eigh_diagonal_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== EIGH DIAGONAL TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	A.data = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0}

	W, V := l.eigh(&A, .Ascending, allocator)

	// Eigenvalues should be [1,2,3]
	assert_close(W[0], 1.0, 1e-12, "λ0")
	assert_close(W[1], 2.0, 1e-12, "λ1")
	assert_close(W[2], 3.0, 1e-12, "λ2")

	// V should be identity up to sign
	for i := 0; i < 3; i += 1 {
		for j := 0; j < 3; j += 1 {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			// allow sign flips: |v_ij| ≈ expected
			vij := V.data[i * V.cols + j]
			if expected == 1.0 {
				assert_close(math.abs(vij), 1.0, 1e-12, "V diag entry")
			} else {
				assert_close(vij, 0.0, 1e-12, "V off-diag entry")
			}
		}
	}

	fmt.println("EIGH DIAGONAL TEST OK")
}

eigh_symmetric_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== EIGH SYMMETRIC TEST ===")

	// Simple symmetric 2x2 with known eigenvalues
	// A = [2 1; 1 2] → eigenvalues 1,3
	A := l.matrix_new(f64, 2, 2, allocator)
	A.data = {2.0, 1.0, 1.0, 2.0}

	W, V := l.eigh(&A, .Ascending, allocator)

	assert_close(W[0], 1.0, 1e-12, "λ_min")
	assert_close(W[1], 3.0, 1e-12, "λ_max")

	// Check A * v_i ≈ λ_i * v_i
	for j := 0; j < 2; j += 1 {
		vx := V.data[0 * V.cols + j]
		vy := V.data[1 * V.cols + j]

		Ax0 := 2.0 * vx + 1.0 * vy
		Ax1 := 1.0 * vx + 2.0 * vy

		lambda := W[j]
		assert_close(Ax0, lambda * vx, 1e-12, "A v = λ v (0)")
		assert_close(Ax1, lambda * vy, 1e-12, "A v = λ v (1)")
	}

	fmt.println("EIGH SYMMETRIC TEST OK")
}

// -------------------- COND / RCOND TESTS --------------------

cond2_svd_identity_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== COND2_SVD IDENTITY TEST ===")

	A := l.matrix_new(f64, 3, 3, allocator)
	for i := 0; i < 3; i += 1 {
		for j := 0; j < 3; j += 1 {
			value: f64
			if (i == j) {
				value = 1.0
			} else {
				value = 0.0
			}
			A.data[i * A.cols + j] = value
		}
	}

	c := l.cond2_svd(&A, allocator)
	rc := l.rcond2_svd(&A, allocator)

	assert_close(c, 1.0, 1e-12, "cond2_svd(I)")
	assert_close(rc, 1.0, 1e-12, "rcond2_svd(I)")

	fmt.println("COND2_SVD IDENTITY TEST OK")
}

cond2_sym_identity_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== COND2_SYM IDENTITY TEST ===")

	A := l.matrix_new(f64, 4, 4, allocator)
	for i := 0; i < 4; i += 1 {
		for j := 0; j < 4; j += 1 {
			value: f64
			if (i == j) {
				value = 1.0
			} else {
				value = 0.0
			}
			A.data[i * A.cols + j] = value
		}
	}

	c := l.cond2_sym(&A, allocator)
	rc := l.rcond2_sym(&A, allocator)

	assert_close(c, 1.0, 1e-12, "cond2_sym(I)")
	assert_close(rc, 1.0, 1e-12, "rcond2_sym(I)")

	fmt.println("COND2_SYM IDENTITY TEST OK")
}

cond2_sym_spd_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== COND2_SYM SPD TEST ===")

	// Diagonal SPD: diag(1, 2, 10) → κ = 10 / 1 = 10
	A := l.matrix_new(f64, 3, 3, allocator)
	A.data = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 10.0}

	c := l.cond2_sym(&A, allocator)
	rc := l.rcond2_sym(&A, allocator)

	assert_close(c, 10.0, 1e-12, "cond2_sym(diag)")
	assert_close(rc, 0.1, 1e-12, "rcond2_sym(diag)")

	fmt.println("COND2_SYM SPD TEST OK")
}

cond2_svd_rank_deficient_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== COND2_SVD RANK-DEFICIENT TEST ===")

	// Rank-1 matrix: [1 2; 2 4]
	A := l.matrix_new(f64, 2, 2, allocator)
	A.data = {1.0, 2.0, 2.0, 4.0}

	c := l.cond2_svd(&A, allocator)
	rc := l.rcond2_svd(&A, allocator)

	if !math.is_inf(c) {
		panic("cond2_svd(rank-deficient) should be +Inf")
	}
	assert_close(rc, 0.0, 0.0, "rcond2_svd(rank-deficient)")

	fmt.println("COND2_SVD RANK-DEFICIENT TEST OK")
}

eigh_cond_full_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== EIGH + COND TEST SUITE ===")

	eigh_diagonal_test(allocator)
	eigh_symmetric_test(allocator)

	cond2_svd_identity_test(allocator)
	cond2_sym_identity_test(allocator)
	cond2_sym_spd_test(allocator)
	cond2_svd_rank_deficient_test(allocator)

	fmt.println("=== EIGH + COND TESTS PASSED ===")
}


wls_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== WLS TEST ===")

	// Simple weighted regression: y = 3 + 2*x + noise
	// Give more weight to precise observations
	X := l.matrix_new(f64, 4, 2, allocator)
	defer l.matrix_free(&X)
	X.data = {1, 1, 1, 2, 1, 3, 1, 4} // intercept + x

	y := []f64{5, 7, 9, 11} // perfect fit: y = 3 + 2*x
	weights := []f64{1, 1, 100, 100} // trust last 2 obs more

	res := ml.wls_fit(&X, y, weights, .Cholesky, allocator)
	//defer ml._ols_result_free(&res, allocator) // your existing cleanup

	fmt.printf("WLS beta = %v (expected ~[3, 2])\n", res.beta)
	// With high weights on last 2 points, should still get ~[3, 2]

	assert_close(res.beta[0], 3.0, 1e-6, "WLS intercept")
	assert_close(res.beta[1], 2.0, 1e-6, "WLS slope")
	fmt.println("WLS test OK")
}
gls_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== GLS TEST ===")

	// Simple case: Ω = I (identity) → GLS = OLS
	// y = 3 + 2*x + ε, ε ~ N(0, I)
	X := l.matrix_new(f64, 3, 2, allocator)
	defer l.matrix_free(&X)
	X.data = {1, 1, 1, 2, 1, 3} // intercept + x

	y := []f64{5, 7, 9} // perfect fit

	// Ω = I (identity covariance)
	Omega := l.matrix_new(f64, 3, 3, allocator)
	defer l.matrix_free(&Omega)
	for i in 0 ..< 3 {Omega.data[i * Omega.cols + i] = 1.0}

	res := ml.gls_fit(&X, y, &Omega, .Cholesky, allocator)
	// defer _ols_result_free(&res, allocator)

	fmt.printf("GLS beta  (Ω=I) = %v (expected ~[3, 2])\n", res.beta)
	assert_close(res.beta[0], 3.0, 1e-9, "GLS intercept")
	assert_close(res.beta[1], 2.0, 1e-9, "GLS slope")

	// Test with non-identity Ω (correlated errors)
	// Ω = [[2, 1, 0], [1, 2, 1], [0, 1, 2]] (SPD tridiagonal)
	Omega2 := l.matrix_new(f64, 3, 3, allocator)
	defer l.matrix_free(&Omega2)
	Omega2.data = {2, 1, 0, 1, 2, 1, 0, 1, 2}

	res2 := ml.gls_fit(&X, y, &Omega2, .QR, allocator)
	// defer _ols_result_free(&res2, allocator)

	fmt.printf("GLS beta (correlated Ω) = %v\n", res2.beta)
	// Should still be close to [3, 2] for this perfect-fit case

	fmt.println("GLS test OK")
}
ridge_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== RIDGE TEST ===")

	// Simple case: y = 3 + 2*x, λ = 0 → should match OLS
	X := l.matrix_new(f64, 3, 2, allocator)
	defer l.matrix_free(&X)
	X.data = {1, 1, 1, 2, 1, 3} // intercept + x

	y := []f64{5, 7, 9} // perfect fit

	// λ = 0 → Ridge = OLS
	res := ml.ridge_fit(&X, y, 0.0, .Cholesky, allocator)
	// defer _ols_result_free(&res, allocator)

	fmt.printf("Ridge beta ( λ = 0 ) = %v (expected ~[3, 2])\n", res.beta)
	assert_close(res.beta[0], 3.0, 1e-9, "Ridge intercept  λ = 0")
	assert_close(res.beta[1], 2.0, 1e-9, "Ridge slope λ = 0")

	// λ > 0 → coefficients shrink toward zero
	res2 := ml.ridge_fit(&X, y, 1.0, .Cholesky, allocator)
	// defer _ols_result_free(&res2, allocator)

	fmt.printf("Ridge beta  ( λ = 1) = %v (shrunk toward 0)\n", res2.beta)
	// With λ=1, expect |β| < |OLS β| (but still close for this perfect-fit case)

	fmt.println("Ridge test OK")
}

lasso_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LASSO TEST ===")

	// Simple case: y = 3 + 2*x, λ = 0 → should match OLS
	X := l.matrix_new(f64, 3, 2, allocator)
	defer l.matrix_free(&X)
	X.data = {1, 1, 1, 2, 1, 3} // intercept + x

	y := []f64{5, 7, 9} // perfect fit

	// λ = 0 → Lasso = OLS (no penalty)
	res := ml.lasso_fit(&X, y, 0.0, 1000, 1e-4, allocator)
	// defer _ols_result_free(&res, allocator)

	fmt.printf("Lasso beta (λ=0) = %v (expected ~[3, 2])\n", res.beta)
	assert_close(res.beta[0], 3.0, 1e-6, "Lasso intercept λ=0")
	assert_close(res.beta[1], 2.0, 1e-6, "Lasso slope λ=0")

	// λ > 0 → coefficients shrink, some may become exactly 0
	// For this perfect-fit case with small n, λ=10 should shrink heavily
	res2 := ml.lasso_fit(&X, y, 10.0, 1000, 1e-4, allocator)
	// defer _ols_result_free(&res2, allocator)

	fmt.printf("Lasso beta (λ=10) = %v (shrunk, possibly sparse)\n", res2.beta)
	// With high λ, expect coefficients closer to 0 (may be exactly 0)

	// Test sparsity: with larger λ, some coefficients should be exactly 0
	X3 := l.matrix_new(f64, 10, 4, allocator) // More features, some irrelevant
	defer l.matrix_free(&X3)
	// Fill X3: first 2 cols relevant, last 2 noise
	for i in 0 ..< 10 {
		X3.data[i * 4 + 0] = 1.0 // intercept
		X3.data[i * 4 + 1] = f64(i + 1) // relevant: x
		X3.data[i * 4 + 2] = rand.float64_uniform(0.0, 1.0) // noise
		X3.data[i * 4 + 3] = rand.float64_uniform(0.0, 1.0) // noise
	}
	y3 := make([]f64, 10, allocator)
	for i in 0 ..< 10 {
		y3[i] = 3.0 + 2.0 * f64(i + 1) + 0.1 * rand.float64_uniform(0.0, 1.0) // true model: intercept + x
	}
	defer delete(y3)

	res3 := ml.lasso_fit(&X3, y3, 1.0, 1000, 1e-4, allocator)
	// defer _ols_result_free(&res3, allocator)

	fmt.printf("Lasso beta (sparse test) = %v\n", res3.beta)
	// Expect beta[2] and/or beta[3] to be exactly 0 (noise features)
	// beta[0] ≈ 3, beta[1] ≈ 2 (signal features)

	fmt.println("Lasso test OK")
}
