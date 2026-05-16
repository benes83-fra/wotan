package tests

import ml "../wotan/ML"
import w "../wotan/core"
import l "../wotan/linalg"
import "core:fmt"
import "core:mem"

matrix_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LINALG TEST ===")

	// Build DataFrame
	df := w.dataframe_new(allocator)
	defer w.destroy_dataframe(&df)

	col_x := w.column_new("X", .Float, 3, allocator)
	w.append_float(&col_x, 1.0)
	w.append_float(&col_x, 2.0)
	w.append_float(&col_x, 3.0)

	col_y := w.column_new("Y", .Float, 3, allocator)
	w.append_float(&col_y, 10.0)
	w.append_float(&col_y, 20.0)
	w.append_float(&col_y, 30.0)

	w.add_column(&df, col_x)
	w.add_column(&df, col_y)

	fmt.println("DataFrame:")
	w.dataframe_pretty_print(&df)

	// DF → Matrix
	X := l.matrix_from_df(&df, []string{"X", "Y"}, allocator) // 3x2 matrix
	defer l.matrix_free(&X)

	beta := []f64{2.0, 0.5}
	y_hat := l.matvec_dyn_simd(&X, beta, allocator)
	fmt.printf("X (3x2) * beta (2x1) = %v\n", y_hat)
	// Expected: [1*2 + 10*0.5, 2*2 + 20*0.5, 3*2 + 30*0.5] -> [7, 14, 21]

	// 2. Test Dynamic Matrix-Matrix (A * B)
	// Let's create a 2x3 and a 3x2 to get a 2x2 result
	A_dyn := l.matrix_new(f64, 2, 3, allocator)
	defer l.matrix_free(&A_dyn)
	A_dyn.data = {1, 2, 3, 4, 5, 6}

	B_dyn := l.matrix_new(f64, 3, 2, allocator)
	defer l.matrix_free(&B_dyn)
	B_dyn.data = {7, 8, 9, 10, 11, 12}

	C_dyn := l.matmul_dyn_simd(&A_dyn, &B_dyn, allocator)
	defer l.matrix_free(&C_dyn)

	fmt.println("Dynamic matmul result:")
	fmt.printf("[%f, %f]\n", C_dyn.data[0], C_dyn.data[1])
	fmt.printf("[%f, %f]\n", C_dyn.data[2], C_dyn.data[3])
	// Expected: [58, 64] / [139, 154]

	// 3. Comparison with Fixed-size (Ground Truth)
	a_fixed: matrix[2, 3]f64 = {1, 2, 3, 4, 5, 6}
	b_fixed: matrix[3, 2]f64 = {7, 8, 9, 10, 11, 12}
	c_fixed := a_fixed * b_fixed
	fmt.println("Fixed-size comparison:", c_fixed)

	fmt.println("=== END LINALG TEST ===")
}
// Utility: absolute difference
abs_f64 :: proc(x: f64) -> f64 {
	if x < 0 do return -x
	return x
}

// Utility: assert with tolerance
assert_close :: proc(a, b: f64, eps: f64, msg: string) {
	if abs_f64(a - b) > eps {
		fmt.printf("ASSERT FAILED: %s\nExpected: %f\nGot:      %f\n", msg, b, a)
		panic("assert_close failed")
	}
}

simd_linalg_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== SIMD LINALG TEST ===")

	// ------------------------------------------------------------
	// 1. Test dot_simd directly
	// ------------------------------------------------------------
	a := []f64{1, 2, 3, 4, 5, 6}
	b := []f64{7, 8, 9, 10, 11, 12}

	dot_ref := 1 * 7 + 2 * 8 + 3 * 9 + 4 * 10 + 5 * 11 + 6 * 12
	dot_simd_val := l.dot_simd(a, b)

	assert_close(dot_simd_val, f64(dot_ref), 1e-9, "dot_simd mismatch")
	fmt.println("dot_simd OK")

	// ------------------------------------------------------------
	// 2. Test matvec_dyn_simd
	// ------------------------------------------------------------
	M := l.matrix_new(f64, 3, 2, allocator)
	M.data = {1, 2, 3, 4, 5, 6}

	x := []f64{10, 20}

	y := l.matvec_dyn_simd(&M, x, allocator)

	assert_close(y[0], 1 * 10 + 2 * 20, 1e-9, "matvec row 0")
	assert_close(y[1], 3 * 10 + 4 * 20, 1e-9, "matvec row 1")
	assert_close(y[2], 5 * 10 + 6 * 20, 1e-9, "matvec row 2")

	fmt.println("matvec_dyn_simd OK")

	// ------------------------------------------------------------
	// 3. Test matmul_dyn_simd vs fixed-size matrix multiply
	// ------------------------------------------------------------
	A_dyn := l.matrix_new(f64, 2, 3, allocator)
	A_dyn.data = {1, 2, 3, 4, 5, 6}

	B_dyn := l.matrix_new(f64, 3, 2, allocator)
	B_dyn.data = {7, 8, 9, 10, 11, 12}

	C_dyn := l.matmul_dyn_simd(&A_dyn, &B_dyn, allocator)

	// Ground truth using Odin fixed-size matrices
	A_fix: matrix[2, 3]f64 = {1, 2, 3, 4, 5, 6}
	B_fix: matrix[3, 2]f64 = {7, 8, 9, 10, 11, 12}
	C_fix := A_fix * B_fix

	assert_close(C_dyn.data[0], C_fix[0, 0], 1e-9, "C[0,0]")
	assert_close(C_dyn.data[1], C_fix[0, 1], 1e-9, "C[0,1]")
	assert_close(C_dyn.data[2], C_fix[1, 0], 1e-9, "C[1,0]")
	assert_close(C_dyn.data[3], C_fix[1, 1], 1e-9, "C[1,1]")

	fmt.println("matmul_dyn_simd OK")

	fmt.println("=== ALL SIMD TESTS PASSED ===")
}

ols_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== OLS TEST ===")

	// Model: y = 3 + 2*x
	X := l.matrix_new(f64, 3, 2, allocator)
	defer l.matrix_free(&X)

	// First column = intercept, second = x
	X.data = {1, 1, 1, 2, 1, 3}

	y := []f64{5, 7, 9}

	res := ml.ols_fit(&X, y, .Cholesky, allocator)
	beta := res.beta

	fmt.printf("OLS beta = %v\n", beta)

	assert_close(beta[0], 3.0, 1e-9, "OLS intercept")
	assert_close(beta[1], 2.0, 1e-9, "OLS slope")

	fmt.println("OLS test OK")
	fmt.println("=== END OLS TEST ===")
}


ols_full_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== OLS FULL TEST ===")

	// Model: y = 3 + 2x
	X := l.matrix_new(f64, 3, 2, allocator)
	X.data = {1, 1, 1, 2, 1, 3}
	y := []f64{5, 7, 9}

	res := ml.ols_fit_full(&X, y, .QR, allocator)

	assert_close(res.beta[0], 3.0, 1e-9, "intercept")
	assert_close(res.beta[1], 2.0, 1e-9, "slope")
	fmt.printf("beta = %v\n", res.beta)
	fmt.printf("fitted = %v\n", res.fitted)
	fmt.printf("stderr = %v\n", res.stderr)
	fmt.printf("tvalues = %v\n", res.tvalues)
	fmt.printf("sigma2 = %f\n", res.sigma2)
	fmt.printf("residuals = %f\n", res.residuals)
	fmt.printf("r2 = %f\n", res.r2)
	fmt.printf("fstat = %f\n", res.fstat)

	fmt.println("=== OLS FULL TEST OK ===")
}


correlation_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== CORRELATION TEST ===")

	// X = [1 2; 2 4; 3 6] → perfect correlation
	X := l.matrix_new(f64, 3, 2, allocator)
	X.data = {1, 2, 2, 4, 3, 6}

	R := l.correlation(&X, allocator)

	assert_close(R.data[0], 1.0, 1e-9, "corr(0,0)")
	assert_close(R.data[3], 1.0, 1e-9, "corr(1,1)")
	assert_close(R.data[1], 1.0, 1e-9, "corr(0,1)")
	assert_close(R.data[2], 1.0, 1e-9, "corr(1,0)")

	fmt.println("Correlation test OK")
}

qr_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== QR TEST ===")

	A := l.matrix_new(f64, 3, 2, allocator)
	A.data = {1, 2, 3, 4, 5, 6}

	Q, R := l.qr_decompose(&A, allocator)

	// Check A ≈ Q*R
	QR := l.matmul_dyn_simd(&Q, &R, allocator)

	for i := 0; i < 6; i += 1 {
		assert_close(QR.data[i], A.data[i], 1e-9, "QR reconstruction")
	}
	m := 3
	// Check Qᵀ Q ≈ I
	QT := l.matrix_new(f64, m, m, allocator)
	for i := 0; i < m; i += 1 {
		for j := 0; j < m; j += 1 {
			QT.data[i * m + j] = Q.data[j * m + i] // transpose
		}
	}

	QTQ := l.matmul_dyn_simd(&QT, &Q, allocator)

	// Check identity
	for i := 0; i < m; i += 1 {
		for j := 0; j < m; j += 1 {
			temp: f64
			if i == j {
				temp = 1.0
			} else {
				temp = 0.0
			}
			expected := temp
			assert_close(QTQ.data[i * m + j], expected, 1e-9, "QᵀQ orthogonality")
		}
	}


	fmt.println("QR test OK")

}


svd_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== SVD TEST ===")

	// A: 3x2 (tall, non-square)
	A := l.matrix_new(f64, 3, 2, allocator)
	A.data = {1, 2, 3, 4, 5, 6}

	U, S, V := l.svd_jacobi(&A, allocator)

	// --- 1) Check singular values are non-negative and sorted descending
	assert_close(S[0] >= S[1] ? 1 : 0, 1, 1e-9, "S[0] >= S[1]")
	assert_close(S[0] >= 0 ? 1 : 0, 1, 1e-9, "S[0] >= 0")
	assert_close(S[1] >= 0 ? 1 : 0, 1, 1e-9, "S[1] >= 0")

	// --- 2) Check Uᵀ U ≈ I (n x n)
	n := A.cols
	UT := l.matrix_new(f64, n, U.rows, allocator)
	for i := 0; i < U.rows; i += 1 {
		for j := 0; j < n; j += 1 {
			UT.data[j * UT.cols + i] = U.data[i * U.cols + j]
		}
	}
	UTU := l.matmul_dyn_simd(&UT, &U, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			expected: f64
			if i == j {
				expected = 1.0
			} else {
				expected = 0.0
			}
			assert_close(UTU.data[i * UTU.cols + j], expected, 1e-6, "UᵀU orthogonality")
		}
	}

	// --- 3) Check Vᵀ V ≈ I (n x n)
	VT := l.matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			VT.data[j * n + i] = V.data[i * V.cols + j]
		}
	}
	VTV := l.matmul_dyn_simd(&VT, &V, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			expected: f64
			if i == j {
				expected = 1.0
			} else {
				expected = 0.0
			}
			assert_close(VTV.data[i * VTV.cols + j], expected, 1e-6, "VᵀV orthogonality")
		}
	}

	// --- 4) Check reconstruction A ≈ U * diag(S) * Vᵀ
	// Build Σ as n x n
	Sigma := l.matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		Sigma.data[i * Sigma.cols + i] = S[i]
	}

	US := l.matmul_dyn_simd(&U, &Sigma, allocator)
	VT2 := l.matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			VT2.data[i * n + j] = V.data[j * V.cols + i]
		}
	}
	USVT := l.matmul_dyn_simd(&US, &VT2, allocator)

	for i := 0; i < A.rows; i += 1 {
		for j := 0; j < A.cols; j += 1 {
			assert_close(
				USVT.data[i * USVT.cols + j],
				A.data[i * A.cols + j],
				1e-6,
				"SVD reconstruction",
			)
		}
	}

	fmt.println("SVD test OK")
}
svd_golub_reinsch_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== SVD GOLUB - REINSCH TEST ===")

	// A: 3x2 (tall, non-square)
	A := l.matrix_new(f64, 3, 2, allocator)
	A.data = {1, 2, 3, 4, 5, 6}

	U, S, V := l.svd_golub_reinsch(&A, allocator)

	// --- 1) Check singular values are non-negative and sorted descending
	assert_close(S[0] >= S[1] ? 1 : 0, 1, 1e-9, "S[0] >= S[1]")
	assert_close(S[0] >= 0 ? 1 : 0, 1, 1e-9, "S[0] >= 0")
	assert_close(S[1] >= 0 ? 1 : 0, 1, 1e-9, "S[1] >= 0")

	// --- 2) Check Uᵀ U ≈ I (n x n)
	n := A.cols
	UT := l.matrix_new(f64, n, U.rows, allocator)
	for i := 0; i < U.rows; i += 1 {
		for j := 0; j < n; j += 1 {
			UT.data[j * UT.cols + i] = U.data[i * U.cols + j]
		}
	}
	UTU := l.matmul_dyn_simd(&UT, &U, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			expected: f64 = (i == j) ? 1.0 : 0.0
			assert_close(UTU.data[i * UTU.cols + j], expected, 1e-6, "UᵀU orthogonality")
		}
	}

	// --- 3) Check Vᵀ V ≈ I (n x n)
	VT := l.matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			VT.data[j * n + i] = V.data[i * V.cols + j]
		}
	}
	VTV := l.matmul_dyn_simd(&VT, &V, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			expected: f64 = (i == j) ? 1.0 : 0.0
			assert_close(VTV.data[i * VTV.cols + j], expected, 1e-6, "VᵀV orthogonality")
		}
	}

	// --- 4) Check reconstruction A ≈ U * diag(S) * Vᵀ
	Sigma := l.matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		Sigma.data[i * Sigma.cols + i] = S[i]
	}

	US := l.matmul_dyn_simd(&U, &Sigma, allocator)

	VT2 := l.matrix_new(f64, n, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < n; j += 1 {
			VT2.data[i * n + j] = V.data[j * V.cols + i]
		}
	}

	USVT := l.matmul_dyn_simd(&US, &VT2, allocator)

	for i := 0; i < A.rows; i += 1 {
		for j := 0; j < A.cols; j += 1 {
			assert_close(
				USVT.data[i * USVT.cols + j],
				A.data[i * A.cols + j],
				1e-6,
				"SVD GR reconstruction",
			)
		}
	}

	// --- 5) Compare singular values to Jacobi SVD
	Uj, Sj, Vj := l.svd_jacobi(&A, allocator)
	_ = Uj
	_ = Vj
	for i := 0; i < n; i += 1 {
		assert_close(S[i], Sj[i], 1e-6, "S (Golub - Reinsch vs Jacobi)")
	}

	fmt.println("SVD GOLUB - REINSCH test OK")
}

thin_svd_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== THIN SVD TEST ===")

	// A: 3x2 (rank 2)
	A := l.matrix_new(f64, 3, 2, allocator)
	A.data = {1, 2, 3, 4, 5, 6}

	U_t, S_t, V_t := l.svd_thin(&A, .GolubReinsch, allocator)

	m := A.rows
	n := A.cols
	r := len(S_t)

	// --- 1) Check rank r is correct (should be 2 for this A)
	assert_close(f64(r), 2.0, 1e-9, "thin rank")

	// --- 2) Check shapes
	if U_t.rows != m || U_t.cols != r {
		panic("U_t shape mismatch")
	}
	if V_t.rows != n || V_t.cols != r {
		panic("V_t shape mismatch")
	}
	if len(S_t) != r {
		panic("S_t length mismatch")
	}

	// --- 3) Check U_tᵀ U_t ≈ I_r
	UT := l.matrix_new(f64, r, m, allocator)
	for i := 0; i < m; i += 1 {
		for j := 0; j < r; j += 1 {
			UT.data[j * UT.cols + i] = U_t.data[i * U_t.cols + j]
		}
	}
	UTU := l.matmul_dyn_simd(&UT, &U_t, allocator)
	for i := 0; i < r; i += 1 {
		for j := 0; j < r; j += 1 {
			expected: f64 = (i == j) ? 1.0 : 0.0
			assert_close(UTU.data[i * UTU.cols + j], expected, 1e-6, "U_tᵀ U_t orthogonality")
		}
	}

	// --- 4) Check V_tᵀ V_t ≈ I_r
	VT := l.matrix_new(f64, r, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < r; j += 1 {
			VT.data[j * VT.cols + i] = V_t.data[i * V_t.cols + j]
		}
	}
	VTV := l.matmul_dyn_simd(&VT, &V_t, allocator)
	for i := 0; i < r; i += 1 {
		for j := 0; j < r; j += 1 {
			expected: f64 = (i == j) ? 1.0 : 0.0
			assert_close(VTV.data[i * VTV.cols + j], expected, 1e-6, "V_tᵀ V_t orthogonality")
		}
	}

	// --- 5) Reconstruction A ≈ U_t * diag(S_t) * V_tᵀ
	Sigma := l.matrix_new(f64, r, r, allocator)
	for i := 0; i < r; i += 1 {
		Sigma.data[i * Sigma.cols + i] = S_t[i]
	}

	US := l.matmul_dyn_simd(&U_t, &Sigma, allocator)

	VT2 := l.matrix_new(f64, r, n, allocator)
	for i := 0; i < n; i += 1 {
		for j := 0; j < r; j += 1 {
			VT2.data[j * VT2.cols + i] = V_t.data[i * V_t.cols + j]
		}
	}

	USVT := l.matmul_dyn_simd(&US, &VT2, allocator)

	for i := 0; i < m; i += 1 {
		for j := 0; j < n; j += 1 {
			assert_close(
				USVT.data[i * USVT.cols + j],
				A.data[i * A.cols + j],
				1e-6,
				"thin SVD reconstruction",
			)
		}
	}

	// --- 6) Compare thin S with full SVD
	_, S_full, _ := l.svd_golub_reinsch(&A, allocator)
	for i := 0; i < r; i += 1 {
		assert_close(S_t[i], S_full[i], 1e-9, "thin vs full singular values")
	}

	fmt.println("THIN SVD test OK")
}
