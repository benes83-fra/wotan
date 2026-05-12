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

	res := ml.ols_fit(&X, y, allocator)
	beta := res.beta

	fmt.printf("OLS beta = %v\n", beta)

	assert_close(beta[0], 3.0, 1e-9, "OLS intercept")
	assert_close(beta[1], 2.0, 1e-9, "OLS slope")

	fmt.println("OLS test OK")
	fmt.println("=== END OLS TEST ===")
}
