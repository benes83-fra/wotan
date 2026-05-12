package tests

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
	A_dyn.data = {1, 2, 3,  4, 5, 6}

	B_dyn := l.matrix_new(f64, 3, 2, allocator)
	defer l.matrix_free(&B_dyn)
	B_dyn.data = {7, 8,  9, 10,  11, 12}

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
