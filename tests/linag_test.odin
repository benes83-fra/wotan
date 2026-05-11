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
	X := l.matrix_from_df(&df, []string{"X"}, allocator)
	defer l.matrix_free(&X)

	y := l.vector_from_df(&df, "Y", allocator)

	fmt.println("X:", X.data)
	fmt.println("y:", y)

	// Dynamic matvec
	beta := []f64{2.0}
	y_hat := l.matvec_dyn_simd(&X, beta, allocator)
	fmt.println("y_hat:", y_hat)

	// Fixed-size matmul
	a: matrix[2, 2]f64 = {1, 2, 3, 4}
	b: matrix[2, 2]f64 = {5, 6, 7, 8}
	c := l.matmul(a, b)
	fmt.println("fixed matmul:", c)

	// Fixed-size matvec
	v := [2]f64{1, 2}
	r := l.matvec(a, v)
	fmt.println("fixed matvec:", r)

	fmt.println("=== END LINALG TEST ===")
}
