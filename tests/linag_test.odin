package tests

import l "../wotan/linalg"
import "core:fmt"
import "core:mem"

matrix_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LINALG TEST ===")

	// Dynamic matrix test: 3x1 * [2] → 3
	X := l.matrix_new(f64, 3, 1, allocator)
	defer l.matrix_free(&X)

	X.data[0] = 1.0
	X.data[1] = 2.0
	X.data[2] = 3.0

	beta := []f64{2.0}
	y_hat := l.matvec_dyn(&X, beta, allocator)
	fmt.println("y_hat = X * beta:", y_hat)

	// Fixed-size matmul
	a: matrix[2, 2]f64 = {1, 2, 3, 4}
	b: matrix[2, 2]f64 = {5, 6, 7, 8}
	c := l.matmul(a, b)
	fmt.println("Fixed matmul:", c)

	// Fixed-size matvec
	v := [2]f64{1, 2}
	r := l.matvec(a, v)
	fmt.println("Fixed matvec:", r)

	fmt.println("=== END LINALG TEST ===")
}
