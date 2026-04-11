package main

import w "./wotan/core"
import "core:fmt"
import "core:mem"

kalman_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== KALMAN FILTER TEST ===")

	x0: [2]f64 = {0.0, 1.0}

	// Matrices are FLAT and COLUMN-MAJOR.
	// For P0 (Identity), it's {col1_row1, col1_row2, col2_row1, col2_row2}
	P0 := matrix[2, 2]f64{
		1.0, 0.0,
		0.0, 1.0,
	}

	// F = {{1, 1}, {0, 1}} -> Column-major: {1.0, 0.0, 1.0, 1.0}
	F := matrix[2, 2]f64{
		1.0, 0.0,
		1.0, 1.0,
	}

	// H = {{1, 0}} (1 row, 2 cols) -> {1.0, 0.0}
	H := matrix[1, 2]f64{
		1.0, 0.0,
	}

	Q := matrix[2, 2]f64{
		0.01, 0.0,
		0.0, 0.01,
	}
	R := matrix[1, 1]f64{
		1.0,
	}

	// REMOVED "2, 1". The compiler infers N and M from the arguments.
	kf := w.kalman_init(x0, P0, F, H, Q, R)

	measurements := [6]f64{1.2, 2.1, 2.9, 4.2, 5.1, 6.05}

	for i in 0 ..< len(measurements) {
		fmt.printf("\n--- Step %d ---\n", i)

		z: [1]f64 = {measurements[i]}

		fmt.println("Predicting...")
		// REMOVED "2, 1"
		w.kalman_predict(&kf)
		fmt.println("Predicted state:", kf.x)

		fmt.println("Updating with measurement:", z)
		// REMOVED "2, 1"
		w.kalman_update(&kf, z)
		fmt.println("Updated state:", kf.x)
	}

	fmt.println("\n=== END KALMAN FILTER TEST ===")
}
