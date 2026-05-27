package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"


// Simple test
dt_test :: proc(allocator: mem.Allocator) {
	// Create tiny dataset: y = 2*x + noise
	X := l.matrix_new(f64, 100, 1, allocator)
	y := make([]f64, 100, allocator)

	for i in 0 ..< 100 {
		X.data[i] = f64(i) / 100.0
		y[i] = 2.0 * X.data[i] + 0.1 * rand.float64_normal(0, 1)
	}

	params := ml.TreeParams {
		max_depth   = 5,
		min_samples = 10,
	}
	tree := ml.dt_fit(&X, y, params, allocator)
	defer ml.dt_free(&tree)

	// Predict
	preds := ml.dt_predict(&tree, &X, allocator)
	defer delete(preds, allocator)

	// Check MSE
	mse := 0.0
	for i in 0 ..< 100 {
		err := y[i] - preds[i]
		mse += err * err
	}
	mse /= 100.0

	fmt.printf("Test MSE: %.4f\n", mse)

	l.matrix_free(&X)
	delete(y, allocator)
}
