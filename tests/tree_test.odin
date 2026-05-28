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
	fmt.println("Predictions: %v", preds)
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


rf_test :: proc(allocator: mem.Allocator) {
	// Generate synthetic data: y = 2*x1 + 3*x2 + noise
	n := 200
	p := 5
	X := l.matrix_new(f64, n, p, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		for j in 0 ..< p {
			X.data[i * p + j] = rand.float64_normal(0, 1)
		}
		y[i] = 2.0 * X.data[i * p + 0] + 3.0 * X.data[i * p + 1] + 0.1 * rand.float64_normal(0, 1)
	}

	params := ml.RFParams {
		n_trees      = 10,
		max_features = 0, // auto = sqrt(p)
		min_samples  = 5,
		max_depth    = 6,
		bootstrap    = true,
	}

	forest := ml.rf_fit(&X, y, params, allocator)
	defer ml.rf_free(&forest)

	// Predict on training data
	preds := ml.rf_predict(&forest, &X, allocator)

	fmt.println("Random Forest Predictions: %v", preds)
	defer delete(preds, allocator)

	// Compute MSE
	mse := 0.0
	for i in 0 ..< n {
		err := y[i] - preds[i]
		mse += err * err
	}
	mse /= f64(n)

	fmt.printf("Random Forest Test MSE: %.4f\n", mse)

	l.matrix_free(&X)
	delete(y, allocator)
}
