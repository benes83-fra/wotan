package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
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

gb_test :: proc(allocator: mem.Allocator) {
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

	params := ml.GBParams {
		n_estimators  = 50,
		learning_rate = 0.1,
		max_depth     = 4, // Shallow trees for boosting
		min_samples   = 5,
		subsample     = 1.0, // Full data (set <1.0 for stochastic GB)
	}

	model := ml.gb_fit(&X, y, params, allocator)
	defer ml.gb_free(&model)

	// Predict on training data
	preds := ml.gb_predict(&model, &X, allocator)
	defer delete(preds, allocator)
	fmt.println("Gradient Boosting Predictions: %v", preds)

	// Compute MSE
	mse := 0.0
	for i in 0 ..< n {
		err := y[i] - preds[i]
		mse += err * err
	}
	mse /= f64(n)

	fmt.printf("Gradient Boosting Test MSE: %.4f\n", mse)

	l.matrix_free(&X)
	delete(y, allocator)
}

vec_sub_simd_test :: proc(allocator: mem.Allocator) {
	n := 100
	a := make([]f64, n, allocator)
	b := make([]f64, n, allocator)
	out := make([]f64, n, allocator)
	defer {
		delete(a, allocator)
		delete(b, allocator)
		delete(out, allocator)
	}

	// Fill with random values
	for i in 0 ..< n {
		a[i] = rand.float64_normal(0, 10)
		b[i] = rand.float64_normal(0, 10)
	}

	// Compute with SIMD
	l.vec_sub_simd(a, b, out)

	// Verify against scalar version
	max_err := 0.0
	for i in 0 ..< n {
		expected := a[i] - b[i]
		err := math.abs(out[i] - expected)
		if err > max_err {max_err = err}
	}

	fmt.printf("vec_sub_simd max error: %.2e\n", max_err)
	assert(max_err < 1e-10, "SIMD subtraction failed")
}
