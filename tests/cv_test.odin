package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"

ridge_cv_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== RIDGE CV TEST ===")

	// Generate synthetic data: y = 3 + 2*x + noise
	n := 50
	X := l.matrix_new(f64, n, 2, allocator)
	defer l.matrix_free(&X)

	for i in 0 ..< n {
		X.data[i * 2 + 0] = 1.0 // intercept
		X.data[i * 2 + 1] = f64(i) / 10.0 // x = 0, 0.1, ..., 4.9
	}

	y := make([]f64, n, allocator)
	defer delete(y, allocator)
	for i in 0 ..< n {
		// True model: y = 3 + 2*x + small noise
		noise := 0.1 * (f64(i) - 25.0) / 25.0 // deterministic "noise"
		y[i] = 3.0 + 2.0 * X.data[i * 2 + 1] + noise
	}

	// Search over lambda values
	lambdas := []f64{0.0, 0.1, 1.0, 10.0}

	best_lambda, best_result, cv_errors := ml.ridge_cv(&X, y, lambdas, 5, 42, .Cholesky, allocator)
	// defer _ols_result_free(&best_result, allocator)
	defer delete(cv_errors, allocator)

	fmt.printf("Best lambda: %f\n", best_lambda)
	fmt.printf("CV errors: %v\n", cv_errors)
	fmt.printf("Final beta: %v\n", best_result.beta)

	// With low noise, best lambda should be small (close to OLS)
	assert(best_lambda <= 1.0, "Ridge CV: best lambda should be small for low-noise data")
	assert_close(best_result.beta[0], 3.0, 0.5, "Ridge CV: intercept")
	assert_close(best_result.beta[1], 2.0, 0.5, "Ridge CV: slope")

	fmt.println("Ridge CV test OK ✅")
}

lasso_cv_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LASSO CV TEST ===")

	// Generate data with 2 signal + 2 noise features
	n := 100
	p := 4
	X := l.matrix_new(f64, n, p, allocator)
	defer l.matrix_free(&X)

	for i in 0 ..< n {
		X.data[i * p + 0] = 1.0 // intercept
		X.data[i * p + 1] = f64(i) / 20.0 // signal feature
		X.data[i * p + 2] = f64(i % 7) / 7.0 // noise feature 1
		X.data[i * p + 3] = f64(i % 11) / 11.0 // noise feature 2
	}

	y := make([]f64, n, allocator)
	defer delete(y, allocator)
	for i in 0 ..< n {
		// True model: only intercept + feature 1 matter
		noise := 0.05 * (f64(i) - 50.0) / 50.0
		y[i] = 3.0 + 2.0 * X.data[i * p + 1] + noise
	}

	// Search over lambda values (log-spaced)
	lambdas := []f64{0.0, 0.01, 0.1, 1.0, 10.0}

	best_lambda, best_result, cv_errors := ml.lasso_cv(
		&X,
		y,
		lambdas,
		5,
		42,
		2000,
		1e-4,
		allocator,
	)
	// defer _ols_result_free(&best_result, allocator)
	defer delete(cv_errors, allocator)

	fmt.printf("Best lambda: %f\n", best_lambda)
	fmt.printf("CV errors: %v\n", cv_errors)
	fmt.printf("Final beta: %v\n", best_result.beta)

	// Lasso should shrink noise features toward zero
	assert(
		best_result.beta[2] == 0.0 || math.abs(best_result.beta[2]) < 0.1,
		"Lasso CV: noise feature 1 should be zero/small",
	)
	assert(
		best_result.beta[3] == 0.0 || math.abs(best_result.beta[3]) < 0.1,
		"Lasso CV: noise feature 2 should be zero/small",
	)
	assert_close(best_result.beta[0], 3.0, 0.5, "Lasso CV: intercept")
	assert_close(best_result.beta[1], 2.0, 0.5, "Lasso CV: signal slope")

	fmt.println("Lasso CV test OK ✅")
}
