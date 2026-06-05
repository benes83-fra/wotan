package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import optim "../wotan/optimize"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
grid_search_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Generic Grid Search ===")

	// Generate data
	n := 200
	X := l.matrix_new(f64, n, 4, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		class_idx := i / 100
		y[i] = f64(class_idx)
		for f in 0 ..< 4 {
			center := f64(class_idx) * 5.0 + f64(f)
			X.data[i * 4 + f] = center + rand.float64_normal(0, 1.5)
		}
	}

	// ---------------------------------------------------------
	// 1. Logistic Regression (Using the clean wrapper)
	// ---------------------------------------------------------
	log_grid := []ml.LogisticParams {
		{
			C = 0.1,
			learning_rate = 0.01,
			optimizer_type = .LBFGS,
			max_iter = 100,
			tol = 1e-4,
			fit_intercept = true,
		},
		{
			C = 1.0,
			learning_rate = 0.01,
			optimizer_type = .LBFGS,
			max_iter = 100,
			tol = 1e-4,
			fit_intercept = true,
		},
		{
			C = 10.0,
			learning_rate = 0.01,
			optimizer_type = .LBFGS,
			max_iter = 100,
			tol = 1e-4,
			fit_intercept = true,
		},
		{
			C = 1.0,
			learning_rate = 0.1,
			optimizer_type = .Adam,
			max_iter = 100,
			tol = 1e-4,
			fit_intercept = true,
		},
	}

	// ✅ Call the wrapper directly! No need to know about evaluators.
	best_log_params, best_log_score := ml.grid_search_logistic(&X, y, 3, log_grid, allocator)

	fmt.printf("Logistic Best CV Accuracy: %.2f%%\n", best_log_score * 100)
	fmt.printf(
		"Logistic Best Params: C=%v, LR=%v, Opt=%v\n\n",
		best_log_params.C,
		best_log_params.learning_rate,
		best_log_params.optimizer_type,
	)


	// ---------------------------------------------------------
	// 2. KNN (Using the clean wrapper)
	// ---------------------------------------------------------
	knn_grid := []ml.KNNParams {
		{k = 3, weights = .Uniform},
		{k = 5, weights = .Uniform},
		{k = 5, weights = .Distance},
		{k = 7, weights = .Distance},
	}

	// ✅ Call the wrapper directly!
	best_knn_params, best_knn_score := ml.grid_search_knn(&X, y, 3, knn_grid, allocator)

	fmt.printf("KNN Best CV Accuracy: %.2f%%\n", best_knn_score * 100)
	fmt.printf("KNN Best Params: k=%v, Weights=%v\n\n", best_knn_params.k, best_knn_params.weights)

	// Cleanup
	l.matrix_free(&X)
	delete(y, allocator)
}
