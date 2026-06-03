package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"

km_test :: proc(allocator: mem.Allocator) {
	// Generate synthetic clustered data
	n_samples := 300
	n_features := 2
	k := 3

	X := l.matrix_new(f64, n_samples, n_features, allocator)

	// Create 3 clusters
	for i in 0 ..< n_samples {
		cluster := i / (n_samples / k)
		mean_x := f64(cluster * 5 - 5)
		mean_y := f64(cluster * 3 - 3)
		X.data[i * n_features + 0] = mean_x + rand.float64_normal(0, 0.5)
		X.data[i * n_features + 1] = mean_y + rand.float64_normal(0, 0.5)
	}

	params := ml.KMParams {
		n_clusters   = k,
		max_iter     = 100,
		tol          = 1e-4,
		init         = .KMeansPlusPlus,
		n_init       = 50,
		random_state = 42,
	}

	model := ml.km_fit(&X, params, allocator)
	defer ml.km_free(&model)

	fmt.printf("K-Means converged: %v after %v iterations\n", model.converged, model.n_iter)
	fmt.printf("Final inertia: %.4f\n", model.inertia)

	// Compute and print stats
	stats := ml.km_compute_stats(&model, &X, allocator)
	defer ml.km_stats_free(&stats, allocator)
	ml.km_stats_print(&stats)

	// Predict on new data
	new_n := 10
	X_new := l.matrix_new(f64, new_n, n_features, allocator)
	for i in 0 ..< new_n {
		X_new.data[i * n_features + 0] = rand.float64_normal(0, 3)
		X_new.data[i * n_features + 1] = rand.float64_normal(0, 3)
	}

	preds := ml.km_predict(&model, &X_new, allocator)
	defer delete(preds, allocator)

	fmt.println("\nPredictions on new data:")
	for i in 0 ..< new_n {
		fmt.printf("  Point %2v → Cluster %v\n", i, preds[i])
	}

	l.matrix_free(&X)
	l.matrix_free(&X_new)
}
knn_test :: proc(allocator: mem.Allocator) {
	// Generate 3 distinct clusters (Classes: 0.0, 1.0, 2.0)
	n := 150
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		class_idx := i / 50 // 0, 1, or 2
		y[i] = f64(class_idx)

		center_x := f64(class_idx) * 5.0
		center_y := f64(class_idx) * 5.0

		X.data[i * 2 + 0] = center_x + rand.float64_normal(0, 0.5)
		X.data[i * 2 + 1] = center_y + rand.float64_normal(0, 0.5)
	}

	// 1. Test Uniform Weighting
	params := ml.KNNParams {
		k       = 5,
		weights = .Uniform,
	}
	model := ml.knn_fit(&X, y, params, allocator)
	defer ml.knn_free(&model)

	preds := ml.knn_predict(&model, &X, allocator)
	defer delete(preds, allocator)

	correct := 0
	for i in 0 ..< n {if preds[i] == y[i] {correct += 1}}
	fmt.printf("KNN (Uniform) Accuracy: %.2f%%\n", f64(correct) / f64(n) * 100)

	// 2. Test Distance Weighting
	params_dist := ml.KNNParams {
		k       = 5,
		weights = .Distance,
	}
	model_dist := ml.knn_fit(&X, y, params_dist, allocator)
	defer ml.knn_free(&model_dist)

	preds_dist := ml.knn_predict(&model_dist, &X, allocator)
	defer delete(preds_dist, allocator)

	correct = 0
	for i in 0 ..< n {if preds_dist[i] == y[i] {correct += 1}}
	fmt.printf("KNN (Distance) Accuracy: %.2f%%\n", f64(correct) / f64(n) * 100)

	l.matrix_free(&X)
	delete(y, allocator)
}
