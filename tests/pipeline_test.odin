package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"


pipeline_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing ML Pipeline ===")

	// 1. Generate data with vastly different scales
	n := 200
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		class_idx := i / 100
		y[i] = f64(class_idx)

		// Feature 0: Small scale (0 to 1)
		X.data[i * 2 + 0] = f64(class_idx) * 0.5 + rand.float64_normal(0, 0.1)

		// Feature 1: Massive scale (0 to 1000)
		X.data[i * 2 + 1] = f64(class_idx) * 500.0 + rand.float64_normal(0, 50.0)
	}

	// 2. Train/Test Split
	split := ml.train_test_split(&X, y, 0.2, true, 42, allocator)
	defer ml.train_test_split_free(&split, allocator)

	// 3. Build Pipeline: StandardScaler -> KNN
	pipe := ml.pipeline_new(allocator)
	defer ml.pipeline_free(&pipe)

	ml.pipeline_add_standard_scaler(&pipe)

	knn_params := ml.KNNParams {
		k       = 5,
		weights = .Uniform,
	}
	ml.pipeline_set_knn(&pipe, knn_params)

	// 4. Fit ONLY on training data!
	// The scaler learns the mean/std of X_train, completely blind to X_test.
	ml.pipeline_fit(&pipe, &split.X_train, split.y_train)

	// 5. Predict on test data
	// The scaler applies the training mean/std to the test data.
	preds := ml.pipeline_predict(&pipe, &split.X_test, allocator)
	defer delete(preds, allocator)

	acc := ml.metrics_accuracy(split.y_test, preds)
	fmt.printf("Pipeline (Scaler -> KNN) Test Accuracy: %.2f%%\n", acc * 100)

	// Cleanup
	l.matrix_free(&X)
	delete(y, allocator)
}
