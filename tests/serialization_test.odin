package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

serialization_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Model Serialization ===")

	// 1. Generate Data
	n := 100
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)
	for i in 0 ..< n {
		X.data[i * 2 + 0] = rand.float64_normal(0, 1)
		X.data[i * 2 + 1] = rand.float64_normal(0, 1)
		y[i] = f64(i / 50) // 2 classes
	}

	// 2. Train a Pipeline
	pipe := ml.pipeline_new(allocator)
	defer ml.pipeline_free(&pipe)

	ml.pipeline_add_standard_scaler(&pipe)
	ml.pipeline_set_kernel_svm(
		&pipe,
		{
			C = 1.0,
			gamma = 0.5,
			kernel_type = .RBF,
			max_iter = 100,
			learning_rate = 1.0,
			optimizer_type = .LBFGS,
		},
	)
	ml.pipeline_fit(&pipe, &X, y)

	// 3. Predict BEFORE saving
	preds_before := ml.pipeline_predict(&pipe, &X, allocator)
	defer delete(preds_before, allocator)

	// 4. Save to disk
	ok := ml.pipeline_save(&pipe, "test_model.wotan")
	fmt.printf("Saved model to disk: %v\n", ok)

	// 5. Load from disk
	loaded_pipe, load_ok := ml.pipeline_load("test_model.wotan", allocator)
	defer ml.pipeline_free(&loaded_pipe)
	fmt.printf("Loaded model from disk: %v\n", load_ok)

	// 6. Predict AFTER loading
	preds_after := ml.pipeline_predict(&loaded_pipe, &X, allocator)
	defer delete(preds_after, allocator)

	// 7. Compare predictions
	match := true
	for i in 0 ..< n {
		if preds_before[i] != preds_after[i] {
			match = false
			break
		}
	}
	fmt.printf("Predictions match exactly: %v\n", match)

	// Cleanup
	l.matrix_free(&X)
	delete(y, allocator)
	os.remove("test_model.wotan")
}

tree_serialization_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Tree Serialization ===")

	// 1. Generate a simple non-linear dataset
	n := 100
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		x0 := rand.float64_normal(0, 1)
		x1 := rand.float64_normal(0, 1)
		X.data[i * 2 + 0] = x0
		X.data[i * 2 + 1] = x1

		// Non-linear rule (a circle)
		if x0 * x0 + x1 * x1 < 1.0 {
			y[i] = 1.0
		} else {
			y[i] = 0.0
		}
	}

	// 2. Build and Fit Pipeline with a Decision Tree
	pipe := ml.pipeline_new(allocator)
	defer ml.pipeline_free(&pipe)

	ml.pipeline_add_standard_scaler(&pipe)
	ml.pipeline_set_decision_tree(&pipe, {max_depth = 5, min_samples = 2})

	ml.pipeline_fit(&pipe, &X, y)

	// 3. Predict BEFORE saving
	preds_before := ml.pipeline_predict(&pipe, &X, allocator)
	defer delete(preds_before, allocator)

	// 4. Save to disk
	ok := ml.pipeline_save(&pipe, "test_tree.wotan")
	fmt.printf("Saved tree pipeline to disk: %v\n", ok)

	// 5. Load from disk
	loaded_pipe, load_ok := ml.pipeline_load("test_tree.wotan", allocator)
	defer ml.pipeline_free(&loaded_pipe)
	fmt.printf("Loaded tree pipeline from disk: %v\n", load_ok)

	// 6. Predict AFTER loading
	preds_after := ml.pipeline_predict(&loaded_pipe, &X, allocator)
	defer delete(preds_after, allocator)

	// 7. Compare predictions
	match := true
	for i in 0 ..< n {
		if preds_before[i] != preds_after[i] {
			match = false
			fmt.printf("Mismatch at index %v: %v vs %v\n", i, preds_before[i], preds_after[i])
			break
		}
	}
	fmt.printf("Predictions match exactly: %v\n", match)

	// 8. Cleanup
	l.matrix_free(&X)
	delete(y, allocator)
	os.remove("test_tree.wotan")
}
