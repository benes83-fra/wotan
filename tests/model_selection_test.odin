
package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"


model_selection_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Model Selection Module ===")

	// Generate data for Gaussian NB (3 classes, 4 features)
	n := 300
	X := l.matrix_new(f64, n, 4, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		class_idx := i / 100 // 0, 1, or 2
		y[i] = f64(class_idx)
		for f in 0 ..< 4 {
			center := f64(class_idx) * 5.0 + f64(f)
			X.data[i * 4 + f] = center + rand.float64_normal(0, 1.0)
		}
	}

	// ---------------------------------------------------------
	// 1. Test Train/Test Split
	// ---------------------------------------------------------
	split := ml.train_test_split(&X, y, 0.2, true, 42, allocator)
	defer ml.train_test_split_free(&split, allocator)

	fmt.printf(
		"Train/Test Split: %v train samples, %v test samples\n",
		split.X_train.rows,
		split.X_test.rows,
	)

	// Fit on train, predict on test
	gnb_model := ml.gnb_fit(&split.X_train, split.y_train, 1e-9, allocator)
	defer ml.gnb_free(&gnb_model)

	preds := ml.gnb_predict(&gnb_model, &split.X_test, allocator)
	defer delete(preds, allocator)

	acc := ml.metrics_accuracy(split.y_test, preds)
	fmt.printf("Holdout Test Accuracy: %.2f%%\n", acc * 100)


	// ---------------------------------------------------------
	// 2. Test K-Fold Cross Validation
	// ---------------------------------------------------------

	// Define a custom evaluator for Gaussian NB
	// user_data is cast to ^f64 to pass the epsilon parameter
	// Define a custom evaluator for Gaussian NB
	gnb_evaluator :: proc(
		X: ^l.Matrix(f64),
		y: []f64,
		train_idx: []int,
		val_idx: []int,
		user_data: rawptr,
		allocator: mem.Allocator,
	) -> f64 {
		// ✅ FIX 1: Dereference rawptr safely using a temporary variable
		eps_ptr := cast(^f64)user_data
		epsilon := eps_ptr^

		// Subset data for this fold
		X_train := ml.subset_matrix(X, train_idx, allocator)
		defer l.matrix_free(&X_train)
		y_train := ml.subset_slice(y, train_idx, allocator)
		defer delete(y_train, allocator)

		// ✅ FIX 2: Fix the typo 'X_valÕÉä' -> 'X_val'
		X_val := ml.subset_matrix(X, val_idx, allocator)
		defer l.matrix_free(&X_val)
		y_val := ml.subset_slice(y, val_idx, allocator)
		defer delete(y_val, allocator)

		// Fit and predict
		model := ml.gnb_fit(&X_train, y_train, epsilon, allocator)
		defer ml.gnb_free(&model)

		preds := ml.gnb_predict(&model, &X_val, allocator)
		defer delete(preds, allocator)

		return ml.metrics_accuracy(y_val, preds)
	}

	epsilon_val := 1e-9
	scores := ml.cross_val_score(
		&X,
		y,
		n_splits = 5,
		evaluator = gnb_evaluator,
		user_data = &epsilon_val,
		shuffle = true,
		random_state = 42,
		allocator = allocator,
	)
	defer delete(scores, allocator)

	// Compute mean and std of scores
	mean_score := 0.0
	for s in scores {mean_score += s}
	mean_score /= f64(len(scores))

	fmt.printf("5-Fold CV Accuracy: %.2f%% +/- ", mean_score * 100)

	// (Optional: compute std dev here if desired, keeping it simple for now)
	fmt.printf(
		" (Scores: %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%)\n",
		scores[0] * 100,
		scores[1] * 100,
		scores[2] * 100,
		scores[3] * 100,
		scores[4] * 100,
	)

	l.matrix_free(&X)
	delete(y, allocator)
}
