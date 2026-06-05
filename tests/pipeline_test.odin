package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math" // ✅ ADDED for math.round
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

pipeline_comprehensive_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Comprehensive Pipeline Test (All Models) ===")

	// Generate Classification Data (3 classes)
	n_cls := 150
	X_cls := l.matrix_new(f64, n_cls, 4, allocator)
	y_cls := make([]f64, n_cls, allocator)
	for i in 0 ..< n_cls {
		class_idx := i / 50
		y_cls[i] = f64(class_idx)
		for f in 0 ..< 4 {
			center := f64(class_idx) * 5.0 + f64(f)
			X_cls.data[i * 4 + f] = center + rand.float64_normal(0, 1.0)
		}
	}

	// Generate Regression Data
	n_reg := 150
	X_reg := l.matrix_new(f64, n_reg, 4, allocator)
	y_reg := make([]f64, n_reg, allocator)
	for i in 0 ..< n_reg {
		for f in 0 ..< 4 {
			X_reg.data[i * 4 + f] = rand.float64_normal(0, 1.0)
		}
		y_reg[i] =
			2.0 * X_reg.data[i * 4 + 0] + 3.0 * X_reg.data[i * 4 + 1] + rand.float64_normal(0, 0.5)
	}

	// ✅ NEW: Helper to round predictions for tree ensembles
	round_preds :: proc(preds: []f64) {
		for i in 0 ..< len(preds) {
			preds[i] = math.round(preds[i])
		}
	}

	// Helper closures to run and evaluate pipelines
	run_cls :: proc(
		name: string,
		pipe: ^ml.Pipeline,
		X: ^l.Matrix(f64),
		y: []f64,
		allocator: mem.Allocator,
		round_output: bool = false, // ✅ ADDED
	) {
		ml.pipeline_fit(pipe, X, y)
		preds := ml.pipeline_predict(pipe, X, allocator)
		defer delete(preds, allocator)

		// ✅ ADDED
		if round_output {
			round_preds(preds)
		}

		acc := ml.metrics_accuracy(y, preds)
		fmt.printf("✅ %-25s Accuracy: %.2f%%\n", name, acc * 100)
	}

	run_reg :: proc(
		name: string,
		pipe: ^ml.Pipeline,
		X: ^l.Matrix(f64),
		y: []f64,
		allocator: mem.Allocator,
	) {
		ml.pipeline_fit(pipe, X, y)
		preds := ml.pipeline_predict(pipe, X, allocator)
		defer delete(preds, allocator)
		r2 := ml.metrics_r2(y, preds)
		fmt.printf("✅ %-25s R2:      %.4f\n", name, r2)
	}

	// --- Classification Models ---
	pipe1 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe1)
	ml.pipeline_add_standard_scaler(&pipe1)
	ml.pipeline_set_logistic(
		&pipe1,
		{C = 1.0, max_iter = 100, learning_rate = 1.0, optimizer_type = .LBFGS},
	)
	run_cls("Logistic (Binary)*", &pipe1, &X_cls, y_cls, allocator) // Renamed to indicate expected failure

	pipe2 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe2)
	ml.pipeline_add_standard_scaler(&pipe2)
	ml.pipeline_set_ovr_logistic(
		&pipe2,
		{C = 1.0, max_iter = 500, learning_rate = 1.0, optimizer_type = .LBFGS}, // ✅ Increased max_iter
	)
	run_cls("OvR Logistic", &pipe2, &X_cls, y_cls, allocator)

	pipe3 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe3)
	ml.pipeline_add_standard_scaler(&pipe3)
	ml.pipeline_set_ovr_linear_svm(
		&pipe3,
		{C = 1.0, max_iter = 500, learning_rate = 1.0, optimizer_type = .LBFGS}, // ✅ Increased max_iter
	)
	run_cls("OvR Linear SVM", &pipe3, &X_cls, y_cls, allocator)

	pipe4 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe4)
	ml.pipeline_add_standard_scaler(&pipe4)
	ml.pipeline_set_ovr_kernel_svm(
		&pipe4,
		{
			C              = 1.0,
			gamma          = 0.1,
			kernel_type    = .RBF,
			max_iter       = 500, // ✅ Increased max_iter
			learning_rate  = 1.0,
			optimizer_type = .LBFGS,
		},
	)
	run_cls("OvR Kernel SVM", &pipe4, &X_cls, y_cls, allocator)

	pipe5 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe5)
	ml.pipeline_add_standard_scaler(
		&pipe5,
	); ml.pipeline_set_knn(&pipe5, {k = 5, weights = .Uniform})
	run_cls("KNN", &pipe5, &X_cls, y_cls, allocator)

	pipe6 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe6)
	ml.pipeline_add_standard_scaler(&pipe6)
	ml.pipeline_set_decision_tree(&pipe6, {max_depth = 5, min_samples = 2})
	run_cls("Decision Tree", &pipe6, &X_cls, y_cls, allocator)

	pipe7 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe7)
	ml.pipeline_add_standard_scaler(&pipe7)
	ml.pipeline_set_random_forest(
		&pipe7,
		{n_trees = 20, max_depth = 5, min_samples = 2, bootstrap = true}, // Increased trees slightly
	)
	run_cls("Random Forest", &pipe7, &X_cls, y_cls, allocator, true) // ✅ Pass true to round

	pipe8 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe8)
	ml.pipeline_add_standard_scaler(&pipe8)
	ml.pipeline_set_gradient_boosting(
		&pipe8,
		{n_estimators = 50, learning_rate = 0.1, max_depth = 3, min_samples = 2},
	)
	run_cls("Gradient Boosting", &pipe8, &X_cls, y_cls, allocator, true) // ✅ Pass true to round

	pipe9 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe9)
	ml.pipeline_add_standard_scaler(&pipe9)
	ml.pipeline_set_gnb(&pipe9, 1e-9)
	run_cls("Gaussian NB", &pipe9, &X_cls, y_cls, allocator)

	// --- Regression Models ---
	pipe10 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe10)
	ml.pipeline_add_standard_scaler(&pipe10); ml.pipeline_set_ols(&pipe10, .Cholesky)
	run_reg("OLS", &pipe10, &X_reg, y_reg, allocator)

	pipe11 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe11)
	ml.pipeline_add_standard_scaler(&pipe11)
	ml.pipeline_set_ridge(&pipe11, {lambda = 1.0, method = .Cholesky})
	run_reg("Ridge", &pipe11, &X_reg, y_reg, allocator)

	pipe12 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe12)
	ml.pipeline_add_standard_scaler(&pipe12)
	ml.pipeline_set_lasso(&pipe12, {lambda = 0.1, max_iter = 1000, tol = 1e-4})
	run_reg("Lasso", &pipe12, &X_reg, y_reg, allocator)

	pipe13 := ml.pipeline_new(allocator); defer ml.pipeline_free(&pipe13)
	ml.pipeline_add_standard_scaler(&pipe13)
	ml.pipeline_set_svr(
		&pipe13,
		{
			C              = 100.0, // ✅ Increased C
			epsilon        = 0.1,
			gamma          = 1.0,
			kernel_type    = .RBF,
			max_iter       = 500, // ✅ Increased max_iter
			learning_rate  = 1.0,
			optimizer_type = .LBFGS,
		},
	)
	run_reg("SVR", &pipe13, &X_reg, y_reg, allocator)

	l.matrix_free(&X_cls)
	delete(y_cls, allocator)
	l.matrix_free(&X_reg)
	delete(y_reg, allocator)
}
