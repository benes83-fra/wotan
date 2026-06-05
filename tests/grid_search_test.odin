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

extended_grid_search_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Extended Grid Search Tests (Full Library) ===")

	// ---------------------------------------------------------
	// 1. Regression Dataset (for Ridge, Lasso, SVR)
	// ---------------------------------------------------------
	n_reg := 150
	X_reg := l.matrix_new(f64, n_reg, 5, allocator)
	y_reg := make([]f64, n_reg, allocator)

	for i in 0 ..< n_reg {
		x0 := rand.float64_normal(0, 1)
		x1 := rand.float64_normal(0, 1)
		X_reg.data[i * 5 + 0] = x0
		X_reg.data[i * 5 + 1] = x1
		X_reg.data[i * 5 + 2] = rand.float64_normal(0, 1) // noise
		X_reg.data[i * 5 + 3] = rand.float64_normal(0, 1) // noise
		X_reg.data[i * 5 + 4] = rand.float64_normal(0, 1) // noise

		// True signal only depends on x0 and x1
		y_reg[i] = 3.0 * x0 + 1.5 * x1 + rand.float64_normal(0, 0.5)
	}

	// --- Ridge Regression ---
	ridge_grid := []ml.RidgeParams {
		{lambda = 0.01, method = .Cholesky},
		{lambda = 0.1, method = .Cholesky},
		{lambda = 1.0, method = .Cholesky},
	}
	best_ridge, ridge_score := ml.grid_search_ridge(&X_reg, y_reg, 3, ridge_grid, allocator)
	fmt.printf("Ridge Best CV R2: %.4f (Lambda: %v)\n", ridge_score, best_ridge.lambda)

	// --- Lasso Regression ---
	lasso_grid := []ml.LassoParams {
		{lambda = 0.01, max_iter = 1000, tol = 1e-4},
		{lambda = 0.1, max_iter = 1000, tol = 1e-4},
		{lambda = 1.0, max_iter = 1000, tol = 1e-4},
	}
	best_lasso, lasso_score := ml.grid_search_lasso(&X_reg, y_reg, 3, lasso_grid, allocator)
	fmt.printf("Lasso Best CV R2: %.4f (Lambda: %v)\n", lasso_score, best_lasso.lambda)

	// --- Support Vector Regression (SVR) ---
	svr_grid := []ml.SVRParams {
		{
			C = 1.0,
			epsilon = 0.1,
			gamma = 0.5,
			kernel_type = .RBF,
			max_iter = 50,
			tol = 1e-3,
			learning_rate = 0.01,
			optimizer_type = .LBFGS,
		},
		{
			C = 10.0,
			epsilon = 0.1,
			gamma = 1.0,
			kernel_type = .RBF,
			max_iter = 50,
			tol = 1e-3,
			learning_rate = 0.01,
			optimizer_type = .LBFGS,
		},
	}
	best_svr, svr_score := ml.grid_search_svr(&X_reg, y_reg, 3, svr_grid, allocator)
	fmt.printf("SVR Best CV R2: %.4f (C: %v, Gamma: %v)\n", svr_score, best_svr.C, best_svr.gamma)


	// ---------------------------------------------------------
	// 2. Classification Dataset (for SVMs, Trees, Ensembles)
	// ---------------------------------------------------------
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

	// --- Linear SVM ---
	lsvm_grid := []ml.SVMParams {
		{
			C = 0.1,
			max_iter = 100,
			tol = 1e-3,
			learning_rate = 0.01,
			fit_intercept = true,
			optimizer_type = .LBFGS,
		},
		{
			C = 1.0,
			max_iter = 100,
			tol = 1e-3,
			learning_rate = 0.01,
			fit_intercept = true,
			optimizer_type = .LBFGS,
		},
		{
			C = 10.0,
			max_iter = 100,
			tol = 1e-3,
			learning_rate = 0.01,
			fit_intercept = true,
			optimizer_type = .LBFGS,
		},
	}
	best_lsvm, lsvm_score := ml.grid_search_linear_svm(&X_cls, y_cls, 3, lsvm_grid, allocator)
	fmt.printf("Linear SVM Best CV Acc: %.2f%% (C: %v)\n", lsvm_score * 100, best_lsvm.C)

	// --- Kernel SVM ---
	ksvm_grid := []ml.KernelSVMParams {
		{
			C = 1.0,
			gamma = 0.1,
			kernel_type = .RBF,
			max_iter = 100,
			tol = 1e-3,
			learning_rate = 0.01,
			optimizer_type = .LBFGS,
		},
		{
			C = 10.0,
			gamma = 1.0,
			kernel_type = .RBF,
			max_iter = 100,
			tol = 1e-3,
			learning_rate = 0.01,
			optimizer_type = .LBFGS,
		},
	}
	best_ksvm, ksvm_score := ml.grid_search_kernel_svm(&X_cls, y_cls, 3, ksvm_grid, allocator)
	fmt.printf(
		"Kernel SVM Best CV Acc: %.2f%% (C: %v, Gamma: %v)\n",
		ksvm_score * 100,
		best_ksvm.C,
		best_ksvm.gamma,
	)

	// --- Decision Tree ---
	dt_grid := []ml.TreeParams {
		{max_depth = 3, min_samples = 2},
		{max_depth = 5, min_samples = 2},
		{max_depth = 10, min_samples = 2},
	}
	best_dt, dt_score := ml.grid_search_decision_tree(&X_cls, y_cls, 3, dt_grid, allocator)
	fmt.printf(
		"Decision Tree Best CV Acc: %.2f%% (Depth: %v)\n",
		dt_score * 100,
		best_dt.max_depth,
	)

	// --- Random Forest ---
	rf_grid := []ml.RFParams {
		{n_trees = 10, max_depth = 5, min_samples = 2, bootstrap = true},
		{n_trees = 50, max_depth = 10, min_samples = 2, bootstrap = true},
	}
	best_rf, rf_score := ml.grid_search_random_forest(&X_cls, y_cls, 3, rf_grid, allocator)
	fmt.printf(
		"Random Forest Best CV Acc: %.2f%% (Trees: %v, Depth: %v)\n",
		rf_score * 100,
		best_rf.n_trees,
		best_rf.max_depth,
	)

	// --- Gradient Boosting ---
	// Note: Adjust field names (n_estimators, learning_rate, max_depth, min_samples) if your GBParams struct differs
	gb_grid := []ml.GBParams {
		{n_estimators = 50, learning_rate = 0.1, max_depth = 3, min_samples = 2},
		{n_estimators = 100, learning_rate = 0.05, max_depth = 5, min_samples = 2},
	}
	best_gb, gb_score := ml.grid_search_gradient_boosting(&X_cls, y_cls, 3, gb_grid, allocator)
	fmt.printf(
		"Gradient Boosting Best CV Acc: %.2f%% (Estimators: %v, LR: %v)\n",
		gb_score * 100,
		best_gb.n_estimators,
		best_gb.learning_rate,
	)

	// Cleanup
	l.matrix_free(&X_reg)
	delete(y_reg, allocator)
	l.matrix_free(&X_cls)
	delete(y_cls, allocator)
}
