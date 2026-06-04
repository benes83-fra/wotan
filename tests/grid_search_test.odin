package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import optim "../wotan/optimize"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

grid_search_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Grid Search ===")

	// ---------------------------------------------------------
	// 1. Grid Search for Logistic Regression
	// ---------------------------------------------------------
	n := 200
	X := l.matrix_new(f64, n, 4, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		class_idx := i / 100
		y[i] = f64(class_idx)
		for f in 0 ..< 4 {
			center := f64(class_idx) * 5.0 + f64(f)
			X.data[i * 4 + f] = center + rand.float64_normal(0, 1.5) // Add some noise
		}
	}

	// Define the grid of hyperparameters to search
	log_grid := ml.LogisticGrid {
		C_values             = []f64{0.1, 1.0, 10.0},
		learning_rate_values = []f64{0.01, 0.1},
		optimizer_types      = []optim.OptimizerType{.LBFGS},
	}

	// Base params that remain constant across the grid search
	base_log_params := ml.LogisticParams {
		max_iter      = 100,
		tol           = 1e-4,
		fit_intercept = true,
	}

	fmt.println("Searching Logistic Regression hyperparameters...")
	log_res := ml.grid_search_logistic(&X, y, log_grid, 3, base_log_params, allocator)

	fmt.printf("✅ Best CV Accuracy: %.2f%%\n", log_res.best_score * 100)
	fmt.printf(
		"✅ Best Params: C=%.2f, LR=%.2f, Optimizer=%v\n\n",
		log_res.best_params.C,
		log_res.best_params.learning_rate,
		log_res.best_params.optimizer_type,
	)


	// ---------------------------------------------------------
	// 2. Grid Search for SVR
	// ---------------------------------------------------------
	n_reg := 100
	X_reg := l.matrix_new(f64, n_reg, 1, allocator)
	y_reg := make([]f64, n_reg, allocator)

	for i in 0 ..< n_reg {
		x_val := f64(i) / f64(n_reg) * 4.0 * math.PI
		X_reg.data[i] = x_val
		y_reg[i] = math.sin(x_val) + rand.float64_normal(0, 0.2)
	}

	svr_grid := ml.SVRGrid {
		C_values       = []f64{1.0, 10.0},
		epsilon_values = []f64{0.1, 0.5},
		gamma_values   = []f64{0.1, 1.0},
	}

	base_svr_params := ml.SVRParams {
		kernel_type    = .RBF,
		max_iter       = 50,
		tol            = 1e-3,
		optimizer_type = .LBFGS,
	}

	fmt.println("Searching SVR hyperparameters...")
	svr_res := ml.grid_search_svr(&X_reg, y_reg, svr_grid, 3, base_svr_params, allocator)

	fmt.printf("✅ Best CV R2: %.4f\n", svr_res.best_score)
	fmt.printf(
		"✅ Best Params: C=%.2f, Epsilon=%.2f, Gamma=%.2f\n",
		svr_res.best_params.C,
		svr_res.best_params.epsilon,
		svr_res.best_params.gamma,
	)

	// Cleanup
	l.matrix_free(&X)
	delete(y, allocator)
	l.matrix_free(&X_reg)
	delete(y_reg, allocator)
}
