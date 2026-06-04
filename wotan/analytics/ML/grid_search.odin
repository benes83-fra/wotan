package ML

import l "../../linalg"
import optim "../../optimize"
import "core:math"
import "core:mem"

// ============================================================================
// Internal Evaluators for Grid Search
// These wrap the model fitting and prediction, returning the metric score.
// ============================================================================

// Evaluator for Logistic Regression (Optimizes Accuracy)
_logistic_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^LogisticParams)user_data

	X_train := subset_matrix(X, train_idx, allocator)
	defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator)
	defer delete(y_train, allocator)

	X_val := subset_matrix(X, val_idx, allocator)
	defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator)
	defer delete(y_val, allocator)

	model := logistic_fit(&X_train, y_train, params^, allocator)
	defer logistic_free(&model)

	preds := logistic_predict(&model, &X_val, allocator)
	defer delete(preds, allocator)

	return metrics_accuracy(y_val, preds)
}

// Evaluator for SVR (Optimizes R2)
_svr_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^SVRParams)user_data

	X_train := subset_matrix(X, train_idx, allocator)
	defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator)
	defer delete(y_train, allocator)

	X_val := subset_matrix(X, val_idx, allocator)
	defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator)
	defer delete(y_val, allocator)

	model := svr_fit(&X_train, y_train, params^, allocator)
	defer svr_free(&model)

	preds := svr_predict(&model, &X_val, allocator)
	defer delete(preds, allocator)

	return metrics_r2(y_val, preds)
}

// ============================================================================
// Grid Search Structures
// ============================================================================

LogisticGrid :: struct {
	C_values:             []f64,
	learning_rate_values: []f64,
	optimizer_types:      []optim.OptimizerType,
}

LogisticGridResult :: struct {
	best_params: LogisticParams,
	best_score:  f64,
	allocator:   mem.Allocator,
}

SVRGrid :: struct {
	C_values:       []f64,
	epsilon_values: []f64,
	gamma_values:   []f64,
}

SVRGridResult :: struct {
	best_params: SVRParams,
	best_score:  f64,
	allocator:   mem.Allocator,
}

// ============================================================================
// Public API: Grid Search for Logistic Regression
// ============================================================================

grid_search_logistic :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	grid: LogisticGrid,
	n_splits: int,
	base_params: LogisticParams, // Base params to copy defaults from (max_iter, tol, etc.)
	allocator: mem.Allocator = context.allocator,
) -> LogisticGridResult {

	res: LogisticGridResult
	res.allocator = allocator
	res.best_score = -math.F64_MAX

	// If grid arrays are empty, safely fallback to the base param value using stack arrays
	c_vals := grid.C_values
	if len(c_vals) == 0 {
		temp := [1]f64{base_params.C}
		c_vals = temp[:]
	}

	lr_vals := grid.learning_rate_values
	if len(lr_vals) == 0 {
		temp := [1]f64{base_params.learning_rate}
		lr_vals = temp[:]
	}

	opt_vals := grid.optimizer_types
	if len(opt_vals) == 0 {
		temp := [1]optim.OptimizerType{base_params.optimizer_type}
		opt_vals = temp[:]
	}

	// Exhaustive search over all combinations
	for c_val in c_vals {
		for lr_val in lr_vals {
			for opt_type in opt_vals {

				params := base_params
				params.C = c_val
				params.learning_rate = lr_val
				params.optimizer_type = opt_type

				// Run K-Fold Cross Validation
				scores := cross_val_score(
					X,
					y,
					n_splits,
					_logistic_cv_evaluator,
					&params,
					true,
					42,
					allocator,
				)

				// Compute mean score
				mean_score := 0.0
				for s in scores {mean_score += s}
				mean_score /= f64(len(scores))

				// Update best if this combination is better
				if mean_score > res.best_score {
					res.best_score = mean_score
					res.best_params = params
				}

				delete(scores, allocator)
			}
		}
	}

	return res
}

// ============================================================================
// Public API: Grid Search for SVR
// ============================================================================

grid_search_svr :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	grid: SVRGrid,
	n_splits: int,
	base_params: SVRParams,
	allocator: mem.Allocator = context.allocator,
) -> SVRGridResult {

	res: SVRGridResult
	res.allocator = allocator
	res.best_score = -math.F64_MAX

	c_vals := grid.C_values
	if len(c_vals) == 0 {
		temp := [1]f64{base_params.C}
		c_vals = temp[:]
	}

	eps_vals := grid.epsilon_values
	if len(eps_vals) == 0 {
		temp := [1]f64{base_params.epsilon}
		eps_vals = temp[:]
	}

	gamma_vals := grid.gamma_values
	if len(gamma_vals) == 0 {
		temp := [1]f64{base_params.gamma}
		gamma_vals = temp[:]
	}

	for c_val in c_vals {
		for eps_val in eps_vals {
			for gamma_val in gamma_vals {

				params := base_params
				params.C = c_val
				params.epsilon = eps_val
				params.gamma = gamma_val

				scores := cross_val_score(
					X,
					y,
					n_splits,
					_svr_cv_evaluator,
					&params,
					true,
					42,
					allocator,
				)

				mean_score := 0.0
				for s in scores {mean_score += s}
				mean_score /= f64(len(scores))

				if mean_score > res.best_score {
					res.best_score = mean_score
					res.best_params = params
				}

				delete(scores, allocator)
			}
		}
	}

	return res
}
