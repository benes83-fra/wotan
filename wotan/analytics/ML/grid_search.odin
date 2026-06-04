package ML

import l "../../linalg"
import optim "../../optimize"
import "core:math"
import "core:mem"

// ============================================================================
// Internal Evaluators for Grid Search
// ============================================================================

_logistic_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^LogisticParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	model := logistic_fit(&X_train, y_train, params^, allocator); defer logistic_free(&model)
	preds := logistic_predict(&model, &X_val, allocator); defer delete(preds, allocator)
	return metrics_accuracy(y_val, preds)
}

_svr_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^SVRParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	model := svr_fit(&X_train, y_train, params^, allocator); defer svr_free(&model)
	preds := svr_predict(&model, &X_val, allocator); defer delete(preds, allocator)
	return metrics_r2(y_val, preds)
}

// ✅ NEW: KNN Evaluator
_knn_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^KNNParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	model := knn_fit(&X_train, y_train, params^, allocator); defer knn_free(&model)
	preds := knn_predict(&model, &X_val, allocator); defer delete(preds, allocator)
	return metrics_accuracy(y_val, preds)
}

// ✅ NEW: Random Forest Evaluator
_rf_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^RFParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	model := rf_fit(&X_train, y_train, params^, allocator); defer rf_free(&model)
	preds := rf_predict(&model, &X_val, allocator); defer delete(preds, allocator)
	return metrics_accuracy(y_val, preds)
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

// ✅ NEW: KNN Grid
KNNGrid :: struct {
	k_values:     []int,
	weight_types: []KNNWeights,
}

KNNGridResult :: struct {
	best_params: KNNParams,
	best_score:  f64,
	allocator:   mem.Allocator,
}

// ✅ NEW: Random Forest Grid
RFGrid :: struct {
	n_trees_values:     []int,
	max_depth_values:   []int,
	min_samples_values: []int,
}

RFGridResult :: struct {
	best_params: RFParams,
	best_score:  f64,
	allocator:   mem.Allocator,
}

// ============================================================================
// Public API: Grid Search for Specific Models
// ============================================================================

grid_search_logistic :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	grid: LogisticGrid,
	n_splits: int,
	base_params: LogisticParams,
	allocator: mem.Allocator = context.allocator,
) -> LogisticGridResult {
	res: LogisticGridResult; res.allocator = allocator; res.best_score = -math.F64_MAX

	c_vals := grid.C_values; if len(c_vals) == 0 {temp := [1]f64{base_params.C}; c_vals = temp[:]}
	lr_vals :=
		grid.learning_rate_values; if len(lr_vals) == 0 {temp := [1]f64{base_params.learning_rate}; lr_vals = temp[:]}
	opt_vals :=
		grid.optimizer_types; if len(opt_vals) == 0 {temp := [1]optim.OptimizerType{base_params.optimizer_type}; opt_vals = temp[:]}

	for c_val in c_vals {
		for lr_val in lr_vals {
			for opt_type in opt_vals {
				params :=
					base_params; params.C = c_val; params.learning_rate = lr_val; params.optimizer_type = opt_type
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
				mean_score := 0.0; for s in scores {mean_score += s}; mean_score /= f64(len(scores))
				if mean_score >
				   res.best_score {res.best_score = mean_score; res.best_params = params}
				delete(scores, allocator)
			}
		}
	}
	return res
}

grid_search_svr :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	grid: SVRGrid,
	n_splits: int,
	base_params: SVRParams,
	allocator: mem.Allocator = context.allocator,
) -> SVRGridResult {
	res: SVRGridResult; res.allocator = allocator; res.best_score = -math.F64_MAX

	c_vals := grid.C_values; if len(c_vals) == 0 {temp := [1]f64{base_params.C}; c_vals = temp[:]}
	eps_vals :=
		grid.epsilon_values; if len(eps_vals) == 0 {temp := [1]f64{base_params.epsilon}; eps_vals = temp[:]}
	gamma_vals :=
		grid.gamma_values; if len(gamma_vals) == 0 {temp := [1]f64{base_params.gamma}; gamma_vals = temp[:]}

	for c_val in c_vals {
		for eps_val in eps_vals {
			for gamma_val in gamma_vals {
				params :=
					base_params; params.C = c_val; params.epsilon = eps_val; params.gamma = gamma_val
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
				mean_score := 0.0; for s in scores {mean_score += s}; mean_score /= f64(len(scores))
				if mean_score >
				   res.best_score {res.best_score = mean_score; res.best_params = params}
				delete(scores, allocator)
			}
		}
	}
	return res
}

// ✅ NEW: Grid Search for KNN
grid_search_knn :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	grid: KNNGrid,
	n_splits: int,
	base_params: KNNParams,
	allocator: mem.Allocator = context.allocator,
) -> KNNGridResult {
	res: KNNGridResult; res.allocator = allocator; res.best_score = -math.F64_MAX

	k_vals := grid.k_values; if len(k_vals) == 0 {temp := [1]int{base_params.k}; k_vals = temp[:]}
	w_vals :=
		grid.weight_types; if len(w_vals) == 0 {temp := [1]KNNWeights{base_params.weights}; w_vals = temp[:]}

	for k_val in k_vals {
		for w_val in w_vals {
			params := base_params; params.k = k_val; params.weights = w_val
			scores := cross_val_score(
				X,
				y,
				n_splits,
				_knn_cv_evaluator,
				&params,
				true,
				42,
				allocator,
			)
			mean_score := 0.0; for s in scores {mean_score += s}; mean_score /= f64(len(scores))
			if mean_score > res.best_score {res.best_score = mean_score; res.best_params = params}
			delete(scores, allocator)
		}
	}
	return res
}

// ✅ NEW: Grid Search for Random Forest
// ✅ NEW: Grid Search for Random Forest
grid_search_random_forest :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	grid: RFGrid,
	n_splits: int,
	base_params: RFParams,
	allocator: mem.Allocator = context.allocator,
) -> RFGridResult {
	res: RFGridResult; res.allocator = allocator; res.best_score = -math.F64_MAX

	trees_vals :=
		grid.n_trees_values; if len(trees_vals) == 0 {temp := [1]int{base_params.n_trees}; trees_vals = temp[:]}
	depth_vals :=
		grid.max_depth_values; if len(depth_vals) == 0 {temp := [1]int{base_params.max_depth}; depth_vals = temp[:]}

	// ✅ FIX: Use min_samples_values and base_params.min_samples
	split_vals :=
		grid.min_samples_values; if len(split_vals) == 0 {temp := [1]int{base_params.min_samples}; split_vals = temp[:]}

	for n_trees in trees_vals {
		for max_depth in depth_vals {
			// ✅ FIX: Loop variable renamed to min_samples
			for min_samples in split_vals {
				params :=
					base_params; params.n_trees = n_trees; params.max_depth = max_depth; params.min_samples = min_samples // ✅ FIX: Assign to min_samples

				scores := cross_val_score(
					X,
					y,
					n_splits,
					_rf_cv_evaluator,
					&params,
					true,
					42,
					allocator,
				)
				mean_score := 0.0; for s in scores {mean_score += s}; mean_score /= f64(len(scores))
				if mean_score >
				   res.best_score {res.best_score = mean_score; res.best_params = params}
				delete(scores, allocator)
			}
		}
	}
	return res
}

// ============================================================================
// Public API: Generic Grid Search (For ANY Model)
// Use this for OLS, Ridge, Lasso, GNB, etc.
// ============================================================================

GridSearchResult :: struct {
	best_score:      f64,
	best_params_raw: rawptr, // User must cast this back to their param type
	allocator:       mem.Allocator,
}

grid_search_custom :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	evaluator: ModelEvaluator,
	param_grid: []rawptr, // Slice of pointers to your param structs
	n_params: int, // Number of param combinations in the grid
	allocator: mem.Allocator = context.allocator,
) -> GridSearchResult {
	res: GridSearchResult; res.allocator = allocator; res.best_score = -math.F64_MAX; res.best_params_raw = nil

	for i in 0 ..< n_params {
		params_ptr := param_grid[i]
		scores := cross_val_score(X, y, n_splits, evaluator, params_ptr, true, 42, allocator)

		mean_score := 0.0
		for s in scores {mean_score += s}
		mean_score /= f64(len(scores))

		if mean_score > res.best_score {
			res.best_score = mean_score
			res.best_params_raw = params_ptr
		}
		delete(scores, allocator)
	}
	return res
}
