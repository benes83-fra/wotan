package ML

import l "../../linalg"
import optim "../../optimize"
import "core:math"
import "core:mem"

// ============================================================================
// 1. Public Evaluators
// These know how to cast the rawptr and call the specific model.
// (Removed the leading underscore to make them public)
// ============================================================================

logistic_cv_evaluator :: proc(
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

svr_cv_evaluator :: proc(
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

knn_cv_evaluator :: proc(
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

rf_cv_evaluator :: proc(
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
	// regression forest (criterion .MSE): R², as the other regression evaluators;
	// classification forest (.Gini / .Entropy): accuracy
	if params.criterion == .MSE do return metrics_r2(y_val, preds)
	return metrics_accuracy(y_val, preds)
}

// ============================================================================
// 2. The Generic Grid Search (The Magic)
// ============================================================================

grid_search :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	evaluator: ModelEvaluator,
	param_grid: []$P, // Accepts a slice of ANY param type
	allocator: mem.Allocator = context.allocator,
) -> (
	best_params: P,
	best_score: f64,
) {
	cv := CVSplitter {
		kind         = .KFold,
		n_splits     = n_splits,
		shuffle      = true,
		random_state = 42,
	}
	return grid_search_cv(X, y, cv, evaluator, param_grid, allocator)
}

// Grid search with any splitter (e.g. Time_Series for walk-forward validation of
// financial models, Stratified_KFold for imbalanced classes): the parameter set
// with the highest mean score over the splits
grid_search_cv :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	cv: CVSplitter,
	evaluator: ModelEvaluator,
	param_grid: []$P,
	allocator: mem.Allocator = context.allocator,
) -> (
	best_params: P,
	best_score: f64,
) {
	best_score = -math.F64_MAX

	if len(param_grid) == 0 {
		return
	}

	for i in 0 ..< len(param_grid) {
		scores := cross_val_score_cv(X, y, cv, evaluator, rawptr(&param_grid[i]), allocator)

		mean_score := 0.0
		for s in scores {mean_score += s}
		mean_score /= f64(len(scores))

		if mean_score > best_score {
			best_score = mean_score
			best_params = param_grid[i]
		}
		delete(scores, allocator)
	}

	return best_params, best_score
}

// ============================================================================
// 3. Convenience Wrappers
// These provide the cleanest possible API for the most common models.
// They just call the generic grid_search under the hood!
// ============================================================================

grid_search_logistic :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []LogisticParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	LogisticParams,
	f64,
) {
	return grid_search(X, y, n_splits, logistic_cv_evaluator, param_grid, allocator)
}

grid_search_svr :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []SVRParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	SVRParams,
	f64,
) {
	return grid_search(X, y, n_splits, svr_cv_evaluator, param_grid, allocator)
}

grid_search_knn :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []KNNParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	KNNParams,
	f64,
) {
	return grid_search(X, y, n_splits, knn_cv_evaluator, param_grid, allocator)
}

grid_search_random_forest :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []RFParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	RFParams,
	f64,
) {
	return grid_search(X, y, n_splits, rf_cv_evaluator, param_grid, allocator)
}
// ============================================================================
// 4. Additional Evaluators (Ridge, Lasso, SVMs, Trees, Boosting)
// ============================================================================


// Training design of the linear CV evaluators: like the pipelines (and sklearn's
// fit_intercept=True) the models get an unpenalised intercept. A column of ones
// is prepended unless X (all rows, not just the fold) already has a constant
// column, which then is the intercept. Predict with _linear_predict, which reads
// β[0] as the intercept when β has one entry more than X has columns.
_cv_linear_design :: proc(X, X_train: ^l.Matrix(f64), allocator: mem.Allocator) -> l.Matrix(f64) {
	if _ols_has_constant(X) {
		out := l.matrix_new(f64, X_train.rows, X_train.cols, allocator)
		copy(out.data, X_train.data)
		return out
	}
	return _with_intercept(X_train, allocator)
}

// Ridge with an unpenalised intercept (see _cv_linear_design); R² on the validation rows
ridge_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^RidgeParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	Xi := _cv_linear_design(X, &X_train, allocator); defer l.matrix_free(&Xi)
	model := ridge_fit(&Xi, y_train, params.lambda, params.method, allocator)
	defer _ols_result_free(&model, allocator)
	preds := _linear_predict(&X_val, model.beta, allocator)
	defer delete(preds, allocator)
	return metrics_r2(y_val, preds)
}

// Lasso with an unpenalised intercept (see _cv_linear_design); max_iter ≤ 0: 1000,
// tol ≤ 0: 1e-4; R² on the validation rows
lasso_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^LassoParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	Xi := _cv_linear_design(X, &X_train, allocator); defer l.matrix_free(&Xi)
	model := lasso_fit(
		&Xi,
		y_train,
		params.lambda,
		params.max_iter if params.max_iter > 0 else 1000,
		params.tol if params.tol > 0 else 1e-4,
		allocator,
	)
	defer _ols_result_free(&model, allocator)
	preds := _linear_predict(&X_val, model.beta, allocator)
	defer delete(preds, allocator)
	return metrics_r2(y_val, preds)
}

// Elastic net with an unpenalised intercept (see _cv_linear_design); R² on the validation rows
elastic_net_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^ElasticNetParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	Xi := _cv_linear_design(X, &X_train, allocator); defer l.matrix_free(&Xi)
	model := elastic_net_fit(
		&Xi,
		y_train,
		params.lambda,
		params.l1_ratio,
		params.max_iter if params.max_iter > 0 else 1000,
		params.tol if params.tol > 0 else 1e-4,
		allocator,
	)
	defer _ols_result_free(&model, allocator)
	preds := _linear_predict(&X_val, model.beta, allocator)
	defer delete(preds, allocator)
	return metrics_r2(y_val, preds)
}

grid_search_elastic_net :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []ElasticNetParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	ElasticNetParams,
	f64,
) {
	return grid_search(X, y, n_splits, elastic_net_cv_evaluator, param_grid, allocator)
}

linear_svm_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^SVMParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	// ✅ FIX: Use One-vs-Rest for multi-class support!
	model := ovr_linear_svm_fit(&X_train, y_train, params^, allocator)
	defer ovr_free(&model)

	preds := ovr_predict(&model, &X_val, allocator)
	defer delete(preds, allocator)

	return metrics_accuracy(y_val, preds)
}

kernel_svm_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^KernelSVMParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	// ✅ FIX: Use One-vs-Rest for multi-class support!
	model := ovr_kernel_svm_fit(&X_train, y_train, params^, allocator)
	defer ovr_free(&model)

	preds := ovr_predict(&model, &X_val, allocator)
	defer delete(preds, allocator)

	return metrics_accuracy(y_val, preds)
}

gb_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^GBParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	model := gb_fit(&X_train, y_train, params^, allocator); defer gb_free(&model)
	preds := gb_predict(&model, &X_val, allocator); defer delete(preds, allocator)

	// squared-error boosting is a regressor: R²; Log_Loss boosting: accuracy
	if params.loss == .Squared_Error do return metrics_r2(y_val, preds)

	return metrics_accuracy(y_val, preds)
}

dt_cv_evaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 {
	params := cast(^TreeParams)user_data
	X_train := subset_matrix(X, train_idx, allocator); defer l.matrix_free(&X_train)
	y_train := subset_slice(y, train_idx, allocator); defer delete(y_train, allocator)
	X_val := subset_matrix(X, val_idx, allocator); defer l.matrix_free(&X_val)
	y_val := subset_slice(y, val_idx, allocator); defer delete(y_val, allocator)

	model := dt_fit(&X_train, y_train, params^, allocator); defer dt_free(&model)
	preds := dt_predict(&model, &X_val, allocator); defer delete(preds, allocator)
	// regression tree (criterion .MSE): R²; classification tree: accuracy
	if params.criterion == .MSE do return metrics_r2(y_val, preds)
	return metrics_accuracy(y_val, preds)
}


// ============================================================================
// 5. Additional Convenience Wrappers
// ============================================================================

grid_search_ridge :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []RidgeParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	RidgeParams,
	f64,
) {
	return grid_search(X, y, n_splits, ridge_cv_evaluator, param_grid, allocator)
}

grid_search_lasso :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []LassoParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	LassoParams,
	f64,
) {
	return grid_search(X, y, n_splits, lasso_cv_evaluator, param_grid, allocator)
}

grid_search_linear_svm :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []SVMParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	SVMParams,
	f64,
) {
	return grid_search(X, y, n_splits, linear_svm_cv_evaluator, param_grid, allocator)
}

grid_search_kernel_svm :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []KernelSVMParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	KernelSVMParams,
	f64,
) {
	return grid_search(X, y, n_splits, kernel_svm_cv_evaluator, param_grid, allocator)
}

grid_search_decision_tree :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []TreeParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	TreeParams,
	f64,
) {
	return grid_search(X, y, n_splits, dt_cv_evaluator, param_grid, allocator)
}

grid_search_gradient_boosting :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	param_grid: []GBParams,
	allocator: mem.Allocator = context.allocator,
) -> (
	GBParams,
	f64,
) {
	return grid_search(X, y, n_splits, gb_cv_evaluator, param_grid, allocator)
}
