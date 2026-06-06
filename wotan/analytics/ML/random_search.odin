package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// 1. The Generic Random Search (The Magic)
// ============================================================================

random_search :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	evaluator: ModelEvaluator,
	n_iter: int,
	sampler: proc(seed: u64) -> $P, // The magic: a function that returns a random $P
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	best_params: P,
	best_score: f64,
) {
	best_score = -math.F64_MAX

	if n_iter <= 0 {
		return
	}

	for i in 0 ..< n_iter {
		// Generate a random parameter set using the iteration index as the seed
		current_seed := seed + u64(i)
		params := sampler(current_seed)

		// ✅ Keep the CV seed constant (42) so every parameter set is evaluated
		// on the EXACT same folds. This ensures a fair comparison!
		scores := cross_val_score(X, y, n_splits, evaluator, rawptr(&params), true, 42, allocator)

		mean_score := 0.0
		for s in scores {mean_score += s}
		mean_score /= f64(len(scores))

		if mean_score > best_score {
			best_score = mean_score
			best_params = params
		}
		delete(scores, allocator)
	}

	return best_params, best_score
}

// ============================================================================
// 2. Convenience Wrappers
// ============================================================================

random_search_logistic :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> LogisticParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	LogisticParams,
	f64,
) {
	return random_search(X, y, n_splits, logistic_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_svr :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> SVRParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	SVRParams,
	f64,
) {
	return random_search(X, y, n_splits, svr_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_knn :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> KNNParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	KNNParams,
	f64,
) {
	return random_search(X, y, n_splits, knn_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_random_forest :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> RFParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	RFParams,
	f64,
) {
	return random_search(X, y, n_splits, rf_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_ridge :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> RidgeParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	RidgeParams,
	f64,
) {
	return random_search(X, y, n_splits, ridge_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_lasso :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> LassoParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	LassoParams,
	f64,
) {
	return random_search(X, y, n_splits, lasso_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_linear_svm :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> SVMParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	SVMParams,
	f64,
) {
	return random_search(X, y, n_splits, linear_svm_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_kernel_svm :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> KernelSVMParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	KernelSVMParams,
	f64,
) {
	return random_search(X, y, n_splits, kernel_svm_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_decision_tree :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> TreeParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	TreeParams,
	f64,
) {
	return random_search(X, y, n_splits, dt_cv_evaluator, n_iter, sampler, seed, allocator)
}

random_search_gradient_boosting :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	n_iter: int,
	sampler: proc(seed: u64) -> GBParams,
	seed: u64 = 42,
	allocator: mem.Allocator = context.allocator,
) -> (
	GBParams,
	f64,
) {
	return random_search(X, y, n_splits, gb_cv_evaluator, n_iter, sampler, seed, allocator)
}
