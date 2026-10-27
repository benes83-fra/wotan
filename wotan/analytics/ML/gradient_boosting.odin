package ML

import l "../../linalg"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Gradient Boosting Structures
// ============================================================================

GradientBoosting :: struct {
	trees:         [dynamic]DecisionTree,
	n_estimators:  int,
	learning_rate: f64,
	initial_pred:  f64,
	max_depth:     int,
	min_samples:   int,
	allocator:     mem.Allocator,
}

GBParams :: struct {
	n_estimators:  int,
	learning_rate: f64,
	max_depth:     int,
	min_samples:   int,
	subsample:     f64, // 1.0 = use all data, <1.0 = stochastic GB
}

// ============================================================================
// Public API
// ============================================================================

// Least-squares gradient boosting (sklearn GradientBoostingRegressor,
// loss="squared_error"): F₀ = mean(y), each tree fits the residuals y - F and
// F += learning_rate · tree. subsample < 1 fits each tree on
// int(subsample · n) rows drawn without replacement (stochastic GB).
gb_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: GBParams,
	allocator: mem.Allocator = context.allocator,
) -> GradientBoosting {
	n_samples := len(y)
	if X.rows != n_samples do panic("gb_fit: y length mismatch")
	if n_samples == 0 do panic("gb_fit: no samples")

	model: GradientBoosting
	model.allocator = allocator
	model.n_estimators = params.n_estimators
	model.learning_rate = params.learning_rate
	model.max_depth = params.max_depth
	model.min_samples = params.min_samples

	// Initial prediction: mean of y
	model.initial_pred = 0.0
	for val in y {model.initial_pred += val}
	model.initial_pred /= f64(n_samples)

	model.trees = make([dynamic]DecisionTree, 0, max(params.n_estimators, 0), allocator)

	// Current predictions (start with initial_pred)
	preds := make([]f64, n_samples, allocator)
	defer delete(preds, allocator)
	for i in 0 ..< n_samples {preds[i] = model.initial_pred}

	residuals := make([]f64, n_samples, allocator)
	defer delete(residuals, allocator)

	// Row subsample size and a permutation buffer for sampling without replacement
	n_sub := n_samples
	if params.subsample > 0.0 && params.subsample < 1.0 {
		n_sub = max(1, int(f64(n_samples) * params.subsample))
	}
	perm := make([]int, n_samples, allocator)
	defer delete(perm, allocator)
	for i in 0 ..< n_samples {perm[i] = i}

	tree_params := TreeParams {
		max_depth   = params.max_depth,
		min_samples = params.min_samples,
	}

	for _ in 0 ..< params.n_estimators {
		// 1. Residuals (negative gradient of ½(y - F)²)
		for i in 0 ..< n_samples {
			residuals[i] = y[i] - preds[i]
		}

		// 2. Optional subsampling: partial Fisher-Yates over perm
		if n_sub < n_samples {
			for i in 0 ..< n_sub {
				j := rand.int_range(i, n_samples)
				perm[i], perm[j] = perm[j], perm[i]
			}
		}

		// 3. Train tree on residuals
		tree := _fit_tree_on_residuals(X, residuals, perm[:n_sub], tree_params, allocator)
		append(&model.trees, tree)

		// 4. Update predictions on all rows: F += learning_rate · tree(x)
		tree_pred := dt_predict(&model.trees[len(model.trees) - 1], X, allocator)
		for i in 0 ..< n_samples {
			preds[i] += params.learning_rate * tree_pred[i]
		}
		delete(tree_pred, allocator)
	}

	return model
}

gb_predict :: proc(
	model: ^GradientBoosting,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := X.rows
	n_trees := len(model.trees)

	out := make([]f64, n, allocator)

	// Start with initial prediction
	for i in 0 ..< n {
		out[i] = model.initial_pred
	}

	// Add contribution from each tree: out += lr * tree_pred
	for &tree, t in model.trees {
		tree_pred := dt_predict(&tree, X, allocator)

		// SIMD-optimized accumulation for large n_trees
		if n_trees >= 8 && n >= 8 {
			// Could use SIMD here for batch updates, but simple loop is fine
			for i in 0 ..< n {
				out[i] += model.learning_rate * tree_pred[i]
			}
		} else {
			for i in 0 ..< n {
				out[i] += model.learning_rate * tree_pred[i]
			}
		}

		delete(tree_pred, allocator)
	}

	return out
}

gb_free :: proc(model: ^GradientBoosting) {
	for i in 0 ..< len(model.trees) {
		dt_free(&model.trees[i])
	}
	if len(model.trees) > 0 {
		delete(model.trees) // No allocator for [dynamic]
	}
}

// ============================================================================
// Internal Helpers
// ============================================================================

// Fit a tree on residuals (with optional subsampling via sample_indices)
_fit_tree_on_residuals :: proc(
	X: ^l.Matrix(f64),
	residuals: []f64,
	sample_indices: []int,
	params: TreeParams,
	allocator: mem.Allocator,
) -> DecisionTree {
	n_samples := len(sample_indices)

	// Create subsampled views
	X_sub := l.matrix_new(f64, n_samples, X.cols, allocator)
	y_sub := make([]f64, n_samples, allocator)

	for idx, i in sample_indices {
		for j in 0 ..< X.cols {
			X_sub.data[i * X_sub.cols + j] = X.data[idx * X.cols + j]
		}
		y_sub[i] = residuals[idx]
	}

	// Train regression tree on residuals
	tree := dt_fit(&X_sub, y_sub, params, allocator)

	l.matrix_free(&X_sub)
	delete(y_sub, allocator)

	return tree
}

// Bootstrap sample (n draws with replacement from [0, n))
_bootstrap_sample_gb :: proc(n: int, allocator: mem.Allocator) -> []int {
	indices := make([]int, n, allocator)
	for i in 0 ..< n {
		indices[i] = rand.int_max(n)
	}
	return indices
}
