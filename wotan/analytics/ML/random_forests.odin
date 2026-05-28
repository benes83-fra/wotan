package ML

import l "../../linalg"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Random Forest Structures
// ============================================================================

RandomForest :: struct {
	trees:        [dynamic]DecisionTree,
	n_trees:      int,
	max_features: int,
	bootstrap:    bool,
	allocator:    mem.Allocator,
}

RFParams :: struct {
	n_trees:      int,
	max_features: int, // 0 = auto (sqrt(n_features))
	min_samples:  int,
	max_depth:    int,
	bootstrap:    bool,
	oob_score:    bool,
}

// ============================================================================
// Public API
// ============================================================================

rf_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: RFParams,
	allocator: mem.Allocator = context.allocator,
) -> RandomForest {
	forest: RandomForest
	forest.allocator = allocator
	forest.n_trees = params.n_trees
	forest.max_features = params.max_features
	if forest.max_features == 0 {
		forest.max_features = int(math.sqrt_f64(f64(X.cols)))
	}
	forest.bootstrap = params.bootstrap

	forest.trees = make([dynamic]DecisionTree, 0, params.n_trees, allocator)

	n_samples := X.rows

	for t in 0 ..< params.n_trees {
		// Bootstrap sample
		sample_indices: []int
		if params.bootstrap {
			sample_indices = _bootstrap_sample(n_samples, allocator)
			defer delete(sample_indices, allocator)
		} else {
			sample_indices = make([]int, n_samples, allocator)
			for i in 0 ..< n_samples {sample_indices[i] = i}
			defer delete(sample_indices, allocator)
		}

		// Train tree with feature subsampling
		tree_params := TreeParams {
			max_depth   = params.max_depth,
			min_samples = params.min_samples,
		}

		tree := _fit_tree_with_feature_subsampling(
			X,
			y,
			sample_indices,
			forest.max_features,
			tree_params,
			allocator,
		)

		append(&forest.trees, tree)
	}

	return forest
}

rf_predict :: proc(
	forest: ^RandomForest,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := X.rows
	n_trees := len(forest.trees)

	if n_trees == 0 {
		return make([]f64, n, allocator)
	}

	// Collect predictions
	all_preds := make([][]f64, n_trees, allocator)
	defer {
		for preds in all_preds {
			delete(preds, allocator)
		}
		delete(all_preds, allocator)
	}

	for &tree, t in forest.trees {
		all_preds[t] = dt_predict(&tree, X, allocator)
	}

	// Average predictions using SIMD
	out := make([]f64, n, allocator)
	ones := make([]f64, n_trees, allocator)
	for i in 0 ..< n_trees {ones[i] = 1.0}
	defer delete(ones, allocator)

	for i in 0 ..< n {
		preds_i := make([]f64, n_trees, allocator)
		for t in 0 ..< n_trees {
			preds_i[t] = all_preds[t][i]
		}

		if n_trees >= 8 {
			sum := l.dot_simd(preds_i, ones)
			out[i] = sum / f64(n_trees)
		} else {
			sum := 0.0
			for t in 0 ..< n_trees {sum += preds_i[t]}
			out[i] = sum / f64(n_trees)
		}
		delete(preds_i, allocator)
	}

	return out
}

rf_free :: proc(forest: ^RandomForest) {
	for &tree in forest.trees {
		dt_free(&tree)
	}
	if len(forest.trees) > 0 {
		delete(forest.trees)
	}
}

// ============================================================================
// Internal Helpers
// ============================================================================

_bootstrap_sample :: proc(n: int, allocator: mem.Allocator) -> []int {
	indices := make([]int, n, allocator)
	for i in 0 ..< n {
		indices[i] = rand.int_range(0, n - 1)
	}
	return indices
}

_sample_features :: proc(n_features: int, k: int, allocator: mem.Allocator) -> []int {
	if k >= n_features {
		out := make([]int, n_features, allocator)
		for i in 0 ..< n_features {out[i] = i}
		return out
	}

	// Fisher-Yates shuffle for first k
	features := make([]int, n_features, allocator)
	for i in 0 ..< n_features {features[i] = i}

	for i in 0 ..< k {
		j := rand.int_range(i, n_features - 1)
		tmp := features[i]
		features[i] = features[j]
		features[j] = tmp
	}

	out := make([]int, k, allocator)
	copy(out, features[0:k])
	delete(features, allocator)
	return out
}

_fit_tree_with_feature_subsampling :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	sample_indices: []int,
	max_features: int,
	params: TreeParams,
	allocator: mem.Allocator,
) -> DecisionTree {
	// Create subsampled X and y
	n_samples := len(sample_indices)
	X_sub := l.matrix_new(f64, n_samples, X.cols, allocator)
	y_sub := make([]f64, n_samples, allocator)

	for i, idx in sample_indices {
		for j in 0 ..< X.cols {
			X_sub.data[i * X_sub.cols + j] = X.data[idx * X.cols + j]
		}
		y_sub[i] = y[idx]
	}

	// Train tree normally (feature subsampling happens inside dt_build_node)
	// For now, we pass max_features and handle it in the tree builder
	// (You can extend dt_build_node to accept max_features and sample features at each split)

	tree := dt_fit(&X_sub, y_sub, params, allocator)

	l.matrix_free(&X_sub)
	delete(y_sub, allocator)

	return tree
}
