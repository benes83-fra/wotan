package ML

import l "../../linalg"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

// ============================================================================
// Random Forest Structures
// ============================================================================

RandomForest :: struct {
	trees:        [dynamic]DecisionTree,
	n_trees:      int,
	max_features: int,
	bootstrap:    bool,
	// out-of-bag R² (regression) / accuracy (classification); NaN unless params.oob_score && bootstrap
	oob_score:    f64,
	classes:      []f64, // sorted class labels (classification forests only, else nil)
	allocator:    mem.Allocator,
}

RFParams :: struct {
	n_trees:      int,
	max_features: int, // features tried per split; 0 = auto (sqrt(n_features))
	min_samples:  int,
	max_depth:    int,
	bootstrap:    bool,
	oob_score:    bool,
	// .MSE: regression forest (mean of the trees); .Gini / .Entropy: classification
	// forest (sklearn RandomForestClassifier: average of the trees' class
	// probabilities, predict = class with the highest average)
	criterion:    TreeCriterion,
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
	n_samples := X.rows
	if len(y) != n_samples do panic("rf_fit: y length mismatch")
	if n_samples == 0 do panic("rf_fit: no samples")

	forest: RandomForest
	forest.allocator = allocator
	forest.n_trees = params.n_trees
	forest.max_features = params.max_features
	if forest.max_features <= 0 {
		forest.max_features = max(1, int(math.sqrt_f64(f64(X.cols))))
	}
	forest.bootstrap = params.bootstrap
	forest.oob_score = math.nan_f64()

	forest.trees = make([dynamic]DecisionTree, 0, max(params.n_trees, 0), allocator)

	// Classification: classes of the whole training set, shared by every tree
	// (a bootstrap sample can miss a class; its probability is then 0)
	codes: []int
	K := 0
	if params.criterion != .MSE {
		forest.classes, codes = _tree_encode_classes(y, allocator)
		K = len(forest.classes)
	}
	defer delete(codes, allocator)

	// Out-of-bag accumulators (per class for classification)
	want_oob := params.oob_score && params.bootstrap
	n_oob_cols := max(K, 1)
	oob_sum := make([]f64, n_samples * n_oob_cols if want_oob else 0, context.temp_allocator)
	oob_cnt := make([]f64, n_samples if want_oob else 0, context.temp_allocator)
	in_bag := make([]bool, n_samples if want_oob else 0, context.temp_allocator)
	defer {
		delete(oob_sum, context.temp_allocator)
		delete(oob_cnt, context.temp_allocator)
		delete(in_bag, context.temp_allocator)
	}

	tree_params := TreeParams {
		max_depth    = params.max_depth,
		min_samples  = params.min_samples,
		max_features = forest.max_features,
		criterion    = params.criterion,
	}

	sample_indices := make([]int, n_samples, allocator)
	defer delete(sample_indices, allocator)

	for _ in 0 ..< params.n_trees {
		// Bootstrap sample (n draws with replacement) or all rows
		if params.bootstrap {
			for i in 0 ..< n_samples {sample_indices[i] = rand.int_max(n_samples)}
		} else {
			for i in 0 ..< n_samples {sample_indices[i] = i}
		}

		// Train tree; feature subsampling happens at every split
		tree := _fit_tree_with_feature_subsampling(
			X,
			y,
			sample_indices,
			forest.max_features,
			tree_params,
			allocator,
			codes,
			forest.classes,
		)
		append(&forest.trees, tree)

		if want_oob {
			for i in 0 ..< n_samples {in_bag[i] = false}
			for idx in sample_indices {in_bag[idx] = true}
			last := &forest.trees[len(forest.trees) - 1]
			for i in 0 ..< n_samples {
				if in_bag[i] do continue
				oob_cnt[i] += 1
				if K == 0 {
					oob_sum[i] += dt_predict_single(last, X, i)
				} else {
					leaf := _dt_leaf_index(last, X, i)
					for k in 0 ..< K {oob_sum[i * K + k] += last.proba[leaf * K + k]}
				}
			}
		}
	}

	if want_oob && K > 0 {
		// accuracy of the averaged OOB class probabilities (sklearn oob_score_)
		hit, cnt := 0.0, 0.0
		for i in 0 ..< n_samples {
			if oob_cnt[i] == 0 do continue
			cnt += 1
			if _argmax(oob_sum[i * K:(i + 1) * K]) == codes[i] do hit += 1
		}
		if cnt > 0 do forest.oob_score = hit / cnt
	} else if want_oob {
		// R² over the samples that were out of bag at least once
		mean, cnt := 0.0, 0.0
		for i in 0 ..< n_samples {
			if oob_cnt[i] > 0 {mean += y[i]; cnt += 1}
		}
		if cnt > 0 {
			mean /= cnt
			sse, sst := 0.0, 0.0
			for i in 0 ..< n_samples {
				if oob_cnt[i] == 0 do continue
				e := y[i] - oob_sum[i] / oob_cnt[i]
				sse += e * e
				sst += (y[i] - mean) * (y[i] - mean)
			}
			forest.oob_score = 1.0 - sse / sst
		}
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

	if len(forest.classes) > 0 {
		// soft voting: class with the highest mean probability
		proba := rf_predict_proba(forest, X, allocator)
		defer l.matrix_free(&proba)
		K := len(forest.classes)
		out := make([]f64, n, allocator)
		for i in 0 ..< n {out[i] = forest.classes[_argmax(proba.data[i * K:(i + 1) * K])]}
		return out
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

// Mean of the trees' class probabilities (n × n_classes, forest.classes order)
rf_predict_proba :: proc(
	forest: ^RandomForest,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	K := len(forest.classes)
	if K == 0 do panic("rf_predict_proba: not a classification forest (criterion = .MSE)")
	out := l.matrix_new(f64, X.rows, K, allocator)
	n_trees := len(forest.trees)
	if n_trees == 0 do return out
	for &tree in forest.trees {
		if len(tree.classes) != K do panic("rf_predict_proba: tree classes do not match the forest")
		for i in 0 ..< X.rows {
			leaf := _dt_leaf_index(&tree, X, i)
			if leaf < 0 do continue
			for k in 0 ..< K {out.data[i * K + k] += tree.proba[leaf * K + k]}
		}
	}
	for &v in out.data {v /= f64(n_trees)}
	return out
}

rf_free :: proc(forest: ^RandomForest) {
	for &tree in forest.trees {
		dt_free(&tree)
	}
	delete(forest.trees)
	delete(forest.classes, forest.allocator)
	forest.trees, forest.classes = nil, nil
}

// Index of the largest entry (first one on ties, as numpy argmax)
_argmax :: proc(v: []f64) -> int {
	best := 0
	for x, i in v {
		if x > v[best] do best = i
	}
	return best
}

// ============================================================================
// Internal Helpers
// ============================================================================

// Bootstrap sample: n draws with replacement from [0, n)
_bootstrap_sample :: proc(n: int, allocator: mem.Allocator) -> []int {
	indices := make([]int, n, allocator)
	for i in 0 ..< n {
		indices[i] = rand.int_max(n)
	}
	return indices
}

// k distinct features drawn uniformly from [0, n_features)
_sample_features :: proc(n_features: int, k: int, allocator: mem.Allocator) -> []int {
	if k >= n_features {
		out := make([]int, n_features, allocator)
		for i in 0 ..< n_features {out[i] = i}
		return out
	}

	// Partial Fisher-Yates shuffle for the first k
	features := make([]int, n_features, allocator)
	for i in 0 ..< n_features {features[i] = i}

	for i in 0 ..< k {
		j := rand.int_range(i, n_features)
		features[i], features[j] = features[j], features[i]
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
	codes: []int = nil, // classification: class index per row of X (into classes)
	classes: []f64 = nil,
) -> DecisionTree {
	// Create subsampled X and y
	n_samples := len(sample_indices)
	X_sub := l.matrix_new(f64, n_samples, X.cols, allocator)
	y_sub := make([]f64, n_samples, allocator)
	codes_sub := make([]int, n_samples if codes != nil else 0, allocator)

	for idx, i in sample_indices {
		for j in 0 ..< X.cols {
			X_sub.data[i * X_sub.cols + j] = X.data[idx * X.cols + j]
		}
		y_sub[i] = y[idx]
		if codes != nil do codes_sub[i] = codes[idx]
	}

	// Feature subsampling happens at every split inside the tree builder
	tree_params := params
	tree_params.max_features = max_features
	tree: DecisionTree
	if codes != nil {
		// every tree owns a copy of the forest's classes
		tree = _dt_fit_codes(&X_sub, y_sub, codes_sub, slice.clone(classes, allocator), tree_params, allocator)
	} else {
		tree = _dt_fit_codes(&X_sub, y_sub, nil, nil, tree_params, allocator)
	}

	l.matrix_free(&X_sub)
	delete(y_sub, allocator)
	delete(codes_sub, allocator)

	return tree
}
