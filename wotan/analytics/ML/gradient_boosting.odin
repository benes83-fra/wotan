package ML

import l "../../linalg"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Gradient Boosting Structures
// ============================================================================

GradientBoosting :: struct {
	trees:           [dynamic]DecisionTree, // Log_Loss: stage s, column k at trees[s·trees_per_stage + k]
	n_estimators:    int,
	learning_rate:   f64,
	initial_pred:    f64, // F₀ (Squared_Error)
	max_depth:       int,
	min_samples:     int,
	loss:            GBLoss,
	classes:         []f64, // sorted class labels (Log_Loss)
	init_raw:        []f64, // F₀ per raw-score column (Log_Loss)
	trees_per_stage: int, // raw-score columns: 1 (regression, binary) or n_classes (multiclass)
	allocator:       mem.Allocator,
}

// The zero value Squared_Error keeps existing GBParams literals regressors.
GBLoss :: enum {
	Squared_Error, // regression (GradientBoostingRegressor)
	Log_Loss, // classification (GradientBoostingClassifier, loss="log_loss")
}

GBParams :: struct {
	n_estimators:  int,
	learning_rate: f64,
	max_depth:     int,
	min_samples:   int,
	subsample:     f64, // 1.0 = use all data, <1.0 = stochastic GB
	loss:          GBLoss,
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
	if params.loss == .Log_Loss do return _gb_fit_classifier(X, y, params, allocator)

	model: GradientBoosting
	model.trees_per_stage = 1
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

	if model.loss == .Log_Loss {
		proba := gb_predict_proba(model, X, allocator)
		defer l.matrix_free(&proba)
		K := len(model.classes)
		out := make([]f64, n, allocator)
		for i in 0 ..< n {out[i] = model.classes[_argmax(proba.data[i * K:(i + 1) * K])]}
		return out
	}

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

// Raw scores F(x) (n × trees_per_stage) of a Log_Loss model: log-odds of the
// second class (binary) or one score per class (multiclass, softmax link)
gb_decision_function :: proc(
	model: ^GradientBoosting,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	if model.loss != .Log_Loss do panic("gb_decision_function: needs a Log_Loss model")
	C := model.trees_per_stage
	out := l.matrix_new(f64, X.rows, C, allocator)
	for i in 0 ..< X.rows {
		for k in 0 ..< C {out.data[i * C + k] = model.init_raw[k]}
	}
	for &tree, t in model.trees {
		k := t % C
		for i in 0 ..< X.rows {
			out.data[i * C + k] += model.learning_rate * dt_predict_single(&tree, X, i)
		}
	}
	return out
}

// Class probabilities (n × n_classes, model.classes order), as sklearn predict_proba
gb_predict_proba :: proc(
	model: ^GradientBoosting,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	raw := gb_decision_function(model, X, allocator)
	defer l.matrix_free(&raw)
	K := len(model.classes)
	out := l.matrix_new(f64, X.rows, K, allocator)
	for i in 0 ..< X.rows {
		_gb_link(raw.data[i * raw.cols:(i + 1) * raw.cols], out.data[i * K:(i + 1) * K])
	}
	return out
}

gb_free :: proc(model: ^GradientBoosting) {
	for i in 0 ..< len(model.trees) {
		dt_free(&model.trees[i])
	}
	delete(model.trees) // No allocator for [dynamic]
	delete(model.classes, model.allocator)
	delete(model.init_raw, model.allocator)
	model.trees, model.classes, model.init_raw = nil, nil, nil
}

// ============================================================================
// Log-loss (classification) boosting
// ============================================================================

// sklearn GradientBoostingClassifier (loss="log_loss"):
// - F₀ = log(p₁ / (1 - p₁)) (binary) or log p_k (multiclass) from the class priors
// - every stage fits one regression tree per raw-score column to the residuals
//   y_k - p_k(x) (p from the scores at the start of the stage), then replaces each
//   leaf value by one Newton step  Σ r / Σ p(1 - p)  over the in-bag rows of the
//   leaf (times (K-1)/K for multiclass), and F_k += learning_rate · tree(x)
// - binary: p = sigmoid(F); multiclass: p = softmax(F)
_gb_fit_classifier :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: GBParams,
	allocator: mem.Allocator,
) -> GradientBoosting {
	n := X.rows
	model: GradientBoosting
	model.allocator = allocator
	model.loss = .Log_Loss
	model.n_estimators = params.n_estimators
	model.learning_rate = params.learning_rate
	model.max_depth = params.max_depth
	model.min_samples = params.min_samples

	classes, codes := _tree_encode_classes(y, allocator)
	defer delete(codes, allocator)
	K := len(classes)
	if K < 2 do panic("gb_fit: log loss needs at least 2 classes")
	model.classes = classes
	C := 1 if K == 2 else K
	model.trees_per_stage = C

	// Initial raw scores from the class priors (clipped as sklearn, float32 eps)
	EPS :: 1.1920928955078125e-07
	model.init_raw = make([]f64, C, allocator)
	prior := make([]f64, K, allocator)
	defer delete(prior, allocator)
	for c in codes {prior[c] += 1}
	for &v in prior {v = math.clamp(v / f64(n), EPS, 1 - EPS)}
	if K == 2 {
		model.init_raw[0] = math.ln(prior[1] / (1 - prior[1]))
	} else {
		for k in 0 ..< K {model.init_raw[k] = math.ln(prior[k])}
	}

	model.trees = make([dynamic]DecisionTree, 0, max(params.n_estimators, 0) * C, allocator)

	F := make([]f64, n * C, allocator)
	P := make([]f64, n * K, allocator)
	yk := make([]f64, n, allocator)
	resid := make([]f64, n, allocator)
	defer {
		delete(F, allocator)
		delete(P, allocator)
		delete(yk, allocator)
		delete(resid, allocator)
	}
	for i in 0 ..< n {
		for k in 0 ..< C {F[i * C + k] = model.init_raw[k]}
	}

	n_sub := n
	if params.subsample > 0.0 && params.subsample < 1.0 {
		n_sub = max(1, int(f64(n) * params.subsample))
	}
	perm := make([]int, n, allocator)
	defer delete(perm, allocator)
	for i in 0 ..< n {perm[i] = i}

	tree_params := TreeParams {
		max_depth   = params.max_depth,
		min_samples = params.min_samples,
	}
	scale := 1.0 if K == 2 else f64(K - 1) / f64(K)

	for _ in 0 ..< params.n_estimators {
		// probabilities at the start of the stage (shared by all K trees)
		for i in 0 ..< n {_gb_link(F[i * C:(i + 1) * C], P[i * K:(i + 1) * K])}

		// one row subsample per stage
		if n_sub < n {
			for i in 0 ..< n_sub {
				j := rand.int_range(i, n)
				perm[i], perm[j] = perm[j], perm[i]
			}
		}
		rows := perm[:n_sub]

		for k in 0 ..< C {
			cls := 1 if K == 2 else k
			for i in 0 ..< n {
				yk[i] = 1 if codes[i] == cls else 0
				resid[i] = yk[i] - P[i * K + cls]
			}
			tree := _fit_tree_on_residuals(X, resid, rows, tree_params, allocator)
			_gb_newton_leaves(&tree, X, yk, resid, rows, scale, allocator)
			append(&model.trees, tree)

			last := &model.trees[len(model.trees) - 1]
			for i in 0 ..< n {
				F[i * C + k] += params.learning_rate * dt_predict_single(last, X, i)
			}
		}
	}
	return model
}

// Replace every leaf value by  scale · Σ r / Σ (y - r)(1 - y + r)  over the rows
// (in-bag samples) that fall into the leaf; 0 if the denominator vanishes
_gb_newton_leaves :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	y, resid: []f64,
	rows: []int,
	scale: f64,
	allocator: mem.Allocator,
) {
	num := make([]f64, len(tree.nodes), allocator)
	den := make([]f64, len(tree.nodes), allocator)
	defer {
		delete(num, allocator)
		delete(den, allocator)
	}
	for i in rows {
		leaf := _dt_leaf_index(tree, X, i)
		num[leaf] += resid[i]
		den[leaf] += (y[i] - resid[i]) * (1 - y[i] + resid[i])
	}
	for &node, j in tree.nodes {
		if node.left_idx >= 0 do continue
		node.value = 0 if math.abs(den[j]) < 1e-150 else scale * num[j] / den[j]
	}
}

// Raw scores → class probabilities: sigmoid (1 column) or softmax (K columns)
_gb_link :: proc(raw: []f64, p: []f64) {
	if len(raw) == 1 {
		z := raw[0]
		// numerically stable sigmoid
		p1 := 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
		p[0], p[1] = 1 - p1, p1
		return
	}
	m := raw[0]
	for v in raw {m = max(m, v)}
	sum := 0.0
	for v, k in raw {
		p[k] = math.exp(v - m)
		sum += p[k]
	}
	for &v in p {v /= sum}
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
