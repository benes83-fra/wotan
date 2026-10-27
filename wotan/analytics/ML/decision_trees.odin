package ML

import l "../../linalg"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

// ============================================================================
// Decision Tree Structures
// ============================================================================

DecisionTree :: struct {
	nodes:     [dynamic]Node, // Dynamic array (managed by runtime)
	root_idx:  int,
	classes:   []f64, // sorted class labels (classification trees only, else nil)
	proba:     [dynamic]f64, // class fractions of node i at [i·n_classes ..] (classification only)
	allocator: mem.Allocator,
}

Node :: struct {
	// Split info (internal nodes)
	feature_idx:  int,
	threshold:    f64,

	// Prediction info (leaf nodes): mean (regression) or majority class label
	value:        f64,

	// Tree structure
	left_idx:     int, // -1 if leaf
	right_idx:    int, // -1 if leaf

	// Statistics
	impurity:     f64,
	sample_count: int,
}

// Split criterion. The zero value MSE keeps existing TreeParams literals regression trees.
TreeCriterion :: enum {
	MSE, // regression (sklearn DecisionTreeRegressor): leaf = mean of y
	Gini, // classification (DecisionTreeClassifier): impurity 1 - Σ p_k², leaf = majority class
	Entropy, // classification: impurity -Σ p_k log₂ p_k
}

// CART as sklearn DecisionTreeRegressor / DecisionTreeClassifier:
// split at the midpoint of consecutive distinct sorted values, x <= t goes left.
// Equal-gain splits go to the lowest feature index / first threshold (sklearn
// breaks such ties by its random feature order, so deep trees can differ there).
// Classification leaves predict the most frequent class (ties: smallest label)
// and store the class fractions for dt_predict_proba.
TreeParams :: struct {
	max_depth:    int, // a node at this depth becomes a leaf (0 = single leaf)
	min_samples:  int, // min samples to split a node (sklearn min_samples_split)
	max_features: int, // features drawn per split (0 or >= n_features = all), like sklearn:
	// if none of them gives a valid split, the remaining features are tried too
	criterion:    TreeCriterion,
}

// ============================================================================
// Public API
// ============================================================================

dt_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: TreeParams,
	allocator: mem.Allocator = context.allocator,
) -> DecisionTree {
	n_samples := X.rows
	if len(y) != n_samples do panic("dt_fit: y length mismatch")
	if n_samples == 0 do panic("dt_fit: no samples")

	if params.criterion != .MSE {
		classes, codes := _tree_encode_classes(y, allocator)
		defer delete(codes, allocator)
		return _dt_fit_codes(X, y, codes, classes, params, allocator)
	}
	return _dt_fit_codes(X, y, nil, nil, params, allocator)
}

dt_predict :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := X.rows
	out := make([]f64, n, allocator)

	for i in 0 ..< n {
		out[i] = dt_predict_single(tree, X, i)
	}
	return out
}

// Class probabilities (n × n_classes, columns in tree.classes order): the class
// fractions of the training samples in the leaf, as sklearn predict_proba.
dt_predict_proba :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	K := len(tree.classes)
	if K == 0 do panic("dt_predict_proba: not a classification tree (criterion = .MSE)")
	out := l.matrix_new(f64, X.rows, K, allocator)
	for i in 0 ..< X.rows {
		leaf := _dt_leaf_index(tree, X, i)
		if leaf < 0 do continue
		copy(out.data[i * K:(i + 1) * K], tree.proba[leaf * K:(leaf + 1) * K])
	}
	return out
}

// ============================================================================
// Internal: Tree Building
// ============================================================================

// Sorted distinct labels and the class index of every sample
_tree_encode_classes :: proc(y: []f64, allocator: mem.Allocator) -> (classes: []f64, codes: []int) {
	sorted := slice.clone(y, allocator)
	defer delete(sorted, allocator)
	slice.sort(sorted)
	uniq := make([dynamic]f64, 0, 8, allocator)
	for v, i in sorted {
		if math.is_nan(v) do panic("tree classifier: NaN label")
		if i == 0 || v != sorted[i - 1] do append(&uniq, v)
	}
	classes = make([]f64, len(uniq), allocator)
	copy(classes, uniq[:])
	delete(uniq)
	codes = make([]int, len(y), allocator)
	for v, i in y {
		k, _ := slice.binary_search(classes, v)
		codes[i] = k
	}
	return
}

// Fit on all rows of X. codes == nil: regression on y; otherwise classification on
// the class indices codes (into classes, whose ownership passes to the tree).
_dt_fit_codes :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	codes: []int,
	classes: []f64,
	params: TreeParams,
	allocator: mem.Allocator,
) -> DecisionTree {
	n_samples := X.rows
	tree: DecisionTree
	tree.allocator = allocator
	tree.nodes = make([dynamic]Node, 0, 2 * n_samples, allocator) // len=0, cap=2*n
	tree.root_idx = -1
	tree.classes = classes
	if codes != nil {
		tree.proba = make([dynamic]f64, 0, 2 * n_samples * len(classes), allocator)
	}

	indices := make([]int, n_samples, allocator)
	defer delete(indices, allocator)
	for i in 0 ..< n_samples {indices[i] = i}

	tree.root_idx = _dt_build(&tree, X, y, codes, indices, params, 0, allocator)
	return tree
}

// Regression node builder (kept for API compatibility)
dt_build_node :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	y: []f64,
	indices: []int,
	params: TreeParams,
	depth: int,
	allocator: mem.Allocator,
) -> int {
	return _dt_build(tree, X, y, nil, indices, params, depth, allocator)
}

_dt_build :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	y: []f64,
	codes: []int, // nil: regression
	indices: []int,
	params: TreeParams,
	depth: int,
	allocator: mem.Allocator,
) -> int {
	n := len(indices)
	classify := codes != nil
	K := len(tree.classes)

	// Create new node (index, not pointer: the recursion appends to tree.nodes)
	node_idx := len(tree.nodes)
	append(&tree.nodes, Node{feature_idx = -1, left_idx = -1, right_idx = -1, sample_count = n})

	// --- 1. Node statistics ---
	impurity, mean_y := 0.0, 0.0
	pure := false
	if classify {
		counts := make([]int, K, allocator)
		defer delete(counts, allocator)
		for idx in indices {counts[codes[idx]] += 1}
		base := len(tree.proba)
		resize(&tree.proba, base + K)
		best_k := 0
		for c, k in counts {
			tree.proba[base + k] = f64(c) / f64(n)
			if c > counts[best_k] do best_k = k
		}
		impurity = _class_impurity(counts, n, params.criterion)
		tree.nodes[node_idx].value = tree.classes[best_k]
		pure = counts[best_k] == n
	} else {
		// two-pass: mean, then variance
		sum_y := 0.0
		for idx in indices {sum_y += y[idx]}
		mean_y = sum_y / f64(n)
		scale := 0.0
		for idx in indices {
			d := y[idx] - mean_y
			impurity += d * d
			scale += y[idx] * y[idx]
		}
		impurity /= f64(n)
		scale /= f64(n)
		tree.nodes[node_idx].value = mean_y
		// pure node: variance at rounding level of y²
		pure = impurity <= 1e-14 * scale
	}
	tree.nodes[node_idx].impurity = impurity

	// --- 2. Stopping criteria ---
	if depth >= params.max_depth || n < params.min_samples || n < 2 || pure {
		return node_idx
	}

	// --- 3. Find best split ---
	best_feature := -1
	best_threshold := 0.0
	best_gain := 0.0
	n_features := X.cols

	order := make([]int, n_features, allocator)
	defer delete(order, allocator)
	for f in 0 ..< n_features {order[f] = f}
	k_feat := n_features
	if params.max_features > 0 && params.max_features < n_features {
		k_feat = params.max_features
		rand.shuffle(order)
	}

	for fi in 0 ..< n_features {
		// past the first k_feat features only keep looking while no split was found
		if fi >= k_feat && best_feature >= 0 do break
		f := order[fi]
		sorted := _sort_indices_by_feature(X, f, indices, allocator)
		defer delete(sorted, allocator)

		gain, threshold: f64
		ok: bool
		if classify {
			gain, threshold, ok = _find_best_class_split(X, codes, sorted, f, K, impurity, params.criterion, allocator)
		} else {
			// on y centred at the node mean: no cancellation
			gain, threshold, ok = _find_best_split_on_feature(X, y, sorted, f, mean_y, n, impurity)
		}

		// any valid split beats none, even at ~zero gain (sklearn: XOR-like data)
		if ok && (best_feature < 0 || gain > best_gain) {
			best_gain = gain
			best_feature = f
			best_threshold = threshold
		}
	}

	// --- 4. Split or leaf ---
	if best_feature < 0 {
		return node_idx
	}

	tree.nodes[node_idx].feature_idx = best_feature
	tree.nodes[node_idx].threshold = best_threshold

	// Partition indices
	left_idx, right_idx := _partition_indices(X, best_feature, best_threshold, indices, allocator)
	defer {
		delete(left_idx, allocator)
		delete(right_idx, allocator)
	}

	l_child := _dt_build(tree, X, y, codes, left_idx, params, depth + 1, allocator)
	r_child := _dt_build(tree, X, y, codes, right_idx, params, depth + 1, allocator)
	tree.nodes[node_idx].left_idx = l_child
	tree.nodes[node_idx].right_idx = r_child

	return node_idx
}

// ============================================================================
// Internal: Split Finding
// ============================================================================

_find_best_split_on_feature :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	sorted_indices: []int,
	feature: int,
	node_mean: f64,
	n: int,
	parent_variance: f64,
) -> (
	gain: f64,
	threshold: f64,
	found: bool,
) {
	// Running sums of y - node_mean (sums over the node are 0 and n·variance)
	total_sum := 0.0
	total_sum2 := 0.0
	for idx in sorted_indices {
		d := y[idx] - node_mean
		total_sum += d
		total_sum2 += d * d
	}

	best_gain := 0.0
	best_thresh := 0.0

	left_sum, left_sum2 := 0.0, 0.0
	for k in 0 ..< n - 1 {
		idx := sorted_indices[k]
		d := y[idx] - node_mean
		left_sum += d
		left_sum2 += d * d

		// Only split between distinct feature values
		curr_val := X.data[idx * X.cols + feature]
		next_val := X.data[sorted_indices[k + 1] * X.cols + feature]
		if curr_val == next_val {continue}

		n_l := f64(k + 1)
		n_r := f64(n - k - 1)
		right_sum := total_sum - left_sum
		right_sum2 := total_sum2 - left_sum2
		// n·impurity of each child = Σd² - (Σd)²/count
		weighted_impurity := ((left_sum2 - left_sum * left_sum / n_l) + (right_sum2 - right_sum * right_sum / n_r)) / f64(n)
		g := parent_variance - weighted_impurity

		if !found || g > best_gain {
			found = true
			best_gain = g
			best_thresh = _split_threshold(curr_val, next_val)
		}
	}

	return best_gain, best_thresh, found
}

// Gini / entropy split: gain = impurity(node) - (n_l·imp_l + n_r·imp_r) / n
_find_best_class_split :: proc(
	X: ^l.Matrix(f64),
	codes: []int,
	sorted_indices: []int,
	feature: int,
	n_classes: int,
	parent_impurity: f64,
	criterion: TreeCriterion,
	allocator: mem.Allocator,
) -> (
	gain: f64,
	threshold: f64,
	found: bool,
) {
	n := len(sorted_indices)
	left := make([]int, n_classes, allocator)
	right := make([]int, n_classes, allocator)
	defer {
		delete(left, allocator)
		delete(right, allocator)
	}
	for idx in sorted_indices {right[codes[idx]] += 1}

	for k in 0 ..< n - 1 {
		idx := sorted_indices[k]
		left[codes[idx]] += 1
		right[codes[idx]] -= 1

		curr_val := X.data[idx * X.cols + feature]
		next_val := X.data[sorted_indices[k + 1] * X.cols + feature]
		if curr_val == next_val {continue}

		n_l := k + 1
		n_r := n - n_l
		weighted :=
			(f64(n_l) * _class_impurity(left, n_l, criterion) + f64(n_r) * _class_impurity(right, n_r, criterion)) /
			f64(n)
		g := parent_impurity - weighted

		if !found || g > gain {
			found = true
			gain = g
			threshold = _split_threshold(curr_val, next_val)
		}
	}
	return
}

_class_impurity :: proc(counts: []int, n: int, criterion: TreeCriterion) -> f64 {
	if criterion == .Entropy {
		h := 0.0
		for c in counts {
			if c > 0 {
				p := f64(c) / f64(n)
				h -= p * math.log2(p)
			}
		}
		return h
	}
	s := 0.0
	for c in counts {
		p := f64(c) / f64(n)
		s += p * p
	}
	return 1 - s
}

// midpoint, but never rounded up onto next_val (it must go right)
_split_threshold :: proc(curr_val, next_val: f64) -> f64 {
	t := curr_val + (next_val - curr_val) / 2.0
	if t >= next_val do t = curr_val
	return t
}

// ============================================================================
// Internal: Sorting & Partitioning
// ============================================================================

// _sort_indices_by_feature :: proc(
// 	X: ^l.Matrix(f64),
// 	feature: int,
// 	indices: []int,
// 	allocator: mem.Allocator,
// ) -> []int {
// 	out := make([]int, len(indices), allocator)
// 	copy(out, indices)

// 	// Simple insertion sort (replace with quicksort for large data)
// 	for i in 1 ..< len(out) {
// 		key := out[i]
// 		key_val := X.data[key * X.cols + feature]
// 		j := i - 1

// 		for j >= 0 && X.data[out[j] * X.cols + feature] > key_val {
// 			out[j + 1] = out[j]
// 			j -= 1
// 		}
// 		out[j + 1] = key
// 	}

// 	return out
// }


// Replace insertion sort with quicksort
_sort_indices_by_feature :: proc(
	X: ^l.Matrix(f64),
	feature: int,
	indices: []int,
	allocator: mem.Allocator,
) -> []int {
	// Create a temporary buffer of feature values for the current indices
	values := make([]f64, len(indices), allocator)
	defer delete(values, allocator)

	for idx, i in indices {
		values[i] = X.data[idx * X.cols + feature]
	}

	// Use slice.sort_with_indices: returns permutation p such that
	// values[p[0]] <= values[p[1]] <= ...
	sorted_perm := slice.sort_with_indices(values[:], allocator)
	defer delete(sorted_perm, allocator)

	// Apply the permutation: out[i] = indices[sorted_perm[i]]
	out := make([]int, len(indices), allocator)
	for i in 0 ..< len(sorted_perm) {
		out[i] = indices[sorted_perm[i]]
	}

	return out
}


_partition_indices :: proc(
	X: ^l.Matrix(f64),
	feature: int,
	threshold: f64,
	indices: []int,
	allocator: mem.Allocator,
) -> (
	[]int,
	[]int,
) { 	// Return two slices

	// Use dynamic arrays for flexible growth
	left_dyn := make([dynamic]int, 0, len(indices), allocator)
	right_dyn := make([dynamic]int, 0, len(indices), allocator)

	for idx in indices {
		if X.data[idx * X.cols + feature] <= threshold {
			append(&left_dyn, idx)
		} else {
			append(&right_dyn, idx)
		}
	}

	// Convert dynamic arrays to slices for return
	left := make([]int, len(left_dyn), allocator)
	right := make([]int, len(right_dyn), allocator)
	copy(left, left_dyn[:])
	copy(right, right_dyn[:])

	// Clean up dynamic arrays
	delete(left_dyn)
	delete(right_dyn)

	return left, right
}

// ============================================================================
// Internal: Prediction
// ============================================================================

dt_predict_single :: proc(tree: ^DecisionTree, X: ^l.Matrix(f64), sample_idx: int) -> f64 {
	leaf := _dt_leaf_index(tree, X, sample_idx)
	return tree.nodes[leaf].value if leaf >= 0 else 0.0
}

// Index of the leaf that row sample_idx of X falls into (-1 for an empty tree)
_dt_leaf_index :: proc(tree: ^DecisionTree, X: ^l.Matrix(f64), sample_idx: int) -> int {
	curr_idx := tree.root_idx
	for curr_idx >= 0 {
		node := &tree.nodes[curr_idx]
		if node.left_idx < 0 do return curr_idx
		if X.data[sample_idx * X.cols + node.feature_idx] <= node.threshold {
			curr_idx = node.left_idx
		} else {
			curr_idx = node.right_idx
		}
	}
	return -1
}

// ============================================================================
// Cleanup
// ============================================================================

dt_free :: proc(tree: ^DecisionTree) {
	delete(tree.nodes) // Odin: delete dynamic array (no allocator needed)
	delete(tree.proba)
	delete(tree.classes, tree.allocator)
	tree.nodes, tree.proba, tree.classes = nil, nil, nil
}
// Add batch prediction function
dt_predict_batch :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator,
) -> []f64 {
	n := X.rows
	out := make([]f64, n, allocator)

	// For each leaf, find which samples reach it
	// (more complex but enables SIMD traversal)

	// Simple version: still loop but use SIMD for feature comparisons
	for i in 0 ..< n {
		out[i] = dt_predict_single(tree, X, i)
	}
	return out
}
