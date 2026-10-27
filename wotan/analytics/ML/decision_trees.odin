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
	allocator: mem.Allocator,
}

Node :: struct {
	// Split info (internal nodes)
	feature_idx:  int,
	threshold:    f64,

	// Prediction info (leaf nodes)
	value:        f64,

	// Tree structure
	left_idx:     int, // -1 if leaf
	right_idx:    int, // -1 if leaf

	// Statistics
	impurity:     f64,
	sample_count: int,
}

// Regression tree (MSE criterion, mean leaves), as sklearn DecisionTreeRegressor:
// split at the midpoint of consecutive distinct sorted values, x <= t goes left.
// Equal-gain splits go to the lowest feature index / first threshold (sklearn
// breaks such ties by its random feature order, so deep trees can differ there).
TreeParams :: struct {
	max_depth:    int, // a node at this depth becomes a leaf (0 = single leaf)
	min_samples:  int, // min samples to split a node (sklearn min_samples_split)
	max_features: int, // features drawn per split (0 or >= n_features = all), like sklearn:
	// if none of them gives a valid split, the remaining features are tried too
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

	tree: DecisionTree
	tree.allocator = allocator
	tree.nodes = make([dynamic]Node, 0, 2 * n_samples, allocator) // len=0, cap=2*n
	tree.root_idx = -1

	// Initial indices
	indices := make([]int, n_samples, allocator)
	defer delete(indices, allocator)
	for i in 0 ..< n_samples {indices[i] = i}

	// Build tree
	root_idx := dt_build_node(&tree, X, y, indices, params, 0, allocator)
	tree.root_idx = root_idx

	return tree
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

// ============================================================================
// Internal: Tree Building
// ============================================================================

dt_build_node :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	y: []f64,
	indices: []int,
	params: TreeParams,
	depth: int,
	allocator: mem.Allocator,
) -> int {
	n := len(indices)

	// Create new node (index, not pointer: the recursion appends to tree.nodes)
	node_idx := len(tree.nodes)
	append(&tree.nodes, Node{})

	// --- 1. Node statistics (two-pass: mean, then variance) ---
	sum_y := 0.0
	for k in 0 ..< n {
		sum_y += y[indices[k]]
	}
	mean_y := sum_y / f64(n)
	variance := 0.0
	scale := 0.0
	for k in 0 ..< n {
		d := y[indices[k]] - mean_y
		variance += d * d
		scale += y[indices[k]] * y[indices[k]]
	}
	variance /= f64(n)
	scale /= f64(n)

	tree.nodes[node_idx] = Node {
		impurity     = variance,
		sample_count = n,
		value        = mean_y,
		feature_idx  = -1,
		left_idx     = -1,
		right_idx    = -1,
	}

	// --- 2. Stopping criteria (pure node: variance at rounding level of y²) ---
	if depth >= params.max_depth || n < params.min_samples || n < 2 || variance <= 1e-14 * scale {
		return node_idx
	}

	// --- 3. Find best split (on y centered at the node mean: no cancellation) ---
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

		gain, threshold, ok := _find_best_split_on_feature(X, y, sorted, f, mean_y, n, variance)

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

	l_child := dt_build_node(tree, X, y, left_idx, params, depth + 1, allocator)
	r_child := dt_build_node(tree, X, y, right_idx, params, depth + 1, allocator)
	tree.nodes[node_idx].left_idx = l_child
	tree.nodes[node_idx].right_idx = r_child

	return node_idx
}

// ============================================================================
// Internal: Split Finding (Optimized with SIMD)
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
			// midpoint, but never rounded up onto next_val (it must go right)
			t := curr_val + (next_val - curr_val) / 2.0
			if t >= next_val do t = curr_val
			best_thresh = t
		}
	}

	return best_gain, best_thresh, found
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
	curr_idx := tree.root_idx

	for curr_idx >= 0 {
		node := &tree.nodes[curr_idx]

		if node.left_idx < 0 {
			// Leaf node
			return node.value
		}

		// Internal node: traverse
		feat_val := X.data[sample_idx * X.cols + node.feature_idx]
		if feat_val <= node.threshold {
			curr_idx = node.left_idx
		} else {
			curr_idx = node.right_idx
		}
	}

	return 0.0 // Should not reach here
}

// ============================================================================
// Cleanup
// ============================================================================

dt_free :: proc(tree: ^DecisionTree) {
	if len(tree.nodes) > 0 {
		delete(tree.nodes) // Odin: delete dynamic array (no allocator needed)
	}
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
