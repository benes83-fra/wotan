package ML

import l "../../linalg"
import "core:math"
import "core:mem"

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

TreeParams :: struct {
	max_depth:   int,
	min_samples: int,
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

	// Create new node
	node_idx := len(tree.nodes)
	append(&tree.nodes, Node{}) // Odin: append to [dynamic] works
	node := &tree.nodes[node_idx]

	// --- 1. Compute node statistics ---
	sum_y := 0.0
	sum_y2 := 0.0
	for k in 0 ..< n {
		val := y[indices[k]]
		sum_y += val
		sum_y2 += val * val
	}

	mean_y := sum_y / f64(n)
	variance := (sum_y2 / f64(n)) - (mean_y * mean_y)

	node.impurity = variance
	node.sample_count = n
	node.value = mean_y
	node.left_idx = -1
	node.right_idx = -1

	// --- 2. Stopping criteria ---
	if depth >= params.max_depth || n < params.min_samples || variance < 1e-10 {
		return node_idx
	}

	// --- 3. Find best split ---
	best_feature := -1
	best_threshold := 0.0
	best_gain := -1.0
	n_features := X.cols

	for f in 0 ..< n_features {
		sorted := _sort_indices_by_feature(X, f, indices, allocator)
		defer delete(sorted, allocator)

		gain, threshold := _find_best_split_on_feature(X, y, sorted, f, sum_y, sum_y2, n, variance)

		if gain > best_gain {
			best_gain = gain
			best_feature = f
			best_threshold = threshold
		}
	}

	// --- 4. Split or leaf ---
	if best_feature < 0 || best_gain <= 0.0 {
		return node_idx
	}

	node.feature_idx = best_feature
	node.threshold = best_threshold

	// Partition indices
	left_idx, right_idx := _partition_indices(X, best_feature, best_threshold, indices, allocator)
	defer {
		delete(left_idx, allocator)
		delete(right_idx, allocator)
	}

	node.left_idx = dt_build_node(tree, X, y, left_idx, params, depth + 1, allocator)
	node.right_idx = dt_build_node(tree, X, y, right_idx, params, depth + 1, allocator)

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
	total_sum_y: f64,
	total_sum_y2: f64,
	n: int,
	parent_variance: f64,
) -> (
	gain: f64,
	threshold: f64,
) {

	best_gain := -1.0
	best_thresh := 0.0

	left_sum_y := 0.0
	left_sum_y2 := 0.0
	left_count := 0

	right_sum_y := total_sum_y
	right_sum_y2 := total_sum_y2
	right_count := n

	for k in 0 ..< n - 1 {
		idx := sorted_indices[k]
		val := y[idx]

		left_sum_y += val
		left_sum_y2 += val * val
		left_count += 1

		right_sum_y -= val
		right_sum_y2 -= val * val
		right_count -= 1

		if left_count < 1 || right_count < 1 {continue}

		// Skip identical feature values
		curr_val := X.data[idx * X.cols + feature]
		next_idx := sorted_indices[k + 1]
		next_val := X.data[next_idx * X.cols + feature]
		if curr_val == next_val {continue}

		// Compute variances
		left_mean := left_sum_y / f64(left_count)
		left_var := (left_sum_y2 / f64(left_count)) - (left_mean * left_mean)

		right_mean := right_sum_y / f64(right_count)
		right_var := (right_sum_y2 / f64(right_count)) - (right_mean * right_mean)

		weighted_impurity := (f64(left_count) * left_var + f64(right_count) * right_var) / f64(n)
		gain := parent_variance - weighted_impurity

		if gain > best_gain {
			best_gain = gain
			best_thresh = (curr_val + next_val) / 2.0
		}
	}

	return best_gain, best_thresh
}

// ============================================================================
// Internal: Sorting & Partitioning
// ============================================================================

_sort_indices_by_feature :: proc(
	X: ^l.Matrix(f64),
	feature: int,
	indices: []int,
	allocator: mem.Allocator,
) -> []int {
	out := make([]int, len(indices), allocator)
	copy(out, indices)

	// Simple insertion sort (replace with quicksort for large data)
	for i in 1 ..< len(out) {
		key := out[i]
		key_val := X.data[key * X.cols + feature]
		j := i - 1

		for j >= 0 && X.data[out[j] * X.cols + feature] > key_val {
			out[j + 1] = out[j]
			j -= 1
		}
		out[j + 1] = key
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
