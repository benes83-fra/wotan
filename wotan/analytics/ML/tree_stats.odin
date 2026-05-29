package ML

import l "../../linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// Tree-Based Model Statistics
// ============================================================================

TreeStats :: struct {
	// Basic fit metrics
	n_samples:          int,
	n_features:         int,
	n_nodes:            int,
	n_leaves:           int,

	// Prediction quality
	mse:                f64,
	rmse:               f64,
	mae:                f64,
	r2:                 f64,
	r2_adj:             f64,

	// Residuals
	residuals:          []f64,
	fitted:             []f64,

	// Feature importance
	feature_importance: []f64,

	// Tree structure stats
	avg_depth:          f64,
	max_depth:          int,
	splits_per_feature: []int,

	// Memory
	total_memory_bytes: int,
}

// Compute regression metrics for any model
tree_compute_metrics :: proc(
	y_true: []f64,
	y_pred: []f64,
	n_features: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	mse: f64,
	rmse: f64,
	mae: f64,
	r2: f64,
	r2_adj: f64,
	residuals: []f64,
	fitted: []f64,
) {
	n := len(y_true)
	if n != len(y_pred) do panic("tree_compute_metrics: length mismatch")

	residuals = make([]f64, n, allocator) // ✅ Fixed: slice make doesn't need allocator arg here
	fitted = make([]f64, n, allocator)
	for i in 0 ..< n {
		residuals[i] = y_true[i] - y_pred[i]
		fitted[i] = y_pred[i]
	}

	sse := 0.0
	sae := 0.0
	for i in 0 ..< n {
		err := residuals[i]
		sse += err * err
		sae += math.abs(err)
	}
	mse = sse / f64(n)
	rmse = math.sqrt(mse)
	mae = sae / f64(n)

	mean_y := 0.0
	for val in y_true {mean_y += val}
	mean_y /= f64(n)

	ss_tot := 0.0
	for val in y_true {
		dy := val - mean_y
		ss_tot += dy * dy
	}

	if ss_tot > 1e-10 {
		r2 = 1.0 - sse / ss_tot
	} else {
		r2 = 0.0
	}

	if n > n_features + 1 {
		r2_adj = 1.0 - (1.0 - r2) * f64(n - 1) / f64(n - n_features - 1)
	} else {
		r2_adj = r2
	}

	return
}

// ============================================================================
// FIXED: Standalone helper procs (no nested captures)
// ============================================================================

// Helper for feature importance traversal (standalone, all params explicit)
_traverse_for_importance_helper :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	y: []f64,
	node_idx: int,
	samples: []int,
	importance: []f64,
	allocator: mem.Allocator,
) {
	node := &tree.nodes[node_idx] // ✅ Now tree is explicit param
	n := len(samples)

	if node.left_idx < 0 {
		return
	}

	// Compute impurity
	sum_y := 0.0
	sum_y2 := 0.0
	for idx in samples {
		val := y[idx] // ✅ y is explicit param
		sum_y += val
		sum_y2 += val * val
	}
	mean_y := sum_y / f64(n)
	node_impurity := (sum_y2 / f64(n)) - (mean_y * mean_y)

	// Partition samples
	left_samples := make([dynamic]int, 0, allocator)
	right_samples := make([dynamic]int, 0, allocator)
	defer {
		delete(left_samples) // ✅ [dynamic] delete: no allocator
		delete(right_samples)
	}

	feature := node.feature_idx
	threshold := node.threshold
	for idx in samples {
		if X.data[idx * X.cols + feature] <= threshold { 	// ✅ X is explicit
			append(&left_samples, idx)
		} else {
			append(&right_samples, idx)
		}
	}

	n_left := len(left_samples)
	n_right := len(right_samples)

	if n_left > 0 && n_right > 0 {
		// Simplified gain calculation
		gain := node_impurity // Placeholder - expand as needed
		if feature >= 0 && feature < len(importance) {
			importance[feature] += f64(n) * gain // ✅ importance is explicit
		}
	}

	if n_left > 0 {
		_traverse_for_importance_helper(
			tree,
			X,
			y,
			node.left_idx,
			left_samples[:],
			importance,
			allocator,
		)
	}
	if n_right > 0 {
		_traverse_for_importance_helper(
			tree,
			X,
			y,
			node.right_idx,
			right_samples[:],
			importance,
			allocator,
		)
	}
}

_tree_feature_importance :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	y: []f64,
	indices: []int,
	allocator: mem.Allocator,
) -> []f64 {
	n_features := X.cols
	importance := make([]f64, n_features, allocator) // ✅ Fixed: slice make

	if tree.root_idx >= 0 {
		_traverse_for_importance_helper(tree, X, y, tree.root_idx, indices, importance, allocator)
	}

	total := 0.0
	for val in importance {total += val}
	if total > 1e-10 {
		for i in 0 ..< n_features {
			importance[i] /= total
		}
	}

	return importance
}

// Helper for structure stats traversal
_traverse_stats_helper :: proc(
	tree: ^DecisionTree,
	node_idx: int,
	depth: int,
	n_features: int,
	splits_per_feature: []int, // ✅ Explicit param
) -> (
	nodes: int,
	leaves: int,
	depth_sum: int,
	max_d: int,
) {
	node := &tree.nodes[node_idx]

	if node.left_idx < 0 {
		return 1, 1, depth, depth
	}

	if node.feature_idx >= 0 && node.feature_idx < n_features {
		splits_per_feature[node.feature_idx] += 1 // ✅ Explicit param
	}

	left_n, left_l, left_ds, left_md := _traverse_stats_helper(
		tree,
		node.left_idx,
		depth + 1,
		n_features,
		splits_per_feature,
	)
	right_n, right_l, right_ds, right_md := _traverse_stats_helper(
		tree,
		node.right_idx,
		depth + 1,
		n_features,
		splits_per_feature,
	)

	return 1 + left_n + right_n, left_l + right_l, left_ds + right_ds, max(left_md, right_md)
}

_tree_structure_stats :: proc(
	tree: ^DecisionTree,
	n_features: int,
) -> (
	n_nodes: int,
	n_leaves: int,
	avg_depth: f64,
	max_depth: int,
	splits_per_feature: []int,
) {
	splits_per_feature = make([]int, n_features, context.temp_allocator) // ✅ Fixed

	if tree.root_idx < 0 {
		return 0, 0, 0.0, 0, splits_per_feature
	}

	total_nodes, total_leaves, total_depth_sum, max_d := _traverse_stats_helper(
		tree,
		tree.root_idx,
		0,
		n_features,
		splits_per_feature,
	)

	avg_d := f64(total_depth_sum) / f64(total_leaves)

	return total_nodes, total_leaves, avg_d, max_d, splits_per_feature
}

// ============================================================================
// FIXED: rf_compute_stats with correct iteration and memory management
// ============================================================================

rf_compute_stats :: proc(
	forest: ^RandomForest,
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> TreeStats {
	n_samples := X.rows
	n_features := X.cols
	n_trees := len(forest.trees)

	fitted := rf_predict(forest, X, allocator)
	defer delete(fitted, allocator) // ✅ slice delete needs allocator

	mse, rmse, mae, r2, r2_adj, residuals, fitted_copy := tree_compute_metrics(
		y,
		fitted,
		n_features,
		allocator,
	)

	avg_importance := make([]f64, n_features, allocator)

	// ✅ FIXED: Use index-based loop to get pointer to tree element
	for t in 0 ..< n_trees {
		tree_ptr := &forest.trees[t] // ✅ Get pointer to element in array

		all_indices := make([]int, n_samples, allocator)
		for i in 0 ..< n_samples {all_indices[i] = i}

		imp := _tree_feature_importance(tree_ptr, X, y, all_indices, allocator)

		for i in 0 ..< n_features {
			avg_importance[i] += imp[i]
		}
		delete(imp, allocator)
		delete(all_indices, allocator)
	}

	for i in 0 ..< n_features {
		avg_importance[i] /= f64(n_trees)
	}
	total := 0.0
	for val in avg_importance {total += val}
	if total > 1e-10 {
		for i in 0 ..< n_features {
			avg_importance[i] /= total
		}
	}

	total_nodes := 0
	total_leaves := 0
	total_depth := 0.0
	max_depth := 0

	// ✅ FIXED: Index-based loop for tree pointers
	for t in 0 ..< n_trees {
		tree_ptr := &forest.trees[t]
		n_n, n_l, avg_d, max_d_t, _ := _tree_structure_stats(tree_ptr, n_features)
		total_nodes += n_n
		total_leaves += n_l
		total_depth += avg_d
		if max_d_t > max_depth {max_depth = max_d_t}
	}
	avg_depth := total_depth / f64(n_trees)

	memory_bytes := 0
	for t in 0 ..< n_trees {
		memory_bytes += len(forest.trees[t].nodes) * size_of(Node) // ✅ Access via index
	}
	memory_bytes += n_samples * size_of(f64) * 2

	stats := TreeStats {
		n_samples          = n_samples,
		n_features         = n_features,
		n_nodes            = total_nodes,
		n_leaves           = total_leaves,
		mse                = mse,
		rmse               = rmse,
		mae                = mae,
		r2                 = r2,
		r2_adj             = r2_adj,
		residuals          = residuals,
		fitted             = fitted_copy,
		feature_importance = avg_importance,
		avg_depth          = avg_depth,
		max_depth          = max_depth,
		splits_per_feature = make([]int, n_features, allocator),
		total_memory_bytes = memory_bytes,
	}

	return stats
}
// Compute stats for a single Decision Tree
dt_compute_stats :: proc(
	tree: ^DecisionTree,
	X: ^l.Matrix(f64),
	y: []f64,
	allocator: mem.Allocator = context.allocator,
) -> TreeStats {
	n_samples := X.rows
	n_features := X.cols

	// Get predictions
	fitted := dt_predict(tree, X, allocator)
	defer delete(fitted, allocator)

	// Compute basic metrics
	mse, rmse, mae, r2, r2_adj, residuals, fitted_copy := tree_compute_metrics(
		y,
		fitted,
		n_features,
		allocator,
	)

	// Feature importance: create indices slice properly
	indices := make([]int, n_samples, allocator)
	for i in 0 ..< n_samples {
		indices[i] = i
	}
	importance := _tree_feature_importance(tree, X, y, indices, allocator)
	delete(indices, allocator) // Clean up temporary slice

	// Tree structure stats
	n_nodes, n_leaves, avg_depth, max_depth, splits_per_feat := _tree_structure_stats(
		tree,
		n_features,
	)

	// Memory estimate
	memory_bytes := len(tree.nodes) * size_of(Node) + n_samples * size_of(f64) * 2

	stats := TreeStats {
		n_samples          = n_samples,
		n_features         = n_features,
		n_nodes            = n_nodes,
		n_leaves           = n_leaves,
		mse                = mse,
		rmse               = rmse,
		mae                = mae,
		r2                 = r2,
		r2_adj             = r2_adj,
		residuals          = residuals,
		fitted             = fitted_copy,
		feature_importance = importance,
		avg_depth          = avg_depth,
		max_depth          = max_depth,
		splits_per_feature = splits_per_feat,
		total_memory_bytes = memory_bytes,
	}

	return stats
}
// Pretty-print
tree_stats_print :: proc(stats: ^TreeStats) {
	fmt.println("=== TREE MODEL STATISTICS ===")
	fmt.printf("Samples: %v\n", stats.n_samples)
	fmt.printf("Features: %v\n", stats.n_features)
	fmt.printf("Nodes/Leaves: %v / %v\n", stats.n_nodes, stats.n_leaves)
	fmt.println()
	fmt.println("Prediction Quality:")
	fmt.printf("  MSE: %.6f\n", stats.mse)
	fmt.printf("  RMSE: %.6f\n", stats.rmse)
	fmt.printf("  MAE: %.6f\n", stats.mae)
	fmt.printf("  R²: %.6f\n", stats.r2)
	fmt.printf("  R²_adj: %.6f\n", stats.r2_adj)
	fmt.println("=== END TREE STATS ===")
}
