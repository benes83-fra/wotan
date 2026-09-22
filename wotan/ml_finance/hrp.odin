package ml_finance

import l "../linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// ============================================================================
// HRP Data Structures
// ============================================================================

HRPResult :: struct {
	weights:       []f64,
	cluster_order: []int, // Quasi-diagonalized leaf order
	allocator:     mem.Allocator,
}

hrp_result_free :: proc(res: ^HRPResult) {
	if res.weights != nil {delete(res.weights, res.allocator)}
	if res.cluster_order != nil {delete(res.cluster_order, res.allocator)}
}

// ============================================================================
// Step 1 & 2 & 3: Full HRP Pipeline
// ============================================================================

// hrp_allocate computes Hierarchical Risk Parity weights.
// returns: A matrix of shape [T, N] where T is time steps and N is assets.
hrp_allocate :: proc(
	returns: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
	limit: bool = true,
) -> HRPResult {
	n_assets := returns.cols
	if n_assets == 0 {
		return HRPResult{allocator = allocator}
	}

	// 1. Compute Covariance and Correlation Matrices using linalg
	cov_mat := l.covariance(returns, allocator)
	defer l.matrix_free(&cov_mat)

	corr_mat := l.correlation(returns, allocator)
	defer l.matrix_free(&corr_mat)

	// 2. Compute Distance Matrix: D_ij = sqrt(0.5 * (1 - rho_ij))
	// We use a flat array for the distance matrix to allow dynamic expansion during clustering
	max_nodes := 2 * n_assets - 1
	full_dist := make([]f64, max_nodes * max_nodes, allocator)
	defer delete(full_dist, allocator)

	for i in 0 ..< n_assets {
		for j in 0 ..< n_assets {
			rho := corr_mat.data[i * n_assets + j]
			// Clamp rho to [-1, 1] to prevent NaN from floating point drift
			if rho > 1.0 {rho = 1.0}
			if rho < -1.0 {rho = -1.0}
			dist := math.sqrt(0.5 * (1.0 - rho))
			full_dist[i * max_nodes + j] = dist
		}
	}

	// 3. Hierarchical Agglomerative Clustering (Single Linkage)
	n_merges := n_assets - 1
	left_child := make([]int, n_merges, allocator)
	right_child := make([]int, n_merges, allocator)
	defer {delete(left_child, allocator); delete(right_child, allocator)}

	active := make([]bool, max_nodes, allocator)
	defer delete(active, allocator)
	for i in 0 ..< n_assets {active[i] = true}

	for step in 0 ..< n_merges {
		min_d := math.F64_MAX
		best_i, best_j := -1, -1

		// Find the two closest active clusters
		for i in 0 ..< n_assets + step {
			if !active[i] {continue}
			for j in i + 1 ..< n_assets + step {
				if !active[j] {continue}
				d := full_dist[i * max_nodes + j]
				if d < min_d {
					min_d = d
					best_i = i
					best_j = j
				}
			}
		}

		// Merge best_i and best_j into a new node
		new_node := n_assets + step
		active[best_i] = false
		active[best_j] = false
		active[new_node] = true

		left_child[step] = best_i
		right_child[step] = best_j

		// Update distances using Single Linkage: D_new,k = min(D_i,k, D_j,k)
		for k in 0 ..< max_nodes {
			if !active[k] || k == new_node {continue}

			d_ik := full_dist[best_i * max_nodes + k]
			d_jk := full_dist[best_j * max_nodes + k]
			new_d := math.min(d_ik, d_jk)

			full_dist[new_node * max_nodes + k] = new_d
			full_dist[k * max_nodes + new_node] = new_d
		}
	}

	// 4. Quasi-Diagonalization (Get Leaf Order via DFS)
	root_node := n_assets + n_merges - 1
	order: [dynamic]int
	stack: [dynamic]int
	append(&stack, root_node)

	for len(stack) > 0 {
		node := stack[len(stack) - 1]
		pop(&stack) // pop

		if node < n_assets {
			append(&order, node)
		} else {
			step := node - n_assets
			// Push right then left so left is processed first (DFS)
			append(&stack, right_child[step])
			append(&stack, left_child[step])
		}
	}
	delete(stack)

	// 5. Recursive Bisection
	weights := make([]f64, n_assets, allocator)
	for i in 0 ..< n_assets {weights[i] = 1.0}

	hrp_bisect(root_node, n_assets, left_child, right_child, cov_mat.data, weights, allocator)
	// In hrp_allocate, after the recursive bisection and normalization:

	// ✅ Max-weight constraint (e.g., no single asset > 30%)
	if limit {
		max_weight := 0.30
		for iter in 0 ..< 10 { 	// Iterate to redistribute excess
			excess := 0.0
			n_uncapped := 0
			for i in 0 ..< n_assets {
				if weights[i] > max_weight {
					excess += weights[i] - max_weight
					weights[i] = max_weight
				} else {
					n_uncapped += 1
				}
			}
			if excess < 1e-8 {break}
			// Redistribute excess to uncapped assets proportionally
			if n_uncapped > 0 {
				redistribute := excess / f64(n_uncapped)
				for i in 0 ..< n_assets {
					if weights[i] < max_weight {
						weights[i] += redistribute
					}
				}
			}
		}
	}

	// Normalize weights to sum to 1.0
	sum_w := 0.0
	for w in weights {sum_w += w}
	if sum_w > 1e-10 {
		for i in 0 ..< n_assets {weights[i] /= sum_w}
	}

	// Convert order to standard slice for the result
	cluster_order := make([]int, len(order), allocator)
	copy(cluster_order, order[:])
	delete(order)

	return HRPResult{weights = weights, cluster_order = cluster_order, allocator = allocator}
}

// ============================================================================
// Recursive Bisection Helper
// ============================================================================

hrp_bisect :: proc(
	node: int,
	n_assets: int,
	left_child, right_child: []int,
	cov: []f64,
	weights: []f64,
	alloc: mem.Allocator,
) {
	if node < n_assets {
		return // Leaf node, nothing to split
	}

	step := node - n_assets
	left_node := left_child[step]
	right_node := right_child[step]

	// Get leaves for left and right clusters
	left_leaves := _hrp_get_leaves(left_node, n_assets, left_child, right_child, alloc)
	defer delete(left_leaves)

	right_leaves := _hrp_get_leaves(right_node, n_assets, left_child, right_child, alloc)
	defer delete(right_leaves)

	// Calculate cluster variances: V = w^T * Sigma * w
	var_left := _hrp_cluster_variance(left_leaves, weights, cov, n_assets)
	var_right := _hrp_cluster_variance(right_leaves, weights, cov, n_assets)

	// Allocation factor (inversely proportional to variance)
	// alpha = 1 - (V_left / (V_left + V_right))
	alpha := 1.0
	denom := var_left + var_right
	if denom > 1e-10 {
		alpha = 1.0 - (var_left / denom)
	}

	// Adjust weights down the tree
	for idx in left_leaves {
		weights[idx] *= alpha
	}
	for idx in right_leaves {
		weights[idx] *= (1.0 - alpha)
	}

	// Recurse
	hrp_bisect(left_node, n_assets, left_child, right_child, cov, weights, alloc)
	hrp_bisect(right_node, n_assets, left_child, right_child, cov, weights, alloc)
}

// ============================================================================
// Tree Traversal & Math Helpers
// ============================================================================

_hrp_get_leaves :: proc(
	node: int,
	n_assets: int,
	left_child, right_child: []int,
	alloc: mem.Allocator,
) -> [dynamic]int {
	leaves: [dynamic]int
	stack: [dynamic]int
	append(&stack, node)

	for len(stack) > 0 {
		curr := stack[len(stack) - 1]
		pop(&stack)


		if curr < n_assets {
			append(&leaves, curr)
		} else {
			s := curr - n_assets
			append(&stack, right_child[s])
			append(&stack, left_child[s])
		}
	}
	delete(stack)
	return leaves
}

_hrp_cluster_variance :: proc(
	leaves: [dynamic]int,
	weights: []f64,
	cov: []f64,
	n_assets: int,
) -> f64 {
	variance := 0.0
	for i in leaves {
		for j in leaves {
			variance += weights[i] * weights[j] * cov[i * n_assets + j]
		}
	}
	return variance
}
