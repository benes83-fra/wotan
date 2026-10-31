package ml_finance

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Hierarchical Risk Parity (López de Prado 2016)
// ============================================================================
// 1. Correlation distance d_ij = √(½(1 − ρ_ij)).
// 2. Single-linkage clustering, by default on the distance between the
//    distance columns d̃_ij = ‖d_·i − d_·j‖ (the paper's step; PyPortfolioOpt
//    clusters on d itself: .Correlation_Distance). Merges are numbered as in
//    scipy's linkage (cluster n + k from step k, smaller id first).
// 3. Quasi-diagonalization: the dendrogram's leaves in order.
// 4. Recursive bisection of that ordered list into halves; each half gets a
//    share inversely proportional to the variance of its inverse-variance
//    portfolio, α = 1 − V₀/(V₀ + V₁).

HRPResult :: struct {
	weights:       []f64,
	cluster_order: []int, // Quasi-diagonalized leaf order
	allocator:     mem.Allocator,
}

HRPLinkageInput :: enum u8 {
	Distance_Of_Distances, // d̃, as in the paper
	Correlation_Distance, // d
}

hrp_result_free :: proc(res: ^HRPResult) {
	if res.weights != nil {delete(res.weights, res.allocator)}
	if res.cluster_order != nil {delete(res.cluster_order, res.allocator)}
}

// HRP weights from a matrix of returns [T, N] (sample covariance). With
// `limit`, no weight exceeds 30% (the excess is spread over the other assets;
// a heuristic outside HRP, off by default).
hrp_allocate :: proc(
	returns: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
	limit: bool = false,
	linkage: HRPLinkageInput = .Distance_Of_Distances,
) -> HRPResult {
	if returns.cols == 0 {return HRPResult{allocator = allocator}}
	cov := l.covariance(returns, context.temp_allocator)
	defer l.matrix_free(&cov)
	res := hrp_allocate_cov(&cov, allocator, linkage)
	if limit {_hrp_cap(res.weights, 0.30)}
	return res
}

// HRP weights from a covariance matrix
hrp_allocate_cov :: proc(
	cov: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
	linkage: HRPLinkageInput = .Distance_Of_Distances,
) -> HRPResult {
	n := cov.rows
	if n == 0 {return HRPResult{allocator = allocator}}
	C := cov.data

	// 1. correlation distance
	d := make([]f64, n * n, context.temp_allocator)
	defer delete(d, context.temp_allocator)
	for i in 0 ..< n {
		for j in 0 ..< n {
			rho := C[i * n + j] / math.sqrt(C[i * n + i] * C[j * n + j])
			d[i * n + j] = math.sqrt(0.5 * (1.0 - clamp(rho, -1.0, 1.0)))
		}
	}

	// 2. single linkage on the chosen distance; nodes 0..n-1 leaves, n+k merges
	m := 2 * n - 1
	D := make([]f64, m * m, context.temp_allocator)
	defer delete(D, context.temp_allocator)
	for i in 0 ..< n {
		for j in 0 ..< n {
			if linkage == .Correlation_Distance {
				D[i * m + j] = d[i * n + j]
			} else {
				s := 0.0
				for k in 0 ..< n {
					x := d[k * n + i] - d[k * n + j]
					s += x * x
				}
				D[i * m + j] = math.sqrt(s)
			}
		}
	}
	left := make([]int, max(n - 1, 0), context.temp_allocator)
	right := make([]int, max(n - 1, 0), context.temp_allocator)
	active := make([]bool, m, context.temp_allocator)
	defer {
		delete(left, context.temp_allocator)
		delete(right, context.temp_allocator)
		delete(active, context.temp_allocator)
	}
	for i in 0 ..< n {active[i] = true}
	for step in 0 ..< n - 1 {
		best, bi, bj := math.F64_MAX, -1, -1
		for i in 0 ..< n + step {
			if !active[i] {continue}
			for j in i + 1 ..< n + step {
				if active[j] && D[i * m + j] < best {best, bi, bj = D[i * m + j], i, j}
			}
		}
		node := n + step
		active[bi], active[bj], active[node] = false, false, true
		left[step], right[step] = bi, bj
		for k in 0 ..< node {
			if !active[k] {continue}
			v := min(D[bi * m + k], D[bj * m + k])
			D[node * m + k], D[k * m + node] = v, v
		}
	}

	// 3. leaf order (left before right)
	order := make([]int, n, allocator)
	{
		stack := make([dynamic]int, 0, n, context.temp_allocator)
		defer delete(stack)
		append(&stack, 2 * n - 2)
		k := 0
		for len(stack) > 0 {
			node := pop(&stack)
			if node < n {
				order[k] = node
				k += 1
			} else {
				append(&stack, right[node - n], left[node - n])
			}
		}
	}

	// 4. recursive bisection of the ordered list
	weights := make([]f64, n, allocator)
	for &x in weights {x = 1}
	_hrp_bisect(order, C, n, weights)

	return HRPResult{weights = weights, cluster_order = order, allocator = allocator}
}

// Variance of the inverse-variance portfolio of the assets `items`
@(private = "file")
_hrp_cluster_variance :: proc(items: []int, C: []f64, n: int) -> f64 {
	s := 0.0
	for i in items {s += 1.0 / C[i * n + i]}
	v := 0.0
	for i in items {
		for j in items {v += C[i * n + j] / (C[i * n + i] * C[j * n + j])}
	}
	return v / (s * s)
}

@(private = "file")
_hrp_bisect :: proc(items: []int, C: []f64, n: int, weights: []f64) {
	if len(items) < 2 {return}
	h := len(items) / 2
	a, b := items[:h], items[h:]
	va, vb := _hrp_cluster_variance(a, C, n), _hrp_cluster_variance(b, C, n)
	alpha := 1.0 - va / (va + vb)
	for i in a {weights[i] *= alpha}
	for i in b {weights[i] *= 1.0 - alpha}
	_hrp_bisect(a, C, n, weights)
	_hrp_bisect(b, C, n, weights)
}

// Caps every weight at `cap`, spreading the excess equally over the others,
// then renormalizes
@(private = "file")
_hrp_cap :: proc(weights: []f64, cap: f64) {
	for _ in 0 ..< 10 {
		excess, n_free := 0.0, 0
		for &x in weights {
			if x > cap {excess += x - cap; x = cap} else {n_free += 1}
		}
		if excess < 1e-8 || n_free == 0 {break}
		for &x in weights {
			if x < cap {x += excess / f64(n_free)}
		}
	}
	s := 0.0
	for x in weights {s += x}
	if s > 1e-10 {for &x in weights {x /= s}}
}
