package ML

import l "../../linalg"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// K-Means Structures
// ============================================================================

KMeans :: struct {
	centroids: l.Matrix(f64),
	labels:    []int,
	inertia:   f64,
	n_iter:    int,
	converged: bool,
	allocator: mem.Allocator,
}

KMParams :: struct {
	n_clusters:   int,
	max_iter:     int,
	tol:          f64,
	init:         KMInit,
	n_init:       int,
	random_state: int,
}

KMInit :: enum {
	Random,
	KMeansPlusPlus,
}

// ============================================================================
// Public API
// ============================================================================

km_fit :: proc(
	X: ^l.Matrix(f64),
	params: KMParams,
	allocator: mem.Allocator = context.allocator,
) -> KMeans {
	n_samples := X.rows
	k := params.n_clusters

	if k < 1 || k > n_samples {
		panic(fmt.aprintf("km_fit: invalid n_clusters=%v for %v samples", k, n_samples))
	}

	best_model: KMeans
	// ✅ FIXED: Use proper f64 infinity
	best_inertia := math.inf_f64(1)

	for init_run in 0 ..< params.n_init {
		model := _km_fit_single(X, params, allocator)

		if model.inertia < best_inertia {
			if best_model.centroids.data != nil {
				l.matrix_free(&best_model.centroids)
				if len(best_model.labels) > 0 {
					delete(best_model.labels, allocator)
				}
			}
			best_model = model
			best_inertia = model.inertia
		} else {
			l.matrix_free(&model.centroids)
			if len(model.labels) > 0 {
				delete(model.labels, allocator)
			}
		}
	}

	return best_model
}

km_predict :: proc(
	model: ^KMeans,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []int {
	n := X.rows
	k := model.centroids.rows
	n_features := model.centroids.cols

	labels := make([]int, n, allocator)

	for i in 0 ..< n {
		// ✅ FIXED: Proper f64 infinity
		best_dist := math.inf_f64(1)
		best_label := 0

		for c in 0 ..< k {
			dist := _squared_euclidean_simd(
				X.data[i * n_features:i * n_features + n_features],
				model.centroids.data[c * n_features:c * n_features + n_features],
				n_features,
			)
			if dist < best_dist {
				best_dist = dist
				best_label = c
			}
		}
		labels[i] = best_label
	}

	return labels
}

km_free :: proc(model: ^KMeans) {
	if model.centroids.data != nil {
		l.matrix_free(&model.centroids)
	}
	if len(model.labels) > 0 {
		delete(model.labels, model.allocator)
	}
}

// ============================================================================
// Internal: Single Fit Run
// ============================================================================

_km_fit_single :: proc(X: ^l.Matrix(f64), params: KMParams, allocator: mem.Allocator) -> KMeans {
	n_samples := X.rows
	n_features := X.cols
	k := params.n_clusters

	// ✅ FIXED: Correct function name (single underscore)
	centroids := _init_centroids(X, k, params.init, params.random_state, allocator)

	labels := make([]int, n_samples, allocator)
	prev_centroids := l.matrix_new(f64, k, n_features, context.temp_allocator)

	inertia := 0.0
	converged := false
	n_iter := 0

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1

		_assign_clusters(X, &centroids, labels, allocator)
		_update_centroids(X, labels, k, &centroids, allocator)
		inertia = _compute_inertia(X, &centroids, labels, allocator)

		if _check_convergence(&centroids, &prev_centroids, params.tol) {
			converged = true
			break
		}

		// ✅ FIXED: Explicit slice bounds for copy
		total_elements := k * n_features
		copy(prev_centroids.data[0:total_elements], centroids.data[0:total_elements])
	}

	l.matrix_free(&prev_centroids)

	return KMeans {
		centroids = centroids,
		labels = labels,
		inertia = inertia,
		n_iter = n_iter,
		converged = converged,
		allocator = allocator,
	}
}

// ============================================================================
// Internal: Initialization (FIXED for Odin rand API)
// ============================================================================

_init_centroids :: proc(
	X: ^l.Matrix(f64),
	k: int,
	init: KMInit,
	seed: int,
	allocator: mem.Allocator,
) -> l.Matrix(f64) {
	n_samples := X.rows
	n_features := X.cols

	centroids := l.matrix_new(f64, k, n_features, allocator)

	// Use context.random_generator (Odin's standard approach)
	gen := context.random_generator

	switch init {
	case .Random:
		selected := make([]bool, n_samples, context.temp_allocator)
		defer delete(selected, context.temp_allocator)

		count := 0
		for count < k {
			// ✅ FIXED: Use rand.int63 % n for [0, n) range
			idx := int(rand.int63(gen) % i64(n_samples))
			if !selected[idx] {
				selected[idx] = true
				for f in 0 ..< n_features {
					centroids.data[count * n_features + f] = X.data[idx * n_features + f]
				}
				count += 1
			}
		}

	case .KMeansPlusPlus:
		// Pick first centroid randomly
		first := int(rand.int63(gen) % i64(n_samples))
		for f in 0 ..< n_features {
			centroids.data[0 * n_features + f] = X.data[first * n_features + f]
		}

		for c in 1 ..< k {
			dists := make([]f64, n_samples, context.temp_allocator)
			for i in 0 ..< n_samples {
				// ✅ FIXED: Proper f64 infinity
				min_dist := math.inf_f64(1)
				for existing in 0 ..< c {
					d := _squared_euclidean_simd(
						X.data[i * n_features:i * n_features + n_features],
						centroids.data[existing * n_features:existing * n_features + n_features],
						n_features,
					)
					if d < min_dist {min_dist = d}
				}
				dists[i] = min_dist
			}

			total := 0.0
			for d in dists {total += d}
			if total < 1e-10 {
				// Fallback: pick random unselected point
				for i in 0 ..< n_samples {
					already := false
					for existing in 0 ..< c {
						if _vectors_equal(
							X.data[i * n_features:i * n_features + n_features],
							centroids.data[existing *
							n_features:existing * n_features +
							n_features],
							n_features,
						) {
							already = true
							break
						}
					}
					if !already {
						for f in 0 ..< n_features {
							centroids.data[c * n_features + f] = X.data[i * n_features + f]
						}
						break
					}
				}
			} else {
				r := rand.float64(gen) * total
				cumsum := 0.0
				for i in 0 ..< n_samples {
					cumsum += dists[i]
					if cumsum >= r {
						for f in 0 ..< n_features {
							centroids.data[c * n_features + f] = X.data[i * n_features + f]
						}
						break
					}
				}
			}
			delete(dists, context.temp_allocator)
		}
	}

	return centroids
}

// ============================================================================
// Internal: Core Algorithm Steps
// ============================================================================

_assign_clusters :: proc(
	X: ^l.Matrix(f64),
	centroids: ^l.Matrix(f64),
	labels: []int,
	allocator: mem.Allocator,
) {
	n_samples := X.rows
	n_features := X.cols
	k := centroids.rows

	for i in 0 ..< n_samples {
		x := X.data[i * n_features:i * n_features + n_features]
		// ✅ FIXED: Proper f64 infinity
		best_dist := math.inf_f64(1)
		best_label := 0

		for c in 0 ..< k {
			c_vec := centroids.data[c * n_features:c * n_features + n_features]
			dist := _squared_euclidean_simd(x, c_vec, n_features)
			if dist < best_dist {
				best_dist = dist
				best_label = c
			}
		}
		labels[i] = best_label
	}
}

_update_centroids :: proc(
	X: ^l.Matrix(f64),
	labels: []int,
	k: int,
	centroids: ^l.Matrix(f64),
	allocator: mem.Allocator,
) {
	n_features := centroids.cols

	counts := make([]int, k, context.temp_allocator)
	defer delete(counts, context.temp_allocator)

	sums := make([][]f64, k, context.temp_allocator)
	defer {
		for s in sums {delete(s, context.temp_allocator)}
		delete(sums, context.temp_allocator)
	}
	for c in 0 ..< k {
		sums[c] = make([]f64, n_features, context.temp_allocator)
	}

	for i, label in labels {
		if label < 0 || label >= k {continue}
		counts[label] += 1
		x := X.data[i * n_features:i * n_features + n_features]
		for f in 0 ..< n_features {
			sums[label][f] += x[f]
		}
	}

	for c in 0 ..< k {
		if counts[c] == 0 {
			for f in 0 ..< n_features {
				centroids.data[c * n_features + f] = rand.float64_normal(
					0,
					1,
					context.random_generator,
				)
			}
		} else {
			inv_n := 1.0 / f64(counts[c])
			for f in 0 ..< n_features {
				centroids.data[c * n_features + f] = sums[c][f] * inv_n
			}
		}
	}
}

_compute_inertia :: proc(
	X: ^l.Matrix(f64),
	centroids: ^l.Matrix(f64),
	labels: []int,
	allocator: mem.Allocator,
) -> f64 {
	n_samples := X.rows
	inertia := 0.0

	for i in 0 ..< n_samples {
		label := labels[i]
		if label < 0 || label >= centroids.rows {continue}

		x := X.data[i * X.cols:i * X.cols + X.cols]
		c := centroids.data[label * centroids.cols:label * centroids.cols + centroids.cols]
		inertia += _squared_euclidean_simd(x, c, X.cols)
	}

	return inertia
}

_check_convergence :: proc(curr: ^l.Matrix(f64), prev: ^l.Matrix(f64), tol: f64) -> bool {
	k := curr.rows
	n_features := curr.cols

	max_shift := 0.0
	for c in 0 ..< k {
		curr_vec := curr.data[c * n_features:c * n_features + n_features]
		prev_vec := prev.data[c * n_features:c * n_features + n_features]
		shift := 0.0
		for f in 0 ..< n_features {
			diff := curr_vec[f] - prev_vec[f]
			shift += diff * diff
		}
		shift = math.sqrt(shift)
		if shift > max_shift {max_shift = shift}
	}

	return max_shift < tol
}

// ============================================================================
// SIMD Helper: Squared Euclidean Distance
// ============================================================================

_squared_euclidean_simd :: proc(a, b: []f64, n: int) -> f64 {
	if len(a) != n || len(b) != n do panic("_squared_euclidean_simd: length mismatch")

	diff := make([]f64, n, context.temp_allocator)
	defer delete(diff, context.temp_allocator)

	l.vec_sub_simd(a, b, diff)
	return l.dot_simd(diff, diff)
}

_vectors_equal :: proc(a, b: []f64, n: int) -> bool {
	for i in 0 ..< n {
		if math.abs(a[i] - b[i]) > 1e-10 {
			return false
		}
	}
	return true
}

// ============================================================================
// K-Means Statistics
// ============================================================================

KMStats :: struct {
	n_samples:       int,
	n_features:      int,
	n_clusters:      int,
	inertia:         f64,
	n_iter:          int,
	converged:       bool,
	cluster_sizes:   []int,
	cluster_inertia: []f64,
}

km_compute_stats :: proc(
	model: ^KMeans,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> KMStats {
	n_samples := X.rows
	n_features := X.cols
	k := model.centroids.rows

	sizes := make([]int, k, allocator)
	for label in model.labels {
		if label >= 0 && label < k {
			sizes[label] += 1
		}
	}

	cluster_inertia := make([]f64, k, allocator)
	for i in 0 ..< n_samples {
		label := model.labels[i]
		if label < 0 || label >= k {continue}
		x := X.data[i * n_features:i * n_features + n_features]
		c := model.centroids.data[label * n_features:label * n_features + n_features]
		cluster_inertia[label] += _squared_euclidean_simd(x, c, n_features)
	}

	return KMStats {
		n_samples = n_samples,
		n_features = n_features,
		n_clusters = k,
		inertia = model.inertia,
		n_iter = model.n_iter,
		converged = model.converged,
		cluster_sizes = sizes,
		cluster_inertia = cluster_inertia,
	}
}

km_stats_print :: proc(stats: ^KMStats) {
	fmt.println("=== K-MEANS STATISTICS ===")
	fmt.printf("Samples: %v\n", stats.n_samples)
	fmt.printf("Features: %v\n", stats.n_features)
	fmt.printf("Clusters: %v\n", stats.n_clusters)
	fmt.printf("Iterations: %v\n", stats.n_iter)
	fmt.printf("Converged: %v\n", stats.converged)
	fmt.printf("Inertia: %.6f\n", stats.inertia)
	fmt.println()

	fmt.println("Cluster Sizes:")
	for c in 0 ..< stats.n_clusters {
		fmt.printf(
			"  Cluster %2v: %4v points, inertia=%.4f\n",
			c,
			stats.cluster_sizes[c],
			stats.cluster_inertia[c],
		)
	}
	fmt.println("=== END K-MEANS STATS ===")
}

km_stats_free :: proc(stats: ^KMStats, allocator: mem.Allocator) {
	if len(stats.cluster_sizes) > 0 {
		delete(stats.cluster_sizes, allocator)
	}
	if len(stats.cluster_inertia) > 0 {
		delete(stats.cluster_inertia, allocator)
	}
}
