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
// K-Means++ Initialization (Uses SIMD for distance computation)
// ============================================================================

_init_centroids_kmeanspp :: proc(
	X: ^l.Matrix(f64),
	k: int,
	seed: int,
	allocator: mem.Allocator,
) -> l.Matrix(f64) {
	n_samples := X.rows
	n_features := X.cols

	centroids := l.matrix_new(f64, k, n_features, allocator)
	gen := context.random_generator
	_ = seed // Odin's rand uses context.random_generator

	// Pick first centroid uniformly at random
	first := int(rand.int63(gen) % i64(n_samples))
	for f in 0 ..< n_features {
		centroids.data[0 * n_features + f] = X.data[first * n_features + f]
	}

	// Pick remaining k-1 centroids
	for c in 1 ..< k {
		// Compute D(x)^2 = min distance^2 to any existing centroid
		dists := make([]f64, n_samples, context.temp_allocator)

		for i in 0 ..< n_samples {
			min_d := math.inf_f64(1)
			x_vec := X.data[i * n_features:i * n_features + n_features]

			for existing in 0 ..< c {
				c_vec := centroids.data[existing * n_features:existing * n_features + n_features]
				d := _squared_euclidean_simd(x_vec, c_vec, n_features)
				if d < min_d {min_d = d}
			}
			dists[i] = min_d
		}

		// Choose next centroid with probability proportional to D(x)^2
		total := 0.0
		for d in dists {total += d}

		if total < 1e-10 {
			// Fallback: pick any point not already chosen
			for i in 0 ..< n_samples {
				already := false
				for e in 0 ..< c {
					same := true
					for f in 0 ..< n_features {
						if math.abs(
							   X.data[i * n_features + f] - centroids.data[e * n_features + f],
						   ) >
						   1e-10 {
							same = false
							break
						}
					}
					if same {already = true; break}
				}
				if !already {
					for f in 0 ..< n_features {
						centroids.data[c * n_features + f] = X.data[i * n_features + f]
					}
					break
				}
			}
		} else {
			// Sample from distribution proportional to dists
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

	return centroids
}
// ============================================================================
// Random Initialization Helper
// ============================================================================

_init_centroids_random :: proc(
	X: ^l.Matrix(f64),
	k: int,
	seed: int,
	allocator: mem.Allocator,
) -> l.Matrix(f64) {
	n_samples := X.rows
	n_features := X.cols

	centroids := l.matrix_new(f64, k, n_features, allocator)
	gen := context.random_generator
	selected := make([]bool, n_samples, context.temp_allocator)
	defer delete(selected, context.temp_allocator)

	count := 0
	for count < k {
		idx := int(rand.int63(gen) % i64(n_samples))
		if !selected[idx] {
			selected[idx] = true
			for f in 0 ..< n_features {
				centroids.data[count * n_features + f] = X.data[idx * n_features + f]
			}
			count += 1
		}
	}

	return centroids
}
// ============================================================================
// Internal: Single Fit Run
// ============================================================================

_km_fit_single :: proc(X: ^l.Matrix(f64), params: KMParams, allocator: mem.Allocator) -> KMeans {
	n_samples := X.rows
	n_features := X.cols
	k := params.n_clusters

	// Simple random initialization (no fancy logic)
	centroids := l.Matrix(f64){} // Declare first

	switch params.init {
	case .Random:
		centroids = _init_centroids_random(X, k, params.random_state, allocator)
	case .KMeansPlusPlus:
		centroids = _init_centroids_kmeanspp(X, k, params.random_state, allocator)
	}

	labels := make([]int, n_samples, allocator)
	prev_centroids := l.matrix_new(f64, k, n_features, context.temp_allocator)

	inertia := 0.0
	converged := false
	n_iter := 0

	for iter in 0 ..< params.max_iter {
		n_iter = iter + 1

		// Assign clusters
		for i in 0 ..< n_samples {
			x_start := i * n_features
			best_dist := math.inf_f64(1)
			best_label := 0

			for c in 0 ..< k {
				c_start := c * n_features
				dist := _squared_euclidean_simd(
					X.data[x_start:x_start + n_features],
					centroids.data[c_start:c_start + n_features],
					n_features,
				)
				if dist < best_dist {
					best_dist = dist
					best_label = c
				}
			}
			labels[i] = best_label
		}

		// Update centroids
		counts := make([]int, k, context.temp_allocator)
		sums := make([][]f64, k, context.temp_allocator)
		for c in 0 ..< k {
			sums[c] = make([]f64, n_features, context.temp_allocator)
		}

		for i in 0 ..< n_samples {
			label := labels[i]
			if label < 0 || label >= k {continue}
			counts[label] += 1
			x_start := i * n_features
			for f in 0 ..< n_features {
				sums[label][f] += X.data[x_start + f]
			}
		}

		for c in 0 ..< k {
			if counts[c] == 0 {
				// Reinitialize empty cluster to random point
				idx := int(rand.int63(context.random_generator) % i64(n_samples))
				for f in 0 ..< n_features {
					centroids.data[c * n_features + f] = X.data[idx * n_features + f]
				}
			} else {
				inv_n := 1.0 / f64(counts[c])
				for f in 0 ..< n_features {
					centroids.data[c * n_features + f] = sums[c][f] * inv_n
				}
			}
		}

		for c in 0 ..< k {
			delete(sums[c], context.temp_allocator)
		}
		delete(sums, context.temp_allocator)
		delete(counts, context.temp_allocator)

		// Compute inertia
		inertia = 0.0
		for i in 0 ..< n_samples {
			label := labels[i]
			if label < 0 || label >= k {continue}
			x_start := i * n_features
			c_start := label * n_features
			inertia += _squared_euclidean_simd(
				X.data[x_start:x_start + n_features],
				centroids.data[c_start:c_start + n_features],
				n_features,
			)
		}

		// Check convergence
		max_shift := 0.0
		for c in 0 ..< k {
			c_start := c * n_features
			shift := 0.0
			for f in 0 ..< n_features {
				diff := centroids.data[c_start + f] - prev_centroids.data[c_start + f]
				shift += diff * diff
			}
			shift = math.sqrt(shift)
			if shift > max_shift {max_shift = shift}
		}

		if max_shift < params.tol {
			converged = true
			break
		}

		// Save centroids for next iteration
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
// SIMD Distance (Uses your wotan_linalg module)
// ============================================================================

_squared_euclidean_simd :: proc(a, b: []f64, n: int) -> f64 {
	if len(a) != n || len(b) != n do panic("_squared_euclidean_simd: length mismatch")

	// Compute diff = a - b using your SIMD subtraction
	diff := make([]f64, n, context.temp_allocator)
	defer delete(diff, context.temp_allocator)

	l.vec_sub_simd(a, b, diff)

	// Compute squared norm using your SIMD dot product
	return l.dot_simd(diff, diff)
}
// ============================================================================
// Statistics (Optional - Can Add Later)
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
		x_start := i * n_features
		c_start := label * n_features
		cluster_inertia[label] += _squared_euclidean_simd(
			X.data[x_start:x_start + n_features],
			model.centroids.data[c_start:c_start + n_features],
			n_features,
		)
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
