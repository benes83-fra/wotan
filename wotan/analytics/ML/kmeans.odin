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
	n_init:       int, // ≤ 0 → 1
	random_state: int, // ≠ 0: reproducible runs from this seed; 0: global RNG
	init_centers: []f64, // optional k×p starting centres (sklearn init=array)
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

	if X.cols == 0 do panic("km_fit: X has no features")

	// Seeded local generator: does not touch the global RNG. (A context change
	// only lasts for its block, so it must be made at procedure scope.)
	rng_state := rand.create_u64(u64(params.random_state))
	gen := context.random_generator
	if params.random_state != 0 do gen = rand.default_random_generator(&rng_state)
	context.random_generator = gen

	best_model: KMeans
	best_inertia := math.inf_f64(1)
	n_init := params.n_init if params.n_init > 0 else 1
	if len(params.init_centers) > 0 do n_init = 1

	for init_run in 0 ..< n_init {
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
	if X.cols != n_features do panic("km_predict: X has the wrong number of columns")

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

		if !(total > 0) {
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
			pick := -1
			for i in 0 ..< n_samples {
				if dists[i] <= 0 do continue // never re-pick an existing centre
				pick = i
				cumsum += dists[i]
				if cumsum > r do break
			}
			for f in 0 ..< n_features {
				centroids.data[c * n_features + f] = X.data[pick * n_features + f]
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
// Internal: Single Fit Run (Lloyd, as sklearn KMeans(algorithm="lloyd"))
//   E-step: label = nearest centre; M-step: centre = mean of its points.
//   Stops when the labels no longer change or Σ_c ||Δcentre_c||² ≤ tol·mean(Var(X_j))
//   (sklearn's scaled tolerance). Empty clusters are moved to the points farthest
//   from their centres. Labels and inertia always refer to the final centres.
// ============================================================================

_km_fit_single :: proc(X: ^l.Matrix(f64), params: KMParams, allocator: mem.Allocator) -> KMeans {
	n_samples := X.rows
	n_features := X.cols
	k := params.n_clusters

	centroids := l.matrix_new(f64, k, n_features, allocator)
	if len(params.init_centers) > 0 {
		if len(params.init_centers) != k * n_features do panic("km_fit: init_centers must have n_clusters × n_features values")
		copy(centroids.data, params.init_centers)
	} else {
		init := _init_centroids_random(X, k, params.random_state, context.temp_allocator) if params.init == .Random else _init_centroids_kmeanspp(X, k, params.random_state, context.temp_allocator)
		copy(centroids.data, init.data)
		l.matrix_free(&init)
	}

	// sklearn: tol is relative to the mean per-feature variance of X
	tol_abs := 0.0
	if params.tol > 0 && n_samples > 0 {
		for f in 0 ..< n_features {
			m, v := 0.0, 0.0
			for i in 0 ..< n_samples {m += X.data[i * n_features + f]}
			m /= f64(n_samples)
			for i in 0 ..< n_samples {d := X.data[i * n_features + f] - m; v += d * d}
			tol_abs += v / f64(n_samples)
		}
		tol_abs = params.tol * tol_abs / f64(max(n_features, 1))
	}

	labels := make([]int, n_samples, allocator)
	prev_labels := make([]int, n_samples, context.temp_allocator)
	defer delete(prev_labels, context.temp_allocator)
	dist := make([]f64, n_samples, context.temp_allocator)
	defer delete(dist, context.temp_allocator)
	new_c := make([]f64, k * n_features, context.temp_allocator)
	defer delete(new_c, context.temp_allocator)
	counts := make([]int, k, context.temp_allocator)
	defer delete(counts, context.temp_allocator)

	converged := false
	n_iter := 0
	max_iter := params.max_iter if params.max_iter > 0 else 300

	_km_assign(X, &centroids, labels, dist)
	for iter in 0 ..< max_iter {
		n_iter = iter + 1

		// M-step
		for &v in new_c do v = 0
		for &c in counts do c = 0
		for i in 0 ..< n_samples {
			c := labels[i]
			counts[c] += 1
			for f in 0 ..< n_features {new_c[c * n_features + f] += X.data[i * n_features + f]}
		}
		for c in 0 ..< k {
			if counts[c] > 0 {
				for f in 0 ..< n_features {new_c[c * n_features + f] /= f64(counts[c])}
			}
		}
		_km_relocate_empty(X, labels, dist, counts, new_c, k)

		shift := 0.0
		for i in 0 ..< k * n_features {
			d := new_c[i] - centroids.data[i]
			shift += d * d
		}
		copy(centroids.data, new_c)

		// E-step
		copy(prev_labels, labels)
		_km_assign(X, &centroids, labels, dist)

		same := true
		for i in 0 ..< n_samples {if labels[i] != prev_labels[i] {same = false; break}}
		if same || shift <= tol_abs {
			converged = true
			break
		}
	}

	inertia := 0.0
	for d in dist do inertia += d

	return KMeans {
		centroids = centroids,
		labels = labels,
		inertia = inertia,
		n_iter = n_iter,
		converged = converged,
		allocator = allocator,
	}
}

// labels[i] = nearest centre, dist[i] = its squared distance (ties → lowest index)
_km_assign :: proc(X: ^l.Matrix(f64), C: ^l.Matrix(f64), labels: []int, dist: []f64) {
	p := X.cols
	for i in 0 ..< X.rows {
		x := X.data[i * p:(i + 1) * p]
		best, best_d := 0, math.inf_f64(1)
		for c in 0 ..< C.rows {
			d := _squared_euclidean_simd(x, C.data[c * p:(c + 1) * p], p)
			if d < best_d {best_d = d; best = c}
		}
		labels[i] = best
		dist[i] = best_d
	}
}

// Empty clusters take the points farthest from their current centres (sklearn)
_km_relocate_empty :: proc(X: ^l.Matrix(f64), labels: []int, dist: []f64, counts: []int, centres: []f64, k: int) {
	p := X.cols
	taken := make([]bool, X.rows, context.temp_allocator)
	defer delete(taken, context.temp_allocator)
	for c in 0 ..< k {
		if counts[c] > 0 do continue
		far, far_d := -1, -1.0
		for i in 0 ..< X.rows {
			if !taken[i] && counts[labels[i]] > 1 && dist[i] > far_d {far_d = dist[i]; far = i}
		}
		if far < 0 do continue
		taken[far] = true
		counts[labels[far]] -= 1
		counts[c] = 1
		for f in 0 ..< p {centres[c * p + f] = X.data[far * p + f]}
	}
}

// ============================================================================
// SIMD Distance (Uses your wotan_linalg module)
// ============================================================================

_squared_euclidean_simd :: proc(a, b: []f64, n: int) -> f64 {
	if len(a) != n || len(b) != n do panic("_squared_euclidean_simd: length mismatch")
	s := 0.0
	for i in 0 ..< n {
		d := a[i] - b[i]
		s += d * d
	}
	return s
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
