package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// K-Nearest Neighbors (Lazy Learner)
// ============================================================================

KNNWeights :: enum {
	Uniform, // Standard majority vote
	Distance, // Weighted by 1/distance
}

KNNParams :: struct {
	k:       int,
	weights: KNNWeights,
}

KNN :: struct {
	X_train:   l.Matrix(f64),
	y_train:   []f64,
	k:         int,
	weights:   KNNWeights,
	allocator: mem.Allocator,
}

// Fit: KNN just stores a copy of the training data
knn_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: KNNParams,
	allocator: mem.Allocator = context.allocator,
) -> KNN {
	// Copy data to ensure lifetime safety
	X_copy := l.matrix_new(f64, X.rows, X.cols, allocator)
	copy(X_copy.data, X.data)

	y_copy := make([]f64, len(y), allocator)
	copy(y_copy, y)

	return KNN {
		X_train = X_copy,
		y_train = y_copy,
		k = params.k,
		weights = params.weights,
		allocator = allocator,
	}
}

// Predict: Finds the K nearest neighbors and does a majority vote
knn_predict :: proc(
	model: ^KNN,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_test := X.rows
	n_train := model.X_train.rows
	n_features := X.cols
	k := math.min(model.k, n_train)

	preds := make([]f64, n_test, allocator)
	if n_train == 0 || k == 0 {
		return preds
	}

	// Pre-allocate buffers outside the loop for maximum performance (Zero allocations in hot loop)
	dists := make([]f64, n_train, allocator)
	defer delete(dists, allocator)

	indices := make([]int, n_train, allocator)
	defer delete(indices, allocator)

	diff := make([]f64, n_features, allocator)
	defer delete(diff, allocator)

	// ✅ FIX: Use standard slices with fixed capacity `k`.
	// The maximum number of unique classes in the top K neighbors is exactly K.
	// We use a manual `num_unique` counter to track the logical length.
	unique_classes := make([]f64, k, allocator)
	class_weights := make([]f64, k, allocator)
	defer {
		delete(unique_classes, allocator)
		delete(class_weights, allocator)
	}

	for i in 0 ..< n_test {
		x_test := X.data[i * n_features:i * n_features + n_features]

		// 1. Compute all distances using SIMD
		for j in 0 ..< n_train {
			x_train := model.X_train.data[j * n_features:j * n_features + n_features]
			l.vec_sub_simd(x_test, x_train, diff)
			dists[j] = l.dot_simd(diff, diff) // Squared Euclidean
			indices[j] = j
		}

		// 2. Partial sort to find top K smallest distances (O(N * K))
		for c in 0 ..< k {
			min_idx := c
			min_dist := dists[c]
			for j in c + 1 ..< n_train {
				if dists[j] < min_dist {
					min_dist = dists[j]
					min_idx = j
				}
			}
			// Swap
			if min_idx != c {
				dists[c], dists[min_idx] = dists[min_idx], dists[c]
				indices[c], indices[min_idx] = indices[min_idx], indices[c]
			}
		}

		// 3. Majority vote or distance weighting
		if model.weights == .Uniform {
			best_class := model.y_train[indices[0]]
			max_count := 0

			// O(K^2) counting is extremely fast for small K
			for c in 0 ..< k {
				current_class := model.y_train[indices[c]]
				count := 0
				for c2 in 0 ..< k {
					if model.y_train[indices[c2]] == current_class {
						count += 1
					}
				}
				if count > max_count {
					max_count = count
					best_class = current_class
				}
			}
			preds[i] = best_class
		} else {
			// Distance weighting
			// ✅ Reset logical length for this sample (keeps backing array, zero allocation!)
			num_unique := 0

			for c in 0 ..< k {
				cls := model.y_train[indices[c]]
				dist := dists[c]
				weight := 1.0 / (math.sqrt(dist) + 1e-10) // Use actual distance, not squared

				found_idx := -1
				for u_idx in 0 ..< num_unique {
					if unique_classes[u_idx] == cls {
						found_idx = u_idx
						break
					}
				}

				if found_idx >= 0 {
					class_weights[found_idx] += weight
				} else {
					unique_classes[num_unique] = cls
					class_weights[num_unique] = weight
					num_unique += 1
				}
			}

			best_class := unique_classes[0]
			max_weight := class_weights[0]
			for u_idx in 1 ..< num_unique {
				if class_weights[u_idx] > max_weight {
					max_weight = class_weights[u_idx]
					best_class = unique_classes[u_idx]
				}
			}
			preds[i] = best_class
		}
	}

	return preds
}

knn_free :: proc(model: ^KNN) {
	l.matrix_free(&model.X_train)
	if len(model.y_train) > 0 {
		delete(model.y_train, model.allocator)
	}
}
