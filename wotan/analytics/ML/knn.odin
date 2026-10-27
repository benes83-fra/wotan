package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// K-Nearest Neighbors (Lazy Learner)
// ============================================================================
// Classifier as sklearn KNeighborsClassifier(algorithm="brute"): Euclidean
// distance, uniform or 1/distance votes, ties between classes → smallest label.
// Neighbours tied at the k-th distance are taken in index order (sklearn's
// choice there is unspecified).

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
	if len(y) != X.rows do panic("knn_fit: y length mismatch")
	if params.k < 1 do panic("knn_fit: k must be >= 1")
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
	if n_train > 0 && n_features != model.X_train.cols do panic("knn_predict: X has the wrong number of columns")
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

		// 3. (Weighted) vote, as sklearn KNeighborsClassifier: uniform weights, or
		//    1/distance where neighbours at distance 0 get all the weight; ties go
		//    to the smallest class label
		any_zero := false
		if model.weights == .Distance {
			for c in 0 ..< k {if dists[c] == 0 {any_zero = true; break}}
		}
		{
			num_unique := 0

			for c in 0 ..< k {
				cls := model.y_train[indices[c]]
				weight := 1.0
				if model.weights == .Distance {
					if any_zero {
						weight = 1.0 if dists[c] == 0 else 0.0
					} else {
						weight = 1.0 / math.sqrt(dists[c])
					}
				}

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
				w := class_weights[u_idx]
				if w > max_weight || (w == max_weight && unique_classes[u_idx] < best_class) {
					max_weight = w
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
