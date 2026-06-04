package ML

import l "../../linalg"
import "core:math"
import "core:math/rand"
import "core:mem"
// ============================================================================
// Helper: Self-contained RNG for shuffling
// Uses u64 casting to prevent debug-mode overflow panics.
// This guarantees reproducible shuffles without mutating the global rand state.
// ============================================================================

_shuffle_indices :: proc(indices: []int, seed: int) {
	state := u32(seed) | 1 // Ensure non-zero
	n := len(indices)
	for i := n - 1; i > 0; i -= 1 {
		// ✅ FIX: Do math in u64 space (guaranteed no overflow), then truncate back to u32
		state = u32(u64(state) * 1664525 + 1013904223)

		j := int(state % u32(i + 1))
		indices[i], indices[j] = indices[j], indices[i]
	}
}

// ============================================================================
// Helper: Subset Matrix and Slice by Indices
// ============================================================================

subset_matrix :: proc(
	X: ^l.Matrix(f64),
	indices: []int,
	allocator: mem.Allocator = context.allocator,
) -> l.Matrix(f64) {
	n_new := len(indices)
	n_features := X.cols

	out := l.matrix_new(f64, n_new, n_features, allocator)
	for idx, i in indices {
		for j in 0 ..< n_features {
			out.data[i * n_features + j] = X.data[idx * n_features + j]
		}
	}
	return out
}

subset_slice :: proc(
	y: []f64,
	indices: []int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_new := len(indices)
	out := make([]f64, n_new, allocator)
	for idx, i in indices {
		out[i] = y[idx]
	}
	return out
}

// ============================================================================
// Train / Test Split
// ============================================================================

TrainTestSplit :: struct {
	X_train: l.Matrix(f64),
	X_test:  l.Matrix(f64),
	y_train: []f64,
	y_test:  []f64,
}

// Contract: 0.0 < test_size < 1.0
train_test_split :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	test_size: f64,
	shuffle: bool = true,
	random_state: int = 42,
	allocator: mem.Allocator = context.allocator,
) -> TrainTestSplit {
	n := X.rows
	test_count := math.max(1, int(f64(n) * test_size))
	train_count := n - test_count

	// 1. Generate indices
	indices := make([]int, n, context.temp_allocator)
	for i in 0 ..< n {indices[i] = i}

	// 2. Shuffle if requested
	if shuffle {
		_shuffle_indices(indices, random_state)
	}

	// 3. Split indices
	train_idx := indices[0:train_count]
	test_idx := indices[train_count:n]

	// 4. Subset data
	return TrainTestSplit {
		X_train = subset_matrix(X, train_idx, allocator),
		X_test = subset_matrix(X, test_idx, allocator),
		y_train = subset_slice(y, train_idx, allocator),
		y_test = subset_slice(y, test_idx, allocator),
	}
}

train_test_split_free :: proc(split: ^TrainTestSplit, allocator: mem.Allocator) {
	l.matrix_free(&split.X_train)
	l.matrix_free(&split.X_test)
	delete(split.y_train, allocator)
	delete(split.y_test, allocator)
}

// ============================================================================
// K-Fold Cross Validation
// ============================================================================

// Function pointer type for evaluating a model on a single fold.
// user_data allows passing model params or state without closures.
ModelEvaluator :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	train_idx: []int,
	val_idx: []int,
	user_data: rawptr,
	allocator: mem.Allocator,
) -> f64 // Returns the metric score (e.g., accuracy or R2)

// Contract: n_splits >= 2
cross_val_score :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	n_splits: int,
	evaluator: ModelEvaluator,
	user_data: rawptr,
	shuffle: bool = true,
	random_state: int = 42,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := X.rows
	scores := make([]f64, n_splits, allocator)

	// 1. Generate and shuffle indices
	indices := make([]int, n, context.temp_allocator)
	for i in 0 ..< n {indices[i] = i}

	if shuffle {
		_shuffle_indices(indices, random_state)
	}

	// 2. Split into K folds
	fold_size := n / n_splits
	remainder := n % n_splits

	for k in 0 ..< n_splits {
		// Calculate val bounds for this fold
		val_start := k * fold_size + math.min(k, remainder)

		val_end := val_start + fold_size
		if k < remainder {
			val_end += 1
		}

		// Build train and val indices for this fold
		val_idx := make([]int, val_end - val_start, context.temp_allocator)
		copy(val_idx, indices[val_start:val_end])

		train_idx := make([]int, n - len(val_idx), context.temp_allocator)
		copy(train_idx[0:val_start], indices[0:val_start])
		copy(train_idx[val_start:n - len(val_idx)], indices[val_end:n])

		// 3. Evaluate
		scores[k] = evaluator(X, y, train_idx, val_idx, user_data, allocator)

		delete(val_idx, context.temp_allocator)
		delete(train_idx, context.temp_allocator)
	}

	delete(indices, context.temp_allocator)
	return scores
}
