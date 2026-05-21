package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// K-Fold Cross-Validation Indices
// Returns a slice of fold assignments: fold[i] ∈ {0, 1, ..., k-1}
// Uses deterministic shuffling based on seed (no external rand dependency)
// ============================================================================

// ============================================================================
// K-Fold Cross-Validation Indices
// Returns a slice of fold assignments: fold[i] ∈ {0, 1, ..., k-1}
// Uses deterministic shuffling based on seed (no external rand dependency)
// ============================================================================
cv_folds :: proc(
	n: int, // number of samples
	k: int, // number of folds
	seed: u64 = 0, // random seed for reproducibility
	allocator: mem.Allocator = context.allocator,
) -> []int {
	folds := make([]int, n, allocator)

	// Initialize: assign each sample to a fold (balanced)
	for i in 0 ..< n {
		folds[i] = i % k
	}

	// Simple deterministic shuffle using seed (Fisher-Yates)
	if seed != 0 {
		// Simple LCG for reproducibility (inline, no closure)
		state := seed
		for i := n - 1; i > 0; i -= 1 {
			// LCG step: state = state * a + c (mod 2^64, implicit)
			state = state * 6364136223846793005 + 1
			// Generate random j in [0, i]
			j := int(state % u64(i + 1))
			// Swap
			folds[i], folds[j] = folds[j], folds[i]
		}
	}

	return folds
}

// ============================================================================
// Split data into train/val indices given fold assignments
// ============================================================================
cv_split :: proc(
	folds: []int,
	val_fold: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	train_idx: []int,
	val_idx: []int,
) {
	n := len(folds)
	train_count := 0
	val_count := 0

	// Count sizes first
	for i in 0 ..< n {
		if folds[i] == val_fold {
			val_count += 1
		} else {
			train_count += 1
		}
	}

	train_idx = make([]int, train_count, allocator)
	val_idx = make([]int, val_count, allocator)

	ti, vi := 0, 0
	for i in 0 ..< n {
		if folds[i] == val_fold {
			val_idx[vi] = i
			vi += 1
		} else {
			train_idx[ti] = i
			ti += 1
		}
	}

	return train_idx, val_idx
}

// ============================================================================
// Extract submatrix X[train_idx, :] and subvector y[train_idx]
// ============================================================================
cv_subset :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	idx: []int,
	allocator: mem.Allocator = context.allocator,
) -> (
	X_sub: l.Matrix(f64),
	y_sub: []f64,
) {
	n_sub := len(idx)
	p := X.cols

	X_sub = l.matrix_new(f64, n_sub, p, allocator)
	y_sub = make([]f64, n_sub, allocator)

	for i in 0 ..< n_sub {
		row := idx[i]
		for j in 0 ..< p {
			X_sub.data[i * p + j] = X.data[row * p + j]
		}
		y_sub[i] = y[row]
	}

	return X_sub, y_sub
}

// ============================================================================
// Free CV subset (helper)
// ============================================================================
cv_subset_free :: proc(X_sub: ^l.Matrix(f64), y_sub: []f64, allocator: mem.Allocator) {
	if X_sub.data != nil {l.matrix_free(X_sub)}
	if len(y_sub) > 0 {delete(y_sub, allocator)}
}
