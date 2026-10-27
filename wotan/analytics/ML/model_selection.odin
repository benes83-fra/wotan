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

// Contract: 0.0 < test_size < 1.0. As sklearn: n_test = ceil(test_size · n),
// n_train = n - n_test; both must be >= 1.
train_test_split :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	test_size: f64,
	shuffle: bool = true,
	random_state: int = 42,
	allocator: mem.Allocator = context.allocator,
) -> TrainTestSplit {
	n := X.rows
	if len(y) != n do panic("train_test_split: y length mismatch")
	if !(test_size > 0 && test_size < 1) do panic("train_test_split: test_size must be in (0, 1)")
	test_count := int(math.ceil(f64(n) * test_size))
	train_count := n - test_count
	if test_count < 1 || train_count < 1 do panic("train_test_split: too few samples for this test_size")

	// 1. Generate indices
	indices := make([]int, n, context.temp_allocator)
	defer delete(indices, context.temp_allocator)
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
	if n_splits < 2 || n_splits > X.rows do panic("cross_val_score: need 2 <= n_splits <= n")
	// KFold: contiguous folds of the shuffled samples, first n % n_splits folds larger
	cv := CVSplitter {
		kind         = .KFold,
		n_splits     = n_splits,
		shuffle      = shuffle,
		random_state = random_state,
	}
	return cross_val_score_cv(X, y, cv, evaluator, user_data, allocator)
}

// ============================================================================
// Hold-out split (validation sets for early stopping)
// ============================================================================

// Shuffled train / validation split with n_val = ceil(fraction · n) (as sklearn
// train_test_split), shuffled by _shuffle_indices(seed). stratify: every class
// gets its share of the validation rows (largest remainder), taken in shuffled
// order. Both index sets are returned ascending. Needs 1 <= n_val < n.
holdout_split :: proc(
	n: int,
	y: []f64,
	fraction: f64,
	stratify: bool,
	seed: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	train: []int,
	val: []int,
) {
	if !(fraction > 0 && fraction < 1) do panic("holdout_split: fraction must be in (0, 1)")
	n_val := int(math.ceil(fraction * f64(n)))
	if n_val < 1 || n_val >= n do panic("holdout_split: too few samples for this fraction")
	if stratify && len(y) != n do panic("holdout_split: stratify needs y of length n")

	order := make([]int, n, context.temp_allocator)
	defer delete(order, context.temp_allocator)
	for i in 0 ..< n {order[i] = i}
	_shuffle_indices(order, seed)

	in_val := make([]bool, n, context.temp_allocator)
	defer delete(in_val, context.temp_allocator)
	if !stratify {
		for i in 0 ..< n_val {in_val[order[i]] = true}
	} else {
		classes, codes := _tree_encode_classes(y, context.temp_allocator)
		defer {
			delete(classes, context.temp_allocator)
			delete(codes, context.temp_allocator)
		}
		K := len(classes)
		count := make([]int, K, context.temp_allocator)
		quota := make([]int, K, context.temp_allocator)
		rem := make([]f64, K, context.temp_allocator)
		defer {
			delete(count, context.temp_allocator)
			delete(quota, context.temp_allocator)
			delete(rem, context.temp_allocator)
		}
		for c in codes {count[c] += 1}
		assigned := 0
		for k in 0 ..< K {
			exact := f64(n_val) * f64(count[k]) / f64(n)
			quota[k] = int(math.floor(exact))
			rem[k] = exact - f64(quota[k])
			assigned += quota[k]
		}
		// remaining rows to the classes with the largest remainders (lowest class first on ties)
		for assigned < n_val {
			best := -1
			for k in 0 ..< K {
				if quota[k] < count[k] && (best < 0 || rem[k] > rem[best]) do best = k
			}
			quota[best] += 1
			rem[best] = -1
			assigned += 1
		}
		for i in order {
			c := codes[i]
			if quota[c] > 0 {in_val[i] = true; quota[c] -= 1}
		}
	}
	train = make([]int, n - n_val, allocator)
	val = make([]int, n_val, allocator)
	a, b := 0, 0
	for i in 0 ..< n {
		if in_val[i] {val[b] = i; b += 1} else {train[a] = i; a += 1}
	}
	return
}
