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
	// every fold must be non-empty (sklearn KFold: 2 <= k <= n)
	if k < 2 || k > n do panic("cv_folds: need 2 <= k <= n")
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

// ============================================================================
// Cross-Validation Splitters (sklearn KFold / StratifiedKFold / TimeSeriesSplit)
// ============================================================================

CVKind :: enum {
	KFold, // contiguous folds of the (optionally shuffled) samples
	Stratified_KFold, // every fold keeps the class proportions of y
	Time_Series, // walk-forward: train on the past, test on the next block
}

CVSplitter :: struct {
	kind:           CVKind,
	n_splits:       int, // number of train/test splits (>= 2)
	// KFold / Stratified_KFold
	shuffle:        bool,
	random_state:   int, // seed of the shuffle (own LCG, not sklearn's RNG)
	// Time_Series
	test_size:      int, // samples per test block (0 = n / (n_splits + 1))
	gap:            int, // samples dropped between train end and test start (purging
	// against look-ahead from overlapping labels, e.g. multi-day returns)
	max_train_size: int, // 0 = expanding window; > 0 = rolling window of this length
}

CVFold :: struct {
	train, test: []int,
}

// Train/test index sets of every split, as sklearn's splitter.split(X, y):
// - KFold: test folds are contiguous blocks of the sample order (shuffled if
//   shuffle), the first n % n_splits folds one sample larger
// - Stratified_KFold (sklearn's algorithm, exact when shuffle = false): classes
//   numbered by first appearance, samples sorted by class are dealt round-robin
//   to the folds, so every fold gets ⌊n_c/k⌋ or ⌈n_c/k⌉ samples of class c;
//   within a class the folds are assigned in sample order (shuffled if shuffle).
//   Indices ascending. Needs y.
// - Time_Series: test blocks of test_size samples at the end of the series, the
//   training set is everything before (block start - gap), the last
//   max_train_size samples of it if set. Indices ascending; never shuffled.
cv_make_folds :: proc(
	cv: CVSplitter,
	n: int,
	y: []f64 = nil,
	allocator: mem.Allocator = context.allocator,
) -> []CVFold {
	k := cv.n_splits
	if k < 2 do panic("cv_make_folds: need n_splits >= 2")
	folds := make([]CVFold, k, allocator)

	switch cv.kind {
	case .KFold:
		if k > n do panic("cv_make_folds: KFold needs n_splits <= n_samples")
		indices := make([]int, n, context.temp_allocator)
		defer delete(indices, context.temp_allocator)
		for i in 0 ..< n {indices[i] = i}
		if cv.shuffle do _shuffle_indices(indices, cv.random_state)
		fold_size, remainder := n / k, n % k
		for f in 0 ..< k {
			start := f * fold_size + min(f, remainder)
			end := start + fold_size + (1 if f < remainder else 0)
			folds[f].test = make([]int, end - start, allocator)
			copy(folds[f].test, indices[start:end])
			folds[f].train = make([]int, n - (end - start), allocator)
			copy(folds[f].train[:start], indices[:start])
			copy(folds[f].train[start:], indices[end:])
		}

	case .Stratified_KFold:
		if len(y) != n do panic("cv_make_folds: Stratified_KFold needs y of length n_samples")
		if k > n do panic("cv_make_folds: Stratified_KFold needs n_splits <= n_samples")
		// classes in order of first appearance
		code := make([]int, n, context.temp_allocator)
		defer delete(code, context.temp_allocator)
		seen := make([dynamic]f64, 0, 8, context.temp_allocator)
		defer delete(seen)
		for v, i in y {
			c := -1
			for s, j in seen {if s == v {c = j; break}}
			if c < 0 {c = len(seen); append(&seen, v)}
			code[i] = c
		}
		C := len(seen)
		count := make([]int, C, context.temp_allocator)
		defer delete(count, context.temp_allocator)
		for c in code {count[c] += 1}
		max_count := 0
		for c in count {max_count = max(max_count, c)}
		if k > max_count do panic("cv_make_folds: n_splits is larger than every class count")

		// allocation[f][c] = #samples of class c among y_sorted[f], y_sorted[f + k], ...
		alloc := make([]int, k * C, context.temp_allocator)
		defer delete(alloc, context.temp_allocator)
		pos := 0
		for c in 0 ..< C {
			for _ in 0 ..< count[c] {
				alloc[(pos % k) * C + c] += 1
				pos += 1
			}
		}
		test_fold := make([]int, n, context.temp_allocator)
		defer delete(test_fold, context.temp_allocator)
		for c in 0 ..< C {
			// fold numbers for the members of class c: 0 repeated alloc[0][c] times, 1 ...
			ff := make([]int, count[c], context.temp_allocator)
			defer delete(ff, context.temp_allocator)
			p := 0
			for f in 0 ..< k {
				for _ in 0 ..< alloc[f * C + c] {ff[p] = f; p += 1}
			}
			if cv.shuffle do _shuffle_indices(ff, cv.random_state + c)
			p = 0
			for i in 0 ..< n {
				if code[i] == c {test_fold[i] = ff[p]; p += 1}
			}
		}
		for f in 0 ..< k {
			nt := 0
			for t in test_fold {if t == f do nt += 1}
			folds[f].test = make([]int, nt, allocator)
			folds[f].train = make([]int, n - nt, allocator)
			a, b := 0, 0
			for t, i in test_fold {
				if t == f {folds[f].test[a] = i; a += 1} else {folds[f].train[b] = i; b += 1}
			}
		}

	case .Time_Series:
		if k + 1 > n do panic("cv_make_folds: Time_Series needs n_splits + 1 <= n_samples")
		if cv.gap < 0 || cv.test_size < 0 || cv.max_train_size < 0 do panic("cv_make_folds: negative gap / test_size / max_train_size")
		ts := cv.test_size if cv.test_size > 0 else n / (k + 1)
		if n - cv.gap - ts * k <= 0 do panic("cv_make_folds: too few samples for n_splits, test_size and gap")
		first := n - k * ts
		for f in 0 ..< k {
			test_start := first + f * ts
			train_end := test_start - cv.gap
			train_start := 0
			if cv.max_train_size > 0 && cv.max_train_size < train_end do train_start = train_end - cv.max_train_size
			folds[f].train = make([]int, train_end - train_start, allocator)
			for i in 0 ..< train_end - train_start {folds[f].train[i] = train_start + i}
			folds[f].test = make([]int, ts, allocator)
			for i in 0 ..< ts {folds[f].test[i] = test_start + i}
		}
	}
	return folds
}

cv_folds_free :: proc(folds: []CVFold, allocator: mem.Allocator = context.allocator) {
	for f in folds {
		delete(f.train, allocator)
		delete(f.test, allocator)
	}
	delete(folds, allocator)
}

// Score of the evaluator on every split of cv (one entry per split)
cross_val_score_cv :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	cv: CVSplitter,
	evaluator: ModelEvaluator,
	user_data: rawptr,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	if len(y) != X.rows do panic("cross_val_score_cv: y length mismatch")
	folds := cv_make_folds(cv, X.rows, y, context.temp_allocator)
	defer cv_folds_free(folds, context.temp_allocator)
	scores := make([]f64, len(folds), allocator)
	for f, i in folds {
		scores[i] = evaluator(X, y, f.train, f.test, user_data, allocator)
	}
	return scores
}
