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
	Purged_KFold, // contiguous folds; training samples whose labels overlap the test labels are purged, an embargo follows each test block
	Combinatorial_Purged, // CPCV: n_splits groups, every choice of n_test_groups of them as the test set, purged and embargoed
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
	// Purged_KFold / Combinatorial_Purged (samples in time order, sample i observed at i)
	label_horizon:  int, // sample i's label uses observations i ..= i + label_horizon
	label_end:      []int, // per sample, the last observation its label uses (>= i); overrides label_horizon
	embargo:        int, // samples after each test block's last label also kept out of training
	n_test_groups:  int, // Combinatorial_Purged: groups per test set, 1 <= k < n_splits (0 = 2)
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
// - Purged_KFold (López de Prado 2018, ch. 7): the KFold blocks, never shuffled. Sample
//   i's label spans observations [i, e_i] (e_i = label_end[i], or i + label_horizon).
//   A test block [a, b) covers [a, max e_i over the block]; a training sample is purged
//   when its span intersects that interval (closed), and the `embargo` samples after the
//   interval's end are dropped as well. With label_horizon = embargo = 0 it is KFold
//   without shuffling.
// - Combinatorial_Purged (CPCV, ch. 12): n_splits = N contiguous groups (the KFold blocks);
//   each of the C(N, k) choices of k = n_test_groups groups (lexicographic, as
//   itertools.combinations) is a test set, purged and embargoed around every run of
//   adjacent test groups. cv_cpcv_paths assembles the splits into C(N-1, k-1) backtest
//   paths. Returns C(N, k) folds, not n_splits.
cv_make_folds :: proc(
	cv: CVSplitter,
	n: int,
	y: []f64 = nil,
	allocator: mem.Allocator = context.allocator,
) -> []CVFold {
	k := cv.n_splits
	if k < 2 do panic("cv_make_folds: need n_splits >= 2")
	if cv.kind == .Purged_KFold || cv.kind == .Combinatorial_Purged {
		return _purged_folds(cv, n, allocator)
	}
	folds := make([]CVFold, k, allocator)

	switch cv.kind {
	case .Purged_KFold, .Combinatorial_Purged:
		unreachable()
	case .KFold:
		if k > n do panic("cv_make_folds: KFold needs n_splits <= n_samples")
		indices := make([]int, n, context.temp_allocator)
		defer delete(indices, context.temp_allocator)
		for i in 0 ..< n {indices[i] = i}
		if cv.shuffle do _shuffle_indices(indices, cv.random_state)
		bounds := cv_block_starts(n, k, context.temp_allocator)
		defer delete(bounds, context.temp_allocator)
		for f in 0 ..< k {
			start, end := bounds[f], bounds[f + 1]
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

// Boundaries of k contiguous blocks of n samples, as numpy.array_split / KFold: block b
// is [starts[b], starts[b + 1]), the first n % k blocks one sample longer (k + 1 entries)
cv_block_starts :: proc(n, k: int, allocator: mem.Allocator = context.allocator) -> []int {
	if k < 1 || k > n do panic("cv_block_starts: need 1 <= k <= n")
	starts := make([]int, k + 1, allocator)
	for b in 0 ..< k {starts[b + 1] = starts[b] + n / k + (1 if b < n % k else 0)}
	return starts
}

// Number of k-element subsets of n elements
binomial :: proc(n, k: int) -> int {
	if k < 0 || k > n do return 0
	c := 1
	for i in 0 ..< min(k, n - k) {c = c * (n - i) / (i + 1)}
	return c
}

// Purged_KFold and Combinatorial_Purged
@(private = "file")
_purged_folds :: proc(cv: CVSplitter, n: int, allocator: mem.Allocator) -> []CVFold {
	N := cv.n_splits
	if N > n do panic("cv_make_folds: purged CV needs n_splits <= n_samples")
	if cv.shuffle do panic("cv_make_folds: purged CV keeps the time order (shuffle must be false)")
	if cv.label_horizon < 0 || cv.embargo < 0 do panic("cv_make_folds: negative label_horizon / embargo")
	if cv.label_end != nil && len(cv.label_end) != n do panic("cv_make_folds: label_end needs one entry per sample")
	k := 1
	if cv.kind == .Combinatorial_Purged {
		k = cv.n_test_groups if cv.n_test_groups > 0 else 2
		if k >= N do panic("cv_make_folds: need n_test_groups < n_splits")
	}
	label_end :: proc(cv: CVSplitter, i: int) -> int {
		e := cv.label_end[i] if cv.label_end != nil else i + cv.label_horizon
		if e < i do panic("cv_make_folds: label_end[i] < i")
		return e
	}
	starts := cv_block_starts(n, N, context.temp_allocator)
	defer delete(starts, context.temp_allocator)
	folds := make([]CVFold, binomial(N, k), allocator)
	groups := make([]int, k, context.temp_allocator)
	defer delete(groups, context.temp_allocator)
	for i in 0 ..< k {groups[i] = i}
	is_test := make([]bool, N, context.temp_allocator)
	defer delete(is_test, context.temp_allocator)
	keep := make([]bool, n, context.temp_allocator)
	defer delete(keep, context.temp_allocator)
	for &fold in folds {
		for &v in is_test {v = false}
		for g in groups {is_test[g] = true}
		n_test := 0
		for g in groups {n_test += starts[g + 1] - starts[g]}
		for &v in keep {v = true}
		// every run of adjacent test groups [a, b): purge spans meeting [a, last label],
		// embargo the samples after it
		g := 0
		for g < N {
			if !is_test[g] {g += 1; continue}
			h := g
			for h < N && is_test[h] {h += 1}
			a, b := starts[g], starts[h]
			last := a
			for i in a ..< b {last = max(last, label_end(cv, i))}
			for j in 0 ..< n {
				if j <= last && label_end(cv, j) >= a {keep[j] = false}
				if j > last && j <= last + cv.embargo {keep[j] = false}
			}
			g = h
		}
		n_train := 0
		for v in keep {if v do n_train += 1}
		fold.train = make([]int, n_train, allocator)
		fold.test = make([]int, n_test, allocator)
		a, b := 0, 0
		for i in 0 ..< n {
			if keep[i] {fold.train[a] = i; a += 1}
		}
		for gg in groups {
			for i in starts[gg] ..< starts[gg + 1] {fold.test[b] = i; b += 1}
		}
		// next combination
		i := k - 1
		for i >= 0 && groups[i] == N - k + i {i -= 1}
		if i < 0 do break
		groups[i] += 1
		for j in i + 1 ..< k {groups[j] = groups[j - 1] + 1}
	}
	return folds
}

// CPCV backtest paths: every group is a test group in C(N-1, k-1) of the C(N, k) splits;
// path p takes, for each group g, the p-th of those splits (in split order). Returns
// paths[p * N + g] = the split whose out-of-sample values form group g of path p, and the
// number of paths, C(N-1, k-1) (López de Prado 2018, ch. 12.4).
cv_cpcv_paths :: proc(
	n_groups, n_test_groups: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	paths: []int,
	n_paths: int,
) {
	N, k := n_groups, n_test_groups
	if k < 1 || k >= N do panic("cv_cpcv_paths: need 1 <= n_test_groups < n_groups")
	n_paths = binomial(N - 1, k - 1)
	paths = make([]int, n_paths * N, allocator)
	used := make([]int, N, context.temp_allocator) // paths filled so far per group
	defer delete(used, context.temp_allocator)
	groups := make([]int, k, context.temp_allocator)
	defer delete(groups, context.temp_allocator)
	for i in 0 ..< k {groups[i] = i}
	for s in 0 ..< binomial(N, k) {
		for g in groups {
			paths[used[g] * N + g] = s
			used[g] += 1
		}
		i := k - 1
		for i >= 0 && groups[i] == N - k + i {i -= 1}
		if i < 0 do break
		groups[i] += 1
		for j in i + 1 ..< k {groups[j] = groups[j - 1] + 1}
	}
	return
}

// One CPCV backtest path: split_values[s] holds split s's out-of-sample values at its test
// indices (length n_samples, other entries ignored); the path takes group g from split
// paths[path * N + g]
cv_cpcv_path :: proc(
	split_values: [][]f64,
	paths: []int,
	n_groups, path: int,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n := len(split_values[0])
	starts := cv_block_starts(n, n_groups, context.temp_allocator)
	defer delete(starts, context.temp_allocator)
	out := make([]f64, n, allocator)
	for g in 0 ..< n_groups {
		src := split_values[paths[path * n_groups + g]]
		copy(out[starts[g]:starts[g + 1]], src[starts[g]:starts[g + 1]])
	}
	return out
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
