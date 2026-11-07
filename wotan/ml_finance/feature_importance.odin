package ml_finance

import ml "../analytics/ML"
import l "../linalg"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// FEATURE IMPORTANCE (López de Prado, AFML ch. 8; MLAM ch. 6)
// ============================================================================
//   MDI  mean decrease impurity of a forest: per-tree impurity importances, a feature
//        a tree never split on (max_features = 1: never offered) excluded from that
//        tree's mean (AFML 8.2); in sample, every feature gets some.
//   MDA  mean decrease accuracy: out of sample, per (purged) CV fold, the score lost
//        when a feature's test column is permuted, relative to the permuted score
//        (AFML 8.3).
//   SFI  single feature importance: the CV score of the model on each feature alone
//        (AFML 8.4); no substitution effects, no joint effects.
//   Clustered MDI / MDA (MLAM 6.4 / 6.5): the same over groups of features found by
//        cluster_features (k-means on the correlation distance, the number of
//        clusters by the silhouette t-statistic, MLAM 4.1); substitutes share a group,
//        so their importance is no longer split between them.
// Results: mean and standard error (std over trees / folds, ddof 1, times n^-1/2) per
// feature or group; MDI normalised so the means sum to 1.

ImportanceScoring :: enum {
	Neg_Log_Loss, // -log loss of the class probabilities (AFML's default)
	Accuracy,
}

ImportanceResult :: struct {
	mean:      []f64, // per feature, or per group when groups are given
	std_err:   []f64,
	allocator: mem.Allocator,
}

importance_result_free :: proc(r: ^ImportanceResult) {
	delete(r.mean, r.allocator)
	delete(r.std_err, r.allocator)
}

// A classifier the out-of-sample methods fit and query. fit returns the fitted model;
// predict_proba gives the [n, K] class probabilities and the model's class labels
// (borrowed from the model).
ImportanceModel :: struct {
	fit:           proc(
		X: ^l.Matrix(f64),
		y: []f64,
		sample_weight: []f64,
		user: rawptr,
		allocator: mem.Allocator,
	) -> rawptr,
	predict_proba: proc(
		model: rawptr,
		X: ^l.Matrix(f64),
		allocator: mem.Allocator,
	) -> (
		l.Matrix(f64),
		[]f64,
	),
	free:          proc(model: rawptr, allocator: mem.Allocator),
	user:          rawptr,
}

// A classification random forest (params.criterion .Gini / .Entropy); params must
// outlive the model's use
rf_importance_model :: proc(params: ^ml.RFParams) -> ImportanceModel {
	assert(params.criterion != .MSE, "rf_importance_model: needs a classification criterion")
	return ImportanceModel {
		fit = proc(
			X: ^l.Matrix(f64),
			y: []f64,
			w: []f64,
			user: rawptr,
			allocator: mem.Allocator,
		) -> rawptr {
			m := new(ml.RandomForest, allocator)
			m^ = ml.rf_fit(X, y, (^ml.RFParams)(user)^, allocator, w)
			return m
		},
		predict_proba = proc(
			model: rawptr,
			X: ^l.Matrix(f64),
			allocator: mem.Allocator,
		) -> (
			l.Matrix(f64),
			[]f64,
		) {
			m := (^ml.RandomForest)(model)
			return ml.rf_predict_proba(m, X, allocator), m.classes
		},
		free = proc(model: rawptr, allocator: mem.Allocator) {
			m := (^ml.RandomForest)(model)
			ml.rf_free(m)
			free(m, allocator)
		},
		user = params,
	}
}

// A classification tree (params.criterion .Gini / .Entropy); params must outlive the use
dt_importance_model :: proc(params: ^ml.TreeParams) -> ImportanceModel {
	assert(params.criterion != .MSE, "dt_importance_model: needs a classification criterion")
	return ImportanceModel {
		fit = proc(
			X: ^l.Matrix(f64),
			y: []f64,
			w: []f64,
			user: rawptr,
			allocator: mem.Allocator,
		) -> rawptr {
			m := new(ml.DecisionTree, allocator)
			m^ = ml.dt_fit(X, y, (^ml.TreeParams)(user)^, allocator, w)
			return m
		},
		predict_proba = proc(
			model: rawptr,
			X: ^l.Matrix(f64),
			allocator: mem.Allocator,
		) -> (
			l.Matrix(f64),
			[]f64,
		) {
			m := (^ml.DecisionTree)(model)
			return ml.dt_predict_proba(m, X, allocator), m.classes
		},
		free = proc(model: rawptr, allocator: mem.Allocator) {
			m := (^ml.DecisionTree)(model)
			ml.dt_free(m)
			free(m, allocator)
		},
		user = params,
	}
}

// ----------------------------------------------------------------------------
// Aggregation (pure)
// ----------------------------------------------------------------------------

// Column means and standard errors of a [n, k] matrix, NaN skipped (pandas mean / std
// with ddof 1), the standard error with n = all rows (AFML: std * n^-0.5)
@(private = "file")
_mean_std_err :: proc(m: ^l.Matrix(f64), allocator: mem.Allocator) -> ImportanceResult {
	n, k := m.rows, m.cols
	r := ImportanceResult {
		mean      = make([]f64, k, allocator),
		std_err   = make([]f64, k, allocator),
		allocator = allocator,
	}
	for j in 0 ..< k {
		s, c := 0.0, 0
		for i in 0 ..< n {
			v := m.data[i * k + j]
			if !math.is_nan(v) {s += v; c += 1}
		}
		if c == 0 {
			r.mean[j], r.std_err[j] = math.nan_f64(), math.nan_f64()
			continue
		}
		mu := s / f64(c)
		ss := 0.0
		for i in 0 ..< n {
			v := m.data[i * k + j]
			if !math.is_nan(v) {ss += (v - mu) * (v - mu)}
		}
		r.mean[j] = mu
		r.std_err[j] = math.sqrt(ss / f64(c - 1)) / math.sqrt(f64(n)) if c > 1 else math.nan_f64()
	}
	return r
}

// MDI from per-tree importances [n_trees, n_features] (each row from
// ml.tree_feature_importance). Without groups (AFML 8.2) a zero means the tree never
// used the feature and is skipped; with groups (MLAM 6.4) a group's importance in a
// tree is the sum over its features. Means and standard errors divided by the sum of
// the means (NaN means skipped).
mdi_from_tree_importances :: proc(
	imp: ^l.Matrix(f64),
	groups: [][]int = nil,
	allocator := context.allocator,
) -> ImportanceResult {
	T, F := imp.rows, imp.cols
	k := F if groups == nil else len(groups)
	m := l.matrix_new(f64, T, k, context.temp_allocator)
	defer l.matrix_free(&m)
	for t in 0 ..< T {
		row := imp.data[t * F:][:F]
		if groups == nil {
			for v, j in row {m.data[t * k + j] = math.nan_f64() if v == 0 else v}
		} else {
			for g, j in groups {
				s := 0.0
				for f in g {s += row[f]}
				m.data[t * k + j] = s
			}
		}
	}
	r := _mean_std_err(&m, allocator)
	total := 0.0
	for v in r.mean {if !math.is_nan(v) {total += v}}
	if total > 0 {
		for j in 0 ..< k {r.mean[j] /= total; r.std_err[j] /= total}
	}
	return r
}

// MDA from the fold scores: base[i] the score of fold i, permuted [n_folds, k] with the
// feature (group) j permuted. Importance (base - permuted) / -permuted for the negative
// log loss, / (1 - permuted) for accuracy (AFML 8.3): the share of the remaining loss
// (or error) the feature explains. Mean and standard error over the folds.
mda_from_scores :: proc(
	base: []f64,
	permuted: ^l.Matrix(f64),
	scoring: ImportanceScoring,
	allocator := context.allocator,
) -> ImportanceResult {
	n, k := permuted.rows, permuted.cols
	assert(len(base) == n, "mda_from_scores: one base score per fold")
	m := l.matrix_new(f64, n, k, context.temp_allocator)
	defer l.matrix_free(&m)
	for i in 0 ..< n {
		for j in 0 ..< k {
			p := permuted.data[i * k + j]
			m.data[i * k + j] = (base[i] - p) / (-p if scoring == .Neg_Log_Loss else 1 - p)
		}
	}
	return _mean_std_err(&m, allocator)
}

// ----------------------------------------------------------------------------
// Methods
// ----------------------------------------------------------------------------

// MDI of a fitted forest (fit it with max_features = 1 so that no feature is masked)
mdi_importance :: proc(
	forest: ^ml.RandomForest,
	n_features: int,
	groups: [][]int = nil,
	allocator := context.allocator,
) -> ImportanceResult {
	T := len(forest.trees)
	imp := l.matrix_new(f64, T, n_features, context.temp_allocator)
	defer l.matrix_free(&imp)
	for &tree, t in forest.trees {
		ti := ml.tree_feature_importance(&tree, n_features, context.temp_allocator)
		copy(imp.data[t * n_features:][:n_features], ti)
		delete(ti, context.temp_allocator)
	}
	return mdi_from_tree_importances(&imp, groups, allocator)
}

// Score of a fitted model on (X, y, w): -log loss over the classes of the whole data
// set (a class the model never saw gets probability 0, clipped), or accuracy of the
// most probable class
@(private = "file")
_score :: proc(
	model: ImportanceModel,
	fitted: rawptr,
	X: ^l.Matrix(f64),
	y, w, classes: []f64,
	scoring: ImportanceScoring,
) -> f64 {
	p, mc := model.predict_proba(fitted, X, context.temp_allocator)
	defer l.matrix_free(&p)
	K := len(classes)
	full := l.matrix_new(f64, X.rows, K, context.temp_allocator)
	defer l.matrix_free(&full)
	pred := make([]f64, X.rows, context.temp_allocator)
	defer delete(pred, context.temp_allocator)
	for c, k in mc {
		g := -1
		for v, j in classes {if v == c {g = j; break}}
		if g < 0 {continue}
		for i in 0 ..< X.rows {full.data[i * K + g] = p.data[i * len(mc) + k]}
	}
	if scoring == .Neg_Log_Loss {
		return -ml.metrics_log_loss(y, &full, classes, sample_weight = w)
	}
	for i in 0 ..< X.rows {
		best := 0
		for k in 1 ..< K {if full.data[i * K + k] > full.data[i * K + best] {best = k}}
		pred[i] = classes[best]
	}
	return ml.metrics_accuracy(y, pred, w)
}

@(private = "file")
_sorted_classes :: proc(y: []f64, allocator: mem.Allocator) -> []f64 {
	c := make([dynamic]f64, allocator)
	for v in y {
		found := false
		for u in c {if u == v {found = true; break}}
		if !found {append(&c, v)}
	}
	for i in 1 ..< len(c) {
		for j := i; j > 0 && c[j - 1] > c[j]; j -= 1 {c[j - 1], c[j] = c[j], c[j - 1]}
	}
	return c[:]
}

// In-place Fisher-Yates shuffle; with state != nil from the LCG
// x <- 6364136223846793005 x + 1442695040888963407, u = (x >> 11) 2^-53, swapping
// i = n-1 .. 1 with j = floor(u (i + 1)) (reproducible elsewhere), else core:math/rand
@(private = "file")
_shuffle :: proc(v: []f64, state: ^u64) {
	if state == nil {rand.shuffle(v); return}
	for i := len(v) - 1; i > 0; i -= 1 {
		state^ = 6364136223846793005 * state^ + 1442695040888963407
		j := int(f64(state^ >> 11) * 1.1102230246251565e-16 * f64(i + 1))
		v[i], v[j] = v[j], v[i]
	}
}

// Rows `idx` of X (copied) and of y / w (w nil: nil)
@(private = "file")
_rows :: proc(
	X: ^l.Matrix(f64),
	y, w: []f64,
	idx: []int,
	allocator: mem.Allocator,
) -> (
	Xs: l.Matrix(f64),
	ys, ws: []f64,
) {
	Xs = ml.subset_matrix(X, idx, allocator)
	ys = ml.subset_slice(y, idx, allocator)
	if w != nil {ws = ml.subset_slice(w, idx, allocator)}
	return
}

// MDA over the splits of cv (use .Purged_KFold with the label ends for overlapping
// labels). Every feature of a group is permuted (independently) within the test rows;
// groups nil: every feature alone. seed != 0 makes the permutations reproducible.
// Also returns the mean unpermuted out-of-sample score.
mda_importance :: proc(
	model: ImportanceModel,
	X: ^l.Matrix(f64),
	y: []f64,
	cv: ml.CVSplitter,
	sample_weight: []f64 = nil,
	scoring := ImportanceScoring.Neg_Log_Loss,
	groups: [][]int = nil,
	seed: u64 = 0,
	allocator := context.allocator,
) -> (
	result: ImportanceResult,
	oos_score: f64,
) {
	assert(len(y) == X.rows, "mda_importance: y length mismatch")
	F := X.cols
	k := F if groups == nil else len(groups)
	classes := _sorted_classes(y, context.temp_allocator)
	defer delete(classes, context.temp_allocator)
	folds := ml.cv_make_folds(cv, X.rows, y, context.temp_allocator)
	defer ml.cv_folds_free(folds, context.temp_allocator)
	base := make([]f64, len(folds), context.temp_allocator)
	defer delete(base, context.temp_allocator)
	perm := l.matrix_new(f64, len(folds), k, context.temp_allocator)
	defer l.matrix_free(&perm)
	state := seed
	st: ^u64 = &state if seed != 0 else nil
	for f, i in folds {
		X0, y0, w0 := _rows(X, y, sample_weight, f.train, context.temp_allocator)
		X1, y1, w1 := _rows(X, y, sample_weight, f.test, context.temp_allocator)
		fitted := model.fit(&X0, y0, w0, model.user, context.temp_allocator)
		base[i] = _score(model, fitted, &X1, y1, w1, classes, scoring)
		oos_score += base[i] / f64(len(folds))
		Xp := l.matrix_new(f64, X1.rows, F, context.temp_allocator)
		col := make([]f64, X1.rows, context.temp_allocator)
		for j in 0 ..< k {
			copy(Xp.data, X1.data)
			one := [1]int{j}
			members := groups[j] if groups != nil else one[:]
			for c in members {
				for r in 0 ..< X1.rows {col[r] = X1.data[r * F + c]}
				_shuffle(col, st)
				for r in 0 ..< X1.rows {Xp.data[r * F + c] = col[r]}
			}
			perm.data[i * k + j] = _score(model, fitted, &Xp, y1, w1, classes, scoring)
		}
		model.free(fitted, context.temp_allocator)
		l.matrix_free(&X0)
		l.matrix_free(&X1)
		l.matrix_free(&Xp)
		delete(col, context.temp_allocator)
		delete(y0, context.temp_allocator)
		delete(y1, context.temp_allocator)
		if w0 != nil {delete(w0, context.temp_allocator); delete(w1, context.temp_allocator)}
	}
	return mda_from_scores(base, &perm, scoring, allocator), oos_score
}

// SFI: per feature, the mean and standard error of the model's CV scores on that
// feature alone (not normalised; compare with the score of a constant prediction)
sfi_importance :: proc(
	model: ImportanceModel,
	X: ^l.Matrix(f64),
	y: []f64,
	cv: ml.CVSplitter,
	sample_weight: []f64 = nil,
	scoring := ImportanceScoring.Neg_Log_Loss,
	allocator := context.allocator,
) -> ImportanceResult {
	assert(len(y) == X.rows, "sfi_importance: y length mismatch")
	F := X.cols
	classes := _sorted_classes(y, context.temp_allocator)
	defer delete(classes, context.temp_allocator)
	folds := ml.cv_make_folds(cv, X.rows, y, context.temp_allocator)
	defer ml.cv_folds_free(folds, context.temp_allocator)
	scores := l.matrix_new(f64, len(folds), F, context.temp_allocator)
	defer l.matrix_free(&scores)
	Xj := l.matrix_new(f64, X.rows, 1, context.temp_allocator)
	defer l.matrix_free(&Xj)
	for j in 0 ..< F {
		for r in 0 ..< X.rows {Xj.data[r] = X.data[r * F + j]}
		for f, i in folds {
			X0, y0, w0 := _rows(&Xj, y, sample_weight, f.train, context.temp_allocator)
			X1, y1, w1 := _rows(&Xj, y, sample_weight, f.test, context.temp_allocator)
			fitted := model.fit(&X0, y0, w0, model.user, context.temp_allocator)
			scores.data[i * F + j] = _score(model, fitted, &X1, y1, w1, classes, scoring)
			model.free(fitted, context.temp_allocator)
			l.matrix_free(&X0)
			l.matrix_free(&X1)
			delete(y0, context.temp_allocator)
			delete(y1, context.temp_allocator)
			if w0 != nil {delete(w0, context.temp_allocator); delete(w1, context.temp_allocator)}
		}
	}
	return _mean_std_err(&scores, allocator)
}

// ----------------------------------------------------------------------------
// Feature clusters (MLAM 4.1, clusterKMeansBase)
// ----------------------------------------------------------------------------

FeatureClusters :: struct {
	labels:     []int, // cluster of each feature
	groups:     [][]int, // features of each cluster, in label order
	silhouette: []f64, // per feature
	allocator:  mem.Allocator,
}

feature_clusters_free :: proc(c: ^FeatureClusters) {
	for g in c.groups {delete(g, c.allocator)}
	delete(c.groups, c.allocator)
	delete(c.labels, c.allocator)
	delete(c.silhouette, c.allocator)
}

// Rows of the correlation distance sqrt((1 - corr) / 2) of the features (NaN
// correlations as 0) clustered by k-means for k = 2 .. max_clusters (at most
// n_features - 1), n_init times each with one k-means++ start; kept: the clustering
// with the highest silhouette mean / std (std with ddof 0).
cluster_features :: proc(
	X: ^l.Matrix(f64),
	max_clusters := 10,
	n_init := 10,
	allocator := context.allocator,
) -> FeatureClusters {
	F := X.cols
	assert(F >= 3, "cluster_features: needs at least 3 features")
	corr := l.correlation(X, context.temp_allocator)
	defer l.matrix_free(&corr)
	D := l.matrix_new(f64, F, F, context.temp_allocator)
	defer l.matrix_free(&D)
	for v, i in corr.data {
		c := 0.0 if math.is_nan(v) else clamp(v, -1, 1)
		D.data[i] = math.sqrt((1 - c) / 2)
	}
	// exactly 0 on the diagonal: the computed self-correlation is 1 up to an ulp, and
	// the square root turns that ulp into a 1e-8 self-distance
	for i in 0 ..< F {D.data[i * F + i] = 0}
	best_stat := math.nan_f64()
	out := FeatureClusters {
		allocator = allocator,
	}
	for _ in 0 ..< max(n_init, 1) {
		for k in 2 ..= min(max_clusters, F - 1) {
			km := ml.km_fit(
				&D,
				{n_clusters = k, max_iter = 300, tol = 1e-4, init = .KMeansPlusPlus, n_init = 1},
				context.temp_allocator,
			)
			defer ml.km_free(&km)
			// k-means can leave a cluster empty only if the data has fewer distinct rows
			used := make([]bool, k, context.temp_allocator)
			n_used := 0
			for c in km.labels {if !used[c] {used[c] = true; n_used += 1}}
			if n_used < 2 {continue}
			s := ml.metrics_silhouette_samples(&D, km.labels, context.temp_allocator)
			m, v := 0.0, 0.0
			for x in s {m += x / f64(F)}
			for x in s {v += (x - m) * (x - m) / f64(F)}
			stat := m / math.sqrt(v)
			if math.is_nan(best_stat) || stat > best_stat {
				best_stat = stat
				delete(out.labels, allocator)
				delete(out.silhouette, allocator)
				out.labels = make([]int, F, allocator)
				copy(out.labels, km.labels)
				out.silhouette = make([]f64, F, allocator)
				copy(out.silhouette, s)
			}
		}
	}
	// groups in label order, empty labels dropped and the labels renumbered
	n_lab := 0
	for c in out.labels {n_lab = max(n_lab, c + 1)}
	remap := make([]int, n_lab, context.temp_allocator)
	for &r in remap {r = -1}
	groups := make([dynamic][]int, allocator)
	for c in 0 ..< n_lab {
		members := make([dynamic]int, allocator)
		for lab, f in out.labels {if lab == c {append(&members, f)}}
		if len(members) == 0 {delete(members); continue}
		remap[c] = len(groups)
		append(&groups, members[:])
	}
	for &lab in out.labels {lab = remap[lab]}
	out.groups = groups[:]
	return out
}
