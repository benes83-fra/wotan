package ML

import l "../../linalg"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// Regression Metrics
// ============================================================================

// Mean Squared Error
metrics_mse :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n != len(y_pred) do panic("metrics_mse: length mismatch")
	if n == 0 do return math.nan_f64()

	sum_sq := 0.0
	for i in 0 ..< n {
		err := y_true[i] - y_pred[i]
		sum_sq += err * err
	}
	return sum_sq / f64(n)
}

// Mean Absolute Error
metrics_mae :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n != len(y_pred) do panic("metrics_mae: length mismatch")
	if n == 0 do return math.nan_f64()

	sum_abs := 0.0
	for i in 0 ..< n {
		sum_abs += math.abs(y_true[i] - y_pred[i])
	}
	return sum_abs / f64(n)
}

// R-squared (Coefficient of Determination)
metrics_r2 :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n != len(y_pred) do panic("metrics_r2: length mismatch")
	if n == 0 do return math.nan_f64()

	// Pass 1: Compute mean of y_true
	mean_y := 0.0
	for i in 0 ..< n {
		mean_y += y_true[i]
	}
	mean_y /= f64(n)

	// Pass 2: Compute Sum of Squared Residuals (SS_res) and Total Sum of Squares (SS_tot)
	ss_res := 0.0
	ss_tot := 0.0
	for i in 0 ..< n {
		residual := y_true[i] - y_pred[i]
		deviation := y_true[i] - mean_y
		ss_res += residual * residual
		ss_tot += deviation * deviation
	}

	if ss_tot == 0 {
		return 1.0 if ss_res == 0 else 0.0 // constant y_true (sklearn r2_score)
	}

	return 1.0 - (ss_res / ss_tot)
}

// ============================================================================
// Classification Metrics
// ============================================================================

// Accuracy
metrics_accuracy :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n != len(y_pred) do panic("metrics_accuracy: length mismatch")
	if n == 0 do return math.nan_f64()

	correct := 0
	for i in 0 ..< n {
		if y_true[i] == y_pred[i] {
			correct += 1
		}
	}
	return f64(correct) / f64(n)
}

// Confusion Matrix
// Returns a flat slice of size n_classes * n_classes (row-major: true_class, pred_class)
metrics_confusion_matrix :: proc(
	y_true: []f64,
	y_pred: []f64,
	classes: []f64,
	allocator: mem.Allocator = context.allocator,
) -> []int {
	n := len(y_true)
	n_classes := len(classes)
	if n == 0 || n != len(y_pred) || n_classes == 0 {
		return make([]int, 0, allocator)
	}

	cm := make([]int, n_classes * n_classes, allocator)

	for i in 0 ..< n {
		// Find true class index
		true_idx := -1
		for c_val, c_idx in classes {
			if c_val == y_true[i] {
				true_idx = c_idx
				break
			}
		}

		// Find predicted class index
		pred_idx := -1
		for c_val, c_idx in classes {
			if c_val == y_pred[i] {
				pred_idx = c_idx
				break
			}
		}

		// Increment count (ignore predictions for unknown classes)
		if true_idx >= 0 && pred_idx >= 0 {
			cm[true_idx * n_classes + pred_idx] += 1
		}
	}

	return cm
}

// Classification Report (Macro-averaged Precision, Recall, F1)
ClassificationReport :: struct {
	accuracy:        f64,
	macro_precision: f64,
	macro_recall:    f64,
	macro_f1:        f64,
	// Per-class metrics (slices of length n_classes)
	precision:       []f64,
	recall:          []f64,
	f1:              []f64,
	allocator:       mem.Allocator,
}

metrics_classification_report :: proc(
	y_true: []f64,
	y_pred: []f64,
	classes: []f64,
	allocator: mem.Allocator = context.allocator,
) -> ClassificationReport {
	n := len(y_true)
	n_classes := len(classes)

	report: ClassificationReport
	report.allocator = allocator
	report.accuracy = metrics_accuracy(y_true, y_pred)

	if n == 0 || n_classes == 0 {
		return report
	}

	// Allocate per-class slices
	report.precision = make([]f64, n_classes, allocator)
	report.recall = make([]f64, n_classes, allocator)
	report.f1 = make([]f64, n_classes, allocator)

	cm := metrics_confusion_matrix(y_true, y_pred, classes, context.temp_allocator)
	defer delete(cm, context.temp_allocator)

	sum_precision := 0.0
	sum_recall := 0.0
	sum_f1 := 0.0

	for i in 0 ..< n_classes {
		// True Positives: diagonal
		tp := f64(cm[i * n_classes + i])

		// False Positives: sum of column i, excluding diagonal
		fp := 0.0
		for r in 0 ..< n_classes {
			if r != i {
				fp += f64(cm[r * n_classes + i])
			}
		}

		// False Negatives: sum of row i, excluding diagonal
		fn := 0.0
		for c in 0 ..< n_classes {
			if c != i {
				fn += f64(cm[i * n_classes + c])
			}
		}

		// Compute metrics with division-by-zero protection
		p := 0.0
		if (tp + fp) > 0.0 {
			p = tp / (tp + fp)
		}

		r := 0.0
		if (tp + fn) > 0.0 {
			r = tp / (tp + fn)
		}

		f := 0.0
		if (p + r) > 0.0 {
			f = 2.0 * (p * r) / (p + r)
		}

		report.precision[i] = p
		report.recall[i] = r
		report.f1[i] = f

		sum_precision += p
		sum_recall += r
		sum_f1 += f
	}

	// Macro averages
	report.macro_precision = sum_precision / f64(n_classes)
	report.macro_recall = sum_recall / f64(n_classes)
	report.macro_f1 = sum_f1 / f64(n_classes)

	return report
}

// Free resources allocated by metrics_classification_report
metrics_free_classification_report :: proc(report: ^ClassificationReport) {
	if len(report.precision) > 0 {delete(report.precision, report.allocator)}
	if len(report.recall) > 0 {delete(report.recall, report.allocator)}
	if len(report.f1) > 0 {delete(report.f1, report.allocator)}
}

// ============================================================================
// More Regression Metrics
// ============================================================================

// Root Mean Squared Error
metrics_rmse :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	return math.sqrt(metrics_mse(y_true, y_pred))
}

// Mean Absolute Percentage Error as a fraction (sklearn: |y - ŷ| / max(|y|, eps),
// so y = 0 gives a huge but finite term)
metrics_mape :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n != len(y_pred) do panic("metrics_mape: length mismatch")
	if n == 0 do return math.nan_f64()
	s := 0.0
	for i in 0 ..< n {
		s += math.abs(y_true[i] - y_pred[i]) / max(math.abs(y_true[i]), math.F64_EPSILON)
	}
	return s / f64(n)
}

// Explained variance 1 - Var(y - ŷ) / Var(y) (unlike R², ignores a constant bias).
// Constant y_true: 1 if the residual variance is 0, else 0 (sklearn).
metrics_explained_variance :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n != len(y_pred) do panic("metrics_explained_variance: length mismatch")
	if n == 0 do return math.nan_f64()
	mean_y, mean_e := 0.0, 0.0
	for i in 0 ..< n {
		mean_y += y_true[i]
		mean_e += y_true[i] - y_pred[i]
	}
	mean_y /= f64(n)
	mean_e /= f64(n)
	var_y, var_e := 0.0, 0.0
	for i in 0 ..< n {
		dy := y_true[i] - mean_y
		de := y_true[i] - y_pred[i] - mean_e
		var_y += dy * dy
		var_e += de * de
	}
	if var_y == 0 do return 1.0 if var_e == 0 else 0.0
	return 1.0 - var_e / var_y
}

// ============================================================================
// Binary Classification: positive-class metrics
// ============================================================================

// Counts for "y == pos_label" (true) vs "pred == pos_label" (predicted)
_binary_counts :: proc(y_true, y_pred: []f64, pos_label: f64, name: string) -> (tp, fp, fn: f64) {
	if len(y_true) != len(y_pred) do panic(name)
	for i in 0 ..< len(y_true) {
		t := y_true[i] == pos_label
		p := y_pred[i] == pos_label
		if t && p do tp += 1
		if !t && p do fp += 1
		if t && !p do fn += 1
	}
	return
}

// Precision TP / (TP + FP) of the positive class; 0 if nothing is predicted
// positive (sklearn zero_division=0, without the warning)
metrics_precision :: proc(y_true, y_pred: []f64, pos_label: f64 = 1) -> f64 {
	tp, fp, _ := _binary_counts(y_true, y_pred, pos_label, "metrics_precision: length mismatch")
	return tp / (tp + fp) if tp + fp > 0 else 0
}

// Recall TP / (TP + FN) of the positive class; 0 if there are no positives
metrics_recall :: proc(y_true, y_pred: []f64, pos_label: f64 = 1) -> f64 {
	tp, _, fn := _binary_counts(y_true, y_pred, pos_label, "metrics_recall: length mismatch")
	return tp / (tp + fn) if tp + fn > 0 else 0
}

// F1 = 2TP / (2TP + FP + FN) of the positive class; 0 if undefined
metrics_f1 :: proc(y_true, y_pred: []f64, pos_label: f64 = 1) -> f64 {
	tp, fp, fn := _binary_counts(y_true, y_pred, pos_label, "metrics_f1: length mismatch")
	return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
}

// ============================================================================
// Probability / Score Based Metrics
// ============================================================================
// Scores can be probabilities of the positive class or any decision value
// (higher = more positive). Probability matrices are n × n_classes with columns
// in the order of `classes` (= the sorted labels, as *_predict_proba return).

// Area under the ROC curve (sklearn roc_auc_score, binary). Computed as the
// Mann-Whitney statistic: P(score of a positive > score of a negative), ties
// counted ½ — identical to the trapezoidal area under roc_curve.
// NaN if y_true contains only one class (sklearn raises).
metrics_roc_auc :: proc(y_true: []f64, scores: []f64, pos_label: f64 = 1) -> f64 {
	n := len(y_true)
	if n != len(scores) do panic("metrics_roc_auc: length mismatch")
	order := make([]int, n, context.temp_allocator)
	defer delete(order, context.temp_allocator)
	for i in 0 ..< n {
		if math.is_nan(scores[i]) do panic("metrics_roc_auc: NaN score")
		order[i] = i
	}
	_sort_by_score(order, scores, false)

	// Σ ranks of positives, ties get their average rank (1-based)
	n_pos, rank_sum := 0.0, 0.0
	i := 0
	for i < n {
		j := i
		for j + 1 < n && scores[order[j + 1]] == scores[order[i]] do j += 1
		avg_rank := f64(i + j + 2) / 2
		for k in i ..= j {
			if y_true[order[k]] == pos_label {
				n_pos += 1
				rank_sum += avg_rank
			}
		}
		i = j + 1
	}
	n_neg := f64(n) - n_pos
	if n_pos == 0 || n_neg == 0 do return math.nan_f64()
	return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

// Multiclass ROC AUC, one-vs-rest, macro average (sklearn multi_class="ovr",
// average="macro"): mean over classes of AUC(y == class, proba[:, class]).
metrics_roc_auc_ovr :: proc(y_true: []f64, proba: ^l.Matrix(f64), classes: []f64) -> f64 {
	n, K := len(y_true), len(classes)
	if proba.rows != n || proba.cols != K do panic("metrics_roc_auc_ovr: proba must be n × n_classes")
	col := make([]f64, n, context.temp_allocator)
	defer delete(col, context.temp_allocator)
	s := 0.0
	for k in 0 ..< K {
		for i in 0 ..< n {col[i] = proba.data[i * K + k]}
		s += metrics_roc_auc(y_true, col, classes[k])
	}
	return s / f64(K)
}

// ROC curve (sklearn roc_curve): false / true positive rates at every distinct
// score threshold, highest threshold first; the first point is (0, 0) with
// threshold +Inf (sklearn >= 1.3; 1.2 used max score + 1). drop_intermediate
// removes collinear points, as the sklearn default does.
ROCCurve :: struct {
	fpr, tpr, thresholds: []f64,
	allocator:            mem.Allocator,
}

metrics_roc_curve :: proc(
	y_true: []f64,
	scores: []f64,
	pos_label: f64 = 1,
	drop_intermediate := true,
	allocator: mem.Allocator = context.allocator,
) -> ROCCurve {
	n := len(y_true)
	if n != len(scores) do panic("metrics_roc_curve: length mismatch")
	tps, fps, thr := _cumulative_counts(y_true, scores, pos_label, context.temp_allocator)
	defer {
		delete(tps, context.temp_allocator)
		delete(fps, context.temp_allocator)
		delete(thr, context.temp_allocator)
	}
	m := len(thr)

	// keep point k unless it lies on the straight line through its neighbours
	keep := make([dynamic]int, 0, m, context.temp_allocator)
	defer delete(keep)
	for k in 0 ..< m {
		if drop_intermediate && k > 0 && k < m - 1 {
			d2_f := fps[k + 1] - 2 * fps[k] + fps[k - 1]
			d2_t := tps[k + 1] - 2 * tps[k] + tps[k - 1]
			if d2_f == 0 && d2_t == 0 do continue
		}
		append(&keep, k)
	}

	P := tps[m - 1] if m > 0 else 0
	N := fps[m - 1] if m > 0 else 0
	c: ROCCurve
	c.allocator = allocator
	c.fpr = make([]f64, len(keep) + 1, allocator)
	c.tpr = make([]f64, len(keep) + 1, allocator)
	c.thresholds = make([]f64, len(keep) + 1, allocator)
	c.thresholds[0] = math.inf_f64(1)
	for k, j in keep {
		// sklearn: NaN rates when a class is absent
		c.fpr[j + 1] = fps[k] / N if N > 0 else math.nan_f64()
		c.tpr[j + 1] = tps[k] / P if P > 0 else math.nan_f64()
		c.thresholds[j + 1] = thr[k]
	}
	if N == 0 do c.fpr[0] = math.nan_f64()
	if P == 0 do c.tpr[0] = math.nan_f64()
	return c
}

metrics_roc_curve_free :: proc(c: ^ROCCurve) {
	delete(c.fpr, c.allocator)
	delete(c.tpr, c.allocator)
	delete(c.thresholds, c.allocator)
}

// Average precision (sklearn average_precision_score): Σ (R_k - R_{k-1}) · P_k
// over the distinct thresholds, highest first (step-wise area under the
// precision-recall curve, no interpolation). NaN if there are no positives.
metrics_average_precision :: proc(y_true: []f64, scores: []f64, pos_label: f64 = 1) -> f64 {
	if len(y_true) != len(scores) do panic("metrics_average_precision: length mismatch")
	tps, fps, thr := _cumulative_counts(y_true, scores, pos_label, context.temp_allocator)
	defer {
		delete(tps, context.temp_allocator)
		delete(fps, context.temp_allocator)
		delete(thr, context.temp_allocator)
	}
	m := len(thr)
	if m == 0 || tps[m - 1] == 0 do return math.nan_f64()
	P := tps[m - 1]
	ap, prev_r := 0.0, 0.0
	for k in 0 ..< m {
		r := tps[k] / P
		ap += (r - prev_r) * tps[k] / (tps[k] + fps[k])
		prev_r = r
	}
	return ap
}

// Log loss / cross-entropy (sklearn log_loss): -mean log p(true class), with
// probabilities clipped to [eps, 1 - eps] and every row renormalised to sum 1
// (eps = machine epsilon, the sklearn default eps="auto"). Labels must be in `classes`.
metrics_log_loss :: proc(y_true: []f64, proba: ^l.Matrix(f64), classes: []f64, eps := math.F64_EPSILON) -> f64 {
	n, K := len(y_true), len(classes)
	if proba.rows != n || proba.cols != K do panic("metrics_log_loss: proba must be n × n_classes")
	if n == 0 do return math.nan_f64()
	loss := 0.0
	for i in 0 ..< n {
		t := -1
		for c, k in classes {if c == y_true[i] {t = k; break}}
		if t < 0 do panic("metrics_log_loss: y_true contains a label that is not in classes")
		row_sum, p_t := 0.0, 0.0
		for k in 0 ..< K {
			p := math.clamp(proba.data[i * K + k], eps, 1 - eps)
			row_sum += p
			if k == t do p_t = p
		}
		loss -= math.ln(p_t / row_sum)
	}
	return loss / f64(n)
}

// Binary log loss from the positive-class probability p
metrics_log_loss_binary :: proc(y_true: []f64, p_pos: []f64, pos_label: f64 = 1, eps := math.F64_EPSILON) -> f64 {
	n := len(y_true)
	if n != len(p_pos) do panic("metrics_log_loss_binary: length mismatch")
	if n == 0 do return math.nan_f64()
	loss := 0.0
	for i in 0 ..< n {
		p := math.clamp(p_pos[i], eps, 1 - eps)
		loss -= math.ln(p) if y_true[i] == pos_label else math.ln(1 - p)
	}
	return loss / f64(n)
}

// Brier score (sklearn brier_score_loss): mean (1[y == pos_label] - p)²
metrics_brier_score :: proc(y_true: []f64, p_pos: []f64, pos_label: f64 = 1) -> f64 {
	n := len(y_true)
	if n != len(p_pos) do panic("metrics_brier_score: length mismatch")
	if n == 0 do return math.nan_f64()
	s := 0.0
	for i in 0 ..< n {
		t := 1.0 if y_true[i] == pos_label else 0.0
		s += (t - p_pos[i]) * (t - p_pos[i])
	}
	return s / f64(n)
}

// Samples sorted by score (descending); at every distinct score: cumulative
// true / false positives and the score itself (sklearn _binary_clf_curve)
_cumulative_counts :: proc(
	y_true, scores: []f64,
	pos_label: f64,
	allocator: mem.Allocator,
) -> (
	tps, fps, thresholds: []f64,
) {
	n := len(y_true)
	order := make([]int, n, allocator)
	defer delete(order, allocator)
	for i in 0 ..< n {
		if math.is_nan(scores[i]) do panic("metrics: NaN score")
		order[i] = i
	}
	_sort_by_score(order, scores, true)
	t := make([dynamic]f64, 0, n, allocator)
	f := make([dynamic]f64, 0, n, allocator)
	h := make([dynamic]f64, 0, n, allocator)
	tp, fp := 0.0, 0.0
	for k in 0 ..< n {
		if y_true[order[k]] == pos_label {tp += 1} else {fp += 1}
		if k == n - 1 || scores[order[k + 1]] != scores[order[k]] {
			append(&t, tp)
			append(&f, fp)
			append(&h, scores[order[k]])
		}
	}
	return t[:], f[:], h[:]
}

// Sort sample indices by score (stable; descending if desc)
_sort_by_score :: proc(order: []int, scores: []f64, desc: bool) {
	Ctx :: struct {
		scores: []f64,
		desc:   bool,
	}
	ctx := Ctx{scores, desc}
	context.user_ptr = &ctx
	slice.stable_sort_by(order, proc(a, b: int) -> bool {
		c := (^Ctx)(context.user_ptr)
		return c.scores[a] > c.scores[b] if c.desc else c.scores[a] < c.scores[b]
	})
}

// ============================================================================
// Clustering
// ============================================================================

// Mean silhouette coefficient (sklearn silhouette_score, Euclidean):
// s_i = (b_i - a_i) / max(a_i, b_i), a_i = mean distance to the other members of
// its cluster, b_i = smallest mean distance to another cluster; s_i = 0 for
// singleton clusters. Needs 2 <= n_clusters <= n - 1. O(n²) distances.
metrics_silhouette :: proc(X: ^l.Matrix(f64), labels: []int) -> f64 {
	n, d := X.rows, X.cols
	if len(labels) != n do panic("metrics_silhouette: labels length mismatch")
	n_lab := 0
	for v in labels {
		if v < 0 do panic("metrics_silhouette: negative label")
		n_lab = max(n_lab, v + 1)
	}
	size := make([]int, n_lab, context.temp_allocator)
	sums := make([]f64, n_lab, context.temp_allocator)
	defer {
		delete(size, context.temp_allocator)
		delete(sums, context.temp_allocator)
	}
	for v in labels {size[v] += 1}
	n_clusters := 0
	for c in size {if c > 0 do n_clusters += 1}
	if n_clusters < 2 || n_clusters > n - 1 do panic("metrics_silhouette: need 2 <= n_clusters <= n - 1")

	total := 0.0
	for i in 0 ..< n {
		for &s in sums {s = 0}
		for j in 0 ..< n {
			if j == i do continue
			dist2 := 0.0
			for k in 0 ..< d {
				diff := X.data[i * d + k] - X.data[j * d + k]
				dist2 += diff * diff
			}
			sums[labels[j]] += math.sqrt(dist2)
		}
		own := labels[i]
		if size[own] == 1 do continue // s_i = 0
		a := sums[own] / f64(size[own] - 1)
		b := math.inf_f64(1)
		for c in 0 ..< n_lab {
			if c == own || size[c] == 0 do continue
			b = min(b, sums[c] / f64(size[c]))
		}
		if max(a, b) > 0 do total += (b - a) / max(a, b)
	}
	return total / f64(n)
}
