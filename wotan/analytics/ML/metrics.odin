package ML

import "core:math"
import "core:mem"

// ============================================================================
// Regression Metrics
// ============================================================================

// Mean Squared Error
metrics_mse :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n == 0 || n != len(y_pred) {return 0.0}

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
	if n == 0 || n != len(y_pred) {return 0.0}

	sum_abs := 0.0
	for i in 0 ..< n {
		sum_abs += math.abs(y_true[i] - y_pred[i])
	}
	return sum_abs / f64(n)
}

// R-squared (Coefficient of Determination)
metrics_r2 :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n == 0 || n != len(y_pred) {return 0.0}

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

	if ss_tot < 1e-12 {
		return 1.0 // Perfect prediction if variance is zero
	}

	return 1.0 - (ss_res / ss_tot)
}

// ============================================================================
// Classification Metrics
// ============================================================================

// Accuracy
metrics_accuracy :: proc(y_true: []f64, y_pred: []f64) -> f64 {
	n := len(y_true)
	if n == 0 || n != len(y_pred) {return 0.0}

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
