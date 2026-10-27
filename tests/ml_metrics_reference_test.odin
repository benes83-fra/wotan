package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math"
import "core:mem"

// Probability / score based and additional metrics (reference: sklearn 1.2.1):
// ROC AUC (ties, non-0/1 labels), ROC curve (with and without dropping collinear
// points), average precision, log loss (clipping, renormalisation), Brier score,
// precision / recall / F1 for a positive label, multiclass OvR AUC, RMSE, MAPE
// (zero target), explained variance, silhouette (singleton cluster).

@(private = "file")
MT_YB := [?]f64{-1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, 2.0, 2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}
@(private = "file")
MT_SB := [?]f64{0.2, 0.5, 0.5, 0.7, 0.5, 0.7, 0.5, 0.4, 0.8, 0.2, 0.8, 0.9, 0.3, 0.6, 0.2, 0.5, 0.8, 0.0, 0.4, 0.7, 0.7, 0.5, 0.7, 0.5, 0.4, 0.3, 1.0, 0.6, 1.0, 0.9}
@(private = "file")
MT_PRED_B := [?]f64{-1.0, -1.0, -1.0, 2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 2.0, 2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, 2.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0}
@(private = "file")
MT_YC := [?]f64{0.0, 1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 3.0, 0.0}
@(private = "file")
MT_P := [?]f64{0.5809682096141975, 0.05545785650691532, 0.3635739338788872, 0.19424149516407344, 0.059926514302716796, 0.7458319905332097, 0.0, 0.3, 0.7, 0.037670692248748514, 0.27834873917962966, 0.6839805685716219, 0.054404730571310955, 0.7794116494902666, 0.1661836199384224, 0.0, 0.5, 0.5, 0.47558344935381985, 0.0355539745868449, 0.48886257605933536, 0.2, 0.2, 0.4, 0.16883166755549955, 0.39924997095605713, 0.4319183614884433, 0.49812886680576884, 0.00031586396866349327, 0.5015552692255677, 0.09700176694524582, 0.7493054197855954, 0.15369281326915885, 0.24158825322076852, 0.24660146110313094, 0.5118102856761007}
@(private = "file")
MT_PN := [?]f64{0.5809682096141975, 0.05545785650691532, 0.3635739338788872, 0.19424149516407344, 0.059926514302716796, 0.7458319905332097, 0.0, 0.3, 0.7, 0.037670692248748514, 0.27834873917962966, 0.6839805685716219, 0.054404730571310955, 0.7794116494902666, 0.1661836199384224, 0.0, 0.5, 0.5, 0.47558344935381985, 0.0355539745868449, 0.48886257605933536, 0.25, 0.25, 0.5, 0.16883166755549955, 0.39924997095605713, 0.4319183614884433, 0.49812886680576884, 0.00031586396866349327, 0.5015552692255677, 0.09700176694524582, 0.7493054197855954, 0.15369281326915885, 0.24158825322076852, 0.24660146110313094, 0.5118102856761007}
@(private = "file")
MT_YR := [?]f64{3.0, -1.5, 0.0, 2.2, 7.1, 4.4, -0.3, 1.0}
@(private = "file")
MT_PR := [?]f64{3.8, -1.2, 0.6, 2.3000000000000003, 7.8, 4.9, 0.30000000000000004, 1.4}
@(private = "file")
MT_XS := [?]f64{0.265, -0.464, -0.479, -0.721, -0.52, 0.16, -0.38, 0.1, 1.901, 0.479, 1.424, 4.734, 3.348, 2.059, 3.907, 3.018, 2.385, 2.366, 2.007, 3.048, 1.069, 3.675, 0.421, 5.876, -1.215, 4.258, -0.306, 2.941, -1.026, 3.985}
@(private = "file")
MT_LAB := [?]int{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3}
@(private = "file")
REF_FPR := [?]f64{0.0, 0.0625, 0.0625, 0.125, 0.25, 0.375, 0.5625, 0.6875, 0.75, 0.9375, 1.0}
@(private = "file")
REF_TPR := [?]f64{0.0, 0.07142857142857142, 0.21428571428571427, 0.35714285714285715, 0.5714285714285714, 0.5714285714285714, 0.8571428571428571, 0.9285714285714286, 1.0, 1.0, 1.0}
@(private = "file")
REF_THR := [?]f64{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0} // sklearn 1.2 thresholds[0] = max + 1; Odin uses +Inf
@(private = "file")
REF_FPR_ALL := [?]f64{0.0, 0.0625, 0.0625, 0.125, 0.25, 0.375, 0.5625, 0.6875, 0.75, 0.9375, 1.0}

ml_metrics_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== ML Metrics vs sklearn ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	near :: proc(got, want: f64) -> bool {return math.abs(got - want) <= 1e-13 * max(1, math.abs(want))}
	close :: proc(got, want: []f64) -> bool {
		if len(got) != len(want) do return false
		for g, i in got {if !(math.abs(g - want[i]) <= 1e-15) do return false}
		return true
	}

	yb, sb := MT_YB[:], MT_SB[:]
	expect(&passed, &total, near(ml.metrics_roc_auc(yb, sb, 2), 0.7053571428571428), "ROC AUC (tied scores, pos_label 2)")
	expect(&passed, &total, near(ml.metrics_average_precision(yb, sb, 2), 0.62817242281528), "average precision")
	expect(&passed, &total, near(ml.metrics_brier_score(yb, sb, 2), 0.22800000000000004), "Brier score")
	expect(&passed, &total, near(ml.metrics_log_loss_binary(yb, sb, 2), 1.7546900661748253), "binary log loss (p = 0 / 1 clipped)")
	roc := ml.metrics_roc_curve(yb, sb, 2)
	expect(&passed, &total, close(roc.fpr, REF_FPR[:]) && close(roc.tpr, REF_TPR[:]) && close(roc.thresholds[1:], REF_THR[:]) && math.is_inf(roc.thresholds[0], 1), "ROC curve (collinear points dropped)")
	ml.metrics_roc_curve_free(&roc)
	roc_all := ml.metrics_roc_curve(yb, sb, 2, drop_intermediate = false)
	expect(&passed, &total, close(roc_all.fpr, REF_FPR_ALL[:]), "ROC curve (all thresholds)")
	ml.metrics_roc_curve_free(&roc_all)
	rc := ml.metrics_roc_curve([]f64{1, 0, 0, 0, 1, 1, 0}, []f64{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3})
	expect(&passed, &total, close(rc.fpr, {0, 0, 0.75, 0.75, 1}) && close(rc.tpr, {0, 1.0 / 3, 1.0 / 3, 1, 1}) && close(rc.thresholds[1:], {0.9, 0.6, 0.4, 0.3}), "ROC curve drops collinear points like sklearn (8 → 5 points)")
	ml.metrics_roc_curve_free(&rc)
	pb := MT_PRED_B[:]
	expect(&passed, &total, near(ml.metrics_precision(yb, pb, 2), 0.5714285714285714) && near(ml.metrics_recall(yb, pb, 2), 0.5714285714285714) && near(ml.metrics_f1(yb, pb, 2), 0.5714285714285714), "precision / recall / F1 for pos_label 2")

	classes := []f64{0, 1, 3}
	P := l.Matrix(f64){data = MT_P[:], rows = 12, cols = 3}
	expect(&passed, &total, near(ml.metrics_log_loss(MT_YC[:], &P, classes), 4.507925992510541), "multiclass log loss (p = 0 clipped, row renormalised)")
	Pn := l.Matrix(f64){data = MT_PN[:], rows = 12, cols = 3}
	expect(&passed, &total, near(ml.metrics_roc_auc_ovr(MT_YC[:], &Pn, classes), 0.5833333333333334), "multiclass ROC AUC (OvR, macro)")

	yr, pr := MT_YR[:], MT_PR[:]
	expect(&passed, &total, near(ml.metrics_rmse(yr, pr), 0.5431390245600107), "RMSE")
	expect(&passed, &total, near(ml.metrics_mape(yr, pr), 337769972052787.56), "MAPE (zero target → eps denominator)")
	expect(&passed, &total, near(ml.metrics_explained_variance(yr, pr), 0.9934957880710946), "explained variance (biased predictions)")

	Xs := l.Matrix(f64){data = MT_XS[:], rows = 15, cols = 2}
	expect(&passed, &total, near(ml.metrics_silhouette(&Xs, MT_LAB[:]), 0.23313103481476347), "silhouette score (with a singleton cluster)")

	// edge cases
	expect(&passed, &total, math.is_nan(ml.metrics_roc_auc([]f64{1, 1, 1}, []f64{0.2, 0.5, 0.9})), "ROC AUC with one class → NaN")
	expect(&passed, &total, ml.metrics_roc_auc([]f64{0, 0, 1, 1}, []f64{0.5, 0.5, 0.5, 0.5}) == 0.5, "ROC AUC of constant scores = 0.5")
	expect(&passed, &total, ml.metrics_precision([]f64{1, 0}, []f64{0, 0}) == 0, "precision with no positive predictions = 0")

	fmt.printf("ML metrics summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
