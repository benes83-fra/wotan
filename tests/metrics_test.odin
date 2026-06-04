package tests
import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"

metrics_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Metrics Module ===")

	// ---------------------------------------------------------
	// 1. Regression Metrics Test (using PCR data)
	// ---------------------------------------------------------
	n_reg := 200
	X_reg := l.matrix_new(f64, n_reg, 5, allocator)
	y_reg := make([]f64, n_reg, allocator)
	y_pred_reg := make([]f64, n_reg, allocator)

	for i in 0 ..< n_reg {
		base := rand.float64_normal(0, 1)
		X_reg.data[i * 5 + 0] = base + rand.float64_normal(0, 0.1)
		X_reg.data[i * 5 + 1] = base + rand.float64_normal(0, 0.1)
		X_reg.data[i * 5 + 2] = base + rand.float64_normal(0, 0.1)
		X_reg.data[i * 5 + 3] = rand.float64_normal(0, 1)
		X_reg.data[i * 5 + 4] = rand.float64_normal(0, 1)

		y_reg[i] = 3.0 * base + rand.float64_normal(0, 0.5)
	}

	// Fit PCR
	pcr_params := ml.PCRParams {
		n_components       = 0,
		variance_threshold = 0.95,
	}
	pcr_model := ml.pcr_fit(&X_reg, y_reg, pcr_params, allocator)
	defer ml.pcr_free(&pcr_model)

	// Predict
	preds_reg := ml.pcr_predict(&pcr_model, &X_reg, allocator)
	defer delete(preds_reg, allocator)

	// Evaluate
	mse := ml.metrics_mse(y_reg, preds_reg)
	mae := ml.metrics_mae(y_reg, preds_reg)
	r2 := ml.metrics_r2(y_reg, preds_reg)

	fmt.printf("PCR Regression Metrics:\n")
	fmt.printf("  MSE:  %.4f\n", mse)
	fmt.printf("  MAE:  %.4f\n", mae)
	fmt.printf("  R2:   %.4f\n", r2)


	// ---------------------------------------------------------
	// 2. Classification Metrics Test (using Gaussian NB data)
	// ---------------------------------------------------------
	n_cls := 300
	X_cls := l.matrix_new(f64, n_cls, 4, allocator)
	y_cls := make([]f64, n_cls, allocator)
	classes := []f64{0.0, 1.0, 2.0}

	for i in 0 ..< n_cls {
		class_idx := i / 100 // 0, 1, or 2
		y_cls[i] = f64(class_idx)
		for f in 0 ..< 4 {
			center := f64(class_idx) * 5.0 + f64(f)
			X_cls.data[i * 4 + f] = center + rand.float64_normal(0, 1.0)
		}
	}

	// Fit & Predict Gaussian NB
	gnb_model := ml.gnb_fit(&X_cls, y_cls, 1e-9, allocator)
	defer ml.gnb_free(&gnb_model)

	preds_cls := ml.gnb_predict(&gnb_model, &X_cls, allocator)
	defer delete(preds_cls, allocator)

	// Evaluate
	acc := ml.metrics_accuracy(y_cls, preds_cls)
	report := ml.metrics_classification_report(y_cls, preds_cls, classes, allocator)
	defer ml.metrics_free_classification_report(&report)

	fmt.printf("\nGaussian NB Classification Metrics:\n")
	fmt.printf("  Accuracy:        %.2f%%\n", acc * 100)
	fmt.printf("  Macro Precision: %.4f\n", report.macro_precision)
	fmt.printf("  Macro Recall:    %.4f\n", report.macro_recall)
	fmt.printf("  Macro F1:        %.4f\n", report.macro_f1)

	// Cleanup
	l.matrix_free(&X_reg)
	delete(y_reg, allocator)
	l.matrix_free(&X_cls)
	delete(y_cls, allocator)
}
