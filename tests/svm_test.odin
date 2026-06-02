
package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import optim "../wotan/optimize"
import "core:fmt"
import "core:math/rand"
import "core:mem"

svm_test :: proc(allocator: mem.Allocator) {
	// Generate linearly separable 2D data
	n := 100
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		x0 := rand.float64_normal(0, 1)
		x1 := rand.float64_normal(0, 1)
		X.data[i * 2 + 0] = x0
		X.data[i * 2 + 1] = x1
		// Label: +1 if x0 + x1 > 0, else -1
		tmp: f64
		if x0 + x1 > 0 {
			tmp = 1.0
		} else {
			tmp = -1.0
		}
		y[i] = tmp
	}

	params := ml.SVMParams {
		C             = 1.0,
		max_iter      = 10000,
		tol           = 1e-3,
		learning_rate = 0.01,
		fit_intercept = true,
	}

	model := ml.svm_fit_linear(&X, y, params, allocator)
	defer ml.svm_free(&model)

	fmt.printf("Linear SVM converged: %v after %v iterations\n", model.converged, model.n_iter)
	fmt.printf(
		"Weights: [%.3f, %.3f], Bias: %.3f\n",
		model.weights[0],
		model.weights[1],
		model.bias,
	)

	// Predict on training data
	preds := ml.svm_predict_linear(&model, &X, allocator)
	defer delete(preds, allocator)

	// Compute accuracy
	correct := 0
	for i in 0 ..< n {
		tmp: f64
		if preds[i] > 0 {
			tmp = 1.0
		} else {
			tmp = -1.0
		}
		pred_class := tmp
		if pred_class == y[i] {correct += 1}
	}
	accuracy := f64(correct) / f64(n)
	fmt.printf("Training accuracy: %.2f%%\n", accuracy * 100)

	l.matrix_free(&X)
	delete(y, allocator)
}

kernel_svm_test :: proc(allocator: mem.Allocator) {
	// Generate non-linearly separable data (XOR-like)
	n := 100
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		x0 := rand.float64_normal(0, 1)
		x1 := rand.float64_normal(0, 1)
		X.data[i * 2 + 0] = x0
		X.data[i * 2 + 1] = x1
		// XOR pattern: +1 if x0*x1 > 0, else -1
		tmp: f64
		if x0 + x1 > 0 {
			tmp = 1.0
		} else {
			tmp = -1.0
		}
		y[i] = tmp
	}

	params := ml.KernelSVMParams {
		C             = 10.0,
		gamma         = 1.0, // RBF width
		kernel_type   = .RBF,
		max_iter      = 500,
		tol           = 1e-3,
		learning_rate = 0.01,
	}

	model := ml.kernel_svm_fit(&X, y, params, allocator)
	defer ml.kernel_svm_free(&model)

	fmt.printf(
		"Kernel SVM: %v support vectors, bias=%.3f\n",
		len(model.support_vectors),
		model.bias,
	)

	// Predict on training data
	preds := ml.kernel_svm_predict(&model, &X, allocator)
	defer delete(preds, allocator)

	// Compute accuracy
	correct := 0
	for i in 0 ..< n {
		tmp: f64
		if preds[i] > 0 {
			tmp = 1.0
		} else {
			tmp = -1.0
		}
		pred_class := tmp
		if pred_class == y[i] {correct += 1}
	}
	accuracy := f64(correct) / f64(n)
	fmt.printf("Training accuracy (RBF kernel): %.2f%%\n", accuracy * 100)

	l.matrix_free(&X)
	delete(y, allocator)
}
