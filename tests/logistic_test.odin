package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"


logistic_test :: proc(allocator: mem.Allocator) {
	// Generate linearly separable 2D data (Labels: 0.0 or 1.0)
	n := 200
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		x0 := rand.float64_normal(0, 1)
		x1 := rand.float64_normal(0, 1)
		X.data[i * 2 + 0] = x0
		X.data[i * 2 + 1] = x1
		// Label: 1.0 if x0 + x1 > 0, else 0.0
		if x0 + x1 > 0 {
			y[i] = 1.0
		} else {
			y[i] = 0.0
		}
	}

	// Try L-BFGS (It should converge in ~10-20 iterations!)
	params := ml.LogisticParams {
		C              = 1.0,
		max_iter       = 100, // L-BFGS needs very few iterations
		tol            = 1e-5,
		learning_rate  = 1.0, // L-BFGS default step size
		fit_intercept  = true,
		optimizer_type = .LBFGS, // ← Try .SGD or .Adam here too!
	}

	model := ml.logistic_fit(&X, y, params, allocator)
	defer ml.logistic_free(&model)

	fmt.printf(
		"Logistic Regression converged: %v after %v iterations\n",
		model.converged,
		model.n_iter,
	)
	fmt.printf(
		"Weights: [%.3f, %.3f], Bias: %.3f\n",
		model.weights[0],
		model.weights[1],
		model.bias,
	)

	// Predict and compute accuracy
	preds := ml.logistic_predict(&model, &X, allocator)
	defer delete(preds, allocator)

	correct := 0
	for i in 0 ..< n {
		if preds[i] == y[i] {
			correct += 1
		}
	}
	accuracy := f64(correct) / f64(n)
	fmt.printf("Training accuracy: %.2f%%\n", accuracy * 100)

	l.matrix_free(&X)
	delete(y, allocator)
}
