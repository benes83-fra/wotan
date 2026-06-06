package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"


mlp_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Multi-Layer Perceptron ===")

	// Generate non-linear data (XOR problem scaled up)
	n := 200
	X := l.matrix_new(f64, n, 2, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		x0 := rand.float64_normal(0, 1)
		x1 := rand.float64_normal(0, 1)
		X.data[i * 2 + 0] = x0
		X.data[i * 2 + 1] = x1

		// XOR-like decision boundary
		if (x0 > 0 && x1 > 0) || (x0 < 0 && x1 < 0) {
			y[i] = 1.0
		} else {
			y[i] = 0.0
		}
	}

	// Build Pipeline: StandardScaler -> MLP
	pipe := ml.pipeline_new(allocator)
	defer ml.pipeline_free(&pipe)

	ml.pipeline_add_standard_scaler(&pipe)

	mlp_params := ml.MLPParams {
		hidden_layers     = []int{16, 16}, // Two hidden layers
		activation        = .ReLU,
		output_activation = .Sigmoid,
		task              = .BinaryClassification,
		learning_rate     = 0.01,
		max_iter          = 500,
		batch_size        = 0, // Full batch
		optimizer_type    = .Adam,
	}
	ml.pipeline_set_mlp(&pipe, mlp_params)

	ml.pipeline_fit(&pipe, &X, y)

	preds := ml.pipeline_predict(&pipe, &X, allocator)
	defer delete(preds, allocator)

	// Round predictions for accuracy
	for i in 0 ..< len(preds) {
		if preds[i] >= 0.5 {preds[i] = 1.0} else {preds[i] = 0.0}
	}

	// ✅ DEBUG: See exactly what the network is predicting vs the true labels
	fmt.println("\n--- First 10 Predictions vs Labels ---")
	for i in 0 ..< 10 {
		fmt.printf("pred: %.2f | true: %.2f\n", preds[i], y[i])
	}
	fmt.println("--------------------------------------")

	acc := ml.metrics_accuracy(y, preds)
	fmt.printf("MLP (ReLU/Adam) Accuracy: %.2f%%\n", acc * 100)

	l.matrix_free(&X)
	delete(y, allocator)
}
