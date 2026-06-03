package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:mem"


gnb_test :: proc(allocator: mem.Allocator) {
	// Generate 3 distinct clusters (Classes: 0.0, 1.0, 2.0)
	n := 300
	X := l.matrix_new(f64, n, 4, allocator)
	y := make([]f64, n, allocator)

	for i in 0 ..< n {
		class_idx := i / 100 // 0, 1, or 2
		y[i] = f64(class_idx)

		// Center the clusters far apart in 4D space
		for f in 0 ..< 4 {
			center := f64(class_idx) * 5.0 + f64(f)
			X.data[i * 4 + f] = center + rand.float64_normal(0, 1.0)
		}
	}

	// Fit Gaussian Naive Bayes
	model := ml.gnb_fit(&X, y, 1e-9, allocator)
	defer ml.gnb_free(&model)

	preds := ml.gnb_predict(&model, &X, allocator)
	defer delete(preds, allocator)

	correct := 0
	for i in 0 ..< n {if preds[i] == y[i] {correct += 1}}
	fmt.printf("Gaussian NB Accuracy: %.2f%%\n", f64(correct) / f64(n) * 100)

	l.matrix_free(&X)
	delete(y, allocator)
}
