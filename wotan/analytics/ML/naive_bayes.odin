package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Gaussian Naive Bayes Structures
// ============================================================================

GaussianNB :: struct {
	classes:    []f64, // Unique class labels [n_classes]
	theta:      l.Matrix(f64), // Means [n_classes, n_features]
	inv_sigma:  l.Matrix(f64), // 1 / (2 * variance) [n_classes, n_features]
	joint_bias: []f64, // log(prior) - 0.5 * sum(log(2*pi*var)) [n_classes]
	allocator:  mem.Allocator,
}

// ============================================================================
// Public API: Fit Gaussian Naive Bayes
// ============================================================================

gnb_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	epsilon: f64 = 1e-9, // Smoothing term to prevent division by zero
	allocator: mem.Allocator = context.allocator,
) -> GaussianNB {
	n_samples := X.rows
	n_features := X.cols

	// 1. Find unique classes
	unique_classes := make([dynamic]f64, 0, allocator)
	for val in y {
		found := false
		for u in unique_classes {
			if u == val {found = true; break}
		}
		if !found {append(&unique_classes, val)}
	}
	n_classes := len(unique_classes)

	classes := make([]f64, n_classes, allocator)
	copy(classes, unique_classes[:])
	delete(unique_classes)

	// 2. Allocate matrices
	theta := l.matrix_new(f64, n_classes, n_features, allocator)
	inv_sigma := l.matrix_new(f64, n_classes, n_features, allocator)
	joint_bias := make([]f64, n_classes, allocator)

	two_pi := 2.0 * math.PI

	// 3. Compute statistics for each class
	for k in 0 ..< n_classes {
		cls := classes[k]

		// Count samples in this class
		count := 0
		for i in 0 ..< n_samples {
			if y[i] == cls {count += 1}
		}

		if count == 0 {continue}

		prior := f64(count) / f64(n_samples)
		bias := math.ln_f64(prior)

		// Two-pass algorithm for numerical stability
		// Pass 1: Compute means
		for j in 0 ..< n_features {
			sum := 0.0
			for i in 0 ..< n_samples {
				if y[i] == cls {
					sum += X.data[i * n_features + j]
				}
			}
			mean := sum / f64(count)
			theta.data[k * n_features + j] = mean

			// Pass 2: Compute variance
			sum_sq_diff := 0.0
			for i in 0 ..< n_samples {
				if y[i] == cls {
					diff := X.data[i * n_features + j] - mean
					sum_sq_diff += diff * diff
				}
			}
			var := (sum_sq_diff / f64(count)) + epsilon

			// Precompute inverse variance for fast prediction
			inv_sigma.data[k * n_features + j] = 1.0 / (2.0 * var)

			// Accumulate the constant log term into the joint bias
			bias -= 0.5 * math.ln_f64(two_pi * var)
		}

		joint_bias[k] = bias
	}

	return GaussianNB {
		classes = classes,
		theta = theta,
		inv_sigma = inv_sigma,
		joint_bias = joint_bias,
		allocator = allocator,
	}
}

// ============================================================================
// Public API: Predict
// Highly optimized: reduces to a simple subtraction, square, and dot product.
// ============================================================================

gnb_predict :: proc(
	model: ^GaussianNB,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_samples := X.rows
	n_features := X.cols
	n_classes := len(model.classes)

	preds := make([]f64, n_samples, allocator)
	if n_samples == 0 {return preds}

	for i in 0 ..< n_samples {
		x_row := X.data[i * n_features:i * n_features + n_features]

		best_class := model.classes[0]
		best_score := math.F64_MIN // Safe lower bound for f64

		for k in 0 ..< n_classes {
			theta_row := model.theta.data[k * n_features:k * n_features + n_features]
			inv_sig_row := model.inv_sigma.data[k * n_features:k * n_features + n_features]

			// Compute sum( (x - mu)^2 * inv_sigma )
			sum := 0.0
			for j in 0 ..< n_features {
				d := x_row[j] - theta_row[j]
				sum += d * d * inv_sig_row[j]
			}

			// Score = joint_bias - penalty
			score := model.joint_bias[k] - sum

			if score > best_score {
				best_score = score
				best_class = model.classes[k]
			}
		}
		preds[i] = best_class
	}

	return preds
}

// ============================================================================
// Public API: Free Resources
// ============================================================================

gnb_free :: proc(model: ^GaussianNB) {
	if len(model.classes) > 0 {delete(model.classes, model.allocator)}
	if model.theta.data != nil {l.matrix_free(&model.theta)}
	if model.inv_sigma.data != nil {l.matrix_free(&model.inv_sigma)}
	if len(model.joint_bias) > 0 {delete(model.joint_bias, model.allocator)}
}
