package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Principal Component Regression (PCR) Structures
// ============================================================================

PCRegression :: struct {
	components:   l.Matrix(f64), // [n_components, n_features] (Top K eigenvectors)
	means:        []f64, // [n_features] (Training feature means for centering)
	betas:        []f64, // [n_components] (OLS coefficients on components)
	bias:         f64, // Intercept (mean of y)
	n_components: int,
	allocator:    mem.Allocator,
}

PCRParams :: struct {
	n_components:       int, // If <= 0, automatically selects based on variance_threshold
	variance_threshold: f64, // e.g., 0.95 to retain 95% of variance
}

// ============================================================================
// Public API: Fit Principal Component Regression
// ============================================================================

pcr_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: PCRParams,
	allocator: mem.Allocator = context.allocator,
) -> PCRegression {
	n_samples := X.rows
	n_features := X.cols
	if n_samples <= 1 {panic("pcr_fit: need at least 2 samples")}

	// 1. Compute means and center X (using temp allocator)
	means := make([]f64, n_features, allocator)
	X_centered := l.matrix_new(f64, n_samples, n_features, context.temp_allocator)
	defer l.matrix_free(&X_centered)

	for j in 0 ..< n_features {
		sum := 0.0
		for i in 0 ..< n_samples {sum += X.data[i * n_features + j]}
		mean := sum / f64(n_samples)
		means[j] = mean
		for i in 0 ..< n_samples {
			X_centered.data[i * n_features + j] = X.data[i * n_features + j] - mean
		}
	}

	// 2. Compute Covariance Matrix: cov = (X_c^T X_c) / (n - 1)
	cov := l.xtx_simd(&X_centered, context.temp_allocator)
	defer l.matrix_free(&cov)

	scale := 1.0 / f64(n_samples - 1)
	for i in 0 ..< cov.rows * cov.cols {cov.data[i] *= scale}

	// 3. Eigendecomposition (Ascending order)
	eigenvalues, eigenvectors := l.eigh(&cov, .Ascending, allocator)
	defer delete(eigenvalues, allocator)
	defer l.matrix_free(&eigenvectors)

	// 4. Determine n_components
	n_comp := params.n_components
	if n_comp <= 0 {
		total_var := 0.0
		for v in eigenvalues {total_var += v}

		if total_var < 1e-12 {
			n_comp = 1 // Fallback if data has zero variance
		} else {
			cum_var := 0.0
			n_comp = 0
			for i in 0 ..< len(eigenvalues) {
				cum_var += eigenvalues[len(eigenvalues) - 1 - i] // Iterate descending
				n_comp += 1
				if cum_var / total_var >= params.variance_threshold {break}
			}
		}
	}
	n_comp = math.min(n_comp, n_features)
	n_comp = math.max(n_comp, 1)

	// 5. Extract top K eigenvectors as rows of `components`
	// eigenvectors matrix is column-major for eigenvectors.
	// Largest eigenvalue is at index n_features - 1.
	components := l.matrix_new(f64, n_comp, n_features, allocator)
	for c in 0 ..< n_comp {
		src_col := n_features - 1 - c
		for r in 0 ..< n_features {
			components.data[c * n_features + r] = eigenvectors.data[r * n_features + src_col]
		}
	}

	// 6. Project X_centered to Z (scores)
	Z := l.matrix_new(f64, n_samples, n_comp, context.temp_allocator)
	defer l.matrix_free(&Z)

	for i in 0 ..< n_samples {
		for c in 0 ..< n_comp {
			sum := 0.0
			for j in 0 ..< n_features {
				sum += X_centered.data[i * n_features + j] * components.data[c * n_features + j]
			}
			Z.data[i * n_comp + c] = sum
		}
	}

	// 7. Fit OLS on Z and y
	// Because Z columns are orthogonal, beta_c = dot(Z_c, y_centered) / dot(Z_c, Z_c)
	y_mean := 0.0
	for val in y {y_mean += val}
	y_mean /= f64(n_samples)

	betas := make([]f64, n_comp, allocator)
	for c in 0 ..< n_comp {
		zty := 0.0
		ztz := 0.0
		for i in 0 ..< n_samples {
			z_val := Z.data[i * n_comp + c]
			y_val := y[i] - y_mean
			zty += z_val * y_val
			ztz += z_val * z_val
		}

		if ztz > 1e-12 {
			betas[c] = zty / ztz
		} else {
			betas[c] = 0.0 // Prevent division by zero for null components
		}
	}

	return PCRegression {
		components = components,
		means = means,
		betas = betas,
		bias = y_mean,
		n_components = n_comp,
		allocator = allocator,
	}
}

// ============================================================================
// Public API: Predict with PCR
// Fused loop: Centers data on-the-fly, uses SIMD dot product for projection.
// ============================================================================

pcr_predict :: proc(
	model: ^PCRegression,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_samples := X.rows
	n_features := X.cols
	n_comp := model.n_components

	preds := make([]f64, n_samples, allocator)
	if n_samples == 0 {return preds}

	// Single temporary buffer for centering the row (Zero allocation in hot loop)
	x_centered := make([]f64, n_features, context.temp_allocator)
	defer delete(x_centered, context.temp_allocator)

	for i in 0 ..< n_samples {
		x_row := X.data[i * n_features:i * n_features + n_features]

		// Center the row
		for j in 0 ..< n_features {
			x_centered[j] = x_row[j] - model.means[j]
		}

		sum := model.bias
		// Project and predict
		for c in 0 ..< n_comp {
			comp_row := model.components.data[c * n_features:c * n_features + n_features]
			// ✅ SIMD-accelerated projection
			z_ic := l.dot_simd(x_centered, comp_row)
			sum += model.betas[c] * z_ic
		}
		preds[i] = sum
	}

	return preds
}

// ============================================================================
// Public API: Free Resources
// ============================================================================

pcr_free :: proc(model: ^PCRegression) {
	if model.components.data != nil {
		l.matrix_free(&model.components)
	}
	if len(model.means) > 0 {
		delete(model.means, model.allocator)
	}
	if len(model.betas) > 0 {
		delete(model.betas, model.allocator)
	}
}
