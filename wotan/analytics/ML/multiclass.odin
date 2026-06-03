package ML

import l "../../linalg"
import "core:mem"

// ============================================================================
// Binary Classifier Tagged Union
// ============================================================================

BinaryClassifierType :: enum {
	Logistic,
	LinearSVM,
	KernelSVM,
}

BinaryClassifier :: struct {
	type:       BinaryClassifierType,
	logistic:   LogisticRegression,
	linear_svm: LinearSVM,
	kernel_svm: KernelSVM,
	allocator:  mem.Allocator,
}

// ============================================================================
// One-vs-Rest (OvR) Classifier
// ============================================================================

OvRClassifier :: struct {
	classes:   []f64, // Unique class labels (e.g., 0.0, 1.0, 2.0)
	models:    []BinaryClassifier, // One binary model per class
	allocator: mem.Allocator,
}

// Function pointer type for fitting
_BinaryFitFunc :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	target_class: f64,
	params: rawptr,
	allocator: mem.Allocator,
) -> BinaryClassifier

// --- Logistic Wrappers ---
_logistic_fit_wrapper :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	target_class: f64,
	params: rawptr,
	allocator: mem.Allocator,
) -> BinaryClassifier {
	p := cast(^LogisticParams)(params)
	n := X.rows
	y_binary := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {
		tmp: f64
		if y[i] == target_class {
			tmp = 1.0
		} else {
			tmp = 0.0 // Logistic requires 0/1
		}
		y_binary[i] = tmp
	}
	model := logistic_fit(X, y_binary, p^, allocator)
	delete(y_binary, context.temp_allocator)
	return BinaryClassifier{type = .Logistic, logistic = model, allocator = allocator}
}

// --- Linear SVM Wrappers ---
_linear_svm_fit_wrapper :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	target_class: f64,
	params: rawptr,
	allocator: mem.Allocator,
) -> BinaryClassifier {
	p := cast(^SVMParams)(params)
	n := X.rows
	y_binary := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {
		tmp: f64
		if y[i] == target_class {
			tmp = 1.0
		} else {
			tmp = -1.0 // ✅ FIX: SVM requires -1/1
		}
		y_binary[i] = tmp
	}
	model := svm_fit_linear(X, y_binary, p^, allocator)
	delete(y_binary, context.temp_allocator)
	return BinaryClassifier{type = .LinearSVM, linear_svm = model, allocator = allocator}
}

// --- Kernel SVM Wrappers ---
_kernel_svm_fit_wrapper :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	target_class: f64,
	params: rawptr,
	allocator: mem.Allocator,
) -> BinaryClassifier {
	p := cast(^KernelSVMParams)(params)
	n := X.rows
	y_binary := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {
		tmp: f64
		if y[i] == target_class {
			tmp = 1.0
		} else {
			tmp = -1.0 // ✅ FIX: SVM requires -1/1
		}
		y_binary[i] = tmp
	}
	model := kernel_svm_fit(X, y_binary, p^, allocator)
	delete(y_binary, context.temp_allocator)
	return BinaryClassifier{type = .KernelSVM, kernel_svm = model, allocator = allocator}
}

// ============================================================================
// Internal: Core OvR Logic
// ============================================================================

_ovr_fit_internal :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: rawptr,
	fit_fn: _BinaryFitFunc,
	allocator: mem.Allocator,
) -> OvRClassifier {
	// 1. Find unique classes
	unique_classes := make([dynamic]f64, 0, allocator)
	for val in y {
		found := false
		for u in unique_classes {
			if u == val {
				found = true
				break
			}
		}
		if !found {
			append(&unique_classes, val)
		}
	}

	n_classes := len(unique_classes)

	// 2. Train one model per class
	models := make([]BinaryClassifier, n_classes, allocator)

	for c_idx in 0 ..< n_classes {
		target_class := unique_classes[c_idx]
		models[c_idx] = fit_fn(X, y, target_class, params, allocator)
	}

	// 3. Create final struct
	classes_final := make([]f64, n_classes, allocator)
	copy(classes_final, unique_classes[:])
	delete(unique_classes) // Free the dynamic array buffer

	return OvRClassifier{classes = classes_final, models = models, allocator = allocator}
}

// ============================================================================
// Public API: Fit One-vs-Rest Models
// ============================================================================

ovr_logistic_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: LogisticParams,
	allocator: mem.Allocator = context.allocator,
) -> OvRClassifier {
	local_params := params // ✅ FIX: Create a local copy to make it addressable
	return _ovr_fit_internal(X, y, &local_params, _logistic_fit_wrapper, allocator)
}

ovr_linear_svm_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: SVMParams,
	allocator: mem.Allocator = context.allocator,
) -> OvRClassifier {
	local_params := params // ✅ FIX: Create a local copy to make it addressable
	return _ovr_fit_internal(X, y, &local_params, _linear_svm_fit_wrapper, allocator)
}

ovr_kernel_svm_fit :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	params: KernelSVMParams,
	allocator: mem.Allocator = context.allocator,
) -> OvRClassifier {
	local_params := params // ✅ FIX: Create a local copy to make it addressable
	return _ovr_fit_internal(X, y, &local_params, _kernel_svm_fit_wrapper, allocator)
}

// ============================================================================
// Public API: Predict
// ============================================================================

ovr_predict :: proc(
	model: ^OvRClassifier,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	n_samples := X.rows
	n_classes := len(model.classes)

	if n_classes == 0 {
		return make([]f64, 0, allocator)
	}

	preds := make([]f64, n_samples, allocator)

	// Allocate matrix to hold all scores: [n_samples, n_classes]
	scores := make([]f64, n_samples * n_classes, allocator)
	defer delete(scores, allocator)

	// Get scores from each binary model
	for c_idx in 0 ..< n_classes {
		bin_model := &model.models[c_idx]

		bin_preds: []f64 = nil // Initialize to avoid uninitialized variable error
		switch bin_model.type {
		case .Logistic:
			bin_preds = logistic_predict_proba(&bin_model.logistic, X, allocator)
		case .LinearSVM:
			bin_preds = svm_predict_linear(&bin_model.linear_svm, X, allocator)
		case .KernelSVM:
			bin_preds = kernel_svm_predict(&bin_model.kernel_svm, X, allocator)
		}

		// Copy scores into the matrix
		for i in 0 ..< n_samples {
			scores[i * n_classes + c_idx] = bin_preds[i]
		}

		delete(bin_preds, allocator)
	}

	// Argmax over the classes for each sample
	for i in 0 ..< n_samples {
		best_score := scores[i * n_classes + 0]
		best_class := model.classes[0]

		for c_idx in 1 ..< n_classes {
			score := scores[i * n_classes + c_idx]
			if score > best_score {
				best_score = score
				best_class = model.classes[c_idx]
			}
		}
		preds[i] = best_class
	}

	return preds
}

// ============================================================================
// Public API: Free Resources
// ============================================================================

ovr_free :: proc(model: ^OvRClassifier) {
	if len(model.models) > 0 {
		for i in 0 ..< len(model.models) {
			bin_model := &model.models[i]
			switch bin_model.type {
			case .Logistic:
				logistic_free(&bin_model.logistic)
			case .LinearSVM:
				svm_free(&bin_model.linear_svm)
			case .KernelSVM:
				kernel_svm_free(&bin_model.kernel_svm)
			}
		}
		delete(model.models, model.allocator)
	}
	if len(model.classes) > 0 {
		delete(model.classes, model.allocator)
	}
}
