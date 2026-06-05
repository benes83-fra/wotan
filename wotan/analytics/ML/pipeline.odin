package ML

import l "../../linalg"
import "core:mem"

// ============================================================================
// Pipeline Structures
// ============================================================================

PipelineStepType :: enum {
	StandardScaler,
	MinMaxScaler,
}

PipelineStep :: struct {
	type:            PipelineStepType,
	standard_scaler: StandardScaler,
	minmax_scaler:   MinMaxScaler,
}

PipelineModelType :: enum {
	None,
	Logistic,
	LinearSVM,
	KernelSVM,
	KNN,
	Ridge,
	Lasso,
	SVR,
}

PipelineModel :: struct {
	type:            PipelineModelType,

	// Parameters for fitting
	logistic_params: LogisticParams,
	lsvm_params:     SVMParams,
	ksvm_params:     KernelSVMParams,
	knn_params:      KNNParams,
	ridge_params:    RidgeParams,
	lasso_params:    LassoParams,
	svr_params:      SVRParams,

	// Fitted models
	logistic:        LogisticRegression,
	linear_svm:      LinearSVM,
	kernel_svm:      KernelSVM,
	knn:             KNN,
	ridge:           OLSResult,
	lasso:           OLSResult,
	svr:             SupportVectorRegression,
}

Pipeline :: struct {
	steps:     [dynamic]PipelineStep,
	model:     PipelineModel,
	allocator: mem.Allocator,
	is_fitted: bool,
}

// ============================================================================
// Public API: Construction & Adding Steps
// ============================================================================

pipeline_new :: proc(allocator: mem.Allocator = context.allocator) -> Pipeline {
	pipe: Pipeline
	pipe.allocator = allocator
	pipe.steps = make([dynamic]PipelineStep, 0, 4, allocator)
	pipe.model.type = .None
	return pipe
}

pipeline_add_standard_scaler :: proc(pipe: ^Pipeline) {
	step := PipelineStep {
		type = .StandardScaler,
	}
	append(&pipe.steps, step)
}

pipeline_add_minmax_scaler :: proc(pipe: ^Pipeline) {
	step := PipelineStep {
		type = .MinMaxScaler,
	}
	append(&pipe.steps, step)
}

// Model setters
pipeline_set_logistic :: proc(pipe: ^Pipeline, params: LogisticParams) {
	pipe.model.type = .Logistic
	pipe.model.logistic_params = params
}

pipeline_set_linear_svm :: proc(pipe: ^Pipeline, params: SVMParams) {
	pipe.model.type = .LinearSVM
	pipe.model.lsvm_params = params
}

pipeline_set_kernel_svm :: proc(pipe: ^Pipeline, params: KernelSVMParams) {
	pipe.model.type = .KernelSVM
	pipe.model.ksvm_params = params
}

pipeline_set_knn :: proc(pipe: ^Pipeline, params: KNNParams) {
	pipe.model.type = .KNN
	pipe.model.knn_params = params
}

pipeline_set_ridge :: proc(pipe: ^Pipeline, params: RidgeParams) {
	pipe.model.type = .Ridge
	pipe.model.ridge_params = params
}

pipeline_set_lasso :: proc(pipe: ^Pipeline, params: LassoParams) {
	pipe.model.type = .Lasso
	pipe.model.lasso_params = params
}

pipeline_set_svr :: proc(pipe: ^Pipeline, params: SVRParams) {
	pipe.model.type = .SVR
	pipe.model.svr_params = params
}

// ============================================================================
// Public API: Fit
// ============================================================================

pipeline_fit :: proc(pipe: ^Pipeline, X: ^l.Matrix(f64), y: []f64) {
	if pipe.model.type == .None {
		panic("pipeline_fit: No model set in pipeline")
	}

	current_X := X^
	owns_X := false

	// 1. Fit and transform through all steps
	for i in 0 ..< len(pipe.steps) {
		step := &pipe.steps[i]
		switch step.type {
		case .StandardScaler:
			step.standard_scaler = standard_scaler_fit(&current_X, pipe.allocator)
			new_X := standard_scaler_transform(&step.standard_scaler, &current_X, pipe.allocator)
			if owns_X {l.matrix_free(&current_X)}
			current_X = new_X
			owns_X = true

		case .MinMaxScaler:
			step.minmax_scaler = minmax_scaler_fit(&current_X, [2]f64{0.0, 1.0}, pipe.allocator)
			new_X := minmax_scaler_transform(&step.minmax_scaler, &current_X, pipe.allocator)
			if owns_X {l.matrix_free(&current_X)}
			current_X = new_X
			owns_X = true
		}
	}

	// 2. Fit the final model
	#partial switch pipe.model.type {
	case .Logistic:
		pipe.model.logistic = logistic_fit(
			&current_X,
			y,
			pipe.model.logistic_params,
			pipe.allocator,
		)
	case .LinearSVM:
		pipe.model.linear_svm = svm_fit_linear(
			&current_X,
			y,
			pipe.model.lsvm_params,
			pipe.allocator,
		)
	case .KernelSVM:
		pipe.model.kernel_svm = kernel_svm_fit(
			&current_X,
			y,
			pipe.model.ksvm_params,
			pipe.allocator,
		)
	case .KNN:
		pipe.model.knn = knn_fit(&current_X, y, pipe.model.knn_params, pipe.allocator)
	case .Ridge:
		pipe.model.ridge = ridge_fit(
			&current_X,
			y,
			pipe.model.ridge_params.lambda,
			pipe.model.ridge_params.method,
			pipe.allocator,
		)
	case .Lasso:
		pipe.model.lasso = lasso_fit(
			&current_X,
			y,
			pipe.model.lasso_params.lambda,
			pipe.model.lasso_params.max_iter,
			pipe.model.lasso_params.tol,
			pipe.allocator,
		)
	case .SVR:
		pipe.model.svr = svr_fit(&current_X, y, pipe.model.svr_params, pipe.allocator)
	}

	if owns_X {l.matrix_free(&current_X)}
	pipe.is_fitted = true
}

// ============================================================================
// Public API: Predict
// ============================================================================

pipeline_predict :: proc(
	pipe: ^Pipeline,
	X: ^l.Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	if !pipe.is_fitted {
		panic("pipeline_predict: Pipeline not fitted")
	}

	current_X := X^
	owns_X := false

	// 1. Transform through all steps
	for i in 0 ..< len(pipe.steps) {
		step := &pipe.steps[i]
		switch step.type {
		case .StandardScaler:
			new_X := standard_scaler_transform(&step.standard_scaler, &current_X, pipe.allocator)
			if owns_X {l.matrix_free(&current_X)}
			current_X = new_X
			owns_X = true

		case .MinMaxScaler:
			new_X := minmax_scaler_transform(&step.minmax_scaler, &current_X, pipe.allocator)
			if owns_X {l.matrix_free(&current_X)}
			current_X = new_X
			owns_X = true
		}
	}

	// 2. Predict with the final model
	preds: []f64
	#partial switch pipe.model.type {
	case .Logistic:
		preds = logistic_predict(&pipe.model.logistic, &current_X, allocator)
	case .LinearSVM:
		raw_preds := svm_predict_linear(&pipe.model.linear_svm, &current_X, pipe.allocator)
		preds = make([]f64, len(raw_preds), allocator)
		for val, i in raw_preds {
			tmp: f64
			if val >= 0.0 {
				tmp = 1.0
			} else {
				tmp = -1.0
			}

			preds[i] = tmp
		}
		delete(raw_preds, pipe.allocator)
	case .KernelSVM:
		raw_preds := kernel_svm_predict(&pipe.model.kernel_svm, &current_X, pipe.allocator)
		preds = make([]f64, len(raw_preds), allocator)
		for val, i in raw_preds {
			tmp: f64
			if val >= 0.0 {
				tmp = 1.0
			} else {
				tmp = -1.0
			}

			preds[i] = tmp
		}
		delete(raw_preds, pipe.allocator)
	case .KNN:
		preds = knn_predict(&pipe.model.knn, &current_X, allocator)
	case .Ridge:
		preds = l.matvec_dyn_simd(&current_X, pipe.model.ridge.beta, allocator)
	case .Lasso:
		preds = l.matvec_dyn_simd(&current_X, pipe.model.lasso.beta, allocator)
	case .SVR:
		preds = svr_predict(&pipe.model.svr, &current_X, allocator)
	}

	if owns_X {l.matrix_free(&current_X)}
	return preds
}

// ============================================================================
// Public API: Free
// ============================================================================

pipeline_free :: proc(pipe: ^Pipeline) {
	// Free steps
	for i in 0 ..< len(pipe.steps) {
		step := &pipe.steps[i]
		switch step.type {
		case .StandardScaler:
			standard_scaler_free(&step.standard_scaler)
		case .MinMaxScaler:
			minmax_scaler_free(&step.minmax_scaler)
		}
	}
	delete(pipe.steps)

	// Free model
	#partial switch pipe.model.type {
	case .Logistic:
		logistic_free(&pipe.model.logistic)
	case .LinearSVM:
		svm_free(&pipe.model.linear_svm)
	case .KernelSVM:
		kernel_svm_free(&pipe.model.kernel_svm)
	case .KNN:
		knn_free(&pipe.model.knn)
	case .Ridge:
		_ols_result_free(&pipe.model.ridge, pipe.allocator)
	case .Lasso:
		_ols_result_free(&pipe.model.lasso, pipe.allocator)
	case .SVR:
		svr_free(&pipe.model.svr)
	}
}
