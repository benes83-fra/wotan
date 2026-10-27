package ML

import l "../../linalg"
import "core:mem"

// ============================================================================
// Pipeline Structures
// ============================================================================

PipelineStepType :: enum {
	StandardScaler,
	MinMaxScaler,
	RobustScaler, // appended: saved files keep their step codes
}

PipelineStep :: struct {
	type:            PipelineStepType,
	standard_scaler: StandardScaler,
	minmax_scaler:   MinMaxScaler,
	robust_scaler:   RobustScaler,
}

PipelineModelType :: enum {
	None,
	// Regression
	OLS,
	Ridge,
	Lasso,
	SVR,
	MLP,
	// Binary Classification
	Logistic,
	LinearSVM,
	KernelSVM,
	KNN,
	DecisionTree,
	GradientBoosting,
	GaussianNB,
	// Multi-class Classification (One-vs-Rest)
	OvR_Logistic,
	OvR_LinearSVM,
	OvR_KernelSVM,
	// Ensembles
	RandomForest,
	// appended (saved files keep their model codes)
	ElasticNet,
}

PipelineModel :: struct {
	type:                PipelineModelType,
	mlp_params:          MLPParams,
	mlp:                 MLP,

	// Parameters for fitting
	logistic_params:     LogisticParams,
	lsvm_params:         SVMParams,
	ksvm_params:         KernelSVMParams,
	knn_params:          KNNParams,
	ridge_params:        RidgeParams,
	lasso_params:        LassoParams,
	elastic_net_params:  ElasticNetParams,
	svr_params:          SVRParams,
	ols_method:          OLSMethod,
	dt_params:           TreeParams,
	rf_params:           RFParams,
	gb_params:           GBParams, // ⚠️ Change to GradientBoostingParams if your struct is named differently
	gnb_epsilon:         f64,
	ovr_logistic_params: LogisticParams,
	ovr_lsvm_params:     SVMParams,
	ovr_ksvm_params:     KernelSVMParams,

	// Fitted models
	logistic:            LogisticRegression,
	linear_svm:          LinearSVM,
	kernel_svm:          KernelSVM,
	knn:                 KNN,
	ridge:               OLSResult,
	lasso:               OLSResult,
	elastic_net:         OLSResult,
	svr:                 SupportVectorRegression,
	ols:                 OLSResult,
	dt:                  DecisionTree,
	rf:                  RandomForest,
	gb:                  GradientBoosting, // ⚠️ Change to GBModel if your struct is named differently
	gnb:                 GaussianNB,
	ovr_logistic:        OvRClassifier,
	ovr_linear_svm:      OvRClassifier,
	ovr_kernel_svm:      OvRClassifier,
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

// RobustScaler with the default quantile range (25, 75)
pipeline_add_robust_scaler :: proc(pipe: ^Pipeline) {
	append(&pipe.steps, PipelineStep{type = .RobustScaler})
}

pipeline_add_minmax_scaler :: proc(pipe: ^Pipeline) {
	step := PipelineStep {
		type = .MinMaxScaler,
	}
	append(&pipe.steps, step)
}

// Model setters
pipeline_set_logistic :: proc(pipe: ^Pipeline, params: LogisticParams) {
	pipe.model.type = .Logistic; pipe.model.logistic_params = params
}
pipeline_set_linear_svm :: proc(pipe: ^Pipeline, params: SVMParams) {
	pipe.model.type = .LinearSVM; pipe.model.lsvm_params = params
}
pipeline_set_kernel_svm :: proc(pipe: ^Pipeline, params: KernelSVMParams) {
	pipe.model.type = .KernelSVM; pipe.model.ksvm_params = params
}
pipeline_set_knn :: proc(pipe: ^Pipeline, params: KNNParams) {
	pipe.model.type = .KNN; pipe.model.knn_params = params
}
pipeline_set_ridge :: proc(pipe: ^Pipeline, params: RidgeParams) {
	pipe.model.type = .Ridge; pipe.model.ridge_params = params
}
pipeline_set_lasso :: proc(pipe: ^Pipeline, params: LassoParams) {
	pipe.model.type = .Lasso; pipe.model.lasso_params = params
}
pipeline_set_elastic_net :: proc(pipe: ^Pipeline, params: ElasticNetParams) {
	pipe.model.type = .ElasticNet; pipe.model.elastic_net_params = params
}
pipeline_set_svr :: proc(pipe: ^Pipeline, params: SVRParams) {
	pipe.model.type = .SVR; pipe.model.svr_params = params
}
pipeline_set_ols :: proc(pipe: ^Pipeline, method: OLSMethod = .Cholesky) {
	pipe.model.type = .OLS; pipe.model.ols_method = method
}
pipeline_set_decision_tree :: proc(pipe: ^Pipeline, params: TreeParams) {
	pipe.model.type = .DecisionTree; pipe.model.dt_params = params
}
pipeline_set_random_forest :: proc(pipe: ^Pipeline, params: RFParams) {
	pipe.model.type = .RandomForest; pipe.model.rf_params = params
}
pipeline_set_gradient_boosting :: proc(pipe: ^Pipeline, params: GBParams) {
	pipe.model.type = .GradientBoosting; pipe.model.gb_params = params
}
pipeline_set_gnb :: proc(pipe: ^Pipeline, epsilon: f64 = 1e-9) {
	pipe.model.type = .GaussianNB; pipe.model.gnb_epsilon = epsilon
}
pipeline_set_ovr_logistic :: proc(pipe: ^Pipeline, params: LogisticParams) {
	pipe.model.type = .OvR_Logistic; pipe.model.ovr_logistic_params = params
}
pipeline_set_ovr_linear_svm :: proc(pipe: ^Pipeline, params: SVMParams) {
	pipe.model.type = .OvR_LinearSVM; pipe.model.ovr_lsvm_params = params
}
pipeline_set_ovr_kernel_svm :: proc(pipe: ^Pipeline, params: KernelSVMParams) {
	pipe.model.type = .OvR_KernelSVM; pipe.model.ovr_ksvm_params = params
}
pipeline_set_mlp :: proc(pipe: ^Pipeline, params: MLPParams) {
	pipe.model.type = .MLP; pipe.model.mlp_params = params
}
// ============================================================================
// Public API: Fit
// ============================================================================

// sample_weight (optional) goes to the final model, as sklearn's
// pipeline.fit(X, y, model__sample_weight=w); the scalers are unweighted.
// Supported by Logistic, LinearSVM, KernelSVM, SVR, DecisionTree, RandomForest
// and GradientBoosting; other models panic when weights are given.
pipeline_fit :: proc(pipe: ^Pipeline, X: ^l.Matrix(f64), y: []f64, sample_weight: []f64 = nil) {
	if pipe.model.type == .None {
		panic("pipeline_fit: No model set in pipeline")
	}
	if len(y) != X.rows do panic("pipeline_fit: y length mismatch")
	#partial switch pipe.model.type {
	case .Logistic, .LinearSVM, .KernelSVM, .SVR, .DecisionTree, .RandomForest, .GradientBoosting:
	case:
		if sample_weight != nil do panic("pipeline_fit: this model does not support sample_weight")
	}
	// Refit: release the previously fitted scalers and model first
	if pipe.is_fitted {
		_pipeline_free_fitted(pipe)
		pipe.is_fitted = false
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

		case .RobustScaler:
			step.robust_scaler = robust_scaler_fit(&current_X, allocator = pipe.allocator)
			new_X := robust_scaler_transform(&step.robust_scaler, &current_X, pipe.allocator)
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
			sample_weight,
		)
	case .LinearSVM:
		pipe.model.linear_svm = svm_fit_linear(
			&current_X,
			y,
			pipe.model.lsvm_params,
			pipe.allocator,
			sample_weight,
		)
	case .KernelSVM:
		pipe.model.kernel_svm = kernel_svm_fit(
			&current_X,
			y,
			pipe.model.ksvm_params,
			pipe.allocator,
			sample_weight,
		)
	case .KNN:
		pipe.model.knn = knn_fit(&current_X, y, pipe.model.knn_params, pipe.allocator)
	case .Ridge:
		// Linear models get an (unpenalised) intercept column, like sklearn's
		// fit_intercept=True: β[0] is the intercept
		Xi := _with_intercept(&current_X, pipe.allocator)
		defer l.matrix_free(&Xi)
		pipe.model.ridge = ridge_fit(
			&Xi,
			y,
			pipe.model.ridge_params.lambda,
			pipe.model.ridge_params.method,
			pipe.allocator,
		)
	case .Lasso:
		Xi := _with_intercept(&current_X, pipe.allocator)
		defer l.matrix_free(&Xi)
		pipe.model.lasso = lasso_fit(
			&Xi,
			y,
			pipe.model.lasso_params.lambda,
			pipe.model.lasso_params.max_iter,
			pipe.model.lasso_params.tol,
			pipe.allocator,
		)
	case .ElasticNet:
		Xi := _with_intercept(&current_X, pipe.allocator)
		defer l.matrix_free(&Xi)
		ep := pipe.model.elastic_net_params
		pipe.model.elastic_net = elastic_net_fit(
			&Xi,
			y,
			ep.lambda,
			ep.l1_ratio,
			ep.max_iter if ep.max_iter > 0 else 1000,
			ep.tol if ep.tol > 0 else 1e-4,
			pipe.allocator,
		)
	case .SVR:
		pipe.model.svr = svr_fit(&current_X, y, pipe.model.svr_params, pipe.allocator, sample_weight)
	case .OLS:
		Xi := _with_intercept(&current_X, pipe.allocator)
		defer l.matrix_free(&Xi)
		pipe.model.ols = ols_fit(&Xi, y, pipe.model.ols_method, pipe.allocator)
	case .DecisionTree:
		pipe.model.dt = dt_fit(&current_X, y, pipe.model.dt_params, pipe.allocator, sample_weight)
	case .RandomForest:
		pipe.model.rf = rf_fit(&current_X, y, pipe.model.rf_params, pipe.allocator, sample_weight)
	case .GradientBoosting:
		pipe.model.gb = gb_fit(&current_X, y, pipe.model.gb_params, pipe.allocator, sample_weight)
	case .GaussianNB:
		pipe.model.gnb = gnb_fit(&current_X, y, pipe.model.gnb_epsilon, pipe.allocator)
	case .OvR_Logistic:
		pipe.model.ovr_logistic = ovr_logistic_fit(
			&current_X,
			y,
			pipe.model.ovr_logistic_params,
			pipe.allocator,
		)
	case .OvR_LinearSVM:
		pipe.model.ovr_linear_svm = ovr_linear_svm_fit(
			&current_X,
			y,
			pipe.model.ovr_lsvm_params,
			pipe.allocator,
		)
	case .MLP:
		pipe.model.mlp = mlp_fit(&current_X, y, pipe.model.mlp_params, pipe.allocator)
	case .OvR_KernelSVM:
		pipe.model.ovr_kernel_svm = ovr_kernel_svm_fit(
			&current_X,
			y,
			pipe.model.ovr_ksvm_params,
			pipe.allocator,
		)
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

		case .RobustScaler:
			new_X := robust_scaler_transform(&step.robust_scaler, &current_X, pipe.allocator)
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
			preds[i] = tmp}
		delete(raw_preds, pipe.allocator)
	case .KNN:
		preds = knn_predict(&pipe.model.knn, &current_X, allocator)
	case .Ridge:
		preds = _linear_predict(&current_X, pipe.model.ridge.beta, allocator)
	case .Lasso:
		preds = _linear_predict(&current_X, pipe.model.lasso.beta, allocator)
	case .ElasticNet:
		preds = _linear_predict(&current_X, pipe.model.elastic_net.beta, allocator)
	case .SVR:
		preds = svr_predict(&pipe.model.svr, &current_X, allocator)
	case .OLS:
		preds = _linear_predict(&current_X, pipe.model.ols.beta, allocator)
	case .DecisionTree:
		preds = dt_predict(&pipe.model.dt, &current_X, allocator)
	case .RandomForest:
		preds = rf_predict(&pipe.model.rf, &current_X, allocator)
	case .GradientBoosting:
		preds = gb_predict(&pipe.model.gb, &current_X, allocator)
	case .GaussianNB:
		preds = gnb_predict(&pipe.model.gnb, &current_X, allocator)
	case .OvR_Logistic:
		preds = ovr_predict(&pipe.model.ovr_logistic, &current_X, allocator)
	case .OvR_LinearSVM:
		preds = ovr_predict(&pipe.model.ovr_linear_svm, &current_X, allocator)
	case .OvR_KernelSVM:
		preds = ovr_predict(&pipe.model.ovr_kernel_svm, &current_X, allocator)
	case .MLP:
		preds = mlp_predict(&pipe.model.mlp, &current_X, allocator)
	}

	if owns_X {l.matrix_free(&current_X)}
	return preds
}

// ============================================================================
// Public API: Free
// ============================================================================

pipeline_free :: proc(pipe: ^Pipeline) {
	if pipe.is_fitted do _pipeline_free_fitted(pipe)
	delete(pipe.steps)
}

// [1, X]: X with a leading column of ones
_with_intercept :: proc(X: ^l.Matrix(f64), allocator: mem.Allocator) -> l.Matrix(f64) {
	n, p := X.rows, X.cols
	out := l.matrix_new(f64, n, p + 1, allocator)
	for i in 0 ..< n {
		out.data[i * (p + 1)] = 1.0
		copy(out.data[i * (p + 1) + 1:(i + 1) * (p + 1)], X.data[i * p:(i + 1) * p])
	}
	return out
}

// X·β, with β[0] as intercept when β has one entry more than X has columns
// (pipelines saved before the intercept was added have exactly X.cols entries)
_linear_predict :: proc(X: ^l.Matrix(f64), beta: []f64, allocator: mem.Allocator) -> []f64 {
	n, p := X.rows, X.cols
	if len(beta) != p && len(beta) != p + 1 do panic("pipeline_predict: X does not match the fitted linear model")
	off := len(beta) - p
	out := make([]f64, n, allocator)
	for i in 0 ..< n {
		s := beta[0] if off == 1 else 0.0
		for j in 0 ..< p {s += X.data[i * p + j] * beta[j + off]}
		out[i] = s
	}
	return out
}

// Free fitted scaler state and the fitted model (the steps themselves stay)
_pipeline_free_fitted :: proc(pipe: ^Pipeline) {
	// Free steps
	for i in 0 ..< len(pipe.steps) {
		step := &pipe.steps[i]
		switch step.type {
		case .StandardScaler:
			standard_scaler_free(&step.standard_scaler)
		case .MinMaxScaler:
			minmax_scaler_free(&step.minmax_scaler)
		case .RobustScaler:
			robust_scaler_free(&step.robust_scaler)
		}
		step.standard_scaler = {}
		step.minmax_scaler = {}
		step.robust_scaler = {}
	}

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
	case .ElasticNet:
		_ols_result_free(&pipe.model.elastic_net, pipe.allocator)
	case .SVR:
		svr_free(&pipe.model.svr)
	case .OLS:
		_ols_result_free(&pipe.model.ols, pipe.allocator)
	case .DecisionTree:
		dt_free(&pipe.model.dt)
	case .RandomForest:
		rf_free(&pipe.model.rf)
	case .GradientBoosting:
		gb_free(&pipe.model.gb)
	case .GaussianNB:
		gnb_free(&pipe.model.gnb)
	case .OvR_Logistic:
		ovr_free(&pipe.model.ovr_logistic)
	case .OvR_LinearSVM:
		ovr_free(&pipe.model.ovr_linear_svm)
	case .OvR_KernelSVM:
		ovr_free(&pipe.model.ovr_kernel_svm)
	case .MLP:
		mlp_free(&pipe.model.mlp)
	}
}
