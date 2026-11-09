package ML

import optim "../../optimize"

// ============================================================================
// Pipeline hyper-parameters in model files (format version 6)
// ============================================================================
// Since version 6 a pipeline file ends with the parameters of its model type
// (after the fitted model, or right after the header of an unfitted pipeline),
// so that a loaded pipeline can be refitted. Files of older versions do not have
// them: the loaded parameters stay zero, as before. Only the block of the
// pipeline's own model type is written.

_write_logistic_params :: proc(w: ^ModelWriter, p: LogisticParams) {
	write_f64(w, p.C)
	write_i64(w, i64(p.max_iter))
	write_f64(w, p.tol)
	write_f64(w, p.learning_rate)
	write_bool(w, p.fit_intercept)
	write_u32(w, u32(p.optimizer_type))
	write_u32(w, u32(p.class_weight))
}

_read_logistic_params :: proc(r: ^ModelReader) -> (p: LogisticParams) {
	p.C = read_f64(r)
	p.max_iter = int(read_i64(r))
	p.tol = read_f64(r)
	p.learning_rate = read_f64(r)
	p.fit_intercept = read_bool(r)
	p.optimizer_type = _read_enum(r, optim.OptimizerType)
	p.class_weight = _read_enum(r, ClassWeight)
	return
}

_write_svm_params :: proc(w: ^ModelWriter, p: SVMParams) {
	write_f64(w, p.C)
	write_i64(w, i64(p.max_iter))
	write_f64(w, p.tol)
	write_f64(w, p.learning_rate)
	write_bool(w, p.fit_intercept)
	write_u32(w, u32(p.optimizer_type))
	write_u32(w, u32(p.class_weight))
}

_read_svm_params :: proc(r: ^ModelReader) -> (p: SVMParams) {
	p.C = read_f64(r)
	p.max_iter = int(read_i64(r))
	p.tol = read_f64(r)
	p.learning_rate = read_f64(r)
	p.fit_intercept = read_bool(r)
	p.optimizer_type = _read_enum(r, optim.OptimizerType)
	p.class_weight = _read_enum(r, ClassWeight)
	return
}

_write_kernel_svm_params :: proc(w: ^ModelWriter, p: KernelSVMParams) {
	write_f64(w, p.C)
	write_u32(w, u32(p.kernel_type))
	write_f64(w, p.gamma)
	write_i64(w, i64(p.degree))
	write_f64(w, p.coef0)
	write_i64(w, i64(p.max_iter))
	write_f64(w, p.tol)
	write_f64(w, p.learning_rate)
	write_u32(w, u32(p.optimizer_type))
	write_u32(w, u32(p.class_weight))
}

_read_kernel_svm_params :: proc(r: ^ModelReader) -> (p: KernelSVMParams) {
	p.C = read_f64(r)
	p.kernel_type = _read_enum(r, SVMKernelType)
	p.gamma = read_f64(r)
	p.degree = int(read_i64(r))
	p.coef0 = read_f64(r)
	p.max_iter = int(read_i64(r))
	p.tol = read_f64(r)
	p.learning_rate = read_f64(r)
	p.optimizer_type = _read_enum(r, optim.OptimizerType)
	p.class_weight = _read_enum(r, ClassWeight)
	return
}

_write_model_params :: proc(w: ^ModelWriter, m: ^PipelineModel) {
	switch m.type {
	case .None:
	case .OLS:
		write_u32(w, u32(m.ols_method))
	case .Ridge:
		write_f64(w, m.ridge_params.lambda)
		write_u32(w, u32(m.ridge_params.method))
	case .Lasso:
		write_f64(w, m.lasso_params.lambda)
		write_i64(w, i64(m.lasso_params.max_iter))
		write_f64(w, m.lasso_params.tol)
	case .ElasticNet:
		p := m.elastic_net_params
		write_f64(w, p.lambda)
		write_f64(w, p.l1_ratio)
		write_i64(w, i64(p.max_iter))
		write_f64(w, p.tol)
	case .SVR:
		p := m.svr_params
		write_f64(w, p.C)
		write_f64(w, p.epsilon)
		write_u32(w, u32(p.kernel_type))
		write_f64(w, p.gamma)
		write_i64(w, i64(p.degree))
		write_f64(w, p.coef0)
		write_i64(w, i64(p.max_iter))
		write_f64(w, p.tol)
		write_f64(w, p.learning_rate)
		write_u32(w, u32(p.optimizer_type))
	case .MLP:
		p := m.mlp_params
		write_slice_int(w, p.hidden_layers)
		write_u32(w, u32(p.activation))
		write_u32(w, u32(p.output_activation))
		write_u32(w, u32(p.task))
		write_f64(w, p.learning_rate)
		write_i64(w, i64(p.max_iter))
		write_i64(w, i64(p.batch_size))
		write_u32(w, u32(p.optimizer_type))
		write_bool(w, p.verbose)
		write_bool(w, p.early_stopping)
		write_f64(w, p.validation_fraction)
		write_i64(w, i64(p.n_iter_no_change))
		write_f64(w, p.tol)
		write_i64(w, i64(p.random_state))
	case .Logistic:
		_write_logistic_params(w, m.logistic_params)
	case .LinearSVM:
		_write_svm_params(w, m.lsvm_params)
	case .KernelSVM:
		_write_kernel_svm_params(w, m.ksvm_params)
	case .KNN:
		write_i64(w, i64(m.knn_params.k))
		write_u32(w, u32(m.knn_params.weights))
	case .DecisionTree:
		p := m.dt_params
		write_i64(w, i64(p.max_depth))
		write_i64(w, i64(p.min_samples))
		write_i64(w, i64(p.max_features))
		write_u32(w, u32(p.criterion))
		write_u32(w, u32(p.class_weight))
	case .GradientBoosting:
		p := m.gb_params
		write_i64(w, i64(p.n_estimators))
		write_f64(w, p.learning_rate)
		write_i64(w, i64(p.max_depth))
		write_i64(w, i64(p.min_samples))
		write_f64(w, p.subsample)
		write_u32(w, u32(p.loss))
		write_i64(w, i64(p.n_iter_no_change))
		write_f64(w, p.validation_fraction)
		write_f64(w, p.tol)
		write_i64(w, i64(p.random_state))
	case .GaussianNB:
		write_f64(w, m.gnb_epsilon)
	case .OvR_Logistic:
		_write_logistic_params(w, m.ovr_logistic_params)
	case .OvR_LinearSVM:
		_write_svm_params(w, m.ovr_lsvm_params)
	case .OvR_KernelSVM:
		_write_kernel_svm_params(w, m.ovr_ksvm_params)
	case .RandomForest:
		p := m.rf_params
		write_i64(w, i64(p.n_trees))
		write_i64(w, i64(p.max_features))
		write_i64(w, i64(p.min_samples))
		write_i64(w, i64(p.max_depth))
		write_bool(w, p.bootstrap)
		write_bool(w, p.oob_score)
		write_u32(w, u32(p.criterion))
		write_u32(w, u32(p.class_weight))
	}
}

// The MLP's hidden_layers are allocated with the reader's allocator and owned by
// the pipeline (pipe.loaded_hidden_layers, freed by pipeline_free)
_read_model_params :: proc(r: ^ModelReader, pipe: ^Pipeline) {
	m := &pipe.model
	switch m.type {
	case .None:
	case .OLS:
		m.ols_method = _read_enum(r, OLSMethod)
	case .Ridge:
		m.ridge_params.lambda = read_f64(r)
		m.ridge_params.method = _read_enum(r, OLSMethod)
	case .Lasso:
		m.lasso_params.lambda = read_f64(r)
		m.lasso_params.max_iter = int(read_i64(r))
		m.lasso_params.tol = read_f64(r)
	case .ElasticNet:
		p := &m.elastic_net_params
		p.lambda = read_f64(r)
		p.l1_ratio = read_f64(r)
		p.max_iter = int(read_i64(r))
		p.tol = read_f64(r)
	case .SVR:
		p := &m.svr_params
		p.C = read_f64(r)
		p.epsilon = read_f64(r)
		p.kernel_type = _read_enum(r, SVMKernelType)
		p.gamma = read_f64(r)
		p.degree = int(read_i64(r))
		p.coef0 = read_f64(r)
		p.max_iter = int(read_i64(r))
		p.tol = read_f64(r)
		p.learning_rate = read_f64(r)
		p.optimizer_type = _read_enum(r, optim.OptimizerType)
	case .MLP:
		p := &m.mlp_params
		pipe.loaded_hidden_layers = read_slice_int(r)
		p.hidden_layers = pipe.loaded_hidden_layers
		p.activation = _read_enum(r, Activation)
		p.output_activation = _read_enum(r, Activation)
		p.task = _read_enum(r, MLPTask)
		p.learning_rate = read_f64(r)
		p.max_iter = int(read_i64(r))
		p.batch_size = int(read_i64(r))
		p.optimizer_type = _read_enum(r, optim.OptimizerType)
		p.verbose = read_bool(r)
		p.early_stopping = read_bool(r)
		p.validation_fraction = read_f64(r)
		p.n_iter_no_change = int(read_i64(r))
		p.tol = read_f64(r)
		p.random_state = int(read_i64(r))
	case .Logistic:
		m.logistic_params = _read_logistic_params(r)
	case .LinearSVM:
		m.lsvm_params = _read_svm_params(r)
	case .KernelSVM:
		m.ksvm_params = _read_kernel_svm_params(r)
	case .KNN:
		m.knn_params.k = int(read_i64(r))
		m.knn_params.weights = _read_enum(r, KNNWeights)
	case .DecisionTree:
		p := &m.dt_params
		p.max_depth = int(read_i64(r))
		p.min_samples = int(read_i64(r))
		p.max_features = int(read_i64(r))
		p.criterion = _read_enum(r, TreeCriterion)
		p.class_weight = _read_enum(r, ClassWeight)
	case .GradientBoosting:
		p := &m.gb_params
		p.n_estimators = int(read_i64(r))
		p.learning_rate = read_f64(r)
		p.max_depth = int(read_i64(r))
		p.min_samples = int(read_i64(r))
		p.subsample = read_f64(r)
		p.loss = _read_enum(r, GBLoss)
		p.n_iter_no_change = int(read_i64(r))
		p.validation_fraction = read_f64(r)
		p.tol = read_f64(r)
		p.random_state = int(read_i64(r))
	case .GaussianNB:
		m.gnb_epsilon = read_f64(r)
	case .OvR_Logistic:
		m.ovr_logistic_params = _read_logistic_params(r)
	case .OvR_LinearSVM:
		m.ovr_lsvm_params = _read_svm_params(r)
	case .OvR_KernelSVM:
		m.ovr_ksvm_params = _read_kernel_svm_params(r)
	case .RandomForest:
		p := &m.rf_params
		p.n_trees = int(read_i64(r))
		p.max_features = int(read_i64(r))
		p.min_samples = int(read_i64(r))
		p.max_depth = int(read_i64(r))
		p.bootstrap = read_bool(r)
		p.oob_score = read_bool(r)
		p.criterion = _read_enum(r, TreeCriterion)
		p.class_weight = _read_enum(r, ClassWeight)
	}
}
