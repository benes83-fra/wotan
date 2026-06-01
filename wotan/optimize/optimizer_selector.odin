package optimize

import l "../linalg"
import "core:mem"

// ============================================================================
// Unified Optimizer API
// ============================================================================

OptimizerType :: enum {
	SGD,
	Adam,
	RMSProp,
	LBFGS,
}

OptimizerConfig :: struct {
	type:          OptimizerType,
	learning_rate: f64,
	momentum:      f64,
	beta1:         f64,
	beta2:         f64,
	epsilon:       f64,
	lbfgs_m:       int,
	lbfgs_tol:     f64,
	weight_decay:  f64,
}

optimizer_default_config :: proc(type: OptimizerType) -> OptimizerConfig {
	switch type {
	case .SGD:
		return OptimizerConfig {
			type = .SGD,
			learning_rate = 0.01,
			momentum = 0.0,
			weight_decay = 0.0,
		}
	case .Adam:
		return OptimizerConfig {
			type = .Adam,
			learning_rate = 0.001,
			beta1 = 0.9,
			beta2 = 0.999,
			epsilon = 1e-8,
			weight_decay = 0.0,
		}
	case .RMSProp:
		return OptimizerConfig {
			type = .RMSProp,
			learning_rate = 0.01,
			beta2 = 0.99,
			epsilon = 1e-8,
			weight_decay = 0.0,
		}
	case .LBFGS:
		return OptimizerConfig {
			type = .LBFGS,
			learning_rate = 1.0,
			lbfgs_m = 10,
			lbfgs_tol = 1e-5,
			weight_decay = 0.0,
		}
	}
	return OptimizerConfig{}
}

Optimizer :: struct {
	config:    OptimizerConfig,
	sgd:       OptimizerSGD,
	adam:      OptimizerAdam,
	rmsprop:   OptimizerRMSProp,
	lbfgs:     OptimizerLBFGS,
	allocator: mem.Allocator,
}

optimizer_init :: proc(
	config: OptimizerConfig,
	n_params: int,
	allocator: mem.Allocator = context.allocator,
) -> Optimizer {
	opt: Optimizer
	opt.config = config
	opt.allocator = allocator

	switch config.type {
	case .SGD:
		opt.sgd = optimizer_sgd_init(
			n_params,
			config.learning_rate,
			config.momentum,
			config.weight_decay,
			allocator,
		)
	case .Adam:
		opt.adam = optimizer_adam_init(
			n_params,
			config.learning_rate,
			config.beta1,
			config.beta2,
			config.epsilon,
			config.weight_decay,
			allocator,
		)
	case .RMSProp:
		opt.rmsprop = optimizer_rmsprop_init(
			n_params,
			config.learning_rate,
			config.beta2,
			config.epsilon,
			config.weight_decay,
			allocator,
		)
	case .LBFGS:
		opt.lbfgs = optimizer_lbfgs_init(
			n_params,
			config.learning_rate,
			config.lbfgs_m,
			config.lbfgs_tol,
			allocator,
		)
	}
	return opt
}

optimizer_init_mat :: proc(
	config: OptimizerConfig,
	rows, cols: int,
	allocator: mem.Allocator = context.allocator,
) -> Optimizer {
	opt: Optimizer
	opt.config = config
	opt.allocator = allocator

	switch config.type {
	case .SGD:
		opt.sgd = optimizer_sgd_init_mat(
			rows,
			cols,
			config.learning_rate,
			config.momentum,
			config.weight_decay,
			allocator,
		)
	case .Adam:
		opt.adam = optimizer_adam_init_mat(
			rows,
			cols,
			config.learning_rate,
			config.beta1,
			config.beta2,
			config.epsilon,
			config.weight_decay,
			allocator,
		)
	case .RMSProp:
		opt.rmsprop = optimizer_rmsprop_init_mat(
			rows,
			cols,
			config.learning_rate,
			config.beta2,
			config.epsilon,
			config.weight_decay,
			allocator,
		)
	case .LBFGS:
		opt.lbfgs = optimizer_lbfgs_init(
			rows * cols,
			config.learning_rate,
			config.lbfgs_m,
			config.lbfgs_tol,
			allocator,
		)
	}
	return opt
}

optimizer_free :: proc(opt: ^Optimizer) {
	switch opt.config.type {
	case .SGD:
		optimizer_sgd_free(&opt.sgd)
	case .Adam:
		optimizer_adam_free(&opt.adam)
	case .RMSProp:
		optimizer_rmsprop_free(&opt.rmsprop)
	case .LBFGS:
		optimizer_lbfgs_free(&opt.lbfgs)
	}
}

optimizer_step :: proc(
	opt: ^Optimizer,
	weights: []f64,
	gradient: []f64,
	loss: f64 = 0.0,
	obj_fn: ObjectiveFunc = nil,
	obj_user_data: rawptr = nil,
) {
	switch opt.config.type {
	case .SGD:
		optimizer_sgd_step(&opt.sgd, weights, gradient)
	case .Adam:
		optimizer_adam_step(&opt.adam, weights, gradient)
	case .RMSProp:
		optimizer_rmsprop_step(&opt.rmsprop, weights, gradient)
	case .LBFGS:
		optimizer_lbfgs_step(&opt.lbfgs, weights, gradient, loss, obj_fn, obj_user_data)
	}
}

optimizer_step_mat :: proc(
	opt: ^Optimizer,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
	loss: f64 = 0.0,
	obj_fn: ObjectiveFunc = nil,
	obj_user_data: rawptr = nil,
) {
	switch opt.config.type {
	case .SGD:
		optimizer_sgd_step_mat(&opt.sgd, weights, gradient)
	case .Adam:
		optimizer_adam_step_mat(&opt.adam, weights, gradient)
	case .RMSProp:
		optimizer_rmsprop_step_mat(&opt.rmsprop, weights, gradient)
	case .LBFGS:
		optimizer_lbfgs_step_mat(&opt.lbfgs, weights, gradient, loss, obj_fn, obj_user_data)
	}
}

optimizer_set_learning_rate :: proc(opt: ^Optimizer, lr: f64) {
	opt.config.learning_rate = lr
	switch opt.config.type {
	case .SGD:
		opt.sgd.learning_rate = lr
	case .Adam:
		opt.adam.learning_rate = lr
	case .RMSProp:
		opt.rmsprop.learning_rate = lr
	case .LBFGS:
		opt.lbfgs.learning_rate = lr
	}
}

OptimizerState :: struct {
	type:          OptimizerType,
	learning_rate: f64,
	step_count:    int,
}

optimizer_state :: proc(opt: ^Optimizer) -> OptimizerState {
	state: OptimizerState
	state.type = opt.config.type
	state.learning_rate = opt.config.learning_rate

	switch opt.config.type {
	case .Adam:
		state.step_count = opt.adam.t
	case .RMSProp:
		state.step_count = 0
	case .SGD:
		state.step_count = 0
	case .LBFGS:
		state.step_count = opt.lbfgs.k
	}
	return state
}
