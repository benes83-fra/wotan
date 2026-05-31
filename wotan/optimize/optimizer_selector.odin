package optimize

import l "../linalg"
import "core:mem"

// ============================================================================
// Unified Optimizer API (Tagged Union + Explicit Dispatch)
// ============================================================================

// Supported optimizer types
OptimizerType :: enum {
	SGD,
	Adam,
	RMSProp,
	// Add new optimizers here
}

// Unified optimizer config (passed to init)
OptimizerConfig :: struct {
	type:          OptimizerType,
	learning_rate: f64,
	// SGD-specific
	momentum:      f64, // [0, 1), default 0.0
	// Adam/RMSProp-specific
	beta1:         f64, // Adam: first moment decay, default 0.9
	beta2:         f64, // Adam/RMSProp: second moment decay, default 0.999
	epsilon:       f64, // Numerical stability, default 1e-8
	// Common
	weight_decay:  f64, // L2 regularization (AdamW-style), default 0.0
}

// Default config helper
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
			type          = .RMSProp,
			learning_rate = 0.01,
			beta2         = 0.99, // alpha in RMSProp
			epsilon       = 1e-8,
			weight_decay  = 0.0,
		}
	}
	return OptimizerConfig{}
}

// ============================================================================
// Unified Optimizer Struct (Tagged Union)
// Holds state for exactly one optimizer type
// ============================================================================

Optimizer :: struct {
	config:    OptimizerConfig,
	// Internal state (only one branch is valid)
	sgd:       OptimizerSGD,
	adam:      OptimizerAdam,
	rmsprop:   OptimizerRMSProp,
	allocator: mem.Allocator,
}

// Contract: config.type is valid, n_params >= 0
// Initializes unified optimizer for slice-based updates
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
			config.beta2, // beta2 = alpha for RMSProp
			config.epsilon,
			config.weight_decay,
			allocator,
		)
	}
	return opt
}

// Contract: config.type is valid, rows/cols >= 0
// Initializes unified optimizer for matrix-based updates
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
	}
	return opt
}

// Contract: opt initialized via optimizer_init or optimizer_init_mat
optimizer_free :: proc(opt: ^Optimizer) {
	switch opt.config.type {
	case .SGD:
		optimizer_sgd_free(&opt.sgd)
	case .Adam:
		optimizer_adam_free(&opt.adam)
	case .RMSProp:
		optimizer_rmsprop_free(&opt.rmsprop)
	}
}

// ============================================================================
// Unified Step API: Slice Version
// Contract: len(weights) == len(gradient) > 0
// Updates weights in-place using the configured optimizer
// ============================================================================

optimizer_step :: proc(opt: ^Optimizer, weights: []f64, gradient: []f64) {
	switch opt.config.type {
	case .SGD:
		optimizer_sgd_step(&opt.sgd, weights, gradient)
	case .Adam:
		optimizer_adam_step(&opt.adam, weights, gradient)
	case .RMSProp:
		optimizer_rmsprop_step(&opt.rmsprop, weights, gradient)
	}
}

// ============================================================================
// Unified Step API: Matrix Version
// Contract: weights.rows == gradient.rows, weights.cols == gradient.cols
// Updates each row independently using the configured optimizer
// ============================================================================

optimizer_step_mat :: proc(opt: ^Optimizer, weights: ^l.Matrix(f64), gradient: ^l.Matrix(f64)) {
	switch opt.config.type {
	case .SGD:
		optimizer_sgd_step_mat(&opt.sgd, weights, gradient)
	case .Adam:
		optimizer_adam_step_mat(&opt.adam, weights, gradient)
	case .RMSProp:
		optimizer_rmsprop_step_mat(&opt.rmsprop, weights, gradient)
	}
}

// ============================================================================
// Convenience: Update Learning Rate at Runtime
// Contract: lr > 0
// ============================================================================

optimizer_set_learning_rate :: proc(opt: ^Optimizer, lr: f64) {
	opt.config.learning_rate = lr
	switch opt.config.type {
	case .SGD:
		opt.sgd.learning_rate = lr
	case .Adam:
		opt.adam.learning_rate = lr
	case .RMSProp:
		opt.rmsprop.learning_rate = lr
	}
}

// ============================================================================
// Optional: Query Optimizer State (for debugging/logging)
// ============================================================================

OptimizerState :: struct {
	type:          OptimizerType,
	learning_rate: f64,
	step_count:    int, // For Adam/RMSProp bias correction
}

optimizer_state :: proc(opt: ^Optimizer) -> OptimizerState {
	state: OptimizerState
	state.type = opt.config.type
	state.learning_rate = opt.config.learning_rate

	switch opt.config.type {
	case .Adam:
		state.step_count = opt.adam.t
	case .RMSProp:
		state.step_count = 0 // RMSProp has no global step
	case .SGD:
		state.step_count = 0
	}
	return state
}
