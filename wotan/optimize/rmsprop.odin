package optimize

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// RMSProp Optimizer (Slice + Matrix APIs)
// ============================================================================

OptimizerRMSProp :: struct {
	learning_rate: f64,
	alpha:         f64,
	epsilon:       f64,
	weight_decay:  f64,
	cache:         []f64, // Slice buffer
	cache_mat:     l.Matrix(f64), // Matrix buffer
	allocator:     mem.Allocator,
}

optimizer_rmsprop_init :: proc(
	n_params: int,
	lr: f64 = 0.01,
	alpha: f64 = 0.99,
	epsilon: f64 = 1e-8,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerRMSProp {
	opt: OptimizerRMSProp
	opt.learning_rate = lr
	opt.alpha = alpha
	opt.epsilon = epsilon
	opt.weight_decay = weight_decay
	opt.allocator = allocator

	opt.cache = make([]f64, n_params, allocator)
	for i in 0 ..< n_params {opt.cache[i] = 0.0}
	return opt
}

optimizer_rmsprop_init_mat :: proc(
	rows, cols: int,
	lr: f64 = 0.01,
	alpha: f64 = 0.99,
	epsilon: f64 = 1e-8,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerRMSProp {
	opt: OptimizerRMSProp
	opt.learning_rate = lr
	opt.alpha = alpha
	opt.epsilon = epsilon
	opt.weight_decay = weight_decay
	opt.allocator = allocator

	opt.cache_mat = l.matrix_new(f64, rows, cols, allocator)
	for i in 0 ..< rows * cols {opt.cache_mat.data[i] = 0.0}
	return opt
}

optimizer_rmsprop_free :: proc(opt: ^OptimizerRMSProp) {
	if len(opt.cache) > 0 {delete(opt.cache, opt.allocator)}
	if opt.cache_mat.data != nil {l.matrix_free(&opt.cache_mat)}
}

// ============================================================================
// Slice API
// ============================================================================
optimizer_rmsprop_step :: proc(opt: ^OptimizerRMSProp, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}

	if opt.weight_decay > 0.0 {
		for i in 0 ..< n {
			weights[i] *= 1.0 - opt.learning_rate * opt.weight_decay
		}
	}

	for i in 0 ..< n {
		opt.cache[i] = opt.alpha * opt.cache[i] + (1.0 - opt.alpha) * gradient[i] * gradient[i]
	}

	for i in 0 ..< n {
		weights[i] -= opt.learning_rate * gradient[i] / (math.sqrt(opt.cache[i]) + opt.epsilon)
	}
}

// ============================================================================
// Matrix API
// ============================================================================
optimizer_rmsprop_step_mat :: proc(
	opt: ^OptimizerRMSProp,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
) {
	if weights.rows != gradient.rows || weights.cols != gradient.cols {return}
	if weights.rows == 0 || weights.cols == 0 {return}

	n := weights.rows * weights.cols

	if opt.weight_decay > 0.0 {
		for i in 0 ..< n {
			weights.data[i] *= 1.0 - opt.learning_rate * opt.weight_decay
		}
	}

	for i in 0 ..< n {
		opt.cache_mat.data[i] =
			opt.alpha * opt.cache_mat.data[i] +
			(1.0 - opt.alpha) * gradient.data[i] * gradient.data[i]
	}

	for i in 0 ..< n {
		weights.data[i] -=
			opt.learning_rate * gradient.data[i] / (math.sqrt(opt.cache_mat.data[i]) + opt.epsilon)
	}
}
