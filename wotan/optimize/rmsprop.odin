package optimize

import "core:math"
import "core:mem"

// ============================================================================
// RMSProp Optimizer
// ============================================================================

OptimizerRMSProp :: struct {
	learning_rate: f64,
	alpha:         f64, // Decay rate for squared gradients (default 0.99)
	epsilon:       f64, // Numerical stability (default 1e-8)
	weight_decay:  f64, // L2 regularization
	cache:         []f64, // Running average of squared gradients
	allocator:     mem.Allocator,
}

// Contract: n_params >= 0, lr > 0, alpha in [0,1), epsilon > 0
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

// Contract: opt must be initialized via optimizer_rmsprop_init
optimizer_rmsprop_free :: proc(opt: ^OptimizerRMSProp) {
	if len(opt.cache) > 0 {delete(opt.cache, opt.allocator)}
}

// Contract: len(weights) == len(gradient) > 0, opt initialized
// Updates weights using RMSProp algorithm with optional weight decay
optimizer_rmsprop_step :: proc(opt: ^OptimizerRMSProp, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}

	// L2 regularization (applied like AdamW)
	if opt.weight_decay > 0.0 {
		for i in 0 ..< n {
			weights[i] *= 1.0 - opt.learning_rate * opt.weight_decay
		}
	}

	// Update cache: cache = α*cache + (1-α)*g²
	for i in 0 ..< n {
		opt.cache[i] = opt.alpha * opt.cache[i] + (1.0 - opt.alpha) * gradient[i] * gradient[i]
	}

	// Update weights: w -= lr * g / (sqrt(cache) + eps)
	for i in 0 ..< n {
		weights[i] -= opt.learning_rate * gradient[i] / (math.sqrt(opt.cache[i]) + opt.epsilon)
	}
}
