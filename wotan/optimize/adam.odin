package optimize

import "core:math"
import "core:mem"

// ============================================================================
// Adam Optimizer (Adaptive Moments)
// ============================================================================

OptimizerAdam :: struct {
	learning_rate: f64,
	beta1:         f64, // First moment decay (default 0.9)
	beta2:         f64, // Second moment decay (default 0.999)
	epsilon:       f64, // Numerical stability (default 1e-8)
	weight_decay:  f64, // L2 regularization (AdamW style)
	t:             int, // Time step for bias correction
	m:             []f64, // First moment estimate
	v:             []f64, // Second moment estimate (uncentered variance)
	allocator:     mem.Allocator,
}

// Contract: n_params >= 0, lr > 0, beta1/beta2 in [0,1), epsilon > 0
optimizer_adam_init :: proc(
	n_params: int,
	lr: f64 = 0.001,
	beta1: f64 = 0.9,
	beta2: f64 = 0.999,
	epsilon: f64 = 1e-8,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerAdam {
	opt: OptimizerAdam
	opt.learning_rate = lr
	opt.beta1 = beta1
	opt.beta2 = beta2
	opt.epsilon = epsilon
	opt.weight_decay = weight_decay
	opt.t = 0
	opt.allocator = allocator

	opt.m = make([]f64, n_params, allocator)
	opt.v = make([]f64, n_params, allocator)
	for i in 0 ..< n_params {
		opt.m[i] = 0.0
		opt.v[i] = 0.0
	}
	return opt
}

// Contract: opt must be initialized via optimizer_adam_init
optimizer_adam_free :: proc(opt: ^OptimizerAdam) {
	if len(opt.m) > 0 {delete(opt.m, opt.allocator)}
	if len(opt.v) > 0 {delete(opt.v, opt.allocator)}
}

// Contract: len(weights) == len(gradient) > 0, opt initialized
// Updates weights using Adam algorithm with optional AdamW-style weight decay
optimizer_adam_step :: proc(opt: ^OptimizerAdam, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}

	opt.t += 1

	// AdamW-style weight decay: apply directly to weights
	if opt.weight_decay > 0.0 {
		for i in 0 ..< n {
			weights[i] *= 1.0 - opt.learning_rate * opt.weight_decay
		}
	}

	// Update biased first and second moment estimates
	for i in 0 ..< n {
		opt.m[i] = opt.beta1 * opt.m[i] + (1.0 - opt.beta1) * gradient[i]
		opt.v[i] = opt.beta2 * opt.v[i] + (1.0 - opt.beta2) * gradient[i] * gradient[i]
	}

	// Bias correction
	m_hat_scale := 1.0 / (1.0 - math.pow(opt.beta1, f64(opt.t)))
	v_hat_scale := 1.0 / (1.0 - math.pow(opt.beta2, f64(opt.t)))

	// Update weights: w -= lr * m_hat / (sqrt(v_hat) + eps)
	for i in 0 ..< n {
		m_hat := opt.m[i] * m_hat_scale
		v_hat := opt.v[i] * v_hat_scale
		weights[i] -= opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)
	}
}
