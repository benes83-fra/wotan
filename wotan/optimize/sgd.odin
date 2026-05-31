package optimize

import l "../linalg"
import "core:mem"

// ============================================================================
// SGD Optimizer (SIMD-Accelerated)
// ============================================================================

OptimizerSGD :: struct {
	learning_rate: f64,
	momentum:      f64,
	weight_decay:  f64,
	velocity:      []f64, // Momentum buffer (allocated if momentum > 0)
	allocator:     mem.Allocator,
}

// Contract: n_params >= 0, lr > 0, momentum in [0,1), weight_decay >= 0
optimizer_sgd_init :: proc(
	n_params: int,
	lr: f64,
	momentum: f64 = 0.0,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerSGD {
	opt: OptimizerSGD
	opt.learning_rate = lr
	opt.momentum = momentum
	opt.weight_decay = weight_decay
	opt.allocator = allocator

	if momentum > 0.0 {
		opt.velocity = make([]f64, n_params, allocator)
		for i in 0 ..< n_params {opt.velocity[i] = 0.0}
	}
	return opt
}

// Contract: opt must be initialized via optimizer_sgd_init
optimizer_sgd_free :: proc(opt: ^OptimizerSGD) {
	if len(opt.velocity) > 0 {
		delete(opt.velocity, opt.allocator)
	}
}

// Contract: len(weights) == len(gradient) > 0, opt initialized
// Updates weights in-place using: w -= lr * (gradient + weight_decay * w)
// If momentum > 0: uses velocity buffer for momentum update
optimizer_sgd_step :: proc(opt: ^OptimizerSGD, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}

	if opt.momentum > 0.0 {
		// Momentum: v = μ*v + g + λ*w; then w -= lr * v
		for i in 0 ..< n {
			opt.velocity[i] =
				opt.momentum * opt.velocity[i] + gradient[i] + opt.weight_decay * weights[i]
		}
		// SIMD: weights -= lr * velocity
		l.axpy_simd(-opt.learning_rate, opt.velocity, weights)
	} else {
		if opt.weight_decay > 0.0 {
			// weights = (1 - lr*λ)*weights - lr*gradient
			decay_factor := 1.0 - opt.learning_rate * opt.weight_decay
			for i in 0 ..< n {weights[i] *= decay_factor}
			l.axpy_simd(-opt.learning_rate, gradient, weights)
		} else {
			// Simple: weights -= lr * gradient (SIMD)
			l.axpy_simd(-opt.learning_rate, gradient, weights)
		}
	}
}
