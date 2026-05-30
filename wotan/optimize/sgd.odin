package optimize

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// SGD Optimizer (Reusable across all gradient-based models)
// ============================================================================

OptimizerSGD :: struct {
	learning_rate: f64,
	momentum:      f64,
	weight_decay:  f64,
	velocity:      []f64, // Exponential moving average buffer
	allocator:     mem.Allocator,
}

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
	if momentum != 0.0 {
		opt.velocity = make([]f64, n_params, allocator)
	}
	return opt
}

optimizer_sgd_free :: proc(opt: ^OptimizerSGD) {
	if len(opt.velocity) > 0 {
		delete(opt.velocity, opt.allocator)
	}
}

// Core update: weights -= lr * (gradient + weight_decay * weights)
// If momentum > 0, applies Nesterov-style velocity tracking
optimizer_sgd_step :: proc(opt: ^OptimizerSGD, weights: []f64, gradient: []f64) {
	n := len(weights)
	if opt.momentum > 0.0 {
		for i in 0 ..< n {
			opt.velocity[i] =
				opt.momentum * opt.velocity[i] + gradient[i] + opt.weight_decay * weights[i]
			weights[i] -= opt.learning_rate * opt.velocity[i]
		}
	} else {
		if opt.weight_decay > 0.0 {
			for i in 0 ..< n {
				weights[i] -= opt.learning_rate * (gradient[i] + opt.weight_decay * weights[i])
			}
		} else {
			for i in 0 ..< n {
				weights[i] -= opt.learning_rate * gradient[i]
			}
		}
	}
}

// ============================================================================
// Learning Rate Scheduler (Linear Decay)
// ============================================================================

LinearDecayScheduler :: struct {
	initial_lr:  f64,
	current_lr:  f64,
	total_steps: int,
	step:        int,
}

scheduler_linear_decay_init :: proc(start_lr: f64, total_steps: int) -> LinearDecayScheduler {
	return LinearDecayScheduler {
		initial_lr = start_lr,
		current_lr = start_lr,
		total_steps = total_steps,
		step = 0,
	}
}

scheduler_step :: proc(sched: ^LinearDecayScheduler) {
	sched.step += 1
	if sched.total_steps > 0 {
		sched.current_lr = sched.initial_lr * f64(1.0 - f64(sched.step) / f64(sched.total_steps))
	}
}
