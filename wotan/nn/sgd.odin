package nn

import optim "../optimize"
import t "../tensor"
import "core:mem"

// ============================================================================
// 1. Stochastic Gradient Descent (SGD) Optimizer
// ============================================================================
// torch.optim.SGD (dampening 0) over registered tensors; the update itself is
// optimize.sgd_update.

SGD :: struct {
	parameters:    [dynamic]^t.Tensor, // Pointers to weights and biases
	velocity:      [dynamic][]f64, // momentum buffers (nil while momentum = 0)
	learning_rate: f64,
	momentum:      f64,
	weight_decay:  f64, // L2 penalty added to the gradient
	nesterov:      bool,
	allocator:     mem.Allocator,
}

// sgd_new creates a new SGD optimizer
sgd_new :: proc(
	learning_rate: f64,
	allocator: mem.Allocator = context.allocator,
	momentum: f64 = 0.0,
	weight_decay: f64 = 0.0,
	nesterov: bool = false,
) -> SGD {
	return SGD {
		parameters = make([dynamic]^t.Tensor, 0, allocator),
		velocity = make([dynamic][]f64, 0, allocator),
		learning_rate = learning_rate,
		momentum = momentum,
		weight_decay = weight_decay,
		nesterov = nesterov,
		allocator = allocator,
	}
}

// sgd_add_param registers a tensor (like weights or bias) to be optimized
sgd_add_param :: proc(opt: ^SGD, param: ^t.Tensor) {
	append(&opt.parameters, param)
	append(&opt.velocity, nil)
}

// sgd_step performs one step: w ← w − lr·d (see optimize.sgd_update)
sgd_step :: proc(opt: ^SGD) {
	for param, i in opt.parameters {
		if !param.requires_grad || param.grad.data == nil {continue}
		if opt.momentum > 0.0 && opt.velocity[i] == nil {
			opt.velocity[i] = make([]f64, len(param.data.data), opt.allocator)
		}
		optim.sgd_update(
			param.data.data,
			param.grad.data,
			opt.velocity[i],
			opt.learning_rate,
			opt.momentum,
			opt.weight_decay,
			opt.nesterov,
		)
	}
}

// sgd_zero_grad resets all gradients to 0.0 (required before the next backward pass)
sgd_zero_grad :: proc(opt: ^SGD) {
	for param in opt.parameters {
		t.tensor_zero_grad(param)
	}
}

// sgd_free cleans up the optimizer's internal lists
sgd_free :: proc(opt: ^SGD) {
	for v in opt.velocity {if v != nil {delete(v, opt.allocator)}}
	delete(opt.velocity)
	delete(opt.parameters)
}
