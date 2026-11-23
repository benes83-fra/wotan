package nn

import optim "../optimize"
import t "../tensor"
import "core:mem"

// ============================================================================
// 1. Stochastic Gradient Descent (SGD) Optimizer
// ============================================================================
// torch.optim.SGD (dampening 0) over registered tensors; the update itself is
// optimize.sgd_update. As torch, a parameter without a gradient (dropped by
// sgd_zero_grad and not reached by the backward pass) is skipped: no momentum
// or weight-decay update. The velocity of a parameter on the GPU lives on the
// device (the host slice is then nil).

SGD :: struct {
	parameters:    [dynamic]^t.Tensor, // Pointers to weights and biases
	velocity:      [dynamic][]f64, // momentum buffers (nil while momentum = 0 or on the device)
	device_keys:   [dynamic]rawptr, // per parameter: identifies its velocity on the device
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
		device_keys = make([dynamic]rawptr, 0, allocator),
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
	append(&opt.device_keys, new(u8, opt.allocator))
}

// sgd_step performs one step: w ← w − lr·d (see optimize.sgd_update)
sgd_step :: proc(opt: ^SGD) {
	for param, i in opt.parameters {
		if !param.requires_grad || !t.tensor_has_grad(param) {continue}
		key := opt.device_keys[i] if opt.momentum > 0.0 else nil
		if param.device != nil && t._device_hooks.sgd != nil {
			// a velocity from a host step goes to the device with its values;
			// a new one starts as zeros there
			t._device_hooks.sgd(
				param,
				key,
				opt.velocity[i],
				opt.learning_rate,
				opt.momentum,
				opt.weight_decay,
				opt.nesterov,
			)
			if opt.velocity[i] !=
			   nil {delete(opt.velocity[i], opt.allocator); opt.velocity[i] = nil}
			continue
		}
		if opt.momentum > 0.0 && opt.velocity[i] == nil {
			opt.velocity[i] = make([]f64, t.tensor_numel(param), opt.allocator)
		}
		if key != nil && t._device_hooks.optim_sync != nil {
			// the parameter left the device: continue from its device velocity
			t._device_hooks.optim_sync(key, opt.velocity[i], nil, true)
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

// sgd_zero_grad drops all gradients (as adam_zero_grad); call it before the
// next backward pass
sgd_zero_grad :: proc(opt: ^SGD) {
	_drop_grads(opt.parameters[:])
}

// sgd_free cleans up the optimizer's internal lists
sgd_free :: proc(opt: ^SGD) {
	if t._device_hooks.optim_release != nil {
		for k in opt.device_keys {t._device_hooks.optim_release(k)}
	}
	for v in opt.velocity {if v != nil {delete(v, opt.allocator)}}
	for k in opt.device_keys {free(k, opt.allocator)}
	delete(opt.velocity)
	delete(opt.device_keys)
	delete(opt.parameters)
}
