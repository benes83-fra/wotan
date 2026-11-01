package nn

import l "../linalg"
import optim "../optimize"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// 1. Adam Optimizer
// ============================================================================
// torch.optim.Adam / AdamW (decoupled weight decay) over registered tensors;
// the update itself is optimize.adam_update. Gradients are used as they are:
// clip them with clip_grad_norm before adam_step if needed.

Adam :: struct {
	parameters:     [dynamic]^t.Tensor,
	moment_1:       [dynamic][]f64,
	moment_2:       [dynamic][]f64,
	learning_rate:  f64,
	learning_rates: [dynamic]Maybe(f64), // per-parameter override of learning_rate
	beta_1:         f64,
	beta_2:         f64,
	epsilon:        f64,
	weight_decay:   f64, // decoupled (AdamW); 0 is plain Adam
	timestep:       int,
	allocator:      mem.Allocator,
}

adam_new :: proc(
	learning_rate: f64 = 0.001,
	beta_1: f64 = 0.9,
	beta_2: f64 = 0.999,
	epsilon: f64 = 1e-8,
	allocator: mem.Allocator = context.allocator,
	weight_decay: f64 = 0.0,
) -> Adam {
	return Adam {
		parameters = make([dynamic]^t.Tensor, 0, allocator),
		moment_1 = make([dynamic][]f64, 0, allocator),
		moment_2 = make([dynamic][]f64, 0, allocator),
		learning_rates = make([dynamic]Maybe(f64), 0, allocator),
		learning_rate = learning_rate,
		beta_1 = beta_1,
		beta_2 = beta_2,
		epsilon = epsilon,
		weight_decay = weight_decay,
		allocator = allocator,
	}
}

adam_add_param :: proc(opt: ^Adam, param: ^t.Tensor) {
	_adam_register(opt, param, nil)
}

// Registers a parameter with its own learning rate
adam_add_param_with_lr :: proc(opt: ^Adam, param: ^t.Tensor, lr: f64) {
	_adam_register(opt, param, lr)
}

@(private = "file")
_adam_register :: proc(opt: ^Adam, param: ^t.Tensor, lr: Maybe(f64)) {
	n := len(param.data.data)
	append(&opt.parameters, param)
	append(&opt.learning_rates, lr)
	append(&opt.moment_1, make([]f64, n, opt.allocator))
	append(&opt.moment_2, make([]f64, n, opt.allocator))
}

adam_step :: proc(opt: ^Adam) {
	opt.timestep += 1
	for param, i in opt.parameters {
		if !param.requires_grad || param.grad.data == nil {continue}
		lr := opt.learning_rates[i].? or_else opt.learning_rate
		optim.adam_update(
			param.data.data,
			param.grad.data,
			opt.moment_1[i],
			opt.moment_2[i],
			lr,
			opt.beta_1,
			opt.beta_2,
			opt.epsilon,
			opt.weight_decay,
			opt.timestep,
		)
	}
}

adam_free :: proc(opt: ^Adam) {
	for m in opt.moment_1 {delete(m, opt.allocator)}
	for v in opt.moment_2 {delete(v, opt.allocator)}
	delete(opt.moment_1)
	delete(opt.moment_2)
	delete(opt.learning_rates)
	delete(opt.parameters)
}

adam_zero_grad :: proc(opt: ^Adam) {
	for param in opt.parameters {
		t.tensor_zero_grad(param)
	}
}

// Clips the global norm of all gradients as torch.nn.utils.clip_grad_norm_:
// when ‖g‖ > max_norm every gradient is scaled by max_norm / (‖g‖ + 1e-6).
// Returns the norm before clipping.
clip_grad_norm_params :: proc(params: []^t.Tensor, max_norm: f64) -> f64 {
	total_sq := 0.0
	for param in params {
		if param.grad.data != nil {total_sq += l.dot_simd(param.grad.data, param.grad.data)}
	}
	total := math.sqrt(total_sq)
	coef := max_norm / (total + 1e-6)
	if coef < 1.0 {
		for param in params {
			if param.grad.data != nil {l.vec_scale_simd(param.grad.data, coef, param.grad.data)}
		}
	}
	return total
}

clip_grad_norm_adam :: proc(opt: ^Adam, max_norm: f64) -> f64 {
	return clip_grad_norm_params(opt.parameters[:], max_norm)
}

clip_grad_norm_sgd :: proc(opt: ^SGD, max_norm: f64) -> f64 {
	return clip_grad_norm_params(opt.parameters[:], max_norm)
}

clip_grad_norm :: proc {
	clip_grad_norm_params,
	clip_grad_norm_adam,
	clip_grad_norm_sgd,
}
