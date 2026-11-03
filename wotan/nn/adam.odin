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
// clip them with clip_grad_norm before adam_step if needed. A parameter with
// its own learning rate keeps its ratio to learning_rate, so a learning-rate
// schedule (optimize.lr_schedule_at) scales it as well; adam_step_scheduled
// schedules it as its own torch param group instead (the same cosine floor).
// As torch, every parameter counts its own steps for the bias correction:
// only steps that update it (it has a gradient) count, so a parameter added
// later or left out of some steps starts its correction from its first update.

Adam :: struct {
	parameters:    [dynamic]^t.Tensor,
	moment_1:      [dynamic][]f64,
	moment_2:      [dynamic][]f64,
	learning_rate: f64,
	lr_scales:     [dynamic]f64, // per parameter: its learning rate / learning_rate
	beta_1:        f64,
	beta_2:        f64,
	epsilon:       f64,
	weight_decay:  f64, // decoupled (AdamW); 0 is plain Adam
	timestep:      int, // calls of adam_step
	steps:         [dynamic]int, // per parameter: its updates so far (bias correction)
	allocator:     mem.Allocator,
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
		lr_scales = make([dynamic]f64, 0, allocator),
		steps = make([dynamic]int, 0, allocator),
		learning_rate = learning_rate,
		beta_1 = beta_1,
		beta_2 = beta_2,
		epsilon = epsilon,
		weight_decay = weight_decay,
		allocator = allocator,
	}
}

adam_add_param :: proc(opt: ^Adam, param: ^t.Tensor) {
	_adam_register(opt, param, 1.0)
}

// Registers a parameter with its own learning rate (relative to the current
// learning_rate from then on)
adam_add_param_with_lr :: proc(opt: ^Adam, param: ^t.Tensor, lr: f64) {
	assert(
		opt.learning_rate > 0,
		"adam_add_param_with_lr: the optimizer's learning rate must be positive",
	)
	_adam_register(opt, param, lr / opt.learning_rate)
}

@(private = "file")
_adam_register :: proc(opt: ^Adam, param: ^t.Tensor, lr_scale: f64) {
	n := len(param.data.data)
	append(&opt.parameters, param)
	append(&opt.lr_scales, lr_scale)
	append(&opt.steps, 0)
	append(&opt.moment_1, make([]f64, n, opt.allocator))
	append(&opt.moment_2, make([]f64, n, opt.allocator))
}

adam_step :: proc(opt: ^Adam) {
	_adam_step(opt, nil, 0)
}

// adam_step at step `step` of schedule s, every parameter scheduled as its own
// torch param group from its base rate s.base_lr · (its rate / learning_rate
// when it was added): warmup / step / linear decay scale all rates alike, a
// cosine takes each one to the same final_lr (as CosineAnnealingLR's eta_min,
// where scaling the shared rate would end at final_lr · ratio). Sets
// learning_rate to the schedule's rate.
adam_step_scheduled :: proc(opt: ^Adam, s: optim.LR_Schedule, step: int) {
	opt.learning_rate = optim.lr_schedule_at(s, step)
	s := s
	_adam_step(opt, &s, step)
}

@(private = "file")
_adam_step :: proc(opt: ^Adam, schedule: ^optim.LR_Schedule, step: int) {
	opt.timestep += 1
	for param, i in opt.parameters {
		if !param.requires_grad || param.grad.data == nil {continue}
		opt.steps[i] += 1
		lr := opt.learning_rate * opt.lr_scales[i]
		if schedule != nil && opt.lr_scales[i] != 1 {
			lr = optim.lr_schedule_at(schedule^, step, schedule.base_lr * opt.lr_scales[i])
		}
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
			opt.steps[i],
		)
	}
}

adam_free :: proc(opt: ^Adam) {
	for m in opt.moment_1 {delete(m, opt.allocator)}
	for v in opt.moment_2 {delete(v, opt.allocator)}
	delete(opt.moment_1)
	delete(opt.moment_2)
	delete(opt.lr_scales)
	delete(opt.steps)
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
