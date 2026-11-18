package gpu

import t "../tensor"
import "core:math"

// ============================================================================
// Optimizer kernels for device parameters (called by wotan/nn)
// ============================================================================
//
// The optimizer state of a device parameter (Adam moments, SGD velocity) lives
// in device buffers keyed by the host state slice of the optimizer, which stays
// allocated: when the parameter returns to the host, or a checkpoint is saved,
// optim_sync copies the state back into it.

@(private)
_optim_state :: proc(key: rawptr, a, b: []f64) -> Optim_State {
	if s, ok := G.optim[key]; ok {return s}
	s := Optim_State {
		n = len(a),
	}
	s.a = _acquire(len(a))
	_upload(s.a, a)
	if b != nil {
		s.b = _acquire(len(b))
		_upload(s.b, b)
	}
	G.optim[key] = s
	return s
}

// the parameter's data and gradient on the device
@(private)
_param :: proc(param: ^t.Tensor) -> (w, g: Bind) {
	t._data_to_device(param)
	t._grad_to_device(param)
	return _data(param), _grad(param)
}

// One launch updates up to ADAM_GROUP parameters (4 storage buffers each) whose
// coefficients agree, as they do when the parameters share the learning rate
// and the step count. The group is smaller if the device binds fewer storage
// buffers per kernel.
@(private)
_hook_adam :: proc(updates: []t.Device_Adam_Update, beta_1, beta_2, epsilon, weight_decay: f64) {
	// the coefficients of optim.adam_update, computed in f64
	coefs :: proc(u: t.Device_Adam_Update, beta_1, beta_2, epsilon, weight_decay: f64) -> [8]f32 {
		step_size := u.lr / (1.0 - math.pow(beta_1, f64(u.step)))
		sqrt_bc2 := math.sqrt(1.0 - math.pow(beta_2, f64(u.step)))
		decay := 1.0 - u.lr * weight_decay
		return {
			f32(beta_1),
			f32(beta_2),
			f32(epsilon),
			f32(step_size),
			f32(sqrt_bc2),
			f32(decay),
			0,
			0,
		}
	}
	group_max := clamp(int(G.limits.maxStorageBuffersPerShaderStage) / 4, 1, ADAM_GROUP)
	i := 0
	for i < len(updates) {
		f := coefs(updates[i], beta_1, beta_2, epsilon, weight_decay)
		binds: [4 * ADAM_GROUP]Bind
		counts: [8]u32
		group: [ADAM_GROUP]^t.Tensor
		count, total := 0, 0
		group_loop: for i < len(updates) && count < group_max {
			u := updates[i]
			if count > 0 && coefs(u, beta_1, beta_2, epsilon, weight_decay) != f {break}
			for x in group[:count] {if x == u.param {break group_loop}} 	// listed twice
			n := t.tensor_numel(u.param)
			s := _optim_state(u.key, u.m, u.v)
			w, g := _param(u.param)
			binds[4 * count + 0] = g
			binds[4 * count + 1] = w
			binds[4 * count + 2] = {s.a, n}
			binds[4 * count + 3] = {s.b, n}
			counts[count] = u32(n)
			group[count] = u.param
			count += 1
			total += n
			i += 1
		}
		_dispatch_1d(_adam_kernel(count), _p(counts, f), binds[:4 * count], total)
		for x in group[:count] {t._data_written_on_device(x)}
	}
}

@(private)
_hook_sgd :: proc(
	param: ^t.Tensor,
	key: rawptr,
	velocity: []f64,
	lr, momentum, weight_decay: f64,
	nesterov: bool,
) {
	n := t.tensor_numel(param)
	vel := Bind{G.dummy, 1}
	if momentum > 0 && key != nil {vel = {_optim_state(key, velocity, nil).a, n}}
	w, g := _param(param)
	u := [8]u32{u32(n), u32(momentum > 0), u32(nesterov), 0, 0, 0, 0, 0}
	_dispatch_1d(
		.Sgd,
		_p(u, {f32(lr), f32(momentum), f32(weight_decay), 0, 0, 0, 0, 0}),
		{g, w, vel},
		n,
	)
	t._data_written_on_device(param)
}

@(private)
_hook_optim_sync :: proc(key: rawptr, a, b: []f64, release: bool) -> bool {
	s, ok := G.optim[key]
	if !ok {return false}
	_download(s.a, a)
	if s.b != nil && b != nil {_download(s.b, b)}
	if release {_hook_optim_release(key)}
	return true
}

@(private)
_hook_optim_release :: proc(key: rawptr) {
	s, ok := G.optim[key]
	if !ok {return}
	_release_buf(s.a)
	if s.b != nil {_release_buf(s.b)}
	delete_key(&G.optim, key)
}

// Σ of the squared gradients of `params` (device gradients reduced on the
// device into one slot each, read back once)
@(private)
_hook_grad_sq_sum :: proc(params: []^t.Tensor) -> f64 {
	total := 0.0
	slots := _acquire(len(params))
	_zero(slots, len(params))
	for x, i in params {
		if x.device.grad_where == .Host {
			for v in x.grad.data {total += v * v}
			continue
		}
		n := t.tensor_numel(x)
		_reduce(_grad(x), n, {slots, len(params)}, i, 1, true)
	}
	host := make([]f64, len(params), context.temp_allocator)
	_download(slots, host)
	_release_buf(slots)
	for v in host {total += v}
	return total
}

@(private)
_hook_grad_scale :: proc(x: ^t.Tensor, coef: f64) {
	if x.device.grad_where == .Host {
		for &v in x.grad.data {v *= coef}
		return
	}
	n := t.tensor_numel(x)
	_dispatch_1d(
		.Scale_Inplace,
		_p({u32(n), 0, 0, 0, 0, 0, 0, 0}, {f32(coef), 0, 0, 0, 0, 0, 0, 0}),
		{_grad(x)},
		n,
	)
	t._grad_written_on_device(x)
}
