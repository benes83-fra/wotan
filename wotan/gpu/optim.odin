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

@(private)
_hook_adam :: proc(
	param: ^t.Tensor,
	key: rawptr,
	m, v: []f64,
	lr, beta_1, beta_2, epsilon, weight_decay: f64,
	step: int,
) {
	n := t.tensor_numel(param)
	s := _optim_state(key, m, v)
	w, g := _param(param)
	// the coefficients of optim.adam_update, computed in f64
	step_size := lr / (1.0 - math.pow(beta_1, f64(step)))
	sqrt_bc2 := math.sqrt(1.0 - math.pow(beta_2, f64(step)))
	decay := 1.0 - lr * weight_decay
	f := [8]f32 {
		f32(beta_1),
		f32(beta_2),
		f32(epsilon),
		f32(step_size),
		f32(sqrt_bc2),
		f32(decay),
		0,
		0,
	}
	_dispatch_1d(.Adam, _p({u32(n), 0, 0, 0, 0, 0, 0, 0}, f), {g, w, {s.a, n}, {s.b, n}}, n)
	t._data_written_on_device(param)
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
