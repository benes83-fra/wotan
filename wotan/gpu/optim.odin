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

// Σ of the squared gradients of `params`: Grad_Sq_Sum adds up to 7 device
// gradients per launch into one partial sum per workgroup; the partial sums
// are read back once and added in f64
@(private)
_hook_grad_sq_sum :: proc(params: []^t.Tensor) -> f64 {
	total := 0.0
	dev := make([dynamic]^t.Tensor, 0, len(params), context.temp_allocator)
	for x in params {
		if x.device.grad_where == .Host {
			for v in x.grad.data {total += v * v}
		} else {
			append(&dev, x)
		}
	}
	if len(dev) == 0 {return total}
	// workgroups per launch: one per 256 elements, at most 256 (grid-stride)
	launches := (len(dev) + GRAD_SUM_GROUP - 1) / GRAD_SUM_GROUP
	groups := make([]int, launches, context.temp_allocator)
	slots_n := 0
	for &g, j in groups {
		n := 0
		for x in dev[j * GRAD_SUM_GROUP:min(len(dev), (j + 1) * GRAD_SUM_GROUP)] {
			n += t.tensor_numel(x)
		}
		g = clamp((n + 255) / 256, 1, 256)
		slots_n += g
	}
	slots := _acquire(slots_n)
	offset := 0
	for g, j in groups {
		chunk := dev[j * GRAD_SUM_GROUP:min(len(dev), (j + 1) * GRAD_SUM_GROUP)]
		binds: [GRAD_SUM_GROUP + 1]Bind
		u: [8]u32
		for k in 0 ..< GRAD_SUM_GROUP {
			if k < len(chunk) {
				binds[k] = _grad(chunk[k])
				u[k] = u32(t.tensor_numel(chunk[k]))
			} else {
				binds[k] = {G.dummy, 1}
			}
		}
		binds[GRAD_SUM_GROUP] = {slots, slots_n}
		u[7] = u32(offset)
		_dispatch(.Grad_Sq_Sum, _p(u), binds[:], u32(g), 1)
		offset += g
	}
	host := make([]f64, slots_n, context.temp_allocator)
	_download(slots, host)
	_release_buf(slots)
	for v in host {total += v}
	return total
}

// g *= coef for the gradients of `params`: Grad_Scale scales up to 8 device
// gradients per launch. A parameter listed twice is scaled twice, as on the host.
@(private)
_hook_grad_scale :: proc(params: []^t.Tensor, coef: f64) {
	binds: [GRAD_SCALE_GROUP]Bind
	group: [GRAD_SCALE_GROUP]^t.Tensor
	u: [8]u32
	count, total := 0, 0
	flush :: proc(
		binds: ^[GRAD_SCALE_GROUP]Bind,
		group: []^t.Tensor,
		u: [8]u32,
		total: int,
		coef: f64,
	) {
		if len(group) == 0 {return}
		for k in len(group) ..< GRAD_SCALE_GROUP {binds[k] = {G.dummy, 1}}
		_dispatch_1d(.Grad_Scale, _p(u, {f32(coef), 0, 0, 0, 0, 0, 0, 0}), binds[:], total)
		for x in group {t._grad_written_on_device(x)}
	}
	for x in params {
		if x.device.grad_where == .Host {
			for &v in x.grad.data {v *= coef}
			continue
		}
		listed := false
		for y in group[:count] {if y == x {listed = true}}
		if count == GRAD_SCALE_GROUP || listed {
			flush(&binds, group[:count], u, total, coef)
			u = {}
			count, total = 0, 0
		}
		n := t.tensor_numel(x)
		binds[count] = _grad(x)
		group[count] = x
		u[count] = u32(n)
		count += 1
		total += n
	}
	flush(&binds, group[:count], u, total, coef)
}
