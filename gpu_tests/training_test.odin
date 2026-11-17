package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

// _mlp: a Sequential MLP with a fixed seed; parameters rounded to f32 so the
// CPU and GPU runs start from identical values
_mlp :: proc(sizes: []int, act: nn.Activation, seed: u64) -> ^nn.Sequential {
	rand.reset(seed)
	model := nn.sequential_new()
	for i in 0 ..< len(sizes) - 1 {
		nn.sequential_add(model, nn.linear_layer_new(sizes[i], sizes[i + 1]))
		if i < len(sizes) - 2 {nn.sequential_add(model, act)}
	}
	params := nn.sequential_parameters(model, context.temp_allocator)
	for p in params {
		for &v in p.data.data {v = f64(f32(v))}
	}
	return model
}

// _regression_data: X [n, d_in] in [-1, 1], y [n, d_out] a smooth function of X
_regression_data :: proc(n, d_in, d_out: int, seed: u64) -> (x, y: ^t.Tensor) {
	rand.reset(seed)
	xm := l.matrix_new(f64, n, d_in)
	ym := l.matrix_new(f64, n, d_out)
	for i in 0 ..< n {
		s := 0.0
		for j in 0 ..< d_in {
			v := f64(f32(2 * rand.float64() - 1))
			xm.data[i * d_in + j] = v
			s += v * f64(j % 3 - 1)
		}
		for k in 0 ..< d_out {ym.data[i * d_out + k] = f64(f32(math.sin(s + f64(k))))}
	}
	return t.tensor_new(xm), t.tensor_new(ym)
}

Train_Opts :: struct {
	on_gpu:   bool,
	use_sgd:  bool,
	clip:     f64, // > 0: clip_grad_norm before the step
	steps:    int,
	no_scope: bool, // free the graph with tensor_free_graph instead of a graph scope
}

// _train runs `steps` training steps and returns the loss per step and the
// final parameters (host copies)
_train :: proc(
	model: ^nn.Sequential,
	x, y: ^t.Tensor,
	o: Train_Opts,
) -> (
	losses: []f64,
	params: [][]f64,
) {
	ps := nn.sequential_parameters(model, context.temp_allocator)
	adam := nn.adam_new(0.01)
	defer nn.adam_free(&adam)
	sgd := nn.sgd_new(0.05, momentum = 0.9, nesterov = true)
	defer nn.sgd_free(&sgd)
	for p in ps {
		if o.use_sgd {nn.sgd_add_param(&sgd, p)} else {nn.adam_add_param(&adam, p)}
	}
	if o.on_gpu {
		gpu.to_device(ps[:])
		gpu.to_device(x)
		gpu.to_device(y)
	}
	losses = make([]f64, o.steps, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for step in 0 ..< o.steps {
		if !o.no_scope {t.graph_begin(&scope)}
		loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
		t.tensor_backward(loss)
		losses[step] = t.tensor_item(loss)
		if o.no_scope {t.tensor_free_graph(loss)} else {t.graph_end(&scope)}
		if o.clip > 0 {nn.clip_grad_norm(ps[:], o.clip)}
		if o.use_sgd {nn.sgd_step(&sgd); nn.sgd_zero_grad(&sgd)} else {nn.adam_step(&adam); nn.adam_zero_grad(&adam)}
	}
	params = make([][]f64, len(ps), context.temp_allocator)
	for p, i in ps {params[i] = _clone(t.tensor_values(p))}
	if o.on_gpu {
		gpu.to_host(ps[:])
		gpu.to_host(x)
		gpu.to_host(y)
	}
	return
}

// skip: indices of parameters left out of the parameter comparison
_compare_runs :: proc(
	name: string,
	ref_l, got_l: []f64,
	ref_p, got_p: [][]f64,
	tol: f64,
	skip: []int = nil,
) {
	worst_l := _max_rel(got_l, ref_l)
	worst_p := 0.0
	outer: for p, i in got_p {
		for k in skip {if k == i {continue outer}}
		worst_p = max(worst_p, _max_rel(p, ref_p[i]))
	}
	check(worst_l <= tol, "%s: losses differ from the CPU by %.2e (> %.0e)", name, worst_l, tol)
	check(
		worst_p <= tol,
		"%s: parameters differ from the CPU by %.2e (> %.0e)",
		name,
		worst_p,
		tol,
	)
	fmt.printfln(
		"    %-40s loss %.5f -> %.5f, max rel diff to CPU: losses %.1e, params %.1e",
		name,
		got_l[0],
		got_l[len(got_l) - 1],
		worst_l,
		worst_p,
	)
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

gpu_mlp_training_test :: proc() {
	fmt.println("gpu_mlp_training_test: MLP + Adam trained on the GPU vs the CPU")
	sizes := []int{6, 32, 32, 2}
	for act in ([]nn.Activation{.ReLU, .Tanh, .Sigmoid, .LeakyReLU, .Softplus}) {
		x, y := _regression_data(128, 6, 2, 7)
		cpu_model := _mlp(sizes, act, 11)
		gpu_model := _mlp(sizes, act, 11)
		ref_l, ref_p := _train(cpu_model, x, y, {steps = 40})
		gpu.reset_stats()
		got_l, got_p := _train(gpu_model, x, y, {steps = 40, on_gpu = true})
		st := gpu.stats()
		fallbacks := 0
		for n in st.forward_fallbacks {fallbacks += n}
		for n in st.backward_fallbacks {fallbacks += n}
		_compare_runs(fmt.tprintf("%v", act), ref_l, got_l, ref_p, got_p, 2e-3)
		check(fallbacks == 0, "%v: %d CPU fallbacks in an all-kernel model", act, fallbacks)
		check(got_l[len(got_l) - 1] < got_l[0], "%v: the loss did not fall", act)
		check(
			st.auto_uploads == 0,
			"%v: %d automatic uploads (everything was moved explicitly)",
			act,
			st.auto_uploads,
		)
		nn.sequential_free(cpu_model)
		nn.sequential_free(gpu_model)
		t.tensor_free(x)
		t.tensor_free(y)
	}
}

gpu_all_fallback_training_test :: proc() {
	fmt.println(
		"gpu_all_fallback_training_test: every kernel disabled, the model still trains through the CPU fallback",
	)
	sizes := []int{5, 16, 1}
	x, y := _regression_data(64, 5, 1, 3)
	defer {t.tensor_free(x); t.tensor_free(y)}
	cpu_model := _mlp(sizes, .Tanh, 5)
	gpu_model := _mlp(sizes, .Tanh, 5)
	defer {nn.sequential_free(cpu_model); nn.sequential_free(gpu_model)}
	ref_l, ref_p := _train(cpu_model, x, y, {steps = 25})
	for op in t.Op {gpu.set_kernel_enabled(op, false)}
	gpu.reset_stats()
	got_l, got_p := _train(gpu_model, x, y, {steps = 25, on_gpu = true})
	st := gpu.stats()
	for op in t.Op {gpu.set_kernel_enabled(op, true)}
	_compare_runs("Tanh MLP, all ops on the fallback", ref_l, got_l, ref_p, got_p, 2e-3)
	// two linear layers: two matmuls per step
	check(
		st.forward_fallbacks[.MatMul] == 50,
		"matmul ran %d times on the fallback, want 50",
		st.forward_fallbacks[.MatMul],
	)
	check(st.backward_fallbacks[.MatMul] > 0, "matmul backward never used the fallback")
}

gpu_sgd_clip_test :: proc() {
	fmt.println(
		"gpu_sgd_clip_test: SGD (Nesterov momentum) with clip_grad_norm on the GPU vs the CPU",
	)
	sizes := []int{4, 24, 3}
	x, y := _regression_data(96, 4, 3, 9)
	defer {t.tensor_free(x); t.tensor_free(y)}
	for clip in ([]f64{0, 0.05}) {
		cpu_model := _mlp(sizes, .ReLU, 13)
		gpu_model := _mlp(sizes, .ReLU, 13)
		ref_l, ref_p := _train(cpu_model, x, y, {steps = 30, use_sgd = true, clip = clip})
		got_l, got_p := _train(
			gpu_model,
			x,
			y,
			{steps = 30, use_sgd = true, clip = clip, on_gpu = true},
		)
		_compare_runs(fmt.tprintf("SGD, clip %v", clip), ref_l, got_l, ref_p, got_p, 2e-3)
		nn.sequential_free(cpu_model)
		nn.sequential_free(gpu_model)
	}
	// graphs freed with tensor_free_graph instead of a graph scope
	cpu_model := _mlp(sizes, .ReLU, 13)
	gpu_model := _mlp(sizes, .ReLU, 13)
	defer {nn.sequential_free(cpu_model); nn.sequential_free(gpu_model)}
	ref_l, ref_p := _train(cpu_model, x, y, {steps = 10, no_scope = true})
	got_l, got_p := _train(gpu_model, x, y, {steps = 10, no_scope = true, on_gpu = true})
	_compare_runs("Adam, tensor_free_graph", ref_l, got_l, ref_p, got_p, 2e-3)
}

gpu_residency_test :: proc() {
	fmt.println("gpu_residency_test: host mirrors, sync accessors, auto-upload, to_host")
	m := l.matrix_new(f64, 2, 3)
	for &v, i in m.data {v = f64(i) - 2.5}
	x := t.tensor_new(m, true)
	gpu.to_device(x)
	check(t.tensor_is_on_device(x), "to_device: tensor not on the device")
	check(x.data.data == nil, "to_device: the host slice is not parked")
	v := t.tensor_values(x)
	check(len(v) == 6 && v[0] == -2.5 && v[5] == 2.5, "tensor_values: got %v", v)

	// a device op output has no host data until it is read
	y := t.tensor_scale(x, 2)
	check(y.data.data == nil && t.tensor_is_on_device(y), "device op output has host data")
	s := t.tensor_sum(y)
	check(t.tensor_item(s) == 0, "sum of the scaled values: %v, want 0", t.tensor_item(s))

	// writing a synced host copy: mark it dirty, the next device op sees it
	t.tensor_values(x)[0] = 10
	t.tensor_mark_host_dirty(x)
	s2 := t.tensor_sum(x)
	check(
		t.tensor_item(s2) == 12.5,
		"after mark_host_dirty the device sum is %v, want 12.5",
		t.tensor_item(s2),
	)

	// gradients: has_grad, values, drop
	t.tensor_backward(s2)
	check(t.tensor_has_grad(x), "no gradient after backward")
	check(x.grad.data == nil, "the device gradient has a host copy before it was read")
	g := t.tensor_grad_values(x)
	check(len(g) == 6 && g[0] == 1 && g[5] == 1, "gradient %v, want all 1", g)
	t.tensor_drop_grad(x)
	check(!t.tensor_has_grad(x), "drop_grad: still has a gradient")

	// detach stays on the device
	d := t.tensor_detach(x)
	check(
		t.tensor_is_on_device(d) && t.tensor_values(d)[0] == 10,
		"detach: not an equal device copy",
	)

	// a host tensor used by a device op is uploaded automatically
	gpu.reset_stats()
	h := t.tensor_new(l.matrix_new(f64, 2, 3))
	sum := t.tensor_add(x, h)
	check(gpu.stats().auto_uploads == 1, "auto uploads %d, want 1", gpu.stats().auto_uploads)
	check(t.tensor_is_on_device(sum), "the result of a mixed op is not on the device")

	// to_host brings everything back and releases the device buffers
	gpu.to_host(x)
	check(
		!t.tensor_is_on_device(x) && x.data.data != nil && x.data.data[0] == 10,
		"to_host: data not back on the host",
	)

	t.tensor_free_graph(sum, s, s2) // also frees y, an op node of s
	for z in ([]^t.Tensor{x, d, h}) {t.tensor_free(z)}
}

gpu_scope_memory_test :: proc() {
	fmt.println(
		"gpu_scope_memory_test: graph scopes recycle device buffers (pool size flat across steps)",
	)
	x, y := _regression_data(64, 8, 1, 21)
	defer {t.tensor_free(x); t.tensor_free(y)}
	model := _mlp({8, 64, 64, 1}, .ReLU, 3)
	defer nn.sequential_free(model)
	ps := nn.sequential_parameters(model, context.temp_allocator)
	adam := nn.adam_new(0.01)
	defer nn.adam_free(&adam)
	for p in ps {nn.adam_add_param(&adam, p)}
	gpu.to_device(ps[:])
	gpu.to_device(x)
	gpu.to_device(y)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	created := t.global_tensors_created - t.global_tensors_freed
	pool_after: [40]u64
	for step in 0 ..< 40 {
		t.graph_begin(&scope)
		loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
		t.tensor_backward(loss)
		_ = t.tensor_item(loss)
		t.graph_end(&scope)
		nn.adam_step(&adam)
		nn.adam_zero_grad(&adam)
		pool_after[step] = gpu.stats().pool_bytes
	}
	check(
		pool_after[39] == pool_after[2],
		"device pool grew from %d to %d bytes between steps 3 and 40",
		pool_after[2],
		pool_after[39],
	)
	check(
		t.global_tensors_created - t.global_tensors_freed == created,
		"tensors leaked by the scoped loop",
	)
	fmt.printfln(
		"    pool after step 3: %d KB, after step 40: %d KB",
		pool_after[2] >> 10,
		pool_after[39] >> 10,
	)
	gpu.to_host(ps[:])
	gpu.to_host(x)
	gpu.to_host(y)
}

gpu_checkpoint_test :: proc() {
	fmt.println("gpu_checkpoint_test: save_checkpoint of a GPU-trained model, load on the CPU")
	if !os.exists("model_checkpoints") {os.make_directory("model_checkpoints")}
	path := "model_checkpoints/gpu_checkpoint_test.ckpt"
	x, y := _regression_data(32, 3, 1, 4)
	defer {t.tensor_free(x); t.tensor_free(y)}
	model := _mlp({3, 8, 1}, .Tanh, 8)
	defer nn.sequential_free(model)
	ps := nn.sequential_parameters(model, context.temp_allocator)
	adam := nn.adam_new(0.01)
	defer nn.adam_free(&adam)
	for p in ps {nn.adam_add_param(&adam, p)}
	gpu.to_device(ps[:])
	for _ in 0 ..< 5 {
		loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
		t.tensor_backward(loss)
		t.tensor_free_graph(loss)
		nn.adam_step(&adam)
		nn.adam_zero_grad(&adam)
	}
	check(nn.save_checkpoint(model, &adam, path, 5), "save_checkpoint failed")
	loaded, opt, epoch, ok := nn.load_checkpoint(path)
	check(ok && epoch == 5, "load_checkpoint failed")
	if ok {
		lp := nn.sequential_parameters(loaded, context.temp_allocator)
		worst := 0.0
		for p, i in ps {worst = max(worst, _max_rel(lp[i].data.data, t.tensor_values(p)))}
		check(worst == 0, "loaded parameters differ from the device ones by %.2e", worst)
		// the moments were copied from the device: a CPU step from the loaded
		// state matches a GPU step from the live one
		m_worst := 0.0
		for m, i in opt.moment_1 {
			dev := make([]f64, len(m), context.temp_allocator)
			_ = t._device_hooks.optim_sync(raw_data(adam.moment_1[i]), dev, nil, false)
			m_worst = max(m_worst, _max_rel(m, dev))
		}
		check(m_worst == 0, "loaded Adam moments differ from the device ones by %.2e", m_worst)
		nn.sequential_free(loaded)
		nn.adam_free(opt)
		free(opt)
	}
	gpu.to_host(ps[:])
	os.remove(path)
}

// Not a check: step times on the CPU and the GPU for a small and a large MLP
gpu_speed_report :: proc() {
	fmt.println(
		"gpu_speed_report: Adam training step, CPU vs GPU (mean of 10 steps after 2 warm-up steps)",
	)
	Config :: struct {
		sizes: []int,
		batch: int,
	}
	configs := []Config {
		{{16, 64, 64, 1}, 64},
		{{64, 256, 256, 1}, 256},
		{{256, 1024, 1024, 1}, 512},
		{{512, 2048, 2048, 1}, 1024},
	}
	for c in configs {
		ms: [2]f64
		for on_gpu, k in ([]bool{false, true}) {
			x, y := _regression_data(c.batch, c.sizes[0], 1, 1)
			model := _mlp(c.sizes, .ReLU, 1)
			_train(model, x, y, {steps = 2, on_gpu = on_gpu})
			start := time.now()
			_train(model, x, y, {steps = 10, on_gpu = on_gpu})
			ms[k] = time.duration_milliseconds(time.since(start)) / 10
			nn.sequential_free(model)
			t.tensor_free(x)
			t.tensor_free(y)
		}
		fmt.printfln(
			"    sizes %v, batch %4d: CPU %8.2f ms   GPU %7.2f ms   (%.1fx)",
			c.sizes,
			c.batch,
			ms[0],
			ms[1],
			ms[0] / ms[1],
		)
	}
}
