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
	// two linear layers: two linear ops per step
	check(
		st.forward_fallbacks[.Linear] == 50,
		"linear ran %d times on the fallback, want 50",
		st.forward_fallbacks[.Linear],
	)
	check(st.backward_fallbacks[.Linear] > 0, "linear backward never used the fallback")
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
	check(x.data.data == nil, "to_device: the host copy was kept")
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
			_ = t._device_hooks.optim_sync(adam.device_keys[i], dev, nil, false)
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

// Gradient bookkeeping and Adam on the GPU vs the CPU: two backward passes
// into the same gradients (the second adds to the first), zeroed gradients
// read back, and Adam with parameters that differ in learning rate and in step
// count (one parameter's gradient is dropped in some steps), so the device
// updates parameters in groups of different makeup
gpu_grad_accumulation_adam_test :: proc() {
	fmt.println(
		"gpu_grad_accumulation_adam_test: accumulated and zeroed gradients, Adam with mixed learning rates and step counts",
	)
	Result :: struct {
		grads:  [][]f64,
		zeroed: f64,
		losses: []f64,
		params: [][]f64,
	}
	run :: proc(on_gpu: bool) -> (r: Result) {
		x, y := _regression_data(32, 5, 2, 21)
		defer {t.tensor_free(x); t.tensor_free(y)}
		model := _mlp({5, 12, 12, 2}, .Tanh, 17)
		defer nn.sequential_free(model)
		ps := nn.sequential_parameters(model, context.temp_allocator)
		if on_gpu {
			gpu.to_device(ps[:])
			gpu.to_device(x)
			gpu.to_device(y)
		}
		// two backward passes, no zeroing in between
		for _ in 0 ..< 2 {
			loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
		}
		r.grads = make([][]f64, len(ps), context.temp_allocator)
		for p, i in ps {r.grads[i] = _clone(t.tensor_grad_values(p))}
		// zeroed gradients read back as zeros
		for p in ps {t.tensor_zero_grad(p)}
		for p in ps {for v in t.tensor_grad_values(p) {r.zeroed = max(r.zeroed, abs(v))}}
		// Adam: the last layer with its own learning rate; the first weight's
		// gradient dropped after the backward pass of steps 2 and 5
		opt := nn.adam_new(0.01)
		defer nn.adam_free(&opt)
		for p, i in ps {
			if i <
			   len(ps) -
				   2 {nn.adam_add_param(&opt, p)} else {nn.adam_add_param_with_lr(&opt, p, 0.004)}
		}
		r.losses = make([]f64, 8, context.temp_allocator)
		for step in 0 ..< 8 {
			loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
			t.tensor_backward(loss)
			r.losses[step] = t.tensor_item(loss)
			t.tensor_free_graph(loss)
			if step == 2 || step == 5 {t.tensor_drop_grad(ps[0])}
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)
		}
		r.params = make([][]f64, len(ps), context.temp_allocator)
		for p, i in ps {r.params[i] = _clone(t.tensor_values(p))}
		if on_gpu {gpu.to_host(ps[:])}
		return
	}
	ref := run(false)
	got := run(true)
	worst_g := 0.0
	for g, i in got.grads {worst_g = max(worst_g, _max_rel(g, ref.grads[i]))}
	check(worst_g <= 1e-4, "accumulated gradients differ from the CPU by %.2e (> 1e-4)", worst_g)
	check(got.zeroed == 0, "a zeroed gradient read back as %v", got.zeroed)
	_compare_runs(
		"Adam, mixed learning rates and step counts",
		ref.losses,
		got.losses,
		ref.params,
		got.params,
		2e-3,
	)
	fmt.printfln("    accumulated gradients: max rel diff to CPU %.1e", worst_g)
}

// Not a check: the time of one Adam training step on the CPU and the GPU, for
// MLPs of growing size and for the GPT of gpu_gpt_training_test (without
// dropout and gradient clipping). Only the step loop is timed, after warm-up
// steps. The GPU runs twice: reading the loss after every step (each read
// waits until the GPU has finished the step) and reading it once at the end.
gpu_speed_report :: proc() {
	fmt.println(
		"gpu_speed_report: Adam training step in ms, CPU vs GPU (GPU: loss read every step / once at the end)",
	)
	Config :: struct {
		sizes: []int,
		batch: int,
		steps: int,
	}
	configs := []Config {
		{{16, 64, 64, 1}, 64, 200},
		{{64, 256, 256, 1}, 256, 50},
		{{256, 1024, 1024, 1}, 512, 10},
		{{512, 2048, 2048, 1}, 1024, 4},
	}
	for c in configs {
		cpu := _mlp_step_ms(c.sizes, c.batch, c.steps, false, true)
		each := _mlp_step_ms(c.sizes, c.batch, c.steps, true, true)
		once := _mlp_step_ms(c.sizes, c.batch, c.steps, true, false)
		fmt.printfln(
			"    MLP %-20s batch %4d: CPU %8.3f   GPU %7.3f / %7.3f   (%.1fx / %.1fx)",
			fmt.tprint(c.sizes),
			c.batch,
			cpu,
			each,
			once,
			cpu / each,
			cpu / once,
		)
	}
	cpu := _gpt_step_ms(100, false, true)
	each := _gpt_step_ms(100, true, true)
	once := _gpt_step_ms(100, true, false)
	fmt.printfln(
		"    GPT (2 blocks, d 32, %d x %d tokens): CPU %8.3f   GPU %7.3f / %7.3f   (%.1fx / %.1fx)",
		TF_BATCH,
		TF_SEQ,
		cpu,
		each,
		once,
		cpu / each,
		cpu / once,
	)
}

// _mlp_step_ms: mean time of `steps` Adam steps of a ReLU MLP (after 3 warm-up
// steps); read_each: the loss is read after every step
_mlp_step_ms :: proc(sizes: []int, batch, steps: int, on_gpu, read_each: bool) -> f64 {
	x, y := _regression_data(batch, sizes[0], 1, 1)
	defer {t.tensor_free(x); t.tensor_free(y)}
	model := _mlp(sizes, .ReLU, 1)
	defer nn.sequential_free(model)
	ps := nn.sequential_parameters(model, context.temp_allocator)
	opt := nn.adam_new(0.001)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	if on_gpu {
		gpu.to_device(ps[:])
		gpu.to_device(x)
		gpu.to_device(y)
	}
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	run :: proc(
		model: ^nn.Sequential,
		opt: ^nn.Adam,
		x, y: ^t.Tensor,
		scope: ^t.Graph_Scope,
		n: int,
		read_each: bool,
	) {
		for _ in 0 ..< n {
			t.graph_begin(scope)
			loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
			t.tensor_backward(loss)
			if read_each {_ = t.tensor_item(loss)}
			t.graph_end(scope)
			nn.adam_step(opt)
			nn.adam_zero_grad(opt)
		}
	}
	run(model, &opt, x, y, &scope, 3, read_each)
	if on_gpu {gpu.synchronize()}
	start := time.now()
	run(model, &opt, x, y, &scope, steps, read_each)
	if on_gpu {gpu.synchronize()} 	// the last steps' work, as one final read would
	ms := time.duration_milliseconds(time.since(start)) / f64(steps)
	if on_gpu {
		gpu.to_host(ps[:])
		gpu.to_host(x)
		gpu.to_host(y)
	}
	return ms
}

// _gpt_step_ms: as _mlp_step_ms, for the GPT of gpu_gpt_training_test
_gpt_step_ms :: proc(steps: int, on_gpu, read_each: bool) -> f64 {
	rand.reset(99)
	model := nn.gpt_model_new(TF_VOCAB, 32, 4, 64, 2, TF_SEQ)
	defer nn.gpt_model_free(&model)
	ps := nn.gpt_model_parameters(&model, context.temp_allocator)
	opt := nn.adam_new(0.003)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	if on_gpu {gpu.to_device(ps[:])}
	mask := nn.create_causal_mask(TF_SEQ, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	run :: proc(
		model: ^nn.GPTModel,
		opt: ^nn.Adam,
		mask: []f64,
		scope: ^t.Graph_Scope,
		n: int,
		read_each: bool,
	) {
		for step in 0 ..< n {
			ids, targets := _token_batch(step % 8)
			t.graph_begin(scope)
			loss := t.tensor_cross_entropy_loss(
				nn.gpt_model_forward(model, ids, mask, true),
				targets,
			)
			t.tensor_backward(loss)
			if read_each {_ = t.tensor_item(loss)}
			t.graph_end(scope)
			nn.adam_step(opt)
			nn.adam_zero_grad(opt)
			t.tensor_free(ids)
		}
	}
	run(&model, &opt, mask, &scope, 3, read_each)
	if on_gpu {gpu.synchronize()}
	start := time.now()
	run(&model, &opt, mask, &scope, steps, read_each)
	if on_gpu {gpu.synchronize()}
	ms := time.duration_milliseconds(time.since(start)) / f64(steps)
	if on_gpu {gpu.to_host(ps[:])}
	return ms
}

// _param_with_grad: a [1, n] parameter whose gradient is `grad`, set by a
// backward pass (d/dp Σ p ⊙ c = c); on the device if `on_device`
_param_with_grad :: proc(grad: []f64, on_device: bool) -> ^t.Tensor {
	p := t.tensor_new(l.matrix_new(f64, 1, len(grad)), true)
	cm := l.matrix_new(f64, 1, len(grad))
	copy(cm.data, grad)
	c := t.tensor_new(cm)
	if on_device {gpu.to_device(p); gpu.to_device(c)}
	loss := t.tensor_sum(t.tensor_mul(p, c))
	t.tensor_backward(loss)
	t.tensor_free_graph(loss)
	t.tensor_free(c)
	return p
}

gpu_clip_grad_norm_test :: proc() {
	fmt.println(
		"gpu_clip_grad_norm_test: clip_grad_norm over 18 gradients (grouped kernels) vs the CPU",
	)
	// sizes from 1 element to several hundred workgroups
	sizes := []int{3, 1, 17, 256, 257, 5, 70000, 9, 33, 2, 1000, 4, 6, 7, 300, 11, 513}
	rand.reset(21)
	grads := make([][]f64, len(sizes), context.temp_allocator)
	for n, i in sizes {
		grads[i] = make([]f64, n, context.temp_allocator)
		for &v in grads[i] {v = f64(f32(2 * rand.float64() - 1))}
	}
	for max_norm in ([]f64{5, 1e6}) {
		cpu := make([dynamic]^t.Tensor, context.temp_allocator)
		dev := make([dynamic]^t.Tensor, context.temp_allocator)
		for g, i in grads {
			append(&cpu, _param_with_grad(g, false))
			append(&dev, _param_with_grad(g, i != 0)) // parameter 0 stays on the host
		}
		// parameter 1: its gradient was read and marked as changed on the host
		_ = t.tensor_grad_values(dev[1])
		t.tensor_mark_grad_host_dirty(dev[1])
		// parameter 4 is listed twice: counted and scaled twice, as on the CPU
		inject_at(&cpu, 5, cpu[4])
		inject_at(&dev, 5, dev[4])
		ref := nn.clip_grad_norm(cpu[:], max_norm)
		gpu.reset_stats()
		got := nn.clip_grad_norm(dev[:], max_norm)
		launches := gpu.stats().dispatches
		worst := 0.0
		for x, i in cpu {
			worst = max(worst, _max_rel(t.tensor_grad_values(dev[i]), t.tensor_grad_values(x)))
		}
		clipped := max_norm < ref
		fmt.printfln(
			"    max_norm %-8v norm %.6f (CPU %.6f), %d launches, max rel diff of the gradients %.1e",
			max_norm,
			got,
			ref,
			launches,
			worst,
		)
		check(abs(got - ref) <= 1e-6 * ref, "clip_grad_norm: norm %v, CPU %v", got, ref)
		check(worst < 1e-6, "clip_grad_norm: gradients differ from the CPU by %v", worst)
		// 16 gradients on the device (one listed twice): 3 launches for the sum,
		// and when clipping 3 to scale them (a group ends before the repeat)
		want := 3 + (3 if clipped else 0)
		check(launches == want, "clip_grad_norm: %d launches, want %d", launches, want)
		ordered_remove(&cpu, 5)
		ordered_remove(&dev, 5)
		for x in cpu {t.tensor_free(x)}
		for x in dev {t.tensor_free(x)}
	}
}
