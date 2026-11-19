package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:time"

// Convolution, pooling and batch-norm models on the GPU against the CPU

// the op cases whose forward runs on the GPU and whose backward runs on the CPU
// (the device forward keeps the batch-norm and layer-norm statistics and the
// Sharpe mean and std on the device, and no max-pool argmax; the CPU backward
// must fetch, recompute or find them. Multi-head attention is one node on the
// GPU, whose CPU backward runs the attention backward on head-layout copies.)
gpu_device_forward_cpu_backward_test :: proc() {
	fmt.println(
		"gpu_device_forward_cpu_backward_test: GPU forward, CPU backward (pooling, norms, conv, Sharpe, multi-head attention, GELU, LSTM, GRU, RNN, SSM)",
	)
	rand.reset(3)
	for c in op_cases {
		#partial switch c.op {
		case .MaxPool2d,
		     .AvgPool2d,
		     .BatchNorm2d,
		     .LayerNorm,
		     .RMSNorm,
		     .Conv2d,
		     .Conv1d,
		     .SharpeLoss,
		     .MultiHeadAttention,
		     .Gelu,
		     .LSTM,
		     .GRU,
		     .RNN,
		     .SSM:
		case:
			continue
		}
		worst, fallbacks := _compare_case(c, false, cpu_backward = true)
		check(
			worst <= c.tol,
			"%s (CPU backward): max relative error %.2e > %.0e",
			c.name,
			worst,
			c.tol,
		)
		check(fallbacks > 0, "%s (CPU backward): the CPU backward did not run", c.name)
		fmt.printfln("    %-34s max rel err %.2e", c.name, worst)
	}
}

// running statistics: training-mode steps update them on the device exactly as
// on the CPU; evaluation mode then normalizes with them
gpu_batch_norm_running_stats_test :: proc() {
	fmt.println(
		"gpu_batch_norm_running_stats_test: running mean / variance updated on the GPU vs the CPU",
	)
	run :: proc(on_gpu: bool) -> (rm, rv, eval_out: []f64) {
		bn := nn.batch_norm_2d_layer_new(4, momentum = 0.2)
		defer nn.batch_norm_2d_layer_free(&bn)
		params := []^t.Tensor{bn.weight, bn.bias}
		if on_gpu {gpu.to_device(params)}
		scope: t.Graph_Scope
		defer t.graph_scope_destroy(&scope)
		rand.reset(21)
		for step in 0 ..< 4 {
			m := l.matrix_new(f64, 1, 5 * 4 * 3 * 3)
			for &v in m.data {v = f64(f32(3 * rand.float64() + f64(step)))}
			x := t.tensor_new(m)
			x.shape = {5, 4, 3, 3}
			t.graph_begin(&scope)
			train := step < 3
			y := t.tensor_batch_norm_2d(
				x,
				bn.weight,
				bn.bias,
				bn.running_mean,
				bn.running_var,
				train,
				bn.momentum,
				bn.eps,
			)
			if !train {eval_out = _clone(t.tensor_values(y))}
			t.graph_end(&scope)
			t.tensor_free(x)
		}
		rm = _clone(t.tensor_values(bn.running_mean))
		rv = _clone(t.tensor_values(bn.running_var))
		if on_gpu {gpu.to_host(params)}
		return
	}
	ref_m, ref_v, ref_e := run(false)
	gpu.reset_stats()
	got_m, got_v, got_e := run(true)
	st := gpu.stats()
	err := max(_max_rel(got_m, ref_m), _max_rel(got_v, ref_v), _max_rel(got_e, ref_e))
	check(err <= 1e-5, "running statistics differ from the CPU by %.2e", err)
	check(st.forward_fallbacks[.BatchNorm2d] == 0, "batch norm fell back to the CPU")
	fmt.printfln(
		"    running mean / var and eval output: max rel diff %.1e (running stats uploaded automatically: %d)",
		err,
		st.auto_uploads,
	)
}

// ----------------------------------------------------------------------------
// Training: a small CNN and a TCN
// ----------------------------------------------------------------------------

CNN_N :: 8
CNN_STEPS :: 10

// The first convolution has no bias: batch norm subtracts every per-channel
// constant, so such a bias has a gradient of exactly zero. In f32 the rounding
// residue (~1e-8) is above Adam's epsilon and Adam turns it into steps of the
// learning rate: the bias (and the running mean that tracks it) would drift on
// the GPU without changing any output.
_cnn_model :: proc() -> ^nn.Sequential {
	rand.reset(31)
	m := nn.sequential_new()
	nn.sequential_add(
		m,
		nn.conv2d_layer_new(1, 6, 3, 1, 1, use_bias = false),
		nn.batch_norm_2d_layer_new(6),
		nn.Activation.ReLU,
		nn.maxpool2d_layer_new(2),
		nn.conv2d_layer_new(6, 8, 3),
		nn.Activation.ReLU,
		nn.avgpool2d_layer_new(2),
		nn.FlattenLayer{},
		nn.dropout_layer_new(0.2),
		nn.linear_layer_new(32, 3),
	)
	_round_params(nn.sequential_parameters(m, context.temp_allocator)[:])
	return m
}

// 12x12 single-channel images of three patterns (bars, columns, a diagonal)
_cnn_batch :: proc() -> (x: ^t.Tensor, labels: []int) {
	rand.reset(32)
	m := l.matrix_new(f64, 1, CNN_N * 144)
	labels = make([]int, CNN_N, context.temp_allocator)
	for n in 0 ..< CNN_N {
		labels[n] = n % 3
		for i in 0 ..< 12 {
			for j in 0 ..< 12 {
				on :=
					(labels[n] == 0 && i % 4 == 0) ||
					(labels[n] == 1 && j % 4 == 0) ||
					(labels[n] == 2 && i == j)
				v := (1.0 if on else 0.0) + 0.2 * rand.float64()
				m.data[n * 144 + i * 12 + j] = f64(f32(v))
			}
		}
	}
	x = t.tensor_new(m)
	x.shape = {CNN_N, 1, 12, 12}
	return
}

_cnn_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, ms: f64) {
	model := _cnn_model()
	defer nn.sequential_free(model)
	ps := nn.sequential_parameters(model, context.temp_allocator)
	opt := nn.adam_new(0.01)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	x, labels := _cnn_batch()
	defer t.tensor_free(x)
	if on_gpu {
		gpu.to_device(ps[:])
		gpu.to_device(x)
	}
	losses = make([]f64, CNN_STEPS, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	start := time.now()
	for step in 0 ..< CNN_STEPS {
		rand.reset(u64(700 + step)) // the same dropout masks on both runs
		t.graph_begin(&scope)
		loss := t.tensor_cross_entropy_loss(nn.sequential_forward(model, x), labels)
		t.tensor_backward(loss)
		losses[step] = t.tensor_item(loss)
		t.graph_end(&scope)
		nn.adam_step(&opt)
		nn.adam_zero_grad(&opt)
	}
	ms = time.duration_milliseconds(time.since(start)) / CNN_STEPS
	// parameters, then the batch norm's running statistics
	bn := model.layers[1].(nn.BatchNorm2dLayer)
	params = make([][]f64, len(ps) + 2, context.temp_allocator)
	for p, i in ps {params[i] = _clone(t.tensor_values(p))}
	params[len(ps)] = _clone(t.tensor_values(bn.running_mean))
	params[len(ps) + 1] = _clone(t.tensor_values(bn.running_var))
	if on_gpu {gpu.to_host(ps[:])}
	return
}

gpu_cnn_training_test :: proc() {
	fmt.println(
		"gpu_cnn_training_test: a CNN (conv, batch norm, max / avg pool, flatten, dropout) on the GPU vs the CPU",
	)
	ref_l, ref_p, cpu_ms := _cnn_run(false)
	gpu.reset_stats()
	got_l, got_p, gpu_ms := _cnn_run(true)
	n_fb, fb_ops := _fallback_ops()
	_compare_runs("CNN (params + running stats)", ref_l, got_l, ref_p, got_p, 2e-3)
	check(got_l[CNN_STEPS - 1] < got_l[0], "CNN: the loss did not fall")
	check(n_fb == 0, "CNN: %d CPU fallbacks (%s)", n_fb, fb_ops)
	fmt.printfln("        step time CPU %.1f ms, GPU %.1f ms", cpu_ms, gpu_ms)
}

TCN_B :: 4
TCN_T :: 16
TCN_STEPS :: 10

_tcn_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, ms: f64) {
	rand.reset(41)
	model := nn.sequential_new()
	defer nn.sequential_free(model)
	nn.sequential_add_tcn(model, 3, 8, levels = 2, kernel_size = 3, dropout = 0.1)
	nn.sequential_add(model, nn.linear_layer_new(8, 1))
	ps := nn.sequential_parameters(model, context.temp_allocator)
	_round_params(ps[:])
	opt := nn.adam_new(0.01)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}

	// three input series; the target is a delayed mix of them
	rand.reset(42)
	xm := l.matrix_new(f64, 1, TCN_B * TCN_T * 3)
	ym := l.matrix_new(f64, 1, TCN_B * TCN_T)
	for &v in xm.data {v = f64(f32(2 * rand.float64() - 1))}
	for b in 0 ..< TCN_B {
		for s in 0 ..< TCN_T {
			r := (b * TCN_T + max(s - 2, 0)) * 3
			ym.data[b * TCN_T + s] = f64(f32(0.5 * xm.data[r] - 0.3 * xm.data[r + 2]))
		}
	}
	x, y := t.tensor_new(xm), t.tensor_new(ym)
	x.shape = {TCN_B, TCN_T, 3, 1}
	defer t.tensor_free(x)
	defer t.tensor_free(y)
	if on_gpu {
		gpu.to_device(ps[:])
		gpu.to_device(x)
		gpu.to_device(y)
	}
	losses = make([]f64, TCN_STEPS, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	start := time.now()
	for step in 0 ..< TCN_STEPS {
		rand.reset(u64(800 + step))
		t.graph_begin(&scope)
		pred := nn.sequential_forward(model, x)
		loss := t.tensor_mse_loss(t.tensor_reshape(pred, y.shape), y)
		t.tensor_backward(loss)
		losses[step] = t.tensor_item(loss)
		t.graph_end(&scope)
		nn.adam_step(&opt)
		nn.adam_zero_grad(&opt)
	}
	ms = time.duration_milliseconds(time.since(start)) / TCN_STEPS
	params = make([][]f64, len(ps), context.temp_allocator)
	for p, i in ps {params[i] = _clone(t.tensor_values(p))}
	if on_gpu {gpu.to_host(ps[:])}
	return
}

gpu_tcn_training_test :: proc() {
	fmt.println(
		"gpu_tcn_training_test: a TCN (dilated causal conv1d, dropout, residual) on the GPU vs the CPU",
	)
	ref_l, ref_p, cpu_ms := _tcn_run(false)
	gpu.reset_stats()
	got_l, got_p, gpu_ms := _tcn_run(true)
	n_fb, fb_ops := _fallback_ops()
	_compare_runs("TCN", ref_l, got_l, ref_p, got_p, 2e-3)
	check(got_l[TCN_STEPS - 1] < got_l[0], "TCN: the loss did not fall")
	check(n_fb == 0, "TCN: %d CPU fallbacks (%s)", n_fb, fb_ops)
	fmt.printfln("        step time CPU %.1f ms, GPU %.1f ms", cpu_ms, gpu_ms)
}
