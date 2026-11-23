package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:time"

// tensor_dropout with set_dropout_masks(.Counter) on the GPU: the kernel makes
// the same mask as the CPU (same zeros, output and gradient to f32 rounding)
// without uploading anything, a CPU backward case downloads the device mask,
// and the time of dropout-heavy steps with Stream (masks drawn on the host and
// uploaded) and Counter masks (printed, not checked)
gpu_dropout_masks_test :: proc() {
	fmt.println(
		"gpu_dropout_masks_test: Counter dropout masks made on the GPU equal the CPU's, with no upload",
	)
	previous := t.set_dropout_masks(.Counter)
	defer t.set_dropout_masks(previous)
	n :: 100_003
	p :: 0.25
	make_x :: proc() -> ^t.Tensor {
		x := t.tensor_new(l.matrix_new(f64, 1, n), true)
		for &v, i in x.data.data {v = f64(f32(math.sin(f64(i)) + 2))}
		return x
	}
	xc, xg := make_x(), make_x()
	defer {t.tensor_free(xc); t.tensor_free(xg)}
	gpu.to_device(xg)

	rand.reset(5)
	yc := t.tensor_dropout(xc, p, true)
	lc := t.tensor_sum(yc)
	t.tensor_backward(lc)
	rand.reset(5)
	up := gpu.stats().bytes_up
	yg := t.tensor_dropout(xg, p, true)
	uploaded := gpu.stats().bytes_up - up
	lg := t.tensor_sum(yg)
	t.tensor_backward(lg)
	out_g, grad_g := t.tensor_values(yg), t.tensor_grad_values(xg)
	same_zeros := true
	for v, i in yc.data.data {same_zeros = same_zeros && ((v == 0) == (out_g[i] == 0))}
	out_err := _max_rel(out_g, yc.data.data)
	grad_err := _max_rel(grad_g, xc.grad.data)
	check(same_zeros, "the GPU dropped other elements than the CPU")
	check(
		out_err <= 1e-6 && grad_err <= 1e-6,
		"dropout output %.1e / gradient %.1e differ from the CPU",
		out_err,
		grad_err,
	)
	check(uploaded == 0, "the Counter mask uploaded %d bytes", uploaded)
	fmt.printfln(
		"    %d elements, p = %.2f: same zeros as the CPU %v, output %.1e, gradient %.1e, bytes uploaded %d",
		n,
		p,
		same_zeros,
		out_err,
		grad_err,
		uploaded,
	)
	t.tensor_free_graph(lc)
	t.tensor_free_graph(lg)
	t.tensor_drop_grad(xc)
	t.tensor_drop_grad(xg)

	// the backward on the CPU (kernel disabled after the forward): the node
	// downloads the device mask
	rand.reset(5)
	yg = t.tensor_dropout(xg, p, true)
	lg = t.tensor_sum(yg)
	fallbacks := gpu.stats().backward_fallbacks[.Dropout]
	gpu.set_kernel_enabled(.Dropout, false)
	t.tensor_backward(lg)
	gpu.set_kernel_enabled(.Dropout, true)
	check(
		gpu.stats().backward_fallbacks[.Dropout] > fallbacks,
		"the dropout backward did not run on the CPU",
	)
	t.tensor_free_graph(lg)
	rand.reset(5)
	yc = t.tensor_dropout(xc, p, true)
	lc = t.tensor_sum(yc)
	t.tensor_backward(lc)
	cpu_bwd_err := _max_rel(t.tensor_grad_values(xg), xc.grad.data)
	check(
		cpu_bwd_err <= 1e-6,
		"CPU backward of a GPU dropout differs from the CPU by %.1e",
		cpu_bwd_err,
	)
	t.tensor_free_graph(lc)
	gpu.to_host(xg)

	// time: 17 dropout sites on [256, 1024] per step, forward and backward
	h := t.tensor_new(l.matrix_new(f64, 256, 1024), true)
	defer t.tensor_free(h)
	gpu.to_device(h)
	step :: proc(h: ^t.Tensor) {
		z := h
		for _ in 0 ..< 17 {z = t.tensor_dropout(z, 0.1, true)}
		loss := t.tensor_sum(z)
		t.tensor_backward(loss)
		_ = t.tensor_item(loss)
		t.tensor_free_graph(loss)
	}
	ms: [2]f64
	for round in 0 ..< 3 {
		for m, k in ([]t.Dropout_Masks{.Stream, .Counter}) {
			t.set_dropout_masks(m)
			step(h)
			start := time.tick_now()
			for _ in 0 ..< 5 {step(h)}
			if round > 0 {ms[k] += time.duration_milliseconds(time.tick_since(start)) / 10}
		}
	}
	fmt.printfln(
		"    17 dropout sites on [256, 1024], forward + backward: Stream %.2f ms, Counter %.2f ms per step",
		ms[0],
		ms[1],
	)
	gpu.to_host(h)
}
