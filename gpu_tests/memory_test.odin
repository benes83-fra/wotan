package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

// GPU training keeps no f64 host copies: parameters, gradients, Adam moments
// and SGD velocity live only on the device (host slices nil, the host memory
// of model and optimizer shrinks by at least 24 bytes per parameter with Adam,
// 8 with SGD). save_checkpoint brings the moments over for the file and frees
// them again. After to_host, a CPU step continues from the device state: three
// GPU steps, to_host and one CPU step end where four GPU steps end (lost
// moments or velocity would move the parameters by about the learning rate).
gpu_host_copies_test :: proc() {
	fmt.println(
		"gpu_host_copies_test: no host copies of parameters, gradients, Adam moments and SGD velocity during GPU training",
	)
	if !os.exists("model_checkpoints") {os.make_directory("model_checkpoints")}
	path := "model_checkpoints/gpu_host_copies_test.ckpt"
	x, y := _regression_data(64, 16, 1, 31)
	defer {t.tensor_free(x); t.tensor_free(y)}
	for use_sgd in ([]bool{false, true}) {
		name := "SGD with momentum" if use_sgd else "Adam"
		final: [2][][]f64 // 4 GPU steps; 3 GPU steps, to_host, 1 CPU step
		for split, r in ([]bool{false, true}) {
			track: mem.Tracking_Allocator
			mem.tracking_allocator_init(&track, context.allocator)
			defer mem.tracking_allocator_destroy(&track)
			context.allocator = mem.tracking_allocator(&track)

			model := _mlp({16, 128, 128, 1}, .Tanh, 9)
			ps := nn.sequential_parameters(model, context.temp_allocator)
			n_params := 0
			for p in ps {n_params += t.tensor_numel(p)}
			adam := nn.adam_new(0.01)
			sgd := nn.sgd_new(0.05, momentum = 0.9)
			for p in ps {
				if use_sgd {nn.sgd_add_param(&sgd, p)} else {nn.adam_add_param(&adam, p)}
			}
			host_before := track.current_memory_allocated
			gpu.to_device(ps[:])
			gpu.to_device(x)
			gpu.to_device(y)
			scope: t.Graph_Scope
			step :: proc(
				model: ^nn.Sequential,
				x, y: ^t.Tensor,
				scope: ^t.Graph_Scope,
				adam: ^nn.Adam,
				sgd: ^nn.SGD,
				use_sgd: bool,
			) {
				t.graph_begin(scope)
				t.tensor_backward(t.tensor_mse_loss(nn.sequential_forward(model, x), y))
				t.graph_end(scope)
				if use_sgd {nn.sgd_step(sgd); nn.sgd_zero_grad(sgd)} else {nn.adam_step(adam); nn.adam_zero_grad(adam)}
			}
			for _ in 0 ..< 3 {step(model, x, y, &scope, &adam, &sgd, use_sgd)}

			if !split {
				// the host state after three device steps
				none := true
				for p, i in ps {
					none &&= p.data.data == nil && p.grad.data == nil
					if use_sgd {none &&= sgd.velocity[i] == nil} else {none &&= adam.moment_1[i] == nil && adam.moment_2[i] == nil}
				}
				host_after := track.current_memory_allocated
				want := (8 if use_sgd else 24) * n_params
				check(none, "%s: host copies left after device steps", name)
				check(
					host_before - host_after >= i64(want),
					"%s: host memory of model and optimizer fell by %d bytes, want at least %d (%d parameters)",
					name,
					host_before - host_after,
					want,
					n_params,
				)
				fmt.printfln(
					"    %-18s host memory of model and optimizer %.2f MB -> %.2f MB (%d parameters)",
					name,
					f64(host_before) / 1e6,
					f64(host_after) / 1e6,
					n_params,
				)
				if !use_sgd {
					// a checkpoint brings the moments over and frees them again
					check(nn.save_checkpoint(model, &adam, path, 3), "save_checkpoint failed")
					freed := true
					for m in adam.moment_1 {freed &&= m == nil}
					check(freed, "Adam: save_checkpoint left host moments")
					loaded, lopt, _, ok := nn.load_checkpoint(path)
					check(ok, "load_checkpoint failed")
					if ok {
						worst := 0.0
						for m, i in lopt.moment_1 {
							dev := make([]f64, len(m), context.temp_allocator)
							_ = t._device_hooks.optim_sync(adam.device_keys[i], dev, nil, false)
							worst = max(worst, _max_rel(m, dev))
						}
						check(
							worst == 0,
							"Adam: saved moments differ from the device ones by %.2e",
							worst,
						)
						nn.sequential_free(loaded)
						nn.adam_free(lopt)
						free(lopt)
					}
					os.remove(path)
				}
				step(model, x, y, &scope, &adam, &sgd, use_sgd)
				gpu.to_host(ps[:])
				gpu.to_host(x)
				gpu.to_host(y)
			} else {
				gpu.to_host(ps[:])
				gpu.to_host(x)
				gpu.to_host(y)
				step(model, x, y, &scope, &adam, &sgd, use_sgd) // on the CPU, from the device state
			}
			final[r] = make([][]f64, len(ps), context.temp_allocator)
			for p, i in ps {final[r][i] = _clone(p.data.data)}
			t.graph_scope_destroy(&scope)
			nn.adam_free(&adam)
			nn.sgd_free(&sgd)
			nn.sequential_free(model)
		}
		worst := 0.0
		for p, i in final[1] {worst = max(worst, _max_rel(p, final[0][i]))}
		check(
			worst <= 1e-4,
			"%s: 3 GPU steps + to_host + 1 CPU step differ from 4 GPU steps by %.2e",
			name,
			worst,
		)
		fmt.printfln(
			"    %-18s 3 GPU steps + 1 CPU step vs 4 GPU steps: max diff %.1e",
			name,
			worst,
		)
	}
}

// The buffer pool: a request is rounded up to its size class (at most a
// quarter above 4 KB), and when shapes change every step the pool gives back
// long-unused buffers instead of keeping a buffer of every size it saw: it
// never holds more than 1.25 times the peak in use. Training with a different
// number of rows each step matches the CPU while buffers are given back.
gpu_pool_test :: proc() {
	fmt.println(
		"gpu_pool_test: size classes, buffers given back when shapes change, results unchanged",
	)
	gpu.release_cached()
	// size classes, seen in the memory in use
	sizes := [?][2]int {
		{1000, 4096},
		{300_000, 1_310_720},
		{1 << 20, 4 << 20},
		{1_100_000, 5_242_880},
	}
	for s in sizes {
		live := gpu.stats().live_bytes
		a := t.tensor_new(l.matrix_new(f64, 1, s[0]))
		gpu.to_device(a)
		got := gpu.stats().live_bytes - live
		check(
			got == u64(s[1]),
			"a tensor of %d floats occupies %d bytes, want %d",
			s[0],
			got,
			s[1],
		)
		t.tensor_free(a)
	}

	// training with 500 to 8000 rows per step, on the CPU and the GPU
	cpu_model := _mlp({32, 256, 32}, .ReLU, 17)
	gpu_model := _mlp({32, 256, 32}, .ReLU, 17)
	defer {nn.sequential_free(cpu_model); nn.sequential_free(gpu_model)}
	models := [2]^nn.Sequential{cpu_model, gpu_model}
	opts: [2]nn.Adam
	for &o, k in opts {
		o = nn.adam_new(0.01)
		for p in nn.sequential_parameters(models[k], context.temp_allocator) {nn.adam_add_param(&o, p)}
	}
	defer for &o in opts {nn.adam_free(&o)}
	gpu.to_device(nn.sequential_parameters(gpu_model, context.temp_allocator)[:])
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	gpu.reset_stats()
	STEPS :: 40
	worst_loss, worst_ratio := 0.0, 0.0
	rand.reset(5)
	for _ in 0 ..< STEPS {
		n := 500 + rand.int_max(7501)
		xm, ym := l.matrix_new(f64, n, 32), l.matrix_new(f64, n, 32)
		for &v in xm.data {v = f64(f32(rand.float64() * 2 - 1))}
		for &v, i in ym.data {v = f64(f32(xm.data[i] * xm.data[i]))}
		losses: [2]f64
		for k in 0 ..< 2 {
			x := t.tensor_new(l.matrix_new(f64, n, 32))
			y := t.tensor_new(l.matrix_new(f64, n, 32))
			copy(x.data.data, xm.data)
			copy(y.data.data, ym.data)
			if k == 1 {gpu.to_device(x); gpu.to_device(y)}
			t.graph_begin(&scope)
			loss := t.tensor_mse_loss(nn.sequential_forward(models[k], x), y)
			t.tensor_backward(loss)
			losses[k] = t.tensor_item(loss)
			t.graph_end(&scope)
			nn.adam_step(&opts[k])
			nn.adam_zero_grad(&opts[k])
			t.tensor_free(x)
			t.tensor_free(y)
		}
		l.matrix_free(&xm)
		l.matrix_free(&ym)
		worst_loss = max(worst_loss, abs(losses[1] - losses[0]) / (1 + abs(losses[0])))
		st := gpu.stats()
		worst_ratio = max(worst_ratio, f64(st.pool_bytes) / f64(st.peak_live_bytes))
	}
	st := gpu.stats()
	check(worst_loss <= 2e-3, "losses differ from the CPU by %.2e", worst_loss)
	check(
		worst_ratio <= 1.25,
		"the pool held %.2f times the peak in use (at most 1.25)",
		worst_ratio,
	)
	check(st.buffers_given_back > 0, "no buffer was given back although shapes changed every step")
	fmt.printfln(
		"    %d steps of 500 to 8000 rows: losses match the CPU to %.1e; pool at most %.2f x the peak in use (%.1f MB), %d buffers given back",
		STEPS,
		worst_loss,
		worst_ratio,
		f64(st.peak_live_bytes) / 1e6,
		st.buffers_given_back,
	)
	gpu.to_host(nn.sequential_parameters(gpu_model, context.temp_allocator)[:])
}
