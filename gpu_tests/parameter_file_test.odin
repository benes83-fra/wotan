package main

import gpu "../wotan/gpu"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:os"

// save_parameters / load_parameters with the parameters and Adam moments on
// the GPU: a run restarted from the file repeats the uninterrupted GPU run bit
// for bit (a parameter with its own learning rate, a frozen one, and an
// optimizer that already holds device moments when the file is loaded into it);
// the moments stay on the device around the save and after the load.
gpu_parameter_file_test :: proc() {
	fmt.println(
		"gpu_parameter_file_test: save_parameters / load_parameters of a GPU-trained parameter list resume bit for bit",
	)
	if !os.exists("model_checkpoints") {os.make_directory("model_checkpoints")}
	path := "model_checkpoints/gpu_parameter_file_test.bin"
	defer os.remove(path)
	x, y := _regression_data(32, 4, 3, 5)
	defer {t.tensor_free(x); t.tensor_free(y)}
	gpu.to_device(x)
	gpu.to_device(y)

	// 4 -> 8 -> layer norm -> tanh -> 3, values rounded to f32 from `seed`
	build :: proc(seed: u64) -> (^nn.Sequential, []^t.Tensor) {
		rand.reset(seed)
		model := nn.sequential_new()
		nn.sequential_add(model, nn.linear_layer_new(4, 8))
		nn.sequential_add(model, nn.layer_norm_layer_new(8))
		nn.sequential_add(model, nn.Activation.Tanh)
		nn.sequential_add(model, nn.linear_layer_new(8, 3))
		ps := nn.sequential_parameters(model, context.temp_allocator, all = true)
		for p in ps {for &v in p.data.data {v = f64(f32(rand.float64() - 0.5))}}
		gpu.to_device(ps[:])
		return model, ps[:]
	}
	// ps[3] (the norm's beta) frozen, ps[4] (the second weights) at its own rate
	adam :: proc(ps: []^t.Tensor, lr: f64) -> nn.Adam {
		opt := nn.adam_new(lr, weight_decay = 0.01)
		for p, i in ps {
			if i == 4 {nn.adam_add_param_with_lr(&opt, p, lr / 2)} else {nn.adam_add_param(&opt, p)}
		}
		return opt
	}
	train :: proc(model: ^nn.Sequential, opt: ^nn.Adam, x, y: ^t.Tensor, n: int, losses: ^[dynamic]f64) {
		for _ in 0 ..< n {
			loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
			append(losses, t.tensor_item(loss))
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
			nn.adam_step(opt)
			nn.adam_zero_grad(opt)
		}
	}

	a, pa := build(31)
	defer nn.sequential_free(a)
	opt_a := adam(pa, 0.02)
	defer nn.adam_free(&opt_a)
	nn.tensor_set_requires_grad(pa[3], false)
	losses_a := make([dynamic]f64, context.temp_allocator)
	train(a, &opt_a, x, y, 3, &losses_a)
	check(
		nn.save_parameters(path, pa, &opt_a, transmute([]u8)string("cfg")),
		"save_parameters failed",
	)
	on_device := true
	for p, i in pa {on_device = on_device && p.device != nil && (i == 3 || opt_a.moment_1[i] == nil)}
	check(on_device, "after save_parameters the parameters or their moments left the device")
	clear(&losses_a)
	train(a, &opt_a, x, y, 4, &losses_a)

	b, pb := build(32)
	defer nn.sequential_free(b)
	opt_b := adam(pb, 0.5)
	defer nn.adam_free(&opt_b)
	scratch := make([dynamic]f64, context.temp_allocator)
	train(b, &opt_b, x, y, 2, &scratch) // device moments that the file replaces
	meta, ok := nn.load_parameters(path, pb, &opt_b)
	defer delete(meta)
	check(ok && string(meta) == "cfg", "load_parameters failed")
	check(!pb[3].requires_grad, "the frozen flag was not restored")
	losses_b := make([dynamic]f64, context.temp_allocator)
	train(b, &opt_b, x, y, 4, &losses_b)

	same := len(losses_a) == len(losses_b)
	for v, i in losses_a {same = same && v == losses_b[i]}
	check(same, "resumed GPU losses differ: %v vs %v", losses_a[:], losses_b[:])
	worst := 0.0
	for p, i in pa {worst = max(worst, _max_rel(t.tensor_values(p), t.tensor_values(pb[i])))}
	check(worst == 0, "resumed GPU parameters differ by %.2e", worst)
	moments_on_device := true
	for p, i in pb {if i != 3 {moments_on_device = moments_on_device && opt_b.moment_1[i] == nil}}
	check(moments_on_device, "after the resumed steps the moments are not on the device")
	fmt.printfln(
		"    losses %v (uninterrupted) vs %v (resumed); parameters max diff %.1e",
		losses_a[:],
		losses_b[:],
		worst,
	)
	gpu.to_host(pa)
	gpu.to_host(pb)
	gpu.to_host(x)
	gpu.to_host(y)
}
