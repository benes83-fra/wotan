package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

// save_parameters / load_parameters on a model that is not a Sequential (two
// linear layers around a layer norm, parameters in a list): a run restarted
// from the file repeats the uninterrupted run bit for bit (Adam state with a
// parameter of its own learning rate and a frozen one), weights and metadata
// round-trip without an optimizer, and broken or mismatching files change
// nothing and leak nothing.

@(private = "file")
_Pf_Model :: struct {
	fc1, fc2: nn.LinearLayer,
	norm:     nn.LayerNormLayer,
}

// 4 -> 8 -> norm -> tanh -> 3, every parameter random from `seed`
@(private = "file")
_pf_model :: proc(seed: u64) -> (m: _Pf_Model) {
	rand.reset(seed)
	m.fc1 = nn.linear_layer_new(4, 8)
	m.norm = nn.layer_norm_layer_new(8)
	m.fc2 = nn.linear_layer_new(8, 3)
	for p in _pf_params(&m) {for &v in p.data.data {v = rand.float64() - 0.5}}
	return
}

@(private = "file")
_pf_params :: proc(m: ^_Pf_Model) -> []^t.Tensor {
	ps := make([]^t.Tensor, 6, context.temp_allocator)
	ps[0], ps[1], ps[2] = m.fc1.weights, m.fc1.bias, m.norm.gamma
	ps[3], ps[4], ps[5] = m.norm.beta, m.fc2.weights, m.fc2.bias
	return ps
}

@(private = "file")
_pf_free :: proc(m: ^_Pf_Model) {
	nn.linear_layer_free(&m.fc1)
	nn.layer_norm_layer_free(&m.norm)
	nn.linear_layer_free(&m.fc2)
}

// Adam with fc2.weights at half the learning rate and norm.beta frozen (still
// registered)
@(private = "file")
_pf_adam :: proc(m: ^_Pf_Model, lr: f64) -> nn.Adam {
	opt := nn.adam_new(lr, weight_decay = 0.01)
	ps := _pf_params(m)
	for p, i in ps {
		if i == 4 {nn.adam_add_param_with_lr(&opt, p, lr / 2)} else {nn.adam_add_param(&opt, p)}
	}
	nn.tensor_set_requires_grad(m.norm.beta, false)
	return opt
}

// Fixed inputs and targets
@(private = "file")
_pf_data :: proc() -> (x, y: ^t.Tensor) {
	x = t.tensor_new(l.matrix_new(f64, 16, 4), false)
	y = t.tensor_new(l.matrix_new(f64, 16, 3), false)
	for &v, i in x.data.data {v = 0.7 * f64((i * 7) % 11) / 11 - 0.3}
	for &v, i in y.data.data {v = f64((i * 5) % 9) / 9 - 0.4}
	return
}

// n training steps; their losses are appended to `losses`
@(private = "file")
_pf_train :: proc(m: ^_Pf_Model, opt: ^nn.Adam, x, y: ^t.Tensor, n: int, losses: ^[dynamic]f64) {
	for _ in 0 ..< n {
		h := t.tensor_tanh(nn.layer_norm_layer_forward(&m.norm, nn.linear_forward(&m.fc1, x)))
		loss := t.tensor_mse_loss(nn.linear_forward(&m.fc2, h), y)
		append(losses, t.tensor_item(loss))
		t.tensor_backward(loss)
		t.tensor_free_graph(loss)
		nn.adam_step(opt)
		nn.adam_zero_grad(opt)
	}
}

// The parameters' values, copied
@(private = "file")
_pf_values :: proc(ps: []^t.Tensor) -> [][]f64 {
	out := make([][]f64, len(ps), context.temp_allocator)
	for p, i in ps {
		out[i] = make([]f64, t.tensor_numel(p), context.temp_allocator)
		copy(out[i], t.tensor_values(p))
	}
	return out
}

@(private = "file")
_pf_same :: proc(a, b: [][]f64) -> bool {
	if len(a) != len(b) {return false}
	for x, i in a {
		if len(x) != len(b[i]) {return false}
		for v, j in x {if v != b[i][j] {return false}}
	}
	return true
}

parameter_file_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Parameter files (save_parameters / load_parameters) ===")
	context.allocator = allocator
	os.make_directory("model_checkpoints")
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	path := "model_checkpoints/parameter_file_test.bin"
	broken := "model_checkpoints/parameter_file_test_broken.bin"
	defer {_ = os.remove(path); _ = os.remove(broken)}
	x, y := _pf_data()
	meta := transmute([]u8)string("d_model=8 step=3")

	// 1. Resume: 3 steps, save, 4 more; a fresh model loaded from the file and
	// trained the same 4 steps gives the same losses and parameters
	a := _pf_model(11)
	defer _pf_free(&a)
	opt_a := _pf_adam(&a, 0.02)
	defer nn.adam_free(&opt_a)
	losses_a := make([dynamic]f64)
	_pf_train(&a, &opt_a, x, y, 3, &losses_a)
	expect(
		&passed,
		&total,
		nn.save_parameters(path, _pf_params(&a), &opt_a, meta),
		"save_parameters",
	)
	clear(&losses_a)
	_pf_train(&a, &opt_a, x, y, 4, &losses_a)

	b := _pf_model(12) // other values, beta not frozen, other learning rate
	defer _pf_free(&b)
	opt_b := nn.adam_new(0.5)
	defer nn.adam_free(&opt_b)
	for p, i in _pf_params(&b) {
		if i == 4 {nn.adam_add_param_with_lr(&opt_b, p, 0.1)} else {nn.adam_add_param(&opt_b, p)}
	}
	got, ok := nn.load_parameters(path, _pf_params(&b), &opt_b)
	expect(&passed, &total, ok && string(got) == string(meta), "load_parameters: metadata")
	expect(
		&passed,
		&total,
		!b.norm.beta.requires_grad && b.norm.gamma.requires_grad,
		"load_parameters: frozen flags restored",
	)
	expect(
		&passed,
		&total,
		opt_b.learning_rate == 0.02 &&
		opt_b.weight_decay == 0.01 &&
		opt_b.timestep == 3 &&
		opt_b.lr_scales[4] == 0.5 &&
		opt_b.steps[3] == 0 &&
		opt_b.steps[0] == 3,
		"load_parameters: Adam hyperparameters, timestep, step counts, learning-rate scales",
	)
	losses_b := make([dynamic]f64)
	_pf_train(&b, &opt_b, x, y, 4, &losses_b)
	same_losses := len(losses_a) == len(losses_b)
	for v, i in losses_a {same_losses = same_losses && v == losses_b[i]}
	expect(&passed, &total, same_losses, "resumed run: the same losses bit for bit")
	expect(
		&passed,
		&total,
		_pf_same(_pf_values(_pf_params(&a)), _pf_values(_pf_params(&b))),
		"resumed run: the same parameters bit for bit",
	)

	// 2. Weights only: values, frozen flags and metadata round-trip
	expect(
		&passed,
		&total,
		nn.save_parameters(path, _pf_params(&a), nil, meta),
		"save_parameters without an optimizer",
	)
	c := _pf_model(13)
	defer _pf_free(&c)
	got_c, ok_c := nn.load_parameters(path, _pf_params(&c))
	expect(
		&passed,
		&total,
		ok_c &&
		string(got_c) == string(meta) &&
		_pf_same(_pf_values(_pf_params(&a)), _pf_values(_pf_params(&c))) &&
		!c.norm.beta.requires_grad,
		"weights only: values, frozen flag and metadata",
	)
	got_m, ok_m := nn.read_parameters_metadata(path)
	expect(&passed, &total, ok_m && string(got_m) == string(meta), "read_parameters_metadata")
	// a file with Adam state also loads without an optimizer, and an
	// optimizer is left alone by a file without Adam state
	opt_c := _pf_adam(&c, 0.3)
	defer nn.adam_free(&opt_c)
	_, ok_c = nn.load_parameters(path, _pf_params(&c), &opt_c)
	expect(
		&passed,
		&total,
		ok_c && opt_c.learning_rate == 0.3 && opt_c.timestep == 0,
		"file without Adam state: optimizer unchanged",
	)
	_ = nn.save_parameters(path, _pf_params(&a), &opt_a, nil)
	got_c, ok_c = nn.load_parameters(path, _pf_params(&c))
	expect(
		&passed,
		&total,
		ok_c && got_c == nil,
		"file with Adam state, no optimizer; no metadata gives nil",
	)

	// 3. Broken or mismatching files: ok = false, nothing changed, nothing leaked
	{
		data, _ := os.read_entire_file(path, context.allocator)
		d := _pf_model(14)
		defer _pf_free(&d)
		opt_d := _pf_adam(&d, 0.3)
		defer nn.adam_free(&opt_d)
		ps := _pf_params(&d)
		before := _pf_values(ps)
		Case :: struct {
			name:  string,
			bytes: []u8, // written to `broken` (nil: use `path`)
			drop:  int, // parameters left out of the list
			short: bool, // fc2.weights replaced by a tensor of another shape
			unreg: bool, // an optimizer without fc1.bias
		}
		wrong_magic := make([]u8, len(data))
		copy(wrong_magic, data)
		wrong_magic[0] = 'X'
		wrong_id := make([]u8, len(data))
		copy(wrong_id, data)
		wrong_id[len(nn.CHECKPOINT_MAGIC)] = 38 // a GPT file
		trailing := make([]u8, len(data) + 1)
		copy(trailing, data)
		cases := []Case {
			{name = "truncated file", bytes = data[:len(data) - 5]},
			{name = "file cut inside the parameters", bytes = data[:len(data) / 3]},
			{name = "wrong magic", bytes = wrong_magic},
			{name = "another file type", bytes = wrong_id},
			{name = "trailing bytes", bytes = trailing},
			{name = "count mismatch", drop = 1},
			{name = "shape mismatch", short = true},
			{name = "registration mismatch", unreg = true},
			{name = "missing file", bytes = {}},
		}
		other := t.tensor_new(l.matrix_new(f64, 3, 8), true)
		for c in cases {
			file := path
			if c.bytes != nil {
				file = broken
				_ = os.write_entire_file(broken, c.bytes)
			} else if c.name == "missing file" {
				file = "model_checkpoints/parameter_file_test_missing.bin"
			}
			list := ps[:len(ps) - c.drop]
			if c.short {
				list = make([]^t.Tensor, len(ps), context.temp_allocator)
				copy(list, ps)
				list[4] = other
			}
			opt := &opt_d
			opt_u: nn.Adam
			if c.unreg {
				opt_u = nn.adam_new(0.3)
				for p, i in ps {if i != 1 {nn.adam_add_param(&opt_u, p)}}
				opt = &opt_u
			}
			track: mem.Tracking_Allocator
			mem.tracking_allocator_init(&track, runtime.heap_allocator()) // the arena frees nothing singly
			context.allocator = mem.tracking_allocator(&track)
			meta_out, ok_out := nn.load_parameters(file, list, opt)
			context.allocator = track.backing
			leaked := len(track.allocation_map)
			mem.tracking_allocator_destroy(&track)
			if c.unreg {nn.adam_free(&opt_u)}
			unchanged :=
				_pf_same(before, _pf_values(ps)) &&
				opt_d.timestep == 0 &&
				opt_d.learning_rate == 0.3 &&
				d.norm.beta.requires_grad == false &&
				d.norm.gamma.requires_grad
			expect(
				&passed,
				&total,
				!ok_out && meta_out == nil && unchanged && leaked == 0,
				fmt.tprintf("%s: ok = false, nothing changed, no leaks (%d left)", c.name, leaked),
			)
		}
	}
	record_failures(total - passed)
	fmt.printf("Parameter files: %d/%d passed\n", passed, total)
}
