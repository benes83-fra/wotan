package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import ml "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"

// ml_finance models trained on the GPU against the CPU: the same seed, data and
// steps on both devices. Each run returns its losses and final parameters.

// skip: parameters left out of the comparison (attention key biases, whose true
// gradient is zero: see _key_biases)
ML_Run :: #type proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int)

_values_of :: proc(ps: []^t.Tensor) -> [][]f64 {
	out := make([][]f64, len(ps), context.temp_allocator)
	for p, i in ps {out[i] = _clone(t.tensor_values(p))}
	return out
}

_host_tensor :: proc(rows, cols: int, values: []f64 = nil) -> ^t.Tensor {
	m := l.matrix_new(f64, rows, cols)
	if values !=
	   nil {copy(m.data, values)} else {for &v in m.data {v = f64(f32(2 * rand.float64() - 1))}}
	return t.tensor_new(m)
}

// _ml_case: CPU run, GPU run, comparison; `allowed` lists the ops that may fall
// back (none for most models)
_ml_case :: proc(name: string, run: ML_Run, tol: f64, allowed: bit_set[t.Op] = {}) {
	ref_l, ref_p, skip := run(false)
	gpu.reset_stats()
	got_l, got_p, _ := run(true)
	st := gpu.stats()
	other, listed := 0, make([dynamic]string, context.temp_allocator)
	for op in t.Op {
		n := st.forward_fallbacks[op] + st.backward_fallbacks[op]
		if n == 0 {continue}
		append(&listed, fmt.tprintf("%v", op))
		if op not_in allowed {other += n}
	}
	_compare_runs(name, ref_l, got_l, ref_p, got_p, tol, skip)
	check(other == 0, "%s: unexpected CPU fallbacks %v", name, listed[:])
	if len(listed) > 0 {fmt.printfln("        CPU fallbacks (expected): %v", listed[:])}
}

// ----------------------------------------------------------------------------

_hedging_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	B, S :: 64, 6
	rand.reset(61)
	h := ml.deep_hedger_new(
		{
			state_size = 2,
			hidden_size = 16,
			num_layers = 2,
			risk_measure = .CVaR,
			cvar_alpha = 0.2,
			transaction_cost = 0.001,
			num_assets = 1,
		},
	)
	defer ml.deep_hedger_free(h)
	ps := nn.sequential_parameters(h.network, context.temp_allocator)
	_round_params(ps[:])
	opt := nn.adam_new(0.01)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	// GBM paths: (spot, time) per date; a call payoff
	pv := make([]f64, B * S * 2, context.temp_allocator)
	pay := make([]f64, B, context.temp_allocator)
	for b in 0 ..< B {
		s := 1.0
		for k in 0 ..< S {
			pv[(b * S + k) * 2] = f64(f32(s))
			pv[(b * S + k) * 2 + 1] = f64(f32(f64(k) / f64(S)))
			s *= math.exp(0.2 * math.sqrt(1.0 / f64(S)) * rand.float64_normal(0, 1))
		}
		pay[b] = f64(f32(max(pv[(b * S + S - 1) * 2] - 1, 0)))
	}
	paths := _host_tensor(1, B * S * 2, pv)
	paths.shape = {B, S, 2, 1}
	payoffs := _host_tensor(B, 1, pay)
	defer {t.tensor_free(paths); t.tensor_free(payoffs)}
	if on_gpu {gpu.to_device(ps[:])}
	losses = make([]f64, 8, context.temp_allocator)
	for i in 0 ..< 8 {losses[i] = ml.deep_hedger_train_step(h, paths, payoffs, &opt)}
	params = _values_of(ps[:])
	if on_gpu {gpu.to_host(ps[:])}
	return
}

_nbeats_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	rand.reset(62)
	m := ml.nbeats_new({input_size = 12, horizon = 4, blocks = 2, layers = 2, layer_size = 32})
	defer ml.nbeats_free(m)
	ps := ml.nbeats_parameters(m, context.temp_allocator)
	_round_params(ps[:])
	opt := nn.adam_new(0.005)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	xv := make([]f64, 32 * 16, context.temp_allocator)
	for b in 0 ..< 32 {for j in 0 ..< 16 {xv[b * 16 + j] = f64(f32(math.sin(0.4 * f64(j + b)) + 0.05 * rand.float64()))}}
	x := _host_tensor(32, 12)
	y := _host_tensor(32, 4)
	for b in 0 ..< 32 {
		copy(x.data.data[b * 12:][:12], xv[b * 16:][:12])
		copy(y.data.data[b * 4:][:4], xv[b * 16 + 12:][:4])
	}
	defer {t.tensor_free(x); t.tensor_free(y)}
	if on_gpu {gpu.to_device(ps[:]); gpu.to_device(x); gpu.to_device(y)}
	losses = make([]f64, 10, context.temp_allocator)
	for i in 0 ..< 10 {losses[i] = ml.nbeats_train_step(m, x, y, &opt)}
	params = _values_of(ps[:])
	if on_gpu {gpu.to_host(ps[:]); gpu.to_host(x); gpu.to_host(y)}
	return
}

_patchtst_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	rand.reset(63)
	m := ml.patchtst_new(
		{
			input_size = 24,
			horizon = 4,
			patch_len = 8,
			stride = 4,
			d_model = 16,
			n_heads = 2,
			d_ff = 32,
			n_layers = 1,
			channels = 1,
			revin = true,
		},
	)
	defer ml.patchtst_free(m)
	ps := ml.patchtst_parameters(m, context.temp_allocator)
	_round_params(ps[:])
	mhas := make([dynamic]^nn.MultiHeadAttentionLayer, context.temp_allocator)
	for &b in m.encoder.blocks {append(&mhas, &b.mha)}
	skip = _key_biases(ps[:], mhas[:])
	opt := nn.adam_new(0.005)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	x := _host_tensor(16, 24)
	y := _host_tensor(16, 4)
	for r in 0 ..< 16 {
		for j in 0 ..< 28 {
			v := f64(f32(math.sin(0.5 * f64(j + 3 * r))))
			if j < 24 {x.data.data[r * 24 + j] = v} else {y.data.data[r * 4 + j - 24] = v}
		}
	}
	defer {t.tensor_free(x); t.tensor_free(y)}
	if on_gpu {gpu.to_device(ps[:])}
	losses = make([]f64, 10, context.temp_allocator)
	for i in 0 ..< 10 {losses[i] = ml.patchtst_train_step(m, x, y, nil, &opt)}
	params = _values_of(ps[:])
	if on_gpu {gpu.to_host(ps[:])}
	return
}

_sdf_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	rand.reset(64)
	m := ml.neural_sdf_new(
		{
			input_size = 3,
			hidden_size = 16,
			num_layers = 2,
			num_assets = 5,
			epsilon = 0.01,
			variance_penalty = 0.1,
		},
	)
	defer ml.neural_sdf_free(m)
	ps := nn.sequential_parameters(m.network, context.temp_allocator)
	_round_params(ps[:])
	opt := nn.adam_new(0.005)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	z := _host_tensor(40, 3)
	r := _host_tensor(40, 5)
	for &v in r.data.data {v = f64(f32(1 + 0.05 * v))}
	defer {t.tensor_free(z); t.tensor_free(r)}
	if on_gpu {gpu.to_device(ps[:]); gpu.to_device(z); gpu.to_device(r)}
	losses = make([]f64, 10, context.temp_allocator)
	for i in 0 ..< 10 {losses[i] = ml.neural_sdf_train_step(m, z, r, &opt)}
	params = _values_of(ps[:])
	if on_gpu {gpu.to_host(ps[:]); gpu.to_host(z); gpu.to_host(r)}
	return
}

// conditional autoencoder: L1 penalty and early stopping on a validation panel
// (the best weights are restored at the end)
_ca_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	L :: 3
	panel :: proc(T, n: int) -> (X: l.Matrix(f64), y: []f64, times: []int) {
		X = l.matrix_new(f64, T * n, L, context.temp_allocator)
		y = make([]f64, T * n, context.temp_allocator)
		times = make([]int, T * n, context.temp_allocator)
		for d in 0 ..< T {
			f := rand.float64_normal(0, 0.05)
			for i in 0 ..< n {
				r := d * n + i
				times[r] = d
				for c in 0 ..< L {X.data[r * L + c] = f64(f32(2 * rand.float64() - 1))}
				beta := 1 + 0.5 * X.data[r * L] - 0.3 * X.data[r * L + 1] * X.data[r * L + 1]
				y[r] = f64(f32(beta * f + 0.01 * rand.float64_normal(0, 1)))
			}
		}
		return
	}
	rand.reset(65)
	Xa, ya, ta := panel(16, 20)
	Xv, yv, tv := panel(6, 20)
	train := ml.ca_panel_new(&Xa, ya, ta)
	valid := ml.ca_panel_new(&Xv, yv, tv)
	defer {ml.ca_panel_free(&train); ml.ca_panel_free(&valid)}
	rand.reset(66)
	m := ml.ca_new({n_chars = L, n_factors = 2, hidden = []int{8}, l1 = 1e-4})
	defer ml.ca_free(m)
	ps := ml.ca_parameters(m, context.temp_allocator)
	_round_params(ps[:])
	if on_gpu {gpu.to_device(ps[:])}
	res := ml.ca_fit(m, &train, &valid, epochs = 30, learning_rate = 0.01, patience = 5)
	losses = make([]f64, 2, context.temp_allocator)
	losses[0], losses[1] = res.last_loss, res.valid_mse
	// the parameters, then the epoch whose weights were kept
	params = make([][]f64, len(ps) + 1, context.temp_allocator)
	copy(params, _values_of(ps[:]))
	params[len(ps)] = make([]f64, 1, context.temp_allocator)
	params[len(ps)][0] = f64(res.best_epoch)
	if on_gpu {gpu.to_host(ps[:])}
	return
}

// CQL: TD + conservative loss on random transitions, soft target updates
_cql_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	cfg := ml.CQLConfig {
		state_dim   = 4,
		num_actions = 3,
		hidden_dim  = 16,
		gamma       = 0.9,
		alpha       = 0.5,
		tau         = 0.1,
	}
	rand.reset(67)
	main_net := ml.cql_network_new(cfg)
	target := ml.cql_network_new(cfg)
	defer {ml.cql_network_free(&main_net); ml.cql_network_free(&target)}
	nets := []^nn.LinearLayer{&main_net.fc1, &main_net.fc2, &target.fc1, &target.fc2}
	ps := make([dynamic]^t.Tensor, context.temp_allocator)
	for n in nets {append(&ps, n.weights, n.bias)}
	_round_params(ps[:])
	ml.cql_soft_update(&main_net, &target, 1)
	opt := nn.adam_new(0.01)
	defer nn.adam_free(&opt)
	ml.cql_network_add_to_adam(&main_net, &opt)
	B :: 16
	states := _host_tensor(B, 4)
	next := _host_tensor(B, 4)
	rewards := _host_tensor(B, 1)
	dones := _host_tensor(B, 1)
	for &v in dones.data.data {v = 1 if v > 0.6 else 0}
	actions := make([]int, B, context.temp_allocator)
	for &a in actions {a = rand.int_max(3)}
	defer {t.tensor_free(states); t.tensor_free(next); t.tensor_free(rewards)
		t.tensor_free(dones)}
	if on_gpu {gpu.to_device(ps[:])}
	losses = make([]f64, 16, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for i in 0 ..< 8 {
		t.graph_begin(&scope)
		alloc := t.graph_allocator(&scope)
		nq := ml.cql_network_forward(&target, next)
		loss, td, reg := ml.cql_loss(
			ml.cql_network_forward(&main_net, states),
			actions,
			rewards,
			nq,
			dones,
			cfg,
			alloc,
		)
		nn.adam_zero_grad(&opt)
		t.tensor_backward(loss)
		t.graph_end(&scope)
		nn.adam_step(&opt)
		ml.cql_soft_update(&main_net, &target, cfg.tau)
		losses[2 * i], losses[2 * i + 1] = td, reg
	}
	params = _values_of(ps[:])
	if on_gpu {gpu.to_host(ps[:])}
	return
}

// DeepLOB with the TCN sequence layer: convolutions, inception, LOB permute, TCN
_deeplob_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	return _deeplob_backend_run(on_gpu, .TCN)
}

// DeepLOB with the LSTM sequence layer
_deeplob_lstm_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	return _deeplob_backend_run(on_gpu, .LSTM)
}

// DeepLOB with the GRU sequence layer
_deeplob_gru_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	return _deeplob_backend_run(on_gpu, .GRU)
}

_deeplob_backend_run :: proc(
	on_gpu: bool,
	backend: nn.SeqBackend,
) -> (
	losses: []f64,
	params: [][]f64,
	skip: []int,
) {
	rand.reset(68)
	m := ml.deeplob_new(
		{
			time_steps = 12,
			price_levels = 6,
			num_classes = 3,
			hidden_dim = 8,
			seq_backend = backend,
		},
	)
	defer ml.deeplob_free(m)
	opt := nn.adam_new(0.003)
	defer nn.adam_free(&opt)
	ml.deeplob_add_to_adam(m, &opt)
	_round_params(opt.parameters[:])
	x := _host_tensor(1, 4 * 3 * 12 * 6)
	x.shape = {4, 3, 12, 6}
	defer t.tensor_free(x)
	if on_gpu {gpu.to_device(opt.parameters[:]); gpu.to_device(x)}
	losses = make([]f64, 5, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for i in 0 ..< 5 {
		rand.reset(u64(900 + i)) // the same dropout mask on both runs
		t.graph_begin(&scope)
		loss := t.tensor_cross_entropy_loss(ml.deeplob_forward(m, x), {0, 1, 2, 1})
		nn.adam_zero_grad(&opt)
		t.tensor_backward(loss)
		losses[i] = t.tensor_item(loss)
		t.graph_end(&scope)
		nn.adam_step(&opt)
	}
	params = _values_of(opt.parameters[:])
	if on_gpu {gpu.to_host(opt.parameters[:]); gpu.to_host(x)}
	return
}

// volatility forecaster: LSTM or GRU over 10 steps, flatten, two linear layers
_lstm_vola_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	return _vola_run(on_gpu, .LSTM)
}

_gru_vola_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	return _vola_run(on_gpu, .GRU)
}

_vola_run :: proc(
	on_gpu: bool,
	backend: nn.SeqBackend,
) -> (
	losses: []f64,
	params: [][]f64,
	skip: []int,
) {
	rand.reset(70)
	m := ml.lstm_volatility_forecaster_new(
		{input_size = 3, hidden_size = 16, seq_len = 10, seq_backend = backend},
	)
	defer ml.lstm_volatility_forecaster_free(&m)
	opt := nn.adam_new(0.005)
	defer nn.adam_free(&opt)
	ml.lstm_volatility_add_to_optimizer(&m, &opt)
	_round_params(opt.parameters[:])
	x := _host_tensor(1, 8 * 10 * 3)
	x.shape = {8, 10, 3, 1}
	y := _host_tensor(8, 1)
	defer {t.tensor_free(x); t.tensor_free(y)}
	if on_gpu {gpu.to_device(opt.parameters[:]); gpu.to_device(x); gpu.to_device(y)}
	losses = make([]f64, 10, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for i in 0 ..< 10 {
		t.graph_begin(&scope)
		loss := t.tensor_mse_loss(ml.lstm_volatility_forecaster_forward(&m, x), y)
		nn.adam_zero_grad(&opt)
		t.tensor_backward(loss)
		losses[i] = t.tensor_item(loss)
		t.graph_end(&scope)
		nn.adam_step(&opt)
	}
	params = _values_of(opt.parameters[:])
	if on_gpu {gpu.to_host(opt.parameters[:]); gpu.to_host(x); gpu.to_host(y)}
	return
}

// VAE: encoder, reparameterization (seeded noise), decoder, MSE + KL
_vae_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	rand.reset(69)
	enc := nn.vae_encoder_new(6, 16, 2)
	dec := nn.vae_decoder_new(2, 16, 6)
	defer {nn.vae_encoder_free(&enc); nn.vae_decoder_free(&dec)}
	opt := nn.adam_new(0.005)
	defer nn.adam_free(&opt)
	nn.vae_encoder_add_to_optimizer(&enc, &opt)
	nn.vae_decoder_add_to_optimizer(&dec, &opt)
	_round_params(opt.parameters[:])
	x := _host_tensor(24, 6)
	for &v in x.data.data {v = f64(f32(0.8 * v))}
	defer t.tensor_free(x)
	if on_gpu {gpu.to_device(opt.parameters[:]); gpu.to_device(x)}
	losses = make([]f64, 10, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for i in 0 ..< 10 {
		rand.reset(u64(950 + i))
		t.graph_begin(&scope)
		mu, lv := nn.vae_encoder_forward(&enc, x)
		z := nn.reparameterize(mu, lv, context.allocator)
		recon := nn.vae_decoder_forward(&dec, z)
		loss := t.tensor_add(
			t.tensor_mse_loss(recon, x),
			t.tensor_scale(t.tensor_kl_divergence(mu, lv), 0.1),
		)
		nn.adam_zero_grad(&opt)
		t.tensor_backward(loss)
		losses[i] = t.tensor_item(loss)
		t.graph_end(&scope)
		nn.adam_step(&opt)
	}
	params = _values_of(opt.parameters[:])
	if on_gpu {gpu.to_host(opt.parameters[:]); gpu.to_host(x)}
	return
}

gpu_ml_finance_test :: proc() {
	fmt.println("gpu_ml_finance_test: ml_finance models trained on the GPU vs the CPU")
	_ml_case("deep hedging (CVaR)", _hedging_run, 2e-3)
	_ml_case("N-BEATS", _nbeats_run, 2e-3)
	_ml_case("PatchTST (RevIN)", _patchtst_run, 2e-3)
	_ml_case("neural SDF", _sdf_run, 2e-3)
	_ml_case("conditional autoencoder (L1, early stop)", _ca_run, 2e-3)
	_ml_case("CQL (soft target updates)", _cql_run, 2e-3)
	_ml_case("DeepLOB (TCN)", _deeplob_run, 2e-3)
	_ml_case("DeepLOB (LSTM)", _deeplob_lstm_run, 2e-3)
	_ml_case("DeepLOB (GRU)", _deeplob_gru_run, 2e-3)
	_ml_case("LSTM volatility forecaster", _lstm_vola_run, 2e-3)
	_ml_case("volatility forecaster (GRU)", _gru_vola_run, 2e-3)
	_ml_case("VAE (reparameterize, KL)", _vae_run, 2e-3)
}
