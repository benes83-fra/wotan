package ml_finance

import l "../linalg"
import nn "../nn"
import tensor "../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// TimeGAN (Yoon, Jarrett & van der Schaar 2019; as in the authors' code)
//
// Five recurrent networks (LSTM + linear + sigmoid), sequences [B, T, F] min-max scaled
// to [0, 1] (timegan_scaler_*):
//   embedder      X -> H        recovery  H -> X~
//   generator     Z -> E^ (Z ~ U[0,1])   supervisor H -> next H (one step ahead)
//   discriminator H -> P(real) per step
// Training, in three phases:
//   1. train_embedder_recovery: 10 sqrt(MSE(X, R(E(X))))
//   2. train_supervisor:        L_S = MSE(H[:, 1:], S(H)[:, :-1]) with H = E(X)
//   3. timegan_train_joint_step: twice (generator step, embedder step), then the
//      discriminator if its loss is above 0.15:
//        generator  BCE(1, D(S(E^))) + γ BCE(1, D(E^)) + 100 sqrt(L_S) + 100 L_V
//                   (L_V: mean |std diff| + mean |mean diff| over the batch, per step
//                   and feature, of X^ = R(S(E^)) against X)
//        embedder   10 sqrt(MSE(X, R(E(X)))) + 0.1 L_S
//        discrim.   BCE(1, D(H)) + BCE(0, D(S(E^))) + γ BCE(0, D(E^))
// Synthetic sequences are R(S(G(Z))).
//
// The previous version trained the supervisor to map latents to the features (no
// temporal supervision at all), gave the generator a "supervised" term that did not
// depend on it, generated R(G(Z)) without the supervisor, had no moment loss and no
// joint embedder step, and slowed the discriminator 10x instead of gating it.
// ============================================================================

TIMEGAN_GAMMA :: 1.0
TIMEGAN_D_THRESHOLD :: 0.15

TimeGAN :: struct {
	embedder:      ^nn.Sequential,
	recovery:      ^nn.Sequential,
	generator:     ^nn.Sequential,
	discriminator: ^nn.Sequential,
	supervisor:    ^nn.Sequential,
	optimizer_e:   nn.Adam,
	optimizer_r:   nn.Adam,
	optimizer_g:   nn.Adam,
	optimizer_d:   nn.Adam,
	optimizer_s:   nn.Adam,
	seq_len:       int,
	feature_dim:   int,
	latent_dim:    int,
	allocator:     mem.Allocator,
	// One arena per training step / generated batch, reused across them
	scope:         tensor.Graph_Scope,
}

new_timegan :: proc(
	seq_len: int,
	feature_dim: int,
	latent_dim: int = 32,
	hidden_dim: int = 64,
	lr: f64 = 1e-3,
	alloc: mem.Allocator = context.allocator,
) -> ^TimeGAN {
	assert(
		seq_len >= 2 && feature_dim > 0 && latent_dim > 0 && hidden_dim > 0,
		"new_timegan: need seq_len >= 2 and positive sizes",
	)
	tg := new(TimeGAN, alloc)
	tg.seq_len = seq_len
	tg.feature_dim = feature_dim
	tg.latent_dim = latent_dim
	tg.allocator = alloc

	rnn :: proc(in_dim, hidden, out_dim: int, alloc: mem.Allocator) -> ^nn.Sequential {
		s := nn.sequential_new(alloc)
		nn.sequential_add(s, nn.lstm_layer_new(in_dim, hidden, alloc))
		nn.sequential_add(s, nn.linear_layer_new(hidden, out_dim, alloc))
		nn.sequential_add(s, nn.Activation.Sigmoid)
		return s
	}
	tg.embedder = rnn(feature_dim, hidden_dim, latent_dim, alloc)
	tg.recovery = rnn(latent_dim, hidden_dim, feature_dim, alloc)
	tg.generator = rnn(latent_dim, hidden_dim, latent_dim, alloc)
	tg.supervisor = rnn(latent_dim, hidden_dim, latent_dim, alloc)
	tg.discriminator = rnn(latent_dim, hidden_dim, 1, alloc)

	tg.optimizer_e = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)
	tg.optimizer_r = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)
	tg.optimizer_g = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)
	tg.optimizer_d = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)
	tg.optimizer_s = nn.adam_new(lr, 0.9, 0.999, 1e-8, alloc)

	nn.sequential_add_to_adam(tg.embedder, &tg.optimizer_e)
	nn.sequential_add_to_adam(tg.recovery, &tg.optimizer_r)
	nn.sequential_add_to_adam(tg.generator, &tg.optimizer_g)
	nn.sequential_add_to_adam(tg.discriminator, &tg.optimizer_d)
	nn.sequential_add_to_adam(tg.supervisor, &tg.optimizer_s)

	return tg
}

timegan_free :: proc(tg: ^TimeGAN) {
	nn.sequential_free(tg.embedder)
	nn.sequential_free(tg.recovery)
	nn.sequential_free(tg.generator)
	nn.sequential_free(tg.discriminator)
	nn.sequential_free(tg.supervisor)
	nn.adam_free(&tg.optimizer_e)
	nn.adam_free(&tg.optimizer_r)
	nn.adam_free(&tg.optimizer_g)
	nn.adam_free(&tg.optimizer_d)
	nn.adam_free(&tg.optimizer_s)
	tensor.graph_scope_destroy(&tg.scope)
	free(tg, tg.allocator)
}

// ============================================================================
// Data Preparation
// ============================================================================

// Per-feature min-max scaling to [0, 1], as the recovery's sigmoid output needs
TimeGANScaler :: struct {
	lo, span: []f64, // x_scaled = (x - lo) / span
}

timegan_scaler_fit :: proc(
	data: []f64,
	feature_dim: int,
	allocator: mem.Allocator,
) -> TimeGANScaler {
	s := TimeGANScaler{make([]f64, feature_dim, allocator), make([]f64, feature_dim, allocator)}
	hi := make([]f64, feature_dim, context.temp_allocator)
	for f in 0 ..< feature_dim {s.lo[f], hi[f] = math.INF_F64, -math.INF_F64}
	for v, k in data {
		f := k % feature_dim
		s.lo[f], hi[f] = min(s.lo[f], v), max(hi[f], v)
	}
	for f in 0 ..< feature_dim {s.span[f] = hi[f] - s.lo[f] + 1e-7}
	return s
}

timegan_scale :: proc(data: []f64, s: TimeGANScaler) {
	F := len(s.lo)
	for &v, k in data {v = (v - s.lo[k % F]) / s.span[k % F]}
}

timegan_unscale :: proc(data: []f64, s: TimeGANScaler) {
	F := len(s.lo)
	for &v, k in data {v = v * s.span[k % F] + s.lo[k % F]}
}

timegan_scaler_free :: proc(s: ^TimeGANScaler, allocator: mem.Allocator) {
	delete(s.lo, allocator)
	delete(s.span, allocator)
}

// Overlapping windows of seq_len days of [log return, log volume change, indicators...]
// (returns clipped to ±0.1, volume changes to ±2, indicators standardised over the
// sample and clipped to ±3). The first day has no return, so the windows start on day
// 1: n_days - seq_len of them. (The indicators used to be standardised in the caller's
// buffer, and every window set started on day 0's placeholder return of 0.)
prepare_market_data :: proc(
	prices: []f64,
	volumes: []f64,
	indicators: []f64,
	n_indicators: int,
	seq_len: int,
	alloc: mem.Allocator,
) -> (
	sequences: []f64,
	n_sequences: int,
	feature_dim: int,
) {
	n_days := len(prices)
	assert(
		len(volumes) == n_days && len(indicators) == n_days * n_indicators,
		"prepare_market_data: lengths differ",
	)
	feature_dim = 2 + n_indicators
	n_sequences = max(n_days - seq_len, 0)
	if n_sequences == 0 {return nil, 0, feature_dim}

	log_change :: proc(a, b, lim: f64) -> f64 {
		return clamp(math.ln(b / a), -lim, lim) if a > 0 && b > 0 else 0
	}
	ind_mean := make([]f64, n_indicators, context.temp_allocator)
	ind_std := make([]f64, n_indicators, context.temp_allocator)
	for ind in 0 ..< n_indicators {
		for i in 0 ..< n_days {ind_mean[ind] += indicators[i * n_indicators + ind] / f64(n_days)}
		for i in 0 ..< n_days {
			d := indicators[i * n_indicators + ind] - ind_mean[ind]
			ind_std[ind] += d * d / f64(n_days)
		}
		ind_std[ind] = math.sqrt(ind_std[ind])
		if ind_std[ind] < 1e-6 {ind_std[ind] = 1}
	}

	sequences = make([]f64, n_sequences * seq_len * feature_dim, alloc)
	for s in 0 ..< n_sequences {
		for t in 0 ..< seq_len {
			row := sequences[(s * seq_len + t) * feature_dim:][:feature_dim]
			day := 1 + s + t
			row[0] = log_change(prices[day - 1], prices[day], 0.1)
			row[1] = log_change(volumes[day - 1], volumes[day], 2.0)
			for ind in 0 ..< n_indicators {
				z := (indicators[day * n_indicators + ind] - ind_mean[ind]) / ind_std[ind]
				row[2 + ind] = clamp(z, -3, 3)
			}
		}
	}
	return sequences, n_sequences, feature_dim
}

// ============================================================================
// Batches and losses (graph operations)
// ============================================================================

// A constant of the graph (it must live until backward, so the graph owns it)
@(private = "file")
_tg_const :: proc(n: int, shape: [4]int, alloc: mem.Allocator) -> ^tensor.Tensor {
	c := tensor.tensor_new(l.matrix_new(f64, 1, n, alloc), false, alloc)
	c.owned_by_graph = true
	c.shape = shape
	return c
}

// B sequences drawn uniformly with replacement, [B, T, F]
@(private = "file")
_tg_batch :: proc(tg: ^TimeGAN, data: []f64, B: int, alloc: mem.Allocator) -> ^tensor.Tensor {
	W := tg.seq_len * tg.feature_dim
	n_seqs := len(data) / W
	x := _tg_const(B * W, {B, tg.seq_len, tg.feature_dim, 1}, alloc)
	for b in 0 ..< B {
		i := rand.int_max(n_seqs)
		copy(x.data.data[b * W:][:W], data[i * W:][:W])
	}
	return x
}

// U[0, 1] noise [B, T, latent]
@(private = "file")
_tg_noise :: proc(tg: ^TimeGAN, B: int, alloc: mem.Allocator) -> ^tensor.Tensor {
	z := _tg_const(B * tg.seq_len * tg.latent_dim, {B, tg.seq_len, tg.latent_dim, 1}, alloc)
	for &v in z.data.data {v = rand.float64()}
	return z
}

@(private = "file")
_tg_full :: proc(like: ^tensor.Tensor, value: f64, alloc: mem.Allocator) -> ^tensor.Tensor {
	c := _tg_const(len(like.data.data), like.shape, alloc)
	for &v in c.data.data {v = value}
	return c
}

// x [B, T, D] (any layout of B·T·D values) as a [B, T·D] matrix
@(private = "file")
_tg_rows :: proc(x: ^tensor.Tensor, B: int) -> ^tensor.Tensor {
	n := len(x.data.data)
	r := tensor.tensor_reshape(x, {B, n / B, 1, 1})
	r.data.rows, r.data.cols = B, n / B
	return r
}

// The supervised loss MSE(H[:, 1:], H_sup[:, :-1]) of latents [B, T, L]: the supervisor's
// output at step t predicts the embedding at t + 1
timegan_supervised_loss :: proc(h, h_sup: ^tensor.Tensor, B: int) -> ^tensor.Tensor {
	L := h.shape[2]
	T := len(h.data.data) / (B * L)
	next := tensor.tensor_slice_cols(_tg_rows(h, B), L, (T - 1) * L)
	pred := tensor.tensor_slice_cols(_tg_rows(h_sup, B), 0, (T - 1) * L)
	return tensor.tensor_mse_loss(pred, next)
}

// The moment loss L_V: mean over (step, feature) of |sqrt(var(X^) + 1e-6) -
// sqrt(var(X) + 1e-6)| plus mean |mean(X^) - mean(X)|, the moments over the batch
// (population variance, as tf.nn.moments). x_hat and x [B, T, F].
timegan_moment_loss :: proc(
	x_hat, x: ^tensor.Tensor,
	B: int,
	alloc: mem.Allocator,
) -> ^tensor.Tensor {
	avg := tensor.tensor_new(l.matrix_new(f64, 1, B, alloc), false, alloc)
	avg.owned_by_graph = true
	for &v in avg.data.data {v = 1 / f64(B)}
	moments :: proc(
		x, avg: ^tensor.Tensor,
		B: int,
		alloc: mem.Allocator,
	) -> (
		mean, sd: ^tensor.Tensor,
	) {
		r := _tg_rows(x, B)
		mean = tensor.tensor_matmul(avg, r) // [1, T·F]
		variance := tensor.tensor_sub(
			tensor.tensor_matmul(avg, tensor.tensor_mul(r, r)),
			tensor.tensor_mul(mean, mean),
		)
		eps := _tg_full(variance, 1e-6, alloc)
		eps.data.rows, eps.data.cols = variance.data.rows, variance.data.cols
		sd = tensor.tensor_sqrt(tensor.tensor_add(variance, eps))
		return
	}
	abs_mean :: proc(d: ^tensor.Tensor, alloc: mem.Allocator) -> ^tensor.Tensor {
		// |d| = d · sign(d), the sign a constant (the gradient of |d| away from 0)
		s := tensor.tensor_new(l.matrix_new(f64, d.data.rows, d.data.cols, alloc), false, alloc)
		s.owned_by_graph = true
		s.shape = d.shape
		for v, i in d.data.data {s.data.data[i] = 1 if v > 0 else (-1 if v < 0 else 0)}
		return tensor.tensor_mean(tensor.tensor_mul(d, s))
	}
	m_hat, sd_hat := moments(x_hat, avg, B, alloc)
	m, sd := moments(x, avg, B, alloc)
	return tensor.tensor_add(
		abs_mean(tensor.tensor_sub(sd_hat, sd), alloc),
		abs_mean(tensor.tensor_sub(m_hat, m), alloc),
	)
}

// BCE of the discriminator's per-step probabilities against an all-`target` label
@(private = "file")
_tg_bce :: proc(p: ^tensor.Tensor, target: f64, alloc: mem.Allocator) -> ^tensor.Tensor {
	return tensor.tensor_bce_loss(p, _tg_full(p, target, alloc))
}

@(private = "file")
_tg_check :: proc(tg: ^TimeGAN, data: []f64) -> bool {
	if len(data) / (tg.seq_len * tg.feature_dim) == 0 {
		fmt.printf(
			"TimeGAN: no complete sequence in the data (seq_len=%d, feature_dim=%d)\n",
			tg.seq_len,
			tg.feature_dim,
		)
		return false
	}
	return true
}

// The joint-phase generator loss on real sequences x [B, T, F] and noise z [B, T, latent]:
// BCE(1, D(S(E^))) + γ BCE(1, D(E^)) + 100 sqrt(L_S) + 100 L_V, with E^ = G(z); also L_S
timegan_generator_loss :: proc(
	tg: ^TimeGAN,
	x, z: ^tensor.Tensor,
	alloc: mem.Allocator,
) -> (
	g_loss, l_s: ^tensor.Tensor,
) {
	B := x.shape[0]
	e_hat := nn.sequential_forward(tg.generator, z)
	h_hat := nn.sequential_forward(tg.supervisor, e_hat)
	x_hat := nn.sequential_forward(tg.recovery, h_hat)
	h := nn.sequential_forward(tg.embedder, x)
	l_s = timegan_supervised_loss(h, nn.sequential_forward(tg.supervisor, h), B)
	adv := tensor.tensor_add(
		_tg_bce(nn.sequential_forward(tg.discriminator, h_hat), 1, alloc),
		tensor.tensor_scale(
			_tg_bce(nn.sequential_forward(tg.discriminator, e_hat), 1, alloc),
			TIMEGAN_GAMMA,
		),
	)
	g_loss = tensor.tensor_add(
		adv,
		tensor.tensor_add(
			tensor.tensor_scale(tensor.tensor_sqrt(l_s), 100),
			tensor.tensor_scale(timegan_moment_loss(x_hat, x, B, alloc), 100),
		),
	)
	return
}

// The discriminator loss BCE(1, D(E(x))) + BCE(0, D(S(G(z)))) + γ BCE(0, D(G(z)))
timegan_discriminator_loss :: proc(
	tg: ^TimeGAN,
	x, z: ^tensor.Tensor,
	alloc: mem.Allocator,
) -> ^tensor.Tensor {
	h := nn.sequential_forward(tg.embedder, x)
	e_hat := nn.sequential_forward(tg.generator, z)
	h_hat := nn.sequential_forward(tg.supervisor, e_hat)
	return tensor.tensor_add(
		tensor.tensor_add(
			_tg_bce(nn.sequential_forward(tg.discriminator, h), 1, alloc),
			_tg_bce(nn.sequential_forward(tg.discriminator, h_hat), 0, alloc),
		),
		tensor.tensor_scale(
			_tg_bce(nn.sequential_forward(tg.discriminator, e_hat), 0, alloc),
			TIMEGAN_GAMMA,
		),
	)
}

// ============================================================================
// Training Steps (each on a batch of batch_size sequences of the scaled data; temporaries live in the model's graph scope)
// ============================================================================

// Phase 1: autoencoder; returns MSE(X, R(E(X))) (the step is on 10 sqrt of it)
train_embedder_recovery :: proc(tg: ^TimeGAN, real_data: []f64, batch_size: int) -> f64 {
	if !_tg_check(tg, real_data) {return 0}
	tensor.graph_begin(&tg.scope)
	defer tensor.graph_end(&tg.scope)
	alloc := tensor.graph_allocator(&tg.scope)
	x := _tg_batch(tg, real_data, batch_size, alloc)
	mse := tensor.tensor_mse_loss(
		nn.sequential_forward(tg.recovery, nn.sequential_forward(tg.embedder, x)),
		x,
	)
	nn.adam_zero_grad(&tg.optimizer_e)
	nn.adam_zero_grad(&tg.optimizer_r)
	tensor.tensor_backward(tensor.tensor_scale(tensor.tensor_sqrt(mse), 10))
	nn.adam_step(&tg.optimizer_e)
	nn.adam_step(&tg.optimizer_r)
	return mse.data.data[0]
}

// Phase 2: the supervisor on the embeddings of real data; returns L_S
train_supervisor :: proc(tg: ^TimeGAN, real_data: []f64, batch_size: int) -> f64 {
	if !_tg_check(tg, real_data) {return 0}
	tensor.graph_begin(&tg.scope)
	defer tensor.graph_end(&tg.scope)
	alloc := tensor.graph_allocator(&tg.scope)
	h := nn.sequential_forward(tg.embedder, _tg_batch(tg, real_data, batch_size, alloc))
	loss := timegan_supervised_loss(h, nn.sequential_forward(tg.supervisor, h), batch_size)
	nn.adam_zero_grad(&tg.optimizer_s)
	tensor.tensor_backward(loss)
	nn.adam_step(&tg.optimizer_s)
	return loss.data.data[0]
}

// Joint phase, generator and supervisor; returns the generator loss
train_generator :: proc(tg: ^TimeGAN, real_data: []f64, batch_size: int) -> f64 {
	if !_tg_check(tg, real_data) {return 0}
	tensor.graph_begin(&tg.scope)
	defer tensor.graph_end(&tg.scope)
	alloc := tensor.graph_allocator(&tg.scope)
	g_loss, _ := timegan_generator_loss(
		tg,
		_tg_batch(tg, real_data, batch_size, alloc),
		_tg_noise(tg, batch_size, alloc),
		alloc,
	)
	nn.adam_zero_grad(&tg.optimizer_g)
	nn.adam_zero_grad(&tg.optimizer_s)
	tensor.tensor_backward(g_loss)
	nn.adam_step(&tg.optimizer_g)
	nn.adam_step(&tg.optimizer_s)
	return g_loss.data.data[0]
}

// Joint phase, embedder and recovery: 10 sqrt(MSE) + 0.1 L_S; returns that loss
train_embedder_joint :: proc(tg: ^TimeGAN, real_data: []f64, batch_size: int) -> f64 {
	if !_tg_check(tg, real_data) {return 0}
	tensor.graph_begin(&tg.scope)
	defer tensor.graph_end(&tg.scope)
	alloc := tensor.graph_allocator(&tg.scope)
	x := _tg_batch(tg, real_data, batch_size, alloc)
	h := nn.sequential_forward(tg.embedder, x)
	mse := tensor.tensor_mse_loss(nn.sequential_forward(tg.recovery, h), x)
	l_s := timegan_supervised_loss(h, nn.sequential_forward(tg.supervisor, h), batch_size)
	loss := tensor.tensor_add(
		tensor.tensor_scale(tensor.tensor_sqrt(mse), 10),
		tensor.tensor_scale(l_s, 0.1),
	)
	nn.adam_zero_grad(&tg.optimizer_e)
	nn.adam_zero_grad(&tg.optimizer_r)
	tensor.tensor_backward(loss)
	nn.adam_step(&tg.optimizer_e)
	nn.adam_step(&tg.optimizer_r)
	return loss.data.data[0]
}

// Joint phase, discriminator: stepped only while its loss is above 0.15 (it would
// otherwise overpower the generator); returns the loss
train_discriminator :: proc(tg: ^TimeGAN, real_data: []f64, batch_size: int) -> f64 {
	if !_tg_check(tg, real_data) {return 0}
	tensor.graph_begin(&tg.scope)
	defer tensor.graph_end(&tg.scope)
	alloc := tensor.graph_allocator(&tg.scope)
	loss := timegan_discriminator_loss(
		tg,
		_tg_batch(tg, real_data, batch_size, alloc),
		_tg_noise(tg, batch_size, alloc),
		alloc,
	)
	if loss.data.data[0] > TIMEGAN_D_THRESHOLD {
		nn.adam_zero_grad(&tg.optimizer_d)
		tensor.tensor_backward(loss)
		nn.adam_step(&tg.optimizer_d)
	}
	return loss.data.data[0]
}

// One joint iteration as in the paper: twice (generator, embedder), then the discriminator
timegan_train_joint_step :: proc(
	tg: ^TimeGAN,
	real_data: []f64,
	batch_size: int,
) -> (
	g_loss, e_loss, d_loss: f64,
) {
	for _ in 0 ..< 2 {
		g_loss = train_generator(tg, real_data, batch_size)
		e_loss = train_embedder_joint(tg, real_data, batch_size)
	}
	d_loss = train_discriminator(tg, real_data, batch_size)
	return
}

// ============================================================================
// Generation
// ============================================================================

// n_sequences synthetic sequences R(S(G(Z))) [n, T, F], in the scaled units
generate_synthetic_data :: proc(tg: ^TimeGAN, n_sequences: int, alloc: mem.Allocator) -> []f64 {
	W := tg.seq_len * tg.feature_dim
	output := make([]f64, n_sequences * W, alloc)
	CHUNK :: 256
	for start := 0; start < n_sequences; start += CHUNK {
		B := min(CHUNK, n_sequences - start)
		tensor.graph_begin(&tg.scope, no_grad = true)
		a := tensor.graph_allocator(&tg.scope)
		e_hat := nn.sequential_forward(tg.generator, _tg_noise(tg, B, a))
		x_hat := nn.sequential_forward(tg.recovery, nn.sequential_forward(tg.supervisor, e_hat))
		copy(output[start * W:][:B * W], x_hat.data.data)
		tensor.graph_end(&tg.scope)
	}
	return output
}

// timegan_init_weights re-initializes the Linear and LSTM layers of all five
// TimeGAN networks: Xavier-uniform weights (fans = matrix rows and columns),
// zero biases.
timegan_init_weights :: proc(tg: ^TimeGAN) {
	init_weights :: proc(weights: ^tensor.Tensor) {
		if weights == nil {return}
		nn.xavier_uniform(weights.data.data, weights.data.rows, weights.data.cols)
	}
	zero :: proc(bias: ^tensor.Tensor) {
		if bias != nil {for &v in bias.data.data {v = 0}}
	}
	init_sequential :: proc(seq: ^nn.Sequential) {
		for layer in seq.layers {
			#partial switch l in layer {
			case nn.LinearLayer:
				init_weights(l.weights)
				zero(l.bias)
			case nn.LSTMLayer:
				init_weights(l.w_ih)
				init_weights(l.w_hh)
				zero(l.bias)
			}
		}
	}
	init_sequential(tg.embedder)
	init_sequential(tg.recovery)
	init_sequential(tg.generator)
	init_sequential(tg.discriminator)
	init_sequential(tg.supervisor)
}
