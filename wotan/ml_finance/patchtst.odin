package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// PatchTST (Nie, Nguyen, Sinthong & Kalagnanam 2023), supervised, as the official
// PatchTST_backbone with norm = "LayerNorm", res_attention = False, no dropout
// ============================================================================
// Channel independence: every row of x [R, input_size] is one channel of one sample,
// forecast on its own with shared weights; channels[r] picks RevIN's affine parameters.
//   RevIN       x <- (x - mean) / sqrt(var + 1e-5) (statistics per row, not trained
//               through), then x w_c + b_c; the forecast is mapped back:
//               (y - b_c) / (w_c + 1e-10) sqrt(var + 1e-5) + mean
//   patching    the series padded at the end with stride copies of its last value,
//               n_patches = (input_size - patch_len) / stride + 2 windows of patch_len
//   embedding   linear patch_len -> d_model, plus a learned positional encoding
//               [n_patches, d_model] (initialised U(-0.02, 0.02))
//   encoder     nn.TransformerEncoder: post-norm blocks, multi-head attention, LayerNorm,
//               GELU feed-forward (the official TSTEncoderLayer with LayerNorm)
//   head        flatten the patches' encodings, linear -> horizon. The features are
//               ordered patch-major (the official flattens d_model-major: the same model
//               up to a permutation of the head's weight rows)

PatchTSTConfig :: struct {
	input_size: int,
	horizon:    int,
	patch_len:  int, // 16 in the paper
	stride:     int, // 8
	d_model:    int, // 128
	n_heads:    int, // 16
	d_ff:       int, // 256
	n_layers:   int, // 3
	channels:   int, // RevIN affine parameter pairs (1 for univariate)
	revin:      bool,
}

PatchTST :: struct {
	config:    PatchTSTConfig,
	n_patches: int,
	embed:     nn.LinearLayer, // [patch_len, d_model]
	pos:       ^t.Tensor, // [n_patches, d_model]
	encoder:   nn.TransformerEncoder,
	head:      nn.LinearLayer, // [n_patches d_model, horizon]
	revin_w:   ^t.Tensor, // [channels, 1]
	revin_b:   ^t.Tensor, // [channels, 1]
	gather:    ^t.Tensor, // [input_size, n_patches patch_len]: padding and unfolding
	allocator: mem.Allocator,
	scope:     t.Graph_Scope,
}

patchtst_new :: proc(config: PatchTSTConfig, allocator := context.allocator) -> ^PatchTST {
	c := config
	assert(
		c.input_size >= c.patch_len && c.patch_len > 0 && c.stride > 0 && c.horizon > 0,
		"patchtst_new: need input_size >= patch_len > 0, stride > 0, horizon > 0",
	)
	assert(c.d_model % c.n_heads == 0 && c.n_layers > 0 && c.channels > 0, "patchtst_new: sizes")
	m := new(PatchTST, allocator)
	m.config = c
	m.allocator = allocator
	m.n_patches = (c.input_size - c.patch_len) / c.stride + 2
	np, P := m.n_patches, c.patch_len
	m.embed = nn.linear_layer_new(P, c.d_model, allocator)
	pm := l.matrix_new(f64, np, c.d_model, allocator)
	for &v in pm.data {v = rand.float64_uniform(-0.02, 0.02)}
	m.pos = t.tensor_new(pm, true, allocator)
	m.encoder = nn.transformer_encoder_new(c.n_layers, c.d_model, c.n_heads, c.d_ff, allocator)
	m.head = nn.linear_layer_new(np * c.d_model, c.horizon, allocator)
	w := l.matrix_new(f64, c.channels, 1, allocator)
	for &v in w.data {v = 1}
	m.revin_w = t.tensor_new(w, true, allocator)
	m.revin_b = t.tensor_new(l.matrix_new(f64, c.channels, 1, allocator), true, allocator)
	// padded[j] = x[min(j, L - 1)], patch p takes padded[p S .. p S + P - 1]
	g := l.matrix_new(f64, c.input_size, np * P, allocator)
	for p in 0 ..< np {
		for k in 0 ..< P {
			src := min(p * c.stride + k, c.input_size - 1)
			g.data[src * (np * P) + p * P + k] = 1
		}
	}
	m.gather = t.tensor_new(g, false, allocator)
	return m
}

patchtst_free :: proc(m: ^PatchTST) {
	nn.linear_layer_free(&m.embed)
	t.tensor_free(m.pos)
	nn.transformer_encoder_free(&m.encoder)
	nn.linear_layer_free(&m.head)
	t.tensor_free(m.revin_w)
	t.tensor_free(m.revin_b)
	t.tensor_free(m.gather)
	t.graph_scope_destroy(&m.scope)
	free(m, m.allocator)
}

// Trainable tensors: embedding (W, b), positional encoding, per encoder block attention
// (q, k, v, out: W, b), feed-forward (fc1, fc2: W, b), LayerNorms (gamma, beta) x 2, head
// (W, b), RevIN (w, b)
patchtst_parameters :: proc(m: ^PatchTST, allocator := context.allocator) -> [dynamic]^t.Tensor {
	p := make([dynamic]^t.Tensor, allocator)
	append(&p, m.embed.weights, m.embed.bias, m.pos)
	for &b in m.encoder.blocks {
		for lin in ([]^nn.LinearLayer{&b.mha.q_proj, &b.mha.k_proj, &b.mha.v_proj, &b.mha.out_proj, &b.ffn.fc1, &b.ffn.fc2}) {
			append(&p, lin.weights, lin.bias)
		}
		append(&p, b.ln1.gamma, b.ln1.beta, b.ln2.gamma, b.ln2.beta)
	}
	append(&p, m.head.weights, m.head.bias)
	if m.config.revin {append(&p, m.revin_w, m.revin_b)}
	return p
}

@(private = "file")
_pt_const :: proc(rows, cols: int, alloc: mem.Allocator) -> ^t.Tensor {
	c := t.tensor_new(l.matrix_new(f64, rows, cols, alloc), false, alloc)
	c.owned_by_graph = true
	return c
}

// Per-row RevIN affine parameter broadcast to [R, width] (graph): onehot(channels) w 1'
@(private = "file")
_pt_affine :: proc(
	param: ^t.Tensor,
	channels: []int,
	R, width: int,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	C := param.data.rows
	E := _pt_const(R, C, alloc)
	for r in 0 ..< R {E.data.data[r * C + (channels[r] if channels != nil else 0)] = 1}
	ones := _pt_const(1, width, alloc)
	for &v in ones.data.data {v = 1}
	return t.tensor_matmul(t.tensor_matmul(E, param), ones)
}

// Forecasts (graph), data [1, R horizon] with shape [R, horizon, 1, 1], for x [R, input_size]
patchtst_forward :: proc(
	m: ^PatchTST,
	x: ^t.Tensor,
	channels: []int,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	c := m.config
	R, L, H, np, d := x.data.rows, c.input_size, c.horizon, m.n_patches, c.d_model
	assert(x.data.cols == L, "patchtst_forward: x must be [rows, input_size]")
	assert(channels == nil || len(channels) == R, "patchtst_forward: one channel per row")
	// RevIN statistics (constants)
	mean := make([]f64, R, context.temp_allocator)
	defer delete(mean, context.temp_allocator)
	sd := make([]f64, R, context.temp_allocator)
	defer delete(sd, context.temp_allocator)
	xn := _pt_const(R, L, alloc)
	for r in 0 ..< R {
		row := x.data.data[r * L:][:L]
		for v in row {mean[r] += v / f64(L)}
		vv := 0.0
		for v in row {vv += (v - mean[r]) * (v - mean[r]) / f64(L)}
		sd[r] = math.sqrt(vv + 1e-5) if c.revin else 1
		if !c.revin {mean[r] = 0}
		for v, j in row {xn.data.data[r * L + j] = (v - mean[r]) / sd[r]}
	}
	z: ^t.Tensor = xn
	if c.revin {
		z = t.tensor_add(
			t.tensor_mul(xn, _pt_affine(m.revin_w, channels, R, L, alloc)),
			_pt_affine(m.revin_b, channels, R, L, alloc),
		)
	}
	// patches [R, np P] -> [R, np, P] -> embedding [R, np, d]
	u := t.tensor_reshape(t.tensor_matmul(z, m.gather), {R, np, c.patch_len, 1})
	e := nn.linear_forward(&m.embed, u)
	// + positional encoding, the same for every row
	rep := _pt_const(R * np, np, alloc)
	for r in 0 ..< R {for p in 0 ..< np {rep.data.data[(r * np + p) * np + p] = 1}}
	pos := t.tensor_reshape(t.tensor_matmul(rep, m.pos), {R, np, d, 1})
	h := nn.transformer_encoder_forward(&m.encoder, t.tensor_add(e, pos))
	y := nn.linear_forward(&m.head, t.tensor_reshape(h, {R, np * d, 1, 1})) // [R, H]
	if R == 1 {y = t.tensor_reshape(y, {1, H, 1, 1})}
	// back to the series' scale
	sdm := _pt_const(1, R * H, alloc)
	mm := _pt_const(1, R * H, alloc)
	for r in 0 ..< R {for j in 0 ..< H {sdm.data.data[r * H + j], mm.data.data[r * H + j] = sd[r], mean[r]}}
	sdm.shape, mm.shape = {R, H, 1, 1}, {R, H, 1, 1}
	if c.revin {
		eps := _pt_const(1, R * H, alloc)
		for &v in eps.data.data {v = 1e-10}
		eps.shape = {R, H, 1, 1}
		bw := t.tensor_reshape(_pt_affine(m.revin_b, channels, R, H, alloc), {R, H, 1, 1})
		ww := t.tensor_reshape(_pt_affine(m.revin_w, channels, R, H, alloc), {R, H, 1, 1})
		y = t.tensor_div(t.tensor_sub(y, bw), t.tensor_add(ww, eps))
	}
	return t.tensor_add(t.tensor_mul(y, sdm), mm)
}

// MSE against y [R, horizon] and one Adam step; the loss before the step
patchtst_train_step :: proc(m: ^PatchTST, x, y: ^t.Tensor, channels: []int, opt: ^nn.Adam) -> f64 {
	t.graph_begin(&m.scope)
	defer t.graph_end(&m.scope)
	alloc := t.graph_allocator(&m.scope)
	loss := patchtst_loss(m, x, y, channels, alloc)
	nn.adam_zero_grad(opt)
	t.tensor_backward(loss)
	nn.adam_step(opt)
	return loss.data.data[0]
}

// MSE of the forecasts against y [R, horizon] (graph)
patchtst_loss :: proc(
	m: ^PatchTST,
	x, y: ^t.Tensor,
	channels: []int,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	f := patchtst_forward(m, x, channels, alloc)
	target := _pt_const(f.data.rows, f.data.cols, alloc)
	copy(target.data.data, y.data.data)
	return t.tensor_mse_loss(f, target, alloc)
}

// Forecasts [R * horizon] row-major, no graph
patchtst_predict :: proc(
	m: ^PatchTST,
	x: ^t.Tensor,
	channels: []int,
	allocator := context.allocator,
) -> []f64 {
	t.graph_begin(&m.scope, no_grad = true)
	defer t.graph_end(&m.scope)
	f := patchtst_forward(m, x, channels, t.graph_allocator(&m.scope))
	out := make([]f64, len(f.data.data), allocator)
	copy(out, f.data.data)
	return out
}
