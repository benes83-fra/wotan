package audio_profile

import l "../../wotan/linalg"
import nn "../../wotan/nn"
import t "../../wotan/tensor"
import "core:math"
import "core:math/rand"

// FastSpeech 2 with the sizes of ming024/FastSpeech2 (LJSpeech config, 100 mel
// bins): phoneme embedding 256 + positions, encoder of 4 and decoder of 6 FFT
// blocks (self-attention with 2 heads and a padding mask, conv1d feed-forward
// 256 -> 1024 -> 256 with kernel 9 and 1, post-LayerNorm), duration, pitch and
// energy predictors (2 conv1d of kernel 3 with LayerNorm, then a linear),
// phoneme-level pitch and energy embeddings (256 bins), the length regulator,
// a mel projection and a 5-layer conv1d postnet (512 channels, kernel 5, tanh;
// without its batch norm). Losses: masked L1 on the mel before and after the
// postnet, masked MSE on the log durations, pitch and energy. Dropout as in the
// reference (0.2 in the blocks, 0.5 in the predictors).
FS2_PHONEMES :: 120 // 10 s of speech, about 12 phonemes per second
FS2_FRAMES :: 938 // 10 s at 24 kHz with hop 256
FS2_D :: 256
FS2_HEADS :: 2
FS2_FF :: 1024
FS2_MELS :: 100
FS2_VOCAB :: 100
FS2_BINS :: 256

FFT_Block :: struct {
	mha:          nn.MultiHeadAttentionLayer,
	ln1, ln2:     nn.LayerNormLayer,
	conv1, conv2: nn.Conv1dLayer,
}

Variance_Predictor :: struct {
	conv1, conv2: nn.Conv1dLayer,
	ln1, ln2:     nn.LayerNormLayer,
	out:          nn.LinearLayer,
}

FS2 :: struct {
	embed:                      nn.EmbeddingLayer,
	pos:                        nn.PositionalEncoding,
	encoder, decoder:           [dynamic]FFT_Block,
	duration, pitch, energy:    Variance_Predictor,
	pitch_embed, energy_embed:  nn.EmbeddingLayer,
	mel_out:                    nn.LinearLayer,
	postnet:                    [5]nn.Conv1dLayer,
	opt:                        nn.Adam,
	// one batch: phoneme ids, durations, pitch / energy bins and targets
	ids, pitch_ids, energy_ids: ^t.Tensor,
	src_key_mask, mel_key_mask: ^t.Tensor, // [B, S] 1 at real steps (attention)
	log_dur, pitch_t, energy_t: ^t.Tensor, // [B, S, 1] targets
	mel_t:                      ^t.Tensor, // [B, T, 100]
	durations:                  []int,
	src_mask, mel_mask:         []f64, // loss masks per step
}

fft_block_new :: proc() -> FFT_Block {
	return {
		mha = nn.multi_head_attention_layer_new(FS2_D, FS2_HEADS),
		ln1 = nn.layer_norm_layer_new(FS2_D),
		ln2 = nn.layer_norm_layer_new(FS2_D),
		conv1 = nn.conv1d_layer_new(FS2_D, FS2_FF, 9, padding = 4),
		conv2 = nn.conv1d_layer_new(FS2_FF, FS2_D, 1),
	}
}

fft_block_forward :: proc(b: ^FFT_Block, x, key_mask: ^t.Tensor) -> ^t.Tensor {
	a := t.tensor_dropout(nn.multi_head_attention_layer_forward(&b.mha, x, key_mask), 0.2, true)
	h := nn.layer_norm_layer_forward(&b.ln1, t.tensor_add(x, a))
	f := nn.conv1d_layer_forward(&b.conv2, t.tensor_relu(nn.conv1d_layer_forward(&b.conv1, h)))
	f = t.tensor_dropout(f, 0.2, true)
	return nn.layer_norm_layer_forward(&b.ln2, t.tensor_add(h, f))
}

variance_predictor_new :: proc() -> Variance_Predictor {
	return {
		conv1 = nn.conv1d_layer_new(FS2_D, FS2_D, 3, padding = 1),
		conv2 = nn.conv1d_layer_new(FS2_D, FS2_D, 3, padding = 1),
		ln1 = nn.layer_norm_layer_new(FS2_D),
		ln2 = nn.layer_norm_layer_new(FS2_D),
		out = nn.linear_layer_new(FS2_D, 1),
	}
}

variance_predictor_forward :: proc(v: ^Variance_Predictor, x: ^t.Tensor) -> ^t.Tensor {
	h := t.tensor_relu(nn.conv1d_layer_forward(&v.conv1, x))
	h = t.tensor_dropout(nn.layer_norm_layer_forward(&v.ln1, h), 0.5, true)
	h = t.tensor_relu(nn.conv1d_layer_forward(&v.conv2, h))
	h = t.tensor_dropout(nn.layer_norm_layer_forward(&v.ln2, h), 0.5, true)
	return nn.linear_forward(&v.out, h)
}

fs2_model :: proc() -> Model {
	rand.reset(7)
	f := new(FS2)
	f.embed = nn.embedding_layer_new(FS2_VOCAB, FS2_D)
	f.pos = nn.positional_encoding_new(FS2_FRAMES, FS2_D)
	for _ in 0 ..< 4 {append(&f.encoder, fft_block_new())}
	for _ in 0 ..< 6 {append(&f.decoder, fft_block_new())}
	f.duration, f.pitch, f.energy =
		variance_predictor_new(), variance_predictor_new(), variance_predictor_new()
	f.pitch_embed = nn.embedding_layer_new(FS2_BINS, FS2_D)
	f.energy_embed = nn.embedding_layer_new(FS2_BINS, FS2_D)
	f.mel_out = nn.linear_layer_new(FS2_D, FS2_MELS)
	widths := [6]int{FS2_MELS, 512, 512, 512, 512, FS2_MELS}
	for &c, i in f.postnet {c = nn.conv1d_layer_new(widths[i], widths[i + 1], 5, padding = 2)}

	m := Model {
		name = "FastSpeech 2 (10-s clips: 120 phonemes, 938 frames)",
		step = fs2_step,
		data = f,
	}
	ps := &m.params
	append(ps, f.embed.weight, f.pitch_embed.weight, f.energy_embed.weight)
	for &b in f.encoder {_fft_params(ps, &b)}
	for &b in f.decoder {_fft_params(ps, &b)}
	for v in ([]^Variance_Predictor{&f.duration, &f.pitch, &f.energy}) {
		_conv_params(ps, &v.conv1)
		_conv_params(ps, &v.conv2)
		_ln_params(ps, &v.ln1)
		_ln_params(ps, &v.ln2)
		_linear_params(ps, &v.out)
	}
	_linear_params(ps, &f.mel_out)
	for &c in f.postnet {_conv_params(ps, &c)}
	f.opt = nn.adam_new(1e-4)
	for p in m.params {nn.adam_add_param(&f.opt, p)}
	append(&m.opts, &f.opt)

	// a batch of lengths spread over 70 to 100 % of the longest
	B, S, T := batch, FS2_PHONEMES, FS2_FRAMES
	ids := make([]f64, B * S)
	pitch_ids, energy_ids := make([]f64, B * S), make([]f64, B * S)
	log_dur, pitch_t, energy_t := make([]f64, B * S), make([]f64, B * S), make([]f64, B * S)
	f.durations = make([]int, B * S)
	f.src_mask, f.mel_mask = make([]f64, B * S), make([]f64, B * T)
	for b in 0 ..< B {
		frac := 0.7 + 0.3 * f64(b) / f64(max(B - 1, 1))
		n_ph := int(frac * f64(S))
		n_fr := int(frac * f64(T))
		// durations: random weights scaled to n_fr frames, at least 1 each
		w := make([]f64, n_ph, context.temp_allocator)
		total := 0.0
		for &v in w {v = 0.5 + rand.float64(); total += v}
		left := n_fr
		for s in 0 ..< n_ph {
			d := max(1, int(w[s] / total * f64(n_fr)))
			if s == n_ph - 1 {d = max(1, left)}
			d = min(d, max(left - (n_ph - 1 - s), 1))
			f.durations[b * S + s] = d
			left -= d
		}
		for s in 0 ..< n_ph {
			i := b * S + s
			ids[i] = f64(1 + rand.int_max(FS2_VOCAB - 1))
			pitch_ids[i] = f64(rand.int_max(FS2_BINS))
			energy_ids[i] = f64(rand.int_max(FS2_BINS))
			log_dur[i] = math.ln(f64(f.durations[i]) + 1)
			pitch_t[i], energy_t[i] = rand.norm_float64(), rand.norm_float64()
			f.src_mask[i] = 1
		}
		for s in 0 ..< n_fr {f.mel_mask[b * T + s] = 1}
	}
	mel := make([]f64, B * T * FS2_MELS)
	for &v in mel {v = rand.norm_float64()}
	f.ids = _const(ids, {B, S, 1, 1}, B, S)
	f.pitch_ids = _const(pitch_ids, {B, S, 1, 1}, B, S)
	f.energy_ids = _const(energy_ids, {B, S, 1, 1}, B, S)
	f.src_key_mask = _const(f.src_mask, {B, S, 1, 1}, B, S)
	f.mel_key_mask = _const(f.mel_mask, {B, T, 1, 1}, B, T)
	f.log_dur = _const(log_dur, {B, S, 1, 1})
	f.pitch_t = _const(pitch_t, {B, S, 1, 1})
	f.energy_t = _const(energy_t, {B, S, 1, 1})
	f.mel_t = _const(mel, {B, T, FS2_MELS, 1})
	append(
		&m.inputs,
		f.ids,
		f.pitch_ids,
		f.energy_ids,
		f.src_key_mask,
		f.mel_key_mask,
		f.log_dur,
		f.pitch_t,
		f.energy_t,
		f.mel_t,
	)
	return m
}

fs2_step :: proc(m: ^Model, scope: ^t.Graph_Scope) -> f64 {
	f := (^FS2)(m.data)
	t.graph_begin(scope)
	x := nn.positional_encoding_forward(&f.pos, nn.embedding_layer_forward(&f.embed, f.ids))
	for &b in f.encoder {x = fft_block_forward(&b, x, f.src_key_mask)}

	// variance adaptor (teacher forcing: the target durations, pitch and energy)
	dur := variance_predictor_forward(&f.duration, x)
	pitch := variance_predictor_forward(&f.pitch, x)
	energy := variance_predictor_forward(&f.energy, x)
	x = t.tensor_add(x, nn.embedding_layer_forward(&f.pitch_embed, f.pitch_ids))
	x = t.tensor_add(x, nn.embedding_layer_forward(&f.energy_embed, f.energy_ids))
	y, _ := t.tensor_length_regulate(
		x,
		f.durations,
		FS2_FRAMES,
		allocator = t.graph_allocator(scope),
	)

	y = nn.positional_encoding_forward(&f.pos, y)
	for &b in f.decoder {y = fft_block_forward(&b, y, f.mel_key_mask)}
	mel := nn.linear_forward(&f.mel_out, y)
	mel = t.tensor_reshape(mel, f.mel_t.shape)
	h := mel
	for &c, i in f.postnet {
		h = nn.conv1d_layer_forward(&c, h)
		if i < 4 {h = t.tensor_dropout(t.tensor_tanh(h), 0.5, true)}
	}
	post := t.tensor_add(mel, h)

	sq :: proc(a, b: ^t.Tensor) -> ^t.Tensor {d := t.tensor_sub(a, b); return t.tensor_mul(d, d)}
	l1 :: proc(a, b: ^t.Tensor) -> ^t.Tensor {return t.tensor_abs(t.tensor_sub(a, b))}
	loss := t.tensor_masked_mean(l1(mel, f.mel_t), f.mel_mask)
	loss = t.tensor_add(loss, t.tensor_masked_mean(l1(post, f.mel_t), f.mel_mask))
	loss = t.tensor_add(
		loss,
		t.tensor_masked_mean(sq(t.tensor_reshape(dur, f.log_dur.shape), f.log_dur), f.src_mask),
	)
	loss = t.tensor_add(
		loss,
		t.tensor_masked_mean(sq(t.tensor_reshape(pitch, f.pitch_t.shape), f.pitch_t), f.src_mask),
	)
	loss = t.tensor_add(
		loss,
		t.tensor_masked_mean(
			sq(t.tensor_reshape(energy, f.energy_t.shape), f.energy_t),
			f.src_mask,
		),
	)
	t.tensor_backward(loss)
	value := t.tensor_item(loss)
	t.graph_end(scope)
	nn.adam_step(&f.opt)
	nn.adam_zero_grad(&f.opt)
	return value
}

// ---- shared helpers ----

// _const: a tensor without gradient holding a copy of `values`, with `shape`
// (stored as a [rows, cols] matrix when given, else flat)
_const :: proc(values: []f64, shape: [4]int, rows := 0, cols := 0) -> ^t.Tensor {
	r, c := 1, len(values)
	if rows > 0 {r, c = rows, cols}
	m := l.matrix_new(f64, r, c)
	copy(m.data, values)
	x := t.tensor_new(m, false)
	x.shape = shape
	return x
}

_linear_params :: proc(ps: ^[dynamic]^t.Tensor, x: ^nn.LinearLayer) {
	append(ps, x.weights)
	if x.bias != nil {append(ps, x.bias)}
}

_conv_params :: proc(ps: ^[dynamic]^t.Tensor, x: ^nn.Conv1dLayer) {
	append(ps, x.weight)
	if x.bias != nil {append(ps, x.bias)}
}

_conv2d_params :: proc(ps: ^[dynamic]^t.Tensor, x: ^nn.Conv2dLayer) {
	append(ps, x.weight)
	if x.bias != nil {append(ps, x.bias)}
}

_ln_params :: proc(ps: ^[dynamic]^t.Tensor, x: ^nn.LayerNormLayer) {
	append(ps, x.gamma)
	if x.beta != nil {append(ps, x.beta)}
}

_fft_params :: proc(ps: ^[dynamic]^t.Tensor, b: ^FFT_Block) {
	for lin in ([]^nn.LinearLayer{&b.mha.q_proj, &b.mha.k_proj, &b.mha.v_proj, &b.mha.out_proj}) {
		_linear_params(ps, lin)
	}
	_ln_params(ps, &b.ln1)
	_ln_params(ps, &b.ln2)
	_conv_params(ps, &b.conv1)
	_conv_params(ps, &b.conv2)
}
