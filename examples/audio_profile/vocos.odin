package audio_profile

import nn "../../wotan/nn"
import t "../../wotan/tensor"
import "core:math"
import "core:math/rand"

// Vocos with the sizes of the published 24 kHz model (charactr/vocos-mel-24khz):
// conv1d embedding 100 -> 512 (kernel 7), LayerNorm, 8 ConvNeXt blocks
// (depthwise conv1d kernel 7, LayerNorm, 512 -> 1536 -> 512 with GELU, residual;
// without the layer scale), LayerNorm, a linear head 512 -> 1026 (log magnitude
// and phase of 513 bins), magnitude exp(·) clamped at 100, and the inverse STFT
// (n_fft 1024, hop 256, Hann, "same" padding) as a matmul with the inverse DFT
// basis plus tensor_overlap_add. Mel loss: STFT of the output (reflect padding,
// strided conv1d with the windowed DFT basis), magnitude, a 100-band mel
// projection, log, L1 against the target log-mel.
// The GAN step adds the multi-period discriminator (periods 2, 3, 5, 7, 11;
// channels 32, 128, 512, 1024, 1024, kernel (5, 1), stride (3, 1)) and the
// multi-resolution discriminator (n_fft 512, 1024, 2048; 64 channels, kernels
// (7, 5), (5, 3), (5, 3), (3, 3), (3, 3)), leaky ReLU 0.1, least-squares GAN
// losses and feature matching, as Vocos trains: a discriminator update on the
// real and the (detached) generated audio, then a generator update with
// 45 · mel L1 + GAN + 2 · feature matching.
VOC_DIM :: 512
VOC_HIDDEN :: 1536
VOC_LAYERS :: 8
VOC_MELS :: 100
VOC_NFFT :: 1024
VOC_HOP :: 256
VOC_BINS :: VOC_NFFT / 2 + 1

ConvNeXt_Block :: struct {
	dw:       nn.Conv1dLayer,
	ln:       nn.LayerNormLayer,
	up, down: nn.LinearLayer,
}

Disc_Stack :: struct {
	convs:  [dynamic]nn.Conv2dLayer,
	post:   nn.Conv2dLayer,
	// multi-period: the period; multi-resolution: the STFT
	period: int,
	stft:   ^STFT,
}

STFT :: struct {
	n_fft, hop: int,
	basis:      ^t.Tensor, // [2·bins, 1, n_fft]: window·cos rows, then -window·sin
}

Vocos :: struct {
	embed:         nn.Conv1dLayer,
	ln_in, ln_out: nn.LayerNormLayer,
	blocks:        [dynamic]ConvNeXt_Block,
	head:          nn.LinearLayer,
	ibasis:        ^t.Tensor, // [2·bins, n_fft] inverse real DFT
	window:        []f64,
	stft:          ^STFT, // the mel loss's STFT (n_fft 1024, hop 256)
	mel_fb:        ^t.Tensor, // [bins, 100]
	discs:         [dynamic]Disc_Stack,
	opt_g, opt_d:  nn.Adam,
	frames:        int,
	mel_in:        ^t.Tensor, // [B, frames, 100] generator input
	audio:         ^t.Tensor, // [B, frames·hop, 1] real audio
	mel_t:         ^t.Tensor, // [B·(frames + 1), 100] target log-mel of the real audio
	ones:          map[[4]int]^t.Tensor, // targets of 1 for the least-squares losses
	zeros:         map[[4]int]^t.Tensor,
	gan:           bool,
}

_hann :: proc(n: int) -> []f64 {
	w := make([]f64, n)
	for &v, i in w {v = 0.5 - 0.5 * math.cos(2 * math.PI * f64(i) / f64(n))}
	return w
}

stft_new :: proc(n_fft, hop: int) -> ^STFT {
	s := new(STFT)
	s.n_fft, s.hop = n_fft, hop
	bins := n_fft / 2 + 1
	w := _hann(n_fft)
	defer delete(w)
	basis := make([]f64, 2 * bins * n_fft, context.temp_allocator)
	for k in 0 ..< bins {
		for n in 0 ..< n_fft {
			arg := 2 * math.PI * f64(k * n) / f64(n_fft)
			basis[k * n_fft + n] = w[n] * math.cos(arg)
			basis[(bins + k) * n_fft + n] = -w[n] * math.sin(arg)
		}
	}
	s.basis = _const(basis, {2 * bins, 1, n_fft, 1})
	return s
}

// magnitude spectrogram [B·F, bins] (as a matrix) of signals [B, L, 1], centred
// with reflect padding; F = L / hop + 1
stft_magnitude :: proc(s: ^STFT, x: ^t.Tensor) -> (mag: ^t.Tensor, F: int) {
	bins := s.n_fft / 2 + 1
	padded := t.tensor_pad_time(x, s.n_fft / 2, s.n_fft / 2, .Reflect)
	spec := t.tensor_conv1d(padded, s.basis, nil, stride = s.hop) // [B, F, 2·bins]
	F = spec.shape[1]
	sm := t.tensor_as_matrix(spec, spec.shape[0] * F, 2 * bins)
	re, im := t.tensor_slice_cols(sm, 0, bins), t.tensor_slice_cols(sm, bins, bins)
	power := t.tensor_add(t.tensor_mul(re, re), t.tensor_mul(im, im))
	return t.tensor_sqrt(t.tensor_clamp(power, 1e-9, math.F64_MAX)), F
}

// Slaney-style triangular mel filters [bins, mels] (0 to 12 kHz)
_mel_filterbank :: proc(bins, mels: int, sr: f64) -> []f64 {
	hz_to_mel :: proc(f: f64) -> f64 {return 2595 * math.log10(1 + f / 700)}
	mel_to_hz :: proc(m: f64) -> f64 {return 700 * (math.pow(10, m / 2595) - 1)}
	fb := make([]f64, bins * mels, context.temp_allocator)
	top := hz_to_mel(sr / 2)
	for m in 0 ..< mels {
		lo := mel_to_hz(top * f64(m) / f64(mels + 1))
		mid := mel_to_hz(top * f64(m + 1) / f64(mels + 1))
		hi := mel_to_hz(top * f64(m + 2) / f64(mels + 1))
		for k in 0 ..< bins {
			f := sr / 2 * f64(k) / f64(bins - 1)
			v := 0.0
			if f > lo &&
			   f <
				   mid {v = (f - lo) / (mid - lo)} else if f >= mid && f < hi {v = (hi - f) / (hi - mid)}
			fb[k * mels + m] = v
		}
	}
	return fb
}

vocos_model :: proc(gan: bool) -> Model {
	rand.reset(11)
	v := new(Vocos)
	v.gan = gan
	// 10-s clips (938 frames); the GAN step trains on 1-s segments (93 frames)
	v.frames = 93 if gan else 938
	v.embed = nn.conv1d_layer_new(VOC_MELS, VOC_DIM, 7, padding = 3)
	v.ln_in = nn.layer_norm_layer_new(VOC_DIM, 1e-6)
	v.ln_out = nn.layer_norm_layer_new(VOC_DIM, 1e-6)
	for _ in 0 ..< VOC_LAYERS {
		append(
			&v.blocks,
			ConvNeXt_Block {
				dw = nn.conv1d_layer_new(VOC_DIM, VOC_DIM, 7, padding = 3, groups = VOC_DIM),
				ln = nn.layer_norm_layer_new(VOC_DIM, 1e-6),
				up = nn.linear_layer_new(VOC_DIM, VOC_HIDDEN),
				down = nn.linear_layer_new(VOC_HIDDEN, VOC_DIM),
			},
		)
	}
	v.head = nn.linear_layer_new(VOC_DIM, 2 * VOC_BINS)
	v.window = _hann(VOC_NFFT)
	ib := make([]f64, 2 * VOC_BINS * VOC_NFFT, context.temp_allocator)
	for k in 0 ..< VOC_BINS {
		c := 1.0 if k == 0 || k == VOC_BINS - 1 else 2.0
		for n in 0 ..< VOC_NFFT {
			arg := 2 * math.PI * f64(k * n) / VOC_NFFT
			ib[k * VOC_NFFT + n] = c * math.cos(arg) / VOC_NFFT
			ib[(VOC_BINS + k) * VOC_NFFT + n] = -c * math.sin(arg) / VOC_NFFT
		}
	}
	v.ibasis = _const(ib, {2 * VOC_BINS, VOC_NFFT, 1, 1}, 2 * VOC_BINS, VOC_NFFT)
	v.stft = stft_new(VOC_NFFT, VOC_HOP)
	fb := _mel_filterbank(VOC_BINS, VOC_MELS, 24000)
	v.mel_fb = _const(fb, {VOC_BINS, VOC_MELS, 1, 1}, VOC_BINS, VOC_MELS)

	m := Model {
		name = "Vocos GAN step (1-s segments: 93 frames)" if gan else "Vocos generator + mel loss (10-s clips: 938 frames)",
		step = vocos_step,
		data = v,
	}
	gen := &m.params
	_conv_params(gen, &v.embed)
	_ln_params(gen, &v.ln_in)
	_ln_params(gen, &v.ln_out)
	for &b in v.blocks {
		_conv_params(gen, &b.dw)
		_ln_params(gen, &b.ln)
		_linear_params(gen, &b.up)
		_linear_params(gen, &b.down)
	}
	_linear_params(gen, &v.head)
	n_gen := len(m.params)
	v.opt_g = nn.adam_new(2e-4, 0.8, 0.9)
	for p in m.params[:n_gen] {nn.adam_add_param(&v.opt_g, p)}
	append(&m.opts, &v.opt_g)

	if gan {
		for period in ([]int{2, 3, 5, 7, 11}) {
			d := Disc_Stack {
				period = period,
			}
			ch := [5]int{1, 32, 128, 512, 1024}
			for i in 0 ..< 4 {
				append(
					&d.convs,
					nn.conv2d_layer_new(
						ch[i],
						ch[i + 1],
						5,
						3,
						2,
						kernel_w = 1,
						stride_w = 1,
						padding_w = 0,
					),
				)
			}
			append(
				&d.convs,
				nn.conv2d_layer_new(
					1024,
					1024,
					5,
					1,
					2,
					kernel_w = 1,
					stride_w = 1,
					padding_w = 0,
				),
			)
			d.post = nn.conv2d_layer_new(
				1024,
				1,
				3,
				1,
				1,
				kernel_w = 1,
				stride_w = 1,
				padding_w = 0,
			)
			append(&v.discs, d)
		}
		for n_fft in ([]int{512, 1024, 2048}) {
			d := Disc_Stack {
				stft = stft_new(n_fft, n_fft / 4),
			}
			append(
				&d.convs,
				nn.conv2d_layer_new(1, 64, 7, 2, 3, kernel_w = 5, stride_w = 2, padding_w = 2),
			)
			append(
				&d.convs,
				nn.conv2d_layer_new(64, 64, 5, 2, 2, kernel_w = 3, stride_w = 1, padding_w = 1),
			)
			append(
				&d.convs,
				nn.conv2d_layer_new(64, 64, 5, 2, 2, kernel_w = 3, stride_w = 2, padding_w = 1),
			)
			append(
				&d.convs,
				nn.conv2d_layer_new(64, 64, 3, 2, 1, kernel_w = 3, stride_w = 1, padding_w = 1),
			)
			append(
				&d.convs,
				nn.conv2d_layer_new(64, 64, 3, 2, 1, kernel_w = 3, stride_w = 2, padding_w = 1),
			)
			d.post = nn.conv2d_layer_new(64, 1, 3, 1, 1)
			append(&v.discs, d)
			append(&m.inputs, d.stft.basis)
		}
		for &d in v.discs {
			for &c in d.convs {_conv2d_params(&m.params, &c)}
			_conv2d_params(&m.params, &d.post)
		}
		v.opt_d = nn.adam_new(2e-4, 0.8, 0.9)
		for p in m.params[n_gen:] {nn.adam_add_param(&v.opt_d, p)}
		append(&m.opts, &v.opt_d)
	}

	// a batch: input mels, real audio and its target log-mel (random values;
	// the cost does not depend on them)
	B, T := batch, v.frames
	mel := make([]f64, B * T * VOC_MELS, context.temp_allocator)
	for &x in mel {x = rand.norm_float64()}
	audio := make([]f64, B * T * VOC_HOP, context.temp_allocator)
	for &x in audio {x = 0.1 * rand.norm_float64()}
	target := make([]f64, B * (T + 1) * VOC_MELS, context.temp_allocator)
	for &x in target {x = rand.norm_float64() - 4}
	v.mel_in = _const(mel, {B, T, VOC_MELS, 1})
	v.audio = _const(audio, {B, T * VOC_HOP, 1, 1})
	v.mel_t = _const(target, {B * (T + 1), VOC_MELS, 1, 1}, B * (T + 1), VOC_MELS)
	append(&m.inputs, v.ibasis, v.stft.basis, v.mel_fb, v.mel_in, v.audio, v.mel_t)
	return m
}

// generated audio [B, frames·hop, 1]
vocos_generate :: proc(v: ^Vocos) -> ^t.Tensor {
	B, T := batch, v.frames
	x := nn.layer_norm_layer_forward(&v.ln_in, nn.conv1d_layer_forward(&v.embed, v.mel_in))
	for &b in v.blocks {
		h := nn.conv1d_layer_forward(&b.dw, x)
		h = nn.layer_norm_layer_forward(&b.ln, h)
		h = nn.linear_forward(&b.down, t.tensor_gelu(nn.linear_forward(&b.up, h)))
		x = t.tensor_add(x, t.tensor_reshape(h, x.shape))
	}
	x = nn.layer_norm_layer_forward(&v.ln_out, x)
	h := t.tensor_as_matrix(nn.linear_forward(&v.head, x), B * T, 2 * VOC_BINS)
	mag := t.tensor_clamp(t.tensor_exp(t.tensor_slice_cols(h, 0, VOC_BINS)), 0, 100)
	phase := t.tensor_slice_cols(h, VOC_BINS, VOC_BINS)
	re := t.tensor_mul(mag, t.tensor_cos(phase))
	im := t.tensor_mul(mag, t.tensor_sin(phase))
	spec := t.tensor_as_matrix(t.tensor_concat({re, im}, 1), B * T, 2 * VOC_BINS)
	frames := t.tensor_reshape(t.tensor_matmul(spec, v.ibasis), {B, T, VOC_NFFT, 1})
	// "same" padding: drop (n_fft - hop) / 2 samples at each end
	return t.tensor_overlap_add(frames, VOC_HOP, v.window, start = (VOC_NFFT - VOC_HOP) / 2)
}

vocos_mel_loss :: proc(v: ^Vocos, audio: ^t.Tensor) -> ^t.Tensor {
	mag, _ := stft_magnitude(v.stft, audio)
	logmel := t.tensor_log(t.tensor_clamp(t.tensor_matmul(mag, v.mel_fb), 1e-5, math.F64_MAX))
	return t.tensor_l1_loss(logmel, v.mel_t)
}

// a discriminator's score map and its feature maps (every conv output)
disc_forward :: proc(d: ^Disc_Stack, audio: ^t.Tensor, feats: ^[dynamic]^t.Tensor) -> ^t.Tensor {
	x: ^t.Tensor
	if d.stft == nil {
		// multi-period: [B, L] -> [B, 1, L / p, p] (reflect-padded to a multiple of p)
		B, L := audio.shape[0], audio.shape[1]
		pad := (d.period - L % d.period) % d.period
		a := audio
		if pad > 0 {a = t.tensor_pad_time(audio, 0, pad, .Reflect)}
		x = t.tensor_reshape(a, {B, 1, (L + pad) / d.period, d.period})
	} else {
		mag, F := stft_magnitude(d.stft, audio)
		x = t.tensor_reshape(mag, {audio.shape[0], 1, F, d.stft.n_fft / 2 + 1})
	}
	for &c in d.convs {
		x = t.tensor_leaky_relu(nn.conv2d_layer_forward(&c, x), 0.1)
		append(feats, x)
	}
	x = nn.conv2d_layer_forward(&d.post, x)
	append(feats, x)
	return x
}

// a constant tensor of `value` shaped like x (cached per shape)
_filled :: proc(
	cache: ^map[[4]int]^t.Tensor,
	x: ^t.Tensor,
	value: f64,
	on_device: bool,
) -> ^t.Tensor {
	if c, ok := cache[x.shape]; ok {return c}
	vals := make([]f64, t.tensor_numel(x), context.temp_allocator)
	for &e in vals {e = value}
	c := _const(vals, x.shape)
	if on_device {_to_device(c)}
	cache[x.shape] = c
	return c
}

vocos_step :: proc(m: ^Model, scope: ^t.Graph_Scope) -> f64 {
	v := (^Vocos)(m.data)
	on_device := t.tensor_is_on_device(v.mel_in)
	if !v.gan {
		t.graph_begin(scope)
		loss := vocos_mel_loss(v, vocos_generate(v))
		t.tensor_backward(loss)
		value := t.tensor_item(loss)
		t.graph_end(scope)
		nn.adam_step(&v.opt_g)
		nn.adam_zero_grad(&v.opt_g)
		return value
	}

	// discriminator update on the real and the detached generated audio
	t.graph_begin(scope)
	fake := t.tensor_detach(vocos_generate(v))
	feats := make([dynamic]^t.Tensor, t.graph_allocator(scope))
	loss_d: ^t.Tensor
	for &d in v.discs {
		real_score := disc_forward(&d, v.audio, &feats)
		fake_score := disc_forward(&d, fake, &feats)
		l := t.tensor_add(
			t.tensor_mse_loss(real_score, _filled(&v.ones, real_score, 1, on_device)),
			t.tensor_mse_loss(fake_score, _filled(&v.zeros, fake_score, 0, on_device)),
		)
		loss_d = l if loss_d == nil else t.tensor_add(loss_d, l)
	}
	t.tensor_backward(loss_d)
	value := t.tensor_item(loss_d)
	t.graph_end(scope)
	nn.adam_step(&v.opt_d)
	nn.adam_zero_grad(&v.opt_d)

	// generator update: 45 · mel L1 + least-squares GAN + 2 · feature matching
	t.graph_begin(scope)
	audio := vocos_generate(v)
	loss := t.tensor_scale(vocos_mel_loss(v, audio), 45)
	for &d in v.discs {
		real_feats := make([dynamic]^t.Tensor, t.graph_allocator(scope))
		fake_feats := make([dynamic]^t.Tensor, t.graph_allocator(scope))
		disc_forward(&d, v.audio, &real_feats)
		score := disc_forward(&d, audio, &fake_feats)
		loss = t.tensor_add(loss, t.tensor_mse_loss(score, _filled(&v.ones, score, 1, on_device)))
		for f, i in fake_feats {
			fm := t.tensor_l1_loss(f, t.tensor_detach(real_feats[i]))
			loss = t.tensor_add(loss, t.tensor_scale(fm, 2))
		}
	}
	t.tensor_backward(loss)
	value += t.tensor_item(loss)
	t.graph_end(scope)
	nn.adam_step(&v.opt_g)
	nn.adam_zero_grad(&v.opt_g)
	// the discriminators' gradients from the generator update are not used
	nn.adam_zero_grad(&v.opt_d)
	return value
}
