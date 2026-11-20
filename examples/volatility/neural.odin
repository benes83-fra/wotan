package main

// Stage 2b: the sequence models. The last SEQ days of five features go
// through a sequence block (LSTM or Mamba), then two dense layers, to a
// forecast of ln(volatility over the next HORIZON days). The features and the
// target are standardised with the training days' mean and standard deviation.
// The forecast is exp(·) of the scaled-back output times Duan's smearing factor:
// exp of a mean log is the median, not the mean, of a skewed volatility.
// Trained with Adam on the MSE, on the GPU when there is one.

import gpu "../../wotan/gpu"
import l "../../wotan/linalg"
import nn "../../wotan/nn"
import t "../../wotan/tensor"
import "core:math"
import "core:math/rand"

SEQ :: 22 // days the model sees
N_FEATURES :: 5
HIDDEN :: 32
EPOCHS :: 15
BATCH :: 128
LEARNING_RATE :: 1e-3
FIRST_FEATURE_DAY :: 22 // the 22-day volatility needs 22 returns
FIRST_SAMPLE :: FIRST_FEATURE_DAY + SEQ - 1

FEATURE_NAMES := [N_FEATURES]string {
	"return",
	"|return|",
	"ln realised vol 5d",
	"ln realised vol 22d",
	"ln VIX",
}

// feature: the five features of day d, known at its close
feature :: proc(m: ^Market, d: int, out: []f64) {
	r := 100 * m.ret[d]
	out[0], out[1] = r, abs(r)
	out[2] = math.ln(realised_vol(m, d, 5))
	out[3] = math.ln(realised_vol(m, d, 22))
	out[4] = math.ln(m.vix[d])
}

Net :: struct {
	model:      ^nn.Sequential,
	fmean, fsd: [N_FEATURES]f64,
	ymean, ysd: f64,
	smear:      f64,
}

net_new :: proc(backend: nn.SeqBackend) -> ^nn.Sequential {
	model := nn.sequential_new()
	// LSTM: features -> HIDDEN; Mamba: a linear map to HIDDEN, then the block
	nn.sequential_add_seq_block(model, backend, N_FEATURES, HIDDEN)
	nn.sequential_add(model, nn.FlattenLayer{}) // [B, SEQ · HIDDEN]
	nn.sequential_add(model, nn.linear_layer_new(SEQ * HIDDEN, HIDDEN))
	nn.sequential_add(model, nn.Activation.ReLU)
	nn.sequential_add(model, nn.linear_layer_new(HIDDEN, 1))
	return model
}

// window_into: the standardised features of days d - SEQ + 1 ..= d
window_into :: proc(net: ^Net, m: ^Market, d: int, dst: []f64) {
	for k in 0 ..< SEQ {
		row := dst[k * N_FEATURES:][:N_FEATURES]
		feature(m, d - SEQ + 1 + k, row)
		for j in 0 ..< N_FEATURES {row[j] = (row[j] - net.fmean[j]) / net.fsd[j]}
	}
}

// net_fold trains a fresh model on the samples ending on days
// FIRST_SAMPLE ..= s - HORIZON - 1 and writes its forecasts for days s ..< e
// to out. Returns the last epoch's mean loss (in units of the target's variance).
net_fold :: proc(
	m: ^Market,
	backend: nn.SeqBackend,
	s, e: int,
	on_gpu: bool,
	seed: u64,
	out: []f64,
) -> (
	loss: f64,
) {
	rand.reset(seed)
	net := Net {
		model = net_new(backend),
	}
	defer nn.sequential_free(net.model)
	last := s - HORIZON - 1
	n := last - FIRST_SAMPLE + 1

	// standardisation from the training days
	feat: [N_FEATURES]f64
	rows := 0
	for d in FIRST_FEATURE_DAY ..= last {
		feature(m, d, feat[:])
		for j in 0 ..< N_FEATURES {net.fmean[j] += feat[j]; net.fsd[j] += feat[j] * feat[j]}
		rows += 1
	}
	for j in 0 ..< N_FEATURES {
		net.fmean[j] /= f64(rows)
		net.fsd[j] = math.sqrt(max(net.fsd[j] / f64(rows) - net.fmean[j] * net.fmean[j], 1e-12))
	}
	y := make([]f64, n, context.temp_allocator)
	for i in 0 ..< n {y[i] = math.ln(m.rv_fwd[FIRST_SAMPLE + i]); net.ymean += y[i]}
	net.ymean /= f64(n)
	for v in y {net.ysd += (v - net.ymean) * (v - net.ymean)}
	net.ysd = math.sqrt(net.ysd / f64(n))
	for &v in y {v = (v - net.ymean) / net.ysd}

	params := nn.sequential_parameters(net.model)
	if on_gpu {gpu.to_device(params[:])}
	opt := nn.adam_new(LEARNING_RATE)
	defer nn.adam_free(&opt)
	for p in params {nn.adam_add_param(&opt, p)}

	W :: SEQ * N_FEATURES
	x := t.tensor_new(l.matrix_new(f64, 1, BATCH * W))
	x.shape = {BATCH, SEQ, N_FEATURES, 1}
	yt := t.tensor_new(l.matrix_new(f64, BATCH, 1))
	defer {t.tensor_free(x); t.tensor_free(yt)}
	order := make([]int, n, context.temp_allocator)
	for &v, i in order {v = i}
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for epoch in 0 ..< EPOCHS {
		rand.shuffle(order)
		sum, nb := 0.0, 0
		for b := 0; b + BATCH <= n; b += BATCH {
			xs, ys := t.tensor_values(x), t.tensor_values(yt)
			for i, k in order[b:][:BATCH] {
				window_into(&net, m, FIRST_SAMPLE + i, xs[k * W:][:W])
				ys[k] = y[i]
			}
			t.tensor_mark_host_dirty(x) // the next GPU op uploads the new batch
			t.tensor_mark_host_dirty(yt)
			t.graph_begin(&scope)
			batch_loss := t.tensor_mse_loss(nn.sequential_forward(net.model, x), yt)
			t.tensor_backward(batch_loss)
			// reading the loss waits for the GPU: only in the last epoch
			if epoch == EPOCHS - 1 {sum += t.tensor_item(batch_loss); nb += 1}
			t.graph_end(&scope)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)
		}
		loss = sum / f64(max(nb, 1))
	}

	// Duan's smearing factor: the mean of exp(residual) on the training days
	train_days := make([]int, n, context.temp_allocator)
	for &d, i in train_days {d = FIRST_SAMPLE + i}
	fit := predict_z(&net, m, train_days)
	net.smear = 0
	for z, i in fit {net.smear += math.exp(net.ysd * (y[i] - z)) / f64(n)}

	test_days := make([]int, e - s, context.temp_allocator)
	for &d, i in test_days {d = s + i}
	for z, i in predict_z(&net, m, test_days) {
		out[s + i] = math.exp(net.ymean + net.ysd * z) * net.smear
	}
	return
}

// predict_z: the model's standardised outputs for the samples ending on `days`,
// in chunks (a Mamba block holds SEQ · HIDDEN · d_state values per sample)
predict_z :: proc(net: ^Net, m: ^Market, days: []int) -> []f64 {
	CHUNK :: 512
	W :: SEQ * N_FEATURES
	out := make([]f64, len(days), context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for lo := 0; lo < len(days); lo += CHUNK {
		k := min(CHUNK, len(days) - lo)
		x := t.tensor_new(l.matrix_new(f64, 1, k * W))
		x.shape = {k, SEQ, N_FEATURES, 1}
		for d, i in days[lo:][:k] {window_into(net, m, d, x.data.data[i * W:][:W])}
		t.graph_begin(&scope, no_grad = true)
		copy(out[lo:][:k], t.tensor_values(nn.sequential_forward(net.model, x)))
		t.graph_end(&scope)
		t.tensor_free(x)
	}
	return out
}
