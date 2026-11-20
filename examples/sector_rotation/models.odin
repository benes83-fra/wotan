package main

// Stage 3: three ways to rank the sectors.
// - LSTM: the last WINDOW days of a sector's features -> LSTM -> two dense
//   layers -> the forecast. One model for all sectors, trained with Adam on
//   the MSE, on the GPU when there is one.
// - Ridge: a linear regression on the features of the last day.
// - Momentum: no fitting at all; the 60-day momentum feature is the forecast.

import ml "../../wotan/analytics/ML"
import gpu "../../wotan/gpu"
import l "../../wotan/linalg"
import nn "../../wotan/nn"
import t "../../wotan/tensor"
import "core:math"
import "core:math/rand"
import "core:slice"

LSTM_HIDDEN :: 32
EPOCHS :: 12
BATCH :: 256
LEARNING_RATE :: 1e-3
RIDGE_LAMBDA :: 10.0

// Sample: one (day, asset) pair
Sample :: struct {
	day, asset: int,
}

// training_samples: every (day, asset) with days lo ..= hi whose target is known
training_samples :: proc(ds: ^Dataset, lo, hi: int) -> []Sample {
	out := make([dynamic]Sample, 0, (hi - lo + 1) * N_ASSETS)
	for d in lo ..= hi {
		for a in 0 ..< N_ASSETS {
			if !math.is_nan(ds.target[d * N_ASSETS + a]) {append(&out, Sample{d, a})}
		}
	}
	return out[:]
}

lstm_model_new :: proc() -> ^nn.Sequential {
	m := nn.sequential_new()
	nn.sequential_add(m, nn.lstm_layer_new(N_FEATURES, LSTM_HIDDEN)) // [B, WINDOW, HIDDEN]
	nn.sequential_add(m, nn.FlattenLayer{}) // [B, WINDOW · HIDDEN]
	nn.sequential_add(m, nn.linear_layer_new(WINDOW * LSTM_HIDDEN, LSTM_HIDDEN))
	nn.sequential_add(m, nn.Activation.ReLU)
	nn.sequential_add(m, nn.linear_layer_new(LSTM_HIDDEN, 1))
	return m
}

// train_lstm trains a fresh model for EPOCHS passes over the samples, in
// shuffled batches of BATCH (the last, incomplete batch of an epoch is
// dropped). Returns the mean loss of the last epoch.
train_lstm :: proc(
	ds: ^Dataset,
	samples: []Sample,
	on_gpu: bool,
	seed: u64,
) -> (
	model: ^nn.Sequential,
	loss: f64,
) {
	rand.reset(seed)
	model = lstm_model_new()
	params := nn.sequential_parameters(model)
	if on_gpu {gpu.to_device(params[:])}
	opt := nn.adam_new(LEARNING_RATE)
	defer nn.adam_free(&opt)
	for p in params {nn.adam_add_param(&opt, p)}

	// one input and one target tensor, refilled for every batch
	x := t.tensor_new(l.matrix_new(f64, 1, BATCH * WINDOW * N_FEATURES))
	x.shape = {BATCH, WINDOW, N_FEATURES, 1}
	y := t.tensor_new(l.matrix_new(f64, BATCH, 1))
	defer {t.tensor_free(x); t.tensor_free(y)}

	order := make([]int, len(samples))
	defer delete(order)
	for &v, i in order {v = i}
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for epoch in 0 ..< EPOCHS {
		rand.shuffle(order)
		sum, n := 0.0, 0
		for b := 0; b + BATCH <= len(order); b += BATCH {
			xs, ys := t.tensor_values(x), t.tensor_values(y)
			for s, i in order[b:][:BATCH] {
				window_into(
					ds,
					samples[s].day,
					samples[s].asset,
					xs[i * WINDOW * N_FEATURES:][:WINDOW * N_FEATURES],
				)
				ys[i] = ds.target[samples[s].day * N_ASSETS + samples[s].asset]
			}
			t.tensor_mark_host_dirty(x) // the next GPU op uploads the new batch
			t.tensor_mark_host_dirty(y)

			t.graph_begin(&scope)
			batch_loss := t.tensor_mse_loss(nn.sequential_forward(model, x), y)
			t.tensor_backward(batch_loss)
			// reading the loss waits for the GPU: only in the last epoch
			if epoch == EPOCHS - 1 {sum += t.tensor_item(batch_loss); n += 1}
			t.graph_end(&scope)
			nn.adam_step(&opt)
			nn.adam_zero_grad(&opt)
		}
		loss = sum / f64(max(n, 1))
	}
	return
}

// predict_lstm writes the model's forecasts for days lo ..< hi into
// pred [day][asset]
predict_lstm :: proc(model: ^nn.Sequential, ds: ^Dataset, lo, hi: int, pred: []f64) {
	n := (hi - lo) * N_ASSETS
	x := t.tensor_new(l.matrix_new(f64, 1, n * WINDOW * N_FEATURES))
	x.shape = {n, WINDOW, N_FEATURES, 1}
	defer t.tensor_free(x)
	for d in lo ..< hi {
		for a in 0 ..< N_ASSETS {
			i := (d - lo) * N_ASSETS + a
			window_into(ds, d, a, x.data.data[i * WINDOW * N_FEATURES:][:WINDOW * N_FEATURES])
		}
	}
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	t.graph_begin(&scope, no_grad = true)
	copy(pred[lo * N_ASSETS:hi * N_ASSETS], t.tensor_values(nn.sequential_forward(model, x)))
	t.graph_end(&scope)
}

// fit_ridge: ridge regression of the target on [1, features of the day]
fit_ridge :: proc(ds: ^Dataset, samples: []Sample) -> []f64 {
	X := l.matrix_new(f64, len(samples), 1 + N_FEATURES)
	y := make([]f64, len(samples))
	defer {l.matrix_free(&X); delete(y)}
	for s, i in samples {
		X.data[i * (1 + N_FEATURES)] = 1
		copy(
			X.data[i * (1 + N_FEATURES) + 1:][:N_FEATURES],
			ds.x[(s.day * N_ASSETS + s.asset) * N_FEATURES:][:N_FEATURES],
		)
		y[i] = ds.target[s.day * N_ASSETS + s.asset]
	}
	res := ml.ridge_fit(&X, y, RIDGE_LAMBDA)
	defer ml.ols_result_free(&res, context.allocator)
	return slice.clone(res.beta)
}

predict_ridge :: proc(beta: []f64, ds: ^Dataset, lo, hi: int, pred: []f64) {
	for d in lo ..< hi {
		for a in 0 ..< N_ASSETS {
			f := ds.x[(d * N_ASSETS + a) * N_FEATURES:][:N_FEATURES]
			p := beta[0]
			for v, k in f {p += beta[k + 1] * v}
			pred[d * N_ASSETS + a] = p
		}
	}
}

// predict_momentum: the 60-day momentum feature (already z-scored across sectors)
predict_momentum :: proc(ds: ^Dataset, lo, hi: int, pred: []f64) {
	for d in lo ..< hi {
		for a in 0 ..< N_ASSETS {pred[d * N_ASSETS + a] = ds.x[(d * N_ASSETS + a) * N_FEATURES + 5]}
	}
}
