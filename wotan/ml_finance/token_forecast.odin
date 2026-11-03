package ml_finance

import l "../linalg"
import nn "../nn"
import tk "../nn/tokenizers"
import t "../tensor"
import "core:mem"

// ============================================================================
// Probabilistic forecasting with a language model over value tokens (Chronos)
// ============================================================================
// A decoder-only Chronos (Ansari et al. 2024): each window of
// context_length + prediction_length values is scaled by its context's mean
// |value| and quantized into tokens (nn/tokenizers TimeSeriesTokenizer), and a
// GPT learns to predict every next token (cross-entropy). Forecasting samples
// prediction_length tokens after the context, path by path (KV cache), and
// maps them back to values: n_samples paths are draws from the predicted
// joint distribution of the future, so any statistic of them (quantiles per
// step, the sum over the horizon, the path's minimum, …) gets a forecast.
// Score them with prob_forecast_sample_quantiles / _sample_crps.

TokenForecasterConfig :: struct {
	context_length:    int,
	prediction_length: int,
	n_tokens:          int, // vocabulary (bins + 3 special tokens); 0: 512
	low, high:         f64, // range of the bin centers in scaled units; both 0: [-15, 15]
	d_model:           int,
	num_heads:         int,
	d_ff:              int,
	num_layers:        int,
	dropout:           f64,
}

TokenForecaster :: struct {
	gpt:       nn.GPTModel,
	tok:       tk.TimeSeriesTokenizer,
	config:    TokenForecasterConfig,
	allocator: mem.Allocator,
}

token_forecaster_new :: proc(
	config: TokenForecasterConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^TokenForecaster {
	model := new(TokenForecaster, allocator)
	model.config = config
	model.allocator = allocator
	c := &model.config
	if c.n_tokens == 0 {c.n_tokens = 512}
	if c.low == 0 && c.high == 0 {c.low, c.high = -15, 15}
	assert(c.context_length > 0 && c.prediction_length > 0, "token_forecaster_new: empty window")
	model.tok = tk.ts_tokenizer_new(
		c.n_tokens,
		c.low,
		c.high,
		c.context_length,
		allocator = allocator,
	)
	model.gpt = nn.gpt_model_new(
		c.n_tokens,
		c.d_model,
		c.num_heads,
		c.d_ff,
		c.num_layers,
		c.context_length + c.prediction_length,
		dropout = c.dropout,
		allocator = allocator,
	)
	return model
}

token_forecaster_free :: proc(model: ^TokenForecaster) {
	nn.gpt_model_free(&model.gpt)
	tk.ts_tokenizer_free(&model.tok)
	free(model, model.allocator)
}

token_forecaster_add_to_adam :: proc(model: ^TokenForecaster, opt: ^nn.Adam) {
	nn.gpt_model_add_to_optimizer(&model.gpt, opt)
}

// Next-token cross-entropy of a batch of windows [batch, context_length +
// prediction_length] (row-major; each scaled by its first context_length
// values): every token is predicted from the ones before it. Call it in a
// graph scope (graph_begin … backward, step … graph_end): the tokens and the
// mask live in it.
token_forecaster_loss :: proc(
	model: ^TokenForecaster,
	windows: []f64,
	batch: int,
	training := true,
) -> ^t.Tensor {
	C, H := model.config.context_length, model.config.prediction_length
	T := C + H
	assert(
		len(windows) == batch * T,
		"token_forecaster_loss: windows must be [batch, context + prediction]",
	)
	assert(t.graph_active_scope() != nil, "token_forecaster_loss: call it in a graph scope")
	alloc := t._scope_or(context.allocator)
	inputs := l.matrix_new(f64, 1, batch * (T - 1), alloc)
	targets := make([]int, batch * (T - 1), alloc)
	for b in 0 ..< batch {
		w := windows[b * T:(b + 1) * T]
		scale := tk.ts_scale(w[:C])
		for i in 0 ..< T - 1 {
			inputs.data[b * (T - 1) + i] = f64(tk.ts_token(&model.tok, w[i], scale))
			targets[b * (T - 1) + i] = tk.ts_token(&model.tok, w[i + 1], scale)
		}
	}
	ids := t.tensor_new(inputs, false, alloc)
	ids.shape = {batch, T - 1, 1, 1}
	mask := nn.create_causal_mask(T - 1, alloc)
	logits := nn.gpt_model_forward(&model.gpt, ids, mask, training)
	return t.tensor_cross_entropy_loss(logits, targets)
}

// n_samples sample paths of the prediction_length values after context (its
// last context_length values count), [n_samples, prediction_length]
// row-major. Only value tokens are sampled (the special ones are excluded);
// top_k > 0 samples among the k likeliest tokens.
token_forecaster_sample :: proc(
	model: ^TokenForecaster,
	context_values: []f64,
	n_samples: int,
	temperature := 1.0,
	top_k := 0,
	allocator := context.allocator,
) -> []f64 {
	H, V := model.config.prediction_length, model.config.n_tokens
	ctx_ids, ctx_mask, scale := tk.ts_encode_context(
		&model.tok,
		context_values,
		context.temp_allocator,
	)
	defer delete(ctx_ids, context.temp_allocator)
	defer delete(ctx_mask, context.temp_allocator)
	n := len(ctx_ids)
	cache := nn.gpt_kv_cache_new(&model.gpt, n_samples)
	defer nn.gpt_kv_cache_free(&cache)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)

	paths := make([]f64, n_samples * H, allocator)
	tokens := make([]int, n_samples, context.temp_allocator)
	defer delete(tokens, context.temp_allocator)
	row := make([]f64, V, context.temp_allocator)
	defer delete(row, context.temp_allocator)
	for step in 0 ..< H {
		t.graph_begin(&scope, no_grad = true)
		alloc := t.graph_allocator(&scope)
		// the context for every path first, then each path's last token
		m := n if step == 0 else 1
		in_data := l.matrix_new(f64, 1, n_samples * m, alloc)
		for s in 0 ..< n_samples {
			for i in 0 ..< m {
				in_data.data[s * m + i] = f64(ctx_ids[i] if step == 0 else tokens[s])
			}
		}
		ids := t.tensor_new(in_data, false, alloc)
		ids.shape = {n_samples, m, 1, 1}
		logits := nn.gpt_model_forward_cached(&model.gpt, &cache, ids)
		for s in 0 ..< n_samples {
			copy(row, logits.data.data[(s * m + m - 1) * V:][:V])
			for &v in row[:model.tok.n_special + 1] {v = -1e300} 	// value tokens only
			tokens[s] = nn.gpt_sample_top_k(row, top_k, temperature, nil, 1.0)
		}
		t.graph_end(&scope)
		values := tk.ts_decode(&model.tok, tokens, scale, context.temp_allocator)
		for s in 0 ..< n_samples {paths[s * H + step] = values[s]}
		delete(values, context.temp_allocator)
	}
	return paths
}
