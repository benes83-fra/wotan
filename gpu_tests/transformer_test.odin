package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:time"

// GPT and BERT trained on the GPU against the CPU. Dropout stays on: the random
// generator is reset before every step, so both runs draw the same masks.

GPT_Case :: struct {
	name:                                 string,
	flash, moe, rms_norm, tie, gelu_tanh: bool,
}

TF_VOCAB :: 23
TF_SEQ :: 12
TF_BATCH :: 6
TF_STEPS :: 12

_round_params :: proc(ps: []^t.Tensor) {
	for p in ps {
		for &v in p.data.data {v = f64(f32(v))}
	}
}

// fixed token batches: ids [batch, seq] and next-token targets (0 is padding)
_token_batch :: proc(step: int) -> (ids: ^t.Tensor, targets: []int) {
	rand.reset(u64(500 + step))
	m := l.matrix_new(f64, 1, TF_BATCH * TF_SEQ)
	targets = make([]int, TF_BATCH * TF_SEQ, context.temp_allocator)
	for i in 0 ..< TF_BATCH * TF_SEQ {
		v := 1 + rand.int_max(TF_VOCAB - 1)
		if i % TF_SEQ >= TF_SEQ - 2 && i / TF_SEQ % 2 == 0 {v = 0} 	// some padding
		m.data[i] = f64(v)
	}
	for i in 0 ..< TF_BATCH * TF_SEQ {
		targets[i] = int(m.data[i + 1]) if (i + 1) % TF_SEQ != 0 else 0
	}
	ids = t.tensor_new(m)
	ids.shape = {TF_BATCH, TF_SEQ, 1, 1}
	return
}

// _key_biases: indices in ps of the attention key-projection biases. Adding a
// constant to every key shifts all scores of a softmax row equally, so their
// true gradient is exactly zero. The rounding residue (~1e-17 in f64 on the
// CPU, ~1e-8 in f32 on the GPU) is below Adam's epsilon only on the CPU: on the
// GPU Adam turns it into steps of the learning rate. That moves the biases
// without changing the model's output, so they are left out of the parameter
// comparison; the losses compare the outputs.
_key_biases :: proc(ps: []^t.Tensor, mhas: []^nn.MultiHeadAttentionLayer) -> []int {
	out := make([dynamic]int, context.temp_allocator)
	for p, i in ps {
		for m in mhas {if p == m.k_proj.bias {append(&out, i)}}
	}
	return out[:]
}

_fallback_ops :: proc() -> (total: int, names: string) {
	st := gpu.stats()
	list := make([dynamic]string, context.temp_allocator)
	for op in t.Op {
		n := st.forward_fallbacks[op] + st.backward_fallbacks[op]
		if n > 0 {
			total += n
			append(&list, fmt.tprintf("%v", op))
		}
	}
	return total, fmt.tprint(list[:])
}

_gpt_run :: proc(
	c: GPT_Case,
	on_gpu: bool,
) -> (
	losses: []f64,
	params: [][]f64,
	ms: f64,
	skip: []int,
) {
	rand.reset(99)
	model := nn.gpt_model_new(
		TF_VOCAB,
		32,
		4,
		64,
		2,
		TF_SEQ,
		use_flash = c.flash,
		use_moe = c.moe,
		num_experts = 4,
		top_k = 2,
		dropout = 0.1,
		rms_norm = c.rms_norm,
		tie_embeddings = c.tie,
		gelu_tanh = c.gelu_tanh,
	)
	defer nn.gpt_model_free(&model)
	ps := nn.gpt_model_parameters(&model, context.temp_allocator)
	_round_params(ps[:])
	mhas := make([dynamic]^nn.MultiHeadAttentionLayer, context.temp_allocator)
	for &b in model.blocks {append(&mhas, &b.mha)}
	skip = _key_biases(ps[:], mhas[:])
	opt := nn.adam_new(0.003)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	if on_gpu {gpu.to_device(ps[:])}
	mask := nn.create_causal_mask(TF_SEQ, context.temp_allocator)

	losses = make([]f64, TF_STEPS, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	start := time.now()
	for step in 0 ..< TF_STEPS {
		ids, targets := _token_batch(step)
		rand.reset(u64(1000 + step)) // same dropout masks on both runs
		t.graph_begin(&scope)
		logits := nn.gpt_model_forward(&model, ids, mask, true, aux_loss = c.moe)
		loss := t.tensor_cross_entropy_loss(logits, targets)
		if c.moe {loss = t.tensor_add(loss, t.tensor_scale(model.aux_loss, 0.01))}
		t.tensor_backward(loss)
		losses[step] = t.tensor_item(loss)
		t.graph_end(&scope)
		nn.clip_grad_norm(&opt, 1.0)
		nn.adam_step(&opt)
		nn.adam_zero_grad(&opt)
		t.tensor_free(ids)
	}
	ms = time.duration_milliseconds(time.since(start)) / TF_STEPS
	params = make([][]f64, len(ps), context.temp_allocator)
	for p, i in ps {params[i] = _clone(t.tensor_values(p))}
	if on_gpu {gpu.to_host(ps[:])}
	return
}

gpu_gpt_training_test :: proc() {
	fmt.println("gpu_gpt_training_test: small GPTs (dropout on) trained on the GPU vs the CPU")
	cases := []GPT_Case {
		{name = "GPT (layer norm, GELU)"},
		{name = "GPT, flash attention", flash = true},
		{name = "GPT, RMSNorm + tied + tanh GELU", rms_norm = true, tie = true, gelu_tanh = true},
		{name = "GPT, mixture of experts", moe = true},
	}
	for c in cases {
		ref_l, ref_p, cpu_ms, skip := _gpt_run(c, false)
		gpu.reset_stats()
		got_l, got_p, gpu_ms, _ := _gpt_run(c, true)
		n_fb, fb_ops := _fallback_ops()
		_compare_runs(c.name, ref_l, got_l, ref_p, got_p, 2e-3, skip)
		check(got_l[TF_STEPS - 1] < got_l[0], "%s: the loss did not fall", c.name)
		check(n_fb == 0, "%s: %d CPU fallbacks (%s)", c.name, n_fb, fb_ops)
		fmt.printfln("        step time CPU %.1f ms, GPU %.1f ms", cpu_ms, gpu_ms)
	}
}

_bert_run :: proc(on_gpu: bool) -> (losses: []f64, params: [][]f64, skip: []int) {
	rand.reset(7)
	model := nn.bert_model_new(TF_VOCAB, 32, 4, 64, 2, TF_SEQ, dropout = 0.1)
	defer nn.bert_model_free(&model)
	ps := nn.bert_model_parameters(&model, context.temp_allocator)
	_round_params(ps[:])
	mhas := make([dynamic]^nn.MultiHeadAttentionLayer, context.temp_allocator)
	for &b in model.encoder_blocks {append(&mhas, &b.mha)}
	skip = _key_biases(ps[:], mhas[:])
	opt := nn.adam_new(0.003)
	defer nn.adam_free(&opt)
	for p in ps {nn.adam_add_param(&opt, p)}
	if on_gpu {gpu.to_device(ps[:])}

	losses = make([]f64, TF_STEPS, context.temp_allocator)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for step in 0 ..< TF_STEPS {
		ids, targets := _token_batch(step)
		seg := l.matrix_new(f64, 1, TF_BATCH * TF_SEQ)
		for i in 0 ..< TF_BATCH * TF_SEQ {seg.data[i] = 1 if i % TF_SEQ >= TF_SEQ / 2 else 0}
		segments := t.tensor_new(seg)
		segments.shape = {TF_BATCH, TF_SEQ, 1, 1}
		nsp_labels := make([]int, TF_BATCH, context.temp_allocator)
		for b in 0 ..< TF_BATCH {nsp_labels[b] = b % 2}
		rand.reset(u64(2000 + step))
		t.graph_begin(&scope)
		mlm, nsp := nn.bert_model_forward(&model, ids, segments, true)
		loss := t.tensor_add(
			t.tensor_cross_entropy_loss(mlm, targets),
			t.tensor_cross_entropy_loss(nsp, nsp_labels),
		)
		t.tensor_backward(loss)
		losses[step] = t.tensor_item(loss)
		t.graph_end(&scope)
		nn.adam_step(&opt)
		nn.adam_zero_grad(&opt)
		t.tensor_free(ids)
		t.tensor_free(segments)
	}
	params = make([][]f64, len(ps), context.temp_allocator)
	for p, i in ps {params[i] = _clone(t.tensor_values(p))}
	if on_gpu {gpu.to_host(ps[:])}
	return
}

gpu_bert_training_test :: proc() {
	fmt.println(
		"gpu_bert_training_test: a small BERT (MLM + NSP, padding masks, dropout on) on the GPU vs the CPU",
	)
	ref_l, ref_p, skip := _bert_run(false)
	gpu.reset_stats()
	got_l, got_p, _ := _bert_run(true)
	n_fb, fb_ops := _fallback_ops()
	_compare_runs("BERT", ref_l, got_l, ref_p, got_p, 2e-3, skip)
	check(got_l[TF_STEPS - 1] < got_l[0], "BERT: the loss did not fall")
	check(n_fb == 0, "BERT: %d CPU fallbacks (%s)", n_fb, fb_ops)
}
