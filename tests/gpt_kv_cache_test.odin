package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:time"

// GPT KV cache (gpt_model_forward_cached):
// - the logits of a sequence fed in pieces (a prefill, then token by token)
//   equal those of gpt_model_forward over the whole sequence, for standard,
//   FlashAttention and MoE blocks; after gpt_kv_cache_reset the cache is reused
// - greedy generation with the cache yields the same tokens as recomputing the
//   whole prefix for every token, and is faster
// - the cache and every forward pass release all their memory

@(private = "file")
_kv_ids :: proc(ids: []int, batch: int) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, len(ids), t._scope_or(context.allocator))
	for id, i in ids {m.data[i] = f64(id)}
	x := t.tensor_new(m, false, t._scope_or(context.allocator))
	x.shape = {batch, len(ids) / batch, 1, 1}
	return x
}

// Largest logit of the last position of every sequence in logits [batch, n, vocab]
@(private = "file")
_kv_greedy :: proc(logits: ^t.Tensor, vocab: int) -> int {
	row := logits.data.data[len(logits.data.data) - vocab:]
	best := 0
	for v, i in row {if v > row[best] {best = i}}
	return best
}

// Greedy continuation of prompt by n tokens, recomputing the whole prefix for
// every token (cache == nil) or with the KV cache
@(private = "file")
_kv_generate :: proc(
	model: ^nn.GPTModel,
	prompt: []int,
	n: int,
	cache: ^nn.GPT_KV_Cache,
	scope: ^t.Graph_Scope,
) -> [dynamic]int {
	seq := make([dynamic]int, 0, len(prompt) + n)
	append(&seq, ..prompt)
	if cache != nil {nn.gpt_kv_cache_reset(cache)}
	for _ in 0 ..< n {
		t.graph_begin(scope, no_grad = true)
		logits: ^t.Tensor
		if cache == nil {
			mask := nn.create_causal_mask(len(seq), context.temp_allocator)
			logits = nn.gpt_model_forward(model, _kv_ids(seq[:], 1), mask, false)
			delete(mask, context.temp_allocator)
		} else {
			// the prompt at once, then the token generated last
			logits = nn.gpt_model_forward_cached(model, cache, _kv_ids(seq[cache.length:], 1))
		}
		append(&seq, _kv_greedy(logits, model.vocab_size))
		t.graph_end(scope)
	}
	return seq
}

gpt_kv_cache_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== GPT KV cache ===")
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// Models, caches and inputs on the heap, so anything left behind is visible
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, runtime.heap_allocator())
	defer mem.tracking_allocator_destroy(&track)
	context.allocator = mem.tracking_allocator(&track)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)

	// 1. Logits in pieces = logits of the whole sequence (batch 2, 8 positions)
	B, L, V :: 2, 8, 11
	ids := make([]int, B * L)
	for &id in ids {id = rand.int_max(V)}
	for variant in 0 ..< 3 {
		model := nn.gpt_model_new(
			V,
			8,
			2,
			16,
			2,
			L,
			use_flash = variant == 1,
			use_moe = variant == 2,
			num_experts = 4,
			top_k = 2,
		)
		defer nn.gpt_model_free(&model)
		cache := nn.gpt_kv_cache_new(&model, B)
		defer nn.gpt_kv_cache_free(&cache)

		t.graph_begin(&scope, no_grad = true)
		mask := nn.create_causal_mask(L, context.temp_allocator)
		full := nn.gpt_model_forward(&model, _kv_ids(ids, B), mask, false).data.data

		// Pieces: positions 0-2, then one per call; twice, with a reset between
		worst, worst_reset := 0.0, 0.0
		for pass in 0 ..< 2 {
			nn.gpt_kv_cache_reset(&cache)
			for start := 0; start < L; {
				n := 3 if start == 0 else 1
				piece := make([]int, B * n, context.temp_allocator)
				for b in 0 ..< B {copy(piece[b * n:(b + 1) * n], ids[b * L + start:b * L + start + n])}
				got := nn.gpt_model_forward_cached(&model, &cache, _kv_ids(piece, B)).data.data
				for b in 0 ..< B {
					for i in 0 ..< n * V {
						diff := abs(got[(b * n) * V + i] - full[(b * L + start) * V + i])
						if pass ==
						   0 {worst = max(worst, diff)} else {worst_reset = max(worst_reset, diff)}
					}
				}
				start += n
			}
		}
		full_len := cache.length
		t.graph_end(&scope)
		delete(mask, context.temp_allocator)

		names := [?]string{"standard attention", "FlashAttention", "top-2-of-4 MoE"}
		name := names[variant]
		expect(
			&passed,
			&total,
			worst < 1e-12 && worst_reset < 1e-12 && full_len == L,
			fmt.tprintf(
				"GPT with %s, batch %d: logits fed as 3 + 1 + 1 + … positions vs the whole sequence, max diff %.1e (again after reset %.1e), cache length %d",
				name,
				B,
				worst,
				worst_reset,
				full_len,
			),
		)
	}
	delete(ids)

	// 2-3. Greedy generation: the same tokens, faster
	{
		model := nn.gpt_model_new(50, 64, 4, 256, 4, 128)
		defer nn.gpt_model_free(&model)
		cache := nn.gpt_kv_cache_new(&model, 1)
		defer nn.gpt_kv_cache_free(&cache)
		prompt := []int{3, 1, 4, 1, 5, 9, 2, 6}
		n := 120

		start := time.tick_now()
		plain := _kv_generate(&model, prompt, n, nil, &scope)
		t_plain := time.tick_since(start)
		defer delete(plain)
		start = time.tick_now()
		cached := _kv_generate(&model, prompt, n, &cache, &scope)
		t_cached := time.tick_since(start)
		defer delete(cached)

		same := len(plain) == len(cached)
		for i in 0 ..< min(len(plain), len(cached)) {same &&= plain[i] == cached[i]}
		expect(
			&passed,
			&total,
			same,
			fmt.tprintf(
				"greedy generation of %d tokens (d_model 64, 4 blocks): the same tokens with and without the cache",
				n,
			),
		)
		speedup := time.duration_seconds(t_plain) / time.duration_seconds(t_cached)
		expect(
			&passed,
			&total,
			speedup > 1,
			fmt.tprintf(
				"generation time: %.0f ms recomputing the prefix, %.0f ms with the cache (%.1fx faster)",
				time.duration_milliseconds(t_plain),
				time.duration_milliseconds(t_cached),
				speedup,
			),
		)
	}

	expect(
		&passed,
		&total,
		len(track.allocation_map) == 0 && len(track.bad_free_array) == 0,
		fmt.tprintf(
			"everything freed (%d left, %d bad frees)",
			len(track.allocation_map),
			len(track.bad_free_array),
		),
	)
	fmt.printf("GPT KV cache: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
