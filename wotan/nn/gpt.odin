
package nn


import l "../linalg"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

// ============================================================================
// GPT Block (Pre-LayerNorm, no cross-attention)
// ============================================================================

GPTBlock :: struct {
	d_model:   int,
	num_heads: int,
	d_ff:      int,
	use_moe:   bool,
	ln1:       LayerNormLayer,
	mha:       MultiHeadAttentionLayer,
	ln2:       LayerNormLayer,
	ffn:       FFNLayer,
	moe:       MoELayer,
}

gpt_block_new :: proc(
	d_model: int,
	num_heads: int,
	d_ff: int,
	use_flash: bool = true,
	use_moe: bool = false,
	num_experts: int = 8,
	top_k: int = 2,
	allocator: mem.Allocator = context.allocator,
) -> GPTBlock {
	block: GPTBlock
	block.d_model = d_model
	block.num_heads = num_heads
	block.d_ff = d_ff
	block.use_moe = use_moe

	block.ln1 = layer_norm_layer_new(d_model, 1e-5, allocator)
	block.mha = multi_head_attention_layer_new(
		d_model,
		num_heads,
		use_flash = use_flash,
		is_causal = true,
		allocator = allocator,
	)
	block.ln2 = layer_norm_layer_new(d_model, 1e-5, allocator)

	if use_moe {
		block.moe = moe_layer_new(d_model, d_ff, num_experts, top_k, allocator)
	} else {
		block.ffn = ffn_layer_new(d_model, d_ff, allocator)
	}
	return block
}

gpt_block_free :: proc(block: ^GPTBlock) {
	layer_norm_layer_free(&block.ln1)
	multi_head_attention_layer_free(&block.mha)
	layer_norm_layer_free(&block.ln2)

	if block.use_moe {
		moe_layer_free(&block.moe)
	} else {
		ffn_layer_free(&block.ffn)
	}
}

gpt_block_forward :: proc(
	block: ^GPTBlock,
	x: ^t.Tensor,
	mask: []f64,
	training: bool,
	dropout: f64,
	aux_loss: bool = false,
) -> ^t.Tensor {
	// Pre-LayerNorm
	x_norm := layer_norm_layer_forward(&block.ln1, x)
	attn_out := masked_multi_head_attention_layer_forward(&block.mha, x_norm, mask)
	return _gpt_block_rest(block, x, attn_out, training, dropout, aux_loss)
}

// The block after its attention: residual, LayerNorm, FFN / MoE, residual
@(private = "file")
_gpt_block_rest :: proc(
	block: ^GPTBlock,
	x, attn_out: ^t.Tensor,
	training: bool,
	dropout: f64,
	aux_loss: bool,
) -> ^t.Tensor {
	x1 := t.tensor_add(x, t.tensor_dropout(attn_out, dropout, training))

	x1_norm := layer_norm_layer_forward(&block.ln2, x1)
	ffn_out: ^t.Tensor
	if block.use_moe {
		ffn_out = moe_layer_forward(&block.moe, x1_norm, x1_norm.allocator, aux_loss)
	} else {
		ffn_out = ffn_layer_forward(&block.ffn, x1_norm)
	}
	ffn_out = t.tensor_dropout(ffn_out, dropout, training)
	return t.tensor_add(x1, ffn_out)
}

// ============================================================================
// GPT Model (Full decoder-only Transformer)
// ============================================================================

GPTModel :: struct {
	vocab_size:     int,
	d_model:        int,
	num_heads:      int,
	d_ff:           int,
	num_layers:     int,
	max_seq_len:    int,
	dropout:        f64, // on the attention and FFN outputs of every block (training only)
	token_emb:      EmbeddingLayer,
	pos_emb:        EmbeddingLayer, // Learned positional embeddings
	blocks:         [dynamic]GPTBlock,
	final_ln:       LayerNormLayer,
	output_proj:    LinearLayer,
	total_aux_loss: f64, // MoE load-balancing loss of the last forward pass
	aux_loss:       ^t.Tensor, // the same in the graph (forward with aux_loss = true), else nil
	allocator:      mem.Allocator,
}

gpt_model_new :: proc(
	vocab_size: int,
	d_model: int,
	num_heads: int,
	d_ff: int,
	num_layers: int,
	max_seq_len: int,
	use_flash: bool = false,
	use_moe: bool = false,
	num_experts: int = 8,
	top_k: int = 2,
	dropout: f64 = 0.1,
	allocator: mem.Allocator = context.allocator,
) -> GPTModel {
	model: GPTModel
	model.vocab_size = vocab_size
	model.d_model = d_model
	model.num_heads = num_heads
	model.d_ff = d_ff
	model.num_layers = num_layers
	model.max_seq_len = max_seq_len
	model.dropout = dropout
	model.allocator = allocator

	// Token embeddings
	model.token_emb = embedding_layer_new(vocab_size, d_model, allocator)

	// Positional embeddings (learned)
	model.pos_emb = embedding_layer_new(max_seq_len, d_model, allocator)

	// GPT blocks
	model.blocks = make([dynamic]GPTBlock, 0, allocator)
	for i in 0 ..< num_layers {
		block := gpt_block_new(
			d_model,
			num_heads,
			d_ff,
			use_flash = use_flash,
			use_moe = use_moe,
			num_experts = num_experts,
			top_k = top_k,
			allocator = allocator,
		)
		append(&model.blocks, block)
	}

	// Final LayerNorm
	model.final_ln = layer_norm_layer_new(d_model, 1e-5, allocator)

	// Output projection (tied with token embeddings would be ideal, but separate for now)
	model.output_proj = linear_layer_new(d_model, vocab_size, allocator)

	return model
}

gpt_model_free :: proc(model: ^GPTModel) {
	embedding_layer_free(&model.token_emb)
	embedding_layer_free(&model.pos_emb)

	for i in 0 ..< len(model.blocks) {
		gpt_block_free(&model.blocks[i])
	}
	delete(model.blocks)

	layer_norm_layer_free(&model.final_ln)
	linear_layer_free(&model.output_proj)
}

// Logits [batch, seq_len, vocab_size]. With MoE blocks and aux_loss = true,
// model.aux_loss is the summed load-balancing loss as part of the graph.
gpt_model_forward :: proc(
	model: ^GPTModel,
	input_ids: ^t.Tensor,
	mask: []f64,
	training: bool,
	aux_loss: bool = false,
) -> ^t.Tensor {
	model.total_aux_loss = 0.0
	model.aux_loss = nil

	x := _gpt_embed(model, input_ids, 0)

	// Pass through GPT blocks
	for i in 0 ..< len(model.blocks) {
		block := &model.blocks[i]
		x = gpt_block_forward(block, x, mask, training, model.dropout, aux_loss)
		if block.use_moe {
			model.total_aux_loss += block.moe.aux_loss_val
			if block.moe.aux_loss != nil {
				model.aux_loss =
					block.moe.aux_loss if model.aux_loss == nil else t.tensor_add(model.aux_loss, block.moe.aux_loss)
			}
		}
	}

	return _gpt_head(model, x)
}

// Token + learned positional embeddings of input_ids [batch, seq_len] at
// positions start ..< start + seq_len
@(private = "file")
_gpt_embed :: proc(model: ^GPTModel, input_ids: ^t.Tensor, start: int) -> ^t.Tensor {
	batch, seq_len := input_ids.shape[0], input_ids.shape[1]
	assert(start + seq_len <= model.max_seq_len, "GPT: more positions than max_seq_len")
	token_emb := embedding_layer_forward(&model.token_emb, input_ids)
	pos_ids := _position_ids(batch, seq_len, input_ids.allocator, start)
	pos_emb := embedding_layer_forward(&model.pos_emb, pos_ids)
	_constant_done(pos_ids, pos_emb)
	return t.tensor_add(token_emb, pos_emb)
}

// Final LayerNorm and output projection: logits [batch, seq_len, vocab_size]
@(private = "file")
_gpt_head :: proc(model: ^GPTModel, x: ^t.Tensor) -> ^t.Tensor {
	return linear_forward(&model.output_proj, layer_norm_layer_forward(&model.final_ln, x))
}

// ============================================================================
// KV cache (generation)
// ============================================================================
// Generating token by token with gpt_model_forward recomputes the whole prefix
// for every new token. The cache keeps every block's attention keys and values
// of the positions seen so far, so each call only computes the new positions:
//
//     cache := nn.gpt_kv_cache_new(&model, batch = 1)
//     defer nn.gpt_kv_cache_free(&cache)
//     scope: t.Graph_Scope
//     defer t.graph_scope_destroy(&scope)
//     ids := prompt                                   // [1, n] token ids
//     for step in 0 ..< n_new {
//         t.graph_begin(&scope, no_grad = true)
//         logits := nn.gpt_model_forward_cached(&model, &cache, ids)
//         next := sample(last row of logits)
//         t.graph_end(&scope)
//         ids = [1, 1] tensor of next
//     }
//
// Positions are absolute (learned positional embeddings), so a sequence holds
// at most max_seq_len tokens: when cache.length reaches it, reset the cache and
// prefill it again with the last tokens (gpt_kv_cache_reset).

GPT_KV_Cache :: struct {
	keys, values: [][]f64, // per block [batch, capacity, d_model]
	batch:        int,
	length:       int, // positions cached so far
	capacity:     int, // model.max_seq_len
	allocator:    mem.Allocator,
}

gpt_kv_cache_new :: proc(
	model: ^GPTModel,
	batch: int,
	allocator := context.allocator,
) -> GPT_KV_Cache {
	cache := GPT_KV_Cache {
		batch     = batch,
		capacity  = model.max_seq_len,
		allocator = allocator,
		keys      = make([][]f64, len(model.blocks), allocator),
		values    = make([][]f64, len(model.blocks), allocator),
	}
	for i in 0 ..< len(model.blocks) {
		cache.keys[i] = make([]f64, batch * cache.capacity * model.d_model, allocator)
		cache.values[i] = make([]f64, batch * cache.capacity * model.d_model, allocator)
	}
	return cache
}

gpt_kv_cache_free :: proc(cache: ^GPT_KV_Cache) {
	for i in 0 ..< len(cache.keys) {
		delete(cache.keys[i], cache.allocator)
		delete(cache.values[i], cache.allocator)
	}
	delete(cache.keys, cache.allocator)
	delete(cache.values, cache.allocator)
	cache^ = {}
}

// Forgets all positions (the buffers are kept)
gpt_kv_cache_reset :: proc(cache: ^GPT_KV_Cache) {
	cache.length = 0
}

// Logits [batch, n, vocab_size] of the n new tokens input_ids [batch, n] at
// positions cache.length ..< cache.length + n, attending to all cached
// positions and causally among themselves; their keys and values are appended
// to the cache. Equal to the last n positions of gpt_model_forward over the
// whole sequence (inference: no dropout, no MoE aux loss). Call it in a
// no-grad scope: nothing is back-propagated through the cache.
gpt_model_forward_cached :: proc(
	model: ^GPTModel,
	cache: ^GPT_KV_Cache,
	input_ids: ^t.Tensor,
) -> ^t.Tensor {
	assert(!t.grad_enabled(), "gpt_model_forward_cached: call it in a no-grad scope")
	batch, n := input_ids.shape[0], input_ids.shape[1]
	assert(batch == cache.batch, "gpt_model_forward_cached: batch differs from the cache's")
	assert(
		cache.length + n <= cache.capacity,
		"gpt_model_forward_cached: cache is full (max_seq_len)",
	)
	start, total, d := cache.length, cache.length + n, model.d_model
	alloc := t._scope_or(input_ids.allocator)

	// New queries see every cached position and the new ones up to their own
	mask := make([]f64, n * total, context.temp_allocator)
	defer delete(mask, context.temp_allocator)
	for i in 0 ..< n {
		for j in start + i + 1 ..< total {mask[i * total + j] = _MASKED}
	}

	x := _gpt_embed(model, input_ids, start)
	for &block, i in model.blocks {
		mha := &block.mha
		x_norm := layer_norm_layer_forward(&block.ln1, x)
		q := linear_forward(&mha.q_proj, x_norm)
		k_new := linear_forward(&mha.k_proj, x_norm)
		v_new := linear_forward(&mha.v_proj, x_norm)

		// Append the new keys and values; attend over positions 0 ..< total
		k := _constant(batch * total * d, {batch, total, d, 1}, alloc)
		v := _constant(batch * total * d, {batch, total, d, 1}, alloc)
		for b in 0 ..< batch {
			row := cache.keys[i][b * cache.capacity * d:]
			copy(row[start * d:total * d], k_new.data.data[b * n * d:(b + 1) * n * d])
			copy(k.data.data[b * total * d:(b + 1) * total * d], row[:total * d])
			row = cache.values[i][b * cache.capacity * d:]
			copy(row[start * d:total * d], v_new.data.data[b * n * d:(b + 1) * n * d])
			copy(v.data.data[b * total * d:(b + 1) * total * d], row[:total * d])
		}
		attn_out := _attend(
			&mha.out_proj,
			mha.num_heads,
			mha.head_dim,
			q,
			k,
			v,
			batch,
			n,
			total,
			false,
			false,
			mask,
		)
		_constant_done(k, attn_out)
		_constant_done(v, attn_out)
		x = _gpt_block_rest(&block, x, attn_out, false, 0, false)
	}
	cache.length = total
	return _gpt_head(model, x)
}
// All trainable GPT parameters (the order of the optimizer state; all:
// frozen ones too)
gpt_model_parameters :: proc(
	model: ^GPTModel,
	allocator: mem.Allocator = context.allocator,
	all: bool = false,
) -> [dynamic]^t.Tensor {
	ps := make([dynamic]^t.Tensor, 0, 16, allocator)
	_add_layer_parameters(&ps, all, model.token_emb, model.pos_emb)
	for &block in model.blocks {gpt_block_parameters(&block, &ps, all)}
	_add_layer_parameters(&ps, all, model.final_ln, model.output_proj)
	return ps
}

// Parameters of one GPT block, appended to out
gpt_block_parameters :: proc(block: ^GPTBlock, out: ^[dynamic]^t.Tensor, all: bool = false) {
	_add_layer_parameters(out, all, block.ln1, block.mha, block.ln2)
	if block.use_moe {
		_add_layer_parameters(out, all, block.moe)
	} else {
		_add_layer_parameters(out, all, block.ffn)
	}
}

// layer_parameters of each layer (copies of layer structs share their tensors)
@(private)
_add_layer_parameters :: proc(out: ^[dynamic]^t.Tensor, all: bool, layers: ..Layer) {
	for layer in layers {
		lay := layer
		layer_parameters(&lay, out, all)
	}
}

// Register all GPT parameters with optimizer
gpt_model_add_to_optimizer :: proc(model: ^GPTModel, opt: ^Adam) {
	ps := gpt_model_parameters(model, context.temp_allocator)
	defer delete(ps)
	for p in ps {adam_add_param(opt, p)}
}

// ============================================================================
// Sampling Functions
// ============================================================================
// A repetition penalty > 1 divides positive logits of recently seen tokens by
// `penalty` and multiplies negative ones (once per distinct token, as in CTRL).
// A temperature <= 0 means greedy decoding.

// Logits with the repetition penalty applied (temp allocator)
@(private = "file")
_penalized :: proc(logits: []f64, recent_tokens: []int, penalty: f64) -> []f64 {
	out := make([]f64, len(logits), context.temp_allocator)
	copy(out, logits)
	if penalty > 1.0 {
		seen := make([]bool, len(logits), context.temp_allocator)
		defer delete(seen, context.temp_allocator)
		for tok in recent_tokens {
			if tok < 0 || tok >= len(out) || seen[tok] {continue}
			seen[tok] = true
			out[tok] = out[tok] / penalty if out[tok] > 0 else out[tok] * penalty
		}
	}
	return out
}

@(private = "file")
_argmax :: proc(v: []f64) -> int {
	best := 0
	for x, i in v {if x > v[best] {best = i}}
	return best
}

// Samples index idx[i] with probability ∝ exp(v[idx[i]] / temperature)
@(private = "file")
_sample_softmax :: proc(v: []f64, idx: []int, temperature: f64) -> int {
	max_val := v[idx[0]]
	for i in idx {max_val = max(max_val, v[i])}
	w := make([]f64, len(idx), context.temp_allocator)
	defer delete(w, context.temp_allocator)
	sum := 0.0
	for i, k in idx {
		w[k] = math.exp((v[i] - max_val) / temperature)
		sum += w[k]
	}
	r := rand.float64() * sum
	for k in 0 ..< len(idx) {
		r -= w[k]
		if r < 0 {return idx[k]}
	}
	return idx[len(idx) - 1]
}

// Indices 0..n-1 sorted by v descending (temp allocator)
@(private = "file")
_by_value_desc :: proc(v: []f64) -> []int {
	idx := make([]int, len(v), context.temp_allocator)
	for &i, k in idx {i = k}
	Ctx :: struct {
		v: []f64,
	}
	context.user_ptr = &Ctx{v}
	slice.sort_by(idx, proc(a, b: int) -> bool {
		v := (^Ctx)(context.user_ptr).v
		return v[a] > v[b]
	})
	return idx
}

// Temperature sampling with repetition penalty
gpt_sample_temperature :: proc(
	logits: []f64,
	temperature: f64,
	recent_tokens: []int,
	penalty: f64,
) -> int {
	v := _penalized(logits, recent_tokens, penalty)
	defer delete(v, context.temp_allocator)
	if temperature <= 0.0 {return _argmax(v)}
	idx := make([]int, len(v), context.temp_allocator)
	defer delete(idx, context.temp_allocator)
	for &i, k in idx {i = k}
	return _sample_softmax(v, idx, temperature)
}

// Top-k sampling with repetition penalty
gpt_sample_top_k :: proc(
	logits: []f64,
	k: int,
	temperature: f64,
	recent_tokens: []int,
	penalty: f64,
) -> int {
	if k <= 0 || k >= len(logits) {
		return gpt_sample_temperature(logits, temperature, recent_tokens, penalty)
	}
	v := _penalized(logits, recent_tokens, penalty)
	defer delete(v, context.temp_allocator)
	if temperature <= 0.0 {return _argmax(v)}
	idx := _by_value_desc(v)
	defer delete(idx, context.temp_allocator)
	return _sample_softmax(v, idx[:k], temperature)
}

// Top-p (nucleus) sampling with repetition penalty: the smallest set of most
// likely tokens whose probability reaches p
gpt_sample_top_p :: proc(
	logits: []f64,
	p: f64,
	temperature: f64,
	recent_tokens: []int,
	penalty: f64,
) -> int {
	if p >= 1.0 {
		return gpt_sample_temperature(logits, temperature, recent_tokens, penalty)
	}
	v := _penalized(logits, recent_tokens, penalty)
	defer delete(v, context.temp_allocator)
	if temperature <= 0.0 {return _argmax(v)}
	idx := _by_value_desc(v)
	defer delete(idx, context.temp_allocator)
	// probabilities at this temperature, in descending order
	max_val := v[idx[0]]
	sum := 0.0
	for i in idx {sum += math.exp((v[i] - max_val) / temperature)}
	cum, size := 0.0, 0
	for i in idx {
		cum += math.exp((v[i] - max_val) / temperature) / sum
		size += 1
		if cum >= p {break}
	}
	return _sample_softmax(v, idx[:size], temperature)
}

// Samples from softmax(logits) (logits are left unchanged)
gpt_softmax_sample :: proc(logits: []f64) -> int {
	return gpt_sample_temperature(logits, 1.0, nil, 1.0)
}

gpt_replace_output_head :: proc(model: ^GPTModel, new_vocab_size: int, allocator: mem.Allocator) {
	linear_layer_free(&model.output_proj)
	model.output_proj = linear_layer_new(model.d_model, new_vocab_size, allocator)
	model.vocab_size = new_vocab_size
}
