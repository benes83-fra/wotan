
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
	use_moe: bool = false, // ✅ NEW
	num_experts: int = 8, // ✅ NEW
	top_k: int = 2, // ✅ NEW
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
	aux_loss: bool = false,
) -> ^t.Tensor {
	// Pre-LayerNorm
	x_norm := layer_norm_layer_forward(&block.ln1, x)
	attn_out := masked_multi_head_attention_layer_forward(&block.mha, x_norm, mask)

	// Add dropout if training
	if training {
		attn_out = t.tensor_dropout(attn_out, 0.1, true)
	}

	x1 := t.tensor_add(x, attn_out)

	x1_norm := layer_norm_layer_forward(&block.ln2, x1)
	ffn_out: ^t.Tensor
	if block.use_moe {
		ffn_out = moe_layer_forward(&block.moe, x1_norm, x1_norm.allocator, aux_loss)
	} else {
		ffn_out = ffn_layer_forward(&block.ffn, x1_norm)
	}
	if training {
		ffn_out = t.tensor_dropout(ffn_out, 0.1, true)
	}

	out := t.tensor_add(x1, ffn_out)

	return out
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
	allocator: mem.Allocator = context.allocator,
) -> GPTModel {
	model: GPTModel
	model.vocab_size = vocab_size
	model.d_model = d_model
	model.num_heads = num_heads
	model.d_ff = d_ff
	model.num_layers = num_layers
	model.max_seq_len = max_seq_len
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
	batch := input_ids.shape[0]
	seq_len := input_ids.shape[1]
	model.total_aux_loss = 0.0
	model.aux_loss = nil

	// Token + learned positional embeddings
	token_emb := embedding_layer_forward(&model.token_emb, input_ids)
	pos_ids := _position_ids(batch, seq_len, input_ids.allocator)
	pos_emb := embedding_layer_forward(&model.pos_emb, pos_ids)
	_constant_done(pos_ids, pos_emb)
	x := t.tensor_add(token_emb, pos_emb)

	// Pass through GPT blocks
	for i in 0 ..< len(model.blocks) {
		block := &model.blocks[i]
		x = gpt_block_forward(block, x, mask, training, aux_loss)
		if block.use_moe {
			model.total_aux_loss += block.moe.aux_loss_val
			if block.moe.aux_loss != nil {
				model.aux_loss =
					block.moe.aux_loss if model.aux_loss == nil else t.tensor_add(model.aux_loss, block.moe.aux_loss)
			}
		}
	}

	// Final LayerNorm
	x = layer_norm_layer_forward(&model.final_ln, x)

	// Output projection
	logits := linear_forward(&model.output_proj, x)

	return logits
}
// All trainable GPT parameters (the order of the optimizer state)
gpt_model_parameters :: proc(
	model: ^GPTModel,
	allocator: mem.Allocator = context.allocator,
) -> [dynamic]^t.Tensor {
	ps := make([dynamic]^t.Tensor, 0, 16, allocator)
	add :: proc(ps: ^[dynamic]^t.Tensor, layer: Layer) {
		lay := layer
		layer_parameters(&lay, ps)
	}
	add(&ps, model.token_emb)
	add(&ps, model.pos_emb)
	for &block in model.blocks {
		add(&ps, block.ln1)
		add(&ps, block.mha)
		add(&ps, block.ln2)
		if block.use_moe {add(&ps, block.moe)} else {add(&ps, block.ffn)}
	}
	add(&ps, model.final_ln)
	add(&ps, model.output_proj)
	return ps
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
