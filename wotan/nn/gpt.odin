
package nn


import l "../linalg"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// GPT Block (Pre-LayerNorm, no cross-attention)
// ============================================================================

GPTBlock :: struct {
	d_model:   int,
	num_heads: int,
	d_ff:      int,
	ln1:       LayerNormLayer,
	mha:       MultiHeadAttentionLayer,
	ln2:       LayerNormLayer,
	ffn:       FFNLayer,
}

gpt_block_new :: proc(
	d_model: int,
	num_heads: int,
	d_ff: int,
	allocator: mem.Allocator = context.allocator,
) -> GPTBlock {
	block: GPTBlock
	block.d_model = d_model
	block.num_heads = num_heads
	block.d_ff = d_ff

	// Pre-LayerNorm architecture (GPT-2 style)
	block.ln1 = layer_norm_layer_new(d_model, 1e-5, allocator)
	block.mha = multi_head_attention_layer_new(d_model, num_heads, allocator)
	block.ln2 = layer_norm_layer_new(d_model, 1e-5, allocator)
	block.ffn = ffn_layer_new(d_model, d_ff, allocator)

	return block
}

gpt_block_free :: proc(block: ^GPTBlock) {
	layer_norm_layer_free(&block.ln1)
	multi_head_attention_layer_free(&block.mha)
	layer_norm_layer_free(&block.ln2)
	ffn_layer_free(&block.ffn)
}

gpt_block_forward :: proc(
	block: ^GPTBlock,
	x: ^t.Tensor,
	mask: []f64,
	training: bool,
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
	ffn_out := ffn_layer_forward(&block.ffn, x1_norm)

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
	vocab_size:  int,
	d_model:     int,
	num_heads:   int,
	d_ff:        int,
	num_layers:  int,
	max_seq_len: int,
	token_emb:   EmbeddingLayer,
	pos_emb:     EmbeddingLayer, // Learned positional embeddings
	blocks:      [dynamic]GPTBlock,
	final_ln:    LayerNormLayer,
	output_proj: LinearLayer,
	allocator:   mem.Allocator,
}

gpt_model_new :: proc(
	vocab_size: int,
	d_model: int,
	num_heads: int,
	d_ff: int,
	num_layers: int,
	max_seq_len: int,
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
		block := gpt_block_new(d_model, num_heads, d_ff, allocator)
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

gpt_model_forward :: proc(
	model: ^GPTModel,
	input_ids: ^t.Tensor,
	mask: []f64,
	training: bool,
) -> ^t.Tensor {
	batch := input_ids.shape[0]
	seq_len := input_ids.shape[1]

	// Token embeddings
	token_emb := embedding_layer_forward(&model.token_emb, input_ids)

	// Positional embeddings
	pos_ids_data := l.matrix_new(f64, 1, batch * seq_len, model.allocator)
	for b in 0 ..< batch {
		for s in 0 ..< seq_len {
			pos_ids_data.data[b * seq_len + s] = f64(s)
		}
	}
	pos_ids := t.tensor_new(pos_ids_data, false, model.allocator)
	pos_ids.shape = [4]int{batch, seq_len, 1, 1}
	pos_ids.op = .Constant

	pos_emb := embedding_layer_forward(&model.pos_emb, pos_ids)

	// Add token and positional embeddings
	x := t.tensor_add(token_emb, pos_emb)

	// Pass through GPT blocks
	for i in 0 ..< len(model.blocks) {
		x = gpt_block_forward(&model.blocks[i], x, mask, training) // ✅ Pass training
	}

	// Final LayerNorm
	x = layer_norm_layer_forward(&model.final_ln, x)

	// Output projection
	logits := linear_forward(&model.output_proj, x)

	return logits
}

// Register all GPT parameters with optimizer
gpt_model_add_to_optimizer :: proc(model: ^GPTModel, opt: ^Adam) {
	adam_add_param(opt, model.token_emb.weight)
	adam_add_param(opt, model.pos_emb.weight)

	for i in 0 ..< len(model.blocks) {
		block := &model.blocks[i]
		adam_add_param(opt, block.ln1.gamma)
		adam_add_param(opt, block.ln1.beta)
		adam_add_param(opt, block.mha.q_proj.weights)
		adam_add_param(opt, block.mha.q_proj.bias)
		adam_add_param(opt, block.mha.k_proj.weights)
		adam_add_param(opt, block.mha.k_proj.bias)
		adam_add_param(opt, block.mha.v_proj.weights)
		adam_add_param(opt, block.mha.v_proj.bias)
		adam_add_param(opt, block.mha.out_proj.weights)
		adam_add_param(opt, block.mha.out_proj.bias)
		adam_add_param(opt, block.ln2.gamma)
		adam_add_param(opt, block.ln2.beta)
		adam_add_param(opt, block.ffn.fc1.weights)
		adam_add_param(opt, block.ffn.fc1.bias)
		adam_add_param(opt, block.ffn.fc2.weights)
		adam_add_param(opt, block.ffn.fc2.bias)
	}

	adam_add_param(opt, model.final_ln.gamma)
	adam_add_param(opt, model.final_ln.beta)
	adam_add_param(opt, model.output_proj.weights)
	adam_add_param(opt, model.output_proj.bias)
}

// ============================================================================
// Sampling Functions
// ============================================================================

// Temperature sampling
// Add repetition penalty to temperature sampling
// Temperature sampling with repetition penalty
gpt_sample_temperature :: proc(
	logits: []f64,
	temperature: f64,
	recent_tokens: []int,
	penalty: f64,
) -> int {
	if temperature <= 0.0 {
		// Greedy decoding with penalty
		max_idx := 0
		max_val := logits[0]

		// Apply penalty to recent tokens
		val := logits[0]
		if penalty > 1.0 {
			for recent in recent_tokens {
				if 0 == recent {
					if val > 0 {val /= penalty} else {val *= penalty}
				}
			}
		}
		max_val = val

		for i in 1 ..< len(logits) {
			val = logits[i]
			if penalty > 1.0 {
				for recent in recent_tokens {
					if i == recent {
						if val > 0 {val /= penalty} else {val *= penalty}
					}
				}
			}
			if val > max_val {
				max_val = val
				max_idx = i
			}
		}
		return max_idx
	}

	// Find max for numerical stability
	max_val := logits[0]
	for i in 1 ..< len(logits) {
		if logits[i] > max_val {
			max_val = logits[i]
		}
	}

	// Apply temperature and repetition penalty
	scaled := make([]f64, len(logits), context.temp_allocator)
	defer delete(scaled, context.temp_allocator)

	sum_exp := 0.0
	for i in 0 ..< len(logits) {
		val := logits[i]

		// Apply repetition penalty
		if penalty > 1.0 {
			for recent in recent_tokens {
				if i == recent {
					if val > 0 {
						val /= penalty
					} else {
						val *= penalty
					}
				}
			}
		}

		scaled[i] = math.exp((val - max_val) / temperature)
		sum_exp += scaled[i]
	}

	// Normalize
	for i in 0 ..< len(logits) {
		scaled[i] /= sum_exp
	}

	// Sample
	r := rand.float64()
	cum_prob := 0.0
	for i in 0 ..< len(logits) {
		cum_prob += scaled[i]
		if r < cum_prob {
			return i
		}
	}

	return len(logits) - 1
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

	// Find top-k logits with penalty applied
	top_k_indices := make([]int, k, context.temp_allocator)
	top_k_logits := make([]f64, k, context.temp_allocator)
	defer {
		delete(top_k_indices, context.temp_allocator)
		delete(top_k_logits, context.temp_allocator)
	}

	// Initialize with minimum values
	for i in 0 ..< k {
		top_k_indices[i] = -1
		top_k_logits[i] = -math.F64_MAX
	}

	// Find top-k with penalty
	for i in 0 ..< len(logits) {
		val := logits[i]

		// Apply repetition penalty
		if penalty > 1.0 {
			for recent in recent_tokens {
				if i == recent {
					if val > 0 {val /= penalty} else {val *= penalty}
				}
			}
		}

		// Check if this should be in top-k
		if val > top_k_logits[k - 1] {
			// Insert in sorted position
			for j in 0 ..< k {
				if val > top_k_logits[j] {
					// Shift down
					for m := k - 1; m > j; m -= 1 {
						top_k_indices[m] = top_k_indices[m - 1]
						top_k_logits[m] = top_k_logits[m - 1]
					}
					top_k_indices[j] = i
					top_k_logits[j] = val
					break
				}
			}
		}
	}

	// Apply temperature
	if temperature > 0.0 {
		for i in 0 ..< k {
			top_k_logits[i] /= temperature
		}
	}

	// Softmax over top-k
	max_val := top_k_logits[0]
	for i in 1 ..< k {
		if top_k_logits[i] > max_val {
			max_val = top_k_logits[i]
		}
	}

	sum_exp := 0.0
	for i in 0 ..< k {
		top_k_logits[i] = math.exp(top_k_logits[i] - max_val)
		sum_exp += top_k_logits[i]
	}

	for i in 0 ..< k {
		top_k_logits[i] /= sum_exp
	}

	// Sample
	r := rand.float64()
	cum_prob := 0.0
	for i in 0 ..< k {
		cum_prob += top_k_logits[i]
		if r < cum_prob {
			return top_k_indices[i]
		}
	}

	return top_k_indices[k - 1]
}

// Top-p (nucleus) sampling with repetition penalty
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

	// Apply temperature, penalty and softmax
	scaled := make([]f64, len(logits), context.temp_allocator)
	defer delete(scaled, context.temp_allocator)

	// Find max with penalty
	max_val := -math.F64_MAX
	for i in 0 ..< len(logits) {
		val := logits[i]
		if penalty > 1.0 {
			for recent in recent_tokens {
				if i == recent {
					if val > 0 {val /= penalty} else {val *= penalty}
				}
			}
		}
		if val > max_val {
			max_val = val
		}
	}

	sum_exp := 0.0
	for i in 0 ..< len(logits) {
		val := logits[i]
		if penalty > 1.0 {
			for recent in recent_tokens {
				if i == recent {
					if val > 0 {val /= penalty} else {val *= penalty}
				}
			}
		}
		scaled[i] = math.exp((val - max_val) / temperature)
		sum_exp += scaled[i]
	}

	for i in 0 ..< len(logits) {
		scaled[i] /= sum_exp
	}

	// Sort by probability (descending)
	indices := make([]int, len(logits), context.temp_allocator)
	probs := make([]f64, len(logits), context.temp_allocator)
	defer {
		delete(indices, context.temp_allocator)
		delete(probs, context.temp_allocator)
	}

	for i in 0 ..< len(logits) {
		indices[i] = i
		probs[i] = scaled[i]
	}

	// Simple bubble sort
	for i in 0 ..< len(probs) - 1 {
		for j in 0 ..< len(probs) - i - 1 {
			if probs[j] < probs[j + 1] {
				temp_prob := probs[j]
				probs[j] = probs[j + 1]
				probs[j + 1] = temp_prob

				temp_idx := indices[j]
				indices[j] = indices[j + 1]
				indices[j + 1] = temp_idx
			}
		}
	}

	// Find nucleus
	cum_prob := 0.0
	nucleus_size := 0
	for i in 0 ..< len(probs) {
		cum_prob += probs[i]
		nucleus_size += 1
		if cum_prob >= p {
			break
		}
	}

	// Normalize probabilities within nucleus
	sum_nucleus := 0.0
	for i in 0 ..< nucleus_size {
		sum_nucleus += probs[i]
	}

	for i in 0 ..< nucleus_size {
		probs[i] /= sum_nucleus
	}

	// Sample from nucleus
	r := rand.float64()
	cum_prob = 0.0
	for i in 0 ..< nucleus_size {
		cum_prob += probs[i]
		if r < cum_prob {
			return indices[i]
		}
	}

	return indices[nucleus_size - 1]
}

// Helper: softmax and sample
gpt_softmax_sample :: proc(logits: []f64) -> int {
	max_val := logits[0]
	for i in 1 ..< len(logits) {
		if logits[i] > max_val {
			max_val = logits[i]
		}
	}

	sum_exp := 0.0
	for i in 0 ..< len(logits) {
		logits[i] = math.exp(logits[i] - max_val)
		sum_exp += logits[i]
	}

	for i in 0 ..< len(logits) {
		logits[i] /= sum_exp
	}

	r := rand.float64()
	cum_prob := 0.0
	for i in 0 ..< len(logits) {
		cum_prob += logits[i]
		if r < cum_prob {
			return i
		}
	}

	return len(logits) - 1
}
