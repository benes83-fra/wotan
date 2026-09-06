// wotan/nn/bert.odin
package nn

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:mem"

// ============================================================================
// BERT Encoder Block (Bidirectional, no causal mask)
// ============================================================================

BERTEncoderBlock :: struct {
	d_model:   int,
	num_heads: int,
	d_ff:      int,
	ln1:       LayerNormLayer,
	mha:       MultiHeadAttentionLayer,
	ln2:       LayerNormLayer,
	ffn:       FFNLayer,
}

bert_encoder_block_new :: proc(
	d_model: int,
	num_heads: int,
	d_ff: int,
	allocator: mem.Allocator = context.allocator,
) -> BERTEncoderBlock {
	block: BERTEncoderBlock
	block.d_model = d_model
	block.num_heads = num_heads
	block.d_ff = d_ff

	block.ln1 = layer_norm_layer_new(d_model, 1e-5, allocator)
	block.mha = multi_head_attention_layer_new(d_model, num_heads, allocator)
	block.ln2 = layer_norm_layer_new(d_model, 1e-5, allocator)
	block.ffn = ffn_layer_new(d_model, d_ff, allocator)

	return block
}

bert_encoder_block_free :: proc(block: ^BERTEncoderBlock) {
	layer_norm_layer_free(&block.ln1)
	multi_head_attention_layer_free(&block.mha)
	layer_norm_layer_free(&block.ln2)
	ffn_layer_free(&block.ffn)
}

bert_encoder_block_forward :: proc(
	block: ^BERTEncoderBlock,
	x: ^t.Tensor,
	training: bool,
) -> ^t.Tensor {
	// Pre-LayerNorm
	x_norm := layer_norm_layer_forward(&block.ln1, x)

	// Bidirectional attention (no mask!)
	attn_out := multi_head_attention_layer_forward(&block.mha, x_norm)

	// Dropout if training
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
// BERT Model
// ============================================================================

BERTModel :: struct {
	vocab_size:     int,
	d_model:        int,
	num_heads:      int,
	d_ff:           int,
	num_layers:     int,
	max_seq_len:    int,
	token_emb:      EmbeddingLayer,
	pos_emb:        EmbeddingLayer,
	segment_emb:    EmbeddingLayer, // Sentence A vs B
	emb_ln:         LayerNormLayer, // LayerNorm after embeddings
	encoder_blocks: [dynamic]BERTEncoderBlock,
	pooler:         LinearLayer, // For [CLS] token representation
	mlm_head:       LinearLayer, // For masked language modeling
	nsp_head:       LinearLayer, // For next sentence prediction
	allocator:      mem.Allocator,
}

bert_model_new :: proc(
	vocab_size: int,
	d_model: int,
	num_heads: int,
	d_ff: int,
	num_layers: int,
	max_seq_len: int,
	allocator: mem.Allocator = context.allocator,
) -> BERTModel {
	model: BERTModel
	model.vocab_size = vocab_size
	model.d_model = d_model
	model.num_heads = num_heads
	model.d_ff = d_ff
	model.num_layers = num_layers
	model.max_seq_len = max_seq_len
	model.allocator = allocator

	// Token embeddings
	model.token_emb = embedding_layer_new(vocab_size, d_model, allocator)

	// Positional embeddings
	model.pos_emb = embedding_layer_new(max_seq_len, d_model, allocator)

	// Segment embeddings (0 = sentence A, 1 = sentence B)
	model.segment_emb = embedding_layer_new(2, d_model, allocator)

	// LayerNorm after embeddings
	model.emb_ln = layer_norm_layer_new(d_model, 1e-5, allocator)

	// Encoder blocks
	model.encoder_blocks = make([dynamic]BERTEncoderBlock, 0, allocator)
	for i in 0 ..< num_layers {
		block := bert_encoder_block_new(d_model, num_heads, d_ff, allocator)
		append(&model.encoder_blocks, block)
	}

	// Pooler for [CLS] token
	model.pooler = linear_layer_new(d_model, d_model, allocator)

	// MLM head (predict masked tokens)
	model.mlm_head = linear_layer_new(d_model, vocab_size, allocator)

	// NSP head (predict next sentence)
	model.nsp_head = linear_layer_new(d_model, 2, allocator)

	return model
}

bert_model_free :: proc(model: ^BERTModel) {
	embedding_layer_free(&model.token_emb)
	embedding_layer_free(&model.pos_emb)
	embedding_layer_free(&model.segment_emb)
	layer_norm_layer_free(&model.emb_ln)

	for i in 0 ..< len(model.encoder_blocks) {
		bert_encoder_block_free(&model.encoder_blocks[i])
	}
	delete(model.encoder_blocks)

	linear_layer_free(&model.pooler)
	linear_layer_free(&model.mlm_head)
	linear_layer_free(&model.nsp_head)
}

bert_model_forward :: proc(
	model: ^BERTModel,
	input_ids: ^t.Tensor, // [batch, seq_len]
	segment_ids: ^t.Tensor, // [batch, seq_len]
	training: bool,
) -> (
	mlm_logits: ^t.Tensor,
	nsp_logits: ^t.Tensor,
) {
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

	pos_emb := embedding_layer_forward(&model.pos_emb, pos_ids)

	// Segment embeddings
	segment_emb := embedding_layer_forward(&model.segment_emb, segment_ids)

	// Sum all embeddings
	x := t.tensor_add(token_emb, pos_emb)
	x = t.tensor_add(x, segment_emb)

	// LayerNorm
	x = layer_norm_layer_forward(&model.emb_ln, x)

	// Pass through encoder blocks
	for i in 0 ..< len(model.encoder_blocks) {
		x = bert_encoder_block_forward(&model.encoder_blocks[i], x, training)
	}

	// MLM head (applied to all positions)
	mlm_logits = linear_forward(&model.mlm_head, x)

	// NSP head (applied to [CLS] token only - first position)
	// Extract [CLS] token representation
	cls_data := l.matrix_new(f64, 1, batch * model.d_model, model.allocator)
	for b in 0 ..< batch {
		for d in 0 ..< model.d_model {
			cls_data.data[b * model.d_model + d] = x.data.data[b * seq_len * model.d_model + d]
		}
	}
	cls_tensor := t.tensor_new(cls_data, false, model.allocator)
	cls_tensor.shape = [4]int{batch, 1, model.d_model, 1}

	// Pool and predict
	pooled := linear_forward(&model.pooler, cls_tensor)
	nsp_logits = linear_forward(&model.nsp_head, pooled)

	// Cleanup temporary tensors
	t.tensor_free(pos_ids)
	t.tensor_free(cls_tensor)

	return mlm_logits, nsp_logits
}

// Register all BERT parameters with optimizer
bert_model_add_to_optimizer :: proc(model: ^BERTModel, opt: ^Adam) {
	adam_add_param(opt, model.token_emb.weight)
	adam_add_param(opt, model.pos_emb.weight)
	adam_add_param(opt, model.segment_emb.weight)
	adam_add_param(opt, model.emb_ln.gamma)
	adam_add_param(opt, model.emb_ln.beta)

	for i in 0 ..< len(model.encoder_blocks) {
		block := &model.encoder_blocks[i]
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

	adam_add_param(opt, model.pooler.weights)
	adam_add_param(opt, model.pooler.bias)
	adam_add_param(opt, model.mlm_head.weights)
	adam_add_param(opt, model.mlm_head.bias)
	adam_add_param(opt, model.nsp_head.weights)
	adam_add_param(opt, model.nsp_head.bias)
}


bert_replace_nsp_head :: proc(model: ^BERTModel, new_num_classes: int, allocator: mem.Allocator) {
	// Free old head
	linear_layer_free(&model.nsp_head)

	// Create new head
	model.nsp_head = linear_layer_new(model.d_model, new_num_classes, allocator)
}

bert_replace_mlm_head :: proc(model: ^BERTModel, new_vocab_size: int, allocator: mem.Allocator) {
	linear_layer_free(&model.mlm_head)
	model.mlm_head = linear_layer_new(model.d_model, new_vocab_size, allocator)
	model.vocab_size = new_vocab_size
}
