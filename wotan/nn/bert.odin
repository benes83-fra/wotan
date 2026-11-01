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
	use_flash: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> BERTEncoderBlock {
	block: BERTEncoderBlock
	block.d_model = d_model
	block.num_heads = num_heads
	block.d_ff = d_ff

	block.ln1 = layer_norm_layer_new(d_model, 1e-5, allocator)
	block.mha = multi_head_attention_layer_new(
		d_model,
		num_heads,
		use_flash = use_flash,
		allocator = allocator,
	)
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
	attention_mask: ^t.Tensor = nil,
	training: bool,
) -> ^t.Tensor {
	// Post-LayerNorm, as in the original BERT
	attn_out := multi_head_attention_layer_forward(&block.mha, x, attention_mask)

	// Dropout if training
	if training {
		attn_out = t.tensor_dropout(attn_out, 0.1, true)
	}
	x1 := t.tensor_add(x, attn_out)

	x1_norm := layer_norm_layer_forward(&block.ln1, x1)
	ffn_out := ffn_layer_forward(&block.ffn, x1_norm)

	if training {
		ffn_out = t.tensor_dropout(ffn_out, 0.1, true)
	}

	x2 := t.tensor_add(x1_norm, ffn_out)
	out := layer_norm_layer_forward(&block.ln2, x2)
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
	use_flash: bool = false,
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
		block := bert_encoder_block_new(
			d_model,
			num_heads,
			d_ff,
			use_flash = use_flash,
			allocator = allocator,
		)
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
	alloc := input_ids.allocator

	// Token + position + segment embeddings, LayerNorm
	token_emb := embedding_layer_forward(&model.token_emb, input_ids)
	pos_ids := _position_ids(batch, seq_len, alloc)
	pos_emb := embedding_layer_forward(&model.pos_emb, pos_ids)
	_constant_done(pos_ids, pos_emb)
	segment_emb := embedding_layer_forward(&model.segment_emb, segment_ids)
	x := t.tensor_add(t.tensor_add(token_emb, pos_emb), segment_emb)
	x = layer_norm_layer_forward(&model.emb_ln, x)

	// Token id 0 is padding: no attention to it (read during the forward pass only)
	attention_mask := _constant(batch * seq_len, {batch, seq_len, 1, 1}, alloc)
	for i in 0 ..< batch * seq_len {
		attention_mask.data.data[i] = 0.0 if input_ids.data.data[i] == 0.0 else 1.0
	}
	for i in 0 ..< len(model.encoder_blocks) {
		x = bert_encoder_block_forward(&model.encoder_blocks[i], x, attention_mask, training)
	}
	t.tensor_free(attention_mask)

	// MLM head (applied to all positions)
	mlm_logits = linear_forward(&model.mlm_head, x)

	// NSP head on [CLS] (position 0), gathered with an embedding lookup into the
	// [batch * seq_len, d_model] rows of x so that gradients reach the encoder
	rows := t.tensor_reshape(x, [4]int{batch * seq_len, model.d_model, 1, 1}, alloc)
	cls_ids := _constant(batch, {batch, 1, 1, 1}, alloc)
	for b in 0 ..< batch {cls_ids.data.data[b] = f64(b * seq_len)}
	cls := t.tensor_embedding(cls_ids, rows)
	_constant_done(cls_ids, cls)

	pooled := t.tensor_tanh(linear_forward(&model.pooler, cls))
	nsp_logits = linear_forward(&model.nsp_head, pooled)
	return mlm_logits, nsp_logits
}

// All trainable BERT parameters (the order of the optimizer state)
bert_model_parameters :: proc(
	model: ^BERTModel,
	allocator: mem.Allocator = context.allocator,
) -> [dynamic]^t.Tensor {
	ps := make([dynamic]^t.Tensor, 0, 16, allocator)
	add :: proc(ps: ^[dynamic]^t.Tensor, layer: Layer) {
		lay := layer
		layer_parameters(&lay, ps)
	}
	add(&ps, model.token_emb)
	add(&ps, model.pos_emb)
	add(&ps, model.segment_emb)
	add(&ps, model.emb_ln)
	for &block in model.encoder_blocks {
		add(&ps, block.ln1)
		add(&ps, block.mha)
		add(&ps, block.ln2)
		add(&ps, block.ffn)
	}
	add(&ps, model.pooler)
	add(&ps, model.mlm_head)
	add(&ps, model.nsp_head)
	return ps
}

// Register all BERT parameters with optimizer
bert_model_add_to_optimizer :: proc(model: ^BERTModel, opt: ^Adam) {
	ps := bert_model_parameters(model, context.temp_allocator)
	defer delete(ps)
	for p in ps {adam_add_param(opt, p)}
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
