package nn

import t "../tensor"
import "core:mem"

// ============================================================================
// Freeze/Unfreeze Utilities
// ============================================================================

tensor_set_requires_grad :: proc(tensor: ^t.Tensor, requires_grad: bool) {
	if tensor == nil {return}
	tensor.requires_grad = requires_grad
}

sequential_freeze_all :: proc(seq: ^Sequential) {
	for layer in seq.layers {
		switch l in layer {
		case LinearLayer:
			l.weights.requires_grad = false
			if l.bias != nil {l.bias.requires_grad = false}
		case Conv2dLayer:
			l.weight.requires_grad = false
			if l.bias != nil {l.bias.requires_grad = false}
		case BatchNorm2dLayer:
			l.weight.requires_grad = false
			l.bias.requires_grad = false
		case LayerNormLayer:
			l.gamma.requires_grad = false
			l.beta.requires_grad = false
		case EmbeddingLayer:
			l.weight.requires_grad = false
		case MultiHeadAttentionLayer:
			l.q_proj.weights.requires_grad = false
			l.q_proj.bias.requires_grad = false
			l.k_proj.weights.requires_grad = false
			l.k_proj.bias.requires_grad = false
			l.v_proj.weights.requires_grad = false
			l.v_proj.bias.requires_grad = false
			l.out_proj.weights.requires_grad = false
			l.out_proj.bias.requires_grad = false
		case FFNLayer:
			l.fc1.weights.requires_grad = false
			l.fc1.bias.requires_grad = false
			l.fc2.weights.requires_grad = false
			l.fc2.bias.requires_grad = false
		case RNNLayer:
			l.w_ih.requires_grad = false
			l.w_hh.requires_grad = false
			l.bias.requires_grad = false
		case GRULayer:
			l.w_ih.requires_grad = false
			l.w_hh.requires_grad = false
			l.bias.requires_grad = false
		case LSTMLayer:
			l.w_ih.requires_grad = false
			l.w_hh.requires_grad = false
			l.bias.requires_grad = false
		case TransformerEncoderBlock:
			l.ln1.gamma.requires_grad = false
			l.ln1.beta.requires_grad = false
			l.mha.q_proj.weights.requires_grad = false
			l.mha.q_proj.bias.requires_grad = false
			l.mha.k_proj.weights.requires_grad = false
			l.mha.k_proj.bias.requires_grad = false
			l.mha.v_proj.weights.requires_grad = false
			l.mha.v_proj.bias.requires_grad = false
			l.mha.out_proj.weights.requires_grad = false
			l.mha.out_proj.bias.requires_grad = false
			l.ln2.gamma.requires_grad = false
			l.ln2.beta.requires_grad = false
			l.ffn.fc1.weights.requires_grad = false
			l.ffn.fc1.bias.requires_grad = false
			l.ffn.fc2.weights.requires_grad = false
			l.ffn.fc2.bias.requires_grad = false
		case TransformerEncoder:
			for i in 0 ..< len(l.blocks) {
				block := &l.blocks[i]
				block.ln1.gamma.requires_grad = false
				block.ln1.beta.requires_grad = false
				block.mha.q_proj.weights.requires_grad = false
				block.mha.q_proj.bias.requires_grad = false
				block.mha.k_proj.weights.requires_grad = false
				block.mha.k_proj.bias.requires_grad = false
				block.mha.v_proj.weights.requires_grad = false
				block.mha.v_proj.bias.requires_grad = false
				block.mha.out_proj.weights.requires_grad = false
				block.mha.out_proj.bias.requires_grad = false
				block.ln2.gamma.requires_grad = false
				block.ln2.beta.requires_grad = false
				block.ffn.fc1.weights.requires_grad = false
				block.ffn.fc1.bias.requires_grad = false
				block.ffn.fc2.weights.requires_grad = false
				block.ffn.fc2.bias.requires_grad = false
			}
		case MaxPool2dLayer, AvgPool2dLayer, DropoutLayer, Activation, FlattenLayer:
		// No trainable parameters
		}
	}
}

sequential_unfreeze_all :: proc(seq: ^Sequential) {
	for layer in seq.layers {
		switch l in layer {
		case LinearLayer:
			l.weights.requires_grad = true
			if l.bias != nil {l.bias.requires_grad = true}
		case Conv2dLayer:
			l.weight.requires_grad = true
			if l.bias != nil {l.bias.requires_grad = true}
		case BatchNorm2dLayer:
			l.weight.requires_grad = true
			l.bias.requires_grad = true
		case LayerNormLayer:
			l.gamma.requires_grad = true
			l.beta.requires_grad = true
		case EmbeddingLayer:
			l.weight.requires_grad = true
		case MultiHeadAttentionLayer:
			l.q_proj.weights.requires_grad = true
			l.q_proj.bias.requires_grad = true
			l.k_proj.weights.requires_grad = true
			l.k_proj.bias.requires_grad = true
			l.v_proj.weights.requires_grad = true
			l.v_proj.bias.requires_grad = true
			l.out_proj.weights.requires_grad = true
			l.out_proj.bias.requires_grad = true
		case FFNLayer:
			l.fc1.weights.requires_grad = true
			l.fc1.bias.requires_grad = true
			l.fc2.weights.requires_grad = true
			l.fc2.bias.requires_grad = true
		case RNNLayer:
			l.w_ih.requires_grad = true
			l.w_hh.requires_grad = true
			l.bias.requires_grad = true
		case GRULayer:
			l.w_ih.requires_grad = true
			l.w_hh.requires_grad = true
			l.bias.requires_grad = true
		case LSTMLayer:
			l.w_ih.requires_grad = true
			l.w_hh.requires_grad = true
			l.bias.requires_grad = true
		case TransformerEncoderBlock:
			l.ln1.gamma.requires_grad = true
			l.ln1.beta.requires_grad = true
			l.mha.q_proj.weights.requires_grad = true
			l.mha.q_proj.bias.requires_grad = true
			l.mha.k_proj.weights.requires_grad = true
			l.mha.k_proj.bias.requires_grad = true
			l.mha.v_proj.weights.requires_grad = true
			l.mha.v_proj.bias.requires_grad = true
			l.mha.out_proj.weights.requires_grad = true
			l.mha.out_proj.bias.requires_grad = true
			l.ln2.gamma.requires_grad = true
			l.ln2.beta.requires_grad = true
			l.ffn.fc1.weights.requires_grad = true
			l.ffn.fc1.bias.requires_grad = true
			l.ffn.fc2.weights.requires_grad = true
			l.ffn.fc2.bias.requires_grad = true
		case TransformerEncoder:
			for i in 0 ..< len(l.blocks) {
				block := &l.blocks[i]
				block.ln1.gamma.requires_grad = true
				block.ln1.beta.requires_grad = true
				block.mha.q_proj.weights.requires_grad = true
				block.mha.q_proj.bias.requires_grad = true
				block.mha.k_proj.weights.requires_grad = true
				block.mha.k_proj.bias.requires_grad = true
				block.mha.v_proj.weights.requires_grad = true
				block.mha.v_proj.bias.requires_grad = true
				block.mha.out_proj.weights.requires_grad = true
				block.mha.out_proj.bias.requires_grad = true
				block.ln2.gamma.requires_grad = true
				block.ln2.beta.requires_grad = true
				block.ffn.fc1.weights.requires_grad = true
				block.ffn.fc1.bias.requires_grad = true
				block.ffn.fc2.weights.requires_grad = true
				block.ffn.fc2.bias.requires_grad = true
			}
		case MaxPool2dLayer, AvgPool2dLayer, DropoutLayer, Activation, FlattenLayer:
		// No trainable parameters
		}
	}
}

sequential_freeze_range :: proc(seq: ^Sequential, start: int, end: int) {
	start := start
	end := end
	n := len(seq.layers)
	start = max(0, min(start, n))
	end = max(0, min(end, n))

	for i in start ..< end {
		layer := &seq.layers[i]
		switch l in layer {
		case LinearLayer:
			l.weights.requires_grad = false
			if l.bias != nil {l.bias.requires_grad = false}
		case Conv2dLayer:
			l.weight.requires_grad = false
			if l.bias != nil {l.bias.requires_grad = false}
		case BatchNorm2dLayer:
			l.weight.requires_grad = false
			l.bias.requires_grad = false
		case LayerNormLayer:
			l.gamma.requires_grad = false
			l.beta.requires_grad = false
		case EmbeddingLayer:
			l.weight.requires_grad = false
		case MultiHeadAttentionLayer:
			l.q_proj.weights.requires_grad = false
			l.q_proj.bias.requires_grad = false
			l.k_proj.weights.requires_grad = false
			l.k_proj.bias.requires_grad = false
			l.v_proj.weights.requires_grad = false
			l.v_proj.bias.requires_grad = false
			l.out_proj.weights.requires_grad = false
			l.out_proj.bias.requires_grad = false
		case FFNLayer:
			l.fc1.weights.requires_grad = false
			l.fc1.bias.requires_grad = false
			l.fc2.weights.requires_grad = false
			l.fc2.bias.requires_grad = false
		case RNNLayer:
			l.w_ih.requires_grad = false
			l.w_hh.requires_grad = false
			l.bias.requires_grad = false
		case GRULayer:
			l.w_ih.requires_grad = false
			l.w_hh.requires_grad = false
			l.bias.requires_grad = false
		case LSTMLayer:
			l.w_ih.requires_grad = false
			l.w_hh.requires_grad = false
			l.bias.requires_grad = false
		case TransformerEncoderBlock:
			l.ln1.gamma.requires_grad = false
			l.ln1.beta.requires_grad = false
			l.mha.q_proj.weights.requires_grad = false
			l.mha.q_proj.bias.requires_grad = false
			l.mha.k_proj.weights.requires_grad = false
			l.mha.k_proj.bias.requires_grad = false
			l.mha.v_proj.weights.requires_grad = false
			l.mha.v_proj.bias.requires_grad = false
			l.mha.out_proj.weights.requires_grad = false
			l.mha.out_proj.bias.requires_grad = false
			l.ln2.gamma.requires_grad = false
			l.ln2.beta.requires_grad = false
			l.ffn.fc1.weights.requires_grad = false
			l.ffn.fc1.bias.requires_grad = false
			l.ffn.fc2.weights.requires_grad = false
			l.ffn.fc2.bias.requires_grad = false
		case TransformerEncoder:
			for i in 0 ..< len(l.blocks) {
				block := &l.blocks[i]
				block.ln1.gamma.requires_grad = false
				block.ln1.beta.requires_grad = false
				block.mha.q_proj.weights.requires_grad = false
				block.mha.q_proj.bias.requires_grad = false
				block.mha.k_proj.weights.requires_grad = false
				block.mha.k_proj.bias.requires_grad = false
				block.mha.v_proj.weights.requires_grad = false
				block.mha.v_proj.bias.requires_grad = false
				block.mha.out_proj.weights.requires_grad = false
				block.mha.out_proj.bias.requires_grad = false
				block.ln2.gamma.requires_grad = false
				block.ln2.beta.requires_grad = false
				block.ffn.fc1.weights.requires_grad = false
				block.ffn.fc1.bias.requires_grad = false
				block.ffn.fc2.weights.requires_grad = false
				block.ffn.fc2.bias.requires_grad = false
			}
		case MaxPool2dLayer, AvgPool2dLayer, DropoutLayer, Activation, FlattenLayer:
		// No trainable parameters
		}
	}
}

// GPT model freeze/unfreeze
gpt_freeze_all :: proc(model: ^GPTModel) {
	model.token_emb.weight.requires_grad = false
	model.pos_emb.weight.requires_grad = false

	for i in 0 ..< len(model.blocks) {
		block := &model.blocks[i]
		block.ln1.gamma.requires_grad = false
		block.ln1.beta.requires_grad = false
		block.mha.q_proj.weights.requires_grad = false
		block.mha.q_proj.bias.requires_grad = false
		block.mha.k_proj.weights.requires_grad = false
		block.mha.k_proj.bias.requires_grad = false
		block.mha.v_proj.weights.requires_grad = false
		block.mha.v_proj.bias.requires_grad = false
		block.mha.out_proj.weights.requires_grad = false
		block.mha.out_proj.bias.requires_grad = false
		block.ln2.gamma.requires_grad = false
		block.ln2.beta.requires_grad = false
		block.ffn.fc1.weights.requires_grad = false
		block.ffn.fc1.bias.requires_grad = false
		block.ffn.fc2.weights.requires_grad = false
		block.ffn.fc2.bias.requires_grad = false
	}

	model.final_ln.gamma.requires_grad = false
	model.final_ln.beta.requires_grad = false
	model.output_proj.weights.requires_grad = false
	model.output_proj.bias.requires_grad = false
}

gpt_unfreeze_all :: proc(model: ^GPTModel) {
	model.token_emb.weight.requires_grad = true
	model.pos_emb.weight.requires_grad = true

	for i in 0 ..< len(model.blocks) {
		block := &model.blocks[i]
		block.ln1.gamma.requires_grad = true
		block.ln1.beta.requires_grad = true
		block.mha.q_proj.weights.requires_grad = true
		block.mha.q_proj.bias.requires_grad = true
		block.mha.k_proj.weights.requires_grad = true
		block.mha.k_proj.bias.requires_grad = true
		block.mha.v_proj.weights.requires_grad = true
		block.mha.v_proj.bias.requires_grad = true
		block.mha.out_proj.weights.requires_grad = true
		block.mha.out_proj.bias.requires_grad = true
		block.ln2.gamma.requires_grad = true
		block.ln2.beta.requires_grad = true
		block.ffn.fc1.weights.requires_grad = true
		block.ffn.fc1.bias.requires_grad = true
		block.ffn.fc2.weights.requires_grad = true
		block.ffn.fc2.bias.requires_grad = true
	}

	model.final_ln.gamma.requires_grad = true
	model.final_ln.beta.requires_grad = true
	model.output_proj.weights.requires_grad = true
	model.output_proj.bias.requires_grad = true
}

gpt_freeze_blocks :: proc(model: ^GPTModel, start: int, end: int) {
	start := start
	end := end
	n := len(model.blocks)
	start = max(0, min(start, n))
	end = max(0, min(end, n))

	for i in start ..< end {
		block := &model.blocks[i]
		block.ln1.gamma.requires_grad = false
		block.ln1.beta.requires_grad = false
		block.mha.q_proj.weights.requires_grad = false
		block.mha.q_proj.bias.requires_grad = false
		block.mha.k_proj.weights.requires_grad = false
		block.mha.k_proj.bias.requires_grad = false
		block.mha.v_proj.weights.requires_grad = false
		block.mha.v_proj.bias.requires_grad = false
		block.mha.out_proj.weights.requires_grad = false
		block.mha.out_proj.bias.requires_grad = false
		block.ln2.gamma.requires_grad = false
		block.ln2.beta.requires_grad = false
		block.ffn.fc1.weights.requires_grad = false
		block.ffn.fc1.bias.requires_grad = false
		block.ffn.fc2.weights.requires_grad = false
		block.ffn.fc2.bias.requires_grad = false
	}
}

// BERT model freeze/unfreeze
bert_freeze_all :: proc(model: ^BERTModel) {
	model.token_emb.weight.requires_grad = false
	model.pos_emb.weight.requires_grad = false
	model.segment_emb.weight.requires_grad = false
	model.emb_ln.gamma.requires_grad = false
	model.emb_ln.beta.requires_grad = false

	for i in 0 ..< len(model.encoder_blocks) {
		block := &model.encoder_blocks[i]
		block.ln1.gamma.requires_grad = false
		block.ln1.beta.requires_grad = false
		block.mha.q_proj.weights.requires_grad = false
		block.mha.q_proj.bias.requires_grad = false
		block.mha.k_proj.weights.requires_grad = false
		block.mha.k_proj.bias.requires_grad = false
		block.mha.v_proj.weights.requires_grad = false
		block.mha.v_proj.bias.requires_grad = false
		block.mha.out_proj.weights.requires_grad = false
		block.mha.out_proj.bias.requires_grad = false
		block.ln2.gamma.requires_grad = false
		block.ln2.beta.requires_grad = false
		block.ffn.fc1.weights.requires_grad = false
		block.ffn.fc1.bias.requires_grad = false
		block.ffn.fc2.weights.requires_grad = false
		block.ffn.fc2.bias.requires_grad = false
	}

	model.pooler.weights.requires_grad = false
	model.pooler.bias.requires_grad = false
	model.mlm_head.weights.requires_grad = false
	model.mlm_head.bias.requires_grad = false
	model.nsp_head.weights.requires_grad = false
	model.nsp_head.bias.requires_grad = false
}

bert_unfreeze_all :: proc(model: ^BERTModel) {
	model.token_emb.weight.requires_grad = true
	model.pos_emb.weight.requires_grad = true
	model.segment_emb.weight.requires_grad = true
	model.emb_ln.gamma.requires_grad = true
	model.emb_ln.beta.requires_grad = true

	for i in 0 ..< len(model.encoder_blocks) {
		block := &model.encoder_blocks[i]
		block.ln1.gamma.requires_grad = true
		block.ln1.beta.requires_grad = true
		block.mha.q_proj.weights.requires_grad = true
		block.mha.q_proj.bias.requires_grad = true
		block.mha.k_proj.weights.requires_grad = true
		block.mha.k_proj.bias.requires_grad = true
		block.mha.v_proj.weights.requires_grad = true
		block.mha.v_proj.bias.requires_grad = true
		block.mha.out_proj.weights.requires_grad = true
		block.mha.out_proj.bias.requires_grad = true
		block.ln2.gamma.requires_grad = true
		block.ln2.beta.requires_grad = true
		block.ffn.fc1.weights.requires_grad = true
		block.ffn.fc1.bias.requires_grad = true
		block.ffn.fc2.weights.requires_grad = true
		block.ffn.fc2.bias.requires_grad = true
	}

	model.pooler.weights.requires_grad = true
	model.pooler.bias.requires_grad = true
	model.mlm_head.weights.requires_grad = true
	model.mlm_head.bias.requires_grad = true
	model.nsp_head.weights.requires_grad = true
	model.nsp_head.bias.requires_grad = true
}

bert_freeze_blocks :: proc(model: ^BERTModel, start: int, end: int) {
	start := start
	end := end
	n := len(model.encoder_blocks)
	start = max(0, min(start, n))
	end = max(0, min(end, n))

	for i in start ..< end {
		block := &model.encoder_blocks[i]
		block.ln1.gamma.requires_grad = false
		block.ln1.beta.requires_grad = false
		block.mha.q_proj.weights.requires_grad = false
		block.mha.q_proj.bias.requires_grad = false
		block.mha.k_proj.weights.requires_grad = false
		block.mha.k_proj.bias.requires_grad = false
		block.mha.v_proj.weights.requires_grad = false
		block.mha.v_proj.bias.requires_grad = false
		block.mha.out_proj.weights.requires_grad = false
		block.mha.out_proj.bias.requires_grad = false
		block.ln2.gamma.requires_grad = false
		block.ln2.beta.requires_grad = false
		block.ffn.fc1.weights.requires_grad = false
		block.ffn.fc1.bias.requires_grad = false
		block.ffn.fc2.weights.requires_grad = false
		block.ffn.fc2.bias.requires_grad = false
	}
}
