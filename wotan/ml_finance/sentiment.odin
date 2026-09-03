package ml_finance

import nn "../nn"
import t "../tensor"
import "core:mem"

// ============================================================================
// Financial Sentiment Analyzer (BERT + Classification Head)
// ============================================================================

SentimentLabel :: enum {
	Negative,
	Neutral,
	Positive,
}

SentimentAnalyzer :: struct {
	bert:       ^nn.BERTModel,
	classifier: nn.LinearLayer, // Maps [CLS] token embedding to 3 classes
	allocator:  mem.Allocator,
}

sentiment_analyzer_new :: proc(
	vocab_size: int,
	d_model: int,
	num_heads: int,
	d_ff: int,
	num_layers: int,
	max_seq_len: int,
	allocator: mem.Allocator = context.allocator,
) -> SentimentAnalyzer {
	analyzer: SentimentAnalyzer
	analyzer.allocator = allocator

	// Initialize the base BERT model (Your existing implementation)
	analyzer.bert = new(nn.BERTModel, allocator)
	analyzer.bert^ = nn.bert_model_new(
		vocab_size,
		d_model,
		num_heads,
		d_ff,
		num_layers,
		max_seq_len,
		allocator,
	)

	// Classification head on top of the [CLS] token (index 0)
	// Output is 3 logits: [Negative, Neutral, Positive]
	analyzer.classifier = nn.linear_layer_new(d_model, 3, allocator)

	return analyzer
}

sentiment_analyzer_free :: proc(analyzer: ^SentimentAnalyzer) {
	if analyzer.bert != nil {
		nn.bert_model_free(analyzer.bert)
	}
	nn.linear_layer_free(&analyzer.classifier)
}

// analyze_text takes a batch of tokenized inputs [Batch, Seq_Len] and returns sentiment logits [Batch, 3]
analyze_text :: proc(
	analyzer: ^SentimentAnalyzer,
	input_ids: ^t.Tensor, // [Batch, Seq_Len]
	attention_mask: ^t.Tensor, // [Batch, Seq_Len]
) -> ^t.Tensor {
	// 1. Forward pass through BERT
	// Output shape: [Batch, Seq_Len, d_model]
	bert_out := nn.bert_model_forward(analyzer.bert, input_ids, attention_mask)

	// 2. Extract the [CLS] token embedding (first token of each sequence)
	// We reshape to isolate the first timestep: [Batch, 1, d_model]
	// (Assuming your tensor slicing or a dedicated extract_cls_token utility exists.
	// For simplicity, we assume the first element in the seq_len dim is [CLS])
	batch := bert_out.shape[0]
	d_model := bert_out.shape[2]

	cls_data := l.matrix_new(f64, 1, batch * d_model, analyzer.allocator)
	for b in 0 ..< batch {
		for d in 0 ..< d_model {
			// Extract [CLS] which is at seq_len index 0
			cls_data.data[b * d_model + d] =
				bert_out.data.data[b * (bert_out.shape[1] * d_model) + 0 * d_model + d]
		}
	}

	cls_tensor := t.tensor_new(cls_data, bert_out.requires_grad, analyzer.allocator)
	cls_tensor.shape = [4]int{batch, 1, d_model, 1}
	defer t.tensor_free(cls_tensor)

	// 3. Pass [CLS] through the classification head
	// linear_forward handles the [Batch, 1, d_model, 1] -> [Batch, 1, 3, 1] mapping
	logits := nn.linear_forward(&analyzer.classifier, cls_tensor)

	return logits
}
