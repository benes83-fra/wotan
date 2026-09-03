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
	bert:      ^nn.BERTModel,
	allocator: mem.Allocator,
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

	// 1. Initialize the base BERT model
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

	// 2. Replace the default 2-class NSP head with a 3-class Sentiment head.
	// The BERT forward pass already routes the [CLS] pooled representation through this head,
	// so we can just hijack the `nsp_logits` return value for our sentiment predictions!
	nn.bert_replace_nsp_head(analyzer.bert, 3, allocator)

	return analyzer
}

sentiment_analyzer_free :: proc(analyzer: ^SentimentAnalyzer) {
	if analyzer.bert != nil {
		nn.bert_model_free(analyzer.bert)
	}
}

// analyze_text takes tokenized inputs and returns sentiment logits [Batch, 3]
// input_ids:   [Batch, Seq_Len]
// segment_ids: [Batch, Seq_Len] (typically all 0 for single-sentence classification)
analyze_text :: proc(
	analyzer: ^SentimentAnalyzer,
	input_ids: ^t.Tensor,
	segment_ids: ^t.Tensor,
	training: bool,
) -> ^t.Tensor {
	// bert_model_forward returns (mlm_logits, nsp_logits)
	// Since we replaced the nsp_head with a 3-class sentiment head, nsp_logits contains our sentiment predictions.
	_, sentiment_logits := nn.bert_model_forward(analyzer.bert, input_ids, segment_ids, training)

	return sentiment_logits
}
