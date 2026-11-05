package ml_finance

import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// Financial Sentiment Analyzer (BERT + Classification Head)
// ============================================================================
// BERT's pooled [CLS] representation (tanh of the pooler) through a 3-class head, as
// BertForSequenceClassification: with ProsusAI/finbert's weights
// (nn.load_bert_model, scripts/convert_bert.py) the logits equal Hugging Face's.
// The head's classes follow `labels`: FinBERT's order is positive, negative, neutral
// (SENTIMENT_FINBERT_LABELS); a new analyzer's is negative, neutral, positive. Read the
// logits through sentiment_probabilities rather than by position. (Callers used to
// hard-code one order or the other; the MLM head over the whole vocabulary was computed
// for every text and discarded; freeing leaked the BERT model.)

SentimentLabel :: enum {
	Negative,
	Neutral,
	Positive,
}

// ProsusAI/finbert's id2label
SENTIMENT_FINBERT_LABELS :: [3]SentimentLabel{.Positive, .Negative, .Neutral}
SENTIMENT_DEFAULT_LABELS :: [3]SentimentLabel{.Negative, .Neutral, .Positive}

SentimentAnalyzer :: struct {
	bert:      ^nn.BERTModel,
	allocator: mem.Allocator,
	labels:    [3]SentimentLabel, // the label of each output class
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
	analyzer.labels = SENTIMENT_DEFAULT_LABELS
	analyzer.bert = new(nn.BERTModel, allocator)
	analyzer.bert^ = nn.bert_model_new(
		vocab_size,
		d_model,
		num_heads,
		d_ff,
		num_layers,
		max_seq_len,
		allocator = allocator,
	)
	// the 2-class NSP head becomes the 3-class sentiment head (it reads the pooled [CLS])
	nn.bert_replace_nsp_head(analyzer.bert, 3, allocator)
	return analyzer
}

// A FinBERT checkpoint (scripts/convert_bert.py ProsusAI/finbert ...): its classifier is
// the NSP head, with FinBERT's label order and LayerNorm epsilon (1e-12; the checkpoint
// does not store it, and BERT's default 1e-5 moved the logits by ~5e-4)
sentiment_analyzer_load :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	analyzer: SentimentAnalyzer,
	ok: bool,
) {
	bert: ^nn.BERTModel
	bert, ok = nn.load_bert_model(path, allocator)
	if !ok {return}
	nn.bert_set_layer_norm_eps(bert, 1e-12) // FinBERT's layer_norm_eps
	return SentimentAnalyzer{bert = bert, allocator = allocator, labels = SENTIMENT_FINBERT_LABELS}, true
}

sentiment_analyzer_free :: proc(analyzer: ^SentimentAnalyzer) {
	if analyzer.bert != nil {
		nn.bert_model_free(analyzer.bert)
		free(analyzer.bert, analyzer.allocator)
		analyzer.bert = nil
	}
}

// Sentiment logits [batch, 3] in the order of analyzer.labels
// input_ids:   [batch, seq_len] (id 0 is padding)
// segment_ids: [batch, seq_len] (all 0 for single sentences)
analyze_text :: proc(
	analyzer: ^SentimentAnalyzer,
	input_ids: ^t.Tensor,
	segment_ids: ^t.Tensor,
	training: bool,
) -> ^t.Tensor {
	_, logits := nn.bert_model_forward(analyzer.bert, input_ids, segment_ids, training, skip_mlm = true)
	return logits
}

// Softmax probabilities of row `row` of the logits, by label
sentiment_probabilities :: proc(
	analyzer: ^SentimentAnalyzer,
	logits: ^t.Tensor,
	row := 0,
) -> (
	probs: [SentimentLabel]f64,
) {
	z := logits.data.data[row * 3:][:3]
	m := max(z[0], z[1], z[2])
	s := 0.0
	for v in z {s += math.exp(v - m)}
	for v, c in z {probs[analyzer.labels[c]] = math.exp(v - m) / s}
	return
}

// The most likely label of row `row` and its probability
sentiment_predict :: proc(
	analyzer: ^SentimentAnalyzer,
	logits: ^t.Tensor,
	row := 0,
) -> (
	label: SentimentLabel,
	probability: f64,
) {
	probs := sentiment_probabilities(analyzer, logits, row)
	for p, lab in probs {
		if p > probability {label, probability = lab, p}
	}
	return
}
