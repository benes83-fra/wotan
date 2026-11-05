package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

// Sentiment analyzer against Hugging Face's ProsusAI/finbert (BertForSequenceClassification,
// float64, the cached weights that scripts/convert_bert.py converts): the WordPiece ids and
// the 3 logits of four financial sentences, and the labels read through FinBERT's order
// (positive, negative, neutral). Also: the classification-only forward (no MLM head) gives
// the same logits as the full one. Skips the FinBERT part without the checkpoint. (With
// BERT's default LayerNorm epsilon 1e-5 instead of FinBERT's 1e-12 the logits were 5e-4 off.)

@(private = "file")
FINBERT_SENTENCES := [4]string {
	"Apple reports record-breaking iPhone sales, exceeding all analyst expectations.",
	"The company misses revenue estimates and warns of a weak second half.",
	"The board will meet on Tuesday to review the quarterly report.",
	"Shares fell 12% after the regulator opened an investigation into the bank.",
}
@(private = "file")
FINBERT_IDS := [4][48]int {
	{
		101,
		6207,
		4311,
		2501,
		1011,
		4911,
		18059,
		4341,
		1010,
		17003,
		2035,
		12941,
		10908,
		1012,
		102,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
	},
	{
		101,
		1996,
		2194,
		22182,
		6599,
		10035,
		1998,
		19428,
		1997,
		1037,
		5410,
		2117,
		2431,
		1012,
		102,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
	},
	{
		101,
		1996,
		2604,
		2097,
		3113,
		2006,
		9857,
		2000,
		3319,
		1996,
		12174,
		3189,
		1012,
		102,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
	},
	{
		101,
		6661,
		3062,
		2260,
		1003,
		2044,
		1996,
		21618,
		2441,
		2019,
		4812,
		2046,
		1996,
		2924,
		1012,
		102,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
	},
}
@(private = "file")
FINBERT_LOGITS := [4][3]f64 {
	{1.988761594058239, -1.6108040991940245, -1.5414036305510335},
	{-1.8312010084489407, 3.0334779587806247, -0.8502720125220664},
	{-1.529912436335699, 0.23637974887574684, 2.2829451745017946},
	{-1.845397108174534, 2.9873607468914973, -0.7988593091825178},
}
@(private = "file")
FINBERT_WANT := [4]ml_fin.SentimentLabel{.Positive, .Negative, .Neutral, .Negative}

sentiment_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Sentiment Analyzer Reference Checks (FinBERT) ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// 1. classification-only forward = the full forward's NSP logits (small random model)
	{
		a := ml_fin.sentiment_analyzer_new(50, 16, 2, 32, 2, 12)
		defer ml_fin.sentiment_analyzer_free(&a)
		ids := t.tensor_new(l.matrix_new(f64, 2, 12), false)
		ids.shape = {2, 12, 1, 1}
		for &v, i in ids.data.data {v = f64(1 + rand.int_max(49)) if i % 12 < 9 else 0}
		seg := t.tensor_new(l.matrix_new(f64, 2, 12), false)
		seg.shape = ids.shape
		quick := ml_fin.analyze_text(&a, ids, seg, false)
		_, full := nn.bert_model_forward(a.bert, ids, seg, false)
		same := true
		for v, i in full.data.data {same &&= v == quick.data.data[i]}
		expect(
			&passed,
			&total,
			same && a.labels == ml_fin.SENTIMENT_DEFAULT_LABELS,
			"classification-only forward equals the full forward's head logits; a new analyzer's labels are negative, neutral, positive",
		)
	}

	// 2. FinBERT vs Hugging Face
	CKPT :: "model_checkpoints/finbert_checkpoint.bin"
	VOCAB :: "model_checkpoints/bert_vocab.txt"
	if !os.exists(CKPT) || !os.exists(VOCAB) {
		fmt.println(
			"No FinBERT checkpoint / vocabulary (see docs/test_catalog.md): FinBERT checks skipped",
		)
	} else {
		analyzer, ok := ml_fin.sentiment_analyzer_load(CKPT)
		tokenizer, tok_ok := tok.wordpiece_tokenizer_new(VOCAB, 48)
		if ok && tok_ok {
			defer ml_fin.sentiment_analyzer_free(&analyzer)
			defer tok.wordpiece_tokenizer_free(&tokenizer)
			ids_ok, labels_ok := true, true
			err := 0.0
			for s, k in FINBERT_SENTENCES {
				ids, seg := tok.tokenize_to_tensors(&tokenizer, s)
				for v, i in FINBERT_IDS[k] {ids_ok &&= ids.data.data[i] == f64(v)}
				logits := ml_fin.analyze_text(&analyzer, ids, seg, false)
				for v, c in FINBERT_LOGITS[k] {err = max(err, abs(logits.data.data[c] - v))}
				label, p := ml_fin.sentiment_predict(&analyzer, logits)
				labels_ok &&= label == FINBERT_WANT[k]
				fmt.printf("   %-78s %v (%.3f)\n", s, label, p)
			}
			expect(
				&passed,
				&total,
				ids_ok && err < 1e-12 && labels_ok,
				fmt.tprintf(
					"FinBERT vs Hugging Face on 4 sentences: WordPiece ids equal %v, logits max err %.1e, labels in FinBERT's order %v",
					ids_ok,
					err,
					labels_ok,
				),
			)
		} else {
			expect(&passed, &total, false, "FinBERT checkpoint and vocabulary load")
		}
	}
	fmt.printf("Sentiment reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
