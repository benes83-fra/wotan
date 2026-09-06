package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"

event_study_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== FinBERT Event Study & CAR Pipeline Test ===")

	symbol := "AAPL"
	fmt.printf("Fetching daily data for %s to establish baseline returns...\n", symbol)

	df := net.read_yahoo(symbol, .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("Failed to fetch data. Aborting.")
		return
	}

	close_col := &df.columns[5] // AdjClose
	date_col := &df.columns[0] // Date (Type: .Date)

	all_dates := make([]string, df.rows, allocator)
	returns := make([]f64, df.rows, allocator)
	defer {
		delete(all_dates, allocator)
		delete(returns, allocator)
	}

	fmt.println("Extracting data using w.column_at_date...")

	for i in 0 ..< df.rows {
		// ✅ FIX: Use column_at_date (or column_at_i64) instead of column_at_string
		date_val, is_null := w.column_at_date(date_col, i)
		if is_null {
			all_dates[i] = ""
			continue
		}

		// Format the date value to a string.
		// If w.column_at_date returns an i64 (Unix timestamp), this formats it.
		// If it returns a w.Date struct, you may need to use w.date_to_string(date_val) here.
		date_str := w.date_to_string(date_val, allocator = allocator)

		all_dates[i] = date_str

		if i > 0 {
			prev_close, _ := w.column_at_float(close_col, i - 1)
			curr_close, _ := w.column_at_float(close_col, i)
			if prev_close > 0.0 {
				returns[i] = (curr_close - prev_close) / prev_close
			}
		}
	}

	fmt.printf("Successfully loaded %d days of data.\n", df.rows)

	fmt.println("\nSimulating NLP Sentiment Analysis on Earnings Dates...")

	// Updated to 2025/2026 dates to ensure they fall within the .TwoYears fetch window from Sept 2026
	event_dates := []string {
		"2025-11-03", // Q4 2025
		"2026-02-02", // Q1 2026
		"2026-05-04", // Q2 2026
		"2026-08-03", // Q3 2026
	}

	for date_str in event_dates {
		sentiment := "Positive"
		if date_str == "2026-08-03" {
			sentiment = "Negative" // Mocking a mixed reaction
		}
		fmt.printf("  [%s] Detected Sentiment: %s\n", date_str, sentiment)
	}

	fmt.println("\nRunning Event Study (Cumulative Abnormal Returns)...")
	fmt.println("Estimation Window: -120 to -10 days relative to event")
	fmt.println("Event Window:      -5 to +5 days relative to event")

	car_result := ml_fin.compute_car(
		all_dates,
		returns,
		event_dates,
		-120, // est_start
		-10, // est_end
		-5, // evt_start
		5, // evt_end
		allocator,
	)
	defer ml_fin.event_study_result_free(&car_result, allocator)

	fmt.println("\n=== Event Study Results ===")
	fmt.printf("Events Analyzed: %d\n", car_result.num_events)

	if car_result.num_events > 0 {
		fmt.printf("Mean CAR (-5 to +5 days): %+.4f%%\n", car_result.mean_car * 100.0)
		fmt.printf("T-Statistic:              %.4f\n", car_result.t_statistic)

		fmt.println("\nPer-Event CAR:")
		for i in 0 ..< car_result.num_events {
			fmt.printf("  %s : %+.4f%%\n", car_result.dates[i], car_result.car_values[i] * 100.0)
		}

		fmt.println("\n--- Interpretation ---")
		if math.abs(car_result.t_statistic) > 1.96 {
			fmt.println(
				"✓ STATISTICALLY SIGNIFICANT: The NLP-detected events have a measurable impact on price.",
			)
		} else {
			fmt.println(
				"~ NOT SIGNIFICANT: The average CAR is not statistically different from zero.",
			)
		}
	} else {
		fmt.println("⚠ WARNING: No events matched the dataset. Check date formatting.")
	}

	fmt.println("\n✓ Event Study Pipeline Test Complete!")

}


// ============================================================================
// Lightweight Mock Tokenizer for End-to-End Testing
// (In production, replace with a real WordPiece/BPE tokenizer loading vocab.json)
// ============================================================================
mock_tokenize :: proc(
	text: string,
	max_seq_len: int,
	allocator: mem.Allocator,
) -> (
	input_ids: ^t.Tensor,
	segment_ids: ^t.Tensor,
) {
	batch := 1

	// Allocate flat matrices for [CLS] + tokens + [SEP]
	ids_data := l.matrix_new(f64, 1, batch * max_seq_len, allocator)
	seg_data := l.matrix_new(f64, 1, batch * max_seq_len, allocator)

	// 101 = [CLS], 102 = [SEP] in standard BERT
	ids_data.data[0] = 101.0
	seg_data.data[0] = 0.0 // Segment A

	// Fill middle with dummy token IDs representing the text
	word_count := len(strings.split(text, " "))
	for i in 1 ..< max_seq_len - 1 {
		if i <= word_count {
			ids_data.data[i] = 1000.0 + f64(i) // Mock token ID
			seg_data.data[i] = 0.0
		} else {
			ids_data.data[i] = 0.0 // Padding
			seg_data.data[i] = 0.0
		}
	}

	// Add [SEP] if it fits
	if max_seq_len > 1 {
		ids_data.data[max_seq_len - 1] = 102.0
		seg_data.data[max_seq_len - 1] = 0.0
	}

	// Create tensors with shape [Batch, Seq_Len, 1, 1]
	in_tensor := t.tensor_new(ids_data, false, allocator)
	in_tensor.shape = [4]int{batch, max_seq_len, 1, 1}

	seg_tensor := t.tensor_new(seg_data, false, allocator)
	seg_tensor.shape = [4]int{batch, max_seq_len, 1, 1}

	return in_tensor, seg_tensor
}


event_study_nlp_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Real NLP Integration & Event Study Pipeline ===")

	symbol := "AAPL"
	fmt.printf("Fetching daily data for %s to establish baseline returns...\n", symbol)

	df := net.read_yahoo(symbol, .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("Failed to fetch data. Aborting.")
		return
	}

	close_col := &df.columns[5] // AdjClose
	date_col := &df.columns[0] // Date

	all_dates := make([]string, df.rows, allocator)
	returns := make([]f64, df.rows, allocator)
	defer {
		// Free the strings we allocated with fmt.aprintf
		for i in 0 ..< df.rows {
			if len(all_dates[i]) > 0 {
				delete(all_dates[i], allocator)
			}
		}
		delete(all_dates, allocator)
		delete(returns, allocator)
	}

	fmt.println("Extracting data using w.column_at_date...")
	for i in 0 ..< df.rows {
		date_val, is_null := w.column_at_date(date_col, i)
		if is_null {
			all_dates[i] = ""
			continue
		}
		// Memory-safe string allocation
		all_dates[i] = w.date_to_string(date_val, allocator)

		if i > 0 {
			prev_close, _ := w.column_at_float(close_col, i - 1)
			curr_close, _ := w.column_at_float(close_col, i)
			if prev_close > 0.0 {
				returns[i] = (curr_close - prev_close) / prev_close
			}
		}
	}
	fmt.printf("Successfully loaded %d days of data.\n", df.rows)

	// ========================================================================
	// 1. Initialize FinBERT Sentiment Analyzer
	// ========================================================================
	fmt.println("\nInitializing FinBERT Sentiment Analyzer (2 layers for fast testing)...")
	vocab_size := 30522
	d_model := 768
	num_heads := 12
	d_ff := 3072
	num_layers := 2 // Keep small for quick test execution
	max_seq_len := 128

	analyzer := ml_fin.sentiment_analyzer_new(
		vocab_size,
		d_model,
		num_heads,
		d_ff,
		num_layers,
		max_seq_len,
		allocator,
	)
	defer ml_fin.sentiment_analyzer_free(&analyzer)
	fmt.println("✓ Analyzer initialized.")

	// ========================================================================
	// 2. Simulate NLP Pipeline: Fetch Text -> Tokenize -> Predict Sentiment
	// ========================================================================
	fmt.println("\nRunning NLP Pipeline on Earnings Dates...")

	// Realistic mock earnings summaries
	event_data := []struct {
		date: string,
		text: string,
	} {
		{
			"2025-11-03",
			"Apple reports record-breaking iPhone sales, exceeding all analyst expectations. Services revenue also hit an all-time high, driving strong margin expansion.",
		},
		{
			"2026-02-02",
			"Strong guidance for the upcoming quarter driven by AI integration in macOS. Management raised full-year revenue outlook.",
		},
		{
			"2026-05-04",
			"Solid results, but supply chain constraints in Asia may impact next quarter's hardware shipments slightly. Overall demand remains robust.",
		},
		{
			"2026-08-03",
			"Apple misses revenue estimates due to unexpected slowdown in China market. CEO warns of challenging macroeconomic headwinds and delayed product cycles.",
		},
	}

	nlp_filtered_dates := make([dynamic]string, allocator)
	defer delete(nlp_filtered_dates)

	for event, _ in event_data {
		fmt.printf("\n  Processing: %s\n", event.date)
		fmt.printf("  Text: \"%s...\"\n", event.text[:min(60, len(event.text))])

		// 1. Tokenize
		input_ids, segment_ids := mock_tokenize(event.text, max_seq_len, allocator)
		defer {
			t.tensor_free(input_ids)
			t.tensor_free(segment_ids)
		}

		// 2. Forward Pass through FinBERT
		// Returns logits of shape [Batch, 1, 3, 1] -> [Negative, Neutral, Positive]
		logits := ml_fin.analyze_text(&analyzer, input_ids, segment_ids, false)
		defer t.tensor_free(logits)

		// 3. Argmax to get sentiment label
		// Extract the 3 logits for the single batch item
		batch_offset := 0 * (1 * 3 * 1) // batch=0, seq=0, features=3
		neg_score := logits.data.data[batch_offset + 0]
		neu_score := logits.data.data[batch_offset + 1]
		pos_score := logits.data.data[batch_offset + 2]

		sentiment: ml_fin.SentimentLabel
		sentiment_str: string

		if pos_score >= neg_score && pos_score >= neu_score {
			sentiment = .Positive
			sentiment_str = "Positive"
		} else if neg_score >= pos_score && neg_score >= neu_score {
			sentiment = .Negative
			sentiment_str = "Negative"
		} else {
			sentiment = .Neutral
			sentiment_str = "Neutral"
		}

		fmt.printf(
			"  Scores -> Neg: %.4f, Neu: %.4f, Pos: %.4f\n",
			neg_score,
			neu_score,
			pos_score,
		)
		fmt.printf("  ➔ Detected Sentiment: %s\n", sentiment_str)

		// 4. Filter: Only run Event Study on strong Positive or Negative signals
		if sentiment == .Positive || sentiment == .Negative {
			append(&nlp_filtered_dates, event.date)
		}
	}

	// ========================================================================
	// 3. Run Event Study on NLP-Filtered Dates
	// ========================================================================
	fmt.println("\nRunning Event Study (Cumulative Abnormal Returns)...")
	fmt.println("Estimation Window: -120 to -10 days relative to event")
	fmt.println("Event Window:      -5 to +5 days relative to event")

	// Convert dynamic array to fixed slice for compute_car
	filtered_slice := make([]string, len(nlp_filtered_dates), allocator)
	for i in 0 ..< len(nlp_filtered_dates) {
		filtered_slice[i] = nlp_filtered_dates[i]
	}

	car_result := ml_fin.compute_car(
		all_dates,
		returns,
		filtered_slice,
		-120, // est_start
		-10, // est_end
		-5, // evt_start
		5, // evt_end
		allocator,
	)
	defer ml_fin.event_study_result_free(&car_result, allocator)

	// ========================================================================
	// 4. Print Results
	// ========================================================================
	fmt.println("\n=== NLP-Driven Event Study Results ===")
	fmt.printf("Total Events Scanned: %d\n", len(event_data))
	fmt.printf("Events Triggering Trade (Pos/Neg): %d\n", car_result.num_events)

	if car_result.num_events > 0 {
		fmt.printf("Mean CAR (-5 to +5 days): %+.4f%%\n", car_result.mean_car * 100.0)
		fmt.printf("T-Statistic:              %.4f\n", car_result.t_statistic)

		fmt.println("\nPer-Event CAR:")
		for i in 0 ..< car_result.num_events {
			fmt.printf("  %s : %+.4f%%\n", car_result.dates[i], car_result.car_values[i] * 100.0)
		}

		fmt.println("\n--- Interpretation ---")
		if math.abs(car_result.t_statistic) > 1.96 {
			fmt.println(
				"✓ STATISTICALLY SIGNIFICANT: The NLP-detected sentiment events have a measurable impact on price.",
			)
		} else {
			fmt.println(
				"~ NOT SIGNIFICANT: The average CAR is not statistically different from zero (efficient market).",
			)
		}
	} else {
		fmt.println("⚠ WARNING: No NLP events met the sentiment threshold for analysis.")
	}

	fmt.println("\n✓ Real NLP Integration Pipeline Test Complete!")
}


event_study_tokenizer_nlp_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Real NLP Integration & Event Study Pipeline ===")

	symbol := "AAPL"
	fmt.printf("Fetching daily data for %s to establish baseline returns...\n", symbol)

	df := net.read_yahoo(symbol, .Daily, .TwoYears, allocator)
	defer w.destroy_dataframe(&df)

	if df.rows == 0 {
		fmt.println("Failed to fetch data. Aborting.")
		return
	}

	close_col := &df.columns[5] // AdjClose
	date_col := &df.columns[0] // Date

	all_dates := make([]string, df.rows, allocator)
	returns := make([]f64, df.rows, allocator)
	defer {
		for i in 0 ..< df.rows {
			if len(all_dates[i]) > 0 {
				delete(all_dates[i], allocator)
			}
		}
		delete(all_dates, allocator)
		delete(returns, allocator)
	}

	fmt.println("Extracting data using w.column_at_date...")
	for i in 0 ..< df.rows {
		date_val, is_null := w.column_at_date(date_col, i)
		if is_null {
			all_dates[i] = ""
			continue
		}
		all_dates[i] = w.date_to_string(date_val, allocator)

		if i > 0 {
			prev_close, _ := w.column_at_float(close_col, i - 1)
			curr_close, _ := w.column_at_float(close_col, i)
			if prev_close > 0.0 {
				returns[i] = (curr_close - prev_close) / prev_close
			}
		}
	}
	fmt.printf("Successfully loaded %d days of data.\n", df.rows)

	// ========================================================================
	// 1. Fetch or Load Real WordPiece Vocabulary
	// ========================================================================
	fmt.println("\nLoading WordPiece Tokenizer Vocabulary...")
	vocab_path := "bert_vocab.txt"

	_, err := os.read_entire_file(vocab_path, allocator)
	if err != nil {
		fmt.println(
			"vocab.txt not found locally. Downloading from HuggingFace (bert-base-uncased)...",
		)
		url := "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
		vocab_data_str, ok := net.http_get(url, allocator)
		if !ok {
			fmt.println("Failed to download vocab.txt. Aborting.")
			return
		}

		file, create_err := os.create(vocab_path)
		if create_err != nil {
			fmt.printf("Failed to create vocab file: %v\n", create_err)
			delete(vocab_data_str, allocator)
			return
		}
		os.write(file, transmute([]u8)vocab_data_str)
		os.close(file)
		delete(vocab_data_str, allocator)
		fmt.println("✓ Vocabulary downloaded and cached successfully.")
	} else {
		fmt.println("✓ Found local vocab.txt")
	}

	max_seq_len := 512
	tokenizer, ok := tok.wordpiece_tokenizer_new(vocab_path, max_seq_len, allocator)
	if !ok {
		fmt.println("Failed to load tokenizer.")
		return
	}
	defer tok.wordpiece_tokenizer_free(&tokenizer)
	fmt.printf("✓ Tokenizer loaded with %d tokens.\n", len(tokenizer.ids_to_tokens))

	// ========================================================================
	// 2. Load Pre-trained FinBERT Model (or fallback to random)
	// ========================================================================
	fmt.println("\nLoading FinBERT Model...")
	checkpoint_path := "finbert_checkpoint.bin"

	bert_model: ^nn.BERTModel
	ok_load := false

	// ✅ Try to load pre-trained weights using your existing persistence module
	bert_model, ok_load = nn.load_bert_model(checkpoint_path, allocator)

	if !ok_load {
		fmt.println(
			"⚠ Checkpoint not found or invalid. Initializing random model for demonstration...",
		)
		fmt.println(
			"  (To see real sentiment, place a valid finbert_checkpoint.bin in the directory)",
		)

		// Fallback to random initialization
		vocab_size := len(tokenizer.ids_to_tokens)
		d_model := 768
		num_heads := 12
		d_ff := 3072
		num_layers := 2 // Keep small for quick test execution if random

		bert_model = new(nn.BERTModel, allocator)
		bert_model^ = nn.bert_model_new(
			vocab_size,
			d_model,
			num_heads,
			d_ff,
			num_layers,
			max_seq_len,
			allocator,
		)
		// Replace NSP head with 3-class sentiment head
		nn.bert_replace_nsp_head(bert_model, 3, allocator)
	} else {
		fmt.println("✓ Pre-trained FinBERT model loaded successfully from checkpoint!")
	}

	analyzer := ml_fin.SentimentAnalyzer {
		bert      = bert_model,
		allocator = allocator,
	}
	defer ml_fin.sentiment_analyzer_free(&analyzer)

	// ========================================================================
	// 3. Real NLP Pipeline: Tokenize -> Predict Sentiment
	// ========================================================================
	fmt.println("\nRunning NLP Pipeline on Earnings Dates...")

	event_data := []struct {
		date: string,
		text: string,
	} {
		{
			"2025-11-03",
			"Apple reports record-breaking iPhone sales, exceeding all analyst expectations. Services revenue also hit an all-time high, driving strong margin expansion.",
		},
		{
			"2026-02-02",
			"Strong guidance for the upcoming quarter driven by AI integration in macOS. Management raised full-year revenue outlook.",
		},
		{
			"2026-05-04",
			"Solid results, but supply chain constraints in Asia may impact next quarter's hardware shipments slightly. Overall demand remains robust.",
		},
		{
			"2026-08-03",
			"Apple misses revenue estimates due to unexpected slowdown in China market. CEO warns of challenging macroeconomic headwinds and delayed product cycles.",
		},
	}

	nlp_filtered_dates := make([dynamic]string, allocator)
	defer delete(nlp_filtered_dates)

	for event, _ in event_data {
		fmt.printf("\n  Processing: %s\n", event.date)
		fmt.printf("  Text: \"%s...\"\n", event.text[:min(60, len(event.text))])

		input_ids, segment_ids := tok.tokenize_to_tensors(&tokenizer, event.text, allocator)
		defer {
			t.tensor_free(input_ids)
			t.tensor_free(segment_ids)
		}

		logits := ml_fin.analyze_text(&analyzer, input_ids, segment_ids, false)
		defer t.tensor_free(logits)

		batch_offset := 0 * (1 * 3 * 1)
		neg_score := logits.data.data[batch_offset + 0]
		neu_score := logits.data.data[batch_offset + 1]
		pos_score := logits.data.data[batch_offset + 2]

		sentiment: ml_fin.SentimentLabel
		sentiment_str: string

		if pos_score >= neg_score && pos_score >= neu_score {
			sentiment = .Positive
			sentiment_str = "Positive"
		} else if neg_score >= pos_score && neg_score >= neu_score {
			sentiment = .Negative
			sentiment_str = "Negative"
		} else {
			sentiment = .Neutral
			sentiment_str = "Neutral"
		}

		fmt.printf(
			"  Scores -> Neg: %.4f, Neu: %.4f, Pos: %.4f\n",
			neg_score,
			neu_score,
			pos_score,
		)
		fmt.printf("  ➔ Detected Sentiment: %s\n", sentiment_str)

		if sentiment == .Positive || sentiment == .Negative {
			append(&nlp_filtered_dates, event.date)
		}
	}

	// ========================================================================
	// 4. Run Event Study on NLP-Filtered Dates
	// ========================================================================
	fmt.println("\nRunning Event Study (Cumulative Abnormal Returns)...")
	fmt.println("Estimation Window: -120 to -10 days relative to event")
	fmt.println("Event Window:      -5 to +5 days relative to event")

	filtered_slice := make([]string, len(nlp_filtered_dates), allocator)
	for i in 0 ..< len(nlp_filtered_dates) {
		filtered_slice[i] = nlp_filtered_dates[i]
	}

	car_result := ml_fin.compute_car(
		all_dates,
		returns,
		filtered_slice,
		-120, // est_start
		-10, // est_end
		-5, // evt_start
		5, // evt_end
		allocator,
	)
	defer ml_fin.event_study_result_free(&car_result, allocator)

	// ========================================================================
	// 5. Print Results
	// ========================================================================
	fmt.println("\n=== NLP-Driven Event Study Results ===")
	fmt.printf("Total Events Scanned: %d\n", len(event_data))
	fmt.printf("Events Triggering Trade (Pos/Neg): %d\n", car_result.num_events)

	if car_result.num_events > 0 {
		fmt.printf("Mean CAR (-5 to +5 days): %+.4f%%\n", car_result.mean_car * 100.0)
		fmt.printf("T-Statistic:              %.4f\n", car_result.t_statistic)

		fmt.println("\nPer-Event CAR:")
		for i in 0 ..< car_result.num_events {
			fmt.printf("  %s : %+.4f%%\n", car_result.dates[i], car_result.car_values[i] * 100.0)
		}

		fmt.println("\n--- Interpretation ---")
		if math.abs(car_result.t_statistic) > 1.96 {
			fmt.println(
				"✓ STATISTICALLY SIGNIFICANT: The NLP-detected sentiment events have a measurable impact on price.",
			)
		} else {
			fmt.println(
				"~ NOT SIGNIFICANT: The average CAR is not statistically different from zero (efficient market).",
			)
		}
	} else {
		fmt.println("⚠ WARNING: No NLP events met the sentiment threshold for analysis.")
	}

	fmt.println("\n✓ Real NLP Integration Pipeline Test Complete!")
}
