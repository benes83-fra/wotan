package tests

import ml "../wotan/analytics/ML"
import w "../wotan/core"
import fin "../wotan/finance"
import importer "../wotan/importer"
import l "../wotan/linalg"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strconv"
import "core:strings"

// ============================================================================
// Alternative Data Pipeline: GDELT News Sentiment + Yahoo Finance Prices
// ============================================================================
alt_data_gdelt_pipeline_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    ALTERNATIVE DATA PIPELINE: GDELT NEWS SENTIMENT & YAHOO PRICES")
	fmt.println("======================================================================\n")

	ticker := "AAPL"
	fmt.printf("Target Asset: %s\n", ticker)

	// ========================================================================
	// 1. Scrape Alternative Data (GDELT API)
	// ========================================================================
	fmt.println("\n1. Scraping GDELT News Data (Last 365 Days)...")
	// GDELT DOC API returns daily article counts and average tone for a query
	gdelt_url := fmt.tprintf(
		"https://api.gdeltproject.org/api/v2/doc/doc?query=%s&timespan=365d&format=csv",
		ticker,
	)

	gdelt_text, ok := net.http_get(gdelt_url, context.temp_allocator)
	if !ok {
		fmt.println("ERROR: Failed to fetch GDELT data")
		return
	}
	defer delete(gdelt_text, context.temp_allocator)

	// Debug: Print first 300 chars to see if it's actually CSV or an error message
	fmt.printf(
		"   DEBUG: First 300 chars of response:\n   %s\n",
		gdelt_text[:min(300, len(gdelt_text))],
	)

	// Parse the CSV
	gdelt_df := importer.csv_load_from_string(gdelt_text, allocator)
	fmt.printf("   Scraped %d rows of news data\n", gdelt_df.rows)
	fmt.printf("   GDELT DataFrame has %d columns\n", len(gdelt_df.columns))

	// Debug: Print first few column names
	for i in 0 ..< min(10, len(gdelt_df.columns)) {
		fmt.printf("   Col %d: %s\n", i, gdelt_df.columns[i].name)
	}

	num_articles_col: ^w.Column = nil
	avg_tone_col: ^w.Column = nil
	date_col: ^w.Column = nil

	// Try to find columns by name (case-insensitive)
	for col, i in gdelt_df.columns {
		lower_name := strings.to_lower(col.name)
		if lower_name == "numarticles" {
			num_articles_col = &gdelt_df.columns[i]
		}
		if lower_name == "avgtone" {
			avg_tone_col = &gdelt_df.columns[i]
		}
		if lower_name == "sqldate" || lower_name == "day" {
			date_col = &gdelt_df.columns[i]
		}
	}

	// Fallback to known GDELT v2 DOC API indices:
	// 1: SQLDATE, 34: NumArticles, 35: AvgTone
	if num_articles_col == nil && len(gdelt_df.columns) > 34 {
		num_articles_col = &gdelt_df.columns[34]
	}
	if avg_tone_col == nil && len(gdelt_df.columns) > 35 {
		avg_tone_col = &gdelt_df.columns[35]
	}
	if date_col == nil && len(gdelt_df.columns) > 1 {
		date_col = &gdelt_df.columns[1]
	}

	if num_articles_col == nil || avg_tone_col == nil || date_col == nil {
		fmt.println("ERROR: Could not identify GDELT columns")
		fmt.println("       The API may have returned an error or the CSV format changed.")
		return
	}

	// ========================================================================
	// 2. Fetch Price Data (Yahoo Finance)
	// ========================================================================
	fmt.println("\n2. Fetching Yahoo Finance Price Data...")
	price_df := net.read_yahoo(ticker, .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&price_df)
	fmt.printf("   Fetched %d days of price data\n", price_df.rows)

	// ========================================================================
	// 3. Align Data & Build Feature Matrix
	// ========================================================================
	fmt.println("\n3. Aligning Alternative Data with Price Data...")

	aligned_df := w.dataframe_new()
	articles_col := w.column_new("NumArticles", .Float, 0)
	tone_col := w.column_new("AvgTone", .Float, 0)
	target_col := w.column_new("NextDayReturn", .Float, 0)

	min_rows := min(gdelt_df.rows, price_df.rows - 1)

	for i in 0 ..< min_rows {
		articles, _ := w.column_at_float(num_articles_col, i)
		tone, _ := w.column_at_float(avg_tone_col, i)

		price_today, _ := w.column_at_float(w.column(&price_df, "Close"), i)
		price_tomorrow, _ := w.column_at_float(w.column(&price_df, "Close"), i + 1)

		if price_today > 0 && price_tomorrow > 0 {
			next_day_return := (price_tomorrow - price_today) / price_today

			w.append_float(&articles_col, articles)
			w.append_float(&tone_col, tone)
			w.append_float(&target_col, next_day_return)
		}
	}

	w.add_column(&aligned_df, articles_col)
	w.add_column(&aligned_df, tone_col)
	w.add_column(&aligned_df, target_col)
	aligned_df.rows = articles_col.len

	fmt.printf("   Aligned dataset: %d samples\n", aligned_df.rows)

	if aligned_df.rows < 10 {
		fmt.println("ERROR: Not enough aligned data to train a model.")
		return
	}

	// ========================================================================
	// 4. Train Model (Random Forest for non-linear Alt-Data signals)
	// ========================================================================
	fmt.println("\n4. Training Random Forest Model...")

	split_idx := int(f64(aligned_df.rows) * 0.8)

	X_train := l.matrix_new(f64, split_idx, 2, allocator)
	y_train := make([]f64, split_idx, allocator)

	X_test := l.matrix_new(f64, aligned_df.rows - split_idx, 2, allocator)
	y_test := make([]f64, aligned_df.rows - split_idx, allocator)

	for i in 0 ..< split_idx {
		art, _ := w.column_at_float(&aligned_df.columns[0], i)
		tone, _ := w.column_at_float(&aligned_df.columns[1], i)
		ret, _ := w.column_at_float(&aligned_df.columns[2], i)

		X_train.data[i * 2 + 0] = art
		X_train.data[i * 2 + 1] = tone
		y_train[i] = ret
	}

	for i in 0 ..< aligned_df.rows - split_idx {
		idx := split_idx + i
		art, _ := w.column_at_float(&aligned_df.columns[0], idx)
		tone, _ := w.column_at_float(&aligned_df.columns[1], idx)
		ret, _ := w.column_at_float(&aligned_df.columns[2], idx)

		X_test.data[i * 2 + 0] = art
		X_test.data[i * 2 + 1] = tone
		y_test[i] = ret
	}

	rf_params := ml.RFParams {
		n_trees     = 50,
		max_depth   = 5,
		min_samples = 5,
		bootstrap   = true,
	}

	model := ml.rf_fit(&X_train, y_train, rf_params, allocator)
	defer ml.rf_free(&model)
	fmt.println("   Model trained successfully")

	// ========================================================================
	// 5. Inference & Backtest
	// ========================================================================
	fmt.println("\n5. Running Inference & Backtest...")

	predictions := ml.rf_predict(&model, &X_test, allocator)
	defer delete(predictions, allocator)

	strategy_returns := make([]f64, len(y_test), allocator)
	defer delete(strategy_returns, allocator)

	correct_direction := 0
	for i in 0 ..< len(y_test) {
		if predictions[i] > 0.0 {
			strategy_returns[i] = y_test[i]
			if y_test[i] > 0.0 {
				correct_direction += 1
			}
		} else {
			strategy_returns[i] = 0.0
		}
	}

	sharpe := fin.sharpe_ratio_from_returns(strategy_returns, 0.0, 252.0)
	accuracy := f64(correct_direction) / f64(len(y_test)) * 100.0

	fmt.printf("   Directional Accuracy: %.2f%%\n", accuracy)
	fmt.printf("   Annualized Sharpe Ratio: %.3f\n", sharpe)

	l.matrix_free(&X_train)
	delete(y_train, allocator)
	l.matrix_free(&X_test)
	delete(y_test, allocator)
	w.destroy_dataframe(&aligned_df)

	fmt.println("\n======================================================================")
	fmt.println("[*] Key Insights:")
	fmt.println("  • GDELT provides free, open-source news sentiment (Alternative Data)")
	fmt.println("  • Random Forests capture non-linear relationships in Alt-Data")
	fmt.println("  • This pipeline demonstrates the full loop: Scrape -> Train -> Infer")
	fmt.println("======================================================================\n")
}
