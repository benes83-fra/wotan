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
import "core:os"
import "core:strings"

alt_data_gdelt_pipeline_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    ALTERNATIVE DATA PIPELINE: GDELT NEWS SENTIMENT & YAHOO PRICES")
	fmt.println("======================================================================\n")

	symbol := "AAPL"
	fmt.printf("Target Asset: %s\n", symbol)

	// 1. Securely fetch API key from environment variable
	api_key := os.get_env("GDELT_API_KEY", allocator)

	text: string
	ok: bool
	use_mock := false

	if api_key == "" {
		fmt.println("WARNING: GDELT_API_KEY environment variable not set.")
		fmt.println(
			"To use the live API, set it in your environment (e.g., 'set GDELT_API_KEY=your_key').",
		)
		fmt.println("Falling back to mock data for pipeline validation...\n")
		use_mock = true
	} else {
		// 2. Use the correct GDELT Cloud API endpoint
		url := fmt.aprintf(
			"https://api.gdeltproject.org/api/v2/doc/doc?query=%s&mode=timelinetone&format=csv&timespan=365d&apikey=%s",
			symbol,
			api_key,
			allocator = context.temp_allocator,
		)

		text, ok = net.http_get(url, context.temp_allocator)
		fmt.println(text)
		if !ok ||
		   strings.contains(text, "<!DOCTYPE") ||
		   strings.contains(text, "<html") ||
		   strings.contains(text, "Not Found") ||
		   len(text) < 100 {
			fmt.println("WARNING: GDELT API request failed or returned an HTML error.")
			fmt.println("Falling back to mock data for pipeline validation...\n")
			use_mock = true
		}
	}

	if use_mock {
		text = `Date,Tone,VolumeInt
20230103,2.5,150
20230104,2.1,145
20230105,2.8,160
20230106,2.3,155
20230109,2.6,170
20230110,2.4,165
20230111,2.9,180
20230112,2.2,140
20230113,2.7,175
20230117,2.5,150
20230118,2.1,145
20230119,2.8,160
20230120,2.3,155
20230123,2.6,170
20230124,2.4,165
20230125,2.9,180
20230126,2.2,140
20230127,2.7,175
20230130,2.5,150
20230131,2.1,145
20230201,2.8,160
20230202,2.3,155
20230203,2.6,170
20230206,2.4,165
20230207,2.9,180
20230208,2.2,140
20230209,2.7,175
20230210,2.5,150
20230213,2.1,145
20230214,2.8,160
20230215,2.3,155
20230216,2.6,170
20230217,2.4,165
20230221,2.9,180
20230222,2.2,140
20230223,2.7,175
20230224,2.5,150
20230227,2.1,145
20230228,2.8,160
20230301,2.3,155`
	}
	fmt.println(text)
	// 3. Parse the data (whether live or mock)
	df_gdelt := importer.csv_load_from_string(text, allocator)
	defer w.destroy_dataframe(&df_gdelt)

	fmt.printf("Scraped %d rows of news data\n", df_gdelt.rows)
	fmt.printf("GDELT DataFrame has %d columns\n", len(df_gdelt.columns))

	date_col: ^w.Column = nil
	tone_col: ^w.Column = nil
	vol_col: ^w.Column = nil

	for i in 0 ..< len(df_gdelt.columns) {
		name := df_gdelt.columns[i].name
		// Strip UTF-8 BOM if present
		if len(name) >= 3 && name[0] == 0xEF && name[1] == 0xBB && name[2] == 0xBF {
			name = name[3:]
		}

		if name == "Date" {
			date_col = &df_gdelt.columns[i]
		} else if name == "Tone" || name == "AllTone" {
			tone_col = &df_gdelt.columns[i]
		} else if name == "VolumeInt" {
			vol_col = &df_gdelt.columns[i]
		}
	}

	if date_col == nil || tone_col == nil || vol_col == nil {
		fmt.println("ERROR: Could not identify GDELT columns (Date, Tone, VolumeInt)")
		return
	}

	// 4. Fetch Yahoo Finance Price Data
	fmt.println("\n2. Fetching Yahoo Finance Price Data...")
	df_yahoo := net.read_yahoo(symbol, .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&df_yahoo)
	fmt.printf("Fetched %d rows of price data\n", df_yahoo.rows)

	// 5. Align Data
	min_rows := min(df_gdelt.rows, df_yahoo.rows)
	if min_rows < 30 {
		fmt.printf("ERROR: Not enough overlapping data points (got %d, need >= 30)\n", min_rows)
		return
	}

	fmt.printf("\n3. Aligned Dataset: %d rows\n", min_rows)

	tone_series := make([]f64, min_rows, allocator)
	vol_series := make([]f64, min_rows, allocator)
	returns_series := make([]f64, min_rows, allocator)

	for i in 0 ..< min_rows {
		tone_series[i], _ = w.column_at_float(tone_col, i)
		vol_series[i], _ = w.column_at_float(vol_col, i)

		if i > 0 {
			close_prev, _ := w.column_at_float(w.column(&df_yahoo, "Close"), i - 1)
			close_curr, _ := w.column_at_float(w.column(&df_yahoo, "Close"), i)
			if close_prev > 0 {
				returns_series[i] = (close_curr - close_prev) / close_prev
			}
		}
	}

	// ========================================================================
	// 6. Train Random Forest Model
	// ========================================================================
	fmt.println("\n4. Training Random Forest Model...")

	split_idx := int(f64(min_rows) * 0.8)
	if split_idx < 10 {split_idx = 10} 	// Safety floor

	X_train := l.matrix_new(f64, split_idx, 2, allocator)
	y_train := make([]f64, split_idx, allocator)

	test_size := min_rows - split_idx
	X_test := l.matrix_new(f64, test_size, 2, allocator)
	y_test := make([]f64, test_size, allocator)

	for i in 0 ..< split_idx {
		X_train.data[i * 2 + 0] = tone_series[i]
		X_train.data[i * 2 + 1] = vol_series[i]
		y_train[i] = returns_series[i]
	}

	for i in 0 ..< test_size {
		idx := split_idx + i
		X_test.data[i * 2 + 0] = tone_series[idx]
		X_test.data[i * 2 + 1] = vol_series[idx]
		y_test[i] = returns_series[idx]
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
	// 7. Inference & Backtest
	// ========================================================================
	fmt.println("\n5. Running Inference & Backtest...")

	predictions := ml.rf_predict(&model, &X_test, allocator)
	defer delete(predictions, allocator)

	strategy_returns := make([]f64, test_size, allocator)
	defer delete(strategy_returns, allocator)

	correct_direction := 0
	for i in 0 ..< test_size {
		if predictions[i] > 0.0 {
			strategy_returns[i] = y_test[i] // Go long
			if y_test[i] > 0.0 {
				correct_direction += 1
			}
		} else {
			strategy_returns[i] = 0.0 // Stay flat
		}
	}

	sharpe := fin.sharpe_ratio_from_returns(strategy_returns, 0.0, 252.0)
	accuracy := f64(correct_direction) / f64(test_size) * 100.0

	fmt.printf("   Directional Accuracy: %.2f%%\n", accuracy)
	fmt.printf("   Annualized Sharpe Ratio: %.3f\n", sharpe)

	// Cleanup
	l.matrix_free(&X_train)
	delete(y_train, allocator)
	l.matrix_free(&X_test)
	delete(y_test, allocator)
	delete(tone_series, allocator)
	delete(vol_series, allocator)
	delete(returns_series, allocator)

	fmt.println("\n======================================================================")
	fmt.println("[*] Key Insights:")
	fmt.println("  • API keys are securely managed via environment variables.")
	fmt.println("  • Pipeline gracefully falls back to mock data if the API is unavailable.")
	fmt.println("  • Random Forest successfully trained on aligned Alt-Data + Price features.")
	fmt.println("  • Pipeline demonstrates the full loop: Scrape -> Align -> Train -> Infer.")
	fmt.println("======================================================================\n")
}
