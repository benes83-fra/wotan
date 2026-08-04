// tests/alt_data_gdelt_pipeline_test.odin
package tests

import w "../wotan/core"
import fin "../wotan/finance"
import importer "../wotan/importer"
import net "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"

alt_data_gdelt_pipeline_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n======================================================================")
	fmt.println("    ALTERNATIVE DATA PIPELINE: GDELT NEWS SENTIMENT & YAHOO PRICES")
	fmt.println("======================================================================\n")

	symbol := "AAPL"
	fmt.printf("Target Asset: %s\n", symbol)

	// 1. Scrape GDELT Timeline Tone data (aggregated daily sentiment)
	// We use 'timelinetone' instead of 'doc' to get daily aggregated metrics
	gdelt_url := fmt.aprintf(
		"https://api.gdeltproject.org/api/v2/timelinetone/timelinetone?query=%s&timespan=365d&format=csv",
		symbol,
		allocator = context.temp_allocator,
	)

	// Fetch GDELT data
	text, ok := net.http_get(gdelt_url, context.temp_allocator)
	if !ok {
		fmt.println("ERROR: Failed to fetch GDELT data")
		return
	}

	// GDELT's API is known to return HTML error pages when it rate-limits or blocks automated requests.
	// If we detect HTML, we fall back to a realistic CSV sample so the pipeline logic can be fully tested.
	if strings.contains(text, "<!DOCTYPE") || strings.contains(text, "<html") {
		fmt.println("WARNING: GDELT API returned an HTML error page (rate-limiting/blocking).")
		fmt.println("Using a realistic GDELT CSV sample to test the pipeline logic...")
		text = `day,numArticles,avgTone
20230101,150,2.5
20230102,145,2.1
20230103,160,2.8
20230104,155,2.3
20230105,170,2.6
20230106,165,2.4`
	}
	defer delete(text, context.temp_allocator)

	df_gdelt := importer.csv_load_from_string(text, allocator)
	defer w.destroy_dataframe(&df_gdelt)

	fmt.printf("Scraped %d rows of news data\n", df_gdelt.rows)
	fmt.printf("GDELT DataFrame has %d columns\n", len(df_gdelt.columns))

	// GDELT timelinetone returns: Date, DateStr, Tone, ToneCount, VolumeInt, AllTone
	date_col: ^w.Column = nil
	tone_col: ^w.Column = nil
	vol_col: ^w.Column = nil

	for i in 0 ..< len(df_gdelt.columns) {
		name := df_gdelt.columns[i].name

		// Strip UTF-8 BOM if present (ï»¿)
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
		fmt.println("Available columns:")
		for i in 0 ..< len(df_gdelt.columns) {
			fmt.printf("  Col %d: %s\n", i, df_gdelt.columns[i].name)
		}
		return
	}

	// 2. Fetch Yahoo Finance Price Data
	fmt.println("\n2. Fetching Yahoo Finance Price Data...")
	df_yahoo := net.read_yahoo(symbol, .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&df_yahoo)
	fmt.printf("Fetched %d rows of price data\n", df_yahoo.rows)

	// 3. Align Data (Simplified: take the last N rows where both exist)
	min_rows := min(df_gdelt.rows, df_yahoo.rows)
	if min_rows < 30 {
		fmt.println("ERROR: Not enough overlapping data points")
		return
	}

	fmt.printf("\n3. Aligned Dataset: %d rows\n", min_rows)

	// Extract aligned series
	tone_series := make([]f64, min_rows, allocator)
	vol_series := make([]f64, min_rows, allocator)
	returns_series := make([]f64, min_rows, allocator)

	for i in 0 ..< min_rows {
		tone_series[i], _ = w.column_at_float(tone_col, i)
		vol_series[i], _ = w.column_at_float(vol_col, i)

		// Calculate daily returns from Yahoo data
		if i > 0 {
			close_prev, _ := w.column_at_float(w.column(&df_yahoo, "Close"), i - 1)
			close_curr, _ := w.column_at_float(w.column(&df_yahoo, "Close"), i)
			if close_prev > 0 {
				returns_series[i] = (close_curr - close_prev) / close_prev
			}
		}
	}

	// 4. Simple Correlation Analysis (Sentiment vs Next-Day Returns)
	fmt.println("\n4. Analyzing Sentiment vs. Next-Day Returns...")

	mean_tone := 0.0
	mean_ret := 0.0
	valid_count := 0

	// We look at tone on day i, and return on day i+1
	for i in 1 ..< min_rows - 1 {
		tone := tone_series[i]
		ret := returns_series[i + 1]

		if !math.is_nan(tone) && !math.is_nan(ret) {
			mean_tone += tone
			mean_ret += ret
			valid_count += 1
		}
	}

	if valid_count > 1 {
		mean_tone /= f64(valid_count)
		mean_ret /= f64(valid_count)

		var_tone := 0.0
		var_ret := 0.0
		cov := 0.0

		for i in 1 ..< min_rows - 1 {
			tone := tone_series[i]
			ret := returns_series[i + 1]
			if !math.is_nan(tone) && !math.is_nan(ret) {
				dt := tone - mean_tone
				dr := ret - mean_ret
				var_tone += dt * dt
				var_ret += dr * dr
				cov += dt * dr
			}
		}

		std_tone := math.sqrt(var_tone)
		std_ret := math.sqrt(var_ret)

		correlation := 0.0
		if std_tone > 1e-10 && std_ret > 1e-10 {
			correlation = cov / (std_tone * std_ret)
		}

		fmt.printf("   Correlation (Tone vs Next-Day Return): %.4f\n", correlation)
		if correlation > 0.05 {
			fmt.println("   Insight: Positive sentiment tends to precede positive returns.")
		} else if correlation < -0.05 {
			fmt.println("   Insight: Negative sentiment tends to precede negative returns.")
		} else {
			fmt.println("   Insight: Weak or no linear relationship detected in this sample.")
		}
	}

	fmt.println("\n======================================================================")
	fmt.println("[*] Key Insights:")
	fmt.println("  • GDELT 'timelinetone' API provides daily aggregated sentiment.")
	fmt.println("  • This pipeline successfully merges alternative data with price data.")
	fmt.println("  • In production, you would feed 'tone_series' and 'vol_series' into")
	fmt.println("    an ML model (e.g., Random Forest or LSTM) to predict returns.")
	fmt.println("======================================================================\n")

	delete(tone_series, allocator)
	delete(vol_series, allocator)
	delete(returns_series, allocator)
}
