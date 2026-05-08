package tests

import w "../wotan/core" // your http_get wrapper
import importer "../wotan/importer"
import net "../wotan/net"
import "core:fmt"
import "core:mem"
import "core:strings"

http_get_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== HTTP GET TEST ===")

	url := "https://example.com"

	text, ok := net.http_get(url, allocator)
	if !ok {
		panic("http_get_test: request failed")
	}

	fmt.println("Received", len(text), "bytes")

	// Basic sanity check
	if !strings.contains(text, "Example Domain") {
		panic("http_get_test: response does not contain expected marker")
	}

	fmt.println("http_get_test: OK -  response contains 'Example Domain'")

	fmt.println("=== END HTTP GET TEST ===")
}
yahoo_finance_csv_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== YAHOO FINANCE CSV TEST ===")

	// TEMP: bypass crumb/cookie and just see what the endpoint does
	url := "https://query1.finance.yahoo.com/v7/finance/download/MSFT?period1=0&period2=9999999999&interval=1d&events=history"
	text, ok := net.http_get(url, allocator)
	if !ok {
		panic("Yahoo CSV download failed (raw)")
	}

	fmt.println("Downloaded", len(text), "bytes")
	fmt.println("RAW RESPONSE:\n", text)

	if !strings.contains(text, "Date,Open,High,Low,Close,Adj Close,Volume") {
		panic("Yahoo CSV header missing or unexpected")
	}

	df := importer.csv_load_from_string(text, allocator)
	defer w.destroy_dataframe(&df)

	fmt.println("--- Yahoo Finance MSFT (head) ---")
	w.df_head(&df, 10)

	fmt.println("=== END YAHOO FINANCE CSV TEST ===")
}

yahoo_finance_json_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== YAHOO FINANCE JSON TEST ===")

	df := net.read_yahoo("GOOG", .Daily, .OneMonth, allocator)
	defer w.destroy_dataframe(&df)

	fmt.println("Rows:", df.rows)
	fmt.println("Columns:", len(df.columns))
	w.df_head(&df, 20)

	fmt.println("=== END YAHOO FINANCE JSON TEST ===")
}

yahoo_finance_events_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== YAHOO FINANCE DIVIDENDS & SPLITS TEST ===")

	df := net.read_yahoo("AAPL", .Daily, .TenYears, allocator)
	defer w.destroy_dataframe(&df)

	fmt.println("Rows:", df.rows)
	fmt.println("Columns:", len(df.columns))
	w.df_head(&df, 50)

	// ------------------------------------------------------------
	// Locate Dividend and Split columns by name
	// ------------------------------------------------------------
	div_col_idx := -1
	split_col_idx := -1

	for i in 0 ..< len(df.columns) {
		if df.columns[i].name == "Dividend" {
			div_col_idx = i
		}
		if df.columns[i].name == "Split" {
			split_col_idx = i
		}
	}

	if div_col_idx < 0 {
		panic("Dividend column missing")
	}
	if split_col_idx < 0 {
		panic("Split column missing")
	}

	div_col := &df.columns[div_col_idx]
	split_col := &df.columns[split_col_idx]

	// ------------------------------------------------------------
	// Scan for actual dividend and split events
	// ------------------------------------------------------------
	div_count := 0
	split_count := 0
	div_data := cast([^]f64)div_col.data
	spl_data := cast([^]f64)split_col.data
	for i in 0 ..< df.rows {
		div := div_data[i]
		spl := spl_data[i]

		if div != 0.0 {
			fmt.println("Dividend event at row", i, "=", div)
			div_count += 1
		}
		if spl != 0.0 {
			fmt.println("Split event at row", i, "=", spl)
			split_count += 1
		}
	}

	fmt.println("Total dividends found:", div_count)
	fmt.println("Total splits found:", split_count)

	if div_count == 0 {
		panic("No dividends detected - expected at least one for AAPL")
	}

	if split_count == 0 {
		fmt.println("Warning: No splits detected in last 5 years (this may be correct)")
	}

	fmt.println("=== END YAHOO FINANCE DIVIDENDS & SPLITS TEST ===")
}
