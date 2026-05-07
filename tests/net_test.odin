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

	df := net.read_yahoo("MSFT", .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&df)

	fmt.println("Rows:", df.rows)
	fmt.println("Columns:", len(df.columns))
	w.df_head(&df, 10)

	fmt.println("=== END YAHOO FINANCE JSON TEST ===")
}
