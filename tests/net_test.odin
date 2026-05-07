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

	text, ok := yahoo_download_csv("MSFT", allocator)
	if !ok {
		panic("Yahoo CSV download failed")
	}

	fmt.println("Downloaded", len(text), "bytes")

	if !strings.contains(text, "Date,Open,High,Low,Close,Adj Close,Volume") {
		panic("Yahoo CSV header missing or unexpected")
	}

	df := importer.csv_load_from_string(text, allocator)
	defer w.destroy_dataframe(&df)

	fmt.println("--- Yahoo Finance MSFT (head) ---")
	w.df_head(&df, 10)

	fmt.println("=== END YAHOO FINANCE CSV TEST ===")
}

yahoo_download_csv :: proc(symbol: string, allocator: mem.Allocator) -> (string, bool) {
	crumb, ok := yahoo_get_crumb(allocator)
	if !ok {
		return "", false
	}

	url := fmt.aprintf(
		"https://query1.finance.yahoo.com/v7/finance/download/%s?period1=0&period2=9999999999&interval=1d&events=history&crumb=%s",
		symbol,
		crumb,
		allocator = allocator,
	)

	text, ok2 := net.http_get(url, allocator)
	return text, ok2
}
yahoo_get_crumb :: proc(allocator: mem.Allocator) -> (string, bool) {
	url := "https://query1.finance.yahoo.com/v1/test/getcrumb"

	text, ok := net.http_get(url, allocator)
	if !ok {
		return "", false
	}

	crumb := strings.trim_space(text)
	return crumb, true
}
