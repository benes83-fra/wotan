package net

import w "../core"
import "../importer"

import "core:fmt"
import "core:mem"
import "core:strings"
import curl "vendor:curl"

// ------------------------------
// 1) Fetch crumb + cookie
// ------------------------------
yahoo_get_crumb_and_cookie :: proc(allocator: mem.Allocator) -> (string, string, bool) {
	// (full implementation goes here — you already have the pieces)
	// returns: crumb, cookie, ok
	return "", "", false
}

// ------------------------------
// 2) Download CSV using crumb + cookie
// ------------------------------
yahoo_download_csv :: proc(symbol: string, allocator: mem.Allocator) -> (string, bool) {
	crumb, cookie, ok := yahoo_get_crumb_and_cookie(allocator)
	if !ok {
		return "", false
	}

	url := fmt.aprintf(
		"https://query1.finance.yahoo.com/v7/finance/download/%s?period1=0&period2=9999999999&interval=1d&events=history&crumb=%s",
		symbol,
		crumb,
		allocator = allocator,
	)
	defer delete(url)

	// Use your existing http_get, but with cookie support
	// (you’ll add http_get_with_cookie next)
	text, ok2 := http_get_with_cookie(url, cookie, allocator)
	return text, ok2
}

// ------------------------------
// 3) Convert CSV → DataFrame
// ------------------------------
read_yahoo :: proc(symbol: string, allocator: mem.Allocator = context.allocator) -> w.DataFrame {
	text, ok := yahoo_download_csv(symbol, context.temp_allocator)
	if !ok {
		panic(fmt.tprintf("Yahoo Finance download failed for %s", symbol))
	}

	return importer.csv_load_from_string(text, allocator)
}
read_yahoo_json :: proc(
	symbol: string,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	url := fmt.aprintf(
		"https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d&range=max",
		symbol,
		allocator = context.temp_allocator,
	)

	text, ok := http_get(url, context.temp_allocator)
	if !ok {
		panic(fmt.tprintf("Failed to GET Yahoo JSON for %s", symbol))
	}

	// TODO: parse JSON → DataFrame
	return yahoo_json_to_dataframe(text, allocator)
}
yahoo_json_to_dataframe :: proc(
	json_text: string,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {

	df := w.dataframe_new()

	// ------------------------------
	// Extract arrays from JSON
	// ------------------------------
	timestamps := importer.extract_json_array_i64(json_text, "\"timestamp\"")
	opens := importer.extract_json_array_f64(json_text, "\"open\"")
	highs := importer.extract_json_array_f64(json_text, "\"high\"")
	lows := importer.extract_json_array_f64(json_text, "\"low\"")
	closes := importer.extract_json_array_f64(json_text, "\"close\"")
	volumes := importer.extract_json_array_i64(json_text, "\"volume\"")
	adjclose := importer.extract_json_array_f64(json_text, "\"adjclose\"")

	count := len(timestamps)
	if count == 0 {
		return df
	}

	// ------------------------------
	// Create columns
	// ------------------------------
	date_col := w.column_new("Date", .Date, count)
	open_col := w.column_new("Open", .Float, count)
	high_col := w.column_new("High", .Float, count)
	low_col := w.column_new("Low", .Float, count)
	close_col := w.column_new("Close", .Float, count)
	adjclose_col := w.column_new("AdjClose", .Float, count)
	volume_col := w.column_new("Volume", .Int, count)

	w.add_column(&df, date_col)
	w.add_column(&df, open_col)
	w.add_column(&df, high_col)
	w.add_column(&df, low_col)
	w.add_column(&df, close_col)
	w.add_column(&df, adjclose_col)
	w.add_column(&df, volume_col)

	// ------------------------------
	// Fill rows
	// ------------------------------
	for i in 0 ..< count {
		// Convert UNIX timestamp → Date
		d := w.date_from_unix(timestamps[i])
		w.append_date(&df.columns[0], d)

		w.append_float(&df.columns[1], opens[i])
		w.append_float(&df.columns[2], highs[i])
		w.append_float(&df.columns[3], lows[i])
		w.append_float(&df.columns[4], closes[i])
		w.append_float(&df.columns[5], adjclose[i])
		w.append_int(&df.columns[6], int(volumes[i]))
	}

	df.rows = count
	return df
}
