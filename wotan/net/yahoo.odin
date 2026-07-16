package net

import w "../core"
import "../importer"

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"
import curl "vendor:curl"


yahoo_fields :: enum {
	Price, // OHLCV + AdjClose
	Dividends,
	Splits,
}

interval :: enum {
	Daily, // 1d
	Weekly, // 1wk
	Monthly, // 1mo
}

range :: enum {
	OneMonth, // 1mo
	ThreeMonths, // 3mo
	SixMonths, // 6mo
	OneYear, // 1y
	TwoYears, // 2y
	FiveYears, // 5y
	TenYears, // 10y
	YTD, // ytd
	Max, // max
}
interval_to_string :: proc(i: interval) -> string {
	#partial switch i {
	case .Daily:
		return "1d"
	case .Weekly:
		return "1wk"
	case .Monthly:
		return "1mo"
	}
	return ""
}

range_to_string :: proc(r: range) -> string {
	#partial switch r {
	case .OneMonth:
		return "1mo"
	case .ThreeMonths:
		return "3mo"
	case .SixMonths:
		return "6mo"
	case .OneYear:
		return "1y"
	case .TwoYears:
		return "2y"
	case .FiveYears:
		return "5y"
	case .TenYears:
		return "10y"
	case .YTD:
		return "ytd"
	case .Max:
		return "max"
	}
	return ""
}

// ============================================================================
// Yahoo Crumb & Cookie Fetcher (Using your existing vendor:curl patterns)
// ============================================================================

// Global cache for crumb and cookie
global_crumb: string = ""
global_cookie: string = ""

// Header buffer to capture Set-Cookie (matches your write_cb style exactly)
HeaderBuffer :: struct {
	data: [dynamic]u8,
}

header_cb :: proc "c" (ptr: [^]u8, size: uint, nmemb: uint, userdata: rawptr) -> uint {
	context = runtime.default_context()
	total := int(size * nmemb)
	buf := cast(^HeaderBuffer)userdata
	append_elems(&buf.data, ..ptr[:total])
	return c.size_t(total)
}

yahoo_get_crumb_and_cookie :: proc(allocator: mem.Allocator) -> (string, string, bool) {
	// Return cached values if already fetched
	if global_crumb != "" && global_cookie != "" {
		return global_crumb, global_cookie, true
	}

	// Step 1: Get initial cookie from Yahoo homepage using a HEAD request (headers only)
	url1 := "https://fc.yahoo.com"
	c_url1 := strings.clone_to_cstring(url1, context.temp_allocator)

	curl_handle := curl.easy_init()
	if curl_handle == nil {
		return "", "", false
	}
	defer curl.easy_cleanup(curl_handle)

	curl.easy_setopt(curl_handle, curl.option.URL, c_url1)
	curl.easy_setopt(curl_handle, curl.option.FOLLOWLOCATION, i64(1))
	curl.easy_setopt(
		curl_handle,
		curl.option.USERAGENT,
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
	)

	// HEAD request: we only need headers to extract the Set-Cookie
	curl.easy_setopt(curl_handle, curl.option.NOBODY, i64(1))

	header_buf: HeaderBuffer
	header_buf.data = make([dynamic]u8, allocator)
	defer delete(header_buf.data)

	curl.easy_setopt(curl_handle, curl.option.HEADERFUNCTION, header_cb)
	curl.easy_setopt(curl_handle, curl.option.HEADERDATA, &header_buf)

	res := curl.easy_perform(curl_handle)
	if res != .E_OK {
		return "", "", false
	}

	header_str := string(header_buf.data[:])

	// Find "Set-Cookie:" (case-insensitive check)
	cookie_pos := strings.index(header_str, "Set-Cookie:")
	if cookie_pos < 0 {
		cookie_pos = strings.index(header_str, "set-cookie:")
	}

	if cookie_pos >= 0 {
		start := cookie_pos + len("Set-Cookie:")
		// Find the end of the header line (\r\n or \n)
		end := strings.index(header_str[start:], "\r\n")
		if end < 0 {
			end = strings.index(header_str[start:], "\n")
		}
		if end < 0 {
			end = len(header_str) - start
		}

		raw_cookie := header_str[start:start + end]
		// Take only the first part before ';' (e.g., "B=abc123; expires=...")
		semi_pos := strings.index(raw_cookie, ";")
		if semi_pos >= 0 {
			global_cookie = strings.trim_space(raw_cookie[:semi_pos])
		} else {
			global_cookie = strings.trim_space(raw_cookie)
		}
	}

	if global_cookie == "" {
		return "", "", false
	}

	// Step 2: Get crumb using the cookie and YOUR EXISTING http_get_with_cookie
	url2 := "https://query1.finance.yahoo.com/v1/test/getcrumb"
	crumb_text, ok := http_get_with_cookie(url2, global_cookie, allocator)
	if !ok || len(crumb_text) == 0 {
		return "", "", false
	}

	global_crumb = strings.trim_space(crumb_text)

	return global_crumb, global_cookie, true
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

	timestamps := importer.extract_json_array_i64(json_text, "\"timestamp\"")
	opens := importer.extract_json_array_f64(json_text, "\"open\"")
	highs := importer.extract_json_array_f64(json_text, "\"high\"")
	lows := importer.extract_json_array_f64(json_text, "\"low\"")
	closes := importer.extract_json_array_f64(json_text, "\"close\"")
	volumes := importer.extract_json_array_i64(json_text, "\"volume\"")
	adjclose := importer.extract_json_array_f64(json_text, "\"adjclose\"")

	divs := extract_dividends(json_text)
	spls := extract_splits(json_text)

	count := len(timestamps)
	if count == 0 {
		return df
	}

	// Create columns
	date_col := w.column_new("Date", .Date, count)
	open_col := w.column_new("Open", .Float, count)
	high_col := w.column_new("High", .Float, count)
	low_col := w.column_new("Low", .Float, count)
	close_col := w.column_new("Close", .Float, count)
	adj_col := w.column_new("AdjClose", .Float, count)
	vol_col := w.column_new("Volume", .Int, count)
	div_col := w.column_new("Dividend", .Float, count)
	split_col := w.column_new("Split", .Float, count)

	w.add_column(&df, date_col)
	w.add_column(&df, open_col)
	w.add_column(&df, high_col)
	w.add_column(&df, low_col)
	w.add_column(&df, close_col)
	w.add_column(&df, adj_col)
	w.add_column(&df, vol_col)
	w.add_column(&df, div_col)
	w.add_column(&df, split_col)

	// Fill rows
	for i in 0 ..< count {
		ts := timestamps[i]

		w.append_date(&df.columns[0], w.date_from_unix(ts))
		w.append_float(&df.columns[1], opens[i])
		w.append_float(&df.columns[2], highs[i])
		w.append_float(&df.columns[3], lows[i])
		w.append_float(&df.columns[4], closes[i])
		w.append_float(&df.columns[5], adjclose[i])
		w.append_int(&df.columns[6], int(volumes[i]))

		// Dividends
		if amt, ok := divs[ts]; ok {
			w.append_float(&df.columns[7], amt)
		} else {
			w.append_float(&df.columns[7], 0.0)
		}

		// Splits
		if ratio, ok := spls[ts]; ok {
			w.append_float(&df.columns[8], ratio)
		} else {
			w.append_float(&df.columns[8], 0.0)
		}
	}

	df.rows = count
	return df
}


build_yahoo_url :: proc(
	symbol: string,
	i: interval,
	r: range,
	allocator: mem.Allocator,
) -> string {
	return fmt.aprintf(
		"https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=%s&range=%s&events=div,splits",
		symbol,
		interval_to_string(i),
		range_to_string(r),
		allocator = allocator,
	)

}


read_yahoo :: proc(
	symbol: string,
	i: interval = .Daily,
	r: range = .Max,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {

	url := build_yahoo_url(symbol, i, r, context.temp_allocator)

	text, ok := http_get(url, context.temp_allocator)
	if !ok {
		panic(fmt.tprintf("Failed to GET Yahoo JSON for %s", symbol))
	}

	return yahoo_json_to_dataframe(text, allocator)
}


extract_dividends :: proc(
	json: string,
	allocator: mem.Allocator = context.temp_allocator,
) -> map[i64]f64 {
	out := make(map[i64]f64, allocator)

	key := "\"dividends\""
	pos := strings.index(json, key)
	if pos < 0 {return out}

	// Find the object after "dividends":
	start := strings.index(json[pos:], "{")
	if start < 0 {return out}
	start += pos

	end := strings.index(json[start:], "}")
	if end < 0 {return out}
	end += start

	block := json[start:end]

	// Each entry looks like: "1609459200":{"amount":0.56,...
	entries := strings.split(block, "}", context.temp_allocator)

	for e in entries {
		ts_pos := strings.index(e, "\"")
		if ts_pos < 0 {continue}

		ts_end := strings.index(e[ts_pos + 1:], "\"")
		if ts_end < 0 {continue}
		ts_str := e[ts_pos + 1:ts_pos + 1 + ts_end]

		ts, ok := strconv.parse_i64(ts_str)
		if !ok {continue}

		amt_pos := strings.index(e, "\"amount\"")
		if amt_pos < 0 {continue}

		colon := strings.index(e[amt_pos:], ":")
		if colon < 0 {continue}
		colon += amt_pos

		comma := strings.index(e[colon:], ",")
		if comma < 0 {comma = len(e) - colon}

		amt_str := strings.trim_space(e[colon + 1:colon + comma])
		amt, ok2 := strconv.parse_f64(amt_str)
		if !ok2 {continue}

		out[ts] = amt
	}

	return out
}


extract_splits :: proc(
	json: string,
	allocator: mem.Allocator = context.temp_allocator,
) -> map[i64]f64 {
	out := make(map[i64]f64, allocator)

	key := "\"splits\""
	pos := strings.index(json, key)
	if pos < 0 {return out}

	start := strings.index(json[pos:], "{")
	if start < 0 {return out}
	start += pos

	end := strings.index(json[start:], "}")
	if end < 0 {return out}
	end += start

	block := json[start:end]

	entries := strings.split(block, "}", context.temp_allocator)

	for e in entries {
		ts_pos := strings.index(e, "\"")
		if ts_pos < 0 {continue}

		ts_end := strings.index(e[ts_pos + 1:], "\"")
		if ts_end < 0 {continue}
		ts_str := e[ts_pos + 1:ts_pos + 1 + ts_end]

		ts, ok := strconv.parse_i64(ts_str)
		if !ok {continue}

		num_pos := strings.index(e, "\"numerator\"")
		den_pos := strings.index(e, "\"denominator\"")
		if num_pos < 0 || den_pos < 0 {continue}

		num := parse_json_number_after(e, num_pos)
		den := parse_json_number_after(e, den_pos)

		if den != 0 {
			out[ts] = num / den
		}
	}

	return out
}


parse_json_number_after :: proc(s: string, pos: int) -> f64 {
	// Find ':' after the key
	colon := strings.index(s[pos:], ":")
	if colon < 0 {
		return 0.0
	}
	colon += pos

	// Find end of number (comma or end of string)
	end := strings.index(s[colon + 1:], ",")
	if end < 0 {
		end = len(s) - (colon + 1)
	}

	num_str := strings.trim_space(s[colon + 1:colon + 1 + end])
	v, ok := strconv.parse_f64(num_str)
	if !ok {
		return 0.0
	}
	return v
}
