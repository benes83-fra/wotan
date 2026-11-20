package net

import w "../core"
import ex "../exporter"
import "../importer"

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:strconv"
import "core:strings"
import "core:time"
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

// Global cache for crumb and cookie. Both live for the whole process on the
// heap, independent of the caller's allocator (which may be an arena that is
// reset between uses).
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
		if semi_pos >= 0 {raw_cookie = raw_cookie[:semi_pos]}
		// Copy out of header_buf, which is freed on return
		global_cookie = strings.clone(strings.trim_space(raw_cookie), runtime.heap_allocator())
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
	defer delete(crumb_text, allocator)

	global_crumb = strings.clone(strings.trim_space(crumb_text), runtime.heap_allocator())

	return global_crumb, global_cookie, true
}
// ------------------------------
// 2) History as CSV text
// ------------------------------

// History of `symbol` as CSV text: the columns of read_yahoo (Date, Open, High,
// Low, Close, AdjClose, Volume, Dividend, Split) in exporter.csv_save format,
// so importer.csv_load_from_string reads it back unchanged. Yahoo's own CSV
// download needs a paid login, so the text is built from the chart API. ok is
// false (and the text empty) when read_yahoo fails.
yahoo_download_csv :: proc(
	symbol: string,
	allocator: mem.Allocator = context.allocator,
	i: interval = .Daily,
	r: range = .Max,
) -> (
	string,
	bool,
) {
	df, ok := read_yahoo(symbol, i, r, context.temp_allocator)
	defer w.destroy_dataframe(&df)
	if !ok {return "", false}
	return ex.csv_string(&df, allocator = allocator), true
}

// ------------------------------
// 3) Convert CSV → DataFrame
// ------------------------------

// Daily history of `symbol` over its whole range. ok is false when Yahoo cannot
// be reached or returns no rows (a rate-limited request answers with an error
// text instead of the chart); the DataFrame is then empty
read_yahoo_json :: proc(
	symbol: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	df: w.DataFrame,
	ok: bool,
) {
	url := fmt.aprintf(
		"https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d&range=max",
		symbol,
		allocator = context.temp_allocator,
	)
	text, got := http_get(url, context.temp_allocator)
	if !got {return w.dataframe_new(), false}
	df = yahoo_json_to_dataframe(text, allocator)
	return df, df.rows > 0
}
yahoo_json_to_dataframe :: proc(
	json_text: string,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {

	df := w.dataframe_new(allocator)

	timestamps := importer.extract_json_array_i64(json_text, "\"timestamp\"")
	opens := importer.extract_json_array_f64(json_text, "\"open\"")
	highs := importer.extract_json_array_f64(json_text, "\"high\"")
	lows := importer.extract_json_array_f64(json_text, "\"low\"")
	closes := importer.extract_json_array_f64(json_text, "\"close\"")
	volumes := importer.extract_json_array_f64(json_text, "\"volume\"") // f64 so nulls show as NaN
	adjclose := importer.extract_json_array_f64(json_text, "\"adjclose\"")

	divs := extract_dividends(json_text)
	spls := extract_splits(json_text)

	count := len(timestamps)
	if count == 0 {
		return df
	}

	// Create columns
	date_col := w.column_new("Date", .Date, count, allocator)
	open_col := w.column_new("Open", .Float, count, allocator)
	high_col := w.column_new("High", .Float, count, allocator)
	low_col := w.column_new("Low", .Float, count, allocator)
	close_col := w.column_new("Close", .Float, count, allocator)
	adj_col := w.column_new("AdjClose", .Float, count, allocator)
	vol_col := w.column_new("Volume", .Int, count, allocator)
	div_col := w.column_new("Dividend", .Float, count, allocator)
	split_col := w.column_new("Split", .Float, count, allocator)

	w.add_column(&df, date_col)
	w.add_column(&df, open_col)
	w.add_column(&df, high_col)
	w.add_column(&df, low_col)
	w.add_column(&df, close_col)
	w.add_column(&df, adj_col)
	w.add_column(&df, vol_col)
	w.add_column(&df, div_col)
	w.add_column(&df, split_col)

	// Fill rows. Yahoo sends null for every field on days without a quote (^VIX on
	// US market holidays, for example); those cells become nulls, not 0. A missing
	// or short array (no adjclose for intraday intervals) gives nulls as well.
	put_float :: proc(c: ^w.Column, vals: []f64, i: int) {
		if i < len(vals) && vals[i] == vals[i] {
			w.append_float(c, vals[i])
		} else {
			w.append_null(c)
		}
	}
	for i in 0 ..< count {
		ts := timestamps[i]

		w.append_date(&df.columns[0], w.date_from_unix(ts))
		put_float(&df.columns[1], opens, i)
		put_float(&df.columns[2], highs, i)
		put_float(&df.columns[3], lows, i)
		put_float(&df.columns[4], closes, i)
		put_float(&df.columns[5], adjclose, i)
		if i < len(volumes) && volumes[i] == volumes[i] {
			w.append_int(&df.columns[6], int(volumes[i]))
		} else {
			w.append_null(&df.columns[6])
		}

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


// History of `symbol` at the interval over the range. ok is false when Yahoo
// cannot be reached or returns no rows (a rate-limited request answers with an
// error text instead of the chart); the DataFrame is then empty
read_yahoo :: proc(
	symbol: string,
	i: interval = .Daily,
	r: range = .Max,
	allocator: mem.Allocator = context.allocator,
) -> (
	df: w.DataFrame,
	ok: bool,
) {
	text, got := _yahoo_chart_text(symbol, i, r)
	if !got {return w.dataframe_new(), false}
	df = yahoo_json_to_dataframe(text, allocator)
	return df, df.rows > 0
}

// The chart JSON of `symbol` (temp allocator)
@(private = "file")
_yahoo_chart_text :: proc(symbol: string, i: interval, r: range) -> (string, bool) {
	url := build_yahoo_url(symbol, i, r, context.temp_allocator)
	return http_get(url, context.temp_allocator)
}

// yahoo_session_live reports whether Yahoo sent the chart JSON while the exchange's
// regular session was open and had traded (`now_unix`: the current time,
// time.time_to_unix(time.now())). The last row then holds the running session: the
// current price and the volume so far, not a close. Before the open and after the
// close it is false. read_yahoo_series and read_yahoo_panel report it as last_is_live.
yahoo_session_live :: proc(json_text: string, now_unix: i64) -> bool {
	// meta: "regularMarketTime":T, ..., "currentTradingPeriod":{..., "regular":{..., "end":E, "start":S, ...}}
	int_after :: proc(s, key: string) -> (v: i64, ok: bool) {
		p := strings.index(s, key)
		if p < 0 {return}
		rest := strings.trim_left_space(s[p + len(key):])
		if len(rest) == 0 || rest[0] != ':' {return}
		rest = strings.trim_left_space(rest[1:])
		n := 0
		for n < len(rest) &&
		    (rest[n] >= '0' && rest[n] <= '9' || n == 0 && rest[n] == '-') {n += 1}
		return strconv.parse_i64(rest[:n])
	}
	period := strings.index(json_text, "\"currentTradingPeriod\"")
	if period < 0 {return false}
	reg := strings.index(json_text[period:], "\"regular\"")
	if reg < 0 {return false}
	open := strings.index(json_text[period + reg:], "{")
	if open < 0 {return false}
	open += period + reg
	close := _matching_brace(json_text, open)
	if close < 0 {return false}
	block := json_text[open:close]
	start, s_ok := int_after(block, "\"start\"")
	end, e_ok := int_after(block, "\"end\"")
	last_trade, t_ok := int_after(json_text, "\"regularMarketTime\"")
	if !s_ok || !e_ok || !t_ok {return false}
	return last_trade >= start && now_unix < end
}

@(private = "file")
_now_unix :: proc() -> i64 {
	return time.time_to_unix(time.now())
}

// One ticker's history as plain arrays, oldest first, ready for numeric code.
// Days on which Yahoo has no close or adjusted close (or a price <= 0) are left
// out, so all arrays have the same length and close[i] > 0. Open, high, low and
// volume are NaN on the rare kept day where only they are missing. adj_close is
// adjusted for dividends and splits; close is adjusted for splits only.
// last_is_live is true when the series was loaded during the exchange's regular
// session: the last row is then today's running session (current price, volume so
// far), not a close. Leave it out (the first len - 1 rows) when you need closes.
Yahoo_Series :: struct {
	symbol:       string,
	dates:        []w.Date,
	open:         []f64,
	high:         []f64,
	low:          []f64,
	close:        []f64,
	adj_close:    []f64,
	volume:       []f64,
	last_is_live: bool,
	allocator:    mem.Allocator,
}

yahoo_series_free :: proc(s: ^Yahoo_Series) {
	delete(s.symbol, s.allocator)
	delete(s.dates, s.allocator)
	delete(s.open, s.allocator)
	delete(s.high, s.allocator)
	delete(s.low, s.allocator)
	delete(s.close, s.allocator)
	delete(s.adj_close, s.allocator)
	delete(s.volume, s.allocator)
	s^ = {}
}

// History of `symbol` at the interval over the range as a Yahoo_Series (see there
// for which days are kept). ok is false when Yahoo cannot be reached or no day has
// a price; the series is then empty. Free with yahoo_series_free.
read_yahoo_series :: proc(
	symbol: string,
	i: interval = .Daily,
	r: range = .Max,
	allocator: mem.Allocator = context.allocator,
) -> (
	s: Yahoo_Series,
	ok: bool,
) {
	text, got := _yahoo_chart_text(symbol, i, r)
	if !got {return {allocator = allocator}, false}
	df := yahoo_json_to_dataframe(text, context.temp_allocator)
	defer w.destroy_dataframe(&df)
	s, ok = yahoo_series_from_dataframe(&df, symbol, allocator)
	if ok && yahoo_session_live(text, _now_unix()) {
		// the running session is the frame's last row; it is kept when it has a price
		last, _ := w.column_at_date(w.column(&df, "Date"), df.rows - 1)
		s.last_is_live = s.dates[len(s.dates) - 1] == last
	}
	return
}

// The Yahoo_Series of a DataFrame with read_yahoo's columns (from
// yahoo_json_to_dataframe, for example); ok is false when no day has a price
yahoo_series_from_dataframe :: proc(
	df: ^w.DataFrame,
	symbol: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	s: Yahoo_Series,
	ok: bool,
) {
	cols := [7]^w.Column {
		w.column(df, "Date"),
		w.column(df, "Open"),
		w.column(df, "High"),
		w.column(df, "Low"),
		w.column(df, "Close"),
		w.column(df, "AdjClose"),
		w.column(df, "Volume"),
	}
	keep := make([dynamic]int, context.temp_allocator)
	for row in 0 ..< df.rows {
		c, cn := w.column_at_float(cols[4], row)
		a, an := w.column_at_float(cols[5], row)
		if cn || an || !(c > 0) || !(a > 0) {continue}
		append(&keep, row)
	}
	n := len(keep)
	if n == 0 {return {allocator = allocator}, false}

	s = Yahoo_Series {
		symbol    = strings.clone(symbol, allocator),
		dates     = make([]w.Date, n, allocator),
		open      = make([]f64, n, allocator),
		high      = make([]f64, n, allocator),
		low       = make([]f64, n, allocator),
		close     = make([]f64, n, allocator),
		adj_close = make([]f64, n, allocator),
		volume    = make([]f64, n, allocator),
		allocator = allocator,
	}
	value :: proc(c: ^w.Column, row: int) -> f64 {
		v, is_null := w.column_at_numeric(c, row)
		return math.nan_f64() if is_null else v
	}
	for row, k in keep {
		s.dates[k], _ = w.column_at_date(cols[0], row)
		s.open[k] = value(cols[1], row)
		s.high[k] = value(cols[2], row)
		s.low[k] = value(cols[3], row)
		s.close[k] = value(cols[4], row)
		s.adj_close[k] = value(cols[5], row)
		s.volume[k] = value(cols[6], row)
	}
	return s, true
}

// Adjusted closes (dividends and splits), volumes (shares traded) and dollar volumes
// (adjusted close times volume) of several tickers on the dates all of them have,
// row-major [date * n_tickers + ticker]. A ticker's rows with a missing value or a
// price <= 0 are left out before the dates are matched. last_is_live is true when the
// panel was loaded during the regular session and its last date is today's running
// session (current prices, volumes so far) for at least one ticker; the first
// len(dates) - 1 dates then hold closes only (the arrays are date-major, so their
// first (len(dates) - 1) * n_tickers values).
Yahoo_Panel :: struct {
	dates:         []w.Date,
	adj_close:     []f64,
	volume:        []f64,
	dollar_volume: []f64,
	n_tickers:     int,
	last_is_live:  bool,
	allocator:     mem.Allocator,
}

yahoo_panel_free :: proc(p: ^Yahoo_Panel) {
	delete(p.dates, p.allocator)
	delete(p.adj_close, p.allocator)
	delete(p.volume, p.allocator)
	delete(p.dollar_volume, p.allocator)
	p^ = {}
}

// The panel of `tickers` at the interval over the range (read_yahoo per ticker). ok is
// false when a ticker cannot be loaded (failed is its index) or the tickers share no
// date (failed = -1); the panel is then empty.
read_yahoo_panel :: proc(
	tickers: []string,
	i: interval = .Daily,
	r: range = .Max,
	allocator: mem.Allocator = context.allocator,
) -> (
	panel: Yahoo_Panel,
	ok: bool,
	failed: int,
) {
	NA := len(tickers)
	ta := context.temp_allocator
	prices := make([]map[w.Date]f64, NA, ta)
	vols := make([]map[w.Date]f64, NA, ta)
	first := make([dynamic]w.Date, ta)
	live := make([dynamic]w.Date, ta) // dates of running sessions
	defer {
		for k in 0 ..< NA {delete(prices[k]); delete(vols[k])}
		delete(prices, ta)
		delete(vols, ta)
		delete(first)
		delete(live)
	}
	for sym, k in tickers {
		text, got_text := _yahoo_chart_text(sym, i, r)
		if !got_text {return {allocator = allocator}, false, k}
		df := yahoo_json_to_dataframe(text, ta)
		defer w.destroy_dataframe(&df)
		if df.rows == 0 {return {allocator = allocator}, false, k}
		if yahoo_session_live(text, _now_unix()) {
			d, _ := w.column_at_date(w.column(&df, "Date"), df.rows - 1)
			append(&live, d)
		}
		prices[k] = make(map[w.Date]f64, ta)
		vols[k] = make(map[w.Date]f64, ta)
		dc, pc, vc := w.column(&df, "Date"), w.column(&df, "AdjClose"), w.column(&df, "Volume")
		for row in 0 ..< df.rows {
			d, dn := w.column_at_date(dc, row)
			p, pn := w.column_at_float(pc, row)
			v, vn := w.column_at_numeric(vc, row)
			if dn || pn || vn || !(p > 0) {continue}
			prices[k][d] = p
			vols[k][d] = v
			if k == 0 {append(&first, d)}
		}
	}
	common := make([dynamic]w.Date, ta)
	defer delete(common)
	for d in first {
		all := true
		for m in prices {all &&= d in m}
		if all {append(&common, d)}
	}
	D := len(common)
	if D == 0 {return {allocator = allocator}, false, -1}
	panel = Yahoo_Panel {
		dates         = make([]w.Date, D, allocator),
		adj_close     = make([]f64, D * NA, allocator),
		volume        = make([]f64, D * NA, allocator),
		dollar_volume = make([]f64, D * NA, allocator),
		n_tickers     = NA,
		allocator     = allocator,
	}
	for d in live {panel.last_is_live ||= d == common[D - 1]}
	copy(panel.dates, common[:])
	for d, t in common {
		for k in 0 ..< NA {
			p, v := prices[k][d], vols[k][d]
			panel.adj_close[t * NA + k] = p
			panel.volume[t * NA + k] = v
			panel.dollar_volume[t * NA + k] = p * v
		}
	}
	return panel, true, -1
}

// The adjusted closes of a panel as a DataFrame: a Date column, then one Float column
// per ticker named by `names` (one name per ticker, copied into the frame). Free with
// w.destroy_dataframe.
yahoo_panel_to_dataframe :: proc(
	panel: ^Yahoo_Panel,
	names: []string,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	NA := panel.n_tickers
	if len(names) != NA {
		panic("yahoo_panel_to_dataframe: need one column name per ticker")
	}
	D := len(panel.dates)
	df := w.dataframe_new(allocator)
	dates := w.column_new("Date", .Date, D, allocator)
	for d in panel.dates {w.append_date(&dates, d)}
	w.add_column(&df, dates)
	for name, k in names {
		col := w.column_new("", .Float, D, allocator)
		w.column_own_name(&col, name)
		for t in 0 ..< D {w.append_float(&col, panel.adj_close[t * NA + k])}
		w.add_column(&df, col)
	}
	return df
}


// Index of the '}' that closes the '{' at json[open] (Yahoo's event blocks hold no
// braces inside strings); -1 when it is not closed
@(private = "file")
_matching_brace :: proc(json: string, open: int) -> int {
	depth := 0
	for i in open ..< len(json) {
		switch json[i] {
		case '{':
			depth += 1
		case '}':
			depth -= 1
			if depth == 0 {return i}
		}
	}
	return -1
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

	end := _matching_brace(json, start) // the whole events object, not its first entry
	if end < 0 {return out}

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

	end := _matching_brace(json, start) // the whole events object, not its first entry
	if end < 0 {return out}

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
