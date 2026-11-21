package net

// Daily histories of CBOE's indices from CBOE's own CSV files: the volatility
// indices (VIX, VXN, VVIX, the single-stock VXAPL, VXAZN, VXGOG, VXGS, VXIBM),
// the implied-correlation indices (COR1M, COR3M, ...) and others such as SKEW.
// Yahoo quotes several of them (^COR1M, ^VXAPL) but serves only the latest value,
// without history.

import w "../core"
import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"

CBOE_HISTORY_URL :: "https://cdn.cboe.com/api/global/us_indices/daily_prices/%s_History.csv"

// One index: the closing value of each day, dates ascending
Cboe_Series :: struct {
	symbol:    string,
	dates:     []w.Date,
	close:     []f64,
	allocator: mem.Allocator,
}

cboe_series_free :: proc(s: ^Cboe_Series) {
	delete(s.symbol, s.allocator)
	delete(s.dates, s.allocator)
	delete(s.close, s.allocator)
	s^ = {}
}

// The daily history of a CBOE index, e.g. "COR1M" or "VXAPL" (a leading ^, as in
// Yahoo's tickers, is ignored). ok is false when CBOE cannot be reached, does not
// know the symbol (it answers 403) or the file has no valid row; the series is then
// empty. Free with cboe_series_free.
read_cboe_index :: proc(
	symbol: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	s: Cboe_Series,
	ok: bool,
) {
	sym := strings.trim_prefix(symbol, "^")
	text, got := _get_ok_body(fmt.tprintf(CBOE_HISTORY_URL, sym))
	if !got {return {allocator = allocator}, false}
	return cboe_history_parse(text, sym, allocator)
}

// A CBOE history file as a Cboe_Series. The files have a header row and one row
// per day, the date as MM/DD/YYYY: either DATE,OPEN,HIGH,LOW,CLOSE (the CLOSE
// column is used) or DATE,<value> (the last column). Rows whose date or value does
// not parse, or whose value is not > 0, are left out. ok is false when the header
// has no DATE column or no row is left.
cboe_history_parse :: proc(
	text: string,
	symbol: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	s: Cboe_Series,
	ok: bool,
) {
	s.allocator = allocator
	ta := context.temp_allocator
	lines := strings.split_lines(strings.trim_space(text), ta)
	if len(lines) < 2 {return s, false}
	header := strings.split(strings.trim_space(lines[0]), ",", ta)
	date_col, value_col := -1, len(header) - 1
	for h, k in header {
		name := strings.to_upper(strings.trim_space(h), ta)
		if name == "DATE" {date_col = k}
		if name == "CLOSE" {value_col = k}
	}
	if date_col < 0 || value_col == date_col {return s, false}
	dates := make([dynamic]w.Date, ta)
	values := make([dynamic]f64, ta)
	for line in lines[1:] {
		f := strings.split(strings.trim_space(line), ",", ta)
		if len(f) <= max(date_col, value_col) {continue}
		d, dok := _cboe_date(strings.trim_space(f[date_col]))
		v, vok := strconv.parse_f64(strings.trim_space(f[value_col]))
		if !dok || !vok || !(v > 0) {continue}
		append(&dates, d)
		append(&values, v)
	}
	if len(dates) == 0 {return s, false}
	s.symbol = strings.clone(symbol, allocator)
	s.dates = make([]w.Date, len(dates), allocator)
	s.close = make([]f64, len(values), allocator)
	copy(s.dates, dates[:])
	copy(s.close, values[:])
	return s, true
}

// MM/DD/YYYY
@(private = "file")
_cboe_date :: proc(text: string) -> (w.Date, bool) {
	p := strings.split(text, "/", context.temp_allocator)
	if len(p) != 3 {return {}, false}
	m, mok := strconv.parse_int(p[0])
	d, dok := strconv.parse_int(p[1])
	y, yok := strconv.parse_int(p[2])
	if !mok || !dok || !yok || !w.validate_date(i32(y), i32(m), i32(d)) {return {}, false}
	return w.Date{i32(y), i32(m), i32(d)}, true
}
