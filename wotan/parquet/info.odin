package parquet

import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"

import w "../core"
import util "../util"

// File metadata without the data, and row filters for read_file.
//
// A Parquet file stores, per row group and column, the null count and the
// smallest and largest value ("statistics"). info_file returns them with the
// schema and row counts; a read with filters uses them to skip the row groups
// that cannot hold a matching row, so a date range out of a long history reads
// only the groups that overlap it.

// A cell value of one of the DataFrame types: a bound of a Filter, or a
// min / max in the statistics
Value :: union {
	int,
	f64,
	bool,
	string,
	w.Date,
	w.Time,
	w.Datetime,
}

// Keeps the rows whose `column` cell lies between `lower` and `upper`; a nil
// bound leaves that end open, and `inclusive` says which ends belong to the
// range (both by default). Missing cells and NaN never match. Bounds of an
// Int or Float column are int or f64; of a Bool, String, Date, Time or
// Datetime column a value of that type. lower = upper picks one value.
Filter :: struct {
	column:       string,
	lower, upper: Value,
	inclusive:    w.Inclusive,
}

// A top-level column of the file
Column_Info :: struct {
	name:         string,
	type:         w.ColumnType, // its type in a DataFrame; .Invalid if wotan cannot load it
	parquet_type: string, // as stored, e.g. "INT64 TIMESTAMP(MICROS, UTC)"
	nullable:     bool,
	reason:       string, // why wotan cannot load it; "" if it can
}

// Statistics of one column in one row group, as the writer stored them
Column_Stats :: struct {
	null_count: int, // -1 if the file does not say
	min, max:   Value, // nil if not stored or not usable (see info_file)
}

Row_Group_Info :: struct {
	rows:             int,
	offset:           i64, // file offset of its first column chunk
	compressed_bytes: i64, // size of its column chunks in the file
	stats:            []Column_Stats, // one per entry of File_Info.columns
}

File_Info :: struct {
	rows:       int,
	columns:    []Column_Info,
	row_groups: []Row_Group_Info,
	created_by: string, // the writing library, if it says
	allocator:  mem.Allocator,
}

// Schema, row counts and per-row-group statistics of the Parquet file at
// `path`; only the footer is read. Free with destroy_info. min / max are
// given in the column's DataFrame type (a timestamp as a Datetime, a decimal
// as a Float) and are nil where the file has none, where they are NaN, and
// for INT96 timestamps, UUIDs and nested columns, whose stored order differs
// from the loaded one. Strings may be truncated bounds (a prefix as min, a
// larger string as max), as writers shorten long values. ok = false, with the
// reason on stderr, if the file cannot be read or its metadata is corrupt.
info_file :: proc(path: string, allocator := context.allocator) -> (info: File_Info, ok: bool) {
	src := _open_file(path) or_return
	defer os.close(src.file)
	return _info(&src, allocator)
}

// info_file for a Parquet file held in `data`
info_bytes :: proc(data: []u8, allocator := context.allocator) -> (info: File_Info, ok: bool) {
	src := Source {
		bytes = data,
		size  = i64(len(data)),
	}
	return _info(&src, allocator)
}

// Frees what info_file returned
destroy_info :: proc(info: ^File_Info) {
	a := info.allocator
	for &c in info.columns {
		delete(c.name, a)
		delete(c.parquet_type, a)
		delete(c.reason, a)
	}
	delete(info.columns, a)
	for &g in info.row_groups {
		for &s in g.stats {
			_free_value(s.min, a)
			_free_value(s.max, a)
		}
		delete(g.stats, a)
	}
	delete(info.row_groups, a)
	delete(info.created_by, a)
	info^ = {}
}

// Prints the rows, row groups and one line per column: its types, null count
// and range over the whole file (where every row group has statistics)
print_info :: proc(info: ^File_Info) {
	fmt.printfln(
		"%d rows in %d row group%s%s%s",
		info.rows,
		len(info.row_groups),
		"" if len(info.row_groups) == 1 else "s",
		", written by " if info.created_by != "" else "",
		info.created_by,
	)
	rows := make([][5]string, len(info.columns) + 1, context.temp_allocator)
	rows[0] = {"column", "type", "parquet type", "nulls", "range"}
	for &c, i in info.columns {
		r := &rows[i + 1]
		r[0] = c.name
		r[1] = fmt.tprint(c.type) if c.reason == "" else "-"
		r[2] = c.parquet_type
		nulls, lo, hi, known := 0, Value(nil), Value(nil), true
		for &g in info.row_groups {
			s := g.stats[i]
			if s.null_count < 0 {known = false} else {nulls += s.null_count}
		}
		r[3] = fmt.tprint(nulls) if known && len(info.row_groups) > 0 else "?"
		have := len(info.row_groups) > 0
		for &g in info.row_groups {
			s := g.stats[i]
			if s.null_count >= 0 && s.null_count == g.rows {continue} 	// all missing
			if s.min == nil || s.max == nil {
				have = false
				break
			}
			if lo == nil || _value_cmp(s.min, lo) < 0 {lo = s.min}
			if hi == nil || _value_cmp(s.max, hi) > 0 {hi = s.max}
		}
		r[4] =
			fmt.tprintf("%s .. %s", _value_text(lo), _value_text(hi)) if have && lo != nil else ""
	}
	widths: [5]int
	for r in rows {
		for k in 0 ..< 5 {widths[k] = max(widths[k], len(r[k]))}
	}
	for r in rows {
		fmt.printfln(
			"  %-*s  %-*s  %-*s  %*s  %s",
			widths[0],
			r[0],
			widths[1],
			r[1],
			widths[2],
			r[2],
			widths[3],
			r[3],
			r[4],
		)
	}
	for c in info.columns {
		if c.reason != "" {fmt.printfln("  %s cannot be loaded: %s", c.name, c.reason)}
	}
}

// ---------------------------------------------------------------------------

@(private)
_info :: proc(src: ^Source, allocator: mem.Allocator) -> (info: File_Info, ok: bool) {
	m: Meta
	defer _close_meta(&m)
	_open_meta(src, &m) or_return

	a := allocator
	info.allocator = a
	info.rows = int(m.fm.num_rows)
	info.created_by = strings.clone(m.fm.created_by, a)
	info.columns = make([]Column_Info, len(m.tops), a)
	for top, i in m.tops {
		se := m.fm.schema[top.se]
		c := &info.columns[i]
		c.name = strings.clone(top.name, a)
		c.parquet_type = strings.clone(_describe(se), a)
		c.nullable = se.repetition != 0
		switch {
		case top.leaf < 0:
			c.reason = strings.clone("nested (a list, map or struct)", a)
		case m.leaves[top.leaf].reason != "":
			c.reason = strings.clone(m.leaves[top.leaf].reason, a)
		case:
			c.type = m.leaves[top.leaf].out
		}
	}
	info.row_groups = make([]Row_Group_Info, len(m.fm.row_groups), a)
	for &rg, g in m.fm.row_groups {
		gi := &info.row_groups[g]
		gi.rows = int(rg.num_rows)
		gi.offset = -1
		for &cc in rg.columns {
			if !cc.has_meta {continue}
			start := cc.meta.data_page_offset
			if cc.meta.has_dictionary_offset &&
			   cc.meta.dictionary_page_offset > 0 &&
			   cc.meta.dictionary_page_offset < start {
				start = cc.meta.dictionary_page_offset
			}
			if gi.offset < 0 || start < gi.offset {gi.offset = start}
			gi.compressed_bytes += cc.meta.total_compressed_size
		}
		gi.stats = make([]Column_Stats, len(m.tops), a)
		for top, i in m.tops {
			s := &gi.stats[i]
			s.null_count = -1
			if top.leaf < 0 {continue}
			leaf := &m.leaves[top.leaf]
			cc := &rg.columns[leaf.chunk]
			if !cc.has_meta {continue}
			s.null_count = int(cc.meta.stats.null_count)
			if lo, hi, sok := _stat_bounds(leaf, &cc.meta.stats); sok {
				s.min, s.max = _clone_value(lo, a), _clone_value(hi, a)
			}
		}
	}
	return info, true
}

@(private)
_clone_value :: proc(v: Value, a: mem.Allocator) -> Value {
	if s, is_str := v.(string); is_str {return strings.clone(s, a)}
	return v
}

@(private)
_free_value :: proc(v: Value, a: mem.Allocator) {
	if s, is_str := v.(string); is_str {delete(s, a)}
}

@(private)
_value_text :: proc(v: Value) -> string {
	switch x in v {
	case int:
		return fmt.tprint(x)
	case f64:
		return w.float_to_string(x, context.temp_allocator)
	case bool:
		return "true" if x else "false"
	case string:
		return fmt.tprintf("%q", x)
	case w.Date:
		return w.date_to_string(x, context.temp_allocator)
	case w.Time:
		return w.time_to_string(x, context.temp_allocator)
	case w.Datetime:
		return w.datetime_to_string(x, context.temp_allocator)
	}
	return ""
}

// The type as stored, e.g. "INT64 TIMESTAMP(MICROS, UTC)" (temp memory)
@(private)
_describe :: proc(se: Schema_Element) -> string {
	CONVERTED := [?]string {
		"UTF8",
		"MAP",
		"MAP_KEY_VALUE",
		"LIST",
		"ENUM",
		"DECIMAL",
		"DATE",
		"TIME_MILLIS",
		"TIME_MICROS",
		"TIMESTAMP_MILLIS",
		"TIMESTAMP_MICROS",
		"UINT_8",
		"UINT_16",
		"UINT_32",
		"UINT_64",
		"INT_8",
		"INT_16",
		"INT_32",
		"INT_64",
		"JSON",
		"BSON",
		"INTERVAL",
	}
	UNITS := [?]string{"?", "MILLIS", "MICROS", "NANOS"}
	lt := se.logical
	annotation := ""
	switch lt.kind {
	case 0:
		if se.converted == CT_DECIMAL {
			annotation = fmt.tprintf("DECIMAL(%d, %d)", se.precision, se.scale)
		} else if se.converted >= 0 && int(se.converted) < len(CONVERTED) {
			annotation = CONVERTED[se.converted]
		}
	case LT_STRING:
		annotation = "STRING"
	case 2:
		annotation = "MAP"
	case 3:
		annotation = "LIST"
	case 4:
		annotation = "ENUM"
	case LT_DECIMAL:
		annotation = fmt.tprintf("DECIMAL(%d, %d)", lt.precision, lt.scale)
	case LT_DATE:
		annotation = "DATE"
	case LT_TIME, LT_TIMESTAMP:
		annotation = fmt.tprintf(
			"%s(%s, %s)",
			"TIME" if lt.kind == LT_TIME else "TIMESTAMP",
			UNITS[lt.unit],
			"UTC" if lt.utc else "local",
		)
	case LT_INTEGER:
		annotation = fmt.tprintf(
			"INT(%d, %s)",
			lt.bit_width,
			"signed" if lt.signed else "unsigned",
		)
	case 11:
		annotation = "NULL"
	case 12:
		annotation = "JSON"
	case 13:
		annotation = "BSON"
	case LT_UUID:
		annotation = "UUID"
	case LT_FLOAT16:
		annotation = "FLOAT16"
	case:
		annotation = fmt.tprintf("logical type %d", lt.kind)
	}
	stored: string
	if se.num_children > 0 || se.type < 0 {
		stored = annotation if annotation != "" else "struct"
		annotation = ""
	} else {
		PHYSICAL := [?]string {
			"BOOLEAN",
			"INT32",
			"INT64",
			"INT96",
			"FLOAT",
			"DOUBLE",
			"BYTE_ARRAY",
			"FIXED_LEN_BYTE_ARRAY",
		}
		if se.type >= 0 && int(se.type) < len(PHYSICAL) {
			stored = PHYSICAL[se.type]
		} else {
			stored = fmt.tprintf("type %d", se.type)
		}
		if Physical_Type(se.type) == .Fixed_Len_Byte_Array {
			stored = fmt.tprintf("%s(%d)", stored, se.type_length)
		}
	}
	if se.repetition == 2 {stored = fmt.tprintf("repeated %s", stored)}
	return fmt.tprintf("%s %s", stored, annotation) if annotation != "" else stored
}

// ---------------------------------------------------------------------------
// Statistics

// min and max of a chunk in the leaf's DataFrame type; ok = false if the
// file has none that wotan can use. Strings point into the footer.
@(private)
_stat_bounds :: proc(leaf: ^Leaf, st: ^Raw_Stats) -> (lo, hi: Value, ok: bool) {
	if leaf.nested || leaf.reason != "" {return}
	if st.has_min && st.has_max {
		lo_ok, hi_ok: bool
		lo, lo_ok = _stat_value(leaf, st.min)
		hi, hi_ok = _stat_value(leaf, st.max)
		return lo, hi, lo_ok && hi_ok
	}
	// the deprecated fields are in signed order: right for signed numbers only
	#partial switch leaf.conv {
	case .Bool,
	     .Int32,
	     .Int64,
	     .Float32,
	     .Float64,
	     .Date32,
	     .Time_Millis,
	     .Time_Micros,
	     .Time_Nanos,
	     .Ts_Millis,
	     .Ts_Micros,
	     .Ts_Nanos,
	     .Dec_Int32,
	     .Dec_Int64:
		if leaf.unsigned || !st.has_old_min || !st.has_old_max {return}
		lo_ok, hi_ok: bool
		lo, lo_ok = _stat_value(leaf, st.min_old)
		hi, hi_ok = _stat_value(leaf, st.max_old)
		return lo, hi, lo_ok && hi_ok
	}
	return
}

// One PLAIN-encoded statistics value as a cell value, converted as the
// column's cells are
@(private)
_stat_value :: proc(leaf: ^Leaf, b: []u8) -> (Value, bool) {
	le :: proc(b: []u8) -> (v: u64) {
		for k := len(b) - 1; k >= 0; k -= 1 {v = v << 8 | u64(b[k])}
		return
	}
	switch leaf.conv {
	case .Bool:
		if len(b) != 1 {return nil, false}
		return b[0] != 0, true
	case .Int32, .UInt32, .Date32, .Time_Millis, .Dec_Int32:
		if len(b) != 4 {return nil, false}
		v := i32(u32(le(b)))
		#partial switch leaf.conv {
		case .Int32:
			return int(v), true
		case .UInt32:
			return int(u32(v)), true
		case .Date32:
			return w.civil_from_days(i64(v)), true
		case .Time_Millis:
			return _time_of_day(i64(v), 1000), true
		}
		return f64(v) / _pow10(leaf.scale), true
	case .Int64, .Time_Micros, .Time_Nanos, .Ts_Millis, .Ts_Micros, .Ts_Nanos, .Dec_Int64:
		if len(b) != 8 {return nil, false}
		u := le(b)
		v := i64(u)
		#partial switch leaf.conv {
		case .Int64:
			// UINT64 cells above max(i64) wrap to negative numbers
			if leaf.unsigned && u > u64(max(i64)) {return nil, false}
			return int(v), true
		case .Time_Micros:
			return _time_of_day(v, 1_000_000), true
		case .Time_Nanos:
			return _time_of_day(v, 1_000_000_000), true
		case .Ts_Millis:
			return _ts(v, 1000), true
		case .Ts_Micros:
			return _ts(v, 1_000_000), true
		case .Ts_Nanos:
			return _ts(v, 1_000_000_000), true
		}
		return f64(v) / _pow10(leaf.scale), true
	case .Float32:
		if len(b) != 4 {return nil, false}
		v := f64(transmute(f32)u32(le(b)))
		if v != v {return nil, false}
		return v, true
	case .Float64:
		if len(b) != 8 {return nil, false}
		v := transmute(f64)le(b)
		if v != v {return nil, false}
		return v, true
	case .Float16:
		if len(b) != 2 {return nil, false}
		v := _half(u16(le(b)))
		if v != v {return nil, false}
		return v, true
	case .Dec_Bytes:
		if len(b) == 0 {return nil, false}
		return _be_int(b) / _pow10(leaf.scale), true
	case .Bytes:
		return string(b), true
	case .Int96, .Uuid:
	// INT96 has no defined order; UUID bytes are shown as text
	}
	return nil, false
}

// -1, 0 or 1 as a is below, equal to or above b (an int and an f64 compare
// as numbers; other values only with their own type)
@(private)
_value_cmp :: proc(a, b: Value) -> int {
	cmp :: proc(x, y: $T) -> int {return -1 if x < y else 1 if x > y else 0}
	switch x in a {
	case int:
		#partial switch y in b {
		case int:
			return cmp(x, y)
		case f64:
			return cmp(f64(x), y)
		}
	case f64:
		#partial switch y in b {
		case int:
			return cmp(x, f64(y))
		case f64:
			return cmp(x, y)
		}
	case bool:
		return cmp(int(x), int(b.(bool)))
	case string:
		return cmp(x, b.(string))
	case w.Date:
		return int(w.date_compare(x, b.(w.Date)))
	case w.Time:
		return int(w.time_compare(x, b.(w.Time)))
	case w.Datetime:
		return int(w.datetime_compare(x, b.(w.Datetime)))
	}
	return 0
}

// ---------------------------------------------------------------------------
// Filters

// Whether the bounds of `f` suit a column of type `out`
@(private)
_check_filter :: proc(f: ^Filter, out: w.ColumnType) -> bool {
	for b in ([2]Value{f.lower, f.upper}) {
		if b == nil {continue}
		fits := false
		switch x in b {
		case int:
			fits = out == .Int || out == .Float
		case f64:
			if x != x {return _fail("filter on %q: NaN bound", f.column)}
			fits = out == .Int || out == .Float
		case bool:
			fits = out == .Bool
		case string:
			fits = out == .String
		case w.Date:
			fits = out == .Date
		case w.Time:
			fits = out == .Time
		case w.Datetime:
			fits = out == .Datetime
		}
		if !fits {
			return _fail(
				"filter on %q: a bound of type %v for a %v column",
				f.column,
				_variant_name(b),
				out,
			)
		}
	}
	return true
}

@(private)
_variant_name :: proc(v: Value) -> string {
	switch _ in v {
	case int:
		return "int"
	case f64:
		return "f64"
	case bool:
		return "bool"
	case string:
		return "string"
	case w.Date:
		return "Date"
	case w.Time:
		return "Time"
	case w.Datetime:
		return "Datetime"
	}
	return "nil"
}

// Whether a lower bound / an upper bound admits a value that compares to it
// as `c` (-1 below, 0 equal, 1 above)
@(private)
_above_lower :: #force_inline proc(c: int, inc: w.Inclusive) -> bool {
	return c > 0 || (c == 0 && (inc == .Both || inc == .Left))
}

@(private)
_below_upper :: #force_inline proc(c: int, inc: w.Inclusive) -> bool {
	return c < 0 || (c == 0 && (inc == .Both || inc == .Right))
}

// True if the statistics show that no row of the group matches `f`: every
// cell missing, or the [min, max] range outside the filter's range
@(private)
_group_excluded :: proc(f: ^Filter, leaf: ^Leaf, st: ^Raw_Stats, rows: i64) -> bool {
	if rows == 0 || st.null_count == rows {return true}
	lo, hi, ok := _stat_bounds(leaf, st)
	if !ok {return false}
	if f.lower != nil && !_above_lower(_value_cmp(hi, f.lower), f.inclusive) {return true}
	if f.upper != nil && !_below_upper(_value_cmp(lo, f.upper), f.inclusive) {return true}
	return false
}

@(private)
Match_Work :: struct {
	filters: []Filter,
	cols:    []^w.Column, // the column of each filter
	keep:    []bool, // per row: matches every filter
}

// The rows of the loaded columns that match every filter, as keep[r]
@(private)
_match_all :: proc(filters: []Filter, cols: []^w.Column, keep: []bool) {
	work := Match_Work {
		filters = filters,
		cols    = cols,
		keep    = keep,
	}
	body :: proc(data: rawptr, begin, end: int) {
		work := (^Match_Work)(data)
		keep := work.keep[begin:end]
		for &k in keep {k = true}
		for &f, i in work.filters {_match_rows(&f, work.cols[i], keep, begin)}
	}
	BLOCK :: 1 << 16
	blocks := (len(keep) + BLOCK - 1) / BLOCK
	if blocks <= 1 {
		body(&work, 0, len(keep))
		return
	}
	util.parallel_for(len(keep), &work, body)
}

// Clears keep[i] for the rows first + i whose cell does not match `f`. `col`
// is a column the reader built: cells back to back, one null flag per row.
@(private)
_match_rows :: proc(f: ^Filter, col: ^w.Column, keep: []bool, first: int) {
	switch col.type {
	case .Int:
		_, lo_int := f.lower.(int)
		_, hi_int := f.upper.(int)
		if (f.lower == nil || lo_int) && (f.upper == nil || hi_int) {
			_match_typed(f, col, keep, first, int, int)
		} else {
			_match_typed(f, col, keep, first, int, f64)
		}
	case .Float:
		_match_typed(f, col, keep, first, f64, f64)
	case .Bool:
		_match_typed(f, col, keep, first, bool, bool)
	case .String:
		_match_typed(f, col, keep, first, string, string)
	case .Date:
		_match_typed(f, col, keep, first, w.Date, w.Date)
	case .Time:
		_match_typed(f, col, keep, first, w.Time, w.Time)
	case .Datetime:
		_match_typed(f, col, keep, first, w.Datetime, w.Datetime)
	case .Invalid:
		for &k in keep {k = false}
	}
}

// _match_rows for cells of type C, compared as T (an Int column against f64
// bounds compares as f64)
@(private)
_match_typed :: proc(
	f: ^Filter,
	col: ^w.Column,
	keep: []bool,
	first: int,
	$C: typeid,
	$T: typeid,
) {
	bound :: proc(v: Value) -> (T, bool) {
		when T == f64 {
			#partial switch x in v {
			case int:
				return f64(x), true
			case f64:
				return x, true
			}
			return 0, false
		} else {
			return v.(T)
		}
	}
	cmp :: #force_inline proc(a, b: T) -> int {
		when T == w.Date {
			return int(w.date_compare(a, b))
		} else when T == w.Time {
			return int(w.time_compare(a, b))
		} else when T == w.Datetime {
			return int(w.datetime_compare(a, b))
		} else when T == bool {
			return int(a) - int(b)
		} else when T == string {
			// byte-wise, inline (short values such as tickers dominate)
			n := min(len(a), len(b))
			for i in 0 ..< n {
				if a[i] != b[i] {return -1 if a[i] < b[i] else 1}
			}
			return -1 if len(a) < len(b) else 1 if len(a) > len(b) else 0
		} else {
			return -1 if a < b else 1 if a > b else 0
		}
	}
	lo, has_lo := bound(f.lower)
	hi, has_hi := bound(f.upper)
	cells := ([^]C)(col.data)[first:]
	nulls := col.nulls[first:] if col.nulls != nil else nil
	inc := f.inclusive
	for r in 0 ..< len(keep) {
		if !keep[r] {continue}
		if nulls != nil && nulls[r] {
			keep[r] = false
			continue
		}
		v := T(cells[r])
		when T == f64 {
			if v != v {
				keep[r] = false
				continue
			}
		}
		keep[r] =
			(!has_lo || _above_lower(cmp(v, lo), inc)) &&
			(!has_hi || _below_upper(cmp(v, hi), inc))
	}
}

// Keeps rows `rows` (ascending) of a column the reader built, in place. A
// column left mostly empty is moved to a buffer of its new size; a String
// column then gets a fresh arena with only the kept strings.
@(private)
_keep_rows :: proc(col: ^w.Column, rows: []int) {
	n := len(rows)
	shrink := 2 * n < col.len
	if col.type == .String && shrink {
		taken := w.column_take(col, rows, col.allocator)
		w.destroy_column(col)
		col^ = taken
		return
	}
	compact :: proc(cells: [^]$T, nulls: []bool, rows: []int) {
		for r, j in rows {
			cells[j] = cells[r]
			if nulls != nil {nulls[j] = nulls[r]}
		}
	}
	switch col.type {
	case .Int:
		compact(([^]int)(col.data), col.nulls, rows)
	case .Float:
		compact(([^]f64)(col.data), col.nulls, rows)
	case .Bool:
		compact(([^]bool)(col.data), col.nulls, rows)
	case .String:
		compact(([^]string)(col.data), col.nulls, rows)
	case .Date:
		compact(([^]w.Date)(col.data), col.nulls, rows)
	case .Time:
		compact(([^]w.Time)(col.data), col.nulls, rows)
	case .Datetime:
		compact(([^]w.Datetime)(col.data), col.nulls, rows)
	case .Invalid:
	}
	col.len = n
	if shrink {
		size := w.type_size(col.type)
		cap := max(n, 1)
		data, err := mem.alloc(cap * size, allocator = col.allocator)
		if err != nil {return} 	// keep the larger buffer
		mem.copy_non_overlapping(data, col.data, n * size)
		mem.free(col.data, col.allocator)
		col.data = data
		if col.nulls != nil {
			nulls := make([]bool, cap, col.allocator)
			copy(nulls, col.nulls[:n])
			delete(col.nulls, col.allocator)
			col.nulls = nulls
		}
		col.capacity = cap
	}
}
