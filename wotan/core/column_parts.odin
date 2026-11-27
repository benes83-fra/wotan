package core

import "core:mem"
import "core:strings"
import "core:unicode/utf8"

// Parts of dates and strings as new columns, and a checked comparison mask:
//
//   compare        mask from a comparison with a value (checked mask_gt / _lt / _eq)
//   date_part      year, month, weekday, ... of a Date / Datetime / Time column
//   str_*          upper / lower / strip / len / replace / slice / split_part,
//                  and the masks str_contains / str_startswith / str_endswith
//
// Missing cells give missing (null) results, and false in masks. Column
// results are named like the source, or `name` when given; the input may be
// a view and is never changed. ok = false when the column has a type the
// procedure does not take.

// --- compare ---

Compare_Op :: enum {
	Eq,
	Ne,
	Lt,
	Le,
	Gt,
	Ge,
}

// Mask of `cell op value`; missing cells give false (also for .Ne). Values
// of type int or f64 work on Int and Float columns; string, bool, Date, Time
// and Datetime values on columns of that type. ok = false on a type
// mismatch.
compare :: proc(
	col: ^Column,
	op: Compare_Op,
	value: $T,
	allocator := context.allocator,
) -> (
	[]bool,
	bool,
) {
	when T == int || T == f64 {
		if col.type != .Int && col.type != .Float {return nil, false}
		if T == int && col.type == .Int {
			return _compare_typed(col, op, int(value), allocator), true
		}
		return _compare_typed(col, op, f64(value), allocator), true
	} else {
		if col.type != _type_of(T) {return nil, false}
		return _compare_typed(col, op, value, allocator), true
	}
}

@(private = "file")
_compare_typed :: proc(
	col: ^Column,
	op: Compare_Op,
	value: $T,
	allocator: mem.Allocator,
) -> []bool {
	mask := make([]bool, col.len, allocator)
	for i in 0 ..< col.len {
		if na_at(col, i) {continue}
		when T == bool {
			v := _cell_as(col, i, T)
			c := 0 if v == value else (1 if v else -1)
		} else {
			c := _cmp(_cell_as(col, i, T), value)
		}
		switch op {
		case .Eq:
			mask[i] = c == 0
		case .Ne:
			mask[i] = c != 0
		case .Lt:
			mask[i] = c < 0
		case .Le:
			mask[i] = c <= 0
		case .Gt:
			mask[i] = c > 0
		case .Ge:
			mask[i] = c >= 0
		}
	}
	return mask
}

// --- date parts ---

// Parts of a date or time, as pandas' .dt accessors give them
Date_Part :: enum {
	Year,
	Quarter, // 1..4
	Month, // 1..12
	Day, // 1..31
	Weekday, // 0 = Monday .. 6 = Sunday (pandas dayofweek)
	Day_Of_Year, // 1..366
	Week, // ISO 8601 week number, 1..53 (pandas isocalendar().week)
	Hour,
	Minute,
	Second,
	Microsecond, // 0..999_999 (pandas .dt.microsecond)
	Nanosecond, // 0..999, the nanoseconds beyond the microsecond (pandas .dt.nanosecond)
	// Bool parts
	Is_Month_Start,
	Is_Month_End,
	Is_Quarter_Start,
	Is_Quarter_End,
	Is_Year_Start,
	Is_Year_End,
	Is_Leap_Year,
}

// Column of one part of every cell of a Date, Datetime or Time column: Int
// for the numeric parts, Bool for the Is_ parts. Date columns have no time
// parts, Time columns only Hour to Nanosecond; a Datetime column with a
// time zone gives the parts of its local time. ok = false for another
// column type or a part the type does not have.
date_part :: proc(
	col: ^Column,
	part: Date_Part,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	time_part := part >= .Hour && part <= .Nanosecond
	#partial switch col.type {
	case .Date:
		if time_part {return {}, false}
	case .Time:
		if !time_part {return {}, false}
	case .Datetime:
	case:
		return {}, false
	}
	is_flag := part >= .Is_Month_Start
	out := _new_like(col, .Bool if is_flag else .Int, name, allocator)
	zone := _Zone_Cursor {
		tz = col.tz,
	}
	for i in 0 ..< col.len {
		if na_at(col, i) {
			append_null(&out)
			continue
		}
		d: Date
		t: Time
		#partial switch col.type {
		case .Date:
			d, _ = column_at_date(col, i)
		case .Time:
			t, _ = column_at_time(col, i)
		case .Datetime:
			dt, _ := column_at_datetime(col, i)
			dt = _to_local(&zone, dt) // the zone's local time, as pandas
			d, t =
				Date{dt.year, dt.month, dt.day}, Time{dt.hour, dt.minute, dt.second, dt.nanosecond}
		}
		y, m, day := int(d.year), int(d.month), int(d.day)
		dim := int(get_days_in_month(d.year, Months(d.month))) if !time_part else 0
		leap := y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
		switch part {
		case .Year:
			append_int(&out, y)
		case .Quarter:
			append_int(&out, (m - 1) / 3 + 1)
		case .Month:
			append_int(&out, m)
		case .Day:
			append_int(&out, day)
		case .Weekday:
			append_int(&out, int(date_weekday(d)))
		case .Day_Of_Year:
			n := days_from_civil(i64(y), i64(m), i64(day)) - days_from_civil(i64(y), 1, 1)
			append_int(&out, int(n) + 1)
		case .Week:
			append_int(&out, _iso_week(d))
		case .Hour:
			append_int(&out, int(t.hour))
		case .Minute:
			append_int(&out, int(t.minute))
		case .Second:
			append_int(&out, int(t.second))
		case .Microsecond:
			append_int(&out, int(t.nanosecond) / 1000)
		case .Nanosecond:
			append_int(&out, int(t.nanosecond) % 1000)
		case .Is_Month_Start:
			append_bool(&out, day == 1)
		case .Is_Month_End:
			append_bool(&out, day == dim)
		case .Is_Quarter_Start:
			append_bool(&out, day == 1 && (m - 1) % 3 == 0)
		case .Is_Quarter_End:
			append_bool(&out, day == dim && m % 3 == 0)
		case .Is_Year_Start:
			append_bool(&out, day == 1 && m == 1)
		case .Is_Year_End:
			append_bool(&out, day == 31 && m == 12)
		case .Is_Leap_Year:
			append_bool(&out, leap)
		}
	}
	return out, true
}

// ISO 8601 week number: weeks start on Monday, week 1 holds the year's
// first Thursday
@(private = "file")
_iso_week :: proc(d: Date) -> int {
	n := days_from_civil(i64(d.year), i64(d.month), i64(d.day))
	wd := i64(date_weekday(d)) // 0 = Monday
	thursday := n - wd + 3 // the Thursday of this ISO week decides its year
	ty := civil_from_days(thursday).year
	jan1 := days_from_civil(i64(ty), 1, 1)
	return int((thursday - jan1) / 7) + 1
}

// --- strings ---

// Upper-case copy of a String column (Unicode letters, rune by rune: unlike
// Python, "ß" stays "ß")
str_upper :: proc(col: ^Column, name := "", allocator := context.allocator) -> (Column, bool) {
	return _str_map(col, .Upper, {}, name, allocator)
}

// Lower-case copy of a String column (Unicode letters, rune by rune)
str_lower :: proc(col: ^Column, name := "", allocator := context.allocator) -> (Column, bool) {
	return _str_map(col, .Lower, {}, name, allocator)
}

// Copy with leading and trailing whitespace removed, or with the characters
// in `chars` removed when given (pandas str.strip(chars))
str_strip :: proc(
	col: ^Column,
	chars := "",
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	return _str_map(col, .Strip, {a = chars}, name, allocator)
}

// Copy with every occurrence of `old` replaced by `new` (plain text, not a
// regular expression). ok = false also for an empty `old`.
str_replace :: proc(
	col: ^Column,
	old, new: string,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if old == "" {return {}, false}
	return _str_map(col, .Replace, {a = old, b = new}, name, allocator)
}

// Characters start ..< stop of every string, counted in characters (Unicode
// code points), with Python's slice rules: negative positions count from the
// end, positions beyond the ends are clamped. stop = max(int) means "to the
// end".
str_slice :: proc(
	col: ^Column,
	start: int,
	stop := max(int),
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	return _str_map(col, .Slice, {i = start, j = stop}, name, allocator)
}

// Part `index` of every string split at `sep` (pandas str.split(sep).str[index]);
// a negative index counts from the end, an empty sep splits at runs of
// whitespace. A string with too few parts gives null.
str_split_part :: proc(
	col: ^Column,
	sep: string,
	index: int,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	return _str_map(col, .Split, {a = sep, i = index}, name, allocator)
}

// Length of every string in characters (Unicode code points), as an Int column
str_len :: proc(col: ^Column, name := "", allocator := context.allocator) -> (Column, bool) {
	if col.type != .String {return {}, false}
	out := _new_like(col, .Int, name, allocator)
	for i in 0 ..< col.len {
		s, null := column_at_string(col, i)
		if null {append_null(&out)} else {append_int(&out, utf8.rune_count_in_string(s))}
	}
	return out, true
}

// Mask: does the string contain `pattern` (plain text)? With
// case_sensitive = false, letters are compared in lower case.
str_contains :: proc(
	col: ^Column,
	pattern: string,
	case_sensitive := true,
	allocator := context.allocator,
) -> (
	[]bool,
	bool,
) {
	return _str_test(col, .Contains, pattern, case_sensitive, allocator)
}

// Mask: does the string start with `prefix`?
str_startswith :: proc(
	col: ^Column,
	prefix: string,
	case_sensitive := true,
	allocator := context.allocator,
) -> (
	[]bool,
	bool,
) {
	return _str_test(col, .Starts, prefix, case_sensitive, allocator)
}

// Mask: does the string end with `suffix`?
str_endswith :: proc(
	col: ^Column,
	suffix: string,
	case_sensitive := true,
	allocator := context.allocator,
) -> (
	[]bool,
	bool,
) {
	return _str_test(col, .Ends, suffix, case_sensitive, allocator)
}

@(private = "file")
_Str_Op :: enum {
	Upper,
	Lower,
	Strip,
	Replace,
	Slice,
	Split,
}

@(private = "file")
_Str_Args :: struct {
	a, b: string,
	i, j: int,
}

@(private = "file")
_str_map :: proc(
	col: ^Column,
	op: _Str_Op,
	args: _Str_Args,
	name: string,
	allocator: mem.Allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .String {return {}, false}
	out := _new_like(col, .String, name, allocator)
	ta := context.temp_allocator
	for i in 0 ..< col.len {
		s, null := column_at_string(col, i)
		if null {
			append_null(&out)
			continue
		}
		r := s
		switch op {
		case .Upper:
			r = strings.to_upper(s, ta)
		case .Lower:
			r = strings.to_lower(s, ta)
		case .Strip:
			r = strings.trim_space(s) if args.a == "" else strings.trim(s, args.a)
		case .Replace:
			r, _ = strings.replace_all(s, args.a, args.b, ta)
		case .Slice:
			r = _rune_slice(s, args.i, args.j)
		case .Split:
			parts := strings.fields(s, ta) if args.a == "" else strings.split(s, args.a, ta)
			k := args.i if args.i >= 0 else len(parts) + args.i
			if k < 0 || k >= len(parts) {
				append_null(&out)
				continue
			}
			r = parts[k]
		}
		append_string(&out, r)
	}
	return out, true
}

// s[start:stop] in code points, with Python's rules for negative and
// out-of-range positions
@(private = "file")
_rune_slice :: proc(s: string, start, stop: int) -> string {
	n := utf8.rune_count_in_string(s)
	norm :: proc(p, n: int) -> int {
		q := p + n if p < 0 else p
		return clamp(q, 0, n)
	}
	a, b := norm(start, n), norm(stop, n) if stop != max(int) else n
	if b <= a {return ""}
	// byte offsets of code points a and b
	ba, bb, k := len(s), len(s), 0
	for off in 0 ..= len(s) {
		if off < len(s) && !utf8.rune_start(s[off]) {continue}
		if k == a {ba = off}
		if k == b {
			bb = off
			break
		}
		k += 1
	}
	return s[ba:bb]
}

@(private = "file")
_Str_Test :: enum {
	Contains,
	Starts,
	Ends,
}

@(private = "file")
_str_test :: proc(
	col: ^Column,
	op: _Str_Test,
	pattern: string,
	case_sensitive: bool,
	allocator: mem.Allocator,
) -> (
	[]bool,
	bool,
) {
	if col.type != .String {return nil, false}
	ta := context.temp_allocator
	pat := pattern if case_sensitive else strings.to_lower(pattern, ta)
	mask := make([]bool, col.len, allocator)
	for i in 0 ..< col.len {
		s, null := column_at_string(col, i)
		if null {continue}
		if !case_sensitive {s = strings.to_lower(s, ta)}
		switch op {
		case .Contains:
			mask[i] = strings.contains(s, pat)
		case .Starts:
			mask[i] = strings.has_prefix(s, pat)
		case .Ends:
			mask[i] = strings.has_suffix(s, pat)
		}
	}
	return mask, true
}
