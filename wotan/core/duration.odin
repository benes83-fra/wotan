package core

import "core:strings"

// Durations: lengths of time to the nanosecond (pandas' Timedelta).
//
//   NANOSECONDS .. WEEKS           units to build one: 15 * MINUTES
//   datetime_sub / datetime_add    a - b of two Datetimes, a Datetime moved by a Duration
//   duration_to_string / parse_duration   text as pandas writes it: "1 days 02:03:04.5"
//   duration_between               Duration column of later - earlier (Datetime,
//                                  Date, Time or Duration columns)
//   add_duration / add_durations   a Datetime (or Date, or Duration) column moved
//                                  by one Duration / by a Duration column
//   duration_total                 a Duration column in a unit, as Float (total_seconds)
//   datetime_floor                 Datetimes rounded down to a multiple of a Duration
//
// diff of a Datetime, Date, Time or Duration column gives a Duration column,
// and resample_every bins rows into intervals such as 15 minutes.

NANOSECONDS :: Duration(1)
MICROSECONDS :: Duration(1_000)
MILLISECONDS :: Duration(1_000_000)
SECONDS :: Duration(1_000_000_000)
MINUTES :: 60 * SECONDS
HOURS :: 60 * MINUTES
DAYS :: 24 * HOURS
WEEKS :: 7 * DAYS

// Nanoseconds of a Datetime since 1970-01-01 (any year: an i128 holds them)
@(private)
_datetime_ns :: proc(dt: Datetime) -> i128 {
	days := days_from_civil(i64(dt.year), i64(dt.month), i64(dt.day))
	secs := i128(days) * 86400 + i128(dt.hour) * 3600 + i128(dt.minute) * 60 + i128(dt.second)
	return secs * 1_000_000_000 + i128(dt.nanosecond)
}

@(private)
_time_ns :: proc(t: Time) -> i128 {
	return(
		(i128(t.hour) * 3600 + i128(t.minute) * 60 + i128(t.second)) * 1_000_000_000 +
		i128(t.nanosecond) \
	)
}

// A nanosecond count as a Duration; ok = false outside the i64 range
@(private)
_duration_of :: proc(ns: i128) -> (Duration, bool) {
	if ns < i128(min(i64)) || ns > i128(max(i64)) {return 0, false}
	return Duration(ns), true
}

// a - b; ok = false when the difference is beyond the Duration range (about
// 292 years)
datetime_sub :: proc(a, b: Datetime) -> (Duration, bool) {
	return _duration_of(_datetime_ns(a) - _datetime_ns(b))
}

// `dt` moved by `d` (back for a negative d)
datetime_add :: proc(dt: Datetime, d: Duration) -> Datetime {
	return add_nanoseconds_datetime(dt, i64(d))
}

// The Duration in seconds
duration_to_seconds :: proc(d: Duration) -> f64 {
	return f64(d) / 1e9
}

// The parts pandas shows: whole days (rounded down, so negative for a
// negative Duration) and the time after them, 0 ..< 24 hours
duration_components :: proc(d: Duration) -> (days: i64, time: Time) {
	ns := i64(d)
	day := i64(DAYS)
	days = ns / day
	rest := ns % day
	if rest < 0 {
		rest += day
		days -= 1
	}
	secs := rest / 1_000_000_000
	time = Time{i32(secs / 3600), i32(secs % 3600 / 60), i32(secs % 60), i32(rest % 1_000_000_000)}
	return
}

// "D days HH:MM:SS" as pandas writes a Timedelta, with ".fff", ".ffffff" or
// ".fffffffff" for a fraction of a second; a negative Duration is "-1 days
// +23:55:00" (days rounded down, then the time after them), so that
// pandas.to_timedelta reads the text back to the same value. Into `buf`, no
// allocation.
duration_format_into :: proc(d: Duration, buf: ^[48]u8) -> string {
	days, t := duration_components(d)
	n := 0
	put :: proc(buf: ^[48]u8, n: ^int, s: string) {
		copy(buf[n^:], s)
		n^ += len(s)
	}
	digits: [24]u8
	k := len(digits)
	v := days
	neg := v < 0
	if neg {v = -v}
	for {
		k -= 1
		digits[k] = u8('0' + v % 10)
		v /= 10
		if v == 0 {break}
	}
	if neg {put(buf, &n, "-")}
	put(buf, &n, string(digits[k:]))
	put(buf, &n, " days +" if neg else " days ")
	two :: proc(buf: ^[48]u8, n: ^int, x: i32) {
		buf[n^] = u8('0' + x / 10)
		buf[n^ + 1] = u8('0' + x % 10)
		n^ += 2
	}
	two(buf, &n, t.hour)
	put(buf, &n, ":")
	two(buf, &n, t.minute)
	put(buf, &n, ":")
	two(buf, &n, t.second)
	fb: [10]u8
	put(buf, &n, fraction_string(t.nanosecond, &fb))
	return string(buf[:n])
}

// duration_format_into, allocated
duration_to_string :: proc(d: Duration, allocator := context.allocator) -> string {
	buf: [48]u8
	return strings.clone(duration_format_into(d, &buf), allocator)
}

// A Duration from text as pandas writes and reads it:
//   "D days HH:MM:SS[.f]"   D days (may be negative) plus the time, "+" or
//                           "-" allowed before the time ("-1 days +23:55:00"
//                           is minus 5 minutes); "day" for "days" and a comma
//                           after it are accepted
//   "D days"                whole days
//   "[-]HH:MM:SS[.f]"       hours (any number of digits), minutes, seconds,
//                           one to nine fraction digits
// No allocation.
parse_duration :: proc(text: string) -> (Duration, bool) {
	s := strings.trim_space(text)
	if len(s) == 0 {return 0, false}
	total: i128
	if i := strings.index(s, "day"); i >= 0 {
		head := strings.trim_space(s[:i])
		rest := s[i + 3:]
		if strings.has_prefix(rest, "s") {rest = rest[1:]}
		if strings.has_prefix(rest, ",") {rest = rest[1:]}
		rest = strings.trim_space(rest)
		neg := false
		if strings.has_prefix(head, "-") {
			neg, head = true, head[1:]
		} else if strings.has_prefix(head, "+") {
			head = head[1:]
		}
		days: i128
		if len(head) == 0 || len(head) > 9 {return 0, false}
		for c in transmute([]u8)head {
			if c < '0' || c > '9' {return 0, false}
			days = days * 10 + i128(c - '0')
		}
		total = (-days if neg else days) * i128(DAYS)
		if rest != "" {
			t, ok := _clock_ns(rest)
			if !ok {return 0, false}
			total += t
		}
	} else {
		t, ok := _clock_ns(s)
		if !ok {return 0, false}
		total = t
	}
	return _duration_of(total)
}

// "[+|-]H..H:MM:SS[.f]" in nanoseconds
@(private = "file")
_clock_ns :: proc(text: string) -> (i128, bool) {
	s := text
	neg := false
	if strings.has_prefix(s, "-") {
		neg, s = true, s[1:]
	} else if strings.has_prefix(s, "+") {
		s = s[1:]
	}
	c1 := strings.index_byte(s, ':')
	if c1 <= 0 || c1 > 9 {return 0, false}
	c2 := c1 + 3
	if len(s) < c2 + 3 || s[c2] != ':' {return 0, false}
	num :: proc(s: string) -> (v: i128, ok: bool) {
		if len(s) == 0 {return}
		for c in transmute([]u8)s {
			if c < '0' || c > '9' {return}
			v = v * 10 + i128(c - '0')
		}
		return v, true
	}
	h, hok := num(s[:c1])
	m, mok := num(s[c1 + 1:c2])
	if !hok || !mok || m > 59 {return 0, false}
	sec_text := s[c2 + 1:]
	frac := ""
	if dot := strings.index_byte(sec_text, '.'); dot >= 0 {
		sec_text, frac = sec_text[:dot], sec_text[dot + 1:]
		if len(frac) == 0 || len(frac) > 9 {return 0, false}
	}
	if len(sec_text) != 2 {return 0, false}
	sec, sok := num(sec_text)
	if !sok || sec > 59 {return 0, false}
	ns: i128
	if frac != "" {
		f, fok := num(frac)
		if !fok {return 0, false}
		ns = f
		for _ in len(frac) ..< 9 {ns *= 10}
	}
	v := ((h * 60 + m) * 60 + sec) * 1_000_000_000 + ns
	return -v if neg else v, true
}

// ---------------------------------------------------------------------------
// Columns

// Duration column of `later` - `earlier`, row by row: two Datetime, Date,
// Time or Duration columns of the same type and length (pandas' b - a). A
// missing value on either side, or a difference beyond the Duration range,
// gives a missing result. ok = false for other types or lengths.
duration_between :: proc(
	later, earlier: ^Column,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if later.type != earlier.type || later.len != earlier.len {return {}, false}
	#partial switch later.type {
	case .Datetime, .Date, .Time, .Duration:
	case:
		return {}, false
	}
	out := _new_like(later, .Duration, name, allocator)
	for i in 0 ..< later.len {
		d, ok := _ns_of(later, i)
		e, eok := _ns_of(earlier, i)
		v, vok := _duration_of(d - e)
		if !ok || !eok || !vok {
			append_null(&out)
		} else {
			append_duration(&out, v)
		}
	}
	return out, true
}

// Nanoseconds of a present Datetime / Date (since 1970), Time (since
// midnight) or Duration cell; false for a missing cell
@(private)
_ns_of :: proc(col: ^Column, i: int) -> (i128, bool) {
	#partial switch col.type {
	case .Datetime:
		v, missing := column_at_datetime(col, i)
		return _datetime_ns(v), !missing
	case .Date:
		v, missing := column_at_date(col, i)
		return _datetime_ns(Datetime{v.year, v.month, v.day, 0, 0, 0, 0}), !missing
	case .Time:
		v, missing := column_at_time(col, i)
		return _time_ns(v), !missing
	case .Duration:
		v, missing := column_at_duration(col, i)
		return i128(v), !missing
	}
	return 0, false
}

// Copy of a Datetime or Duration column with `d` added to every cell; a Date
// column gives a Datetime column (midnight plus d), as in pandas. A result
// beyond the Duration range (Duration input) is missing. ok = false for
// other column types.
add_duration :: proc(
	col: ^Column,
	d: Duration,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	#partial switch col.type {
	case .Datetime, .Date, .Duration:
	case:
		return {}, false
	}
	out := _new_like(col, .Duration if col.type == .Duration else .Datetime, name, allocator)
	for i in 0 ..< col.len {_append_moved(&out, col, i, i128(d), true)}
	return out, true
}

// Like add_duration, with the Duration of each row taken from `durations`
// (a Duration column of the same length; a missing one gives a missing
// result). ok = false for other types or lengths.
add_durations :: proc(
	col, durations: ^Column,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if durations.type != .Duration || durations.len != col.len {return {}, false}
	#partial switch col.type {
	case .Datetime, .Date, .Duration:
	case:
		return {}, false
	}
	out := _new_like(col, .Duration if col.type == .Duration else .Datetime, name, allocator)
	for i in 0 ..< col.len {
		d, missing := column_at_duration(durations, i)
		_append_moved(&out, col, i, i128(d), !missing)
	}
	return out, true
}

@(private = "file")
_append_moved :: proc(out, col: ^Column, i: int, d: i128, present: bool) {
	if !present || is_null(col, i) {
		append_null(out)
		return
	}
	#partial switch col.type {
	case .Datetime:
		v, _ := column_at_datetime(col, i)
		append_datetime(out, datetime_add(v, Duration(d)))
	case .Date:
		v, _ := column_at_date(col, i)
		append_datetime(
			out,
			datetime_add(Datetime{v.year, v.month, v.day, 0, 0, 0, 0}, Duration(d)),
		)
	case .Duration:
		v, _ := column_at_duration(col, i)
		r, ok := _duration_of(i128(v) + d)
		if ok {append_duration(out, r)} else {append_null(out)}
	}
}

// Float column of every Duration in units of `unit`: duration_total(col,
// SECONDS) is pandas' dt.total_seconds(), duration_total(col, DAYS) the days
// with their fraction. Missing cells stay missing. ok = false for another
// column type or a unit that is not positive.
duration_total :: proc(
	col: ^Column,
	unit: Duration,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .Duration || unit <= 0 {return {}, false}
	out := _new_like(col, .Float, name, allocator)
	for i in 0 ..< col.len {
		v, missing := column_at_duration(col, i)
		if missing {
			append_null(&out)
		} else {
			// whole units and the rest apart, so that large counts keep
			// every digit a double can show
			q, r := i64(v) / i64(unit), i64(v) % i64(unit)
			append_float(&out, f64(q) + f64(r) / f64(unit))
		}
	}
	return out, true
}

// Copy of a Datetime column with every cell rounded down to a multiple of
// `every` counted from midnight 1970-01-01 (pandas' dt.floor): with every =
// 15 * MINUTES, 09:44:59 becomes 09:30:00. ok = false for another column
// type or an `every` that is not positive.
datetime_floor :: proc(
	col: ^Column,
	every: Duration,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .Datetime || every <= 0 {return {}, false}
	out := _new_like(col, .Datetime, name, allocator)
	for i in 0 ..< col.len {
		v, missing := column_at_datetime(col, i)
		if missing {
			append_null(&out)
			continue
		}
		ns := _datetime_ns(v)
		e := i128(every)
		q := ns / e
		if ns % e < 0 {q -= 1}
		append_datetime(&out, _datetime_of_ns(q * e))
	}
	return out, true
}

// The Datetime `ns` nanoseconds after 1970-01-01 (any year)
@(private)
_datetime_of_ns :: proc(ns: i128) -> Datetime {
	per_day := i128(DAYS)
	days := ns / per_day
	rest := ns % per_day
	if rest < 0 {
		rest += per_day
		days -= 1
	}
	d := civil_from_days(i64(days))
	secs := i64(rest / 1_000_000_000)
	return Datetime {
		d.year,
		d.month,
		d.day,
		i32(secs / 3600),
		i32(secs % 3600 / 60),
		i32(secs % 60),
		i32(rest % 1_000_000_000),
	}
}
