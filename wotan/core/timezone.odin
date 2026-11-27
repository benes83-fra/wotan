package core

import "core:strings"

// Time zones (the IANA database, version tz_data_version(), built in: no
// files or registry needed, the same answers on every machine).
//
// A Datetime column has a time zone in col.tz. Without one (the zero value)
// its values are plain wall-clock times, as before. With one, the values
// are instants stored in UTC, shown in the zone's local time with its UTC
// offset ("2024-03-10 03:30:00-04:00"), as in pandas.
//
//   time_zone("America/New_York")      the zone of an IANA name, "UTC" or a
//                                      fixed offset such as "+05:30"
//   time_zone_name / time_zone_names   names of one zone / of all zones
//   utc_offset                         the zone's offset from UTC at an instant
//   datetime_to_local / datetime_from_local   one instant between UTC and
//                                      local time
//   tz_localize / tz_convert           columns: give wall-clock times a zone /
//                                      show instants in another zone
//   to_local_time                      wall-clock times of a zoned column, no zone

// The time zone of a Datetime column or value: none (the zero value), a
// zone of the IANA database, or a fixed offset from UTC
Time_Zone :: struct {
	id:     i32, // 0 none; > 0 the IANA name number + 1; -1 fixed offset
	offset: i32, // seconds east of UTC, for a fixed offset
}

// What tz_localize does with a wall-clock time that happens twice (the hour
// when clocks go back)
Ambiguous_Time :: enum {
	Raise, // ok = false (pandas' ambiguous="raise")
	Earliest, // the first of the two instants (usually summer time)
	Latest, // the second
	Missing, // a missing value (pandas' ambiguous="NaT")
}

// What tz_localize does with a wall-clock time that never happens (the hour
// skipped when clocks go forward)
Nonexistent_Time :: enum {
	Raise, // ok = false (pandas' nonexistent="raise")
	Shift_Forward, // the first instant after the gap (02:30 -> 03:00)
	Shift_Backward, // the last nanosecond before it (02:30 -> 01:59:59.999999999)
	Missing, // a missing value (pandas' nonexistent="NaT")
}

UTC_ZONE_NAME :: "UTC"

@(private = "file")
TZ_DATA := #load("tzdata.bin")

@(private = "file")
_Db :: struct {
	version: string,
	names:   []string,
	zones:   []u16, // record of each name
	records: []u32, // offset of each record in TZ_DATA
}

@(private = "file")
_db: _Db

@(private = "file")
_names_store: [1024]string

@(private = "file")
_zones_store: [1024]u16

@(private = "file")
_records_store: [1024]u32

// "+HH:MM" for every whole minute from -24:00 to +24:00
@(private = "file")
_fixed_names: [2 * 24 * 60 + 1][6]u8

@(private = "file")
_u16 :: proc "contextless" (p: int) -> u16 {return u16(TZ_DATA[p]) | u16(TZ_DATA[p + 1]) << 8}

@(private = "file")
_u32 :: proc "contextless" (p: int) -> u32 {
	return(
		u32(TZ_DATA[p]) |
		u32(TZ_DATA[p + 1]) << 8 |
		u32(TZ_DATA[p + 2]) << 16 |
		u32(TZ_DATA[p + 3]) << 24 \
	)
}

@(private = "file")
_i64 :: proc "contextless" (p: int) -> i64 {
	return i64(u64(_u32(p)) | u64(_u32(p + 4)) << 32)
}

@(init, private = "file")
_tz_init :: proc "contextless" () {
	d := TZ_DATA
	p := 4
	vlen := int(d[p])
	_db.version = string(d[p + 1:p + 1 + vlen])
	p += 1 + vlen
	n_names := int(_u16(p))
	n_zones := int(_u16(p + 2))
	p += 4
	for k in 0 ..< n_names {
		l := int(d[p])
		_names_store[k] = string(d[p + 1:p + 1 + l])
		_zones_store[k] = _u16(p + 1 + l)
		p += 3 + l
	}
	for k in 0 ..< n_zones {_records_store[k] = _u32(p + 4 * k)}
	_db.names = _names_store[:n_names]
	_db.zones = _zones_store[:n_names]
	_db.records = _records_store[:n_zones]

	for k in 0 ..< len(_fixed_names) {
		m := k - 24 * 60
		a := m if m >= 0 else -m
		_fixed_names[k] = {
			'+' if m >= 0 else '-',
			u8('0' + a / 600),
			u8('0' + (a / 60) % 10),
			':',
			u8('0' + (a % 60) / 10),
			u8('0' + a % 10),
		}
	}
}

// Version of the built-in time zone database, e.g. "2026f"
tz_data_version :: proc() -> string {return _db.version}

// Every IANA zone name the database knows, sorted (static strings; free only
// the slice)
time_zone_names :: proc(allocator := context.allocator) -> []string {
	out := make([]string, len(_db.names), allocator)
	copy(out, _db.names)
	return out
}

// The zone of a name: an IANA name such as "America/New_York" or
// "Europe/Berlin" (letter case is forgiven), "UTC" (also "Z"), or a fixed
// offset "+05:30", "-0800", "+05". ok = false for anything else.
time_zone :: proc(name: string) -> (Time_Zone, bool) {
	if name == "" {return {}, false}
	if name == "Z" || name == "z" {return time_zone(UTC_ZONE_NAME)}
	if name[0] == '+' || name[0] == '-' {return _fixed_zone(name)}
	lo, hi := 0, len(_db.names)
	for lo < hi {
		mid := (lo + hi) / 2
		if _db.names[mid] < name {lo = mid + 1} else {hi = mid}
	}
	if lo < len(_db.names) && _db.names[lo] == name {return {id = i32(lo + 1)}, true}
	for n, k in _db.names {
		if strings.equal_fold(n, name) {return {id = i32(k + 1)}, true}
	}
	return {}, false
}

@(private = "file")
_fixed_zone :: proc(s: string) -> (Time_Zone, bool) {
	sign := 1 if s[0] == '+' else -1
	digits: [4]int
	n := 0
	for c, i in s[1:] {
		if c == ':' && i == 2 && len(s) == 6 {continue}
		if c < '0' || c > '9' || n == 4 {return {}, false}
		digits[n] = int(c - '0')
		n += 1
	}
	if n != 2 && n != 4 {return {}, false}
	h := digits[0] * 10 + digits[1]
	m := digits[2] * 10 + digits[3] if n == 4 else 0
	if m >= 60 || h * 60 + m > 24 * 60 {return {}, false}
	return {id = -1, offset = i32(sign * (h * 3600 + m * 60))}, true
}

// The zone as a fixed offset (seconds east of UTC; |offset| at most 24 hours,
// whole minutes); ok = false otherwise
fixed_offset_zone :: proc(seconds: int) -> (Time_Zone, bool) {
	if seconds % 60 != 0 || abs(seconds) > 86400 {return {}, false}
	return {id = -1, offset = i32(seconds)}, true
}

// Name of a zone: its IANA name, "+05:30" for a fixed offset, "" for none
// (static memory)
time_zone_name :: proc(tz: Time_Zone) -> string {
	switch {
	case tz.id > 0 && int(tz.id) <= len(_db.names):
		return _db.names[tz.id - 1]
	case tz.id == -1:
		return string(_fixed_names[tz.offset / 60 + 24 * 60][:])
	}
	return ""
}

// ---------------------------------------------------------------------------
// Offsets

// A stretch of UTC seconds [start, next) with one UTC offset
@(private)
_Span :: struct {
	start, next: i64,
	offset:      i32,
}

@(private = "file")
_NEVER_BEFORE :: min(i64)

@(private = "file")
_NEVER_AFTER :: max(i64)

// The span of `tz` that holds UTC second `t`
@(private)
_zone_span :: proc(tz: Time_Zone, t: i64) -> _Span {
	if tz.id == -1 {return {_NEVER_BEFORE, _NEVER_AFTER, tz.offset}}
	if tz.id <= 0 || int(tz.id) > len(_db.names) {return {_NEVER_BEFORE, _NEVER_AFTER, 0}}
	rec := int(_db.records[_db.zones[tz.id - 1]])
	n_types := int(TZ_DATA[rec])
	types := rec + 1
	type_offset := proc(types, k: int) -> i32 {return i32(_u32(types + 5 * k))}
	p := types + 5 * n_types
	initial := int(TZ_DATA[p])
	n := int(_u32(p + 1))
	times := p + 5
	idx := times + 8 * n
	rule := idx + n

	// transitions at or before t
	lo, hi := 0, n
	for lo < hi {
		mid := (lo + hi) / 2
		if _i64(times + 8 * mid) <= t {lo = mid + 1} else {hi = mid}
	}
	if lo < n || TZ_DATA[rule] == 0 {
		s := _Span{_NEVER_BEFORE, _NEVER_AFTER, type_offset(types, initial)}
		if lo > 0 {
			s.start = _i64(times + 8 * (lo - 1))
			s.offset = type_offset(types, int(TZ_DATA[idx + lo - 1]))
		}
		if lo < n {s.next = _i64(times + 8 * lo)}
		return s
	}
	last := _i64(times + 8 * (n - 1)) if n > 0 else _NEVER_BEFORE
	std := i32(_u32(rule + 1))
	if TZ_DATA[rule] == 1 {return {last, _NEVER_AFTER, std}}
	dst := i32(_u32(rule + 5))
	// the rule's switches in the years around t, in order
	year := i64(civil_from_days(_floor_div64(t, 86400)).year)
	sw: [6]struct {
		at:     i64,
		offset: i32,
	}
	for k in 0 ..< 3 {
		y := year - 1 + i64(k)
		on := _rule_local(y, rule + 9) - i64(std)
		off := _rule_local(y, rule + 16) - i64(dst)
		a, b := 2 * k, 2 * k + 1
		if on < off {
			sw[a], sw[b] = {on, dst}, {off, std}
		} else {
			sw[a], sw[b] = {off, std}, {on, dst}
		}
	}
	s := _Span{last, _NEVER_AFTER, std}
	for x in sw {
		if x.at <= t {
			s.offset = x.offset
			if x.at > s.start {s.start = x.at}
		} else {
			s.next = x.at
			break
		}
	}
	return s
}

// Local seconds (since 1970) of a POSIX rule date "Mm.w.d/time" in `year`
@(private = "file")
_rule_local :: proc(year: i64, p: int) -> i64 {
	m := i64(TZ_DATA[p])
	w := i64(TZ_DATA[p + 1])
	wd := i64(TZ_DATA[p + 2])
	secs := i64(i32(_u32(p + 3)))
	first := days_from_civil(year, m, 1)
	first_wd := (first + 4) %% 7 // 1970-01-01 was a Thursday; Sunday = 0
	day := 1 + (wd - first_wd + 7) % 7 + (w - 1) * 7
	if w == 5 {
		for day > i64(get_days_in_month(i32(year), Months(m))) {day -= 7}
	}
	return (first + day - 1) * 86400 + secs
}

@(private)
_floor_div64 :: proc(a, b: i64) -> i64 {
	q := a / b
	if (a % b != 0) && ((a < 0) != (b < 0)) {q -= 1}
	return q
}

// Offsets of one zone for many instants in a row: remembers the last span,
// so that sorted times cost one lookup per change of offset
@(private)
_Zone_Cursor :: struct {
	tz:   Time_Zone,
	span: _Span,
	set:  bool,
}

@(private)
_cursor_offset :: proc(c: ^_Zone_Cursor, t: i64) -> i32 {
	if !c.set || t < c.span.start || t >= c.span.next {
		c.span = _zone_span(c.tz, t)
		c.set = true
	}
	return c.span.offset
}

// The zone's offset from UTC at the instant `utc` (summer time included), as
// a Duration: -5 * HOURS for New York in winter
utc_offset :: proc(tz: Time_Zone, utc: Datetime) -> Duration {
	return Duration(_zone_span(tz, datetime_to_unix(utc)).offset) * SECONDS
}

// The local wall-clock time in `tz` of the instant `utc`
datetime_to_local :: proc(utc: Datetime, tz: Time_Zone) -> Datetime {
	c := _Zone_Cursor {
		tz = tz,
	}
	return _to_local(&c, utc)
}

@(private)
_to_local :: proc(c: ^_Zone_Cursor, utc: Datetime) -> Datetime {
	if c.tz.id == 0 {return utc}
	t := datetime_to_unix(utc)
	out := datetime_from_unix(t + i64(_cursor_offset(c, t)))
	out.nanosecond = utc.nanosecond
	return out
}

// The instant (in UTC) of the wall-clock time `local` in `tz`. A time that
// happens twice or never is settled by `ambiguous` / `nonexistent`; ok =
// false for Raise or Missing.
datetime_from_local :: proc(
	local: Datetime,
	tz: Time_Zone,
	ambiguous := Ambiguous_Time.Raise,
	nonexistent := Nonexistent_Time.Raise,
) -> (
	Datetime,
	bool,
) {
	c := _Zone_Cursor {
		tz = tz,
	}
	return _from_local(&c, local, ambiguous, nonexistent)
}

@(private)
_from_local :: proc(
	c: ^_Zone_Cursor,
	local: Datetime,
	ambiguous: Ambiguous_Time,
	nonexistent: Nonexistent_Time,
) -> (
	Datetime,
	bool,
) {
	if c.tz.id == 0 {return local, true}
	l := datetime_to_unix(local)
	// the usual case: no switch of offset within three days, so l - offset
	// is the only answer
	o := i64(_cursor_offset(c, l))
	x := l - o
	if x >= c.span.start + 3 * 86400 && x < c.span.next - 3 * 86400 && _ok_span(c, x, o) {
		return _with_ns(x, local.nanosecond), true
	}
	// every instant t with t + offset(t) = l lies within 26 hours of l: walk
	// the spans there and keep those that hold one
	found: [4]i64
	n := 0
	gap_at: i64 = _NEVER_AFTER
	t := l - 26 * 3600
	for t <= l + 26 * 3600 {
		s := _zone_span(c.tz, t)
		cand := l - i64(s.offset)
		if cand >= s.start && cand < s.next && n < len(found) {
			if n == 0 || found[n - 1] != cand {
				found[n] = cand
				n += 1
			}
		}
		// a switch forward that l falls into: l is past the wall clock
		// before the switch and short of the wall clock after it
		if s.next != _NEVER_AFTER && s.next <= l + 26 * 3600 {
			after := _zone_span(c.tz, s.next)
			if after.offset > s.offset &&
			   l >= s.next + i64(s.offset) &&
			   l < s.next + i64(after.offset) {
				gap_at = s.next
			}
		}
		if s.next == _NEVER_AFTER {break}
		t = s.next
	}
	switch {
	case n == 1:
		return _with_ns(found[0], local.nanosecond), true
	case n >= 2:
		switch ambiguous {
		case .Earliest:
			return _with_ns(found[0], local.nanosecond), true
		case .Latest:
			return _with_ns(found[n - 1], local.nanosecond), true
		case .Raise, .Missing:
		}
		return {}, false
	}
	if gap_at == _NEVER_AFTER {return {}, false}
	switch nonexistent {
	case .Shift_Forward:
		return datetime_from_unix(gap_at), true
	case .Shift_Backward:
		return _with_ns(gap_at - 1, 999_999_999), true
	case .Raise, .Missing:
	}
	return {}, false
}

@(private = "file")
_ok_span :: proc(c: ^_Zone_Cursor, x: i64, o: i64) -> bool {
	return x >= c.span.start && x < c.span.next && i64(c.span.offset) == o
}

@(private = "file")
_with_ns :: proc(secs: i64, ns: i32) -> Datetime {
	dt := datetime_from_unix(secs)
	dt.nanosecond = ns
	return dt
}

// ---------------------------------------------------------------------------
// Text

// "YYYY-MM-DD HH:MM:SS[.fff]+HH:MM": the local time in `tz` of the instant
// `utc` with its UTC offset, as pandas writes it ("2024-03-10
// 03:30:00-04:00"; "+00:00" for UTC)
datetime_zoned_string :: proc(
	utc: Datetime,
	tz: Time_Zone,
	allocator := context.allocator,
) -> string {
	c := _Zone_Cursor {
		tz = tz,
	}
	return _zoned_string(&c, utc, allocator)
}

@(private)
_zoned_string :: proc(c: ^_Zone_Cursor, utc: Datetime, allocator := context.allocator) -> string {
	t := datetime_to_unix(utc)
	off := int(_cursor_offset(c, t))
	local := datetime_from_unix(t + i64(off))
	local.nanosecond = utc.nanosecond
	buf: [48]u8
	s := datetime_format_iso_into(local, buf[:])
	n := len(s)
	a := off if off >= 0 else -off
	buf[n] = '+' if off >= 0 else '-'
	two := proc(buf: []u8, at, v: int) {
		buf[at] = u8('0' + v / 10)
		buf[at + 1] = u8('0' + v % 10)
	}
	two(buf[:], n + 1, a / 3600)
	buf[n + 3] = ':'
	two(buf[:], n + 4, (a / 60) % 60)
	n += 6
	if a % 60 != 0 {
		buf[n] = ':'
		two(buf[:], n + 1, a % 60)
		n += 3
	}
	return strings.clone(string(buf[:n]), allocator)
}

// A date and time with a UTC offset: "2024-03-10 03:30:00-04:00",
// "2024-03-10T07:30:00Z", "...+0530", "...+05" (the time as parse_datetime
// reads it). Returns the instant in UTC and the offset in seconds; ok =
// false without an offset or for anything else.
parse_datetime_offset :: proc(text: string) -> (utc: Datetime, offset: int, ok: bool) {
	sep := strings.index_byte(text, ' ')
	if sep < 0 {sep = strings.index_byte(text, 'T')}
	if sep < 0 {return}
	cut := -1
	for i := len(text) - 1; i > sep; i -= 1 {
		c := text[i]
		if c == '+' || c == '-' || ((c == 'Z' || c == 'z') && i == len(text) - 1) {
			cut = i
			break
		}
	}
	if cut < 0 {return}
	local, lok := parse_datetime(text[:cut])
	if !lok {return}
	if text[cut] == 'Z' || text[cut] == 'z' {
		return local, 0, true
	}
	s := text[cut + 1:]
	d: [6]int
	n := 0
	for i := 0; i < len(s); i += 1 {
		c := s[i]
		if c == ':' && (i == 2 || i == 5) && n == i - i / 3 {continue}
		if c < '0' || c > '9' || n == 6 {return}
		d[n] = int(c - '0')
		n += 1
	}
	if n != 2 && n != 4 && n != 6 {return}
	secs := (d[0] * 10 + d[1]) * 3600 + (d[2] * 10 + d[3]) * 60 + d[4] * 10 + d[5]
	if d[2] * 10 + d[3] >= 60 || d[4] * 10 + d[5] >= 60 || secs > 86400 {return}
	if text[cut] == '-' {secs = -secs}
	out := datetime_from_unix(datetime_to_unix(local) - i64(secs))
	out.nanosecond = local.nanosecond
	return out, secs, true
}

// ---------------------------------------------------------------------------
// Columns

// The wall-clock times of a Datetime column, placed in the zone `zone`
// (pandas' tz_localize): the result keeps the instants in UTC and has
// tz = the zone. Times that happen twice or never are settled by `ambiguous`
// and `nonexistent` (Missing: a missing value). zone = "" does the opposite:
// a zoned column gives its local wall-clock times without a zone
// (tz_localize(None)). ok = false for another column type, an unknown zone,
// a column that already has a zone (use tz_convert), or a time that the
// Raise choices refuse.
tz_localize :: proc(
	col: ^Column,
	zone: string,
	ambiguous := Ambiguous_Time.Raise,
	nonexistent := Nonexistent_Time.Raise,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .Datetime {return {}, false}
	if zone == "" {return to_local_time(col, name, allocator)}
	if col.tz.id != 0 {return {}, false}
	tz, ok := time_zone(zone)
	if !ok {return {}, false}
	out := _new_like(col, .Datetime, name, allocator)
	out.tz = tz
	c := _Zone_Cursor {
		tz = tz,
	}
	for i in 0 ..< col.len {
		v, missing := column_at_datetime(col, i)
		if missing {
			append_null(&out)
			continue
		}
		if !_append_localized(&out, &c, v, ambiguous, nonexistent) {
			destroy_column(&out)
			return {}, false
		}
	}
	return out, true
}

// Appends the instant of wall-clock time `v` (a missing value when the
// choice for it is Missing); false when a Raise choice refuses it
@(private)
_append_localized :: proc(
	out: ^Column,
	c: ^_Zone_Cursor,
	v: Datetime,
	ambiguous: Ambiguous_Time,
	nonexistent: Nonexistent_Time,
) -> bool {
	u, ok := _from_local(c, v, ambiguous, nonexistent)
	if ok {
		append_datetime(out, u)
		return true
	}
	// which choice refused it: Missing gives a missing value
	_, twice := _from_local(c, v, .Earliest, .Raise)
	if twice && ambiguous == .Missing || !twice && nonexistent == .Missing {
		append_null(out)
		return true
	}
	return false
}

// A zoned Datetime column shown in another zone (pandas' tz_convert): the
// same instants, tz = `zone`. ok = false for a column without a zone (give
// it one with tz_localize), another type or an unknown zone.
tz_convert :: proc(
	col: ^Column,
	zone: string,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .Datetime || col.tz.id == 0 {return {}, false}
	tz, ok := time_zone(zone)
	if !ok {return {}, false}
	out := column_clone(col, allocator)
	_rename(&out, name)
	out.tz = tz
	return out, true
}

// The local wall-clock times of a zoned Datetime column, without a zone
// (a copy for a column without one). ok = false for another column type.
to_local_time :: proc(col: ^Column, name := "", allocator := context.allocator) -> (Column, bool) {
	if col.type != .Datetime {return {}, false}
	out := _new_like(col, .Datetime, name, allocator)
	out.tz = {}
	c := _Zone_Cursor {
		tz = col.tz,
	}
	for i in 0 ..< col.len {
		v, missing := column_at_datetime(col, i)
		if missing {append_null(&out)} else {append_datetime(&out, _to_local(&c, v))}
	}
	return out, true
}
