package core

import "core:fmt"
import "core:math"
import "core:strconv"
import "core:strings"

date_compare :: proc(a, b: Date) -> i32 {
	if a.year != b.year {return a.year - b.year}
	if a.month != b.month {return a.month - b.month}
	return a.day - b.day

}
time_compare :: proc(a, b: Time) -> i32 {
	if a.hour != b.hour {return a.hour - b.hour}
	if a.minute != b.minute {return a.minute - b.minute}
	if a.second != b.second {return a.second - b.second}
	return a.nanosecond - b.nanosecond
}
datetime_compare :: proc(a, b: Datetime) -> i32 {
	if a.year != b.year {return a.year - b.year}
	if a.month != b.month {return a.month - b.month}
	if a.day != b.day {return a.day - b.day}
	if a.hour != b.hour {return a.hour - b.hour}
	if a.minute != b.minute {return a.minute - b.minute}
	if a.second != b.second {return a.second - b.second}
	return a.nanosecond - b.nanosecond
}

validate_date :: proc(year: i32, month: i32, day: i32) -> bool {
	if month < 1 || month > 12 || day < 1 {
		return false
	}
	return day <= get_days_in_month(year, Months(month))
}
validate_time :: proc(hour: i32, minute: i32, second: i32) -> bool {
	return hour >= 0 && hour <= 23 && minute >= 0 && minute <= 59 && second >= 0 && second <= 59
}

// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's
// days_from_civil; exact for every year)
days_from_civil :: proc(y, m, d: i64) -> i64 {
	y := y - (1 if m <= 2 else 0)
	era := (y if y >= 0 else y - 399) / 400
	yoe := y - era * 400
	mp := (m + 9) % 12
	doy := (153 * mp + 2) / 5 + d - 1
	doe := yoe * 365 + yoe / 4 - yoe / 100 + doy
	return era * 146097 + doe - 719468
}

// Inverse of days_from_civil
civil_from_days :: proc(z: i64) -> Date {
	z := z + 719468
	era := (z if z >= 0 else z - 146096) / 146097
	doe := z - era * 146097
	yoe := (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
	doy := doe - (365 * yoe + yoe / 4 - yoe / 100)
	mp := (5 * doy + 2) / 153
	d := doy - (153 * mp + 2) / 5 + 1
	m := mp + 3 if mp < 10 else mp - 9
	y := yoe + era * 400 + (1 if m <= 2 else 0)
	return Date{i32(y), i32(m), i32(d)}
}

@(private = "file")
_floor_div :: proc(a, b: i64) -> i64 {
	q := a / b
	if (a % b != 0) && ((a < 0) != (b < 0)) {q -= 1}
	return q
}

// Datetime of a Unix timestamp (seconds since 1970-01-01 00:00:00 UTC;
// negative timestamps are before 1970)
datetime_from_unix :: proc(ts: i64) -> Datetime {
	days := _floor_div(ts, 86400)
	secs := ts - days * 86400
	d := civil_from_days(days)
	return Datetime {
		d.year,
		d.month,
		d.day,
		i32(secs / 3600),
		i32((secs % 3600) / 60),
		i32(secs % 60),
		0,
	}
}

// Datetime of a Unix time in nanoseconds (negative before 1970)
datetime_from_unix_nanos :: proc(ns: i64) -> Datetime {
	secs := _floor_div(ns, 1_000_000_000)
	dt := datetime_from_unix(secs)
	dt.nanosecond = i32(ns - secs * 1_000_000_000)
	return dt
}

// Seconds with an optional fraction ("SS", "SS.f" .. "SS.fffffffff")
@(private = "file")
_parse_seconds :: proc(s: string) -> (second: int, nanosecond: i32, ok: bool) {
	whole := s
	frac := ""
	if dot := strings.index_byte(s, '.'); dot >= 0 {
		whole, frac = s[:dot], s[dot + 1:]
		if len(frac) == 0 || len(frac) > 9 {return}
	}
	sec := _digits(whole) or_return
	second = int(sec)
	ns := 0
	for c in transmute([]u8)frac {
		if c < '0' || c > '9' {return}
		ns = ns * 10 + int(c - '0')
	}
	for _ in len(frac) ..< 9 {ns *= 10}
	return second, i32(ns), true
}

// The fraction of a second as text: "" for none, else 3, 6 or 9 digits
// (milli-, micro- or nanoseconds), whichever shows it exactly
fraction_string :: proc(nanosecond: i32, buf: ^[10]u8) -> string {
	if nanosecond <= 0 {return ""}
	digits := 9
	v := int(nanosecond)
	if v % 1_000_000 == 0 {
		digits, v = 3, v / 1_000_000
	} else if v % 1000 == 0 {
		digits, v = 6, v / 1000
	}
	buf[0] = '.'
	for k := digits; k >= 1; k -= 1 {
		buf[k] = u8('0' + v % 10)
		v /= 10
	}
	return string(buf[:digits + 1])
}

// Digits only (1 to 9 of them) as a number
@(private = "file")
_digits :: #force_inline proc(s: string) -> (v: i32, ok: bool) {
	if len(s) == 0 || len(s) > 9 {return 0, false}
	for c in transmute([]u8)s {
		if c < '0' || c > '9' {return 0, false}
		v = v * 10 + i32(c - '0')
	}
	return v, true
}

// "YYYY-MM-DD" (digits only; month and day need not be zero-padded), a valid
// calendar date. No allocation.
parse_date :: proc(date_str: string) -> (Date, bool) {
	s := date_str
	// the usual zero-padded form, read directly
	if len(s) == 10 && s[4] == '-' && s[7] == '-' {
		year, ok1 := _digits(s[:4])
		month, ok2 := _digits(s[5:7])
		day, ok3 := _digits(s[8:])
		if ok1 &&
		   ok2 &&
		   ok3 &&
		   validate_date(year, month, day) {return Date{year, month, day}, true}
		return {}, false
	}
	d1 := strings.index_byte(s, '-')
	if d1 <= 0 {return {}, false}
	d2 := strings.index_byte(s[d1 + 1:], '-')
	if d2 <= 0 {return {}, false}
	d2 += d1 + 1
	year, ok1 := _digits(s[:d1])
	month, ok2 := _digits(s[d1 + 1:d2])
	day, ok3 := _digits(s[d2 + 1:])
	if !ok1 || !ok2 || !ok3 || !validate_date(year, month, day) {return {}, false}
	return Date{year, month, day}, true
}

// "HH:MM:SS" with an optional fraction of 1 to 9 digits ("HH:MM:SS.fff");
// 00:00:00 ..= 23:59:59.999999999. No allocation.
parse_time :: proc(time_str: string) -> (Time, bool) {
	s := time_str
	// the usual zero-padded form, read directly
	if len(s) == 8 && s[2] == ':' && s[5] == ':' {
		hour, ok1 := _digits(s[:2])
		minute, ok2 := _digits(s[3:5])
		second, ok3 := _digits(s[6:])
		if ok1 &&
		   ok2 &&
		   ok3 &&
		   validate_time(hour, minute, second) {return Time{hour, minute, second, 0}, true}
		return {}, false
	}
	c1 := strings.index_byte(s, ':')
	if c1 <= 0 {return {}, false}
	c2 := strings.index_byte(s[c1 + 1:], ':')
	if c2 <= 0 {return {}, false}
	c2 += c1 + 1
	hour, ok1 := _digits(s[:c1])
	minute, ok2 := _digits(s[c1 + 1:c2])
	second, ns, ok3 := _parse_seconds(s[c2 + 1:])
	if !ok1 || !ok2 || !ok3 || !validate_time(hour, minute, i32(second)) {return {}, false}
	return Time{hour, minute, i32(second), ns}, true
}

// "YYYY-MM-DD HH:MM:SS" or ISO 8601 "YYYY-MM-DDTHH:MM:SS", the seconds with
// an optional fraction of up to 9 digits. No allocation.
parse_datetime :: proc(datetime_str: string) -> (Datetime, bool) {
	sep := strings.index_byte(datetime_str, ' ')
	if sep < 0 {sep = strings.index_byte(datetime_str, 'T')}
	if sep < 0 {return {}, false}
	d, dok := parse_date(datetime_str[:sep])
	t, tok := parse_time(datetime_str[sep + 1:])
	if !dok || !tok {return {}, false}
	return Datetime{d.year, d.month, d.day, t.hour, t.minute, t.second, t.nanosecond}, true
}


// date_to_string :: proc(d: Date) -> string {
// 	// Use a string builder for efficiency
// 	sb := strings.builder_make()


// 	// Format with zero-padding for month/day
// 	fmt.sbprintf(&sb, "%04d-%02d-%02d", d.year, d.month, d.day)

// 	return strings.to_string(sb) // Convert builder to string
// }
// time_to_string :: proc(t: Time) -> string {
// 	sb := strings.builder_make()
// 	defer strings.builder_destroy(&sb)

// 	fmt.sbprintf(&sb, "%02d:%02d:%02d", t.hour, t.minute, t.second)

// 	return strings.to_string(sb)

// }
// datetime_to_string :: proc(dt: Datetime) -> string {
// 	sb := strings.builder_make()
// 	defer strings.builder_destroy(&sb)

// 	fmt.sbprintf(
// 		&sb,
// 		"%04d-%02d-%02d %02d:%02d:%02d",
// 		dt.year,
// 		dt.month,
// 		dt.day,
// 		dt.hour,
// 		dt.minute,
// 		dt.second,
// 	)

// 	return strings.to_string(sb)

// }
date_to_string :: proc(d: Date, allocator := context.allocator) -> string {
	return fmt.aprintf("%04d-%02d-%02d", d.year, d.month, d.day, allocator = allocator)
}

// "HH:MM:SS", with ".fff", ".ffffff" or ".fffffffff" when there is a
// fraction of a second (the fewest digits that show it exactly)
time_to_string :: proc(t: Time, allocator := context.allocator) -> string {
	fb: [10]u8
	return fmt.aprintf(
		"%02d:%02d:%02d%s",
		t.hour,
		t.minute,
		t.second,
		fraction_string(t.nanosecond, &fb),
		allocator = allocator,
	)
}

// "YYYY-MM-DD HH:MM:SS", with a fraction of a second as time_to_string
datetime_to_string :: proc(dt: Datetime, allocator := context.allocator) -> string {
	fb: [10]u8
	return fmt.aprintf(
		"%04d-%02d-%02d %02d:%02d:%02d%s",
		dt.year,
		dt.month,
		dt.day,
		dt.hour,
		dt.minute,
		dt.second,
		fraction_string(dt.nanosecond, &fb),
		allocator = allocator,
	)
}

// datetime_to_string in the temp allocator
datetime_to_string_na :: proc(dt: Datetime) -> string {
	return datetime_to_string(dt, context.temp_allocator)
}

// "YYYY-MM-DD HH:MM:SS[.fff…]" into `buf` (at least 29 bytes), without
// allocating
datetime_format_iso_into :: proc(dt: Datetime, buf: []u8) -> string {
	assert(len(buf) >= 29)

	i := 0

	// Year
	buf[i] = u8('0' + dt.year / 1000); i += 1
	buf[i] = u8('0' + (dt.year / 100) % 10); i += 1
	buf[i] = u8('0' + (dt.year / 10) % 10); i += 1
	buf[i] = u8('0' + dt.year % 10); i += 1

	buf[i] = '-'; i += 1

	// Month
	buf[i] = u8('0' + dt.month / 10); i += 1
	buf[i] = u8('0' + dt.month % 10); i += 1

	buf[i] = '-'; i += 1

	// Day
	buf[i] = u8('0' + dt.day / 10); i += 1
	buf[i] = u8('0' + dt.day % 10); i += 1

	buf[i] = ' '; i += 1

	// Hour
	buf[i] = u8('0' + dt.hour / 10); i += 1
	buf[i] = u8('0' + dt.hour % 10); i += 1

	buf[i] = ':'; i += 1

	// Minute
	buf[i] = u8('0' + dt.minute / 10); i += 1
	buf[i] = u8('0' + dt.minute % 10); i += 1

	buf[i] = ':'; i += 1

	// Second
	buf[i] = u8('0' + dt.second / 10); i += 1
	buf[i] = u8('0' + dt.second % 10); i += 1

	// Fraction
	fb: [10]u8
	for c in transmute([]u8)fraction_string(dt.nanosecond, &fb) {
		buf[i] = c; i += 1
	}

	// No allocation here: just a view into caller's buffer
	return string(buf[:i])
}


date_to_int :: proc(d: Date) -> i32 {
	// Validate ranges
	if d.month < 1 || d.month > 12 {
		fmt.println("Error: month out of range (1-12)")
		return -1
	}
	if d.day < 1 || d.day > 31 {
		fmt.println("Error: day out of range (1-31)")
		return -1
	}
	// Combine into YYYYMMDD
	return d.year * 10000 + d.month * 100 + d.day
}

int_to_date :: proc(date_int: i32) -> Date {
	year := date_int / 10000
	month := (date_int / 100) % 100
	day := date_int % 100
	return Date{year, month, day}
}


// Convert Date → f64 (Julian Day Number)
date_to_f64 :: proc(d: Date) -> f64 {
	// Adjust months so that March is month 1
	y := d.year
	m := d.month
	if m <= 2 {
		y -= 1
		m += 12
	}

	// Julian Day Number calculation (Gregorian calendar)
	a := y / 100
	b := 2 - a + a / 4
	jd :=
		math.floor_f64(365.2500000 * cast(f64)(y + 4716)) +
		math.floor_f64(30.60010000 * cast(f64)(m + 1)) +
		cast(f64)d.day +
		cast(f64)b -
		1524.5

	return jd
}

// Convert f64 (Julian Day Number) → Date (inverse of date_to_f64; a
// fractional day gives the date it falls on)
f64_to_date :: proc(jd: f64) -> Date {
	// JD 2440587.5 is 1970-01-01 00:00 UTC
	return civil_from_days(i64(math.floor_f64(jd - 2440587.5 + 1e-9)))
}

// Whole seconds since midnight (the nanoseconds are left out)
time_to_int :: proc(t: Time) -> i32 {
	return t.hour * HOUR + t.minute * MINUTE + t.second
}

// Seconds → time of day (wraps around midnight in both directions)
int_to_time :: proc(s: i32) -> Time {
	s := s % DAY
	if s < 0 {
		s += DAY
	}
	hour := s / HOUR
	minute := (s % HOUR) / MINUTE
	second := s % MINUTE
	return Time{hour, minute, second, 0}
}
date_less :: proc(a, b: Date) -> bool {
	return date_compare(a, b) < 0
}

time_less :: proc(a, b: Time) -> bool {
	return time_compare(a, b) < 0
}

datetime_less :: proc(a, b: Datetime) -> bool {
	return datetime_compare(a, b) < 0
}
date_from_unix :: proc(ts: i64) -> Date {
	// Unix seconds → calendar date (UTC); negative timestamps are before 1970
	return civil_from_days(_floor_div(ts, 86400))
}
