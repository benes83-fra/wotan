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
	return a.second - b.second
}
datetime_compare :: proc(a, b: Datetime) -> i32 {
	if a.year != b.year {return a.year - b.year}
	if a.month != b.month {return a.month - b.month}
	if a.day != b.day {return a.day - b.day}
	if a.hour != b.hour {return a.hour - b.hour}
	if a.minute != b.minute {return a.minute - b.minute}
	return a.second - b.second
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
	return Datetime{d.year, d.month, d.day, i32(secs / 3600), i32((secs % 3600) / 60), i32(secs % 60)}
}

parse_date :: proc(date_str: string) -> (Date, bool) {
	parts := strings.split(date_str, "-")
	defer delete(parts)
	if len(parts) != 3 {
		return Date{}, false // Invalid format
	}

	year, ok1 := strconv.parse_int(parts[0])
	month, ok2 := strconv.parse_int(parts[1])
	day, ok3 := strconv.parse_int(parts[2])

	if !ok1 || !ok2 || !ok3 {
		return Date{}, false // Non-numeric values
	}

	// Basic validation
	if !validate_date(i32(year), i32(month), i32(day)) {
		return Date{}, false
	}

	return Date{i32(year), i32(month), i32(day)}, true
}
parse_time :: proc(time_str: string) -> (Time, bool) {
	parts := strings.split(time_str, ":")
	defer delete(parts)
	if len(parts) != 3 {
		return Time{}, false
	}
	hour, ok1 := strconv.parse_int(parts[0])
	minute, ok2 := strconv.parse_int(parts[1])
	second, ok3 := strconv.parse_int(parts[2])
	if !ok1 || !ok2 || !ok3 {
		return Time{}, false
	}
	if !validate_time(i32(hour), i32(minute), i32(second)) {

		return Time{}, false
	}
	return Time{i32(hour), i32(minute), i32(second)}, true

}

parse_datetime :: proc(datetime_str: string) -> (Datetime, bool) {
	// "YYYY-MM-DD HH:MM:SS" or ISO 8601 "YYYY-MM-DDTHH:MM:SS"
	sep := strings.index_byte(datetime_str, ' ')
	if sep < 0 {sep = strings.index_byte(datetime_str, 'T')}
	if sep < 0 {
		return Datetime{}, false
	}
	parts := [2]string{datetime_str[:sep], datetime_str[sep + 1:]}
	dparts := strings.split(parts[0], "-")
	defer delete(dparts)
	if len(dparts) != 3 {
		return Datetime{}, false
	}
	year, ok1 := strconv.parse_int(dparts[0])
	month, ok2 := strconv.parse_int(dparts[1])
	day, ok3 := strconv.parse_int(dparts[2])
	if !ok1 || !ok2 || !ok3 {
		return Datetime{}, false
	}
	if !validate_date(i32(year), i32(month), i32(day)) {
		return Datetime{}, false
	}
	tparts := strings.split(parts[1], ":")
	defer delete(tparts)
	if len(tparts) != 3 {
		return Datetime{}, false
	}
	hour, ok4 := strconv.parse_int(tparts[0])
	minute, ok5 := strconv.parse_int(tparts[1])
	second, ok6 := strconv.parse_int(tparts[2])
	if !ok4 || !ok5 || !ok6 {
		return Datetime{}, false
	}
	if !validate_time(i32(hour), i32(minute), i32(second)) {
		return Datetime{}, false
	}
	dt := Datetime{i32(year), i32(month), i32(day), i32(hour), i32(minute), i32(second)}
	return dt, true
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

time_to_string :: proc(t: Time, allocator := context.allocator) -> string {
	return fmt.aprintf("%02d:%02d:%02d", t.hour, t.minute, t.second, allocator = allocator)
}

datetime_to_string :: proc(dt: Datetime, allocator := context.allocator) -> string {
	return fmt.aprintf(
		"%04d-%02d-%02d %02d:%02d:%02d",
		dt.year,
		dt.month,
		dt.day,
		dt.hour,
		dt.minute,
		dt.second,
		allocator = allocator,
	)
}
datetime_to_string_na :: proc(dt: Datetime) -> string {
	// 19 chars for "YYYY-MM-DD HH:MM:SS" + 1 for safety if needed
	buf: [24]byte

	// bprintf formats directly into the slice and returns the string
	return fmt.bprintf(
		buf[:],
		"%04d-%02d-%02d %02d:%02d:%02d",
		dt.year,
		dt.month,
		dt.day,
		dt.hour,
		dt.minute,
		dt.second,
	)

}
datetime_format_iso_into :: proc(dt: Datetime, buf: []u8) -> string {
	// Require at least 19 bytes: "YYYY-MM-DD HH:MM:SS"
	assert(len(buf) >= 19)

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
	return Time{hour, minute, second}
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
