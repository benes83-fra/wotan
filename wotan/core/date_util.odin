package wotan


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
	days_in_month := get_days_in_month(year, Months(month))

	if month < 1 || month > 12 || day < 0 || day > days_in_month {
		return false
	}
	return true
}
validate_time :: proc(hour: i32, minute: i32, second: i32) -> bool {

	if hour < 0 || hour > 24 || minute < 0 || minute > 60 || second < 0 || second > 60 {
		return false
	}
	return true
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
	parts := strings.split(datetime_str, " ")
	defer delete(parts)
	if len(parts) != 2 {
		return Datetime{}, false
	}
	dparts := strings.split(parts[0], "-")
	defer delete(dparts)
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


date_to_string :: proc(d: Date) -> string {
	// Use a string builder for efficiency
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)

	// Format with zero-padding for month/day
	fmt.sbprintf(&sb, "%04d-%02d-%02d", d.year, d.month, d.day)

	return strings.to_string(sb) // Convert builder to string
}
time_to_string :: proc(t: Time) -> string {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)

	fmt.sbprintf(&sb, "%02d:%02d%02d", t.hour, t.minute, t.second)

	return strings.to_string(sb)

}

datetime_to_string :: proc(dt: Datetime) -> string {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)

	fmt.sbprintf(
		&sb,
		"%%04d-%02d-%02d 02d:%02d%02d",
		dt.year,
		dt.month,
		dt.day,
		dt.hour,
		dt.minute,
		dt.second,
	)

	return strings.to_string(sb)

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

// Convert f64 (Julian Day Number) → Date
f64_to_date :: proc(jd: f64) -> Date {
	jd := jd
	jd += 0.5
	z := math.floor_f64(jd)
	f := jd - z

	a := z
	if z >= 2299161 {
		alpha := f64((z - 1867216.25) / 36524.25)
		a += 1 + alpha - alpha / 4
	}

	b: f64 = a + f64(1524)
	c := cast(i64)((f64(b) - f64(122.1)) / f64(365.25))
	d := cast(f64)(365.25 * cast(f64)c)
	e := cast(i64)(f64(b - d) / 30.6001)

	day := f64(b - d - (f64(30.6001) * f64(e)) + f)
	month := e < 14 ? e - 1 : e - 13
	year := month > 2 ? c - 4716 : c - 4715

	return Date{year = cast(i32)year, month = cast(i32)month, day = cast(i32)math.floor_f64(day)}
}


time_to_int :: proc(t: Time) -> i32 {
	return t.hour * HOUR + t.minute * MINUTE + t.second
}

int_to_time :: proc(s: i32) -> Time {
	s := s
	if s < 0 {
		s = 0
	}
	s = s % DAY
	hour := s / HOUR
	minute := (s % HOUR) / MINUTE
	second := s % MINUTE
	return Time{hour, minute, second}
}
