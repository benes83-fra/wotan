package core

import "core:fmt"
import "core:math"
import t "core:time"

Days :: enum {
	Monday,
	Tuesday,
	Wednesday,
	Thursday,
	Friday,
	Saturday,
	Sunday,
}

Months :: enum {
	January   = 1,
	February  = 2,
	March     = 3,
	April     = 4,
	May       = 5,
	June      = 6,
	July      = 7,
	August    = 8,
	September = 9,
	October   = 10,
	November  = 11,
	December  = 12,
}

DAY :: 24 * 60 * 60
HOUR :: 60 * 60
MINUTE :: 60

new_Datetime_from_Date_and_Time :: proc(date: Date, time: Time) -> Datetime {
	dt := Datetime{date.year, date.month, date.day, time.hour, time.minute, time.second}
	if !validate_datetime(dt) {
		panic("datetime: datetime not valid")
	}
	return dt
}

get_Date_from_Datetime :: proc(dt: Datetime) -> Date {
	return Date{dt.year, dt.month, dt.day}
}

get_Time_from_Datetime :: proc(dt: Datetime) -> Time {
	return Time{dt.hour, dt.minute, dt.second}
}

validate_datetime :: proc(dt: Datetime) -> bool {
	year := dt.year
	month := dt.month
	day := dt.day
	hour := dt.hour
	minute := dt.minute
	second := dt.second
	valid_d := validate_date(year, month, day)
	valid_t := validate_time(hour, minute, second)
	return valid_d && valid_t

}

get_days_in_month :: proc(year: i32, month: Months) -> i32 {
	days: i32
	switch month {
	case .January:
		days = 31
	case .March:
		days = 31
	case .May:
		days = 31
	case .July:
		days = 31
	case .August:
		days = 31
	case .October:
		days = 31
	case .December:
		days = 31
	case .April:
		days = 30
	case .June:
		days = 30
	case .September:
		days = 30
	case .November:
		days = 30
	case .February:
		days = get_february_days(year)

	}
	return days
}

get_february_days :: proc(year: i32) -> i32 {
	if (year % 400 == 0) || (year % 4 == 0 && year % 100 != 0) {
		return 29
	}
	return 28
}

get_days_of_the_year :: proc(year: i32) -> i32 {

	if (year % 400 == 0) || (year % 4 == 0 && year % 100 != 0) {
		return 366
	}
	return 365
}
add_day_date :: proc(date: Date, n: i32) -> Date {
	return civil_from_days(
		days_from_civil(i64(date.year), i64(date.month), i64(date.day)) + i64(n),
	)
}
add_day_datetime :: proc(dt: Datetime, n: i32) -> Datetime {
	year := dt.year
	month := dt.month
	day := dt.day
	hour := dt.hour
	minute := dt.minute
	second := dt.second
	new_date := add_day_date(Date{year, month, day}, n)
	return Datetime{new_date.year, new_date.month, new_date.day, hour, minute, second}
}


// Calendar months later (earlier for n < 0); the day is clamped to the
// length of the target month (2024-01-31 + 1 month = 2024-02-29)
add_month_date :: proc(date: Date, n: i32) -> Date {
	total_months := date.year * 12 + date.month - 1 + n
	new_year := math.floor_div(total_months, 12)
	new_month := total_months - new_year * 12 + 1
	day := min(date.day, get_days_in_month(new_year, Months(new_month)))
	return Date{new_year, new_month, day}
}
add_month_datetime :: proc(dt: Datetime, n: i32) -> Datetime {
	year := dt.year
	month := dt.month
	day := dt.day
	hour := dt.hour
	minute := dt.minute
	second := dt.second
	new_date := add_month_date(Date{year, month, day}, n)
	return Datetime{new_date.year, new_date.month, new_date.day, hour, minute, second}
}

// Same month and day n years later; Feb 29 becomes Feb 28 in common years
add_year_date :: proc(date: Date, n: i32) -> Date {
	year := date.year + n
	day := min(date.day, get_days_in_month(year, Months(date.month)))
	return Date{year, date.month, day}
}

add_year_datetime :: proc(dt: Datetime, n: i32) -> Datetime {
	year := dt.year
	month := dt.month
	day := dt.day
	hour := dt.hour
	minute := dt.minute
	second := dt.second
	new_date := add_year_date(Date{year, month, day}, n)
	return Datetime{new_date.year, new_date.month, new_date.day, hour, minute, second}
}

// Time of day n seconds later (earlier for n < 0), wrapping around midnight
add_seconds_time :: proc(time: Time, n: i32) -> Time {
	total := i64(time.hour) * HOUR + i64(time.minute) * MINUTE + i64(time.second) + i64(n)
	total %%= DAY
	return Time{i32(total / HOUR), i32((total % HOUR) / MINUTE), i32(total % MINUTE)}
}

add_second_time :: proc(time: Time) -> Time {
	hour := time.hour
	minute := time.minute
	second := time.second
	if second < 59 {
		second = second + 1

		return Time{hour, minute, second}
	}
	if minute < 59 {

		minute = minute + 1
		second = 0

	} else {
		if hour < 23 {
			hour = hour + 1
			minute = 0
			second = 0
		} else {
			hour = 0
			minute = 0
			second = 0
		}

	}
	return Time{hour, minute, second}
}

add_minutes_time :: proc(time: Time, n: i32) -> Time {

	return add_seconds_time(time, n * MINUTE)
}

add_hour_time :: proc(time: Time, n: i32) -> Time {
	return add_seconds_time(time, n * HOUR)
}

add_minute_time :: proc(time: Time) -> Time {
	hour := time.hour
	minute := time.minute
	second := time.second

	if minute < 59 {
		minute = minute + 1
		return Time{hour, minute, second}
	}
	if hour < 23 {
		hour = hour + 1
		minute = 0

	} else {
		hour = 0
		minute = 0

	}
	return Time{hour, minute, second}
}

add_hour_datetime :: proc(datetime: Datetime) -> Datetime {

	year := datetime.year
	month := datetime.month
	day := datetime.day
	hour := datetime.hour
	minute := datetime.minute
	second := datetime.second
	time := Time{hour, minute, second}
	date := Date{year, month, day}
	if hour < 23 {
		hour = hour + 1
		return Datetime{year, month, day, hour, minute, second}
	} else {
		hour = 0
		date = add_day_date(date, 1)
		return Datetime{date.year, date.month, date.day, hour, minute, second}
	}
}

add_minute_datetime :: proc(datetime: Datetime) -> Datetime {

	year := datetime.year
	month := datetime.month
	day := datetime.day
	hour := datetime.hour
	minute := datetime.minute
	second := datetime.second
	time := Time{hour, minute, second}
	date := Date{year, month, day}
	if hour < 23 {
		time = add_minute_time(time)
		return Datetime{year, month, day, time.hour, time.minute, time.second}
	}
	if minute < 59 {

		time = add_minute_time(time)
		return Datetime{year, month, day, time.hour, time.minute, time.second}
	} else {

		date = add_day_date(date, 1)
		hour = 0
		minute = 0
		return Datetime{date.year, date.month, date.day, hour, minute, second}
	}
}

// Datetime n seconds later (earlier for n < 0), across day / month / year boundaries
add_seconds_datetime :: proc(datetime: Datetime, n: i32) -> Datetime {
	return datetime_from_unix(datetime_to_unix(datetime) + i64(n))
}

add_minutes_datetime :: proc(datetime: Datetime, n: i32) -> Datetime {

	return add_seconds_datetime(datetime, n * MINUTE)

}
add_hours_datetime :: proc(datetime: Datetime, n: i32) -> Datetime {
	return add_seconds_datetime(datetime, n * HOUR)
}


add_second_datetime :: proc(datetime: Datetime) -> Datetime {
	year := datetime.year
	month := datetime.month
	day := datetime.day
	hour := datetime.hour
	minute := datetime.minute
	second := datetime.second

	if second < 59 {
		second = second + 1
		return Datetime{year, month, day, hour, minute, second}
	}
	if minute < 59 {
		minute = minute + 1
		second = 0
	} else {
		if hour < 23 {
			hour = hour + 1
			minute = 0
			second = 0
		} else {
			date := Date{year, month, day}
			date = add_day_date(date, 1)
			year = date.year
			month = date.month
			day = date.day
			hour = 0
			minute = 0
			second = 0
		}
	}
	return Datetime{year, month, day, hour, minute, second}
}
abs_days :: proc(days: i32) -> i32 {
	if days >= 0 {
		return days
	} else {
		return -days
	}
}
get_day_diff_between_months :: proc(a, b: Date) -> i32 {

	a_year := a.year
	a_month := a.month
	a_day := a.day
	b_year := b.year
	b_month := b.month
	b_day := b.day

	month := a_month
	days: i32 = get_days_in_month(a_year, Months(month)) - a_day
	month = month + 1
	for month < b_month {
		days = days + get_days_in_month(a_year, Months(month))
		month = month + 1
	}
	days = days + b_day
	return days
}

get_remaining_days_of_year :: proc(a: Date) -> i32 {
	year := a.year
	days: i32 = 0
	days = get_day_diff_between_months(a, Date{year, i32(12), i32(31)}) + 1
	return days
}

get_day_diff_between_years :: proc(a, b: Date) -> i32 {

	a_year := a.year
	a_month := a.month
	a_day := a.day
	b_year := b.year
	b_month := b.month
	b_day := b.day
	year := a_year
	days := get_remaining_days_of_year(a)
	year = year + 1
	month: i32 = 1
	day: i32 = 1
	for year < b_year {
		date := Date{year, month, day}
		days = days + get_remaining_days_of_year(date)
		year = year + 1
	}
	date := Date{year, month, day}

	days = days + get_day_diff_between_months(date, b)
	return days
}
// Absolute number of days between two dates
get_date_day_diffs :: proc(a, b: Date) -> i32 {
	return abs_days(date_diff(a, b))
}

// Current time (UTC)
now :: proc() -> Datetime {

	now := t.now()
	buf: [20]u8
	res := t.to_string_yyyy_mm_dd(now, buf[:])
	buf2: [20]u8
	res2 := t.to_string_hms(now, buf2[:])

	date, ok := parse_date(res)
	if !ok {
		panic("Cannot parse Date string")
	}

	time, ok2 := parse_time(res2)
	if !ok2 {
		panic("Cannot parese Time string")
	}
	return Datetime{date.year, date.month, date.day, time.hour, time.minute, time.second}
}
// Returns the number of days from date a to date b (b - a)
// Positive if b is after a, negative if b is before a
date_diff :: proc(a, b: Date) -> i32 {
	a_year := a.year
	a_month := a.month
	a_day := a.day
	b_year := b.year
	b_month := b.month
	b_day := b.day

	// Convert both dates to a day count since an epoch
	days_a := date_to_days(a_year, a_month, a_day)
	days_b := date_to_days(b_year, b_month, b_day)

	return days_b - days_a
}

// Convert a date to a day count (similar to Julian Day Number)
date_to_days :: proc(year: i32, month: i32, day: i32) -> i32 {
	// Use a simplified algorithm
	y := year
	m := month

	// Adjust for months January and February
	if m <= 2 {
		y -= 1
		m += 12
	}

	// Calculate days using a formula similar to Julian Day
	days := 365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + day

	return days
}

// Convert Datetime to Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)
datetime_to_unix :: proc(dt: Datetime) -> i64 {
	days := days_from_civil(i64(dt.year), i64(dt.month), i64(dt.day))
	return days * 86400 + i64(dt.hour) * 3600 + i64(dt.minute) * 60 + i64(dt.second)
}

// Get current Unix timestamp
now_unix :: proc() -> i64 {
	dt := now()
	return datetime_to_unix(dt)
}

// Day of the week (1970-01-01 was a Thursday)
date_weekday :: proc(d: Date) -> Days {
	n := days_from_civil(i64(d.year), i64(d.month), i64(d.day))
	return Days((n + 3) %% 7)
}

// Last calendar day of d's month
date_end_of_month :: proc(d: Date) -> Date {
	return Date{d.year, d.month, get_days_in_month(d.year, Months(d.month))}
}

date_is_end_of_month :: proc(d: Date) -> bool {
	return d.day == get_days_in_month(d.year, Months(d.month))
}
