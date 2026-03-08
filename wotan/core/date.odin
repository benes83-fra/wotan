package wotan

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
	n := n
	year := date.year
	month := date.month
	day := date.day
	days_left_in_month: i32

	if n > 0 {
		for n > 0 {
			days_left_in_month = get_days_in_month(year, Months(month)) - day
			if (n > days_left_in_month) {
				n -= (days_left_in_month + 1)
				day = 1
				month += 1
				if (month > 12) {
					month = i32(Months.January)
					year += 1
				}
			} else {
				day += n
				n = 0
			}
		}
		return Date{year, month, day}
	} else if n < 0 {
		n = -n
		for n > 0 {
			if n >= day {
				n -= day
				month -= 1
				if month < 1 {
					month = i32(Months.December)
					year -= 1
				}
				day = get_days_in_month(year, Months(month))

			} else {
				day -= n
				n = 0
			}
		}
		return Date{year, month, day}


	}
	return Date{year, month, day}
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


add_month_date :: proc(date: Date, n: i32) -> Date {
	year := date.year
	month := date.month
	day := date.day

	total_months := year * 12 + month - 1 + n
	new_year := math.floor_div(total_months, 12)
	new_month := (total_months % 12) + 1
	days_in_month := get_days_in_month(new_year, Months(new_month))
	if day > days_in_month {
		new_month += 1
		day = day - days_in_month
	}
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

add_year_date :: proc(date: Date, n: i32) -> Date {
	year := date.year
	month := date.month
	day := date.day
	year = year + n
	return Date{year, month, day}

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

add_seconds_time :: proc(time: Time, n: i32) -> Time {
	hour := time.hour
	minute := time.minute
	second := time.second
	new_second := second + n
	minute_in_seconds := minute * MINUTE
	hour_in_seconds := hour * HOUR
	if new_second >= MINUTE {
		new_minute := minute + new_second / MINUTE

		new_second = new_second % MINUTE
		if new_minute >= HOUR / MINUTE {
			new_hour := hour + new_minute * MINUTE / HOUR
			new_minute = new_minute % 60
			return Time{new_hour, new_minute, new_second}
		}
		return Time{hour, new_minute, new_second}

	} else if new_second < MINUTE && new_second >= 0 {

		return Time{hour, minute, new_second}
	}
	if new_second < 0 && minute > 0 {
		new_minute := minute + new_second / MINUTE - 1
		new_second = MINUTE + new_second % MINUTE
		if new_minute <= 0 {
			new_hour := hour - 1 + new_minute / (60)
			new_minute = 60 + new_minute % 60
			return Time{new_hour, new_minute, new_second}
		}
		return Time{hour, new_minute, new_second}
	}
	return Time{hour, minute, second}
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

add_seconds_datetime :: proc(datetime: Datetime, n: i32) -> Datetime {

	year := datetime.year
	month := datetime.month
	day := datetime.day
	hour := datetime.hour
	minute := datetime.minute
	second := datetime.second

	time := Time{hour, minute, second}
	time = add_seconds_time(time, n)

	if time.hour > 24 {

		days := time.hour / 24
		date := Date{year, month, day}
		date = add_day_date(date, days)
		time.hour = time.hour % 24
		return Datetime{date.year, date.month, date.day, time.hour, time.minute, time.second}
	} else if time.hour < 0 {
		days := -(1 - time.hour / 24)
		fmt.println(days, n)
		date := Date{year, month, day}
		date = add_day_date(date, days)
		time.hour = time.hour % 24 + 24
		return Datetime{date.year, date.month, date.day, time.hour, time.minute, time.second}
	} else {
		return Datetime{year, month, day, time.hour, time.minute, time.second}
	}
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
get_date_day_diffs :: proc(a, b: Date) -> i32 {
	a_year := a.year
	a_month := a.month
	a_day := a.day
	b_year := b.year
	b_month := b.month
	b_day := b.day

	if (a_year == b_year) && (a_month == b_month) {
		return abs_days(b_day - a_day)
	}
	if a_year == b_year {
		if a_month < b_month {
			return get_day_diff_between_months(a, b)
		} else {
			return get_day_diff_between_months(b, a)
		}
	}
	if a_year < b_year {
		return get_day_diff_between_years(a, b)
	} else {
		return get_day_diff_between_years(b, a)
	}

}


now :: proc() -> Datetime {

	now := t.now()
	buf: [20]u8
	res := t.to_string_yyyy_mm_dd(now, buf[:])
	buf2: [20]u8
	res2 := t.to_string_hms(now, buf2[:])
	fmt.println(res2)

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
