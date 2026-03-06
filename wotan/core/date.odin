package wotan

import "core:fmt"

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

new_Datetime_from_Date_and_Time :: proc(date: Date, time: Time) -> Datetime {
	return Datetime{date.year, date.month, date.day, time.hour, time.minute, time.second}
}

get_Date_from_Datetime :: proc(dt: Datetime) -> Date {
	return Date{dt.year, dt.month, dt.day}
}

get_Time_from_Datetime :: proc(dt: Datetime) -> Time {
	return Time{dt.hour, dt.minute, dt.second}
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
add_day_date :: proc(date: Date) -> Date {

	year := date.year
	month := date.month
	day := date.day
	days_in_month := get_days_in_month(year, Months(month))
	if day < days_in_month {
		day = day + 1
		return Date{year, month, day}
	} else {
		if Months(month) != .December {
			month = month + 1
			day = 1
		} else {
			year = year + 1
			month = 1
			day = 1
		}
	}
	return Date{year, month, day}
}


add_month_date :: proc(date: Date) -> Date {
	year := date.year
	month := date.month
	day := date.day

	if Months(month) != .December {
		month = month + 1
		return Date{year, month, day}


	}
	year = year + 1
	month = 1
	day = 1
	return Date{year, month, day}
}

add_year_date :: proc(date: Date) -> Date {
	year := date.year
	month := date.month
	day := date.day
	year = year + 1
	return Date{year, month, day}

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
		hour = hour + 100
		return Datetime{year, month, day, hour, minute, second}
	} else {
		hour = 0
		date = add_day_date(date)
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

		date = add_day_date(date)
		hour = 0
		minute = 0
		return Datetime{date.year, date.month, date.day, hour, minute, second}
	}
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
			date = add_day_date(date)
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
