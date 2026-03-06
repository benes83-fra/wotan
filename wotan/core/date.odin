package wotan


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
/*
days_per_month := map[Months]int {
	.January   = 31,
	.February  = 28,
	.March     = 31,
	.April     = 30,
	.May       = 31,
	.June      = 30,
	.July      = 31,
	.August    = 31,
	.September = 30,
	.October   = 31,
	.November  = 30,
	.December  = 31,
}*/


add_day :: proc(date: Date) -> Date {

	year := date.year
	month := date.month
	day := date.day
	if day < 28 {
		day = day + 1
		return Date{year, month, day}
	}
	mMonth := Months(month)
	switch mMonth {
	case .February:
		month = month + 1
		day = 1

	case .April:
	case .June:
	case .September:
	case .November:
		if day < 30 {
			day = day + 1

		} else {
			month = month + 1
			day = 1
		}
	case .January:
	case .March:
	case .May:
	case .July:
	case .August:
	case .October:
		if day < 31 {
			day = day + 1
		} else {
			month = month + 1
			day = 1

		}
	case .December:
		if day < 31 {
			day = day + 1
		} else {
			year = year + 1
			month = 1
			day = 1
		}
	}
	return Date{year, month, day}

}


add_month :: proc(date: Date) -> Date {
	year := date.year
	month := date.month
	day := date.day

	if month < 12 {
		month = month + 1
		return Date{year, month, day}


	}
	year = year + 1
	month = 1
	day = 1
	return Date{year, month, day}
}

add_year :: proc(date: Date) -> Date {
	year := date.year
	month := date.month
	day := date.day
	year = year + 1
	return Date{year, month, day}

}


add_second :: proc(time: Time) -> Time {
	hour := time.hour
	minute := time.minute
	second := time.second
	if second < 60 {
		second = second + 1

		return Time{hour, minute, second}
	}
	if minute < 60 {

		minute = minute + 1
		second = 1

	} else {
		if hour < 24 {
			hour = hour + 1
			minute = 1
			second = 1
		} else {
			hour = 1
			minute = 1
			second = 1
		}

	}
	return Time{hour, minute, second}
}
