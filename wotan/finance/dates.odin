package finance

import w "../core"
import "core:math"

// ============================================================================
// CALENDARS, DAY COUNTERS AND SCHEDULES ON w.Date
// ============================================================================
// Conventions follow QuantLib (its tests verify against QuantLib 1.44). Dates
// are the core package's w.Date; internally the procedures work on day numbers
// (w.days_from_civil, days since 1970-01-01). A zero w.Date{} is "no date".

TimeUnit :: enum u8 {
	Days,
	Weeks,
	Months,
	Years,
}

// A length of time, e.g. Period{6, .Months}
Period :: struct {
	n:    int,
	unit: TimeUnit,
}

// Day number of a date (days since 1970-01-01), shared within the package
@(private)
_dn :: proc(d: w.Date) -> i64 {
	return w.days_from_civil(i64(d.year), i64(d.month), i64(d.day))
}

@(private = "file")
_add :: proc(n: i64, p: Period) -> i64 {
	return _dn(date_add_period(w.civil_from_days(n), p))
}

// Calendar arithmetic without business-day adjustment (QuantLib Date + Period):
// months and years clamp the day to the end of a shorter month
date_add_period :: proc(d: w.Date, p: Period) -> w.Date {
	switch p.unit {
	case .Days:
		return w.add_day_date(d, i32(p.n))
	case .Weeks:
		return w.add_day_date(d, i32(7 * p.n))
	case .Months:
		return w.add_month_date(d, i32(p.n))
	case .Years:
		return w.add_month_date(d, i32(12 * p.n))
	}
	return d
}

// Western Easter Sunday (anonymous Gregorian algorithm)
easter_sunday :: proc(year: int) -> w.Date {
	a, b, c := year % 19, year / 100, year % 100
	d, e := b / 4, b % 4
	f := (b + 8) / 25
	g := (b - f + 1) / 3
	h := (19 * a + b - d - g + 15) % 30
	i, k := c / 4, c % 4
	l := (32 + 2 * e + 2 * i - h - k) % 7
	m := (a + 11 * h + 22 * l) / 451
	n := h + l - 7 * m + 114
	return w.Date{i32(year), i32(n / 31), i32(n % 31 + 1)}
}

// ============================================================================
// HOLIDAY CALENDARS
// ============================================================================

CalendarMarket :: enum u8 {
	Weekends, // Saturdays and Sundays only
	TARGET, // Euro area TARGET2 settlement
	US_Settlement, // US federal holidays (QuantLib UnitedStates(Settlement))
	US_Government_Bond, // SIFMA recommended bond-market closings
	UK_Settlement, // England & Wales bank holidays
}

// A joint calendar: a day is a business day when it is one in every market of
// the set. The empty set is the null calendar (every day is a business day).
Calendar :: bit_set[CalendarMarket]

is_business_day :: proc(cal: Calendar, d: w.Date) -> bool {
	return _is_bd(cal, _dn(d))
}

is_holiday :: proc(cal: Calendar, d: w.Date) -> bool {
	return !_is_bd(cal, _dn(d))
}

@(private = "file")
_is_bd :: proc(cal: Calendar, n: i64) -> bool {
	if cal == {} {return true}
	wd := w.Days((n + 3) %% 7)
	if wd == .Saturday || wd == .Sunday {return false}
	if cal - {.Weekends} == {} {return true}
	date := w.civil_from_days(n)
	y, m, d := int(date.year), int(date.month), int(date.day)
	jan1 := w.days_from_civil(i64(y), 1, 1)
	doy := int(n - jan1) + 1
	em := int(_dn(easter_sunday(y)) - jan1) + 2 // Easter Monday, day of year
	for mk in cal {
		switch mk {
		case .Weekends:
		case .TARGET:
			if _is_target_holiday(y, m, d, doy, em) {return false}
		case .US_Settlement:
			if _is_us_settlement_holiday(y, m, d, wd) {return false}
		case .US_Government_Bond:
			if _is_us_gov_bond_holiday(y, m, d, wd, doy, em) {return false}
		case .UK_Settlement:
			if _is_uk_settlement_holiday(y, m, d, wd, doy, em) {return false}
		}
	}
	return true
}

@(private = "file")
_is_target_holiday :: proc(y, m, d, doy, em: int) -> bool {
	return(
		(d == 1 && m == 1) ||
		(doy == em - 3 && y >= 2000) ||
		(doy == em && y >= 2000) ||
		(d == 1 && m == 5 && y >= 2000) ||
		(d == 25 && m == 12) ||
		(d == 26 && m == 12 && y >= 2000) ||
		(d == 31 && m == 12 && (y == 1998 || y == 1999 || y == 2001)) \
	) // Good Friday
	// Easter Monday
	// Labour Day

}

// Federal holidays shared by the US calendars; a Saturday holiday moves to
// Friday and a Sunday one to Monday unless noted
@(private = "file")
_us_mlk :: proc(y, m, d: int, wd: w.Days) -> bool {
	return d >= 15 && d <= 21 && wd == .Monday && m == 1 && y >= 1983
}
@(private = "file")
_us_washington :: proc(y, m, d: int, wd: w.Days) -> bool {
	if y >= 1971 {return d >= 15 && d <= 21 && wd == .Monday && m == 2}
	return (d == 22 || (d == 23 && wd == .Monday) || (d == 21 && wd == .Friday)) && m == 2
}
@(private = "file")
_us_memorial :: proc(y, m, d: int, wd: w.Days) -> bool {
	if y >= 1971 {return d >= 25 && wd == .Monday && m == 5}
	return (d == 30 || (d == 31 && wd == .Monday) || (d == 29 && wd == .Friday)) && m == 5
}
@(private = "file")
_us_juneteenth :: proc(y, m, d: int, wd: w.Days) -> bool {
	return(
		(d == 19 || (d == 20 && wd == .Monday) || (d == 18 && wd == .Friday)) &&
		m == 6 &&
		y >= 2022 \
	)
}
@(private = "file")
_us_independence :: proc(m, d: int, wd: w.Days) -> bool {
	return (d == 4 || (d == 5 && wd == .Monday) || (d == 3 && wd == .Friday)) && m == 7
}
@(private = "file")
_us_labor :: proc(m, d: int, wd: w.Days) -> bool {
	return d <= 7 && wd == .Monday && m == 9
}
@(private = "file")
_us_columbus :: proc(y, m, d: int, wd: w.Days) -> bool {
	return d >= 8 && d <= 14 && wd == .Monday && m == 10 && y >= 1971
}
@(private = "file")
_us_veterans :: proc(y, m, d: int, wd: w.Days, saturday_to_friday: bool) -> bool {
	if y <= 1970 || y >= 1978 {
		return(
			(d == 11 ||
				(d == 12 && wd == .Monday) ||
				(saturday_to_friday && d == 10 && wd == .Friday)) &&
			m == 11 \
		)
	}
	return d >= 22 && d <= 28 && wd == .Monday && m == 10
}
@(private = "file")
_us_thanksgiving :: proc(m, d: int, wd: w.Days) -> bool {
	return d >= 22 && d <= 28 && wd == .Thursday && m == 11
}
@(private = "file")
_us_christmas :: proc(m, d: int, wd: w.Days) -> bool {
	return (d == 25 || (d == 26 && wd == .Monday) || (d == 24 && wd == .Friday)) && m == 12
}

@(private = "file")
_is_us_settlement_holiday :: proc(y, m, d: int, wd: w.Days) -> bool {
	return(
		((d == 1 || (d == 2 && wd == .Monday)) && m == 1) ||
		(d == 31 && wd == .Friday && m == 12) ||
		_us_mlk(y, m, d, wd) ||
		_us_washington(y, m, d, wd) ||
		_us_memorial(y, m, d, wd) ||
		_us_juneteenth(y, m, d, wd) ||
		_us_independence(m, d, wd) ||
		_us_labor(m, d, wd) ||
		_us_columbus(y, m, d, wd) ||
		_us_veterans(y, m, d, wd, true) ||
		_us_thanksgiving(m, d, wd) ||// New Year's Day on a Saturday
		_us_christmas(m, d, wd) \
	)
}

// SIFMA: as federal, but no Friday before a Saturday New Year / Veterans Day;
// Good Friday, except from 1996 when it is the first Friday of April (payrolls
// day, early close only); one-off closings
@(private = "file")
_is_us_gov_bond_holiday :: proc(y, m, d: int, wd: w.Days, doy, em: int) -> bool {
	good_friday := doy == em - 3 && !(y >= 1996 && m == 4 && d <= 7)
	return(
		((d == 1 || (d == 2 && wd == .Monday)) && m == 1) ||
		_us_mlk(y, m, d, wd) ||
		_us_washington(y, m, d, wd) ||
		good_friday ||
		_us_memorial(y, m, d, wd) ||
		_us_juneteenth(y, m, d, wd) ||
		_us_independence(m, d, wd) ||
		_us_labor(m, d, wd) ||
		_us_columbus(y, m, d, wd) ||
		_us_veterans(y, m, d, wd, false) ||
		_us_thanksgiving(m, d, wd) ||
		_us_christmas(m, d, wd) ||
		(y == 2004 && m == 6 && d == 11) ||
		(y == 2012 && m == 10 && d == 30) ||
		(y == 2018 && m == 12 && d == 5) \
	) // President Reagan's funeral
	// Hurricane Sandy
	// President Bush's funeral
}

@(private = "file")
_is_uk_settlement_holiday :: proc(y, m, d: int, wd: w.Days, doy, em: int) -> bool {
	bank :=
		(d <= 7 && wd == .Monday && m == 5 && y != 1995 && y != 2020) ||
		(d == 8 && m == 5 && (y == 1995 || y == 2020)) ||
		(d >= 25 && wd == .Monday && m == 5 && y != 2002 && y != 2012 && y != 2022) ||
		(d == 4 && m == 6 && (y == 2002 || y == 2012)) ||
		(d == 2 && m == 6 && y == 2022) ||
		(d >= 25 && wd == .Monday && m == 8) ||
		(d == 3 && m == 6 && y == 2002) ||
		(d == 5 && m == 6 && y == 2012) ||
		(d == 3 && m == 6 && y == 2022) ||// Early May bank holiday, first Monday of May (May 8th in 1995 and 2020, VE day)
		(d == 29 && m == 4 && y == 2011) ||
		(d == 19 && m == 9 && y == 2022) ||// Spring bank holiday, last Monday of May (moved for the jubilees)
		(d == 8 && m == 5 && y == 2023) // Summer bank holiday, last Monday of August // one-offs: jubilees, royal wedding, state funeral, coronation
	return(
		((d == 1 || ((d == 2 || d == 3) && wd == .Monday)) && m == 1) ||
		doy == em - 3 ||
		doy == em ||
		bank ||
		((d == 25 || (d == 27 && (wd == .Monday || wd == .Tuesday))) && m == 12) ||
		((d == 26 || (d == 28 && (wd == .Monday || wd == .Tuesday))) && m == 12) ||
		(d == 31 && m == 12 && y == 1999) \
	)
}

// ============================================================================
// BUSINESS-DAY ADJUSTMENT AND DATE ARITHMETIC
// ============================================================================

BusinessDayConvention :: enum u8 {
	Following, // next business day
	ModifiedFollowing, // next, unless it is in the next month, then previous
	Preceding, // previous business day
	ModifiedPreceding, // previous, unless it is in the previous month, then next
	Unadjusted,
	HalfMonthModifiedFollowing, // modified following, also not crossing the 15th
	Nearest, // nearest business day, following on a tie
}

calendar_adjust :: proc(
	cal: Calendar,
	d: w.Date,
	bdc: BusinessDayConvention = .Following,
) -> w.Date {
	return w.civil_from_days(_adjust(cal, _dn(d), bdc))
}

@(private = "file")
_adjust :: proc(cal: Calendar, n: i64, bdc: BusinessDayConvention) -> i64 {
	switch bdc {
	case .Unadjusted:
		return n
	case .Following, .ModifiedFollowing, .HalfMonthModifiedFollowing:
		n1 := n
		for !_is_bd(cal, n1) {n1 += 1}
		if bdc != .Following {
			d, d1 := w.civil_from_days(n), w.civil_from_days(n1)
			if d1.month != d.month {return _adjust(cal, n, .Preceding)}
			if bdc == .HalfMonthModifiedFollowing && d.day <= 15 && d1.day > 15 {
				return _adjust(cal, n, .Preceding)
			}
		}
		return n1
	case .Preceding, .ModifiedPreceding:
		n1 := n
		for !_is_bd(cal, n1) {n1 -= 1}
		if bdc == .ModifiedPreceding && w.civil_from_days(n1).month != w.civil_from_days(n).month {
			return _adjust(cal, n, .Following)
		}
		return n1
	case .Nearest:
		n1, n2 := n, n
		for !_is_bd(cal, n1) && !_is_bd(cal, n2) {n1 += 1; n2 -= 1}
		return n2 if !_is_bd(cal, n1) else n1
	}
	return n
}

// Last business day of d's month
calendar_end_of_month :: proc(cal: Calendar, d: w.Date) -> w.Date {
	return w.civil_from_days(_eom(cal, _dn(d)))
}

// True when d is the last business day of its month
calendar_is_end_of_month :: proc(cal: Calendar, d: w.Date) -> bool {
	return _is_eom(cal, _dn(d))
}

@(private = "file")
_eom :: proc(cal: Calendar, n: i64) -> i64 {
	return _adjust(cal, _dn(w.date_end_of_month(w.civil_from_days(n))), .Preceding)
}

@(private = "file")
_is_eom :: proc(cal: Calendar, n: i64) -> bool {
	return w.civil_from_days(n).month != w.civil_from_days(_adjust(cal, n + 1, .Following)).month
}

// Moves d by n units (QuantLib Calendar::advance): days count business days;
// weeks, months and years are calendar arithmetic followed by the adjustment.
// With end_of_month, a d at the end of its month (the last business day, or the
// last calendar day when unadjusted) maps to the end of the target month.
calendar_advance :: proc(
	cal: Calendar,
	d: w.Date,
	n: int,
	unit: TimeUnit,
	bdc: BusinessDayConvention = .Following,
	end_of_month := false,
) -> w.Date {
	return w.civil_from_days(_advance(cal, _dn(d), n, unit, bdc, end_of_month))
}

calendar_advance_period :: proc(
	cal: Calendar,
	d: w.Date,
	p: Period,
	bdc: BusinessDayConvention = .Following,
	end_of_month := false,
) -> w.Date {
	return calendar_advance(cal, d, p.n, p.unit, bdc, end_of_month)
}

@(private = "file")
_advance :: proc(
	cal: Calendar,
	n: i64,
	k: int,
	unit: TimeUnit,
	bdc: BusinessDayConvention,
	eom: bool,
) -> i64 {
	if k == 0 {return _adjust(cal, n, bdc)}
	switch unit {
	case .Days:
		n1, left := n, k
		for left > 0 {
			n1 += 1
			for !_is_bd(cal, n1) {n1 += 1}
			left -= 1
		}
		for left < 0 {
			n1 -= 1
			for !_is_bd(cal, n1) {n1 -= 1}
			left += 1
		}
		return n1
	case .Weeks:
		return _adjust(cal, n + i64(7 * k), bdc)
	case .Months, .Years:
		d := w.civil_from_days(n)
		d1 := date_add_period(d, Period{k, unit})
		if eom {
			if bdc == .Unadjusted {
				if w.date_is_end_of_month(d) {return _dn(w.date_end_of_month(d1))}
			} else if _is_eom(cal, n) {
				return _eom(cal, _dn(d1))
			}
		}
		return _adjust(cal, _dn(d1), bdc)
	}
	return n
}

// Business days between the dates, counting `from` when include_first and `to`
// when include_last; negative when to < from
business_days_between :: proc(
	cal: Calendar,
	from, to: w.Date,
	include_first := true,
	include_last := false,
) -> int {
	a, b := _dn(from), _dn(to)
	if a > b {return -business_days_between(cal, to, from, include_last, include_first)}
	if a == b {
		return 1 if include_first && include_last && _is_bd(cal, a) else 0
	}
	n := 0
	for k := a + 1; k < b; k += 1 {
		if _is_bd(cal, k) {n += 1}
	}
	if include_first && _is_bd(cal, a) {n += 1}
	if include_last && _is_bd(cal, b) {n += 1}
	return n
}

// ============================================================================
// DAY COUNTERS ON DATES
// ============================================================================

@(private = "file")
_is_last_of_february :: proc(d: w.Date) -> bool {
	return d.month == 2 && w.date_is_end_of_month(d)
}

// Day count under the convention; `termination` is the maturity for
// THIRTY_E_360_ISDA (whose last day of February is not moved to the 30th)
day_count_days :: proc(dc: DayCountConvention, d1, d2: w.Date, termination := w.Date{}) -> int {
	#partial switch dc {
	case .THIRTY_360, .THIRTY_360_US, .THIRTY_E_360, .THIRTY_E_360_ISDA:
		dd1, dd2 := int(d1.day), int(d2.day)
		#partial switch dc {
		case .THIRTY_360:
			// Bond Basis (ISDA 2006 4.16(f))
			if dd1 == 31 {dd1 = 30}
			if dd2 == 31 && dd1 == 30 {dd2 = 30}
		case .THIRTY_360_US:
			if _is_last_of_february(d2) && _is_last_of_february(d1) {dd2 = 30}
			if _is_last_of_february(d1) {dd1 = 30}
			if dd2 == 31 && dd1 >= 30 {dd2 = 30}
			if dd1 == 31 {dd1 = 30}
		case .THIRTY_E_360:
			// Eurobond basis
			dd1 = min(dd1, 30)
			dd2 = min(dd2, 30)
		case .THIRTY_E_360_ISDA:
			// German
			if dd1 == 31 {dd1 = 30}
			if dd2 == 31 {dd2 = 30}
			if _is_last_of_february(d1) {dd1 = 30}
			if d2 != termination && _is_last_of_february(d2) {dd2 = 30}
		}
		return 360 * int(d2.year - d1.year) + 30 * int(d2.month - d1.month) + dd2 - dd1
	}
	return int(_dn(d2) - _dn(d1))
}

// Year fraction between dates (QuantLib DayCounter::yearFraction). ACT_ACT is
// ISDA; ACT_ACT_ICMA uses the coupon's reference period [ref_start, ref_end]
// (defaulting to [d1, d2]) and handles long stubs by extending it by whole
// periods. ACT_365 is Actual/365 Fixed.
year_fraction :: proc(
	dc: DayCountConvention,
	d1, d2: w.Date,
	ref_start := w.Date{},
	ref_end := w.Date{},
	termination := w.Date{},
) -> f64 {
	switch dc {
	case .ACT_360:
		return f64(_dn(d2) - _dn(d1)) / 360.0
	case .ACT_365:
		return f64(_dn(d2) - _dn(d1)) / 365.0
	case .ACT_ACT:
		if d1 == d2 {return 0.0}
		if _dn(d1) > _dn(d2) {return -year_fraction(dc, d2, d1)}
		dib1 := f64(w.get_days_of_the_year(d1.year))
		dib2 := f64(w.get_days_of_the_year(d2.year))
		return(
			f64(d2.year - d1.year - 1) +
			f64(w.days_from_civil(i64(d1.year) + 1, 1, 1) - _dn(d1)) / dib1 +
			f64(_dn(d2) - w.days_from_civil(i64(d2.year), 1, 1)) / dib2 \
		)
	case .ACT_ACT_ICMA:
		rs := _dn(ref_start) if ref_start != {} else 0
		re := _dn(ref_end) if ref_end != {} else 0
		return _act_act_icma(_dn(d1), _dn(d2), rs, re, ref_start != {}, ref_end != {})
	case .THIRTY_360, .THIRTY_360_US, .THIRTY_E_360, .THIRTY_E_360_ISDA:
		return f64(day_count_days(dc, d1, d2, termination)) / 360.0
	}
	return 0.0
}

@(private = "file")
_act_act_icma :: proc(d1, d2, ref_start, ref_end: i64, has_start, has_end: bool) -> f64 {
	if d1 == d2 {return 0.0}
	if d1 > d2 {return -_act_act_icma(d2, d1, ref_start, ref_end, has_start, has_end)}
	rs := ref_start if has_start else d1
	re := ref_end if has_end else d2
	if !(re > rs && re > d1) {return math.QNAN_F64}
	months := int(math.round(12.0 * f64(re - rs) / 365.0))
	if months == 0 {
		rs = d1
		re = _add(d1, Period{1, .Years})
		months = 12
	}
	period := f64(months) / 12.0
	if d2 <= re {
		if d1 >= rs {
			return period * f64(d2 - d1) / f64(re - rs)
		}
		// long first coupon: the notional period before rs
		prev := _add(rs, Period{-months, .Months})
		if d2 > rs {
			return(
				_act_act_icma(d1, rs, prev, rs, true, true) +
				_act_act_icma(rs, d2, rs, re, true, true) \
			)
		}
		return _act_act_icma(d1, d2, prev, rs, true, true)
	}
	// long last coupon: rs <= d1 < re < d2, whole notional periods after re
	if rs > d1 {return math.QNAN_F64}
	sum := _act_act_icma(d1, re, rs, re, true, true)
	i := 0
	nrs, nre: i64
	for {
		nrs = _add(re, Period{months * i, .Months})
		nre = _add(re, Period{months * (i + 1), .Months})
		if d2 < nre {break}
		sum += period
		i += 1
	}
	return sum + _act_act_icma(nrs, d2, nrs, nre, true, true)
}

// ============================================================================
// IMM AND CDS DATES
// ============================================================================

// Third Wednesday of the month
@(private = "file")
_third_wednesday :: proc(y, m: int) -> i64 {
	first := w.days_from_civil(i64(y), i64(m), 1)
	return first + (i64(w.Days.Wednesday) - (first + 3) %% 7) %% 7 + 14
}

// IMM dates: third Wednesdays of March, June, September and December (of any
// month when main_cycle is false)
is_imm_date :: proc(d: w.Date, main_cycle := true) -> bool {
	if main_cycle && d.month % 3 != 0 {return false}
	return _dn(d) == _third_wednesday(int(d.year), int(d.month))
}

// First IMM date strictly after d
imm_next_date :: proc(d: w.Date, main_cycle := true) -> w.Date {
	step := 3 if main_cycle else 1
	y, m := int(d.year), int(d.month)
	m = ((m + step - 1) / step) * step // this month if in the cycle, else the next cycle month
	if m > 12 {m -= 12; y += 1}
	for {
		t := _third_wednesday(y, m)
		if t > _dn(d) {return w.civil_from_days(t)}
		m += step
		if m > 12 {m -= 12; y += 1}
	}
}

// The IMM 20th (Mar/Jun/Sep/Dec 20) on or before / on or after a day
@(private = "file")
_previous_twentieth :: proc(n: i64) -> i64 {
	d := w.civil_from_days(n)
	r := w.days_from_civil(i64(d.year), i64(d.month), 20)
	if r > n {r = _add(r, Period{-1, .Months})}
	m := int(w.civil_from_days(r).month)
	if m % 3 != 0 {r = _add(r, Period{-(m % 3), .Months})}
	return r
}

@(private = "file")
_next_twentieth :: proc(n: i64) -> i64 {
	d := w.civil_from_days(n)
	r := w.days_from_civil(i64(d.year), i64(d.month), 20)
	if r < n {r = _add(r, Period{1, .Months})}
	m := int(w.civil_from_days(r).month)
	if m % 3 != 0 {r = _add(r, Period{3 - m % 3, .Months})}
	return r
}

// Scheduled termination of a standard CDS traded on trade_date for `tenor`
// (whole years or multiples of 3 months; QuantLib cdsMaturity): the IMM 20th
// on or before the trade date + tenor + 3 months. Under CDS2015 maturities
// roll only on Jun 20 and Dec 20: a trade between Mar 20 and Jun 19 (Sep 20 and
// Dec 19) still matures on a Dec 20 (Jun 20). A zero tenor is only defined
// before the June / December roll (else returns the zero date).
cds_maturity_date :: proc(
	trade_date: w.Date,
	tenor: Period,
	rule: DateGenerationRule = .CDS2015,
) -> w.Date {
	anchor := _previous_twentieth(_dn(trade_date))
	if rule == .CDS2015 {
		m := w.civil_from_days(anchor).month
		if m == 6 || m == 12 {
			if tenor.n == 0 {return {}}
			anchor = _add(anchor, Period{-3, .Months})
		}
	}
	return w.civil_from_days(_add(_add(anchor, tenor), Period{3, .Months}))
}

// ============================================================================
// SCHEDULES
// ============================================================================

DateGenerationRule :: enum u8 {
	Backward, // from the termination date back (stub at the front)
	Forward, // from the effective date forward (stub at the back)
	Zero, // effective and termination dates only
	CDS, // standard CDS: quarterly roll on Mar/Jun/Sep/Dec 20, from the IMM 20th before the effective date
	CDS2015, // CDS dates as CDS; maturities roll semi-annually (see cds_maturity_date)
}

// dates[0] = effective … dates[n] = termination, adjusted; is_regular[k] says
// whether period k (dates[k] → dates[k+1]) is a whole tenor (false for stubs)
Schedule :: struct {
	dates:                  []w.Date,
	is_regular:             []bool,
	tenor:                  Period,
	calendar:               Calendar,
	convention:             BusinessDayConvention,
	termination_convention: BusinessDayConvention,
	end_of_month:           bool,
}

// Generates a schedule as QuantLib's Schedule: unadjusted dates stepped by
// whole multiples of the tenor from the seed (termination for Backward,
// effective for Forward), optional explicit first / next-to-last dates (stubs),
// then adjustment; with end_of_month and a seed on the month's last business
// day the intermediate dates roll to their month end first. The termination
// date is adjusted with termination_convention.
make_schedule :: proc(
	effective, termination: w.Date,
	tenor: Period,
	calendar: Calendar,
	convention: BusinessDayConvention = .ModifiedFollowing,
	termination_convention: BusinessDayConvention = .ModifiedFollowing,
	rule: DateGenerationRule = .Backward,
	end_of_month := false,
	first_date := w.Date{},
	next_to_last_date := w.Date{},
	allocator := context.allocator,
) -> Schedule {
	s := Schedule {
		tenor                  = tenor,
		calendar               = calendar,
		convention             = convention,
		termination_convention = termination_convention,
	}
	r := rule
	if tenor.n == 0 {r = .Zero}
	cds := r == .CDS || r == .CDS2015
	eom := end_of_month && !cds && (tenor.unit == .Months || tenor.unit == .Years)
	s.end_of_month = eom
	eff, term := _dn(effective), _dn(termination)
	first := _dn(first_date) if first_date != {} else 0
	ntl := _dn(next_to_last_date) if next_to_last_date != {} else 0
	has_first, has_ntl := first_date != {}, next_to_last_date != {}

	dates := make([dynamic]i64, 0, 16, context.temp_allocator)
	reg := make([dynamic]bool, 0, 16, context.temp_allocator)
	step :: proc(seed: i64, k: int, tenor: Period, bdc: BusinessDayConvention, eom: bool) -> i64 {
		return _advance({}, seed, k * tenor.n, tenor.unit, bdc, eom)
	}

	seed: i64
	switch r {
	case .Zero:
		append(&dates, eff, term)
		append(&reg, true)
		seed = eff
	case .Backward:
		append(&dates, term)
		seed = term
		periods := 1
		if has_ntl {
			inject_at(&dates, 0, ntl)
			inject_at(&reg, 0, step(seed, -periods, tenor, convention, eom) == ntl)
			seed = ntl
		}
		exit := first if has_first else eff
		for {
			temp := step(seed, -periods, tenor, convention, eom)
			if temp < exit {
				if has_first &&
				   _adjust(calendar, dates[0], convention) !=
					   _adjust(calendar, first, convention) {
					inject_at(&dates, 0, first)
					inject_at(&reg, 0, false)
				}
				break
			}
			if _adjust(calendar, dates[0], convention) != _adjust(calendar, temp, convention) {
				inject_at(&dates, 0, temp)
				inject_at(&reg, 0, true)
			}
			periods += 1
		}
		if _adjust(calendar, dates[0], convention) != _adjust(calendar, eff, convention) {
			inject_at(&dates, 0, eff)
			inject_at(&reg, 0, false)
		}
	case .Forward, .CDS, .CDS2015:
		if cds {
			// start on the IMM 20th on or before the effective date (one roll
			// earlier if that 20th only becomes a business day after it)
			prev20 := _previous_twentieth(eff)
			if _adjust(calendar, prev20, convention) > eff {
				append(&dates, _add(prev20, Period{-3, .Months}))
				append(&reg, true)
			}
			append(&dates, prev20)
		} else {
			append(&dates, eff)
		}
		seed = dates[len(dates) - 1]
		periods := 1
		if has_first {
			append(&dates, first)
			append(&reg, step(seed, periods, tenor, convention, eom) == first)
			seed = first
		} else if cds {
			next20 := _next_twentieth(eff)
			if next20 != eff {
				append(&dates, next20)
				append(&reg, true)
				seed = next20
			}
		}
		exit := ntl if has_ntl else term
		for {
			temp := step(seed, periods, tenor, convention, eom)
			if temp > exit {
				if has_ntl &&
				   _adjust(calendar, dates[len(dates) - 1], convention) !=
					   _adjust(calendar, ntl, convention) {
					append(&dates, ntl)
					append(&reg, false)
				}
				break
			}
			if _adjust(calendar, dates[len(dates) - 1], convention) !=
			   _adjust(calendar, temp, convention) {
				append(&dates, temp)
				append(&reg, true)
			}
			periods += 1
		}
		if _adjust(calendar, dates[len(dates) - 1], termination_convention) !=
		   _adjust(calendar, term, termination_convention) {
			append(&dates, _next_twentieth(term) if cds else term)
			append(&reg, cds)
		}
	}

	// intermediate dates (rolled to the month end under the end-of-month rule),
	// then the effective date and, unless unadjusted or a CDS rule, the
	// termination date
	n := len(dates)
	roll := eom && _is_eom(calendar, seed)
	for i in 1 ..< n - 1 {
		if roll {dates[i] = _dn(w.date_end_of_month(w.civil_from_days(dates[i])))}
		dates[i] = _adjust(calendar, dates[i], convention)
	}
	dates[0] = _adjust(calendar, dates[0], convention)
	if termination_convention != .Unadjusted && !cds {
		dates[n - 1] = _adjust(calendar, dates[n - 1], termination_convention)
	}

	// an adjusted next-to-last / second date may collide with the ends
	if len(dates) >= 2 && dates[len(dates) - 2] >= dates[len(dates) - 1] {
		k := len(dates)
		if len(reg) >= 2 {reg[len(reg) - 2] = dates[k - 2] == dates[k - 1]}
		dates[k - 2] = dates[k - 1]
		pop(&dates)
		pop(&reg)
	}
	if len(dates) >= 2 && dates[1] <= dates[0] {
		if len(reg) >= 2 {reg[1] = dates[1] == dates[0]}
		dates[1] = dates[0]
		ordered_remove(&dates, 0)
		ordered_remove(&reg, 0)
	}

	s.dates = make([]w.Date, len(dates), allocator)
	for d, i in dates {s.dates[i] = w.civil_from_days(d)}
	s.is_regular = make([]bool, len(reg), allocator)
	copy(s.is_regular, reg[:])
	delete(dates)
	delete(reg)
	return s
}

free_schedule :: proc(s: ^Schedule, allocator := context.allocator) {
	delete(s.dates, allocator)
	delete(s.is_regular, allocator)
	s^ = {}
}

// Reference period of accrual period k for coupon day counting (QuantLib
// FixedRateLeg): an irregular first period is measured against the regular
// period ending at its end, an irregular last one against the regular period
// starting at its start
schedule_reference_period :: proc(s: Schedule, k: int) -> (ref_start, ref_end: w.Date) {
	ref_start, ref_end = s.dates[k], s.dates[k + 1]
	if s.is_regular[k] || s.tenor.n == 0 {return}
	if k == 0 {
		ref_start = calendar_advance(
			s.calendar,
			ref_end,
			-s.tenor.n,
			s.tenor.unit,
			s.convention,
			s.end_of_month,
		)
	} else if k == len(s.dates) - 2 {
		ref_end = calendar_advance(
			s.calendar,
			ref_start,
			s.tenor.n,
			s.tenor.unit,
			s.convention,
			s.end_of_month,
		)
	}
	return
}

// Coupons per year of a tenor (0 when it does not divide a year)
period_frequency :: proc(p: Period) -> int {
	months: int
	switch p.unit {
	case .Months:
		months = p.n
	case .Years:
		months = 12 * p.n
	case .Weeks:
		return 52 / p.n if p.n > 0 && 52 % p.n == 0 else 0
	case .Days:
		return 0
	}
	if months <= 0 || 12 % months != 0 {return 0}
	return 12 / months
}

// ACT/365 Fixed time from the reference date, the time axis of dated curves
date_to_time :: proc(reference, d: w.Date) -> f64 {
	return f64(_dn(d) - _dn(reference)) / 365.0
}
