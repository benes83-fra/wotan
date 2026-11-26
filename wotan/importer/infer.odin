package importer

import w "../core"
import "core:strconv"
import "core:strings"

is_null :: proc(s: string) -> bool {
	ret := strings.trim(s, " ")
	return ret == "" || ret == "NA" || ret == "null" || ret == "None" || ret == "NaN"
}

// An optional sign and decimal digits, within the range of an i64 (as pandas
// reads integers: no "0x" prefixes, no "_" separators)
parse_int_strict :: proc(s: string) -> (v: int, ok: bool) {
	i := 0
	neg := false
	if len(s) > 0 && (s[0] == '+' || s[0] == '-') {
		neg = s[0] == '-'
		i = 1
	}
	if i == len(s) {return 0, false}
	// accumulate negatively, so -9223372036854775808 fits
	acc: int
	for ; i < len(s); i += 1 {
		c := s[i]
		if c < '0' || c > '9' {return 0, false}
		d := int(c - '0')
		if acc < (min(int) + d) / 10 {return 0, false}
		acc = acc * 10 - d
	}
	if !neg {
		if acc == min(int) {return 0, false}
		acc = -acc
	}
	return acc, true
}

// Decimal float syntax ([sign] digits [. digits] [e [sign] digits], or a
// leading or trailing point), or inf / infinity / nan in any case. A value
// beyond the f64 range ("1e400") is no float, as in pandas.
parse_float_strict :: proc(s: string) -> (f64, bool) {
	t := s
	if len(t) > 0 && (t[0] == '+' || t[0] == '-') {t = t[1:]}
	if strings.equal_fold(t, "inf") ||
	   strings.equal_fold(t, "infinity") ||
	   strings.equal_fold(t, "nan") {
		v, _ := strconv.parse_f64(s)
		return v, true
	}
	i, digits := 0, 0
	for i < len(t) && t[i] >= '0' && t[i] <= '9' {i += 1; digits += 1}
	if i < len(t) && t[i] == '.' {
		i += 1
		for i < len(t) && t[i] >= '0' && t[i] <= '9' {i += 1; digits += 1}
	}
	if digits == 0 {return 0, false}
	if i < len(t) && (t[i] == 'e' || t[i] == 'E') {
		i += 1
		if i < len(t) && (t[i] == '+' || t[i] == '-') {i += 1}
		e := i
		for i < len(t) && t[i] >= '0' && t[i] <= '9' {i += 1}
		if i == e {return 0, false}
	}
	if i != len(t) {return 0, false}
	return strconv.parse_f64(s) // false only for an overflow
}

is_int :: proc(s: string) -> bool {
	_, ok := parse_int_strict(s)
	return ok
}

is_float :: proc(s: string) -> bool {
	_, ok := parse_float_strict(s)
	return ok
}

is_bool :: proc(s: string) -> bool {
	return strings.equal_fold(s, "true") || strings.equal_fold(s, "false")
}

// Date / time checks use the same validating parsers as the loaders, so an
// inferred type can always be parsed (e.g. "2024-13-45" stays a string)
is_date :: proc(s: string) -> bool {
	_, ok := w.parse_date(s)
	return ok
}

is_time :: proc(s: string) -> bool {
	_, ok := w.parse_time(s)
	return ok
}

// "YYYY-MM-DD HH:MM:SS" or ISO 8601 "YYYY-MM-DDTHH:MM:SS"
is_datetime :: proc(s: string) -> bool {
	_, ok := w.parse_datetime(s)
	return ok
}

// Most specific type that every (non-null) sample parses as. Ints and floats
// combine to Float; any other mix of kinds (e.g. numbers and booleans, dates
// and datetimes) falls back to String, so the inferred type always parses.
infer_column_type :: proc(samples: []string) -> w.ColumnType {
	has_float, has_int, has_bool, has_date, has_time, has_datetime, has_string :=
		false, false, false, false, false, false, false

	for raw in samples {
		s := strings.trim(raw, " 	")
		if s == "" || is_null(s) {
			continue
		}
		switch {
		case is_int(s):
			has_int = true
		case is_float(s):
			has_float = true
		case is_bool(s):
			has_bool = true
		case is_datetime(s):
			has_datetime = true
		case is_date(s):
			has_date = true
		case is_time(s):
			has_time = true
		case:
			has_string = true
		}
	}

	numeric := has_int || has_float
	kinds :=
		int(numeric) +
		int(has_bool) +
		int(has_datetime) +
		int(has_date) +
		int(has_time) +
		int(has_string)
	switch {
	case kinds != 1:
		return .String
	case has_float:
		return .Float
	case has_int:
		return .Int
	case has_bool:
		return .Bool
	case has_datetime:
		return .Datetime
	case has_date:
		return .Date
	case has_time:
		return .Time
	}
	return .String
}
