package importer

import w "../core"
import "core:strconv"
import "core:strings"

is_null :: proc(s: string) -> bool {
	ret := strings.trim(s, " ")
	return ret == "" || ret == "NA" || ret == "null" || ret == "None" || ret == "NaN"
}

is_int :: proc(s: string) -> bool {
	_, err := strconv.parse_int(s)
	return err
}

is_float :: proc(s: string) -> bool {
	_, err := strconv.parse_f64(s)
	return err
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
	has_float, has_int, has_bool, has_date, has_time, has_datetime, has_string := false, false, false, false, false, false, false

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
	kinds := int(numeric) + int(has_bool) + int(has_datetime) + int(has_date) + int(has_time) + int(has_string)
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
