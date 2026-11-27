package core


import "core:strings"

ColumnType :: enum {
	Invalid,
	Int,
	Float,
	Bool,
	String,
	Date,
	Time,
	Datetime,
	Duration,
}

Date :: struct {
	year:  i32,
	month: i32,
	day:   i32,
}

// A time of day to the nanosecond
Time :: struct {
	hour:       i32,
	minute:     i32,
	second:     i32,
	nanosecond: i32, // 0 ..< 1_000_000_000
}

// A calendar date and time of day to the nanosecond (no time zone)
Datetime :: struct {
	year:       i32,
	month:      i32,
	day:        i32,
	hour:       i32,
	minute:     i32,
	second:     i32,
	nanosecond: i32, // 0 ..< 1_000_000_000
}

// A length of time in nanoseconds, negative or positive (pandas' Timedelta):
// about 292 years either way. The difference of two Datetimes, or what is
// added to one. Build one with nanoseconds .. weeks (duration.odin).
Duration :: distinct i64


type_size :: proc(t: ColumnType) -> int {
	switch t {
	case .Int:
		return size_of(int)
	case .Float:
		return size_of(f64)
	case .Bool:
		return size_of(bool)
	case .String:
		return size_of(string)
	case .Date:
		return size_of(Date)
	case .Time:
		return size_of(Time)
	case .Datetime:
		return size_of(Datetime)
	case .Duration:
		return size_of(Duration)
	case .Invalid:
		return 0
	}

	return 0
}


get_column_type_by_type :: proc(type: string) -> ColumnType {
	col: ColumnType
	ltype := strings.to_lower(type)

	defer delete(ltype)
	switch ltype {
	case "int":
		col = .Int
	case "float":
		col = .Float
	case "string":
		col = .String
	case "bool":
		col = .Bool
	case "date":
		col = .Date
	case "time":
		col = .Time
	case "datetime":
		col = .Datetime


	}

	return col
}

get_column_type :: proc($T: typeid) -> ColumnType {
	col: ColumnType
	when T ==
		int || T == u16 || T == u32 || T == u64 || T == u128 || T == i16 || T == i32 || T == i64 || T == i128 {
		col = .Int
	}
	when T == f64 || T == f32 {
		col = .Float
	}
	when T == bool {
		col = .Bool
	}
	when T == string {
		col = .String
	}
	when T == Date {
		col = .Date
	}
	when T == Time {
		col = .Time
	}
	when T == Datetime {
		col = .Datetime
	}


	return col
}
