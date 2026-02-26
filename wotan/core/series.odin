package wotan

import "core:mem"

Series :: struct {
	name:  string,
	type:  ColumnType,
	len:   int,
	data:  rawptr,
	nulls: []bool, // true = null
}

//
// --- Constructors ---
//

series_from_slice :: proc(name: string, slice: []$T, type: ColumnType) -> Series {
	return Series{name = name, type = type, len = slice.len, data = slice.data, nulls = nil}
}

series_empty :: proc(name: string, type: ColumnType, capacity: int) -> Series {
	size := capacity * type_size(type)
	data, err := mem.alloc(size)
	if err != nil {
		panic("Series: allocation failed")
	}

	return Series{name = name, type = type, len = 0, data = data, nulls = nil}
}

//
// --- Typed Accessors ---
//

series_at_int :: proc(s: ^Series, i: int) -> (int, bool) {
	if s.type != .Int {
		panic("series_at_int: wrong type")
	}
	if i < 0 || i >= s.len {
		panic("series_at_int: index out of bounds")
	}
	if s.nulls != nil && s.nulls[i] {
		return 0, true
	}
	base := uintptr(s.data) + uintptr(i * size_of(int))
	return (cast(^int)base)^, false
}

series_at_float :: proc(s: ^Series, i: int) -> (f64, bool) {
	if s.type != .Float {
		panic("series_at_float: wrong type")
	}
	if i < 0 || i >= s.len {
		panic("series_at_float: index out of bounds")
	}
	if s.nulls != nil && s.nulls[i] {
		return 0.0, true
	}
	base := uintptr(s.data) + uintptr(i * size_of(f64))
	return (cast(^f64)base)^, false
}

series_at_bool :: proc(s: ^Series, i: int) -> (bool, bool) {
	if s.type != .Bool {
		panic("series_at_bool: wrong type")
	}
	if i < 0 || i >= s.len {
		panic("series_at_bool: index out of bounds")
	}
	if s.nulls != nil && s.nulls[i] {
		return false, true
	}
	base := uintptr(s.data) + uintptr(i * size_of(bool))
	return (cast(^bool)base)^, false
}

series_at_string :: proc(s: ^Series, i: int) -> (string, bool) {
	if s.type != .String {
		panic("series_at_string: wrong type")
	}
	if i < 0 || i >= s.len {
		panic("series_at_string: index out of bounds")
	}
	if s.nulls != nil && s.nulls[i] {
		return "", true
	}
	base := uintptr(s.data) + uintptr(i * size_of(string))
	return (cast(^string)base)^, false
}

series_at_date :: proc(s: ^Series, i: int) -> (Date, bool) {
	if s.type != .Date {
		panic("series_at_date: wrong type")
	}
	if i < 0 || i >= s.len {
		panic("series_at_date: index out of bounds")
	}
	if s.nulls != nil && s.nulls[i] {
		return Date{}, true
	}
	base := uintptr(s.data) + uintptr(i * size_of(Date))
	return (cast(^Date)base)^, false
}
