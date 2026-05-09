package core

import "base:intrinsics"
import "core:mem"

// ------------------------------------------------------------
// Indexing
// ------------------------------------------------------------

set_index :: proc(df: ^DataFrame, colname: string) {
	if !(colname in df.name_to_index) {
		panic("set_index: no such column")
	}
	df.index_column = colname
	df.has_index = true
}

// ------------------------------------------------------------
// Comparison helpers
// ------------------------------------------------------------

equals_generic :: proc(a, b: $T) -> bool where intrinsics.type_is_numeric(T) {
	return a == b
}

less_generic :: proc(a, b: $T) -> bool where intrinsics.type_is_numeric(T) {
	return a < b
}

less_equal_generic :: proc(a, b: $T) -> bool where intrinsics.type_is_numeric(T) {
	return a <= b
}

// Overload sets

equals :: proc {
	equals_generic,
	equals_date,
	equals_time,
	equals_datetime,
}

less :: proc {
	less_generic,
	less_date,
	less_time,
	less_datetime,
}

less_equal :: proc {
	less_equal_generic,
	less_equal_date,
	less_equal_time,
	less_equal_datetime,
}

// --- Date ---

equals_date :: proc(a, b: Date) -> bool {
	return date_compare(a, b) == 0
}

less_date :: proc(a, b: Date) -> bool {
	return date_compare(a, b) < 0
}

less_equal_date :: proc(a, b: Date) -> bool {
	return date_compare(a, b) <= 0
}

// --- Time ---

equals_time :: proc(a, b: Time) -> bool {
	return time_compare(a, b) == 0
}

less_time :: proc(a, b: Time) -> bool {
	return time_compare(a, b) < 0
}

less_equal_time :: proc(a, b: Time) -> bool {
	return time_compare(a, b) <= 0
}

// --- Datetime ---

equals_datetime :: proc(a, b: Datetime) -> bool {
	return datetime_compare(a, b) == 0
}

less_datetime :: proc(a, b: Datetime) -> bool {
	return datetime_compare(a, b) < 0
}

less_equal_datetime :: proc(a, b: Datetime) -> bool {
	return datetime_compare(a, b) <= 0
}

// ------------------------------------------------------------
// Binary search helpers
// ------------------------------------------------------------

binary_search :: proc(data: []$T, key: T) -> int {
	lo := 0
	hi := len(data) - 1

	for lo <= hi {
		mid := (lo + hi) / 2

		if equals(data[mid], key) {
			return mid
		}
		if less(data[mid], key) {
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}

	return -1
}

lower_bound :: proc(data: []$T, key: T) -> int {
	lo := 0
	hi := len(data)

	for lo < hi {
		mid := (lo + hi) / 2
		if less(data[mid], key) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}

	return lo
}

upper_bound :: proc(data: []$T, key: T) -> int {
	lo := 0
	hi := len(data)

	for lo < hi {
		mid := (lo + hi) / 2
		if less_equal(data[mid], key) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}

	return lo
}

// ------------------------------------------------------------
// LOC (scalar)
// ------------------------------------------------------------

loc :: proc(df: ^DataFrame, key: $T) -> DataFrame {
	if !df.has_index {
		panic("loc: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	// 1. binary search
	i := binary_search(data, key)
	if i >= 0 {
		return df_row(df, i)
	}

	// 2. fallback linear scan
	for j in 0 ..< df.rows {
		if equals(data[j], key) {
			return df_row(df, j)
		}
	}

	panic("loc: key not found")
}

// ------------------------------------------------------------
// LOC (range)
// ------------------------------------------------------------

loc_range :: proc(df: ^DataFrame, start: $T, stop: T) -> DataFrame {
	if !df.has_index {
		panic("loc_range: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	i0 := lower_bound(data, start)
	i1 := upper_bound(data, stop)

	return df_slice(df, i0, i1)
}

// ------------------------------------------------------------
// Row / slice extraction
// ------------------------------------------------------------

df_row :: proc(df: ^DataFrame, i: int) -> DataFrame {
	out := dataframe_new(context.allocator)
	out.rows = 1

	for &col in df.columns {
		dst := column_new(col.name, col.type, 1, context.allocator)

		#partial switch col.type {
		case .Int:
			src := cast([^]int)col.data
			dst_data := cast([^]int)dst.data
			dst_data[0] = src[i]

		case .Float:
			src := cast([^]f64)col.data
			dst_data := cast([^]f64)dst.data
			dst_data[0] = src[i]

		case .String:
			src := cast([^]string)col.data
			dst_data := cast([^]string)dst.data
			dst_data[0] = src[i]

		case .Bool:
			src := cast([^]bool)col.data
			dst_data := cast([^]bool)dst.data
			dst_data[0] = src[i]

		case .Date:
			src := cast([^]Date)col.data
			dst_data := cast([^]Date)dst.data
			dst_data[0] = src[i]

		case .Datetime:
			src := cast([^]Datetime)col.data
			dst_data := cast([^]Datetime)dst.data
			dst_data[0] = src[i]
		}

		dst.len = 1
		add_column(&out, dst)
	}

	return out
}

df_slice :: proc(df: ^DataFrame, i0, i1: int) -> DataFrame {
	i0 := i0
	i1 := i1
	if i0 < 0 do i0 = 0
	if i1 > df.rows do i1 = df.rows
	if i1 <= i0 do return dataframe_new(context.allocator)

	out := dataframe_new(context.allocator)
	out.rows = i1 - i0

	for &col in df.columns {
		dst := column_new(col.name, col.type, out.rows, context.allocator)

		#partial switch col.type {
		case .Int:
			src := cast([^]int)col.data
			dst_data := cast([^]int)dst.data
			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]

		case .Float:
			src := cast([^]f64)col.data
			dst_data := cast([^]f64)dst.data
			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]

		case .String:
			src := cast([^]string)col.data
			dst_data := cast([^]string)dst.data
			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]

		case .Bool:
			src := cast([^]bool)col.data
			dst_data := cast([^]bool)dst.data
			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]

		case .Date:
			src := cast([^]Date)col.data
			dst_data := cast([^]Date)dst.data
			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]

		case .Datetime:
			src := cast([^]Datetime)col.data
			dst_data := cast([^]Datetime)dst.data
			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]
		}

		dst.len = out.rows
		add_column(&out, dst)
	}

	return out
}
