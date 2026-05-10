package core

import "base:intrinsics"

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


reorder_column_inplace :: proc(col: ^Column, index_vec: []int) {
	tmp := column_new(col.name, col.type, col.len, col.allocator)
	copy_column_direct(col, &tmp, index_vec)
	tmp.len = col.len
	destroy_column(col)
	col^ = tmp
}

col_get :: proc(col: ^Column, i: int, $T: typeid) -> T {
	src := cast([^]T)col.data
	row: int
	if col.is_view {
		row = col.offset + i
	} else {
		row = i
	}

	return src[row]
}

copy_column_values :: proc(df: ^DataFrame, col: ^Column, dst: ^Column, indices: []int) {
	#partial switch col.type {
	case .Int:
		dst_data := cast([^]int)dst.data
		for i in 0 ..< len(indices) {
			dst_data[i] = col_get_df(df, col, indices[i], int)
		}

	case .Float:
		dst_data := cast([^]f64)dst.data
		for i in 0 ..< len(indices) {
			dst_data[i] = col_get_df(df, col, indices[i], f64)
		}

	case .String:
		dst_data := cast([^]string)dst.data
		for i in 0 ..< len(indices) {
			dst_data[i] = col_get_df(df, col, indices[i], string)
		}

	case .Bool:
		dst_data := cast([^]bool)dst.data
		for i in 0 ..< len(indices) {
			dst_data[i] = col_get_df(df, col, indices[i], bool)
		}

	case .Date:
		dst_data := cast([^]Date)dst.data
		for i in 0 ..< len(indices) {
			dst_data[i] = col_get_df(df, col, indices[i], Date)
		}

	case .Time:
		dst_data := cast([^]Time)dst.data
		for i in 0 ..< len(indices) {
			dst_data[i] = col_get_df(df, col, indices[i], Time)
		}

	case .Datetime:
		dst_data := cast([^]Datetime)dst.data
		for i in 0 ..< len(indices) {
			dst_data[i] = col_get_df(df, col, indices[i], Datetime)
		}
	}
}

col_get_df :: proc(df: ^DataFrame, col: ^Column, i: int, $T: typeid) -> T {
	src := cast([^]T)col.data

	// logical row -> physical row
	row: int
	if df.index != nil {
		// index-vector view
		row = df.index[i]
	} else if col.is_view {
		// contiguous column view
		row = col.offset + i
	} else {
		// plain materialized
		row = i
	}

	return src[row]
}
insert_column_front :: proc(df: ^DataFrame, col: Column) {
	// shift existing columns right
	append(&df.columns, col)
	for i := len(df.columns) - 1; i > 0; i -= 1 {
		df.columns[i] = df.columns[i - 1]
	}
	df.columns[0] = col

	// rebuild name_to_index
	df.name_to_index = make(map[string]int)
	for c, i in df.columns {
		df.name_to_index[c.name] = i
	}
}


copy_column_direct :: proc(src: ^Column, dst: ^Column, mapping: []int) {
	#partial switch src.type {
	case .Int:
		s := cast([^]int)src.data
		d := cast([^]int)dst.data
		for i in 0 ..< len(mapping) do d[i] = s[mapping[i]]

	case .Float:
		s := cast([^]f64)src.data
		d := cast([^]f64)dst.data
		for i in 0 ..< len(mapping) do d[i] = s[mapping[i]]

	case .String:
		s := cast([^]string)src.data
		d := cast([^]string)dst.data
		for i in 0 ..< len(mapping) do d[i] = s[mapping[i]]

	case .Bool:
		s := cast([^]bool)src.data
		d := cast([^]bool)dst.data
		for i in 0 ..< len(mapping) do d[i] = s[mapping[i]]

	case .Date:
		s := cast([^]Date)src.data
		d := cast([^]Date)dst.data
		for i in 0 ..< len(mapping) do d[i] = s[mapping[i]]

	case .Time:
		s := cast([^]Time)src.data
		d := cast([^]Time)dst.data
		for i in 0 ..< len(mapping) do d[i] = s[mapping[i]]

	case .Datetime:
		s := cast([^]Datetime)src.data
		d := cast([^]Datetime)dst.data
		for i in 0 ..< len(mapping) do d[i] = s[mapping[i]]
	}
}
copy_column_direct_or_null :: proc(src: ^Column, dst: ^Column, mapping: []int) {
	#partial switch src.type {
	case .Int:
		s := cast([^]int)src.data
		d := cast([^]int)dst.data
		for i in 0 ..< len(mapping) {
			if mapping[i] >= 0 {
				d[i] = s[mapping[i]]
			} else {
				dst.nulls[i] = true
			}
		}

	case .Float:
		s := cast([^]f64)src.data
		d := cast([^]f64)dst.data
		for i in 0 ..< len(mapping) {
			if mapping[i] >= 0 {
				d[i] = s[mapping[i]]
			} else {
				dst.nulls[i] = true
			}
		}

	case .String:
		s := cast([^]string)src.data
		d := cast([^]string)dst.data
		for i in 0 ..< len(mapping) {
			if mapping[i] >= 0 {
				d[i] = s[mapping[i]]
			} else {
				dst.nulls[i] = true
			}
		}

	case .Bool:
		s := cast([^]bool)src.data
		d := cast([^]bool)dst.data
		for i in 0 ..< len(mapping) {
			if mapping[i] >= 0 {
				d[i] = s[mapping[i]]
			} else {
				dst.nulls[i] = true
			}
		}

	case .Date:
		s := cast([^]Date)src.data
		d := cast([^]Date)dst.data
		for i in 0 ..< len(mapping) {
			if mapping[i] >= 0 {
				d[i] = s[mapping[i]]
			} else {
				dst.nulls[i] = true
			}
		}

	case .Time:
		s := cast([^]Time)src.data
		d := cast([^]Time)dst.data
		for i in 0 ..< len(mapping) {
			if mapping[i] >= 0 {
				d[i] = s[mapping[i]]
			} else {
				dst.nulls[i] = true
			}
		}

	case .Datetime:
		s := cast([^]Datetime)src.data
		d := cast([^]Datetime)dst.data
		for i in 0 ..< len(mapping) {
			if mapping[i] >= 0 {
				d[i] = s[mapping[i]]
			} else {
				dst.nulls[i] = true
			}
		}
	}
}
