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


loc :: proc {
	loc_single,
	loc_range,
	loc_many,
	loc_mask,
}


// ------------------------------------------------------------
// LOC (scalar)
// ------------------------------------------------------------


loc_single :: proc(df: ^DataFrame, key: $T) -> DataFrame {
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

    case .Time:
      src := cast ([^]Time)col.data
      dst_data := cast ([^]Time) dst.data
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

// df_slice :: proc(df: ^DataFrame, i0, i1: int) -> DataFrame {
// 	i0 := i0
// 	i1 := i1
// 	if i0 < 0 do i0 = 0
// 	if i1 > df.rows do i1 = df.rows
// 	if i1 <= i0 do return dataframe_new(context.allocator)
//
// 	out := dataframe_new(context.allocator)
// 	out.rows = i1 - i0
//
// 	for &col in df.columns {
// 		dst := column_new(col.name, col.type, out.rows, context.allocator)
//
// 		#partial switch col.type {
// 		case .Int:
// 			src := cast([^]int)col.data
// 			dst_data := cast([^]int)dst.data
// 			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]
//
// 		case .Float:
// 			src := cast([^]f64)col.data
// 			dst_data := cast([^]f64)dst.data
// 			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]
//
// 		case .String:
// 			src := cast([^]string)col.data
// 			dst_data := cast([^]string)dst.data
// 			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]
//
// 		case .Bool:
// 			src := cast([^]bool)col.data
// 			dst_data := cast([^]bool)dst.data
// 			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]
//
// 		case .Date:
// 			src := cast([^]Date)col.data
// 			dst_data := cast([^]Date)dst.data
// 			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]
//
// 		case .Datetime:
// 			src := cast([^]Datetime)col.data
// 			dst_data := cast([^]Datetime)dst.data
// 			for i in 0 ..< out.rows do dst_data[i] = src[i0 + i]
// 		}
//
// 		dst.len = out.rows
// 		add_column(&out, dst)
// 	}
//
// 	return out
// }
df_slice :: proc(df: ^DataFrame, i0, i1: int) -> DataFrame {
	// normalize bounds
	i0 := i0
	i1 := i1
	if i0 < 0 do i0 = 0
	if i1 > df.rows do i1 = df.rows
	if i1 <= i0 do return dataframe_new(context.allocator)

	out := dataframe_new(context.allocator)
	out.rows = i1 - i0
	out.index_column = df.index_column
	out.has_index = df.has_index

	for &col in df.columns {
		view_col := col

		view_col.is_view = true
		view_col.orig = &col

		view_col.offset = col.offset + i0 if col.is_view else i0
		view_col.len = out.rows
		view_col.capacity = view_col.len

		// data + nulls are shared; arena/allocator unchanged
		// data stays pointing to the original buffer
		if len(col.nulls) > 0 {
			start := view_col.offset
			stop := view_col.offset + view_col.len
			view_col.nulls = col.nulls[start:stop]
		}

		add_column(&out, view_col)
	}

	return out
}
// ------------------------------------------------------------
// LOC MANY (list of keys)
// ------------------------------------------------------------
// ------------------------------------------------------------
// LOC MANY (list of keys)
// ------------------------------------------------------------

loc_many :: proc(df: ^DataFrame, keys: []$T) -> DataFrame {
	if !df.has_index {
		panic("loc_many: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	// 1) Collect matching row indices
	indices := make([dynamic]int, 0, len(keys), context.allocator)
	defer delete(indices)

	for key in keys {
		// Try binary search first
		idx := binary_search(data, key)

		if idx >= 0 {
			append(&indices, idx)
			continue
		}

		// Fallback: linear scan (for unsorted index)
		for i in 0 ..< df.rows {
			if equals(data[i], key) {
				append(&indices, i)
			}
		}
	}

	// No matches → empty DataFrame
	if len(indices) == 0 {
		return dataframe_new(context.allocator)
	}

	// 2) Build result DataFrame by copying rows at collected indices
	out := dataframe_new(context.allocator)
	out.rows = len(indices)

	for &col in df.columns {
		dst := column_new(col.name, col.type, out.rows, context.allocator)

		#partial switch col.type {
		case .Int:
			src := cast([^]int)col.data
			dst_data := cast([^]int)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = src[indices[i]]
			}

		case .Float:
			src := cast([^]f64)col.data
			dst_data := cast([^]f64)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = src[indices[i]]
			}

		case .String:
			src := cast([^]string)col.data
			dst_data := cast([^]string)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = src[indices[i]]
			}

		case .Bool:
			src := cast([^]bool)col.data
			dst_data := cast([^]bool)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = src[indices[i]]
			}

		case .Date:
			src := cast([^]Date)col.data
			dst_data := cast([^]Date)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = src[indices[i]]
			}
		case .Time:
			src := cast([^]Time)col.data
			dst_data := cast([^]Time)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = src[indices[i]]
			}
		case .Datetime:
			src := cast([^]Datetime)col.data
			dst_data := cast([^]Datetime)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = src[indices[i]]
			}
		}

		dst.len = out.rows
		add_column(&out, dst)
	}

	return out
}
// ------------------------------------------------------------
// LOC_FROM (open-ended slice from a start label)
// ------------------------------------------------------------

loc_from :: proc(df: ^DataFrame, start: $T) -> DataFrame {
	if !df.has_index {
		panic("loc_from: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	// left boundary
	i0 := lower_bound(data, start)

	// open-ended: until end
	i1 := df.rows

	return df_slice(df, i0, i1)
}
// ------------------------------------------------------------
// LOC_UNTIL (open-ended slice until a stop label)
// ------------------------------------------------------------

loc_until :: proc(df: ^DataFrame, stop: $T) -> DataFrame {
	if !df.has_index {
		panic("loc_until: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	// open-ended: from beginning
	i0 := 0

	// right boundary
	i1 := upper_bound(data, stop)

	return df_slice(df, i0, i1)
}
// ------------------------------------------------------------
// LOC_MASK (boolean mask selection)
// ------------------------------------------------------------

loc_mask :: proc(df: ^DataFrame, mask: []bool) -> DataFrame {
	if len(mask) != df.rows {
		panic("loc_mask: mask length does not match DataFrame rows")
	}

	// 1) Collect indices where mask[i] == true
	indices := make([dynamic]int, 0, df.rows, context.allocator)
	defer delete(indices)

	for i in 0 ..< df.rows {
		if mask[i] {
			append(&indices, i)
		}
	}

	// No matches → empty DataFrame
	if len(indices) == 0 {
		return dataframe_new(context.allocator)
	}

	// 2) Build output DataFrame
	out := dataframe_new(context.allocator)
	out.rows = len(indices)

	for &col in df.columns {
		dst := column_new(col.name, col.type, out.rows, context.allocator)

		#partial switch col.type {
		case .Int:
			src := cast([^]int)col.data
			dst_data := cast([^]int)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[indices[j]]

		case .Float:
			src := cast([^]f64)col.data
			dst_data := cast([^]f64)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[indices[j]]

		case .String:
			src := cast([^]string)col.data
			dst_data := cast([^]string)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[indices[j]]

		case .Bool:
			src := cast([^]bool)col.data
			dst_data := cast([^]bool)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[indices[j]]

		case .Date:
			src := cast([^]Date)col.data
			dst_data := cast([^]Date)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[indices[j]]

    case .Time:
			src := cast([^]Time)col.data
			dst_data := cast([^]Time)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[indices[j]]
		
    case .Datetime:
			src := cast([^]Datetime)col.data
			dst_data := cast([^]Datetime)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[indices[j]]
		}

		dst.len = out.rows
		add_column(&out, dst)
	}

	return out
}
// ------------------------------------------------------------
// ILOC (scalar integer indexing)
// ------------------------------------------------------------

iloc :: proc {
	iloc_single,
	iloc_range,
	iloc_many,
}


iloc_single :: proc(df: ^DataFrame, i: int) -> DataFrame {
	if i < 0 || i >= df.rows {
		panic("iloc: index out of bounds")
	}
	return df_row(df, i)
}
// ------------------------------------------------------------
// ILOC_RANGE (integer slice [i0, i1))
// ------------------------------------------------------------

iloc_range :: proc(df: ^DataFrame, i0: int, i1: int) -> DataFrame {
	i0 := i0
	i1 := i1
	if i0 < 0 do i0 = 0
	if i1 > df.rows do i1 = df.rows
	if i1 <= i0 do return dataframe_new(context.allocator)

	return df_slice(df, i0, i1)
}
// ------------------------------------------------------------
// ILOC_MANY (list of integer indices)
// ------------------------------------------------------------

iloc_many :: proc(df: ^DataFrame, idxs: []int) -> DataFrame {
	// Validate indices
	for i in idxs {
		if i < 0 || i >= df.rows {
			panic("iloc_many: index out of bounds")
		}
	}

	// Build output
	out := dataframe_new(context.allocator)
	out.rows = len(idxs)

	for &col in df.columns {
		dst := column_new(col.name, col.type, out.rows, context.allocator)

		#partial switch col.type {
		case .Int:
			src := cast([^]int)col.data
			dst_data := cast([^]int)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[idxs[j]]

		case .Float:
			src := cast([^]f64)col.data
			dst_data := cast([^]f64)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[idxs[j]]

		case .String:
			src := cast([^]string)col.data
			dst_data := cast([^]string)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[idxs[j]]

		case .Bool:
			src := cast([^]bool)col.data
			dst_data := cast([^]bool)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[idxs[j]]

		case .Date:
			src := cast([^]Date)col.data
			dst_data := cast([^]Date)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[idxs[j]]

		case .Time:
			src := cast([^]Time)col.data
			dst_data := cast([^]Time)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[idxs[j]]

		case .Datetime:
			src := cast([^]Datetime)col.data
			dst_data := cast([^]Datetime)dst.data
			for j in 0 ..< out.rows do dst_data[j] = src[idxs[j]]
		}

		dst.len = out.rows
		add_column(&out, dst)
	}

	return out
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
