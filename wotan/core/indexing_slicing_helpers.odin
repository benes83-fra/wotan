package core

import "base:intrinsics"
import "core:mem"

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


// Reorder a column to rows `index_vec` (owned copy: strings cloned, nulls kept)
reorder_column_inplace :: proc(col: ^Column, index_vec: []int) {
	allocator := col.allocator if col.allocator.procedure != nil else context.allocator
	tmp := column_take(col, index_vec, allocator)
	destroy_column(col)
	col^ = tmp
}

// Value at logical row i (view-aware; null rows give the zero value)
col_get :: proc(col: ^Column, i: int, $T: typeid) -> T {
	c, r := column_resolve(col, i)
	return (cast([^]T)c.data)[r]
}

// Copy rows `indices` of col into the preallocated dst (nulls kept; string
// bytes are shared with the source, use column_take for an owned copy)
copy_column_values :: proc(df: ^DataFrame, col: ^Column, dst: ^Column, indices: []int) {
	_copy_rows(col, dst, indices)
}

// Value at logical row i of a DataFrame column (df.index only labels rows)
col_get_df :: proc(df: ^DataFrame, col: ^Column, i: int, $T: typeid) -> T {
	return col_get(col, i, T)
}

@(private = "file")
_copy_rows :: proc(src: ^Column, dst: ^Column, mapping: []int) {
	if dst.nulls == nil || len(dst.nulls) < len(mapping) {
		if dst.allocator.procedure == nil {dst.allocator = context.allocator}
		dst.nulls = make([]bool, max(dst.capacity, len(mapping)), dst.allocator)
	}
	size := type_size(src.type)
	for m, i in mapping {
		if m < 0 || is_null(src, m) {
			dst.nulls[i] = true
			continue
		}
		dst.nulls[i] = false
		c, r := column_resolve(src, m)
		mem.copy(rawptr(uintptr(dst.data) + uintptr(i * size)), rawptr(uintptr(c.data) + uintptr(r * size)), size)
	}
}
insert_column_front :: proc(df: ^DataFrame, col: Column) {
	// shift existing columns right
	append(&df.columns, col)
	for i := len(df.columns) - 1; i > 0; i -= 1 {
		df.columns[i] = df.columns[i - 1]
	}
	df.columns[0] = col
  if df.name_to_index != nil {
    delete (df.name_to_index)
  }
	// rebuild name_to_index
	df.name_to_index = make(map[string]int)
	for c, i in df.columns {
		df.name_to_index[c.name] = i
	}
}


// dst[i] = src[mapping[i]] into a preallocated dst (view-aware, nulls kept;
// string bytes are shared with the source)
copy_column_direct :: proc(src: ^Column, dst: ^Column, mapping: []int) {
	_copy_rows(src, dst, mapping)
}

// As copy_column_direct; mapping[i] < 0 gives a null
copy_column_direct_or_null :: proc(src: ^Column, dst: ^Column, mapping: []int) {
	_copy_rows(src, dst, mapping)
}
