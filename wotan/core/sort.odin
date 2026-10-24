package core

import "core:fmt"
import "core:mem"
import "core:sort"
import "core:strings"

IntSorter :: struct {
	idx:        ^[]int,
	data:       [^]int,
	descending: bool,
}

FloatSorter :: struct {
	idx:        ^[]int,
	data:       [^]f64,
	descending: bool,
}

StringSorter :: struct {
	idx:        ^[]int,
	data:       [^]string,
	descending: bool,
}

BoolSorter :: struct {
	idx:        ^[]int,
	data:       [^]bool,
	descending: bool,
}

DateSorter :: struct {
	idx:        ^[]int,
	data:       [^]Date,
	descending: bool,
}
TimeSorter :: struct {
	idx:        ^[]int,
	data:       [^]Time,
	descending: bool,
}
DatetimeSorter :: struct {
	idx:        ^[]int,
	data:       [^]Datetime,
	descending: bool,
}


// Copy of df with rows ordered by one column (stable; nulls / NaN last)
dataframe_sort :: proc(
	df: ^DataFrame,
	column_name: string,
	descending: bool,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	return dataframe_sort_by(df, []string{column_name}, []bool{descending}, allocator)
}

// Copy of df with rows ordered by several columns (lexicographic, stable;
// nulls / NaN last in every key). `descending` may be shorter than `columns`
// (missing entries are ascending).
dataframe_sort_by :: proc(
	df: ^DataFrame,
	columns: []string,
	descending: []bool,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	keys := make([]^Column, len(columns), context.temp_allocator)
	defer delete(keys, context.temp_allocator)
	for name, i in columns {
		keys[i] = column(df, name)
	}
	order := _sort_order(keys, descending, df.rows, context.temp_allocator)
	defer delete(order, context.temp_allocator)

	out := df_take(df, order, allocator)
	if out.index != nil { 	// sorted frames are numbered afresh
		delete(out.index, allocator)
		out.index = nil
	}
	return out
}

// Stable row order of one column (nulls / NaN last)
column_sort_order :: proc(
	col: ^Column,
	descending: bool,
	allocator := context.allocator,
) -> []int {
	keys := []^Column{col}
	return _sort_order(keys, []bool{descending}, col.len, allocator)
}

@(private = "file")
_Sort_Ctx :: struct {
	keys: []^Column,
	desc: []bool,
}

@(private = "file")
_row_less :: proc(ctx: ^_Sort_Ctx, a, b: int) -> bool {
	for col, k in ctx.keys {
		c := column_compare_rows(col, a, b)
		if c == 0 {continue}
		// nulls stay last in both directions: only flip value comparisons
		if k < len(ctx.desc) &&
		   ctx.desc[k] &&
		   !is_null(col, a) &&
		   !is_null(col, b) &&
		   !_is_nan(col, a) &&
		   !_is_nan(col, b) {
			c = -c
		}
		return c < 0
	}
	return false
}

@(private = "file")
_is_nan :: proc(col: ^Column, r: int) -> bool {
	if col.type != .Float {return false}
	v, _ := column_at_float(col, r)
	return v != v
}

// Merge sort of row indices 0..<n (stable)
@(private = "file")
_sort_order :: proc(keys: []^Column, desc: []bool, n: int, allocator: mem.Allocator) -> []int {
	order := make([]int, n, allocator)
	for i in 0 ..< n {order[i] = i}
	if n < 2 || len(keys) == 0 {return order}
	ctx := _Sort_Ctx{keys, desc}
	tmp := make([]int, n, context.temp_allocator)
	defer delete(tmp, context.temp_allocator)
	width := 1
	src, dst := order, tmp
	for width < n {
		for lo := 0; lo < n; lo += 2 * width {
			mid := min(lo + width, n)
			hi := min(lo + 2 * width, n)
			i, j, k := lo, mid, lo
			for i < mid && j < hi {
				if _row_less(&ctx, src[j], src[i]) {
					dst[k] = src[j]
					j += 1
				} else {
					dst[k] = src[i]
					i += 1
				}
				k += 1
			}
			for i < mid {dst[k] = src[i]; i += 1; k += 1}
			for j < hi {dst[k] = src[j]; j += 1; k += 1}
		}
		src, dst = dst, src
		width *= 2
	}
	if raw_data(src) != raw_data(order) {copy(order, src)}
	return order
}


make_sorter :: proc {
	make_int_sorter,
	make_float_sorter,
	make_bool_sorter,
	make_string_sorter,
	make_date_sorter,
	make_time_sorter,
	make_datetime_sorter,
}

make_generic_sorter :: proc(t: ^$T) -> sort.Interface {
	return sort.Interface{collection = rawptr(t), len = proc(it: sort.Interface) -> int {
			s := (^T)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^T)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				s.data[ai] > s.data^[aj]
			}
			return s.data^[ai] > s.data^[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^T)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}

make_int_sorter :: proc(s: ^IntSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^IntSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^IntSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data[ai] > s.data[aj]
			}
			return s.data[ai] < s.data[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^IntSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}
make_float_sorter :: proc(s: ^FloatSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^FloatSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^FloatSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data[ai] > s.data[aj]
			}
			return s.data[ai] < s.data[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^FloatSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}


make_string_sorter :: proc(s: ^StringSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^StringSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^StringSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			a := s.data[ai]
			b := s.data[aj]
			cmp := strings.compare(a, b)
			if s.descending {
				return cmp > 0
			}
			return cmp < 0
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^StringSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}

// --- Bool ---

make_bool_sorter :: proc(s: ^BoolSorter) -> sort.Interface {
	return sort.Interface {
		collection = rawptr(s),
		len = proc(it: sort.Interface) -> int {
			s := (^BoolSorter)(it.collection)
			return len(s.idx^)
		},
		less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^BoolSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			a := s.data[ai]
			b := s.data[aj]
			if s.descending {
				// true > false
				return a && !b
			}
			// false < true
			return !a && b
		},
		swap = proc(it: sort.Interface, i, j: int) {
			s := (^BoolSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		},
	}
}

// --- Date ---

make_date_sorter :: proc(s: ^DateSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^DateSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^DateSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			a := s.data[ai]
			b := s.data[aj]
			cmp := date_compare(a, b)
			if s.descending {
				return cmp > 0
			}
			return cmp < 0
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^DateSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}

// --- Time ---

make_time_sorter :: proc(s: ^TimeSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^TimeSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^TimeSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			a := s.data[ai]
			b := s.data[aj]
			cmp := time_compare(a, b)
			if s.descending {
				return cmp > 0
			}
			return cmp < 0
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^TimeSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}

// --- Datetime ---

make_datetime_sorter :: proc(s: ^DatetimeSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^DatetimeSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^DatetimeSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			a := s.data[ai]
			b := s.data[aj]
			cmp := datetime_compare(a, b)
			if s.descending {
				return cmp > 0
			}
			return cmp < 0
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^DatetimeSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}
