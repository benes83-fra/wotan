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


dataframe_sort :: proc(
	df: ^DataFrame,
	column_name: string,
	descending: bool,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {

	col := column(df, column_name)
	if col == nil {
		panic("dataframe_sort: column not found")
	}

	// 1. Build index array
	idx := make([]int, df.rows, allocator)
	defer delete(idx)
	for i in 0 ..< df.rows {
		idx[i] = i
	}

	// 2. Sort index array based on column type
	#partial switch col.type {

	case .Int:
		sorter := IntSorter {
			idx        = &idx,
			data       = ([^]int)(col.data),
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case .Float:
		sorter := FloatSorter {
			idx        = &idx,
			data       = ([^]f64)(col.data),
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case .String:
		sorter := StringSorter {
			idx        = &idx,
			data       = ([^]string)(col.data),
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case .Bool:
		sorter := BoolSorter {
			idx        = &idx,
			data       = ([^]bool)(col.data),
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case .Date:
		sorter := DateSorter {
			idx        = &idx,
			data       = ([^]Date)(col.data),
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))
	case .Time:
		sorter := TimeSorter {
			idx        = &idx,
			data       = ([^]Time)(col.data),
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))


	case .Datetime:
		sorter := DatetimeSorter {
			idx        = &idx,
			data       = ([^]Datetime)(col.data),
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case:
		panic("dataframe_sort: unsupported column type")
	}


	// 3. Build sorted DataFrame (copy-based)
	out := dataframe_new(allocator)
	out.rows = df.rows

	for &src_col in df.columns {
		dst_col := column_new(src_col.name, src_col.type, df.rows, allocator)

		#partial switch src_col.type {
		case .Int:
			src := cast([^]int)src_col.data
			dst := cast([^]int)dst_col.data
			for i in 0 ..< df.rows do dst[i] = src[idx[i]]

		case .Float:
			src := cast([^]f64)src_col.data
			dst := cast([^]f64)dst_col.data
			for i in 0 ..< df.rows do dst[i] = src[idx[i]]

		case .String:
			src := cast([^]string)src_col.data
			dst := cast([^]string)dst_col.data
			for i in 0 ..< df.rows do dst[i] = src[idx[i]]

		case .Bool:
			src := cast([^]bool)src_col.data
			dst := cast([^]bool)dst_col.data
			for i in 0 ..< df.rows do dst[i] = src[idx[i]]

		case .Date:
			src := cast([^]Date)src_col.data
			dst := cast([^]Date)dst_col.data
			for i in 0 ..< df.rows do dst[i] = src[idx[i]]

		case .Datetime:
			src := cast([^]Datetime)src_col.data
			dst := cast([^]Datetime)dst_col.data
			for i in 0 ..< df.rows do dst[i] = src[idx[i]]
		}
		dst_col.len = df.rows
		add_column(&out, dst_col)
	}

	return out
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
