package core

import "core:fmt"
import "core:mem"
import "core:sort"
import "core:strings"

IntSorter :: struct {
	idx:        ^[]int,
	data:       ^[]int,
	descending: bool,
}

FloatSorter :: struct {
	idx:        ^[]int,
	data:       ^[]f64,
	descending: bool,
}

StringSorter :: struct {
	idx:        ^[]int,
	data:       ^[]string,
	descending: bool,
}

BoolSorter :: struct {
	idx:        ^[]int,
	data:       ^[]bool,
	descending: bool,
}

DateSorter :: struct {
	idx:        ^[]int,
	data:       ^[]Date,
	descending: bool,
}
TimeSorter :: struct {
	idx:        ^[]int,
	data:       ^[]Date,
	descending: bool,
}
DatetimeSorter :: struct {
	idx:        ^[]int,
	data:       ^[]Datetime,
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
	for i in 0 ..< df.rows {
		idx[i] = i
	}

	// 2. Sort index array based on column type
	#partial switch col.type {

	case .Int:
		sorter := IntSorter {
			idx        = &idx,
			data       = cast(^[]int)col.data,
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case .Float:
		sorter := FloatSorter {
			idx        = &idx,
			data       = cast(^[]f64)col.data,
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case .String:
		sorter := StringSorter {
			idx        = &idx,
			data       = cast(^[]string)col.data,
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case .Bool:
		sorter := BoolSorter {
			idx        = &idx,
			data       = cast(^[]bool)col.data,
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))

	case .Date:
		sorter := DateSorter {
			idx        = &idx,
			data       = cast(^[]Date)col.data,
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))
	case .Time:
		sorter := TimeSorter {
			idx        = &idx,
			data       = cast(^[]Date)col.data,
			descending = descending,
		}
		sort.sort(make_sorter(&sorter))


	case .Datetime:
		sorter := DatetimeSorter {
			idx        = &idx,
			data       = cast(^[]Datetime)col.data,
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

make_int_sorter :: proc(s: ^IntSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^IntSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^IntSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data^[ai] > s.data^[aj]
			}
			return s.data^[ai] < s.data^[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^IntSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}
make_float_sorter :: proc(s: ^FloatSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^IntSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^IntSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data^[ai] > s.data^[aj]
			}
			return s.data^[ai] < s.data^[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^IntSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}

make_string_sorter :: proc(s: ^StringSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^IntSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^IntSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data^[ai] > s.data^[aj]
			}
			return s.data^[ai] < s.data^[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^IntSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}
make_bool_sorter :: proc(s: ^BoolSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^IntSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^IntSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data^[ai] > s.data^[aj]
			}
			return s.data^[ai] < s.data^[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^IntSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}
make_date_sorter :: proc(s: ^DateSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^IntSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^IntSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data^[ai] > s.data^[aj]
			}
			return s.data^[ai] < s.data^[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^IntSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}
make_time_sorter :: proc(s: ^TimeSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^IntSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^IntSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data^[ai] > s.data^[aj]
			}
			return s.data^[ai] < s.data^[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^IntSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}
make_datetime_sorter :: proc(s: ^DatetimeSorter) -> sort.Interface {
	return sort.Interface{collection = rawptr(s), len = proc(it: sort.Interface) -> int {
			s := (^IntSorter)(it.collection)
			return len(s.idx^)
		}, less = proc(it: sort.Interface, i, j: int) -> bool {
			s := (^IntSorter)(it.collection)
			ai := s.idx^[i]
			aj := s.idx^[j]
			if s.descending {
				return s.data^[ai] > s.data^[aj]
			}
			return s.data^[ai] < s.data^[aj]
		}, swap = proc(it: sort.Interface, i, j: int) {
			s := (^IntSorter)(it.collection)
			s.idx^[i], s.idx^[j] = s.idx^[j], s.idx^[i]
		}}
}
