package wotan

import "core:fmt"
import "core:mem"

JoinKind :: enum {
	Inner,
	Left,
	Right,
	Outer,
}

get_string :: proc(c: ^Column, row: int) -> string {
	base := uintptr(c.data) + uintptr(row * size_of(string))
	return (cast(^string)base)^
}

get_int :: proc(c: ^Column, row: int) -> int {
	base := uintptr(c.data) + uintptr(row * size_of(int))
	return (cast(^int)base)^
}

get_float :: proc(c: ^Column, row: int) -> f64 {
	base := uintptr(c.data) + uintptr(row * size_of(f64))
	return (cast(^f64)base)^
}

get_date :: proc(c: ^Column, row: int) -> Date {
	base := uintptr(c.data) + uintptr(row * size_of(Date))
	return (cast(^Date)base)^
}
get_time :: proc(c: ^Column, row: int) -> Time {
	base := uintptr(c.data) + uintptr(row * size_of(Time))
	return (cast(^Time)base)^
}

get_datetime :: proc(c: ^Column, row: int) -> Datetime {
	base := uintptr(c.data) + uintptr(row * size_of(Datetime))
	return (cast(^Datetime)base)^
}
get_bool :: proc(c: ^Column, row: int) -> bool {
	base := uintptr(c.data) + uintptr(row * size_of(bool))
	return (cast(^bool)base)^
}


get_key :: proc($T: typeid, c: ^Column, row: int) -> T {
	base := uintptr(c.data) + uintptr(row * size_of(T))
	return (cast(^T)base)^
}


JoinIndex :: struct($T: typeid) {
	key:   string,
	col:   ^Column,
	table: map[T]int, // key → row indices in right DF
}

build_join_index :: proc(
	$T: typeid,
	right: ^DataFrame,
	key: string,
	allocator: mem.Allocator = context.allocator,
) -> JoinIndex(T) {
	idx: JoinIndex(T)
	idx.key = key
	idx.col = column(right, key)
	idx.table = make(map[T]int, allocator) // if you later want arena-backed, change allocator here

	for r in 0 ..< right.rows {
		k := get_key(T, idx.col, r)
		idx.table[k] = r
	}

	return idx
}

join_inner :: proc(
	$T: typeid,
	left: ^DataFrame,
	right: ^DataFrame,
	key: string,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	result := dataframe_new()

	// 1) left columns
	for i in 0 ..< len(left.columns) {
		src := &left.columns[i]
		dst := column_new(src.name, src.type, 0) // capacity will grow as we append
		add_column(&result, dst)
	}

	// 2) right columns except key
	for i in 0 ..< len(right.columns) {
		src := &right.columns[i]
		if src.name == key {
			continue
		}
		dst := column_new(src.name, src.type, 0)
		add_column(&result, dst)
	}

	// 3) perform the join
	idx := build_join_index(T, right, key, allocator)
  left_key_col := column(left, key)
	for li in 0 ..< left.rows {
		k := get_key(T, left_key_col, li)

		ri, ok := idx.table[k]
		if !ok {
			continue // inner join: skip non-matches
		}

		emit_joined_row(&result, left, right, li, ri, key)
	}


	if len(result.columns) > 0 {
		result.rows = result.columns[0].len
	} else {
		result.rows = 0
	}

	return result
}

emit_joined_row :: proc(
	out: ^DataFrame,
	left: ^DataFrame,
	right: ^DataFrame,
	li: int,
	ri: int,
	key: string,
	allocator: mem.Allocator = context.allocator,
) {
	// left columns first
	for ci in 0 ..< len(left.columns) {
		src := &left.columns[ci]
		dst := &out.columns[ci]
		copy_value(dst, src, li)
	}

	// right columns (skipping key)
	out_offset := len(left.columns)
	out_idx := out_offset
	for ci in 0 ..< len(right.columns) {
		src := &right.columns[ci]
		if src.name == key {
			continue
		}
		dst := &out.columns[out_idx]
		copy_value(dst, src, ri)
		out_idx += 1
	}
}

copy_value :: proc(dst, src: ^Column, row: int) {
	#partial switch src.type {
	case .Int:
		append_int(dst, get_int(src, row))
	case .Float:
		append_float(dst, get_float(src, row))
	case .Date:
		append_date(dst, get_date(src, row))
	case .Time:
		append_time(dst, get_time(src, row))
	case .Datetime:
		append_datetime(dst, get_datetime(src, row))
	case .Bool:
		append_bool(dst, get_bool(src, row))
	case .String:
		append_string(dst, get_string(src, row))
	}
}

join :: proc(
	$T: typeid,
	left: ^DataFrame,
	right: ^DataFrame,
	key: string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	switch kind {
	case .Inner:
		fmt.println("Joining")
		return join_inner(T, left, right, key, allocator)
	case .Left:
	// TODO
	case .Right:
	// TODO
	case .Outer:
	// TODO
	}
	return dataframe_new()
}
