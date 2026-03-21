package core

import "core:fmt"
import "core:mem"
import "core:strings"

JoinKind :: enum {
	Inner,
	Left,
	Right,
	Outer,
}


JoinRowKind :: enum {
	LeftOnly,
	RightOnly,
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

join_generic :: proc(
	$T: typeid,
	left: ^DataFrame,
	right: ^DataFrame,
	key: string,
	kind: JoinKind,
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
	matched_right := make([]bool, right.rows, allocator)
	left_key_col := column(left, key)
	for li in 0 ..< left.rows {
		k := get_key(T, left_key_col, li)

		ri, ok := idx.table[k]
		if ok {
			matched_right[ri] = true
			emit_joined_row(&result, left, right, li, ri, key)
		} else {
			if kind == .Inner {
				continue
			}
			if kind == .Left || kind == .Outer {
				emit_unmatched_row(&result, left, right, li, .LeftOnly, key, allocator)

			}
		}
	}

	if kind == .Right || kind == .Outer {
		for ri in 0 ..< right.rows {
			if !matched_right[ri] {
				emit_unmatched_row(&result, left, right, ri, .RightOnly, key, allocator)

			}
		}
	}
	if len(result.columns) > 0 {
		result.rows = result.columns[0].len
	} else {
		result.rows = 0
	}

	return result
}


emit_unmatched_row :: proc(
	out: ^DataFrame,
	left: ^DataFrame,
	right: ^DataFrame,
	row: int, // row index in whichever side is real
	kind: JoinRowKind,
	key: string,
	allocator: mem.Allocator = context.allocator,
) {
	// 1. LEFT side
	for ci in 0 ..< len(left.columns) {
		dst := &out.columns[ci]

		if kind == .LeftOnly {
			// real left row
			src := &left.columns[ci]
			copy_value(dst, src, row)
		} else {
			// right-only row → left side is NULL
			append_null(dst)
		}
	}

	// 2. RIGHT side
	out_offset := len(left.columns)
	out_idx := out_offset

	for ci in 0 ..< len(right.columns) {
		src := &right.columns[ci]
		if src.name == key {
			continue
		}

		dst := &out.columns[out_idx]

		if kind == .RightOnly {
			// real right row
			copy_value(dst, src, row)
		} else {
			// left-only row → right side is NULL
			append_null(dst)
		}

		out_idx += 1
	}
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

join_single :: proc(
	$T: typeid,
	left: ^DataFrame,
	right: ^DataFrame,
	key: string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	return join_generic(T, left, right, key, kind, allocator)
}


join :: proc(
	left: ^DataFrame,
	right: ^DataFrame,
	key: []string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	return join_generic_multi(left, right, key, kind, allocator)
}

copy_single_field :: proc(dst: rawptr, col: ^Column, row: int) {
	#partial switch col.type {
	case .Int:
		(cast(^int)dst)^ = get_int(col, row)
	case .Float:
		(cast(^f64)dst)^ = get_float(col, row)
	case .String:
		(cast(^string)dst)^ = get_string(col, row)
	case .Date:
		(cast(^Date)dst)^ = get_date(col, row)
	case .Time:
		(cast(^Time)dst)^ = get_time(col, row)
	case .Datetime:
		(cast(^Datetime)dst)^ = get_datetime(col, row)
	case .Bool:
		(cast(^bool)dst)^ = get_bool(col, row)
	}
}

Join_Multi_Index :: struct {
	bucket_head: map[string]int, // key → head node index, or -1
	rows:        [dynamic]int, // right row index per node
	next:        [dynamic]int, // next node index, or -1
}

join_multi_index_make :: proc(allocator: mem.Allocator) -> Join_Multi_Index {
	return Join_Multi_Index {
		bucket_head = make(map[string]int, allocator),
		rows = make([dynamic]int, 0, allocator),
		next = make([dynamic]int, 0, allocator),
	}
}

build_join_multi_index :: proc(
	right: ^DataFrame,
	key_cols: []^Column,
	allocator: mem.Allocator = context.allocator,
) -> Join_Multi_Index {
	idx := join_multi_index_make(allocator)

	for r in 0 ..< right.rows {
		k := make_composite_key_string(key_cols, r, allocator)

		// new node
		node_index := len(idx.rows)
		append(&idx.rows, r)
		append(&idx.next, -1)

		// link into bucket
		head, ok := idx.bucket_head[k]
		if ok {
			idx.next[node_index] = head
		}
		idx.bucket_head[k] = node_index
	}

	return idx
}


join_generic_multi :: proc(
	left: ^DataFrame,
	right: ^DataFrame,
	keys: []string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	result := dataframe_new()

	// 1) left columns
	for i in 0 ..< len(left.columns) {
		src := &left.columns[i]
		dst := column_new(src.name, src.type, 0)
		add_column(&result, dst)
	}

	// 2) right columns, skipping all key columns
	for i in 0 ..< len(right.columns) {
		src := &right.columns[i]

		skip := false
		for k in keys {
			if src.name == k {
				skip = true
				break
			}
		}
		if skip {
			continue
		}

		dst := column_new(src.name, src.type, 0)
		add_column(&result, dst)
	}

	// 3) key columns
	left_key_cols := make([]^Column, len(keys), allocator)
	right_key_cols := make([]^Column, len(keys), allocator)
	for i in 0 ..< len(keys) {
		left_key_cols[i] = column(left, keys[i])
		right_key_cols[i] = column(right, keys[i])
	}

	// 4) build multi index on right
	idx := build_join_multi_index(right, right_key_cols, allocator)
	matched_right := make([]bool, right.rows, allocator)

	for li in 0 ..< left.rows {
		k := make_composite_key_string(left_key_cols, li, allocator)

		head, ok := idx.bucket_head[k]
		found := false

		if ok {
			i := head
			for i != -1 {
				ri := idx.rows[i]
				matched_right[ri] = true
				emit_joined_row_multi(&result, left, right, li, ri, keys)
				found = true
				i = idx.next[i]
			}
		}

		if !found {
			if kind == .Inner {
				continue
			}
			if kind == .Left || kind == .Outer {
				emit_unmatched_row_multi(&result, left, right, li, .LeftOnly, keys)
			}
		}
	}

	// right‑only for Right/Outer
	if kind == .Right || kind == .Outer {
		for ri in 0 ..< right.rows {
			if !matched_right[ri] {
				emit_unmatched_row_multi(&result, left, right, ri, .RightOnly, keys)
			}
		}
	}


	if len(result.columns) > 0 {
		result.rows = result.columns[0].len
	}

	return result
}


emit_joined_row_multi :: proc(
	out: ^DataFrame,
	left: ^DataFrame,
	right: ^DataFrame,
	li: int,
	ri: int,
	keys: []string,
) {
	// left columns
	for ci in 0 ..< len(left.columns) {
		src := &left.columns[ci]
		dst := &out.columns[ci]
		copy_value(dst, src, li)
	}

	// right columns, skipping all key columns
	out_offset := len(left.columns)
	out_idx := out_offset

	for ci in 0 ..< len(right.columns) {
		src := &right.columns[ci]

		skip := false
		for k in keys {
			if src.name == k {
				skip = true
				break
			}
		}
		if skip {
			continue
		}

		dst := &out.columns[out_idx]
		copy_value(dst, src, ri)
		out_idx += 1
	}
}


emit_unmatched_row_multi :: proc(
	out: ^DataFrame,
	left: ^DataFrame,
	right: ^DataFrame,
	row: int, // row index in whichever side is real
	kind: JoinRowKind,
	keys: []string,
	allocator: mem.Allocator = context.allocator,
) {
	// 1. LEFT side
	for ci in 0 ..< len(left.columns) {
		dst := &out.columns[ci]

		if kind == .LeftOnly {
			// real left row
			src := &left.columns[ci]
			copy_value(dst, src, row)
		} else {
			// right-only row → left side is NULL
			append_null(dst)
		}
	}

	// 2. RIGHT side
	out_offset := len(left.columns)
	out_idx := out_offset

	for ci in 0 ..< len(right.columns) {
		src := &right.columns[ci]
		skip := false
		for k in keys {
			if src.name == k {
				skip = true
				break
			}
		}
		if skip {
			continue
		}
		dst := &out.columns[out_idx]

		if kind == .RightOnly {
			// real right row
			copy_value(dst, src, row)
		} else {
			// left-only row → right side is NULL
			append_null(dst)
		}

		out_idx += 1
	}
}


make_composite_key_string :: proc(cols: []^Column, row: int, allocator: mem.Allocator) -> string {
	b := strings.builder_make(allocator)
	defer strings.builder_destroy(&b)

	for i in 0 ..< len(cols) {
		col := cols[i]

		if i > 0 {
			fmt.sbprint(&b, '|')
		}

		#partial switch col.type {
		case .Int:
			fmt.sbprintf(&b, "i:%v", get_int(col, row))
		case .Float:
			fmt.sbprintf(&b, "i:%v", get_float(col, row))
		case .String:
			fmt.sbprintf(&b, "i:%v", get_string(col, row))
		case .Date:
			d := get_date(col, row)
			fmt.sbprintf(&b, "D:%04d-%02d-%02d", d.year, d.month, d.day)
		case .Time:
			t := get_time(col, row)
			fmt.sbprintf(&b, "T:%02d:%02d:%02d", t.hour, t.minute, t.second)
		case .Datetime:
			dt := get_datetime(col, row)
			fmt.sbprintf(
				&b,
				"DT:%04d-%02d-%02d %02d:%02d:%02d",
				dt.year,
				dt.month,
				dt.day,
				dt.hour,
				dt.minute,
				dt.second,
			)
		case .Bool:
			fmt.sbprintf(&b, "b:%v", get_bool(col, row))
		}
	}

	return strings.to_string(b)
}


join_multi_index_matches :: proc(idx: ^Join_Multi_Index, key: string, proc_row: proc(ri: int)) {
	head, ok := idx.bucket_head[key]
	if !ok {
		return
	}

	i := head
	for i != -1 {
		ri := idx.rows[i]
		proc_row(ri)
		i = idx.next[i]
	}
}
