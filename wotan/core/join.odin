package core

import "core:fmt"
import "core:mem"
import vmem "core:mem/virtual"
import "core:strings"

// Joins (SQL / pandas merge semantics):
// - output: all left columns, then the right columns except the right keys;
//   right names that collide with left names get "_right" (then "_right2", ...)
// - rows: each left row with every matching right row (all pairs for duplicate
//   keys) in left order; Left / Outer keep unmatched left rows (right side
//   null); Right / Outer append unmatched right rows (left side null, the key
//   columns take the right row's key values)
// - keys match on type and value; a null (or NaN) key never matches
// - values keep their nulls; output columns use `allocator`

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

// Raw cell readers (view-aware; a null cell reads as the zero value)
get_string :: proc(c: ^Column, row: int) -> string {v, _ := column_at_string(c, row); return v}
get_int :: proc(c: ^Column, row: int) -> int {v, _ := column_at_int(c, row); return v}
get_float :: proc(c: ^Column, row: int) -> f64 {v, _ := column_at_float(c, row); return v}
get_date :: proc(c: ^Column, row: int) -> Date {v, _ := column_at_date(c, row); return v}
get_time :: proc(c: ^Column, row: int) -> Time {v, _ := column_at_time(c, row); return v}
get_datetime :: proc(c: ^Column, row: int) -> Datetime {v, _ := column_at_datetime(c, row); return v}
get_bool :: proc(c: ^Column, row: int) -> bool {v, _ := column_at_bool(c, row); return v}

get_key :: proc($T: typeid, c: ^Column, row: int) -> T {
	if size_of(T) != type_size(c.type) {
		panic(fmt.tprintf("get_key: %v does not match column type %v", typeid_of(T), c.type))
	}
	ptr, null := column_at_ptr(c, row)
	if null {return T{}}
	return (cast(^T)ptr)^
}


// Single-key index (key -> all right rows with that key, in order)
JoinIndex :: struct($T: typeid) {
	key:   string,
	col:   ^Column,
	table: map[T][dynamic]int,
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
	idx.table = make(map[T][dynamic]int, allocator)
	for r in 0 ..< right.rows {
		if is_null(idx.col, r) {continue}
		k := get_key(T, idx.col, r)
		if k not_in idx.table {idx.table[k] = make([dynamic]int, allocator)}
		append(&idx.table[k], r)
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
	keys := []string{key}
	return join_internal(left, right, keys, keys, kind, allocator)
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
	keys: []string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	return join_internal(left, right, keys, keys, kind, allocator)
}

join_on :: proc(
	left: ^DataFrame,
	right: ^DataFrame,
	left_keys: []string,
	right_keys: []string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	return join_internal(left, right, left_keys, right_keys, kind, allocator)
}

join_generic_multi :: proc(
	left: ^DataFrame,
	right: ^DataFrame,
	keys: []string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	return join_internal(left, right, keys, keys, kind, allocator)
}

join_generic_multi_on :: proc(
	left: ^DataFrame,
	right: ^DataFrame,
	left_keys: []string,
	right_keys: []string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	return join_internal(left, right, left_keys, right_keys, kind, allocator)
}


// Append src[row] to dst (nulls kept)
copy_value :: proc(dst, src: ^Column, row: int) {
	copy_cell(dst, src, row)
}

copy_single_field :: proc(dst: rawptr, col: ^Column, row: int) {
	ptr, null := column_at_ptr(col, row)
	if null {
		mem.zero(dst, type_size(col.type))
		return
	}
	mem.copy(dst, rawptr(ptr), type_size(col.type))
}


// ------------------------------------------------------------
// Composite keys
// ------------------------------------------------------------

// Encode the key of `row` into b: per column a type tag and the value, strings
// length-prefixed, so different types or separator characters cannot collide.
// Null (and NaN) cells encode as "N;" and make `has_null` true.
@(private = "file")
_write_key :: proc(b: ^strings.Builder, cols: []^Column, row: int) -> (has_null: bool) {
	for col in cols {
		if is_null(col, row) {
			strings.write_string(b, "N;")
			has_null = true
			continue
		}
		switch col.type {
		case .Int:
			fmt.sbprintf(b, "i%d;", get_int(col, row))
		case .Float:
			v := get_float(col, row)
			if v != v {
				strings.write_string(b, "N;")
				has_null = true
				continue
			}
			if v == 0 {v = 0} 	// -0.0 == 0.0
			fmt.sbprintf(b, "f%x;", transmute(u64)v)
		case .String:
			s := get_string(col, row)
			fmt.sbprintf(b, "s%d:", len(s))
			strings.write_string(b, s)
		case .Bool:
			strings.write_string(b, "b1;" if get_bool(col, row) else "b0;")
		case .Date:
			d := get_date(col, row)
			fmt.sbprintf(b, "d%d-%d-%d;", d.year, d.month, d.day)
		case .Time:
			t := get_time(col, row)
			fmt.sbprintf(b, "t%d:%d:%d;", t.hour, t.minute, t.second)
		case .Datetime:
			t := get_datetime(col, row)
			fmt.sbprintf(b, "D%d-%d-%d %d:%d:%d;", t.year, t.month, t.day, t.hour, t.minute, t.second)
		case .Invalid:
			strings.write_string(b, "?;")
		}
	}
	return
}

// Composite key of `row` as an owned string (allocated with `allocator`)
make_composite_key_string :: proc(cols: []^Column, row: int, allocator: mem.Allocator) -> string {
	b := strings.builder_make(allocator)
	_write_key(&b, cols, row)
	return strings.to_string(b) // the builder's buffer is the result
}


// Multi-key index: key -> chain of rows (in row order); keys in first-seen
// order. Rows whose key contains a null are indexed under that key too
// (groupby2 groups them); joins never look such keys up.
Join_Multi_Index :: struct {
	bucket_head:  map[string]int, // key → first node index
	bucket_tail:  map[string]int, // key → last node index
	rows:         [dynamic]int,
	next:         [dynamic]int,
	ordered_keys: [dynamic]string,
	arena:        ^vmem.Arena, // owns the key strings
	allocator:    mem.Allocator,
}


join_multi_index_make :: proc(allocator: mem.Allocator) -> Join_Multi_Index {
	arena := new(vmem.Arena, allocator)
	if err := vmem.arena_init_growing(arena); err != nil {
		panic("join_multi_index_make: cannot allocate key arena")
	}
	return Join_Multi_Index {
		bucket_head = make(map[string]int, allocator),
		bucket_tail = make(map[string]int, allocator),
		rows = make([dynamic]int, 0, allocator),
		next = make([dynamic]int, 0, allocator),
		ordered_keys = make([dynamic]string, 0, allocator),
		arena = arena,
		allocator = allocator,
	}
}

join_multi_index_destroy :: proc(idx: ^Join_Multi_Index) {
	delete(idx.bucket_head)
	delete(idx.bucket_tail)
	delete(idx.rows)
	delete(idx.next)
	delete(idx.ordered_keys)
	if idx.arena != nil {
		vmem.arena_destroy(idx.arena)
		free(idx.arena, idx.allocator)
		idx.arena = nil
	}
}


build_join_multi_index :: proc(
	right: ^DataFrame,
	key_cols: []^Column,
	allocator: mem.Allocator = context.allocator,
) -> Join_Multi_Index {
	idx := join_multi_index_make(allocator)
	key_alloc := vmem.arena_allocator(idx.arena)
	b := strings.builder_make(context.temp_allocator)
	defer strings.builder_destroy(&b)

	for r in 0 ..< right.rows {
		strings.builder_reset(&b)
		_write_key(&b, key_cols, r)

		node_index := len(idx.rows)
		append(&idx.rows, r)
		append(&idx.next, -1)

		if tail, ok := idx.bucket_tail[strings.to_string(b)]; ok {
			idx.next[tail] = node_index
			idx.bucket_tail[strings.to_string(b)] = node_index
		} else {
			k := strings.clone(strings.to_string(b), key_alloc)
			append(&idx.ordered_keys, k)
			idx.bucket_head[k] = node_index
			idx.bucket_tail[k] = node_index
		}
	}
	return idx
}


join_multi_index_matches :: proc(idx: ^Join_Multi_Index, key: string, proc_row: proc(ri: int)) {
	head, ok := idx.bucket_head[key]
	if !ok {
		return
	}
	for i := head; i != -1; i = idx.next[i] {
		proc_row(idx.rows[i])
	}
}


// ------------------------------------------------------------
// Row emission
// ------------------------------------------------------------

@(private = "file")
_is_key :: proc(name: string, keys: []string) -> bool {
	for k in keys {
		if name == k {return true}
	}
	return false
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
	emit_joined_row_multi(out, left, right, li, ri, []string{key})
}

emit_unmatched_row :: proc(
	out: ^DataFrame,
	left: ^DataFrame,
	right: ^DataFrame,
	row: int,
	kind: JoinRowKind,
	key: string,
	allocator: mem.Allocator = context.allocator,
) {
	keys := []string{key}
	_emit_unmatched(out, left, right, row, kind, keys, keys)
}

emit_joined_row_multi :: proc(
	out: ^DataFrame,
	left: ^DataFrame,
	right: ^DataFrame,
	li: int,
	ri: int,
	keys: []string,
) {
	for ci in 0 ..< len(left.columns) {
		copy_cell(&out.columns[ci], &left.columns[ci], li)
	}
	out_idx := len(left.columns)
	for ci in 0 ..< len(right.columns) {
		if _is_key(right.columns[ci].name, keys) {continue}
		copy_cell(&out.columns[out_idx], &right.columns[ci], ri)
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
	_emit_unmatched(out, left, right, row, kind, keys, keys)
}

@(private = "file")
_emit_unmatched :: proc(
	out: ^DataFrame,
	left: ^DataFrame,
	right: ^DataFrame,
	row: int,
	kind: JoinRowKind,
	left_keys: []string,
	right_keys: []string,
) {
	// left side: the real row, or nulls (key columns take the right key values)
	for ci in 0 ..< len(left.columns) {
		dst := &out.columns[ci]
		if kind == .LeftOnly {
			copy_cell(dst, &left.columns[ci], row)
			continue
		}
		filled := false
		for lk, k in left_keys {
			if left.columns[ci].name != lk {continue}
			rk := column(right, right_keys[k])
			if rk.type == dst.type {
				copy_cell(dst, rk, row)
				filled = true
			}
			break
		}
		if !filled {append_null(dst)}
	}

	// right side: the real row, or nulls
	out_idx := len(left.columns)
	for ci in 0 ..< len(right.columns) {
		if _is_key(right.columns[ci].name, right_keys) {continue}
		dst := &out.columns[out_idx]
		if kind == .RightOnly {
			copy_cell(dst, &right.columns[ci], row)
		} else {
			append_null(dst)
		}
		out_idx += 1
	}
}


// ------------------------------------------------------------
// Join driver
// ------------------------------------------------------------

join_internal :: proc(
	left: ^DataFrame,
	right: ^DataFrame,
	left_keys: []string,
	right_keys: []string,
	kind: JoinKind = .Inner,
	allocator: mem.Allocator = context.allocator,
) -> DataFrame {
	assert(len(left_keys) == len(right_keys), "left_keys and right_keys must match")
	assert(len(left_keys) > 0, "join needs at least one key")

	result := dataframe_new(allocator)

	// 1) columns: left, then right non-key columns with unique names
	taken := make(map[string]bool, allocator = context.temp_allocator)
	defer delete(taken)
	for &src in left.columns {
		taken[src.name] = true
		add_column(&result, column_new(src.name, src.type, 0, allocator))
	}
	for &src in right.columns {
		if _is_key(src.name, right_keys) {continue}
		name := src.name
		suffix := 1
		for taken[name] {
			if suffix == 1 {
				name = fmt.aprintf("%s_right", src.name, allocator = allocator)
			} else {
				name = fmt.aprintf("%s_right%d", src.name, suffix, allocator = allocator)
			}
			suffix += 1
		}
		taken[name] = true
		add_column(&result, column_new(name, src.type, 0, allocator))
	}

	// 2) key columns (types must agree pairwise)
	left_key_cols := make([]^Column, len(left_keys), context.temp_allocator)
	right_key_cols := make([]^Column, len(right_keys), context.temp_allocator)
	defer {
		delete(left_key_cols, context.temp_allocator)
		delete(right_key_cols, context.temp_allocator)
	}
	for i in 0 ..< len(left_keys) {
		left_key_cols[i] = column(left, left_keys[i])
		right_key_cols[i] = column(right, right_keys[i])
	}

	// 3) index the right side
	idx := build_join_multi_index(right, right_key_cols, context.temp_allocator)
	defer join_multi_index_destroy(&idx)
	matched_right := make([]bool, right.rows, context.temp_allocator)
	defer delete(matched_right, context.temp_allocator)

	// 4) left rows in order
	b := strings.builder_make(context.temp_allocator)
	defer strings.builder_destroy(&b)
	for li in 0 ..< left.rows {
		strings.builder_reset(&b)
		has_null := _write_key(&b, left_key_cols, li)

		found := false
		if !has_null {
			if head, ok := idx.bucket_head[strings.to_string(b)]; ok {
				for i := head; i != -1; i = idx.next[i] {
					ri := idx.rows[i]
					matched_right[ri] = true
					emit_joined_row_multi(&result, left, right, li, ri, right_keys)
					found = true
				}
			}
		}
		if !found && (kind == .Left || kind == .Outer) {
			_emit_unmatched(&result, left, right, li, .LeftOnly, left_keys, right_keys)
		}
	}

	// 5) unmatched right rows (incl. right rows with null keys)
	if kind == .Right || kind == .Outer {
		for ri in 0 ..< right.rows {
			if !matched_right[ri] {
				_emit_unmatched(&result, left, right, ri, .RightOnly, left_keys, right_keys)
			}
		}
	}

	result.rows = result.columns[0].len if len(result.columns) > 0 else 0
	return result
}

