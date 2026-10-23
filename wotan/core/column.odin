package core

import "core:fmt"
import "core:mem"
import "core:strings"

import vmem "core:mem/virtual"
Column :: struct {
	name:      string,
	type:      ColumnType,
	len:       int,
	capacity:  int,
	data:      rawptr,
	nulls:     []bool,
	orig:      ^Column, // pointer to original column if this is a view
	offset:    int, // start offset in original
	is_view:   bool,
	arena:     vmem.Arena,
	allocator: mem.Allocator,
}


column_new :: proc(
	name: string,
	type: ColumnType,
	capacity: int,
	allocator: mem.Allocator = context.allocator,
) -> Column {
	size := capacity * type_size(type)
	data, err := mem.alloc(size, allocator = allocator)
	if err != nil {
		panic("Column: allocation failed")
	}
	nulls := make([]bool, capacity, allocator = allocator)
	return Column {
		name = name,
		type = type,
		len = 0,
		capacity = capacity,
		data = data,
		nulls = nulls,
		arena = vmem.Arena{},
		is_view = false,
		allocator = allocator,
	}
}

destroy_column :: proc(col: ^Column) {
	if col.is_view {
		return
	}
	if col.data != nil {

		mem.free(col.data, col.allocator)
		col.data = nil
	}
	if col.nulls != nil {
		delete(col.nulls, col.allocator)
		col.nulls = nil
	}
	if col.type == .String && col.arena.kind != nil {
		vmem.arena_destroy(&col.arena)
	}
	col.len = 0.0
	col.capacity = 0.0
}


append_int :: proc(c: ^Column, v: int) {
	if c.is_view {
		panic("append_int: cannot append to view column")
	}
	if c.type != .Int {
		panic("append_int: wrong column type")
	}
	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}

	idx := c.len
	c.len += 1

	base := uintptr(c.data) + uintptr(idx * size_of(int))
	(cast(^int)base)^ = v
}


grow :: proc(c: ^Column, new_capacity: int) {
	if new_capacity <= c.capacity {
		return
	}

	// buffers belong to the column's allocator (columns built by literal fall
	// back to the context allocator)
	if c.allocator.procedure == nil {c.allocator = context.allocator}
	new_size := new_capacity * type_size(c.type)
	new_data, err := mem.alloc(new_size, allocator = c.allocator)
	if err != nil {
		panic("allocation failed")
	}

	old_size := c.len * type_size(c.type)
	if c.data != nil {
		mem.copy(new_data, c.data, old_size)
		mem.free(c.data, c.allocator)
	}
	c.data = new_data
	c.capacity = new_capacity

	if c.nulls != nil {
		new_nulls := make([]bool, new_capacity, c.allocator)
		copy(new_nulls, c.nulls)
		delete(c.nulls, c.allocator)
		c.nulls = new_nulls
	}
}

append_colum :: proc {
	append_bool,
	append_int,
	append_null,
	append_float,
	append_date,
	append_string,
	append_time,
	append_datetime,
}


append_fake_float :: proc(c: ^Column, v: f64) {
	if c.is_view {
		panic("append_float: cannot append to view column")
	}

	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}

	idx := c.len
	c.len += 1

	base := uintptr(c.data) + uintptr(idx * size_of(f64))
	(cast(^f64)base)^ = v
}


append_float :: proc(c: ^Column, v: f64) {
	if c.is_view {
		panic("append_float: cannot append to view column")
	}
	if c.type != .Float {
		panic("append_float: wrong column type")
	}
	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}

	idx := c.len
	c.len += 1

	base := uintptr(c.data) + uintptr(idx * size_of(f64))
	(cast(^f64)base)^ = v
}

append_bool :: proc(c: ^Column, v: bool) {
	if c.is_view {
		panic("append_bool: cannot append to view column")
	}
	if c.type != .Bool {
		panic("append_bool: wrong column type")
	}
	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}

	idx := c.len
	c.len += 1

	base := uintptr(c.data) + uintptr(idx * size_of(bool))
	(cast(^bool)base)^ = v
}

append_string :: proc(c: ^Column, v: string, should_clone: bool = true) {
	if c.is_view {
		panic("append_string: cannot append to view column")
	}
	if c.type != .String {
		panic("append_string: wrong column type")
	}
	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}
	final_str := v
	if should_clone {
		// Initialize arena on first use if needed
		if c.arena.kind == nil {

			err := vmem.arena_init_growing(&c.arena)
			if err != nil {
				panic("column_append_string: Cannot allocate Arena")
			}
		}
		allocator := vmem.arena_allocator(&c.arena)
		final_str = strings.clone(v, allocator)
	}
	idx := c.len
	c.len += 1

	base := uintptr(c.data) + uintptr(idx * size_of(string))
	(cast(^string)base)^ = final_str
}

append_date :: proc(c: ^Column, v: Date) {
	if c.is_view {
		panic("append_date: cannot append to view column")
	}
	if c.type != .Date {
		panic("append_date: wrong column type")
	}
	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}

	idx := c.len
	c.len += 1

	base := uintptr(c.data) + uintptr(idx * size_of(Date))
	(cast(^Date)base)^ = v
}


append_time :: proc(c: ^Column, v: Time) {
	if c.is_view {
		panic("append_time: cannot append to view column")
	}
	if c.type != .Time {
		panic("append_time: wrong column type")
	}
	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}

	idx := c.len
	c.len += 1

	base := uintptr(c.data) + uintptr(idx * size_of(Time))
	(cast(^Time)base)^ = v
}

append_datetime :: proc(c: ^Column, v: Datetime) {
	if c.is_view {
		panic("append_datetime: cannot append to view column")
	}
	if c.type != .Datetime {
		panic("append_datetime: wrong column type")
	}
	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}

	idx := c.len
	c.len += 1

	base := uintptr(c.data) + uintptr(idx * size_of(Datetime))
	(cast(^Datetime)base)^ = v
}
append_null :: proc(c: ^Column) {
	if c.is_view {
		panic("append_null: cannot append to view column")
	}
	if c.len >= c.capacity {
		grow(c, max(8, c.capacity * 2))
	}

	idx := c.len
	c.len += 1

	if c.nulls == nil {
		if c.allocator.procedure == nil {c.allocator = context.allocator}
		c.nulls = make([]bool, c.capacity, c.allocator)
	}
	c.nulls[idx] = true
}

// Physical column and row behind row `i` of `col`: bounds-checked against the
// (view) column itself, then resolved through views (nested views add up
// their offsets)
column_resolve :: proc(col: ^Column, i: int, loc := #caller_location) -> (^Column, int) {
	if i < 0 || i >= col.len {
		panic(fmt.tprintf("column %q: row %d out of range 0..<%d", col.name, i, col.len), loc)
	}
	c, r := col, i
	for c.is_view {
		r += c.offset
		c = c.orig
	}
	return c, r
}

@(private = "file")
_cell_ptr :: #force_inline proc(c: ^Column, r: int) -> uintptr {
	return uintptr(c.data) + uintptr(r * type_size(c.type))
}

@(private = "file")
_null_at :: #force_inline proc(c: ^Column, r: int) -> bool {
	return c.nulls != nil && r < len(c.nulls) && c.nulls[r]
}

column_at_ptr :: proc(col: ^Column, i: int) -> (uintptr, bool) {
	c, r := column_resolve(col, i)
	if _null_at(c, r) {
		return 0, true
	}
	return _cell_ptr(c, r), false
}


// Column read helpers (safe, copy-based readers; second result = is_null)
column_at_int :: proc(col: ^Column, i: int) -> (int, bool) {
	c, r := column_resolve(col, i)
	if _null_at(c, r) {return 0, true}
	return (cast(^int)_cell_ptr(c, r))^, false
}

column_at_float :: proc(col: ^Column, i: int) -> (f64, bool) {
	c, r := column_resolve(col, i)
	if _null_at(c, r) {return 0.0, true}
	return (cast(^f64)_cell_ptr(c, r))^, false
}

column_at_bool :: proc(col: ^Column, i: int) -> (bool, bool) {
	c, r := column_resolve(col, i)
	if _null_at(c, r) {return false, true}
	return (cast(^bool)_cell_ptr(c, r))^, false
}

column_at_string :: proc(col: ^Column, i: int) -> (string, bool) {
	c, r := column_resolve(col, i)
	if _null_at(c, r) {return "", true}
	return (cast(^string)_cell_ptr(c, r))^, false
}

column_at_date :: proc(col: ^Column, i: int) -> (Date, bool) {
	c, r := column_resolve(col, i)
	if _null_at(c, r) {return Date{0, 0, 0}, true}
	return (cast(^Date)_cell_ptr(c, r))^, false
}

column_at_time :: proc(col: ^Column, i: int) -> (Time, bool) {
	c, r := column_resolve(col, i)
	if _null_at(c, r) {return Time{0, 0, 0}, true}
	return (cast(^Time)_cell_ptr(c, r))^, false
}

column_at_datetime :: proc(col: ^Column, i: int) -> (Datetime, bool) {
	c, r := column_resolve(col, i)
	if _null_at(c, r) {return Datetime{0, 0, 0, 0, 0, 0}, true}
	return (cast(^Datetime)_cell_ptr(c, r))^, false
}

// New owned column with the given rows of `src` (view-aware; nulls kept;
// strings cloned into the new column). A row index of -1 yields null.
column_take :: proc(src: ^Column, rows: []int, allocator := context.allocator) -> Column {
	out := column_new(src.name, src.type, max(len(rows), 1), allocator)
	for r in rows {
		if r < 0 || is_null(src, r) {
			append_null(&out)
			continue
		}
		switch src.type {
		case .Int:
			v, _ := column_at_int(src, r)
			append_int(&out, v)
		case .Float:
			v, _ := column_at_float(src, r)
			append_float(&out, v)
		case .Bool:
			v, _ := column_at_bool(src, r)
			append_bool(&out, v)
		case .String:
			v, _ := column_at_string(src, r)
			append_string(&out, v)
		case .Date:
			v, _ := column_at_date(src, r)
			append_date(&out, v)
		case .Time:
			v, _ := column_at_time(src, r)
			append_time(&out, v)
		case .Datetime:
			v, _ := column_at_datetime(src, r)
			append_datetime(&out, v)
		case .Invalid:
			append_null(&out)
		}
	}
	return out
}

// Owned copy of all rows of a column (materializes views)
column_clone :: proc(src: ^Column, allocator := context.allocator) -> Column {
	rows := make([]int, src.len, context.temp_allocator)
	defer delete(rows, context.temp_allocator)
	for i in 0 ..< src.len {rows[i] = i}
	return column_take(src, rows, allocator)
}

// Three-way comparison of two cells of one column; nulls (and float NaN)
// order after every value. Returns -1 / 0 / +1.
column_compare_rows :: proc(col: ^Column, a, b: int) -> int {
	na, nb := is_null(col, a), is_null(col, b)
	if col.type == .Float {
		fa, _ := column_at_float(col, a)
		fb, _ := column_at_float(col, b)
		na, nb = na || fa != fa, nb || fb != fb
	}
	if na || nb {
		if na && nb {return 0}
		return 1 if na else -1
	}
	cmp3 :: proc(x, y: $T) -> int {
		if x < y {return -1}
		if x > y {return 1}
		return 0
	}
	switch col.type {
	case .Int:
		x, _ := column_at_int(col, a)
		y, _ := column_at_int(col, b)
		return cmp3(x, y)
	case .Float:
		x, _ := column_at_float(col, a)
		y, _ := column_at_float(col, b)
		return cmp3(x, y)
	case .Bool:
		x, _ := column_at_bool(col, a)
		y, _ := column_at_bool(col, b)
		return cmp3(int(x), int(y))
	case .String:
		x, _ := column_at_string(col, a)
		y, _ := column_at_string(col, b)
		return strings.compare(x, y)
	case .Date:
		x, _ := column_at_date(col, a)
		y, _ := column_at_date(col, b)
		return int(date_compare(x, y))
	case .Time:
		x, _ := column_at_time(col, a)
		y, _ := column_at_time(col, b)
		return int(time_compare(x, y))
	case .Datetime:
		x, _ := column_at_datetime(col, a)
		y, _ := column_at_datetime(col, b)
		return int(datetime_compare(x, y))
	case .Invalid:
	}
	return 0
}

column_slice_view :: proc(orig: ^Column, start: int, end: int) -> Column {
	if start < 0 || end < start || end > orig.len {
		panic("column_slice_view: invalid range")
	}

	n := end - start
	base, offset := orig, start
	for base.is_view { 	// view of a view: point straight at the physical column
		offset += base.offset
		base = base.orig
	}

	return Column {
		name     = orig.name, // shared, not owned
		type     = orig.type,
		len      = n,
		capacity = n, // logical capacity for the view
		data     = nil, // no storage; read through column_at_* / is_null
		nulls    = nil,
		orig     = base,
		offset   = offset,
		is_view  = true,
	}
}


column_slice_copy :: proc(orig: ^Column, start: int, end: int) -> Column {
	if start < 0 || end < start || end > orig.len {
		panic("column_slice_copy: invalid range")
	}

	n := end - start
	c := column_new(orig.name, orig.type, n)

	for i in start ..< end {
		#partial switch orig.type {
		case .Int:
			v, is_null := column_at_int(orig, i)
			if is_null {append_null(&c)} else {append_int(&c, v)}

		case .Float:
			v, is_null := column_at_float(orig, i)
			if is_null {append_null(&c)} else {append_float(&c, v)}

		case .Bool:
			v, is_null := column_at_bool(orig, i)
			if is_null {append_null(&c)} else {append_bool(&c, v)}

		case .String:
			v, is_null := column_at_string(orig, i)
			if is_null {append_null(&c)} else {append_string(&c, v)}

		case .Date:
			v, is_null := column_at_date(orig, i)
			if is_null {append_null(&c)} else {append_date(&c, v)}
		case .Time:
			v, is_null := column_at_time(orig, i)
			if is_null {append_null(&c)} else {append_time(&c, v)}

		case .Datetime:
			v, is_null := column_at_datetime(orig, i)
			if is_null {append_null(&c)} else {append_datetime(&c, v)}
		}
	}

	return c
}

column_new_bool_mask :: proc(len: int) -> Column {
	c := column_new("mask", .Bool, len)
	return c
}

column_gt :: proc {
	column_gt_int,
	column_gt_float,
	column_gt_date,
	column_gt_time,
	column_gt_datetime,
}

column_lt :: proc {
	column_lt_int,
	column_lt_float,
	column_lt_date,
	column_lt_time,
	column_lt_datetime,
}

column_eq :: proc {
	column_eq_int,
	column_eq_float,
	column_eq_date,
	column_eq_string,
	column_eq_time,
	column_eq_datetime,
}


column_gt_int :: proc(col: ^Column, value: int) -> Column {
	if col.type != .Int {
		panic("column_gt_int: wrong column type")
	}

	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_int(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, v > value)
		}
	}
	return out

}


column_lt_int :: proc(col: ^Column, value: int) -> Column {
	if col.type != .Int {
		panic("column_lt_int: wrong column type")
	}

	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_int(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, v < value)
		}
	}
	return out

}


column_eq_int :: proc(col: ^Column, value: int) -> Column {
	if col.type != .Int {
		panic("column_eq_int: wrong column type")
	}

	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_int(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, v == value)
		}
	}
	return out

}


column_gt_float :: proc(col: ^Column, value: f64) -> Column {
	if col.type != .Float {
		panic("column_gt_float: wrong column type")
	}

	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_float(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, v > value)
		}
	}
	return out

}


column_lt_float :: proc(col: ^Column, value: f64) -> Column {
	if col.type != .Float {
		panic("column_lt_float: wrong column type")
	}

	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_float(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, v < value)
		}
	}
	return out

}


column_eq_float :: proc(col: ^Column, value: f64) -> Column {
	if col.type != .Float {
		panic("column_eq_float: wrong column type")
	}

	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_float(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, v == value)
		}
	}
	return out

}

column_eq_string :: proc(col: ^Column, value: string) -> Column {
	if col.type != .String {
		panic("column_eq_string: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_string(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, v == value)
		}
	}
	return out
}
column_contains_string :: proc(col: ^Column, value: string) -> Column {
	if col.type != .String {
		panic("column_contains_string: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_string(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, strings.contains(v, value))
		}
	}
	return out
}

column_gt_date :: proc(col: ^Column, value: Date) -> Column {
	if col.type != .Date {
		panic("column_gt_date: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_date(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, date_compare(v, value) > 0)
		}
	}
	return out
}

column_lt_date :: proc(col: ^Column, value: Date) -> Column {
	if col.type != .Date {
		panic("column_lt_date: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_date(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, date_compare(v, value) < 0)
		}
	}
	return out
}

column_eq_date :: proc(col: ^Column, value: Date) -> Column {
	if col.type != .Date {
		panic("column_eq_date: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_date(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, date_compare(v, value) == 0)
		}
	}
	return out
}

column_gt_time :: proc(col: ^Column, value: Time) -> Column {
	if col.type != .Time {
		panic("column_gt_time: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_time(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, time_compare(v, value) > 0)
		}
	}
	return out
}

column_lt_time :: proc(col: ^Column, value: Time) -> Column {
	if col.type != .Time {
		panic("column_lt_time: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_time(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, time_compare(v, value) < 0)
		}
	}
	return out
}

column_eq_time :: proc(col: ^Column, value: Time) -> Column {
	if col.type != .Time {
		panic("column_eq_time: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_time(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, time_compare(v, value) == 0)
		}
	}
	return out
}

column_gt_datetime :: proc(col: ^Column, value: Datetime) -> Column {
	if col.type != .Datetime {
		panic("column_gt_datetime: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_datetime(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, datetime_compare(v, value) > 0)
		}
	}
	return out
}

column_lt_datetime :: proc(col: ^Column, value: Datetime) -> Column {
	if col.type != .Datetime {
		panic("column_lt_datetime: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_datetime(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, datetime_compare(v, value) < 0)
		}
	}
	return out
}

column_eq_datetime :: proc(col: ^Column, value: Datetime) -> Column {
	if col.type != .Datetime {
		panic("column_eq_datetime: wrong column type")
	}
	out := column_new_bool_mask(col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_datetime(col, i)
		if is_null {
			append_null(&out)
		} else {
			append_bool(&out, datetime_compare(v, value) == 0)
		}
	}
	return out
}
//converts bool column into a bool array mask, treating nulls as false
column_mask :: proc(col: ^Column) -> []bool {
	if col.type != .Bool {
		panic("column_mask: wrong column type")
	}
	barr := make([]bool, col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_bool(col, i)
		if (is_null) {
			barr[i] = false
		} else {
			barr[i] = v
		}
	}

	return barr
}
column_from_ints :: proc(name: string, values: []int) -> Column {
	col := column_new(name, .Int, len(values))
	for v in values {
		append_int(&col, v)
	}
	return col
}
column_from_strings :: proc(name: string, values: []string) -> Column {
	col := column_new(name, .String, len(values))
	for v in values {
		append_string(&col, v)
	}
	return col
}
column_from_bools :: proc(name: string, values: []bool) -> Column {
	col := column_new(name, .Bool, len(values))
	for v in values {
		append_bool(&col, v)
	}
	return col
}


column_from_floats :: proc(name: string, values: []f64) -> Column {
	col := column_new(name, .Float, len(values))
	for v in values {
		append_float(&col, v)
	}
	return col
}

column_from_dates :: proc(name: string, values: []Date) -> Column {
	col := column_new(name, .Date, len(values))
	for v in values {
		append_date(&col, v)
	}
	return col
}
column_from_times :: proc(name: string, values: []Time) -> Column {
	col := column_new(name, .Time, len(values))
	for v in values {
		append_time(&col, v)
	}
	return col
}

column_from_datetimes :: proc(name: string, values: []Datetime) -> Column {
	col := column_new(name, .Datetime, len(values))
	for v in values {
		append_datetime(&col, v)
	}
	return col
}


column_from_f64 :: proc(name: string, values: []f64) -> Column {
	col := column_new(name, .Float, len(values))

	for v in values {
		append_float(&col, v)
	}

	return col
}
column_from_f64_nullable :: proc(name: string, values: []f64, nulls: []bool) -> Column {
	col := column_new(name, .Float, len(values))

	for v, i in values {
		if nulls[i] {
			append_null(&col)
		} else {
			append_float(&col, v)
		}
	}

	return col
}

// Returns true if the cell at (col, row) is null, respecting views and bounds.
is_null :: proc(col: ^Column, row: int) -> bool {
	c, r := column_resolve(col, row)
	return _null_at(c, r)
}

// Returns the value at (col, row) as a string, using your typed accessors.
// Null -> "" (caller decides what to do with that).
value_as_string :: proc(col: ^Column, row: int, allocator: mem.Allocator) -> string {
	if is_null(col, row) {
		return ""
	}

	// Canonical text form, readable back by the importers without loss:
	// shortest exact floats (always with '.'/exponent so they stay Float),
	// true/false, ISO dates "YYYY-MM-DD", times "HH:MM:SS",
	// datetimes "YYYY-MM-DD HH:MM:SS"
	switch col.type {
	case .Int:
		v, _ := column_at_int(col, row)
		return fmt.aprintf("%d", v, allocator = allocator)
	case .Float:
		v, _ := column_at_float(col, row)
		return float_to_string(v, allocator)
	case .Bool:
		v, _ := column_at_bool(col, row)
		return "true" if v else "false"
	case .String:
		v, _ := column_at_string(col, row)
		return v
	case .Date:
		v, _ := column_at_date(col, row)
		return date_to_string(v, allocator)
	case .Time:
		v, _ := column_at_time(col, row)
		return time_to_string(v, allocator)
	case .Datetime:
		v, _ := column_at_datetime(col, row)
		return datetime_to_string(v, allocator)
	case .Invalid:
	}
	return ""
}

// Shortest decimal text that parses back to exactly `v`; integral values keep
// a ".0" so that a re-imported column is still inferred as Float.
// Non-finite values: "NaN", "Inf", "-Inf".
float_to_string :: proc(v: f64, allocator: mem.Allocator) -> string {
	if v != v {return "NaN"}
	if v > max(f64) {return "Inf"}
	if v < -max(f64) {return "-Inf"}
	s := fmt.aprintf("%v", v, allocator = allocator)
	for i in 0 ..< len(s) {
		if s[i] == '.' || s[i] == 'e' || s[i] == 'E' {return s}
	}
	out := fmt.aprintf("%s.0", s, allocator = allocator)
	delete(s, allocator)
	return out
}
