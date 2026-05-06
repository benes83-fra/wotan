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

	new_size := new_capacity * type_size(c.type)
	new_data, err := mem.alloc(new_size)
	if err != nil {
		panic("allocation failed")
	}


	old_size := c.len * type_size(c.type)
	mem.copy(new_data, c.data, old_size)

	mem.free(c.data)
	c.data = new_data
	c.capacity = new_capacity

	if c.nulls != nil {
		new_nulls := make([]bool, new_capacity)
		copy(new_nulls, c.nulls)
		delete(c.nulls)
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

	base := uintptr(c.data) + uintptr(idx * size_of(Date))
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
		c.nulls = make([]bool, c.capacity)
	}
	c.nulls[idx] = true
}

column_at_ptr :: proc(col: ^Column, i: int) -> (uintptr, bool) {

	if i < 0 || i >= col.len {
		panic("column_at_int: index out of range")
	}

	if col.is_view {
		return column_at_ptr(col.orig, col.offset + i)
	}

	if col.nulls != nil && col.nulls[i] {
		return 0, true
	}
	base := uintptr(col.data) + uintptr(i * type_size(col.type))

	return base, false

}


// Column read helpers (safe, copy-based readers)
column_at_int :: proc(col: ^Column, i: int) -> (int, bool) {
	if col.is_view {
		return column_at_int(col.orig, col.offset + i)
	}

	if i < 0 || i >= col.len {
		fmt.println(" Index is :", i)
		panic("column_at_int: index out of range")
	}

	if col.nulls != nil && col.nulls[i] {
		return 0, true
	}
	base := uintptr(col.data) + uintptr(i * type_size(col.type))
	return (cast(^int)base)^, false
}


column_at_float :: proc(col: ^Column, i: int) -> (f64, bool) {
	if i < 0 || i >= col.len {
		panic("column_at_float: index out of range")
	}
	if col.is_view {
		return column_at_float(col.orig, col.offset + i)
	}
	if col.nulls != nil && col.nulls[i] {
		return 0.0, true
	}
	base := uintptr(col.data) + uintptr(i * type_size(col.type))
	return (cast(^f64)base)^, false
}

column_at_bool :: proc(col: ^Column, i: int) -> (bool, bool) {
	if i < 0 || i >= col.len {
		panic("column_at_bool: index out of range")
	}
	if col.is_view {
		return column_at_bool(col.orig, col.offset + i)
	}
	if col.nulls != nil && col.nulls[i] {
		return false, true
	}
	base := uintptr(col.data) + uintptr(i * type_size(col.type))
	return (cast(^bool)base)^, false
}

column_at_string :: proc(col: ^Column, i: int) -> (string, bool) {
	if i < 0 || i >= col.len {
		panic("column_at_string: index out of range")
	}
	if col.is_view {
		return column_at_string(col.orig, col.offset + i)
	}
	if col.nulls != nil && col.nulls[i] {
		return "", true
	}
	base := uintptr(col.data) + uintptr(i * type_size(col.type))
	return (cast(^string)base)^, false
}

column_at_date :: proc(col: ^Column, i: int) -> (Date, bool) {
	if i < 0 || i >= col.len {
		panic("column_at_date: index out of range")
	}
	if col.is_view {
		return column_at_date(col.orig, col.offset + i)
	}
	if col.nulls != nil && col.nulls[i] {
		return Date{0, 0, 0}, true
	}
	base := uintptr(col.data) + uintptr(i * type_size(col.type))
	return (cast(^Date)base)^, false
}

column_at_time :: proc(col: ^Column, i: int) -> (Time, bool) {
	if i < 0 || i >= col.len {
		panic("column_at_date: index out of range")
	}
	if col.is_view {
		return column_at_time(col.orig, col.offset + i)
	}
	if col.nulls != nil && col.nulls[i] {
		return Time{0, 0, 0}, true
	}
	base := uintptr(col.data) + uintptr(i * type_size(col.type))
	return (cast(^Time)base)^, false
}

column_at_datetime :: proc(col: ^Column, i: int) -> (Datetime, bool) {
	if i < 0 || i >= col.len {
		panic("column_at_datetime: index out of range")
	}
	if col.is_view {
		return column_at_datetime(col.orig, col.offset + i)
	}
	if col.nulls != nil && col.nulls[i] {
		return Datetime{0, 0, 0, 0, 0, 0}, true
	}
	base := uintptr(col.data) + uintptr(i * type_size(col.type))
	return (cast(^Datetime)base)^, false
}
column_slice_view :: proc(orig: ^Column, start: int, end: int) -> Column {
	if start < 0 || end < start || end > orig.len {
		panic("column_slice_view: invalid range")
	}

	n := end - start

	return Column {
		name     = orig.name, // shared, not owned
		type     = orig.type,
		len      = n,
		capacity = n, // logical capacity for the view
		data     = nil, // no storage
		nulls    = nil, // no bitmap; delegate to orig
		orig     = orig,
		offset   = start,
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
	if col.type != .Date {
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
	col := column_new(name, .Date, len(values))
	for v in values {
		append_time(&col, v)
	}
	return col
}

column_from_datetimes :: proc(name: string, values: []Datetime) -> Column {
	col := column_new(name, .Date, len(values))
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
	if col.is_view {
		return is_null(col.orig, col.offset + row)
	}

	if row < 0 || row >= col.len {
		panic("is_null: index out of range")
	}

	if col.nulls == nil {
		return false
	}

	return col.nulls[row]
}

// Returns the value at (col, row) as a string, using your typed accessors.
// Null -> "" (caller decides what to do with that).
value_as_string :: proc(col: ^Column, row: int, allocator: mem.Allocator) -> string {
	if col.is_view {
		return value_as_string(col.orig, col.offset + row, allocator)
	}

	if row < 0 || row >= col.len {
		panic("value_as_string: index out of range")
	}

	// Adjust the cases to your actual ColumnType enum names.
	#partial switch col.type {
	case .Int:
		v, is_null := column_at_int(col, row)
		if is_null {
			return ""
		}
		return fmt.aprintf("%v", v, allocator = allocator)

	case .Float:
		v, is_null := column_at_float(col, row)
		if is_null {
			return ""
		}
		return fmt.aprintf("%v", v, allocator = allocator)

	case .Bool:
		v, is_null := column_at_bool(col, row)
		if is_null {
			return ""
		}
		if v {
			return "1"
		}
		return "0"

	case .String:
		v, is_null := column_at_string(col, row)
		if is_null {
			return ""
		}
		return v

	case .Datetime:
		v, is_null := column_at_datetime(col, row)
		if is_null {
			return ""
		}
		return fmt.aprintf("%v", v, allocator = allocator)
	}

	// Fallback for unsupported types (Date, Time, etc. if you want to add them later)
	panic("value_as_string: unsupported column type")
}
