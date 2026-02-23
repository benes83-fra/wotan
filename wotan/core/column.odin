package wotan

import "core:mem"

Column :: struct {
    name:     string,
    type:     ColumnType,
    len:      int,
    capacity: int,
    data:     rawptr,
    nulls:    []bool,
    orig:     ^Column,   // pointer to original column if this is a view
    offset:   int,     // start offset in original
    is_view:  bool,
}


column_new :: proc(name: string, type: ColumnType, capacity: int) -> Column {
    size := capacity * type_size(type)
    data, err := mem.alloc(size)
    if err != nil {
        panic("Column: allocation failed")
    }

    return Column{
        name     = name,
        type     = type,
        len      = 0,
        capacity = capacity,
        data     = data,
        nulls    = nil,
    }
}

append_int :: proc(c: ^Column, v: int) {
    if c.type != .Int {
        panic("append_int: wrong column type")
    }
    if c.len >= c.capacity {
        grow(c, max(8, c.capacity * 2))
    }

    idx := c.len
    c.len += 1

    base := uintptr(c.data) + uintptr(idx * size_of(int))
    (cast(^int) base)^ = v
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
        c.nulls = new_nulls
    }
}

append_float :: proc(c: ^Column, v: f64) {
    if c.type != .Float {
        panic("append_float: wrong column type")
    }
    if c.len >= c.capacity {
        grow(c, max(8, c.capacity * 2))
    }

    idx := c.len
    c.len += 1

    base := uintptr(c.data) + uintptr(idx * size_of(f64))
    (cast(^f64) base)^ = v
}

append_bool :: proc(c: ^Column, v: bool) {
    if c.type != .Bool {
        panic("append_bool: wrong column type")
    }
    if c.len >= c.capacity {
        grow(c, max(8, c.capacity * 2))
    }

    idx := c.len
    c.len += 1

    base := uintptr(c.data) + uintptr(idx * size_of(bool))
    (cast(^bool) base)^ = v
}

append_string :: proc(c: ^Column, v: string) {
    if c.type != .String {
        panic("append_string: wrong column type")
    }
    if c.len >= c.capacity {
        grow(c, max(8, c.capacity * 2))
    }

    idx := c.len
    c.len += 1

    base := uintptr(c.data) + uintptr(idx * size_of(string))
    (cast(^string) base)^ = v
}

append_date :: proc(c: ^Column, v: Date) {
    if c.type != .Date {
        panic("append_date: wrong column type")
    }
    if c.len >= c.capacity {
        grow(c, max(8, c.capacity * 2))
    }

    idx := c.len
    c.len += 1

    base := uintptr(c.data) + uintptr(idx * size_of(Date))
    (cast(^Date) base)^ = v
}

append_null :: proc(c: ^Column) {
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



// Column read helpers (safe, copy-based readers)
column_at_int :: proc(col: ^Column, i: int) -> (int, bool) {
    if i < 0 || i >= col.len {
        panic("column_at_int: index out of range")
    }
    if col.nulls != nil && col.nulls[i] {
        return 0, true
    }
    base := uintptr(col.data) + uintptr(i * type_size(col.type))
    return (cast(^int) base)^, false
}

column_at_float :: proc(col: ^Column, i: int) -> (f64, bool) {
    if i < 0 || i >= col.len {
        panic("column_at_float: index out of range")
    }
    if col.nulls != nil && col.nulls[i] {
        return 0.0, true
    }
    base := uintptr(col.data) + uintptr(i * type_size(col.type))
    return (cast(^f64) base)^, false
}

column_at_bool :: proc(col: ^Column, i: int) -> (bool, bool) {
    if i < 0 || i >= col.len {
        panic("column_at_bool: index out of range")
    }
    if col.nulls != nil && col.nulls[i] {
        return false, true
    }
    base := uintptr(col.data) + uintptr(i * type_size(col.type))
    return (cast(^bool) base)^, false
}

column_at_string :: proc(col: ^Column, i: int) -> (string, bool) {
    if i < 0 || i >= col.len {
        panic("column_at_string: index out of range")
    }
    if col.nulls != nil && col.nulls[i] {
        return "", true
    }
    base := uintptr(col.data) + uintptr(i * type_size(col.type))
    return (cast(^string) base)^, false
}

column_at_date :: proc(col: ^Column, i: int) -> (Date, bool) {
    if i < 0 || i >= col.len {
        panic("column_at_date: index out of range")
    }
    if col.nulls != nil && col.nulls[i] {
        return Date{0,0,0}, true
    }
    base := uintptr(col.data) + uintptr(i * type_size(col.type))
    return (cast(^Date) base)^, false
}


column_slice_view :: proc(orig: ^Column, start: int, end: int) -> Column {
    if start < 0 || end < start || end > orig.len {
        panic("column_slice_view: invalid range")
    }
    c := column_new(orig.name, orig.type, end - start)
    c.orig = orig
    c.offset = start
    c.len = end - start
    c.is_view = true
    // do not allocate data/nulls for view
    c.data = nil
    c.nulls = nil
    return c
}
