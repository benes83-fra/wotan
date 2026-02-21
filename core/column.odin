package wotan

import "core:mem"

Column :: struct {
    name:     string,
    type:     ColumnType,
    len:      int,
    capacity: int,
    data:     rawptr,
    nulls:    []bool,
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
