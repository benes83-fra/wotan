package wotan

import "core:strings"

and :: proc {
    mask_and,
    column_and,
    mask_and_column,
    column_and_mask
}

or :: proc {
    mask_or,
    column_or,
    mask_or_column,
    column_or_mask,
}

not :: proc {
    mask_not,
    column_not,
}



// helpers
make_mask :: proc(n: int) -> []bool {
    return make([]bool, n)
}

mask_and :: proc(a: []bool, b: []bool) -> []bool {
    if len(a) != len(b) { panic("mask_and: length mismatch") }
    out := make_mask(len(a))
    for i in 0..<len(a) { out[i] = a[i] && b[i] }
    return out
}

mask_or :: proc(a: []bool, b: []bool) -> []bool {
    if len(a) != len(b) { panic("mask_or: length mismatch") }
    out := make_mask(len(a))
    for i in 0..<len(a) { out[i] = a[i] || b[i] }
    return out
}

mask_not :: proc(a: []bool) -> []bool {
    out := make_mask(len(a))
    for i in 0..<len(a) { out[i] = !a[i] }
    return out
}



column_and :: proc (a, b: ^Column) -> []bool {
 
    if a.len != b.len {
        panic ("column_and: Length mismatch")
    }
    out := make_mask(a.len)
    for i in 0..<a.len {
        a_val, a_ok := column_at_bool(a,i)
        b_val, b_ok := column_at_bool(b,i)
        if a_ok && b_ok {
            out[i] = a_val && b_val
        } else {
            out[i] = false
        }
    }
    return out
}

column_or :: proc (a, b: ^Column) -> []bool {
 
    if a.len != b.len {
        panic ("column_and: Length mismatch")
    }
    out := make_mask(a.len)
    for i in 0..< a.len {
        a_val, a_ok := column_at_bool(a,i)
        b_val, b_ok := column_at_bool(b,i)
        if a_ok && b_ok {
            out[i] = a_val || b_val
        } else {
            out[i] = false
        }
    }
    return out
}

column_not :: proc (a: ^Column) -> []bool {

   
    out:= make_mask(a.len)
    for i in 0..<a.len {
        a_val, a_ok := column_at_bool(a,i)
        if a_ok {
            out[i] = !a_val
        } else {
            out[i] = false
        }
    }
    return out
}

mask_and_column :: proc (a: []bool, b: ^Column) -> []bool{

    if len(a) != b.len {
        panic("mask_and_column: Length mismatch")
    }
    out := make_mask(len(a))
    for i in 0..<len(a) {
        b_val, b_ok := column_at_bool(b,i)
        if b_ok {
            out[i] = a[i] && b_val
        } else {
            out[i] = false
        }
    }
    return out


}

mask_or_column :: proc (a: []bool, b: ^Column) -> []bool{

    if len(a) != b.len {
        panic("mask_or_column: Length mismatch")
    }
    out := make_mask(len(a))
    for i in 0..<len(a) {
        b_val, b_ok := column_at_bool(b,i)
        if b_ok {
            out[i] = a[i] || b_val
        } else {
            out[i] = false
        }
    }
    return out


}


column_and_mask :: proc (a: ^Column, b: []bool) -> []bool{
    return mask_and_column(b,a)
}


column_or_mask :: proc (a: ^Column, b: []bool) -> []bool{
    return mask_or_column(b,a)
}



mask_gt :: proc{
    mask_gt_int,
    mask_gt_float,
    mask_gt_date,
}

mask_lt :: proc{
    mask_lt_int,
    mask_lt_float,
    mask_lt_date,
}
mask_eq :: proc{
    mask_eq_int,
    mask_eq_float,
    mask_eq_date,
    mask_eq_string,
}

mask_gt_int :: proc(col: ^Column, value: int) -> []bool {
    if col.type != .Int {
        panic("mask_gt_int: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_int(col, i)
        mask[i] = (!is_null && v > value)
    }
    return mask
}

mask_lt_int :: proc(col: ^Column, value: int) -> []bool {
    if col.type != .Int {
        panic("mask_lt_int: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_int(col, i)
        mask[i] = (!is_null && v < value)
    }
    return mask
}

mask_eq_int :: proc(col: ^Column, value: int) -> []bool {
    if col.type != .Int {
        panic("mask_eq_int: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_int(col, i)
        mask[i] = (!is_null && v == value)
    }
    return mask
}




mask_gt_float :: proc(col: ^Column, value: f64) -> []bool {
    if col.type != .Float {
        panic("mask_gt_float: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_float(col, i)
        mask[i] = (!is_null && v > value)
    }
    return mask
}

mask_lt_float :: proc(col: ^Column, value: f64) -> []bool {
    if col.type != .Float {
        panic("mask_lt_float: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_float(col, i)
        mask[i] = (!is_null && v < value)
    }
    return mask
}

mask_eq_float :: proc(col: ^Column, value: f64) -> []bool {
    if col.type != .Float {
        panic("mask_eq_float: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_float(col, i)
        mask[i] = (!is_null && v == value)
    }
    return mask
}

mask_gt_date :: proc(col: ^Column, value: Date) -> []bool {
    if col.type != .Date {
        panic("mask_gt_date: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_date(col, i)
        mask[i] = (!is_null && date_compare(v,value) > 0)
    }
    return mask
}

mask_lt_date :: proc(col: ^Column, value: Date) -> []bool {
    if col.type != .Date {
        panic("mask_lt_date: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_date(col, i)
        mask[i] = (!is_null && date_compare(v,value) < 0)
    }
    return mask
}

mask_eq_date :: proc(col: ^Column, value: Date) -> []bool {
    if col.type != .Date {
        panic("mask_eq_date: wrong column type")
    }

    mask := make([]bool, col.len)
    for i in 0..<col.len {
        v, is_null := column_at_date(col, i)
        mask[i] = (!is_null && date_compare(v,value) == 0)
    }
    return mask
}



mask_eq_string :: proc(col: ^Column, value: string) -> []bool {
	if col.type != .String {
		panic("mask_eq_string: wrong column type")
	}
	mask := make([]bool, col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_string(col, i)
		mask[i] = (!is_null && v == value)
	}
	return mask
}
mask_contains_string :: proc(col: ^Column, value: string) -> []bool {
	if col.type != .String {
		panic("mask_contains_string: wrong column type")
	}
	mask := make([]bool, col.len)
	for i in 0 ..< col.len {
		v, is_null := column_at_string(col, i)
		mask[i] = (!is_null && strings.contains(v, value))
	}
	return mask
}