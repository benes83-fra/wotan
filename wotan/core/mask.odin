package wotan


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



column_and :: proc (a, b: Column) -> []bool {
    a:=a
    b:=b
    if a.len != b.len {
        panic ("column_and: Length mismatch")
    }
    out := make_mask(a.len)
    for i in 0..<a.len {
        a_val, a_ok := column_at_bool(&a,i)
        b_val, b_ok := column_at_bool(&b,i)
        if a_ok && b_ok {
            out[i] = a_val && b_val
        } else {
            out[i] = false
        }
    }
    return out
}

column_or :: proc (a, b: Column) -> []bool {
    a:=a
    b:=b
    if a.len != b.len {
        panic ("column_and: Length mismatch")
    }
    out := make_mask(a.len)
    for i in 0..< a.len {
        a_val, a_ok := column_at_bool(&a,i)
        b_val, b_ok := column_at_bool(&b,i)
        if a_ok && b_ok {
            out[i] = a_val || b_val
        } else {
            out[i] = false
        }
    }
    return out
}

column_not :: proc (a: Column) -> []bool {

    a:=a
    out:= make_mask(a.len)
    for i in 0..<a.len {
        a_val, a_ok := column_at_bool(&a,i)
        if a_ok {
            out[i] = !a_val
        } else {
            out[i] = false
        }
    }
    return out
}

mask_and_column :: proc (a: []bool, b: Column) -> []bool{
    b:=b
    if len(a) != b.len {
        panic("mask_and_column: Length mismatch")
    }
    out := make_mask(len(a))
    for i in 0..<len(a) {
        b_val, b_ok := column_at_bool(&b,i)
        if b_ok {
            out[i] = a[i] && b_val
        } else {
            out[i] = false
        }
    }
    return out


}

mask_or_column :: proc (a: []bool, b: Column) -> []bool{
    b:=b
    if len(a) != b.len {
        panic("mask_or_column: Length mismatch")
    }
    out := make_mask(len(a))
    for i in 0..<len(a) {
        b_val, b_ok := column_at_bool(&b,i)
        if b_ok {
            out[i] = a[i] || b_val
        } else {
            out[i] = false
        }
    }
    return out


}


column_and_mask :: proc (a: Column, b: []bool) -> []bool{
    return mask_and_column(b,a)
}


column_or_mask :: proc (a: Column, b: []bool) -> []bool{
    return mask_or_column(b,a)
}