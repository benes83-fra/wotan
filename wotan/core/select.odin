package wotan

//
// --- Select Expression Types ------------------------------------------------
//

Select_Kind :: enum {
    Column,        // copy an existing column
    Mask,          // convert []bool mask to Bool column
    IntAdd,        // col + int
    IntSub,        // col - int
    FloatAdd,      // col + float
    FloatSub,      // col - float
    // extend later (String ops, Date ops, etc.)
}

Select_Expr :: struct {
    name:         string,
    kind:         Select_Kind,
    col:          ^Column,   // for column-based expressions
    mask:         []bool,    // for mask-based expressions
    int_value:    int,
    float_value:  f64,
    string_value: string,
}

//
// --- Convenience Constructors -----------------------------------------------
//

col_expr :: proc(name: string, col: ^Column) -> Select_Expr {
    return Select_Expr{
        name = name,
        kind = .Column,
        col  = col,
    }
}

mask_expr :: proc(name: string, mask: []bool) -> Select_Expr {
    return Select_Expr{
        name = name,
        kind = .Mask,
        mask = mask,
    }
}

add_int_expr :: proc(name: string, col: ^Column, v: int) -> Select_Expr {
    return Select_Expr{
        name      = name,
        kind      = .IntAdd,
        col       = col,
        int_value = v,
    }
}

sub_int_expr :: proc(name: string, col: ^Column, v: int) -> Select_Expr {
    return Select_Expr{
        name      = name,
        kind      = .IntSub,
        col       = col,
        int_value = v,
    }
}

add_float_expr :: proc(name: string, col: ^Column, v: f64) -> Select_Expr {
    return Select_Expr{
        name        = name,
        kind        = .FloatAdd,
        col         = col,
        float_value = v,
    }
}

sub_float_expr :: proc(name: string, col: ^Column, v: f64) -> Select_Expr {
    return Select_Expr{
        name        = name,
        kind        = .FloatSub,
        col         = col,
        float_value = v,
    }
}

//
// --- Select Overload Group --------------------------------------------------
//

select :: proc {
    select_columns,
    select_exprs,
}

//
// --- Column-only Select -----------------------------------------------------
//

select_columns :: proc(df: ^DataFrame, names: []string) -> DataFrame {
    return dataframe_select_columns(df, names, false)
}

//
// --- Expression-based Select ------------------------------------------------
//

select_exprs :: proc(df: ^DataFrame, exprs: []Select_Expr) -> DataFrame {
    out := dataframe_new()

    for expr in exprs {
        #partial switch expr.kind {

        // ---------------------------------------------------------------------
        case .Column:
            orig := expr.col
            new_col := column_new(expr.name, orig.type, orig.len)

            for i in 0..<orig.len {
                #partial switch orig.type {
                case .Int:
                    v, n := column_at_int(orig, i)
                    if n { append_null(&new_col) } else { append_int(&new_col, v) }

                case .Float:
                    v, n := column_at_float(orig, i)
                    if n { append_null(&new_col) } else { append_float(&new_col, v) }

                case .Bool:
                    v, n := column_at_bool(orig, i)
                    if n { append_null(&new_col) } else { append_bool(&new_col, v) }

                case .String:
                    v, n := column_at_string(orig, i)
                    if n { append_null(&new_col) } else { append_string(&new_col, v) }

                case .Date:
                    v, n := column_at_date(orig, i)
                    if n { append_null(&new_col) } else { append_date(&new_col, v) }
                }
            }

            add_column(&out, new_col)

        // ---------------------------------------------------------------------
        case .Mask:
            m := expr.mask
            new_col := column_new(expr.name, .Bool, len(m))

            for i in 0..<len(m) {
                append_bool(&new_col, m[i])
            }

            add_column(&out, new_col)

        // ---------------------------------------------------------------------
        case .IntAdd:
            orig := expr.col
            new_col := column_new(expr.name, .Int, orig.len)

            for i in 0..<orig.len {
                v, n := column_at_int(orig, i)
                if n { append_null(&new_col) } else { append_int(&new_col, v + expr.int_value) }
            }

            add_column(&out, new_col)

        // ---------------------------------------------------------------------
        case .IntSub:
            orig := expr.col
            new_col := column_new(expr.name, .Int, orig.len)

            for i in 0..<orig.len {
                v, n := column_at_int(orig, i)
                if n { append_null(&new_col) } else { append_int(&new_col, v - expr.int_value) }
            }

            add_column(&out, new_col)

        // ---------------------------------------------------------------------
        case .FloatAdd:
            orig := expr.col
            new_col := column_new(expr.name, .Float, orig.len)

            for i in 0..<orig.len {
                v, n := column_at_float(orig, i)
                if n { append_null(&new_col) } else { append_float(&new_col, v + expr.float_value) }
            }

            add_column(&out, new_col)

        // ---------------------------------------------------------------------
        case .FloatSub:
            orig := expr.col
            new_col := column_new(expr.name, .Float, orig.len)

            for i in 0..<orig.len {
                v, n := column_at_float(orig, i)
                if n { append_null(&new_col) } else { append_float(&new_col, v - expr.float_value) }
            }

            add_column(&out, new_col)
        }
    }

    return out
}
