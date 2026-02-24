package wotan

import "core:fmt"

DataFrame :: struct {
    columns:       [dynamic]Column,
    name_to_index: map[string]int,
    rows:          int,
}

//
// --- Constructors ---
//

dataframe_new :: proc() -> DataFrame {
    return DataFrame{
        columns       = make([dynamic]Column),
        name_to_index = make(map[string]int),
        rows          = 0,
    }
}

//
// --- Column management ---
//

add_column :: proc(df: ^DataFrame, col: Column) {
    if len(df.columns) == 0 {
        df.rows = col.len
    } else if col.len != df.rows {
        panic("DataFrame.add_column: column length mismatch")
    }

    if col.name in df.name_to_index {
        panic("DataFrame.add_column: duplicate column name")
    }

    df.name_to_index[col.name] = len(df.columns)
    _, err := append(&df.columns, col)
    if err != nil {
        panic("DataFrame.add_column: append failed")
    }
}

column :: proc(df: ^DataFrame, name: string) -> ^Column {
    idx, ok := df.name_to_index[name]
    if !ok {
        panic(fmt.tprintf("DataFrame.column: no such column '%s'", name))
    }
    return &df.columns[idx]
}

df_series :: proc(df: ^DataFrame, name: string) -> Series {
    col := column(df, name)
    return Series{
        name  = col.name,
        type  = col.type,
        len   = col.len,
        data  = col.data,
        nulls = col.nulls,
    }
}



//
// --- Debug printing ---
//

dataframe_print :: proc(df: ^DataFrame) {
    fmt.println("DataFrame:")
    fmt.printf("  rows: %d\n", df.rows)
    fmt.printf("  columns: %d\n", len(df.columns))

    for col in df.columns {
        fmt.printf("    %s (%v)\n", col.name, col.type)
    }
}


df_head :: proc(df: ^DataFrame, n: int) {
    en :=n
    if en > df.rows {
        en = df.rows
    }

    // Print header
    fmt.print("    ")
    for col in df.columns {
        fmt.printf("%s\t", col.name)
    }
    fmt.println()

    // Print rows
    for i in 0..<en {
        fmt.printf("%3d ", i)

        for &col in df.columns {
            if col.is_view {
                v, is_null := column_at_int(&col, i)
                if is_null {
                    fmt.print("NULL\t")
                } else {
                    fmt.printf("%v\t", v)
                }
                continue
            }
            base := uintptr(col.data) + uintptr(i * type_size(col.type))

            if col.nulls != nil && col.nulls[i] {
                fmt.print("NULL\t")
                continue
            }

            #partial switch col.type {
            case .Int:
                fmt.printf("%v\t", (cast(^int) base)^)
            case .Float:
                fmt.printf("%v\t", (cast(^f64) base)^)
            case .Bool:
                fmt.printf("%v\t", (cast(^bool) base)^)
            case .String:
                fmt.printf("%v\t", (cast(^string) base)^)
            case .Date:
                d := (cast(^Date) base)^
                fmt.printf("%04d-%02d-%02d\t", d.year, d.month, d.day)
            }
        }

        fmt.println()
    }
}



dataframe_select_columns :: proc(df: ^DataFrame, names: []string, copy: bool) -> DataFrame {
    out := dataframe_new()

    for name in names {
        col := column(df, name) // existing lookup

        new_col: Column
        if copy {
            // deep copy using your existing copy slice
            new_col = column_slice_copy(col, 0, col.len)
        } else {
            // view of entire column
            new_col = column_slice_view(col, 0, col.len)
        }

        add_column(&out, new_col)
    }

    return out
}
