package wotan

import "core:fmt"

// Basic accessors

dataframe_column_count :: proc(df: ^DataFrame) -> int {
    return len(df.columns)
}

dataframe_row_count :: proc(df: ^DataFrame) -> int {
    return df.rows
}

dataframe_column_name :: proc(df: ^DataFrame, idx: int) -> string {
    if idx < 0 || idx >= len(df.columns) {
        panic(fmt.tprintf("dataframe_column_name: index out of range %d", idx))
    }
    return df.columns[idx].name
}

dataframe_column_type :: proc(df: ^DataFrame, idx: int) -> ColumnType {
    if idx < 0 || idx >= len(df.columns) {
        panic(fmt.tprintf("dataframe_column_type: index out of range %d", idx))
    }
    return df.columns[idx].type
}

dataframe_column_ptr :: proc(df: ^DataFrame, idx: int) -> ^Column {
    if idx < 0 || idx >= len(df.columns) {
        panic(fmt.tprintf("dataframe_column_ptr: index out of range %d", idx))
    }
    return &df.columns[idx]
}
