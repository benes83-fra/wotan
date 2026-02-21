package main

import "core:fmt"
import "wotan:core"

main :: proc() {
    df := core.dataframe_new()

    col_age  := core.column_new("age", .Int, 4)
    col_name := core.column_new("name", .String, 4)

    core.append_int(&col_age, 10)
    core.append_string(&col_name, "Hubert")

    core.append_int(&col_age, 20)
    core.append_string(&col_name, "Anna")

    core.append_int(&col_age, 30)
    core.append_string(&col_name, "Markus")

    core.append_int(&col_age, 40)
    core.append_string(&col_name, "Julia")

    core.add_column(&df, col_age)
    core.add_column(&df, col_name)

    core.df_head(&df, 5)


    core.dataframe_print(&df)

    s := core.df_series(&df, "age")
    v, null := core.series_at_int(&s, 3)
    fmt.printf("age[2] = %d (null=%v)\n", v, null)
}
