package main

import "core:fmt"
import w "./wotan/core"
import csv "./wotan/importer"

main :: proc() {
    df := w.dataframe_new()

    col_age  := w.column_new("age", .Int, 4)
    col_name := w.column_new("name", .String, 4)

    w.append_int(&col_age, 10)
    w.append_string(&col_name, "Hubert")

    w.append_int(&col_age, 20)
    w.append_string(&col_name, "Anna")

    w.append_int(&col_age, 30)
    w.append_string(&col_name, "Markus")

    w.append_int(&col_age, 40)
    w.append_string(&col_name, "Julia")

    w.add_column(&df, col_age)
    w.add_column(&df, col_name)

    w.df_head(&df, 5)


    w.dataframe_print(&df)

    s := w.df_series(&df, "age")
    v, null := w.series_at_int(&s, 3)
    fmt.printf("age[2] = %d (null=%v)\n", v, null)


    types := []w.ColumnType{ .Int, .String }
    df2 := csv.csv_load("people.csv", types)

    w.dataframe_print(&df2)
    w.df_head(&df2, 5)

    df3 := csv.csv_load_auto("people_dates.csv")
    w.dataframe_print(&df3)
    w.df_head(&df3, 10)

}
