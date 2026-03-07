package main

import w "./wotan/core"
import csv "./wotan/importer"
import "core:fmt"
import "core:mem"
import "core:strings"

main :: proc() {
	when ODIN_DEBUG {
		default_allocator := context.allocator
		tracking_allocator: mem.Tracking_Allocator
		mem.tracking_allocator_init(&tracking_allocator, default_allocator)
		context.allocator = mem.tracking_allocator(&tracking_allocator)

		defer {
			if len(tracking_allocator.allocation_map) > 0 {
				for _, entry in tracking_allocator.allocation_map {
					fmt.eprintf("%v leaked %v bytes\n", entry.location, entry.size)

				}
			}
			mem.tracking_allocator_destroy(&tracking_allocator)
		}

	}

	df := w.dataframe_new()

	col_age := w.column_new("age", .Int, 4)
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

	w.destroy_dataframe(&df)
	types := []w.ColumnType{.Int, .String}
	df2 := csv.csv_load("people.csv", types)

	fmt.println("Here not here yet")
	w.dataframe_print(&df2)
	w.df_head(&df2, 5)
	w.destroy_dataframe(&df2)

	df3 := csv.csv_load("people_dates.csv")
	w.dataframe_print(&df3)
	w.df_head(&df3, 10)
	w.dataframe_pretty_print(&df3)


	df4 := w.dataframe_new()
	c1 := w.column_new("age", .Int, 4)
	w.append_int(&c1, 10)
	w.append_int(&c1, 20)
	w.append_int(&c1, 30)
	w.append_int(&c1, 40)
	w.add_column(&df4, c1)

	slice := w.dataframe_slice_rows_copy(&df4, 1, 3)
	w.df_head(&slice, 10) // should print rows 1 and 2 (20, 30)
	w.destroy_dataframe(&slice)

	fmt.println("Showing column selection:")
	df5 := w.dataframe_slice_rows(&df4, 1, 2, false)
	w.df_head(&df5, 5)
	w.destroy_dataframe(&df4)
	w.destroy_dataframe(&df5)
	df_age := w.dataframe_select_columns(&df3, []string{"age"}, false)
	w.df_head(&df_age, 5)

	w.destroy_dataframe(&df3)
	w.destroy_dataframe(&df_age)


	fmt.println("Showing Boolean filtering:")
	fmt.println("Without filter:")
	df6 := csv.csv_load("people_dates.csv")
	w.dataframe_pretty_print(&df6, 20)
	fmt.println("With filter:")
	// assume "active" is Bool
	df_active := w.dataframe_filter_bool_column(&df6, "active")
	w.dataframe_pretty_print(&df_active, 20)

	w.destroy_dataframe(&df6)
	w.destroy_dataframe(&df_active)
	fmt.println("Showing column filtering:")
	df7 := csv.csv_load("people_dates.csv")

	df_active2 := w.filter(&df7, "active")
	w.dataframe_pretty_print(&df_active2, 20)

	w.destroy_dataframe(&df7)
	w.destroy_dataframe(&df_active2)

	fmt.println("Filtering more complex booleans")
	df8 := csv.csv_load("people_dates.csv")
	c_mask := (w.column_lt(w.column(&df8, "age"), 31))
	bmask := w.column_mask(&c_mask)
	mask2 := w.column_mask(w.column(&df8, "active"))
	mask := w.mask_and(mask2, bmask)
	df_active3 := w.filter(&df8, mask)
	w.dataframe_pretty_print(&df_active3, 20)
	delete(mask)

	fmt.println("Using wobei/where with masks")
	//memory safe implementation. The syntax is more flexible otherwise, but there is some risk of leaks
	m1 := w.mask_lt(w.column(&df8, "age"), 31)
	m2 := w.column_mask(w.column(&df8, "active"))
	mask = w.and(m1, m2)
	delete(m1)
	delete(m2)

	df9 := w.wobei(&df8, mask)
	w.dataframe_pretty_print(&df9, 20)
	delete(mask)
	delete(mask2)
	delete(bmask)
	defer w.destroy_column(&c_mask)

	fmt.println("A simple select example")
	exprs := []w.Select_Expr {
		w.col_expr("age", w.column(&df8, "age")),
		w.add_expr("age_plus_10", w.column(&df8, "age"), 10),
		w.mask_expr("is_young", w.mask_lt(w.column(&df8, "age"), 30)),
	}

	df10 := w.select(&df8, exprs)
	w.dataframe_pretty_print(&df10, 20)

	fmt.println("Apply expressions on select:")
	exprs2 := []w.Select_Expr {
		w.col_expr("age", w.column(&df8, "age")),
		w.apply_expr("age_plus_5", w.column(&df8, "age"), proc(x: int) -> int {
			return x + 5
		}),
		w.apply_expr("upper_name", w.column(&df8, "name"), proc(s: string) -> string {
			return strings.to_upper(s, context.temp_allocator)
		}),
		w.apply_expr("is_even", w.column(&df8, "age"), proc(x: bool) -> bool {
			return x
		}),
		w.div_expr("Breaking Salaries", w.column(&df8, "salary"), 10),
		w.conv_int_to_f64_expr("Floating Salaries", w.column(&df8, "salary")),
		w.conv_expr("Conv Age", w.column(&df8, "age"), "float"),
		w.conv_expr("Birthday madness", w.column(&df8, "age"), "float"),
		w.col_expr("datetime", w.column(&df8, "birthday")),
	}
	df11 := w.select(&df8, exprs2)
	w.dataframe_pretty_print(&df11, 20)

	fmt.println("Some test with Dates")
	date1 := w.Date{2020, 2, 7}
	date2 := w.Date{2024, 2, 6}
	days := w.get_date_day_diffs(date1, date2)
	fmt.println("A lot of days:", days)
	for i in 0 ..= 5 {
		date1 = w.add_month_date(date1, -7)
		fmt.printf("Date incremeneted to %v\n", date1)
	}

	for i in 0 ..= 100 {
		date2 = w.add_day_date(date2, -100)
		fmt.printf("Date incremeneted to %v\n", date2)
	}
	fmt.println("Now we test some times")
	time1 := w.Time{16, 2, 58}

	for i in 0 ..= 20 {
		time1 = w.add_seconds_time(time1,-600)
		fmt.printf("Time increased to %v\n", time1)
	}

	w.destroy_dataframe(&df_active3)
	w.destroy_dataframe(&df8)
	w.destroy_dataframe(&df9)
	w.destroy_dataframe(&df10)
	w.destroy_dataframe(&df11)
	defer w.free_select_exprs(exprs)
	defer w.free_select_exprs(exprs2)

}
