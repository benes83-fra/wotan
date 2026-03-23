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
	defer free_all(context.temp_allocator)
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

	for i in 0 ..= 20 {
		date2 = w.add_day_date(date2, -100)
		fmt.printf("Date incremeneted to %v\n", date2)
	}
	fmt.println("Now we test some times")
	time1 := w.Time{16, 2, 58}

	for i in 0 ..= 20 {
		time1 = w.add_seconds_time(time1, -600)
		fmt.printf("Time increased to %v\n", time1)
	}
	dt := w.Datetime{1983, 7, 20, 13, 13, 13}
	for i in 0 ..= 20 {
		dt = w.add_hours_datetime(dt, -49)
		fmt.printf("Time elapsed %v \n", dt)
	}
	fmt.println(w.now())
	gdf := w.groupby(&df8, []string{"age"})
	exprs3 := []w.Agg_Expr {
		w.count("n"),
		w.sum_agg("avg_age", w.column(&df8, "age")),
		w.avg_agg("total_salary", w.column(&df8, "salary")),
	}
	out := w.agg(&gdf, exprs3)
	w.dataframe_pretty_print(&out, 20)


	w.destroy_grouped_dataframe(&gdf)
	w.destroy_dataframe(&out)

	// --- Insert this right before calling groupby(&df8, ...) ---

	people := w.dataframe_new()
	w.add_column(&people, w.column_from_ints("id", []int{1, 2, 3}))
	w.add_column(&people, w.column_from_strings("name", []string{"Alice", "Bob", "Charlie"}))
	w.add_column(&people, w.column_from_ints("age", []int{30, 20, 40}))
	people.rows = 3
	w.dataframe_pretty_print(&people, 20)
	salary := w.dataframe_new()
	w.add_column(&salary, w.column_from_ints("id", []int{1, 2, 4}))
	w.add_column(&salary, w.column_from_floats("salary", []f64{50000.00, 42000.00, 90000.00}))
	salary.rows = 3
	w.dataframe_pretty_print(&salary, 20)

	joined := w.join_single(int, &people, &salary, "id", .Outer, context.temp_allocator)

	fmt.println("Joined DF1:")

	w.dataframe_pretty_print(&joined, 20)


	left := w.dataframe_new()
	w.add_column(&left, w.column_from_ints("id", []int{1, 1, 2}))
	w.add_column(&left, w.column_from_ints("dept", []int{10, 20, 10}))
	w.add_column(&left, w.column_from_strings("name", []string{"Alice", "Bob", "Carol"}))
	left.rows = 3

	right := w.dataframe_new()
	w.add_column(&right, w.column_from_ints("id", []int{1, 2, 1}))
	w.add_column(&right, w.column_from_ints("dept", []int{10, 10, 30}))
	w.add_column(&right, w.column_from_floats("salary", []f64{50000, 60000, 70000}))
	right.rows = 3

	joined2 := w.join(&left, &right, []string{"id", "dept"}, .Inner, context.temp_allocator)

	fmt.println("Joined DF2:")
	w.dataframe_pretty_print(&joined2, 20)

	left2 := w.dataframe_new()
	w.add_column(&left2, w.column_from_ints("id", []int{1, 1, 2}))
	w.add_column(&left2, w.column_from_strings("country", []string{"DE", "US", "DE"}))
	w.add_column(&left2, w.column_from_strings("name", []string{"Alice", "Bob", "Carol"}))
	left2.rows = 3

	right2 := w.dataframe_new()
	w.add_column(&right2, w.column_from_ints("id", []int{1, 2, 1}))
	w.add_column(&right2, w.column_from_strings("country", []string{"DE", "DE", "FR"}))
	w.add_column(&right2, w.column_from_floats("salary", []f64{50000, 60000, 70000}))
	right2.rows = 3
	w.dataframe_pretty_print(&left2, 20)
	w.dataframe_pretty_print(&right2, 20)

	joined3 := w.join(
		&left2,
		&right2,
		[]string{"id", "country"},
		.Inner,
		allocator = context.temp_allocator,
	)
	join_test()


	w.dataframe_pretty_print(&joined3, 20)
	dfx := w.df_from(
		w.column_from_ints("id", []int{1, 2, 3}),
		w.column_from_strings("name", []string{"A", "B", "C"}),
		w.column_from_floats("salary", []f64{10, 20, 30}),
	)
	w.dataframe_pretty_print(&dfx, 20)


	w.destroy_dataframe(&dfx)
	w.destroy_dataframe(&left2)
	w.destroy_dataframe(&right2)
	w.destroy_dataframe(&joined3)
	w.destroy_dataframe(&left)
	w.destroy_dataframe(&right)
	w.destroy_dataframe(&joined2)

	w.destroy_dataframe(&joined)
	w.destroy_dataframe(&salary)
	w.destroy_dataframe(&people)
	w.destroy_dataframe(&df_active3)
	w.destroy_dataframe(&df8)
	w.destroy_dataframe(&df9)
	w.destroy_dataframe(&df10)
	w.destroy_dataframe(&df11)
	defer w.free_select_exprs(exprs)
	defer w.free_select_exprs(exprs2)

}
