package tests

import w "../wotan/core"
import "core:fmt"
import "core:mem"

loc_slice_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LOC / LOC_RANGE TEST ===")

	df := w.dataframe_new()

	// Date index column
	col_date := w.column_new("Date", .Date, 5)
	w.append_date(&col_date, w.Date{2024, 1, 1})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 3})
	w.append_date(&col_date, w.Date{2024, 1, 4})
	w.append_date(&col_date, w.Date{2024, 1, 5})

	// Some payload column
	col_val := w.column_new("Value", .Int, 5)
	w.append_int(&col_val, 10)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 30)
	w.append_int(&col_val, 40)
	w.append_int(&col_val, 50)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	// Set index
	w.set_index(&df, "Date")

	fmt.println("Original DataFrame:")
	w.dataframe_pretty_print(&df, max_rows = 20)

	// Scalar loc
	row := w.loc(&df, w.Date{2024, 1, 3})
	fmt.println("\nloc(Date{2024-01-03}):")
	w.dataframe_pretty_print(&row, max_rows = 20)

	// Range loc
	slice := w.loc_range(&df, w.Date{2024, 1, 2}, w.Date{2024, 1, 4})
	fmt.println("\nloc_range(Date{2024-01-02}, Date{2024-01-04}):")
	w.dataframe_pretty_print(&slice, max_rows = 20)

	w.destroy_dataframe(&df)
	w.destroy_dataframe(&row)
	w.destroy_dataframe(&slice)

	fmt.println("=== END LOC / LOC_RANGE TEST ===")
}


loc_many_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LOC_MANY TEST ===")

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	col_date := w.column_new("Date", .Date, 5)
	w.append_date(&col_date, w.Date{2024, 1, 1})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 3})
	w.append_date(&col_date, w.Date{2024, 1, 4})
	w.append_date(&col_date, w.Date{2024, 1, 5})

	col_val := w.column_new("Value", .Int, 5)
	w.append_int(&col_val, 10)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 30)
	w.append_int(&col_val, 40)
	w.append_int(&col_val, 50)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	w.set_index(&df, "Date")

	fmt.println("Original DataFrame:")
	w.dataframe_pretty_print(&df)

	keys := []w.Date {
		{2024, 1, 4},
		{2024, 1, 2},
		{2024, 1, 4}, // duplicate
		{2024, 1, 1},
	}

	out := w.loc_many(&df, keys)
	defer w.destroy_dataframe(&out)
	fmt.println("\nloc_many result:")
	w.dataframe_pretty_print(&out)

	fmt.println("=== END LOC_MANY TEST ===")
}

loc_from_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LOC_FROM TEST ===")

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	col_date := w.column_new("Date", .Date, 5)
	w.append_date(&col_date, w.Date{2024, 1, 1})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 3})
	w.append_date(&col_date, w.Date{2024, 1, 4})
	w.append_date(&col_date, w.Date{2024, 1, 5})

	col_val := w.column_new("Value", .Int, 5)
	w.append_int(&col_val, 10)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 30)
	w.append_int(&col_val, 40)
	w.append_int(&col_val, 50)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	w.set_index(&df, "Date")

	fmt.println("Original DataFrame:")
	w.dataframe_pretty_print(&df)

	out := w.loc_from(&df, w.Date{2024, 1, 3})
	defer w.destroy_dataframe(&out)

	fmt.println("\nloc_from(Date{2024,1,3}):")
	w.dataframe_pretty_print(&out)

	fmt.println("=== END LOC_FROM TEST ===")
}
loc_until_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LOC_UNTIL TEST ===")

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	col_date := w.column_new("Date", .Date, 5)
	w.append_date(&col_date, w.Date{2024, 1, 1})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 3})
	w.append_date(&col_date, w.Date{2024, 1, 4})
	w.append_date(&col_date, w.Date{2024, 1, 5})

	col_val := w.column_new("Value", .Int, 5)
	w.append_int(&col_val, 10)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 30)
	w.append_int(&col_val, 40)
	w.append_int(&col_val, 50)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	w.set_index(&df, "Date")

	fmt.println("Original DataFrame:")
	w.dataframe_pretty_print(&df)

	out := w.loc_until(&df, w.Date{2024, 1, 3})
	defer w.destroy_dataframe(&out)

	fmt.println("\nloc_until(Date{2024,1,3}):")
	w.dataframe_pretty_print(&out)

	fmt.println("=== END LOC_UNTIL TEST ===")
}
loc_mask_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== LOC_MASK TEST ===")

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	col_date := w.column_new("Date", .Date, 5)
	w.append_date(&col_date, w.Date{2024, 1, 1})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 3})
	w.append_date(&col_date, w.Date{2024, 1, 4})
	w.append_date(&col_date, w.Date{2024, 1, 5})

	col_val := w.column_new("Value", .Int, 5)
	w.append_int(&col_val, 10)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 30)
	w.append_int(&col_val, 40)
	w.append_int(&col_val, 50)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	w.set_index(&df, "Date")

	fmt.println("Original DataFrame:")
	w.dataframe_pretty_print(&df)

	// Mask: select rows where Value >= 30
	mask := []bool{false, false, true, true, true}

	out := w.loc_mask(&df, mask)
	defer w.destroy_dataframe(&out)

	fmt.println("\nloc_mask(Value >= 30):")
	w.dataframe_pretty_print(&out)

	fmt.println("=== END LOC_MASK TEST ===")
}
