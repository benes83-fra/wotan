package tests

import w "../wotan/core"
import "core:fmt"
import "core:mem"

indexing_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== INDEXING TEST ===")

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	// Unsorted dates
	col_date := w.column_new("Date", .Date, 5)
	w.append_date(&col_date, w.Date{2024, 1, 5})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 4})
	w.append_date(&col_date, w.Date{2024, 1, 1})
	w.append_date(&col_date, w.Date{2024, 1, 3})

	col_val := w.column_new("Value", .Int, 5)
	w.append_int(&col_val, 50)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 40)
	w.append_int(&col_val, 10)
	w.append_int(&col_val, 30)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	fmt.println("Original (unsorted):")
	w.dataframe_pretty_print(&df)

	// ------------------------------------------------------------
	// 1) SET INDEX (should sort by Date)
	// ------------------------------------------------------------
	w.set_index(&df, "Date")

	fmt.println("\nAfter set_index(Date):")
	w.dataframe_pretty_print(&df)

	// Expected order:
	// 2024-01-01 → 10
	// 2024-01-02 → 20
	// 2024-01-03 → 30
	// 2024-01-04 → 40
	// 2024-01-05 → 50

	// ------------------------------------------------------------
	// 2) Test loc on sorted index
	// ------------------------------------------------------------
	row := w.loc(&df, w.Date{2024, 1, 3})
	defer w.destroy_dataframe(&row)

	fmt.println("\nloc(Date{2024,1,3}):")
	w.dataframe_pretty_print(&row)

	// ------------------------------------------------------------
	// 3) Test loc_range
	// ------------------------------------------------------------
	slice := w.loc(&df, w.Date{2024, 1, 2}, w.Date{2024, 1, 4})
	defer w.destroy_dataframe(&slice)

	fmt.println("\nloc_range(Date{2024,1,2}, Date{2024,1,4}):")
	w.dataframe_pretty_print(&slice)

	// ------------------------------------------------------------
	// 4) Test iloc still works
	// ------------------------------------------------------------
	il := w.iloc(&df, 2)
	defer w.destroy_dataframe(&il)

	fmt.println("\niloc(2):")
	w.dataframe_pretty_print(&il)

	fmt.println("=== END INDEXING TEST ===")
}
