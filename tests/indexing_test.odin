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
reset_index_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== RESET INDEX TEST ===")

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	// Build sorted frame
	col_date := w.column_new("Date", .Date, 3)
	w.append_date(&col_date, w.Date{2024, 1, 1})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 3})

	col_val := w.column_new("Value", .Int, 3)
	w.append_int(&col_val, 10)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 30)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	// Set index
	w.set_index(&df, "Date")

	fmt.println("After set_index(Date):")
	w.dataframe_pretty_print(&df)

	// ------------------------------------------------------------
	// 1) reset_index(drop=false)
	// ------------------------------------------------------------
	df_copy := w.materialize(&df) // isolate test
	defer w.destroy_dataframe(&df_copy)

	w.reset_index(&df_copy, false)

	fmt.println("\nAfter reset_index(drop=false):")
	w.dataframe_pretty_print(&df_copy)

	// Expected:
	// Columns: [Date, Date, Value]
	// First Date = old index
	// Second Date = original column
	// Index metadata cleared

	// ------------------------------------------------------------
	// 2) reset_index(drop=true)
	// ------------------------------------------------------------
	df_copy2 := w.materialize(&df) // isolate test
	defer w.destroy_dataframe(&df_copy2)

	w.reset_index(&df_copy2, true)

	fmt.println("\nAfter reset_index(drop=true):")
	w.dataframe_pretty_print(&df_copy2)

	// Expected:
	// Columns: [Value]
	// Index column removed entirely
	// Index metadata cleared

	fmt.println("=== END RESET INDEX TEST ===")
}

reindex_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== REINDEX TEST ===")

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	// Build sorted frame
	col_date := w.column_new("Date", .Date, 3)
	w.append_date(&col_date, w.Date{2024, 1, 1})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 3})

	col_val := w.column_new("Value", .Int, 3)
	w.append_int(&col_val, 10)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 30)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	// Set index
	w.set_index(&df, "Date")

	fmt.println("Original (indexed):")
	w.dataframe_pretty_print(&df)

	// New index with:
	// - reordered rows
	// - missing row (NULL)
	// - extra row (NULL)
	new_idx := []w.Date {
		w.Date{2024, 1, 3}, // existing
		w.Date{2024, 1, 1}, // existing
		w.Date{2024, 1, 5}, // missing
		w.Date{2024, 1, 2}, // existing
	}

	out := w.reindex(&df, new_idx)
	defer w.destroy_dataframe(&out)

	fmt.println("\nAfter reindex([...]):")
	w.dataframe_pretty_print(&out)

	// Expected:
	// 2024-01-03 → 30
	// 2024-01-01 → 10
	// 2024-01-05 → NULL
	// 2024-01-02 → 20

	fmt.println("=== END REINDEX TEST ===")
}


set_index_drop_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("=== SET INDEX DROP TEST ===")

	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)

	// Build unsorted frame
	col_date := w.column_new("Date", .Date, 4)
	w.append_date(&col_date, w.Date{2024, 1, 4})
	w.append_date(&col_date, w.Date{2024, 1, 2})
	w.append_date(&col_date, w.Date{2024, 1, 3})
	w.append_date(&col_date, w.Date{2024, 1, 1})

	col_val := w.column_new("Value", .Int, 4)
	w.append_int(&col_val, 40)
	w.append_int(&col_val, 20)
	w.append_int(&col_val, 30)
	w.append_int(&col_val, 10)

	w.add_column(&df, col_date)
	w.add_column(&df, col_val)

	fmt.println("Original (unsorted):")
	w.dataframe_pretty_print(&df)

	// ------------------------------------------------------------
	// 1) set_index(drop=false)
	// ------------------------------------------------------------
	df_copy := w.materialize(&df)
	defer w.destroy_dataframe(&df_copy)

	w.set_index(&df_copy, "Date", false)

	fmt.println("\nAfter set_index(Date, drop=false):")
	w.dataframe_pretty_print(&df_copy)

	// Expected:
	// Columns: [Date, Value]
	// Sorted by Date ascending
	// Index column still present

	// ------------------------------------------------------------
	// 2) set_index(drop=true)
	// ------------------------------------------------------------
	df_copy2 := w.materialize(&df)
	defer w.destroy_dataframe(&df_copy2)

	w.set_index(&df_copy2, "Date", true)

	fmt.println("\nAfter set_index(Date, drop=true):")
	w.dataframe_pretty_print(&df_copy2)

	// Expected:
	// Columns: [Value]
	// Sorted by Date ascending
	// Index column removed entirely

	fmt.println("=== END SET INDEX DROP TEST ===")
}
