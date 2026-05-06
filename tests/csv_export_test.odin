package tests

import w "../wotan/core"
import exporter "../wotan/exporter"
import importer "../wotan/importer"
import "core:fmt"
import "core:mem"

csv_export_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== CSV EXPORT TEST ===")

	// 1) Build a small DataFrame
	df := w.dataframe_new()

	defer w.destroy_dataframe(&df)

	w.add_column(&df, w.column_new("Name", .String, 3, allocator))
	w.add_column(&df, w.column_new("Age", .Int, 3, allocator))
	w.add_column(&df, w.column_new("Score", .Float, 3, allocator))

	w.append_string(&df.columns[0], "Alice")
	w.append_int(&df.columns[1], 30)
	w.append_float(&df.columns[2], 88.5)

	w.append_string(&df.columns[0], "Bob")
	w.append_int(&df.columns[1], 20)
	w.append_float(&df.columns[2], 91.0)

	w.append_string(&df.columns[0], "Charlie")
	w.append_null(&df.columns[1])
	w.append_float(&df.columns[2], 77.25)

	df.rows = 3

	// 2) Save to CSV
	ok := exporter.csv_save(&df, "export_test.csv", allocator)
	fmt.println("csv_save ok:", ok)

	// 3) Load back
	df2 := importer.csv_load("export_test.csv")

	// 4) Print both
	fmt.println("--- Original DF ---")
	w.dataframe_pretty_print(&df, 20)

	fmt.println("--- Loaded DF ---")
	w.dataframe_pretty_print(&df2, 20)

	// 5) Cleanup
	defer w.destroy_dataframe(&df2)

	fmt.println("=== END CSV EXPORT TEST ===")
}
