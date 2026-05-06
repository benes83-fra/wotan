
package tests


import w "../wotan/core"
import exporter "../wotan/exporter"
import importer "../wotan/importer"
import zm "../wotan/zip_min"
import "core:fmt"
import "core:mem"

excel_basic_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== EXCEL LOAD TEST ===")

	df := importer.xlsx_load("example.xlsx", allocator)
	w.dataframe_pretty_print(&df, 20)
	w.df_head(&df, 10)

	w.destroy_dataframe(&df)

	fmt.println("=== END EXCEL LOAD TEST ===")
}


zip_smoke_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== ZIP SMOKE TEST ===")
	bytes, ok := zm.zip_read_file("example.xlsx", "xl/workbook.xml", allocator)
	fmt.println("ok:", ok, "len:", len(bytes))
	if ok {
		s := string(bytes)
		fmt.println("First 200 chars of workbook.xml:")
		fmt.println(s[:min(200, len(s))])
	}
	fmt.println("=== END ZIP SMOKE TEST ===")
}

excel_date_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== EXCEL DATE TEST ===")

	df := importer.xlsx_load("datetime.xlsx", allocator)

	w.dataframe_pretty_print(&df, 20)
	w.df_head(&df, 10)

	w.destroy_dataframe(&df)

	fmt.println("=== END EXCEL DATE TEST ===")
}


excel_export_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== EXCEL EXPORT TEST ===")

	// 1) Build a small DataFrame
	df := w.dataframe_new()

	col_name := w.column_new("Name", .String, 3)
	col_age := w.column_new("Age", .Int, 3)
	col_score := w.column_new("Score", .Float, 3)

	w.add_column(&df, col_name)
	w.add_column(&df, col_age)
	w.add_column(&df, col_score)

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

	// 2) Save to Excel
	ok := exporter.xlsx_save(&df, "export_test.xlsx", allocator)
	fmt.println("xlsx_save ok:", ok)

	// 3) Load back
	df2 := importer.xlsx_load("export_test.xlsx", allocator)

	// 4) Print both
	fmt.println("--- Original DF ---")
	w.dataframe_pretty_print(&df, 20)

	fmt.println("--- Loaded DF ---")
	w.dataframe_pretty_print(&df2, 20)

	// cleanup
	w.destroy_dataframe(&df)
	w.destroy_dataframe(&df2)

	fmt.println("=== END EXCEL EXPORT TEST ===")
}
