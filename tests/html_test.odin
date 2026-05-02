package tests
import w "../wotan/core"
import exporter "../wotan/exporter"
import importer "../wotan/importer"
import "core:fmt"
import "core:mem"

html_basic_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== HTML load test ===")

	df := importer.html_load("example_table.html", allocator)
	w.dataframe_pretty_print(&df, 20)
	w.df_head(&df, 5)

	w.destroy_dataframe(&df)

	fmt.println("=== END HTML load test ===")
}


test_extract_tables :: proc() {
	contents, _ := importer.read_file("test_multi.html")
	html := string(contents)
	defer delete(contents)

	tables := importer.extract_tables(html)
	fmt.println("tables.len =", len(tables))

	for t, i in tables {
		fmt.println("--- TABLE", i, "---")
		fmt.println(t)
	}
}


test_index :: proc(allocator: mem.Allocator) {
	df0 := importer.html_load_table_index("test_multi.html", 0, allocator)
	df1 := importer.html_load_table_index("test_multi.html", 1, allocator)
	df2 := importer.html_load_table_index("test_multi.html", 2, allocator)
	defer {
		w.destroy_dataframe(&df0)
		w.destroy_dataframe(&df1)
		w.destroy_dataframe(&df2)
	}
	fmt.println("=== TABLE 0 ===")
	w.dataframe_pretty_print(&df0, 10)

	fmt.println("=== TABLE 1 ===")
	w.dataframe_pretty_print(&df1, 10)

	fmt.println("=== TABLE 2 ===")
	w.dataframe_pretty_print(&df2, 10)
}
test_id :: proc(allocator: mem.Allocator) {
	df := importer.html_load_table_id("test_multi.html", "second", allocator)
	defer w.destroy_dataframe(&df)
	fmt.println("=== TABLE second ===")
	w.dataframe_pretty_print(&df, 10)
}

test_all :: proc(allocator: mem.Allocator) {
	dfs := importer.html_load_all("test_multi.html", allocator)

	fmt.println("dfs.len =", len(dfs))

	for &df, i in dfs {
		fmt.println("=== TABLE", i, "===")
		w.dataframe_pretty_print(&df, 10)
		w.destroy_dataframe(&df)
	}
}

html_extended_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	test_extract_tables()
	test_id(allocator)
	test_index(allocator)
	test_all(allocator)
}

html_tags_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	df := importer.html_load_table_id("tag_table.html", "test", allocator)
	defer w.destroy_dataframe(&df)
	w.dataframe_pretty_print(&df, 20)


}


html_export_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== HTML EXPORT TEST ===")

	// Build a small DF
	df := w.dataframe_new()

	col_name := w.column_new("Name", .String, 4)
	col_age := w.column_new("Age", .Int, 4)
	col_score := w.column_new("Score", .Float, 4)

	w.append_string(&col_name, "Alice")
	w.append_int(&col_age, 30)
	w.append_float(&col_score, 88.5)

	w.append_string(&col_name, "Bob")
	w.append_int(&col_age, 20)
	w.append_float(&col_score, 91)

	w.append_string(&col_name, "Charlie")
	w.append_null(&col_age)
	w.append_float(&col_score, 77.25)

	w.append_string(&col_name, "Dora")
	w.append_int(&col_age, 40)
	w.append_null(&col_score)

	w.add_column(&df, col_name)
	w.add_column(&df, col_age)
	w.add_column(&df, col_score)
	df.rows = 4

	fmt.println("Original DF:")
	w.dataframe_pretty_print(&df, 20)

	// Export to HTML
	exporter.html_write(&df, "export_test.html", allocator)
	fmt.println("Wrote export_test.html")

	// Import again
	df2 := importer.html_load("export_test.html", allocator)
	fmt.println("Re-imported DF:")
	w.dataframe_pretty_print(&df2, 20)

	// Cleanup
	w.destroy_dataframe(&df)
	w.destroy_dataframe(&df2)

	fmt.println("=== END HTML EXPORT TEST ===")
}
