package main
import w "./wotan/core"
import importer "./wotan/importer"
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
