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
