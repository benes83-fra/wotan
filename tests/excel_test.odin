
package tests


import w "../wotan/core"
import importer "../wotan/importer"
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
