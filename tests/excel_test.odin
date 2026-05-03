
package tests


import w "../wotan/core"
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
