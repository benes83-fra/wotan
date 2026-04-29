package main

import "core:fmt"
import "core:mem"
import w "wotan/core"
import importer "wotan/importer"

json_basic_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== JSON load test ===")

	df_json := importer.json_load("example_data.json", importer.DEFAULT_NULL_TOKEN, allocator)
	w.dataframe_pretty_print(&df_json, 20)
	w.df_head(&df_json, 5)

	w.destroy_dataframe(&df_json)

	fmt.println("=== END JSON load test ===")
}
