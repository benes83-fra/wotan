package tests

import w "../wotan/core"
import exporter "../wotan/exporter"
import importer "../wotan/importer"
import "core:fmt"
import "core:mem"
import "core:os"

json_basic_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== JSON load test ===")

	df_json := importer.json_load("example_data.json", importer.DEFAULT_NULL_TOKEN, allocator)
	w.dataframe_pretty_print(&df_json, 20)
	w.df_head(&df_json, 5)

	w.destroy_dataframe(&df_json)

	fmt.println("=== END JSON load test ===")
}


jsonl_basic_test :: proc(allocator: mem.Allocator = context.temp_allocator) {
	fmt.println("=== JSONL load test ===")

	df_jsonl := importer.jsonl_load("example_data2.jsonl", importer.DEFAULT_NULL_TOKEN, allocator)
	w.dataframe_pretty_print(&df_jsonl, 20)
	w.df_head(&df_jsonl, 5)

	w.destroy_dataframe(&df_jsonl)

	fmt.println("=== END JSONL load test ===")
}

json_export_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== JSON Export Test ===")

	// Create a DataFrame
	df := w.dataframe_new()
	defer w.destroy_dataframe(&df)
	// Create columns
	col_name := w.column_new("name", .String, 4)
	col_age := w.column_new("age", .Int, 4)
	col_score := w.column_new("score", .Float, 4)

	// Append rows
	w.append_string(&col_name, "Alice")
	w.append_int(&col_age, 30)
	w.append_float(&col_score, 88.5)

	w.append_string(&col_name, "Bob")
	w.append_int(&col_age, 20)
	w.append_float(&col_score, 91.0)

	w.append_string(&col_name, "Charlie")
	w.append_null(&col_age)
	w.append_float(&col_score, 77.25)

	w.append_string(&col_name, "Dora")
	w.append_int(&col_age, 40)
	w.append_null(&col_score)

	// Add columns to DataFrame
	w.add_column(&df, col_name)
	w.add_column(&df, col_age)
	w.add_column(&df, col_score)

	// Write JSON array
	exporter.json_write(&df, "out.json", allocator)

	// Write NDJSON
	exporter.jsonl_write(&df, "out.jsonl", allocator)

	fmt.println("Wrote out.json and out.jsonl")

	// Read back and print for verification
	contents_json, _ := importer.read_file("out.json")
	contents_jsonl, _ := importer.read_file("out.jsonl")
	defer delete(contents_json)
	defer delete(contents_jsonl)
	fmt.println("--- out.json ---")
	fmt.println(string(contents_json))

	fmt.println("--- out.jsonl ---")
	fmt.println(string(contents_jsonl))

	fmt.println("=== END JSON Export Test ===")
}
