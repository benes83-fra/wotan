package exporter

import w "../core"
import "core:mem"
import "core:os"
import "core:strings"

// One JSON object per row, keys in column order. Numbers keep their exact
// value (floats always carry a '.' or exponent so they stay floats), missing
// and non-finite values are null, dates / times are ISO strings.
@(private = "file")
_json_write_row :: proc(b: ^strings.Builder, df: ^w.DataFrame, row: int, allocator: mem.Allocator) {
	strings.write_byte(b, '{')
	for &col, c in df.columns {
		if c > 0 {strings.write_byte(b, ',')}
		json_write_string(b, col.name)
		strings.write_byte(b, ':')

		text, is_null := cell_text(&col, row, allocator)
		switch {
		case is_null:
			strings.write_string(b, "null")
		case col.type == .Int || col.type == .Bool:
			strings.write_string(b, text)
		case col.type == .Float:
			v, _ := w.column_at_float(&col, row)
			if v != v || v > max(f64) || v < -max(f64) {
				strings.write_string(b, "null") // JSON has no NaN / Inf
			} else {
				strings.write_string(b, text)
			}
		case:
			json_write_string(b, text)
		}
	}
	strings.write_byte(b, '}')
}

// ------------------------------------------------------------
// JSON Writer (DataFrame → JSON array of objects)
// ------------------------------------------------------------
json_write :: proc(df: ^w.DataFrame, path: string, allocator: mem.Allocator) {
	builder := strings.Builder{}
	strings.builder_init(&builder, allocator)
	defer strings.builder_destroy(&builder)

	strings.write_byte(&builder, '[')
	for row in 0 ..< df.rows {
		if row > 0 {strings.write_byte(&builder, ',')}
		_json_write_row(&builder, df, row, allocator)
	}
	strings.write_byte(&builder, ']')

	err := os.write_entire_file(path, transmute([]u8)strings.to_string(builder))
	if err != nil {
		panic("json_write: failed to write file")
	}
}

// ------------------------------------------------------------
// JSON Lines Writer (DataFrame → NDJSON)
// ------------------------------------------------------------
jsonl_write :: proc(df: ^w.DataFrame, path: string, allocator: mem.Allocator) {
	builder := strings.Builder{}
	strings.builder_init(&builder, allocator)
	defer strings.builder_destroy(&builder)

	for row in 0 ..< df.rows {
		_json_write_row(&builder, df, row, allocator)
		strings.write_byte(&builder, '\n')
	}

	err := os.write_entire_file(path, transmute([]u8)strings.to_string(builder))
	if err != nil {
		panic("jsonl_write: failed to write file")
	}
}
