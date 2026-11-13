package exporter

import w "../core"
import "core:mem"
import vmem "core:mem/virtual"
import "core:os"
import "core:strings"

// ------------------------------------------------------------
// HTML Writer (DataFrame → HTML <table>)
// Cell text is HTML-escaped; null cells are empty <td></td>.
// ------------------------------------------------------------
html_write :: proc(df: ^w.DataFrame, path: string, allocator: mem.Allocator) {
	builder := strings.Builder{}
	strings.builder_init(&builder, allocator)
	defer strings.builder_destroy(&builder)

	strings.write_string(&builder, "<meta charset=\"utf-8\">\n<table>\n")

	// HEADER
	strings.write_string(&builder, "  <thead>\n    <tr>")
	for &col in df.columns {
		strings.write_string(&builder, "<th>")
		xml_escape_into(&builder, col.name)
		strings.write_string(&builder, "</th>")
	}
	strings.write_string(&builder, "</tr>\n  </thead>\n")

	// BODY
	strings.write_string(&builder, "  <tbody>\n")
	scratch: Cell_Scratch
	cell_alloc := cell_scratch_init(&scratch)
	defer vmem.arena_destroy(&scratch.arena)
	for row in 0 ..< df.rows {
		row_mark := vmem.arena_temp_begin(&scratch.arena)
		defer vmem.arena_temp_end(row_mark)
		strings.write_string(&builder, "    <tr>")
		for &col in df.columns {
			strings.write_string(&builder, "<td>")
			text, is_null := cell_text(&col, row, cell_alloc)
			if !is_null {xml_escape_into(&builder, text)}
			strings.write_string(&builder, "</td>")
		}
		strings.write_string(&builder, "</tr>\n")
	}
	strings.write_string(&builder, "  </tbody>\n</table>\n")

	err := os.write_entire_file(path, transmute([]u8)strings.to_string(builder))
	if err != nil {
		panic("html_write: failed to write file")
	}
}
