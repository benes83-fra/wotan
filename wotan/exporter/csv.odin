package exporter

import w "../core"

import "core:fmt"
import "core:mem"
import "core:strings"

csv_save :: proc(
	df: ^w.DataFrame,
	path: string,
	sep: u8 = ',',
	allocator: mem.Allocator = context.allocator,
) -> bool {
	b, _ := strings.builder_make(allocator = allocator)
	defer strings.builder_destroy(&b)

	cols := len(df.columns)
	rows := df.rows

	// --- Write header ---
	for c in 0 ..< cols {
		write_csv_field(&b, df.columns[c].name, sep)
		if c < cols - 1 {
			strings.write_byte(&b, sep)
		}
	}
	strings.write_byte(&b, '\n')

	// --- Write rows ---
	for r in 0 ..< rows {
		for c in 0 ..< cols {
			col := &df.columns[c]

			if !w.is_null(col, r) {
				v := w.value_as_string(col, r, allocator)
				write_csv_field(&b, v, sep)
			}

			if c < cols - 1 {
				strings.write_byte(&b, sep)
			}
		}
		strings.write_byte(&b, '\n')
	}

	csv_text := strings.to_string(b)
	return write_file(path, csv_text)
}


write_csv_field :: proc(b: ^strings.Builder, s: string, sep: u8) {
	needs_quote := false

	for i in 0 ..< len(s) {
		ch := s[i]
		if ch == sep || ch == '"' || ch == '\n' || ch == '\r' {
			needs_quote = true
			break
		}
	}

	if !needs_quote {
		strings.write_string(b, s)
		return
	}

	// quoted field
	strings.write_byte(b, '"')

	for i in 0 ..< len(s) {
		ch := s[i]
		if ch == '"' {
			strings.write_string(b, `""`)
		} else {
			strings.write_byte(b, ch)
		}
	}

	strings.write_byte(b, '"')
}
