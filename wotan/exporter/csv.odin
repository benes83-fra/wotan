package exporter

import w "../core"
import "core:mem"
import "core:strings"

// DataFrame → CSV. Round-trips through importer.csv_load: null cells are
// written as empty unquoted fields, strings that would otherwise read back
// differently (empty, padded, null tokens, separators, quotes, newlines) are
// quoted, floats keep their exact value and type, dates use ISO format.
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
			text, is_null := cell_text(col, r, allocator)
			if !is_null {
				if col.type == .String {
					write_csv_field(&b, text, sep)
				} else {
					strings.write_string(&b, text)
				}
			}
			if c < cols - 1 {
				strings.write_byte(&b, sep)
			}
		}
		strings.write_byte(&b, '\n')
	}

	return write_file(path, strings.to_string(b))
}


write_csv_field :: proc(b: ^strings.Builder, s: string, sep: u8) {
	if !csv_field_needs_quotes(s, sep) {
		strings.write_string(b, s)
		return
	}

	// quoted field, embedded quotes doubled
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
