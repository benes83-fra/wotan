package importer


import w "../core"
import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"

html_load :: proc(path: string, allocator: mem.Allocator) -> w.DataFrame {
	df := w.dataframe_new()

	contents, err := read_file(path)
	if err != nil {
		panic("html_load: cannot read file")
	}
	defer delete(contents)
	html := string(contents)

	start := strings.index(html, "<table")
	if start < 0 {return df}
	end := strings.index(html[start:], "</table>")
	if end < 0 {return df}

	table := html[start:start + end]

	rows := extract_rows(table)
	if len(rows) == 0 {return df}

	// PASS 1: parse all cells as strings
	tmp := make([][]string, len(rows), allocator)

	for r, r_i in rows {
		cells := extract_cells(r)

		tmp[r_i] = make([]string, len(cells), allocator)

		for cell, c_i in cells {
			clean := strip_html(cell)
			tmp[r_i][c_i] = clean
		}
	}

	header := tmp[0]
	row_count := len(tmp) - 1
	col_count := len(header)

	// PASS 2: infer column types
	inferred := infer_column_types(tmp)
	// PASS 3: create columns
	for name, i in header {
		col := w.column_new(name, inferred[i], row_count)
		w.add_column(&df, col)
	}

	// PASS 4: append values
	for r_i in 1 ..< len(tmp) {
		row := tmp[r_i]
		for value, c_i in row {
			append_value(&df.columns[c_i], value)
		}
	}
	df.rows = row_count
	return df
}

extract_rows :: proc(table: string) -> []string {
	rows := make([dynamic]string, 0, context.temp_allocator)

	start := 0
	for {
		i := strings.index(table[start:], "<tr")
		if i < 0 {
			break
		}
		i += start

		j := strings.index(table[i:], "</tr>")
		if j < 0 {
			break
		}
		j += i + len("</tr>")

		row := table[i:j]
		append(&rows, row)

		start = j
	}

	return rows[:]
}
extract_header :: proc(row: string) -> []string {
	headers := make([dynamic]string, 0, context.temp_allocator)

	cells := extract_cells(row)
	for c in cells {
		clean := strip_html(c)
		append(&headers, clean)
	}


	return headers[:]
}
extract_cells :: proc(row: string) -> []string {
	cells := make([dynamic]string, 0, context.temp_allocator)

	// Try <th> first
	tag := "<th"
	end_tag := "</th>"

	if strings.index(row, "<td") >= 0 {
		tag = "<td"
		end_tag = "</td>"
	}

	start := 0
	for {
		i := strings.index(row[start:], tag)
		if i < 0 {
			break
		}
		i += start

		// find '>' of opening tag
		gt := strings.index(row[i:], ">")
		if gt < 0 {
			break
		}
		gt += i + 1

		j := strings.index(row[gt:], end_tag)
		if j < 0 {
			break
		}
		j += gt

		cell := row[gt:j]
		append(&cells, cell)

		start = j + len(end_tag)
	}

	return cells[:]
}
strip_html :: proc(s: string, allocator: mem.Allocator = context.temp_allocator) -> string {
	// Remove tags: <...>
	out := strings.Builder{}
	strings.builder_init(&out, allocator)

	inside := false
	for b in s {
		if b == '<' {
			inside = true
			continue
		}
		if b == '>' {
			inside = false
			continue
		}
		if !inside {
			strings.write_byte(&out, u8(b))
		}
	}

	raw := strings.to_string(out)
	strings.builder_destroy(&out)
	err: bool
	// Decode minimal HTML entities
	raw, _ = strings.replace_all(raw, "&nbsp;", " ")
	raw, _ = strings.replace_all(raw, "&amp;", "&")
	raw, _ = strings.replace_all(raw, "&lt;", "<")
	raw, _ = strings.replace_all(raw, "&gt;", ">")
	raw, _ = strings.replace_all(raw, "&quot;", "\"")

	return raw
}
infer_and_append :: proc(col: ^w.Column, s: string) {
	trimmed := strings.trim_space(s)

	if trimmed == "" {
		w.append_null(col)
		return
	}

	// Try int
	if v, ok := strconv.parse_int(trimmed); ok {
		w.append_int(col, v)
		return
	}

	// Try float
	if v, ok := strconv.parse_f64(trimmed); ok {
		w.append_float(col, v)
		return
	}

	// Try bool
	if trimmed == "true" || trimmed == "TRUE" {
		w.append_bool(col, true)
		return
	}
	if trimmed == "false" || trimmed == "FALSE" {
		w.append_bool(col, false)
		return
	}

	// Try date/datetime/time?
	// If you want, we can add this later.

	// Fallback: string
	w.append_string(col, trimmed)
}
infer_column_types :: proc(tmp: [][]string) -> []w.ColumnType {
	col_count := len(tmp[0])
	types := make([]w.ColumnType, col_count, context.temp_allocator)

	for c in 0 ..< col_count {
		all_int := true
		all_float := true
		all_bool := true

		for r in 1 ..< len(tmp) {
			v := strings.trim_space(tmp[r][c])

			if v == "" {
				continue
			}

			if _, ok := strconv.parse_int(v); !ok {
				all_int = false
			}

			if _, ok := strconv.parse_f64(v); !ok {
				all_float = false
			}

			if !(v == "true" || v == "false" || v == "TRUE" || v == "FALSE") {
				all_bool = false
			}
		}

		if all_int {
			types[c] = .Int
		} else if all_float {
			types[c] = .Float
		} else if all_bool {
			types[c] = .Bool
		} else {
			types[c] = .String
		}
	}

	return types[:]
}
append_value :: proc(col: ^w.Column, s: string) {
	trimmed := strings.trim_space(s)

	if trimmed == "" {
		w.append_null(col)
		return
	}

	#partial switch col.type {
	case .Int:
		v, _ := strconv.parse_int(trimmed)
		w.append_int(col, v)

	case .Float:
		v, _ := strconv.parse_f64(trimmed)
		w.append_float(col, v)

	case .Bool:
		w.append_bool(col, trimmed == "true" || trimmed == "TRUE")

	case .String:
		w.append_string(col, trimmed)
	}
}
