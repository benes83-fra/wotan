package importer

import w "../core"
import zm "../zip_min"

import "core:mem"
import "core:strconv"
import "core:strings"

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

xlsx_load :: proc(path: string, allocator: mem.Allocator) -> w.DataFrame {
	df := w.dataframe_new()

	// 1) sharedStrings (optional)
	shared_bytes, ok_shared := zm.zip_read_file(path, "xl/sharedStrings.xml", allocator)
	defer if ok_shared {delete(shared_bytes)}


	shared := []string{}
	if ok_shared {
		shared = parse_shared_strings(string(shared_bytes), allocator)
	}

	// 2) first worksheet: sheet1.xml
	sheet_bytes, ok_sheet := zm.zip_read_file(path, "xl/worksheets/sheet1.xml", allocator)
	if !ok_sheet {
		return df
	}
	defer delete(sheet_bytes)

	rows := parse_sheet_to_grid(string(sheet_bytes), shared, allocator)
	if len(rows) == 0 {
		return df
	}

	// 3) Build DataFrame (same pattern as HTML importer)
	header := rows[0]
	row_count := len(rows) - 1

	inferred := infer_column_types_excel(rows)

	for name, i in header {
		col := w.column_new(name, inferred[i], row_count)
		w.add_column(&df, col)
	}

	for r_i in 1 ..< len(rows) {
		row := rows[r_i]
		for value, c_i in row {
			append_value_excel(&df.columns[c_i], value)
		}
	}

	df.rows = row_count
	return df
}

// ------------------------------------------------------------
// Shared strings
// ------------------------------------------------------------

parse_shared_strings :: proc(xml: string, allocator: mem.Allocator) -> []string {
	out := make([dynamic]string, 0, allocator)

	start := 0
	for {
		i := strings.index(xml[start:], "<si")
		if i < 0 {
			break
		}
		i += start

		j := strings.index(xml[i:], "</si>")
		if j < 0 {
			break
		}
		j += i + len("</si>")

		si := xml[i:j]

		// find <t>...</t> inside <si>
		t_start := strings.index(si, "<t")
		if t_start >= 0 {
			gt := strings.index(si[t_start:], ">")
			if gt >= 0 {
				gt += t_start + 1
				t_end := strings.index(si[gt:], "</t>")
				if t_end >= 0 {
					t_end += gt
					text := si[gt:t_end]
					// minimal entity decoding
					text, _ = strings.replace_all(text, "&amp;", "&")
					text, _ = strings.replace_all(text, "&lt;", "<")
					text, _ = strings.replace_all(text, "&gt;", ">")
					text, _ = strings.replace_all(text, "&quot;", "\"")
					text, _ = strings.replace_all(text, "&apos;", "'")
					append(&out, text)
				}
			}
		}

		start = j
	}

	return out[:]
}

// ------------------------------------------------------------
// Sheet parsing
// ------------------------------------------------------------

parse_sheet_to_grid :: proc(
	xml: string,
	shared: []string,
	allocator: mem.Allocator,
) -> [][]string {
	rows_dyn := make([dynamic][]string, 0, allocator)

	start := 0
	for {
		i := strings.index(xml[start:], "<row")
		if i < 0 {
			break
		}
		i += start

		j := strings.index(xml[i:], "</row>")
		if j < 0 {
			break
		}
		j += i + len("</row>")

		row_xml := xml[i:j]
		row := parse_row(row_xml, shared, allocator)
		append(&rows_dyn, row)

		start = j
	}

	return rows_dyn[:]
}

parse_row :: proc(row_xml: string, shared: []string, allocator: mem.Allocator) -> []string {
	row := make([dynamic]string, 0, allocator)

	start := 0
	for {
		i := strings.index(row_xml[start:], "<c")
		if i < 0 {
			break
		}
		i += start

		j := strings.index(row_xml[i:], "</c>")
		if j < 0 {
			break
		}
		j += i + len("</c>")

		cell_xml := row_xml[i:j]

		col_idx := cell_ref_to_col(cell_xml)
		val := cell_value(cell_xml, shared)

		// ensure capacity
		if col_idx >= len(row) {
			needed := col_idx + 1 - len(row)
			for k in 0 ..< needed {
				append(&row, "")
			}
		}
		row[col_idx] = val

		start = j
	}

	return row[:]
}

// Extract column index from r="A1", r="BC12", etc.
cell_ref_to_col :: proc(cell_xml: string) -> int {
	r_pos := strings.index(cell_xml, " r=\"")
	if r_pos < 0 {
		return 0
	}
	r_pos += len(" r=\"")
	end := r_pos
	for end < len(cell_xml) && cell_xml[end] != '"' {
		end += 1
	}
	ref := cell_xml[r_pos:end]

	// letters prefix
	col := 0
	for i in 0 ..< len(ref) {
		ch := ref[i]
		if ch >= 'A' && ch <= 'Z' {
			col = col * 26 + int(ch - 'A' + 1)
		} else if ch >= 'a' && ch <= 'z' {
			col = col * 26 + int(ch - 'a' + 1)
		} else {
			break
		}
	}
	return col - 1 // zero-based
}

cell_value :: proc(cell_xml: string, shared: []string) -> string {
	// type attribute: t="s" => shared string
	is_shared := false
	t_pos := strings.index(cell_xml, " t=\"")
	if t_pos >= 0 {
		t_pos += len(" t=\"")
		end := t_pos
		for end < len(cell_xml) && cell_xml[end] != '"' {
			end += 1
		}
		tval := cell_xml[t_pos:end]
		if tval == "s" {
			is_shared = true
		}
	}

	v_start := strings.index(cell_xml, "<v>")
	if v_start < 0 {
		return ""
	}
	v_start += len("<v>")
	v_end := strings.index(cell_xml[v_start:], "</v>")
	if v_end < 0 {
		return ""
	}
	v_end += v_start

	raw := cell_xml[v_start:v_end]

	if is_shared {
		idx, ok := strconv.parse_int(raw)
		if ok && idx >= 0 && idx < len(shared) {
			return shared[idx]
		}
		return ""
	}

	return raw
}

// ------------------------------------------------------------
// Type inference + append (reuse pattern from HTML importer)
// ------------------------------------------------------------

infer_column_types_excel :: proc(tmp: [][]string) -> []w.ColumnType {
	col_count := len(tmp[0])
	types := make([]w.ColumnType, col_count, context.temp_allocator)

	for c in 0 ..< col_count {
		all_int := true
		all_float := true
		all_bool := true

		for r in 1 ..< len(tmp) {
			if c >= len(tmp[r]) {
				continue
			}
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

append_value_excel :: proc(col: ^w.Column, s: string) {
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
