package importer

import w "../core"
import zm "../zip_min"

import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"


SheetInfo :: struct {
	name: string,
	rid:  string,
	path: string, // resolved "xl/worksheets/....xml"
}

RelInfo :: struct {
	id:     string,
	target: string,
}


// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

xlsx_load :: proc(path: string, allocator: mem.Allocator) -> w.DataFrame {
	df := w.dataframe_new()

	fmt.println(path)
	sheets := discover_sheets(path, allocator)
	if len(sheets) == 0 {
		fmt.println("No sheets found.")
		return df
	}
	first_sheet := sheets[0]

	// 1) sharedStrings (optional)
	shared_bytes, ok_shared := zm.zip_read_file(path, "xl/sharedStrings.xml", allocator)
	// defer if ok_shared {delete(shared_bytes)}

	shared := []string{}
	if ok_shared {
		shared = parse_shared_strings(string(shared_bytes), allocator)
	}

	// 2) load first worksheet by resolved path
	sheet_bytes, ok_sheet := zm.zip_read_file(path, first_sheet.path, allocator)
	if !ok_sheet {
		return df
	}

	styles_bytes, ok_styles := zm.zip_read_file(path, "xl/styles.xml", allocator)
	styles_is_date := []bool{}
	if ok_styles {
		styles_is_date = parse_styles_date_formats(string(styles_bytes), allocator)
	}
	rows := parse_sheet_to_grid(string(sheet_bytes), shared, styles_is_date, allocator)

	if len(rows) == 0 {
		return df
	}


	header := rows[0]
	row_count := len(rows) - 1

	inferred := infer_column_types_excel(rows)

	for name, i in header {
		safe_name := name

		// Optional: avoid empty names
		if strings.trim_space(safe_name) == "" {
			safe_name = fmt.aprintf("col_%d", i, allocator = allocator) // allocates a fresh string

		} else {
			// Force a copy so we don't keep a pointer into the XML buffer arena
			safe_name = fmt.aprintf("%s", safe_name, allocator = allocator)
		}

		col := w.column_new(safe_name, inferred[i], row_count)
		w.add_column(&df, col)
	}

	for r_i in 1 ..< len(rows) {
		row := rows[r_i]

		// iterate over all columns defined by the header
		for c_i in 0 ..< len(header) {
			val := ""
			if c_i < len(row) {
				val = row[c_i]
			}
			append_value_excel(&df.columns[c_i], val)
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
	styles_is_date: []bool,
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
		row := parse_row(row_xml, shared, styles_is_date, allocator)

		append(&rows_dyn, row)

		start = j
	}

	return rows_dyn[:]
}

parse_row :: proc(
	row_xml: string,
	shared: []string,
	styles_is_date: []bool,
	allocator: mem.Allocator,
) -> []string {
	row := make([dynamic]string, 0, allocator)

	start := 0
	for {
		i := strings.index(row_xml[start:], "<c")
		if i < 0 {
			break
		}
		i += start

		// find end of the <c ...> tag
		tag_end := strings.index(row_xml[i:], ">")
		if tag_end < 0 {
			break
		}
		tag_end += i + 1

		// check if self-closing: .../>
		self_closing := tag_end >= 2 && row_xml[tag_end - 2] == '/'

		cell_end := tag_end
		if !self_closing {
			// normal cell: find </c>
			j := strings.index(row_xml[tag_end:], "</c>")
			if j < 0 {
				break
			}
			cell_end = tag_end + j + len("</c>")
		}

		cell_xml := row_xml[i:cell_end]

		col_idx := cell_ref_to_col(cell_xml)

		val := ""
		if !self_closing {
			raw, is_dt := cell_value_raw(cell_xml, shared, styles_is_date)
			if is_dt {
				val = cell_value_datetime(raw, allocator)
			} else {
				val = raw
			}
		}

		if col_idx >= len(row) {
			needed := col_idx + 1 - len(row)
			for k in 0 ..< needed {
				append(&row, "")
			}
		}
		row[col_idx] = val

		start = cell_end
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
parse_attrs :: proc(tag: string, allocator: mem.Allocator) -> map[string]string {
	attrs := make(map[string]string, allocator)
	i := 0
	// skip until first space or end
	for i < len(tag) &&
	    tag[i] != ' ' &&
	    tag[i] != '\t' &&
	    tag[i] != '\r' &&
	    tag[i] != '\n' &&
	    tag[i] != '>' &&
	    tag[i] != '/' {
		i += 1
	}
	for i < len(tag) {
		// skip whitespace
		for i < len(tag) && (tag[i] == ' ' || tag[i] == '\t' || tag[i] == '\r' || tag[i] == '\n') {
			i += 1
		}
		if i >= len(tag) || tag[i] == '>' {
			break
		}
		// parse key
		key_start := i
		for i < len(tag) &&
		    tag[i] != '=' &&
		    tag[i] != ' ' &&
		    tag[i] != '\t' &&
		    tag[i] != '\r' &&
		    tag[i] != '\n' &&
		    tag[i] != '>' {
			i += 1
		}
		key_end := i
		// skip whitespace
		for i < len(tag) && (tag[i] == ' ' || tag[i] == '\t' || tag[i] == '\r' || tag[i] == '\n') {
			i += 1
		}
		if i >= len(tag) || tag[i] != '=' {
			break
		}
		i += 1 // skip '='
		// skip whitespace
		for i < len(tag) && (tag[i] == ' ' || tag[i] == '\t' || tag[i] == '\r' || tag[i] == '\n') {
			i += 1
		}
		if i >= len(tag) || tag[i] != '"' {
			break
		}
		i += 1 // skip opening quote
		val_start := i
		for i < len(tag) && tag[i] != '"' {
			i += 1
		}
		val_end := i
		if i < len(tag) && tag[i] == '"' {
			i += 1
		}

		key := tag[key_start:key_end]
		val := tag[val_start:val_end]
		attrs[key] = val
	}
	return attrs
}
parse_workbook_sheets :: proc(xml: string, allocator: mem.Allocator) -> []SheetInfo {
	sheets := make([dynamic]SheetInfo, 0, allocator)

	start := 0
	for {
		tag, next := extract_xml_tag(xml, start, "sheet")
		if next < 0 {
			break
		}
		attrs := parse_attrs(tag, allocator)

		si := SheetInfo{}
		if name, ok := attrs["name"]; ok {
			si.name = name
		}
		if rid, ok := attrs["r:id"]; ok {
			si.rid = rid
		}

		// only keep real sheets
		if si.name != "" && si.rid != "" {
			append(&sheets, si)
		}

		start = next
	}

	return sheets[:]
}


parse_rels :: proc(xml: string, allocator: mem.Allocator) -> []RelInfo {
	rels := make([dynamic]RelInfo, 0, allocator)

	start := 0
	for {
		tag, next := extract_xml_tag(xml, start, "Relationship")
		if next < 0 {
			break
		}
		attrs := parse_attrs(tag, allocator)

		r := RelInfo{}
		if id, ok := attrs["Id"]; ok {
			r.id = id
		}
		if target, ok := attrs["Target"]; ok {
			r.target = target
		}

		if r.id != "" && r.target != "" {
			append(&rels, r)
		}

		start = next
	}

	return rels[:]
}


discover_sheets :: proc(path: string, allocator: mem.Allocator) -> []SheetInfo {
	sheets := make([dynamic]SheetInfo, 0, allocator)
	wb_bytes, ok_wb := zm.zip_read_file(path, "xl/workbook.xml", allocator)
	if !ok_wb {
		return sheets[:]
	}
	//defer delete(wb_bytes)
	rel_bytes, ok_rels := zm.zip_read_file(path, "xl/_rels/workbook.xml.rels", allocator)
	if !ok_rels {
		rel_bytes, ok_rels = zm.zip_read_file(path, "xl/rels/workbook.xml.rels", allocator)
	}
	if !ok_rels {
		fmt.println("Warning: workbook.xml.rels not found in either location")
		return sheets[:]
	}

	//defer delete(rel_bytes)
	wb_xml := string(wb_bytes)
	rel_xml := string(rel_bytes)

	wb_sheets := parse_workbook_sheets(wb_xml, allocator)
	rels := parse_rels(rel_xml, allocator)

	for &s in wb_sheets {
		for r in rels {
			if s.rid == r.id {
				// only take worksheet targets
				if strings.index(r.target, "worksheets/") >= 0 {
					pth := fmt.aprintf("xl/%s", r.target, allocator = allocator)
					s.path = pth
					append(&sheets, s)
				}
				break
			}
		}
	}

	return sheets[:]
}
parse_styles_date_formats :: proc(xml: string, allocator: mem.Allocator) -> []bool {
	is_date := make([dynamic]bool, 0, allocator)

	// 1) custom formats
	custom := make(map[int]string, allocator)
	start := 0
	for {
		tag, next := extract_xml_tag(xml, start, "numFmt")
		if next < 0 {
			break
		}
		attrs := parse_attrs(tag, allocator)
		start = next

		if id_str, ok := attrs["numFmtId"]; ok {
			if code, ok2 := attrs["formatCode"]; ok2 {
				id, ok3 := strconv.parse_int(id_str)
				if ok3 {
					custom[id] = code
				}
			}
		}
	}

	// 2) xf entries
	start = 0
	for {
		tag, next := extract_xml_tag(xml, start, "xf")
		if next < 0 {
			break
		}
		attrs := parse_attrs(tag, allocator)
		start = next

		is_date_format := false

		if numFmtId_str, ok := attrs["numFmtId"]; ok {
			numFmtId, ok2 := strconv.parse_int(numFmtId_str)
			if ok2 {
				if numFmtId == 14 ||
				   numFmtId == 15 ||
				   numFmtId == 16 ||
				   numFmtId == 17 ||
				   numFmtId == 22 ||
				   numFmtId == 18 ||
				   numFmtId == 19 ||
				   numFmtId == 20 ||
				   numFmtId == 21 ||
				   numFmtId == 45 ||
				   numFmtId == 46 ||
				   numFmtId == 47 {
					is_date_format = true
				}

				if code, ok3 := custom[numFmtId]; ok3 {
					if looks_like_date_format(code) {
						is_date_format = true
					}
				}
			}
		}

		append(&is_date, is_date_format)
	}

	return is_date[:]
}

looks_like_date_format :: proc(code: string) -> bool {
	lower := strings.to_lower(code)
	defer delete(lower)
	// detect date formats (must contain day + month)
	if strings.contains(lower, "d") && strings.contains(lower, "m") {
		return true
	}

	// detect time formats (must contain hour or second)
	if strings.contains(lower, "h") || strings.contains(lower, "s") {
		return true
	}

	return false
}


excel_serial_to_datetime :: proc(n: f64) -> w.Datetime {
	// Excel epoch: 1899-12-31
	base := w.Date {
		year  = 1899,
		month = 12,
		day   = 31,
	}

	days := i32(n)
	frac := n - f64(days)

	// Excel's fake leap day: serial 60 = 1900-02-29
	if days >= 60 {
		days -= 1
	}

	// add days
	date := w.add_day_date(base, days)

	// time of day
	total_seconds := int(frac * 86400.0)
	hour := i32(total_seconds / 3600)
	minute := i32((total_seconds % 3600) / 60)
	second := i32(total_seconds % 60)

	time := w.Time{hour, minute, second}

	return w.new_Datetime_from_Date_and_Time(date, time)
}

cell_value_raw :: proc(
	cell_xml: string,
	shared: []string,
	styles_is_date: []bool,
) -> (
	string,
	bool,
) {
	// detect type attribute
	t_pos := strings.index(cell_xml, " t=\"")
	cell_type := ""
	if t_pos >= 0 {
		t_pos += len(" t=\"")
		end := t_pos
		for end < len(cell_xml) && cell_xml[end] != '"' {
			end += 1
		}
		cell_type = cell_xml[t_pos:end]
	}

	// --- inline string ---
	if cell_type == "inlineStr" {
		t_start := strings.index(cell_xml, "<t")
		if t_start >= 0 {
			gt := strings.index(cell_xml[t_start:], ">")
			if gt >= 0 {
				gt += t_start + 1
				t_end := strings.index(cell_xml[gt:], "</t>")
				if t_end >= 0 {
					t_end += gt
					return cell_xml[gt:t_end], false
				}
			}
		}
		return "", false
	}

	// --- shared string ---
	if cell_type == "s" {
		v_start := strings.index(cell_xml, "<v>")
		if v_start < 0 {return "", false}
		v_start += len("<v>")
		v_end := strings.index(cell_xml[v_start:], "</v>")
		if v_end < 0 {return "", false}
		v_end += v_start

		raw := cell_xml[v_start:v_end]
		idx, ok := strconv.parse_int(raw)
		if ok && idx >= 0 && idx < len(shared) {
			return shared[idx], false
		}
		return "", false
	}

	// --- normal numeric / boolean / text ---
	v_start := strings.index(cell_xml, "<v>")
	if v_start < 0 {return "", false}
	v_start += len("<v>")
	v_end := strings.index(cell_xml[v_start:], "</v>")
	if v_end < 0 {return "", false}
	v_end += v_start

	raw := cell_xml[v_start:v_end]

	// --- DATE DETECTION ---
	s_pos := strings.index(cell_xml, " s=\"")
	if s_pos >= 0 {
		s_pos += len(" s=\"")
		end := s_pos
		for end < len(cell_xml) && cell_xml[end] != '"' {
			end += 1
		}
		s_idx_str := cell_xml[s_pos:end]
		s_idx, ok := strconv.parse_int(s_idx_str)
		if ok && s_idx >= 0 && s_idx < len(styles_is_date) {
			if styles_is_date[s_idx] {
				// Only treat as datetime if the raw value parses as float
				if _, ok2 := strconv.parse_f64(raw); ok2 {
					return raw, true
				}
			}
		}
	}

	// fallback: raw view
	return raw, false
}

cell_value_datetime :: proc(raw: string, allocator: mem.Allocator) -> string {
	f, ok := strconv.parse_f64(raw)
	if !ok {
		// keep original numeric if it wasn't a valid float after all
		return raw
	}

	dt := excel_serial_to_datetime(f)

	// allocate backing storage with the *same* allocator as the DataFrame
	buf := make([]u8, 19, allocator) // "YYYY-MM-DD HH:MM:SS" = 19 bytes
	s := w.datetime_format_iso_into(dt, buf[:])
	return s
} // Helper: find "<tag_name" or "<x:tag_name" where the tag name is *exactly* tag_name
// i.e. the next character after the name must be a terminator: space, '>', '/', or whitespace.
find_exact_tag_start :: proc(xml: string, from: int, pattern: string) -> int {
	from := from
	i := strings.index(xml[from:], pattern)
	for i >= 0 {
		i += from
		next_idx := i + len(pattern)
		if next_idx >= len(xml) {
			return -1
		}

		ch := xml[next_idx]
		if ch == ' ' || ch == '>' || ch == '/' || ch == '\t' || ch == '\r' || ch == '\n' {
			return i
		}

		// false positive (e.g. "<sheets"), continue searching after this position
		from = next_idx
		i = strings.index(xml[from:], pattern)
	}

	return -1
}

extract_xml_tag :: proc(xml: string, start: int, tag_name: string) -> (string, int) {
	bigger := fmt.aprintf("<%s", tag_name)
	bigger_x := fmt.aprintf("<x:%s", tag_name)
	defer delete(bigger)
	defer delete(bigger_x)

	i := find_exact_tag_start(xml, start, bigger)
	if i < 0 {
		i = find_exact_tag_start(xml, start, bigger_x)
		if i < 0 {
			return "", -1
		}
	}

	// Find the closing '>'
	j := strings.index(xml[i:], ">")
	if j < 0 {
		return "", -1
	}
	j += i + 1

	return xml[i:j], j
}
