package importer

import w "../core"
import zm "../zip_min"

import "core:fmt"
import "core:math"
import "core:mem"
import "core:strconv"
import "core:strings"

// .xlsx → DataFrame. The first row is the header. Each cell keeps the kind
// Excel stored: numbers become Int / Float, booleans Bool, numbers with a date
// / time number format Date, Datetime or Time. A column with a text cell is a
// String column, except that "numbers stored as text" in canonical form
// ("88.5", "-3") count as numbers; codes like "007" keep the column text. Error cells and missing
// cells are null. Sheet, shared-string and style parts are located through the
// workbook relationships (absolute or relative targets).

SheetInfo :: struct {
	name: string,
	rid:  string,
	path: string, // resolved zip path, e.g. "xl/worksheets/sheet1.xml"
}

RelInfo :: struct {
	id:     string,
	target: string,
	type:   string,
}

Xlsx_Cell_Kind :: enum {
	Empty,
	Text,
	Number,
	Bool,
	Date,
	Datetime,
	Time,
	Error,
}

Xlsx_Cell :: struct {
	kind: Xlsx_Cell_Kind,
	text: string,
}

Xlsx_Date_Kind :: enum {
	None,
	Date,
	Datetime,
	Time,
}


// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

// First sheet of the workbook
xlsx_load :: proc(path: string, allocator: mem.Allocator) -> w.DataFrame {
	sheets := discover_sheets(path, context.temp_allocator)
	if len(sheets) == 0 {
		return w.dataframe_new(allocator)
	}
	return _xlsx_load_sheet(path, sheets[0], allocator)
}

// Sheet by name; ok = false if the workbook has no such sheet
xlsx_load_sheet :: proc(
	path: string,
	sheet_name: string,
	allocator: mem.Allocator,
) -> (
	w.DataFrame,
	bool,
) {
	for s in discover_sheets(path, context.temp_allocator) {
		if s.name == sheet_name {
			return _xlsx_load_sheet(path, s, allocator), true
		}
	}
	return w.dataframe_new(allocator), false
}

// Names of the worksheets, in workbook order
xlsx_sheet_names :: proc(path: string, allocator: mem.Allocator) -> []string {
	sheets := discover_sheets(path, context.temp_allocator)
	names := make([]string, len(sheets), allocator)
	for s, i in sheets {names[i] = strings.clone(s.name, allocator)}
	return names
}

@(private = "file")
_xlsx_load_sheet :: proc(path: string, sheet: SheetInfo, allocator: mem.Allocator) -> w.DataFrame {
	df := w.dataframe_new(allocator)
	scratch := context.temp_allocator

	rels := _workbook_rels(path)
	shared_path := _rel_target_of_type(rels, "/sharedStrings", "xl/sharedStrings.xml")
	styles_path := _rel_target_of_type(rels, "/styles", "xl/styles.xml")

	shared := []string{}
	if shared_bytes, ok := zm.zip_read_file(path, shared_path, scratch); ok {
		shared = parse_shared_strings(string(shared_bytes), scratch)
	}
	date_kinds := []Xlsx_Date_Kind{}
	if styles_bytes, ok := zm.zip_read_file(path, styles_path, scratch); ok {
		date_kinds = parse_styles_date_kinds(string(styles_bytes), scratch)
	}
	sheet_bytes, ok_sheet := zm.zip_read_file(path, sheet.path, scratch)
	if !ok_sheet {
		return df
	}

	rows := parse_sheet_cells(string(sheet_bytes), shared, date_kinds, scratch)
	if len(rows) == 0 {
		return df
	}

	n_cols := 0
	for r in rows {n_cols = max(n_cols, len(r))}
	header := rows[0]
	row_count := len(rows) - 1
	cell :: proc(row: []Xlsx_Cell, c: int) -> Xlsx_Cell {return(
			row[c] if c < len(row) else Xlsx_Cell{} \
		)}

	for c in 0 ..< n_cols {
		h := cell(header, c)
		name :=
			h.text if h.kind != .Empty && h.kind != .Error && strings.trim_space(h.text) != "" else fmt.tprintf("col_%d", c)

		// column type from the stored cell kinds
		any_text := false
		samples := make([dynamic]string, 0, row_count, scratch)
		for r in 1 ..< len(rows) {
			v := cell(rows[r], c)
			#partial switch v.kind {
			case .Text:
				if _canonical_number(v.text) {
					append(&samples, v.text)
				} else {
					any_text = true
				}
			case .Number, .Bool, .Date, .Datetime, .Time:
				append(&samples, v.text)
			}
		}
		type := w.ColumnType.String if any_text else infer_column_type(samples[:])

		col := w.column_new(strings.clone(name, allocator), type, row_count, allocator)
		for r in 1 ..< len(rows) {
			v := cell(rows[r], c)
			if v.kind == .Empty || v.kind == .Error {
				w.append_null(&col)
				continue
			}
			if type == .String || (v.kind == .Text && !_canonical_number(v.text)) {
				w.append_string(&col, v.text)
				continue
			}
			s := v.text
			ok := true
			#partial switch type {
			case .Int:
				x: int
				x, ok = strconv.parse_int(s)
				if ok {w.append_int(&col, x)}
			case .Float:
				x: f64
				x, ok = strconv.parse_f64(s)
				if ok {w.append_float(&col, x)}
			case .Bool:
				w.append_bool(&col, s == "true")
			case .Date:
				d: w.Date
				d, ok = w.parse_date(s)
				if ok {w.append_date(&col, d)}
			case .Datetime:
				dt: w.Datetime
				dt, ok = w.parse_datetime(s)
				if ok {w.append_datetime(&col, dt)}
			case .Time:
				t: w.Time
				t, ok = w.parse_time(s)
				if ok {w.append_time(&col, t)}
			}
			if !ok {w.append_null(&col)} 	// unreachable: types are inferred from these values
		}
		w.add_column(&df, col)
	}

	df.rows = row_count
	return df
}


// Text that is a plain decimal number as a number cell would print it: no
// padding, no '+', no leading zeros ("007", "01234" are codes, not numbers)
@(private = "file")
_canonical_number :: proc(s: string) -> bool {
	if len(s) == 0 || s[0] == '+' || s != strings.trim_space(s) {return false}
	_, ok := strconv.parse_f64(s)
	if !ok {return false}
	t := s[1:] if s[0] == '-' else s
	if len(t) > 1 && t[0] == '0' && t[1] >= '0' && t[1] <= '9' {return false}
	for i in 0 ..< len(t) { 	// digits, one '.', exponent only (no inf / nan words)
		ch := t[i]
		if !((ch >= '0' && ch <= '9') ||
			   ch == '.' ||
			   ch == 'e' ||
			   ch == 'E' ||
			   ch == '-' ||
			   ch == '+') {return false}
	}
	return true
}

// ------------------------------------------------------------
// Workbook structure
// ------------------------------------------------------------

@(private = "file")
_workbook_rels :: proc(path: string) -> []RelInfo {
	rel_bytes, ok := zm.zip_read_file(path, "xl/_rels/workbook.xml.rels", context.temp_allocator)
	if !ok {
		rel_bytes, ok = zm.zip_read_file(path, "xl/rels/workbook.xml.rels", context.temp_allocator)
	}
	if !ok {return {}}
	return parse_rels(string(rel_bytes), context.temp_allocator)
}

// Zip path of a relationship target relative to xl/ ("/xl/..." is absolute)
@(private = "file")
_resolve_target :: proc(target: string, allocator: mem.Allocator) -> string {
	t := target
	if strings.has_prefix(t, "/") {return strings.clone(t[1:], allocator)}
	for strings.has_prefix(t, "./") {t = t[2:]}
	return fmt.aprintf("xl/%s", t, allocator = allocator)
}

@(private = "file")
_rel_target_of_type :: proc(rels: []RelInfo, type_suffix: string, fallback: string) -> string {
	for r in rels {
		if strings.has_suffix(r.type, type_suffix) {
			return _resolve_target(r.target, context.temp_allocator)
		}
	}
	return fallback
}

discover_sheets :: proc(path: string, allocator: mem.Allocator) -> []SheetInfo {
	sheets := make([dynamic]SheetInfo, 0, allocator)
	wb_bytes, ok_wb := zm.zip_read_file(path, "xl/workbook.xml", context.temp_allocator)
	if !ok_wb {
		return sheets[:]
	}
	rels := _workbook_rels(path)
	if len(rels) == 0 {
		return sheets[:]
	}

	for &s in parse_workbook_sheets(string(wb_bytes), context.temp_allocator) {
		for r in rels {
			if s.rid != r.id {continue}
			if strings.has_suffix(r.type, "/worksheet") ||
			   strings.contains(r.target, "worksheets/") {
				append(
					&sheets,
					SheetInfo {
						name = strings.clone(s.name, allocator),
						rid = strings.clone(s.rid, allocator),
						path = _resolve_target(r.target, allocator),
					},
				)
			}
			break
		}
	}
	return sheets[:]
}


// ------------------------------------------------------------
// Shared strings / rich text
// ------------------------------------------------------------

// Concatenated text of all <t> runs in a fragment (rich text), phonetic
// <rPh> runs excluded, entities decoded
@(private = "file")
_rich_text :: proc(xml: string, allocator: mem.Allocator) -> string {
	b := strings.builder_make(0, 32, context.temp_allocator)
	pos := 0
	for {
		i := find_exact_tag_start(xml, pos, "<t")
		if i < 0 {break}
		if ph := strings.index(xml[pos:i], "<rPh"); ph >= 0 { 	// skip phonetic run
			close := strings.index(xml[pos + ph:], "</rPh>")
			if close < 0 {break}
			pos = pos + ph + close + len("</rPh>")
			continue
		}
		gt := strings.index_byte(xml[i:], '>')
		if gt < 0 {break}
		gt += i
		if xml[gt - 1] == '/' { 	// <t/>
			pos = gt + 1
			continue
		}
		end := strings.index(xml[gt + 1:], "</t>")
		if end < 0 {break}
		strings.write_string(&b, xml[gt + 1:gt + 1 + end])
		pos = gt + 1 + end + len("</t>")
	}
	return strings.clone(decode_entities(strings.to_string(b)), allocator)
}

parse_shared_strings :: proc(xml: string, allocator: mem.Allocator) -> []string {
	out := make([dynamic]string, 0, allocator)
	start := 0
	for {
		i := find_exact_tag_start(xml, start, "<si")
		if i < 0 {break}
		j := strings.index(xml[i:], "</si>")
		if j < 0 {break}
		j += i + len("</si>")
		append(&out, _rich_text(xml[i:j], allocator))
		start = j
	}
	return out[:]
}


// ------------------------------------------------------------
// Styles: which cell formats are dates / times
// ------------------------------------------------------------

// Date kind of every <cellXfs> entry (the index used by a cell's s="...")
parse_styles_date_kinds :: proc(xml: string, allocator: mem.Allocator) -> []Xlsx_Date_Kind {
	kinds := make([dynamic]Xlsx_Date_Kind, 0, allocator)

	custom := make(map[int]string, allocator = context.temp_allocator)
	start := 0
	for {
		tag, next := extract_xml_tag(xml, start, "numFmt")
		if next < 0 {break}
		start = next
		attrs := parse_attrs(tag, context.temp_allocator)
		if id, ok := strconv.parse_int(attrs["numFmtId"]); ok {
			custom[id] = decode_entities(attrs["formatCode"])
		}
	}

	// only <cellXfs> (cellStyleXfs, which usually comes first, has other indices)
	sec_start := find_exact_tag_start(xml, 0, "<cellXfs")
	if sec_start < 0 {return kinds[:]}
	sec_end := strings.index(xml[sec_start:], "</cellXfs>")
	section := xml[sec_start:] if sec_end < 0 else xml[sec_start:sec_start + sec_end]

	start = len("<cellXfs")
	for {
		tag, next := extract_xml_tag(section, start, "xf")
		if next < 0 {break}
		start = next
		attrs := parse_attrs(tag, context.temp_allocator)
		kind := Xlsx_Date_Kind.None
		if id, ok := strconv.parse_int(attrs["numFmtId"]); ok {
			if code, has := custom[id]; has {
				kind = classify_number_format(code)
			} else {
				kind = _builtin_date_kind(id)
			}
		}
		append(&kinds, kind)
	}
	return kinds[:]
}

@(private = "file")
_builtin_date_kind :: proc(id: int) -> Xlsx_Date_Kind {
	switch id {
	case 14 ..= 17, 27 ..= 36, 50 ..= 58:
		return .Date
	case 22:
		return .Datetime
	case 18 ..= 21, 45 ..= 47:
		return .Time
	}
	return .None
}

// Date / time kind of a custom number format code. Quoted literals, escaped
// characters and [...] sections (colours, locales) are ignored, so a format
// like `#,##0 "USD"` is not mistaken for a time.
classify_number_format :: proc(code: string) -> Xlsx_Date_Kind {
	has_date, has_time, has_m := false, false, false
	i := 0
	for i < len(code) {
		ch := code[i]
		switch ch {
		case '"':
			j := strings.index_byte(code[i + 1:], '"')
			i = len(code) if j < 0 else i + j + 2
			continue
		case '\\', '_', '*':
			i += 2
			continue
		case '[':
			j := strings.index_byte(code[i:], ']')
			inner := strings.to_lower(code[i + 1:i + j] if j > 0 else "", context.temp_allocator)
			if inner == "h" ||
			   inner == "hh" ||
			   inner == "m" ||
			   inner == "mm" ||
			   inner == "s" ||
			   inner == "ss" {
				has_time = true // elapsed time
			}
			i = len(code) if j < 0 else i + j + 1
			continue
		case 'y', 'Y', 'd', 'D':
			has_date = true
		case 'h', 'H', 's', 'S':
			has_time = true
		case 'm', 'M':
			has_m = true
		}
		i += 1
	}
	switch {
	case has_date && has_time:
		return .Datetime
	case has_date || (has_m && !has_time):
		return .Date
	case has_time:
		return .Time
	}
	return .None
}

// Kept for compatibility: true for date or time formats
looks_like_date_format :: proc(code: string) -> bool {
	return classify_number_format(code) != .None
}


// ------------------------------------------------------------
// Sheet parsing
// ------------------------------------------------------------

parse_sheet_cells :: proc(
	xml: string,
	shared: []string,
	date_kinds: []Xlsx_Date_Kind,
	allocator: mem.Allocator,
) -> [][]Xlsx_Cell {
	rows := make([dynamic][]Xlsx_Cell, 0, allocator)
	start := 0
	for {
		i := find_exact_tag_start(xml, start, "<row")
		if i < 0 {break}
		gt := strings.index_byte(xml[i:], '>')
		if gt < 0 {break}
		gt += i
		if xml[gt - 1] == '/' { 	// <row .../> : empty row
			start = gt + 1
			continue
		}
		j := strings.index(xml[gt:], "</row>")
		if j < 0 {break}
		j += gt
		append(&rows, parse_row_cells(xml[gt + 1:j], shared, date_kinds, allocator))
		start = j + len("</row>")
	}
	return rows[:]
}

// Value of attribute `name` in a start tag (double or single quotes)
@(private = "file")
_attr :: proc(tag: string, name: string) -> string {
	pos := 0
	for {
		k := strings.index(tag[pos:], name)
		if k < 0 {return ""}
		k += pos
		after := k + len(name)
		before_ok :=
			k > 0 &&
			(tag[k - 1] == ' ' || tag[k - 1] == '\t' || tag[k - 1] == '\n' || tag[k - 1] == '\r')
		if before_ok &&
		   after + 1 < len(tag) &&
		   tag[after] == '=' &&
		   (tag[after + 1] == '"' || tag[after + 1] == '\'') {
			q := tag[after + 1]
			end := strings.index_byte(tag[after + 2:], q)
			if end < 0 {return ""}
			return tag[after + 2:after + 2 + end]
		}
		pos = after
	}
}

parse_row_cells :: proc(
	row_xml: string,
	shared: []string,
	date_kinds: []Xlsx_Date_Kind,
	allocator: mem.Allocator,
) -> []Xlsx_Cell {
	row := make([dynamic]Xlsx_Cell, 0, allocator)
	next_col := 0
	start := 0
	for {
		i := find_exact_tag_start(row_xml, start, "<c")
		if i < 0 {break}
		gt := strings.index_byte(row_xml[i:], '>')
		if gt < 0 {break}
		gt += i
		tag := row_xml[i:gt + 1]
		self_closing := row_xml[gt - 1] == '/'
		content := ""
		cell_end := gt + 1
		if !self_closing {
			j := strings.index(row_xml[gt + 1:], "</c>")
			if j < 0 {break}
			content = row_xml[gt + 1:gt + 1 + j]
			cell_end = gt + 1 + j + len("</c>")
		}
		start = cell_end

		col := next_col
		if ref := _attr(tag, "r"); ref != "" {col = cell_ref_to_col(fmt.tprintf(` r="%s"`, ref))}
		next_col = col + 1

		c := Xlsx_Cell{}
		t := _attr(tag, "t")
		v := ""
		if vs := strings.index(content, "<v>"); vs >= 0 {
			ve := strings.index(content[vs + 3:], "</v>")
			if ve >= 0 {v = content[vs + 3:vs + 3 + ve]}
		}
		switch t {
		case "s":
			if idx, ok := strconv.parse_int(v); ok && idx >= 0 && idx < len(shared) {
				c = {.Text, shared[idx]}
			}
		case "inlineStr":
			c = {.Text, _rich_text(content, allocator)}
		case "str":
			c = {.Text, strings.clone(decode_entities(v), allocator)}
		case "b":
			if v != "" {c = {.Bool, "true" if strings.trim_space(v) == "1" else "false"}}
		case "e":
			c = {.Error, v}
		case:
			if v == "" {break}
			kind := Xlsx_Date_Kind.None
			if sidx, ok := strconv.parse_int(_attr(tag, "s"));
			   ok && sidx >= 0 && sidx < len(date_kinds) {
				kind = date_kinds[sidx]
			}
			serial, is_num := strconv.parse_f64(v)
			if kind == .None || !is_num {
				c = {.Number, v}
				break
			}
			dt := excel_serial_to_datetime(serial)
			switch kind {
			case .Date:
				c = {
					.Date,
					fmt.aprintf(
						"%04d-%02d-%02d",
						dt.year,
						dt.month,
						dt.day,
						allocator = allocator,
					),
				}
			case .Time:
				c = {
					.Time,
					fmt.aprintf(
						"%02d:%02d:%02d",
						dt.hour,
						dt.minute,
						dt.second,
						allocator = allocator,
					),
				}
			case .Datetime, .None:
				c = {
					.Datetime,
					fmt.aprintf(
						"%04d-%02d-%02d %02d:%02d:%02d",
						dt.year,
						dt.month,
						dt.day,
						dt.hour,
						dt.minute,
						dt.second,
						allocator = allocator,
					),
				}
			}
		}

		if col >= len(row) {
			for _ in len(row) ..= col {append(&row, Xlsx_Cell{})}
		}
		row[col] = c
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
// Excel dates (1900 date system)
// ------------------------------------------------------------

// Serial → datetime; serial 60 is Excel's fictitious 1900-02-29. The time of
// day is rounded to the nearest second (serials are binary fractions of a day).
excel_serial_to_datetime :: proc(n: f64) -> w.Datetime {
	base := w.Date {
		year  = 1899,
		month = 12,
		day   = 31,
	}

	total := i64(math.round(n * 86400.0)) // seconds since the epoch, rounded
	days := i32(total / 86400)
	secs := i32(total % 86400)

	if days >= 60 {
		days -= 1
	}
	date := w.add_day_date(base, days)
	time := w.Time{secs / 3600, (secs % 3600) / 60, secs % 60}
	return w.new_Datetime_from_Date_and_Time(date, time)
}


// ------------------------------------------------------------
// Small XML helpers
// ------------------------------------------------------------

parse_attrs :: proc(tag: string, allocator: mem.Allocator) -> map[string]string {
	attrs := make(map[string]string, allocator)
	i := 0
	is_ws :: proc(c: u8) -> bool {return c == ' ' || c == '\t' || c == '\r' || c == '\n'}
	// skip the tag name
	for i < len(tag) && !is_ws(tag[i]) && tag[i] != '>' && tag[i] != '/' {
		i += 1
	}
	for i < len(tag) {
		for i < len(tag) && is_ws(tag[i]) {i += 1}
		if i >= len(tag) || tag[i] == '>' || tag[i] == '/' {break}
		key_start := i
		for i < len(tag) && tag[i] != '=' && !is_ws(tag[i]) && tag[i] != '>' {i += 1}
		key_end := i
		for i < len(tag) && is_ws(tag[i]) {i += 1}
		if i >= len(tag) || tag[i] != '=' {break}
		i += 1
		for i < len(tag) && is_ws(tag[i]) {i += 1}
		if i >= len(tag) || (tag[i] != '"' && tag[i] != '\'') {break}
		q := tag[i]
		i += 1
		val_start := i
		for i < len(tag) && tag[i] != q {i += 1}
		val_end := i
		if i < len(tag) {i += 1}
		attrs[tag[key_start:key_end]] = tag[val_start:val_end]
	}
	return attrs
}

parse_workbook_sheets :: proc(xml: string, allocator: mem.Allocator) -> []SheetInfo {
	sheets := make([dynamic]SheetInfo, 0, allocator)
	start := 0
	for {
		tag, next := extract_xml_tag(xml, start, "sheet")
		if next < 0 {break}
		start = next
		attrs := parse_attrs(tag, context.temp_allocator)
		si := SheetInfo {
			name = decode_entities(attrs["name"], allocator),
			rid  = attrs["r:id"],
		}
		if si.name != "" && si.rid != "" {
			append(&sheets, si)
		}
	}
	return sheets[:]
}

parse_rels :: proc(xml: string, allocator: mem.Allocator) -> []RelInfo {
	rels := make([dynamic]RelInfo, 0, allocator)
	start := 0
	for {
		tag, next := extract_xml_tag(xml, start, "Relationship")
		if next < 0 {break}
		start = next
		attrs := parse_attrs(tag, context.temp_allocator)
		r := RelInfo {
			id     = attrs["Id"],
			target = attrs["Target"],
			type   = attrs["Type"],
		}
		if r.id != "" && r.target != "" {
			append(&rels, r)
		}
	}
	return rels[:]
}

// Helper: find "<tag_name" where the tag name is *exactly* tag_name, i.e. the
// next character is a terminator: space, '>', '/', or whitespace.
find_exact_tag_start :: proc(xml: string, from: int, pattern: string) -> int {
	from := from
	for from < len(xml) {
		i := strings.index(xml[from:], pattern)
		if i < 0 {return -1}
		i += from
		next_idx := i + len(pattern)
		if next_idx >= len(xml) {return -1}
		ch := xml[next_idx]
		if ch == ' ' || ch == '>' || ch == '/' || ch == '\t' || ch == '\r' || ch == '\n' {
			return i
		}
		from = next_idx // false positive (e.g. "<sheets"), continue after it
	}
	return -1
}

extract_xml_tag :: proc(xml: string, start: int, tag_name: string) -> (string, int) {
	i := find_exact_tag_start(xml, start, fmt.tprintf("<%s", tag_name))
	if i < 0 {
		i = find_exact_tag_start(xml, start, fmt.tprintf("<x:%s", tag_name))
		if i < 0 {
			return "", -1
		}
	}
	j := strings.index(xml[i:], ">")
	if j < 0 {
		return "", -1
	}
	j += i + 1
	return xml[i:j], j
}
