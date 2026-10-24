package importer


import w "../core"
import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"

// HTML <table> → DataFrame. Tags are matched case-insensitively, <th> and <td>
// cells are taken in document order (row headers included), colspan repeats
// the cell, closing tags may be omitted, text is UTF-8 safe with HTML entities
// decoded. The first row is the header; short rows are padded with nulls and
// extra cells are ignored. Cell text is trimmed; empty cells are null.

html_load :: proc(path: string, allocator: mem.Allocator) -> w.DataFrame {
	contents, err := read_file(path)
	if err != nil {
		panic("html_load: cannot read file")
	}
	defer delete(contents)
	return html_load_from_string(contents, allocator)
}

// ASCII lower-case copy (byte offsets unchanged) for case-insensitive search
@(private = "file")
_lower_ascii :: proc(s: string, allocator := context.temp_allocator) -> string {
	b := make([]u8, len(s), allocator)
	for i in 0 ..< len(s) {
		ch := s[i]
		b[i] = ch + 32 if ch >= 'A' && ch <= 'Z' else ch
	}
	return string(b)
}

// Start of the next "<name" tag at or after `from` (whole tag name only, so
// "<th" does not match "<thead"); `lower` is the lower-cased text
@(private = "file")
_find_tag :: proc(lower: string, from: int, name: string) -> int {
	pos := from
	for pos < len(lower) {
		i := strings.index(lower[pos:], name)
		if i < 0 {return -1}
		i += pos
		after := i + len(name)
		if after >= len(lower) {return -1}
		ch := lower[after]
		if ch == '>' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '/' {return i}
		pos = after
	}
	return -1
}

// Blocks "<tag ...>...</tag>" of the text, in order (outer tag included)
@(private = "file")
_extract_blocks :: proc(text: string, tag: string) -> []string {
	out := make([dynamic]string, 0, context.temp_allocator)
	lower := _lower_ascii(text)
	open := fmt.tprintf("<%s", tag)
	close := fmt.tprintf("</%s>", tag)
	start := 0
	for {
		i := _find_tag(lower, start, open)
		if i < 0 {break}
		j := strings.index(lower[i:], close)
		if j < 0 {break}
		j += i + len(close)
		append(&out, text[i:j])
		start = j
	}
	return out[:]
}

extract_rows :: proc(table: string) -> []string {
	return _extract_blocks(table, "tr")
}

extract_tables :: proc(html: string) -> []string {
	return _extract_blocks(html, "table")
}

extract_header :: proc(row: string) -> []string {
	headers := make([dynamic]string, 0, context.temp_allocator)
	for c in extract_cells(row) {
		append(&headers, strip_html(c))
	}
	return headers[:]
}

// Inner HTML of the <th> / <td> cells of a row, in document order
extract_cells :: proc(row: string) -> []string {
	cells, _ := _extract_cells_spans(row)
	return cells
}

// Cells plus their colspan
@(private = "file")
_extract_cells_spans :: proc(row: string) -> (cells: []string, spans: []int) {
	out := make([dynamic]string, 0, context.temp_allocator)
	out_spans := make([dynamic]int, 0, context.temp_allocator)
	lower := _lower_ascii(row)

	next_cell :: proc(lower: string, from: int) -> int {
		td := _find_tag(lower, from, "<td")
		th := _find_tag(lower, from, "<th")
		if td < 0 {return th}
		if th < 0 {return td}
		return min(td, th)
	}

	pos := next_cell(lower, 0)
	for pos >= 0 {
		gt := strings.index_byte(row[pos:], '>')
		if gt < 0 {break}
		open_tag := lower[pos:pos + gt]
		content_start := pos + gt + 1

		// content ends at the closing tag, the next cell or the end of the row
		is_th := strings.has_prefix(open_tag, "<th")
		close := "</th" if is_th else "</td"
		end := len(row)
		if j := strings.index(lower[content_start:], close); j >= 0 {end = content_start + j}
		if j := strings.index(lower[content_start:], "</tr");
		   j >= 0 {end = min(end, content_start + j)}
		nxt := next_cell(lower, content_start)
		if nxt >= 0 {end = min(end, nxt)}

		span := 1
		if k := strings.index(open_tag, "colspan"); k >= 0 {
			rest := open_tag[k + len("colspan"):]
			digits := strings.trim_left(rest, " \t=\"'")
			n := 0
			for n < len(digits) && digits[n] >= '0' && digits[n] <= '9' {n += 1}
			if v, ok := strconv.parse_int(digits[:n]); ok && v > 1 && v < 1000 {span = v}
		}

		append(&out, row[content_start:end])
		append(&out_spans, span)
		pos = nxt if nxt >= 0 && nxt >= end else next_cell(lower, end)
	}
	return out[:], out_spans[:]
}

// Text content of an HTML fragment: tags removed (UTF-8 preserved), entities
// decoded, surrounding whitespace trimmed
strip_html :: proc(s: string, allocator: mem.Allocator = context.temp_allocator) -> string {
	out := strings.builder_make(0, len(s), context.temp_allocator)
	inside := false
	for i in 0 ..< len(s) {
		b := s[i]
		if b == '<' {
			inside = true
			continue
		}
		if b == '>' && inside {
			inside = false
			continue
		}
		if !inside {
			strings.write_byte(&out, b)
		}
	}
	decoded := decode_entities(strings.to_string(out), context.temp_allocator)
	return strings.clone(strings.trim_space(decoded), allocator)
}

infer_and_append :: proc(col: ^w.Column, s: string) {
	trimmed := strings.trim_space(s)

	if trimmed == "" {
		w.append_null(col)
		return
	}
	if v, ok := strconv.parse_int(trimmed); ok {
		w.append_int(col, v)
		return
	}
	if v, ok := strconv.parse_f64(trimmed); ok {
		w.append_float(col, v)
		return
	}
	if strings.equal_fold(trimmed, "true") {
		w.append_bool(col, true)
		return
	}
	if strings.equal_fold(trimmed, "false") {
		w.append_bool(col, false)
		return
	}
	w.append_string(col, trimmed)
}

// Column types from the data rows (row 0 is the header); ragged rows allowed
infer_column_types :: proc(tmp: [][]string) -> []w.ColumnType {
	col_count := len(tmp[0])
	types := make([]w.ColumnType, col_count, context.temp_allocator)

	for c in 0 ..< col_count {
		all_int := true
		all_float := true
		all_bool := true
		all_date := true
		all_time := true
		all_datetime := true
		any_value := false

		for r in 1 ..< len(tmp) {
			if c >= len(tmp[r]) {continue}
			v := strings.trim_space(tmp[r][c])
			if v == "" {continue}
			any_value = true

			if _, ok := strconv.parse_int(v); !ok {all_int = false}
			if _, ok := strconv.parse_f64(v); !ok {all_float = false}
			if !(strings.equal_fold(v, "true") ||
				   strings.equal_fold(v, "false")) {all_bool = false}
			if !is_date(v) {all_date = false}
			if !is_time(v) {all_time = false}
			if !is_datetime(v) {all_datetime = false}
		}

		// precedence: datetime > date > time > int > float > bool > string
		switch {
		case !any_value:
			types[c] = .String
		case all_datetime:
			types[c] = .Datetime
		case all_date:
			types[c] = .Date
		case all_time:
			types[c] = .Time
		case all_int:
			types[c] = .Int
		case all_float:
			types[c] = .Float
		case all_bool:
			types[c] = .Bool
		case:
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
		w.append_bool(col, strings.equal_fold(trimmed, "true"))
	case .String:
		w.append_string(col, trimmed)
	case .Date:
		date, _ := w.parse_date(trimmed)
		w.append_date(col, date)
	case .Time:
		time, _ := w.parse_time(trimmed)
		w.append_time(col, time)
	case .Datetime:
		datetime, _ := w.parse_datetime(trimmed)
		w.append_datetime(col, datetime)
	}
}

html_load_table_index :: proc(path: string, index: int, allocator: mem.Allocator) -> w.DataFrame {
	contents, err := read_file(path)
	if err != nil {panic("html_load: cannot read file")}
	defer delete(contents)

	tables := extract_tables(contents)
	if index < 0 || index >= len(tables) {
		return w.dataframe_new(allocator)
	}
	return html_parse_table(tables[index], allocator)
}

html_load_table_id :: proc(path: string, id: string, allocator: mem.Allocator) -> w.DataFrame {
	contents, err := read_file(path)
	if err != nil {panic("html_load: cannot read file")}
	defer delete(contents)

	tables := extract_tables(contents)
	needle := fmt.tprintf("id=\"%s\"", id)
	needle_sq := fmt.tprintf("id='%s'", id)
	for t in tables {
		open_tag := t[:max(strings.index_byte(t, '>'), 0)]
		if strings.contains(open_tag, needle) || strings.contains(open_tag, needle_sq) {
			return html_parse_table(t, allocator)
		}
	}
	return w.dataframe_new(allocator)
}

html_load_all :: proc(path: string, allocator: mem.Allocator) -> []w.DataFrame {
	contents, err := read_file(path)
	if err != nil {panic("html_load: cannot read file")}
	defer delete(contents)
	return html_load_all_from_string(contents, allocator)
}

html_parse_table :: proc(table: string, allocator: mem.Allocator) -> w.DataFrame {
	df := w.dataframe_new(allocator)

	// Rows in logical order: <thead> rows, then <tbody> rows (or all rows)
	rows_dyn := make([dynamic]string, 0, context.temp_allocator)
	thead := extract_section(table, "thead")
	tbody := extract_section(table, "tbody")
	if thead != "" {
		for r in extract_rows(thead) {append(&rows_dyn, r)}
	}
	if tbody != "" {
		for r in extract_rows(tbody) {append(&rows_dyn, r)}
	} else if thead == "" {
		for r in extract_rows(table) {append(&rows_dyn, r)}
	} else {
		// header in <thead>, body rows directly in <table>
		after := strings.index(table, thead) + len(thead)
		for r in extract_rows(table[after:]) {append(&rows_dyn, r)}
	}
	rows := rows_dyn[:]
	if len(rows) == 0 {
		return df
	}

	// PASS 1: all cells as text (colspan expanded)
	tmp := make([][]string, len(rows), context.temp_allocator)
	for r, r_i in rows {
		cells, spans := _extract_cells_spans(r)
		line := make([dynamic]string, 0, len(cells), context.temp_allocator)
		for cell, c_i in cells {
			clean := strip_html(cell)
			for _ in 0 ..< spans[c_i] {append(&line, clean)}
		}
		tmp[r_i] = line[:]
	}

	header := tmp[0]
	row_count := len(tmp) - 1

	// PASS 2: infer column types
	inferred := infer_column_types(tmp)

	// PASS 3: create columns
	for name, i in header {
		col := w.column_new(strings.clone(name, allocator), inferred[i], row_count, allocator)
		w.add_column(&df, col)
	}

	// PASS 4: append values (short rows padded with nulls, extra cells ignored)
	for r_i in 1 ..< len(tmp) {
		row := tmp[r_i]
		for c_i in 0 ..< len(header) {
			append_value(&df.columns[c_i], row[c_i] if c_i < len(row) else "")
		}
	}

	df.rows = row_count
	return df
}


extract_section :: proc(
	table: string,
	tag: string,
	allocator: mem.Allocator = context.temp_allocator,
) -> string {
	blocks := _extract_blocks(table, tag)
	return blocks[0] if len(blocks) > 0 else ""
}

html_load_from_string :: proc(
	html: string,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	tables := extract_tables(html)
	if len(tables) == 0 {
		return w.dataframe_new(allocator)
	}
	return html_parse_table(tables[0], allocator)
}

html_load_all_from_string :: proc(
	html: string,
	allocator: mem.Allocator = context.allocator,
) -> []w.DataFrame {
	tables := extract_tables(html)
	dfs := make([]w.DataFrame, len(tables), allocator)
	for t, i in tables {
		dfs[i] = html_parse_table(t, allocator)
	}
	return dfs[:]
}
