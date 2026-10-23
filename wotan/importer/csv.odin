package importer

import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"

import w "../core"
import infer "../importer" // or whatever path matches your layout

DEFAULT_NULL_TOKEN :: []string{"NA", "null", "None", ""}

// Null semantics (lossless with exporter.csv_save):
// - an unquoted field that is empty or a null token is null
// - in a String column a quoted field is always the literal string
//   ("" is the empty string, "NA" the text NA)
// - in typed columns quoted content is parsed like unquoted content


// Types inferred from every data row (fields trimmed; null tokens skipped)
infer_col_types :: proc(
	records: [][]string,
	null_tokens: []string = DEFAULT_NULL_TOKEN,
) -> []w.ColumnType {
	header := records[0]
	col_count := len(header)
	samples := make([][dynamic]string, col_count)
	defer {
		for i in 0 ..< col_count {delete(samples[i])}
		delete(samples)
	}

	for row_i in 1 ..< len(records) {
		fields := records[row_i]
		for col_i in 0 ..< min(col_count, len(fields)) {
			if is_null_field(fields[col_i], null_tokens) {
				continue
			}
			append(&samples[col_i], strings.trim(fields[col_i], " \t"))
		}
	}
	types := make([]w.ColumnType, col_count, context.temp_allocator)
	for i in 0 ..< col_count {
		types[i] = infer.infer_column_type(samples[i][:])
	}
	return types
}


csv_load :: proc(
	path: string,
	types: []w.ColumnType = nil,
	sep: u8 = ',',
	null_tokens: []string = DEFAULT_NULL_TOKEN,
) -> w.DataFrame {
	contents, err := read_file(path)
	if err != nil {
		panic("csv_load: failed to read file")
	}
	defer delete(contents)
	return _csv_build(contents, context.allocator, types, sep, null_tokens, "csv_load")
}

csv_load_from_string :: proc(
	text: string,
	allocator: mem.Allocator = context.allocator,
	types: []w.ColumnType = nil,
	sep: u8 = ',',
	null_tokens: []string = DEFAULT_NULL_TOKEN,
) -> w.DataFrame {
	return _csv_build(text, allocator, types, sep, null_tokens, "csv_load_from_string")
}

@(private = "file")
_csv_build :: proc(
	text: string,
	allocator: mem.Allocator,
	types: []w.ColumnType,
	sep: u8,
	null_tokens: []string,
	label: string,
) -> w.DataFrame {
	records, quoted := _parse_csv(text, sep)
	defer {
		csv_free(records)
		for q in quoted {delete(q)}
		delete(quoted)
	}
	if len(records) == 0 {
		panic(fmt.tprintf("%s: empty input", label))
	}

	header := records[0]
	types := types
	explicit := types != nil
	if !explicit {
		types = infer_col_types(records, null_tokens)
	}
	if len(header) != len(types) {
		panic(fmt.tprintf("%s: header/type count mismatch", label))
	}
	n_cols := len(types)

	df := w.dataframe_new(allocator)
	cols := make([]w.Column, n_cols)
	defer delete(cols)
	for i in 0 ..< n_cols {
		cols[i] = w.column_new(header[i], types[i], len(records), allocator)
	}

	long_rows := 0
	for row_i in 1 ..< len(records) {
		fields := records[row_i]
		if len(fields) > n_cols {long_rows += 1}

		for col_i in 0 ..< n_cols {
			col := &cols[col_i]
			if col_i >= len(fields) { 	// short row: missing trailing fields are null
				w.append_null(col)
				continue
			}
			raw := fields[col_i]
			was_quoted := quoted[row_i][col_i]

			if types[col_i] == .String {
				if was_quoted {
					w.append_string(col, raw)
				} else if is_null_field(raw, null_tokens) {
					w.append_null(col)
				} else {
					w.append_string(col, strings.trim(raw, " \t"))
				}
				continue
			}

			field := strings.trim(raw, " \t")
			if is_null_field(field, null_tokens) {
				w.append_null(col)
				continue
			}
			ok := true
			#partial switch types[col_i] {
			case .Int:
				v: int
				v, ok = strconv.parse_int(field)
				if ok {w.append_int(col, v)}
			case .Float:
				v: f64
				v, ok = strconv.parse_f64(field)
				if ok {w.append_float(col, v)}
			case .Bool:
				truthy := strings.equal_fold(field, "true") || field == "1"
				ok = truthy || strings.equal_fold(field, "false") || field == "0"
				if ok {w.append_bool(col, truthy)}
			case .Date:
				d: w.Date
				d, ok = w.parse_date(field)
				if ok {w.append_date(col, d)}
			case .Time:
				t: w.Time
				t, ok = w.parse_time(field)
				if ok {w.append_time(col, t)}
			case .Datetime:
				dt: w.Datetime
				dt, ok = w.parse_datetime(field)
				if ok {w.append_datetime(col, dt)}
			}
			if !ok {
				// inferred types cover every row, so this only happens with explicit types
				panic(fmt.tprintf("%s: row %d, column %q: cannot parse '%s' as %v", label, row_i, header[col_i], field, types[col_i]))
			}
		}
	}
	if long_rows > 0 {
		fmt.eprintf("%s: %d row(s) had more than %d fields; extra fields ignored\n", label, long_rows, n_cols)
	}

	for col in cols {
		w.add_column(&df, col)
	}
	df.rows = len(records) - 1
	return df
}


// Parse the entire CSV text into records, honoring quoted fields and newlines inside quotes.
// Returns an array of records, each record is an array of fields.
parse_csv_records :: proc(text: string, seperator: u8 = ',') -> [][]string {
	records, quoted := _parse_csv(text, seperator)
	for q in quoted {delete(q)}
	delete(quoted)
	return records
}

// RFC 4180 parsing plus: UTF-8 BOM skipped, CRLF / LF / CR line ends, blank
// lines skipped, a quote only opens a quoted field at the start of the field
// (elsewhere it is a literal character). `quoted` marks fields that were quoted.
@(private = "file")
_parse_csv :: proc(text: string, seperator: u8) -> (records: [][]string, quoted: [][]bool) {
	recs := make([dynamic][]string)
	quots := make([dynamic][]bool)
	cur_fields := make([dynamic]string)
	cur_quoted := make([dynamic]bool)
	cur_field_bytes := make([dynamic]u8, context.temp_allocator)
	field_quoted := false
	in_quote := false

	end_field :: proc(cur_fields: ^[dynamic]string, cur_quoted: ^[dynamic]bool, bytes: ^[dynamic]u8, field_quoted: ^bool) {
		append(cur_fields, string(bytes[:]))
		append(cur_quoted, field_quoted^)
		bytes^ = make([dynamic]u8, context.temp_allocator)
		field_quoted^ = false
	}
	end_record :: proc(recs: ^[dynamic][]string, quots: ^[dynamic][]bool, cur_fields: ^[dynamic]string, cur_quoted: ^[dynamic]bool) {
		// a blank line is a single empty unquoted field: skip it
		if len(cur_fields) == 1 && len(cur_fields[0]) == 0 && !cur_quoted[0] {
			clear(cur_fields)
			clear(cur_quoted)
			return
		}
		append(recs, cur_fields[:])
		append(quots, cur_quoted[:])
		cur_fields^ = make([dynamic]string)
		cur_quoted^ = make([dynamic]bool)
	}

	i := 0
	n := len(text)
	if strings.has_prefix(text, "\xEF\xBB\xBF") {i = 3} 	// UTF-8 byte order mark
	for i < n {
		b := text[i]

		if b == '"' {
			if in_quote {
				if i + 1 < n && text[i + 1] == '"' { 	// escaped quote
					append(&cur_field_bytes, u8('"'))
					i += 2
				} else { 	// closing quote
					in_quote = false
					i += 1
				}
				continue
			}
			if len(cur_field_bytes) == 0 && !field_quoted { 	// opening quote at field start
				in_quote = true
				field_quoted = true
				i += 1
				continue
			}
			// a quote inside an unquoted field is a literal character
		}

		if !in_quote {
			if b == seperator {
				end_field(&cur_fields, &cur_quoted, &cur_field_bytes, &field_quoted)
				i += 1
				continue
			}
			if b == '\n' || b == '\r' {
				end_field(&cur_fields, &cur_quoted, &cur_field_bytes, &field_quoted)
				end_record(&recs, &quots, &cur_fields, &cur_quoted)
				i += 2 if b == '\r' && i + 1 < n && text[i + 1] == '\n' else 1
				continue
			}
		}

		append(&cur_field_bytes, b)
		i += 1
	}
	// End of file: flush remaining field/record if any (an unterminated quote is closed leniently)
	if len(cur_field_bytes) > 0 || len(cur_fields) > 0 || field_quoted {
		end_field(&cur_fields, &cur_quoted, &cur_field_bytes, &field_quoted)
		end_record(&recs, &quots, &cur_fields, &cur_quoted)
	}
	delete(cur_fields)
	delete(cur_quoted)
	return recs[:], quots[:]
}


csv_free :: proc(records: [][]string) {
	for record in records {
		if record != nil {
			for &s in record {
				if s != "" {
					delete(s, context.temp_allocator)
				}
			}
			delete(record)
		}
	}
	if records != nil {
		delete(records)
	}
}


// Helper to trim optional surrounding whitespace and quotes, and unescape double quotes.
// Use this when you want to treat quoted empty string as empty string and remove outer quotes.
unquote_and_trim :: proc(str: string) -> string {
	s := strings.trim(str, " \t")
	if len(s) >= 2 && s[0] == '"' && s[len(s) - 1] == '"' {
		// remove surrounding quotes and unescape double quotes
		inner := s[1:len(s) - 1]
		out_bytes := make([dynamic]u8, context.temp_allocator)
		i := 0
		for i < len(inner) {
			if inner[i] == '"' && i + 1 < len(inner) && inner[i + 1] == '"' {
				append(&out_bytes, u8('"'))
				i += 2
			} else {
				append(&out_bytes, u8(inner[i]))
				i += 1
			}
		}
		return string(out_bytes[:])
	}
	return s
}


is_token_equal :: proc(a: string, b: string) -> bool {
	// exact match; you can add case-insensitive variant if desired
	return a == b
}


// True if the (unquoted) field text is a null token after trimming
is_null_field :: proc(field: string, null_tokens: []string) -> bool {
	trimmed := strings.trim(field, " \t")
	if len(trimmed) >= 2 && trimmed[0] == '"' && trimmed[len(trimmed) - 1] == '"' {
		trimmed = unquote_and_trim(trimmed)
	}
	for t in null_tokens {
		if is_token_equal(trimmed, t) {
			return true
		}
	}
	return false
}
