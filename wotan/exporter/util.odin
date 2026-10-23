package exporter

import w "../core"
import "core:mem"
import "core:os"
import "core:strings"


write_file :: proc(path: string, contents: string) -> bool {
	flags := os.O_CREATE + os.O_TRUNC + os.O_WRONLY
	perm := os.perm_number(0o644)

	f, err := os.open(path, flags, perm)
	if err != nil {
		return false
	}
	defer os.close(f)

	// SAFE here because we only *read* the bytes
	bytes := transmute([]u8)contents

	n, err2 := os.write(f, bytes)
	return err2 == nil && n == len(bytes)
}

// ------------------------------------------------------------
// Shared cell formatting (view-aware: always goes through the column accessors)
// ------------------------------------------------------------

// Canonical text of a cell (see w.value_as_string); `is_null` for missing values
cell_text :: proc(col: ^w.Column, row: int, allocator: mem.Allocator) -> (text: string, is_null: bool) {
	if w.is_null(col, row) {
		return "", true
	}
	return w.value_as_string(col, row, allocator), false
}

// Unquoted fields the CSV importer reads as null (importer.DEFAULT_NULL_TOKEN)
CSV_NULL_TOKENS :: []string{"NA", "null", "None"}

// A string field must be quoted when it contains the separator, a quote or a
// line break, when it is empty (an unquoted empty field is null), when it has
// outer whitespace (the importer trims unquoted fields) or when it reads as a
// null token
csv_field_needs_quotes :: proc(s: string, sep: u8) -> bool {
	if len(s) == 0 {return true}
	if s[0] == ' ' || s[0] == '\t' || s[len(s) - 1] == ' ' || s[len(s) - 1] == '\t' {return true}
	for i in 0 ..< len(s) {
		ch := s[i]
		if ch == sep || ch == '"' || ch == '\n' || ch == '\r' {return true}
	}
	for t in CSV_NULL_TOKENS {
		if s == t {return true}
	}
	return false
}

// JSON string literal with the required escapes (UTF-8 passes through)
json_write_string :: proc(b: ^strings.Builder, s: string) {
	HEX := "0123456789abcdef"
	strings.write_byte(b, '"')
	for i in 0 ..< len(s) {
		ch := s[i]
		switch ch {
		case '"':
			strings.write_string(b, `\"`)
		case '\\':
			strings.write_string(b, `\\`)
		case '\n':
			strings.write_string(b, `\n`)
		case '\r':
			strings.write_string(b, `\r`)
		case '\t':
			strings.write_string(b, `\t`)
		case 0x08:
			strings.write_string(b, `\b`)
		case 0x0C:
			strings.write_string(b, `\f`)
		case:
			if ch < 0x20 {
				strings.write_string(b, `\u00`)
				strings.write_byte(b, HEX[ch >> 4])
				strings.write_byte(b, HEX[ch & 0xF])
			} else {
				strings.write_byte(b, ch)
			}
		}
	}
	strings.write_byte(b, '"')
}

// Text content escaped for XML / HTML (&, <, > and, in attributes, quotes).
// Control characters other than tab / newline / carriage return are not
// allowed in XML 1.0 and are dropped.
xml_escape_into :: proc(b: ^strings.Builder, s: string, attribute := false) {
	for i in 0 ..< len(s) {
		ch := s[i]
		switch ch {
		case '&':
			strings.write_string(b, "&amp;")
		case '<':
			strings.write_string(b, "&lt;")
		case '>':
			strings.write_string(b, "&gt;")
		case '"':
			if attribute {strings.write_string(b, "&quot;")} else {strings.write_byte(b, ch)}
		case:
			if ch < 0x20 && ch != '\t' && ch != '\n' && ch != '\r' {continue}
			strings.write_byte(b, ch)
		}
	}
}
