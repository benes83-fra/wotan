package exporter

import w "../core"
import "core:fmt"
import "core:mem"
import "core:strings"

xlsx_save :: proc(df: ^w.DataFrame, path: string, allocator: mem.Allocator) -> bool {
	ws := worksheet_xml_from_df(df, allocator)
	wb := workbook_xml(allocator)
	wb_rel := workbook_rels_xml(allocator)
	root_rel := root_rels_xml(allocator)
	ct := content_types_xml(allocator)

	entries := []ZipEntry {
		{name = "[Content_Types].xml", data = transmute([]u8)ct},
		{name = "_rels/.rels", data = transmute([]u8)root_rel},
		{name = "xl/workbook.xml", data = transmute([]u8)wb},
		{name = "xl/_rels/workbook.xml.rels", data = transmute([]u8)wb_rel},
		{name = "xl/worksheets/sheet1.xml", data = transmute([]u8)ws},
	}

	return zip_write(path, entries, allocator)
}

excel_col_name :: proc(col_idx: int, allocator: mem.Allocator) -> string {
	// 0 -> "A", 1 -> "B", ..., 25 -> "Z", 26 -> "AA", ...
	n := col_idx
	buf := make([dynamic]u8, 0, allocator)

	for {
		rem := n % 26
		n = n / 26 - 1
		ch := u8('A' + rem)

		// prepend
		append(&buf, 0)
		copy(buf[1:], buf[:len(buf) - 1])
		buf[0] = ch

		if n < 0 {
			break
		}
	}

	return string(buf[:])
}

worksheet_xml_from_df :: proc(df: ^w.DataFrame, allocator: mem.Allocator) -> string {
	b, _ := strings.builder_make(allocator = allocator)
	defer strings.builder_destroy(&b)

	strings.write_string(
		&b,
		`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
		`<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">` +
		`<sheetData>`,
	)

	rows := df.rows
	cols := len(df.columns)

	// Header row
	strings.write_string(&b, `<row r="1">`)
	for c in 0 ..< cols {
		col_name := excel_col_name(c, allocator)


		strings.write_string(&b, `<c r="`)
		strings.write_string(&b, col_name)
		strings.write_string(&b, `1" t="inlineStr"><is><t>`)

		strings.write_string(&b, df.columns[c].name)

		strings.write_string(&b, `</t></is></c>`)
	}
	strings.write_string(&b, `</row>`)

	// Data rows
	for r in 0 ..< rows {
		row_idx := r + 2

		strings.write_string(&b, `<row r="`)
		strings.write_int(&b, row_idx)
		strings.write_string(&b, `">`)

		for c in 0 ..< cols {
			col := &df.columns[c]
			col_name := excel_col_name(c, allocator)


			strings.write_string(&b, `<c r="`)
			strings.write_string(&b, col_name)
			strings.write_int(&b, row_idx)
			strings.write_string(&b, `"`)

			if w.is_null(col, r) {
				strings.write_string(&b, `/>`)
				continue
			}

			#partial switch col.type {
			case .Int, .Float:
				strings.write_string(&b, `><v>`)
				v := w.value_as_string(col, r, allocator)
				strings.write_string(&b, v)
				strings.write_string(&b, `</v></c>`)

			case .Bool:
				strings.write_string(&b, ` t="b"><v>`)
				if w.get_bool(col, r) {
					strings.write_string(&b, "1")
				} else {
					strings.write_string(&b, "0")
				}
				strings.write_string(&b, `</v></c>`)

			case .String, .Datetime:
				strings.write_string(&b, ` t="inlineStr"><is><t>`)
				v := w.value_as_string(col, r, allocator)
				strings.write_string(&b, v)
				strings.write_string(&b, `</t></is></c>`)
			}
		}

		strings.write_string(&b, `</row>`)
	}

	strings.write_string(&b, `</sheetData></worksheet>`)

	return strings.to_string(b)
}


workbook_xml :: proc(allocator: mem.Allocator) -> string {
	return fmt.aprintf(
		`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
		`<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ` +
		`xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">` +
		`<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>` +
		`</workbook>`,
		allocator = allocator,
	)
}

workbook_rels_xml :: proc(allocator: mem.Allocator) -> string {
	return fmt.aprintf(
		`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
		`<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
		`<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>` +
		`</Relationships>`,
		allocator = allocator,
	)
}

root_rels_xml :: proc(allocator: mem.Allocator) -> string {
	return fmt.aprintf(
		`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
		`<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
		`<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>` +
		`</Relationships>`,
		allocator = allocator,
	)
}

content_types_xml :: proc(allocator: mem.Allocator) -> string {
	return fmt.aprintf(
		`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
		`<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">` +
		`<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>` +
		`<Default Extension="xml" ContentType="application/xml"/>` +
		`<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>` +
		`<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>` +
		`</Types>`,
		allocator = allocator,
	)
}
