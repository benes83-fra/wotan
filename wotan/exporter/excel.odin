package exporter

import w "../core"
import "core:fmt"
import "core:mem"
import "core:strings"

// DataFrame → .xlsx (single sheet "Sheet1"). Numbers are written exactly,
// booleans as t="b", strings as inline strings (whitespace preserved), dates /
// times / datetimes as Excel serial numbers with ISO display formats, so Excel
// shows them as dates and importer.xlsx_load reads them back with their type.
// Null cells are omitted; non-finite floats are written as empty cells.
xlsx_save :: proc(df: ^w.DataFrame, path: string, allocator: mem.Allocator) -> bool {
	ws := worksheet_xml_from_df(df, allocator)
	wb := workbook_xml(allocator)
	wb_rel := workbook_rels_xml(allocator)
	root_rel := root_rels_xml(allocator)
	ct := content_types_xml(allocator)
	st := styles_xml(allocator)
	defer {
		delete(ws, allocator)
		delete(wb, allocator)
		delete(wb_rel, allocator)
		delete(root_rel, allocator)
		delete(ct, allocator)
		delete(st, allocator)
	}

	entries := []ZipEntry {
		{name = "[Content_Types].xml", data = transmute([]u8)ct},
		{name = "_rels/.rels", data = transmute([]u8)root_rel},
		{name = "xl/workbook.xml", data = transmute([]u8)wb},
		{name = "xl/_rels/workbook.xml.rels", data = transmute([]u8)wb_rel},
		{name = "xl/styles.xml", data = transmute([]u8)st},
		{name = "xl/worksheets/sheet1.xml", data = transmute([]u8)ws},
	}

	return zip_write(path, entries, allocator)
}

// Cell style indices into <cellXfs> of styles_xml
XLSX_STYLE_DATE :: 1
XLSX_STYLE_DATETIME :: 2
XLSX_STYLE_TIME :: 3

excel_col_name :: proc(col_idx: int, allocator: mem.Allocator) -> string {
	// 0 -> "A", 1 -> "B", ..., 25 -> "Z", 26 -> "AA", ...
	buf: [8]u8
	i := len(buf)
	n := col_idx
	for {
		i -= 1
		buf[i] = u8('A' + n % 26)
		n = n / 26 - 1
		if n < 0 {break}
	}
	return strings.clone(string(buf[i:]), allocator)
}

// Excel serial day number (1900 date system, including Excel's fictitious
// 1900-02-29 so that serials before March 1900 match Excel)
excel_serial_from_date :: proc(y, m, d: int) -> int {
	n := int(w.days_from_civil(i64(y), i64(m), i64(d)) - w.days_from_civil(1899, 12, 30))
	if n < 61 {n -= 1}
	return n
}

worksheet_xml_from_df :: proc(df: ^w.DataFrame, allocator: mem.Allocator) -> string {
	b, _ := strings.builder_make(allocator = allocator)
	// the builder's buffer becomes the returned string (not destroyed here)

	strings.write_string(
		&b,
		`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
		`<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">` +
		`<sheetData>`,
	)

	rows := df.rows
	cols := len(df.columns)
	col_names := make([]string, cols, context.temp_allocator)
	defer delete(col_names, context.temp_allocator)
	for c in 0 ..< cols {col_names[c] = excel_col_name(c, context.temp_allocator)}

	open_cell :: proc(b: ^strings.Builder, col_name: string, row_idx: int) {
		strings.write_string(b, `<c r="`)
		strings.write_string(b, col_name)
		strings.write_int(b, row_idx)
		strings.write_byte(b, '"')
	}
	inline_string :: proc(b: ^strings.Builder, s: string) {
		strings.write_string(b, ` t="inlineStr"><is><t xml:space="preserve">`)
		xml_escape_into(b, s)
		strings.write_string(b, `</t></is></c>`)
	}
	serial_cell :: proc(b: ^strings.Builder, style: int, serial: f64) {
		strings.write_string(b, ` s="`)
		strings.write_int(b, style)
		strings.write_string(b, `"><v>`)
		s := w.float_to_string(serial, context.temp_allocator)
		strings.write_string(b, s)
		delete(s, context.temp_allocator)
		strings.write_string(b, `</v></c>`)
	}

	// Header row
	strings.write_string(&b, `<row r="1">`)
	for c in 0 ..< cols {
		open_cell(&b, col_names[c], 1)
		inline_string(&b, df.columns[c].name)
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
			if w.is_null(col, r) {continue} 	// missing cells are simply absent

			switch col.type {
			case .Int:
				v, _ := w.column_at_int(col, r)
				open_cell(&b, col_names[c], row_idx)
				strings.write_string(&b, `><v>`)
				strings.write_int(&b, v)
				strings.write_string(&b, `</v></c>`)
			case .Float:
				v, _ := w.column_at_float(col, r)
				if v != v || v > max(f64) || v < -max(f64) {continue} 	// no NaN / Inf in xlsx
				open_cell(&b, col_names[c], row_idx)
				strings.write_string(&b, `><v>`)
				s := w.float_to_string(v, context.temp_allocator)
				strings.write_string(&b, s)
				delete(s, context.temp_allocator)
				strings.write_string(&b, `</v></c>`)
			case .Bool:
				v, _ := w.column_at_bool(col, r)
				open_cell(&b, col_names[c], row_idx)
				strings.write_string(&b, ` t="b"><v>`)
				strings.write_byte(&b, '1' if v else '0')
				strings.write_string(&b, `</v></c>`)
			case .String:
				v, _ := w.column_at_string(col, r)
				open_cell(&b, col_names[c], row_idx)
				inline_string(&b, v)
			case .Date:
				v, _ := w.column_at_date(col, r)
				open_cell(&b, col_names[c], row_idx)
				serial_cell(
					&b,
					XLSX_STYLE_DATE,
					f64(excel_serial_from_date(int(v.year), int(v.month), int(v.day))),
				)
			case .Datetime:
				v, _ := w.column_at_datetime(col, r)
				open_cell(&b, col_names[c], row_idx)
				day := f64(excel_serial_from_date(int(v.year), int(v.month), int(v.day)))
				secs := f64(v.hour * 3600 + v.minute * 60 + v.second)
				serial_cell(&b, XLSX_STYLE_DATETIME, day + secs / 86400.0)
			case .Time:
				v, _ := w.column_at_time(col, r)
				open_cell(&b, col_names[c], row_idx)
				serial_cell(
					&b,
					XLSX_STYLE_TIME,
					f64(v.hour * 3600 + v.minute * 60 + v.second) / 86400.0,
				)
			case .Invalid:
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
		`<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>` +
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
		`<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>` +
		`</Types>`,
		allocator = allocator,
	)
}

// Minimal stylesheet: default style + ISO date / datetime / time formats
// (cellXfs indices XLSX_STYLE_DATE / _DATETIME / _TIME)
styles_xml :: proc(allocator: mem.Allocator) -> string {
	return fmt.aprintf(
		`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
		`<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">` +
		`<numFmts count="3">` +
		`<numFmt numFmtId="164" formatCode="yyyy-mm-dd"/>` +
		`<numFmt numFmtId="165" formatCode="yyyy-mm-dd hh:mm:ss"/>` +
		`<numFmt numFmtId="166" formatCode="hh:mm:ss"/>` +
		`</numFmts>` +
		`<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>` +
		`<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>` +
		`<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>` +
		`<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>` +
		`<cellXfs count="4">` +
		`<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>` +
		`<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>` +
		`<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>` +
		`<xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>` +
		`</cellXfs>` +
		`<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>` +
		`</styleSheet>`,
		allocator = allocator,
	)
}
