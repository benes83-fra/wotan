package tests

import w "../wotan/core"
import exporter "../wotan/exporter"
import importer "../wotan/importer"
import "core:fmt"
import "core:mem"
import "core:os"

// Importer / exporter contracts:
// - CSV, JSON, JSONL and XLSX round-trip a DataFrame exactly: types, nulls,
//   empty vs null strings, padded / quoted / multi-line / non-ASCII strings,
//   2^53+1, integral and tiny floats, Date / Time / Datetime
// - HTML round-trips everything except what HTML cannot express (empty
//   string vs null, outer whitespace)
// - exporters write the rows of a sliced view, not of the underlying frame
// - CSV / JSON / HTML / XLSX importers handle the usual real-world input quirks

io_roundtrip_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Import / Export Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	df := _io_sample_frame(allocator)
	defer w.destroy_dataframe(&df)
	files := []string{"io_rt_test.csv", "io_rt_test.json", "io_rt_test.jsonl", "io_rt_test.html", "io_rt_test.xlsx", "io_edge_test.xlsx", "io_edge_test.json", "io_view_test.csv", "io_view_test.json"}
	defer for f in files {os.remove(f)}

	// 1. exact round trips
	{
		ok_csv := exporter.csv_save(&df, "io_rt_test.csv", allocator = allocator)
		back_csv := importer.csv_load("io_rt_test.csv")
		defer w.destroy_dataframe(&back_csv)
		exporter.json_write(&df, "io_rt_test.json", allocator)
		back_json := importer.json_load("io_rt_test.json", allocator = allocator)
		defer w.destroy_dataframe(&back_json)
		exporter.jsonl_write(&df, "io_rt_test.jsonl", allocator)
		back_jsonl := importer.jsonl_load("io_rt_test.jsonl", allocator = allocator)
		defer w.destroy_dataframe(&back_jsonl)
		ok_x := exporter.xlsx_save(&df, "io_rt_test.xlsx", allocator)
		back_x := importer.xlsx_load("io_rt_test.xlsx", allocator)
		defer w.destroy_dataframe(&back_x)
		m_csv, m_json, m_jsonl, m_x := _io_diff(&df, &back_csv), _io_diff(&df, &back_json), _io_diff(&df, &back_jsonl), _io_diff(&df, &back_x)
		expect(&passed, &total, ok_csv && ok_x && m_csv == "" && m_json == "" && m_jsonl == "" && m_x == "",
			fmt.tprintf("CSV / JSON / JSONL / XLSX round trips are exact%s%s%s%s", m_csv, m_json, m_jsonl, m_x))
	}

	// 2. HTML: exact except empty string -> null and outer whitespace
	{
		exporter.html_write(&df, "io_rt_test.html", allocator)
		back := importer.html_load("io_rt_test.html", allocator)
		defer w.destroy_dataframe(&back)
		diff := _io_diff(&df, &back, html = true)
		expect(&passed, &total, diff == "", fmt.tprintf("HTML round trip (modulo empty / padded strings)%s", diff))
	}

	// 3. sliced view: rows 3..5 are exported, not rows 0..2
	{
		view := w.df_slice(&df, 3, 6)
		exporter.csv_save(&view, "io_view_test.csv", allocator = allocator)
		exporter.json_write(&view, "io_view_test.json", allocator)
		a := importer.csv_load("io_view_test.csv")
		defer w.destroy_dataframe(&a)
		b := importer.json_load("io_view_test.json", allocator = allocator)
		defer w.destroy_dataframe(&b)
		sa, _ := w.column_at_string(w.column(&a, "name"), 0)
		sb, _ := w.column_at_string(w.column(&b, "name"), 0)
		expect(&passed, &total, a.rows == 3 && b.rows == 3 && sa == "line1\nline2" && sb == "line1\nline2",
			fmt.tprintf("view export writes the view's rows (first name %q / %q)", sa, sb))
	}

	// 4. CSV input quirks
	{
		text := "\xEF\xBB\xBFid,name,x,y\r\n1,\"a,b\", 1 ,NA\r\n\r\n2,\"\",2.5,\"NA\"\r\n3,\"NA\",3\r\n4,,4,7,extra\r\n"
		d := importer.csv_load_from_string(text, allocator)
		defer w.destroy_dataframe(&d)
		name := w.column(&d, "name")
		n1, null1 := w.column_at_string(name, 1)
		n2, _ := w.column_at_string(name, 2)
		ok := d.rows == 4 && d.columns[0].name == "id" && w.column(&d, "x").type == .Float &&
			n1 == "" && !null1 && n2 == "NA" && w.is_null(name, 3) &&
			w.is_null(w.column(&d, "y"), 0) && w.is_null(w.column(&d, "y"), 1) && w.is_null(w.column(&d, "y"), 2)
		expect(&passed, &total, ok, "CSV: BOM, CRLF, blank line, short / long rows, quoted \"\" and \"NA\" are strings, mixed int/float column")
	}

	// 5. JSON: keys in file order, union over records
	{
		_ = os.write_entire_file("io_edge_test.json", transmute([]u8)string(`[{"z":1,"a":"xé"},{"a":"","m":true},{"m":false,"z":2.5}]`))
		d := importer.json_load("io_edge_test.json", allocator = allocator)
		defer w.destroy_dataframe(&d)
		a1, a1_null := w.column_at_string(w.column(&d, "a"), 1)
		ok := len(d.columns) == 3 && d.columns[0].name == "z" && d.columns[1].name == "a" && d.columns[2].name == "m" &&
			d.columns[0].type == .Float && a1 == "" && !a1_null && w.is_null(w.column(&d, "m"), 0)
		expect(&passed, &total, ok, "JSON: columns in file order over all records, \"\" stays a string, missing keys are null")
	}

	// 6. HTML: entities, row headers, colspan, upper-case tags, omitted </td>
	{
		html := `<TABLE><TR><TH>Key</TH><TH>A</TH><TH>B</TH></TR>` +
			`<TR><TH>r1</TH><TD>caf&eacute; &amp;lt;x&amp;gt; &#x263A;<TD>2</TR>` +
			`<tr><th>r2</th><td colspan="2">same</td></tr></TABLE>`
		d := importer.html_load_from_string(html, allocator)
		defer w.destroy_dataframe(&d)
		a0, _ := w.column_at_string(w.column(&d, "A"), 0)
		b1, _ := w.column_at_string(w.column(&d, "B"), 1)
		k1, _ := w.column_at_string(w.column(&d, "Key"), 1)
		expect(&passed, &total, d.rows == 2 && a0 == "café &lt;x&gt; ☺" && k1 == "r2" && b1 == "same",
			fmt.tprintf("HTML: entities decoded once (%q), row headers kept, colspan repeated (%q)", a0, b1))
	}

	// 7. Excel-style workbook (absolute part paths, rich text, cellStyleXfs before
	//    cellXfs, literal text in a number format, bool / error / r-less cells)
	{
		ok_zip := _io_write_excel_style_xlsx("io_edge_test.xlsx", allocator)
		d := importer.xlsx_load("io_edge_test.xlsx", allocator)
		defer w.destroy_dataframe(&d)
		ok := ok_zip && d.rows == 3 && len(d.columns) == 6
		if ok {
			name0, _ := w.column_at_string(w.column(&d, "name"), 0)
			amt := w.column(&d, "amount")
			a0, _ := w.column_at_float(amt, 0)
			when_ := w.column(&d, "when")
			dt0, _ := w.column_at_date(when_, 0)
			flag := w.column(&d, "flag")
			f0, _ := w.column_at_bool(flag, 0)
			code := w.column(&d, "code")
			c0, _ := w.column_at_string(code, 0)
			score := w.column(&d, "score")
			s0, _ := w.column_at_float(score, 0)
			ok = name0 == "Äpfel & <Birnen>" && amt.type == .Float && a0 == 1234.5 &&
				when_.type == .Date && dt0 == w.Date{2024, 3, 1} && flag.type == .Bool && f0 &&
				w.is_null(flag, 1) && code.type == .String && c0 == "007" && score.type == .Float && s0 == 88.5
		}
		expect(&passed, &total, ok, "XLSX (Excel-style parts): rich text, styles, USD format is not a date, bool / error / text-number cells")
	}

	fmt.printf("Import / export summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

_io_sample_frame :: proc(allocator: mem.Allocator) -> w.DataFrame {
	names := []string{"Alice", "O'Brien, Jr.", "say \"hi\"", "line1\nline2", "", "ünïcødé € 日本", "  padded  ", "007", "<b>&amp;</b>", "NA"}
	n := len(names)
	df := w.dataframe_new(allocator)
	w.add_column(&df, w.column_new("name", .String, n, allocator))
	w.add_column(&df, w.column_new("count", .Int, n, allocator))
	w.add_column(&df, w.column_new("value", .Float, n, allocator))
	w.add_column(&df, w.column_new("flag", .Bool, n, allocator))
	w.add_column(&df, w.column_new("day", .Date, n, allocator))
	w.add_column(&df, w.column_new("clock", .Time, n, allocator))
	w.add_column(&df, w.column_new("stamp", .Datetime, n, allocator))
	floats := []f64{0.1, -1e-300, 1.5e10, 2.0, 0, 2.0 / 3.0, 1e21, -2.5, 123456.789, 1e-7}
	for i in 0 ..< n {
		w.append_string(&df.columns[0], names[i])
		if i == 4 {w.append_null(&df.columns[1])} else {w.append_int(&df.columns[1], 9007199254740993 if i == 2 else i * 7 - 5)}
		if i == 4 {w.append_null(&df.columns[2])} else {w.append_float(&df.columns[2], floats[i])}
		if i == 2 {w.append_null(&df.columns[3])} else {w.append_bool(&df.columns[3], i % 3 == 0)}
		if i == 6 {w.append_null(&df.columns[4])} else {w.append_date(&df.columns[4], w.Date{i32(1999 + i), i32(1 + i), i32(28 - i)})}
		w.append_time(&df.columns[5], w.Time{i32(i), 30, i32(59 - i)})
		w.append_datetime(&df.columns[6], w.Datetime{2024, 2, i32(20 + i % 9), i32(i), 45, 7})
	}
	df.rows = n
	return df
}

// "" when b equals a cell by cell (type and value); otherwise a short report.
// html = true allows the two HTML limitations for String cells.
_io_diff :: proc(a, b: ^w.DataFrame, html := false) -> string {
	if a.rows != b.rows || len(a.columns) != len(b.columns) {
		return fmt.tprintf(" [shape %dx%d -> %dx%d]", a.rows, len(a.columns), b.rows, len(b.columns))
	}
	for &ca, c in a.columns {
		cb := &b.columns[c]
		if ca.name != cb.name || ca.type != cb.type {
			return fmt.tprintf(" [column %d: %q %v -> %q %v]", c, ca.name, ca.type, cb.name, cb.type)
		}
		for r in 0 ..< a.rows {
			na, nb := w.is_null(&ca, r), w.is_null(cb, r)
			sa := "" if na else w.value_as_string(&ca, r, context.temp_allocator)
			sb := "" if nb else w.value_as_string(cb, r, context.temp_allocator)
			if html && ca.type == .String && !na {
				if sa == "" && nb {continue}
				if sa == "  padded  " && sb == "padded" {continue}
			}
			if na != nb || sa != sb {
				return fmt.tprintf(" [%s row %d: %q -> %q]", ca.name, r, sa, sb)
			}
		}
	}
	return ""
}

// A workbook laid out the way Excel / openpyxl write them
_io_write_excel_style_xlsx :: proc(path: string, allocator: mem.Allocator) -> bool {
	ct := `<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/></Types>`
	root := `<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>`
	wb := `<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Data" sheetId="1" r:id="rId7"/></sheets></workbook>`
	rels := `<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
		`<Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="/xl/worksheets/data.xml"/>` +
		`<Relationship Id="rId8" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="/xl/strings.xml"/>` +
		`<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>`
	sst := `<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">` +
		`<si><t>name</t></si><si><t>amount</t></si><si><t>when</t></si><si><t>flag</t></si><si><t>code</t></si><si><t>score</t></si>` +
		`<si><r><rPr><b/></rPr><t>Äpfel</t></r><r><t xml:space="preserve"> &amp; &lt;Birnen&gt;</t></r></si>` +
		`<si><t>007</t></si><si><t>88.5</t></si><si><t>Kiwi</t></si></sst>`
	styles := `<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">` +
		`<numFmts count="2"><numFmt numFmtId="164" formatCode="#,##0.00 &quot;USD&quot;"/><numFmt numFmtId="165" formatCode="dd/mm/yyyy"/></numFmts>` +
		`<cellStyleXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/></cellStyleXfs>` +
		`<cellXfs count="3"><xf numFmtId="0" xfId="0"/><xf numFmtId="164" xfId="0"/><xf numFmtId="165" xfId="0"/></cellXfs></styleSheet>`
	sheet := `<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>` +
		`<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c><c r="E1" t="s"><v>4</v></c><c r="F1" t="s"><v>5</v></c></row>` +
		`<row r="2"><c r="A2" t="s"><v>6</v></c><c r="B2" s="1"><v>1234.5</v></c><c r="C2" s="2"><v>45352</v></c><c r="D2" t="b"><v>1</v></c><c r="E2" t="s"><v>7</v></c><c r="F2" t="s"><v>8</v></c></row>` +
		`<row r="3" spans="1:6"/>` +
		`<row r="4"><c t="s"><v>9</v></c><c s="1"><v>2</v></c><c s="2"><v>45353.5</v></c><c t="e"><v>#N/A</v></c><c r="F4"><v>91</v></c></row>` +
		`<row r="5"><c r="A5" t="str"><v>formula &amp; text</v></c><c r="B5"><v>-3</v></c></row>` +
		`</sheetData></worksheet>`
	entries := []exporter.ZipEntry {
		{name = "[Content_Types].xml", data = transmute([]u8)ct},
		{name = "_rels/.rels", data = transmute([]u8)root},
		{name = "xl/workbook.xml", data = transmute([]u8)wb},
		{name = "xl/_rels/workbook.xml.rels", data = transmute([]u8)rels},
		{name = "xl/strings.xml", data = transmute([]u8)sst},
		{name = "xl/styles.xml", data = transmute([]u8)styles},
		{name = "xl/worksheets/data.xml", data = transmute([]u8)sheet},
	}
	return exporter.zip_write(path, entries, allocator)
}
