package exporter

import w "../core"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"

// ------------------------------------------------------------
// HTML Writer (DataFrame → HTML <table>)
// ------------------------------------------------------------
html_write :: proc(df: ^w.DataFrame, path: string, allocator: mem.Allocator) {
	builder := strings.Builder{}
	strings.builder_init(&builder, allocator)
	defer strings.builder_destroy(&builder)

	// <table>
	strings.write_string(&builder, "<table>\n")

	// HEADER
	strings.write_string(&builder, "  <thead>\n    <tr>")
	for &col in df.columns {
		strings.write_string(&builder, "<th>")
		strings.write_string(&builder, col.name)
		strings.write_string(&builder, "</th>")
	}
	strings.write_string(&builder, "</tr>\n  </thead>\n")

	// BODY
	strings.write_string(&builder, "  <tbody>\n")

	for row in 0 ..< df.rows {
		strings.write_string(&builder, "    <tr>")

		for &col in df.columns {
			strings.write_string(&builder, "<td>")

			#partial switch col.type {
			case .Int:
				if col.nulls != nil && col.nulls[row] {
					// empty cell
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(int))
					v := (cast(^int)base)^
					v_int := fmt.aprint(v)
					defer delete(v_int)
					strings.write_string(&builder, v_int)
				}

			case .Float:
				if col.nulls != nil && col.nulls[row] {
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(f64))
					v := (cast(^f64)base)^
					v_f64 := fmt.aprint(v)
					defer delete(v_f64)
					strings.write_string(&builder, v_f64)
				}

			case .Bool:
				if col.nulls != nil && col.nulls[row] {
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(bool))
					v := (cast(^bool)base)^
					if v {
						strings.write_string(&builder, "true")
					} else {
						strings.write_string(&builder, "false")
					}
				}

			case .String:
				if col.nulls != nil && col.nulls[row] {
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(string))
					v := (cast(^string)base)^
					strings.write_string(&builder, v)
				}

			case .Date:
				if col.nulls != nil && col.nulls[row] {
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Date))
					v := (cast(^w.Date)base)^
					strings.write_string(&builder, w.date_to_string(v))
				}

			case .Time:
				if col.nulls != nil && col.nulls[row] {
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Time))
					v := (cast(^w.Time)base)^
					strings.write_string(&builder, w.time_to_string(v))
				}

			case .Datetime:
				if col.nulls != nil && col.nulls[row] {
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Datetime))
					v := (cast(^w.Datetime)base)^
					strings.write_string(&builder, w.datetime_to_string(v))
				}
			}

			strings.write_string(&builder, "</td>")
		}

		strings.write_string(&builder, "</tr>\n")
	}

	strings.write_string(&builder, "  </tbody>\n</table>\n")

	// Write file
	final := strings.to_string(builder)
	err := os.write_entire_file(path, transmute([]u8)final)
	if err != nil {
		panic("html_write: failed to write file")
	}
}
