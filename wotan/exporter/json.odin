package exporter

import "core:encoding/json"
import "core:mem"
import "core:os"
import "core:strings"

import w "../core"

// ------------------------------------------------------------
// JSON Writer (DataFrame → JSON array of objects)
// ------------------------------------------------------------
json_write :: proc(df: ^w.DataFrame, path: string, allocator: mem.Allocator) {
	builder := strings.Builder{}
	strings.builder_init(&builder, allocator)
	defer strings.builder_destroy(&builder)

	// opening bracket
	strings.write_byte(&builder, '[')

	for row in 0 ..< df.rows {
		obj := json.Object{}

		for &col in df.columns {
			#partial switch col.type {
			case .Int:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(int))
					v := (cast(^int)base)^
					obj[col.name] = json.Integer(v)
				}

			case .Float:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(f64))
					v := (cast(^f64)base)^
					obj[col.name] = json.Float(v)
				}

			case .Bool:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(bool))
					v := (cast(^bool)base)^
					obj[col.name] = json.Boolean(v)
				}

			case .String:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(string))
					v := (cast(^string)base)^
					obj[col.name] = json.String(v)
				}

			case .Date:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Date))
					v := (cast(^w.Date)base)^
					obj[col.name] = json.String(w.date_to_string(v))
				}

			case .Time:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Time))
					v := (cast(^w.Time)base)^
					obj[col.name] = json.String(w.time_to_string(v))
				}

			case .Datetime:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Datetime))
					v := (cast(^w.Datetime)base)^
					obj[col.name] = json.String(w.datetime_to_string(v))
				}
			}
		}

		bytes, err := json.marshal(obj, allocator = allocator)
		if err != nil {
			panic("json_write: marshal failed")
		}

		if row > 0 {
			strings.write_byte(&builder, ',')
		}
		strings.write_string(&builder, string(bytes))


		delete(obj) // now safe: no array holds it
	}

	// closing bracket
	strings.write_byte(&builder, ']')

	final := strings.to_string(builder)
	err2 := os.write_entire_file(path, transmute([]u8)final)
	if err2 != nil {
		panic("json_write: failed to write file")
	}
}

// ------------------------------------------------------------
// JSON Lines Writer (DataFrame → NDJSON)
// ------------------------------------------------------------
jsonl_write :: proc(df: ^w.DataFrame, path: string, allocator: mem.Allocator) {
	builder := strings.Builder{}
	strings.builder_init(&builder, allocator)

	for row in 0 ..< df.rows {
		obj := json.Object{}
		defer delete(obj)

		for &col in df.columns {
			#partial switch col.type {
			case .Int:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(int))
					v := (cast(^int)base)^
					obj[col.name] = json.Integer(v)
				}

			case .Float:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(f64))
					v := (cast(^f64)base)^
					obj[col.name] = json.Float(v)
				}

			case .Bool:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(bool))
					v := (cast(^bool)base)^
					obj[col.name] = json.Boolean(v)
				}

			case .String:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(string))
					v := (cast(^string)base)^
					obj[col.name] = json.String(v)
				}

			case .Date:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Date))
					v := (cast(^w.Date)base)^
					obj[col.name] = json.String(w.date_to_string(v))
				}

			case .Time:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Time))
					v := (cast(^w.Time)base)^
					obj[col.name] = json.String(w.time_to_string(v))
				}

			case .Datetime:
				if col.nulls != nil && col.nulls[row] {
					obj[col.name] = json.Null{}
				} else {
					base := uintptr(col.data) + uintptr(row * size_of(w.Datetime))
					v := (cast(^w.Datetime)base)^
					obj[col.name] = json.String(w.datetime_to_string(v))
				}
			}
		}

		bytes, err := json.marshal(obj, allocator = allocator)
		if err != nil {
			panic("jsonl_write: marshal failed")
		}

		strings.write_string(&builder, string(bytes))
		strings.write_byte(&builder, '\n')
	}

	final := strings.to_string(builder)
	err2 := os.write_entire_file(path, transmute([]u8)final)
	if err2 != nil {
		panic("json_write: failed to write file")
	}
}
