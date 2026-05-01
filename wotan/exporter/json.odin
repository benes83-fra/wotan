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
	arr := json.Array{}

	for row in 0 ..< df.rows {
		obj := json.Object{}

		for &col in df.columns {
			s := cast(^w.Series)&col // <-- THIS is the correct bridge

			#partial switch col.type {
			case .Int:
				v, is_null := w.series_at_int(s, row)
				obj[col.name] = is_null ? json.Null{} : json.Integer(v)

			case .Float:
				v, is_null := w.series_at_float(s, row)
				obj[col.name] = is_null ? json.Null{} : json.Float(v)

			case .Bool:
				v, is_null := w.series_at_bool(s, row)
				obj[col.name] = is_null ? json.Null{} : json.Boolean(v)

			case .String:
				v, is_null := w.series_at_string(s, row)
				obj[col.name] = is_null ? json.Null{} : json.String(v)

			case .Date:
				v, is_null := w.series_at_date(s, row)
				obj[col.name] = is_null ? json.Null{} : json.String(w.date_to_string(v))

			case .Time:
				v, is_null := w.series_at_time(s, row)
				obj[col.name] = is_null ? json.Null{} : json.String(w.time_to_string(v))

			case .Datetime:
				v, is_null := w.series_at_datetime(s, row)
				obj[col.name] = is_null ? json.Null{} : json.String(w.datetime_to_string(v))
			}
		}

		append(&arr, obj)
	}

	bytes, err := json.marshal(arr, allocator = allocator)
	if err != nil {
		panic("json_write: marshal failed")
	}

	err2 := os.write_entire_file(path, bytes)
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

		for &col in df.columns {
			s := cast(^w.Series)&col

			#partial switch col.type {
			case .Int:
				v, is_null := w.series_at_int(s, row)
				obj[col.name] = is_null ? json.Null{} : json.Integer(v)

			case .Float:
				v, is_null := w.series_at_float(s, row)
				obj[col.name] = is_null ? json.Null{} : json.Float(v)

			case .Bool:
				v, is_null := w.series_at_bool(s, row)
				obj[col.name] = is_null ? json.Null{} : json.Boolean(v)

			case .String:
				v, is_null := w.series_at_string(s, row)
				obj[col.name] = is_null ? json.Null{} : json.String(v)

			case .Date:
				v, is_null := w.series_at_date(s, row)
				obj[col.name] = is_null ? json.Null{} : json.String(w.date_to_string(v))

			case .Time:
				v, is_null := w.series_at_time(s, row)
				obj[col.name] = is_null ? json.Null{} : json.String(w.time_to_string(v))

			case .Datetime:
				v, is_null := w.series_at_datetime(s, row)
				obj[col.name] = is_null ? json.Null{} : json.String(w.datetime_to_string(v))
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
