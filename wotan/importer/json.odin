package importer

import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"

import w "../core"
import infer "../importer"


// ------------------------------------------------------------
// JSON Loader (array of objects)
// ------------------------------------------------------------
json_load :: proc(
	path: string,
	null_tokens: []string = DEFAULT_NULL_TOKEN,
	allocator: mem.Allocator,
) -> w.DataFrame {

	contents, err := read_file(path)
	if err != nil {
		panic("json_load: failed to read file")
	}
	defer delete(contents)

	text := string(contents)

	root, err2 := json.parse_string(text, json.DEFAULT_SPECIFICATION, true, allocator)
	if err2 != .None {
		panic("json_load: invalid JSON")
	}

	// Declare arr here so it exists in the outer scope
	arr: json.Array

	#partial switch v in root {
	case json.Array:
		arr = v
	case:
		panic("json_load: expected top-level JSON array")
	}

	if len(arr) == 0 {
		panic("json_load: empty JSON array")
	}

	// Use type assertion to ensure elements are Objects
	for val, i in arr {
		if _, ok := val.(json.Object); !ok {
			panic(fmt.tprintf("json_load: element %d is not an object", i + 1))
		}
	}

	// Determine schema from first object
	first_obj := arr[0].(json.Object)
	col_count := len(first_obj)
	keys := make([]string, col_count, allocator)

	i := 0
	for key in first_obj {
		keys[i] = key
		i += 1
	}
	samples := make([][dynamic]string, col_count)

	sample_limit := min(100, len(arr))
	for i in 0 ..< sample_limit {
		obj := arr[i].(json.Object) // Cast to Object

		for col_i in 0 ..< col_count {
			key := keys[col_i]
			val := obj[key]

			// Check for Null type in union
			if _, is_null := val.(json.Null); is_null {
				continue
			}

			s := json_value_to_string(val)
			if is_null_field(s, null_tokens) {
				continue
			}

			append(&samples[col_i], s)
		}
	}

	types := make([]w.ColumnType, col_count, allocator)
	for i in 0 ..< col_count {
		types[i] = infer.infer_column_type(samples[i][:])
	}

	for i in 0 ..< col_count {
		delete(samples[i])
	}
	delete(samples)

	df := w.dataframe_new()
	cols := make([]w.Column, col_count, allocator)

	for i in 0 ..< col_count {
		cols[i] = w.column_new(keys[i], types[i], len(arr))
	}

	for row_i in 0 ..< len(arr) {
		obj := arr[row_i].(json.Object) // Cast to Object

		for col_i in 0 ..< col_count {
			key := keys[col_i]
			val := obj[key]

			if _, is_null := val.(json.Null); is_null {
				w.append_null(&cols[col_i])
				continue
			}

			s := json_value_to_string(val)
			if is_null_field(s, null_tokens) {
				w.append_null(&cols[col_i])
				continue
			}

			#partial switch types[col_i] {
			case .Int:
				v, ok := strconv.parse_int(s)
				if !ok {panic(fmt.tprintf("json_load: invalid int '%s'", s))}
				w.append_int(&cols[col_i], v)

			case .Float:
				v, ok := strconv.parse_f64(s)
				if !ok {panic(fmt.tprintf("json_load: invalid float '%s'", s))}
				w.append_float(&cols[col_i], v)

			case .Bool:
				w.append_bool(&cols[col_i], s == "true")

			case .String:
				w.append_string(&cols[col_i], s)

			case .Date:
				d, ok := w.parse_date(s)
				if !ok {panic(fmt.tprintf("json_load: invalid date '%s'", s))}
				w.append_date(&cols[col_i], d)

			case .Time:
				t, ok := w.parse_time(s)
				if !ok {panic(fmt.tprintf("json_load: invalid time '%s'", s))}
				w.append_time(&cols[col_i], t)

			case .Datetime:
				dt, ok := w.parse_datetime(s)
				if !ok {panic(fmt.tprintf("json_load: invalid datetime '%s'", s))}
				w.append_datetime(&cols[col_i], dt)
			}
		}
	}

	for col in cols {
		w.add_column(&df, col)
	}

	return df
}


// ------------------------------------------------------------
// Convert JSON value to string for type inference + parsing
// ------------------------------------------------------------
json_value_to_string :: proc(v: json.Value) -> string {
	#partial switch val in v {
	case json.String:
		return val
	case json.Integer:
		// write_int requires a buffer. 32 bytes is plenty for an i64.
		buf: [32]byte
		return strings.clone(strconv.write_int(buf[:], i64(val), 10), context.temp_allocator)
	case json.Float:
		// write_float requires a buffer.
		buf: [64]byte
		return strings.clone(
			strconv.write_float(buf[:], f64(val), 'f', -1, 64),
			context.temp_allocator,
		)
	case json.Boolean:
		return val ? "true" : "false"
	case json.Null:
		return ""
	case json.Object, json.Array:
		// Marshal returns []u8, cast it to string
		bytes, _ := json.marshal(v, allocator = context.temp_allocator)
		return string(bytes)
	}
	return ""
}
jsonl_load :: proc(
	path: string,
	null_tokens: []string = DEFAULT_NULL_TOKEN,
	allocator: mem.Allocator,
) -> w.DataFrame {

	contents, err := read_file(path)
	if err != nil {
		panic("jsonl_load: failed to read file")
	}
	defer delete(contents)

	text := string(contents)
	lines := strings.split(text, "\n")
	defer delete(lines)

	// --------------------------------------------------------
	// First pass: collect sample objects for schema inference
	// --------------------------------------------------------
	sample_objects := make([dynamic]json.Object, 0, allocator)

	for i in 0 ..< len(lines) {
		line := strings.trim(lines[i], " \t\r")
		if len(line) == 0 {
			continue
		}

		root, err2 := json.parse_string(line, json.DEFAULT_SPECIFICATION, true, allocator)
		if err2 != .None {
			panic(fmt.tprintf("jsonl_load: invalid JSON on line %d", i + 1))
		}

		obj, ok := root.(json.Object)
		if !ok {
			panic(fmt.tprintf("jsonl_load: line %d is not a JSON object", i + 1))
		}

		append(&sample_objects, obj)
		if len(sample_objects) >= 100 {
			break
		}
	}

	if len(sample_objects) == 0 {
		panic("jsonl_load: no valid JSON objects found")
	}

	// --------------------------------------------------------
	// Determine schema from first object
	// --------------------------------------------------------
	first_obj := sample_objects[0]
	col_count := len(first_obj)
	keys := make([]string, col_count, allocator)

	idx := 0
	for key in first_obj {
		keys[idx] = key
		idx += 1
	}

	samples := make([][dynamic]string, col_count)

	// Collect sample values for type inference
	for &obj in sample_objects {
		for col_i in 0 ..< col_count {
			key := keys[col_i]
			val := obj[key]

			if _, is_null := val.(json.Null); is_null {
				continue
			}

			s := json_value_to_string(val)
			if is_null_field(s, null_tokens) {
				continue
			}

			append(&samples[col_i], s)
		}
	}

	// Infer column types
	types := make([]w.ColumnType, col_count, allocator)
	for i in 0 ..< col_count {
		types[i] = infer.infer_column_type(samples[i][:])
	}

	// Cleanup sample buffers
	for i in 0 ..< col_count {
		delete(samples[i])
	}
	delete(samples)

	// --------------------------------------------------------
	// Create DataFrame + columns
	// --------------------------------------------------------
	df := w.dataframe_new()
	cols := make([]w.Column, col_count, allocator)

	for i in 0 ..< col_count {
		cols[i] = w.column_new(keys[i], types[i], len(lines))
	}

	// --------------------------------------------------------
	// Second pass: parse all lines and append rows
	// --------------------------------------------------------
	for line_i in 0 ..< len(lines) {
		line := strings.trim(lines[line_i], " \t\r")
		if len(line) == 0 {
			continue
		}

		root, err2 := json.parse_string(line, json.DEFAULT_SPECIFICATION, true, allocator)
		if err2 != .None {
			panic(fmt.tprintf("jsonl_load: invalid JSON on line %d", line_i + 1))
		}

		obj, ok := root.(json.Object)
		if !ok {
			panic(fmt.tprintf("jsonl_load: line %d is not a JSON object", line_i + 1))
		}

		for col_i in 0 ..< col_count {
			key := keys[col_i]
			val := obj[key]

			if _, is_null := val.(json.Null); is_null {
				w.append_null(&cols[col_i])
				continue
			}

			s := json_value_to_string(val)
			if is_null_field(s, null_tokens) {
				w.append_null(&cols[col_i])
				continue
			}

			#partial switch types[col_i] {
			case .Int:
				v, ok := strconv.parse_int(s)
				if !ok {panic(fmt.tprintf("jsonl_load: invalid int '%s'", s))}
				w.append_int(&cols[col_i], v)

			case .Float:
				v, ok := strconv.parse_f64(s)
				if !ok {panic(fmt.tprintf("jsonl_load: invalid float '%s'", s))}
				w.append_float(&cols[col_i], v)

			case .Bool:
				w.append_bool(&cols[col_i], s == "true")

			case .String:
				w.append_string(&cols[col_i], s)

			case .Date:
				d, ok := w.parse_date(s)
				if !ok {panic(fmt.tprintf("jsonl_load: invalid date '%s'", s))}
				w.append_date(&cols[col_i], d)

			case .Time:
				t, ok := w.parse_time(s)
				if !ok {panic(fmt.tprintf("jsonl_load: invalid time '%s'", s))}
				w.append_time(&cols[col_i], t)

			case .Datetime:
				dt, ok := w.parse_datetime(s)
				if !ok {panic(fmt.tprintf("jsonl_load: invalid datetime '%s'", s))}
				w.append_datetime(&cols[col_i], dt)
			}
		}
	}

	// Add columns to DataFrame
	for col in cols {
		w.add_column(&df, col)
	}

	return df
}
extract_json_array_i64 :: proc(json: string, key: string) -> []i64 {
	start := strings.index(json, key)
	if start < 0 {return []i64{}}

	start = strings.index(json[start:], "[")
	if start < 0 {return []i64{}}
	start += strings.index(json, key) + 1

	end := strings.index(json[start:], "]")
	if end < 0 {return []i64{}}
	end += start

	slice := json[start:end]
	parts := strings.split(slice, ",", context.temp_allocator)

	out := make([]i64, len(parts), context.temp_allocator)
	for p, i in parts {
		v, _ := strconv.parse_i64(strings.trim_space(p))
		out[i] = v
	}
	return out[:]
}

extract_json_array_f64 :: proc(json: string, key: string) -> []f64 {
	start := strings.index(json, key)
	if start < 0 {return []f64{}}

	start = strings.index(json[start:], "[")
	if start < 0 {return []f64{}}
	start += strings.index(json, key) + 1

	end := strings.index(json[start:], "]")
	if end < 0 {return []f64{}}
	end += start

	slice := json[start:end]
	parts := strings.split(slice, ",", context.temp_allocator)

	out := make([]f64, len(parts), context.temp_allocator)
	for p, i in parts {
		v, _ := strconv.parse_f64(strings.trim_space(p))
		out[i] = v
	}
	return out[:]
}
