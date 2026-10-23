package importer

import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"

import w "../core"
import infer "../importer"

// Columns are the union of the keys of all records, in order of first
// appearance in the file; a record without a key contributes null. JSON null
// is null; in String columns a JSON string is always the literal text (so ""
// and "NA" stay strings); in typed columns empty strings and null tokens are
// null. Nested objects / arrays are kept as their JSON text.


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

	root, err2 := json.parse_string(contents, json.DEFAULT_SPECIFICATION, true, allocator)
	if err2 != .None {
		panic("json_load: invalid JSON")
	}
	defer json.destroy_value(root, allocator)

	arr, is_arr := root.(json.Array)
	if !is_arr {
		panic("json_load: expected top-level JSON array")
	}
	if len(arr) == 0 {
		panic("json_load: empty JSON array")
	}

	objects := make([]json.Object, len(arr), context.temp_allocator)
	defer delete(objects, context.temp_allocator)
	for val, i in arr {
		obj, ok := val.(json.Object)
		if !ok {
			panic(fmt.tprintf("json_load: element %d is not an object", i + 1))
		}
		objects[i] = obj
	}

	keys := make([dynamic]string, 0, 16, context.temp_allocator)
	seen := make(map[string]bool, allocator = context.temp_allocator)
	defer {
		delete(keys)
		delete(seen)
	}
	_json_collect_keys(contents, 2, &keys, &seen) // array = depth 1, records = depth 2
	return _json_build(objects, keys[:], null_tokens, allocator, "json_load")
}


// ------------------------------------------------------------
// JSON Lines Loader (one object per line)
// ------------------------------------------------------------
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

	lines := strings.split(contents, "\n")
	defer delete(lines)

	values := make([dynamic]json.Value, 0, len(lines), context.temp_allocator)
	objects := make([dynamic]json.Object, 0, len(lines), context.temp_allocator)
	keys := make([dynamic]string, 0, 16, context.temp_allocator)
	seen := make(map[string]bool, allocator = context.temp_allocator)
	defer {
		for v in values {json.destroy_value(v, allocator)}
		delete(values)
		delete(objects)
		delete(keys)
		delete(seen)
	}

	for raw_line, i in lines {
		line := strings.trim(raw_line, " \t\r")
		if len(line) == 0 {
			continue
		}
		root, err2 := json.parse_string(line, json.DEFAULT_SPECIFICATION, true, allocator)
		if err2 != .None {
			panic(fmt.tprintf("jsonl_load: invalid JSON on line %d", i + 1))
		}
		append(&values, root)
		obj, ok := root.(json.Object)
		if !ok {
			panic(fmt.tprintf("jsonl_load: line %d is not a JSON object", i + 1))
		}
		append(&objects, obj)
		_json_collect_keys(line, 1, &keys, &seen)
	}

	if len(objects) == 0 {
		panic("jsonl_load: no valid JSON objects found")
	}
	return _json_build(objects[:], keys[:], null_tokens, allocator, "jsonl_load")
}


// Object keys found at nesting `depth` (1 = top-level object), appended to
// `keys` in order of first appearance. A string at that depth followed by ':'
// is a key; escaped keys are unescaped with the JSON parser.
@(private = "file")
_json_collect_keys :: proc(text: string, depth: int, keys: ^[dynamic]string, seen: ^map[string]bool) {
	d := 0
	i := 0
	n := len(text)
	for i < n {
		ch := text[i]
		switch ch {
		case '{', '[':
			d += 1
			i += 1
		case '}', ']':
			d -= 1
			i += 1
		case '"':
			start := i
			i += 1
			escaped := false
			for i < n && text[i] != '"' {
				if text[i] == '\\' {escaped = true; i += 1}
				i += 1
			}
			end := min(i, n) // index of the closing quote
			i += 1
			if d != depth {continue}
			j := i
			for j < n && (text[j] == ' ' || text[j] == '\t' || text[j] == '\r' || text[j] == '\n') {j += 1}
			if j >= n || text[j] != ':' {continue}
			key := text[start + 1:end]
			if escaped {
				v, err := json.parse_string(text[start:end + 1], json.DEFAULT_SPECIFICATION, true, context.temp_allocator)
				if s, ok := v.(json.String); ok && err == .None {key = s}
			}
			if !seen[key] {
				seen[key] = true
				append(keys, key)
			}
		case:
			i += 1
		}
	}
}

@(private = "file")
_json_build :: proc(
	objects: []json.Object,
	keys: []string,
	null_tokens: []string,
	allocator: mem.Allocator,
	label: string,
) -> w.DataFrame {
	col_count := len(keys)

	// Infer types from every record
	samples := make([][dynamic]string, col_count)
	defer {
		for i in 0 ..< col_count {delete(samples[i])}
		delete(samples)
	}
	for &obj in objects {
		for key, col_i in keys {
			val, present := obj[key]
			if !present {continue}
			if _, is_null := val.(json.Null); is_null {continue}
			s := json_value_to_string(val)
			if is_null_field(s, null_tokens) {continue}
			append(&samples[col_i], s)
		}
	}
	types := make([]w.ColumnType, col_count, context.temp_allocator)
	defer delete(types, context.temp_allocator)
	for i in 0 ..< col_count {
		types[i] = infer.infer_column_type(samples[i][:])
	}

	df := w.dataframe_new(allocator)
	cols := make([]w.Column, col_count, context.temp_allocator)
	defer delete(cols, context.temp_allocator)
	for i in 0 ..< col_count {
		cols[i] = w.column_new(strings.clone(keys[i], allocator), types[i], len(objects), allocator)
	}

	for &obj, row_i in objects {
		for key, col_i in keys {
			col := &cols[col_i]
			val, present := obj[key]
			if !present {
				w.append_null(col)
				continue
			}
			if _, is_null := val.(json.Null); is_null {
				w.append_null(col)
				continue
			}

			s := json_value_to_string(val)
			if types[col_i] == .String {
				if _, is_str := val.(json.String); !is_str && is_null_field(s, null_tokens) {
					w.append_null(col)
				} else {
					w.append_string(col, s)
				}
				continue
			}

			field := strings.trim(s, " \t")
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
				w.append_bool(col, strings.equal_fold(field, "true"))
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
				panic(fmt.tprintf("%s: record %d, key %q: cannot parse '%s' as %v", label, row_i + 1, key, field, types[col_i]))
			}
		}
	}

	for col in cols {
		w.add_column(&df, col)
	}
	df.rows = len(objects)
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
		// shortest text that parses back exactly; always marked as a float
		return w.float_to_string(f64(val), context.temp_allocator)
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
