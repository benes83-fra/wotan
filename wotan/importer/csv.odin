package importer

import "core:fmt"
import "core:strings"
import "core:strconv"
import w "../core"
import infer "../importer"   // or whatever path matches your layout

DEFAULT_NULL_TOKEN::[]string{"NA","null","None",""}

csv_load :: proc(path: string, types: []w.ColumnType, null_tokens: [] string =DEFAULT_NULL_TOKEN) -> w.DataFrame {
   
    
    contents, err2 := read_file(path)
    if err2 !=nil{
        panic("csv_load: failed to read file")
    }
    defer delete(contents)

    text := string(contents)
    records := parse_csv_records(text)
    if len(records)== 0 {
        panic ("csv_load: empty file")

    }
    lines := strings.split(text, "\n")

    header := records[0]
    if len(header) != len (types){
        panic ("csv_load: header/type count missmatch")
    }
    

    df := w.dataframe_new()

    // Create columns
    cols := make([]w.Column, len(types))
    for i in 0..<len(types) {
        cols[i] = w.column_new(header[i], types[i], len(records))
    }
    
    // Parse rows
    for row_i in 1..<len(records) {
        

        fields := records[row_i]
        
        if len(fields) != len(types) {
            panic(fmt.tprintf("csv_load: row %d has wrong number of fields", row_i))
        }

        for col_i in 0..<len(types) {
            field_raw := unquote_and_trim(fields[col_i])
            if is_null_field (field_raw,null_tokens){
                w.append_null(&cols[col_i])
                continue
            }
            field := unquote_and_trim(strings.trim(field_raw,"\t"))

            #partial switch types[col_i] {
            case .Int:
                v, ok :=strconv.parse_int(field)
                if !ok {
                    panic(fmt.tprintf("csv_load: invalid int '%s'", field))
                }
                w.append_int(&cols[col_i], v)

            case .Float:
                v, ok := strconv.parse_f64(field)
                if !ok {
                    panic(fmt.tprintf("csv_load: invalid float '%s'", field))
                }
                w.append_float(&cols[col_i], v)

            case .Bool:
                w.append_bool(&cols[col_i], field == "true")

            case .String:
                w.append_string(&cols[col_i], field)

            case .Date:
                // Very simple YYYY-MM-DD parser
                parts := strings.split(field, "-")
                if len(parts) != 3 {
                    panic(fmt.tprintf("csv_load: invalid date '%s'", field))
                }
                year,_ := strconv.parse_int(parts[0])
                month, _ := strconv.parse_int((parts[1]))
                day, _ := strconv.parse_int(parts[2])
                w.append_date(&cols[col_i], w.Date{i32(year), i32(month), i32(day)})
            }
        }
    }

    // Add columns to DataFrame
    for col in cols {
        w.add_column(&df, col)
    }

    return df
}
csv_load_auto :: proc(path: string, null_tokens: [] string =[]string{"NA","null","None"}) -> w.DataFrame {
    // Read file

    contents, err := read_file(path)
    if err!=nil {
        panic("csv_load_auto: failed to read file")
    }

    text := string(contents)
    records := parse_csv_records(text)
    if len(records) <=1{
        panic ("csv_auto_load: no data rows")
    }

    header :=records[0]
    col_count := len(header)

    // Collect samples
    sample_limit := min(100, len(records)-1)
    samples := make([][dynamic]string, col_count)
    defer delete(samples)
    for i in 0..<col_count {
        samples[i] = make([dynamic]string)
    }

    for row_i in 1..=sample_limit {
        fields := records [row_i]
        if len(fields) != col_count {
            continue
        }
        for col_i in 0..<col_count {
            if is_null_field(fields[col_i],null_tokens){
                _ =append(&samples[col_i],"")
            }else {
             _ = append(&samples[col_i], fields[col_i])
            }
        }
    }

    // Infer types
    types := make([]w.ColumnType, col_count)
    for i in 0..<col_count {
        types[i] = infer.infer_column_type(samples[i][:])
    }

    // Delegate to typed loader
    return csv_load(path, types)
}



// Parse the entire CSV text into records, honoring quoted fields and newlines inside quotes.
// Returns an array of records, each record is an array of fields.
parse_csv_records :: proc(text: string) -> [][]string {
    records := make([dynamic][]string)
    cur_fields := make([dynamic]string)
    cur_field_bytes := make([dynamic]u8)

    in_quote := false
    i := 0
    n := len(text)

    for i < n {
        b := text[i]

        // CRLF normalization: treat '\r' as part of newline handling
        if b == '"' {
            // Quote handling
            if !in_quote {
                in_quote = true
                i += 1
                continue
            } else {
                // If next char is also a quote, it's an escaped quote -> append one quote
                if i+1 < n && text[i+1] == '"' {
                    _, _ = append(&cur_field_bytes, u8('"'))
                    i += 2
                    continue
                } else {
                    // Closing quote
                    in_quote = false
                    i += 1
                    continue
                }
            }
        }

        if !in_quote {
            if b == ',' {
                // end of field
                field := string(cur_field_bytes[:])
                _, _ = append(&cur_fields, field)
                // reset field buffer
                cur_field_bytes = make([dynamic]u8)
                i += 1
                continue
            }

            if b == '\n' {
                // end of record
                field := string(cur_field_bytes[:])
                _, _ = append(&cur_fields, field)
                _, _ = append(&records, cur_fields[:])
                // reset for next record
                cur_fields = make([dynamic]string)
                cur_field_bytes = make([dynamic]u8)
                i += 1
                continue
            }

            if b == '\r' {
                // handle CRLF or lone CR
                // if next is '\n', skip both; otherwise treat CR as newline
                if i+1 < n && text[i+1] == '\n' {
                    field := string(cur_field_bytes[:])
                    _, _ = append(&cur_fields, field)
                    _, _ = append(&records, cur_fields[:])
                    cur_fields = make([dynamic]string)
                    cur_field_bytes = make([dynamic]u8)
                    i += 2
                    continue
                } else {
                    field := string(cur_field_bytes[:])
                    _, _ = append(&cur_fields, field)
                    _, _ = append(&records, cur_fields[:])
                    cur_fields = make([dynamic]string)
                    cur_field_bytes = make([dynamic]u8)
                    i += 1
                    continue
                }
            }
        }

        // default: append byte to current field
        _, _ = append(&cur_field_bytes, u8(b))
        i += 1
    }

    // End of file: flush remaining field/record if any
    // If we are still in a quote at EOF, we treat it as closed (lenient)
    if len(cur_field_bytes) > 0 || len(cur_fields) > 0 {
        field := string(cur_field_bytes[:])
        _, _ = append(&cur_fields, field)
        _, _ = append(&records, cur_fields[:])
    }

    return records[:]
}

// Helper to trim optional surrounding whitespace and quotes, and unescape double quotes.
// Use this when you want to treat quoted empty string as empty string and remove outer quotes.
unquote_and_trim :: proc(str: string) -> string {
    s:= strings.trim(str, " \t")
    if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
        // remove surrounding quotes and unescape double quotes
        inner := s[1:len(s)-1]
        // replace "" with "
        // simple implementation: scan and build
        out_bytes := make([dynamic]u8)
        defer delete(out_bytes)
        i := 0
        for i < len(inner) {
            if inner[i] == '"' && i+1 < len(inner) && inner[i+1] == '"' {
                _, _ = append(&out_bytes, u8('"'))
                i += 2
            } else {
                _, _ = append(&out_bytes, u8(inner[i]))
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



is_null_field :: proc(field: string, null_tokens: []string) -> bool {
    // Trim whitespace first
    trimmed := strings.trim(field, " \t")
    // If quoted, unquote and treat inner value as the field content
    if len(trimmed) >= 2 && trimmed[0] == '"' && trimmed[len(trimmed)-1] == '"' {
        inner := unquote_and_trim(trimmed)
        // If you want quoted empty string to be considered empty string (not null),
        // remove "" from default null_tokens. Current behavior: compare inner to tokens.
        for t in null_tokens {
            if is_token_equal(inner, t) {
                return true
            }
        }
        return false
    }
    // Unquoted: compare trimmed value to tokens
    for t in null_tokens {
        if is_token_equal(trimmed, t) {
            return true
        }
    }
    return false
}
