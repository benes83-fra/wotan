package io

import "core:fmt"
import "core:strings"
import "core:os"
import "core:strconv"
import w "../core"


csv_load :: proc(path: string, types: []w.ColumnType) -> w.DataFrame {
    file, err := os.open(path)
    if err != nil {
        panic(fmt.tprintf("csv_load: cannot open file '%s'", path))
    }
    defer os.close(file)

    contents, ok := os.read_entire_file(file)
    if !ok{
        panic("csv_load: failed to read file")
    }

    text := string(contents)
    lines := strings.split(text, "\n")

    if len(lines) == 0 {
        panic("csv_load: empty file")
    }

    // Parse header
    header := strings.split(strings.trim(lines[0], "\r"), ",")
    if len(header) != len(types) {
        panic("csv_load: header/type count mismatch")
    }

    df := w.dataframe_new()

    // Create columns
    cols := make([]w.Column, len(types))
    for i in 0..<len(types) {
        cols[i] = w.column_new(header[i], types[i], len(lines))
    }
    
    // Parse rows
    for row_i in 1..<len(lines) {
        line := strings.trim(lines[row_i], "\r")
        if line == "" {
            continue
        }

        fields := strings.split(line, ",")
        if len(fields) != len(types) {
            panic(fmt.tprintf("csv_load: row %d has wrong number of fields", row_i))
        }

        for col_i in 0..<len(types) {
            field := fields[col_i]

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
