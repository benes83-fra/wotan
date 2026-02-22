package importer

import "core:strings"
import "core:strconv"
import w "../core"

is_null :: proc(s: string) -> bool {
    ret := strings.trim(s, " ")
    return ret == "" || ret == "NA" || ret == "null" || ret == "None"
}

is_int :: proc(s: string) -> bool {
    _, err := strconv.parse_int(s)
    return err
}

is_float :: proc(s: string) -> bool {
    _, err := strconv.parse_f64(s)
    return err
}

is_bool :: proc(s: string) -> bool {
    return s == "true" || s == "false"
}

is_date :: proc(s: string) -> bool {
    parts := strings.split(s, "-")
    if len(parts) != 3 {
        return false
    }
    _, err1 := strconv.parse_int(parts[0])
    _, err2 := strconv.parse_int(parts[1])
    _, err3 := strconv.parse_int(parts[2])
    return err1 && err2  && err3 
}

infer_column_type :: proc(samples: []string) -> w.ColumnType {
    has_string := false
    has_float  := false
    has_int    := false
    has_bool   := false
    has_date   := false

    for s in samples {
        if is_null(s) {
            continue
        }
        if is_int(s) {
            has_int = true
            continue
        }
        if is_float(s) {
            has_float = true
            continue
        }
        if is_bool(s) {
            has_bool = true
            continue
        }
        if is_date(s) {
            has_date = true
            continue
        }
        has_string = true
    }

    if has_string { return .String }
    if has_float  { return .Float }
    if has_int    { return .Int }
    if has_bool   { return .Bool }
    if has_date   { return .Date }

    return .String
}
