package wotan

ColumnType :: enum {
    Invalid,
    Int,
    Float,
    Bool,
    String,
    Date,
}

Date :: struct {
    year:  i32,
    month: i32,
    day:   i32,
}

type_size :: proc(t: ColumnType) -> int {
    switch t {
    case .Int:    return size_of(int)
    case .Float:  return size_of(f64)
    case .Bool:   return size_of(bool)
    case .String: return size_of(string)
    case .Date:   return size_of(Date)
    case .Invalid:
        return 0
    }
    return 0
}
