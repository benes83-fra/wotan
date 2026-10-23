package core

import "core:fmt"
import "core:mem"
import vmem "core:mem/virtual"
import "core:strings"

DataFrame :: struct {
	columns:       [dynamic]Column,
	name_to_index: map[string]int,
	index:         []int, // row labels for printing (source row numbers); not used for data access
	rows:          int,
	index_column:  string, // name of the index column
	has_index:     bool,
	index_values:  Column, // index keys after set_index(drop=true) removed the column
}

//
// --- Constructors ---
//

dataframe_new :: proc(allocator: mem.Allocator = context.allocator) -> DataFrame {
	return DataFrame {
		columns = make([dynamic]Column, allocator),
		name_to_index = make(map[string]int, allocator),
		rows = 0,
	}
}
//DataFrame destructor for cleanup
destroy_dataframe :: proc(df: ^DataFrame) {
	for &col in df.columns {
		destroy_column(&col)

	}
	if df.index != nil {
		delete(df.index)
	}
	destroy_column(&df.index_values)
	df.index_values = {}

	if df.columns != nil {
		delete(df.columns)
	}
	if df.name_to_index != nil {
		delete(df.name_to_index)
	}
	df.columns = nil
	df.name_to_index = nil
	df.rows = 0
}

//
// --- Column management ---
//

add_column :: proc(df: ^DataFrame, col: Column) {
	if len(df.columns) == 0 {
		df.rows = col.len
	} else if col.len != df.rows {
		panic("DataFrame.add_column: column length mismatch")
	}

	if col.name in df.name_to_index {
		panic("DataFrame.add_column: duplicate column name")
	}

	df.name_to_index[col.name] = len(df.columns)
	_, err := append(&df.columns, col)
	if err != nil {
		panic("DataFrame.add_column: append failed")
	}
}

column :: proc(df: ^DataFrame, name: string) -> ^Column {
	idx, ok := df.name_to_index[name]
	if !ok {
		panic(fmt.tprintf("DataFrame.column: no such column '%s'", name))
	}
	return &df.columns[idx]
}

// Series over a column's storage (views resolved to their rows of the
// physical column; no copy)
df_series :: proc(df: ^DataFrame, name: string) -> Series {
	col := column(df, name)
	base, offset := col, 0
	for base.is_view {
		offset += base.offset
		base = base.orig
	}
	nulls: []bool
	if base.nulls != nil && len(base.nulls) >= offset + col.len {
		nulls = base.nulls[offset:offset + col.len]
	}
	return Series {
		name = col.name,
		type = col.type,
		len = col.len,
		data = rawptr(uintptr(base.data) + uintptr(offset * type_size(col.type))),
		nulls = nulls,
	}
}


//
// --- Debug printing ---
//

dataframe_print :: proc(df: ^DataFrame) {
	fmt.println("DataFrame:")
	fmt.printf("  rows: %d\n", df.rows)
	fmt.printf("  columns: %d\n", len(df.columns))

	for col in df.columns {
		fmt.printf("    %s (%v)\n", col.name, col.type)
	}
}


df_head :: proc(df: ^DataFrame, n: int) {
	en := n
	if en > df.rows {
		en = df.rows
	}

	// Print header
	fmt.print("    ")
	for col in df.columns {
		fmt.printf("%s\t", col.name)
	}
	fmt.println()

	// Print rows
	for i in 0 ..< en {
		fmt.printf("%3d ", i)

		for &col in df.columns {
			fmt.printf("%s\t", column_value_string(&col, i))
		}

		fmt.println()
	}
}


dataframe_select_columns :: proc(df: ^DataFrame, names: []string, copy: bool) -> DataFrame {
	out := dataframe_new()

	for name in names {
		col := column(df, name) // existing lookup by name

		if copy {
			// deep copy column
			add_column(&out, column_clone(col))
		} else {
			// view: full-row slice of this column
			view_col := column_slice_view(col, 0, col.len)
			add_column(&out, view_col)
		}
	}

	return out
}


// Helper: convert a cell to string, view-aware via column_* readers
column_value_string :: proc(col: ^Column, row: int) -> string {
	#partial switch col.type {
	case .Int:
		v, is_null := column_at_int(col, row)
		if is_null {return "NULL"}
		return fmt.tprintf("%v", v)

	case .Float:
		v, is_null := column_at_float(col, row)
		if is_null {return "NULL"}
		return fmt.tprintf("%v", v)

	case .Bool:
		v, is_null := column_at_bool(col, row)
		if is_null {return "NULL"}
		return fmt.tprintf("%v", v)

	case .String:
		v, is_null := column_at_string(col, row)
		if is_null {return "NULL"}
		return v

	case .Date:
		v, is_null := column_at_date(col, row)
		if is_null {return "NULL"}
		return fmt.tprintf("%04d-%02d-%02d", v.year, v.month, v.day)
	case .Time:
		v, is_null := column_at_time(col, row)
		if is_null {return "NULL"}
		return fmt.tprintf("%02d:%02d:%02d", v.hour, v.minute, v.second)
	case .Datetime:
		v, is_null := column_at_datetime(col, row)
		if is_null {return "NULL"}

		return fmt.tprintf(
			"%04d-%02d-%02d %02d:%02d:%02d",
			v.year,
			v.month,
			v.day,
			v.hour,
			v.minute,
			v.second,
		)
	}

	return "<?>"
}


df_from :: proc(cols: ..Column, allocator: mem.Allocator = context.allocator) -> DataFrame {
	df := dataframe_new(allocator)
	rows := -1

	for col in cols {

		// Determine row count
		if rows == -1 {
			rows = col.len
		} else {
			assert(col.len == rows, "All columns must have same length")
		}


		add_column(&df, col)
	}

	if rows >= 0 {
		df.rows = rows
	}

	return df
}
