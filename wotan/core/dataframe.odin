package wotan

import "core:fmt"
import "core:strings"

DataFrame :: struct {
	columns:       [dynamic]Column,
	name_to_index: map[string]int,
	rows:          int,
}

//
// --- Constructors ---
//

dataframe_new :: proc() -> DataFrame {
	return DataFrame {
		columns = make([dynamic]Column),
		name_to_index = make(map[string]int),
		rows = 0,
	}
}
//DataFrame destructor for cleanup
destroy_dataframe :: proc(df: ^DataFrame) {
	for &col in df.columns {
		destroy_column(&col)

	}
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

df_series :: proc(df: ^DataFrame, name: string) -> Series {
	col := column(df, name)
	return Series {
		name = col.name,
		type = col.type,
		len = col.len,
		data = col.data,
		nulls = col.nulls,
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
			if col.is_view {
				v, is_null := column_at_int(&col, i)
				if is_null {
					fmt.print("NULL\t")
				} else {
					fmt.printf("%v\t", v)
				}
				continue
			}
			base := uintptr(col.data) + uintptr(i * type_size(col.type))

			if col.nulls != nil && col.nulls[i] {
				fmt.print("NULL\t")
				continue
			}

			#partial switch col.type {
			case .Int:
				fmt.printf("%v\t", (cast(^int)base)^)
			case .Float:
				fmt.printf("%v\t", (cast(^f64)base)^)
			case .Bool:
				fmt.printf("%v\t", (cast(^bool)base)^)
			case .String:
				fmt.printf("%v\t", (cast(^string)base)^)
			case .Date:
				d := (cast(^Date)base)^
				fmt.printf("%04d-%02d-%02d\t", d.year, d.month, d.day)
			}
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
			new_col := column_new(col.name, col.type, col.len)
			for i in 0 ..< col.len {
				#partial switch col.type {
				case .Int:
					v, is_null := column_at_int(col, i)
					if is_null {append_null(&new_col)} else {append_int(&new_col, v)}
				case .Float:
					v, is_null := column_at_float(col, i)
					if is_null {append_null(&new_col)} else {append_float(&new_col, v)}
				case .Bool:
					v, is_null := column_at_bool(col, i)
					if is_null {append_null(&new_col)} else {append_bool(&new_col, v)}
				case .String:
					v, is_null := column_at_string(col, i)
					if is_null {append_null(&new_col)} else {append_string(&new_col, v)}
				case .Date:
					v, is_null := column_at_date(col, i)
					if is_null {append_null(&new_col)} else {append_date(&new_col, v)}
				}
			}
			add_column(&out, new_col)
		} else {
			// view: full-row slice of this column
			view_col := column_slice_view(col, 0, col.len)
			add_column(&out, view_col)
		}
	}

	return out
}


dataframe_pretty_print :: proc(df: ^DataFrame, max_rows: int = 20) {
	if df.rows == 0 || len(df.columns) == 0 {
		fmt.println("Empty DataFrame")
		return
	}

	// 1) Determine row range
	en := df.rows
	if max_rows > 0 && en > max_rows {
		en = max_rows
	}

	// 2) Compute column widths: max(len(name), len(value))
	col_widths := make([]int, len(df.columns))
	defer delete(col_widths)

	for &col, i in df.columns {
		w := len(col.name)
		// sample all printed rows for width
		for r in 0 ..< en {
			val_str := column_value_string(&col, r)
			if len(val_str) > w {
				w = len(val_str)
			}
		}
		col_widths[i] = w
	}

	// 3) Print header
	fmt.print("    ") // row index column
	for col, i in df.columns {
		name_padded := pad_right(col.name, col_widths[i])
		fmt.printf("%s  ", name_padded)
	}
	fmt.println()

	// 4) Print rows
	for r in 0 ..< en {
		fmt.printf("%3d ", r)
		for &col, i in df.columns {
			val_str := column_value_string(&col, r)
			val_padded := pad_right(val_str, col_widths[i])
			fmt.printf("%s  ", val_padded)
		}
		fmt.println()
	}

	if en < df.rows {
		fmt.printf("... (%d more rows)\n", df.rows - en)
	}
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
	}

	return "<?>"
}

// Simple right-padding helper
pad_right :: proc(s: string, width: int) -> string {
	if len(s) >= width {
		return s
	}
	rep := strings.repeat(" ", width - len(s))
	defer delete(rep)
	ret := fmt.tprintf("%s%s", s, rep)
	return ret
}

dataframe_filter_bool_column :: proc(df: ^DataFrame, mask_col_name: string) -> DataFrame {
	// 1) Lookup mask column
	mask_col := column(df, mask_col_name)
	if mask_col.type != .Bool {
		panic("dataframe_filter_bool_column: mask column must be Bool")
	}

	// 2) Collect matching row indices
	idxs := make([dynamic]int, 0, df.rows)
	for i in 0 ..< df.rows {
		v, is_null := column_at_bool(mask_col, i)
		if !is_null && v {
			_, err := append(&idxs, i)
			if err != nil {
				panic("dataframe_filter_bool_column: append idx failed")
			}
		}
	}

	// 3) Build output DataFrame with copied rows
	out := dataframe_new()

	for &col in df.columns {
		new_col := column_new(col.name, col.type, len(idxs))

		for idx in idxs {
			#partial switch col.type {
			case .Int:
				v, is_null := column_at_int(&col, idx)
				if is_null {append_null(&new_col)} else {append_int(&new_col, v)}

			case .Float:
				v, is_null := column_at_float(&col, idx)
				if is_null {append_null(&new_col)} else {append_float(&new_col, v)}

			case .Bool:
				v, is_null := column_at_bool(&col, idx)
				if is_null {append_null(&new_col)} else {append_bool(&new_col, v)}

			case .String:
				v, is_null := column_at_string(&col, idx)
				if is_null {append_null(&new_col)} else {append_string(&new_col, v)}

			case .Date:
				v, is_null := column_at_date(&col, idx)
				if is_null {append_null(&new_col)} else {append_date(&new_col, v)}
			}
		}

		add_column(&out, new_col)
	}

	delete(idxs)
	return out
}
filter :: proc {
  dataframe_filter_by_bool_mask,
  dataframe_filter_column_by_col_name,
  dataframe_filter_column,
}

dataframe_filter_by_bool_mask :: proc(df: ^DataFrame, mask: []bool) -> DataFrame {
	if len(mask) != df.rows {
		panic("dataframe_filter: mask length mismatch")
	}

	// Collect indices of rows to keep
	idxs := make([dynamic]int, 0, df.rows)
	for i in 0 ..< df.rows {
		if mask[i] {
			_, err := append(&idxs, i)
			if err != nil {
				panic("dataframe_filter: append failed")
			}
		}
	}
	out := dataframe_new()

	// Build filtered columns
	for &col in df.columns {
		new_col := column_new(col.name, col.type, len(idxs))

		for idx in idxs {
			#partial switch col.type {
			case .Int:
				v, is_null := column_at_int(&col, idx)
				if is_null {append_null(&new_col)} else {append_int(&new_col, v)}

			case .Float:
				v, is_null := column_at_float(&col, idx)
				if is_null {append_null(&new_col)} else {append_float(&new_col, v)}

			case .Bool:
				v, is_null := column_at_bool(&col, idx)
				if is_null {append_null(&new_col)} else {append_bool(&new_col, v)}

			case .String:
				v, is_null := column_at_string(&col, idx)
				if is_null {append_null(&new_col)} else {append_string(&new_col, v)}

			case .Date:
				v, is_null := column_at_date(&col, idx)
				if is_null {append_null(&new_col)} else {append_date(&new_col, v)}
			}
		}

		add_column(&out, new_col)
	}
	delete(idxs)
	return out
}


dataframe_filter_column_by_col_name :: proc(df: ^DataFrame, col_name: string) -> DataFrame {
	col := column(df, col_name)
	if col.type != .Bool {
		panic("dataframe_filter_column: column must be Bool")
	}

	mask := make([]bool, df.rows)
	for i in 0 ..< df.rows {
		v, is_null := column_at_bool(col, i)
		mask[i] = (!is_null && v)
	}
	out := dataframe_filter_by_bool_mask(df, mask)
	delete(mask)
	return out
}

dataframe_filter_column :: proc(df: ^DataFrame, mask_col: Column) -> DataFrame {
	col := mask_col
  defer destroy_column(&col)
	if col.type != .Bool {
		panic("dataframe_filter_column: column must be Bool")
	}

	mask := make([]bool, df.rows)
	for i in 0 ..< df.rows {
		v, is_null := column_at_bool(&col, i)
		mask[i] = (!is_null && v)
	}
	out := dataframe_filter_by_bool_mask(df, mask)
	delete(mask)
	return out
}


dataframe_filter_series :: proc(df: ^DataFrame, s: Series) -> DataFrame {
	s := s
	if s.type != .Bool {
		panic("dataframe_filter_series: series must be Bool")
	}

	mask := make([]bool, s.len)
	for i in 0 ..< s.len {
		v, is_null := series_at_bool(&s, i)
		mask[i] = (!is_null && v)
	}

	out := filter(df, mask)
	delete(mask)
	return out
}
