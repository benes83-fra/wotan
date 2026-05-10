package core

import "core:fmt"
import "core:mem"
import "core:strings"


dataframe_pretty_print :: proc(df: ^DataFrame, max_rows: int = 20) {
	if df.rows == 0 || len(df.columns) == 0 {
		fmt.println("Empty DataFrame")
		return
	}

	en := df.rows
	if max_rows > 0 && en > max_rows {
		en = max_rows
	}

	// compute column widths
	col_widths := make([]int, len(df.columns))
	defer delete(col_widths)

	for &col, i in df.columns {
		w := len(col.name)
		for r in 0 ..< en {
			val_str := column_value_string_df(df, &col, r)
			if len(val_str) > w {
				w = len(val_str)
			}
		}
		col_widths[i] = w
	}

	// header
	fmt.print("    ")
	for col, i in df.columns {
		name_padded := pad_right(col.name, col_widths[i])
		fmt.printf("%s  ", name_padded)
	}
	fmt.println()

	// rows
	for r in 0 ..< en {
		phys := pretty_row_label(df, r)
		fmt.printf("%3d ", phys)

		for &col, i in df.columns {
			val_str := column_value_string_df(df, &col, r)
			val_padded := pad_right(val_str, col_widths[i])
			fmt.printf("%s  ", val_padded)
		}
		fmt.println()
	}

	if en < df.rows {
		fmt.printf("... (%d more rows)\n", df.rows - en)
	}
}


pretty_row_label :: proc(df: ^DataFrame, r: int) -> int {
	if df.index != nil {
		return df.index[r]
	}

	// contiguous view?
	// any column will do; they all share the same offset
	if len(df.columns) > 0 && df.columns[0].is_view {
		return df.columns[0].offset + r
	}

	// materialized
	return r
}


column_value_string_df :: proc(df: ^DataFrame, col: ^Column, r: int) -> string {
	#partial switch col.type {
	case .Int:
		v := col_get_df(df, col, r, int)
		return fmt.tprintf("%v", v)

	case .Float:
		v := col_get_df(df, col, r, f64)
		return fmt.tprintf("%v", v)

	case .Bool:
		v := col_get_df(df, col, r, bool)
		return fmt.tprintf("%v", v)

	case .String:
		v := col_get_df(df, col, r, string)
		return v

	case .Date:
		v := col_get_df(df, col, r, Date)
		return fmt.tprintf("%04d-%02d-%02d", v.year, v.month, v.day)

	case .Time:
		v := col_get_df(df, col, r, Time)
		return fmt.tprintf("%02d:%02d:%02d", v.hour, v.minute, v.second)

	case .Datetime:
		v := col_get_df(df, col, r, Datetime)
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
