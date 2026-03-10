package wotan

import "core:fmt"
import "core:strings"

// A single group: key values + row indices
Group :: struct {
	key_values:  [dynamic]string,
	row_indices: [dynamic]int,
}

// The result of groupby()
GroupedDataFrame :: struct {
	df:     ^DataFrame,
	keys:   []string,
	groups: [dynamic]Group,
}

// -----------------------------------------------------------------------------
// Helper: stringify a row's key columns into a stable composite key
// -----------------------------------------------------------------------------
make_group_key :: proc(df: ^DataFrame, row: int, keys: []string) -> string {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)

	for key, i in keys { 	// <-- FIXED ORDER
		col := column(df, key)

		#partial switch col.type {
		case .Int:
			v, n := column_at_int(col, row)
			if n {fmt.sbprintf(&sb, "NULL")} else {fmt.sbprintf(&sb, "%d", v)}

		case .Float:
			v, n := column_at_float(col, row)
			if n {fmt.sbprintf(&sb, "NULL")} else {fmt.sbprintf(&sb, "%g", v)}

		case .Bool:
			v, n := column_at_bool(col, row)
			if n {fmt.sbprintf(&sb, "NULL")} else {fmt.sbprintf(&sb, "%v", v)}

		case .String:
			v, n := column_at_string(col, row)
			if n {fmt.sbprintf(&sb, "NULL")} else {fmt.sbprintf(&sb, "%s", v)}

		case .Date:
			v, n := column_at_date(col, row)
			if n {fmt.sbprintf(&sb, "NULL")} else {fmt.sbprintf(&sb, "%s", date_to_string(v))}

		case .Time:
			v, n := column_at_time(col, row)
			if n {fmt.sbprintf(&sb, "NULL")} else {fmt.sbprintf(&sb, "%s", time_to_string(v))}

		case .Datetime:
			v, n := column_at_datetime(col, row)
			if n {fmt.sbprintf(&sb, "NULL")} else {fmt.sbprintf(&sb, "%s", datetime_to_string(v))}
		}

		if i < len(keys) - 1 {
			fmt.sbprintf(&sb, "|")
		}
	}

	return strings.to_string(sb)
}

// -----------------------------------------------------------------------------
// GROUPBY ENGINE
// -----------------------------------------------------------------------------
groupby :: proc(df: ^DataFrame, keys: []string) -> GroupedDataFrame {
	gdf := GroupedDataFrame {
		df     = df,
		keys   = keys,
		groups = make([dynamic]Group, 0, 16),
	}

	groups_map := make(map[string]int)

	for row in 0 ..< df.rows {
		key := make_group_key(df, row, keys)

		idx, exists := groups_map[key]
		if !exists {
			parts := strings.split(key, "|")

			dyn_parts := make([dynamic]string, 0, len(parts))
			for p in parts { 	// <-- FIXED ORDER
				append(&dyn_parts, p)
			}

			g := Group {
				key_values  = dyn_parts,
				row_indices = make([dynamic]int, 0, 8),
			}
			append(&g.row_indices, row)

			append(&gdf.groups, g)
			groups_map[key] = len(gdf.groups) - 1
		} else {
			append(&gdf.groups[idx].row_indices, row)
		}
	}

	return gdf
}

// -----------------------------------------------------------------------------
// Debug helper
// -----------------------------------------------------------------------------
groupby_debug_print :: proc(gdf: ^GroupedDataFrame) {
	fmt.println("Grouped by:", gdf.keys)

	for g, i in gdf.groups {
		fmt.printf("Group %d: keys=%v rows=%v\n", i, g.key_values, g.row_indices)
	}
}
destroy_grouped_dataframe :: proc(gdf: ^GroupedDataFrame) {
	for g in gdf.groups {
		// key_values holds string slices; do NOT delete each string
		delete(g.key_values)
		delete(g.row_indices)
	}

	delete(gdf.groups)
}
