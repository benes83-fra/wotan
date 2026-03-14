package wotan


import "core:fmt"
import "core:mem"
import vmem "core:mem/virtual"
import "core:strings"

Group :: struct {
	key_values:  []string, // Switched to slice for easier arena management
	row_indices: [dynamic]int,
}

GroupedDataFrame :: struct {
	df:     ^DataFrame,
	keys:   []string,
	groups: [dynamic]Group,
	arena:  vmem.Arena, // Keeps all group-related strings/arrays alive
}

// Helper to format keys into the arena
make_group_key_in_arena :: proc(
	df: ^DataFrame,
	row: int,
	keys: []string,
	allocator: mem.Allocator,
) -> string {
	sb := strings.builder_make(allocator)
	// We don't destroy the builder because its buffer will live in the arena

	for key, i in keys {
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

		if i < len(keys) - 1 {fmt.sbprint(&sb, "|")}
	}
	return strings.to_string(sb)
}

groupby :: proc(df: ^DataFrame, keys: []string) -> GroupedDataFrame {
	gdf := GroupedDataFrame {
		df     = df,
		keys   = keys,
		groups = make([dynamic]Group),
	}

	// Initialize Arena for all string/group data
	arene_err := vmem.arena_init_growing(&gdf.arena, 4000)
	allocator := vmem.arena_allocator(&gdf.arena)

	// Map to track unique keys during construction
	groups_map := make(map[string]int)
	defer delete(groups_map)

	for row in 0 ..< df.rows {
		// Temporary key for map lookup
		temp_key := make_group_key_in_arena(df, row, keys, allocator)

		idx, exists := groups_map[temp_key]
		if !exists {
			// Persist the key and values into the arena
			persistent_key := strings.clone(temp_key, allocator)
			parts := strings.split(persistent_key, "|", allocator)

			g := Group {
				key_values  = parts,
				row_indices = make([dynamic]int, allocator),
			}
			append(&g.row_indices, row)
			append(&gdf.groups, g)

			groups_map[persistent_key] = len(gdf.groups) - 1
		} else {
			append(&gdf.groups[idx].row_indices, row)
		}
	}

	return gdf
}

destroy_grouped_dataframe :: proc(gdf: ^GroupedDataFrame) {
	// 1. Delete the top-level dynamic array
	delete(gdf.groups)
	// 2. Free everything else (strings, row_indices, etc.) in one go
	vmem.arena_destroy(&gdf.arena)
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
