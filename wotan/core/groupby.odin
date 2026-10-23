package core


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

// Groups rows by the values of `keys` (groups in order of first appearance;
// null key values form their own group). Keys are compared by type and value.
groupby :: proc(df: ^DataFrame, keys: []string) -> GroupedDataFrame {
	gdf := GroupedDataFrame {
		df     = df,
		keys   = keys,
		groups = make([dynamic]Group),
	}

	// Arena for all group data (key texts, row lists)
	if err := vmem.arena_init_growing(&gdf.arena, 4000); err != nil {
		panic("groupby: cannot allocate arena")
	}
	allocator := vmem.arena_allocator(&gdf.arena)

	key_cols := make([]^Column, len(keys), context.temp_allocator)
	defer delete(key_cols, context.temp_allocator)
	for k, i in keys {key_cols[i] = column(df, k)}

	groups_map := make(map[string]int)
	defer delete(groups_map)

	for row in 0 ..< df.rows {
		key := make_composite_key_string(key_cols, row, context.temp_allocator)
		if idx, exists := groups_map[key]; exists {
			append(&gdf.groups[idx].row_indices, row)
			delete(key, context.temp_allocator)
			continue
		}
		// readable key values (NULL for missing)
		parts := make([]string, len(keys), allocator)
		for col, i in key_cols {
			parts[i] = strings.clone(column_value_string(col, row), allocator)
		}
		g := Group {
			key_values  = parts,
			row_indices = make([dynamic]int, allocator),
		}
		append(&g.row_indices, row)
		append(&gdf.groups, g)
		groups_map[strings.clone(key, allocator)] = len(gdf.groups) - 1
		delete(key, context.temp_allocator)
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
