package core

import "base:intrinsics"
import "core:mem"
import "core:sort"

// ------------------------------------------------------------
// Indexing
// ------------------------------------------------------------

set_index :: proc(df: ^DataFrame, colname: string) {
	if !(colname in df.name_to_index) {
		panic("set_index: no such column")
	}

	col_idx := df.name_to_index[colname]
	col := &df.columns[col_idx]

	// 1. Build index vector [0,1,2,...]
	index_vec := make([]int, df.rows, context.allocator)
	for i in 0 ..< df.rows {
		index_vec[i] = i
	}

	// 2. Sort index_vec using your existing sorter system
	#partial switch col.type {

	case .Int:
		sorter := IntSorter {
			idx        = &index_vec,
			data       = cast([^]int)col.data,
			descending = false,
		}
		sort.sort(make_int_sorter(&sorter))

	case .Float:
		sorter := FloatSorter {
			idx        = &index_vec,
			data       = cast([^]f64)col.data,
			descending = false,
		}
		sort.sort(make_float_sorter(&sorter))

	case .String:
		sorter := StringSorter {
			idx        = &index_vec,
			data       = cast([^]string)col.data,
			descending = false,
		}
		sort.sort(make_string_sorter(&sorter))

	case .Bool:
		sorter := BoolSorter {
			idx        = &index_vec,
			data       = cast([^]bool)col.data,
			descending = false,
		}
		sort.sort(make_bool_sorter(&sorter))

	case .Date:
		sorter := DateSorter {
			idx        = &index_vec,
			data       = cast([^]Date)col.data,
			descending = false,
		}
		sort.sort(make_date_sorter(&sorter))

	case .Time:
		sorter := TimeSorter {
			idx        = &index_vec,
			data       = cast([^]Time)col.data,
			descending = false,
		}
		sort.sort(make_time_sorter(&sorter))

	case .Datetime:
		sorter := DatetimeSorter {
			idx        = &index_vec,
			data       = cast([^]Datetime)col.data,
			descending = false,
		}
		sort.sort(make_datetime_sorter(&sorter))

	case:
		panic("set_index: unsupported index type")
	}

	// 3. Reorder all columns in-place
	for &c in df.columns {
		reorder_column_inplace(&c, index_vec)
	}

	// 4. Mark index metadata
	df.index_column = colname
	df.has_index = true

	delete(index_vec)
}


materialize :: proc(df: ^DataFrame) -> DataFrame {
	out := dataframe_new(context.allocator)
	out.rows = df.rows
	out.index = nil
	out.index_column = df.index_column
	out.has_index = df.has_index

	// logical indices 0..rows-1
	indices := make([]int, out.rows)
	for i in 0 ..< out.rows do indices[i] = i

	for &col in df.columns {
		dst := column_new(col.name, col.type, out.rows, context.allocator)

		// this internally uses col_get_df(df, col, indices[i], T)
		copy_column_values(df, &col, &dst, indices)

		dst.is_view = false
		dst.orig = nil
		dst.offset = 0
		dst.len = out.rows
		dst.capacity = out.rows

		add_column(&out, dst)
	}

	delete(indices)
	return out
}


reset_index :: proc(df: ^DataFrame) {
	if !df.has_index {
		return
	}

	// 1. Extract the index column name
	name := df.index_column
	idx := df.name_to_index[name]
	col := df.columns[idx]

	// 2. Create a new column with the same values
	new_col := column_new(name, col.type, df.rows, context.allocator)
	mapping := make([]int, df.rows)
	defer delete(mapping)
	for i in 0 ..< df.rows do mapping[i] = i
	copy_column_direct(&col, &new_col, mapping)


	new_col.len = df.rows

	// 3. Insert new column at front
	insert_column_front(df, new_col)

	// 4. Clear index metadata
	df.index_column = ""
	df.has_index = false
} // Date-specific reindex
reindex_date :: proc(df: ^DataFrame, new_index: []Date) -> DataFrame {
	if !df.has_index {
		panic("reindex: DataFrame has no index")
	}

	// 1. Ensure index column is Date
	idx_col := &df.columns[df.name_to_index[df.index_column]]
	if idx_col.type != .Date {
		panic("reindex_date: index column is not Date")
	}

	// 2. Build lookup: Date -> row index
	lookup := make(map[Date]int)
	src := cast([^]Date)idx_col.data
	for i in 0 ..< df.rows {
		lookup[src[i]] = i
	}

	// 3. Build mapping: logical position -> physical row (or -1 for missing)
	mapping := make([]int, len(new_index))
	for i in 0 ..< len(new_index) {
		if row, ok := lookup[new_index[i]]; ok {
			mapping[i] = row
		} else {
			mapping[i] = -1
		}
	}

	// 4. Create output DataFrame
	out := dataframe_new(context.allocator)
	out.rows = len(new_index)
	out.index_column = df.index_column
	out.has_index = true

	// 5. For each column, build a new column using mapping
	for &col in df.columns {
		dst := column_new(col.name, col.type, out.rows, context.allocator)

		// uses mapping[i] == -1 -> NULL inside
		copy_column_direct_or_null(&col, &dst, mapping)

		dst.len = out.rows
		add_column(&out, dst)
	}

	delete(mapping)
	delete(lookup)
	return out
}

// Overload entry point
reindex :: proc {
	reindex_date,
}
