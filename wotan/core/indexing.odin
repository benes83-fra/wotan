package core

// ------------------------------------------------------------
// Indexing
// ------------------------------------------------------------

// Make `colname` the index: rows are reordered by it (stable, nulls last) so
// loc_* can binary-search. drop=true removes the column from the data and
// keeps its values as the hidden index (df.index_values).
set_index :: proc(df: ^DataFrame, colname: string, drop: bool = false) {
	if !(colname in df.name_to_index) {
		panic("set_index: no such column")
	}
	col_idx := df.name_to_index[colname]
	key_col := &df.columns[col_idx]
	if key_col.type == .Invalid {
		panic("set_index: unsupported index type")
	}

	// 1. stable order by the key column
	order := column_sort_order(key_col, false, context.temp_allocator)
	defer delete(order, context.temp_allocator)

	// 2. reorder every column (owned copies: strings cloned, nulls kept)
	if df.index_values.type != .Invalid { 	// an earlier hidden index follows the rows
		reordered := column_take(
			&df.index_values,
			order,
			df.columns[0].allocator if len(df.columns) > 0 else context.allocator,
		)
		destroy_column(&df.index_values)
		df.index_values = reordered
	}
	for &c in df.columns {
		reorder_column_inplace(&c, order)
	}
	if df.index != nil { 	// row labels no longer apply to the new order
		delete(df.index)
		df.index = nil
	}

	// 3. drop: the column becomes the hidden index
	if drop {
		destroy_column(&df.index_values)
		df.index_values = df.columns[col_idx]
		ordered_remove(&df.columns, col_idx)
		_rebuild_name_index(df)
	}

	df.index_column = colname
	df.has_index = true
}

@(private = "file")
_rebuild_name_index :: proc(df: ^DataFrame) {
	if df.name_to_index != nil {
		delete(df.name_to_index)
	}
	df.name_to_index = make(map[string]int)
	for c, i in df.columns {
		df.name_to_index[c.name] = i
	}
}


// Owned copy of a DataFrame (views materialized, index metadata kept)
materialize :: proc(df: ^DataFrame) -> DataFrame {
	rows := make([]int, df.rows, context.temp_allocator)
	defer delete(rows, context.temp_allocator)
	for i in 0 ..< df.rows {rows[i] = i}
	out := df_take(df, rows)
	if out.index != nil { 	// a materialized frame is numbered afresh
		delete(out.index)
		out.index = nil
	}
	return out
}


// Clear the index. drop=false: the index becomes the first column again if it
// was dropped by set_index (a column that is still present stays where it is);
// drop=true: the index values are removed from the frame.
reset_index :: proc(df: ^DataFrame, drop: bool = false) {
	if !df.has_index {
		return
	}
	name := df.index_column
	idx, present := df.name_to_index[name]

	if drop {
		if present {
			destroy_column(&df.columns[idx])
			ordered_remove(&df.columns, idx)
		}
	} else if !present && df.index_values.type != .Invalid {
		restored := column_clone(&df.index_values)
		insert_column_front(df, restored)
	}
	destroy_column(&df.index_values)
	df.index_values = {}
	_rebuild_name_index(df)

	df.index_column = ""
	df.has_index = false
}

// Rows for the given Date keys, in that order; keys not in the index give a
// row of nulls (the index column itself holds the requested key)
reindex_date :: proc(df: ^DataFrame, new_index: []Date) -> DataFrame {
	if !df.has_index {
		panic("reindex: DataFrame has no index")
	}
	idx_col: ^Column
	if i, ok := df.name_to_index[df.index_column]; ok {
		idx_col = &df.columns[i]
	} else {
		idx_col = &df.index_values
	}
	if idx_col.type != .Date {
		panic("reindex_date: index column is not Date")
	}

	// Date -> first row with that key
	lookup := make(map[Date]int)
	defer delete(lookup)
	for i := df.rows - 1; i >= 0; i -= 1 {
		if d, null := column_at_date(idx_col, i); !null {
			lookup[d] = i
		}
	}

	mapping := make([]int, len(new_index))
	defer delete(mapping)
	for d, i in new_index {
		mapping[i] = lookup[d] if d in lookup else -1
	}

	out := dataframe_new(context.allocator)
	for &col in df.columns {
		if col.name == df.index_column { 	// the index column shows the requested keys
			keys := column_new(col.name, .Date, len(new_index))
			for d in new_index {append_date(&keys, d)}
			add_column(&out, keys)
			continue
		}
		add_column(&out, column_take(&col, mapping))
	}
	out.rows = len(new_index)
	out.index_column = df.index_column
	out.has_index = true
	if df.index_values.type != .Invalid {
		out.index_values = column_new(df.index_column, .Date, len(new_index))
		for d in new_index {append_date(&out.index_values, d)}
	}
	return out
}

// Overload entry point
reindex :: proc {
	reindex_date,
}
