package core

import "core:fmt"

// Row selection. Contiguous ranges (df_slice, iloc_range, loc_range,
// loc_from, loc_until) return views that read through to the source columns;
// row lists and masks (iloc_many, loc_many, loc_mask) return owned copies.
// Empty selections keep the columns (with zero rows). df.index carries the
// source row labels for printing; it is never used to address data.


loc :: proc {
	loc_single,
	loc_range,
	loc_many,
	loc_mask,
}


// The index keys: the index column, or the hidden copy kept by set_index(drop=true)
@(private = "file")
_index_col :: proc(df: ^DataFrame, label: string) -> ^Column {
	if !df.has_index {
		panic(fmt.tprintf("%s: DataFrame has no index", label))
	}
	if idx, ok := df.name_to_index[df.index_column]; ok {
		return &df.columns[idx]
	}
	if df.index_values.type != .Invalid {
		return &df.index_values
	}
	panic(fmt.tprintf("%s: index column %q is missing", label, df.index_column))
}

// Index keys as a typed slice (temp memory) + the number of leading non-null
// keys (set_index sorts nulls last, so binary searches use that prefix)
@(private = "file")
_index_keys :: proc(df: ^DataFrame, $T: typeid, label: string) -> (keys: []T, n_valid: int, valid: []bool) {
	col := _index_col(df, label)
	if col.type != get_column_type(T) || size_of(T) != type_size(col.type) {
		panic(fmt.tprintf("%s: key type %v does not match index column type %v", label, typeid_of(T), col.type))
	}
	keys = make([]T, df.rows, context.temp_allocator)
	valid = make([]bool, df.rows, context.temp_allocator)
	n_valid = df.rows
	for i in 0 ..< df.rows {
		ptr, null := column_at_ptr(col, i)
		valid[i] = !null
		if null {
			if n_valid == df.rows {n_valid = i}
			continue
		}
		keys[i] = (cast(^T)ptr)^
	}
	return
}

// Row labels of `src` for printing (source row numbers)
@(private = "file")
_row_label :: proc(src: ^DataFrame, r: int) -> int {
	return pretty_row_label(src, r)
}

// Owned copy of the given rows of `src`, with index metadata and labels
df_take :: proc(src: ^DataFrame, rows: []int, allocator := context.allocator) -> DataFrame {
	out := dataframe_new(allocator)
	for &col in src.columns {
		add_column(&out, column_take(&col, rows, allocator))
	}
	out.rows = len(rows)
	out.index_column = src.index_column
	out.has_index = src.has_index
	if src.index_values.type != .Invalid {
		out.index_values = column_take(&src.index_values, rows, allocator)
	}
	if len(rows) > 0 {
		labels := make([]int, len(rows), allocator)
		for r, i in rows {labels[i] = _row_label(src, r) if r >= 0 else -1}
		out.index = labels
	}
	return out
}


// ------------------------------------------------------------
// LOC (scalar): first row whose index equals `key`
// ------------------------------------------------------------

loc_single :: proc(df: ^DataFrame, key: $T) -> DataFrame {
	keys, n_valid, valid := _index_keys(df, T, "loc")

	// 1. binary search (sorted, non-null prefix)
	i := lower_bound(keys[:n_valid], key)
	if i < n_valid && equals(keys[i], key) {
		return df_row(df, i)
	}

	// 2. fallback linear scan (unsorted index)
	for j in 0 ..< df.rows {
		if valid[j] && equals(keys[j], key) {
			return df_row(df, j)
		}
	}

	panic("loc: key not found")
}

// ------------------------------------------------------------
// LOC (range, inclusive): rows with start <= index <= stop
// ------------------------------------------------------------

loc_range :: proc(df: ^DataFrame, start: $T, stop: T) -> DataFrame {
	keys, n_valid, _ := _index_keys(df, T, "loc_range")
	i0 := lower_bound(keys[:n_valid], start)
	i1 := upper_bound(keys[:n_valid], stop)
	return df_slice(df, i0, max(i0, i1))
}

// ------------------------------------------------------------
// Row / slice extraction
// ------------------------------------------------------------

// One row as an owned DataFrame (nulls and strings preserved)
df_row :: proc(df: ^DataFrame, i: int) -> DataFrame {
	if i < 0 || i >= df.rows {
		panic(fmt.tprintf("df_row: row %d out of range 0..<%d", i, df.rows))
	}
	out := dataframe_new(context.allocator)
	rows := []int{i}
	for &col in df.columns {
		add_column(&out, column_take(&col, rows, context.allocator))
	}
	out.rows = 1
	return out
}


// Rows [i0, i1) as views of the source columns (clamped; empty keeps the schema)
df_slice :: proc(df: ^DataFrame, i0, i1: int) -> DataFrame {
	i0 := clamp(i0, 0, df.rows)
	i1 := clamp(i1, i0, df.rows)

	out := dataframe_new(context.allocator)
	for &col in df.columns {
		add_column(&out, column_slice_view(&col, i0, i1))
	}
	out.rows = i1 - i0
	out.index_column = df.index_column
	out.has_index = df.has_index
	if df.index_values.type != .Invalid {
		out.index_values = column_slice_view(&df.index_values, i0, i1)
	}
	if df.index != nil && i1 > i0 {
		labels := make([]int, i1 - i0, context.allocator)
		copy(labels, df.index[i0:i1])
		out.index = labels
	}
	return out
}

// ------------------------------------------------------------
// LOC MANY (list of keys): all rows matching each key, in key order
// ------------------------------------------------------------

loc_many :: proc(df: ^DataFrame, keys: []$T) -> DataFrame {
	data, n_valid, valid := _index_keys(df, T, "loc_many")

	indices := make([dynamic]int, 0, len(keys), context.temp_allocator)
	defer delete(indices)

	for key in keys {
		i0 := lower_bound(data[:n_valid], key)
		i1 := upper_bound(data[:n_valid], key)
		if i1 > i0 && equals(data[i0], key) {
			for i in i0 ..< i1 {append(&indices, i)}
			continue
		}
		for i in 0 ..< df.rows { 	// unsorted index
			if valid[i] && equals(data[i], key) {
				append(&indices, i)
			}
		}
	}
	return df_take(df, indices[:])
}

// ------------------------------------------------------------
// LOC_FROM (open-ended slice from a start label)
// ------------------------------------------------------------

loc_from :: proc(df: ^DataFrame, start: $T) -> DataFrame {
	keys, n_valid, _ := _index_keys(df, T, "loc_from")
	return df_slice(df, lower_bound(keys[:n_valid], start), df.rows)
}

// ------------------------------------------------------------
// LOC_UNTIL (open-ended slice until a stop label, inclusive)
// ------------------------------------------------------------

loc_until :: proc(df: ^DataFrame, stop: $T) -> DataFrame {
	keys, n_valid, _ := _index_keys(df, T, "loc_until")
	return df_slice(df, 0, upper_bound(keys[:n_valid], stop))
}

// ------------------------------------------------------------
// LOC_MASK (boolean mask selection)
// ------------------------------------------------------------

loc_mask :: proc(df: ^DataFrame, mask: []bool) -> DataFrame {
	if len(mask) != df.rows {
		panic("loc_mask: mask length does not match DataFrame rows")
	}
	indices := make([dynamic]int, 0, df.rows, context.temp_allocator)
	defer delete(indices)
	for i in 0 ..< df.rows {
		if mask[i] {
			append(&indices, i)
		}
	}
	return df_take(df, indices[:])
}

// ------------------------------------------------------------
// ILOC (scalar integer indexing)
// ------------------------------------------------------------

iloc :: proc {
	iloc_single,
	iloc_range,
	iloc_many,
}


iloc_single :: proc(df: ^DataFrame, i: int) -> DataFrame {
	if i < 0 || i >= df.rows {
		panic("iloc: index out of bounds")
	}
	return df_row(df, i)
}
// ------------------------------------------------------------
// ILOC_RANGE (integer slice [i0, i1))
// ------------------------------------------------------------

iloc_range :: proc(df: ^DataFrame, i0: int, i1: int) -> DataFrame {
	return df_slice(df, i0, i1)
}
// ------------------------------------------------------------
// ILOC_MANY (list of integer indices)
// ------------------------------------------------------------

iloc_many :: proc(df: ^DataFrame, idxs: []int) -> DataFrame {
	for i in idxs {
		if i < 0 || i >= df.rows {
			panic("iloc_many: index out of bounds")
		}
	}
	return df_take(df, idxs)
}


// Kept for compatibility: rows by position as an owned DataFrame
dataframe_view_indices :: proc(base: ^DataFrame, indices: []int) -> DataFrame {
	return df_take(base, indices)
}

