package core

import "core:fmt"
import "core:mem"


loc :: proc {
	loc_single,
	loc_range,
	loc_many,
	loc_mask,
}


// ------------------------------------------------------------
// LOC (scalar)
// ------------------------------------------------------------


loc_single :: proc(df: ^DataFrame, key: $T) -> DataFrame {
	if !df.has_index {
		panic("loc: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	// 1. binary search
	i := binary_search(data, key)
	if i >= 0 {
		return df_row(df, i)
	}

	// 2. fallback linear scan
	for j in 0 ..< df.rows {
		if equals(data[j], key) {
			return df_row(df, j)
		}
	}

	panic("loc: key not found")
}

// ------------------------------------------------------------
// LOC (range)
// ------------------------------------------------------------

loc_range :: proc(df: ^DataFrame, start: $T, stop: T) -> DataFrame {
	if !df.has_index {
		panic("loc_range: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	i0 := lower_bound(data, start)
	i1 := upper_bound(data, stop)

	return df_slice(df, i0, i1)
}

// ------------------------------------------------------------
// Row / slice extraction
// ------------------------------------------------------------

df_row :: proc(df: ^DataFrame, i: int) -> DataFrame {
	out := dataframe_new(context.allocator)
	out.rows = 1

	for &col in df.columns {
		dst := column_new(col.name, col.type, 1, context.allocator)

		#partial switch col.type {
		case .Int:
			dst_data := cast([^]int)dst.data
			dst_data[0] = col_get_df(df, &col, i, int)

		case .Float:
			dst_data := cast([^]f64)dst.data
			dst_data[0] = col_get_df(df, &col, i, f64)

		case .String:
			dst_data := cast([^]string)dst.data
			dst_data[0] = col_get_df(df, &col, i, string)

		case .Bool:
			dst_data := cast([^]bool)dst.data
			dst_data[0] = col_get_df(df, &col, i, bool)

		case .Date:
			dst_data := cast([^]Date)dst.data
			dst_data[0] = col_get_df(df, &col, i, Date)

		case .Time:
			dst_data := cast([^]Time)dst.data
			dst_data[0] = col_get_df(df, &col, i, Time)

		case .Datetime:
			dst_data := cast([^]Datetime)dst.data
			dst_data[0] = col_get_df(df, &col, i, Datetime)
		}

		dst.len = 1
		add_column(&out, dst)
	}

	return out
}


df_slice :: proc(df: ^DataFrame, i0, i1: int) -> DataFrame {
	// normalize bounds
	i0 := i0
	i1 := i1
	if i0 < 0 do i0 = 0
	if i1 > df.rows do i1 = df.rows
	if i1 <= i0 do return dataframe_new(context.allocator)

	out := dataframe_new(context.allocator)
	out.rows = i1 - i0
	out.index_column = df.index_column
	out.has_index = df.has_index

	for &col in df.columns {
		view_col := col

		view_col.is_view = true
		view_col.orig = &col

		view_col.offset = col.offset + i0 if col.is_view else i0
		view_col.len = out.rows
		view_col.capacity = view_col.len

		// data + nulls are shared; arena/allocator unchanged
		// data stays pointing to the original buffer
		if len(col.nulls) > 0 {
			start := view_col.offset
			stop := view_col.offset + view_col.len
			view_col.nulls = col.nulls[start:stop]
		}

		add_column(&out, view_col)
	}

	return out
}
// ------------------------------------------------------------
// LOC MANY (list of keys)
// ------------------------------------------------------------
// ------------------------------------------------------------
// LOC MANY (list of keys)
// ------------------------------------------------------------

loc_many :: proc(df: ^DataFrame, keys: []$T) -> DataFrame {
	if !df.has_index {
		panic("loc_many: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	indices := make([dynamic]int, 0, len(keys), context.allocator)

	for key in keys {
		idx := binary_search(data, key)
		if idx >= 0 {
			append(&indices, idx)
			continue
		}
		for i in 0 ..< df.rows {
			if equals(data[i], key) {
				append(&indices, i)
			}
		}
	}

	if len(indices) == 0 {
		return dataframe_new(context.allocator)
	}

	return dataframe_view_indices(df, indices[:])
}

// ------------------------------------------------------------
// LOC_FROM (open-ended slice from a start label)
// ------------------------------------------------------------

loc_from :: proc(df: ^DataFrame, start: $T) -> DataFrame {
	if !df.has_index {
		panic("loc_from: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	// left boundary
	i0 := lower_bound(data, start)

	// open-ended: until end
	i1 := df.rows

	return df_slice(df, i0, i1)
}
// ------------------------------------------------------------
// LOC_UNTIL (open-ended slice until a stop label)
// ------------------------------------------------------------

loc_until :: proc(df: ^DataFrame, stop: $T) -> DataFrame {
	if !df.has_index {
		panic("loc_until: DataFrame has no index")
	}

	col := column(df, df.index_column)
	data := mem.slice_ptr(cast(^T)col.data, df.rows)

	// open-ended: from beginning
	i0 := 0

	// right boundary
	i1 := upper_bound(data, stop)

	return df_slice(df, i0, i1)
}
// ------------------------------------------------------------
// LOC_MASK (boolean mask selection)
// ------------------------------------------------------------

loc_mask :: proc(df: ^DataFrame, mask: []bool) -> DataFrame {
	if len(mask) != df.rows {
		panic("loc_mask: mask length does not match DataFrame rows")
	}

	indices := make([dynamic]int, 0, df.rows, context.allocator)
	// DO NOT delete(indices) — ownership moves to the view

	for i in 0 ..< df.rows {
		if mask[i] {
			append(&indices, i)
		}
	}

	if len(indices) == 0 {
		return dataframe_new(context.allocator)
	}

	return dataframe_view_indices(df, indices[:])
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
	i0 := i0
	i1 := i1
	if i0 < 0 do i0 = 0
	if i1 > df.rows do i1 = df.rows
	if i1 <= i0 do return dataframe_new(context.allocator)

	return df_slice(df, i0, i1)
}
// ------------------------------------------------------------
// ILOC_MANY (list of integer indices)
// ------------------------------------------------------------

iloc_many :: proc(df: ^DataFrame, idxs: []int) -> DataFrame {
	// Validate indices
	for i in idxs {
		if i < 0 || i >= df.rows {
			panic("iloc_many: index out of bounds")
		}
	}

	if len(idxs) == 0 {
		return dataframe_new(context.allocator)
	}

	// Own a heap-allocated index vector
	indices := make([dynamic]int, len(idxs), len(idxs), context.allocator)
	for i in 0 ..< len(idxs) {
		indices[i] = idxs[i]
	}

	return dataframe_view_indices(df, indices[:])
}


dataframe_view_indices :: proc(base: ^DataFrame, indices: []int) -> DataFrame {
	out := dataframe_new(context.allocator)
	out.rows = len(indices)
	out.index = indices
	out.index_column = base.index_column
	out.has_index = base.has_index

	for i in 0 ..< len(base.columns) {
		base_col := &base.columns[i]
		view_col := base_col^
		view_col.is_view = true
		view_col.orig = base_col
		view_col.offset = 0
		view_col.len = out.rows
		view_col.capacity = out.rows
		add_column(&out, view_col)
	}

	return out
}
