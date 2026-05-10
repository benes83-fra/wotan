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

	for &col in df.columns {
		dst := column_new(col.name, col.type, out.rows, context.allocator)

		// no nulls handling for now – column_new already gave us all-false nulls

		#partial switch col.type {
		case .Int:
			dst_data := cast([^]int)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = col_get_df(df, &col, i, int)
			}

		case .Float:
			dst_data := cast([^]f64)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = col_get_df(df, &col, i, f64)
			}

		case .String:
			dst_data := cast([^]string)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = col_get_df(df, &col, i, string)
			}

		case .Bool:
			dst_data := cast([^]bool)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = col_get_df(df, &col, i, bool)
			}

		case .Date:
			dst_data := cast([^]Date)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = col_get_df(df, &col, i, Date)
			}

		case .Time:
			dst_data := cast([^]Time)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = col_get_df(df, &col, i, Time)
			}

		case .Datetime:
			dst_data := cast([^]Datetime)dst.data
			for i in 0 ..< out.rows {
				dst_data[i] = col_get_df(df, &col, i, Datetime)
			}
		}

		dst.is_view = false
		dst.orig = nil
		dst.offset = 0
		dst.len = out.rows
		dst.capacity = out.rows

		add_column(&out, dst)
	}

	return out
}
