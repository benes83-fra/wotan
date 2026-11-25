package core

// Checked variants of the core entry points that panic on a bad name, key,
// position or type. Each try_ procedure checks every condition under which
// the original panics, returns ok = false instead, and otherwise calls the
// original, so the results are identical. Use them where the input comes from
// outside the program (file headers, user settings, configuration), so that a
// bad name ends in an error you can report rather than a crash.
//
//   try_column        column             unknown name
//   try_add_column    add_column         length mismatch, name taken
//   try_set_index     set_index          unknown column
//   try_loc           loc / loc_single   no index, wrong key type, key not found
//   try_iloc          iloc_single        position out of range
//   try_iloc_many     iloc_many          a position out of range
//   try_select        select (names)     unknown name
//   try_sort          dataframe_sort_by  unknown name
//   try_join          join               unknown key, no keys
//   try_join_on       join_on            unknown key, key lists of different length
//   try_groupby       groupby2 + groupby2_agg (one call): unknown key or column,
//                     a missing second column, quantile or alpha out of range
//
// For masks, `compare` (column_ops.odin) is the checked counterpart of the
// mask_gt / mask_lt / mask_eq family.

// The column named `name`
try_column :: proc(df: ^DataFrame, name: string) -> (^Column, bool) {
	idx, found := df.name_to_index[name]
	if !found {return nil, false}
	return &df.columns[idx], true
}

// Adds `col` to `df` (which then owns it). On false, `col` still belongs to
// the caller.
try_add_column :: proc(df: ^DataFrame, col: Column) -> bool {
	if len(df.columns) > 0 && col.len != df.rows {return false}
	if col.name in df.name_to_index {return false}
	add_column(df, col)
	return true
}

// set_index(df, name, drop)
try_set_index :: proc(df: ^DataFrame, name: string, drop := false) -> bool {
	idx, found := df.name_to_index[name]
	if !found || df.columns[idx].type == .Invalid {return false}
	set_index(df, name, drop)
	return true
}

// The first row whose index key equals `key`, as loc_single returns it
try_loc :: proc(df: ^DataFrame, key: $T) -> (DataFrame, bool) {
	if !_index_usable(df, T) {return {}, false}
	row, found := _loc_find(df, key)
	if !found {return {}, false}
	return df_row(df, row), true
}

// Row i as a one-row frame, as iloc_single returns it
try_iloc :: proc(df: ^DataFrame, i: int) -> (DataFrame, bool) {
	if i < 0 || i >= df.rows {return {}, false}
	return iloc_single(df, i), true
}

// The rows at positions `idxs`, as iloc_many returns them
try_iloc_many :: proc(df: ^DataFrame, idxs: []int) -> (DataFrame, bool) {
	for i in idxs {
		if i < 0 || i >= df.rows {return {}, false}
	}
	return iloc_many(df, idxs), true
}

// View of the named columns, as select(df, names) returns it
try_select :: proc(df: ^DataFrame, names: []string) -> (DataFrame, bool) {
	if !_all_known(df, names) {return {}, false}
	return select_columns(df, names), true
}

// Copy of `df` sorted by `columns`, as dataframe_sort_by returns it
try_sort :: proc(
	df: ^DataFrame,
	columns: []string,
	descending: []bool = nil,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	if !_all_known(df, columns) {return {}, false}
	return dataframe_sort_by(df, columns, descending, allocator), true
}

// join(left, right, keys, kind)
try_join :: proc(
	left, right: ^DataFrame,
	keys: []string,
	kind := JoinKind.Inner,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	return try_join_on(left, right, keys, keys, kind, allocator)
}

// join_on(left, right, left_keys, right_keys, kind)
try_join_on :: proc(
	left, right: ^DataFrame,
	left_keys, right_keys: []string,
	kind := JoinKind.Inner,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	if len(left_keys) == 0 || len(left_keys) != len(right_keys) {return {}, false}
	if !_all_known(left, left_keys) || !_all_known(right, right_keys) {return {}, false}
	return join_on(left, right, left_keys, right_keys, kind, allocator), true
}

// groupby2(df, keys) followed by groupby2_agg(aggs), as one call: one row per
// group with the key columns and one column per aggregator
try_groupby :: proc(
	df: ^DataFrame,
	keys: []string,
	aggs: []Aggregator,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	if len(keys) == 0 || !_all_known(df, keys) {return {}, false}
	for a in aggs {
		if !(a.column in df.name_to_index) {return {}, false}
		#partial switch a.kind {
		case .Cov, .Corr, .EWM_Cov, .EWM_Corr:
			if !(a.other in df.name_to_index) {return {}, false}
		case .Quantile:
			if !(a.quantile >= 0 && a.quantile <= 1) {return {}, false}
		}
		#partial switch a.kind {
		case .EWM_Mean, .EWM_Var, .EWM_Std, .EWM_Cov, .EWM_Corr:
			if !(a.alpha > 0 && a.alpha <= 1) {return {}, false}
		}
	}
	gb := groupby2(df, keys, allocator)
	defer destroy_groupby2(&gb)
	return groupby2_agg(&gb, aggs, allocator), true
}

@(private = "file")
_all_known :: proc(df: ^DataFrame, names: []string) -> bool {
	for n in names {
		if !(n in df.name_to_index) {return false}
	}
	return true
}
