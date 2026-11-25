package core

import "core:math/rand"
import "core:mem"

// Reshaping and sampling:
//
//   melt          wide -> long: one row per (id, value column) pair
//   pivot         long -> wide: one row per index key, one column per column key
//   pivot_table   pivot that aggregates repeated (index, column) pairs
//   resample      (resample.odin) one row per day / week / month / quarter / year
//   sample        random rows
//
// Results are new owned frames. Inputs may be views and are never changed.

// --- melt ---

// Long form of `df`: for every value column (in the given order) and every
// row, one output row with the id columns, a String column `var_name` holding
// the value column's name and a column `value_name` holding its value. With
// no value_vars given, every column that is not an id column is a value
// column. The value columns must share one type, except that Int and Float
// mix into Float. Rows are numbered afresh. ok = false for an unknown column,
// no value columns, mixed types, or an output name that clashes with an id
// column.
melt :: proc(
	df: ^DataFrame,
	id_vars: []string,
	value_vars: []string = nil,
	var_name := "variable",
	value_name := "value",
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	ids := make([dynamic]^Column, 0, len(id_vars), context.temp_allocator)
	defer delete(ids)
	is_id := make([]bool, len(df.columns), context.temp_allocator)
	defer delete(is_id, context.temp_allocator)
	for name in id_vars {
		idx, found := df.name_to_index[name]
		if !found || name == var_name || name == value_name {return {}, false}
		append(&ids, &df.columns[idx])
		is_id[idx] = true
	}
	vals := make([dynamic]^Column, 0, len(df.columns), context.temp_allocator)
	defer delete(vals)
	if len(value_vars) == 0 {
		for &c, i in df.columns {
			if !is_id[i] {append(&vals, &c)}
		}
	} else {
		for name in value_vars {
			idx, found := df.name_to_index[name]
			if !found {return {}, false}
			append(&vals, &df.columns[idx])
		}
	}
	if len(vals) == 0 || var_name == value_name {return {}, false}
	type := vals[0].type
	for v in vals[1:] {
		if v.type == type {continue}
		if (v.type == .Int || v.type == .Float) && (type == .Int || type == .Float) {
			type = .Float
		} else {
			return {}, false
		}
	}

	n := df.rows * len(vals)
	rows := make([]int, n, context.temp_allocator)
	defer delete(rows, context.temp_allocator)
	for k in 0 ..< n {rows[k] = k % max(df.rows, 1)}
	out := dataframe_new(allocator)
	for c in ids {add_column(&out, column_take(c, rows, allocator))}
	names := column_new("", .String, max(n, 1), allocator)
	column_own_name(&names, var_name)
	values := column_new("", type, max(n, 1), allocator)
	column_own_name(&values, value_name)
	for v in vals {
		for r in 0 ..< df.rows {
			append_string(&names, v.name)
			if na_at(v, r) && type == .Float && v.type == .Int {
				append_null(&values)
			} else if type == .Float && v.type == .Int {
				x, _ := column_at_int(v, r)
				append_float(&values, f64(x))
			} else {
				copy_cell(&values, v, r)
			}
		}
	}
	add_column(&out, names)
	add_column(&out, values)
	out.rows = n
	return out, true
}

// --- pivot / pivot_table ---

// How pivot_table and resample combine the values that fall into one cell
Agg_Func :: enum {
	Mean, // Float mean of the present values (Int or Float values)
	Sum, // sum, same type as the values (Int or Float)
	Count, // Int number of present values
	Min, // smallest present value (any orderable type)
	Max, // largest present value
	First, // first present value (in row order; resample: in time order)
	Last, // last present value (likewise)
}

// Wide form of `df`: one row per distinct key of `index`, one column per
// distinct key of `columns` (named by the key's text, as value_as_string
// writes it), holding the `values` cell of that pair. Both keys are sorted
// ascending; the first output column holds the index keys. Pairs that do not
// occur are null, and rows with a missing index or column key are left out.
// ok = false for an unknown column, a pair that occurs twice (use pivot_table
// to aggregate), or a column key whose text is the name of the index column.
pivot :: proc(
	df: ^DataFrame,
	index, columns, values: string,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	return _pivot(df, index, columns, values, .First, true, allocator)
}

// pivot that combines the values of a repeated (index, column) pair with
// `agg`; missing values are left out before aggregating. A pair whose values
// are all missing has a Sum and a Count of 0 and is null otherwise; a pair
// that does not occur is null. ok = false as for pivot (without the
// repeated-pair case), or when `agg` needs numbers (Mean, Sum) and `values`
// is not an Int or Float column.
pivot_table :: proc(
	df: ^DataFrame,
	index, columns, values: string,
	agg := Agg_Func.Mean,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	return _pivot(df, index, columns, values, agg, false, allocator)
}

@(private = "file")
_pivot :: proc(
	df: ^DataFrame,
	index, columns, values: string,
	agg: Agg_Func,
	strict: bool,
	allocator: mem.Allocator,
) -> (
	DataFrame,
	bool,
) {
	ii, f1 := df.name_to_index[index]
	ci, f2 := df.name_to_index[columns]
	vi, f3 := df.name_to_index[values]
	if !f1 || !f2 || !f3 {return {}, false}
	icol, ccol, vcol := &df.columns[ii], &df.columns[ci], &df.columns[vi]
	if !_agg_supported(vcol.type, agg) {return {}, false}

	// distinct keys, sorted: rank[id] = output position of key id (-1: missing)
	key_ranks :: proc(col: ^Column) -> (codes: []int, rank: []int, firsts: []int) {
		c, f, na := _value_codes(col)
		keys := column_take(col, f[:], context.temp_allocator)
		defer destroy_column(&keys)
		order := column_sort_order(&keys, false, context.temp_allocator)
		defer delete(order, context.temp_allocator)
		rank = make([]int, len(f), context.temp_allocator)
		sorted_firsts := make([dynamic]int, 0, len(f), context.temp_allocator)
		for id in order {
			if id == na {
				rank[id] = -1
				continue
			}
			rank[id] = len(sorted_firsts)
			append(&sorted_firsts, f[id])
		}
		delete(f)
		return c, rank, sorted_firsts[:]
	}
	icodes, irank, ifirst := key_ranks(icol)
	ccodes, crank, cfirst := key_ranks(ccol)
	defer {
		delete(icodes, context.temp_allocator)
		delete(irank, context.temp_allocator)
		delete(ifirst, context.temp_allocator)
		delete(ccodes, context.temp_allocator)
		delete(crank, context.temp_allocator)
		delete(cfirst, context.temp_allocator)
	}
	nr, nc := len(ifirst), len(cfirst)

	// column names: the key texts, which must not clash with the index name
	names := make([]string, nc, context.temp_allocator)
	defer delete(names, context.temp_allocator)
	for r, k in cfirst {
		names[k] = value_as_string(ccol, r, context.temp_allocator)
		if names[k] == icol.name {return {}, false}
	}

	// output cell of every row, column-major (cell = column * nr + row) so
	// that each output column is one run of cells; -1: left out
	cell_of := make([]int, df.rows, context.temp_allocator)
	defer delete(cell_of, context.temp_allocator)
	for r in 0 ..< df.rows {
		ri, ck := irank[icodes[r]], crank[ccodes[r]]
		cell_of[r] = ck * nr + ri if ri >= 0 && ck >= 0 else -1
	}
	cells := _cells_build(cell_of, nc * nr)
	defer _cells_free(&cells)
	if strict {
		for k in 0 ..< nc * nr {
			if cells.start[k + 1] - cells.start[k] > 1 {return {}, false}
		}
	}

	out := dataframe_new(allocator)
	add_column(&out, column_take(icol, ifirst, allocator))
	for k in 0 ..< nc {
		col := _agg_cells(vcol, agg, &cells, k * nr, nr, false, allocator)
		column_own_name(&col, names[k])
		add_column(&out, col)
	}
	return out, true
}

// --- cell aggregation (pivot_table, resample) ---

// Rows grouped by output cell: the rows of cell k are rows[start[k]:start[k+1]],
// in row order. Temp-allocated.
@(private)
_Cells :: struct {
	start: []int,
	rows:  []int,
}

// Cells from the output cell of every row (-1: the row is left out); within
// a cell, rows come in row order, or in the order of `order` (a permutation
// of the rows) when given
@(private)
_cells_build :: proc(cell_of: []int, n_cells: int, order: []int = nil) -> _Cells {
	c: _Cells
	c.start = make([]int, n_cells + 1, context.temp_allocator)
	for k in cell_of {if k >= 0 {c.start[k + 1] += 1}}
	for k in 0 ..< n_cells {c.start[k + 1] += c.start[k]}
	c.rows = make([]int, c.start[n_cells], context.temp_allocator)
	fill := make([]int, n_cells, context.temp_allocator)
	defer delete(fill, context.temp_allocator)
	copy(fill, c.start[:n_cells])
	for i in 0 ..< len(cell_of) {
		r := order[i] if order != nil else i
		k := cell_of[r]
		if k < 0 {continue}
		c.rows[fill[k]] = r
		fill[k] += 1
	}
	return c
}

@(private)
_cells_free :: proc(c: ^_Cells) {
	delete(c.start, context.temp_allocator)
	delete(c.rows, context.temp_allocator)
}

// Can `agg` combine values of this type?
@(private)
_agg_supported :: proc(type: ColumnType, agg: Agg_Func) -> bool {
	if type == .Invalid {return false}
	if agg == .Mean || agg == .Sum {return type == .Int || type == .Float}
	return true
}

// Column of `n` cells (cells first ..< first + n) of `vcol` combined with
// `agg`, missing values left out. A cell that exists but holds no present
// value has a Sum and Count of 0 and is null otherwise (as in pandas); a cell
// that does not exist is null. With `all_exist` every cell exists (resample:
// every period in range), otherwise a cell exists when it has rows (pivot: a
// pair that occurs). Unnamed.
@(private)
_agg_cells :: proc(
	vcol: ^Column,
	agg: Agg_Func,
	cells: ^_Cells,
	first, n: int,
	all_exist: bool,
	allocator: mem.Allocator,
) -> Column {
	if agg == .Mean || agg == .Sum || agg == .Count {
		type: ColumnType = .Float if agg == .Mean else .Int if agg == .Count else vcol.type
		col := column_new("", type, max(n, 1), allocator)
		for k in first ..< first + n {
			cnt, si, sf := 0, 0, 0.0
			exists := all_exist || cells.start[k + 1] > cells.start[k]
			for r in cells.rows[cells.start[k]:cells.start[k + 1]] {
				if na_at(vcol, r) {continue}
				cnt += 1
				if agg == .Count {continue}
				if vcol.type == .Int {
					v, _ := column_at_int(vcol, r)
					si += v
					sf += f64(v)
				} else {
					v, _ := column_at_float(vcol, r)
					sf += v
				}
			}
			switch {
			case agg == .Count:
				if exists {append_int(&col, cnt)} else {append_null(&col)}
			case cnt == 0 && !(agg == .Sum && exists):
				append_null(&col)
			case agg == .Mean:
				append_float(&col, sf / f64(cnt))
			case type == .Int:
				append_int(&col, si)
			case:
				append_float(&col, sf)
			}
		}
		return col
	}
	// Min / Max / First / Last pick one source row per cell
	pick := make([]int, n, context.temp_allocator)
	defer delete(pick, context.temp_allocator)
	for k in first ..< first + n {
		best := -1
		for r in cells.rows[cells.start[k]:cells.start[k + 1]] {
			if na_at(vcol, r) {continue}
			switch agg {
			case .First:
				if best < 0 {best = r}
			case .Last:
				best = r
			case .Min:
				if best < 0 || column_compare_rows(vcol, r, best) < 0 {best = r}
			case .Max:
				if best < 0 || column_compare_rows(vcol, r, best) > 0 {best = r}
			case .Mean, .Sum, .Count:
			}
		}
		pick[k - first] = best
	}
	col := column_take(vcol, pick, allocator)
	column_own_name(&col, "")
	return col
}

// --- sample ---

// n rows of `df` drawn at random, in the order drawn; without replacement
// unless `replace`. Uses context.random_generator, so rand.reset (or your own
// generator) makes the draw repeatable. Rows keep their labels. ok = false for
// a negative n, or n > rows without replacement.
sample :: proc(
	df: ^DataFrame,
	n: int,
	replace := false,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	if n < 0 || (!replace && n > df.rows) || (replace && n > 0 && df.rows == 0) {return {}, false}
	rows := make([]int, n, context.temp_allocator)
	defer delete(rows, context.temp_allocator)
	if replace {
		for &r in rows {r = rand.int_max(df.rows)}
	} else {
		// partial Fisher-Yates over all row numbers
		perm := make([]int, df.rows, context.temp_allocator)
		defer delete(perm, context.temp_allocator)
		for i in 0 ..< df.rows {perm[i] = i}
		for i in 0 ..< n {
			j := i + rand.int_max(df.rows - i)
			perm[i], perm[j] = perm[j], perm[i]
			rows[i] = perm[i]
		}
	}
	return df_take(df, rows, allocator), true
}
