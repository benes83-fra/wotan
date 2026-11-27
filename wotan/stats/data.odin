package stats

import w "../core"
import "core:fmt"
import "core:mem"
import "core:slice"
import "core:strings"

// DataFrame columns as test input. Numeric columns are Int, Float or Bool
// (as 0 / 1); null cells and NaN are left out. Group columns are String,
// Int or Bool, their labels sorted (numerically for Int).

// The numeric values of a column, without nulls and NaN.
column_values :: proc(
	df: ^w.DataFrame,
	name: string,
	allocator := context.allocator,
) -> (
	values: []f64,
	ok: bool,
) {
	c := _numeric_column(df, name) or_return
	out := make([dynamic]f64, 0, c.len, allocator)
	for i in 0 ..< c.len {
		v, null := w.column_at_numeric(c, i)
		if !null {append(&out, v)}
	}
	return out[:], true
}

// The rows where both columns hold a number, as two slices of equal length
// (for paired tests and correlations).
paired_values :: proc(
	df: ^w.DataFrame,
	a, b: string,
	allocator := context.allocator,
) -> (
	x, y: []f64,
	ok: bool,
) {
	ca := _numeric_column(df, a) or_return
	cb := _numeric_column(df, b) or_return
	xs := make([dynamic]f64, 0, ca.len, allocator)
	ys := make([dynamic]f64, 0, ca.len, allocator)
	for i in 0 ..< ca.len {
		u, nu := w.column_at_numeric(ca, i)
		v, nv := w.column_at_numeric(cb, i)
		if !nu && !nv {
			append(&xs, u)
			append(&ys, v)
		}
	}
	return xs[:], ys[:], true
}

// Samples of one column split by the labels of another.
Groups :: struct {
	labels:  []string,
	samples: [][]f64, // samples[i] holds the values of labels[i]
}

// Splits the numeric column `value` by the labels in column `group` (rows
// with a null in either are left out), for k-sample tests:
//
//     g, _ := stats.grouped_values(&df, "yield", "fertilizer")
//     defer stats.groups_destroy(&g)
//     res, _ := stats.kruskal_wallis(g.samples)
grouped_values :: proc(
	df: ^w.DataFrame,
	value, group: string,
	allocator := context.allocator,
) -> (
	g: Groups,
	ok: bool,
) {
	cv := _numeric_column(df, value) or_return
	codes, labels := _group_codes(df, group, allocator) or_return
	defer delete(codes, allocator)
	lists := make([][dynamic]f64, len(labels), allocator)
	defer delete(lists, allocator)
	for &l in lists {l = make([dynamic]f64, 0, allocator)}
	for i in 0 ..< cv.len {
		v, null := w.column_at_numeric(cv, i)
		if !null && codes[i] >= 0 {append(&lists[codes[i]], v)}
	}
	samples := make([][]f64, len(labels), allocator)
	for l, i in lists {samples[i] = l[:]}
	return Groups{labels, samples}, true
}

groups_destroy :: proc(g: ^Groups, allocator := context.allocator) {
	for s in g.samples {delete(s, allocator)}
	for l in g.labels {delete(l, allocator)}
	delete(g.samples, allocator)
	delete(g.labels, allocator)
	g^ = {}
}

// A contingency table of counts, row-major.
Table :: struct {
	counts:     []f64, // rows * cols, counts[i * cols + j]
	rows, cols: int,
	row_labels: []string,
	col_labels: []string,
}

// Counts of the label pairs of two columns (rows with a null in either are
// left out), for chi_square_contingency and fisher_exact.
crosstab :: proc(
	df: ^w.DataFrame,
	row, col: string,
	allocator := context.allocator,
) -> (
	t: Table,
	ok: bool,
) {
	rc, rl := _group_codes(df, row, allocator) or_return
	defer delete(rc, allocator)
	cc, cl, cok := _group_codes(df, col, allocator)
	if !cok {
		for l in rl {delete(l, allocator)}
		delete(rl, allocator)
		return {}, false
	}
	defer delete(cc, allocator)
	t = Table {
		counts     = make([]f64, len(rl) * len(cl), allocator),
		rows       = len(rl),
		cols       = len(cl),
		row_labels = rl,
		col_labels = cl,
	}
	for i in 0 ..< len(rc) {
		if rc[i] >= 0 && cc[i] >= 0 {t.counts[rc[i] * t.cols + cc[i]] += 1}
	}
	return t, true
}

table_destroy :: proc(t: ^Table, allocator := context.allocator) {
	delete(t.counts, allocator)
	for l in t.row_labels {delete(l, allocator)}
	for l in t.col_labels {delete(l, allocator)}
	delete(t.row_labels, allocator)
	delete(t.col_labels, allocator)
	t^ = {}
}

// ---------------------------------------------------------------------------

@(private)
_numeric_column :: proc(df: ^w.DataFrame, name: string) -> (c: ^w.Column, ok: bool) {
	idx, found := df.name_to_index[name]
	if !found {return nil, false}
	c = &df.columns[idx]
	#partial switch c.type {
	case .Int, .Float, .Bool:
		return c, true
	}
	return nil, false
}

// group index of each row (-1 for null) and the sorted labels
@(private)
_group_codes :: proc(
	df: ^w.DataFrame,
	name: string,
	allocator: mem.Allocator,
) -> (
	codes: []int,
	labels: []string,
	ok: bool,
) {
	w.TEMP_GUARD(ignore = _is_temp(allocator))
	idx, found := df.name_to_index[name]
	if !found {return nil, nil, false}
	c := &df.columns[idx]
	n := c.len
	#partial switch c.type {
	case .String:
		keys := make([dynamic]string, 0, context.temp_allocator)
		for i in 0 ..< n {
			s, null := w.column_at_string(c, i)
			if !null {append(&keys, s)}
		}
		slice.sort(keys[:])
		uniq := slice.unique(keys[:])
		codes = make([]int, n, allocator)
		for i in 0 ..< n {
			s, null := w.column_at_string(c, i)
			codes[i] = -1
			if !null {codes[i], _ = slice.binary_search(uniq, s)}
		}
		labels = make([]string, len(uniq), allocator)
		for s, i in uniq {labels[i] = strings.clone(s, allocator)}
		return codes, labels, true
	case .Int, .Bool:
		keys := make([dynamic]int, 0, context.temp_allocator)
		for i in 0 ..< n {
			v, null := _int_key(c, i)
			if !null {append(&keys, v)}
		}
		slice.sort(keys[:])
		uniq := slice.unique(keys[:])
		codes = make([]int, n, allocator)
		for i in 0 ..< n {
			v, null := _int_key(c, i)
			codes[i] = -1
			if !null {codes[i], _ = slice.binary_search(uniq, v)}
		}
		labels = make([]string, len(uniq), allocator)
		for v, i in uniq {
			if c.type == .Bool {
				labels[i] = strings.clone("true" if v == 1 else "false", allocator)
			} else {
				labels[i] = fmt.aprint(v, allocator = allocator)
			}
		}
		return codes, labels, true
	}
	return nil, nil, false
}

@(private)
_int_key :: proc(c: ^w.Column, i: int) -> (int, bool) {
	if c.type == .Bool {
		b, null := w.column_at_bool(c, i)
		return int(b), null
	}
	return w.column_at_int(c, i)
}

@(private)
_is_temp :: proc(a: mem.Allocator) -> bool {
	return a.data == context.temp_allocator.data && a.procedure == context.temp_allocator.procedure
}
