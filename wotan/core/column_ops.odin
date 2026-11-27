package core

import "core:math"
import "core:mem"
import "core:slice"

// Column calculations familiar from pandas Series:
//
//   shift / diff / pct_change          lags, differences, relative changes
//   cumsum / cumprod / cummax / cummin running totals and extremes
//   clip                               bound values to a range
//   rank                               rank values (average, min, max, first, dense)
//   nlargest / nsmallest               rows with the largest / smallest values
//   describe                           summary statistics of the numeric columns
//
// A missing cell (null, or NaN in a Float column) gives a missing result
// (null). The lag, cumulative, rank and interpolate procedures take an
// optional `by`: key columns (such as a ticker) that split the rows into
// groups, so that each group is computed on its own, like pandas'
// groupby(...).pct_change() etc. Rows with a missing key give a missing
// result. Column results are new owned columns named like the source, or
// `name` when given, so that they can be added next to the source with
// add_column. The input may be a view and is never changed.

// --- shift / diff / pct_change ---

// Copy of `col` moved down by n rows (up for a negative n); the rows that
// have no source value are null. Works for every column type. With `by`
// (one or more key columns), rows move only within their group.
// ok = false when a key column's length differs from col's.
shift :: proc(
	col: ^Column,
	n := 1,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	partner, ok := _partners(col.len, n, by)
	if !ok {return {}, false}
	defer delete(partner, context.temp_allocator)
	out := column_take(col, partner, allocator)
	_rename(&out, name)
	return out, true
}

// Difference to the value n rows earlier (later for a negative n), as an Int
// column for Int input and a Float column for Float input; for Datetime,
// Date, Time and Duration input a Duration column (the time between the two
// rows, as pandas' diff of datetimes). Rows without a partner, or with a
// missing value on either side, are null. With `by`, the partner is n rows
// earlier within the row's group. ok = false for other column types, or a
// key column of another length.
diff :: proc(
	col: ^Column,
	n := 1,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	timed :=
		col.type == .Datetime || col.type == .Date || col.type == .Time || col.type == .Duration
	if col.type != .Int && col.type != .Float && !timed {return {}, false}
	partner, ok := _partners(col.len, n, by)
	if !ok {return {}, false}
	defer delete(partner, context.temp_allocator)
	out := _new_like(col, .Duration if timed else col.type, name, allocator)
	for i in 0 ..< col.len {
		j := partner[i]
		if j < 0 || na_at(col, i) || na_at(col, j) {
			append_null(&out)
			continue
		}
		if timed {
			a, _ := _ns_of(col, i)
			b, _ := _ns_of(col, j)
			d, dok := _duration_of(a - b)
			if dok {append_duration(&out, d)} else {append_null(&out)}
		} else if col.type == .Int {
			a, _ := column_at_int(col, i)
			b, _ := column_at_int(col, j)
			append_int(&out, a - b)
		} else {
			a, _ := column_at_float(col, i)
			b, _ := column_at_float(col, j)
			append_float(&out, a - b)
		}
	}
	return out, true
}

// Relative change to the value n rows earlier, x[i] / x[i-n] - 1, as a Float
// column (Int or Float input). Rows without a partner or with a missing value
// on either side are null; gaps are not filled first (call ffill for that).
// A zero earlier value gives +-Inf, or NaN for 0 / 0. With `by`, the partner
// is n rows earlier within the row's group, so that returns of a long table
// with several tickers never mix two tickers. ok = false for other column
// types, or a key column of another length.
pct_change :: proc(
	col: ^Column,
	n := 1,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .Int && col.type != .Float {return {}, false}
	partner, ok := _partners(col.len, n, by)
	if !ok {return {}, false}
	defer delete(partner, context.temp_allocator)
	out := _new_like(col, .Float, name, allocator)
	for i in 0 ..< col.len {
		j := partner[i]
		if j < 0 || na_at(col, i) || na_at(col, j) {
			append_null(&out)
			continue
		}
		append_float(&out, _num(col, i) / _num(col, j) - 1)
	}
	return out, true
}

// --- cumulative ---

// Running sum (Int or Float column; same type out). With skipna, a missing
// cell is missing in the result and the sum carries on past it; without
// skipna, every row from the first missing cell on is missing. With `by`,
// each group has its own running sum. ok = false for other column types, or a
// key column of another length.
cumsum :: proc(
	col: ^Column,
	skipna := true,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	return _cum_arith(col, false, skipna, name, by, allocator)
}

// Running product (Int or Float column); skipna and by as for cumsum
cumprod :: proc(
	col: ^Column,
	skipna := true,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	return _cum_arith(col, true, skipna, name, by, allocator)
}

// Running maximum; works for every orderable type (Int, Float, Bool, String,
// Date, Time, Datetime). skipna and by as for cumsum.
cummax :: proc(
	col: ^Column,
	skipna := true,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	return _cum_extreme(col, true, skipna, name, by, allocator)
}

// Running minimum; as cummax
cummin :: proc(
	col: ^Column,
	skipna := true,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	return _cum_extreme(col, false, skipna, name, by, allocator)
}

// --- clip ---

// Copy of `col` (Int or Float) with values below `lower` raised to it and
// values above `upper` lowered to it. A bound that is nil or NaN is not
// applied. An Int column stays Int, so its bounds must be whole numbers in
// the int range. ok = false for other column types, lower > upper, or a
// fractional bound on an Int column.
clip :: proc(
	col: ^Column,
	lower: Maybe(f64) = nil,
	upper: Maybe(f64) = nil,
	name := "",
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .Int && col.type != .Float {return {}, false}
	lo, has_lo := lower.?
	hi, has_hi := upper.?
	has_lo &&= lo == lo
	has_hi &&= hi == hi
	if has_lo && has_hi && lo > hi {return {}, false}
	if col.type == .Int {
		whole :: proc(v: f64) -> bool {
			return v == math.trunc(v) && v >= -9223372036854775808.0 && v < 9223372036854775808.0
		}
		if (has_lo && !whole(lo)) || (has_hi && !whole(hi)) {return {}, false}
	}
	out := _new_like(col, col.type, name, allocator)
	for i in 0 ..< col.len {
		if na_at(col, i) {
			append_null(&out)
			continue
		}
		if col.type == .Int {
			v, _ := column_at_int(col, i)
			if has_lo {v = max(v, int(lo))}
			if has_hi {v = min(v, int(hi))}
			append_int(&out, v)
		} else {
			v, _ := column_at_float(col, i)
			if has_lo {v = max(v, lo)}
			if has_hi {v = min(v, hi)}
			append_float(&out, v)
		}
	}
	return out, true
}

// --- rank ---

// How rank numbers tied values
Rank_Method :: enum {
	Average, // mean of the positions the ties occupy
	Min, // lowest of those positions
	Max, // highest of those positions
	First, // positions in order of appearance
	Dense, // like Min, but the next distinct value gets the next number
}

// Rank (1 = smallest, or largest when ascending is false) of every value, as
// a Float column; missing cells stay missing. Works for every orderable type.
// With `by`, values are ranked within their group (cross-sectional ranks:
// rank the returns of each date with by = {date}). ok = false for a column
// of type Invalid, or a key column of another length.
rank :: proc(
	col: ^Column,
	method := Rank_Method.Average,
	ascending := true,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if col.type == .Invalid {return {}, false}
	g, ok := _groups(col.len, by)
	if !ok {return {}, false}
	defer _groups_free(&g)
	order := column_sort_order(col, !ascending, context.temp_allocator)
	defer delete(order, context.temp_allocator)
	ranks := make([]f64, col.len, context.temp_allocator)
	defer delete(ranks, context.temp_allocator)
	ranked := make([]bool, col.len, context.temp_allocator)
	defer delete(ranked, context.temp_allocator)

	// the sorted order split by group (each part stays sorted)
	sorted := make([]int, col.len, context.temp_allocator)
	defer delete(sorted, context.temp_allocator)
	start := make([]int, len(g.lists) + 1, context.temp_allocator)
	defer delete(start, context.temp_allocator)
	for list, k in g.lists {start[k + 1] = start[k] + len(list)}
	fill := make([]int, len(g.lists), context.temp_allocator)
	defer delete(fill, context.temp_allocator)
	copy(fill, start[:len(g.lists)])
	for r in order {
		k := g.of[r]
		if k < 0 || na_at(col, r) {continue}
		sorted[fill[k]] = r
		fill[k] += 1
	}

	for k in 0 ..< len(g.lists) {
		part := sorted[start[k]:fill[k]]
		i, dense := 0, 0
		for i < len(part) {
			j := i + 1
			for j < len(part) && column_compare_rows(col, part[i], part[j]) == 0 {j += 1}
			dense += 1
			for q in i ..< j {
				r: f64
				switch method {
				case .Average:
					r = f64(i + 1 + j) / 2
				case .Min:
					r = f64(i + 1)
				case .Max:
					r = f64(j)
				case .First:
					r = f64(q + 1)
				case .Dense:
					r = f64(dense)
				}
				ranks[part[q]] = r
				ranked[part[q]] = true
			}
			i = j
		}
	}

	out := _new_like(col, .Float, name, allocator)
	for r in 0 ..< col.len {
		if ranked[r] {append_float(&out, ranks[r])} else {append_null(&out)}
	}
	return out, true
}

// --- nlargest / nsmallest ---

// The n rows with the largest values in `column`, largest first; ties keep
// their order of appearance and missing values are skipped. Rows keep their
// labels. ok = false for an unknown column or a negative n.
nlargest :: proc(
	df: ^DataFrame,
	n: int,
	column_name: string,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	return _n_extreme(df, n, column_name, true, allocator)
}

// The n rows with the smallest values in `column`, smallest first; as nlargest
nsmallest :: proc(
	df: ^DataFrame,
	n: int,
	column_name: string,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	return _n_extreme(df, n, column_name, false, allocator)
}

// --- describe ---

// Summary statistics of every Int and Float column, as pandas describe() gives
// them: a String column "stat" with the rows count, mean, std (sample, ddof
// 1), min, 25%, 50%, 75% and max (quantiles linearly interpolated), and one
// Float column per numeric column. Missing cells are left out; a statistic
// that cannot be computed (the mean of no values, the std of one) is NaN. The
// label column is named "statistic" instead if a numeric column is named
// "stat".
describe :: proc(df: ^DataFrame, allocator := context.allocator) -> DataFrame {
	STATS :: [8]string{"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
	label_name := "stat"
	for &col in df.columns {
		if col.name == "stat" &&
		   (col.type == .Int || col.type == .Float) {label_name = "statistic"}
	}
	out := dataframe_new(allocator)
	labels := column_new(label_name, .String, len(STATS), allocator)
	for s in STATS {append_string(&labels, s)}
	add_column(&out, labels)

	for &col in df.columns {
		if col.type != .Int && col.type != .Float {continue}
		vals := make([dynamic]f64, 0, col.len, context.temp_allocator)
		defer delete(vals)
		for i in 0 ..< col.len {
			if !na_at(&col, i) {append(&vals, _num(&col, i))}
		}
		n := len(vals)
		nan := math.nan_f64()
		mean, std := nan, nan
		lo, q1, q2, q3, hi := nan, nan, nan, nan, nan
		if n > 0 {
			sum := 0.0
			for v in vals {sum += v}
			mean = sum / f64(n)
			if n > 1 {
				ss := 0.0
				for v in vals {ss += (v - mean) * (v - mean)}
				std = math.sqrt(ss / f64(n - 1))
			}
			sorted := vals[:]
			slice.sort(sorted)
			lo, hi = sorted[0], sorted[n - 1]
			q1, q2, q3 = _quantile(sorted, 0.25), _quantile(sorted, 0.5), _quantile(sorted, 0.75)
		}
		stat := column_new("", .Float, len(STATS), allocator)
		column_own_name(&stat, col.name)
		for v in ([8]f64{f64(n), mean, std, lo, q1, q2, q3, hi}) {append_float(&stat, v)}
		add_column(&out, stat)
	}
	return out
}

// --- isin / between ---

// Mask that is true where the cell equals one of `values`. A missing cell
// gives false, unless `values` contains NaN, which matches every missing cell
// (as in pandas). The values' type must match the column (int, f64, bool, string,
// Date, Time, Datetime); int values also match a Float column. ok = false on
// a type mismatch.
isin :: proc(col: ^Column, values: []$T, allocator := context.allocator) -> ([]bool, bool) {
	when T == int {
		if col.type == .Float {
			as_f := make([]f64, len(values), context.temp_allocator)
			defer delete(as_f, context.temp_allocator)
			for v, i in values {as_f[i] = f64(v)}
			return isin(col, as_f, allocator)
		}
	}
	if col.type != _type_of(T) {return nil, false}
	set := make(map[T]bool, allocator = context.temp_allocator)
	defer delete(set)
	match_na := false
	for v in values {
		key := v
		when T == f64 {
			if key != key {
				match_na = true
				continue
			}
			if key == 0 {key = 0} 	// -0.0 matches 0.0
		}
		set[key] = true
	}
	mask := make([]bool, col.len, allocator)
	for i in 0 ..< col.len {
		if na_at(col, i) {
			mask[i] = match_na
			continue
		}
		v := _cell_as(col, i, T)
		when T == f64 {if v == 0 {v = 0}}
		mask[i] = v in set
	}
	return mask, true
}

// Which ends of a `between` range belong to it
Inclusive :: enum {
	Both,
	Neither,
	Left,
	Right,
}

// Mask that is true where lower <= cell <= upper (ends as `inclusive` says);
// missing cells give false. Bounds of type int or f64 work on Int and Float
// columns; string, Date, Time and Datetime bounds on columns of that type.
// ok = false on a type mismatch.
between :: proc(
	col: ^Column,
	lower, upper: $T,
	inclusive := Inclusive.Both,
	allocator := context.allocator,
) -> (
	[]bool,
	bool,
) {
	when T == int || T == f64 {
		if col.type != .Int && col.type != .Float {return nil, false}
		if T == int && col.type == .Int {
			return _between_typed(col, int(lower), int(upper), inclusive, allocator), true
		}
		return _between_typed(col, f64(lower), f64(upper), inclusive, allocator), true
	} else {
		if col.type != _type_of(T) {return nil, false}
		return _between_typed(col, lower, upper, inclusive, allocator), true
	}
}

@(private = "file")
_between_typed :: proc(
	col: ^Column,
	lower, upper: $T,
	inclusive: Inclusive,
	allocator: mem.Allocator,
) -> []bool {
	mask := make([]bool, col.len, allocator)
	for i in 0 ..< col.len {
		if na_at(col, i) {continue}
		v := _cell_as(col, i, T)
		lo, hi := _cmp(v, lower), _cmp(v, upper)
		in_lo := lo >= 0 if inclusive == .Both || inclusive == .Left else lo > 0
		in_hi := hi <= 0 if inclusive == .Both || inclusive == .Right else hi < 0
		mask[i] = in_lo && in_hi
	}
	return mask
}

// Column type that holds values of T (.Invalid for other types)
@(private)
_type_of :: proc($T: typeid) -> ColumnType {
	when T == int {return .Int} else when T == f64 {return .Float} else when T == bool {return .Bool} else when T == string {return .String} else when T == Date {return .Date} else when T == Time {return .Time} else when T == Datetime {return .Datetime} else when T == Duration {return .Duration} else {return .Invalid}
}

// Present cell as a T: the column's own type, or an Int / Float cell as f64
@(private)
_cell_as :: proc(col: ^Column, i: int, $T: typeid) -> T {
	when T == int {
		v, _ := column_at_int(col, i); return v
	} else when T == f64 {
		return _num(col, i)
	} else when T == bool {
		v, _ := column_at_bool(col, i); return v
	} else when T == string {
		v, _ := column_at_string(col, i); return v
	} else when T == Date {
		v, _ := column_at_date(col, i); return v
	} else when T == Time {
		v, _ := column_at_time(col, i); return v
	} else when T == Datetime {
		v, _ := column_at_datetime(col, i); return v
	} else when T == Duration {
		v, _ := column_at_duration(col, i); return v
	}
}

@(private)
_cmp :: proc(a, b: $T) -> int {
	when T == Date {
		return int(date_compare(a, b))
	} else when T == Time {
		return int(time_compare(a, b))
	} else when T == Datetime {
		return int(datetime_compare(a, b))
	} else {
		return -1 if a < b else 1 if a > b else 0
	}
}

// --- interpolate ---

// Copy of `col` (Int or Float; Float out) with each gap between two present
// values filled linearly. Without `x`, rows are equally spaced; with `x` (an
// Int, Float, Date or Datetime column of the same length, without missing
// cells), the fill follows its values, so that uneven dates are weighted by
// their distance (days for Date, seconds for Datetime). limit > 0 fills at
// most `limit` cells of each gap from its start. Cells before the first or
// after the last present value stay missing. With `by`, gaps are filled
// within each group only, from the group's own values. ok = false for another
// column type, an unusable `x`, a negative limit, or a key column of another
// length.
interpolate :: proc(
	col: ^Column,
	x: ^Column = nil,
	limit := 0,
	name := "",
	by: []^Column = nil,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .Int && col.type != .Float || limit < 0 {return {}, false}
	if x != nil {
		if x.len != col.len {return {}, false}
		#partial switch x.type {
		case .Int, .Float, .Date, .Datetime:
		case:
			return {}, false
		}
		for i in 0 ..< x.len {if na_at(x, i) {return {}, false}}
	}
	g, ok := _groups(col.len, by)
	if !ok {return {}, false}
	defer _groups_free(&g)
	pos :: proc(x: ^Column, i: int) -> f64 {
		if x == nil {return f64(i)}
		#partial switch x.type {
		case .Date:
			d, _ := column_at_date(x, i)
			return f64(days_from_civil(i64(d.year), i64(d.month), i64(d.day)))
		case .Datetime:
			d, _ := column_at_datetime(x, i)
			return datetime_to_unix_f64(d)
		}
		return _num(x, i)
	}

	vals := make([]f64, col.len, context.temp_allocator)
	present := make([]bool, col.len, context.temp_allocator)
	defer delete(vals, context.temp_allocator)
	defer delete(present, context.temp_allocator)
	for list in g.lists {
		// without x, rows are spaced by their place in the group
		at :: proc(x: ^Column, list: []int, k: int) -> f64 {
			return f64(k) if x == nil else pos(x, list[k])
		}
		last := -1 // place in list of the last present value
		for i, k in list {
			if na_at(col, i) {continue}
			vals[i], present[i] = _num(col, i), true
			if last >= 0 && k - last > 1 {
				xa, xb := at(x, list, last), at(x, list, k)
				ya, yb := vals[list[last]], vals[i]
				for m in last + 1 ..< k {
					if limit > 0 && m - last > limit {break}
					t := (at(x, list, m) - xa) / (xb - xa) if xb != xa else 0
					vals[list[m]], present[list[m]] = ya + (yb - ya) * t, true
				}
			}
			last = k
		}
	}
	out := _new_like(col, .Float, name, allocator)
	for i in 0 ..< col.len {
		if present[i] {append_float(&out, vals[i])} else {append_null(&out)}
	}
	return out, true
}

// --- helpers ---

// Rows of the groups that the key columns `by` form: lists[k] holds the rows
// of group k in row order (groups in order of first appearance), of[r] the
// group of row r. A row with a missing key is in no group (of = -1); without
// keys, all rows form one group. Temp-allocated; free with _groups_free.
@(private)
_Groups :: struct {
	lists: [][]int,
	of:    []int,
	flat:  []int,
}

@(private)
_groups :: proc(n: int, by: []^Column) -> (g: _Groups, ok: bool) {
	g.of = make([]int, n, context.temp_allocator)
	if len(by) == 0 {
		g.flat = make([]int, n, context.temp_allocator)
		for i in 0 ..< n {g.flat[i], g.of[i] = i, 0}
		g.lists = make([][]int, 1, context.temp_allocator)
		g.lists[0] = g.flat
		return g, true
	}
	for c in by {
		if c.len != n {
			delete(g.of, context.temp_allocator)
			return {}, false
		}
	}
	pairs := make(map[[2]int]int, allocator = context.temp_allocator)
	defer delete(pairs)
	for c, k in by {
		codes, firsts, na := _value_codes(c)
		delete(firsts)
		for i in 0 ..< n {
			if k > 0 && g.of[i] < 0 {continue}
			if codes[i] == na {
				g.of[i] = -1
				continue
			}
			if k == 0 {
				g.of[i] = codes[i]
				continue
			}
			key := [2]int{g.of[i], codes[i]}
			id, found := pairs[key]
			if !found {
				id = len(pairs)
				pairs[key] = id
			}
			g.of[i] = id
		}
		clear(&pairs)
		delete(codes, context.temp_allocator)
	}
	// renumber groups in order of first appearance and bucket the rows
	renum := make(map[int]int, allocator = context.temp_allocator)
	defer delete(renum)
	for i in 0 ..< n {
		if g.of[i] < 0 {continue}
		id, found := renum[g.of[i]]
		if !found {
			id = len(renum)
			renum[g.of[i]] = id
		}
		g.of[i] = id
	}
	sizes := make([]int, len(renum) + 1, context.temp_allocator)
	defer delete(sizes, context.temp_allocator)
	for i in 0 ..< n {if g.of[i] >= 0 {sizes[g.of[i] + 1] += 1}}
	for k in 1 ..< len(sizes) {sizes[k] += sizes[k - 1]}
	g.flat = make([]int, sizes[len(sizes) - 1], context.temp_allocator)
	g.lists = make([][]int, len(renum), context.temp_allocator)
	fill := make([]int, len(renum), context.temp_allocator)
	defer delete(fill, context.temp_allocator)
	copy(fill, sizes[:len(renum)])
	for i in 0 ..< n {
		k := g.of[i]
		if k < 0 {continue}
		g.flat[fill[k]] = i
		fill[k] += 1
	}
	for k in 0 ..< len(renum) {g.lists[k] = g.flat[sizes[k]:sizes[k + 1]]}
	return g, true
}

@(private)
_groups_free :: proc(g: ^_Groups) {
	delete(g.lists, context.temp_allocator)
	delete(g.of, context.temp_allocator)
	delete(g.flat, context.temp_allocator)
}

// For every row, the row n places earlier in its group (-1: none)
@(private = "file")
_partners :: proc(len_: int, n: int, by: []^Column) -> ([]int, bool) {
	g, ok := _groups(len_, by)
	if !ok {return nil, false}
	defer _groups_free(&g)
	partner := make([]int, len_, context.temp_allocator)
	for &p in partner {p = -1}
	for list in g.lists {
		for r, k in list {
			j := k - n
			if j >= 0 && j < len(list) {partner[r] = list[j]}
		}
	}
	return partner, true
}


@(private)
_rename :: proc(col: ^Column, name: string) {
	if name != "" {column_own_name(col, name)}
}

// Empty owned column of `type`, named `name` or like `src`
@(private)
_new_like :: proc(
	src: ^Column,
	type: ColumnType,
	name: string,
	allocator: mem.Allocator,
) -> Column {
	out := column_new("", type, max(src.len, 1), allocator)
	column_own_name(&out, name if name != "" else src.name)
	if type == .Datetime && src.type == .Datetime {out.tz = src.tz}
	return out
}

// Present Int or Float cell as f64
@(private)
_num :: proc(col: ^Column, i: int) -> f64 {
	if col.type == .Int {
		v, _ := column_at_int(col, i)
		return f64(v)
	}
	v, _ := column_at_float(col, i)
	return v
}

@(private = "file")
_cum_arith :: proc(
	col: ^Column,
	product: bool,
	skipna: bool,
	name: string,
	by: []^Column,
	allocator: mem.Allocator,
) -> (
	Column,
	bool,
) {
	if col.type != .Int && col.type != .Float {return {}, false}
	g, ok := _groups(col.len, by)
	if !ok {return {}, false}
	defer _groups_free(&g)
	vi := make([]int, col.len, context.temp_allocator)
	vf := make([]f64, col.len, context.temp_allocator)
	present := make([]bool, col.len, context.temp_allocator)
	defer delete(vi, context.temp_allocator)
	defer delete(vf, context.temp_allocator)
	defer delete(present, context.temp_allocator)
	for list in g.lists {
		acc_i := 1 if product else 0
		acc_f := 1.0 if product else 0.0
		broken := false
		for i in list {
			if broken || na_at(col, i) {
				broken = broken || !skipna
				continue
			}
			if col.type == .Int {
				v, _ := column_at_int(col, i)
				if product {acc_i *= v} else {acc_i += v}
				vi[i] = acc_i
			} else {
				v, _ := column_at_float(col, i)
				if product {acc_f *= v} else {acc_f += v}
				vf[i] = acc_f
			}
			present[i] = true
		}
	}
	out := _new_like(col, col.type, name, allocator)
	for i in 0 ..< col.len {
		switch {
		case !present[i]:
			append_null(&out)
		case col.type == .Int:
			append_int(&out, vi[i])
		case:
			append_float(&out, vf[i])
		}
	}
	return out, true
}

@(private = "file")
_cum_extreme :: proc(
	col: ^Column,
	is_max: bool,
	skipna: bool,
	name: string,
	by: []^Column,
	allocator: mem.Allocator,
) -> (
	Column,
	bool,
) {
	if col.type == .Invalid {return {}, false}
	g, ok := _groups(col.len, by)
	if !ok {return {}, false}
	defer _groups_free(&g)
	// source row for every row: the row holding the extreme so far
	rows := make([]int, col.len, context.temp_allocator)
	defer delete(rows, context.temp_allocator)
	for &r in rows {r = -1}
	for list in g.lists {
		best := -1
		broken := false
		for i in list {
			if broken || na_at(col, i) {
				broken = broken || !skipna
				continue
			}
			if best < 0 {
				best = i
			} else {
				c := column_compare_rows(col, i, best)
				if (is_max && c > 0) || (!is_max && c < 0) {best = i}
			}
			rows[i] = best
		}
	}
	out := column_take(col, rows, allocator)
	_rename(&out, name)
	return out, true
}

@(private = "file")
_n_extreme :: proc(
	df: ^DataFrame,
	n: int,
	column_name: string,
	largest: bool,
	allocator: mem.Allocator,
) -> (
	DataFrame,
	bool,
) {
	idx, found := df.name_to_index[column_name]
	if !found || n < 0 {return {}, false}
	col := &df.columns[idx]
	order := column_sort_order(col, largest, context.temp_allocator) // stable, missing last
	defer delete(order, context.temp_allocator)
	k := 0
	for k < min(n, len(order)) && !na_at(col, order[k]) {k += 1}
	return df_take(df, order[:k], allocator), true
}
