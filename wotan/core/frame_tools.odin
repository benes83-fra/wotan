package core

import "core:math"
import "core:mem"
import "core:strconv"
import "core:strings"

// Everyday DataFrame housekeeping:
//
//   head / tail                      first or last rows (views)
//   drop_columns / rename_column     remove or rename columns
//   concat_rows                      stack frames with the same columns
//   unique / nunique / value_counts  distinct values of a column
//   duplicated / drop_duplicates     repeated rows
//   cast_column / cast_columns       change the type of a column
//
// Values compare the way pandas compares them: null and NaN (in Float
// columns) are one "missing" value that equals itself, and -0.0 equals 0.0.

// Which repeat of a duplicated row counts as the original
Keep :: enum {
	First, // keep the first occurrence, mark the later ones
	Last, // keep the last occurrence, mark the earlier ones
	None, // mark every occurrence of a repeated row
}

// --- head / tail ---

// View of the first n rows; a negative n gives all rows but the last |n|.
// Like every view, it must not outlive `df`; free it with destroy_dataframe.
head :: proc(df: ^DataFrame, n: int) -> DataFrame {
	end := n if n >= 0 else df.rows + n
	return df_slice(df, 0, clamp(end, 0, df.rows))
}

// View of the last n rows; a negative n gives all rows but the first |n|
tail :: proc(df: ^DataFrame, n: int) -> DataFrame {
	start := df.rows - n if n >= 0 else -n
	return df_slice(df, clamp(start, 0, df.rows), df.rows)
}

// --- dropping and renaming columns ---

// Copy of `df` without the named columns. Dropping the column that holds the
// index also removes the index. ok = false for an unknown column name.
drop_columns :: proc(
	df: ^DataFrame,
	names: []string,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	dropped := make([]bool, len(df.columns), context.temp_allocator)
	defer delete(dropped, context.temp_allocator)
	for name in names {
		idx, found := df.name_to_index[name]
		if !found {return {}, false}
		dropped[idx] = true
	}
	kept := make([dynamic]Column, 0, len(df.columns), context.temp_allocator)
	defer delete(kept)
	for &col, i in df.columns {
		if !dropped[i] {append(&kept, column_clone(&col, allocator))}
	}
	out := _frame_like(df, kept[:], allocator)
	if out.has_index &&
	   out.index_values.type == .Invalid &&
	   !(out.index_column in out.name_to_index) {
		out.has_index = false
		out.index_column = ""
	}
	return out, true
}

// Renames a column in place; the frame keeps its own copy of `new_name`.
// ok = false when `old_name` does not exist, `new_name` is taken by another
// column, or the frame is a view (materialize it first).
rename_column :: proc(df: ^DataFrame, old_name, new_name: string) -> bool {
	idx, found := df.name_to_index[old_name]
	if !found {return false}
	if old_name == new_name {return true}
	if new_name in df.name_to_index {return false}
	col := &df.columns[idx]
	if col.is_view {return false}
	is_index := df.has_index && df.index_column == old_name
	delete_key(&df.name_to_index, old_name)
	column_own_name(col, new_name)
	df.name_to_index[col.name] = idx
	if is_index {df.index_column = col.name}
	return true
}

// --- concat_rows ---

// The rows of all frames, one after another, as a new owned frame numbered
// 0..<n. Every frame must have the same column names with the same types; the
// columns come in the order of the first frame. The index is kept when all
// frames have the same one. ok = false for an empty list or mismatched columns.
concat_rows :: proc(frames: []^DataFrame, allocator := context.allocator) -> (DataFrame, bool) {
	if len(frames) == 0 {return {}, false}
	first := frames[0]
	total := 0
	same_index := true
	for f in frames {
		if len(f.columns) != len(first.columns) {return {}, false}
		for &col in first.columns {
			idx, found := f.name_to_index[col.name]
			if !found || f.columns[idx].type != col.type {return {}, false}
		}
		same_index &&=
			f.has_index == first.has_index &&
			f.index_column == first.index_column &&
			f.index_values.type == first.index_values.type
		total += f.rows
	}

	stack :: proc(dst: ^Column, frames: []^DataFrame, name: string, index_values: bool) {
		for f in frames {
			src := &f.index_values if index_values else &f.columns[f.name_to_index[name]]
			for r in 0 ..< f.rows {copy_cell(dst, src, r)}
		}
	}

	out := dataframe_new(allocator)
	for &col in first.columns {
		c := column_new("", col.type, max(total, 1), allocator)
		column_own_name(&c, col.name)
		stack(&c, frames, col.name, false)
		add_column(&out, c)
	}
	out.rows = total
	if same_index && first.has_index {
		out.has_index = true
		if first.index_values.type != .Invalid {
			out.index_values = column_new("", first.index_values.type, max(total, 1), allocator)
			column_own_name(&out.index_values, first.index_values.name)
			stack(&out.index_values, frames, "", true)
			out.index_column = out.index_values.name
		} else {
			out.index_column = out.columns[out.name_to_index[first.index_column]].name
		}
	}
	return out, true
}

// --- distinct values ---

// The distinct values of `col` in order of first appearance, as a new column.
// Missing cells count as one value.
unique :: proc(col: ^Column, allocator := context.allocator) -> Column {
	_, firsts, _ := _value_codes(col)
	defer delete(firsts)
	return column_take(col, firsts[:], allocator)
}

// Number of distinct values; missing cells count as one more value only when
// dropna is false
nunique :: proc(col: ^Column, dropna := true) -> int {
	codes, firsts, na_id := _value_codes(col)
	defer delete(codes, context.temp_allocator)
	defer delete(firsts)
	return len(firsts) - (1 if dropna && na_id >= 0 else 0)
}

// How often each distinct value occurs: a frame with the values (in a column
// named like `col`, or "value" if `col` is itself named "count") and their
// counts (Int column "count"), most frequent first; equal counts keep the
// order of first appearance. Missing cells are counted only when dropna is
// false.
value_counts :: proc(col: ^Column, dropna := true, allocator := context.allocator) -> DataFrame {
	codes, firsts, na_id := _value_codes(col)
	defer delete(codes, context.temp_allocator)
	defer delete(firsts)
	counts := make([]int, len(firsts), context.temp_allocator)
	defer delete(counts, context.temp_allocator)
	for c in codes {counts[c] += 1}

	order := make([dynamic]int, 0, len(firsts), context.temp_allocator)
	defer delete(order)
	for id in 0 ..< len(firsts) {
		if dropna && id == na_id {continue}
		// insertion into a list sorted by count descending, ties after
		// earlier ids: stable
		append(&order, id)
		j := len(order) - 1
		for j > 0 && counts[order[j - 1]] < counts[id] {
			order[j] = order[j - 1]
			j -= 1
		}
		order[j] = id
	}

	rows := make([]int, len(order), context.temp_allocator)
	defer delete(rows, context.temp_allocator)
	n := make([]int, len(order), context.temp_allocator)
	defer delete(n, context.temp_allocator)
	for id, i in order {
		rows[i] = firsts[id]
		n[i] = counts[id]
	}
	values := column_take(col, rows, allocator)
	if col.name == "count" {column_own_name(&values, "value")}
	count_col := column_new("count", .Int, max(len(n), 1), allocator)
	for v in n {append_int(&count_col, v)}
	out := dataframe_new(allocator)
	add_column(&out, values)
	add_column(&out, count_col)
	return out
}

// --- duplicated rows ---

// Mask that is true for rows repeating an earlier (keep = .First) or later
// (.Last) row, or for every row that occurs more than once (.None). Rows are
// compared on `columns` (all columns when empty). ok = false for an unknown
// column name.
duplicated :: proc(
	df: ^DataFrame,
	columns: []string = nil,
	keep := Keep.First,
	allocator := context.allocator,
) -> (
	[]bool,
	bool,
) {
	codes, ok := _row_codes(df, columns)
	if !ok {return nil, false}
	defer delete(codes, context.temp_allocator)
	mask := make([]bool, df.rows, allocator)
	seen := make([]int, df.rows, context.temp_allocator) // occurrences per code
	defer delete(seen, context.temp_allocator)
	switch keep {
	case .First:
		for c, i in codes {
			mask[i] = seen[c] > 0
			seen[c] += 1
		}
	case .Last:
		#reverse for c, i in codes {
			mask[i] = seen[c] > 0
			seen[c] += 1
		}
	case .None:
		for c in codes {seen[c] += 1}
		for c, i in codes {mask[i] = seen[c] > 1}
	}
	return mask, true
}

// Rows of `df` that duplicated does not mark, with their row labels
drop_duplicates :: proc(
	df: ^DataFrame,
	columns: []string = nil,
	keep := Keep.First,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	mask, ok := duplicated(df, columns, keep, context.temp_allocator)
	if !ok {return {}, false}
	defer delete(mask, context.temp_allocator)
	rows := make([dynamic]int, 0, df.rows, context.temp_allocator)
	defer delete(rows)
	for dup, i in mask {
		if !dup {append(&rows, i)}
	}
	return df_take(df, rows[:], allocator), true
}

// --- casting ---

// Copy of `col` converted to type `to`. Missing cells stay missing (null).
//   Int <-> Float <-> Bool   Float to Int truncates toward zero; to Bool,
//                            non-zero is true; from Bool, true is 1
//   any type -> String       the canonical text of value_as_string
//   String -> any type       parsed after trimming spaces: integers, floats,
//                            true/false/1/0 (any case), YYYY-MM-DD, HH:MM:SS,
//                            "YYYY-MM-DD HH:MM:SS" (or with a T, or a date
//                            alone for midnight); blank strings become null
//   Date -> Datetime         midnight
//   Datetime -> Date / Time  the date or the time of day
// A cell that cannot be converted (unparseable text, an infinite or too
// large float for Int) makes the cast fail when `strict`, and becomes null
// otherwise. ok = false for an unsupported pair of types or, when `strict`,
// a cell that cannot be converted.
cast_column :: proc(
	col: ^Column,
	to: ColumnType,
	strict := true,
	allocator := context.allocator,
) -> (
	Column,
	bool,
) {
	if !_cast_supported(col.type, to) {return {}, false}
	out := column_new("", to, max(col.len, 1), allocator)
	column_own_name(&out, col.name)
	for i in 0 ..< col.len {
		if na_at(col, i) {
			append_null(&out)
			continue
		}
		if !_cast_cell(&out, col, i) {
			if strict {
				destroy_column(&out)
				return {}, false
			}
			append_null(&out)
		}
	}
	return out, true
}

// Copy of `df` with the named columns converted by cast_column. ok = false for
// an unknown column name, or when cast_column fails for one of them.
cast_columns :: proc(
	df: ^DataFrame,
	names: []string,
	to: ColumnType,
	strict := true,
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	target := make([]bool, len(df.columns), context.temp_allocator)
	defer delete(target, context.temp_allocator)
	for name in names {
		idx, found := df.name_to_index[name]
		if !found {return {}, false}
		target[idx] = true
	}
	cols := make([dynamic]Column, 0, len(df.columns), context.temp_allocator)
	defer delete(cols)
	for &col, i in df.columns {
		if !target[i] {
			append(&cols, column_clone(&col, allocator))
			continue
		}
		c, ok := cast_column(&col, to, strict, allocator)
		if !ok {
			for &done in cols {destroy_column(&done)}
			return {}, false
		}
		append(&cols, c)
	}
	return _frame_like(df, cols[:], allocator), true
}

// --- helpers ---

// Value id of every row (ids numbered in order of first appearance; all
// missing cells share one id, na_id, or -1 if there are none) and the first
// row of each id. Both temp-allocated.
@(private)
_value_codes :: proc(col: ^Column) -> (codes: []int, firsts: [dynamic]int, na_id: int) {
	codes = make([]int, col.len, context.temp_allocator)
	firsts = make([dynamic]int, 0, 16, context.temp_allocator)
	na_id = -1
	switch col.type {
	case .Int:
		_codes_of(col, int, column_at_int, codes, &firsts, &na_id)
	case .Float:
		_codes_of(col, f64, column_at_float, codes, &firsts, &na_id)
	case .Bool:
		_codes_of(col, bool, column_at_bool, codes, &firsts, &na_id)
	case .String:
		_codes_of(col, string, column_at_string, codes, &firsts, &na_id)
	case .Date:
		_codes_of(col, Date, column_at_date, codes, &firsts, &na_id)
	case .Time:
		_codes_of(col, Time, column_at_time, codes, &firsts, &na_id)
	case .Datetime:
		_codes_of(col, Datetime, column_at_datetime, codes, &firsts, &na_id)
	case .Invalid:
		for i in 0 ..< col.len {codes[i] = 0}
		if col.len > 0 {
			na_id = 0
			append(&firsts, 0)
		}
	}
	return
}

@(private = "file")
_codes_of :: proc(
	col: ^Column,
	$T: typeid,
	at: proc(col: ^Column, i: int) -> (T, bool),
	codes: []int,
	firsts: ^[dynamic]int,
	na_id: ^int,
) {
	ids := make(map[T]int, allocator = context.temp_allocator)
	defer delete(ids)
	for i in 0 ..< col.len {
		v, missing := at(col, i)
		when T == f64 {
			if v != v {missing = true}
			if v == 0 {v = 0} 	// -0.0 and 0.0 are one value
		}
		if missing {
			if na_id^ < 0 {
				na_id^ = len(firsts)
				append(firsts, i)
			}
			codes[i] = na_id^
			continue
		}
		id, found := ids[v]
		if !found {
			id = len(firsts)
			ids[v] = id
			append(firsts, i)
		}
		codes[i] = id
	}
}

// Row id of every row of `df` over `columns` (all when empty): equal rows get
// equal ids. Temp-allocated.
@(private = "file")
_row_codes :: proc(df: ^DataFrame, columns: []string) -> ([]int, bool) {
	cols := make([dynamic]^Column, 0, len(df.columns), context.temp_allocator)
	defer delete(cols)
	if len(columns) == 0 {
		for &c in df.columns {append(&cols, &c)}
	} else {
		for name in columns {
			idx, found := df.name_to_index[name]
			if !found {return nil, false}
			append(&cols, &df.columns[idx])
		}
	}
	codes := make([]int, df.rows, context.temp_allocator)
	pairs := make(map[[2]int]int, allocator = context.temp_allocator)
	defer delete(pairs)
	for c, k in cols {
		next, firsts, _ := _value_codes(c)
		delete(firsts)
		if k == 0 {
			copy(codes, next)
		} else {
			clear(&pairs)
			for i in 0 ..< df.rows {
				key := [2]int{codes[i], next[i]}
				id, found := pairs[key]
				if !found {
					id = len(pairs)
					pairs[key] = id
				}
				codes[i] = id
			}
		}
		delete(next, context.temp_allocator)
	}
	return codes, true
}

@(private = "file")
_cast_supported :: proc(from, to: ColumnType) -> bool {
	if from == .Invalid || to == .Invalid {return false}
	if from == to || from == .String || to == .String {return true}
	numeric :: bit_set[ColumnType]{.Int, .Float, .Bool}
	if from in numeric && to in numeric {return true}
	if from == .Date && to == .Datetime {return true}
	if from == .Datetime && (to == .Date || to == .Time) {return true}
	return false
}

// Appends the converted cell `i` of `src` (present, not missing) to `dst`;
// false (nothing appended) when the value cannot be converted
@(private = "file")
_cast_cell :: proc(dst, src: ^Column, i: int) -> bool {
	if src.type == dst.type {
		copy_cell(dst, src, i)
		return true
	}
	if dst.type == .String {
		append_string(dst, value_as_string(src, i, context.temp_allocator))
		return true
	}
	if src.type == .String {
		s, _ := column_at_string(src, i)
		return _parse_into(dst, strings.trim_space(s))
	}
	#partial switch src.type {
	case .Int:
		v, _ := column_at_int(src, i)
		#partial switch dst.type {
		case .Float:
			append_float(dst, f64(v))
		case .Bool:
			append_bool(dst, v != 0)
		}
	case .Float:
		v, _ := column_at_float(src, i)
		#partial switch dst.type {
		case .Int:
			// [-2^63, 2^63): the floats that truncate to an int
			if !(v >= -9223372036854775808.0 && v < 9223372036854775808.0) {return false}
			append_int(dst, int(math.trunc(v)))
		case .Bool:
			append_bool(dst, v != 0)
		}
	case .Bool:
		v, _ := column_at_bool(src, i)
		#partial switch dst.type {
		case .Int:
			append_int(dst, 1 if v else 0)
		case .Float:
			append_float(dst, 1 if v else 0)
		}
	case .Date:
		v, _ := column_at_date(src, i)
		append_datetime(dst, Datetime{v.year, v.month, v.day, 0, 0, 0})
	case .Datetime:
		v, _ := column_at_datetime(src, i)
		if dst.type == .Date {
			append_date(dst, Date{v.year, v.month, v.day})
		} else {
			append_time(dst, Time{v.hour, v.minute, v.second})
		}
	}
	return true
}

// Parses trimmed text `s` into a cell of dst's type; blank text gives null
@(private = "file")
_parse_into :: proc(dst: ^Column, s: string) -> bool {
	if s == "" {
		append_null(dst)
		return true
	}
	#partial switch dst.type {
	case .Int:
		v, ok := strconv.parse_int(s, 10)
		if !ok {return false}
		append_int(dst, v)
	case .Float:
		v, ok := strconv.parse_f64(s)
		if !ok {return false}
		append_float(dst, v)
	case .Bool:
		switch {
		case strings.equal_fold(s, "true") || s == "1":
			append_bool(dst, true)
		case strings.equal_fold(s, "false") || s == "0":
			append_bool(dst, false)
		case:
			return false
		}
	case .Date:
		v, ok := parse_date(s)
		if !ok {return false}
		append_date(dst, v)
	case .Time:
		v, ok := parse_time(s)
		if !ok {return false}
		append_time(dst, v)
	case .Datetime:
		v, ok := parse_datetime(s)
		if !ok {
			d, dok := parse_date(s)
			if !dok {return false}
			v = Datetime{d.year, d.month, d.day, 0, 0, 0}
		}
		append_datetime(dst, v)
	case:
		return false
	}
	return true
}
