package core

import "core:fmt"
import "core:slice"

// Which right rows merge_asof may match
Asof_Direction :: enum {
	Backward, // the last right row whose key is <= the left key
	Forward, // the first right row whose key is >= the left key
	Nearest, // the closer of the two (the backward one on a tie)
}

// Joins every row of `left` to the nearest row of `right` by the ordered key
// `on`, as pandas.merge_asof does: typically the last quote at or before each
// trade, or the last published value at or before each date.
//
// The result has one row per left row, in left order and with left's labels
// and index: all left columns, then the right columns except the key and the
// `by` columns. A right column whose name is taken gets "_right" (then
// "_right2", ...). Left rows without a match get nulls.
//
//   on                   key column, the same type in both frames: Int, Float,
//                        Date, Time or Datetime (`right_on` names right's key
//                        if it is named differently)
//   by                   key columns that must match exactly (such as a
//                        ticker), the same names and types in both frames
//   direction            .Backward, .Forward or .Nearest
//   tolerance            largest allowed key distance: in the key's units, so
//                        days for Date and seconds for Time / Datetime
//   allow_exact_matches  false: a right key equal to the left key does not count
//
// Unlike pandas, neither frame has to be sorted. Of several right rows with
// the same key, Backward takes the last and Forward the first. Missing keys
// never match. Int keys are compared as f64, exactly up to 2^53.
// ok = false for an unknown column, key columns of different or unsupported
// types, a negative tolerance, or a right column whose suffixed name is taken.
merge_asof :: proc(
	left, right: ^DataFrame,
	on: string,
	by: []string = nil,
	direction := Asof_Direction.Backward,
	tolerance: Maybe(f64) = nil,
	allow_exact_matches := true,
	right_on := "",
	allocator := context.allocator,
) -> (
	DataFrame,
	bool,
) {
	r_on := right_on if right_on != "" else on
	li, f1 := left.name_to_index[on]
	ri, f2 := right.name_to_index[r_on]
	if !f1 || !f2 {return {}, false}
	lkey, rkey := &left.columns[li], &right.columns[ri]
	if lkey.type != rkey.type || (lkey.tz.id == 0) != (rkey.tz.id == 0) {return {}, false}
	#partial switch lkey.type {
	case .Int, .Float, .Date, .Time, .Datetime:
	case:
		return {}, false
	}
	tol, has_tol := tolerance.?
	if has_tol && !(tol >= 0) {return {}, false}

	// group code of every left row (lcode) and right row (rcode); -1: missing
	lcode := make([]int, left.rows, context.temp_allocator)
	rcode := make([]int, right.rows, context.temp_allocator)
	defer delete(lcode, context.temp_allocator)
	defer delete(rcode, context.temp_allocator)
	if !_asof_codes(left, right, by, lcode, rcode) {return {}, false}

	// right rows with a key, sorted by (group, key, row)
	Entry :: struct {
		code: int,
		key:  f64,
		row:  int,
	}
	entries := make([dynamic]Entry, 0, right.rows, context.temp_allocator)
	defer delete(entries)
	for r in 0 ..< right.rows {
		if rcode[r] < 0 || na_at(rkey, r) {continue}
		append(&entries, Entry{rcode[r], _asof_pos(rkey, r), r})
	}
	slice.sort_by(entries[:], proc(a, b: Entry) -> bool {
		if a.code != b.code {return a.code < b.code}
		if a.key != b.key {return a.key < b.key}
		return a.row < b.row
	})
	// range of entries per group
	ranges := make(map[int][2]int, allocator = context.temp_allocator)
	defer delete(ranges)
	for e, k in entries {
		rg, found := ranges[e.code]
		if !found {rg[0] = k}
		rg[1] = k + 1
		ranges[e.code] = rg
	}

	match := make([]int, left.rows, context.temp_allocator)
	defer delete(match, context.temp_allocator)
	for l in 0 ..< left.rows {
		match[l] = -1
		if lcode[l] < 0 || na_at(lkey, l) {continue}
		rg, found := ranges[lcode[l]]
		if !found {continue}
		part := entries[rg[0]:rg[1]]
		x := _asof_pos(lkey, l)
		// back: last entry with key <= x (< x without exact matches);
		// fwd: first entry with key >= x (> x)
		back, fwd := -1, -1
		{
			lo, hi := 0, len(part) // first entry beyond x
			for lo < hi {
				mid := (lo + hi) / 2
				if part[mid].key < x ||
				   (allow_exact_matches && part[mid].key == x) {lo = mid + 1} else {hi = mid}
			}
			back = lo - 1
		}
		{
			lo, hi := 0, len(part) // first entry at or beyond x
			for lo < hi {
				mid := (lo + hi) / 2
				if part[mid].key < x ||
				   (!allow_exact_matches && part[mid].key == x) {lo = mid + 1} else {hi = mid}
			}
			if lo < len(part) {fwd = lo}
		}
		pick := -1
		switch direction {
		case .Backward:
			pick = back
		case .Forward:
			pick = fwd
		case .Nearest:
			switch {
			case back < 0:
				pick = fwd
			case fwd < 0:
				pick = back
			case:
				pick = back if x - part[back].key <= part[fwd].key - x else fwd
			}
		}
		if pick < 0 {continue}
		if has_tol && abs(part[pick].key - x) > tol {continue}
		match[l] = part[pick].row
	}

	// result: left columns, then right columns except the keys
	skip := make([]bool, len(right.columns), context.temp_allocator)
	defer delete(skip, context.temp_allocator)
	skip[ri] = true
	for name in by {skip[right.name_to_index[name]] = true}
	cols := make(
		[dynamic]Column,
		0,
		len(left.columns) + len(right.columns),
		context.temp_allocator,
	)
	defer delete(cols)
	taken := make(map[string]bool, allocator = context.temp_allocator)
	defer delete(taken)
	for &c in left.columns {
		append(&cols, column_clone(&c, allocator))
		taken[c.name] = true
	}
	for &c, k in right.columns {
		if skip[k] {continue}
		name := c.name
		for suffix := 1; (name in taken); suffix += 1 {
			name =
				fmt.tprintf("%s_right", c.name) if suffix == 1 else fmt.tprintf("%s_right%d", c.name, suffix)
		}
		taken[name] = true
		col := column_take(&c, match, allocator)
		column_own_name(&col, name)
		append(&cols, col)
	}
	return _frame_like(left, cols[:], allocator), true
}

// Key of a present cell as f64: the number, days for Date, seconds for Time
// and Datetime
@(private = "file")
_asof_pos :: proc(col: ^Column, i: int) -> f64 {
	#partial switch col.type {
	case .Int:
		v, _ := column_at_int(col, i)
		return f64(v)
	case .Float:
		v, _ := column_at_float(col, i)
		return v
	case .Date:
		d, _ := column_at_date(col, i)
		return f64(days_from_civil(i64(d.year), i64(d.month), i64(d.day)))
	case .Time:
		t, _ := column_at_time(col, i)
		return f64(t.hour) * 3600 + f64(t.minute) * 60 + f64(t.second) + f64(t.nanosecond) / 1e9
	case .Datetime:
		d, _ := column_at_datetime(col, i)
		return datetime_to_unix_f64(d)
	}
	return 0
}

// Common group codes of the `by` key columns across both frames (equal key
// tuples get equal codes, -1 for a row with a missing key); all 0 without
// keys. False for an unknown key name or keys of different types.
@(private = "file")
_asof_codes :: proc(left, right: ^DataFrame, by: []string, lcode, rcode: []int) -> bool {
	pairs := make(map[[2]int]int, allocator = context.temp_allocator)
	defer delete(pairs)
	nl := left.rows
	for name, k in by {
		li, f1 := left.name_to_index[name]
		ri, f2 := right.name_to_index[name]
		if !f1 || !f2 {return false}
		lc, rc := &left.columns[li], &right.columns[ri]
		if lc.type != rc.type || lc.type == .Invalid || (lc.tz.id == 0) != (rc.tz.id == 0) {
			return false
		}
		// both frames' keys in one column, so that equal keys share a code
		both := column_new("", lc.type, max(nl + right.rows, 1), context.temp_allocator)
		defer destroy_column(&both)
		for r in 0 ..< nl {copy_cell(&both, lc, r)}
		for r in 0 ..< right.rows {copy_cell(&both, rc, r)}
		codes, firsts, na := _value_codes(&both)
		delete(firsts)
		defer delete(codes, context.temp_allocator)
		for i in 0 ..< len(codes) {
			code := &lcode[i] if i < nl else &rcode[i - nl]
			if k > 0 && code^ < 0 {continue}
			if codes[i] == na {
				code^ = -1
				continue
			}
			if k == 0 {
				code^ = codes[i]
				continue
			}
			key := [2]int{code^, codes[i]}
			id, found := pairs[key]
			if !found {
				id = len(pairs)
				pairs[key] = id
			}
			code^ = id
		}
		clear(&pairs)
	}
	return true
}
