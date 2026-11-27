package parquet

import "base:intrinsics"
import "core:fmt"
import "core:strconv"
import "core:strings"

import w "../core"

// Nested columns (lists, maps, structs, and any nesting of them) are read as
// String columns of JSON text, one value per row:
//   list    -> [1,2,3]          (an empty list is [])
//   struct  -> {"a":1,"b":"x"}  (fields in schema order)
//   map     -> {"key":value}    (keys that are not strings become their text)
//   null    -> null inside a value; a missing top-level value is a null cell
// Leaf values are written as in json_write: numbers as numbers (NaN and
// infinities as null, decimals as their nearest double, unsigned 64-bit
// integers exactly), strings escaped,
// dates, times and timestamps as their wotan text ("2024-01-02",
// "09:30:00.5", "2024-01-02 09:30:00"), binary without a text annotation
// with each byte of 0x80 and above as \u00XX.
//
// Each leaf of a nested field is read on its own, as its repetition and
// definition levels plus its present values (the Dremel encoding); the rows
// are then assembled from all leaves of the field together.

@(private)
Node_Kind :: enum u8 {
	Leaf,
	Struct,
	List,
	Map,
}

// A field of a nested top-level column
@(private)
Node :: struct {
	name:             string,
	kind:             Node_Kind,
	optional:         bool,
	repeated:         bool,
	def, rep:         int, // the levels at which this field is present / repeats
	leaf:             int, // Leaf: index into the leaves
	leaf_lo, leaf_hi: int, // the leaves under this field
	kids_lo, kids_n:  int, // children: kids[kids_lo:][:kids_n]
	elem:             int, // List: the element field (may be the repeated field itself)
	rnode:            int, // List / Map: the repeated field
	key, value:       int, // Map: key and value fields (value -1 if absent)
}

MAX_NESTING :: 64

// Plans the field at schema[i^] and everything under it (i^ moves past it);
// returns its node index. `reason` gets the first thing that makes the field
// unreadable.
@(private)
_plan_node :: proc(
	schema: []Schema_Element,
	i: ^int,
	parent_def, parent_rep: int,
	path: string,
	depth: int,
	m: ^Meta,
	reason: ^string,
) -> int {
	se := schema[i^]
	i^ += 1
	n := Node {
		name     = se.name,
		optional = se.repetition == 1,
		repeated = se.repetition == 2,
		leaf     = -1,
		elem     = -1,
		rnode    = -1,
		key      = -1,
		value    = -1,
	}
	n.def = parent_def + (1 if n.optional || n.repeated else 0)
	n.rep = parent_rep + (1 if n.repeated else 0)
	full := se.name if path == "" else fmt.tprintf("%s.%s", path, se.name)
	if depth > MAX_NESTING && reason^ == "" {reason^ = "nested too deeply"}
	n.leaf_lo = len(m.leaves)
	idx := len(m.nodes)
	append(&m.nodes, n)

	if se.num_children <= 0 {
		leaf := _plan_leaf(se)
		leaf.name = full
		leaf.chunk = len(m.leaves)
		leaf.nested = true
		leaf.max_def = n.def
		leaf.max_rep = n.rep
		lt, ct := se.logical.kind, se.converted
		leaf.text = lt == LT_STRING || lt == 4 || lt == 12 || ct == CT_UTF8 || ct == 4 || ct == 19
		if n.def > 255 || n.rep > 255 {leaf.reason = "nested too deeply"}
		if leaf.reason != "" && reason^ == "" {reason^ = fmt.tprintf("%s: %s", full, leaf.reason)}
		m.nodes[idx].kind = .Leaf
		m.nodes[idx].leaf = len(m.leaves)
		append(&m.leaves, leaf)
		m.nodes[idx].leaf_hi = len(m.leaves)
		return idx
	}

	kids := make([dynamic]int, context.temp_allocator)
	for _ in 0 ..< se.num_children {
		if i^ >= len(schema) {break}
		append(&kids, _plan_node(schema, i, n.def, n.rep, full, depth + 1, m, reason))
	}
	nd := &m.nodes[idx]
	nd.kind = .Struct
	nd.kids_lo = len(m.kids)
	nd.kids_n = len(kids)
	append(&m.kids, ..kids[:])
	nd.leaf_hi = len(m.leaves)

	lt, ct := se.logical.kind, se.converted
	switch {
	case (lt == 3 || ct == 3) && len(kids) == 1 && m.nodes[kids[0]].repeated:
		// LIST: <list> repeated group list { element }, or the older layouts
		// where the repeated field is the element itself
		r := kids[0]
		rn := &m.nodes[r]
		elem := r
		if rn.kind != .Leaf &&
		   rn.kids_n == 1 &&
		   rn.name != "array" &&
		   rn.name != fmt.tprintf("%s_tuple", se.name) {
			elem = m.kids[rn.kids_lo]
		}
		nd.kind, nd.rnode, nd.elem = .List, r, elem
	case (lt == 2 || ct == 1 || ct == 2) && len(kids) == 1:
		// MAP: <map> repeated group key_value { key; value }
		r := kids[0]
		rn := &m.nodes[r]
		if rn.repeated && rn.kind != .Leaf && (rn.kids_n == 1 || rn.kids_n == 2) {
			nd.kind, nd.rnode = .Map, r
			nd.key = m.kids[rn.kids_lo]
			nd.value = m.kids[rn.kids_lo + 1] if rn.kids_n == 2 else -1
		}
	}
	return idx
}

// ---------------------------------------------------------------------------
// Leaf streams

// The levels and present values of one leaf of a nested field, over all row
// groups read
@(private)
Leaf_Stream :: struct {
	rep, def: [dynamic]u8,
	vals:     [dynamic]u8, // present values, back to back, as cells of leaf.out
	nvals:    int,
	full:     u8, // the leaf's highest definition level: a present value
	strings:  w.Column, // holds the bytes of String values
}

@(private)
_stream_destroy :: proc(s: ^Leaf_Stream) {
	delete(s.rep)
	delete(s.def)
	delete(s.vals)
	w.destroy_column(&s.strings)
}

// Bits that hold the levels 0 ..= max
@(private)
_level_width :: proc(max: int) -> int {
	return 0 if max == 0 else int(64 - intrinsics.count_leading_zeros(u64(max)))
}

// A data page of a nested leaf: levels appended to the stream, present values
// decoded after the earlier ones; st.pos counts the rows that start in it
@(private)
_nested_page :: proc(st: ^Chunk_State, ph: ^Page_Header, payload: []u8) -> bool {
	leaf, s := st.leaf, st.stream
	n: int
	enc: Encoding
	rep_src, def_src, data: []u8
	switch Page_Type(ph.type) {
	case .Index_Page:
		return true
	case .Dictionary_Page:
		return _read_page(st, ph, payload, 0)
	case .Data_Page:
		if !ph.has_data {return _fail("column %q: data page without header", leaf.name)}
		n = int(ph.data.num_values)
		enc = Encoding(ph.data.encoding)
		body, ok := _decompress(st, payload, int(ph.uncompressed_size))
		if !ok {return _fail("column %q: cannot decompress a data page", leaf.name)}
		off := 0
		section :: proc(body: []u8, off: ^int) -> ([]u8, bool) {
			if len(body) - off^ < 4 {return nil, false}
			ln := int(intrinsics.unaligned_load((^u32le)(&body[off^])))
			if ln > len(body) - off^ - 4 {return nil, false}
			out := body[off^ + 4:off^ + 4 + ln]
			off^ += 4 + ln
			return out, true
		}
		if leaf.max_rep > 0 {
			rep_src, ok = section(body, &off)
			if !ok {return _fail("column %q: corrupt repetition levels", leaf.name)}
		}
		if leaf.max_def > 0 {
			def_src, ok = section(body, &off)
			if !ok {return _fail("column %q: corrupt definition levels", leaf.name)}
		}
		data = body[off:]
	case .Data_Page_V2:
		if !ph.has_v2 {return _fail("column %q: data page without header", leaf.name)}
		v2 := ph.v2
		n = int(v2.num_values)
		enc = Encoding(v2.encoding)
		rl, dl := int(v2.rep_len), int(v2.def_len)
		if rl < 0 || dl < 0 || rl + dl > len(payload) {
			return _fail("column %q: corrupt level lengths", leaf.name)
		}
		rep_src, def_src = payload[:rl], payload[rl:rl + dl]
		data = payload[rl + dl:]
		if v2.is_compressed && st.codec != .Uncompressed {
			ok: bool
			data, ok = _decompress(st, data, int(ph.uncompressed_size) - rl - dl)
			if !ok {return _fail("column %q: cannot decompress a data page", leaf.name)}
		}
	case:
		return true // unknown page types are skipped
	}
	if n < 0 || n > 1 << 31 {return _fail("column %q: corrupt value count", leaf.name)}

	base := len(s.def)
	resize(&s.rep, base + n)
	resize(&s.def, base + n)
	reps, defs := s.rep[base:], s.def[base:]
	if !_rle_hybrid(rep_src, _level_width(leaf.max_rep), reps) {
		return _fail("column %q: corrupt repetition levels", leaf.name)
	}
	if !_rle_hybrid(def_src, _level_width(leaf.max_def), defs) {
		return _fail("column %q: corrupt definition levels", leaf.name)
	}
	nn, rows := 0, 0
	for k in 0 ..< n {
		if int(defs[k]) > leaf.max_def || int(reps[k]) > leaf.max_rep {
			return _fail("column %q: level out of range", leaf.name)
		}
		if int(defs[k]) == leaf.max_def {nn += 1}
		if reps[k] == 0 {rows += 1}
	}
	esz := w.type_size(leaf.out)
	resize(&s.vals, (s.nvals + nn) * esz)
	if nn > 0 && !_decode_values(st, enc, data, nn, &s.vals[s.nvals * esz]) {return false}
	s.nvals += nn
	st.pos += rows
	return true
}

// ---------------------------------------------------------------------------
// Assembly

@(private)
Assembler :: struct {
	m:       ^Meta,
	streams: []Leaf_Stream, // indexed by leaf index - lo
	cur, vi: []int, // per stream: next level entry, next present value
	lo:      int,
	sb:      ^strings.Builder,
	bad:     bool,
}

// A nested column being assembled: its rows are split into ranges that are
// assembled in parallel, each into its own text buffer, then copied into the
// column
@(private)
Nested_Column :: struct {
	top:     ^Top,
	streams: []Leaf_Stream,
	col:     ^w.Column,
	ranges:  []Row_Range,
}

@(private)
Row_Range :: struct {
	r0, r1:  int, // rows
	cur, vi: []int, // per stream: next level entry, next present value (from row r0 on)
	start:   []int, // cur and vi at row r0, for the check against the previous range
	text:    [dynamic]u8, // the rows' JSON text, back to back
	ends:    []int, // per row: end of its text in `text`; -1 for a null cell
	ok:      bool,
}

// Splits the `total` rows of a nested column into ranges of about
// `range_rows`, with each stream's position at every range start. false if
// the streams do not hold `total` rows.
@(private)
_plan_ranges :: proc(nc: ^Nested_Column, total, range_rows: int) -> bool {
	n := max(1, (total + range_rows - 1) / range_rows)
	nc.ranges = make([]Row_Range, n)
	for &rg, k in nc.ranges {
		rg.r0 = k * total / n
		rg.r1 = (k + 1) * total / n
		rg.cur = make([]int, len(nc.streams))
		rg.vi = make([]int, len(nc.streams))
		rg.start = make([]int, 2 * len(nc.streams))
		rg.ends = make([]int, rg.r1 - rg.r0)
	}
	for &s, k in nc.streams {
		// a row starts at every repetition level 0
		next, row, vals := 0, 0, 0
		for i in 0 ..< len(s.rep) {
			if s.rep[i] == 0 {
				for next < n && nc.ranges[next].r0 == row {
					nc.ranges[next].cur[k] = i
					nc.ranges[next].vi[k] = vals
					next += 1
				}
				row += 1
			}
			if s.def[i] == s.full {vals += 1}
		}
		if row != total || vals != s.nvals {
			return _fail("column %q: levels do not match the rows", nc.top.name)
		}
		for next < n {
			// empty trailing ranges (total = 0)
			nc.ranges[next].cur[k] = len(s.rep)
			nc.ranges[next].vi[k] = vals
			next += 1
		}
	}
	for &rg in nc.ranges {
		copy(rg.start, rg.cur)
		copy(rg.start[len(rg.cur):], rg.vi)
	}
	return true
}

// Assembles the rows of one range (a parallel task: it allocates only the
// text buffer, a dynamic array that keeps its allocator)
@(private)
_assemble_range :: proc(m: ^Meta, nc: ^Nested_Column, rg: ^Row_Range) {
	top := nc.top
	root := &m.nodes[top.node]
	a := Assembler {
		m       = m,
		streams = nc.streams,
		lo      = root.leaf_lo,
		cur     = rg.cur,
		vi      = rg.vi,
	}
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	a.sb = &sb
	for r in rg.r0 ..< rg.r1 {
		for &s, k in nc.streams {
			if a.cur[k] >= len(s.rep) || s.rep[a.cur[k]] != 0 {return}
		}
		switch {
		case root.repeated:
			_emit_list(&a, top.node, top.node)
		case root.optional && _peek_def(&a, top.node) < root.def:
			_skip(&a, top.node)
			rg.ends[r - rg.r0] = -1
			continue
		case:
			_emit_value(&a, top.node, false)
		}
		if a.bad {return}
		rg.ends[r - rg.r0] = strings.builder_len(sb)
	}
	rg.text = sb.buf
	sb.buf = nil
	rg.ok = true
}

// After every range is assembled: checks that each range ended where the
// next one starts, and copies the text into the column
@(private)
_finish_column :: proc(nc: ^Nested_Column, total: int) -> bool {
	bytes := 0
	for &rg, k in nc.ranges {
		if !rg.ok {return _fail("column %q: corrupt levels", nc.top.name)}
		for &s, j in nc.streams {
			end_cur, end_vi := len(s.rep), s.nvals
			if k + 1 < len(nc.ranges) {
				next := &nc.ranges[k + 1]
				end_cur, end_vi = next.start[j], next.start[len(nc.streams) + j]
			}
			// rg.cur / rg.vi were advanced by the assembly
			if rg.cur[j] != end_cur || rg.vi[j] != end_vi {
				return _fail("column %q: levels do not match the values", nc.top.name)
			}
		}
		bytes += len(rg.text)
	}
	space, ok := _string_space(nc.col, bytes)
	if !ok {return _fail("column %q: out of memory", nc.top.name)}
	cells := ([^]string)(nc.col.data)
	at := 0
	for &rg in nc.ranges {
		copy(space[at:], rg.text[:])
		start := 0
		for e, i in rg.ends {
			r := rg.r0 + i
			if e < 0 {
				nc.col.nulls[r] = true
				cells[r] = ""
				continue
			}
			cells[r] = string(space[at + start:at + e])
			start = e
		}
		at += len(rg.text)
	}
	nc.col.len = total
	return true
}

@(private)
_destroy_ranges :: proc(nc: ^Nested_Column) {
	for &rg in nc.ranges {
		delete(rg.cur)
		delete(rg.vi)
		delete(rg.start)
		delete(rg.text)
		delete(rg.ends)
	}
	delete(nc.ranges)
}

@(private)
_peek_def :: proc(a: ^Assembler, node: int) -> int {
	k := a.m.nodes[node].leaf_lo - a.lo
	if a.cur[k] >= len(a.streams[k].def) {
		a.bad = true
		return 0
	}
	return int(a.streams[k].def[a.cur[k]])
}

// Consumes the one level entry every leaf under `node` has for a field that
// is null or empty at this point
@(private)
_skip :: proc(a: ^Assembler, node: int) {
	n := &a.m.nodes[node]
	for l in n.leaf_lo ..< n.leaf_hi {
		k := l - a.lo
		if a.cur[k] >= len(a.streams[k].def) {
			a.bad = true
			return
		}
		a.cur[k] += 1
	}
}

// A field in its place inside its parent: a list if it repeats
@(private)
_emit_field :: proc(a: ^Assembler, node: int) {
	n := &a.m.nodes[node]
	if n.repeated {
		_emit_list(a, node, node)
	} else {
		_emit_value(a, node, n.optional)
	}
}

@(private)
_emit_value :: proc(a: ^Assembler, node: int, check_null: bool) {
	if a.bad {return}
	n := &a.m.nodes[node]
	if check_null && _peek_def(a, node) < n.def {
		_skip(a, node)
		strings.write_string(a.sb, "null")
		return
	}
	switch n.kind {
	case .Leaf:
		_emit_leaf(a, node)
	case .Struct:
		strings.write_byte(a.sb, '{')
		for c in 0 ..< n.kids_n {
			kid := a.m.kids[n.kids_lo + c]
			if c > 0 {strings.write_byte(a.sb, ',')}
			_json_string(a.sb, a.m.nodes[kid].name, true)
			strings.write_byte(a.sb, ':')
			_emit_field(a, kid)
		}
		strings.write_byte(a.sb, '}')
	case .List:
		_emit_list(a, n.rnode, n.elem)
	case .Map:
		_emit_map(a, node)
	}
}

// The elements of repeated field `r` at this point, each written as `elem`
// (r itself, or its only child)
@(private)
_emit_list :: proc(a: ^Assembler, r, elem: int) {
	rn := &a.m.nodes[r]
	if _peek_def(a, r) < rn.def {
		_skip(a, r)
		strings.write_string(a.sb, "[]")
		return
	}
	strings.write_byte(a.sb, '[')
	for first := true; !a.bad; first = false {
		if !first {strings.write_byte(a.sb, ',')}
		if elem == r {_emit_value(a, r, false)} else {_emit_field(a, elem)}
		if !_more(a, r) {break}
	}
	strings.write_byte(a.sb, ']')
}

// Whether repeated field `r` has another element at this point
@(private)
_more :: proc(a: ^Assembler, r: int) -> bool {
	rn := &a.m.nodes[r]
	k := rn.leaf_lo - a.lo
	s := &a.streams[k]
	if a.cur[k] >= len(s.rep) {return false}
	l := int(s.rep[a.cur[k]])
	if l > rn.rep {
		a.bad = true
		return false
	}
	return l == rn.rep
}

@(private)
_emit_map :: proc(a: ^Assembler, node: int) {
	n := &a.m.nodes[node]
	r := n.rnode
	if _peek_def(a, r) < a.m.nodes[r].def {
		_skip(a, r)
		strings.write_string(a.sb, "{}")
		return
	}
	key_sb := strings.builder_make()
	defer strings.builder_destroy(&key_sb)
	strings.write_byte(a.sb, '{')
	for first := true; !a.bad; first = false {
		if !first {strings.write_byte(a.sb, ',')}
		// the key: a string as it is, anything else as its JSON text
		out := a.sb
		strings.builder_reset(&key_sb)
		a.sb = &key_sb
		_emit_field(a, n.key)
		a.sb = out
		key := strings.to_string(key_sb)
		if len(key) > 0 && key[0] == '"' {
			strings.write_string(a.sb, key)
		} else {
			_json_string(a.sb, key, true)
		}
		strings.write_byte(a.sb, ':')
		if n.value >= 0 {_emit_field(a, n.value)} else {strings.write_string(a.sb, "null")}
		if !_more(a, r) {break}
	}
	strings.write_byte(a.sb, '}')
}

@(private)
_emit_leaf :: proc(a: ^Assembler, node: int) {
	n := &a.m.nodes[node]
	k := n.leaf_lo - a.lo
	s := &a.streams[k]
	leaf := &a.m.leaves[n.leaf]
	if a.cur[k] >= len(s.def) {
		a.bad = true
		return
	}
	present := int(s.def[a.cur[k]]) == leaf.max_def
	a.cur[k] += 1
	if !present {
		strings.write_string(a.sb, "null")
		return
	}
	i := a.vi[k]
	if i >= s.nvals {
		a.bad = true
		return
	}
	a.vi[k] += 1
	sb := a.sb
	vals := raw_data(s.vals)
	switch leaf.out {
	case .Int:
		v := ([^]int)(vals)[i]
		buf: [32]u8
		if leaf.unsigned {
			// JSON text holds UINT64 values above max(i64)
			strings.write_string(sb, strconv.write_uint(buf[:], u64(v), 10))
		} else {
			strings.write_string(sb, strconv.write_int(buf[:], i64(v), 10))
		}
	case .Float:
		v := ([^]f64)(vals)[i]
		if v != v || v > max(f64) || v < -max(f64) {
			strings.write_string(sb, "null") // JSON has no NaN / Inf
		} else {
			// as w.float_to_string (fmt's %v): shortest exact text, always a float
			buf: [386]u8
			text := strconv.write_float(buf[:], v, 'g', -1, 64)
			if len(text) > 0 && text[0] == '+' {text = text[1:]}
			strings.write_string(sb, text)
			whole := true
			for c in transmute([]u8)text {
				if c == '.' || c == 'e' || c == 'E' {whole = false}
			}
			if whole {strings.write_string(sb, ".0")}
		}
	case .Bool:
		strings.write_string(sb, "true" if ([^]bool)(vals)[i] else "false")
	case .String:
		_json_string(sb, ([^]string)(vals)[i], leaf.text || leaf.conv == .Uuid)
	case .Date:
		d := ([^]w.Date)(vals)[i]
		fmt.sbprintf(sb, `"%04d-%02d-%02d"`, d.year, d.month, d.day)
	case .Time:
		t := ([^]w.Time)(vals)[i]
		fb: [10]u8
		fmt.sbprintf(
			sb,
			`"%02d:%02d:%02d%s"`,
			t.hour,
			t.minute,
			t.second,
			w.fraction_string(t.nanosecond, &fb),
		)
	case .Datetime:
		d := ([^]w.Datetime)(vals)[i]
		fb: [10]u8
		fmt.sbprintf(
			sb,
			`"%04d-%02d-%02d %02d:%02d:%02d%s"`,
			d.year,
			d.month,
			d.day,
			d.hour,
			d.minute,
			d.second,
			w.fraction_string(d.nanosecond, &fb),
		)
	case .Invalid:
		strings.write_string(sb, "null")
	}
}

// `s` as a JSON string. Text is kept as UTF-8; binary (text = false) writes
// each byte of 0x80 and above as \u00XX.
@(private)
_json_string :: proc(sb: ^strings.Builder, s: string, text: bool) {
	HEX := "0123456789abcdef"
	strings.write_byte(sb, '"')
	for i in 0 ..< len(s) {
		ch := s[i]
		switch ch {
		case '"':
			strings.write_string(sb, `\"`)
		case '\\':
			strings.write_string(sb, `\\`)
		case '\n':
			strings.write_string(sb, `\n`)
		case '\r':
			strings.write_string(sb, `\r`)
		case '\t':
			strings.write_string(sb, `\t`)
		case:
			if ch < 0x20 || (ch >= 0x80 && !text) {
				strings.write_string(sb, `\u00`)
				strings.write_byte(sb, HEX[ch >> 4])
				strings.write_byte(sb, HEX[ch & 0xF])
			} else {
				strings.write_byte(sb, ch)
			}
		}
	}
	strings.write_byte(sb, '"')
}
