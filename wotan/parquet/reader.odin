package parquet

import "base:intrinsics"
import "base:runtime"
import "core:compress/deflate"
import lz4 "core:compress/lz4"
import "core:fmt"
import "core:math"
import "core:mem"
import vmem "core:mem/virtual"
import "core:os"

import w "../core"
import util "../util"
import zs "../zstd"

// Parquet reader: flat schemas (every top-level field a primitive column,
// required or optional) into a DataFrame.
//
// Types:
//   BOOLEAN                                   -> Bool
//   INT32, INT64 (also INT(8/16/32/64), signed or unsigned) -> Int
//   FLOAT, DOUBLE, FLOAT16                    -> Float
//   DECIMAL (on INT32, INT64, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY) -> Float
//   BYTE_ARRAY (STRING, ENUM, JSON or plain binary) -> String
//   FIXED_LEN_BYTE_ARRAY                      -> String (raw bytes; UUID as 8-4-4-4-12 text)
//   DATE                                      -> Date
//   TIME (millis, micros, nanos)              -> Time
//   TIMESTAMP (millis, micros, nanos), INT96  -> Datetime (UTC for instants)
// Datetime and Time hold whole seconds; sub-second parts are dropped
// (rounded towards the past).
//
// Encodings: PLAIN, PLAIN_DICTIONARY / RLE_DICTIONARY, RLE (booleans),
// DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY and
// BYTE_STREAM_SPLIT, in data pages v1 and v2. Codecs: uncompressed, Snappy,
// gzip, Zstandard, LZ4_RAW and the Hadoop-framed LZ4. Anything else (nested
// columns, Brotli or LZO, encrypted files) makes the read fail with a message
// on stderr; a read never returns partial or guessed data.
//
// Only the column chunks of the requested columns are read from disk, and the
// columns are decoded in parallel (util.parallel_max_threads caps the
// threads; 1 decodes on the calling thread).

// The Parquet file at `path` as a DataFrame. `columns` selects (and orders)
// the columns to load; nil loads all of them. On failure the reason is
// printed to stderr and ok is false.
read_file :: proc(
	path: string,
	columns: []string = nil,
	allocator := context.allocator,
) -> (
	df: w.DataFrame,
	ok: bool,
) {
	f, err := os.open(path)
	if err != nil {
		_fail("cannot open %q", path)
		return
	}
	defer os.close(f)
	size, serr := os.file_size(f)
	if serr != nil {
		_fail("cannot read the size of %q", path)
		return
	}
	src := Source {
		file = f,
		size = size,
	}
	return _read(&src, columns, allocator)
}

// The Parquet file held in `data` as a DataFrame (see read_file)
read_bytes :: proc(
	data: []u8,
	columns: []string = nil,
	allocator := context.allocator,
) -> (
	df: w.DataFrame,
	ok: bool,
) {
	src := Source {
		bytes = data,
		size  = i64(len(data)),
	}
	return _read(&src, columns, allocator)
}

// ---------------------------------------------------------------------------

@(private)
Source :: struct {
	bytes: []u8,
	file:  ^os.File,
	size:  i64,
}

// Bytes [off, off+n) of the source: a slice of the input for read_bytes, the
// bytes read into `buf` for read_file
@(private)
_source_read :: proc(src: ^Source, off, n: i64, buf: ^[dynamic]u8) -> ([]u8, bool) {
	if off < 0 || n < 0 || off > src.size || n > src.size - off {return nil, false}
	if src.file == nil {return src.bytes[off:off + n], true}
	resize(buf, int(n))
	got, err := os.read_at(src.file, buf[:], off)
	if err != nil || got != int(n) {return nil, false}
	return buf[:], true
}

@(private)
_fail :: proc(format: string, args: ..any) -> bool {
	fmt.eprint("parquet: ")
	fmt.eprintfln(format, ..args)
	return false
}

// How the physical values of a leaf become the cells of its column
@(private)
Conv :: enum u8 {
	Bool,
	Int32,
	UInt32,
	Int64,
	Float32,
	Float64,
	Float16,
	Date32,
	Time_Millis,
	Time_Micros,
	Time_Nanos,
	Ts_Millis,
	Ts_Micros,
	Ts_Nanos,
	Int96,
	Dec_Int32,
	Dec_Int64,
	Dec_Bytes,
	Bytes,
	Uuid,
}

@(private)
Leaf :: struct {
	name:        string,
	chunk:       int, // index of the leaf among all leaves (the column chunk)
	physical:    Physical_Type,
	type_length: int,
	max_def:     int,
	conv:        Conv,
	out:         w.ColumnType,
	scale:       i32,
	nested:      bool,
	reason:      string, // why the leaf cannot be read; "" if it can
}

@(private)
_read :: proc(
	src: ^Source,
	columns: []string,
	allocator: mem.Allocator,
) -> (
	df: w.DataFrame,
	ok: bool,
) {
	buf := make([dynamic]u8)
	defer delete(buf)

	// footer: <metadata> <u32 metadata length> "PAR1"
	if src.size < 12 {
		_fail("not a Parquet file (too short)")
		return
	}
	tail, tok := _source_read(src, src.size - 8, 8, &buf)
	if !tok {
		_fail("cannot read the file footer")
		return
	}
	magic := string(tail[4:8])
	if magic == "PARE" {
		_fail("encrypted Parquet files are not supported")
		return
	}
	if magic != "PAR1" {
		_fail("not a Parquet file (no PAR1 magic)")
		return
	}
	meta_len := i64(intrinsics.unaligned_load((^u32le)(&tail[0])))
	if meta_len <= 0 || meta_len > src.size - 12 {
		_fail("corrupt footer length")
		return
	}
	footer_buf := make([dynamic]u8)
	defer delete(footer_buf)
	footer, fok := _source_read(src, src.size - 8 - meta_len, meta_len, &footer_buf)
	if !fok {
		_fail("cannot read the file metadata")
		return
	}
	t := Thrift {
		data = footer,
	}
	fm := _read_file_meta(&t)
	defer _destroy_file_meta(&fm)
	if t.bad || t.pos != len(footer) || len(fm.schema) == 0 {
		_fail("corrupt file metadata")
		return
	}

	leaves, tops := _plan_schema(fm.schema[:]) or_return
	defer delete(leaves)
	defer delete(tops)

	// selection: indices into `tops`
	picked := make([dynamic]int)
	defer delete(picked)
	if columns == nil {
		for i in 0 ..< len(tops) {append(&picked, i)}
	} else {
		for name in columns {
			found := -1
			for top, i in tops {
				if top.name == name {
					found = i
					break
				}
			}
			if found < 0 {
				_fail("no column %q in the file", name)
				return
			}
			for p in picked {
				if p == found {
					_fail("column %q requested twice", name)
					return
				}
			}
			append(&picked, found)
		}
	}
	for p in picked {
		top := tops[p]
		if top.leaf < 0 {
			_fail(
				"column %q is nested (a list, map or struct), which is not supported; pass `columns` to load the others",
				top.name,
			)
			return
		}
		leaf := &leaves[top.leaf]
		if leaf.reason != "" {
			_fail("column %q: %s", leaf.name, leaf.reason)
			return
		}
		for q in picked {
			if q != p && tops[q].name == top.name {
				_fail("the file has two columns named %q", top.name)
				return
			}
		}
	}

	total := 0
	for rg in fm.row_groups {
		if rg.num_rows < 0 || rg.num_rows > fm.num_rows {
			_fail("corrupt row count")
			return
		}
		if len(rg.columns) != len(leaves) {
			_fail(
				"corrupt row group (%d column chunks for %d columns)",
				len(rg.columns),
				len(leaves),
			)
			return
		}
		total += int(rg.num_rows)
	}
	if i64(total) != fm.num_rows {
		_fail("corrupt row count")
		return
	}

	cols := make([dynamic]w.Column, 0, len(picked))
	defer delete(cols)
	success := false
	defer if !success {
		for &c in cols {w.destroy_column(&c)}
	}
	jobs := make([]Leaf_Job, len(picked))
	defer delete(jobs)
	for p, j in picked {
		leaf := &leaves[tops[p].leaf]
		col := w.column_new("", leaf.out, total, allocator)
		w.column_own_name(&col, leaf.name)
		append(&cols, col)
		jobs[j] = Leaf_Job {
			leaf = leaf,
		}
	}
	for &job, j in jobs {job.col = &cols[j]}
	work := Read_Work {
		src   = src,
		fm    = &fm,
		jobs  = jobs,
		total = total,
	}
	util.parallel_for(len(jobs), &work, _read_leaves, tasks = len(jobs))
	for job in jobs {
		if !job.ok {return}
	}

	df = w.dataframe_new(allocator)
	for c in cols {w.add_column(&df, c)}
	df.rows = total
	success = true
	return df, true
}

@(private)
Leaf_Job :: struct {
	leaf: ^Leaf,
	col:  ^w.Column,
	ok:   bool,
}

@(private)
Read_Work :: struct {
	src:   ^Source,
	fm:    ^File_Meta,
	jobs:  []Leaf_Job,
	total: int,
	next:  int, // next job to take (atomic)
}

// A parallel task: takes columns until none are left, so a long column does
// not hold up a whole range of short ones. Scratch memory comes from the heap
// (thread-safe); the columns were allocated before the threads started.
@(private)
_read_leaves :: proc(data: rawptr, begin, end: int) {
	work := (^Read_Work)(data)
	context.allocator = runtime.heap_allocator()
	for {
		j := intrinsics.atomic_add(&work.next, 1)
		if j >= len(work.jobs) {break}
		job := &work.jobs[j]
		job.ok = _read_leaf(work.src, work.fm, job.leaf, job.col, work.total)
	}
}

@(private)
Top :: struct {
	name: string,
	leaf: int, // index into the leaves, -1 for a nested field
}

// The leaves (primitive columns, in column-chunk order) and the top-level
// fields of the schema
@(private)
_plan_schema :: proc(
	schema: []Schema_Element,
) -> (
	leaves: [dynamic]Leaf,
	tops: [dynamic]Top,
	ok: bool,
) {
	leaves = make([dynamic]Leaf)
	tops = make([dynamic]Top)
	n_top := int(schema[0].num_children)
	i := 1
	for _ in 0 ..< n_top {
		if i >= len(schema) {break}
		se := schema[i]
		if se.num_children > 0 || se.repetition == 2 {
			append(&tops, Top{name = se.name, leaf = -1})
			// count its leaves so chunk indices stay aligned
			pending := 1
			for pending > 0 && i < len(schema) {
				pending += int(schema[i].num_children) - 1
				if schema[i].num_children == 0 {
					append(
						&leaves,
						Leaf{name = schema[i].name, chunk = len(leaves), nested = true},
					)
				}
				i += 1
			}
			continue
		}
		leaf := _plan_leaf(se)
		leaf.chunk = len(leaves)
		append(&tops, Top{name = se.name, leaf = len(leaves)})
		append(&leaves, leaf)
		i += 1
	}
	if len(tops) != n_top || i != len(schema) {
		_fail("corrupt schema")
		delete(leaves)
		delete(tops)
		return nil, nil, false
	}
	return leaves, tops, true
}

@(private)
_plan_leaf :: proc(se: Schema_Element) -> Leaf {
	leaf := Leaf {
		name        = se.name,
		physical    = Physical_Type(se.type),
		type_length = int(se.type_length),
		max_def     = 1 if se.repetition == 1 else 0,
	}
	lt := se.logical
	ct := se.converted
	is_decimal := lt.kind == LT_DECIMAL || ct == CT_DECIMAL
	if is_decimal {
		leaf.scale = lt.scale if lt.kind == LT_DECIMAL else se.scale
	}
	unsigned := (lt.kind == LT_INTEGER && !lt.signed) || (ct >= CT_UINT_8 && ct <= CT_UINT_64)

	switch leaf.physical {
	case .Boolean:
		leaf.conv, leaf.out = .Bool, .Bool
	case .Int32:
		switch {
		case is_decimal:
			leaf.conv, leaf.out = .Dec_Int32, .Float
		case lt.kind == LT_DATE || ct == CT_DATE:
			leaf.conv, leaf.out = .Date32, .Date
		case lt.kind == LT_TIME || ct == CT_TIME_MILLIS:
			leaf.conv, leaf.out = .Time_Millis, .Time
		case unsigned:
			leaf.conv, leaf.out = .UInt32, .Int
		case:
			leaf.conv, leaf.out = .Int32, .Int
		}
	case .Int64:
		switch {
		case is_decimal:
			leaf.conv, leaf.out = .Dec_Int64, .Float
		case lt.kind == LT_TIMESTAMP:
			leaf.out = .Datetime
			switch lt.unit {
			case .Millis:
				leaf.conv = .Ts_Millis
			case .Micros:
				leaf.conv = .Ts_Micros
			case .Nanos, .None:
				leaf.conv = .Ts_Nanos
			}
		case ct == CT_TIMESTAMP_MILLIS:
			leaf.conv, leaf.out = .Ts_Millis, .Datetime
		case ct == CT_TIMESTAMP_MICROS:
			leaf.conv, leaf.out = .Ts_Micros, .Datetime
		case lt.kind == LT_TIME:
			leaf.conv, leaf.out = .Time_Nanos if lt.unit == .Nanos else .Time_Micros, .Time
		case ct == CT_TIME_MICROS:
			leaf.conv, leaf.out = .Time_Micros, .Time
		case:
			// UINT64 values above max(i64) wrap to negative numbers
			leaf.conv, leaf.out = .Int64, .Int
		}
	case .Int96:
		leaf.conv, leaf.out = .Int96, .Datetime
	case .Float:
		leaf.conv, leaf.out = .Float32, .Float
	case .Double:
		leaf.conv, leaf.out = .Float64, .Float
	case .Byte_Array:
		if is_decimal {
			leaf.conv, leaf.out = .Dec_Bytes, .Float
		} else {
			leaf.conv, leaf.out = .Bytes, .String
		}
	case .Fixed_Len_Byte_Array:
		switch {
		case leaf.type_length <= 0:
			leaf.reason = "FIXED_LEN_BYTE_ARRAY without a length"
		case is_decimal:
			leaf.conv, leaf.out = .Dec_Bytes, .Float
		case lt.kind == LT_FLOAT16:
			leaf.conv, leaf.out = .Float16, .Float
			if leaf.type_length != 2 {leaf.reason = "FLOAT16 with a length other than 2"}
		case lt.kind == LT_UUID && leaf.type_length == 16:
			leaf.conv, leaf.out = .Uuid, .String
		case:
			leaf.conv, leaf.out = .Bytes, .String
		}
	case:
		leaf.reason = "unknown physical type"
	}
	if is_decimal && leaf.scale < 0 {leaf.reason = "DECIMAL with a negative scale"}
	return leaf
}

// ---------------------------------------------------------------------------
// Column chunks and pages

@(private)
Chunk_State :: struct {
	leaf:     ^Leaf,
	col:      ^w.Column,
	pos:      int, // rows written so far
	codec:    Codec,
	dict:     []u8, // decoded dictionary, in the column's cell type
	dict_len: int,
	scratch:  [dynamic]u8, // decompressed page
	levels:   [dynamic]u8, // definition levels of a page
	idx:      [dynamic]u32, // dictionary indices of a page
	plain:    [dynamic]u8, // values re-encoded as PLAIN (delta and split encodings)
	ints:     [dynamic]i64, // delta-decoded integers
	ints2:    [dynamic]i64, // second integer run (DELTA_BYTE_ARRAY suffix lengths)
	bytes:    [dynamic]u8, // rebuilt byte-array values, back to back
}

@(private)
_read_leaf :: proc(src: ^Source, fm: ^File_Meta, leaf: ^Leaf, col: ^w.Column, total: int) -> bool {
	st := Chunk_State {
		leaf = leaf,
		col  = col,
	}
	defer {
		delete(st.dict)
		delete(st.scratch)
		delete(st.levels)
		delete(st.idx)
		delete(st.plain)
		delete(st.ints)
		delete(st.ints2)
		delete(st.bytes)
	}
	chunk_buf := make([dynamic]u8)
	defer delete(chunk_buf)

	for &rg in fm.row_groups {
		if leaf.chunk >= len(rg.columns) {return _fail("corrupt row group (missing column chunk)")}
		cc := rg.columns[leaf.chunk]
		if cc.file_path != "" {
			return _fail(
				"column %q lives in another file (%s), which is not supported",
				leaf.name,
				cc.file_path,
			)
		}
		if !cc.has_meta {return _fail("column %q: chunk without metadata", leaf.name)}
		cm := cc.meta
		if Physical_Type(cm.type) != leaf.physical {
			return _fail("column %q: chunk type differs from the schema", leaf.name)
		}
		st.codec = Codec(cm.codec)
		#partial switch st.codec {
		case .Uncompressed, .Snappy, .Gzip, .Lz4, .Lz4_Raw, .Zstd:
		case:
			return _fail(
				"column %q uses %v compression, which is not supported",
				leaf.name,
				st.codec,
			)
		}
		start := cm.data_page_offset
		if cm.has_dictionary_offset &&
		   cm.dictionary_page_offset > 0 &&
		   cm.dictionary_page_offset < start {
			start = cm.dictionary_page_offset
		}
		chunk, cok := _source_read(src, start, cm.total_compressed_size, &chunk_buf)
		if !cok {return _fail("column %q: chunk outside the file", leaf.name)}

		delete(st.dict)
		st.dict, st.dict_len = nil, 0
		end_pos := st.pos + int(rg.num_rows)
		if end_pos > total {return _fail("corrupt row counts")}
		p := 0
		for st.pos < end_pos {
			if p >= len(chunk) {return _fail("column %q: chunk ends early", leaf.name)}
			t := Thrift {
				data = chunk[p:],
			}
			ph := _read_page_header(&t)
			if t.bad || ph.compressed_size < 0 || ph.uncompressed_size < 0 {
				return _fail("column %q: corrupt page header", leaf.name)
			}
			p += t.pos
			if int(ph.compressed_size) > len(chunk) - p {
				return _fail("column %q: page outside the chunk", leaf.name)
			}
			payload := chunk[p:p + int(ph.compressed_size)]
			p += int(ph.compressed_size)
			if !_read_page(&st, &ph, payload, end_pos) {return false}
		}
	}
	if st.pos != total {return _fail("column %q: %d of %d rows", leaf.name, st.pos, total)}
	col.len = total
	return true
}

@(private)
_decompress :: proc(st: ^Chunk_State, src: []u8, n: int) -> ([]u8, bool) {
	switch st.codec {
	case .Uncompressed:
		if len(src) < n {return nil, false}
		return src[:n], true
	case .Snappy:
		resize(&st.scratch, n)
		if !snappy_decompress(st.scratch[:], src) {return nil, false}
		return st.scratch[:], true
	case .Gzip:
		clear(&st.scratch)
		if deflate.inflate_gzip_from_bytes(&st.scratch, src, expected_output_size = n) !=
		   nil {return nil, false}
		if len(st.scratch) != n {return nil, false}
		return st.scratch[:], true
	case .Lz4_Raw:
		resize(&st.scratch, n)
		got, err := lz4.decompress_block_to_slice(st.scratch[:], src)
		if err != nil || got != n {return nil, false}
		return st.scratch[:], true
	case .Lz4:
		// Hadoop framing: blocks of <u32be raw size> <u32be packed size> <data>;
		// some writers used a bare LZ4 block instead
		resize(&st.scratch, n)
		if _lz4_hadoop(st.scratch[:], src) {return st.scratch[:], true}
		got, err := lz4.decompress_block_to_slice(st.scratch[:], src)
		if err != nil || got != n {return nil, false}
		return st.scratch[:], true
	case .Zstd:
		resize(&st.scratch, n)
		got, ok := zs.decompress(st.scratch[:], src)
		if !ok || got != n {return nil, false}
		return st.scratch[:], true
	case .Lzo, .Brotli:
	}
	return nil, false
}

@(private)
_lz4_hadoop :: proc(dst, src: []u8) -> bool {
	d, s := 0, 0
	for s < len(src) {
		if s + 8 > len(src) {return false}
		raw := int(
			u32(src[s]) << 24 | u32(src[s + 1]) << 16 | u32(src[s + 2]) << 8 | u32(src[s + 3]),
		)
		packed := int(
			u32(src[s + 4]) << 24 | u32(src[s + 5]) << 16 | u32(src[s + 6]) << 8 | u32(src[s + 7]),
		)
		s += 8
		if packed > len(src) - s || raw > len(dst) - d {return false}
		got, err := lz4.decompress_block_to_slice(dst[d:d + raw], src[s:s + packed])
		if err != nil || got != raw {return false}
		d += raw
		s += packed
	}
	return d == len(dst)
}

@(private)
_read_page :: proc(st: ^Chunk_State, ph: ^Page_Header, payload: []u8, end_pos: int) -> bool {
	leaf := st.leaf
	switch Page_Type(ph.type) {
	case .Index_Page:
		return true
	case .Dictionary_Page:
		if !ph.has_dict {return _fail("column %q: dictionary page without header", leaf.name)}
		enc := Encoding(ph.dict.encoding)
		if enc != .Plain && enc != .Plain_Dictionary {
			return _fail("column %q: dictionary in %v encoding", leaf.name, enc)
		}
		data, ok := _decompress(st, payload, int(ph.uncompressed_size))
		if !ok {return _fail("column %q: cannot decompress the dictionary page", leaf.name)}
		n := int(ph.dict.num_values)
		if n < 0 {return _fail("column %q: corrupt dictionary", leaf.name)}
		delete(st.dict)
		st.dict = make([]u8, max(n, 1) * w.type_size(leaf.out))
		st.dict_len = n
		if !_plain(st, data, n, raw_data(st.dict)) {
			return _fail("column %q: corrupt dictionary values", leaf.name)
		}
		return true
	case .Data_Page:
		if !ph.has_data {return _fail("column %q: data page without header", leaf.name)}
		n := int(ph.data.num_values)
		if n < 0 ||
		   n > end_pos - st.pos {return _fail("column %q: too many values in a page", leaf.name)}
		data, ok := _decompress(st, payload, int(ph.uncompressed_size))
		if !ok {return _fail("column %q: cannot decompress a data page", leaf.name)}
		off := 0
		levels: []u8
		if leaf.max_def > 0 {
			if len(data) < 4 {return _fail("column %q: corrupt definition levels", leaf.name)}
			ln := int(intrinsics.unaligned_load((^u32le)(&data[0])))
			if ln > len(data) - 4 {return _fail("column %q: corrupt definition levels", leaf.name)}
			resize(&st.levels, n)
			if !_rle_hybrid(data[4:4 + ln], 1, st.levels[:]) {
				return _fail("column %q: corrupt definition levels", leaf.name)
			}
			levels = st.levels[:]
			off = 4 + ln
		}
		return _values(st, Encoding(ph.data.encoding), data[off:], n, levels)
	case .Data_Page_V2:
		if !ph.has_v2 {return _fail("column %q: data page without header", leaf.name)}
		v2 := ph.v2
		n := int(v2.num_values)
		if n < 0 ||
		   n > end_pos - st.pos {return _fail("column %q: too many values in a page", leaf.name)}
		rl, dl := int(v2.rep_len), int(v2.def_len)
		if rl < 0 || dl < 0 || rl + dl > len(payload) {
			return _fail("column %q: corrupt level lengths", leaf.name)
		}
		levels: []u8
		if leaf.max_def > 0 {
			resize(&st.levels, n)
			if !_rle_hybrid(payload[rl:rl + dl], 1, st.levels[:]) {
				return _fail("column %q: corrupt definition levels", leaf.name)
			}
			levels = st.levels[:]
		}
		body := payload[rl + dl:]
		data := body
		if v2.is_compressed && st.codec != .Uncompressed {
			ok: bool
			data, ok = _decompress(st, body, int(ph.uncompressed_size) - rl - dl)
			if !ok {return _fail("column %q: cannot decompress a data page", leaf.name)}
		}
		return _values(st, Encoding(v2.encoding), data, n, levels)
	}
	return true // unknown page types are skipped
}

// Decodes the `n` cells of a data page (nulls where the definition level is
// below the maximum) into the column at st.pos
@(private)
_values :: proc(st: ^Chunk_State, enc: Encoding, data: []u8, n: int, levels: []u8) -> bool {
	leaf, col := st.leaf, st.col
	nn := n
	if levels != nil {
		nn = 0
		for l in levels {nn += int(l)}
	}
	esz := w.type_size(leaf.out)
	dst := rawptr(uintptr(col.data) + uintptr(st.pos * esz))

	#partial switch enc {
	case .Plain:
		if !_plain(st, data, nn, dst) {return _fail("column %q: corrupt values", leaf.name)}
	case .Plain_Dictionary, .Rle_Dictionary:
		if st.dict ==
		   nil {return _fail("column %q: dictionary indices without a dictionary", leaf.name)}
		if nn > 0 {
			if len(data) < 1 {return _fail("column %q: corrupt dictionary indices", leaf.name)}
			bw := int(data[0])
			if bw > 32 {return _fail("column %q: corrupt dictionary indices", leaf.name)}
			resize(&st.idx, nn)
			if !_rle_hybrid(data[1:], bw, st.idx[:]) {
				return _fail("column %q: corrupt dictionary indices", leaf.name)
			}
			if !_gather(leaf.out, dst, raw_data(st.dict), st.dict_len, st.idx[:]) {
				return _fail("column %q: dictionary index out of range", leaf.name)
			}
		}
	case .Rle:
		if leaf.physical !=
		   .Boolean {return _fail("column %q: RLE values on a non-boolean column", leaf.name)}
		if len(data) < 4 {return _fail("column %q: corrupt values", leaf.name)}
		ln := int(intrinsics.unaligned_load((^u32le)(&data[0])))
		if ln > len(data) - 4 {return _fail("column %q: corrupt values", leaf.name)}
		out := ([^]u8)(dst)[:nn] // bool cells are bytes holding 0 or 1
		if !_rle_hybrid(
			data[4:4 + ln],
			1,
			out,
		) {return _fail("column %q: corrupt values", leaf.name)}
	case .Delta_Binary_Packed, .Delta_Length_Byte_Array, .Delta_Byte_Array, .Byte_Stream_Split:
		if nn > 0 && !_decode_other(st, enc, data, nn, dst) {return false}
	case:
		return _fail("column %q uses the %v encoding, which is not supported", leaf.name, enc)
	}
	if nn < n {_spread(leaf.out, col, st.pos, n, nn, levels)}
	st.pos += n
	return true
}

// The dense values sit at the front of the page's cells; move each to its row
// (back to front, so nothing is overwritten before it moves) and mark nulls
@(private)
_spread :: proc(out: w.ColumnType, col: ^w.Column, base, n, nn: int, levels: []u8) {
	spread :: proc(cells: [^]$T, nulls: []bool, base, n, nn: int, levels: []u8) {
		j := nn - 1
		for i := n - 1; i >= 0; i -= 1 {
			if levels[i] != 0 {
				cells[base + i] = cells[base + j]
				j -= 1
			} else {
				cells[base + i] = {}
				nulls[base + i] = true
			}
		}
	}
	#partial switch out {
	case .Int:
		spread(([^]int)(col.data), col.nulls, base, n, nn, levels)
	case .Float:
		spread(([^]f64)(col.data), col.nulls, base, n, nn, levels)
	case .Bool:
		spread(([^]bool)(col.data), col.nulls, base, n, nn, levels)
	case .String:
		spread(([^]string)(col.data), col.nulls, base, n, nn, levels)
	case .Date:
		spread(([^]w.Date)(col.data), col.nulls, base, n, nn, levels)
	case .Time:
		spread(([^]w.Time)(col.data), col.nulls, base, n, nn, levels)
	case .Datetime:
		spread(([^]w.Datetime)(col.data), col.nulls, base, n, nn, levels)
	}
}

@(private)
_gather :: proc(out: w.ColumnType, dst, dict: rawptr, dict_len: int, idx: []u32) -> bool {
	gather :: proc(dst: [^]$T, dict: [^]T, dict_len: int, idx: []u32) -> bool #no_bounds_check {
		for k, i in idx {
			if int(k) >= dict_len {return false}
			dst[i] = dict[k]
		}
		return true
	}
	#partial switch out {
	case .Int:
		return gather(([^]int)(dst), ([^]int)(dict), dict_len, idx)
	case .Float:
		return gather(([^]f64)(dst), ([^]f64)(dict), dict_len, idx)
	case .Bool:
		return gather(([^]bool)(dst), ([^]bool)(dict), dict_len, idx)
	case .String:
		return gather(([^]string)(dst), ([^]string)(dict), dict_len, idx)
	case .Date:
		return gather(([^]w.Date)(dst), ([^]w.Date)(dict), dict_len, idx)
	case .Time:
		return gather(([^]w.Time)(dst), ([^]w.Time)(dict), dict_len, idx)
	case .Datetime:
		return gather(([^]w.Datetime)(dst), ([^]w.Datetime)(dict), dict_len, idx)
	}
	return false
}

// RLE / bit-packing hybrid: runs of <varint header> where an even header
// repeats one value (header>>1 times, stored in ceil(bw/8) bytes) and an odd
// header starts (header>>1) groups of 8 bit-packed values, LSB first. Fills
// all of `out`; false if the data runs out first.
@(private)
_rle_hybrid :: proc(src: []u8, bw: int, out: []$T) -> bool #no_bounds_check {
	n := len(out)
	if bw == 0 {
		for i in 0 ..< n {out[i] = 0}
		return true
	}
	mask := u64(1) << uint(bw) - 1
	vbytes := (bw + 7) / 8
	s, o := 0, 0
	for o < n {
		hdr: u64
		shift: uint
		for {
			if s >= len(src) {return false}
			b := src[s]
			s += 1
			hdr |= u64(b & 0x7f) << shift
			if b & 0x80 == 0 {break}
			shift += 7
			if shift > 63 {return false}
		}
		if hdr & 1 == 0 {
			count := int(hdr >> 1)
			if vbytes > len(src) - s {return false}
			v: u64
			for k in 0 ..< vbytes {v |= u64(src[s + k]) << uint(8 * k)}
			s += vbytes
			if v > mask {return false}
			count = min(count, n - o)
			val := T(v)
			for k in 0 ..< count {out[o + k] = val}
			o += count
		} else {
			groups := int(hdr >> 1)
			nbytes := groups * bw
			if groups < 0 || nbytes > len(src) - s {return false}
			count := min(groups * 8, n - o)
			run := src[s:s + nbytes]
			if bw == 1 {
				for k in 0 ..< count {out[o + k] = T((run[k >> 3] >> uint(k & 7)) & 1)}
			} else {
				for k in 0 ..< count {
					bit := k * bw
					byte_i := bit >> 3
					v: u64
					if byte_i + 8 <= len(run) {
						v = u64(intrinsics.unaligned_load((^u64le)(&run[byte_i])))
					} else {
						for q in 0 ..< len(run) - byte_i {v |= u64(run[byte_i + q]) << uint(8 * q)}
					}
					out[o + k] = T((v >> uint(bit & 7)) & mask)
				}
			}
			o += count
			s += nbytes
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// PLAIN values

@(private)
_pow10 :: proc(scale: i32) -> f64 {
	return math.pow(10, f64(scale))
}

@(private)
_ts :: #force_inline proc(v: i64, per_second: i64) -> w.Datetime {
	s := v / per_second
	if v % per_second < 0 {s -= 1} 	// floor: 1969-12-31 23:59:59.5 stays in that second
	return w.datetime_from_unix(s)
}

@(private)
_time_of_day :: #force_inline proc(v: i64, per_second: i64) -> w.Time {
	s := v / per_second
	if v % per_second < 0 {s -= 1}
	return w.Time{i32(s / 3600), i32((s % 3600) / 60), i32(s % 60)}
}

// Big-endian two's complement integer of 1..16 bytes (decimal storage)
@(private)
_be_int :: proc(b: []u8) -> f64 {
	if len(b) == 0 {return 0}
	if len(b) <= 16 {
		v: i128 = i128(i8(b[0]))
		for k in 1 ..< len(b) {v = v << 8 | i128(b[k])}
		return f64(v)
	}
	v := f64(i8(b[0]))
	for k in 1 ..< len(b) {v = v * 256 + f64(b[k])}
	return v
}

@(private)
_half :: proc(h: u16) -> f64 {
	sign := f64(-1) if h & 0x8000 != 0 else 1
	e := int((h >> 10) & 0x1f)
	m := f64(h & 0x3ff)
	switch e {
	case 0:
		return sign * m * math.pow(f64(2), -24)
	case 31:
		return sign * math.inf_f64(1) if m == 0 else math.nan_f64()
	}
	return sign * (1 + m / 1024) * math.pow(f64(2), f64(e - 15))
}

// Space for `n` bytes in the column's string arena
@(private)
_string_space :: proc(col: ^w.Column, n: int) -> ([]u8, bool) {
	if n == 0 {return nil, true}
	if col.arena.curr_block == nil {
		if vmem.arena_init_growing(&col.arena) != nil {return nil, false}
	}
	b, err := vmem.arena_alloc(&col.arena, uint(n), 1)
	return b, err == nil
}

// Decodes `n` PLAIN values from `src` into `dst` (cells of the leaf's column
// type); false if `src` is too short
@(private)
_plain :: proc(st: ^Chunk_State, src: []u8, n: int, dst: rawptr) -> bool #no_bounds_check {
	leaf := st.leaf
	if n == 0 {return true}
	p := raw_data(src)
	switch leaf.conv {
	case .Bool:
		if len(src) < (n + 7) / 8 {return false}
		out := ([^]bool)(dst)
		for i in 0 ..< n {out[i] = (src[i >> 3] >> uint(i & 7)) & 1 != 0}
	case .Int32, .UInt32, .Date32, .Time_Millis, .Dec_Int32:
		if len(src) < 4 * n {return false}
		vals := ([^]i32le)(p)
		#partial switch leaf.conv {
		case .Int32:
			out := ([^]int)(dst)
			for i in 0 ..< n {out[i] = int(intrinsics.unaligned_load(&vals[i]))}
		case .UInt32:
			out := ([^]int)(dst)
			for i in 0 ..< n {out[i] = int(u32(intrinsics.unaligned_load(&vals[i])))}
		case .Date32:
			out := ([^]w.Date)(dst)
			for i in 0 ..< n {out[i] = w.civil_from_days(i64(intrinsics.unaligned_load(&vals[i])))}
		case .Time_Millis:
			out := ([^]w.Time)(dst)
			for i in 0 ..< n {out[i] = _time_of_day(i64(intrinsics.unaligned_load(&vals[i])), 1000)}
		case .Dec_Int32:
			out := ([^]f64)(dst)
			d := _pow10(leaf.scale)
			for i in 0 ..< n {out[i] = f64(intrinsics.unaligned_load(&vals[i])) / d}
		}
	case .Int64, .Time_Micros, .Time_Nanos, .Ts_Millis, .Ts_Micros, .Ts_Nanos, .Dec_Int64:
		if len(src) < 8 * n {return false}
		vals := ([^]i64le)(p)
		#partial switch leaf.conv {
		case .Int64:
			mem.copy(dst, p, 8 * n)
		case .Time_Micros:
			out := ([^]w.Time)(dst)
			for i in 0 ..< n {out[i] = _time_of_day(i64(intrinsics.unaligned_load(&vals[i])), 1_000_000)}
		case .Time_Nanos:
			out := ([^]w.Time)(dst)
			for i in 0 ..< n {out[i] = _time_of_day(i64(intrinsics.unaligned_load(&vals[i])), 1_000_000_000)}
		case .Ts_Millis:
			out := ([^]w.Datetime)(dst)
			for i in 0 ..< n {out[i] = _ts(i64(intrinsics.unaligned_load(&vals[i])), 1000)}
		case .Ts_Micros:
			out := ([^]w.Datetime)(dst)
			for i in 0 ..< n {out[i] = _ts(i64(intrinsics.unaligned_load(&vals[i])), 1_000_000)}
		case .Ts_Nanos:
			out := ([^]w.Datetime)(dst)
			for i in 0 ..< n {out[i] = _ts(i64(intrinsics.unaligned_load(&vals[i])), 1_000_000_000)}
		case .Dec_Int64:
			out := ([^]f64)(dst)
			d := _pow10(leaf.scale)
			for i in 0 ..< n {out[i] = f64(intrinsics.unaligned_load(&vals[i])) / d}
		}
	case .Float32:
		if len(src) < 4 * n {return false}
		vals := ([^]f32le)(p)
		out := ([^]f64)(dst)
		for i in 0 ..< n {out[i] = f64(intrinsics.unaligned_load(&vals[i]))}
	case .Float64:
		if len(src) < 8 * n {return false}
		mem.copy(dst, p, 8 * n)
	case .Float16:
		if len(src) < 2 * n {return false}
		out := ([^]f64)(dst)
		for i in 0 ..< n {out[i] = _half(u16(src[2 * i]) | u16(src[2 * i + 1]) << 8)}
	case .Int96:
		if len(src) < 12 * n {return false}
		out := ([^]w.Datetime)(dst)
		for i in 0 ..< n {
			nanos := i64(intrinsics.unaligned_load((^i64le)(&src[12 * i])))
			julian := i64(intrinsics.unaligned_load((^u32le)(&src[12 * i + 8])))
			out[i] = _ts((julian - 2_440_588) * 86400 * 1_000_000_000 + nanos, 1_000_000_000)
		}
	case .Dec_Bytes:
		out := ([^]f64)(dst)
		d := _pow10(leaf.scale)
		s := 0
		for i in 0 ..< n {
			ln := leaf.type_length
			if leaf.physical == .Byte_Array {
				if s + 4 > len(src) {return false}
				ln = int(intrinsics.unaligned_load((^u32le)(&src[s])))
				s += 4
			}
			if ln > len(src) - s {return false}
			out[i] = _be_int(src[s:s + ln]) / d
			s += ln
		}
	case .Bytes:
		out := ([^]string)(dst)
		if leaf.physical == .Fixed_Len_Byte_Array {
			ln := leaf.type_length
			if len(src) < ln * n {return false}
			space, ok := _string_space(st.col, ln * n)
			if !ok {return false}
			mem.copy_non_overlapping(raw_data(space), p, ln * n)
			for i in 0 ..< n {out[i] = string(space[i * ln:(i + 1) * ln])}
			return true
		}
		// one copy of the whole value run into the arena; the strings point
		// into it (the 4-byte length prefixes stay as gaps)
		s := 0
		for _ in 0 ..< n {
			if s + 4 > len(src) {return false}
			ln := int(intrinsics.unaligned_load((^u32le)(&src[s])))
			if ln > len(src) - s - 4 {return false}
			s += 4 + ln
		}
		space, ok := _string_space(st.col, s)
		if !ok {return false}
		mem.copy_non_overlapping(raw_data(space), p, s)
		s = 0
		for i in 0 ..< n {
			ln := int(intrinsics.unaligned_load((^u32le)(&space[s])))
			out[i] = string(space[s + 4:s + 4 + ln])
			s += 4 + ln
		}
	case .Uuid:
		if len(src) < 16 * n {return false}
		space, ok := _string_space(st.col, 36 * n)
		if !ok {return false}
		out := ([^]string)(dst)
		HEX := "0123456789abcdef"
		for i in 0 ..< n {
			text := space[36 * i:36 * (i + 1)]
			k := 0
			for b in 0 ..< 16 {
				if b == 4 || b == 6 || b == 8 || b == 10 {
					text[k] = '-'
					k += 1
				}
				v := src[16 * i + b]
				text[k] = HEX[v >> 4]
				text[k + 1] = HEX[v & 15]
				k += 2
			}
			out[i] = string(text)
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Delta and byte-stream-split encodings. Each is turned back into the PLAIN
// layout of the physical type, which _plain then converts like any other page.

@(private)
_decode_other :: proc(st: ^Chunk_State, enc: Encoding, data: []u8, nn: int, dst: rawptr) -> bool {
	leaf := st.leaf
	phys := leaf.physical
	#partial switch enc {
	case .Delta_Binary_Packed:
		if phys != .Int32 && phys != .Int64 {
			return _fail("column %q: DELTA_BINARY_PACKED on a %v column", leaf.name, phys)
		}
		if _, ok := _delta_ints(data, nn, &st.ints); !ok {
			return _fail("column %q: corrupt DELTA_BINARY_PACKED values", leaf.name)
		}
		clear(&st.plain)
		if phys == .Int32 {
			resize(&st.plain, 4 * nn)
			out := ([^]i32le)(raw_data(st.plain))
			for v, i in st.ints[:nn] {intrinsics.unaligned_store(&out[i], i32le(i32(v)))}
		} else {
			resize(&st.plain, 8 * nn)
			out := ([^]i64le)(raw_data(st.plain))
			for v, i in st.ints[:nn] {intrinsics.unaligned_store(&out[i], i64le(v))}
		}
	case .Delta_Length_Byte_Array:
		if phys != .Byte_Array {
			return _fail("column %q: DELTA_LENGTH_BYTE_ARRAY on a %v column", leaf.name, phys)
		}
		used, ok := _delta_ints(data, nn, &st.ints)
		if !ok {return _fail("column %q: corrupt DELTA_LENGTH_BYTE_ARRAY lengths", leaf.name)}
		total := 0
		for l in st.ints[:nn] {
			if l < 0 ||
			   l >
				   i64(
					   len(data),
				   ) {return _fail("column %q: corrupt DELTA_LENGTH_BYTE_ARRAY lengths", leaf.name)}
			total += int(l)
		}
		if total >
		   len(data) -
			   used {return _fail("column %q: DELTA_LENGTH_BYTE_ARRAY data ends early", leaf.name)}
		return _byte_values(st, data[used:used + total], st.ints[:nn], dst)
	case .Delta_Byte_Array:
		if phys != .Byte_Array && phys != .Fixed_Len_Byte_Array {
			return _fail("column %q: DELTA_BYTE_ARRAY on a %v column", leaf.name, phys)
		}
		u1, ok1 := _delta_ints(data, nn, &st.ints)
		if !ok1 {return _fail("column %q: corrupt DELTA_BYTE_ARRAY prefix lengths", leaf.name)}
		u2, ok2 := _delta_ints(data[u1:], nn, &st.ints2)
		if !ok2 {return _fail("column %q: corrupt DELTA_BYTE_ARRAY suffix lengths", leaf.name)}
		suffixes := data[u1 + u2:]
		// value i = the first prefix[i] bytes of value i-1, then suffix i
		clear(&st.bytes)
		s, prev_start, prev_len := 0, 0, 0
		for i in 0 ..< nn {
			pre, suf := st.ints[i], st.ints2[i]
			if pre < 0 || suf < 0 || int(pre) > prev_len || suf > i64(len(suffixes) - s) {
				return _fail("column %q: corrupt DELTA_BYTE_ARRAY values", leaf.name)
			}
			start := len(st.bytes)
			if pre > 0 {
				resize(&st.bytes, start + int(pre))
				copy(st.bytes[start:], st.bytes[prev_start:prev_start + int(pre)])
			}
			append(&st.bytes, ..suffixes[s:s + int(suf)])
			s += int(suf)
			prev_start, prev_len = start, int(pre + suf)
			st.ints[i] = pre + suf // the value's length
		}
		return _byte_values(st, st.bytes[:], st.ints[:nn], dst)
	case .Byte_Stream_Split:
		k: int
		#partial switch phys {
		case .Float, .Int32:
			k = 4
		case .Double, .Int64:
			k = 8
		case .Fixed_Len_Byte_Array:
			k = leaf.type_length
		case:
			return _fail("column %q: BYTE_STREAM_SPLIT on a %v column", leaf.name, phys)
		}
		if len(data) <
		   k * nn {return _fail("column %q: BYTE_STREAM_SPLIT data ends early", leaf.name)}
		// byte b of value i is byte i of stream b
		resize(&st.plain, k * nn)
		for b in 0 ..< k {
			stream := data[b * nn:(b + 1) * nn]
			for i in 0 ..< nn {st.plain[i * k + b] = stream[i]}
		}
	}
	if !_plain(st, st.plain[:], nn, dst) {return _fail("column %q: corrupt values", leaf.name)}
	return true
}

// Byte-array values given back to back with their lengths: strings point
// into one copy in the column's arena; other conversions go through PLAIN
@(private)
_byte_values :: proc(st: ^Chunk_State, values: []u8, lens: []i64, dst: rawptr) -> bool {
	leaf := st.leaf
	if leaf.conv == .Bytes && leaf.physical == .Byte_Array {
		space, ok := _string_space(st.col, len(values))
		if !ok {return _fail("column %q: out of memory", leaf.name)}
		if len(values) > 0 {copy(space, values)}
		out := ([^]string)(dst)
		off := 0
		for l, i in lens {
			out[i] = string(space[off:off + int(l)])
			off += int(l)
		}
		return true
	}
	clear(&st.plain)
	off := 0
	for l in lens {
		if leaf.physical == .Fixed_Len_Byte_Array {
			if int(l) !=
			   leaf.type_length {return _fail("column %q: value of the wrong length", leaf.name)}
		} else {
			n := u32(l)
			append(&st.plain, u8(n), u8(n >> 8), u8(n >> 16), u8(n >> 24))
		}
		append(&st.plain, ..values[off:off + int(l)])
		off += int(l)
	}
	if !_plain(
		st,
		st.plain[:],
		len(lens),
		dst,
	) {return _fail("column %q: corrupt values", leaf.name)}
	return true
}

@(private)
_uvarint :: proc(src: []u8, p: ^int) -> (v: u64, ok: bool) {
	shift: uint
	for {
		if p^ >= len(src) || shift > 63 {return 0, false}
		b := src[p^]
		p^ += 1
		v |= u64(b & 0x7f) << shift
		if b & 0x80 == 0 {return v, true}
		shift += 7
	}
}

// DELTA_BINARY_PACKED: a header (block size, miniblocks per block, value
// count, first value), then blocks of <min delta> <one bit width per
// miniblock> <miniblocks of bit-packed deltas above the minimum>. Exactly
// `want` values are expected. Returns the bytes used, up to the end of the
// last miniblock that holds a value (later miniblocks are not stored).
@(private)
_delta_ints :: proc(
	src: []u8,
	want: int,
	out: ^[dynamic]i64,
) -> (
	used: int,
	ok: bool,
) #no_bounds_check {
	p := 0
	block := _uvarint(src, &p) or_return
	minis := _uvarint(src, &p) or_return
	total := _uvarint(src, &p) or_return
	first_z := _uvarint(src, &p) or_return
	if block == 0 ||
	   block % 128 != 0 ||
	   minis == 0 ||
	   block % minis != 0 ||
	   block > 1 << 20 {return}
	per := int(block / minis)
	if per % 32 != 0 || total != u64(want) {return}
	resize(out, want)
	if want == 0 {return p, true}
	last := u64(i64(first_z >> 1) ~ -i64(first_z & 1))
	out[0] = i64(last)
	k := 1
	for k < want {
		mz := _uvarint(src, &p) or_return
		min_delta := u64(i64(mz >> 1) ~ -i64(mz & 1))
		if int(minis) > len(src) - p {return}
		widths := src[p:p + int(minis)]
		p += int(minis)
		for m in 0 ..< int(minis) {
			if k >= want {break}
			bw := int(widths[m])
			if bw > 64 {return}
			nbytes := per * bw / 8
			if nbytes > len(src) - p {return}
			run := src[p:p + nbytes]
			for i in 0 ..< per {
				if k >= want {break}
				v: u64
				if bw > 0 {
					bit := i * bw
					byte0 := bit >> 3
					if bw <= 56 && byte0 + 8 <= len(run) {
						v = u64(intrinsics.unaligned_load((^u64le)(&run[byte0]))) >> uint(bit & 7)
					} else {
						// up to 64 bits starting anywhere in a byte: nine bytes at most
						acc: u128
						for q in 0 ..< min((bw + (bit & 7) + 7) / 8, len(run) - byte0) {
							acc |= u128(run[byte0 + q]) << uint(8 * q)
						}
						v = u64(acc >> uint(bit & 7))
					}
					if bw < 64 {v &= u64(1) << uint(bw) - 1}
				}
				last = last + min_delta + v // wraps, as the encoder's arithmetic did
				out[k] = i64(last)
				k += 1
			}
			p += nbytes
		}
	}
	return p, true
}
