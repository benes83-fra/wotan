package parquet

import "base:intrinsics"
import "base:runtime"
import "core:compress/deflate"
import lz4 "core:compress/lz4"
import "core:math"
import "core:mem"
import "core:os"

import w "../core"
import util "../util"
import zs "../zstd"

// Parquet writer. Every column becomes an optional (nullable) column of the
// standard type for its values, so pandas, pyarrow, Spark, DuckDB and Polars
// read it back with the same meaning:
//   Int -> INT64             Float -> DOUBLE (NaN stays NaN, not null)
//   Bool -> BOOLEAN          String -> BYTE_ARRAY (STRING)
//   Date -> INT32 (DATE)     Datetime -> INT64 TIMESTAMP(MICROS, local)
//   Time -> INT64 TIME(MICROS, local)
//   Datetime with a time zone -> INT64 TIMESTAMP(MICROS, UTC), the zone
//            named in the Arrow schema (below)
//   Duration -> INT64 nanoseconds, with an Arrow schema in the file's
//            key-value metadata (ARROW:schema) that names it a duration,
//            so that pandas and wotan read it back as one (arrow.odin)
// A Datetime or Time column with a value that microseconds cannot hold is
// written in NANOS instead, if every value fits (Datetime: years 1677 to
// 2262; otherwise the extra digits are dropped).
// A hidden index (set_index with drop = true) is written as the first column,
// under the index's name, as pandas writes its index.
//
// Layout: row groups of `row_group_rows` rows; per column chunk an optional
// dictionary page and data pages (version 1) of about 1 MiB of values. The
// column chunks are encoded and compressed in parallel
// (util.parallel_max_threads caps the threads); the file is the same for any
// number of threads.
// Strings are dictionary-encoded while the dictionary stays under 1 MiB,
// otherwise PLAIN. Each chunk carries statistics (null count, min, max) so
// that readers can skip row groups.

Compression :: enum {
	None,
	Snappy,
	Gzip,
	Lz4, // written as LZ4_RAW
	Zstd,
}

// `df` as the bytes of a Parquet file (free them with delete). ok is false,
// with the reason on stderr, if a column has no type or a hidden index
// shares its name with a column.
write_bytes :: proc(
	df: ^w.DataFrame,
	compression := Compression.Snappy,
	row_group_rows := 1 << 20,
	dictionary := true,
	allocator := context.allocator,
) -> (
	data: []u8,
	ok: bool,
) {
	out := make([dynamic]u8, allocator)
	if !_write(df, &out, compression, max(row_group_rows, 1), dictionary) {
		delete(out)
		return nil, false
	}
	return out[:], true
}

// Writes `df` to the Parquet file at `path` (see write_bytes)
write_file :: proc(
	df: ^w.DataFrame,
	path: string,
	compression := Compression.Snappy,
	row_group_rows := 1 << 20,
	dictionary := true,
) -> bool {
	data, ok := write_bytes(df, compression, row_group_rows, dictionary)
	if !ok {return false}
	defer delete(data)
	if os.write_entire_file(path, data) != nil {
		return _fail("cannot write %q", path)
	}
	return true
}

// ---------------------------------------------------------------------------

// A column to write: its cells as `data[offset ..< offset + rows]`
@(private)
Out_Col :: struct {
	name:   string,
	type:   w.ColumnType,
	data:   rawptr,
	nulls:  []bool, // nil: no nulls
	offset: int,
	nanos:  bool, // Datetime / Time in nanoseconds rather than microseconds
	tz:     w.Time_Zone, // Datetime: a zone = UTC instants
}

@(private)
_out_col :: proc(c: ^w.Column) -> Out_Col {
	b, off := c, 0
	for b.is_view {
		off += b.offset
		b = b.orig
	}
	return Out_Col {
		name = c.name,
		type = c.type,
		data = b.data,
		nulls = b.nulls,
		offset = off,
		tz = c.tz if c.type == .Datetime else {},
	}
}

@(private)
_is_null :: #force_inline proc(c: ^Out_Col, row: int) -> bool {
	return c.nulls != nil && c.nulls[c.offset + row]
}

@(private)
Chunk_Info :: struct {
	start:        i64,
	dict_offset:  i64, // -1: no dictionary page
	data_offset:  i64,
	compressed:   i64,
	uncompressed: i64,
	null_count:   i64,
	has_minmax:   bool,
	min, max:     [8]u8, // PLAIN bytes of fixed-width values
	min_str:      string, // BYTE_ARRAY statistics (point into the column)
	max_str:      string,
	stat_len:     int, // bytes used in min / max
	dict:         bool,
}

@(private)
Writer :: struct {
	out:         ^[dynamic]u8,
	compression: Compression,
	dictionary:  bool,
	values:      [dynamic]u8, // a page's encoded values
	page:        [dynamic]u8, // levels + values
	packed:      [dynamic]u8, // compressed page
	idx:         [dynamic]u32,
	levels:      [dynamic]u32,
}

@(private)
PAGE_BYTES :: 1 << 20
@(private)
DICT_BYTES :: 1 << 20

@(private)
_write :: proc(
	df: ^w.DataFrame,
	out: ^[dynamic]u8,
	compression: Compression,
	rg_rows: int,
	dictionary: bool,
) -> bool {
	cols := make([dynamic]Out_Col)
	defer delete(cols)
	iv := &df.index_values
	if df.has_index && iv.type != .Invalid && iv.len == df.rows {
		if df.index_column in df.name_to_index {
			return _fail("the hidden index %q has the name of a column", df.index_column)
		}
		oc := _out_col(iv)
		oc.name = df.index_column
		append(&cols, oc)
	}
	for &c in df.columns {
		if c.type == .Invalid {return _fail("column %q has no type", c.name)}
		if c.len !=
		   df.rows {return _fail("column %q has %d rows, the frame %d", c.name, c.len, df.rows)}
		append(&cols, _out_col(&c))
	}
	for &c in cols {
		if c.type == .Datetime || c.type == .Time {c.nanos = _wants_nanos(&c, df.rows)}
	}

	// every column chunk is encoded on its own, in parallel, into its own
	// buffer; the buffers are then joined in file order
	n_groups := (df.rows + rg_rows - 1) / rg_rows
	infos := make([]Chunk_Info, n_groups * len(cols))
	defer delete(infos)
	bufs := make([][dynamic]u8, len(infos))
	defer {
		for &b in bufs {delete(b)}
		delete(bufs)
	}
	oks := make([]bool, len(infos))
	defer delete(oks)
	work := Write_Work {
		cols        = cols[:],
		rows        = df.rows,
		rg_rows     = rg_rows,
		compression = compression,
		dictionary  = dictionary,
		infos       = infos,
		bufs        = bufs,
		oks         = oks,
	}
	util.parallel_for(len(infos), &work, _write_chunks, tasks = len(infos))
	for ok in oks {
		if !ok {return false}
	}

	total := 4 + 8
	for b in bufs {total += len(b)}
	reserve(out, total + 4096)
	append(out, "PAR1")
	for &info, k in infos {
		base := i64(len(out))
		append(out, ..bufs[k][:])
		info.start += base
		info.data_offset += base
		if info.dict_offset >= 0 {info.dict_offset += base}
	}

	footer_start := len(out)
	_write_footer(out, df.rows, cols[:], infos, rg_rows, n_groups, compression)
	footer_len := len(out) - footer_start
	n := u32(footer_len)
	append(out, u8(n), u8(n >> 8), u8(n >> 16), u8(n >> 24))
	append(out, "PAR1")
	return true
}

@(private)
Write_Work :: struct {
	cols:        []Out_Col,
	rows:        int,
	rg_rows:     int,
	compression: Compression,
	dictionary:  bool,
	infos:       []Chunk_Info, // per chunk (row group major); offsets relative to its buffer
	bufs:        [][dynamic]u8,
	oks:         []bool,
	next:        int, // next chunk to take (atomic)
}

// A parallel task: encodes chunks until none are left, reusing its page
// buffers. Scratch and chunk buffers come from the heap (thread-safe).
@(private)
_write_chunks :: proc(data: rawptr, begin, end: int) {
	work := (^Write_Work)(data)
	context.allocator = runtime.heap_allocator()
	wr := Writer {
		compression = work.compression,
		dictionary  = work.dictionary,
	}
	defer {
		delete(wr.values)
		delete(wr.page)
		delete(wr.packed)
		delete(wr.idx)
		delete(wr.levels)
	}
	nc := len(work.cols)
	for {
		k := intrinsics.atomic_add(&work.next, 1)
		if k >= len(work.infos) {break}
		g, ci := k / nc, k % nc
		r0 := g * work.rg_rows
		r1 := min(r0 + work.rg_rows, work.rows)
		work.bufs[k] = make([dynamic]u8)
		wr.out = &work.bufs[k]
		work.infos[k], work.oks[k] = _write_chunk(&wr, &work.cols[ci], r0, r1)
	}
}

// ---------------------------------------------------------------------------
// Column chunks

@(private)
_physical :: proc(t: w.ColumnType) -> Physical_Type {
	#partial switch t {
	case .Bool:
		return .Boolean
	case .Date:
		return .Int32
	case .Float:
		return .Double
	case .String:
		return .Byte_Array
	}
	return .Int64 // Int, Datetime, Time, Duration
}

@(private)
_micros_dt :: #force_inline proc(v: w.Datetime) -> i64 {
	return w.datetime_to_unix(v) * 1_000_000 + i64(v.nanosecond / 1000)
}

@(private)
_nanos_dt :: #force_inline proc(v: w.Datetime) -> i64 {
	ns, _ := w.datetime_to_unix_nanos(v) // the column was checked to fit
	return ns
}

// Whether `c` (Datetime or Time) needs nanoseconds: a value with digits
// below the microsecond, and (Datetime) every value in the nanosecond range
@(private)
_wants_nanos :: proc(c: ^Out_Col, rows: int) -> bool {
	finer := false
	#partial switch c.type {
	case .Time:
		cells := ([^]w.Time)(c.data)[c.offset:]
		for r in 0 ..< rows {
			if !_is_null(c, r) && cells[r].nanosecond % 1000 != 0 {return true}
		}
	case .Datetime:
		cells := ([^]w.Datetime)(c.data)[c.offset:]
		for r in 0 ..< rows {
			if _is_null(c, r) {continue}
			if _, fits := w.datetime_to_unix_nanos(cells[r]); !fits {return false}
			if cells[r].nanosecond % 1000 != 0 {finer = true}
		}
	}
	return finer
}

@(private)
_micros_t :: #force_inline proc(v: w.Time) -> i64 {
	return(
		(i64(v.hour) * 3600 + i64(v.minute) * 60 + i64(v.second)) * 1_000_000 +
		i64(v.nanosecond / 1000) \
	)
}

@(private)
_nanos_t :: #force_inline proc(v: w.Time) -> i64 {
	return(
		(i64(v.hour) * 3600 + i64(v.minute) * 60 + i64(v.second)) * 1_000_000_000 +
		i64(v.nanosecond) \
	)
}

@(private)
_days :: #force_inline proc(v: w.Date) -> i32 {
	return i32(w.days_from_civil(i64(v.year), i64(v.month), i64(v.day)))
}

@(private)
_write_chunk :: proc(wr: ^Writer, c: ^Out_Col, r0, r1: int) -> (info: Chunk_Info, ok: bool) {
	info = Chunk_Info {
		start       = i64(len(wr.out)),
		dict_offset = -1,
	}
	_stats(c, r0, r1, &info)

	// strings: a dictionary while it stays small
	dict_map: map[string]u32
	dict_list: [dynamic]string
	defer {
		delete(dict_map)
		delete(dict_list)
	}
	if wr.dictionary && c.type == .String {
		info.dict = true
		cells := ([^]string)(c.data)[c.offset:]
		bytes := 0
		resize(&wr.idx, r1 - r0)
		k := 0
		for r in r0 ..< r1 {
			if _is_null(c, r) {continue}
			s := cells[r]
			id, found := dict_map[s]
			if !found {
				id = u32(len(dict_list))
				dict_map[s] = id
				append(&dict_list, s)
				bytes += 4 + len(s)
				if bytes > DICT_BYTES {
					info.dict = false
					break
				}
			}
			wr.idx[k] = id
			k += 1
		}
		if len(dict_list) == 0 {info.dict = false} 	// nothing to encode
		if info.dict {
			info.dict_offset = i64(len(wr.out))
			clear(&wr.values)
			for s in dict_list {_put_bytes(&wr.values, s)}
			_emit_page(wr, &info, .Dictionary_Page, len(dict_list), wr.values[:], false) or_return
		}
	}

	info.data_offset = i64(len(wr.out))
	// data pages of about PAGE_BYTES of values each
	bw := 1
	for (1 << uint(bw)) < len(dict_list) {bw += 1}
	pos := 0 // index into wr.idx (dictionary) of the page's first non-null cell
	r := r0
	for r < r1 {
		// rows in this page
		end := r
		size := 0
		for end < r1 && size < PAGE_BYTES {
			if !_is_null(c, end) {
				switch {
				case info.dict:
					size += 4
				case c.type == .String:
					size += 4 + len(([^]string)(c.data)[c.offset + end])
				case c.type == .Bool:
					size += 1
				case:
					size += 8
				}
			}
			end += 1
		}
		// definition levels (1 = present), then the values of present cells
		clear(&wr.page)
		append(&wr.page, 0, 0, 0, 0)
		resize(&wr.levels, end - r)
		nn := 0
		for i in r ..< end {
			present := !_is_null(c, i)
			wr.levels[i - r] = 1 if present else 0
			nn += int(present)
		}
		_rle_encode(&wr.page, wr.levels[:], 1)
		ln := u32(len(wr.page) - 4)
		wr.page[0], wr.page[1], wr.page[2], wr.page[3] =
			u8(ln), u8(ln >> 8), u8(ln >> 16), u8(ln >> 24)
		if info.dict {
			append(&wr.page, u8(bw))
			_rle_encode(&wr.page, wr.idx[pos:pos + nn], bw)
			pos += nn
		} else {
			_plain_values(&wr.page, c, r, end)
		}
		_emit_page(wr, &info, .Data_Page, end - r, wr.page[:], info.dict) or_return
		r = end
	}
	return info, true
}

// Appends a page header and the (compressed) page; updates the chunk sizes
@(private)
_emit_page :: proc(
	wr: ^Writer,
	info: ^Chunk_Info,
	kind: Page_Type,
	n: int,
	body: []u8,
	dict_encoded: bool,
) -> bool {
	payload := body
	switch wr.compression {
	case .None:
	case .Snappy:
		snappy_compress(&wr.packed, body)
		payload = wr.packed[:]
	case .Gzip:
		if deflate.deflate_gzip_to_dynamic(&wr.packed, body) !=
		   nil {return _fail("gzip compression failed")}
		payload = wr.packed[:]
	case .Lz4:
		if lz4.compress_block_to_dynamic(&wr.packed, body) !=
		   nil {return _fail("LZ4 compression failed")}
		payload = wr.packed[:]
	case .Zstd:
		clear(&wr.packed)
		if !zs.compress_to_dynamic(&wr.packed, body) {return _fail("Zstandard compression failed")}
		payload = wr.packed[:]
	}

	start := len(wr.out)
	t := Thrift_Writer {
		out = wr.out,
	}
	_tw_i32(&t, 1, i32(kind))
	_tw_i32(&t, 2, i32(len(body)))
	_tw_i32(&t, 3, i32(len(payload)))
	if kind == .Dictionary_Page {
		_tw_struct(&t, 7)
		_tw_i32(&t, 1, i32(n))
		_tw_i32(&t, 2, i32(Encoding.Plain))
		_tw_end(&t)
	} else {
		_tw_struct(&t, 5)
		_tw_i32(&t, 1, i32(n))
		_tw_i32(&t, 2, i32(Encoding.Rle_Dictionary if dict_encoded else Encoding.Plain))
		_tw_i32(&t, 3, i32(Encoding.Rle))
		_tw_i32(&t, 4, i32(Encoding.Rle))
		_tw_end(&t)
	}
	_tw_byte(&t, T_STOP)
	header := len(wr.out) - start
	append(wr.out, ..payload)
	info.compressed += i64(header + len(payload))
	info.uncompressed += i64(header + len(body))
	return true
}

@(private)
_put_bytes :: proc(buf: ^[dynamic]u8, s: string) {
	n := u32(len(s))
	append(buf, u8(n), u8(n >> 8), u8(n >> 16), u8(n >> 24))
	append(buf, s)
}

// PLAIN values of the present cells of rows [r0, r1)
@(private)
_plain_values :: proc(buf: ^[dynamic]u8, c: ^Out_Col, r0, r1: int) #no_bounds_check {
	put :: proc(
		buf: ^[dynamic]u8,
		c: ^Out_Col,
		r0, r1: int,
		cells: [^]$T,
		$V: typeid,
		conv: proc(v: T) -> V,
	) {
		start := len(buf)
		resize(buf, start + (r1 - r0) * size_of(V))
		out := ([^]V)(&buf[start]) if r1 > r0 else nil
		k := 0
		for r in r0 ..< r1 {
			if _is_null(c, r) {continue}
			intrinsics.unaligned_store(&out[k], conv(cells[c.offset + r]))
			k += 1
		}
		resize(buf, start + k * size_of(V))
	}
	switch c.type {
	case .Int, .Duration:
		cells := ([^]int)(c.data) // a Duration is an i64 count of nanoseconds
		if c.nulls == nil || !_any_null(c, r0, r1) {
			append(buf, ..([^]u8)(&cells[c.offset + r0])[:(r1 - r0) * 8])
		} else {
			put(buf, c, r0, r1, cells, i64le, proc(v: int) -> i64le {return i64le(v)})
		}
	case .Float:
		cells := ([^]f64)(c.data)
		if c.nulls == nil || !_any_null(c, r0, r1) {
			append(buf, ..([^]u8)(&cells[c.offset + r0])[:(r1 - r0) * 8])
		} else {
			put(buf, c, r0, r1, cells, f64le, proc(v: f64) -> f64le {return f64le(v)})
		}
	case .Date:
		put(
			buf,
			c,
			r0,
			r1,
			([^]w.Date)(c.data),
			i32le,
			proc(v: w.Date) -> i32le {return i32le(_days(v))},
		)
	case .Datetime:
		if c.nanos {
			put(
				buf,
				c,
				r0,
				r1,
				([^]w.Datetime)(c.data),
				i64le,
				proc(v: w.Datetime) -> i64le {return i64le(_nanos_dt(v))},
			)
		} else {
			put(
				buf,
				c,
				r0,
				r1,
				([^]w.Datetime)(c.data),
				i64le,
				proc(v: w.Datetime) -> i64le {return i64le(_micros_dt(v))},
			)
		}
	case .Time:
		if c.nanos {
			put(
				buf,
				c,
				r0,
				r1,
				([^]w.Time)(c.data),
				i64le,
				proc(v: w.Time) -> i64le {return i64le(_nanos_t(v))},
			)
		} else {
			put(
				buf,
				c,
				r0,
				r1,
				([^]w.Time)(c.data),
				i64le,
				proc(v: w.Time) -> i64le {return i64le(_micros_t(v))},
			)
		}
	case .Bool:
		cells := ([^]bool)(c.data)
		acc: u8
		bit: uint
		for r in r0 ..< r1 {
			if _is_null(c, r) {continue}
			if cells[c.offset + r] {acc |= 1 << bit}
			bit += 1
			if bit == 8 {
				append(buf, acc)
				acc, bit = 0, 0
			}
		}
		if bit > 0 {append(buf, acc)}
	case .String:
		cells := ([^]string)(c.data)
		for r in r0 ..< r1 {
			if _is_null(c, r) {continue}
			_put_bytes(buf, cells[c.offset + r])
		}
	case .Invalid:
	}
}

@(private)
_any_null :: proc(c: ^Out_Col, r0, r1: int) -> bool {
	for r in r0 ..< r1 {if c.nulls[c.offset + r] {return true}}
	return false
}

// RLE / bit-packing hybrid of `vals` (each < 2^bw), appended to `buf`. Works
// in blocks of 8: a block of 8 equal values extends (or starts) a repeated
// run, other blocks are bit-packed; the last, partial block is a short run
// if its values are equal, else a padded bit-packed group (readers stop
// after the page's value count).
@(private)
_rle_encode :: proc(buf: ^[dynamic]u8, vals: []u32, bw: int) {
	n := len(vals)
	vbytes := (bw + 7) / 8
	run_val: u32
	run_len := 0
	packed_from, packed_to := 0, 0 // pending bit-packed blocks: vals[packed_from:packed_to]

	put_varint :: proc(buf: ^[dynamic]u8, v: u64) {
		v := v
		for v >= 0x80 {
			append(buf, u8(v) | 0x80)
			v >>= 7
		}
		append(buf, u8(v))
	}
	flush_run :: proc(buf: ^[dynamic]u8, run_val: u32, run_len: ^int, vbytes: int) {
		if run_len^ == 0 {return}
		put_varint(buf, u64(run_len^) << 1)
		for k in 0 ..< vbytes {append(buf, u8(run_val >> uint(8 * k)))}
		run_len^ = 0
	}
	flush_packed :: proc(buf: ^[dynamic]u8, vals: []u32, from, to: ^int, bw: int) {
		count := to^ - from^
		if count == 0 {return}
		groups := (count + 7) / 8
		put_varint(buf, u64(groups) << 1 | 1)
		acc: u64
		bits: uint
		for k in 0 ..< groups * 8 {
			v := u64(vals[from^ + k]) if from^ + k < to^ else 0
			acc |= v << bits
			bits += uint(bw)
			for bits >= 8 {
				append(buf, u8(acc))
				acc >>= 8
				bits -= 8
			}
		}
		if bits > 0 {append(buf, u8(acc))}
		from^ = to^
	}

	i := 0
	for i < n {
		m := min(8, n - i)
		v0 := vals[i]
		equal := true
		for k in 1 ..< m {if vals[i + k] != v0 {equal = false; break}}
		switch {
		case equal && m == 8:
			if run_len > 0 && run_val == v0 {
				run_len += 8
			} else {
				flush_packed(buf, vals, &packed_from, &packed_to, bw)
				flush_run(buf, run_val, &run_len, vbytes)
				run_val, run_len = v0, 8
			}
			packed_from, packed_to = i + 8, i + 8
		case equal:
			// the last, short block
			flush_packed(buf, vals, &packed_from, &packed_to, bw)
			if run_len > 0 && run_val == v0 {
				run_len += m
			} else {
				flush_run(buf, run_val, &run_len, vbytes)
				run_val, run_len = v0, m
			}
		case:
			flush_run(buf, run_val, &run_len, vbytes)
			if packed_to != i {packed_from = i}
			packed_to = i + m
		}
		i += m
	}
	flush_packed(buf, vals, &packed_from, &packed_to, bw)
	flush_run(buf, run_val, &run_len, vbytes)
}

// Null count, min and max of the present cells (NaN left out; an all-NaN or
// all-null chunk has no min / max)
@(private)
_stats :: proc(c: ^Out_Col, r0, r1: int, info: ^Chunk_Info) {
	put :: proc(b: ^[8]u8, v: u64, n: int) {for k in 0 ..< n {b[k] = u8(v >> uint(8 * k))}}
	nulls := 0
	switch c.type {
	case .Int, .Date, .Datetime, .Time, .Duration:
		lo, hi := max(i64), min(i64)
		for r in r0 ..< r1 {
			if _is_null(c, r) {
				nulls += 1
				continue
			}
			v: i64
			#partial switch c.type {
			case .Int, .Duration:
				v = i64(([^]int)(c.data)[c.offset + r])
			case .Date:
				v = i64(_days(([^]w.Date)(c.data)[c.offset + r]))
			case .Datetime:
				d := ([^]w.Datetime)(c.data)[c.offset + r]
				v = _nanos_dt(d) if c.nanos else _micros_dt(d)
			case .Time:
				t := ([^]w.Time)(c.data)[c.offset + r]
				v = _nanos_t(t) if c.nanos else _micros_t(t)
			}
			lo, hi = min(lo, v), max(hi, v)
		}
		if nulls < r1 - r0 {
			info.has_minmax = true
			info.stat_len = 4 if c.type == .Date else 8
			put(&info.min, u64(lo), info.stat_len)
			put(&info.max, u64(hi), info.stat_len)
		}
	case .Float:
		lo, hi := math.inf_f64(1), math.inf_f64(-1)
		seen := false
		for r in r0 ..< r1 {
			if _is_null(c, r) {
				nulls += 1
				continue
			}
			v := ([^]f64)(c.data)[c.offset + r]
			if math.is_nan(v) {continue}
			seen = true
			lo, hi = min(lo, v), max(hi, v)
		}
		if seen {
			// the format's rule for zeros: a zero minimum is -0, a zero maximum +0
			if lo == 0 {lo = math.copy_sign(f64(0), -1)}
			if hi == 0 {hi = 0}
			info.has_minmax = true
			info.stat_len = 8
			put(&info.min, transmute(u64)lo, 8)
			put(&info.max, transmute(u64)hi, 8)
		}
	case .Bool:
		lo, hi := true, false
		for r in r0 ..< r1 {
			if _is_null(c, r) {
				nulls += 1
				continue
			}
			v := ([^]bool)(c.data)[c.offset + r]
			lo, hi = lo && v, hi || v
		}
		if nulls < r1 - r0 {
			info.has_minmax = true
			info.stat_len = 1
			info.min[0], info.max[0] = u8(lo), u8(hi)
		}
	case .String:
		lo, hi: string
		seen := false
		for r in r0 ..< r1 {
			if _is_null(c, r) {
				nulls += 1
				continue
			}
			v := ([^]string)(c.data)[c.offset + r]
			if !seen {
				lo, hi, seen = v, v, true
			} else {
				if v < lo {lo = v} 	// byte-wise order, as the format specifies for strings
				if v > hi {hi = v}
			}
		}
		// long values are left out rather than truncated
		if seen && len(lo) <= 64 && len(hi) <= 64 {
			info.has_minmax = true
			info.min_str, info.max_str = lo, hi
		}
	case .Invalid:
	}
	info.null_count = i64(nulls)
}

// ---------------------------------------------------------------------------
// Footer

@(private)
_write_footer :: proc(
	out: ^[dynamic]u8,
	rows: int,
	cols: []Out_Col,
	infos: []Chunk_Info,
	rg_rows, n_groups: int,
	compression: Compression,
) {
	t := Thrift_Writer {
		out = out,
	}
	codec: Codec
	switch compression {
	case .None:
		codec = .Uncompressed
	case .Snappy:
		codec = .Snappy
	case .Gzip:
		codec = .Gzip
	case .Lz4:
		codec = .Lz4_Raw
	case .Zstd:
		codec = .Zstd
	}
	_tw_i32(&t, 1, 1) // version
	// schema: the root, then one optional leaf per column
	_tw_list(&t, 2, T_STRUCT, len(cols) + 1)
	_tw_open(&t)
	_tw_string(&t, 4, "schema")
	_tw_i32(&t, 5, i32(len(cols)))
	_tw_end(&t)
	for &c in cols {
		_tw_open(&t)
		_tw_i32(&t, 1, i32(_physical(c.type)))
		_tw_i32(&t, 3, 1) // OPTIONAL
		_tw_string(&t, 4, c.name)
		#partial switch c.type {
		case .String:
			_tw_i32(&t, 6, CT_UTF8)
			_tw_struct(&t, 10)
			_tw_struct(&t, LT_STRING)
			_tw_end(&t)
			_tw_end(&t)
		case .Date:
			_tw_i32(&t, 6, CT_DATE)
			_tw_struct(&t, 10)
			_tw_struct(&t, LT_DATE)
			_tw_end(&t)
			_tw_end(&t)
		case .Datetime, .Time:
			// the legacy converted type has no nanoseconds
			if !c.nanos {_tw_i32(&t, 6, CT_TIMESTAMP_MICROS if c.type == .Datetime else CT_TIME_MICROS)}
			_tw_struct(&t, 10)
			_tw_struct(&t, LT_TIMESTAMP if c.type == .Datetime else LT_TIME)
			_tw_bool(&t, 1, c.tz.id != 0) // isAdjustedToUTC: instants, or wall-clock values
			_tw_struct(&t, 2) // unit
			_tw_struct(&t, 3 if c.nanos else 2) // NANOS or MICROS
			_tw_end(&t)
			_tw_end(&t)
			_tw_end(&t)
			_tw_end(&t)
		}
		_tw_end(&t)
	}
	_tw_i64(&t, 3, i64(rows))
	// row groups
	_tw_list(&t, 4, T_STRUCT, n_groups)
	for g in 0 ..< n_groups {
		r0 := g * rg_rows
		r1 := min(r0 + rg_rows, rows)
		group := infos[g * len(cols):(g + 1) * len(cols)]
		total_unc, total_comp: i64
		for &info in group {
			total_unc += info.uncompressed
			total_comp += info.compressed
		}
		_tw_open(&t)
		_tw_list(&t, 1, T_STRUCT, len(cols))
		for &info, ci in group {
			c := &cols[ci]
			_tw_open(&t) // ColumnChunk
			_tw_i64(&t, 2, info.start)
			_tw_struct(&t, 3) // ColumnMetaData
			_tw_i32(&t, 1, i32(_physical(c.type)))
			if info.dict {
				_tw_list(&t, 2, T_I32, 3)
				_tw_zigzag(&t, i64(Encoding.Plain))
				_tw_zigzag(&t, i64(Encoding.Rle))
				_tw_zigzag(&t, i64(Encoding.Rle_Dictionary))
			} else {
				_tw_list(&t, 2, T_I32, 2)
				_tw_zigzag(&t, i64(Encoding.Plain))
				_tw_zigzag(&t, i64(Encoding.Rle))
			}
			_tw_list(&t, 3, T_BINARY, 1)
			_tw_varint(&t, u64(len(c.name)))
			append(out, c.name)
			_tw_i32(&t, 4, i32(codec))
			_tw_i64(&t, 5, i64(r1 - r0))
			_tw_i64(&t, 6, info.uncompressed)
			_tw_i64(&t, 7, info.compressed)
			_tw_i64(&t, 9, info.data_offset)
			if info.dict {_tw_i64(&t, 11, info.dict_offset)}
			_tw_struct(&t, 12) // Statistics
			_tw_i64(&t, 3, info.null_count)
			if info.has_minmax {
				if c.type == .String {
					_tw_string(&t, 5, info.max_str)
					_tw_string(&t, 6, info.min_str)
				} else {
					_tw_binary(&t, 5, info.max[:info.stat_len])
					_tw_binary(&t, 6, info.min[:info.stat_len])
				}
			}
			_tw_end(&t)
			_tw_end(&t) // ColumnMetaData
			_tw_end(&t) // ColumnChunk
		}
		_tw_i64(&t, 2, total_unc)
		_tw_i64(&t, 3, i64(r1 - r0))
		if len(group) > 0 {_tw_i64(&t, 5, group[0].start)}
		_tw_i64(&t, 6, total_comp)
		_tw_end(&t)
	}
	// durations and time zones: an Arrow schema that names them (Parquet has
	// neither)
	needs_arrow := false
	for &c in cols {
		if c.type == .Duration || c.tz.id != 0 {needs_arrow = true}
	}
	if needs_arrow {
		text := _arrow_schema_text(cols)
		defer delete(text)
		_tw_list(&t, 5, T_STRUCT, 1)
		_tw_open(&t)
		_tw_string(&t, 1, ARROW_SCHEMA_KEY)
		_tw_string(&t, 2, text)
		_tw_end(&t)
	}
	_tw_string(&t, 6, "wotan")
	// column orders: the type-defined order for every column (min / max
	// statistics are valid only with it)
	_tw_list(&t, 7, T_STRUCT, len(cols))
	for _ in cols {
		_tw_open(&t)
		_tw_struct(&t, 1)
		_tw_end(&t)
		_tw_end(&t)
	}
	_tw_byte(&t, T_STOP)
}
