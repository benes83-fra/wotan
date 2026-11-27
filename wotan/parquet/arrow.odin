package parquet

import w "../core"
import "base:intrinsics"
import "core:encoding/base64"

// The "ARROW:schema" entry of a file's key-value metadata: the Arrow schema
// that pyarrow (and so pandas and Polars) writes next to the Parquet one,
// base64 text of an Arrow IPC message holding a flatbuffer. It carries what
// Parquet has no type for; wotan reads and writes it for durations, which
// Parquet stores as plain INT64, and for the time zone of a timestamp column:
//   reading  a top-level INT64 column without a Parquet logical type that
//            the Arrow schema calls Duration(unit) becomes a Duration column;
//            a UTC timestamp column that it gives a time zone gets that zone
//   writing  a frame with a Duration column or a Datetime column with a time
//            zone gets an Arrow schema of all its columns, so that pandas
//            reads them back as timedelta64 / datetime64 in that zone
//
// Flatbuffers in brief: a table starts with an i32 offset back to its
// vtable (u16 vtable size, u16 table size, then one u16 offset per field, 0
// for an absent field); offsets to tables, vectors and strings are u32 and
// point forward from where they are stored; a vector or string is a u32
// count followed by its elements.

ARROW_SCHEMA_KEY :: "ARROW:schema"

// Arrow type union members used here
@(private)
ARROW_INT :: 2
@(private)
ARROW_FLOAT :: 3
@(private)
ARROW_UTF8 :: 5
@(private)
ARROW_BOOL :: 6
@(private)
ARROW_DATE :: 8
@(private)
ARROW_TIME :: 9
@(private)
ARROW_TIMESTAMP :: 10
@(private)
ARROW_DURATION :: 18

// A top-level field of an Arrow schema: its name, type, time unit (0
// seconds .. 3 nanoseconds; Duration, Time and Timestamp only) and time zone
// (Timestamp only; "" for none)
@(private)
Arrow_Field :: struct {
	name: string, // points into the decoded bytes
	type: u8,
	unit: int,
	tz:   string, // points into the decoded bytes
}

// ---------------------------------------------------------------------------
// Reading

@(private)
Flat :: struct {
	b:   []u8,
	bad: bool,
}

@(private)
_fb_u32 :: proc(f: ^Flat, at: int) -> int {
	if at < 0 || at + 4 > len(f.b) {
		f.bad = true
		return 0
	}
	return int(intrinsics.unaligned_load((^u32le)(&f.b[at])))
}

@(private)
_fb_u16 :: proc(f: ^Flat, at: int) -> int {
	if at < 0 || at + 2 > len(f.b) {
		f.bad = true
		return 0
	}
	return int(intrinsics.unaligned_load((^u16le)(&f.b[at])))
}

// Position of field `id` of the table at `tab`, -1 if absent
@(private)
_fb_field :: proc(f: ^Flat, tab, id: int) -> int {
	vt := tab - int(i32(u32(_fb_u32(f, tab))))
	size := _fb_u16(f, vt)
	if f.bad || 4 + 2 * id >= size {return -1}
	off := _fb_u16(f, vt + 4 + 2 * id)
	return -1 if off == 0 else tab + off
}

// The target of the offset field `id` of table `tab`, -1 if absent
@(private)
_fb_ref :: proc(f: ^Flat, tab, id: int) -> int {
	at := _fb_field(f, tab, id)
	if at < 0 {return -1}
	return at + _fb_u32(f, at)
}

@(private)
_fb_scalar :: proc(f: ^Flat, tab, id, size, default: int) -> int {
	at := _fb_field(f, tab, id)
	if at < 0 {return default}
	switch size {
	case 1:
		if at >= len(f.b) {
			f.bad = true
			return 0
		}
		return int(f.b[at])
	case 2:
		return int(i16(u16(_fb_u16(f, at))))
	}
	return int(i32(u32(_fb_u32(f, at))))
}

// The string field `id` of table `tab` ("" if absent)
@(private)
_fb_string :: proc(f: ^Flat, tab, id: int) -> string {
	at := _fb_ref(f, tab, id)
	if at < 0 {return ""}
	ln := _fb_u32(f, at)
	if f.bad || ln > len(f.b) - at - 4 {
		f.bad = true
		return ""
	}
	return string(f.b[at + 4:at + 4 + ln])
}

// The top-level fields of the Arrow schema in `text` (base64), or ok = false
// if it cannot be read. The names point into `bytes`, which the caller frees.
@(private)
_arrow_fields :: proc(text: string) -> (fields: [dynamic]Arrow_Field, bytes: []u8, ok: bool) {
	err: base64.Error
	bytes, err = base64.decode(text)
	if err != nil {return}
	b := bytes
	// IPC message: [0xFFFFFFFF] <i32 length> <flatbuffer Message>
	if len(b) >= 4 && intrinsics.unaligned_load((^u32le)(&b[0])) == 0xFFFFFFFF {b = b[4:]}
	if len(b) < 8 {return}
	n := int(intrinsics.unaligned_load((^u32le)(&b[0])))
	if n > len(b) - 4 {return}
	f := Flat {
		b = b[4:4 + n],
	}
	msg := _fb_u32(&f, 0)
	if _fb_scalar(&f, msg, 1, 1, 0) != 1 {return} 	// header_type: Schema
	schema := _fb_ref(&f, msg, 2)
	vec := _fb_ref(&f, schema, 1)
	if f.bad || schema < 0 || vec < 0 {return}
	count := _fb_u32(&f, vec)
	if count > len(f.b) / 4 {return}
	fields = make([dynamic]Arrow_Field, 0, count)
	for k in 0 ..< count {
		at := vec + 4 + 4 * k
		field := at + _fb_u32(&f, at)
		af: Arrow_Field
		af.name = _fb_string(&f, field, 0)
		af.type = u8(_fb_scalar(&f, field, 2, 1, 0))
		// a field equal to its default is not stored: the unit of Duration
		// and Time defaults to milliseconds, of Timestamp to seconds
		if af.type == ARROW_DURATION || af.type == ARROW_TIMESTAMP || af.type == ARROW_TIME {
			default := 0 if af.type == ARROW_TIMESTAMP else 1
			af.unit = default
			if t := _fb_ref(&f, field, 3); t >= 0 {
				af.unit = _fb_scalar(&f, t, 0, 2, default)
				if af.type == ARROW_TIMESTAMP {af.tz = _fb_string(&f, t, 1)}
			}
		}
		if f.bad {break}
		append(&fields, af)
	}
	if f.bad {
		delete(fields)
		fields = nil
		return
	}
	return fields, bytes, true
}

// ---------------------------------------------------------------------------
// Writing

// A small front-to-back flatbuffer builder: every table is written before
// the tables, vectors and strings it points to, so all offsets point forward
@(private)
Fb_Builder :: struct {
	buf: [dynamic]u8,
}

@(private)
_fbw_pad :: proc(b: ^Fb_Builder, align: int) {
	for len(b.buf) % align != 0 {append(&b.buf, 0)}
}

@(private)
_fbw_put :: proc(b: ^Fb_Builder, at: int, v: $T) {
	intrinsics.unaligned_store((^T)(&b.buf[at]), v)
}

// A field of a table being written: a scalar of 1, 2 or 4 bytes, or an
// offset (size 0) patched later with _fbw_point
@(private)
Fb_Field :: struct {
	id:    int,
	size:  int, // 1, 2, 4 for scalars; 0 for an offset
	value: int,
}

// Writes a table (its vtable first); returns the table's position and the
// positions of its offset fields, in the order given
@(private)
_fbw_table :: proc(b: ^Fb_Builder, fields: []Fb_Field) -> (tab: int, refs: [4]int) {
	n_ids := 0
	for f in fields {n_ids = max(n_ids, f.id + 1)}
	// layout: soffset, then 4-byte fields, then 2-byte, then 1-byte
	at := make([]int, len(fields), context.temp_allocator)
	pos := 4
	for want in ([]int{4, 2, 1}) {
		for f, k in fields {
			size := 4 if f.size == 0 else f.size
			if size != want {continue}
			at[k] = pos
			pos += size
		}
	}
	tsize := (pos + 3) / 4 * 4
	_fbw_pad(b, 2)
	vt := len(b.buf)
	resize(&b.buf, vt + 4 + 2 * n_ids)
	_fbw_put(b, vt, u16le(4 + 2 * n_ids))
	_fbw_put(b, vt + 2, u16le(tsize))
	for k in 0 ..< n_ids {_fbw_put(b, vt + 4 + 2 * k, u16le(0))}
	_fbw_pad(b, 4)
	tab = len(b.buf)
	resize(&b.buf, tab + tsize)
	for k in tab ..< tab + tsize {b.buf[k] = 0}
	_fbw_put(b, tab, i32le(tab - vt))
	r := 0
	for f, k in fields {
		_fbw_put(b, vt + 4 + 2 * f.id, u16le(at[k]))
		switch f.size {
		case 0:
			refs[r] = tab + at[k]
			r += 1
		case 1:
			b.buf[tab + at[k]] = u8(f.value)
		case 2:
			_fbw_put(b, tab + at[k], i16le(f.value))
		case 4:
			_fbw_put(b, tab + at[k], i32le(f.value))
		}
	}
	return
}

// Sets the offset stored at `at` to point to `target` (which follows it)
@(private)
_fbw_point :: proc(b: ^Fb_Builder, at, target: int) {
	_fbw_put(b, at, u32le(target - at))
}

@(private)
_fbw_string :: proc(b: ^Fb_Builder, s: string) -> int {
	_fbw_pad(b, 4)
	at := len(b.buf)
	resize(&b.buf, at + 4)
	_fbw_put(b, at, u32le(len(s)))
	append(&b.buf, s)
	append(&b.buf, 0)
	return at
}

// A vector of `n` offsets; returns its position (elements at +4, +8, ...)
@(private)
_fbw_vector :: proc(b: ^Fb_Builder, n: int) -> int {
	_fbw_pad(b, 4)
	at := len(b.buf)
	resize(&b.buf, at + 4 + 4 * n)
	_fbw_put(b, at, u32le(n))
	return at
}

// The Arrow schema of the columns as the writer stores them, as base64 text
// (allocated)
@(private)
_arrow_schema_text :: proc(cols: []Out_Col) -> string {
	b: Fb_Builder
	defer delete(b.buf)
	resize(&b.buf, 4) // root offset
	// Message { version: V5, header_type: Schema, header }
	msg, mrefs := _fbw_table(&b, {{0, 2, 4}, {1, 1, 1}, {2, 0, 0}})
	_fbw_point(&b, 0, msg)
	// Schema { fields }
	schema, srefs := _fbw_table(&b, {{1, 0, 0}})
	_fbw_point(&b, mrefs[0], schema)
	vec := _fbw_vector(&b, len(cols))
	_fbw_point(&b, srefs[0], vec)
	for &c, k in cols {
		tag, unit := 0, 0
		switch c.type {
		case .Int:
			tag = ARROW_INT
		case .Float:
			tag = ARROW_FLOAT
		case .Bool:
			tag = ARROW_BOOL
		case .String:
			tag = ARROW_UTF8
		case .Date:
			tag = ARROW_DATE
		case .Datetime:
			tag, unit = ARROW_TIMESTAMP, 3 if c.nanos else 2
		case .Time:
			tag, unit = ARROW_TIME, 3 if c.nanos else 2
		case .Duration:
			tag, unit = ARROW_DURATION, 3
		case .Invalid:
		}
		// Field { name, nullable, type_type, type, children }
		field, frefs := _fbw_table(&b, {{0, 0, 0}, {1, 1, 1}, {2, 1, tag}, {3, 0, 0}, {5, 0, 0}})
		_fbw_point(&b, vec + 4 + 4 * k, field)
		_fbw_point(&b, frefs[0], _fbw_string(&b, c.name))
		type_tab: int
		switch tag {
		case ARROW_INT:
			type_tab, _ = _fbw_table(&b, {{0, 4, 64}, {1, 1, 1}}) // 64 bits, signed
		case ARROW_FLOAT:
			type_tab, _ = _fbw_table(&b, {{0, 2, 2}}) // DOUBLE
		case ARROW_DATE:
			type_tab, _ = _fbw_table(&b, {{0, 2, 0}}) // DAY
		case ARROW_TIME:
			type_tab, _ = _fbw_table(&b, {{0, 2, unit}, {1, 4, 64}})
		case ARROW_TIMESTAMP:
			if c.tz.id == 0 {
				type_tab, _ = _fbw_table(&b, {{0, 2, unit}}) // no time zone: wall-clock values
			} else {
				trefs: [4]int
				type_tab, trefs = _fbw_table(&b, {{0, 2, unit}, {1, 0, 0}})
				_fbw_point(&b, trefs[0], _fbw_string(&b, w.time_zone_name(c.tz)))
			}
		case ARROW_DURATION:
			type_tab, _ = _fbw_table(&b, {{0, 2, unit}})
		case:
			type_tab, _ = _fbw_table(&b, {}) // Bool, Utf8: no fields
		}
		_fbw_point(&b, frefs[1], type_tab)
		_fbw_point(&b, frefs[2], _fbw_vector(&b, 0)) // no children
	}
	_fbw_pad(&b, 8)
	msg_bytes := make([]u8, 8 + len(b.buf), context.temp_allocator)
	intrinsics.unaligned_store((^u32le)(&msg_bytes[0]), 0xFFFFFFFF)
	intrinsics.unaligned_store((^u32le)(&msg_bytes[4]), u32le(len(b.buf)))
	copy(msg_bytes[8:], b.buf[:])
	return base64.encode(msg_bytes)
}
