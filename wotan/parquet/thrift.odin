package parquet

// Parquet keeps its metadata (the file footer and one header per page) in
// Thrift's compact protocol. This file decodes the parts of parquet.thrift
// that the reader needs; unknown fields are skipped, so files written by
// newer writers still load.
//
// Compact protocol in brief: a struct is a run of fields ended by a 0 byte.
// A field header byte holds the field-id delta in its high nibble (0 means a
// zigzag i16 id follows) and the type in its low nibble. Integers are zigzag
// varints, binaries a varint length and the bytes, lists a header byte (size
// in the high nibble, 15 meaning a varint size follows) and the elements.

@(private)
T_STOP :: 0
@(private)
T_TRUE :: 1
@(private)
T_FALSE :: 2
@(private)
T_BYTE :: 3
@(private)
T_I16 :: 4
@(private)
T_I32 :: 5
@(private)
T_I64 :: 6
@(private)
T_DOUBLE :: 7
@(private)
T_BINARY :: 8
@(private)
T_LIST :: 9
@(private)
T_SET :: 10
@(private)
T_MAP :: 11
@(private)
T_STRUCT :: 12

// Physical types (parquet.thrift `Type`)
Physical_Type :: enum i32 {
	Boolean              = 0,
	Int32                = 1,
	Int64                = 2,
	Int96                = 3,
	Float                = 4,
	Double               = 5,
	Byte_Array           = 6,
	Fixed_Len_Byte_Array = 7,
}

// Compression codecs (parquet.thrift `CompressionCodec`)
Codec :: enum i32 {
	Uncompressed = 0,
	Snappy       = 1,
	Gzip         = 2,
	Lzo          = 3,
	Brotli       = 4,
	Lz4          = 5, // Hadoop-framed LZ4 (deprecated in the format)
	Zstd         = 6,
	Lz4_Raw      = 7,
}

// Value encodings (parquet.thrift `Encoding`)
Encoding :: enum i32 {
	Plain                   = 0,
	Plain_Dictionary        = 2,
	Rle                     = 3,
	Bit_Packed              = 4,
	Delta_Binary_Packed     = 5,
	Delta_Length_Byte_Array = 6,
	Delta_Byte_Array        = 7,
	Rle_Dictionary          = 8,
	Byte_Stream_Split       = 9,
}

@(private)
Page_Type :: enum i32 {
	Data_Page       = 0,
	Index_Page      = 1,
	Dictionary_Page = 2,
	Data_Page_V2    = 3,
}

// ConvertedType values used by the reader (the legacy annotations)
@(private)
CT_UTF8 :: 0
@(private)
CT_DECIMAL :: 5
@(private)
CT_DATE :: 6
@(private)
CT_TIME_MILLIS :: 7
@(private)
CT_TIME_MICROS :: 8
@(private)
CT_TIMESTAMP_MILLIS :: 9
@(private)
CT_TIMESTAMP_MICROS :: 10
@(private)
CT_UINT_8 :: 11
@(private)
CT_UINT_16 :: 12
@(private)
CT_UINT_32 :: 13
@(private)
CT_UINT_64 :: 14

// LogicalType union members (the field id inside the union)
@(private)
LT_STRING :: 1
@(private)
LT_DECIMAL :: 5
@(private)
LT_DATE :: 6
@(private)
LT_TIME :: 7
@(private)
LT_TIMESTAMP :: 8
@(private)
LT_INTEGER :: 10
@(private)
LT_FLOAT16 :: 15
@(private)
LT_UUID :: 14

// Time units of TIME and TIMESTAMP
@(private)
Time_Unit :: enum u8 {
	None,
	Millis,
	Micros,
	Nanos,
}

@(private)
Logical_Type :: struct {
	kind:      int, // 0 = none, else the LT_ member id
	unit:      Time_Unit,
	utc:       bool,
	bit_width: i32,
	signed:    bool,
	scale:     i32,
	precision: i32,
}

@(private)
Schema_Element :: struct {
	type:         i32, // Physical_Type, -1 for groups
	type_length:  i32,
	repetition:   i32, // 0 required, 1 optional, 2 repeated, -1 unset
	name:         string,
	num_children: i32,
	converted:    i32, // -1 unset
	scale:        i32,
	precision:    i32,
	logical:      Logical_Type,
}

@(private)
Column_Meta :: struct {
	type:                   i32,
	codec:                  i32,
	num_values:             i64,
	total_compressed_size:  i64,
	data_page_offset:       i64,
	dictionary_page_offset: i64, // 0 when absent
	has_dictionary_offset:  bool,
}

@(private)
Column_Chunk :: struct {
	file_path: string,
	has_meta:  bool,
	meta:      Column_Meta,
}

@(private)
Row_Group :: struct {
	columns:  [dynamic]Column_Chunk,
	num_rows: i64,
}

@(private)
File_Meta :: struct {
	version:    i32,
	schema:     [dynamic]Schema_Element,
	num_rows:   i64,
	row_groups: [dynamic]Row_Group,
	created_by: string,
}

@(private)
Data_Page_Header :: struct {
	num_values: i32,
	encoding:   i32,
}

@(private)
Dictionary_Page_Header :: struct {
	num_values: i32,
	encoding:   i32,
}

@(private)
Data_Page_Header_V2 :: struct {
	num_values:    i32,
	num_nulls:     i32,
	num_rows:      i32,
	encoding:      i32,
	def_len:       i32,
	rep_len:       i32,
	is_compressed: bool,
}

@(private)
Page_Header :: struct {
	type:              i32,
	uncompressed_size: i32,
	compressed_size:   i32,
	has_data:          bool,
	data:              Data_Page_Header,
	has_dict:          bool,
	dict:              Dictionary_Page_Header,
	has_v2:            bool,
	v2:                Data_Page_Header_V2,
}

// ---------------------------------------------------------------------------
// Compact protocol reader. Errors set `bad` and make every further read
// return zero values, so the decoders check `bad` once at the end.

@(private)
Thrift :: struct {
	data: []u8,
	pos:  int,
	bad:  bool,
}

@(private)
_t_byte :: #force_inline proc(t: ^Thrift) -> u8 {
	if t.pos >= len(t.data) {
		t.bad = true
		return 0
	}
	b := t.data[t.pos]
	t.pos += 1
	return b
}

@(private)
_t_varint :: proc(t: ^Thrift) -> u64 {
	v: u64
	shift: u32
	for {
		b := _t_byte(t)
		if t.bad {return 0}
		v |= u64(b & 0x7f) << shift
		if b & 0x80 == 0 {break}
		shift += 7
		if shift > 63 {
			t.bad = true
			return 0
		}
	}
	return v
}

@(private)
_t_zigzag :: #force_inline proc(t: ^Thrift, typ: u8 = T_I64) -> i64 {
	// i16, i32 and i64 share the varint encoding; a field of another type is corrupt
	if typ != T_I16 && typ != T_I32 && typ != T_I64 {t.bad = true}
	v := _t_varint(t)
	return i64(v >> 1) ~ -i64(v & 1)
}

@(private)
_t_i32 :: #force_inline proc(t: ^Thrift, typ: u8 = T_I32) -> i32 {return i32(_t_zigzag(t, typ))}

@(private)
_t_binary :: proc(t: ^Thrift) -> []u8 {
	n := int(_t_varint(t))
	if t.bad || n < 0 || n > len(t.data) - t.pos {
		t.bad = true
		return nil
	}
	b := t.data[t.pos:t.pos + n]
	t.pos += n
	return b
}

@(private)
_t_string :: #force_inline proc(t: ^Thrift, typ: u8 = T_BINARY) -> string {
	if typ != T_BINARY {
		t.bad = true
		return ""
	}
	return string(_t_binary(t))
}

// Next field of a struct: (type, id); type T_STOP at the end
@(private)
_t_field :: proc(t: ^Thrift, last: ^i16) -> (typ: u8, id: i16) {
	b := _t_byte(t)
	if t.bad || b == 0 {return T_STOP, 0}
	typ = b & 0x0f
	if typ > T_STRUCT {
		t.bad = true
		return T_STOP, 0
	}
	delta := i16(b >> 4)
	if delta != 0 {
		id = last^ + delta
	} else {
		id = i16(_t_zigzag(t))
	}
	last^ = id
	return
}

@(private)
_t_list :: proc(t: ^Thrift) -> (elem: u8, n: int) {
	b := _t_byte(t)
	elem = b & 0x0f
	n = int(b >> 4)
	if n == 15 {n = int(_t_varint(t))}
	if n < 0 || n > len(t.data) - t.pos { 	// every element takes at least a byte
		t.bad = true
		n = 0
	}
	return
}

// A list of structs in a field of type `typ`: its length
@(private)
_t_struct_list :: proc(t: ^Thrift, typ: u8) -> int {
	if typ != T_LIST {
		t.bad = true
		return 0
	}
	elem, n := _t_list(t)
	if elem != T_STRUCT && n > 0 {
		t.bad = true
		return 0
	}
	return n
}

// A boolean field (its value lives in the field header)
@(private)
_t_bool_field :: #force_inline proc(t: ^Thrift, typ: u8) -> bool {
	if typ != T_TRUE && typ != T_FALSE {t.bad = true}
	return typ == T_TRUE
}

@(private)
_t_skip :: proc(t: ^Thrift, typ: u8, depth := 0) {
	if depth > 64 {
		t.bad = true
		return
	}
	switch typ {
	case T_TRUE, T_FALSE:
	// the value lives in the field header
	case T_BYTE:
		_t_byte(t)
	case T_I16, T_I32, T_I64:
		_t_varint(t)
	case T_DOUBLE:
		if t.pos + 8 > len(t.data) {t.bad = true} else {t.pos += 8}
	case T_BINARY:
		_t_binary(t)
	case T_LIST, T_SET:
		elem, n := _t_list(t)
		for _ in 0 ..< n {
			if t.bad {return}
			if elem == T_TRUE || elem == T_FALSE {
				_t_byte(t) // booleans inside a list take one byte each
			} else {
				_t_skip(t, elem, depth + 1)
			}
		}
	case T_MAP:
		n := int(_t_varint(t))
		if n == 0 {return}
		kv := _t_byte(t)
		for _ in 0 ..< n {
			if t.bad {return}
			_t_skip(t, kv >> 4, depth + 1)
			_t_skip(t, kv & 0x0f, depth + 1)
		}
	case T_STRUCT:
		last: i16
		for {
			ft, _ := _t_field(t, &last)
			if ft == T_STOP || t.bad {return}
			_t_skip(t, ft, depth + 1)
		}
	case:
		t.bad = true
	}
}

// ---------------------------------------------------------------------------
// Struct decoders

@(private)
_read_time_unit :: proc(t: ^Thrift, typ: u8 = T_STRUCT) -> Time_Unit {
	if typ != T_STRUCT {
		t.bad = true
		return .None
	}
	unit := Time_Unit.None
	last: i16
	for {
		typ, id := _t_field(t, &last)
		if typ == T_STOP {break}
		switch id {
		case 1:
			unit = .Millis
		case 2:
			unit = .Micros
		case 3:
			unit = .Nanos
		}
		_t_skip(t, typ) // each member is an empty struct
	}
	return unit
}

@(private)
_read_logical :: proc(t: ^Thrift, typ: u8 = T_STRUCT) -> Logical_Type {
	if typ != T_STRUCT {
		t.bad = true
		return {}
	}
	lt: Logical_Type
	last: i16
	for {
		typ, id := _t_field(t, &last)
		if typ == T_STOP {break}
		lt.kind = int(id)
		if typ != T_STRUCT {
			_t_skip(t, typ)
			continue
		}
		inner: i16
		for {
			ft, fid := _t_field(t, &inner)
			if ft == T_STOP {break}
			switch {
			case id == LT_DECIMAL && fid == 1:
				lt.scale = _t_i32(t, ft)
			case id == LT_DECIMAL && fid == 2:
				lt.precision = _t_i32(t, ft)
			case (id == LT_TIME || id == LT_TIMESTAMP) && fid == 1:
				lt.utc = _t_bool_field(t, ft)
			case (id == LT_TIME || id == LT_TIMESTAMP) && fid == 2:
				lt.unit = _read_time_unit(t, ft)
			case id == LT_INTEGER && fid == 1:
				lt.bit_width = i32(i8(_t_byte(t)))
			case id == LT_INTEGER && fid == 2:
				lt.signed = _t_bool_field(t, ft)
			case:
				_t_skip(t, ft)
			}
		}
	}
	return lt
}

@(private)
_read_schema_element :: proc(t: ^Thrift) -> Schema_Element {
	se := Schema_Element {
		type       = -1,
		repetition = -1,
		converted  = -1,
	}
	last: i16
	seen: bit_set[0 ..< 16]
	for {
		typ, id := _t_field(t, &last)
		if typ == T_STOP {break}
		if id >= 0 && id < 16 {seen += {int(id)}}
		switch id {
		case 1:
			se.type = _t_i32(t, typ)
		case 2:
			se.type_length = _t_i32(t, typ)
		case 3:
			se.repetition = _t_i32(t, typ)
		case 4:
			se.name = _t_string(t, typ)
		case 5:
			se.num_children = _t_i32(t, typ)
		case 6:
			se.converted = _t_i32(t, typ)
		case 7:
			se.scale = _t_i32(t, typ)
		case 8:
			se.precision = _t_i32(t, typ)
		case 10:
			se.logical = _read_logical(t, typ)
		case:
			_t_skip(t, typ)
		}
	}
	if !({4} <= seen) {t.bad = true} 	// a required field is missing
	return se
}

@(private)
_read_column_meta :: proc(t: ^Thrift, typ: u8 = T_STRUCT) -> Column_Meta {
	if typ != T_STRUCT {
		t.bad = true
		return {}
	}
	cm: Column_Meta
	last: i16
	seen: bit_set[0 ..< 16]
	for {
		typ, id := _t_field(t, &last)
		if typ == T_STOP {break}
		if id >= 0 && id < 16 {seen += {int(id)}}
		switch id {
		case 1:
			cm.type = _t_i32(t, typ)
		case 4:
			cm.codec = _t_i32(t, typ)
		case 5:
			cm.num_values = _t_zigzag(t, typ)
		case 7:
			cm.total_compressed_size = _t_zigzag(t, typ)
		case 9:
			cm.data_page_offset = _t_zigzag(t, typ)
		case 11:
			cm.dictionary_page_offset = _t_zigzag(t, typ)
			cm.has_dictionary_offset = true
		case:
			_t_skip(t, typ)
		}
	}
	if !({1, 4, 5, 7, 9} <= seen) {t.bad = true} 	// a required field is missing
	return cm
}

@(private)
_read_column_chunk :: proc(t: ^Thrift) -> Column_Chunk {
	cc: Column_Chunk
	last: i16
	for {
		typ, id := _t_field(t, &last)
		if typ == T_STOP {break}
		switch id {
		case 1:
			cc.file_path = _t_string(t, typ)
		case 3:
			cc.meta = _read_column_meta(t, typ)
			cc.has_meta = true
		case:
			_t_skip(t, typ)
		}
	}
	return cc
}

@(private)
_read_row_group :: proc(t: ^Thrift, allocator := context.allocator) -> Row_Group {
	rg: Row_Group
	rg.columns = make([dynamic]Column_Chunk, allocator)
	last: i16
	seen: bit_set[0 ..< 16]
	for {
		typ, id := _t_field(t, &last)
		if typ == T_STOP {break}
		if id >= 0 && id < 16 {seen += {int(id)}}
		switch id {
		case 1:
			n := _t_struct_list(t, typ)
			reserve(&rg.columns, n)
			for _ in 0 ..< n {
				if t.bad {break}
				append(&rg.columns, _read_column_chunk(t))
			}
		case 3:
			rg.num_rows = _t_zigzag(t, typ)
		case:
			_t_skip(t, typ)
		}
	}
	if !({1, 3} <= seen) {t.bad = true} 	// a required field is missing
	return rg
}

@(private)
_read_file_meta :: proc(t: ^Thrift, allocator := context.allocator) -> File_Meta {
	fm: File_Meta
	fm.schema = make([dynamic]Schema_Element, allocator)
	fm.row_groups = make([dynamic]Row_Group, allocator)
	last: i16
	seen: bit_set[0 ..< 16]
	for {
		typ, id := _t_field(t, &last)
		if typ == T_STOP {break}
		if id >= 0 && id < 16 {seen += {int(id)}}
		switch id {
		case 1:
			fm.version = _t_i32(t, typ)
		case 2:
			n := _t_struct_list(t, typ)
			reserve(&fm.schema, n)
			for _ in 0 ..< n {
				if t.bad {break}
				append(&fm.schema, _read_schema_element(t))
			}
		case 3:
			fm.num_rows = _t_zigzag(t, typ)
		case 4:
			n := _t_struct_list(t, typ)
			reserve(&fm.row_groups, n)
			for _ in 0 ..< n {
				if t.bad {break}
				append(&fm.row_groups, _read_row_group(t, allocator))
			}
		case 6:
			fm.created_by = _t_string(t, typ)
		case:
			_t_skip(t, typ)
		}
	}
	if !({2, 3, 4} <= seen) {t.bad = true} 	// a required field is missing
	return fm
}

@(private)
_destroy_file_meta :: proc(fm: ^File_Meta) {
	for &rg in fm.row_groups {delete(rg.columns)}
	delete(fm.row_groups)
	delete(fm.schema)
}

@(private)
_read_page_header :: proc(t: ^Thrift) -> Page_Header {
	ph: Page_Header
	last: i16
	seen: bit_set[0 ..< 16]
	for {
		typ, id := _t_field(t, &last)
		if typ == T_STOP {break}
		if id >= 0 && id < 16 {seen += {int(id)}}
		switch id {
		case 1:
			ph.type = _t_i32(t, typ)
		case 2:
			ph.uncompressed_size = _t_i32(t, typ)
		case 3:
			ph.compressed_size = _t_i32(t, typ)
		case 5:
			ph.has_data = true
			if typ != T_STRUCT {t.bad = true}
			inner: i16
			for {
				ft, fid := _t_field(t, &inner)
				if ft == T_STOP {break}
				switch fid {
				case 1:
					ph.data.num_values = _t_i32(t, ft)
				case 2:
					ph.data.encoding = _t_i32(t, ft)
				case:
					_t_skip(t, ft)
				}
			}
		case 7:
			ph.has_dict = true
			if typ != T_STRUCT {t.bad = true}
			inner: i16
			for {
				ft, fid := _t_field(t, &inner)
				if ft == T_STOP {break}
				switch fid {
				case 1:
					ph.dict.num_values = _t_i32(t, ft)
				case 2:
					ph.dict.encoding = _t_i32(t, ft)
				case:
					_t_skip(t, ft)
				}
			}
		case 8:
			ph.has_v2 = true
			if typ != T_STRUCT {t.bad = true}
			ph.v2.is_compressed = true // the default when the field is absent
			inner: i16
			for {
				ft, fid := _t_field(t, &inner)
				if ft == T_STOP {break}
				switch fid {
				case 1:
					ph.v2.num_values = _t_i32(t, ft)
				case 2:
					ph.v2.num_nulls = _t_i32(t, ft)
				case 3:
					ph.v2.num_rows = _t_i32(t, ft)
				case 4:
					ph.v2.encoding = _t_i32(t, ft)
				case 5:
					ph.v2.def_len = _t_i32(t, ft)
				case 6:
					ph.v2.rep_len = _t_i32(t, ft)
				case 7:
					ph.v2.is_compressed = _t_bool_field(t, ft)
				case:
					_t_skip(t, ft)
				}
			}
		case:
			_t_skip(t, typ)
		}
	}
	if !({1, 2, 3} <= seen) {t.bad = true} 	// a required field is missing
	return ph
}
