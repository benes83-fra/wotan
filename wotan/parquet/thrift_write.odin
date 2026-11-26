package parquet

// Thrift compact protocol writer (the encoding thrift.odin reads). Fields
// must be written in increasing id order within a struct, as parquet.thrift
// numbers them; the short header form (id delta in the high nibble) is used
// whenever the delta fits.

@(private)
Thrift_Writer :: struct {
	out:   ^[dynamic]u8,
	last:  [32]i16, // last field id per open struct
	depth: int,
}

@(private)
_tw_byte :: #force_inline proc(t: ^Thrift_Writer, b: u8) {append(t.out, b)}

@(private)
_tw_varint :: proc(t: ^Thrift_Writer, v: u64) {
	v := v
	for v >= 0x80 {
		append(t.out, u8(v) | 0x80)
		v >>= 7
	}
	append(t.out, u8(v))
}

@(private)
_tw_zigzag :: #force_inline proc(t: ^Thrift_Writer, v: i64) {
	_tw_varint(t, u64((v << 1) ~ (v >> 63)))
}

@(private)
_tw_field :: proc(t: ^Thrift_Writer, id: i16, typ: u8) {
	delta := id - t.last[t.depth]
	if delta > 0 && delta <= 15 {
		_tw_byte(t, u8(delta) << 4 | typ)
	} else {
		_tw_byte(t, typ)
		_tw_zigzag(t, i64(id))
	}
	t.last[t.depth] = id
}

@(private)
_tw_i32 :: proc(t: ^Thrift_Writer, id: i16, v: i32) {
	_tw_field(t, id, T_I32)
	_tw_zigzag(t, i64(v))
}

@(private)
_tw_i64 :: proc(t: ^Thrift_Writer, id: i16, v: i64) {
	_tw_field(t, id, T_I64)
	_tw_zigzag(t, v)
}

@(private)
_tw_bool :: proc(t: ^Thrift_Writer, id: i16, v: bool) {
	_tw_field(t, id, T_TRUE if v else T_FALSE)
}

@(private)
_tw_binary :: proc(t: ^Thrift_Writer, id: i16, b: []u8) {
	_tw_field(t, id, T_BINARY)
	_tw_varint(t, u64(len(b)))
	append(t.out, ..b)
}

@(private)
_tw_string :: #force_inline proc(t: ^Thrift_Writer, id: i16, s: string) {
	_tw_binary(t, id, transmute([]u8)s)
}

// Opens a struct-typed field; close it with _tw_end
@(private)
_tw_struct :: proc(t: ^Thrift_Writer, id: i16) {
	_tw_field(t, id, T_STRUCT)
	_tw_open(t)
}

// Opens a struct that is a list element (no field header)
@(private)
_tw_open :: proc(t: ^Thrift_Writer) {
	t.depth += 1
	t.last[t.depth] = 0
}

@(private)
_tw_end :: proc(t: ^Thrift_Writer) {
	_tw_byte(t, T_STOP)
	t.depth -= 1
}

// A list-typed field of `n` elements of type `elem`; the elements follow
@(private)
_tw_list :: proc(t: ^Thrift_Writer, id: i16, elem: u8, n: int) {
	_tw_field(t, id, T_LIST)
	if n < 15 {
		_tw_byte(t, u8(n) << 4 | elem)
	} else {
		_tw_byte(t, 0xf0 | elem)
		_tw_varint(t, u64(n))
	}
}
