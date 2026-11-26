package parquet

import "core:mem"

// Snappy raw block format (the form Parquet uses, without the framing
// format's stream header). A block starts with the uncompressed length as a
// varint, followed by elements whose tag byte's low two bits give the kind:
//   00  literal: length-1 in the upper six bits (60..63: the length follows
//       in 1..4 little-endian bytes), then the bytes
//   01  copy, length 4..11 in bits 2..4, offset of 11 bits (3 high bits in the
//       tag, 8 in the next byte)
//   10  copy, length 1..64 in the upper six bits, 2-byte offset
//   11  copy, length 1..64 in the upper six bits, 4-byte offset
// A copy repeats `length` bytes starting `offset` bytes back in the output;
// the ranges may overlap (offset < length repeats a pattern).

// The uncompressed length announced by a Snappy block, and the size of its
// varint
@(private)
_snappy_length :: proc(src: []u8) -> (n: int, header: int, ok: bool) {
	v: u64
	shift: u32
	for i in 0 ..< min(len(src), 10) {
		b := src[i]
		v |= u64(b & 0x7f) << shift
		if b & 0x80 == 0 {
			if v > u64(max(i32)) {return 0, 0, false}
			return int(v), i + 1, true
		}
		shift += 7
	}
	return 0, 0, false
}

// Decompresses `src` into `dst`, which must have exactly the announced length
@(private)
snappy_decompress :: proc(dst, src: []u8) -> bool #no_bounds_check {
	n, s, ok := _snappy_length(src)
	if !ok || n != len(dst) {return false}
	d := 0
	for s < len(src) {
		tag := src[s]
		s += 1
		length, offset: int
		switch tag & 3 {
		case 0:
			length = int(tag >> 2)
			if length >= 60 {
				extra := length - 59
				if s + extra > len(src) {return false}
				length = 0
				for k in 0 ..< extra {length |= int(src[s + k]) << uint(8 * k)}
				s += extra
			}
			length += 1
			if length > len(src) - s || length > n - d {return false}
			mem.copy_non_overlapping(&dst[d], &src[s], length)
			d += length
			s += length
			continue
		case 1:
			if s >= len(src) {return false}
			length = 4 + int((tag >> 2) & 7)
			offset = int(tag >> 5) << 8 | int(src[s])
			s += 1
		case 2:
			if s + 2 > len(src) {return false}
			length = 1 + int(tag >> 2)
			offset = int(src[s]) | int(src[s + 1]) << 8
			s += 2
		case 3:
			if s + 4 > len(src) {return false}
			length = 1 + int(tag >> 2)
			offset =
				int(src[s]) | int(src[s + 1]) << 8 | int(src[s + 2]) << 16 | int(src[s + 3]) << 24
			s += 4
		}
		if offset == 0 || offset > d || length > n - d {return false}
		if offset >= length {
			mem.copy_non_overlapping(&dst[d], &dst[d - offset], length)
		} else {
			// overlapping: the copied bytes feed the copy itself
			for k in 0 ..< length {dst[d + k] = dst[d - offset + k]}
		}
		d += length
	}
	return d == n
}
