package parquet

import "base:intrinsics"
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

// Compresses `src` into `dst` (overwritten) as one Snappy block. The input is
// cut into 64 KiB fragments, as the reference compressor does, so every copy
// offset fits two bytes. Matches are found through a hash table of 4-byte
// sequences; on data without matches the scan skips ahead faster and faster,
// so incompressible input costs little.
@(private)
snappy_compress :: proc(dst: ^[dynamic]u8, src: []u8) #no_bounds_check {
	clear(dst)
	reserve(dst, 32 + len(src) + len(src) / 6)
	v := u64(len(src))
	for v >= 0x80 {
		append(dst, u8(v) | 0x80)
		v >>= 7
	}
	append(dst, u8(v))

	FRAG :: 1 << 16
	HASH_BITS :: 14
	table: [1 << HASH_BITS]u16
	for start := 0; start < len(src); start += FRAG {
		block := src[start:min(start + FRAG, len(src))]
		n := len(block)
		lit := 0
		if n >= 15 {
			table = {}
			load :: #force_inline proc(b: []u8, i: int) -> u32 {
				return u32(intrinsics.unaligned_load((^u32le)(&b[i])))
			}
			hash :: #force_inline proc(x: u32) -> u32 {return (x * 0x1e35a7bd) >> (32 - HASH_BITS)}
			ip := 1
			limit := n - 4
			misses := 32
			for ip <= limit {
				cur := load(block, ip)
				h := hash(cur)
				cand := int(table[h])
				table[h] = u16(ip)
				if cand >= ip || load(block, cand) != cur {
					ip += misses >> 5 // 1 at first, more after a long run without matches
					misses += 1
					continue
				}
				misses = 32
				_snappy_literal(dst, block[lit:ip])
				m := 4
				for ip + m < n && block[cand + m] == block[ip + m] {m += 1}
				_snappy_copy(dst, ip - cand, m)
				ip += m
				lit = ip
				if ip - 1 <= limit {table[hash(load(block, ip - 1))] = u16(ip - 1)}
			}
		}
		_snappy_literal(dst, block[lit:])
	}
}

@(private)
_snappy_literal :: proc(dst: ^[dynamic]u8, b: []u8) {
	n := len(b)
	if n == 0 {return}
	switch {
	case n <= 60:
		append(dst, u8(n - 1) << 2)
	case n <= 1 << 8:
		append(dst, 60 << 2, u8(n - 1))
	case n <= 1 << 16:
		append(dst, 61 << 2, u8(n - 1), u8((n - 1) >> 8))
	case n <= 1 << 24:
		append(dst, 62 << 2, u8(n - 1), u8((n - 1) >> 8), u8((n - 1) >> 16))
	case:
		append(dst, 63 << 2, u8(n - 1), u8((n - 1) >> 8), u8((n - 1) >> 16), u8((n - 1) >> 24))
	}
	append(dst, ..b)
}

// A copy of `length` bytes from `offset` back (offset < 65536)
@(private)
_snappy_copy :: proc(dst: ^[dynamic]u8, offset, length: int) {
	length := length
	for length > 0 {
		k := min(length, 64)
		if length - k > 0 && length - k < 4 {k = length - 4} 	// keep every piece >= 4
		if k >= 4 && k <= 11 && offset < 2048 {
			append(dst, u8(1 | (k - 4) << 2 | (offset >> 8) << 5), u8(offset))
		} else {
			append(dst, u8(2 | (k - 1) << 2), u8(offset), u8(offset >> 8))
		}
		length -= k
	}
}
