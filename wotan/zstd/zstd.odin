package zstd

import "base:intrinsics"
import "core:hash/xxhash"
import "core:mem"

// Zstandard decompression (RFC 8878), written for Parquet pages but usable on
// any Zstandard data: one or more frames, skippable frames, raw, RLE and
// compressed blocks, Huffman-coded literals (1 or 4 streams, with tables
// reused across blocks), FSE-coded sequences in all four table modes, repeat
// offsets, and the optional content checksum (verified). Frames that need a
// dictionary are refused.
//
// Every read is bounds-checked against the input and every write against the
// output, so damaged data gives ok = false, never a crash or an overrun.

// Decompresses `src` into `dst`, which must be large enough for the whole
// content. Returns the number of bytes written.
decompress :: proc(dst, src: []u8) -> (written: int, ok: bool) {
	s := 0
	d := 0
	if len(src) == 0 {return 0, false}
	for s < len(src) {
		if len(src) - s < 4 {return 0, false}
		magic := _u32(src, s)
		if magic & 0xFFFFFFF0 == 0x184D2A50 { 	// skippable frame
			if len(src) - s < 8 {return 0, false}
			size := int(_u32(src, s + 4))
			if size > len(src) - s - 8 {return 0, false}
			s += 8 + size
			continue
		}
		if magic != 0xFD2FB528 {return 0, false}
		n, used, fok := _frame(dst[d:], src[s + 4:])
		if !fok {return 0, false}
		d += n
		s += 4 + used
	}
	return d, true
}

// The content size the frames of `src` declare, if every frame declares one
decompressed_size :: proc(src: []u8) -> (size: int, ok: bool) {
	s := 0
	for s < len(src) {
		if len(src) - s < 5 {return 0, false}
		magic := _u32(src, s)
		if magic & 0xFFFFFFF0 == 0x184D2A50 {
			if len(src) - s < 8 {return 0, false}
			s += 8 + int(_u32(src, s + 4))
			continue
		}
		if magic != 0xFD2FB528 {return 0, false}
		h, hok := _header(src[s + 4:])
		if !hok || h.content_size < 0 {return 0, false}
		size += h.content_size
		// skip the frame's blocks
		p := s + 4 + h.size
		for {
			if len(src) - p < 3 {return 0, false}
			bh := int(src[p]) | int(src[p + 1]) << 8 | int(src[p + 2]) << 16
			bsize := bh >> 3
			if (bh >> 1) & 3 == 1 {bsize = 1}
			p += 3 + bsize
			if bh & 1 != 0 {break}
		}
		if h.checksum {p += 4}
		if p > len(src) {return 0, false}
		s = p
	}
	return size, true
}

// ---------------------------------------------------------------------------

@(private)
_u32 :: #force_inline proc(b: []u8, i: int) -> u32 {
	return u32(b[i]) | u32(b[i + 1]) << 8 | u32(b[i + 2]) << 16 | u32(b[i + 3]) << 24
}

@(private)
Header :: struct {
	size:         int, // bytes of the frame header (after the magic)
	content_size: int, // -1 if not declared
	checksum:     bool,
}

@(private)
_header :: proc(src: []u8) -> (h: Header, ok: bool) {
	if len(src) < 1 {return}
	fhd := src[0]
	fcs_flag := int(fhd >> 6)
	single := fhd & 0x20 != 0
	if fhd & 0x08 != 0 {return} 	// reserved bit
	h.checksum = fhd & 0x04 != 0
	dict_flag := int(fhd & 3)
	p := 1
	if !single {p += 1} 	// window descriptor (the whole output is the window here)
	DICT_BYTES := [4]int{0, 1, 2, 4}
	dict_bytes := DICT_BYTES[dict_flag]
	if len(src) < p + dict_bytes {return}
	dict_id: u32
	for k in 0 ..< dict_bytes {dict_id |= u32(src[p + k]) << uint(8 * k)}
	if dict_id != 0 {return} 	// needs a dictionary
	p += dict_bytes
	fcs_table := [4]int{1 if single else 0, 2, 4, 8}
	fcs_bytes := fcs_table[fcs_flag]
	if len(src) < p + fcs_bytes {return}
	h.content_size = -1
	if fcs_bytes > 0 {
		v: u64
		for k in 0 ..< fcs_bytes {v |= u64(src[p + k]) << uint(8 * k)}
		if fcs_bytes == 2 {v += 256}
		if v > u64(max(int) / 2) {return}
		h.content_size = int(v)
	}
	p += fcs_bytes
	h.size = p
	return h, true
}

// FSE decoding table entry
@(private)
Fse_Entry :: struct {
	symbol:    u8,
	bits:      u8,
	new_state: u16,
}

@(private)
Fse_Table :: struct {
	entries: [512]Fse_Entry, // up to accuracy log 9
	log:     int,
	present: bool,
}

@(private)
Huf_Entry :: struct {
	symbol: u8,
	bits:   u8,
}

@(private)
HUF_MAX_BITS :: 11

@(private)
Frame_State :: struct {
	huf:        [1 << HUF_MAX_BITS]Huf_Entry,
	huf_bits:   int, // 0: no table yet
	ll, of, ml: Fse_Table,
	rep:        [3]int,
	literals:   [1 << 17 + 32]u8, // 32 spare bytes for 16-byte copies
}

// One frame (after its magic) into dst: bytes written and bytes read
@(private)
_frame :: proc(dst, src: []u8) -> (written, used: int, ok: bool) {
	h := _header(src) or_return
	p := h.size
	st := new(Frame_State)
	defer free(st)
	st.rep = {1, 4, 8}
	d := 0
	for {
		if len(src) - p < 3 {return}
		bh := int(src[p]) | int(src[p + 1]) << 8 | int(src[p + 2]) << 16
		p += 3
		last := bh & 1 != 0
		bsize := bh >> 3
		switch (bh >> 1) & 3 {
		case 0:
			// raw
			if bsize > len(src) - p || bsize > len(dst) - d {return}
			intrinsics.mem_copy_non_overlapping(raw_data(dst[d:]), raw_data(src[p:]), bsize)
			d += bsize
			p += bsize
		case 1:
			// RLE: one byte, repeated bsize times
			if p >= len(src) || bsize > len(dst) - d {return}
			mem.set(raw_data(dst[d:]), src[p], bsize)
			d += bsize
			p += 1
		case 2:
			if bsize > len(src) - p || bsize > 1 << 17 {return}
			n := _block(st, dst, d, src[p:p + bsize]) or_return
			d += n
			p += bsize
		case 3:
			return
		}
		if last {break}
	}
	if h.content_size >= 0 && d != h.content_size {return}
	if h.checksum {
		if len(src) - p < 4 {return}
		want := _u32(src, p)
		if u32(xxhash.XXH64(dst[:d])) != want {return}
		p += 4
	}
	return d, p, true
}

// ---------------------------------------------------------------------------
// Bit readers

// Forward little-endian bit reader (FSE table descriptions)
@(private)
Fwd_Bits :: struct {
	data: []u8,
	pos:  int, // bit position
}

@(private)
_fwd_read :: proc(b: ^Fwd_Bits, n: int) -> (v: u32, ok: bool) {
	if b.pos + n > len(b.data) * 8 {return 0, false}
	for k in 0 ..< n {
		p := b.pos + k
		v |= u32((b.data[p >> 3] >> uint(p & 7)) & 1) << uint(k)
	}
	b.pos += n
	return v, true
}

// Backward bit reader (Huffman streams and sequences): the stream is read
// from its last byte towards its first, most significant bits first; the
// last byte's highest set bit marks where the data starts. Reading past the
// start yields zero bits and is detected through `pos < 0`.
@(private)
Back_Bits :: struct {
	data: []u8,
	pos:  int, // bits still unread below this position
}

@(private)
_back_init :: proc(b: ^Back_Bits, data: []u8) -> bool {
	if len(data) == 0 {return false}
	last := data[len(data) - 1]
	if last == 0 {return false}
	b.data = data
	b.pos = (len(data) - 1) * 8 + int(7 - intrinsics.count_leading_zeros(last))
	return true
}

// Bits [p, p + n) of the stream as a little-endian number (n <= 56; bits
// below 0 are zero)
@(private)
_bits_at :: #force_inline proc(data: []u8, p, n: int) -> u64 #no_bounds_check {
	if n == 0 {return 0}
	p, n := p, n
	shift: uint
	if p < 0 {
		if n + p <= 0 {return 0}
		n += p
		shift = uint(-p)
		p = 0
	}
	i := p >> 3
	v: u64
	if i + 8 <= len(data) {
		v = u64(intrinsics.unaligned_load((^u64le)(&data[i])))
	} else {
		for k in 0 ..< len(data) - i {v |= u64(data[i + k]) << uint(8 * k)}
	}
	return ((v >> uint(p & 7)) & (u64(1) << uint(n) - 1)) << shift
}

@(private)
_back_read :: #force_inline proc(b: ^Back_Bits, n: int) -> u64 {
	b.pos -= n
	return _bits_at(b.data, b.pos, n)
}

@(private)
_back_peek :: #force_inline proc(b: ^Back_Bits, n: int) -> u64 {
	return _bits_at(b.data, b.pos - n, n)
}

// ---------------------------------------------------------------------------
// FSE

// Reads a table description (normalized counts) and builds the table;
// returns the bytes it took
@(private)
_fse_read :: proc(
	t: ^Fse_Table,
	src: []u8,
	max_symbol: int,
	max_log: int,
) -> (
	used: int,
	ok: bool,
) {
	b := Fwd_Bits {
		data = src,
	}
	v := _fwd_read(&b, 4) or_return
	log := int(v) + 5
	if log > max_log {return}
	norm: [256]i16
	remaining := (1 << uint(log)) + 1
	threshold := 1 << uint(log)
	nbits := log + 1
	sym := 0
	for remaining > 1 && sym <= max_symbol {
		// the value takes nbits-1 or nbits bits
		max_small := 2 * threshold - 1 - remaining
		low := 0
		if b.pos + nbits - 1 <= len(src) * 8 {
			low = int(_bits_at(src, b.pos, nbits - 1))
		} else {
			return
		}
		value: int
		if low < max_small {
			value = low
			b.pos += nbits - 1
		} else {
			full := _fwd_read(&b, nbits) or_return
			value = int(full)
			if value >= threshold {value -= max_small}
		}
		count := value - 1
		remaining -= abs(count)
		norm[sym] = i16(count)
		sym += 1
		if count == 0 {
			// runs of zero probabilities: 2-bit repeat counts, 3 = continue
			for {
				r := _fwd_read(&b, 2) or_return
				for _ in 0 ..< r {
					if sym > max_symbol {return}
					norm[sym] = 0
					sym += 1
				}
				if r != 3 {break}
			}
		}
		for remaining < threshold && threshold > 1 {
			nbits -= 1
			threshold >>= 1
		}
	}
	if remaining != 1 || sym > max_symbol + 1 {return}
	_fse_build(t, norm[:sym], log) or_return
	return (b.pos + 7) / 8, true
}

@(private)
_fse_build :: proc(t: ^Fse_Table, norm: []i16, log: int) -> bool {
	size := 1 << uint(log)
	if size > len(t.entries) {return false}
	next: [256]u16
	high := size - 1
	for c, s in norm {
		if c == -1 {
			t.entries[high].symbol = u8(s)
			high -= 1
			next[s] = 1
		} else {
			next[s] = u16(max(c, 0))
		}
	}
	step := (size >> 1) + (size >> 3) + 3
	mask := size - 1
	pos := 0
	for c, s in norm {
		for _ in 0 ..< int(c) {
			t.entries[pos].symbol = u8(s)
			pos = (pos + step) & mask
			for pos > high {pos = (pos + step) & mask}
		}
	}
	if pos != 0 {return false}
	for u in 0 ..< size {
		s := t.entries[u].symbol
		ns := int(next[s])
		next[s] += 1
		if ns == 0 {return false}
		bits := log - int(31 - intrinsics.count_leading_zeros(u32(ns)))
		t.entries[u].bits = u8(bits)
		t.entries[u].new_state = u16((ns << uint(bits)) - size)
	}
	t.log = log
	t.present = true
	return true
}

@(private)
_fse_rle :: proc(t: ^Fse_Table, symbol: u8) {
	t.entries[0] = {symbol, 0, 0}
	t.log = 0
	t.present = true
}

// ---------------------------------------------------------------------------
// Literals

// Decodes the literals section; returns the literals and the bytes it took
@(private)
_literals :: proc(st: ^Frame_State, src: []u8) -> (lits: []u8, used: int, ok: bool) {
	if len(src) < 1 {return}
	b0 := int(src[0])
	kind := b0 & 3
	fmt := (b0 >> 2) & 3
	switch kind {
	case 0, 1:
		// raw, RLE
		size, hdr: int
		switch fmt {
		case 0, 2:
			size, hdr = b0 >> 3, 1
		case 1:
			if len(src) < 2 {return}
			size, hdr = b0 >> 4 + int(src[1]) << 4, 2
		case 3:
			if len(src) < 3 {return}
			size, hdr = b0 >> 4 + int(src[1]) << 4 + int(src[2]) << 12, 3
		}
		if size > len(st.literals) {return}
		if kind == 0 {
			if size > len(src) - hdr {return}
			copy(st.literals[:size], src[hdr:hdr + size])
			return st.literals[:size], hdr + size, true
		}
		if hdr >= len(src) {return}
		mem.set(&st.literals[0], src[hdr], size)
		return st.literals[:size], hdr + 1, true
	}
	// compressed (2) or treeless (3)
	streams := 1 if fmt == 0 else 4
	HDR := [4]int{3, 3, 4, 5}
	BITS := [4]uint{10, 10, 14, 18}
	hdr := HDR[fmt]
	bits := BITS[fmt]
	if len(src) < hdr {return}
	hv: u64
	for k in 0 ..< hdr {hv |= u64(src[k]) << uint(8 * k)}
	mask := u64(1) << bits - 1
	regen := int((hv >> 4) & mask)
	csize := int((hv >> (4 + bits)) & mask)
	if regen > len(st.literals) || csize > len(src) - hdr {return}
	body := src[hdr:hdr + csize]
	if kind == 2 {
		n := _huf_read(st, body) or_return
		body = body[n:]
	} else if st.huf_bits == 0 {
		return
	}
	out := st.literals[:regen]
	if streams == 1 {
		_huf_stream(st, body, out) or_return
	} else {
		if len(body) < 6 {return}
		s1 := int(body[0]) | int(body[1]) << 8
		s2 := int(body[2]) | int(body[3]) << 8
		s3 := int(body[4]) | int(body[5]) << 8
		rest := body[6:]
		if s1 + s2 + s3 > len(rest) {return}
		per := (regen + 3) / 4
		if 3 * per > regen {return}
		runs := [4]Huf_Run {
			{data = rest[:s1], out = out[:per]},
			{data = rest[s1:s1 + s2], out = out[per:2 * per]},
			{data = rest[s1 + s2:s1 + s2 + s3], out = out[2 * per:3 * per]},
			{data = rest[s1 + s2 + s3:], out = out[3 * per:]},
		}
		_huf_four(st, &runs) or_return
	}
	return out, hdr + csize, true
}

// Reads a Huffman tree description and builds the decoding table; returns
// the bytes it took
@(private)
_huf_read :: proc(st: ^Frame_State, src: []u8) -> (used: int, ok: bool) {
	if len(src) < 1 {return}
	weights: [256]u8
	n := 0
	hb := int(src[0])
	if hb >= 128 {
		n = hb - 127
		nbytes := (n + 1) / 2
		if nbytes > len(src) - 1 {return}
		for i in 0 ..< n {
			b := src[1 + i / 2]
			weights[i] = b >> 4 if i % 2 == 0 else b & 15
		}
		used = 1 + nbytes
	} else {
		// FSE-compressed weights: two interleaved states on one stream
		if hb == 0 || hb > len(src) - 1 {return}
		data := src[1:1 + hb]
		t: Fse_Table
		tl := _fse_read(&t, data, 255, 6) or_return
		if tl >= len(data) {return}
		br: Back_Bits
		_back_init(&br, data[tl:]) or_return
		s1 := int(_back_read(&br, t.log))
		s2 := int(_back_read(&br, t.log))
		for {
			if n >= 255 {return}
			e := t.entries[s1]
			weights[n] = e.symbol
			n += 1
			s1 = int(e.new_state) + int(_back_read(&br, int(e.bits)))
			if br.pos < 0 {
				if n >= 255 {return}
				weights[n] = t.entries[s2].symbol
				n += 1
				break
			}
			if n >= 255 {return}
			e2 := t.entries[s2]
			weights[n] = e2.symbol
			n += 1
			s2 = int(e2.new_state) + int(_back_read(&br, int(e2.bits)))
			if br.pos < 0 {
				if n >= 255 {return}
				weights[n] = t.entries[s1].symbol
				n += 1
				break
			}
		}
		used = 1 + hb
	}
	// the last symbol's weight completes the total to a power of two
	total := 0
	for i in 0 ..< n {
		w := int(weights[i])
		if w > HUF_MAX_BITS + 1 {return}
		if w > 0 {total += 1 << uint(w - 1)}
	}
	if total == 0 {return}
	max_bits := int(32 - intrinsics.count_leading_zeros(u32(total)))
	if max_bits > HUF_MAX_BITS {return}
	left := (1 << uint(max_bits)) - total
	if left <= 0 || left & (left - 1) != 0 {return}
	weights[n] = u8(int(31 - intrinsics.count_leading_zeros(u32(left))) + 1)
	n += 1
	// fill the table: by weight ascending, then by symbol
	start: [HUF_MAX_BITS + 2]int
	count: [HUF_MAX_BITS + 2]int
	for i in 0 ..< n {count[weights[i]] += 1}
	pos := 0
	for w in 1 ..= max_bits {
		start[w] = pos
		pos += count[w] << uint(w - 1)
	}
	if pos != 1 << uint(max_bits) {return}
	for i in 0 ..< n {
		w := int(weights[i])
		if w == 0 {continue}
		span := 1 << uint(w - 1)
		e := Huf_Entry{u8(i), u8(max_bits + 1 - w)}
		for k in 0 ..< span {st.huf[start[w] + k] = e}
		start[w] += span
	}
	st.huf_bits = max_bits
	return used, true
}

@(private)
_huf_stream :: proc(st: ^Frame_State, src: []u8, out: []u8) -> bool {
	br: Back_Bits
	if !_back_init(&br, src) {return len(out) == 0 && len(src) == 0}
	return _huf_finish(st, br.data, br.pos, out, 0)
}

// One of the four streams of a literals section
@(private)
Huf_Run :: struct {
	data: []u8,
	pos:  int,
	out:  []u8,
	i:    int,
}

// The four streams, interleaved while each has room for a full round (five
// symbols of at most 11 bits from one 8-byte load per stream), so the CPU
// overlaps their independent table lookups; then each stream on its own
@(private)
_huf_four :: proc(st: ^Frame_State, runs: ^[4]Huf_Run) -> bool #no_bounds_check {
	for &r in runs {
		br: Back_Bits
		if !_back_init(&br, r.data) {
			if len(r.out) != 0 || len(r.data) != 0 {return false}
			r.pos = 0
			continue
		}
		r.pos = br.pos
	}
	mb := uint(st.huf_bits)
	mask := u64(1) << mb - 1
	for {
		ready := true
		for &r in runs {
			if r.pos < 64 || len(r.out) - r.i < 5 {ready = false}
		}
		if !ready {break}
		c: [4]u64
		avail: [4]uint
		base: [4]int
		#unroll for k in 0 ..< 4 {
			base[k] = (runs[k].pos >> 3) - 7
			c[k] = u64(intrinsics.unaligned_load((^u64le)(&runs[k].data[base[k]])))
			avail[k] = uint(runs[k].pos - 8 * base[k])
		}
		#unroll for _ in 0 ..< 5 {
			#unroll for k in 0 ..< 4 {
				e := st.huf[(c[k] >> (avail[k] - mb)) & mask]
				runs[k].out[runs[k].i] = e.symbol
				runs[k].i += 1
				avail[k] -= uint(e.bits)
			}
		}
		#unroll for k in 0 ..< 4 {runs[k].pos = 8 * base[k] + int(avail[k])}
	}
	for &r in runs {
		if len(r.data) == 0 {continue}
		_huf_finish(st, r.data, r.pos, r.out, r.i) or_return
	}
	return true
}

// Decodes out[i:] from a backward stream whose unread bits end at `pos`
@(private)
_huf_finish :: proc(
	st: ^Frame_State,
	data: []u8,
	pos: int,
	out: []u8,
	i: int,
) -> bool #no_bounds_check {
	br := Back_Bits {
		data = data,
		pos  = pos,
	}
	mb := st.huf_bits
	mask := u64(1) << uint(mb) - 1
	pos := pos
	i := i
	n := len(out)
	// fast path: one 8-byte load holds the next 56+ bits, enough for several
	// symbols of at most 11 bits each
	for pos >= 64 && i < n {
		s := (pos >> 3) - 7
		c := u64(intrinsics.unaligned_load((^u64le)(&data[s])))
		avail := pos - 8 * s // bits of c below pos: 56 ..= 63
		for avail >= mb && i < n {
			e := st.huf[(c >> uint(avail - mb)) & mask]
			out[i] = e.symbol
			avail -= int(e.bits)
			i += 1
		}
		pos = 8 * s + avail
	}
	br.pos = pos
	for ; i < n; i += 1 {
		e := st.huf[_back_peek(&br, mb)]
		out[i] = e.symbol
		br.pos -= int(e.bits)
	}
	return br.pos == 0
}

// ---------------------------------------------------------------------------
// Sequences

@(private)
LL_BASE := [36]u32 {
	0,
	1,
	2,
	3,
	4,
	5,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
	18,
	20,
	22,
	24,
	28,
	32,
	40,
	48,
	64,
	128,
	256,
	512,
	1024,
	2048,
	4096,
	8192,
	16384,
	32768,
	65536,
}
@(private)
LL_BITS := [36]u8 {
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	1,
	1,
	1,
	1,
	2,
	2,
	3,
	3,
	4,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
}
@(private)
ML_BASE := [53]u32 {
	3,
	4,
	5,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
	17,
	18,
	19,
	20,
	21,
	22,
	23,
	24,
	25,
	26,
	27,
	28,
	29,
	30,
	31,
	32,
	33,
	34,
	35,
	37,
	39,
	41,
	43,
	47,
	51,
	59,
	67,
	83,
	99,
	131,
	259,
	515,
	1027,
	2051,
	4099,
	8195,
	16387,
	32771,
	65539,
}
@(private)
ML_BITS := [53]u8 {
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	1,
	1,
	1,
	1,
	2,
	2,
	3,
	3,
	4,
	4,
	5,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
}
@(private)
LL_DEFAULT := [36]i16 {
	4,
	3,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	3,
	2,
	1,
	1,
	1,
	1,
	1,
	-1,
	-1,
	-1,
	-1,
}
@(private)
ML_DEFAULT := [53]i16 {
	1,
	4,
	3,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
}
@(private)
OF_DEFAULT := [29]i16 {
	1,
	1,
	1,
	1,
	1,
	1,
	2,
	2,
	2,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	-1,
	-1,
	-1,
	-1,
	-1,
}

// One table of the sequences section, by its mode; returns the bytes it took
@(private)
_seq_table :: proc(
	t: ^Fse_Table,
	mode: int,
	src: []u8,
	default: []i16,
	default_log, max_symbol, max_log: int,
) -> (
	used: int,
	ok: bool,
) {
	switch mode {
	case 0:
		_fse_build(t, default, default_log) or_return
		return 0, true
	case 1:
		if len(src) < 1 || int(src[0]) > max_symbol {return}
		_fse_rle(t, src[0])
		return 1, true
	case 2:
		return _fse_read(t, src, max_symbol, max_log)
	case 3:
		return 0, t.present
	}
	return
}

// A compressed block into dst[d:]; returns the bytes written
@(private)
_block :: proc(
	st: ^Frame_State,
	dst: []u8,
	d: int,
	src: []u8,
) -> (
	written: int,
	ok: bool,
) #no_bounds_check {
	lits, p := _literals(st, src) or_return
	if p >= len(src) {return}
	// number of sequences
	b0 := int(src[p])
	nseq: int
	switch {
	case b0 == 0:
		nseq = 0
		p += 1
	case b0 < 128:
		nseq = b0
		p += 1
	case b0 < 255:
		if p + 1 >= len(src) {return}
		nseq = (b0 - 128) << 8 + int(src[p + 1])
		p += 2
	case:
		if p + 2 >= len(src) {return}
		nseq = int(src[p + 1]) + int(src[p + 2]) << 8 + 0x7F00
		p += 3
	}
	o := d
	if nseq == 0 {
		if len(lits) > len(dst) - o {return}
		copy(dst[o:], lits)
		return len(lits), true
	}
	if p >= len(src) {return}
	modes := int(src[p])
	p += 1
	if modes & 3 != 0 {return}
	n := _seq_table(&st.ll, modes >> 6, src[p:], LL_DEFAULT[:], 6, 35, 9) or_return
	p += n
	n = _seq_table(&st.of, (modes >> 4) & 3, src[p:], OF_DEFAULT[:], 5, 31, 8) or_return
	p += n
	n = _seq_table(&st.ml, (modes >> 2) & 3, src[p:], ML_DEFAULT[:], 6, 52, 9) or_return
	p += n
	if p >= len(src) {return}

	br: Back_Bits
	_back_init(&br, src[p:]) or_return
	ll_s := int(_back_read(&br, st.ll.log))
	of_s := int(_back_read(&br, st.of.log))
	ml_s := int(_back_read(&br, st.ml.log))
	li := 0 // next literal
	for k in 0 ..< nseq {
		lle := st.ll.entries[ll_s]
		ofe := st.of.entries[of_s]
		mle := st.ml.entries[ml_s]
		of_code := int(ofe.symbol)
		if of_code > 31 || int(lle.symbol) > 35 || int(mle.symbol) > 52 {return}
		ov := int(1 << uint(of_code)) + int(_back_read(&br, of_code))
		ml := int(ML_BASE[mle.symbol]) + int(_back_read(&br, int(ML_BITS[mle.symbol])))
		ll := int(LL_BASE[lle.symbol]) + int(_back_read(&br, int(LL_BITS[lle.symbol])))
		// offset: a new one, or one of the three most recent
		off: int
		if ov > 3 {
			off = ov - 3
			st.rep[2] = st.rep[1]
			st.rep[1] = st.rep[0]
			st.rep[0] = off
		} else {
			idx := ov - 1 + (1 if ll == 0 else 0)
			if idx == 0 {
				off = st.rep[0]
			} else {
				off = st.rep[0] - 1 if idx == 3 else st.rep[idx]
				if idx >= 2 {st.rep[2] = st.rep[1]}
				st.rep[1] = st.rep[0]
				st.rep[0] = off
			}
		}
		// literals, then the match
		if ll > len(lits) - li || ll + ml > len(dst) - o {return}
		// short copies go 16 bytes at a time while there is room past their
		// end (the literal buffer has 32 spare bytes, the output is checked)
		if ll > 0 {
			if ll <= 32 && o + 32 <= len(dst) {
				_copy16(&dst[o], &st.literals[li])
				if ll > 16 {_copy16(&dst[o + 16], &st.literals[li + 16])}
			} else {
				intrinsics.mem_copy_non_overlapping(&dst[o], &lits[li], ll)
			}
			o += ll
			li += ll
		}
		if off <= 0 || off > o {return}
		if off >= 16 && o + ml + 16 <= len(dst) {
			// each 16-byte step reads bytes written at least 16 bytes earlier
			for j := 0; j < ml; j += 16 {_copy16(&dst[o + j], &dst[o - off + j])}
		} else if off >= ml {
			intrinsics.mem_copy_non_overlapping(&dst[o], &dst[o - off], ml)
		} else {
			for j in 0 ..< ml {dst[o + j] = dst[o - off + j]}
		}
		o += ml
		if k < nseq - 1 {
			ll_s = int(lle.new_state) + int(_back_read(&br, int(lle.bits)))
			ml_s = int(mle.new_state) + int(_back_read(&br, int(mle.bits)))
			of_s = int(ofe.new_state) + int(_back_read(&br, int(ofe.bits)))
		}
		if br.pos < 0 {return}
	}
	if br.pos != 0 {return}
	rest := len(lits) - li
	if rest > len(dst) - o {return}
	if rest > 0 {intrinsics.mem_copy_non_overlapping(&dst[o], &lits[li], rest)}
	o += rest
	if o - d > 1 << 17 {return}
	return o - d, true
}

@(private)
_copy16 :: #force_inline proc "contextless" (dst, src: ^u8) {
	intrinsics.unaligned_store((^u128)(dst), intrinsics.unaligned_load((^u128)(src)))
}
