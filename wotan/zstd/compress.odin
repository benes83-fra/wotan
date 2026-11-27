package zstd

import "base:intrinsics"
import "core:math"
import "core:mem"
import "core:slice"

// Zstandard compression (RFC 8878). The output is one frame that declares the
// content size and has no checksum and no dictionary, as libzstd writes it by
// default, so every Zstandard decoder reads it.
//
// The matcher looks for repeats through two hash tables, one on 8 bytes and
// one on 5, and tries the most recent offset first, like libzstd's fast
// levels. Each block of up to 128 KiB holds Huffman-coded literals and
// FSE-coded sequences, each table predefined, a single symbol or written into
// the block, whichever is smallest. A block that does not get smaller is
// stored as it is, and a block of one repeated byte takes four bytes.

// Appends `src`, compressed into one Zstandard frame, to `dst`. ok is false
// only for input of 2 GiB or more or when memory runs out.
compress_to_dynamic :: proc(dst: ^[dynamic]u8, src: []u8) -> bool {
	n := len(src)
	if n >= 1 << 31 {return false}
	single := n <= 1 << WINDOW_LOG

	// frame header: magic, descriptor, window (unless single segment), size
	hdr: [18]u8
	h := 0
	_put32(hdr[:], 0, 0xFD2FB528)
	h = 4
	fcs_flag: u8
	switch {
	case single && n < 256:
		fcs_flag = 0
	case n < 65536 + 256:
		fcs_flag = 1
	case:
		fcs_flag = 2
	}
	hdr[h] = fcs_flag << 6 | (0x20 if single else 0)
	h += 1
	if !single {
		hdr[h] = u8(WINDOW_LOG - 10) << 3
		h += 1
	}
	switch fcs_flag {
	case 0:
		hdr[h] = u8(n)
		h += 1
	case 1:
		v := n - 256
		hdr[h], hdr[h + 1] = u8(v), u8(v >> 8)
		h += 2
	case:
		_put32(hdr[:], h, u32(n))
		h += 4
	}
	if _, err := append(dst, ..hdr[:h]); err != nil {return false}

	if n == 0 {
		// one empty raw block, the last
		_, err := append(dst, 1, 0, 0)
		return err == nil
	}

	e, ok := _encoder_new(n)
	if !ok {return false}
	defer _encoder_free(e)
	e.src = src
	e.window = n if single else 1 << WINDOW_LOG
	for start := 0; start < n; start += BLOCK_MAX {
		end := min(n, start + BLOCK_MAX)
		_compress_block(e, dst, start, end, end == n) or_return
	}
	return true
}

// `src` compressed into one Zstandard frame (see compress_to_dynamic); free
// the result with delete
compress :: proc(src: []u8, allocator := context.allocator) -> (out: []u8, ok: bool) {
	buf := make([dynamic]u8, 0, 32 + len(src) / 2, allocator)
	if !compress_to_dynamic(&buf, src) {
		delete(buf)
		return nil, false
	}
	return buf[:], true
}

// ---------------------------------------------------------------------------

@(private)
BLOCK_MAX :: 1 << 17

// Content up to this size is one segment (window = the whole content);
// larger content uses a window of this size
@(private)
WINDOW_LOG :: 23

@(private)
Seq :: struct {
	ll, ml, ov: u32, // literal length, match length, offset value as coded
}

// Encoding table of an FSE code (libzstd's layout)
@(private)
Fse_CTable :: struct {
	state:      [512]u16,
	delta_bits: [64]u32,
	delta_find: [64]i32,
	log:        uint,
}

@(private)
Encoder :: struct {
	src:                    []u8,
	window:                 int,
	rep:                    [3]int,
	long_t, short_t:        []u32, // position + 1, 0 when empty
	long_log, short_log:    uint,
	lits:                   []u8,
	nlits:                  int,
	seqs:                   []Seq,
	nseq:                   int,
	ll_c, ml_c, of_c:       []u8,
	out:                    []u8,
	ll_def, of_def, ml_def: Fse_CTable,
	ll_t, of_t, ml_t:       Fse_CTable,
}

@(private)
_encoder_new :: proc(n: int) -> (e: ^Encoder, ok: bool) {
	err: mem.Allocator_Error
	e, err = new(Encoder)
	if err != nil {return nil, false}
	bmax := min(n, BLOCK_MAX)
	hb := uint(_highbit(u32(n)) + 1)
	e.long_log = clamp(hb, 8, 17)
	e.short_log = e.long_log - 1
	e.long_t, err = make([]u32, 1 << e.long_log)
	if err == nil {e.short_t, err = make([]u32, 1 << e.short_log)}
	if err == nil {e.lits, err = make([]u8, bmax + 32)}
	if err == nil {e.seqs, err = make([]Seq, bmax / 4 + 2)}
	if err == nil {e.ll_c, err = make([]u8, bmax / 4 + 2)}
	if err == nil {e.ml_c, err = make([]u8, bmax / 4 + 2)}
	if err == nil {e.of_c, err = make([]u8, bmax / 4 + 2)}
	if err == nil {e.out, err = make([]u8, bmax + 1024)}
	if err != nil {
		_encoder_free(e)
		return nil, false
	}
	e.rep = {1, 4, 8}
	_fse_ctable(&e.ll_def, LL_DEFAULT[:], 6)
	_fse_ctable(&e.of_def, OF_DEFAULT[:], 5)
	_fse_ctable(&e.ml_def, ML_DEFAULT[:], 6)
	return e, true
}

@(private)
_encoder_free :: proc(e: ^Encoder) {
	delete(e.long_t)
	delete(e.short_t)
	delete(e.lits)
	delete(e.seqs)
	delete(e.ll_c)
	delete(e.ml_c)
	delete(e.of_c)
	delete(e.out)
	free(e)
}

@(private)
_put32 :: #force_inline proc(b: []u8, i: int, v: u32) {
	b[i], b[i + 1], b[i + 2], b[i + 3] = u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24)
}

@(private)
_highbit :: #force_inline proc(v: u32) -> int {
	return int(31 - intrinsics.count_leading_zeros(v))
}

@(private)
_r32 :: #force_inline proc(b: []u8, i: int) -> u32 #no_bounds_check {
	return u32(intrinsics.unaligned_load((^u32le)(&b[i])))
}

@(private)
_r64 :: #force_inline proc(b: []u8, i: int) -> u64 #no_bounds_check {
	return u64(intrinsics.unaligned_load((^u64le)(&b[i])))
}

// One block: compressed, else as RLE or raw
@(private)
_compress_block :: proc(e: ^Encoder, dst: ^[dynamic]u8, start, end: int, last: bool) -> bool {
	src := e.src
	size := end - start
	lastbit := 1 if last else 0
	same := true
	for i in start + 1 ..< end {
		if src[i] != src[start] {
			same = false
			break
		}
	}
	if same && size > 1 {
		bh := lastbit | 1 << 1 | size << 3
		_, err := append(dst, u8(bh), u8(bh >> 8), u8(bh >> 16), src[start])
		return err == nil
	}
	saved := e.rep
	_find_sequences(e, start, end)
	n, ok := _encode_block(e)
	if ok && n < size {
		bh := lastbit | 2 << 1 | n << 3
		_, err := append(dst, u8(bh), u8(bh >> 8), u8(bh >> 16))
		if err == nil {_, err = append(dst, ..e.out[:n])}
		return err == nil
	}
	// stored: the decoder's recent offsets stay as they were
	e.rep = saved
	bh := lastbit | size << 3
	_, err := append(dst, u8(bh), u8(bh >> 8), u8(bh >> 16))
	if err == nil {_, err = append(dst, ..src[start:end])}
	return err == nil
}

// ---------------------------------------------------------------------------
// Match finding

@(private)
PRIME8 :: 0xCF1BBCDCB7A56463
@(private)
PRIME5 :: 0xCF1BBCDCBB

@(private)
_hash8 :: #force_inline proc(v: u64, log: uint) -> u32 {
	return u32((v * PRIME8) >> (64 - log))
}

@(private)
_hash5 :: #force_inline proc(v: u64, log: uint) -> u32 {
	return u32(((v << 24) * PRIME5) >> (64 - log))
}

// Equal bytes of src[a:limit] and src[b:] (b < a)
@(private)
_match_len :: #force_inline proc(src: []u8, a, b, limit: int) -> int #no_bounds_check {
	n := 0
	for a + n + 8 <= limit {
		x := _r64(src, a + n) ~ _r64(src, b + n)
		if x != 0 {return n + int(intrinsics.count_trailing_zeros(x)) >> 3}
		n += 8
	}
	for a + n < limit && src[a + n] == src[b + n] {n += 1}
	return n
}

// Records a sequence: the literals src[lit:lit + ll], then `ml` bytes from
// `off` back; codes the offset against the recent ones as the decoder will
@(private)
_emit :: #force_inline proc(e: ^Encoder, lit, ll, off, ml: int) #no_bounds_check {
	if ll > 0 {
		if ll <= 16 && lit + 16 <= len(e.src) {
			_copy16(&e.lits[e.nlits], &e.src[lit])
		} else {
			intrinsics.mem_copy_non_overlapping(&e.lits[e.nlits], &e.src[lit], ll)
		}
		e.nlits += ll
	}
	r := &e.rep
	ov: int
	if ll > 0 {
		switch off {
		case r[0]:
			ov = 1
		case r[1]:
			ov = 2
		case r[2]:
			ov = 3
		case:
			ov = off + 3
		}
	} else {
		switch off {
		case r[1]:
			ov = 1
		case r[2]:
			ov = 2
		case r[0] - 1:
			ov = 3
		case:
			ov = off + 3
		}
	}
	// the decoder's update of the recent offsets
	if ov > 3 {
		r[2] = r[1]
		r[1] = r[0]
		r[0] = off
	} else {
		idx := ov - 1 + (1 if ll == 0 else 0)
		if idx != 0 {
			if idx >= 2 {r[2] = r[1]}
			r[1] = r[0]
			r[0] = off
		}
	}
	e.seqs[e.nseq] = {u32(ll), u32(ml), u32(ov)}
	e.nseq += 1
}

// The sequences and literals of src[start:end]; matches may reach back into
// earlier blocks, within the window
@(private)
_find_sequences :: proc(e: ^Encoder, start, end: int) #no_bounds_check {
	src := e.src
	e.nlits = 0
	e.nseq = 0
	ll_log, sl_log := e.long_log, e.short_log
	lt, st := e.long_t, e.short_t
	ip := start
	anchor := start
	// hashing reads 8 bytes at ip and ip + 1
	ilimit := end - 9
	put_limit := len(src) - 8
	for ip < ilimit {
		lowest := max(0, ip - e.window)
		// the most recent offset, one byte ahead
		r0 := e.rep[0]
		if r0 <= e.window && ip + 1 >= r0 && _r32(src, ip + 1) == _r32(src, ip + 1 - r0) {
			ml := 4 + _match_len(src, ip + 5, ip + 5 - r0, end)
			v := _r64(src, ip)
			lt[_hash8(v, ll_log)] = u32(ip + 1)
			st[_hash5(v, sl_log)] = u32(ip + 1)
			_emit(e, anchor, ip + 1 - anchor, r0, ml)
			ip += 1 + ml
			anchor = ip
		} else {
			v := _r64(src, ip)
			hl := _hash8(v, ll_log)
			hs := _hash5(v, sl_log)
			cl := int(lt[hl]) - 1
			cs := int(st[hs]) - 1
			lt[hl] = u32(ip + 1)
			st[hs] = u32(ip + 1)
			m := -1
			ml := 0
			if cl >= lowest && _r64(src, cl) == v {
				m = cl
				ml = 8 + _match_len(src, ip + 8, cl + 8, end)
			} else if cs >= lowest && _r32(src, cs) == u32(v) {
				// a short match; a long one one byte ahead is better
				v1 := _r64(src, ip + 1)
				h1 := _hash8(v1, ll_log)
				c1 := int(lt[h1]) - 1
				lt[h1] = u32(ip + 2)
				if c1 >= max(0, ip + 1 - e.window) && _r64(src, c1) == v1 {
					ip += 1
					m = c1
					ml = 8 + _match_len(src, ip + 8, c1 + 8, end)
				} else {
					m = cs
					ml = 4 + _match_len(src, ip + 4, cs + 4, end)
				}
			}
			if m < 0 {
				ip += 1 + (ip - anchor) >> 8
				continue
			}
			// extend backwards over the literals
			for ip > anchor && m > 0 && src[ip - 1] == src[m - 1] {
				ip -= 1
				m -= 1
				ml += 1
			}
			_emit(e, anchor, ip - anchor, ip - m, ml)
			mstart := ip
			ip += ml
			anchor = ip
			// fill the tables from inside the match
			for p in ([3]int{mstart + 2, ip - 2, ip - 1}) {
				if p < put_limit && p > mstart {
					pv := _r64(src, p)
					lt[_hash8(pv, ll_log)] = u32(p + 1)
					st[_hash5(pv, sl_log)] = u32(p + 1)
				}
			}
		}
		// the second most recent offset, straight after the match
		for ip < ilimit {
			r1 := e.rep[1]
			if r1 > e.window || r1 > ip || _r32(src, ip) != _r32(src, ip - r1) {break}
			ml := 4 + _match_len(src, ip + 4, ip + 4 - r1, end)
			v := _r64(src, ip)
			lt[_hash8(v, ll_log)] = u32(ip + 1)
			st[_hash5(v, sl_log)] = u32(ip + 1)
			_emit(e, ip, 0, r1, ml)
			ip += ml
			anchor = ip
		}
	}
	// the rest are literals
	rest := end - anchor
	if rest > 0 {
		copy(e.lits[e.nlits:], src[anchor:end])
		e.nlits += rest
	}
}

// ---------------------------------------------------------------------------
// Bit writer: bits go in from the least significant end, as both the forward
// (table descriptions) and the backward readers (streams) of the decoder take
// them

@(private)
Bit_Out :: struct {
	buf:  []u8,
	pos:  int,
	acc:  u64,
	n:    uint,
	over: bool,
}

@(private)
_bo_add :: #force_inline proc(b: ^Bit_Out, v: u64, bits: uint) {
	b.acc |= (v & (u64(1) << bits - 1)) << b.n
	b.n += bits
}

// Stores the whole bytes; at most 56 bits may be added between two flushes
@(private)
_bo_flush :: #force_inline proc(b: ^Bit_Out) #no_bounds_check {
	if b.pos + 8 > len(b.buf) {
		b.over = true
		b.pos = 0
		b.acc = 0
		b.n = 0
		return
	}
	intrinsics.unaligned_store((^u64le)(&b.buf[b.pos]), u64le(b.acc))
	k := b.n >> 3
	b.pos += int(k)
	b.acc >>= k * 8
	b.n &= 7
}

// Bytes written; a backward stream ends with a 1 bit marking its start
@(private)
_bo_close :: proc(b: ^Bit_Out, marker: bool) -> (int, bool) {
	if marker {_bo_add(b, 1, 1)}
	_bo_flush(b)
	if b.over {return 0, false}
	return b.pos + (1 if b.n > 0 else 0), true
}

// ---------------------------------------------------------------------------
// FSE

// Places the symbols in the table exactly as the decoder does (_fse_build)
@(private)
_fse_ctable :: proc(t: ^Fse_CTable, norm: []i16, log: uint) {
	size := 1 << log
	symbols: [512]u8
	high := size - 1
	for c, s in norm {
		if c == -1 {
			symbols[high] = u8(s)
			high -= 1
		}
	}
	step := (size >> 1) + (size >> 3) + 3
	mask := size - 1
	pos := 0
	for c, s in norm {
		for _ in 0 ..< int(c) {
			symbols[pos] = u8(s)
			pos = (pos + step) & mask
			for pos > high {pos = (pos + step) & mask}
		}
	}
	cumul: [65]int
	for c, s in norm {
		cumul[s + 1] = cumul[s] + (1 if c == -1 else max(int(c), 0))
	}
	for u in 0 ..< size {
		s := symbols[u]
		t.state[cumul[s]] = u16(size + u)
		cumul[s] += 1
	}
	total := 0
	for c, s in norm {
		switch c {
		case 0:
			t.delta_bits[s] = u32((log + 1) << 16 - (1 << log))
		case -1, 1:
			t.delta_bits[s] = u32(log << 16 - (1 << log))
			t.delta_find[s] = i32(total - 1)
			total += 1
		case:
			bits_out := log - uint(_highbit(u32(c) - 1))
			t.delta_bits[s] = u32(bits_out << 16) - u32(int(c) << bits_out)
			t.delta_find[s] = i32(total - int(c))
			total += int(c)
		}
	}
	t.log = log
}

// The first state: the one the decoder ends on for symbol `s`
@(private)
_fse_init :: #force_inline proc(t: ^Fse_CTable, s: u8) -> u32 {
	nb := (t.delta_bits[s] + 1 << 15) >> 16
	value := nb << 16 - t.delta_bits[s]
	return u32(t.state[i32(value >> nb) + t.delta_find[s]])
}

@(private)
_fse_encode :: #force_inline proc(
	b: ^Bit_Out,
	t: ^Fse_CTable,
	state: ^u32,
	s: u8,
) #no_bounds_check {
	nb := (state^ + t.delta_bits[s]) >> 16
	_bo_add(b, u64(state^), uint(nb))
	state^ = u32(t.state[i32(state^ >> nb) + t.delta_find[s]])
}

// libzstd's choice of table size for `n` symbols up to `max_sym`
@(private)
_table_log :: proc(max_log: int, n: int, max_sym: int) -> uint {
	src_bits := _highbit(u32(n - 1)) - 2
	min_bits := min(_highbit(u32(n)) + 1, _highbit(u32(max(max_sym, 1))) + 2)
	log := max_log
	if src_bits < log {log = src_bits}
	if min_bits > log {log = min_bits}
	return uint(clamp(log, 5, max_log))
}

// Counts scaled to a total of 1 << log, every present symbol at least 1; the
// rounding is settled one step at a time where it costs the fewest bits
@(private)
_normalize :: proc(norm: []i16, count: []u32, total: int, log: uint) {
	size := 1 << log
	sum := 0
	for c, s in count {
		if c == 0 {
			norm[s] = 0
			continue
		}
		v := max(1, (int(c) * size + total / 2) / total)
		norm[s] = i16(v)
		sum += v
	}
	for sum != size {
		best := -1
		best_cost := max(f64)
		for c, s in count {
			if c == 0 {continue}
			v := f64(norm[s])
			cost: f64
			if sum > size {
				if norm[s] <= 1 {continue}
				cost = f64(c) * math.log2(v / (v - 1))
			} else {
				cost = -f64(c) * math.log2((v + 1) / v)
			}
			if cost < best_cost {
				best_cost = cost
				best = s
			}
		}
		if sum > size {
			norm[best] -= 1
			sum -= 1
		} else {
			norm[best] += 1
			sum += 1
		}
	}
}

// Bits to code the symbols counted in `count` with the table `norm`
@(private)
_cost :: proc(count: []u32, norm: []i16, log: uint) -> f64 {
	bits := 0.0
	for c, s in count {
		if c == 0 {continue}
		n := 1.0 if norm[s] == -1 else f64(norm[s])
		bits += f64(c) * (f64(log) - math.log2(n))
	}
	return bits
}

// A table description, as the decoder's _fse_read takes it; norm ends with
// its last present symbol
@(private)
_write_ncount :: proc(out: []u8, norm: []i16, log: uint) -> (int, bool) {
	b := Bit_Out {
		buf = out,
	}
	_bo_add(&b, u64(log - 5), 4)
	size := 1 << log
	remaining := size + 1
	threshold := size
	nbits := log + 1
	s := 0
	for remaining > 1 && s < len(norm) {
		c := int(norm[s])
		value := c + 1
		max_small := 2 * threshold - 1 - remaining
		if value < max_small {
			_bo_add(&b, u64(value), nbits - 1)
		} else if value < threshold {
			_bo_add(&b, u64(value), nbits)
		} else {
			_bo_add(&b, u64(value + max_small), nbits)
		}
		remaining -= abs(c)
		s += 1
		if c == 0 {
			z := 0
			for s + z < len(norm) && norm[s + z] == 0 {z += 1}
			s += z
			for z >= 3 {
				_bo_add(&b, 3, 2)
				z -= 3
				_bo_flush(&b)
			}
			_bo_add(&b, u64(z), 2)
		}
		for remaining < threshold {
			nbits -= 1
			threshold >>= 1
		}
		_bo_flush(&b)
	}
	if remaining != 1 {return 0, false}
	return _bo_close(&b, false)
}

// ---------------------------------------------------------------------------
// Literals

@(private)
HUF_LIMIT :: 11

// Huffman code lengths of at most HUF_LIMIT bits for the counted bytes (at
// least two present); returns the longest
@(private)
_huf_lengths :: proc(count: ^[256]u32, lens: ^[256]u8) -> int {
	keys: [256]u64
	m := 0
	for c, s in count {
		if c > 0 {
			keys[m] = u64(c) << 8 | u64(s)
			m += 1
		}
	}
	slice.sort(keys[:m])
	// two queues: the sorted leaves, and the inner nodes in order of creation
	weight: [512]u64
	parent: [512]i32
	for i in 0 ..< m {weight[i] = keys[i] >> 8}
	li, ni, nn := 0, m, m
	for nn < 2 * m - 1 {
		pick: [2]int
		for k in 0 ..< 2 {
			if li < m && (ni >= nn || weight[li] <= weight[ni]) {
				pick[k] = li
				li += 1
			} else {
				pick[k] = ni
				ni += 1
			}
		}
		weight[nn] = weight[pick[0]] + weight[pick[1]]
		parent[pick[0]] = i32(nn)
		parent[pick[1]] = i32(nn)
		nn += 1
	}
	depth: [512]int
	num: [512]int
	maxd := 0
	for i := 2 * m - 3; i >= 0; i -= 1 {
		depth[i] = depth[parent[i]] + 1
		if i < m {
			num[depth[i]] += 1
			maxd = max(maxd, depth[i])
		}
	}
	// limit the lengths, keeping the code complete
	L :: HUF_LIMIT
	for i in L + 1 ..= maxd {
		num[L] += num[i]
		num[i] = 0
	}
	total := 0
	for i in 1 ..= L {total += num[i] << uint(L - i)}
	for total != 1 << L {
		num[L] -= 1
		for i := L - 1; i > 0; i -= 1 {
			if num[i] != 0 {
				num[i] -= 1
				num[i + 1] += 2
				break
			}
		}
		total -= 1
	}
	// shortest codes to the most frequent bytes
	lens^ = {}
	k := m - 1
	longest := 0
	for l in 1 ..= L {
		for _ in 0 ..< num[l] {
			lens[keys[k] & 0xFF] = u8(l)
			k -= 1
			longest = l
		}
	}
	return longest
}

// The tree description of the weights (symbols 0 ..< len(weights); the last
// symbol's weight is implied): 4-bit weights, or FSE-coded when shorter or
// when there are more than 128
@(private)
_huf_describe :: proc(out: []u8, weights: []u8) -> (int, bool) {
	n := len(weights)
	direct := 1 + (n + 1) / 2 if n <= 128 else max(int)
	// FSE-coded weights, as libzstd writes them
	fse := max(int)
	tmp: [160]u8
	fse_loop: if n > 2 {
		count: [16]u32
		maxw := 0
		for w in weights {
			count[w] += 1
			maxw = max(maxw, int(w))
		}
		maxc: u32 = 0
		for c in count {maxc = max(maxc, c)}
		if maxc == u32(n) || maxc == 1 {break fse_loop}
		log := _table_log(6, n, maxw)
		norm: [16]i16
		_normalize(norm[:maxw + 1], count[:maxw + 1], n, log)
		hl, hok := _write_ncount(tmp[1:], norm[:maxw + 1], log)
		if !hok {break fse_loop}
		t: Fse_CTable
		_fse_ctable(&t, norm[:maxw + 1], log)
		b := Bit_Out {
			buf = tmp[1 + hl:],
		}
		s1, s2: u32
		ip := n
		if n & 1 != 0 {
			s1 = _fse_init(&t, weights[n - 1])
			s2 = _fse_init(&t, weights[n - 2])
			_fse_encode(&b, &t, &s1, weights[n - 3])
			ip = n - 3
		} else {
			s2 = _fse_init(&t, weights[n - 1])
			s1 = _fse_init(&t, weights[n - 2])
			ip = n - 2
		}
		_bo_flush(&b)
		for ip > 0 {
			_fse_encode(&b, &t, &s2, weights[ip - 1])
			_fse_encode(&b, &t, &s1, weights[ip - 2])
			ip -= 2
			_bo_flush(&b)
		}
		_bo_add(&b, u64(s2), t.log)
		_bo_add(&b, u64(s1), t.log)
		sl, sok := _bo_close(&b, true)
		if !sok || hl + sl >= 128 {break fse_loop}
		tmp[0] = u8(hl + sl)
		fse = 1 + hl + sl
	}
	if direct == max(int) && fse == max(int) {return 0, false}
	if direct <= fse {
		if direct > len(out) {return 0, false}
		out[0] = u8(127 + n)
		for i := 0; i < n; i += 2 {
			lo := weights[i + 1] if i + 1 < n else 0
			out[1 + i / 2] = weights[i] << 4 | lo
		}
		return direct, true
	}
	if fse > len(out) {return 0, false}
	copy(out, tmp[:fse])
	return fse, true
}

// One Huffman stream: the bytes from last to first, so that the decoder,
// reading backwards, gets them in order
@(private)
_huf_stream_out :: proc(
	out: []u8,
	src: []u8,
	code: ^[256]u16,
	lens: ^[256]u8,
) -> (
	int,
	bool,
) #no_bounds_check {
	b := Bit_Out {
		buf = out,
	}
	i := len(src)
	for i & 3 != 0 {
		i -= 1
		s := src[i]
		_bo_add(&b, u64(code[s]), uint(lens[s]))
	}
	_bo_flush(&b)
	for i > 0 {
		#unroll for k in 1 ..= 4 {
			s := src[i - k]
			_bo_add(&b, u64(code[s]), uint(lens[s]))
		}
		i -= 4
		_bo_flush(&b)
		if b.over {return 0, false}
	}
	return _bo_close(&b, true)
}

// Header of raw (kind 0) or RLE (kind 1) literals
@(private)
_lit_header :: proc(out: []u8, kind, size: int) -> int {
	switch {
	case size < 32:
		out[0] = u8(kind | size << 3)
		return 1
	case size < 4096:
		v := kind | 1 << 2 | size << 4
		out[0], out[1] = u8(v), u8(v >> 8)
		return 2
	}
	v := kind | 3 << 2 | size << 4
	out[0], out[1], out[2] = u8(v), u8(v >> 8), u8(v >> 16)
	return 3
}

// The literals section into out; returns its bytes (0 if it does not fit)
@(private)
_encode_literals :: proc(out: []u8, lits: []u8) -> int {
	n := len(lits)
	raw :: proc(out: []u8, lits: []u8) -> int {
		if len(out) < len(lits) + 3 {return 0}
		h := _lit_header(out, 0, len(lits))
		copy(out[h:], lits)
		return h + len(lits)
	}
	if n < 32 {return raw(out, lits)}
	count: [256]u32
	for c in lits {count[c] += 1}
	present := 0
	last := 0
	for c, s in count {
		if c > 0 {
			present += 1
			last = s
		}
	}
	if present == 1 {
		h := _lit_header(out, 1, n)
		out[h] = lits[0]
		return h + 1
	}
	lens: [256]u8
	mb := _huf_lengths(&count, &lens)
	bits := 0
	for c, s in count {bits += int(c) * int(lens[s])}
	// worth it only with a real gain (libzstd's rule)
	if bits / 8 + 6 + 64 >= n - (n >> 6) - 2 {return raw(out, lits)}
	weights: [256]u8
	for s in 0 ..< last {
		if lens[s] > 0 {weights[s] = u8(mb + 1 - int(lens[s]))}
	}
	// canonical codes in the decoder's order: by weight, then by symbol
	per_weight: [HUF_LIMIT + 2]int
	for s in 0 ..= last {
		if lens[s] > 0 {per_weight[mb + 1 - int(lens[s])] += 1}
	}
	next: [HUF_LIMIT + 2]int
	pos := 0
	for w in 1 ..= mb {
		next[w] = pos
		pos += per_weight[w] << uint(w - 1)
	}
	code: [256]u16
	for s in 0 ..= last {
		if lens[s] == 0 {continue}
		w := mb + 1 - int(lens[s])
		code[s] = u16(next[w] >> uint(w - 1))
		next[w] += 1 << uint(w - 1)
	}

	streams := 1 if n < 256 else 4
	fmt, hdr, field: int
	switch {
	case streams == 1:
		fmt, hdr, field = 0, 3, 10
	case n < 1024:
		fmt, hdr, field = 1, 3, 10
	case n < 16384:
		fmt, hdr, field = 2, 4, 14
	case:
		fmt, hdr, field = 3, 5, 18
	}
	if len(out) < hdr + 8 {return raw(out, lits)}
	p := hdr
	tl, tok := _huf_describe(out[p:], weights[:last])
	if !tok {return raw(out, lits)}
	p += tl
	if streams == 1 {
		sl, sok := _huf_stream_out(out[p:], lits, &code, &lens)
		if !sok {return raw(out, lits)}
		p += sl
	} else {
		if len(out) - p < 6 {return raw(out, lits)}
		jump := p
		p += 6
		per := (n + 3) / 4
		for k in 0 ..< 4 {
			part := lits[k * per:min(n, (k + 1) * per)]
			sl, sok := _huf_stream_out(out[p:], part, &code, &lens)
			if !sok {return raw(out, lits)}
			if k < 3 {
				out[jump + 2 * k] = u8(sl)
				out[jump + 2 * k + 1] = u8(sl >> 8)
			}
			p += sl
		}
	}
	csize := p - hdr
	if csize >= n - (n >> 6) - 2 {return raw(out, lits)}
	hv := u64(2) | u64(fmt) << 2 | u64(n) << 4 | u64(csize) << uint(4 + field)
	for k in 0 ..< hdr {out[k] = u8(hv >> uint(8 * k))}
	return p
}

// ---------------------------------------------------------------------------
// Sequences

@(private)
_ll_code :: #force_inline proc(ll: u32) -> u8 {
	if ll < 16 {return u8(ll)}
	if ll >= 64 {return u8(_highbit(ll) + 19)}
	code := 16
	for code < 24 && LL_BASE[code + 1] <= ll {code += 1}
	return u8(code)
}

@(private)
_ml_code :: #force_inline proc(ml: u32) -> u8 {
	b := ml - 3
	if b < 32 {return u8(b)}
	if b >= 128 {return u8(_highbit(b) + 36)}
	code := 32
	for code < 42 && ML_BASE[code + 1] <= ml {code += 1}
	return u8(code)
}

// Picks the table mode of one of the three codes and writes its description;
// returns the mode and the table to code with (nil for a single symbol)
@(private)
_choose_table :: proc(
	out: []u8,
	codes: []u8,
	max_log: int,
	default: []i16,
	default_log: uint,
	def: ^Fse_CTable,
	t: ^Fse_CTable,
) -> (
	mode: int,
	used: int,
	table: ^Fse_CTable,
	ok: bool,
) {
	count: [64]u32
	maxs := 0
	for c in codes {
		count[c] += 1
		maxs = max(maxs, int(c))
	}
	present := 0
	for c in count {if c > 0 {present += 1}}
	if present == 1 {
		if len(out) < 1 {return}
		out[0] = codes[0]
		return 1, 1, nil, true
	}
	pre_ok := maxs < len(default)
	pre_cost := max(f64)
	if pre_ok {pre_cost = _cost(count[:maxs + 1], default[:maxs + 1], default_log)}
	log := _table_log(max_log, len(codes), maxs)
	norm: [64]i16
	_normalize(norm[:maxs + 1], count[:maxs + 1], len(codes), log)
	tmp: [256]u8
	hl, hok := _write_ncount(tmp[:], norm[:maxs + 1], log)
	if !hok {return}
	fse_cost := f64(hl * 8) + _cost(count[:maxs + 1], norm[:maxs + 1], log)
	if pre_ok && pre_cost <= fse_cost {
		return 0, 0, def, true
	}
	if len(out) < hl {return}
	copy(out, tmp[:hl])
	_fse_ctable(t, norm[:maxs + 1], log)
	return 2, hl, t, true
}

// The block's literals and sequences into e.out; ok is false if they do not
// fit (the block is then stored)
@(private)
_encode_block :: proc(e: ^Encoder) -> (size: int, ok: bool) #no_bounds_check {
	out := e.out
	p := _encode_literals(out, e.lits[:e.nlits])
	if p == 0 {return 0, false}
	nseq := e.nseq
	if len(out) - p < 4 {return 0, false}
	switch {
	case nseq < 128:
		out[p] = u8(nseq)
		p += 1
	case nseq < 0x7F00:
		out[p], out[p + 1] = u8(nseq >> 8 + 128), u8(nseq)
		p += 2
	case:
		v := nseq - 0x7F00
		out[p], out[p + 1], out[p + 2] = 255, u8(v), u8(v >> 8)
		p += 3
	}
	if nseq == 0 {return p, true}

	for i in 0 ..< nseq {
		s := e.seqs[i]
		e.ll_c[i] = _ll_code(s.ll)
		e.ml_c[i] = _ml_code(s.ml)
		e.of_c[i] = u8(_highbit(s.ov))
	}
	modes := p
	p += 1
	llm, l1, llt := _choose_table(
		out[p:],
		e.ll_c[:nseq],
		9,
		LL_DEFAULT[:],
		6,
		&e.ll_def,
		&e.ll_t,
	) or_return
	p += l1
	ofm, l2, oft := _choose_table(
		out[p:],
		e.of_c[:nseq],
		8,
		OF_DEFAULT[:],
		5,
		&e.of_def,
		&e.of_t,
	) or_return
	p += l2
	mlm, l3, mlt := _choose_table(
		out[p:],
		e.ml_c[:nseq],
		9,
		ML_DEFAULT[:],
		6,
		&e.ml_def,
		&e.ml_t,
	) or_return
	p += l3
	out[modes] = u8(llm << 6 | ofm << 4 | mlm << 2)

	// the sequences from last to first
	b := Bit_Out {
		buf = out[p:],
	}
	last := nseq - 1
	lls, ofs, mls: u32
	if mlt != nil {mls = _fse_init(mlt, e.ml_c[last])}
	if oft != nil {ofs = _fse_init(oft, e.of_c[last])}
	if llt != nil {lls = _fse_init(llt, e.ll_c[last])}
	extras :: #force_inline proc(b: ^Bit_Out, s: Seq, llc, mlc, ofc: u8) {
		_bo_add(b, u64(s.ll - LL_BASE[llc]), uint(LL_BITS[llc]))
		_bo_add(b, u64(s.ml - ML_BASE[mlc]), uint(ML_BITS[mlc]))
		_bo_flush(b)
		_bo_add(b, u64(s.ov), uint(ofc))
		_bo_flush(b)
	}
	extras(&b, e.seqs[last], e.ll_c[last], e.ml_c[last], e.of_c[last])
	for i := last - 1; i >= 0; i -= 1 {
		if oft != nil {_fse_encode(&b, oft, &ofs, e.of_c[i])}
		if mlt != nil {_fse_encode(&b, mlt, &mls, e.ml_c[i])}
		if llt != nil {_fse_encode(&b, llt, &lls, e.ll_c[i])}
		_bo_flush(&b)
		extras(&b, e.seqs[i], e.ll_c[i], e.ml_c[i], e.of_c[i])
		if b.over {return 0, false}
	}
	if mlt != nil {_bo_add(&b, u64(mls), mlt.log)}
	if oft != nil {_bo_add(&b, u64(ofs), oft.log)}
	if llt != nil {_bo_add(&b, u64(lls), llt.log)}
	n := _bo_close(&b, true) or_return
	return p + n, true
}
