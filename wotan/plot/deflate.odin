package plot

import "core:slice"
import "core:sync"

// ============================================================================
// zlib (RFC 1950) / DEFLATE (RFC 1951) compressor: LZ77 with hash chains and
// one-step lazy matching, dynamic Huffman blocks with length-limited codes.
// Pure Odin (core:compress only provides inflate).
// ============================================================================

@(private = "file")
WINDOW :: 32768
@(private = "file")
HASH_BITS :: 15
@(private = "file")
MIN_MATCH :: 3
@(private = "file")
MAX_MATCH :: 258
@(private = "file")
MAX_CHAIN :: 96 // hash-chain candidates per position (zlib level ~6)
@(private = "file")
NICE_MATCH :: 128 // stop searching once a match this long is found
@(private = "file")
BLOCK_TOKENS :: 1 << 16

@(private = "file")
LEN_BASE := [29]u16{3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258}
@(private = "file")
LEN_EXTRA := [29]u8{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0}
@(private = "file")
DIST_BASE := [30]u16{1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577}
@(private = "file")
DIST_EXTRA := [30]u8{0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13}
// order in which code-length code lengths are transmitted
@(private = "file")
CL_ORDER := [19]u8{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15}

// symbol lookup tables, built once: length 3..258 -> code index, distance 1..32768 -> code
@(private = "file")
_len_code: [MAX_MATCH + 1]u8
@(private = "file")
_dist_code: [WINDOW + 1]u8
@(private = "file")
_tables_once: sync.Once

@(private = "file")
_build_tables :: proc() {
	c := 0
	for l in MIN_MATCH ..= MAX_MATCH {
		for c + 1 < len(LEN_BASE) && int(LEN_BASE[c + 1]) <= l {c += 1}
		_len_code[l] = u8(c)
	}
	c = 0
	for d in 1 ..= WINDOW {
		for c + 1 < len(DIST_BASE) && int(DIST_BASE[c + 1]) <= d {c += 1}
		_dist_code[d] = u8(c)
	}
}

// ---------------------------------------------------------------- bit output
@(private = "file")
Bit_Writer :: struct {
	out: [dynamic]u8,
	buf: u64,
	n:   u32,
}

@(private = "file")
_put :: #force_inline proc(bw: ^Bit_Writer, v: u32, nbits: u32) {
	bw.buf |= u64(v) << bw.n
	bw.n += nbits
	for bw.n >= 8 {
		append(&bw.out, u8(bw.buf))
		bw.buf >>= 8
		bw.n -= 8
	}
}

@(private = "file")
_flush_bits :: proc(bw: ^Bit_Writer) {
	if bw.n > 0 {append(&bw.out, u8(bw.buf))}
	bw.buf, bw.n = 0, 0
}

// ---------------------------------------------------------------- Huffman
// Code lengths for `freq` limited to `max_len` bits. Every used symbol gets a
// code; the code is always complete (callers ensure >= 2 used symbols).
// Optimal Huffman depths, then the miniz length-limiting fix-up.
@(private = "file")
_huffman_lengths :: proc(freq: []u32, lengths: []u8, max_len: int) {
	n := len(freq)
	slice.zero(lengths)
	syms := make([dynamic]int, 0, n, context.temp_allocator)
	for f, s in freq {if f > 0 {append(&syms, s)}}
	m := len(syms)
	if m == 0 {return}
	if m == 1 {lengths[syms[0]] = 1; return}

	// leaves sorted by frequency (ascending), then repeated two-smallest merges
	weight := make([]u64, 2 * m, context.temp_allocator)
	parent := make([]int, 2 * m, context.temp_allocator)
	order := make([]int, m, context.temp_allocator)
	for i in 0 ..< m {order[i] = i; weight[i] = u64(freq[syms[i]])}
	for i in 1 ..< m { 	// insertion sort by weight (m <= 288)
		j := i
		for j > 0 && weight[order[j - 1]] > weight[order[j]] {
			order[j - 1], order[j] = order[j], order[j - 1]
			j -= 1
		}
	}
	// two-queue merge: leaves in `order`, internal nodes are created in increasing weight
	li, ii, next := 0, m, m
	pick :: proc(li, ii: ^int, next, m: int, order: []int, weight: []u64) -> int {
		if li^ < m && (ii^ >= next || weight[order[li^]] <= weight[ii^]) {
			li^ += 1
			return order[li^ - 1]
		}
		ii^ += 1
		return ii^ - 1
	}
	for next < 2 * m - 1 {
		a := pick(&li, &ii, next, m, order, weight)
		b := pick(&li, &ii, next, m, order, weight)
		weight[next] = weight[a] + weight[b]
		parent[a], parent[b] = next, next
		next += 1
	}
	root := 2 * m - 2
	depth := make([]int, 2 * m - 1, context.temp_allocator)
	for k := root - 1; k >= 0; k -= 1 {depth[k] = depth[parent[k]] + 1}

	// count codes per length, clamp to max_len and restore a complete code (miniz)
	num := make([]int, 64, context.temp_allocator)
	for i in 0 ..< m {num[min(depth[i], 63)] += 1}
	for l in max_len + 1 ..< 64 {num[max_len] += num[l]; num[l] = 0}
	total := 0
	for l in 1 ..= max_len {total += num[l] << uint(max_len - l)}
	for total != 1 << uint(max_len) {
		num[max_len] -= 1
		for l := max_len - 1; l > 0; l -= 1 {
			if num[l] != 0 {
				num[l] -= 1
				num[l + 1] += 2
				break
			}
		}
		total -= 1
	}
	// most frequent symbols get the shortest codes
	l := 1
	for k := m - 1; k >= 0; k -= 1 {
		for num[l] == 0 {l += 1}
		lengths[syms[order[k]]] = u8(l)
		num[l] -= 1
	}
}

// Canonical codes (RFC 1951 3.2.2), bit-reversed for LSB-first output
@(private = "file")
_canonical_codes :: proc(lengths: []u8, codes: []u16) {
	bl_count: [16]u16
	for l in lengths {bl_count[l] += 1}
	bl_count[0] = 0
	next_code: [16]u16
	code: u16 = 0
	for bits in 1 ..< 16 {
		code = (code + bl_count[bits - 1]) << 1
		next_code[bits] = code
	}
	for l, s in lengths {
		if l == 0 {codes[s] = 0; continue}
		c := next_code[l]
		next_code[l] += 1
		r: u16 = 0
		for _ in 0 ..< l {r = (r << 1) | (c & 1); c >>= 1}
		codes[s] = r
	}
}

// A complete prefix code needs at least two used symbols: mark unused ones
@(private = "file")
_ensure_two :: proc(freq: []u32) {
	used := 0
	for f in freq {if f > 0 {used += 1}}
	for i := 0; used < 2 && i < len(freq); i += 1 {
		if freq[i] == 0 {freq[i] = 1; used += 1}
	}
}

// ---------------------------------------------------------------- blocks
// token: literal byte (< 256), or 1<<31 | (length-3)<<16 | distance
@(private = "file")
_emit_block :: proc(bw: ^Bit_Writer, tokens: []u32, final: bool) {
	lit_freq: [286]u32
	dist_freq: [30]u32
	for t in tokens {
		if t & (1 << 31) == 0 {
			lit_freq[t] += 1
		} else {
			length := int((t >> 16) & 0x1FF) + 3
			dist := int(t & 0xFFFF)
			lit_freq[257 + int(_len_code[length])] += 1
			dist_freq[_dist_code[dist]] += 1
		}
	}
	lit_freq[256] = 1 // end of block
	_ensure_two(lit_freq[:])
	_ensure_two(dist_freq[:])

	lit_len: [286]u8
	dist_len: [30]u8
	_huffman_lengths(lit_freq[:], lit_len[:], 15)
	_huffman_lengths(dist_freq[:], dist_len[:], 15)
	lit_code: [286]u16
	dist_code: [30]u16
	_canonical_codes(lit_len[:], lit_code[:])
	_canonical_codes(dist_len[:], dist_code[:])

	n_lit := 286
	for n_lit > 257 && lit_len[n_lit - 1] == 0 {n_lit -= 1}
	n_dist := 30
	for n_dist > 1 && dist_len[n_dist - 1] == 0 {n_dist -= 1}

	// run-length encode the concatenated code lengths with symbols 16/17/18
	all := make([]u8, n_lit + n_dist, context.temp_allocator)
	copy(all, lit_len[:n_lit])
	copy(all[n_lit:], dist_len[:n_dist])
	rle := make([dynamic][2]u8, 0, len(all), context.temp_allocator) // (symbol, extra value)
	for i := 0; i < len(all); {
		v := all[i]
		run := 1
		for i + run < len(all) && all[i + run] == v {run += 1}
		i += run
		if v == 0 {
			for run >= 11 {r := min(run, 138); append(&rle, [2]u8{18, u8(r - 11)}); run -= r}
			if run >= 3 {append(&rle, [2]u8{17, u8(run - 3)}); run = 0}
			for _ in 0 ..< run {append(&rle, [2]u8{0, 0})}
		} else {
			append(&rle, [2]u8{v, 0})
			run -= 1
			for run >= 3 {r := min(run, 6); append(&rle, [2]u8{16, u8(r - 3)}); run -= r}
			for _ in 0 ..< run {append(&rle, [2]u8{v, 0})}
		}
	}
	cl_freq: [19]u32
	for e in rle {cl_freq[e[0]] += 1}
	_ensure_two(cl_freq[:])
	cl_len: [19]u8
	cl_code: [19]u16
	_huffman_lengths(cl_freq[:], cl_len[:], 7)
	_canonical_codes(cl_len[:], cl_code[:])
	n_cl := 19
	for n_cl > 4 && cl_len[CL_ORDER[n_cl - 1]] == 0 {n_cl -= 1}

	// header
	_put(bw, 1 if final else 0, 1)
	_put(bw, 2, 2) // dynamic Huffman
	_put(bw, u32(n_lit - 257), 5)
	_put(bw, u32(n_dist - 1), 5)
	_put(bw, u32(n_cl - 4), 4)
	for i in 0 ..< n_cl {_put(bw, u32(cl_len[CL_ORDER[i]]), 3)}
	for e in rle {
		s := e[0]
		_put(bw, u32(cl_code[s]), u32(cl_len[s]))
		switch s {
		case 16:
			_put(bw, u32(e[1]), 2)
		case 17:
			_put(bw, u32(e[1]), 3)
		case 18:
			_put(bw, u32(e[1]), 7)
		}
	}

	// data
	for t in tokens {
		if t & (1 << 31) == 0 {
			_put(bw, u32(lit_code[t]), u32(lit_len[t]))
			continue
		}
		length := int((t >> 16) & 0x1FF) + 3
		dist := int(t & 0xFFFF)
		lc := int(_len_code[length])
		_put(bw, u32(lit_code[257 + lc]), u32(lit_len[257 + lc]))
		if LEN_EXTRA[lc] > 0 {_put(bw, u32(length - int(LEN_BASE[lc])), u32(LEN_EXTRA[lc]))}
		dc := int(_dist_code[dist])
		_put(bw, u32(dist_code[dc]), u32(dist_len[dc]))
		if DIST_EXTRA[dc] > 0 {_put(bw, u32(dist - int(DIST_BASE[dc])), u32(DIST_EXTRA[dc]))}
	}
	_put(bw, u32(lit_code[256]), u32(lit_len[256]))
}

// ---------------------------------------------------------------- LZ77
@(private = "file")
_hash3 :: #force_inline proc(d: []u8, i: int) -> int {
	return int((u32(d[i]) << 10 ~ u32(d[i + 1]) << 5 ~ u32(d[i + 2])) & (1 << HASH_BITS - 1))
}

@(private = "file")
_longest_match :: proc(d: []u8, i: int, head: []i32, prev: []i32, hash: int) -> (best_len, best_dist: int) {
	limit := min(MAX_MATCH, len(d) - i)
	if limit < MIN_MATCH {return 0, 0}
	cand := int(head[hash])
	chain := MAX_CHAIN
	for cand >= 0 && i - cand <= WINDOW && chain > 0 {
		chain -= 1
		// quick reject on the byte that would extend the current best
		if d[cand + best_len] == d[i + best_len] && d[cand] == d[i] {
			l := 0
			for l < limit && d[cand + l] == d[i + l] {l += 1}
			if l > best_len {
				best_len, best_dist = l, i - cand
				if l >= NICE_MATCH || l == limit {break}
			}
		}
		next := int(prev[cand & (WINDOW - 1)])
		if next >= cand {break} 	// stale entry from an older window
		cand = next
	}
	if best_len < MIN_MATCH {return 0, 0}
	return
}

// zlib stream of `data`; the result is allocated with `allocator`
zlib_compress :: proc(data: []u8, allocator := context.allocator) -> []u8 {
	sync.once_do(&_tables_once, _build_tables)
	scratch := context.temp_allocator
	head := make([]i32, 1 << HASH_BITS, scratch)
	prev := make([]i32, WINDOW, scratch)
	tokens := make([dynamic]u32, 0, BLOCK_TOKENS, scratch)
	defer {
		delete(head, scratch)
		delete(prev, scratch)
		delete(tokens)
	}
	slice.fill(head, -1)
	slice.fill(prev, -1)

	bw := Bit_Writer {
		out = make([dynamic]u8, 0, len(data) / 8 + 64, allocator),
	}
	append(&bw.out, 0x78, 0x9C) // zlib header: deflate, 32K window, default level

	n := len(data)
	insert :: #force_inline proc(d: []u8, i: int, head, prev: []i32) {
		if i + MIN_MATCH <= len(d) {
			h := _hash3(d, i)
			prev[i & (WINDOW - 1)] = head[h]
			head[h] = i32(i)
		}
	}

	i := 0
	for i < n {
		cur_len, cur_dist := 0, 0
		if i + MIN_MATCH <= n {
			cur_len, cur_dist = _longest_match(data, i, head, prev, _hash3(data, i))
		}
		insert(data, i, head, prev)
		// lazy evaluation: a longer match one byte later wins over this one
		if cur_len >= MIN_MATCH && cur_len < NICE_MATCH && i + 1 + MIN_MATCH <= n {
			next_len, _ := _longest_match(data, i + 1, head, prev, _hash3(data, i + 1))
			if next_len > cur_len {cur_len = 0}
		}
		if cur_len >= MIN_MATCH {
			append(&tokens, u32(1 << 31) | u32(cur_len - 3) << 16 | u32(cur_dist))
			for k in 1 ..< cur_len {insert(data, i + k, head, prev)}
			i += cur_len
		} else {
			append(&tokens, u32(data[i]))
			i += 1
		}
		if len(tokens) >= BLOCK_TOKENS {
			_emit_block(&bw, tokens[:], false)
			clear(&tokens)
		}
	}
	_emit_block(&bw, tokens[:], true)
	_flush_bits(&bw)

	adler := _adler32_checksum(data)
	append(&bw.out, u8(adler >> 24), u8(adler >> 16), u8(adler >> 8), u8(adler))
	return bw.out[:]
}

// ---------------------------------------------------------------- PNG filters
@(private = "file")
_paeth :: #force_inline proc(a, b, c: u8) -> u8 {
	p := int(a) + int(b) - int(c)
	pa, pb, pc := abs(p - int(a)), abs(p - int(b)), abs(p - int(c))
	if pa <= pb && pa <= pc {return a}
	if pb <= pc {return b}
	return c
}

// Filtered scanlines (filter byte + row) with the per-row filter minimising the
// sum of absolute signed residuals (the libpng heuristic)
png_filter_rows :: proc(px: []u8, width, height, bpp: int, allocator := context.allocator) -> []u8 {
	stride := width * bpp
	out := make([]u8, height * (stride + 1), allocator)
	cand := make([]u8, 5 * stride, context.temp_allocator)
	defer delete(cand, context.temp_allocator)
	zero_row := make([]u8, stride, context.temp_allocator)
	defer delete(zero_row, context.temp_allocator)
	for y in 0 ..< height {
		row := px[y * stride:(y + 1) * stride]
		up := zero_row if y == 0 else px[(y - 1) * stride:y * stride]
		best, best_cost := 0, max(int)
		for f in 0 ..< 5 {
			c := cand[f * stride:(f + 1) * stride]
			cost := 0
			for x in 0 ..< stride {
				a := row[x - bpp] if x >= bpp else 0
				b := up[x]
				cc := up[x - bpp] if x >= bpp else 0
				v: u8
				switch f {
				case 0:
					v = row[x]
				case 1:
					v = row[x] - a
				case 2:
					v = row[x] - b
				case 3:
					v = row[x] - u8((int(a) + int(b)) / 2)
				case 4:
					v = row[x] - _paeth(a, b, cc)
				}
				c[x] = v
				cost += abs(int(i8(v)))
			}
			if cost < best_cost {best, best_cost = f, cost}
		}
		o := out[y * (stride + 1):]
		o[0] = u8(best)
		copy(o[1:stride + 1], cand[best * stride:(best + 1) * stride])
	}
	return out
}
