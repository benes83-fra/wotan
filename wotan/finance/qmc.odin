package finance

import "base:intrinsics"
import "core:math"
import "core:mem"

// ============================================================================
// QUASI-MONTE CARLO: SOBOL' SEQUENCES AND THE BROWNIAN BRIDGE
// ============================================================================

// Sobol' sequence (Joe-Kuo direction numbers, 32 bits, Gray-code order as
// scipy.stats.qmc.Sobol(scramble=False, bits=32)); point n is the XOR of the
// direction integers selected by the Gray code of n, scaled by 2^-32. An
// optional random digital shift (XOR with a random integer per dimension)
// randomizes the sequence for error estimates across independent shifts.
SobolSequence :: struct {
	dims:  int,
	v:     []u32, // direction integers, v[d·32 + j]
	x:     []u32, // current point
	shift: []u32, // digital shift (all zero without one)
	index: u64, // index of the next point
}

// Sequence of `dims` dimensions (at most 2048) starting at point `skip`;
// shift_seed ≠ 0 applies a digital shift drawn from that seed
make_sobol :: proc(
	dims: int,
	skip: u64 = 0,
	shift_seed: u64 = 0,
	allocator := context.allocator,
) -> SobolSequence {
	assert(dims >= 1 && dims <= _SOBOL_MAX_DIM, "make_sobol: 1 to 2048 dimensions")
	s := SobolSequence {
		dims  = dims,
		v     = make([]u32, dims * 32, allocator),
		x     = make([]u32, dims, allocator),
		shift = make([]u32, dims, allocator),
	}
	off := 0
	for d in 0 ..< dims {
		v := s.v[d * 32:(d + 1) * 32]
		p := _SOBOL_POLY[d]
		m := 31 - int(intrinsics.count_leading_zeros(p)) // degree
		if d == 0 {
			for &x in v {x = 1}
		} else {
			for j in 0 ..< m {v[j] = _SOBOL_MINIT[off + j]}
			off += m
			// v_j = 2a_1 v_{j-1} ⊕ … ⊕ 2^{m-1}a_{m-1} v_{j-m+1} ⊕ 2^m v_{j-m} ⊕ v_{j-m}
			for j in m ..< 32 {
				nv := v[j - m]
				for k in 0 ..< m {
					if (p >> u32(m - 1 - k)) & 1 != 0 {nv ~= v[j - k - 1] << u32(k + 1)}
				}
				v[j] = nv
			}
		}
		for j in 0 ..< 32 {v[j] <<= u32(31 - j)}
	}
	if shift_seed != 0 {
		state := shift_seed
		for &sh in s.shift {sh = u32(_splitmix64(&state) >> 32)}
	}
	sobol_skip_to(&s, skip)
	return s
}

free_sobol :: proc(s: ^SobolSequence, allocator := context.allocator) {
	delete(s.v, allocator)
	delete(s.x, allocator)
	delete(s.shift, allocator)
	s^ = {}
}

// Positions the sequence so that the next point is point n
sobol_skip_to :: proc(s: ^SobolSequence, n: u64) {
	g := n ~ (n >> 1)
	for d in 0 ..< s.dims {
		x: u32
		for j in 0 ..< 32 {
			if (g >> u64(j)) & 1 != 0 {x ~= s.v[d * 32 + j]}
		}
		s.x[d] = x
	}
	s.index = n
}

// Next point in [0, 1)^dims
sobol_next :: proc(s: ^SobolSequence, out: []f64) {
	l := int(intrinsics.count_trailing_zeros(~s.index))
	for d in 0 ..< s.dims {
		out[d] = f64(s.x[d] ~ s.shift[d]) * _TWO_M32
		s.x[d] ~= s.v[d * 32 + l]
	}
	s.index += 1
}

@(private = "file")
_TWO_M32 :: 1.0 / 4294967296.0 // 2^-32

@(private = "file")
_splitmix64 :: proc(state: ^u64) -> u64 {
	state^ += 0x9E3779B97F4A7C15
	z := state^
	z = (z ~ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ~ (z >> 27)) * 0x94D049BB133111EB
	return z ~ (z >> 31)
}

// ----------------------------------------------------------------------------
// Brownian bridge (QuantLib's BrownianBridge): builds a Brownian path from
// standard normals in the order terminal value first, then successive
// midpoints, so the first (best-distributed) quasi-random dimensions carry the
// largest-scale features of the path. The output is the standardized
// increments (W(t_i) − W(t_{i−1}))/√(t_i − t_{i−1}), again independent
// N(0, 1), so it can feed any generator expecting i.i.d. normals.
// ----------------------------------------------------------------------------

BrownianBridge :: struct {
	size:                      int,
	bridge_index:              []int,
	left_index, right_index:   []int,
	left_weight, right_weight: []f64,
	std_dev, sqrt_dt:          []f64,
}

// Bridge on the grid t_i = i (equal steps), i = 1 … n_steps
make_brownian_bridge :: proc(n_steps: int, allocator := context.allocator) -> BrownianBridge {
	t := make([]f64, n_steps, context.temp_allocator)
	defer delete(t, context.temp_allocator)
	for &x, i in t {x = f64(i + 1)}
	return make_brownian_bridge_times(t, allocator)
}

// Bridge on the increasing times t_0 < … < t_{n−1} (t_0 > 0)
make_brownian_bridge_times :: proc(
	times: []f64,
	allocator := context.allocator,
) -> BrownianBridge {
	n := len(times)
	b := BrownianBridge {
		size         = n,
		bridge_index = make([]int, n, allocator),
		left_index   = make([]int, n, allocator),
		right_index  = make([]int, n, allocator),
		left_weight  = make([]f64, n, allocator),
		right_weight = make([]f64, n, allocator),
		std_dev      = make([]f64, n, allocator),
		sqrt_dt      = make([]f64, n, allocator),
	}
	if n == 0 {return b}
	t := times
	b.sqrt_dt[0] = math.sqrt_f64(t[0])
	for i in 1 ..< n {b.sqrt_dt[i] = math.sqrt_f64(t[i] - t[i - 1])}
	// map[i] ≠ 0: point i is constructed by variate map[i] − 1
	mp := make([]int, n, context.temp_allocator)
	defer delete(mp, context.temp_allocator)
	mp[n - 1] = 1
	b.bridge_index[0] = n - 1
	b.std_dev[0] = math.sqrt_f64(t[n - 1])
	j := 0
	for i in 1 ..< n {
		for mp[j] != 0 {j += 1}
		k := j
		for mp[k] == 0 {k += 1}
		l := j + ((k - 1 - j) >> 1)
		mp[l] = i
		b.bridge_index[i], b.left_index[i], b.right_index[i] = l, j, k
		if j != 0 {
			b.left_weight[i] = (t[k] - t[l]) / (t[k] - t[j - 1])
			b.right_weight[i] = (t[l] - t[j - 1]) / (t[k] - t[j - 1])
			b.std_dev[i] = math.sqrt_f64((t[l] - t[j - 1]) * (t[k] - t[l]) / (t[k] - t[j - 1]))
		} else {
			b.left_weight[i] = (t[k] - t[l]) / t[k]
			b.right_weight[i] = t[l] / t[k]
			b.std_dev[i] = math.sqrt_f64(t[l] * (t[k] - t[l]) / t[k])
		}
		j = k + 1
		if j >= n {j = 0}
	}
	return b
}

free_brownian_bridge :: proc(b: ^BrownianBridge, allocator := context.allocator) {
	delete(b.bridge_index, allocator)
	delete(b.left_index, allocator)
	delete(b.right_index, allocator)
	delete(b.left_weight, allocator)
	delete(b.right_weight, allocator)
	delete(b.std_dev, allocator)
	delete(b.sqrt_dt, allocator)
	b^ = {}
}

// Standardized path increments from the variates z (both of length size)
brownian_bridge_transform :: proc(b: ^BrownianBridge, z, out: []f64) {
	n := b.size
	if n == 0 {return}
	out[n - 1] = b.std_dev[0] * z[0]
	for i in 1 ..< n {
		j, k, l := b.left_index[i], b.right_index[i], b.bridge_index[i]
		if j != 0 {
			out[l] =
				b.left_weight[i] * out[j - 1] + b.right_weight[i] * out[k] + b.std_dev[i] * z[i]
		} else {
			out[l] = b.right_weight[i] * out[k] + b.std_dev[i] * z[i]
		}
	}
	for i := n - 1; i >= 1; i -= 1 {
		out[i] = (out[i] - out[i - 1]) / b.sqrt_dt[i]
	}
	out[0] /= b.sqrt_dt[0]
}

// n_paths × n_steps × n_factors standard normals from a Sobol' sequence, in the
// layout of the mc_generate_* path generators (path-major, then step, then
// factor: e.g. Heston's (Z_v, Z_⊥) per step). Points are taken from the origin
// on, so 2^k paths form a complete net, and each coordinate is mapped by
// Φ⁻¹((x + ½)/2^32), finite at the origin; shift_seed ≠ 0 randomizes by a
// digital shift (average independent shifts for an error estimate). With brownian_bridge, factor f's
// normals come from a bridge whose variate j uses dimension j·n_factors + f, so
// the terminal values of all factors take the first dimensions. n_steps ·
// n_factors ≤ 2048.
mc_generate_sobol_normals :: proc(
	n_paths, n_steps: int,
	n_factors: int = 1,
	brownian_bridge := true,
	shift_seed: u64 = 0,
	allocator: mem.Allocator = context.allocator,
) -> []f64 {
	dims := n_steps * n_factors
	out := make([]f64, n_paths * dims, allocator)
	s := make_sobol(dims, 0, shift_seed, context.temp_allocator)
	defer free_sobol(&s, context.temp_allocator)
	bb: BrownianBridge
	if brownian_bridge {bb = make_brownian_bridge(n_steps, context.temp_allocator)}
	defer if brownian_bridge {free_brownian_bridge(&bb, context.temp_allocator)}
	u := make([]f64, dims, context.temp_allocator)
	zf := make([]f64, n_steps, context.temp_allocator)
	inc := make([]f64, n_steps, context.temp_allocator)
	defer {
		delete(u, context.temp_allocator)
		delete(zf, context.temp_allocator)
		delete(inc, context.temp_allocator)
	}
	for p in 0 ..< n_paths {
		sobol_next(&s, u)
		row := out[p * dims:(p + 1) * dims]
		for &x in u {x = _bs_ppf(x + 0.5 * _TWO_M32)}
		if !brownian_bridge {
			copy(row, u)
			continue
		}
		for f in 0 ..< n_factors {
			for j in 0 ..< n_steps {zf[j] = u[j * n_factors + f]}
			brownian_bridge_transform(&bb, zf, inc)
			for i in 0 ..< n_steps {row[i * n_factors + f] = inc[i]}
		}
	}
	return out
}
