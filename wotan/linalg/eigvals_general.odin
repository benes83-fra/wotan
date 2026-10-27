package wotan_linalg

import "core:math"
import "core:mem"

// Eigenvalues of a general (non-symmetric) real square matrix, as complex pairs
// (re[i], im[i]); complex eigenvalues come in conjugate pairs. Reduction to
// upper Hessenberg form by stabilised elementary similarity transformations,
// then the Francis double-shift QR iteration (EISPACK elmhes / hqr, as in
// Numerical Recipes 11.5–11.6). A is not modified. ok = false if an eigenvalue
// did not converge within 30 iterations (re / im are then NaN).
// Order: as found by the deflation (use eigvals_modulus_max for the spectral radius).
eigvals_general :: proc(
	A: ^Matrix(f64),
	allocator: mem.Allocator = context.allocator,
) -> (
	re: []f64,
	im: []f64,
	ok: bool,
) {
	n := A.rows
	if A.cols != n do panic("eigvals_general: matrix must be square")
	re = make([]f64, n, allocator)
	im = make([]f64, n, allocator)
	if n == 0 do return re, im, true
	a := make([]f64, n * n, context.temp_allocator)
	defer delete(a, context.temp_allocator)
	copy(a, A.data)
	at :: #force_inline proc(a: []f64, n, i, j: int) -> ^f64 {return &a[i * n + j]}

	// ---- elmhes: reduce to upper Hessenberg form ----
	for m in 1 ..< n - 1 {
		x := 0.0
		i := m
		for j in m ..< n {
			if math.abs(at(a, n, j, m - 1)^) > math.abs(x) {
				x = at(a, n, j, m - 1)^
				i = j
			}
		}
		if i != m {
			for j in m - 1 ..< n {at(a, n, i, j)^, at(a, n, m, j)^ = at(a, n, m, j)^, at(a, n, i, j)^}
			for j in 0 ..< n {at(a, n, j, i)^, at(a, n, j, m)^ = at(a, n, j, m)^, at(a, n, j, i)^}
		}
		if x != 0 {
			for i2 in m + 1 ..< n {
				y := at(a, n, i2, m - 1)^
				if y != 0 {
					y /= x
					at(a, n, i2, m - 1)^ = y
					for j in m ..< n {at(a, n, i2, j)^ -= y * at(a, n, m, j)^}
					for j in 0 ..< n {at(a, n, j, m)^ += y * at(a, n, j, i2)^}
				}
			}
		}
	}
	// zero the multipliers stored below the subdiagonal
	for i in 2 ..< n {
		for j in 0 ..< i - 1 {at(a, n, i, j)^ = 0}
	}

	// ---- hqr: eigenvalues of the Hessenberg matrix ----
	anorm := 0.0
	for i in 0 ..< n {
		for j in max(i - 1, 0) ..< n {anorm += math.abs(at(a, n, i, j)^)}
	}
	nn := n - 1
	t := 0.0
	p, q, r, s, w, x, y, z: f64
	ok = true
	for nn >= 0 {
		its := 0
		l: int
		for {
			// look for a single small subdiagonal element
			for l = nn; l >= 1; l -= 1 {
				s = math.abs(at(a, n, l - 1, l - 1)^) + math.abs(at(a, n, l, l)^)
				if s == 0 do s = anorm
				if math.abs(at(a, n, l, l - 1)^) + s == s {
					at(a, n, l, l - 1)^ = 0
					break
				}
			}
			x = at(a, n, nn, nn)^
			if l == nn {
				// one root found
				re[nn], im[nn] = x + t, 0
				nn -= 1
				break
			}
			y = at(a, n, nn - 1, nn - 1)^
			w = at(a, n, nn, nn - 1)^ * at(a, n, nn - 1, nn)^
			if l == nn - 1 {
				// two roots found
				p = 0.5 * (y - x)
				q = p * p + w
				z = math.sqrt(math.abs(q))
				x += t
				if q >= 0 {
					z = p + math.copy_sign(z, p)
					re[nn - 1], re[nn] = x + z, x + z
					if z != 0 do re[nn] = x - w / z
					im[nn - 1], im[nn] = 0, 0
				} else {
					re[nn - 1], re[nn] = x + p, x + p
					im[nn - 1], im[nn] = -z, z
				}
				nn -= 2
				break
			}
			if its == 30 {
				for i in 0 ..< n {re[i], im[i] = math.nan_f64(), math.nan_f64()}
				return re, im, false
			}
			if its == 10 || its == 20 {
				// exceptional shift
				t += x
				for i in 0 ..= nn {at(a, n, i, i)^ -= x}
				s = math.abs(at(a, n, nn, nn - 1)^) + math.abs(at(a, n, nn - 1, nn - 2)^)
				x = 0.75 * s
				y = x
				w = -0.4375 * s * s
			}
			its += 1
			m: int
			for m = nn - 2; m >= l; m -= 1 {
				z = at(a, n, m, m)^
				r = x - z
				s = y - z
				p = (r * s - w) / at(a, n, m + 1, m)^ + at(a, n, m, m + 1)^
				q = at(a, n, m + 1, m + 1)^ - z - r - s
				r = at(a, n, m + 2, m + 1)^
				s = math.abs(p) + math.abs(q) + math.abs(r)
				p /= s
				q /= s
				r /= s
				if m == l do break
				u := math.abs(at(a, n, m, m - 1)^) * (math.abs(q) + math.abs(r))
				v := math.abs(p) * (math.abs(at(a, n, m - 1, m - 1)^) + math.abs(z) + math.abs(at(a, n, m + 1, m + 1)^))
				if u + v == v do break
			}
			for i in m + 2 ..= nn {
				at(a, n, i, i - 2)^ = 0
				if i != m + 2 do at(a, n, i, i - 3)^ = 0
			}
			for k := m; k <= nn - 1; k += 1 {
				if k != m {
					p = at(a, n, k, k - 1)^
					q = at(a, n, k + 1, k - 1)^
					r = 0
					if k != nn - 1 do r = at(a, n, k + 2, k - 1)^
					x = math.abs(p) + math.abs(q) + math.abs(r)
					if x != 0 {
						p /= x
						q /= x
						r /= x
					}
				}
				s = math.copy_sign(math.sqrt(p * p + q * q + r * r), p)
				if s != 0 {
					if k == m {
						if l != m do at(a, n, k, k - 1)^ = -at(a, n, k, k - 1)^
					} else {
						at(a, n, k, k - 1)^ = -s * x
					}
					p += s
					x = p / s
					y = q / s
					z = r / s
					q /= p
					r /= p
					for j in k ..= nn {
						p = at(a, n, k, j)^ + q * at(a, n, k + 1, j)^
						if k != nn - 1 {
							p += r * at(a, n, k + 2, j)^
							at(a, n, k + 2, j)^ -= p * z
						}
						at(a, n, k + 1, j)^ -= p * y
						at(a, n, k, j)^ -= p * x
					}
					mmin := nn if nn < k + 3 else k + 3
					for i in l ..= mmin {
						p = x * at(a, n, i, k)^ + y * at(a, n, i, k + 1)^
						if k != nn - 1 {
							p += z * at(a, n, i, k + 2)^
							at(a, n, i, k + 2)^ -= p * r
						}
						at(a, n, i, k + 1)^ -= p * q
						at(a, n, i, k)^ -= p
					}
				}
			}
		}
	}
	return re, im, ok
}

// Largest |λ| (spectral radius) of a general square matrix; NaN if the QR iteration failed
eigvals_modulus_max :: proc(A: ^Matrix(f64)) -> f64 {
	re, im, ok := eigvals_general(A, context.temp_allocator)
	defer {
		delete(re, context.temp_allocator)
		delete(im, context.temp_allocator)
	}
	if !ok do return math.nan_f64()
	m := 0.0
	for i in 0 ..< len(re) {m = max(m, math.sqrt(re[i] * re[i] + im[i] * im[i]))}
	return m
}
