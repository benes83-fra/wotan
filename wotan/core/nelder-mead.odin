
package core

import "core:math"
import "core:mem"

ObjectiveWithCtx :: proc(x: []f64, ctx: rawptr) -> f64

vec_copy :: proc(dst, src: []f64) {
	for i in 0 ..< len(dst) {
		dst[i] = src[i]
	}
}

vec_add_scaled :: proc(out, a, b: []f64, alpha: f64) {
	// out = a + alpha * b
	for i in 0 ..< len(out) {
		out[i] = a[i] + alpha * b[i]
	}
}

vec_add :: proc(out, a, b: []f64) {
	for i in 0 ..< len(out) {
		out[i] = a[i] + b[i]
	}
}

vec_sub :: proc(out, a, b: []f64) {
	for i in 0 ..< len(out) {
		out[i] = a[i] - b[i]
	}
}

vec_scale :: proc(out, a: []f64, alpha: f64) {
	for i in 0 ..< len(out) {
		out[i] = alpha * a[i]
	}
}

nelder_mead :: proc(
	f: ObjectiveWithCtx,
	ctx: rawptr,
	x0: []f64,
	max_iter: int,
	tol: f64,
	allocator: mem.Allocator,
) -> (
	best_x: []f64,
	best_f: f64,
) {
	n := len(x0)
	if n == 0 {
		best_x = make([]f64, 0, allocator)
		best_f = 0.0
		return
	}

	// Simplex: n+1 vertices
	simplex := make([][]f64, n + 1, allocator)
	values := make([]f64, n + 1, allocator)

	for i in 0 ..< n + 1 {
		simplex[i] = make([]f64, n, allocator)
	}

	// Initialize simplex around x0
	for j in 0 ..< n {
		simplex[0][j] = x0[j]
	}
	for i in 1 ..< n + 1 {
		vec_copy(simplex[i], simplex[0])
		simplex[i][i - 1] += 0.05 // small perturbation
	}

	for i in 0 ..< n + 1 {
		values[i] = f(simplex[i], ctx)
	}

	alpha := 1.0
	gamma := 2.0
	rho := 0.5
	sigma := 0.5

	centroid := make([]f64, n, allocator)
	xr := make([]f64, n, allocator)
	xe := make([]f64, n, allocator)
	xc := make([]f64, n, allocator)

	for iter in 0 ..< max_iter {
		// sort simplex by value
		for i in 0 ..< n + 1 {
			for j in i + 1 ..< n + 1 {
				if values[j] < values[i] {
					values[i], values[j] = values[j], values[i]
					simplex[i], simplex[j] = simplex[j], simplex[i]
				}
			}
		}

		// check convergence: spread of values
		fmin := values[0]
		fmax := values[n]
		if math.abs(fmax - fmin) < tol {
			break
		}

		// centroid of all but worst
		for j in 0 ..< n {
			centroid[j] = 0.0
		}
		for i in 0 ..< n {
			for j in 0 ..< n {
				centroid[j] += simplex[i][j]
			}
		}
		for j in 0 ..< n {
			centroid[j] /= f64(n)
		}

		// reflection
		for j in 0 ..< n {
			xr[j] = centroid[j] + alpha * (centroid[j] - simplex[n][j])
		}
		fr := f(xr, ctx)

		if fr >= values[0] && fr < values[n - 1] {
			// accept reflection
			vec_copy(simplex[n], xr)
			values[n] = fr
			continue
		}

		if fr < values[0] {
			// expansion
			for j in 0 ..< n {
				xe[j] = centroid[j] + gamma * (xr[j] - centroid[j])
			}
			fe := f(xe, ctx)
			if fe < fr {
				vec_copy(simplex[n], xe)
				values[n] = fe
			} else {
				vec_copy(simplex[n], xr)
				values[n] = fr
			}
			continue
		}

		// contraction
		for j in 0 ..< n {
			xc[j] = centroid[j] + rho * (simplex[n][j] - centroid[j])
		}
		fc := f(xc, ctx)

		if fc < values[n] {
			vec_copy(simplex[n], xc)
			values[n] = fc
			continue
		}

		// shrink
		for i in 1 ..< n + 1 {
			for j in 0 ..< n {
				simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j])
			}
			values[i] = f(simplex[i], ctx)
		}
	}

	best_x = make([]f64, n, allocator)
	vec_copy(best_x, simplex[0])
	best_f = values[0]
	return
}
