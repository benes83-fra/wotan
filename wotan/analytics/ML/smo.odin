package ML

import l "../../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// SMO solver for the SVM duals (LIBSVM's algorithm: Fan, Chen & Lin 2005)
//
//   min ½ αᵀQα + pᵀα   s.t. 0 ≤ α_t ≤ C,  yᵀα = 0   (y_t = ±1)
//   Q_ts = y_t y_s K(x_{t mod n}, x_{s mod n})
//
// SVC uses l = n variables (p = -1); ε-SVR uses l = 2n (α then α*). Working
// set by second-order selection (WSS2), stopping when the maximal KKT
// violation m(α) - M(α) < tol, bias from LIBSVM's ρ rule (decision = Σ ... - ρ).
// ============================================================================

_SMO_TAU :: 1e-12
_SMO_KERNEL_CACHE_MAX :: 4000 // precompute the n×n kernel matrix up to this n

_SMOKernel :: struct {
	X:           ^l.Matrix(f64),
	kernel_type: SVMKernelType,
	gamma:       f64,
	degree:      int,
	coef0:       f64,
	n:           int,
	K:           []f64, // n×n if cached, else nil
	diag:        []f64, // K(x_i, x_i)
	rows:        [2][]f64, // on-demand row buffers
	allocator:   mem.Allocator,
}

_smo_kernel_init :: proc(
	X: ^l.Matrix(f64),
	kernel_type: SVMKernelType,
	gamma: f64,
	degree: int,
	coef0: f64,
	allocator: mem.Allocator,
) -> _SMOKernel {
	n, p := X.rows, X.cols
	k := _SMOKernel {
		X           = X,
		kernel_type = kernel_type,
		gamma       = gamma,
		degree      = degree,
		coef0       = coef0,
		n           = n,
		allocator   = allocator,
	}
	k.diag = make([]f64, n, allocator)
	for i in 0 ..< n {
		xi := X.data[i * p:(i + 1) * p]
		k.diag[i] = _kernel_eval(xi, xi, kernel_type, gamma, degree, coef0)
	}
	if n <= _SMO_KERNEL_CACHE_MAX {
		k.K = make([]f64, n * n, allocator)
		for i in 0 ..< n {
			xi := X.data[i * p:(i + 1) * p]
			for j in i ..< n {
				v := _kernel_eval(xi, X.data[j * p:(j + 1) * p], kernel_type, gamma, degree, coef0)
				k.K[i * n + j] = v
				k.K[j * n + i] = v
			}
		}
	} else {
		k.rows[0] = make([]f64, n, allocator)
		k.rows[1] = make([]f64, n, allocator)
	}
	return k
}

_smo_kernel_free :: proc(k: ^_SMOKernel) {
	delete(k.diag, k.allocator)
	if k.K != nil do delete(k.K, k.allocator)
	if k.rows[0] != nil do delete(k.rows[0], k.allocator)
	if k.rows[1] != nil do delete(k.rows[1], k.allocator)
}

// Row K(x_i, ·) over the n base points (slot selects the on-demand buffer)
_smo_kernel_row :: proc(k: ^_SMOKernel, i: int, slot: int) -> []f64 {
	n := k.n
	if k.K != nil do return k.K[i * n:(i + 1) * n]
	p := k.X.cols
	xi := k.X.data[i * p:(i + 1) * p]
	row := k.rows[slot]
	for j in 0 ..< n {
		row[j] = _kernel_eval(xi, k.X.data[j * p:(j + 1) * p], k.kernel_type, k.gamma, k.degree, k.coef0)
	}
	return row
}

// Solve the dual. `alpha` (length l = len(p)) is the starting point (usually 0)
// and is overwritten. With `equality = false` the constraint yᵀα = 0 is dropped
// (no intercept) and single coordinates are optimised instead of pairs.
// Iteration limit: max_iter pair updates (≤ 0: 10 000 000, as LIBSVM warns).
_smo_solve :: proc(
	k: ^_SMOKernel,
	p: []f64,
	y: []f64,
	C: f64,
	tol: f64,
	max_iter: int,
	equality: bool,
	alpha: []f64,
) -> (
	rho: f64,
	converged: bool,
	n_iter: int,
) {
	n := k.n
	L := len(p)
	eps := tol if tol > 0 else 1e-3
	limit := max_iter if max_iter > 0 else 10_000_000

	G := make([]f64, L, context.temp_allocator)
	defer delete(G, context.temp_allocator)
	copy(G, p)
	// G = Qα + p for a non-zero start
	for s in 0 ..< L {
		if alpha[s] == 0 do continue
		row := _smo_kernel_row(k, s % n, 0)
		for t in 0 ..< L {G[t] += y[t] * y[s] * row[t % n] * alpha[s]}
	}

	upper :: proc(a, C: f64) -> bool {return a >= C}
	lower :: proc(a: f64) -> bool {return a <= 0}

	for n_iter < limit {
		if !equality {
			// pick the coordinate with the largest projected-gradient violation
			best, best_v := -1, 0.0
			for t in 0 ..< L {
				g := G[t]
				pg := g
				if lower(alpha[t]) do pg = math.min(g, 0)
				else if upper(alpha[t], C) do pg = math.max(g, 0)
				if math.abs(pg) > best_v {best_v = math.abs(pg); best = t}
			}
			if best < 0 || best_v < eps {converged = true; break}
			i := best
			qii := math.max(k.diag[i % n], _SMO_TAU)
			old := alpha[i]
			alpha[i] = math.clamp(old - G[i] / qii, 0, C)
			d := alpha[i] - old
			row := _smo_kernel_row(k, i % n, 0)
			for t in 0 ..< L {G[t] += y[t] * y[i] * row[t % n] * d}
			n_iter += 1
			continue
		}

		// ---- working set selection (WSS2) ----
		Gmax, Gmax2 := -math.INF_F64, -math.INF_F64
		i := -1
		for t in 0 ..< L {
			if y[t] > 0 {
				if !upper(alpha[t], C) && -G[t] >= Gmax {Gmax = -G[t]; i = t}
			} else {
				if !lower(alpha[t]) && G[t] >= Gmax {Gmax = G[t]; i = t}
			}
		}
		if i < 0 {converged = true; break}
		Ki := _smo_kernel_row(k, i % n, 0)
		j := -1
		obj_min := math.INF_F64
		for t in 0 ..< L {
			if y[t] > 0 {
				if !lower(alpha[t]) {
					grad_diff := Gmax + G[t]
					if G[t] >= Gmax2 do Gmax2 = G[t]
					if grad_diff > 0 {
						quad := k.diag[i % n] + k.diag[t % n] - 2.0 * Ki[t % n]
						if quad <= 0 do quad = _SMO_TAU
						obj := -(grad_diff * grad_diff) / quad
						if obj <= obj_min {obj_min = obj; j = t}
					}
				}
			} else {
				if !upper(alpha[t], C) {
					grad_diff := Gmax - G[t]
					if -G[t] >= Gmax2 do Gmax2 = -G[t]
					if grad_diff > 0 {
						quad := k.diag[i % n] + k.diag[t % n] - 2.0 * Ki[t % n]
						if quad <= 0 do quad = _SMO_TAU
						obj := -(grad_diff * grad_diff) / quad
						if obj <= obj_min {obj_min = obj; j = t}
					}
				}
			}
		}
		if Gmax + Gmax2 < eps || j < 0 {converged = true; break}

		// ---- two-variable update (LIBSVM Solver::Solve) ----
		Kj := _smo_kernel_row(k, j % n, 1)
		Kij := Ki[j % n]
		Qij := y[i] * y[j] * Kij
		old_ai, old_aj := alpha[i], alpha[j]
		if y[i] != y[j] {
			quad := k.diag[i % n] + k.diag[j % n] + 2.0 * Qij
			if quad <= 0 do quad = _SMO_TAU
			delta := (-G[i] - G[j]) / quad
			diff := alpha[i] - alpha[j]
			alpha[i] += delta
			alpha[j] += delta
			if diff > 0 {
				if alpha[j] < 0 {alpha[j] = 0; alpha[i] = diff}
			} else {
				if alpha[i] < 0 {alpha[i] = 0; alpha[j] = -diff}
			}
			if diff > 0 {
				if alpha[i] > C {alpha[i] = C; alpha[j] = C - diff}
			} else {
				if alpha[j] > C {alpha[j] = C; alpha[i] = C + diff}
			}
		} else {
			quad := k.diag[i % n] + k.diag[j % n] - 2.0 * Qij
			if quad <= 0 do quad = _SMO_TAU
			delta := (G[i] - G[j]) / quad
			sum := alpha[i] + alpha[j]
			alpha[i] -= delta
			alpha[j] += delta
			if sum > C {
				if alpha[i] > C {alpha[i] = C; alpha[j] = sum - C}
			} else {
				if alpha[j] < 0 {alpha[j] = 0; alpha[i] = sum}
			}
			if sum > C {
				if alpha[j] > C {alpha[j] = C; alpha[i] = sum - C}
			} else {
				if alpha[i] < 0 {alpha[i] = 0; alpha[j] = sum}
			}
		}

		dai := alpha[i] - old_ai
		daj := alpha[j] - old_aj
		for t in 0 ..< L {
			G[t] += y[t] * (y[i] * Ki[t % n] * dai + y[j] * Kj[t % n] * daj)
		}
		n_iter += 1
	}

	// Snap rounding-level values onto the bounds, so an α of ~1e-17 is not
	// mistaken for a free support vector (it would pin ρ, LIBSVM gets exact 0)
	snap := 1e-12 * C
	for t in 0 ..< L {
		if alpha[t] <= snap do alpha[t] = 0
		else if alpha[t] >= C - snap do alpha[t] = C
	}

	if !equality do return 0, converged, n_iter

	// ---- ρ (LIBSVM calculate_rho) ----
	ub, lb := math.INF_F64, -math.INF_F64
	nr_free, sum_free := 0, 0.0
	for t in 0 ..< L {
		yG := y[t] * G[t]
		if upper(alpha[t], C) {
			if y[t] < 0 {ub = math.min(ub, yG)} else {lb = math.max(lb, yG)}
		} else if lower(alpha[t]) {
			if y[t] > 0 {ub = math.min(ub, yG)} else {lb = math.max(lb, yG)}
		} else {
			nr_free += 1
			sum_free += yG
		}
	}
	rho = sum_free / f64(nr_free) if nr_free > 0 else (ub + lb) / 2
	return rho, converged, n_iter
}

// sklearn gamma="scale": 1 / (n_features · Var(X)) over all entries of X
_svm_gamma_scale :: proc(X: ^l.Matrix(f64)) -> f64 {
	m := len(X.data)
	if m == 0 do return 1.0
	mean := 0.0
	for v in X.data do mean += v
	mean /= f64(m)
	var := 0.0
	for v in X.data do var += (v - mean) * (v - mean)
	var /= f64(m)
	if var == 0 do return 1.0
	return 1.0 / (f64(X.cols) * var)
}
