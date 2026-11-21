package optimize

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// L-BFGS
// ============================================================================
// Quasi-Newton minimisation with a limited memory: the inverse Hessian is
// approximated from the last m steps s = x_{k+1} - x_k and gradient changes
// y = g_{k+1} - g_k by the two-loop recursion (Nocedal & Wright, algorithm 7.4).
// Two interfaces:
// - lbfgs_minimize runs the whole minimisation from a procedure that returns
//   the objective and its gradient, and stops when the gradient is small;
// - optimizer_lbfgs_step takes one step inside the caller's loop, with the
//   optimizer_step interface shared with SGD, Adam and RMSProp.

// Objective at x (the line search of optimizer_lbfgs_step)
ObjectiveFunc :: proc(x: []f64, user_data: rawptr) -> f64

// Objective and gradient at x: writes the gradient to grad, returns the objective
ValueGradFunc :: proc(x: []f64, grad: []f64, user_data: rawptr) -> f64

// ============================================================================
// History and search direction (shared by both interfaces)
// ============================================================================

// The last m pairs (s, y) in a ring: pair i of the ring is s_hist[i*n:][:n],
// y_hist[i*n:][:n] and rho[i] = 1 / sᵀy; `newest` is the slot of the newest of
// the `filled` stored pairs
_LBFGS_History :: struct {
	n, m:           int,
	s_hist, y_hist: []f64,
	rho, alphas:    []f64,
	filled, newest: int,
}

_lbfgs_history_init :: proc(n, m: int, allocator: mem.Allocator) -> (h: _LBFGS_History) {
	h.n, h.m = n, max(m, 1)
	h.s_hist = make([]f64, h.m * n, allocator)
	h.y_hist = make([]f64, h.m * n, allocator)
	h.rho = make([]f64, h.m, allocator)
	h.alphas = make([]f64, h.m, allocator)
	h.newest = h.m - 1
	return
}

_lbfgs_history_free :: proc(h: ^_LBFGS_History, allocator: mem.Allocator) {
	delete(h.s_hist, allocator)
	delete(h.y_hist, allocator)
	delete(h.rho, allocator)
	delete(h.alphas, allocator)
}

// The slot the next pair is written to (the oldest once the ring is full)
_lbfgs_next_slot :: proc(h: ^_LBFGS_History) -> (s, y: []f64) {
	j := (h.newest + 1) % h.m
	return h.s_hist[j * h.n:][:h.n], h.y_hist[j * h.n:][:h.n]
}

// Keeps the pair written to the next slot if it has positive curvature
// (sᵀy > 0, scaled by yᵀy), which keeps the approximation positive definite
_lbfgs_commit :: proc(h: ^_LBFGS_History) -> bool {
	s, y := _lbfgs_next_slot(h)
	sy := l.dot_simd(s, y)
	yy := l.dot_simd(y, y)
	if !(sy > 1e-12 * yy) || !(sy > 0) {return false}
	h.newest = (h.newest + 1) % h.m
	h.rho[h.newest] = 1 / sy
	h.filled = min(h.filled + 1, h.m)
	return true
}

// d = -H g by the two-loop recursion: newest to oldest, scale by
// H₀ = (sᵀy / yᵀy) I from the newest pair, oldest to newest
_lbfgs_direction :: proc(h: ^_LBFGS_History, g, d: []f64) {
	n, m := h.n, h.m
	copy(d, g)
	for i in 0 ..< h.filled {
		j := (h.newest - i + m) % m
		h.alphas[i] = h.rho[j] * l.dot_simd(h.s_hist[j * n:][:n], d)
		l.axpy_simd(-h.alphas[i], h.y_hist[j * n:][:n], d)
	}
	if h.filled > 0 {
		s := h.s_hist[h.newest * n:][:n]
		y := h.y_hist[h.newest * n:][:n]
		gamma := l.dot_simd(s, y) / l.dot_simd(y, y)
		for &v in d {v *= gamma}
	}
	for i := h.filled - 1; i >= 0; i -= 1 {
		j := (h.newest - i + m) % m
		beta := h.rho[j] * l.dot_simd(h.y_hist[j * n:][:n], d)
		l.axpy_simd(h.alphas[i] - beta, h.s_hist[j * n:][:n], d)
	}
	for &v in d {v = -v}
}

_lbfgs_max_abs :: proc(v: []f64) -> f64 {
	r := 0.0
	for x in v {r = max(r, math.abs(x))}
	return r
}

_lbfgs_finite :: proc(f: f64) -> bool {return !math.is_nan(f) && !math.is_inf(f)}

// ============================================================================
// lbfgs_minimize: the whole minimisation
// ============================================================================

LBFGS_Result :: struct {
	f:         f64, // the objective at the returned x
	n_iter:    int, // accepted steps
	n_eval:    int, // calls of the objective (each with its gradient)
	converged: bool, // no gradient component exceeds gtol at x
}

// Minimises fg from x (overwritten with the minimiser). Each iteration takes
// the L-BFGS direction with a backtracking line search (Armijo condition,
// quadratic interpolation, given up once a step no longer changes x); the
// first step moves no coordinate by more than 1.
// A non-finite objective counts as too large. Stops when no gradient component
// exceeds gtol (converged), after max_iter iterations, when the line search
// finds no decrease, or when three steps in a row change the objective by less
// than its rounding.
lbfgs_minimize :: proc(
	fg: ValueGradFunc,
	x: []f64,
	user_data: rawptr,
	max_iter := 1000,
	gtol := 1e-6,
	m := 10,
	allocator := context.allocator,
) -> (
	res: LBFGS_Result,
) {
	n := len(x)
	if n == 0 {return {converged = true}}
	h := _lbfgs_history_init(n, m, allocator)
	defer _lbfgs_history_free(&h, allocator)
	buf := make([]f64, 4 * n, allocator)
	defer delete(buf, allocator)
	g := buf[:n]
	g_new := buf[n:][:n]
	x_new := buf[2 * n:][:n]
	d := buf[3 * n:][:n]

	f := fg(x, g, user_data)
	res.n_eval = 1
	if !_lbfgs_finite(f) {return {f = math.INF_F64, n_eval = 1}}
	stalls := 0
	for it in 0 ..< max_iter {
		if _lbfgs_max_abs(g) <= gtol {break}
		_lbfgs_direction(&h, g, d)
		gd := l.dot_simd(g, d)
		if !(gd < 0) {
			// not a descent direction: forget the history, steepest descent
			h.filled = 0
			for i in 0 ..< n {d[i] = -g[i]}
			gd = -l.dot_simd(g, g)
		}
		step := 1.0
		if h.filled == 0 {step = min(1.0, 1.0 / _lbfgs_max_abs(d))}
		f_new := math.INF_F64
		accepted := false
		for _ in 0 ..< 60 {
			moved := false
			for i in 0 ..< n {
				x_new[i] = x[i] + step * d[i]
				moved ||= x_new[i] != x[i]
			}
			if !moved {break} 	// the step no longer changes x: no decrease to find
			f_new = fg(x_new, g_new, user_data)
			res.n_eval += 1
			finite := _lbfgs_finite(f_new)
			if finite && f_new <= f + 1e-4 * step * gd {accepted = true; break}
			if finite {
				// minimiser of the quadratic through f, f'(0) and f(step), kept in [0.1, 0.5]·step
				den := 2 * (f_new - f - step * gd)
				step =
					clamp(-gd * step * step / den, 0.1 * step, 0.5 * step) if den > 0 else 0.5 * step
			} else {
				step *= 0.1
			}
		}
		if !accepted {break}
		s, y := _lbfgs_next_slot(&h)
		for i in 0 ..< n {
			s[i] = x_new[i] - x[i]
			y[i] = g_new[i] - g[i]
		}
		_lbfgs_commit(&h)
		stalls = stalls + 1 if f - f_new <= 1e-15 * math.abs(f) else 0
		copy(x, x_new)
		copy(g, g_new)
		f = f_new
		res.n_iter = it + 1
		if stalls >= 3 {break}
	}
	res.f = f
	res.converged = _lbfgs_max_abs(g) <= gtol
	return
}

// ============================================================================
// optimizer_lbfgs_step: one step at a time
// ============================================================================

OptimizerLBFGS :: struct {
	learning_rate:      f64, // the trial step along the L-BFGS direction
	m:                  int, // pairs kept
	tol:                f64,
	n_params:           int,
	is_initialized:     bool,
	k:                  int, // steps taken
	line_search_failed: bool, // the last step found no decrease and left the weights unchanged
	hist:               _LBFGS_History,
	d:                  []f64,
	x_new:              []f64,
	prev_x:             []f64,
	prev_grad:          []f64,
	has_prev:           bool,
	prev_loss:          f64,
	allocator:          mem.Allocator,
}

optimizer_lbfgs_init :: proc(
	n_params: int,
	lr: f64 = 1.0,
	m: int = 10,
	tol: f64 = 1e-5,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerLBFGS {
	opt: OptimizerLBFGS
	opt.learning_rate = lr
	opt.m = m
	opt.tol = tol
	opt.n_params = n_params
	opt.allocator = allocator
	opt.is_initialized = true

	if n_params == 0 {return opt}

	opt.hist = _lbfgs_history_init(n_params, m, allocator)
	opt.d = make([]f64, n_params, allocator)
	opt.x_new = make([]f64, n_params, allocator)
	opt.prev_x = make([]f64, n_params, allocator)
	opt.prev_grad = make([]f64, n_params, allocator)

	return opt
}

optimizer_lbfgs_free :: proc(opt: ^OptimizerLBFGS) {
	if !opt.is_initialized {return}
	_lbfgs_history_free(&opt.hist, opt.allocator)
	delete(opt.d, opt.allocator)
	delete(opt.x_new, opt.allocator)
	delete(opt.prev_x, opt.allocator)
	delete(opt.prev_grad, opt.allocator)
	opt.is_initialized = false
}

// One L-BFGS step from weights, given the gradient (and the loss) there. The
// pair (s, y) from the previous call enters the history first. With obj_fn the
// step along the direction starts at learning_rate and is halved until the
// loss decreases enough (Armijo); if 40 halvings do not get there, the weights
// stay unchanged and line_search_failed is set. Without obj_fn the step is
// learning_rate. While the history is empty the step moves no weight by more
// than learning_rate.
optimizer_lbfgs_step :: proc(
	opt: ^OptimizerLBFGS,
	weights: []f64,
	gradient: []f64,
	loss: f64 = 0.0,
	obj_fn: ObjectiveFunc = nil,
	obj_user_data: rawptr = nil,
) {
	n := len(weights)
	if n == 0 || n != len(gradient) || !opt.is_initialized {return}
	h := &opt.hist

	if opt.has_prev {
		s, y := _lbfgs_next_slot(h)
		for i in 0 ..< n {
			s[i] = weights[i] - opt.prev_x[i]
			y[i] = gradient[i] - opt.prev_grad[i]
		}
		_lbfgs_commit(h)
	}

	_lbfgs_direction(h, gradient, opt.d)
	dir_deriv := l.dot_simd(gradient, opt.d)
	if !(dir_deriv < 0) {
		h.filled = 0
		for i in 0 ..< n {opt.d[i] = -gradient[i]}
		dir_deriv = -l.dot_simd(gradient, gradient)
	}

	step_size := opt.learning_rate
	if h.filled == 0 {
		dmax := _lbfgs_max_abs(opt.d)
		if dmax > 1 {step_size /= dmax}
	}

	opt.line_search_failed = false
	if obj_fn != nil {
		c1 := 1e-4
		accepted := false
		for _ in 0 ..< 40 {
			copy(opt.x_new, weights)
			l.axpy_simd(step_size, opt.d, opt.x_new)
			new_loss := obj_fn(opt.x_new, obj_user_data)
			if _lbfgs_finite(new_loss) && new_loss <= loss + c1 * step_size * dir_deriv {
				accepted = true
				break
			}
			step_size *= 0.5
		}
		if !accepted {
			opt.line_search_failed = true
			opt.has_prev = false
			return
		}
	}

	copy(opt.prev_x, weights)
	copy(opt.prev_grad, gradient)
	opt.prev_loss = loss
	opt.has_prev = true

	l.axpy_simd(step_size, opt.d, weights)

	opt.k += 1
}

optimizer_lbfgs_step_mat :: proc(
	opt: ^OptimizerLBFGS,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
	loss: f64 = 0.0,
	obj_fn: ObjectiveFunc = nil,
	obj_user_data: rawptr = nil,
) {
	if weights.rows != gradient.rows || weights.cols != gradient.cols {return}
	if weights.rows == 0 || weights.cols == 0 {return}
	if !opt.is_initialized {return}

	n := weights.rows * weights.cols
	weights_flat := weights.data[0:n]
	gradient_flat := gradient.data[0:n]

	optimizer_lbfgs_step(opt, weights_flat, gradient_flat, loss, obj_fn, obj_user_data)
}
