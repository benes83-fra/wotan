package optimize

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Objective Function Type (Required for L-BFGS Line Search)
// ============================================================================
ObjectiveFunc :: proc(x: []f64, user_data: rawptr) -> f64

// ============================================================================
// L-BFGS Optimizer
// ============================================================================

OptimizerLBFGS :: struct {
	learning_rate:  f64,
	m:              int,
	tol:            f64,
	n_params:       int,
	is_initialized: bool,
	s_history:      []f64,
	y_history:      []f64,
	rho_history:    []f64,
	q:              []f64,
	d:              []f64,
	alphas:         []f64,
	x_new:          []f64,
	k:              int,
	filled:         int,
	prev_x:         []f64,
	prev_grad:      []f64,
	prev_loss:      f64,
	allocator:      mem.Allocator,
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

	opt.s_history = make([]f64, m * n_params, allocator)
	opt.y_history = make([]f64, m * n_params, allocator)
	opt.rho_history = make([]f64, m, allocator)
	opt.q = make([]f64, n_params, allocator)
	opt.d = make([]f64, n_params, allocator)
	opt.alphas = make([]f64, m, allocator)
	opt.x_new = make([]f64, n_params, allocator)
	opt.prev_x = make([]f64, n_params, allocator)
	opt.prev_grad = make([]f64, n_params, allocator)

	return opt
}

optimizer_lbfgs_free :: proc(opt: ^OptimizerLBFGS) {
	if !opt.is_initialized {return}
	delete(opt.s_history, opt.allocator)
	delete(opt.y_history, opt.allocator)
	delete(opt.rho_history, opt.allocator)
	delete(opt.q, opt.allocator)
	delete(opt.d, opt.allocator)
	delete(opt.alphas, opt.allocator)
	delete(opt.x_new, opt.allocator)
	delete(opt.prev_x, opt.allocator)
	delete(opt.prev_grad, opt.allocator)
	opt.is_initialized = false
}

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

	if opt.k > 0 {
		idx := (opt.k - 1) % opt.m
		s_idx := idx * n
		y_idx := idx * n

		for i in 0 ..< n {
			opt.s_history[s_idx + i] = weights[i] - opt.prev_x[i]
			opt.y_history[y_idx + i] = gradient[i] - opt.prev_grad[i]
		}

		ys := l.dot_simd(opt.s_history[s_idx:s_idx + n], opt.y_history[y_idx:y_idx + n])
		if ys > 1e-10 {
			opt.rho_history[idx] = 1.0 / ys
			if opt.filled < opt.m {opt.filled += 1}
		} else {
			opt.rho_history[idx] = 0.0
		}
	}

	copy(opt.q, gradient)

	for i in 0 ..< opt.filled {
		hist_idx := (opt.k - 1 - i) % opt.m
		if hist_idx < 0 {hist_idx += opt.m}

		s_idx := hist_idx * n
		y_idx := hist_idx * n

		rho := opt.rho_history[hist_idx]
		alpha := rho * l.dot_simd(opt.s_history[s_idx:s_idx + n], opt.q)
		opt.alphas[i] = alpha
		l.axpy_simd(-alpha, opt.y_history[y_idx:y_idx + n], opt.q)
	}

	if opt.filled > 0 {
		hist_idx := (opt.k - 1) % opt.m
		y_idx := hist_idx * n
		s_idx := hist_idx * n

		ys := l.dot_simd(opt.y_history[y_idx:y_idx + n], opt.s_history[s_idx:s_idx + n])
		yy := l.dot_simd(opt.y_history[y_idx:y_idx + n], opt.y_history[y_idx:y_idx + n])
		gamma := ys / (yy + 1e-10)

		for i in 0 ..< n {opt.q[i] *= gamma}
	}

	for i := opt.filled - 1; i >= 0; i -= 1 {
		hist_idx := (opt.k - opt.filled + i) % opt.m
		if hist_idx < 0 {hist_idx += opt.m}

		s_idx := hist_idx * n
		y_idx := hist_idx * n

		rho := opt.rho_history[hist_idx]
		beta := rho * l.dot_simd(opt.y_history[y_idx:y_idx + n], opt.q)
		l.axpy_simd(opt.alphas[i] - beta, opt.s_history[s_idx:s_idx + n], opt.q)
	}

	for i in 0 ..< n {opt.d[i] = -opt.q[i]}

	step_size := opt.learning_rate
	dir_deriv := l.dot_simd(gradient, opt.d)

	if dir_deriv >= 0.0 {
		for i in 0 ..< n {opt.d[i] = -gradient[i]}
		dir_deriv = -l.dot_simd(gradient, gradient)
	}

	if obj_fn != nil {
		c1 := 1e-4
		for _ in 0 ..< 20 {
			copy(opt.x_new, weights)
			l.axpy_simd(step_size, opt.d, opt.x_new)

			new_loss := obj_fn(opt.x_new, obj_user_data)
			if new_loss <= loss + c1 * step_size * dir_deriv {
				break
			}
			step_size *= 0.5
		}
	}

	copy(opt.prev_x, weights)
	copy(opt.prev_grad, gradient)
	opt.prev_loss = loss

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
