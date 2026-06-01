package optimize

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Objective Function Type (Required for L-BFGS Line Search)
// ============================================================================
ObjectiveFunc :: proc(x: []f64) -> f64

// ============================================================================
// L-BFGS Optimizer (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
// ============================================================================

OptimizerLBFGS :: struct {
	learning_rate:  f64, // Max step size / initial learning rate
	m:              int, // Memory size (number of past steps to store)
	tol:            f64, // Gradient norm tolerance for convergence
	n_params:       int,
	is_initialized: bool, // ✅ FIX: Track initialization state safely

	// History buffers (pre-allocated circular buffers)
	s_history:      []f64, // m * n_params (step differences)
	y_history:      []f64, // m * n_params (gradient differences)
	rho_history:    []f64, // m (1 / (y^T s))

	// Working buffers (pre-allocated to avoid allocations in hot loop)
	q:              []f64, // n_params
	d:              []f64, // n_params (search direction)
	alphas:         []f64, // m
	x_new:          []f64, // n_params (for line search)

	// State
	k:              int, // Current iteration
	filled:         int, // How many history slots are filled (up to m)
	prev_x:         []f64, // x_{k-1}
	prev_grad:      []f64, // g_{k-1}
	prev_loss:      f64,
	allocator:      mem.Allocator,
}

// Contract: n_params >= 0, m >= 1, lr > 0
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
	opt.is_initialized = true // ✅ FIX

	if n_params == 0 {return opt}

	// Allocate history buffers
	opt.s_history = make([]f64, m * n_params, allocator)
	opt.y_history = make([]f64, m * n_params, allocator)
	opt.rho_history = make([]f64, m, allocator)

	// Allocate working buffers
	opt.q = make([]f64, n_params, allocator)
	opt.d = make([]f64, n_params, allocator)
	opt.alphas = make([]f64, m, allocator)
	opt.x_new = make([]f64, n_params, allocator)

	// Allocate state buffers
	opt.prev_x = make([]f64, n_params, allocator)
	opt.prev_grad = make([]f64, n_params, allocator)

	return opt
}

// Contract: opt must be initialized via optimizer_lbfgs_init
optimizer_lbfgs_free :: proc(opt: ^OptimizerLBFGS) {
	if !opt.is_initialized {return} 	// ✅ FIX: Safe early exit

	delete(opt.s_history, opt.allocator)
	delete(opt.y_history, opt.allocator)
	delete(opt.rho_history, opt.allocator)
	delete(opt.q, opt.allocator)
	delete(opt.d, opt.allocator)
	delete(opt.alphas, opt.allocator)
	delete(opt.x_new, opt.allocator)
	delete(opt.prev_x, opt.allocator)
	delete(opt.prev_grad, opt.allocator)

	opt.is_initialized = false // Prevent double free
}

// ============================================================================
// L-BFGS Step (Slice API)
// Contract: len(weights) == len(gradient) > 0
// If obj_fn is nil, falls back to fixed step size (learning_rate).
// If obj_fn is provided, performs backtracking line search (Armijo condition).
// ============================================================================
optimizer_lbfgs_step :: proc(
	opt: ^OptimizerLBFGS,
	weights: []f64,
	gradient: []f64,
	loss: f64,
	obj_fn: ObjectiveFunc = nil,
) {
	n := len(weights)
	if n == 0 || n != len(gradient) || !opt.is_initialized {return}

	// 1. Update history (if not the first iteration)
	if opt.k > 0 {
		idx := (opt.k - 1) % opt.m
		s_idx := idx * n
		y_idx := idx * n

		// s_k = x_k - x_{k-1}
		// y_k = g_k - g_{k-1}
		for i in 0 ..< n {
			opt.s_history[s_idx + i] = weights[i] - opt.prev_x[i]
			opt.y_history[y_idx + i] = gradient[i] - opt.prev_grad[i]
		}

		// Compute rho_k = 1 / (y_k^T s_k)
		ys := l.dot_simd(opt.s_history[s_idx:s_idx + n], opt.y_history[y_idx:y_idx + n])

		if ys > 1e-10 {
			opt.rho_history[idx] = 1.0 / ys
			if opt.filled < opt.m {opt.filled += 1}
		} else {
			opt.rho_history[idx] = 0.0 // Skip update if not positive definite
		}
	}

	// 2. Two-loop recursion to compute search direction
	copy(opt.q, gradient)

	// Loop 1: backwards from most recent to oldest
	for i in 0 ..< opt.filled {
		hist_idx := (opt.k - 1 - i) % opt.m
		if hist_idx < 0 {hist_idx += opt.m}

		s_idx := hist_idx * n
		y_idx := hist_idx * n

		rho := opt.rho_history[hist_idx]
		alpha := rho * l.dot_simd(opt.s_history[s_idx:s_idx + n], opt.q)
		opt.alphas[i] = alpha

		// q = q - alpha * y
		l.axpy_simd(-alpha, opt.y_history[y_idx:y_idx + n], opt.q)
	}

	// Initial Hessian approximation: H0 = gamma * I
	if opt.filled > 0 {
		hist_idx := (opt.k - 1) % opt.m
		y_idx := hist_idx * n
		s_idx := hist_idx * n

		ys := l.dot_simd(opt.y_history[y_idx:y_idx + n], opt.s_history[s_idx:s_idx + n])
		yy := l.dot_simd(opt.y_history[y_idx:y_idx + n], opt.y_history[y_idx:y_idx + n])
		gamma := ys / (yy + 1e-10)

		for i in 0 ..< n {opt.q[i] *= gamma}
	}

	// Loop 2: forwards from oldest to most recent
	for i := opt.filled - 1; i >= 0; i -= 1 {
		hist_idx := (opt.k - opt.filled + i) % opt.m
		if hist_idx < 0 {hist_idx += opt.m}

		s_idx := hist_idx * n
		y_idx := hist_idx * n

		rho := opt.rho_history[hist_idx]
		beta := rho * l.dot_simd(opt.y_history[y_idx:y_idx + n], opt.q)

		// q = q + (alphas[i] - beta) * s
		l.axpy_simd(opt.alphas[i] - beta, opt.s_history[s_idx:s_idx + n], opt.q)
	}

	// Search direction d = -q
	for i in 0 ..< n {opt.d[i] = -opt.q[i]}

	// 3. Line Search
	step_size := opt.learning_rate
	dir_deriv := l.dot_simd(gradient, opt.d)

	// If d is not a descent direction, fallback to steepest descent
	if dir_deriv >= 0.0 {
		for i in 0 ..< n {opt.d[i] = -gradient[i]}
		dir_deriv = -l.dot_simd(gradient, gradient)
	}

	if obj_fn != nil {
		// Backtracking line search (Armijo condition)
		c1 := 1e-4
		for _ in 0 ..< 20 { 	// Max 20 backtracks
			// x_new = x + step_size * d
			copy(opt.x_new, weights)
			l.axpy_simd(step_size, opt.d, opt.x_new)

			new_loss := obj_fn(opt.x_new)

			// Armijo condition: f(x_new) <= f(x) + c1 * step_size * dir_deriv
			if new_loss <= loss + c1 * step_size * dir_deriv {
				break // Accept step
			}
			step_size *= 0.5
		}
	}

	// 4. Save state for next iteration
	copy(opt.prev_x, weights)
	copy(opt.prev_grad, gradient)
	opt.prev_loss = loss

	// 5. Apply step to weights: x = x + step_size * d
	l.axpy_simd(step_size, opt.d, weights)

	opt.k += 1
}

// ============================================================================
// L-BFGS Step (Matrix API)
// Contract: weights.rows == gradient.rows, weights.cols == gradient.cols
// Operates on the flat underlying data slice for maximum efficiency.
// ============================================================================
optimizer_lbfgs_step_mat :: proc(
	opt: ^OptimizerLBFGS,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
	loss: f64,
	obj_fn: ObjectiveFunc = nil,
) {
	if weights.rows != gradient.rows || weights.cols != gradient.cols {return}
	if weights.rows == 0 || weights.cols == 0 {return}
	if !opt.is_initialized {return}

	// L-BFGS operates on a flat vector. We pass the underlying data slices directly.
	// Note: The provided obj_fn must expect a flat []f64 representation of the matrix.
	n := weights.rows * weights.cols
	weights_flat := weights.data[0:n]
	gradient_flat := gradient.data[0:n]

	optimizer_lbfgs_step(opt, weights_flat, gradient_flat, loss, obj_fn)
}
