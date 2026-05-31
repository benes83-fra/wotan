package optimize

import l "../linalg"
import "core:mem"

// ============================================================================
// SGD Optimizer (SIMD-Accelerated, Slice + Matrix APIs)
// ============================================================================

OptimizerSGD :: struct {
	learning_rate: f64,
	momentum:      f64,
	weight_decay:  f64,
	velocity:      []f64, // Momentum buffer (for slice API)
	velocity_mat:  l.Matrix(f64), // Momentum buffer (for matrix API)
	allocator:     mem.Allocator,
}

// Contract: n_params >= 0, lr > 0, momentum in [0,1), weight_decay >= 0
// Initializes optimizer for slice-based updates
optimizer_sgd_init :: proc(
	n_params: int,
	lr: f64,
	momentum: f64 = 0.0,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerSGD {
	opt: OptimizerSGD
	opt.learning_rate = lr
	opt.momentum = momentum
	opt.weight_decay = weight_decay
	opt.allocator = allocator

	if momentum > 0.0 {
		opt.velocity = make([]f64, n_params, allocator)
		for i in 0 ..< n_params {opt.velocity[i] = 0.0}
	}
	return opt
}

// Contract: rows, cols >= 0, lr > 0, momentum in [0,1)
// Initializes optimizer for matrix-based (batched) updates
optimizer_sgd_init_mat :: proc(
	rows, cols: int,
	lr: f64,
	momentum: f64 = 0.0,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerSGD {
	opt: OptimizerSGD
	opt.learning_rate = lr
	opt.momentum = momentum
	opt.weight_decay = weight_decay
	opt.allocator = allocator

	if momentum > 0.0 {
		opt.velocity_mat = l.matrix_new(f64, rows, cols, allocator)
		for i in 0 ..< rows * cols {opt.velocity_mat.data[i] = 0.0}
	}
	return opt
}

// Contract: opt initialized via optimizer_sgd_init
optimizer_sgd_free :: proc(opt: ^OptimizerSGD) {
	if len(opt.velocity) > 0 {
		delete(opt.velocity, opt.allocator)
	}
	if opt.velocity_mat.data != nil {
		l.matrix_free(&opt.velocity_mat)
	}
}

// ============================================================================
// Slice API: Single parameter vector update
// Contract: len(weights) == len(gradient) > 0
// Updates: weights -= lr * (gradient + weight_decay * weights)
// If momentum > 0: uses velocity buffer
// ============================================================================
optimizer_sgd_step :: proc(opt: ^OptimizerSGD, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}

	if opt.momentum > 0.0 {
		// Momentum: v = μ*v + g + λ*w; then w -= lr * v
		for i in 0 ..< n {
			opt.velocity[i] =
				opt.momentum * opt.velocity[i] + gradient[i] + opt.weight_decay * weights[i]
		}
		// SIMD: weights -= lr * velocity
		l.axpy_simd(-opt.learning_rate, opt.velocity, weights)
	} else {
		if opt.weight_decay > 0.0 {
			// weights = (1 - lr*λ)*weights - lr*gradient
			decay_factor := 1.0 - opt.learning_rate * opt.weight_decay
			for i in 0 ..< n {weights[i] *= decay_factor}
			l.axpy_simd(-opt.learning_rate, gradient, weights)
		} else {
			// Simple: weights -= lr * gradient (SIMD)
			l.axpy_simd(-opt.learning_rate, gradient, weights)
		}
	}
}

// ============================================================================
// Matrix API: Batched parameter matrix update (rows = batch, cols = params)
// Contract: weights.rows == gradient.rows, weights.cols == gradient.cols
// Updates each row independently: weights[i] -= lr * (gradient[i] + λ*weights[i])
// If momentum > 0: uses velocity_mat buffer
// ============================================================================
optimizer_sgd_step_mat :: proc(
	opt: ^OptimizerSGD,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
) {
	if weights.rows != gradient.rows || weights.cols != gradient.cols {return}
	if weights.rows == 0 || weights.cols == 0 {return}

	n := weights.rows * weights.cols

	if opt.momentum > 0.0 {
		// Momentum update per element (SIMD via axpy on flattened view)
		// v = μ*v + g + λ*w  →  v = μ*v + (g + λ*w)
		if opt.weight_decay > 0.0 {
			// First: temp = g + λ*w (SIMD)
			temp := make([]f64, n, context.temp_allocator)
			defer delete(temp, context.temp_allocator)
			for i in 0 ..< n {temp[i] = gradient.data[i] + opt.weight_decay * weights.data[i]}
			// Then: v = μ*v + temp (SIMD)
			l.axpy_simd(opt.momentum, opt.velocity_mat.data, opt.velocity_mat.data) // v *= μ
			l.axpy_simd(1.0, temp, opt.velocity_mat.data) // v += temp
		} else {
			// v = μ*v + g (SIMD)
			l.axpy_simd(opt.momentum, opt.velocity_mat.data, opt.velocity_mat.data)
			l.axpy_simd(1.0, gradient.data, opt.velocity_mat.data)
		}
		// weights -= lr * v (SIMD)
		l.axpy_simd(-opt.learning_rate, opt.velocity_mat.data, weights.data)
	} else {
		if opt.weight_decay > 0.0 {
			// weights = (1 - lr*λ)*weights - lr*gradient
			decay_factor := 1.0 - opt.learning_rate * opt.weight_decay
			for i in 0 ..< n {weights.data[i] *= decay_factor}
			l.axpy_simd(-opt.learning_rate, gradient.data, weights.data)
		} else {
			// weights -= lr * gradient (SIMD)
			l.axpy_simd(-opt.learning_rate, gradient.data, weights.data)
		}
	}
}

// ============================================================================
// Convenience: Row-wise matrix update (for per-sample gradients)
// Contract: weights.cols == gradient.cols, row < weights.rows
// Updates single row: weights[row] -= lr * (gradient[row] + λ*weights[row])
// ============================================================================
optimizer_sgd_step_row :: proc(
	opt: ^OptimizerSGD,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
	row: int,
) {
	if row < 0 || row >= weights.rows || row >= gradient.rows {return}
	if weights.cols != gradient.cols {return}

	n := weights.cols
	if n == 0 {return}

	w_row := weights.data[row * n:row * n + n]
	g_row := gradient.data[row * n:row * n + n]

	if opt.momentum > 0.0 {
		v_row := opt.velocity_mat.data[row * n:row * n + n]
		for i in 0 ..< n {
			v_row[i] = opt.momentum * v_row[i] + g_row[i] + opt.weight_decay * w_row[i]
		}
		l.axpy_simd(-opt.learning_rate, v_row, w_row)
	} else {
		if opt.weight_decay > 0.0 {
			decay_factor := 1.0 - opt.learning_rate * opt.weight_decay
			for i in 0 ..< n {w_row[i] *= decay_factor}
			l.axpy_simd(-opt.learning_rate, g_row, w_row)
		} else {
			l.axpy_simd(-opt.learning_rate, g_row, w_row)
		}
	}
}
