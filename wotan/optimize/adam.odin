package optimize

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// Adam Optimizer (Adaptive Moments, Slice + Matrix APIs)
// ============================================================================

OptimizerAdam :: struct {
	learning_rate: f64,
	beta1:         f64,
	beta2:         f64,
	epsilon:       f64,
	weight_decay:  f64,
	t:             int,
	// Slice buffers
	m:             []f64,
	v:             []f64,
	// Matrix buffers
	m_mat:         l.Matrix(f64),
	v_mat:         l.Matrix(f64),
	allocator:     mem.Allocator,
}

// Contract: n_params >= 0, lr > 0, beta1/beta2 in [0,1), epsilon > 0
optimizer_adam_init :: proc(
	n_params: int,
	lr: f64 = 0.001,
	beta1: f64 = 0.9,
	beta2: f64 = 0.999,
	epsilon: f64 = 1e-8,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerAdam {
	opt: OptimizerAdam
	opt.learning_rate = lr
	opt.beta1 = beta1
	opt.beta2 = beta2
	opt.epsilon = epsilon
	opt.weight_decay = weight_decay
	opt.t = 0
	opt.allocator = allocator

	opt.m = make([]f64, n_params, allocator)
	opt.v = make([]f64, n_params, allocator)
	for i in 0 ..< n_params {
		opt.m[i] = 0.0
		opt.v[i] = 0.0
	}
	return opt
}

// Contract: rows, cols >= 0, lr > 0, beta1/beta2 in [0,1)
optimizer_adam_init_mat :: proc(
	rows, cols: int,
	lr: f64 = 0.001,
	beta1: f64 = 0.9,
	beta2: f64 = 0.999,
	epsilon: f64 = 1e-8,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerAdam {
	opt: OptimizerAdam
	opt.learning_rate = lr
	opt.beta1 = beta1
	opt.beta2 = beta2
	opt.epsilon = epsilon
	opt.weight_decay = weight_decay
	opt.t = 0
	opt.allocator = allocator

	opt.m_mat = l.matrix_new(f64, rows, cols, allocator)
	opt.v_mat = l.matrix_new(f64, rows, cols, allocator)
	for i in 0 ..< rows * cols {
		opt.m_mat.data[i] = 0.0
		opt.v_mat.data[i] = 0.0
	}
	return opt
}

optimizer_adam_free :: proc(opt: ^OptimizerAdam) {
	if len(opt.m) > 0 {delete(opt.m, opt.allocator)}
	if len(opt.v) > 0 {delete(opt.v, opt.allocator)}
	if opt.m_mat.data != nil {l.matrix_free(&opt.m_mat)}
	if opt.v_mat.data != nil {l.matrix_free(&opt.v_mat)}
}

// ============================================================================
// Slice API
// ============================================================================
optimizer_adam_step :: proc(opt: ^OptimizerAdam, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}

	opt.t += 1

	// AdamW-style weight decay
	if opt.weight_decay > 0.0 {
		for i in 0 ..< n {
			weights[i] *= 1.0 - opt.learning_rate * opt.weight_decay
		}
	}

	// Update moments
	for i in 0 ..< n {
		opt.m[i] = opt.beta1 * opt.m[i] + (1.0 - opt.beta1) * gradient[i]
		opt.v[i] = opt.beta2 * opt.v[i] + (1.0 - opt.beta2) * gradient[i] * gradient[i]
	}

	// Bias correction
	m_hat_scale := 1.0 / (1.0 - math.pow(opt.beta1, f64(opt.t)))
	v_hat_scale := 1.0 / (1.0 - math.pow(opt.beta2, f64(opt.t)))

	// Update weights
	for i in 0 ..< n {
		m_hat := opt.m[i] * m_hat_scale
		v_hat := opt.v[i] * v_hat_scale
		weights[i] -= opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)
	}
}

// ============================================================================
// Matrix API (batched)
// ============================================================================
optimizer_adam_step_mat :: proc(
	opt: ^OptimizerAdam,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
) {
	if weights.rows != gradient.rows || weights.cols != gradient.cols {return}
	if weights.rows == 0 || weights.cols == 0 {return}

	opt.t += 1
	n := weights.rows * weights.cols

	// AdamW weight decay
	if opt.weight_decay > 0.0 {
		for i in 0 ..< n {
			weights.data[i] *= 1.0 - opt.learning_rate * opt.weight_decay
		}
	}

	// Update moments (element-wise, SIMD-ready via loops)
	for i in 0 ..< n {
		opt.m_mat.data[i] = opt.beta1 * opt.m_mat.data[i] + (1.0 - opt.beta1) * gradient.data[i]
		opt.v_mat.data[i] =
			opt.beta2 * opt.v_mat.data[i] + (1.0 - opt.beta2) * gradient.data[i] * gradient.data[i]
	}

	// Bias correction
	m_hat_scale := 1.0 / (1.0 - math.pow(opt.beta1, f64(opt.t)))
	v_hat_scale := 1.0 / (1.0 - math.pow(opt.beta2, f64(opt.t)))

	// Update weights
	for i in 0 ..< n {
		m_hat := opt.m_mat.data[i] * m_hat_scale
		v_hat := opt.v_mat.data[i] * v_hat_scale
		weights.data[i] -= opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)
	}
}
