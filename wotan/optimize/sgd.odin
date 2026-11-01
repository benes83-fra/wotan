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

// One SGD step on a block of parameters, as torch.optim.SGD (dampening 0):
// d = g + λw; with momentum μ > 0, v ← μv + d and d = v (Nesterov: d + μv);
// w ← w − lr·d. `velocity` may be nil when μ = 0. Shared with nn.SGD.
sgd_update :: proc(
	w, g, velocity: []f64,
	lr, momentum, weight_decay: f64,
	nesterov: bool = false,
) {
	for i in 0 ..< len(w) {
		d := g[i] + weight_decay * w[i]
		if momentum > 0.0 {
			velocity[i] = momentum * velocity[i] + d
			d = d + momentum * velocity[i] if nesterov else velocity[i]
		}
		w[i] -= lr * d
	}
}

// ============================================================================
// Slice API: Single parameter vector update
// Contract: len(weights) == len(gradient) > 0
// ============================================================================
optimizer_sgd_step :: proc(opt: ^OptimizerSGD, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}
	sgd_update(weights, gradient, opt.velocity, opt.learning_rate, opt.momentum, opt.weight_decay)
}

// ============================================================================
// Matrix API: Batched parameter matrix update
// Contract: weights.rows == gradient.rows, weights.cols == gradient.cols
// ============================================================================
optimizer_sgd_step_mat :: proc(
	opt: ^OptimizerSGD,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
) {
	if weights.rows != gradient.rows || weights.cols != gradient.cols {return}
	if weights.rows == 0 || weights.cols == 0 {return}
	sgd_update(weights.data, gradient.data, opt.velocity_mat.data, opt.learning_rate, opt.momentum, opt.weight_decay)
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
	v_row: []f64
	if opt.momentum > 0.0 {v_row = opt.velocity_mat.data[row * n:row * n + n]}
	sgd_update(w_row, g_row, v_row, opt.learning_rate, opt.momentum, opt.weight_decay)
}
