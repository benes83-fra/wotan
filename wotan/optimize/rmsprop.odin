package optimize

import l "../linalg"
import "core:math"
import "core:mem"

// ============================================================================
// RMSProp Optimizer (Slice + Matrix APIs)
// ============================================================================

OptimizerRMSProp :: struct {
	learning_rate: f64,
	alpha:         f64,
	epsilon:       f64,
	weight_decay:  f64,
	cache:         []f64, // Slice buffer
	cache_mat:     l.Matrix(f64), // Matrix buffer
	allocator:     mem.Allocator,
}

optimizer_rmsprop_init :: proc(
	n_params: int,
	lr: f64 = 0.01,
	alpha: f64 = 0.99,
	epsilon: f64 = 1e-8,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerRMSProp {
	opt: OptimizerRMSProp
	opt.learning_rate = lr
	opt.alpha = alpha
	opt.epsilon = epsilon
	opt.weight_decay = weight_decay
	opt.allocator = allocator

	opt.cache = make([]f64, n_params, allocator)
	for i in 0 ..< n_params {opt.cache[i] = 0.0}
	return opt
}

optimizer_rmsprop_init_mat :: proc(
	rows, cols: int,
	lr: f64 = 0.01,
	alpha: f64 = 0.99,
	epsilon: f64 = 1e-8,
	weight_decay: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> OptimizerRMSProp {
	opt: OptimizerRMSProp
	opt.learning_rate = lr
	opt.alpha = alpha
	opt.epsilon = epsilon
	opt.weight_decay = weight_decay
	opt.allocator = allocator

	opt.cache_mat = l.matrix_new(f64, rows, cols, allocator)
	for i in 0 ..< rows * cols {opt.cache_mat.data[i] = 0.0}
	return opt
}

optimizer_rmsprop_free :: proc(opt: ^OptimizerRMSProp) {
	if len(opt.cache) > 0 {delete(opt.cache, opt.allocator)}
	if opt.cache_mat.data != nil {l.matrix_free(&opt.cache_mat)}
}

// One RMSprop step on a block of parameters with its square average, as
// torch.optim.RMSprop (no momentum, not centered): weight decay is an L2 term
// of the gradient (as torch, not decoupled), d = g + λw;
// v ← αv + (1 − α)d², w ← w − lr·d / (√v + ε).
rmsprop_update :: proc(w, g, v: []f64, lr, alpha, epsilon, weight_decay: f64) {
	for i in 0 ..< len(w) {
		d := g[i] + weight_decay * w[i]
		v[i] = alpha * v[i] + (1.0 - alpha) * d * d
		w[i] -= lr * d / (math.sqrt(v[i]) + epsilon)
	}
}

// ============================================================================
// Slice API
// ============================================================================
optimizer_rmsprop_step :: proc(opt: ^OptimizerRMSProp, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}
	rmsprop_update(
		weights,
		gradient,
		opt.cache,
		opt.learning_rate,
		opt.alpha,
		opt.epsilon,
		opt.weight_decay,
	)
}

// ============================================================================
// Matrix API
// ============================================================================
optimizer_rmsprop_step_mat :: proc(
	opt: ^OptimizerRMSProp,
	weights: ^l.Matrix(f64),
	gradient: ^l.Matrix(f64),
) {
	if weights.rows != gradient.rows || weights.cols != gradient.cols {return}
	if weights.rows == 0 || weights.cols == 0 {return}
	rmsprop_update(
		weights.data,
		gradient.data,
		opt.cache_mat.data,
		opt.learning_rate,
		opt.alpha,
		opt.epsilon,
		opt.weight_decay,
	)
}
