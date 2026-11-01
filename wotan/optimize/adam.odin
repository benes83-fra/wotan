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

// One Adam step on a block of parameters with its moment buffers (t ≥ 1), as
// torch.optim.Adam / AdamW. Weight decay is decoupled (AdamW, Loshchilov &
// Hutter): w ← (1 − lr·λ) w before the update. Shared with nn.Adam.
adam_update :: proc(w, g, m, v: []f64, lr, beta1, beta2, epsilon, weight_decay: f64, t: int) {
	step := lr / (1.0 - math.pow(beta1, f64(t)))
	sqrt_bc2 := math.sqrt(1.0 - math.pow(beta2, f64(t)))
	decay := 1.0 - lr * weight_decay
	for i in 0 ..< len(w) {
		gi := g[i]
		m[i] = beta1 * m[i] + (1.0 - beta1) * gi
		v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi
		w[i] = w[i] * decay - step * m[i] / (math.sqrt(v[i]) / sqrt_bc2 + epsilon)
	}
}

// ============================================================================
// Slice API
// ============================================================================
optimizer_adam_step :: proc(opt: ^OptimizerAdam, weights: []f64, gradient: []f64) {
	n := len(weights)
	if n == 0 || n != len(gradient) {return}
	opt.t += 1
	adam_update(weights, gradient, opt.m, opt.v, opt.learning_rate, opt.beta1, opt.beta2, opt.epsilon, opt.weight_decay, opt.t)
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
	adam_update(weights.data, gradient.data, opt.m_mat.data, opt.v_mat.data, opt.learning_rate, opt.beta1, opt.beta2, opt.epsilon, opt.weight_decay, opt.t)
}
