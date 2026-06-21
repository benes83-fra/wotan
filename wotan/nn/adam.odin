package nn

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// 1. Adam Optimizer
// ============================================================================

Adam :: struct {
	parameters:     [dynamic]^t.Tensor,
	moment_1:       [dynamic][]f64,
	moment_2:       [dynamic][]f64,
	learning_rate:  f64,
	beta_1:         f64,
	beta_2:         f64,
	epsilon:        f64,
	timestep:       int,
	allocator:      mem.Allocator,
	grad_sq:        []f64, // ✅ Pre-allocated temp buffer
	max_param_size: int, // Track max size needed
}

adam_new :: proc(
	learning_rate: f64 = 0.001,
	beta_1: f64 = 0.9,
	beta_2: f64 = 0.999,
	epsilon: f64 = 1e-8,
	allocator: mem.Allocator = context.allocator,
) -> Adam {
	return Adam {
		parameters     = make([dynamic]^t.Tensor, 0, allocator),
		moment_1       = make([dynamic][]f64, 0, allocator),
		moment_2       = make([dynamic][]f64, 0, allocator),
		learning_rate  = learning_rate,
		beta_1         = beta_1,
		beta_2         = beta_2,
		epsilon        = epsilon,
		timestep       = 0,
		allocator      = allocator,
		grad_sq        = nil, // Will be allocated on first use
		max_param_size = 0,
	}
}

adam_add_param :: proc(opt: ^Adam, param: ^t.Tensor) {
	append(&opt.parameters, param)

	n := len(param.data.data)
	m := make([]f64, n, opt.allocator)
	v := make([]f64, n, opt.allocator)

	append(&opt.moment_1, m)
	append(&opt.moment_2, v)

	// Track max parameter size for temp buffer
	if n > opt.max_param_size {
		opt.max_param_size = n
	}
}

adam_step :: proc(opt: ^Adam) {
	opt.timestep += 1
	t := f64(opt.timestep)

	// ✅ Allocate temp buffer once, reuse for all parameters
	if opt.grad_sq == nil && opt.max_param_size > 0 {
		opt.grad_sq = make([]f64, opt.max_param_size, opt.allocator)
	}

	for i in 0 ..< len(opt.parameters) {
		param := opt.parameters[i]
		if !param.requires_grad || param.grad.data == nil {
			continue
		}

		m := opt.moment_1[i]
		v := opt.moment_2[i]
		grad := param.grad.data
		data := param.data.data
		n := len(data)

		// Update biased first moment: m = beta_1 * m + (1 - beta_1) * grad
		for j in 0 ..< n {
			m[j] *= opt.beta_1
		}
		l.axpy_simd(1.0 - opt.beta_1, grad, m)

		// ✅ Reuse pre-allocated grad_sq buffer
		l.vec_mul_simd(grad, grad, opt.grad_sq[:n])

		// Update biased second moment: v = beta_2 * v + (1 - beta_2) * grad^2
		for j in 0 ..< n {
			v[j] *= opt.beta_2
		}
		l.axpy_simd(1.0 - opt.beta_2, opt.grad_sq[:n], v)

		// Bias correction
		bias_correction_1 := 1.0 - math.pow(opt.beta_1, t)
		bias_correction_2 := 1.0 - math.pow(opt.beta_2, t)

		m_hat_scale := 1.0 / bias_correction_1
		v_hat_scale := 1.0 / bias_correction_2

		// Update parameters
		for j in 0 ..< n {
			m_hat := m[j] * m_hat_scale
			v_hat := v[j] * v_hat_scale
			data[j] -= opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)
		}
	}
}

adam_free :: proc(opt: ^Adam) {
	for m in opt.moment_1 {
		delete(m, opt.allocator)
	}
	for v in opt.moment_2 {
		delete(v, opt.allocator)
	}
	if opt.grad_sq != nil {
		delete(opt.grad_sq, opt.allocator) // ✅ Free temp buffer
	}
	delete(opt.parameters)
	delete(opt.moment_1)
	delete(opt.moment_2)
}

adam_zero_grad :: proc(opt: ^Adam) {
	for param in opt.parameters {
		t.tensor_zero_grad(param)
	}
}
