package nn

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// 1. Adam Optimizer
// ============================================================================

Adam :: struct {
	parameters:    [dynamic]^t.Tensor,
	moment_1:      [dynamic][]f64, // First moment (momentum)
	moment_2:      [dynamic][]f64, // Second moment (RMSprop)
	learning_rate: f64,
	beta_1:        f64, // Exponential decay rate for moment_1 (typically 0.9)
	beta_2:        f64, // Exponential decay rate for moment_2 (typically 0.999)
	epsilon:       f64, // Small constant for numerical stability (typically 1e-8)
	timestep:      int, // Current timestep for bias correction
	allocator:     mem.Allocator,
}

// adam_new creates a new Adam optimizer with default hyperparameters
adam_new :: proc(
	learning_rate: f64 = 0.001,
	beta_1: f64 = 0.9,
	beta_2: f64 = 0.999,
	epsilon: f64 = 1e-8,
	allocator: mem.Allocator = context.allocator,
) -> Adam {
	return Adam {
		parameters = make([dynamic]^t.Tensor, 0, allocator),
		moment_1 = make([dynamic][]f64, 0, allocator),
		moment_2 = make([dynamic][]f64, 0, allocator),
		learning_rate = learning_rate,
		beta_1 = beta_1,
		beta_2 = beta_2,
		epsilon = epsilon,
		timestep = 0,
		allocator = allocator,
	}
}

// adam_add_param registers a tensor (like weights or bias) to be optimized
adam_add_param :: proc(opt: ^Adam, param: ^t.Tensor) {
	append(&opt.parameters, param)

	// Initialize moment vectors to zeros
	n := len(param.data.data)
	m := make([]f64, n, opt.allocator)
	v := make([]f64, n, opt.allocator)

	append(&opt.moment_1, m)
	append(&opt.moment_2, v)
}

// adam_step performs one step of Adam optimization
adam_step :: proc(opt: ^Adam) {
	opt.timestep += 1
	t := f64(opt.timestep)

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

		// Update biased first moment estimate: m = beta_1 * m + (1 - beta_1) * grad
		// Using SIMD: m = beta_1 * m + (1 - beta_1) * grad
		// We can use axpy_simd for this: m = m + (1-beta_1) * grad, then scale by beta_1
		// Actually, let's do it element-wise for clarity, or use a combination of operations

		// Scale existing m by beta_1
		for j in 0 ..< n {
			m[j] *= opt.beta_1
		}
		// Add (1 - beta_1) * grad
		l.axpy_simd(1.0 - opt.beta_1, grad, m)

		// Update biased second moment estimate: v = beta_2 * v + (1 - beta_2) * grad^2
		// First compute grad^2 into a temp buffer
		grad_sq := make([]f64, n, context.temp_allocator)
		l.vec_mul_simd(grad, grad, grad_sq)

		// Scale existing v by beta_2
		for j in 0 ..< n {
			v[j] *= opt.beta_2
		}
		// Add (1 - beta_2) * grad^2
		l.axpy_simd(1.0 - opt.beta_2, grad_sq, v)
		delete(grad_sq, context.temp_allocator)

		// Bias correction
		bias_correction_1 := 1.0 - math.pow(opt.beta_1, t)
		bias_correction_2 := 1.0 - math.pow(opt.beta_2, t)

		m_hat_scale := 1.0 / bias_correction_1
		v_hat_scale := 1.0 / bias_correction_2

		// Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
		// We need to compute this element-wise
		for j in 0 ..< n {
			m_hat := m[j] * m_hat_scale
			v_hat := v[j] * v_hat_scale
			data[j] -= opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)
		}
	}
}

// adam_zero_grad resets all gradients to 0.0
adam_zero_grad :: proc(opt: ^Adam) {
	for param in opt.parameters {
		t.tensor_zero_grad(param)
	}
}

// adam_free cleans up the optimizer's internal state
adam_free :: proc(opt: ^Adam) {
	for m in opt.moment_1 {
		delete(m, opt.allocator)
	}
	for v in opt.moment_2 {
		delete(v, opt.allocator)
	}
	delete(opt.parameters)
	delete(opt.moment_1)
	delete(opt.moment_2)
}
