package nn

import l "../linalg"
import t "../tensor"
import "core:mem"

// ============================================================================
// 1. Stochastic Gradient Descent (SGD) Optimizer
// ============================================================================

SGD :: struct {
	parameters:    [dynamic]^t.Tensor, // Pointers to weights and biases
	learning_rate: f64,
	allocator:     mem.Allocator,
}

// sgd_new creates a new SGD optimizer
sgd_new :: proc(learning_rate: f64, allocator: mem.Allocator = context.allocator) -> SGD {
	return SGD {
		parameters = make([dynamic]^t.Tensor, 0, allocator),
		learning_rate = learning_rate,
		allocator = allocator,
	}
}

// sgd_add_param registers a tensor (like weights or bias) to be optimized
sgd_add_param :: proc(opt: ^SGD, param: ^t.Tensor) {
	append(&opt.parameters, param)
}

// sgd_step performs one step of gradient descent: W = W - lr * grad
sgd_step :: proc(opt: ^SGD) {
	for param in opt.parameters {
		if param.requires_grad && param.grad.data != nil {
			// ✅ SIMD Optimization:
			// axpy does: y += alpha * x
			// We want: param.data += (-learning_rate) * param.grad
			l.axpy_simd(-opt.learning_rate, param.grad.data, param.data.data)
		}
	}
}

// sgd_zero_grad resets all gradients to 0.0 (required before the next backward pass)
sgd_zero_grad :: proc(opt: ^SGD) {
	for param in opt.parameters {
		t.tensor_zero_grad(param)
	}
}

// sgd_free cleans up the optimizer's internal list
sgd_free :: proc(opt: ^SGD) {
	delete(opt.parameters)
}
