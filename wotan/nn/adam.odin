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
	learning_rates: [dynamic]f64,
	beta_1:         f64,
	beta_2:         f64,
	epsilon:        f64,
	timestep:       int,
	allocator:      mem.Allocator,
	grad_sq:        []f64, // ✅ Pre-allocated temp buffer
	max_param_size: int, // Track max size needed
}


adam_add_param_with_lr :: proc(opt: ^Adam, param: ^t.Tensor, lr: f64) {
	append(&opt.parameters, param)
	append(&opt.learning_rates, lr) // Custom LR for this param

	n := len(param.data.data)
	m := make([]f64, n, opt.allocator)
	v := make([]f64, n, opt.allocator)

	append(&opt.moment_1, m)
	append(&opt.moment_2, v)

	if n > opt.max_param_size {
		opt.max_param_size = n
	}
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
		lr := opt.learning_rate
		if i < len(opt.learning_rates) {
			lr = opt.learning_rates[i]
		}
		grad_norm := 0.0
		for j in 0 ..< len(param.grad.data) {
			grad_norm += param.grad.data[j] * param.grad.data[j]
		}
		grad_norm = math.sqrt(grad_norm)

		max_grad_norm := 1.0
		if grad_norm > max_grad_norm {
			scale := max_grad_norm / grad_norm
			for j in 0 ..< len(param.grad.data) {
				param.grad.data[j] *= scale
			}
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
			data[j] -= lr * m_hat / (math.sqrt(v_hat) + opt.epsilon)
		}
	}
}

adam_free :: proc(opt: ^Adam) {
	// Free all parameter moment arrays
	for i in 0 ..< len(opt.moment_1) {
		delete(opt.moment_1[i], opt.allocator)
	}
	for i in 0 ..< len(opt.moment_2) {
		delete(opt.moment_2[i], opt.allocator)
	}

	// ✅ FIX: Free the grad_sq buffer
	if opt.grad_sq != nil {
		delete(opt.grad_sq, opt.allocator)
		opt.grad_sq = nil
	}

	// ✅ FIX: Free the learning_rates array
	if len(opt.learning_rates) > 0 {
		delete(opt.learning_rates)
	}

	// ✅ FIX: Free the parameters array
	if len(opt.parameters) > 0 {
		delete(opt.parameters)
	}

	// Free the moment arrays themselves
	delete(opt.moment_1)
	delete(opt.moment_2)
}

adam_zero_grad :: proc(opt: ^Adam) {
	for param in opt.parameters {
		t.tensor_zero_grad(param)
	}
}
// Helper: Add only trainable parameters
// Helper: Add only trainable parameters
sequential_add_trainable_to_adam :: proc(seq: ^Sequential, opt: ^Adam) {
	for layer in seq.layers {
		switch l in layer {
		case LinearLayer:
			if l.weights.requires_grad {adam_add_param(opt, l.weights)}
			if l.bias != nil && l.bias.requires_grad {adam_add_param(opt, l.bias)}
		case Conv2dLayer:
			if l.weight.requires_grad {adam_add_param(opt, l.weight)}
			if l.bias != nil && l.bias.requires_grad {adam_add_param(opt, l.bias)}
		case BatchNorm2dLayer:
			if l.weight.requires_grad {adam_add_param(opt, l.weight)}
			if l.bias.requires_grad {adam_add_param(opt, l.bias)}
		case LayerNormLayer:
			if l.gamma.requires_grad {adam_add_param(opt, l.gamma)}
			if l.beta.requires_grad {adam_add_param(opt, l.beta)}
		case EmbeddingLayer:
			if l.weight.requires_grad {adam_add_param(opt, l.weight)}
		case MultiHeadAttentionLayer:
			if l.q_proj.weights.requires_grad {adam_add_param(opt, l.q_proj.weights)}
			if l.q_proj.bias != nil &&
			   l.q_proj.bias.requires_grad {adam_add_param(opt, l.q_proj.bias)}
			if l.k_proj.weights.requires_grad {adam_add_param(opt, l.k_proj.weights)}
			if l.k_proj.bias != nil &&
			   l.k_proj.bias.requires_grad {adam_add_param(opt, l.k_proj.bias)}
			if l.v_proj.weights.requires_grad {adam_add_param(opt, l.v_proj.weights)}
			if l.v_proj.bias != nil &&
			   l.v_proj.bias.requires_grad {adam_add_param(opt, l.v_proj.bias)}
			if l.out_proj.weights.requires_grad {adam_add_param(opt, l.out_proj.weights)}
			if l.out_proj.bias != nil &&
			   l.out_proj.bias.requires_grad {adam_add_param(opt, l.out_proj.bias)}
		case FFNLayer:
			if l.fc1.weights.requires_grad {adam_add_param(opt, l.fc1.weights)}
			if l.fc1.bias != nil && l.fc1.bias.requires_grad {adam_add_param(opt, l.fc1.bias)}
			if l.fc2.weights.requires_grad {adam_add_param(opt, l.fc2.weights)}
			if l.fc2.bias != nil && l.fc2.bias.requires_grad {adam_add_param(opt, l.fc2.bias)}
		case RNNLayer:
			if l.w_ih.requires_grad {adam_add_param(opt, l.w_ih)}
			if l.w_hh.requires_grad {adam_add_param(opt, l.w_hh)}
			if l.bias.requires_grad {adam_add_param(opt, l.bias)}
		case GRULayer:
			if l.w_ih.requires_grad {adam_add_param(opt, l.w_ih)}
			if l.w_hh.requires_grad {adam_add_param(opt, l.w_hh)}
			if l.bias.requires_grad {adam_add_param(opt, l.bias)}
		case LSTMLayer:
			if l.w_ih.requires_grad {adam_add_param(opt, l.w_ih)}
			if l.w_hh.requires_grad {adam_add_param(opt, l.w_hh)}
			if l.bias.requires_grad {adam_add_param(opt, l.bias)}
		case TransformerEncoderBlock:
			if l.ln1.gamma.requires_grad {adam_add_param(opt, l.ln1.gamma)}
			if l.ln1.beta.requires_grad {adam_add_param(opt, l.ln1.beta)}
			if l.mha.q_proj.weights.requires_grad {adam_add_param(opt, l.mha.q_proj.weights)}
			if l.mha.q_proj.bias != nil &&
			   l.mha.q_proj.bias.requires_grad {adam_add_param(opt, l.mha.q_proj.bias)}
			if l.mha.k_proj.weights.requires_grad {adam_add_param(opt, l.mha.k_proj.weights)}
			if l.mha.k_proj.bias != nil &&
			   l.mha.k_proj.bias.requires_grad {adam_add_param(opt, l.mha.k_proj.bias)}
			if l.mha.v_proj.weights.requires_grad {adam_add_param(opt, l.mha.v_proj.weights)}
			if l.mha.v_proj.bias != nil &&
			   l.mha.v_proj.bias.requires_grad {adam_add_param(opt, l.mha.v_proj.bias)}
			if l.mha.out_proj.weights.requires_grad {adam_add_param(opt, l.mha.out_proj.weights)}
			if l.mha.out_proj.bias != nil &&
			   l.mha.out_proj.bias.requires_grad {adam_add_param(opt, l.mha.out_proj.bias)}
			if l.ln2.gamma.requires_grad {adam_add_param(opt, l.ln2.gamma)}
			if l.ln2.beta.requires_grad {adam_add_param(opt, l.ln2.beta)}
			if l.ffn.fc1.weights.requires_grad {adam_add_param(opt, l.ffn.fc1.weights)}
			if l.ffn.fc1.bias != nil &&
			   l.ffn.fc1.bias.requires_grad {adam_add_param(opt, l.ffn.fc1.bias)}
			if l.ffn.fc2.weights.requires_grad {adam_add_param(opt, l.ffn.fc2.weights)}
			if l.ffn.fc2.bias != nil &&
			   l.ffn.fc2.bias.requires_grad {adam_add_param(opt, l.ffn.fc2.bias)}
		case TransformerEncoder:
			for i in 0 ..< len(l.blocks) {
				block := &l.blocks[i]
				if block.ln1.gamma.requires_grad {adam_add_param(opt, block.ln1.gamma)}
				if block.ln1.beta.requires_grad {adam_add_param(opt, block.ln1.beta)}
				if block.mha.q_proj.weights.requires_grad {adam_add_param(opt, block.mha.q_proj.weights)}
				if block.mha.q_proj.bias != nil &&
				   block.mha.q_proj.bias.requires_grad {adam_add_param(opt, block.mha.q_proj.bias)}
				if block.mha.k_proj.weights.requires_grad {adam_add_param(opt, block.mha.k_proj.weights)}
				if block.mha.k_proj.bias != nil &&
				   block.mha.k_proj.bias.requires_grad {adam_add_param(opt, block.mha.k_proj.bias)}
				if block.mha.v_proj.weights.requires_grad {adam_add_param(opt, block.mha.v_proj.weights)}
				if block.mha.v_proj.bias != nil &&
				   block.mha.v_proj.bias.requires_grad {adam_add_param(opt, block.mha.v_proj.bias)}
				if block.mha.out_proj.weights.requires_grad {adam_add_param(opt, block.mha.out_proj.weights)}
				if block.mha.out_proj.bias != nil &&
				   block.mha.out_proj.bias.requires_grad {adam_add_param(opt, block.mha.out_proj.bias)}
				if block.ln2.gamma.requires_grad {adam_add_param(opt, block.ln2.gamma)}
				if block.ln2.beta.requires_grad {adam_add_param(opt, block.ln2.beta)}
				if block.ffn.fc1.weights.requires_grad {adam_add_param(opt, block.ffn.fc1.weights)}
				if block.ffn.fc1.bias != nil &&
				   block.ffn.fc1.bias.requires_grad {adam_add_param(opt, block.ffn.fc1.bias)}
				if block.ffn.fc2.weights.requires_grad {adam_add_param(opt, block.ffn.fc2.weights)}
				if block.ffn.fc2.bias != nil &&
				   block.ffn.fc2.bias.requires_grad {adam_add_param(opt, block.ffn.fc2.bias)}
			}
		case GATLayer:
			if l.linear.weights.requires_grad {adam_add_param(opt, l.linear.weights)}
			if l.linear.bias != nil &&
			   l.linear.bias.requires_grad {adam_add_param(opt, l.linear.bias)}
			if l.mha.q_proj.weights.requires_grad {adam_add_param(opt, l.mha.q_proj.weights)}
			if l.mha.k_proj.weights.requires_grad {adam_add_param(opt, l.mha.k_proj.weights)}
			if l.mha.v_proj.weights.requires_grad {adam_add_param(opt, l.mha.v_proj.weights)}
			if l.mha.out_proj.weights.requires_grad {adam_add_param(opt, l.mha.out_proj.weights)}
		case MaxPool2dLayer, AvgPool2dLayer, DropoutLayer, Activation, FlattenLayer:
		// No trainable parameters
		}
	}
}
// clip_grad_norm clips the global norm of all gradients in the optimizer.
// If the total norm exceeds max_norm, all gradients are scaled down so that
// the total norm equals max_norm.
clip_grad_norm :: proc(opt: ^Adam, max_norm: f64) {
	if max_norm <= 0.0 {return}

	total_norm_sq := 0.0
	for param in opt.parameters {
		if param.grad.data != nil && len(param.grad.data) > 0 {
			// Use SIMD dot product for efficient squared sum
			total_norm_sq += l.dot_simd(param.grad.data, param.grad.data)
		}
	}

	total_norm := math.sqrt(total_norm_sq)
	if total_norm > max_norm {
		scale := max_norm / (total_norm + 1e-6) // Add epsilon for stability
		for param in opt.parameters {
			if param.grad.data != nil && len(param.grad.data) > 0 {
				// Scale gradient in-place using SIMD
				l.vec_scale_simd(param.grad.data, scale, param.grad.data)
			}
		}
	}
}
