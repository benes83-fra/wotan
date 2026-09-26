

package nn


import l "../linalg"
import t "../tensor"

import "core:math"
import "core:mem"
// ============================================================================
// Mamba Layer (Selective State Space Model)
// ============================================================================
MambaLayer :: struct {
	d_model:    int,
	d_state:    int,
	proj_x:     LinearLayer,
	proj_B:     LinearLayer,
	proj_C:     LinearLayer,
	proj_Delta: LinearLayer,
	A:          ^t.Tensor, // ✅ CHANGED: Trainable theta_A (was A)
	D:          ^t.Tensor,
	proj_out:   LinearLayer,
}

mamba_layer_new :: proc(
	d_model: int,
	d_state: int = 16,
	allocator: mem.Allocator = context.allocator,
) -> MambaLayer {
	layer: MambaLayer
	layer.d_model = d_model
	layer.d_state = d_state

	layer.proj_x = linear_layer_new(d_model, d_model, allocator)
	layer.proj_B = linear_layer_new(d_model, d_state, allocator)
	layer.proj_C = linear_layer_new(d_model, d_state, allocator)
	layer.proj_Delta = linear_layer_new(d_model, d_model, allocator)
	layer.proj_out = linear_layer_new(d_model, d_model, allocator)

	// ✅ Initialize A_param (theta_A)
	// We want the actual A to be in [-0.5, -4.0] for stable decay.
	// Since A = -exp(theta_A), theta_A = log(-A).
	A_param_data := l.matrix_new(f64, d_model, d_state, allocator)
	for d in 0 ..< d_model {
		for n in 0 ..< d_state {
			frac := f64(n) / f64(d_state - 1)
			target_A := -(0.5 * math.pow(8.0, frac)) // -0.5 to -4.0
			A_param_data.data[d * d_state + n] = math.ln_f64(-target_A)
		}
	}
	layer.A = t.tensor_new(A_param_data, true, allocator)
	layer.A.shape = [4]int{d_model, d_state, 1, 1}

	// Initialize D (Skip connection)
	D_data := l.matrix_new(f64, 1, d_model, allocator)
	for i in 0 ..< d_model {D_data.data[i] = 1.0}
	// ✅ UNFROZEN: Safe to train now that softplus is fixed!
	layer.D = t.tensor_new(D_data, true, allocator)

	if layer.proj_Delta.bias != nil {
		for i in 0 ..< d_model {
			layer.proj_Delta.bias.data.data[i] = -2.0
		}
	}
	return layer
}

mamba_layer_free :: proc(layer: ^MambaLayer) {
	linear_layer_free(&layer.proj_x)
	linear_layer_free(&layer.proj_B)
	linear_layer_free(&layer.proj_C)
	linear_layer_free(&layer.proj_Delta)
	linear_layer_free(&layer.proj_out)
	if layer.A != nil {t.tensor_free(layer.A)}
	if layer.D != nil {t.tensor_free(layer.D)}
}

mamba_layer_forward :: proc(layer: ^MambaLayer, x: ^t.Tensor, h_0: ^t.Tensor) -> ^t.Tensor {
	// 1. Projections
	x_proj := linear_forward(&layer.proj_x, x)
	B_proj := linear_forward(&layer.proj_B, x)
	C_proj := linear_forward(&layer.proj_C, x)
	delta_raw := linear_forward(&layer.proj_Delta, x)
	Delta := t.tensor_softplus(delta_raw)
	if layer.proj_x.weights.requires_grad {
		x_proj.requires_grad = true
		B_proj.requires_grad = true
		C_proj.requires_grad = true
		delta_raw.requires_grad = true
		Delta.requires_grad = true
	}
	// 2. ✅ REPARAMETERIZE A: A = -exp(theta_A)
	// This mathematically guarantees A is ALWAYS negative, preventing explosions!
	A_pos := t.tensor_exp(layer.A)
	A := t.tensor_neg(A_pos)

	// 3. Selective SSM
	ssm_out := t.tensor_ssm(x_proj, h_0, A, B_proj, C_proj, Delta, layer.D)

	// Clean up A intermediates if we aren't building an autograd graph
	if !layer.A.requires_grad {
		t.tensor_free(A_pos)
		t.tensor_free(A)
	}

	// 4. Output projection
	out := linear_forward(&layer.proj_out, ssm_out)

	return out
}
mamba_layer_constrain :: proc(layer: ^MambaLayer) {
	t.tensor_clip_weights(layer.A, -8.0, -0.05)
}
