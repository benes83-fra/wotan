

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
	A:          ^t.Tensor, // [d_model, d_state]
	D:          ^t.Tensor, // [1, d_model] (Skip connection)
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

	// Initialize A (Diagonal state transition matrix)
	A_data := l.matrix_new(f64, d_model, d_state, allocator)
	for d in 0 ..< d_model {
		for n in 0 ..< d_state {
			// Stable initialization
			A_data.data[d * d_state + n] = -math.exp_f64(f64(n) / f64(d_state) * math.ln_f64(10.0))
		}
	}
	layer.A = t.tensor_new(A_data, true, allocator)
	layer.A.shape = [4]int{d_model, d_state, 1, 1}

	// Initialize D (Skip connection)
	D_data := l.matrix_new(f64, 1, d_model, allocator)
	for i in 0 ..< d_model {D_data.data[i] = 1.0}
	layer.D = t.tensor_new(D_data, true, allocator)

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

	// 2. Apply softplus to Delta to ensure it is positive
	Delta := t.tensor_softplus(delta_raw)

	// 3. Selective SSM
	ssm_out := t.tensor_ssm(x_proj, h_0, layer.A, B_proj, C_proj, Delta, layer.D)

	// 4. Output projection
	out := linear_forward(&layer.proj_out, ssm_out)

	return out
}
