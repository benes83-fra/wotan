package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Deep BSDE (Han, Jentzen & E 2018) for Black-Scholes payoffs
// ============================================================================
// d independent assets dX = r X dt + sigma X dW under the pricing measure; the option
// value u(t, X) solves the BS PDE, and Y_t = u(t, X_t) follows the BSDE
//   dY = r Y dt + Z . dW,   Y_T = g(X_T),   Z = sigma X grad_x u.
// Y_0 (one trainable scalar) and Z_t = z_net(t, ln(X_t / S_0)) are trained so that the
// simulated Y_T matches the payoff: loss E[(Y_T - g(X_T))^2]. At the optimum Y_0 is the
// price. Discretisation: X is simulated exactly (log-normal steps), the linear driver is
// integrated exactly (Y_{k+1} = e^{r dt} Y_k + Z_k . dW_k), so the only time-step error
// is that of the hedge Z, which has mean zero. Payoff: call on the maximum of the assets,
// max(max_j X_T^j - K, 0) (for d = 1 a European call).

DeepBSDEModel :: struct {
	y0:        ^t.Tensor, // [1, 1], the price estimate
	z_net:     ^nn.Sequential, // (t, ln(X / S_0)) -> Z, [1 + d] -> [d]
	T:         f64,
	N:         int,
	d:         int,
	allocator: mem.Allocator,
}

// n_paths is not used (Y_0 is a single scalar for any batch size); kept for existing callers
deep_bsde_model_new :: proc(
	initial_y0: f64,
	d: int,
	T: f64,
	N: int,
	hidden_size: int,
	num_layers: int,
	n_paths: int = 0,
	allocator: mem.Allocator = context.allocator,
) -> DeepBSDEModel {
	model: DeepBSDEModel
	model.T = T
	model.N = N
	model.d = d
	model.allocator = allocator

	y0_data := l.matrix_new(f64, 1, 1, allocator)
	y0_data.data[0] = initial_y0
	model.y0 = t.tensor_new(y0_data, true, allocator)

	model.z_net = nn.sequential_new(allocator)
	nn.sequential_add(model.z_net, nn.linear_layer_new(1 + d, hidden_size, allocator))
	nn.sequential_add(model.z_net, nn.Activation.ReLU)
	for _ in 1 ..< num_layers {
		nn.sequential_add(model.z_net, nn.linear_layer_new(hidden_size, hidden_size, allocator))
		nn.sequential_add(model.z_net, nn.Activation.ReLU)
	}
	nn.sequential_add(model.z_net, nn.linear_layer_new(hidden_size, d, allocator))
	return model
}

deep_bsde_model_free :: proc(model: ^DeepBSDEModel) {
	if model.y0 != nil {t.tensor_free(model.y0)}
	if model.z_net != nil {nn.sequential_free(model.z_net)}
}

// The current price estimate Y_0
deep_bsde_y0 :: proc(model: ^DeepBSDEModel) -> f64 {
	return model.y0.data.data[0]
}

// Registers Y_0 and the Z network with Adam
deep_bsde_add_to_adam :: proc(model: ^DeepBSDEModel, opt: ^nn.Adam) {
	nn.adam_add_param(opt, model.y0)
	nn.sequential_add_to_adam(model.z_net, opt)
}

// ============================================================================
// Forward Pass
// ============================================================================
// Simulates n_paths paths over n_steps steps of length T / n_steps; returns Y_T
// [n_paths, 1] (graph) and X_T [n_paths, d] (constant). K is not used here.
deep_bsde_bs_forward :: proc(
	model: ^DeepBSDEModel,
	S_0: []f64,
	K: f64,
	r: f64,
	sigma: []f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> (
	Y_T: ^t.Tensor,
	X_T: ^t.Tensor,
) {
	d := model.d
	assert(len(S_0) == d && len(sigma) == d, "deep_bsde_bs_forward: S_0 and sigma need d entries")
	allocator := t._scope_or(allocator) // the active graph scope's arena, if any
	dt := model.T / f64(n_steps)
	sqrt_dt := math.sqrt(dt)
	growth := math.exp(r * dt)

	X_data := l.matrix_new(f64, n_paths, d, allocator)
	for i in 0 ..< n_paths {for j in 0 ..< d {X_data.data[i * d + j] = S_0[j]}}
	X := t.tensor_new(X_data, false, allocator)

	// Y_0 broadcast to every path: ones [n_paths, 1] @ y0 [1, 1]
	ones_n := l.matrix_new(f64, n_paths, 1, allocator)
	for &v in ones_n.data {v = 1}
	Y := t.tensor_matmul(t.tensor_new(ones_n, false, allocator), model.y0)

	ones_d := l.matrix_new(f64, d, 1, allocator)
	for &v in ones_d.data {v = 1}
	ones := t.tensor_new(ones_d, false, allocator)

	for step in 0 ..< n_steps {
		t_k := f64(step) * dt

		dW_data := l.matrix_new(f64, n_paths, d, allocator)
		for &v in dW_data.data {v = rand.float64_normal(0.0, 1.0) * sqrt_dt}
		dW := t.tensor_new(dW_data, false, allocator)

		// Z-network input [t, ln(X / S_0)]
		input_data := l.matrix_new(f64, n_paths, 1 + d, allocator)
		for i in 0 ..< n_paths {
			input_data.data[i * (1 + d)] = t_k
			for j in 0 ..< d {
				input_data.data[i * (1 + d) + 1 + j] = math.ln(X.data.data[i * d + j] / S_0[j])
			}
		}
		Z_k := nn.sequential_forward(model.z_net, t.tensor_new(input_data, false, allocator))

		// exact GBM step
		X_next_data := l.matrix_new(f64, n_paths, d, allocator)
		for i in 0 ..< n_paths {
			for j in 0 ..< d {
				s := sigma[j]
				X_next_data.data[i * d + j] =
					X.data.data[i * d + j] *
					math.exp((r - 0.5 * s * s) * dt + s * dW.data.data[i * d + j])
			}
		}
		X = t.tensor_new(X_next_data, false, allocator)

		// Y_{k+1} = e^{r dt} Y_k + sum_j Z_k^j dW_k^j
		Z_dW := t.tensor_matmul(t.tensor_mul(Z_k, dW), ones)
		Y = t.tensor_add(t.tensor_scale(Y, growth), Z_dW)
	}
	return Y, X
}

// ============================================================================
// Loss Function
// ============================================================================
// mean over the paths of (Y_T - max(max_j X_T^j - K, 0))^2
deep_bsde_bs_loss :: proc(
	model: ^DeepBSDEModel,
	S_0: []f64,
	K: f64,
	r: f64,
	sigma: []f64,
	n_paths: int,
	n_steps: int,
	allocator: mem.Allocator = context.allocator,
) -> ^t.Tensor {
	d := model.d
	allocator := t._scope_or(allocator)
	Y_T, X_T := deep_bsde_bs_forward(model, S_0, K, r, sigma, n_paths, n_steps, allocator)

	payoff_data := l.matrix_new(f64, n_paths, 1, allocator)
	for i in 0 ..< n_paths {
		m := X_T.data.data[i * d]
		for j in 1 ..< d {m = math.max(m, X_T.data.data[i * d + j])}
		payoff_data.data[i] = math.max(m - K, 0.0)
	}
	diff := t.tensor_sub(Y_T, t.tensor_new(payoff_data, false, allocator))
	return t.tensor_mean(t.tensor_mul(diff, diff))
}
