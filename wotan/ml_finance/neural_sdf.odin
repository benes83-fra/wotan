package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// Neural stochastic discount factor
//
//   M_t = f_θ(Z_t)² + ε   (a network of state variables Z_t observed with the returns
//                          R_t, so M_t > 0; ε is the floor)
//   Euler equations:  E[M_t R_{i,t}] = 1 for every asset i (gross returns)
//   Loss:  Σ_i (mean_t(M_t R_{i,t}) - 1)²  +  λ Var_t(M)
//
// With few assets many SDFs price them exactly; the variance penalty λ picks among them,
// and as λ -> 0 the loss's minimiser tends to the minimum-variance SDF that prices the
// assets (Hansen-Jagannathan). neural_sdf_optimum gives the minimiser over every series
// M_t: no network can reach a lower loss, and a network of the returns themselves can
// reach it (the optimum is affine in R_t). The variance penalty used to be fixed at 1,
// so large next to the squared pricing errors of daily returns (~1e-7) that the learnt
// SDF was a constant, and the network started near M = ε.
// ============================================================================

NeuralSDFConfig :: struct {
	input_size:       int, // Dimension of the state variables Z_t
	hidden_size:      int, // Hidden layer width
	num_layers:       int, // Number of hidden layers
	num_assets:       int, // Number of assets to price
	epsilon:          f64, // Positivity floor ε of M (e.g. 0.01)
	variance_penalty: f64, // λ, the weight of Var(M) in the loss (0: pricing errors only)
}

NeuralSDF :: struct {
	network:   ^nn.Sequential,
	config:    NeuralSDFConfig,
	allocator: mem.Allocator,
	scope:     t.Graph_Scope, // one arena per training step / evaluation, reused
}

// ============================================================================
// Initialization
// ============================================================================

neural_sdf_new :: proc(
	config: NeuralSDFConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^NeuralSDF {
	assert(
		config.input_size > 0 && config.hidden_size > 0 && config.num_assets > 0,
		"neural_sdf_new: sizes must be positive",
	)
	assert(
		config.epsilon >= 0 && config.variance_penalty >= 0,
		"neural_sdf_new: epsilon and variance_penalty must be >= 0",
	)
	model := new(NeuralSDF, allocator)
	model.config = config
	model.allocator = allocator

	seq := nn.sequential_new(allocator)
	nn.sequential_add(seq, nn.linear_layer_new(config.input_size, config.hidden_size, allocator))
	nn.sequential_add(seq, nn.Activation.ReLU)
	for _ in 1 ..< config.num_layers {
		nn.sequential_add(
			seq,
			nn.linear_layer_new(config.hidden_size, config.hidden_size, allocator),
		)
		nn.sequential_add(seq, nn.Activation.ReLU)
	}
	// one output per date: the SDF before squaring, started near M = 1 (small weights,
	// bias sqrt(1 - ε)); from a random start M ~ ε and the loss is ~N, far from the
	// 1e-5 scale of the pricing errors the training has to resolve
	out := nn.linear_layer_new(config.hidden_size, 1, allocator)
	for &v in out.weights.data.data {v *= 0.01}
	out.bias.data.data[0] = math.sqrt(max(1 - config.epsilon, 0))
	nn.sequential_add(seq, out)
	model.network = seq
	return model
}

neural_sdf_free :: proc(model: ^NeuralSDF) {
	if model.network != nil {
		nn.sequential_free(model.network)
	}
	t.graph_scope_destroy(&model.scope)
	free(model, model.allocator)
}

// ============================================================================
// SDF and loss (graph operations)
// ============================================================================

// A constant of the graph (it must live until backward, so the graph owns it)
@(private = "file")
_sdf_const :: proc(rows, cols: int, alloc: mem.Allocator) -> ^t.Tensor {
	c := t.tensor_new(l.matrix_new(f64, rows, cols, alloc), false, alloc)
	c.owned_by_graph = true
	return c
}

// M = f_θ(Z)² + ε, [T, 1], for z [T, input_size]
neural_sdf_values :: proc(model: ^NeuralSDF, z: ^t.Tensor, alloc: mem.Allocator) -> ^t.Tensor {
	raw := nn.sequential_forward(model.network, z)
	floor := _sdf_const(raw.data.rows, 1, alloc)
	floor.shape = raw.shape
	for &v in floor.data.data {v = model.config.epsilon}
	return t.tensor_add(t.tensor_mul(raw, raw), floor)
}

// The loss Σ_i (mean_t(M_t R_{i,t}) - 1)² + λ Var(M) (population variance) of an SDF
// series m [T, 1] and gross returns r [T, N] (row-major), as a graph scalar
neural_sdf_loss :: proc(
	m: ^t.Tensor,
	r: ^t.Tensor,
	variance_penalty: f64,
	alloc: mem.Allocator,
) -> ^t.Tensor {
	n := m.data.rows
	assert(
		len(m.data.data) == n && r.data.rows == n,
		"neural_sdf_loss: m must be [T, 1] and r [T, N]",
	)
	N := r.data.cols
	rt := _sdf_const(N, n, alloc) // R' / T
	for row in 0 ..< n {
		for i in 0 ..< N {rt.data.data[i * n + row] = r.data.data[row * N + i] / f64(n)}
	}
	one := _sdf_const(N, 1, alloc)
	for &v in one.data.data {v = 1}
	e := t.tensor_sub(t.tensor_matmul(rt, m), one) // pricing errors [N, 1]
	loss := t.tensor_sum(t.tensor_mul(e, e))
	if variance_penalty != 0 {
		mean := t.tensor_mean(m)
		variance := t.tensor_sub(t.tensor_mean(t.tensor_mul(m, m)), t.tensor_mul(mean, mean))
		loss = t.tensor_add(loss, t.tensor_scale(variance, variance_penalty))
	}
	return loss
}

// ============================================================================
// Training and evaluation
// ============================================================================
// z: [T, input_size] state variables; r: [T, num_assets] gross returns (1 + net return).
// One full-batch Adam step; returns the loss before the update.
neural_sdf_train_step :: proc(
	model: ^NeuralSDF,
	z: ^t.Tensor,
	r: ^t.Tensor,
	opt: ^nn.Adam,
) -> f64 {
	assert(
		r.data.cols == model.config.num_assets,
		"neural_sdf_train_step: r must have num_assets columns",
	)
	t.graph_begin(&model.scope)
	defer t.graph_end(&model.scope)
	alloc := t.graph_allocator(&model.scope)
	loss := neural_sdf_loss(
		neural_sdf_values(model, z, alloc),
		r,
		model.config.variance_penalty,
		alloc,
	)
	nn.adam_zero_grad(opt)
	t.tensor_backward(loss)
	nn.adam_step(opt)
	return loss.data.data[0]
}

// The SDF series M_t (length T)
neural_sdf_series :: proc(model: ^NeuralSDF, z: ^t.Tensor, allocator: mem.Allocator) -> []f64 {
	t.graph_begin(&model.scope, no_grad = true)
	defer t.graph_end(&model.scope)
	m := neural_sdf_values(model, z, t.graph_allocator(&model.scope))
	out := make([]f64, len(m.data.data), allocator)
	copy(out, m.data.data)
	return out
}

// The pricing errors |mean_t(M_t R_{i,t}) - 1| per asset
neural_sdf_evaluate :: proc(
	model: ^NeuralSDF,
	z: ^t.Tensor,
	r: ^t.Tensor,
	allocator: mem.Allocator,
) -> []f64 {
	m := neural_sdf_series(model, z, context.temp_allocator)
	defer delete(m, context.temp_allocator)
	return sdf_pricing_errors(m, r.data.data, r.data.cols, allocator)
}

// |mean_t(m_t R_{i,t}) - 1| per asset, for returns [T, N] row-major
sdf_pricing_errors :: proc(
	m: []f64,
	returns: []f64,
	num_assets: int,
	allocator: mem.Allocator,
) -> []f64 {
	n := len(m)
	assert(len(returns) == n * num_assets, "sdf_pricing_errors: returns must be [T, N]")
	out := make([]f64, num_assets, allocator)
	for i in 0 ..< num_assets {
		s := 0.0
		for row in 0 ..< n {s += m[row] * returns[row * num_assets + i]}
		out[i] = math.abs(s / f64(n) - 1)
	}
	return out
}

// The minimiser of the neural SDF loss over every series m_t (no network, no positivity
// constraint), for λ > 0. The first-order conditions give m = m̄ - R e / λ with e the
// pricing errors, so with μ = mean_t R_t and S = mean_t R_t R_t':
//   (I + S/λ) e - m̄ μ = -1,   μ'e = 0.
// Returns the series, its loss and its smallest value (if it is below the network's
// floor ε, the network cannot reach it).
neural_sdf_optimum :: proc(
	returns: []f64,
	num_assets: int,
	variance_penalty: f64,
	allocator: mem.Allocator,
) -> (
	m: []f64,
	loss: f64,
	min_m: f64,
	ok: bool,
) {
	N := num_assets
	n := len(returns) / N
	assert(
		variance_penalty > 0 && len(returns) == n * N && n > 0,
		"neural_sdf_optimum: need λ > 0 and returns [T, N]",
	)
	A := l.matrix_new(f64, N + 1, N + 1, context.temp_allocator)
	b := make([]f64, N + 1, context.temp_allocator)
	for row in 0 ..< n {
		R := returns[row * N:][:N]
		for i in 0 ..< N {
			for j in 0 ..< N {A.data[i * (N + 1) + j] += R[i] * R[j] / (f64(n) * variance_penalty)}
			A.data[i * (N + 1) + N] -= R[i] / f64(n) // -m̄ μ_i
			A.data[N * (N + 1) + i] += R[i] / f64(n) // μ'e
		}
	}
	for i in 0 ..< N {
		A.data[i * (N + 1) + i] += 1
		b[i] = -1
	}
	LU, piv, _, solved := l.lu_decompose(&A, context.temp_allocator)
	if !solved {return}
	x := l.lu_solve(&LU, piv, b, context.temp_allocator)
	e, m_bar := x[:N], x[N]
	m = make([]f64, n, allocator)
	min_m = math.INF_F64
	mean, sq := 0.0, 0.0
	for row in 0 ..< n {
		v := m_bar
		for i in 0 ..< N {v -= returns[row * N + i] * e[i] / variance_penalty}
		m[row] = v
		min_m = min(min_m, v)
		mean += v / f64(n)
		sq += v * v / f64(n)
	}
	for v in e {loss += v * v}
	loss += variance_penalty * (sq - mean * mean)
	return m, loss, min_m, true
}

// ============================================================================
// Save / Load
// ============================================================================

neural_sdf_save :: proc(model: ^NeuralSDF, path: string) -> bool {
	dummy_opt := nn.adam_new(0.001, allocator = model.allocator)
	defer nn.adam_free(&dummy_opt)
	nn.sequential_add_to_adam(model.network, &dummy_opt)
	return nn.save_checkpoint(model.network, &dummy_opt, path, 0, model.allocator)
}

// The checkpoint holds the network only: pass the configuration it was trained with
neural_sdf_load :: proc(
	path: string,
	config: NeuralSDFConfig,
	allocator: mem.Allocator = context.allocator,
) -> (
	^NeuralSDF,
	bool,
) {
	loaded_model, loaded_opt, _, ok := nn.load_checkpoint(path, allocator)
	if !ok {
		return nil, false
	}
	if loaded_opt != nil {
		nn.adam_free(loaded_opt)
		free(loaded_opt, allocator) // load_checkpoint allocates the Adam struct itself
	}
	model := new(NeuralSDF, allocator)
	model.network = loaded_model
	model.config = config
	model.allocator = allocator
	return model, true
}
