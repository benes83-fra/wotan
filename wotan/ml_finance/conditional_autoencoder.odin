package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// CONDITIONAL AUTOENCODER (Gu, Kelly & Xiu 2021, "Autoencoder asset pricing models")
// ============================================================================
//   r_{i,t+1} = beta(z_{i,t})' f_{t+1} + e
//   beta network    characteristics z [L] -> hidden ReLU layers -> K betas
//                   (hidden nil: CA0, linear betas as IPCA; {32}: CA1; {32, 16}: CA2)
//   factor network  linear: the L characteristic-managed portfolios
//                   x_t = (Z_t' Z_t)^-1 Z_t' r_{t+1} -> K factors
// Fitted on the whole (unbalanced) panel at once: loss = mean squared error over every
// observation + l1 * sum |w| over the weight matrices (the subgradient l1 sign(w) is
// added to the gradients, as torch's autograd does for abs). Full-batch Adam, early
// stopping on a validation panel (the best epoch's weights are restored). GKX also use
// batch normalisation and ensembles of 10 seeds: not here (an ensemble is a loop).
// Like IPCA: total R² with the realised factors x_t of each date, predictive R² with the
// mean factor of the training period, both 1 - SSE / sum r² (not demeaned).

CAConfig :: struct {
	n_chars:   int,
	n_factors: int,
	hidden:    []int, // widths of the beta network's hidden layers (copied)
	l1:        f64, // L1 penalty on the weights (not the biases)
}

ConditionalAutoencoder :: struct {
	beta_net:   ^nn.Sequential,
	factor_net: ^nn.Sequential,
	config:     CAConfig,
	allocator:  mem.Allocator,
	scope:      t.Graph_Scope,
}

ca_new :: proc(config: CAConfig, allocator := context.allocator) -> ^ConditionalAutoencoder {
	assert(config.n_chars > 0 && config.n_factors > 0, "ca_new: sizes must be positive")
	assert(config.l1 >= 0, "ca_new: l1 >= 0")
	m := new(ConditionalAutoencoder, allocator)
	m.allocator = allocator
	m.config = config
	m.config.hidden = make([]int, len(config.hidden), allocator)
	copy(m.config.hidden, config.hidden)
	m.beta_net = nn.sequential_new(allocator)
	prev := config.n_chars
	for h in config.hidden {
		nn.sequential_add(m.beta_net, nn.linear_layer_new(prev, h, allocator))
		nn.sequential_add(m.beta_net, nn.Activation.ReLU)
		prev = h
	}
	nn.sequential_add(m.beta_net, nn.linear_layer_new(prev, config.n_factors, allocator))
	m.factor_net = nn.sequential_new(allocator)
	nn.sequential_add(
		m.factor_net,
		nn.linear_layer_new(config.n_chars, config.n_factors, allocator),
	)
	return m
}

ca_free :: proc(m: ^ConditionalAutoencoder) {
	nn.sequential_free(m.beta_net)
	nn.sequential_free(m.factor_net)
	t.graph_scope_destroy(&m.scope)
	delete(m.config.hidden, m.allocator)
	free(m, m.allocator)
}

// Every trainable tensor: the beta network's layers, then the factor network's
// (weights before biases within a layer)
ca_parameters :: proc(
	m: ^ConditionalAutoencoder,
	allocator := context.allocator,
) -> [dynamic]^t.Tensor {
	p := nn.sequential_parameters(m.beta_net, allocator)
	q := nn.sequential_parameters(m.factor_net, context.temp_allocator)
	defer delete(q)
	append(&p, ..q[:])
	return p
}

// The panel as constant tensors: Z [n, L], r [n, 1], the date selector S [n, T]
// (one-hot), the managed portfolios P [T, L]
CAPanel :: struct {
	Z, r, S, P: ^t.Tensor,
	times:      []int,
	T:          int,
	allocator:  mem.Allocator,
}

ca_panel_free :: proc(p: ^CAPanel) {
	for x in ([]^t.Tensor{p.Z, p.r, p.S, p.P}) {t.tensor_free(x)}
	delete(p.times, p.allocator)
}

// times: dates 0 ..< T, every date needs observations (and Z_t' Z_t invertible)
ca_panel_new :: proc(
	X: ^l.Matrix(f64),
	y: []f64,
	times: []int,
	allocator := context.allocator,
) -> CAPanel {
	n, L := X.rows, X.cols
	port := ipca_portfolios(X, y, times, context.temp_allocator)
	defer ipca_portfolios_free(&port)
	T := port.T
	p := CAPanel {
		T         = T,
		allocator = allocator,
		times     = make([]int, n, allocator),
	}
	copy(p.times, times)
	Zm := l.matrix_new(f64, n, L, allocator)
	copy(Zm.data, X.data[:n * L])
	p.Z = t.tensor_new(Zm, false, allocator)
	rm := l.matrix_new(f64, n, 1, allocator)
	copy(rm.data, y)
	p.r = t.tensor_new(rm, false, allocator)
	Sm := l.matrix_new(f64, n, T, allocator)
	for d, i in times {Sm.data[i * T + d] = 1}
	p.S = t.tensor_new(Sm, false, allocator)
	Pm := l.matrix_new(f64, T, L, allocator)
	W := l.matrix_new(f64, L, L, context.temp_allocator)
	defer l.matrix_free(&W)
	for d in 0 ..< T {
		copy(W.data, port.W[d * L * L:][:L * L])
		x := l.solve_spd_cholesky(&W, port.Q[d * L:][:L], .Unblocked, context.temp_allocator)
		copy(Pm.data[d * L:][:L], x)
		delete(x, context.temp_allocator)
	}
	p.P = t.tensor_new(Pm, false, allocator)
	return p
}

// Fitted returns [n, 1] (graph): rowsum(beta(Z) * S F(P)); with lambda the factors are
// this vector [K] at every date instead (the predictive fit)
ca_fitted :: proc(
	m: ^ConditionalAutoencoder,
	p: ^CAPanel,
	alloc: mem.Allocator,
	lambda: []f64 = nil,
) -> ^t.Tensor {
	K := m.config.n_factors
	n := p.Z.data.rows
	beta := nn.sequential_forward(m.beta_net, p.Z) // [n, K]
	fo: ^t.Tensor
	if lambda == nil {
		f := nn.sequential_forward(m.factor_net, p.P) // [T, K]
		fo = t.tensor_matmul(p.S, f) // [n, K]
	} else {
		c := t.tensor_new(l.matrix_new(f64, n, K, alloc), false, alloc)
		c.owned_by_graph = true
		for i in 0 ..< n {copy(c.data.data[i * K:][:K], lambda)}
		fo = c
	}
	ones := t.tensor_new(l.matrix_new(f64, K, 1, alloc), false, alloc)
	ones.owned_by_graph = true
	for &v in ones.data.data {v = 1}
	return t.tensor_matmul(t.tensor_mul(beta, fo), ones)
}

// l1 * sum |w| over the weight matrices
@(private = "file")
_ca_l1 :: proc(m: ^ConditionalAutoencoder) -> f64 {
	if m.config.l1 == 0 {return 0}
	s := 0.0
	for net in ([]^nn.Sequential{m.beta_net, m.factor_net}) {
		for &layer in net.layers {
			if lin, ok := layer.(nn.LinearLayer); ok {for v in lin.weights.data.data {s += abs(v)}}
		}
	}
	return m.config.l1 * s
}

// Gradients of MSE + L1 in every parameter's grad (zeroed first); returns the loss
ca_gradients :: proc(m: ^ConditionalAutoencoder, p: ^CAPanel) -> f64 {
	params := ca_parameters(m, context.temp_allocator)
	for q in params {t.tensor_zero_grad(q)}
	delete(params)
	t.graph_begin(&m.scope)
	defer t.graph_end(&m.scope)
	alloc := t.graph_allocator(&m.scope)
	loss := t.tensor_mse_loss(ca_fitted(m, p, alloc), p.r, alloc)
	t.tensor_backward(loss)
	if m.config.l1 > 0 {
		for net in ([]^nn.Sequential{m.beta_net, m.factor_net}) {
			for &layer in net.layers {
				if lin, ok := layer.(nn.LinearLayer); ok {
					for v, i in lin.weights.data.data {
						s := 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
						lin.weights.grad.data[i] += m.config.l1 * s
					}
				}
			}
		}
	}
	return loss.data.data[0] + _ca_l1(m)
}

// One full-batch Adam step (opt holds ca_parameters); the loss before the update
ca_train_step :: proc(m: ^ConditionalAutoencoder, p: ^CAPanel, opt: ^nn.Adam) -> f64 {
	loss := ca_gradients(m, p)
	nn.adam_step(opt)
	return loss
}

// MSE of the fit (no penalty), no graph
ca_mse :: proc(m: ^ConditionalAutoencoder, p: ^CAPanel, lambda: []f64 = nil) -> f64 {
	t.graph_begin(&m.scope, no_grad = true)
	defer t.graph_end(&m.scope)
	fit := ca_fitted(m, p, t.graph_allocator(&m.scope), lambda)
	s := 0.0
	for v, i in fit.data.data {d := v - p.r.data.data[i]; s += d * d}
	return s / f64(len(fit.data.data))
}

// The factors F(P) [T, K] row-major
ca_factors :: proc(
	m: ^ConditionalAutoencoder,
	p: ^CAPanel,
	allocator := context.allocator,
) -> []f64 {
	t.graph_begin(&m.scope, no_grad = true)
	defer t.graph_end(&m.scope)
	f := nn.sequential_forward(m.factor_net, p.P)
	out := make([]f64, len(f.data.data), allocator)
	copy(out, f.data.data)
	return out
}

// Mean factor [K] over the panel's dates (the risk premia for the predictive fit)
ca_mean_factor :: proc(
	m: ^ConditionalAutoencoder,
	p: ^CAPanel,
	allocator := context.allocator,
) -> []f64 {
	K := m.config.n_factors
	f := ca_factors(m, p, context.temp_allocator)
	defer delete(f, context.temp_allocator)
	out := make([]f64, K, allocator)
	for d in 0 ..< p.T {for k in 0 ..< K {out[k] += f[d * K + k] / f64(p.T)}}
	return out
}

// Total and predictive R² (1 - SSE / sum r²); lambda: the training period's mean factor
// (nil: this panel's own)
ca_r2 :: proc(
	m: ^ConditionalAutoencoder,
	p: ^CAPanel,
	lambda: []f64 = nil,
) -> (
	total, predictive: f64,
) {
	lam := lambda
	if lam == nil {lam = ca_mean_factor(m, p, context.temp_allocator)}
	defer if lambda == nil {delete(lam, context.temp_allocator)}
	ss := 0.0
	for v in p.r.data.data {ss += v * v}
	n := f64(len(p.r.data.data))
	return 1 - ca_mse(m, p) * n / ss, 1 - ca_mse(m, p, lam) * n / ss
}

CAFitResult :: struct {
	best_epoch: int, // Adam updates behind the kept weights
	epochs_run: int,
	last_loss:  f64, // MSE + L1 of the last step (before its update)
	valid_mse:  f64, // validation MSE of the kept weights; NaN without validation panel
}

// Full-batch Adam for up to `epochs` steps; with a validation panel, stops after
// `patience` epochs without a lower validation MSE and restores the best weights
ca_fit :: proc(
	m: ^ConditionalAutoencoder,
	train: ^CAPanel,
	valid: ^CAPanel = nil,
	epochs := 500,
	learning_rate := 1e-3,
	patience := 50,
) -> CAFitResult {
	opt := nn.adam_new(learning_rate, allocator = context.temp_allocator)
	defer nn.adam_free(&opt)
	params := ca_parameters(m, context.temp_allocator)
	defer delete(params)
	for q in params {nn.adam_add_param(&opt, q)}
	// snapshot of the best weights
	best := make([][]f64, len(params), context.temp_allocator)
	defer {for b in best {delete(b, context.temp_allocator)}; delete(best, context.temp_allocator)}
	for q, i in params {
		best[i] = make([]f64, len(q.data.data), context.temp_allocator)
		copy(best[i], q.data.data)
	}
	res := CAFitResult {
		valid_mse = math.nan_f64(),
	}
	best_valid := math.INF_F64
	since := 0
	for e in 0 ..< epochs {
		res.last_loss = ca_train_step(m, train, &opt)
		res.epochs_run = e + 1
		if valid == nil {
			res.best_epoch = e + 1
			continue
		}
		v := ca_mse(m, valid)
		if v < best_valid {
			best_valid, since = v, 0
			res.best_epoch, res.valid_mse = e + 1, v
			for q, i in params {copy(best[i], q.data.data)}
		} else {
			since += 1
			if since >= patience {break}
		}
	}
	if valid != nil && res.best_epoch > 0 {
		for q, i in params {copy(q.data.data, best[i])}
	}
	return res
}
