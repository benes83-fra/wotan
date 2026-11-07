package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// N-BEATS (Oreshkin, Carpov, Chapados & Bengio 2020), as the authors' reference code
// (ElementAI/N-BEATS, models/nbeats.py)
// ============================================================================
// A stack of blocks on a window x of the last input_size values. The window is
// time-reversed; the forecast starts at the last observed value. Each block: `layers`
// fully connected ReLU layers, one linear head theta, a basis maps theta to a backcast
// (subtracted from the block's input: the residual goes on) and a forecast (added up):
//   generic         theta = (backcast [input_size], forecast [horizon]) directly
//   trend           polynomials (t / size)^p, p = 0 .. degree, on each grid
//   seasonality     Fourier: frequencies 0 and harmonics .. harmonics·horizon/2 step 1,
//                   divided by harmonics; backcast grid -2π t / horizon
// Interpretable: a trend stack then a seasonality stack, the blocks of a stack sharing
// one set of weights (as the reference). Trained here with mean squared error.
// Option relative (not in the reference): the blocks get x - x_last, so a series that
// drifts to levels never seen in training still gets inputs of the trained scale.

NBeatsBasis :: enum {
	Generic,
	Trend,
	Seasonality,
}

NBeatsConfig :: struct {
	input_size:             int,
	horizon:                int,
	interpretable:          bool,
	blocks:                 int, // generic: number of blocks (paper 30); interpretable: per stack (3)
	layers:                 int, // FC layers per block (4)
	layer_size:             int, // generic (512) / trend stack (256)
	seasonality_layer_size: int, // interpretable only (2048)
	degree:                 int, // trend polynomial degree (3)
	harmonics:              int, // seasonality (1)
	// the blocks see the window minus its last value (the forecast starts from it
	// anyway): level-free inputs for trending series; off = the reference
	relative:               bool,
}

NBeatsBlock :: struct {
	layers:   [dynamic]nn.LinearLayer,
	theta:    nn.LinearLayer,
	basis:    NBeatsBasis,
	back_mat: [2]^t.Tensor, // basis templates [P, input_size] (seasonality: cos, sin)
	fore_mat: [2]^t.Tensor, // [P, horizon]
	p:        int, // basis parameters per part
}

NBeats :: struct {
	config:    NBeatsConfig,
	blocks:    [dynamic]NBeatsBlock, // the distinct blocks
	order:     []int, // the block applied at each position
	allocator: mem.Allocator,
	scope:     t.Graph_Scope,
}

@(private = "file")
_nb_block :: proc(
	c: NBeatsConfig,
	basis: NBeatsBasis,
	layer_size: int,
	allocator: mem.Allocator,
) -> NBeatsBlock {
	b := NBeatsBlock {
		basis  = basis,
		layers = make([dynamic]nn.LinearLayer, allocator),
	}
	prev := c.input_size
	for _ in 0 ..< c.layers {
		append(&b.layers, nn.linear_layer_new(prev, layer_size, allocator))
		prev = layer_size
	}
	theta_size: int
	switch basis {
	case .Generic:
		theta_size = c.input_size + c.horizon
	case .Trend:
		b.p = c.degree + 1
		theta_size = 2 * b.p
		for k in 0 ..< 2 {
			size := c.input_size if k == 0 else c.horizon
			m := l.matrix_new(f64, b.p, size, allocator)
			for i in 0 ..< b.p {
				for j in 0 ..< size {m.data[i * size + j] = math.pow(f64(j) / f64(size), f64(i))}
			}
			if k ==
			   0 {b.back_mat[0] = t.tensor_new(m, false, allocator)} else {b.fore_mat[0] = t.tensor_new(m, false, allocator)}
		}
	case .Seasonality:
		h, H := c.harmonics, c.horizon
		// frequencies: 0, then arange(h, h H / 2) / h
		freq := make([dynamic]f64, context.temp_allocator)
		append(&freq, 0)
		for v := f64(h); v < f64(h) * f64(H) / 2; v += 1 {append(&freq, v / f64(h))}
		b.p = len(freq)
		theta_size = 4 * b.p
		for k in 0 ..< 2 {
			size := c.input_size if k == 0 else H
			sgn := -1.0 if k == 0 else 1.0
			cm := l.matrix_new(f64, b.p, size, allocator)
			sm := l.matrix_new(f64, b.p, size, allocator)
			for i in 0 ..< b.p {
				for j in 0 ..< size {
					g := sgn * 2 * math.PI * (f64(j) / f64(H)) * freq[i]
					cm.data[i * size + j] = math.cos(g)
					sm.data[i * size + j] = math.sin(g)
				}
			}
			pair := [2]^t.Tensor {
				t.tensor_new(cm, false, allocator),
				t.tensor_new(sm, false, allocator),
			}
			if k == 0 {b.back_mat = pair} else {b.fore_mat = pair}
		}
		delete(freq)
	}
	b.theta = nn.linear_layer_new(prev, theta_size, allocator)
	return b
}

nbeats_new :: proc(config: NBeatsConfig, allocator := context.allocator) -> ^NBeats {
	c := config
	assert(
		c.input_size > 0 && c.horizon > 0 && c.blocks > 0 && c.layers > 0 && c.layer_size > 0,
		"nbeats_new: sizes must be positive",
	)
	m := new(NBeats, allocator)
	m.config = c
	m.allocator = allocator
	m.blocks = make([dynamic]NBeatsBlock, allocator)
	if !c.interpretable {
		m.order = make([]int, c.blocks, allocator)
		for i in 0 ..< c.blocks {
			append(&m.blocks, _nb_block(c, .Generic, c.layer_size, allocator))
			m.order[i] = i
		}
	} else {
		assert(
			c.seasonality_layer_size > 0 && c.harmonics > 0 && c.degree >= 0,
			"nbeats_new: interpretable sizes",
		)
		append(&m.blocks, _nb_block(c, .Trend, c.layer_size, allocator))
		append(&m.blocks, _nb_block(c, .Seasonality, c.seasonality_layer_size, allocator))
		m.order = make([]int, 2 * c.blocks, allocator)
		for i in 0 ..< 2 * c.blocks {m.order[i] = 0 if i < c.blocks else 1}
	}
	return m
}

nbeats_free :: proc(m: ^NBeats) {
	for &b in m.blocks {
		for &lin in b.layers {nn.linear_layer_free(&lin)}
		delete(b.layers)
		nn.linear_layer_free(&b.theta)
		for x in b.back_mat {if x != nil {t.tensor_free(x)}}
		for x in b.fore_mat {if x != nil {t.tensor_free(x)}}
	}
	delete(m.blocks)
	delete(m.order, m.allocator)
	t.graph_scope_destroy(&m.scope)
	free(m, m.allocator)
}

// Every trainable tensor (each distinct block: its layers' weights and biases, then the
// theta head's)
nbeats_parameters :: proc(m: ^NBeats, allocator := context.allocator) -> [dynamic]^t.Tensor {
	p := make([dynamic]^t.Tensor, allocator)
	for &b in m.blocks {
		for &lin in b.layers {append(&p, lin.weights, lin.bias)}
		append(&p, b.theta.weights, b.theta.bias)
	}
	return p
}

// Forecast [B, horizon] (graph) of windows x [B, input_size] (oldest value first)
nbeats_forward :: proc(m: ^NBeats, x: ^t.Tensor, alloc: mem.Allocator) -> ^t.Tensor {
	c := m.config
	B := x.data.rows
	assert(x.data.cols == c.input_size, "nbeats_forward: x must be [batch, input_size]")
	// time-reversed input and the naive start (last value) as graph constants
	res := t.tensor_new(l.matrix_new(f64, B, c.input_size, alloc), false, alloc)
	res.owned_by_graph = true
	fc := t.tensor_new(l.matrix_new(f64, B, c.horizon, alloc), false, alloc)
	fc.owned_by_graph = true
	for i in 0 ..< B {
		row := x.data.data[i * c.input_size:][:c.input_size]
		base := row[c.input_size - 1] if c.relative else 0
		for j in 0 ..< c.input_size {res.data.data[i * c.input_size + j] = row[c.input_size - 1 - j] - base}
		for j in 0 ..< c.horizon {fc.data.data[i * c.horizon + j] = row[c.input_size - 1]}
	}
	forecast: ^t.Tensor = fc
	for bi in m.order {
		b := &m.blocks[bi]
		h := res
		for &lin in b.layers {h = t.tensor_relu(nn.linear_forward(&lin, h))}
		theta := nn.linear_forward(&b.theta, h)
		back, fore: ^t.Tensor
		switch b.basis {
		case .Generic:
			back = t.tensor_slice_cols(theta, 0, c.input_size)
			fore = t.tensor_slice_cols(theta, theta.data.cols - c.horizon, c.horizon)
		case .Trend:
			back = t.tensor_matmul(t.tensor_slice_cols(theta, b.p, b.p), b.back_mat[0])
			fore = t.tensor_matmul(t.tensor_slice_cols(theta, 0, b.p), b.fore_mat[0])
		case .Seasonality:
			P := b.p
			back = t.tensor_add(
				t.tensor_matmul(t.tensor_slice_cols(theta, 2 * P, P), b.back_mat[0]),
				t.tensor_matmul(t.tensor_slice_cols(theta, 3 * P, P), b.back_mat[1]),
			)
			fore = t.tensor_add(
				t.tensor_matmul(t.tensor_slice_cols(theta, 0, P), b.fore_mat[0]),
				t.tensor_matmul(t.tensor_slice_cols(theta, P, P), b.fore_mat[1]),
			)
		}
		res = t.tensor_sub(res, back)
		forecast = t.tensor_add(forecast, fore)
	}
	return forecast
}

// MSE of the forecasts for windows x [B, input_size] and targets y [B, horizon]; with
// opt, one Adam step (gradients zeroed first); returns the loss before the step
nbeats_train_step :: proc(m: ^NBeats, x, y: ^t.Tensor, opt: ^nn.Adam) -> f64 {
	t.graph_begin(&m.scope)
	defer t.graph_end(&m.scope)
	alloc := t.graph_allocator(&m.scope)
	loss := t.tensor_mse_loss(nbeats_forward(m, x, alloc), y, alloc)
	nn.adam_zero_grad(opt)
	t.tensor_backward(loss)
	nn.adam_step(opt)
	return loss.data.data[0]
}

// Forecasts [B * horizon] row-major, no graph
nbeats_predict :: proc(m: ^NBeats, x: ^t.Tensor, allocator := context.allocator) -> []f64 {
	t.graph_begin(&m.scope, no_grad = true)
	defer t.graph_end(&m.scope)
	f := nbeats_forward(m, x, t.graph_allocator(&m.scope))
	out := make([]f64, len(f.data.data), allocator)
	copy(out, f.data.data)
	return out
}

// Every window of a series: X [n, input_size] (values s .. s+input_size-1), Y [n, horizon]
// (the next horizon values), n = len - input_size - horizon + 1
forecast_windows :: proc(
	series: []f64,
	input_size, horizon: int,
	allocator := context.allocator,
) -> (
	X, Y: l.Matrix(f64),
) {
	n := len(series) - input_size - horizon + 1
	assert(n > 0, "forecast_windows: series shorter than input_size + horizon")
	X = l.matrix_new(f64, n, input_size, allocator)
	Y = l.matrix_new(f64, n, horizon, allocator)
	for s in 0 ..< n {
		copy(X.data[s * input_size:][:input_size], series[s:][:input_size])
		copy(Y.data[s * horizon:][:horizon], series[s + input_size:][:horizon])
	}
	return
}
