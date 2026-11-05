package ml_finance

import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:mem"

// ============================================================================
// DeepLOB-style classifier of limit order book windows
// ============================================================================
// After Zhang, Zohren & Roberts (2019), simplified: input [batch, 3, T, levels] (mid,
// spread and volume imbalance per level, see compute_lob_microstructure), two 3x3
// convolutions, 2x2 max pooling (T and levels halve), an inception block (1x1, 1x1-3x3,
// 1x1-5x5 branches), a sequence layer (LSTM, GRU or Mamba) over the T / 2 steps, and a
// linear head on its last step: logits of down / stationary / up. The original uses 1x2
// convolutions that pair prices with sizes and temporal (k x 1) inception branches.

DeepLOBConfig :: struct {
	time_steps:   int,
	price_levels: int, // >= 2 (pooled by 2)
	num_classes:  int,
	hidden_dim:   int,
	dropout_prob: f64, // on the last step, training only; 0 means the default 0.3
	seq_backend:  nn.SeqBackend,
}

DeepLOB :: struct {
	cnn:       ^nn.Sequential,
	inception: InceptionBlock,
	seq_layer: ^nn.Sequential,
	fc_head:   nn.LinearLayer,
	config:    DeepLOBConfig,
	allocator: mem.Allocator,
	training:  bool,
}

// ============================================================================
// Inception Block (Multi-Scale Feature Extraction)
// ============================================================================

InceptionBlock :: struct {
	b1_1x1: nn.Conv2dLayer,
	b2_1x1: nn.Conv2dLayer,
	b2_3x3: nn.Conv2dLayer,
	b3_1x1: nn.Conv2dLayer,
	b3_5x5: nn.Conv2dLayer,
}

inception_block_new :: proc(
	in_channels: int,
	out_channels: int,
	allocator: mem.Allocator = context.allocator,
) -> InceptionBlock {
	block: InceptionBlock

	// Split channels to ensure the sum exactly equals out_channels
	c1 := out_channels / 3
	c2 := out_channels / 3
	c3 := out_channels - c1 - c2 // Guarantees c1 + c2 + c3 == out_channels

	block.b1_1x1 = nn.conv2d_layer_new(in_channels, c1, 1, 1, 0, true, allocator)
	block.b2_1x1 = nn.conv2d_layer_new(in_channels, c2, 1, 1, 0, true, allocator)
	block.b2_3x3 = nn.conv2d_layer_new(c2, c2, 3, 1, 1, true, allocator)
	block.b3_1x1 = nn.conv2d_layer_new(in_channels, c3, 1, 1, 0, true, allocator)
	block.b3_5x5 = nn.conv2d_layer_new(c3, c3, 5, 1, 2, true, allocator)

	return block
}

inception_block_free :: proc(block: ^InceptionBlock) {
	nn.conv2d_layer_free(&block.b1_1x1)
	nn.conv2d_layer_free(&block.b2_1x1)
	nn.conv2d_layer_free(&block.b2_3x3)
	nn.conv2d_layer_free(&block.b3_1x1)
	nn.conv2d_layer_free(&block.b3_5x5)
}

inception_block_forward :: proc(block: ^InceptionBlock, x: ^t.Tensor) -> ^t.Tensor {
	b1 := t.tensor_conv2d(
		x,
		block.b1_1x1.weight,
		block.b1_1x1.bias,
		block.b1_1x1.stride,
		block.b1_1x1.padding,
	)
	b1 = t.tensor_relu(b1)

	b2 := t.tensor_conv2d(
		x,
		block.b2_1x1.weight,
		block.b2_1x1.bias,
		block.b2_1x1.stride,
		block.b2_1x1.padding,
	)
	b2 = t.tensor_relu(b2)
	b2 = t.tensor_conv2d(
		b2,
		block.b2_3x3.weight,
		block.b2_3x3.bias,
		block.b2_3x3.stride,
		block.b2_3x3.padding,
	)
	b2 = t.tensor_relu(b2)

	b3 := t.tensor_conv2d(
		x,
		block.b3_1x1.weight,
		block.b3_1x1.bias,
		block.b3_1x1.stride,
		block.b3_1x1.padding,
	)
	b3 = t.tensor_relu(b3)
	b3 = t.tensor_conv2d(
		b3,
		block.b3_5x5.weight,
		block.b3_5x5.bias,
		block.b3_5x5.stride,
		block.b3_5x5.padding,
	)
	b3 = t.tensor_relu(b3)

	branches: [3]^t.Tensor
	branches[0] = b1
	branches[1] = b2
	branches[2] = b3

	out := t.tensor_concat(branches[:], 1)

	return out // b1, b2, b3 belong to the graph
}

// ============================================================================
// Initialization & Lifecycle
// ============================================================================

deeplob_new :: proc(
	config: DeepLOBConfig,
	allocator: mem.Allocator = context.allocator,
) -> ^DeepLOB {
	assert(config.time_steps >= 2 && config.price_levels >= 2, "deeplob_new: the 2x2 pooling needs 2+ time steps and levels")
	model := new(DeepLOB, allocator)
	model.config = config
	model.allocator = allocator
	model.training = true

	if model.config.dropout_prob == 0.0 {
		model.config.dropout_prob = 0.3
	}

	model.cnn = nn.sequential_new(allocator)
	nn.sequential_add(model.cnn, nn.conv2d_layer_new(3, 16, 3, 1, 1, true, allocator))
	nn.sequential_add(model.cnn, nn.Activation.ReLU)
	nn.sequential_add(model.cnn, nn.conv2d_layer_new(16, 32, 3, 1, 1, true, allocator))
	nn.sequential_add(model.cnn, nn.Activation.ReLU)

	inception_out_channels := 32
	model.inception = inception_block_new(32, inception_out_channels, allocator)
	nn.sequential_add(model.cnn, nn.maxpool2d_layer_new(2, 2))

	// sequence layer (LSTM / GRU / Mamba) over the pooled time steps
	seq_feat_dim := inception_out_channels * (config.price_levels / 2)
	model.seq_layer = nn.sequential_new(allocator)
	nn.sequential_add_seq_block(
		model.seq_layer,
		config.seq_backend,
		seq_feat_dim,
		config.hidden_dim,
		16, // d_state for Mamba
		allocator,
	)

	model.fc_head = nn.linear_layer_new(config.hidden_dim, config.num_classes, allocator)
	return model
}


deeplob_free :: proc(model: ^DeepLOB) {
	if model.cnn != nil {nn.sequential_free(model.cnn)}
	inception_block_free(&model.inception)
	if model.seq_layer != nil {nn.sequential_free(model.seq_layer)}
	nn.linear_layer_free(&model.fc_head)
	free(model, model.allocator)
}

deeplob_train :: proc(model: ^DeepLOB) {
	model.training = true
}

deeplob_eval :: proc(model: ^DeepLOB) {
	model.training = false
}
// ============================================================================
// Forward Pass
// ============================================================================

deeplob_forward :: proc(model: ^DeepLOB, input: ^t.Tensor) -> ^t.Tensor {
	alloc := model.allocator
	batch := input.shape[0]

	cnn_out := nn.sequential_forward(model.cnn, input)
	inception_out := inception_block_forward(&model.inception, cnn_out)

	c_out := inception_out.shape[1]
	t_out := inception_out.shape[2]
	l_out := inception_out.shape[3]

	// Permute to [Batch, Time, Features] for the sequence layer
	lstm_in := t.tensor_permute_lob(inception_out, batch, c_out, t_out, l_out, alloc)

	// zero initial states
	seq_out := nn.sequential_forward(model.seq_layer, lstm_in)

	// The last time step (the same layout for LSTM, GRU and Mamba), as a graph
	// op so that gradients reach the sequence layer, inception block and CNN
	// (a plain copy cut the graph here: only fc_head trained)
	// [batch, hidden] rows, so the logits are a [batch, num_classes] matrix
	last_step := nn.as_matrix(
		nn.sequence_step(seq_out, t_out - 1, alloc),
		batch,
		model.config.hidden_dim,
		alloc,
	)

	if model.training && model.config.dropout_prob > 0.0 {
		last_step = t.tensor_dropout(last_step, model.config.dropout_prob, model.training)
	}

	logits := nn.linear_forward(&model.fc_head, last_step)
	return logits
}

// ============================================================================
// Optimizer Integration
// ============================================================================

deeplob_add_to_adam :: proc(model: ^DeepLOB, opt: ^nn.Adam) {
	nn.sequential_add_to_adam(model.cnn, opt)

	nn.adam_add_param(opt, model.inception.b1_1x1.weight)
	if model.inception.b1_1x1.bias != nil {nn.adam_add_param(opt, model.inception.b1_1x1.bias)}

	nn.adam_add_param(opt, model.inception.b2_1x1.weight)
	if model.inception.b2_1x1.bias != nil {nn.adam_add_param(opt, model.inception.b2_1x1.bias)}
	nn.adam_add_param(opt, model.inception.b2_3x3.weight)
	if model.inception.b2_3x3.bias != nil {nn.adam_add_param(opt, model.inception.b2_3x3.bias)}

	nn.adam_add_param(opt, model.inception.b3_1x1.weight)
	if model.inception.b3_1x1.bias != nil {nn.adam_add_param(opt, model.inception.b3_1x1.bias)}
	nn.adam_add_param(opt, model.inception.b3_5x5.weight)
	if model.inception.b3_5x5.bias != nil {nn.adam_add_param(opt, model.inception.b3_5x5.bias)}


	nn.sequential_add_to_adam(model.seq_layer, opt)
	nn.adam_add_param(opt, model.fc_head.weights)
	if model.fc_head.bias != nil {
		nn.adam_add_param(opt, model.fc_head.bias)
	}
}
// raw_data [n_windows, 4, T, levels] (bid price, bid size, ask price, ask size) ->
// [n_windows, 3, T, levels]: mid (ask + bid) / 2, spread ask - bid, volume imbalance
// (bid size - ask size) / (bid size + ask size) of each level
compute_lob_microstructure :: proc(
	raw_data: []f64,
	n_windows: int,
	time_steps: int,
	price_levels: int,
	allocator: mem.Allocator,
) -> []f64 {
	out_C := 3
	in_C := 4
	total_elements := n_windows * out_C * time_steps * price_levels
	out_data := make([]f64, total_elements, allocator)

	TL := time_steps * price_levels

	for n: int = 0; n < n_windows; n += 1 {
		for t: int = 0; t < time_steps; t += 1 {
			for l: int = 0; l < price_levels; l += 1 {
				in_base := n * (in_C * TL) + t * price_levels + l
				out_base := n * (out_C * TL) + t * price_levels + l

				bid_p := raw_data[in_base + 0 * TL]
				bid_v := raw_data[in_base + 1 * TL]
				ask_p := raw_data[in_base + 2 * TL]
				ask_v := raw_data[in_base + 3 * TL]

				// 1. Mid-Price: (Ask + Bid) / 2
				out_data[out_base + 0 * TL] = (ask_p + bid_p) * 0.5

				// 2. Spread: Ask - Bid
				out_data[out_base + 1 * TL] = ask_p - bid_p

				// 3. Order Book Imbalance (OBI): (BidVol - AskVol) / (BidVol + AskVol)
				vol_sum := bid_v + ask_v
				if vol_sum != 0.0 {
					out_data[out_base + 2 * TL] = (bid_v - ask_v) / vol_sum
				} else {
					out_data[out_base + 2 * TL] = 0.0
				}
			}
		}
	}
	return out_data
}

// ============================================================================
// Windows and labels from a LOBSTER order book
// ============================================================================
// book: n_rows events in LOBSTER layout, row-major [n_rows, 4 * levels]: per level
// (ask price, ask size, bid price, bid size). Window w covers events w .. w + T - 1 and is
// labelled with the move after it (Zhang, Zohren & Roberts 2019): m_t = level-1 mid of the
// last event, m_+ = mean level-1 mid of the next `horizon` events, l = m_+ / m_t - 1;
// 2 (up) if l > threshold, 0 (down) if l < -threshold, else 1. future_returns: m_{t+h} / m_t - 1
// (h = horizon). The inputs [N, 3, T, levels] hold only events of the window: mid, spread
// and imbalance per level (compute_lob_microstructure), not normalised; fit a
// DeepLOBNorm on the training windows and apply it to all. Windows: n_rows - T - horizon + 1.
deeplob_lobster_windows :: proc(
	book: []f64,
	n_rows, levels, time_steps, horizon: int,
	threshold: f64,
	allocator: mem.Allocator = context.allocator,
) -> (
	inputs: ^t.Tensor,
	labels: []int,
	future_returns: []f64,
) {
	L, T := levels, time_steps
	assert(len(book) == n_rows * 4 * L, "deeplob_lobster_windows: book must be [n_rows, 4 * levels]")
	assert(T >= 1 && horizon >= 1, "deeplob_lobster_windows: time_steps and horizon must be positive")
	N := n_rows - T - horizon + 1
	if N <= 0 {return}
	mid :: proc(book: []f64, row, L: int) -> f64 {
		return 0.5 * (book[row * 4 * L + 0] + book[row * 4 * L + 2])
	}
	TL := T * L
	raw := make([]f64, N * 4 * TL, allocator)
	defer delete(raw, allocator)
	labels = make([]int, N, allocator)
	future_returns = make([]f64, N, allocator)
	for w in 0 ..< N {
		for k in 0 ..< T {
			r := book[(w + k) * 4 * L:][:4 * L]
			for lv in 0 ..< L {
				base := w * 4 * TL + k * L + lv
				raw[base + 0 * TL] = r[4 * lv + 2] // bid price
				raw[base + 1 * TL] = r[4 * lv + 3] // bid size
				raw[base + 2 * TL] = r[4 * lv + 0] // ask price
				raw[base + 3 * TL] = r[4 * lv + 1] // ask size
			}
		}
		last := w + T - 1
		m_t := mid(book, last, L)
		m_plus := 0.0
		for j in 1 ..= horizon {m_plus += mid(book, last + j, L) / f64(horizon)}
		change := m_plus / m_t - 1
		labels[w] = 2 if change > threshold else (0 if change < -threshold else 1)
		future_returns[w] = mid(book, last + horizon, L) / m_t - 1
	}
	micro := compute_lob_microstructure(raw, N, T, L, allocator)
	inputs = t.tensor_new(
		l.Matrix(f64){rows = 1, cols = len(micro), data = micro, allocator = allocator},
		false,
		allocator,
	)
	inputs.shape = {N, 3, T, L}
	return
}

// z-score of each channel and level with the mean and standard deviation over the
// training windows and their time steps (as DeepLOB normalises with past data; z-scoring
// each window over its own steps would erase a persistent imbalance)
DeepLOBNorm :: struct {
	mean, std: []f64, // [3 * levels], channel-major
}

deeplob_norm_fit :: proc(x: ^t.Tensor, allocator: mem.Allocator = context.allocator) -> DeepLOBNorm {
	N, C, T, L := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	nm := DeepLOBNorm{make([]f64, C * L, allocator), make([]f64, C * L, allocator)}
	cnt := f64(N * T)
	for n in 0 ..< N {for c in 0 ..< C {for k in 0 ..< T {for lv in 0 ..< L {
		nm.mean[c * L + lv] += x.data.data[((n * C + c) * T + k) * L + lv] / cnt
	}}}}
	for n in 0 ..< N {for c in 0 ..< C {for k in 0 ..< T {for lv in 0 ..< L {
		d := x.data.data[((n * C + c) * T + k) * L + lv] - nm.mean[c * L + lv]
		nm.std[c * L + lv] += d * d / cnt
	}}}}
	for &s, i in nm.std { 	// a constant feature (up to rounding) is only centred
		s = math.sqrt(s)
		if s <= 1e-12 * max(abs(nm.mean[i]), 1) {s = 1}
	}
	return nm
}

// In place: x <- (x - mean) / std per channel and level
deeplob_norm_apply :: proc(x: ^t.Tensor, nm: DeepLOBNorm) {
	N, C, T, L := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	assert(len(nm.mean) == C * L, "deeplob_norm_apply: fitted on another number of levels")
	for n in 0 ..< N {for c in 0 ..< C {for k in 0 ..< T {for lv in 0 ..< L {
		i := ((n * C + c) * T + k) * L + lv
		x.data.data[i] = (x.data.data[i] - nm.mean[c * L + lv]) / nm.std[c * L + lv]
	}}}}
}

deeplob_norm_free :: proc(nm: ^DeepLOBNorm, allocator: mem.Allocator = context.allocator) {
	delete(nm.mean, allocator)
	delete(nm.std, allocator)
}
