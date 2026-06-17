package nn

import l "../linalg"
import t "../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// Activation Enum
// ============================================================================

Activation :: enum {
	None,
	ReLU,
	Sigmoid,
	Tanh,
	LeakyReLU,
}

// apply_activation applies an activation function to a tensor
apply_activation :: proc(x: ^t.Tensor, act: Activation) -> ^t.Tensor {
	switch act {
	case .None:
		return x
	case .ReLU:
		return t.tensor_relu(x)
	case .Sigmoid:
		return t.tensor_sigmoid(x)
	case .Tanh:
		return t.tensor_tanh(x)
	case .LeakyReLU:
		return t.tensor_leaky_relu(x, 0.01)
	}
	return x
}

// ============================================================================
// Conv2d Layer
// ============================================================================

Conv2dLayer :: struct {
	weight:       ^t.Tensor, // (C_out, C_in, kH, kW)
	bias:         ^t.Tensor, // (C_out,) or nil
	in_channels:  int,
	out_channels: int,
	kernel_size:  int,
	stride:       int,
	padding:      int,
}

conv2d_layer_new :: proc(
	in_channels: int,
	out_channels: int,
	kernel_size: int,
	stride: int = 1,
	padding: int = 0,
	use_bias: bool = true,
	allocator: mem.Allocator = context.allocator,
) -> Conv2dLayer {
	layer: Conv2dLayer
	layer.in_channels = in_channels
	layer.out_channels = out_channels
	layer.kernel_size = kernel_size
	layer.stride = stride
	layer.padding = padding

	// Initialize weights with Xavier initialization
	col_w := in_channels * kernel_size * kernel_size
	w_data := l.matrix_new(f64, out_channels, col_w, allocator)
	fan_in := in_channels * kernel_size * kernel_size
	fan_out := out_channels * kernel_size * kernel_size
	limit := math.sqrt(6.0 / f64(fan_in + fan_out))

	for i in 0 ..< len(w_data.data) {
		w_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit
	}

	layer.weight = t.tensor_new(w_data, true, allocator)
	layer.weight.shape = [4]int{out_channels, in_channels, kernel_size, kernel_size}

	// Initialize bias to zeros
	if use_bias {
		b_data := l.matrix_new(f64, 1, out_channels, allocator)
		layer.bias = t.tensor_new(b_data, true, allocator)
	}

	return layer
}

conv2d_layer_forward :: proc(layer: ^Conv2dLayer, x: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding)
}

conv2d_layer_free :: proc(layer: ^Conv2dLayer) {
	if layer.weight != nil {t.tensor_free(layer.weight)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

// ============================================================================
// MaxPool2d Layer
// ============================================================================

MaxPool2dLayer :: struct {
	kernel_size: int,
	stride:      int,
}

maxpool2d_layer_new :: proc(kernel_size: int, stride: int = 0) -> MaxPool2dLayer {
	layer: MaxPool2dLayer
	layer.kernel_size = kernel_size
	layer.stride = stride == 0 ? kernel_size : stride
	return layer
}

maxpool2d_layer_forward :: proc(layer: ^MaxPool2dLayer, x: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_max_pool2d(x, layer.kernel_size, layer.kernel_size, layer.stride)
}
AvgPool2dLayer :: struct {
	kernel_size: int,
	stride:      int,
}

avgpool2d_layer_new :: proc(kernel_size: int, stride: int = 0) -> AvgPool2dLayer {
	layer: AvgPool2dLayer
	layer.kernel_size = kernel_size
	layer.stride = stride == 0 ? kernel_size : stride
	return layer
}

// ============================================================================
// Dropout Layer (NEW)
// ============================================================================

DropoutLayer :: struct {
	drop_prob: f64,
}

dropout_layer_new :: proc(drop_prob: f64 = 0.5) -> DropoutLayer {
	layer: DropoutLayer
	layer.drop_prob = drop_prob
	return layer
}

BatchNorm2dLayer :: struct {
	num_features: int,
	eps:          f64,
	momentum:     f64,
	weight:       ^t.Tensor, // gamma (scale)
	bias:         ^t.Tensor, // beta (shift)
	running_mean: ^t.Tensor, // Saved for eval mode
	running_var:  ^t.Tensor, // Saved for eval mode
}

batch_norm_2d_layer_new :: proc(
	num_features: int,
	eps: f64 = 1e-5,
	momentum: f64 = 0.1,
	allocator: mem.Allocator = context.allocator,
) -> BatchNorm2dLayer {
	layer: BatchNorm2dLayer
	layer.num_features = num_features
	layer.eps = eps
	layer.momentum = momentum

	// Initialize weight (gamma) to 1.0
	w_data := l.matrix_new(f64, 1, num_features, allocator)
	for i in 0 ..< num_features {w_data.data[i] = 1.0}
	layer.weight = t.tensor_new(w_data, true, allocator)

	// Initialize bias (beta) to 0.0
	b_data := l.matrix_new(f64, 1, num_features, allocator)
	// (already zeroed by matrix_new)
	layer.bias = t.tensor_new(b_data, true, allocator)

	// Running stats (not requiring grad)
	rm_data := l.matrix_new(f64, 1, num_features, allocator)
	layer.running_mean = t.tensor_new(rm_data, false, allocator)

	rv_data := l.matrix_new(f64, 1, num_features, allocator)
	layer.running_var = t.tensor_new(rv_data, false, allocator)

	return layer
}

batch_norm_2d_layer_free :: proc(layer: ^BatchNorm2dLayer) {
	if layer.weight != nil {t.tensor_free(layer.weight)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
	if layer.running_mean != nil {t.tensor_free(layer.running_mean)}
	if layer.running_var != nil {t.tensor_free(layer.running_var)}
}
// ============================================================================
// RNN Layer
// ============================================================================

RNNLayer :: struct {
	input_size:  int,
	hidden_size: int,
	w_ih:        ^t.Tensor, // [input_size, hidden_size]
	w_hh:        ^t.Tensor, // [hidden_size, hidden_size]
	bias:        ^t.Tensor, // [1, hidden_size]
}

rnn_layer_new :: proc(
	input_size: int,
	hidden_size: int,
	allocator: mem.Allocator = context.allocator,
) -> RNNLayer {
	layer: RNNLayer
	layer.input_size = input_size
	layer.hidden_size = hidden_size

	// Xavier initialization for w_ih
	w_ih_data := l.matrix_new(f64, input_size, hidden_size, allocator)
	limit_ih := math.sqrt(6.0 / f64(input_size + hidden_size))
	for i in 0 ..< len(w_ih_data.data) {
		w_ih_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_ih
	}
	layer.w_ih = t.tensor_new(w_ih_data, true, allocator)
	// ✅ FIX: Set shape so shape[0]=input_size, shape[1]=hidden_size
	layer.w_ih.shape = [4]int{input_size, hidden_size, 1, 1}

	// Orthogonal/Xavier initialization for w_hh
	w_hh_data := l.matrix_new(f64, hidden_size, hidden_size, allocator)
	limit_hh := math.sqrt(6.0 / f64(hidden_size + hidden_size))
	for i in 0 ..< len(w_hh_data.data) {
		w_hh_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_hh
	}
	layer.w_hh = t.tensor_new(w_hh_data, true, allocator)
	// ✅ FIX: Set shape so shape[0]=hidden_size, shape[1]=hidden_size
	layer.w_hh.shape = [4]int{hidden_size, hidden_size, 1, 1}

	// Zero bias
	bias_data := l.matrix_new(f64, 1, hidden_size, allocator)
	layer.bias = t.tensor_new(bias_data, true, allocator)
	// ✅ FIX: Set shape so shape[1]=hidden_size
	layer.bias.shape = [4]int{1, hidden_size, 1, 1}

	return layer
}
rnn_layer_free :: proc(layer: ^RNNLayer) {
	if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
	if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

rnn_layer_forward :: proc(layer: ^RNNLayer, x: ^t.Tensor, h_0: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_rnn(x, h_0, layer.w_ih, layer.w_hh, layer.bias)
}
GRULayer :: struct {
	input_size:  int,
	hidden_size: int,
	w_ih:        ^t.Tensor,
	w_hh:        ^t.Tensor,
	bias:        ^t.Tensor,
}

gru_layer_new :: proc(
	input_size: int,
	hidden_size: int,
	allocator: mem.Allocator = context.allocator,
) -> GRULayer {
	layer: GRULayer
	layer.input_size = input_size
	layer.hidden_size = hidden_size
	H3 := 3 * hidden_size

	w_ih_data := l.matrix_new(f64, input_size, H3, allocator)
	limit_ih := math.sqrt(6.0 / f64(input_size + hidden_size))
	for i in 0 ..< len(w_ih_data.data) {
		w_ih_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_ih
	}
	layer.w_ih = t.tensor_new(w_ih_data, true, allocator)
	layer.w_ih.shape = [4]int{input_size, H3, 1, 1}

	w_hh_data := l.matrix_new(f64, hidden_size, H3, allocator)
	limit_hh := math.sqrt(6.0 / f64(hidden_size + hidden_size))
	for i in 0 ..< len(w_hh_data.data) {
		w_hh_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_hh
	}
	layer.w_hh = t.tensor_new(w_hh_data, true, allocator)
	layer.w_hh.shape = [4]int{hidden_size, H3, 1, 1}

	bias_data := l.matrix_new(f64, 1, H3, allocator)
	layer.bias = t.tensor_new(bias_data, true, allocator)
	layer.bias.shape = [4]int{1, H3, 1, 1}

	return layer
}

gru_layer_free :: proc(layer: ^GRULayer) {
	if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
	if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

gru_layer_forward :: proc(layer: ^GRULayer, x: ^t.Tensor, h_0: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_gru(x, h_0, layer.w_ih, layer.w_hh, layer.bias)
}
LSTMLayer :: struct {
	input_size:  int,
	hidden_size: int,
	w_ih:        ^t.Tensor,
	w_hh:        ^t.Tensor,
	bias:        ^t.Tensor,
}

lstm_layer_new :: proc(
	input_size: int,
	hidden_size: int,
	allocator: mem.Allocator = context.allocator,
) -> LSTMLayer {
	layer: LSTMLayer
	layer.input_size = input_size
	layer.hidden_size = hidden_size
	H4 := 4 * hidden_size

	w_ih_data := l.matrix_new(f64, input_size, H4, allocator)
	limit_ih := math.sqrt(6.0 / f64(input_size + hidden_size))
	for i in 0 ..< len(w_ih_data.data) {
		w_ih_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_ih
	}
	layer.w_ih = t.tensor_new(w_ih_data, true, allocator)
	layer.w_ih.shape = [4]int{input_size, H4, 1, 1}

	w_hh_data := l.matrix_new(f64, hidden_size, H4, allocator)
	limit_hh := math.sqrt(6.0 / f64(hidden_size + hidden_size))
	for i in 0 ..< len(w_hh_data.data) {
		w_hh_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit_hh
	}
	layer.w_hh = t.tensor_new(w_hh_data, true, allocator)
	layer.w_hh.shape = [4]int{hidden_size, H4, 1, 1}

	bias_data := l.matrix_new(f64, 1, H4, allocator)
	layer.bias = t.tensor_new(bias_data, true, allocator)
	layer.bias.shape = [4]int{1, H4, 1, 1}

	return layer
}

lstm_layer_free :: proc(layer: ^LSTMLayer) {
	if layer.w_ih != nil {t.tensor_free(layer.w_ih)}
	if layer.w_hh != nil {t.tensor_free(layer.w_hh)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

lstm_layer_forward :: proc(
	layer: ^LSTMLayer,
	x: ^t.Tensor,
	h_0: ^t.Tensor,
	c_0: ^t.Tensor,
) -> ^t.Tensor {
	return t.tensor_lstm(x, h_0, c_0, layer.w_ih, layer.w_hh, layer.bias)
}
// ============================================================================
// Embedding Layer
// ============================================================================

EmbeddingLayer :: struct {
	num_embeddings: int,
	embedding_dim:  int,
	weight:         ^t.Tensor, // [vocab_size, embedding_dim]
}

embedding_layer_new :: proc(
	num_embeddings: int,
	embedding_dim: int,
	allocator: mem.Allocator = context.allocator,
) -> EmbeddingLayer {
	layer: EmbeddingLayer
	layer.num_embeddings = num_embeddings
	layer.embedding_dim = embedding_dim

	w_data := l.matrix_new(f64, num_embeddings, embedding_dim, allocator)

	// Standard embedding initialization: Uniform(-1/sqrt(dim), 1/sqrt(dim))
	limit := 1.0 / math.sqrt(f64(embedding_dim))
	for i in 0 ..< len(w_data.data) {
		w_data.data[i] = (rand.float64() * 2.0 - 1.0) * limit
	}

	layer.weight = t.tensor_new(w_data, true, allocator)
	layer.weight.shape = [4]int{num_embeddings, embedding_dim, 1, 1}

	return layer
}

embedding_layer_free :: proc(layer: ^EmbeddingLayer) {
	if layer.weight != nil {t.tensor_free(layer.weight)}
}

embedding_layer_forward :: proc(layer: ^EmbeddingLayer, x: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_embedding(x, layer.weight)
}
// ============================================================================
// Positional Encoding (Sinusoidal)
// ============================================================================

PositionalEncoding :: struct {
	max_seq_len: int,
	embed_dim:   int,
	pe:          []f64, // Precomputed [max_seq_len, embed_dim]
	allocator:   mem.Allocator,
}

positional_encoding_new :: proc(
	max_seq_len: int,
	embed_dim: int,
	allocator: mem.Allocator = context.allocator,
) -> PositionalEncoding {
	pe := make([]f64, max_seq_len * embed_dim, allocator)

	// Precompute the denominator: 10000^(2i / embed_dim)
	// Using log/exp for numerical stability and efficiency
	div_term := make([]f64, embed_dim / 2, allocator)
	for i in 0 ..< embed_dim / 2 {
		div_term[i] = math.exp(-f64(2 * i) * math.ln_f64(10000.0) / f64(embed_dim))
	}

	for pos in 0 ..< max_seq_len {
		pos_f := f64(pos)
		for i in 0 ..< embed_dim / 2 {
			pe_idx_even := pos * embed_dim + 2 * i
			pe_idx_odd := pos * embed_dim + 2 * i + 1

			pe[pe_idx_even] = math.sin(pos_f * div_term[i])
			if 2 * i + 1 < embed_dim {
				pe[pe_idx_odd] = math.cos(pos_f * div_term[i])
			}
		}
	}
	delete(div_term, allocator)

	return PositionalEncoding {
		max_seq_len = max_seq_len,
		embed_dim = embed_dim,
		pe = pe,
		allocator = allocator,
	}
}

positional_encoding_free :: proc(pe: ^PositionalEncoding) {
	if pe.pe != nil {
		delete(pe.pe, pe.allocator)
	}
}

// positional_encoding_forward adds the precomputed PE to the input embeddings
// ✅ SIMD-optimized using vec_add_simd
positional_encoding_forward :: proc(pe: ^PositionalEncoding, x: ^t.Tensor) -> ^t.Tensor {
	batch := x.shape[0]
	seq_len := x.shape[1]
	embed_dim := x.shape[2]

	if seq_len > pe.max_seq_len {
		panic(fmt.aprintf("Sequence length %d exceeds max_seq_len %d", seq_len, pe.max_seq_len))
	}
	if embed_dim != pe.embed_dim {
		panic(fmt.aprintf("Embed dim %d does not match PE embed_dim %d", embed_dim, pe.embed_dim))
	}

	out_data := l.matrix_new(f64, 1, batch * seq_len * embed_dim, x.allocator)

	// 1. Copy input embeddings to output
	copy(out_data.data, x.data.data)

	// 2. ✅ SIMD-optimized addition of PE to each batch element
	pe_slice := pe.pe[0:seq_len * embed_dim]
	for b in 0 ..< batch {
		batch_offset := b * seq_len * embed_dim
		l.vec_add_simd(
			out_data.data[batch_offset:batch_offset + seq_len * embed_dim],
			pe_slice,
			out_data.data[batch_offset:batch_offset + seq_len * embed_dim],
		)
	}

	out := t.tensor_new(out_data, x.requires_grad, x.allocator)
	out.shape = x.shape
	return out
}
// ============================================================================
// Multi-Head Attention Layer
// ============================================================================

MultiHeadAttentionLayer :: struct {
	d_model:   int,
	num_heads: int,
	head_dim:  int,
	q_proj:    LinearLayer,
	k_proj:    LinearLayer,
	v_proj:    LinearLayer,
	out_proj:  LinearLayer,
}

multi_head_attention_layer_new :: proc(
	d_model: int,
	num_heads: int,
	allocator: mem.Allocator = context.allocator,
) -> MultiHeadAttentionLayer {
	layer: MultiHeadAttentionLayer
	layer.d_model = d_model
	layer.num_heads = num_heads
	layer.head_dim = d_model / num_heads

	layer.q_proj = linear_layer_new(d_model, d_model, allocator)
	layer.k_proj = linear_layer_new(d_model, d_model, allocator)
	layer.v_proj = linear_layer_new(d_model, d_model, allocator)
	layer.out_proj = linear_layer_new(d_model, d_model, allocator)

	return layer
}

multi_head_attention_layer_free :: proc(layer: ^MultiHeadAttentionLayer) {
	linear_layer_free(&layer.q_proj)
	linear_layer_free(&layer.k_proj)
	linear_layer_free(&layer.v_proj)
	linear_layer_free(&layer.out_proj)
}

multi_head_attention_layer_forward :: proc(
	layer: ^MultiHeadAttentionLayer,
	x: ^t.Tensor,
) -> ^t.Tensor {
	batch := x.shape[0]
	seq_len := x.shape[1]

	// 1. Projections
	q := linear_forward(&layer.q_proj, x)
	k := linear_forward(&layer.k_proj, x)
	v := linear_forward(&layer.v_proj, x)

	// 2. Permute to [batch * num_heads, seq_len, head_dim]
	q_perm := t.tensor_permute_mha(q, batch, seq_len, layer.num_heads, layer.head_dim)
	k_perm := t.tensor_permute_mha(k, batch, seq_len, layer.num_heads, layer.head_dim)
	v_perm := t.tensor_permute_mha(v, batch, seq_len, layer.num_heads, layer.head_dim)

	// 3. Scaled Dot-Product Attention (processes all heads in parallel!)
	att := t.tensor_scaled_dot_product_attention(q_perm, k_perm, v_perm)

	// 4. Inverse permute back to [batch, seq_len, d_model]
	att_inv := t.tensor_permute_mha_inverse(att, batch, seq_len, layer.num_heads, layer.head_dim)

	// 5. Output projection
	out := linear_forward(&layer.out_proj, att_inv)

	// Cleanup intermediate tensors (autograd handles graph, but we free the intermediates)
	// t.tensor_free(q); t.tensor_free(k); t.tensor_free(v)
	// t.tensor_free(q_perm); t.tensor_free(k_perm); t.tensor_free(v_perm)
	// t.tensor_free(att); t.tensor_free(att_inv)

	return out
}
// ============================================================================
// Layer Normalization Layer
// ============================================================================

LayerNormLayer :: struct {
	d_model: int,
	eps:     f64,
	gamma:   ^t.Tensor, // [1, d_model] (scale)
	beta:    ^t.Tensor, // [1, d_model] (shift)
}

layer_norm_layer_new :: proc(
	d_model: int,
	eps: f64 = 1e-5,
	allocator: mem.Allocator = context.allocator,
) -> LayerNormLayer {
	layer: LayerNormLayer
	layer.d_model = d_model
	layer.eps = eps

	// Initialize gamma to 1.0
	gamma_data := l.matrix_new(f64, 1, d_model, allocator)
	for i in 0 ..< d_model {gamma_data.data[i] = 1.0}
	layer.gamma = t.tensor_new(gamma_data, true, allocator)
	layer.gamma.shape = [4]int{1, d_model, 1, 1}

	// Initialize beta to 0.0
	beta_data := l.matrix_new(f64, 1, d_model, allocator)
	layer.beta = t.tensor_new(beta_data, true, allocator)
	layer.beta.shape = [4]int{1, d_model, 1, 1}

	return layer
}

layer_norm_layer_free :: proc(layer: ^LayerNormLayer) {
	if layer.gamma != nil {t.tensor_free(layer.gamma)}
	if layer.beta != nil {t.tensor_free(layer.beta)}
}

layer_norm_layer_forward :: proc(layer: ^LayerNormLayer, x: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_layer_norm(x, layer.gamma, layer.beta, layer.eps)
}
