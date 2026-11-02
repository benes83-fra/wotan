package nn

import l "../linalg"
import t "../tensor"
import "core:mem"

// ============================================================================
// Conv1d Layer
// ============================================================================
// 1-D convolution over channels-last sequences [batch, T, C_in] → [batch,
// T_out, C_out] (see tensor_conv1d). Causal: padded on the left only by
// dilation·(K - 1), so T_out = T and output t depends on inputs up to t.

Conv1dLayer :: struct {
	weight:       ^t.Tensor, // [C_out, C_in, K]
	bias:         ^t.Tensor, // [1, C_out] or nil
	in_channels:  int,
	out_channels: int,
	kernel_size:  int,
	dilation:     int,
	pad_left:     int,
	pad_right:    int,
}

// padding: zeros on both sides (ignored when causal)
conv1d_layer_new :: proc(
	in_channels: int,
	out_channels: int,
	kernel_size: int,
	dilation: int = 1,
	padding: int = 0,
	causal: bool = false,
	use_bias: bool = true,
	allocator: mem.Allocator = context.allocator,
) -> Conv1dLayer {
	layer := Conv1dLayer {
		in_channels  = in_channels,
		out_channels = out_channels,
		kernel_size  = kernel_size,
		dilation     = dilation,
		pad_left     = dilation * (kernel_size - 1) if causal else padding,
		pad_right    = 0 if causal else padding,
	}
	// Xavier initialization (as Conv2d), zero bias
	fan_in, fan_out := in_channels * kernel_size, out_channels * kernel_size
	w_data := l.matrix_new(f64, out_channels, fan_in, allocator)
	xavier_uniform(w_data.data, fan_in, fan_out)
	layer.weight = t.tensor_new(w_data, true, allocator)
	layer.weight.shape = {out_channels, in_channels, kernel_size, 1}
	if use_bias {
		layer.bias = t.tensor_new(l.matrix_new(f64, 1, out_channels, allocator), true, allocator)
	}
	return layer
}

conv1d_layer_forward :: proc(layer: ^Conv1dLayer, x: ^t.Tensor) -> ^t.Tensor {
	return t.tensor_conv1d(
		x,
		layer.weight,
		layer.bias,
		layer.dilation,
		layer.pad_left,
		layer.pad_right,
	)
}

conv1d_layer_free :: proc(layer: ^Conv1dLayer) {
	if layer.weight != nil {t.tensor_free(layer.weight)}
	if layer.bias != nil {t.tensor_free(layer.bias)}
}

// ============================================================================
// TCN (temporal convolutional network, Bai, Kolter & Koltun 2018)
// ============================================================================
// A residual block of two dilated causal convolutions:
//   h = dropout(relu(conv1(x))), h = dropout(relu(conv2(h)))
//   out = relu(h + x), with a 1×1 convolution of x when the widths differ.
// Stacking blocks with dilations 1, 2, 4, … gives a receptive field of
// 1 + 2·(K - 1)·(2^levels - 1) steps. (The paper's weight normalization is
// not used.)

TCNBlock :: struct {
	conv1, conv2: Conv1dLayer,
	downsample:   Conv1dLayer, // 1×1 convolution of the residual; weight nil if in = out
	dropout:      f64,
}

tcn_block_new :: proc(
	in_channels: int,
	out_channels: int,
	kernel_size: int,
	dilation: int,
	dropout: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) -> TCNBlock {
	block := TCNBlock {
		conv1   = conv1d_layer_new(
			in_channels,
			out_channels,
			kernel_size,
			dilation,
			causal = true,
			allocator = allocator,
		),
		conv2   = conv1d_layer_new(
			out_channels,
			out_channels,
			kernel_size,
			dilation,
			causal = true,
			allocator = allocator,
		),
		dropout = dropout,
	}
	if in_channels != out_channels {
		block.downsample = conv1d_layer_new(in_channels, out_channels, 1, allocator = allocator)
	}
	return block
}

tcn_block_forward :: proc(block: ^TCNBlock, x: ^t.Tensor, training: bool) -> ^t.Tensor {
	h := t.tensor_relu(conv1d_layer_forward(&block.conv1, x))
	h = t.tensor_dropout(h, block.dropout, training)
	h = t.tensor_relu(conv1d_layer_forward(&block.conv2, h))
	h = t.tensor_dropout(h, block.dropout, training)
	res := conv1d_layer_forward(&block.downsample, x) if block.downsample.weight != nil else x
	// the input may be a [rows, cols] matrix of the same values (stored differently)
	if res.data.rows != h.data.rows {res = t.tensor_reshape(res, h.shape)}
	return t.tensor_relu(t.tensor_add(h, res))
}

tcn_block_free :: proc(block: ^TCNBlock) {
	conv1d_layer_free(&block.conv1)
	conv1d_layer_free(&block.conv2)
	conv1d_layer_free(&block.downsample)
}

// sequential_add_tcn appends `levels` TCN blocks in_size → hidden → … → hidden
// with dilations 1, 2, 4, …
sequential_add_tcn :: proc(
	s: ^Sequential,
	in_size: int,
	hidden: int,
	levels: int = 3,
	kernel_size: int = 3,
	dropout: f64 = 0.0,
	allocator: mem.Allocator = context.allocator,
) {
	for i in 0 ..< levels {
		sequential_add(
			s,
			tcn_block_new(
				in_size if i == 0 else hidden,
				hidden,
				kernel_size,
				1 << uint(i),
				dropout,
				allocator,
			),
		)
	}
}
