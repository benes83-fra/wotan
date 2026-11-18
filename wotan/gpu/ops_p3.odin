package gpu

import t "../tensor"

// ============================================================================
// Device ops, part 3: convolutions, pooling, batch norm
// ============================================================================
//
// Convolutions are im2col + one matmul. conv2d builds the columns of all
// samples side by side ([C_in·kH·kW, N·L]), multiplies once by the weight
// [C_out, C_in·kH·kW] and permutes the [C_out, N, L] product into [N, C_out, L].
// conv1d (channels-last) builds one row per output step ([B·T_out, C_in·K])
// and multiplies by the transposed weight; the product is already [B, T_out, C_out].
// The backward passes recompute the columns instead of keeping them.

@(private)
Conv2d_Dims :: struct {
	N, C_in, H, W, C_out, kH, kW, stride, pad, oh, ow, L, CK: int,
}

@(private)
_conv2d_dims :: proc(x, w: ^t.Tensor, stride, pad: int) -> (d: Conv2d_Dims) {
	d.N, d.C_in, d.H, d.W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	d.C_out, d.kH, d.kW = w.shape[0], w.shape[2], w.shape[3]
	d.stride, d.pad = stride, pad
	d.oh = (d.H + 2 * pad - d.kH) / stride + 1
	d.ow = (d.W + 2 * pad - d.kW) / stride + 1
	d.L = d.oh * d.ow
	d.CK = d.C_in * d.kH * d.kW
	return
}

@(private)
_im2col_2d :: proc(d: Conv2d_Dims, x: Bind) -> Bind {
	n := d.CK * d.N * d.L
	col := Bind{_acquire(n), n}
	u := [8]u32 {
		u32(d.N),
		u32(d.C_in),
		u32(d.H),
		u32(d.W),
		u32(d.kH),
		u32(d.kW),
		u32(d.stride),
		u32(d.pad),
	}
	_dispatch_1d(.Im2col_2d, _p(u), {x, col}, n)
	return col
}

// dst [B, A, L] = src [A, B, L] with the two outer dimensions swapped
@(private)
_swap_outer :: proc(src, dst: Bind, A, B, L: int) {
	// Permute_Mha with one batch, head <- sequence: sequence layout [S, H, hd] =
	// [A, B, L], head layout [H, S, hd] = [B, A, L]
	u := [8]u32{1, u32(A), u32(B), u32(L), 0, 0, 0, 0}
	_dispatch_1d(.Permute_Mha, _p(u), {src, dst}, A * B * L)
}

@(private)
Conv1d_Dims :: struct {
	B, T, C_in, C_out, K, dilation, pad_left, T_out, R, CK: int,
}

@(private)
_conv1d_dims :: proc(m: []int) -> (d: Conv1d_Dims) {
	d.B, d.T, d.C_in, d.C_out, d.K, d.dilation, d.pad_left, d.T_out =
		m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]
	d.R = d.B * d.T_out
	d.CK = d.C_in * d.K
	return
}

@(private)
_conv1d_params :: proc(d: Conv1d_Dims) -> Params {
	return _p(
		{
			u32(d.B),
			u32(d.T),
			u32(d.C_in),
			u32(d.K),
			u32(d.dilation),
			u32(d.pad_left),
			u32(d.T_out),
			0,
		},
	)
}

@(private)
_im2col_1d :: proc(d: Conv1d_Dims, x: Bind) -> Bind {
	n := d.R * d.CK
	col := Bind{_acquire(n), n}
	_dispatch_1d(.Im2col_1d, _conv1d_params(d), {x, col}, n)
	return col
}

@(private)
_pool_params :: proc(x: ^t.Tensor, kH, kW, stride: int, op: t.Op) -> Params {
	u := [8]u32 {
		u32(x.shape[0] * x.shape[1]),
		u32(x.shape[2]),
		u32(x.shape[3]),
		u32(kH),
		u32(kW),
		u32(stride),
		u32(op == .AvgPool2d),
		0,
	}
	return _p(u)
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

@(private)
_forward_p3 :: proc(op: t.Op, out: ^t.Tensor, inputs: []^t.Tensor, ints: []int, floats: []f64) {
	o := _data(out)
	n := t.tensor_numel(out)
	#partial switch op {
	case .Conv2d:
		x, w := inputs[0], inputs[1]
		d := _conv2d_dims(x, w, ints[0], ints[1])
		col := _im2col_2d(d, _data(x))
		prod := Bind{_acquire(d.C_out * d.N * d.L), d.C_out * d.N * d.L}
		_matmul(_data(w), col, prod, d.C_out, d.N * d.L, d.CK, false, false, false)
		_swap_outer(prod, o, d.C_out, d.N, d.L)
		if len(inputs) > 2 {
			u := [8]u32{u32(n), u32(d.L), u32(d.C_out), 0, 0, 0, 0, 0}
			_dispatch_1d(.Chan_Bias, _p(u), {_data(inputs[2]), o}, n)
		}
		_release_buf(col.buf)
		_release_buf(prod.buf)
	case .Conv1d:
		d := _conv1d_dims(ints)
		col := _im2col_1d(d, _data(inputs[0]))
		_matmul(col, _data(inputs[1]), o, d.R, d.C_out, d.CK, false, true, false)
		if len(inputs) > 2 {
			u := [8]u32{u32(n), 1, u32(d.C_out), 0, 0, 0, 0, 0}
			_dispatch_1d(.Chan_Bias, _p(u), {_data(inputs[2]), o}, n)
		}
		_release_buf(col.buf)
	case .MaxPool2d, .AvgPool2d:
		_dispatch_1d(
			.Pool,
			_pool_params(inputs[0], ints[0], ints[1], ints[2], op),
			{_data(inputs[0]), o},
			n,
		)
	case .BatchNorm2d:
		x, gamma, beta, rm, rv := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
		C, HW := x.shape[1], x.shape[2] * x.shape[3]
		training := floats[0] != 0
		stats := _bind(out.device.aux_buf, 2 * C)
		u := [8]u32{u32(C), u32(HW), u32(x.shape[0]), u32(training), 0, 0, 0, 0}
		f := [8]f32{f32(floats[1]), f32(floats[2]), 0, 0, 0, 0, 0, 0}
		_dispatch_rows(.Bn_Stats, _p(u, f), {_data(x), _data(rm), _data(rv), stats}, C)
		_dispatch_1d(
			.Bn_Apply,
			_p({u32(n), u32(HW), u32(C), 0, 0, 0, 0, 0}),
			{_data(x), _data(gamma), _data(beta), stats, o},
			n,
		)
		if training {
			t._data_written_on_device(rm)
			t._data_written_on_device(rv)
		}
	case:
		_forward_p4(op, out, inputs, ints, floats)
	}
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------

@(private)
_backward_p3 :: proc(node: ^t.Tensor, dy: Bind, n: int) -> bool {
	#partial switch node.op {
	case .Conv2d:
		_conv2d_backward(node, dy)
	case .Conv1d:
		_conv1d_backward(node, dy)
	case .MaxPool2d, .AvgPool2d:
		a := node.inputs[0]
		if !a.requires_grad {return true}
		pp := node.pool_params
		xd := _in_data(a)
		_dispatch_1d(
			.Pool_Bwd,
			_pool_params(a, pp.kH, pp.kW, pp.stride, node.op),
			{xd, dy, _in_grad(a)},
			t.tensor_numel(a),
		)
		t._grad_written_on_device(a)
	case .BatchNorm2d:
		_batch_norm_backward(node, dy)
	case:
		return _backward_p4(node, dy, n)
	}
	return true
}

@(private)
_conv2d_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x, w := node.inputs[0], node.inputs[1]
	b: ^t.Tensor = node.inputs[2] if len(node.inputs) > 2 else nil
	d := _conv2d_dims(x, w, node.conv_params.stride, node.conv_params.pad)
	NL := d.N * d.L
	if b != nil && b.requires_grad {
		u := [8]u32{u32(d.C_out), u32(d.L), u32(d.N), 0, 0, 0, 0, 0}
		_dispatch_rows(.Chan_Sum, _p(u), {dy, _in_grad(b)}, d.C_out)
		t._grad_written_on_device(b)
	}
	if !x.requires_grad && !w.requires_grad {return}
	// dy [N, C_out, L] -> [C_out, N·L]
	dyp := Bind{_acquire(d.C_out * NL), d.C_out * NL}
	_swap_outer(dy, dyp, d.N, d.C_out, d.L)
	if w.requires_grad {
		// dW [C_out, CK] += dy' @ colᵀ
		col := _im2col_2d(d, _in_data(x))
		_matmul(dyp, col, _in_grad(w), d.C_out, d.CK, NL, false, true, true)
		_release_buf(col.buf)
		t._grad_written_on_device(w)
	}
	if x.requires_grad {
		// dcol [CK, N·L] = Wᵀ @ dy', summed back onto the image
		dcol := Bind{_acquire(d.CK * NL), d.CK * NL}
		_matmul(_in_data(w), dyp, dcol, d.CK, NL, d.C_out, true, false, false)
		u := [8]u32 {
			u32(d.N),
			u32(d.C_in),
			u32(d.H),
			u32(d.W),
			u32(d.kH),
			u32(d.kW),
			u32(d.stride),
			u32(d.pad),
		}
		_dispatch_1d(.Col2im_2d, _p(u), {dcol, _in_grad(x)}, t.tensor_numel(x))
		_release_buf(dcol.buf)
		t._grad_written_on_device(x)
	}
	_release_buf(dyp.buf)
}

@(private)
_conv1d_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x, w := node.inputs[0], node.inputs[1]
	b: ^t.Tensor = node.inputs[2] if len(node.inputs) > 2 else nil
	d := _conv1d_dims(node.int_metadata[:])
	if b != nil && b.requires_grad {
		_dispatch_1d(
			.Col_Sum,
			_p({u32(d.R), u32(d.C_out), 0, 0, 0, 0, 0, 0}),
			{dy, _in_grad(b)},
			d.C_out,
		)
		t._grad_written_on_device(b)
	}
	if w.requires_grad {
		// dW [C_out, CK] += dyᵀ @ col
		col := _im2col_1d(d, _in_data(x))
		_matmul(dy, col, _in_grad(w), d.C_out, d.CK, d.R, true, false, true)
		_release_buf(col.buf)
		t._grad_written_on_device(w)
	}
	if x.requires_grad {
		// dcol [R, CK] = dy @ W, summed back onto the sequence
		dcol := Bind{_acquire(d.R * d.CK), d.R * d.CK}
		_matmul(dy, _in_data(w), dcol, d.R, d.CK, d.C_out, false, false, false)
		_dispatch_1d(.Col2im_1d, _conv1d_params(d), {dcol, _in_grad(x)}, t.tensor_numel(x))
		_release_buf(dcol.buf)
		t._grad_written_on_device(x)
	}
}

// batch norm: the statistics the forward used (aux buffer, or uploaded from the
// CPU forward's metadata), per-channel sums, then the input gradient
@(private)
_batch_norm_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x, gamma, beta := node.inputs[0], node.inputs[1], node.inputs[2]
	N, C, HW := x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
	training := node.f64_metadata[0] != 0
	stats: Bind
	temp := false
	if node.device != nil && node.device.aux_buf.ptr != nil {
		stats = _bind(node.device.aux_buf, 2 * C)
	} else {
		// the CPU forward recorded [training, means, inv_stds]
		stats, temp = _upload_values(node.f64_metadata[1:1 + 2 * C]), true
	}
	want_g, want_b := gamma.requires_grad, beta.requires_grad
	xd := _in_data(x)
	sums := Bind{_acquire(2 * C), 2 * C}
	// gradients not wanted go to scratch buffers (never the same buffer twice)
	dg := _in_grad(gamma) if want_g else Bind{_acquire(C), C}
	db := _in_grad(beta) if want_b else Bind{_acquire(C), C}
	u := [8]u32{u32(C), u32(HW), u32(N), u32(want_g), u32(want_b), 0, 0, 0}
	_dispatch_rows(.Bn_Bwd_Sums, _p(u), {xd, dy, stats, sums, dg, db}, C)
	if want_g {t._grad_written_on_device(gamma)} else {_release_buf(dg.buf)}
	if want_b {t._grad_written_on_device(beta)} else {_release_buf(db.buf)}
	if x.requires_grad {
		n := t.tensor_numel(x)
		u2 := [8]u32{u32(n), u32(HW), u32(C), u32(training), u32(N * HW), 0, 0, 0}
		_dispatch_1d(.Bn_Bwd_X, _p(u2), {xd, _in_data(gamma), dy, stats, sums, _in_grad(x)}, n)
		t._grad_written_on_device(x)
	}
	_release_buf(sums.buf)
	if temp {_release_buf(stats.buf)}
}
