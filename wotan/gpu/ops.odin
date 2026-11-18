package gpu

import t "../tensor"
import "core:mem"

// ============================================================================
// Device ops: forward and backward of the ops with GPU kernels
// ============================================================================
//
// wotan/tensor records the graph exactly as on the CPU (op, inputs, metadata)
// and calls _forward for the computation; tensor_backward calls _backward per
// node. Ops not listed in SUPPORTED run on the CPU (the generic fallback).

@(private)
SUPPORTED :: bit_set[t.Op] {
	.Add,
	.Sub,
	.Mul,
	.Div,
	.Neg,
	.Scale,
	.Relu,
	.LeakyReLU,
	.Sigmoid,
	.Tanh,
	.Exp,
	.Log,
	.Sqrt,
	.Clamp,
	.Softplus,
	.AddBias,
	.MatMul,
	.Sum,
	.Mean,
	.SumDim1,
	.MSELoss,
	.Reshape,
	.Transpose,
	.SliceCols,
	.Concat,
	// phase 2 (ops_p2.odin)
	.GeluTanh,
	.NormCDF,
	.Softmax,
	.LogSumExpDim1,
	.CrossEntropy,
	.BCELoss,
	.BinaryCrossEntropy,
	.LayerNorm,
	.RMSNorm,
	.Dropout,
	.Embedding,
	.PermuteMHA,
	.PermuteMHAInverse,
	.ScaledDotProductAttention,
	.MaskedScaledDotProductAttention,
	.FlashAttention,
	.GateMul,
	// phase 3 (ops_p3.odin)
	.Flatten,
	.Conv2d,
	.Conv1d,
	.MaxPool2d,
	.AvgPool2d,
	.BatchNorm2d,
	// phase 4 (ops_p4.odin; lgamma and the prospect transforms are unary codes)
	.Lgamma,
	.ProspectValue,
	.ProbWeighting,
	.KLDivergence,
	.Entropy,
	.ProspectUtility,
	.SharpeLoss,
	.NormalizeTime,
	.PermuteLOB,
	.TopKMask,
	.Reparameterize,
}

@(private)
_unary_code :: proc(op: t.Op) -> (code: u32, ok: bool) {
	#partial switch op {
	case .Neg:
		return U_NEG, true
	case .Scale:
		return U_SCALE, true
	case .Relu:
		return U_RELU, true
	case .LeakyReLU:
		return U_LEAKY, true
	case .Sigmoid:
		return U_SIGMOID, true
	case .Tanh:
		return U_TANH, true
	case .Exp:
		return U_EXP, true
	case .Log:
		return U_LOG, true
	case .Sqrt:
		return U_SQRT, true
	case .Clamp:
		return U_CLAMP, true
	case .Softplus:
		return U_SOFTPLUS, true
	case .GeluTanh:
		return U_GELU_TANH, true
	case .NormCDF:
		return U_NORM_CDF, true
	case .Lgamma:
		return U_LGAMMA, true
	case .ProspectValue:
		return U_PROSPECT, true
	case .ProbWeighting:
		return U_PROB_WEIGHT, true
	}
	return 0, false
}

@(private)
_binary_code :: proc(op: t.Op) -> u32 {
	#partial switch op {
	case .Sub:
		return 1
	case .Mul:
		return 2
	case .Div:
		return 3
	}
	return 0
}

@(private)
_hook_has_forward :: proc(op: t.Op) -> bool {
	return op in SUPPORTED && !G.disabled[op]
}

@(private)
_data :: proc(x: ^t.Tensor) -> Bind {
	return _bind(x.device.data_buf, t.tensor_numel(x))
}

@(private)
_grad :: proc(x: ^t.Tensor) -> Bind {
	return _bind(x.device.grad_buf, t.tensor_numel(x))
}

@(private)
_p :: proc(u: [8]u32 = {}, f: [8]f32 = {}) -> Params {
	return {u = u, f = f}
}

// _matmul records C[M, N] (+)= op(A) @ op(B), in row chunks of the grid limit
@(private)
_matmul :: proc(A, B, C: Bind, M, N, K: int, trans_a, trans_b, accumulate: bool, batch := 1) {
	if M == 0 || N == 0 {return}
	if K == 0 {
		if !accumulate {_zero(C.buf, M * N * batch)}
		return
	}
	gx := u32((N + 15) / 16)
	CHUNK :: 65535 * 16
	for r0 := 0; r0 < M; r0 += CHUNK {
		rows := min(CHUNK, M - r0)
		p := _p({u32(M), u32(N), u32(K), u32(trans_a), u32(trans_b), u32(accumulate), u32(r0), 0})
		_dispatch(.Matmul, p, {A, B, C}, gx, u32((rows + 15) / 16), u32(batch))
	}
}

// _reduce: out[index] = scale * Σ x (squares: Σ x²), two passes for large n
@(private)
_reduce :: proc(x: Bind, n: int, out: Bind, index: int, scale: f32, squares: bool) {
	sq := u32(squares)
	groups := min((n + 255) / 256, 256)
	if groups <= 1 {
		_dispatch(
			.Reduce,
			_p({u32(n), sq, u32(index), 0, 0, 0, 0, 0}, {scale, 0, 0, 0, 0, 0, 0, 0}),
			{x, out},
			1,
			1,
		)
		return
	}
	partial := _acquire(groups)
	_dispatch(
		.Reduce,
		_p({u32(n), sq, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}),
		{x, {partial, groups}},
		u32(groups),
		1,
	)
	_dispatch(
		.Reduce,
		_p({u32(groups), 0, u32(index), 0, 0, 0, 0, 0}, {scale, 0, 0, 0, 0, 0, 0, 0}),
		{{partial, groups}, out},
		1,
		1,
	)
	_release_buf(partial) // later reuse is recorded after these dispatches
}

// _strided: dst[r * dst_stride + dst_off + c] (+)= src[r * src_stride + src_off + c]
@(private)
_strided :: proc(
	src, dst: Bind,
	rows, length, src_stride, src_off, dst_stride, dst_off: int,
	accumulate: bool,
) {
	p := _p(
		{
			u32(rows),
			u32(length),
			u32(src_stride),
			u32(src_off),
			u32(dst_stride),
			u32(dst_off),
			u32(accumulate),
			0,
		},
	)
	_dispatch_1d(.Strided_Copy, p, {src, dst}, rows * length)
}

// _concat_blocks: outer block count and inner block size of a concat along dim
@(private)
_concat_blocks :: proc(shape: [4]int, dim: int) -> (outer, inner: int) {
	outer, inner = 1, 1
	for d in 0 ..< dim {outer *= shape[d]}
	for d in dim + 1 ..< 4 {inner *= shape[d]}
	return
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

@(private)
_hook_forward :: proc(op: t.Op, out: ^t.Tensor, inputs: []^t.Tensor, ints: []int, floats: []f64) {
	n := t.tensor_numel(out)
	o := _data(out)
	f0 := f32(floats[0]) if len(floats) > 0 else 0
	f1 := f32(floats[1]) if len(floats) > 1 else 0
	f2 := f32(floats[2]) if len(floats) > 2 else 0
	if code, ok := _unary_code(op); ok {
		_dispatch_1d(
			.Unary,
			_p({u32(n), code, 0, 0, 0, 0, 0, 0}, {f0, f1, f2, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), o},
			n,
		)
		return
	}
	#partial switch op {
	case .Add, .Sub, .Mul, .Div:
		_dispatch_1d(
			.Binary,
			_p({u32(n), _binary_code(op), 0, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), _data(inputs[1]), o},
			n,
		)
	case .MatMul:
		_matmul(
			_data(inputs[0]),
			_data(inputs[1]),
			o,
			ints[0],
			ints[2],
			ints[1],
			false,
			false,
			false,
		)
	case .AddBias:
		_dispatch_1d(
			.Bias_Add,
			_p({u32(n), u32(ints[1]), 0, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), _data(inputs[1]), o},
			n,
		)
	case .Sum, .Mean:
		a := inputs[0]
		an := t.tensor_numel(a)
		scale := f32(1) if op == .Sum else f32(1.0 / f64(an))
		_reduce(_data(a), an, o, 0, scale, false)
	case .MSELoss:
		a, b := inputs[0], inputs[1]
		an := t.tensor_numel(a)
		diff := _acquire(an)
		_dispatch_1d(
			.Binary,
			_p({u32(an), 1, 0, 0, 0, 0, 0, 0}),
			{_data(a), _data(b), {diff, an}},
			an,
		)
		_reduce({diff, an}, an, o, 0, f32(1.0 / f64(an)), true)
		_release_buf(diff)
	case .SumDim1:
		_dispatch_1d(
			.Row_Sum,
			_p({u32(ints[0]), u32(ints[1]), 0, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), o},
			ints[0],
		)
	case .Reshape, .Flatten:
		_copy(o.buf, _buf(inputs[0].device.data_buf), n)
	case .Transpose:
		a := inputs[0]
		_dispatch_1d(
			.Transpose,
			_p({u32(a.data.rows), u32(a.data.cols), 0, 0, 0, 0, 0, 0}),
			{_data(a), o},
			n,
		)
	case .SliceCols:
		a := inputs[0]
		start, C := ints[0], ints[1]
		count := out.data.cols
		_strided(_data(a), o, a.data.rows, count, C, start, count, 0, false)
	case .Concat:
		dim := ints[0]
		total := 0
		for i in 0 ..< len(inputs) {total += ints[1 + i]}
		off := 0
		for x, i in inputs {
			size := ints[1 + i]
			outer, inner := _concat_blocks(x.shape, dim)
			_strided(
				_data(x),
				o,
				outer,
				size * inner,
				size * inner,
				0,
				total * inner,
				off * inner,
				false,
			)
			off += size
		}
	case:
		_forward_p2(op, out, inputs, ints, floats)
	}
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------

// _in_grad: the input's gradient on the device, ready for accumulation
@(private)
_in_grad :: proc(x: ^t.Tensor) -> Bind {
	t._grad_to_device(x)
	return _grad(x)
}

@(private)
_in_data :: proc(x: ^t.Tensor) -> Bind {
	t._data_to_device(x)
	return _data(x)
}

@(private)
_hook_backward :: proc(node: ^t.Tensor, allocator: mem.Allocator) -> bool {
	op := node.op
	if op not_in SUPPORTED || G.disabled[op] {return false}
	t._grad_to_device(node)
	dy := _grad(node)
	n := t.tensor_numel(node)
	written: [8]^t.Tensor
	nw := 0
	wrote :: proc(w: ^[8]^t.Tensor, nw: ^int, x: ^t.Tensor) {
		w[nw^] = x
		nw^ += 1
	}

	// dx += f * dy
	axpy :: proc(dy: Bind, x: ^t.Tensor, f: f32, n: int) {
		g := _in_grad(x)
		_dispatch_1d(
			.Unary_Bwd,
			_p({u32(n), U_COPY, 0, 0, 0, 0, 0, 0}, {f, 0, 0, 0, 0, 0, 0, 0}),
			{dy, dy, dy, g},
			n,
		)
	}

	if code, ok := _unary_code(op); ok {
		a := node.inputs[0]
		if !a.requires_grad {return true}
		f0 := f32(node.f64_metadata[0]) if len(node.f64_metadata) > 0 else 0
		f1 := f32(node.f64_metadata[1]) if len(node.f64_metadata) > 1 else 0
		f2 := f32(node.f64_metadata[2]) if len(node.f64_metadata) > 2 else 0
		switch code {
		case U_NEG:
			axpy(dy, a, -1, n)
		case U_SCALE:
			axpy(dy, a, f0, n)
		case:
			x := _in_data(a)
			y := x
			if code == U_EXP || code == U_SQRT || code == U_PROB_WEIGHT {y = _in_data(node)}
			g := _in_grad(a)
			_dispatch_1d(
				.Unary_Bwd,
				_p({u32(n), code, 0, 0, 0, 0, 0, 0}, {f0, f1, f2, 0, 0, 0, 0, 0}),
				{x, y, dy, g},
				n,
			)
		}
		t._grad_written_on_device(a)
		return true
	}

	#partial switch op {
	case .Add, .Sub:
		for x, i in node.inputs {
			if !x.requires_grad {continue}
			axpy(dy, x, -1 if op == .Sub && i == 1 else 1, n)
			wrote(&written, &nw, x)
		}
	case .Mul, .Div:
		a, b := node.inputs[0], node.inputs[1]
		for x, side in node.inputs {
			if !x.requires_grad {continue}
			ad, bd := _in_data(a), _in_data(b)
			g := _in_grad(x)
			_dispatch_1d(
				.Binary_Bwd,
				_p({u32(n), _binary_code(op), u32(side), 0, 0, 0, 0, 0}),
				{ad, bd, dy, g},
				n,
			)
			wrote(&written, &nw, x)
		}
	case .MatMul:
		a, b := node.inputs[0], node.inputs[1]
		N, in_f, out_f := a.data.rows, a.data.cols, b.data.cols
		if len(node.int_metadata) == 3 {
			N, in_f, out_f = node.int_metadata[0], node.int_metadata[1], node.int_metadata[2]
		}
		if a.requires_grad {
			// dA[N, in] += dC[N, out] @ Bᵀ
			bd := _in_data(b)
			_matmul(dy, bd, _in_grad(a), N, in_f, out_f, false, true, true)
			wrote(&written, &nw, a)
		}
		if b.requires_grad {
			// dB[in, out] += Aᵀ @ dC
			ad := _in_data(a)
			_matmul(ad, dy, _in_grad(b), in_f, out_f, N, true, false, true)
			wrote(&written, &nw, b)
		}
	case .AddBias:
		a, bias := node.inputs[0], node.inputs[1]
		N, D := node.data.rows, node.data.cols
		if len(node.int_metadata) == 2 {N, D = node.int_metadata[0], node.int_metadata[1]}
		if a.requires_grad {
			axpy(dy, a, 1, n)
			wrote(&written, &nw, a)
		}
		if bias.requires_grad {
			_dispatch_1d(.Col_Sum, _p({u32(N), u32(D), 0, 0, 0, 0, 0, 0}), {dy, _in_grad(bias)}, D)
			wrote(&written, &nw, bias)
		}
	case .Sum, .Mean:
		a := node.inputs[0]
		if a.requires_grad {
			an := t.tensor_numel(a)
			scale := f32(1) if op == .Sum else f32(1.0 / f64(an))
			_dispatch_1d(
				.Bcast_Add,
				_p({u32(an), 0, 0, 0, 0, 0, 0, 0}, {scale, 0, 0, 0, 0, 0, 0, 0}),
				{dy, _in_grad(a)},
				an,
			)
			wrote(&written, &nw, a)
		}
	case .MSELoss:
		pred, target := node.inputs[0], node.inputs[1]
		an := t.tensor_numel(pred)
		for x, i in node.inputs {
			if !x.requires_grad {continue}
			sign := f32(1) if i == 0 else f32(-1)
			pd, td := _in_data(pred), _in_data(target)
			f := sign * f32(2.0 / f64(an))
			_dispatch_1d(
				.Mse_Bwd,
				_p({u32(an), 0, 0, 0, 0, 0, 0, 0}, {f, 0, 0, 0, 0, 0, 0, 0}),
				{pd, td, dy, _in_grad(x)},
				an,
			)
			wrote(&written, &nw, x)
		}
	case .SumDim1:
		a := node.inputs[0]
		if a.requires_grad {
			an := t.tensor_numel(a)
			width := an / node.data.rows
			_dispatch_1d(
				.Row_Bcast_Add,
				_p({u32(an), u32(width), 0, 0, 0, 0, 0, 0}),
				{dy, _in_grad(a)},
				an,
			)
			wrote(&written, &nw, a)
		}
	case .Reshape, .Flatten:
		a := node.inputs[0]
		if a.requires_grad {
			axpy(dy, a, 1, n)
			wrote(&written, &nw, a)
		}
	case .Transpose:
		a := node.inputs[0]
		if a.requires_grad {
			// dy is [cols, rows]: transposing it back accumulates into dx [rows, cols]
			_dispatch_1d(
				.Transpose,
				_p({u32(a.data.cols), u32(a.data.rows), 1, 0, 0, 0, 0, 0}),
				{dy, _in_grad(a)},
				n,
			)
			wrote(&written, &nw, a)
		}
	case .SliceCols:
		a := node.inputs[0]
		if a.requires_grad {
			start, C := node.int_metadata[0], node.int_metadata[1]
			count := node.data.cols
			_strided(dy, _in_grad(a), a.data.rows, count, count, 0, C, start, true)
			wrote(&written, &nw, a)
		}
	case .Concat:
		dim := node.int_metadata[0]
		total := 0
		for i in 0 ..< len(node.inputs) {total += node.int_metadata[1 + i]}
		off := 0
		for x, i in node.inputs {
			size := node.int_metadata[1 + i]
			if x.requires_grad {
				outer, inner := _concat_blocks(x.shape, dim)
				_strided(
					dy,
					_in_grad(x),
					outer,
					size * inner,
					total * inner,
					off * inner,
					size * inner,
					0,
					true,
				)
				t._grad_written_on_device(x)
			}
			off += size
		}
	case:
		if !_backward_p2(node, dy, n) {return false}
	}
	for x in written[:nw] {t._grad_written_on_device(x)}
	return true
}
