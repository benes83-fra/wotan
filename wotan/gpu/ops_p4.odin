package gpu

import t "../tensor"
import "core:fmt"

// ============================================================================
// Device ops, part 4: finance losses and transforms, VAE ops, top-k routing
// ============================================================================
//
// lgamma, prospect_value and prob_weighting are unary codes (ops.odin). The
// scalar losses write one term per element (Loss_Terms) and reduce it; their
// gradients are one Loss_Grad pass each. The Sharpe ratio keeps its mean and
// standard deviation in the output's aux buffer for the backward pass.

@(private)
_loss_terms :: proc(mode: u32, a, b: Bind, n: int, out: Bind, scale: f32, f: [3]f32 = {}) {
	terms := Bind{_acquire(n), n}
	_dispatch_1d(
		.Loss_Terms,
		_p({u32(n), mode, 0, 0, 0, 0, 0, 0}, {0, f[0], f[1], f[2], 0, 0, 0, 0}),
		{a, b, terms},
		n,
	)
	_reduce(terms, n, out, 0, scale, false)
	_release_buf(terms.buf)
}

// dx += g[0] · scale · ∂term (see Loss_Grad)
@(private)
_loss_grad :: proc(mode: u32, x: ^t.Tensor, a, b, aux, g: Bind, scale: f32, f: [3]f32 = {}) {
	n := t.tensor_numel(x)
	_dispatch_1d(
		.Loss_Grad,
		_p({u32(n), mode, 0, 0, 0, 0, 0, 0}, {scale, f[0], f[1], f[2], 0, 0, 0, 0}),
		{a, b, aux, g, _in_grad(x)},
		n,
	)
	t._grad_written_on_device(x)
}

@(private)
_f3 :: proc(v: []f64) -> [3]f32 {
	return {f32(v[0]), f32(v[1]), f32(v[2])}
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

@(private)
_forward_p4 :: proc(op: t.Op, out: ^t.Tensor, inputs: []^t.Tensor, ints: []int, floats: []f64) {
	o := _data(out)
	n := t.tensor_numel(out)
	#partial switch op {
	case .KLDivergence:
		mu, lv := inputs[0], inputs[1]
		m := t.tensor_numel(mu)
		_loss_terms(0, _data(mu), _data(lv), m, o, f32(-0.5 / floats[0]))
	case .Entropy:
		a := inputs[0]
		_loss_terms(1, _data(a), _dummy(), t.tensor_numel(a), o, 1)
	case .ProspectUtility:
		x := inputs[0]
		m := t.tensor_numel(x)
		_loss_terms(2, _data(x), _dummy(), m, o, f32(-1.0 / f64(m)), _f3(floats))
	case .SharpeLoss:
		x := inputs[0]
		aux := _bind(out.device.aux_buf, 2)
		p := _p(
			{u32(t.tensor_numel(x)), 0, 0, 0, 0, 0, 0, 0},
			{f32(floats[0]), 0, 0, 0, 0, 0, 0, 0},
		)
		_dispatch(.Sharpe, p, {_data(x), o, aux}, 1, 1)
	case .NormalizeTime:
		N, C, T, L := ints[0], ints[1], ints[2], ints[3]
		u := [8]u32{u32(N * C * L), u32(C), u32(T), u32(L), 0, 0, 0, 0}
		f := [8]f32{f32(floats[0]), 0, 0, 0, 0, 0, 0, 0}
		_dispatch_rows(.Norm_Time, _p(u, f), {_data(inputs[0]), _dummy(), _dummy(), o}, N * C * L)
	case .PermuteLOB:
		// [B, C, T, L] -> [B, T, C·L]: the head permute with S = C, H = T, hd = L
		u := [8]u32{u32(ints[0]), u32(ints[1]), u32(ints[2]), u32(ints[3]), 0, 0, 0, 0}
		_dispatch_1d(.Permute_Mha, _p(u), {_data(inputs[0]), o}, n)
	case .TopKMask:
		N, E, k := ints[0], ints[1], ints[2]
		_dispatch_1d(.Top_K, _p({u32(N), u32(E), u32(k), 0, 0, 0, 0, 0}), {_data(inputs[0]), o}, N)
	case .Reparameterize:
		eps := _bind(out.device.aux_buf, n)
		_dispatch_1d(
			.Reparam,
			_p({u32(n), 0, 0, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), _data(inputs[1]), eps, o},
			n,
		)
	case:
		panic(fmt.tprintf("gpu: no forward kernel for %v", op))
	}
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------

@(private)
_backward_p4 :: proc(node: ^t.Tensor, dy: Bind, n: int) -> bool {
	#partial switch node.op {
	case .KLDivergence:
		mu, lv := node.inputs[0], node.inputs[1]
		batch := node.f64_metadata[0]
		md, ld := _in_data(mu), _in_data(lv)
		if mu.requires_grad {_loss_grad(0, mu, md, ld, _dummy(), dy, f32(1 / batch))}
		if lv.requires_grad {_loss_grad(1, lv, md, ld, _dummy(), dy, f32(1 / (2 * batch)))}
	case .Entropy:
		a := node.inputs[0]
		if a.requires_grad {_loss_grad(2, a, _in_data(a), _dummy(), _dummy(), dy, 1)}
	case .ProspectUtility:
		x := node.inputs[0]
		if x.requires_grad {
			m := t.tensor_numel(x)
			f := _f3(node.f64_metadata[:])
			_loss_grad(3, x, _in_data(x), _dummy(), _dummy(), dy, f32(1.0 / f64(m)), f)
		}
	case .SharpeLoss:
		x := node.inputs[0]
		if !x.requires_grad {return true}
		aux: Bind
		temp := false
		if node.device != nil && node.device.aux_buf.ptr != nil {
			aux = _bind(node.device.aux_buf, 2)
		} else {
			// the CPU forward recorded [rf, mean, std]
			aux, temp = _upload_values(node.f64_metadata[1:3]), true
		}
		rf := f32(node.f64_metadata[0])
		_loss_grad(4, x, _in_data(x), _dummy(), aux, dy, 1, {rf, 0, 0})
		if temp {_release_buf(aux.buf)}
	case .NormalizeTime:
		x := node.inputs[0]
		if !x.requires_grad {return true}
		m := node.int_metadata
		N, C, T, L := m[0], m[1], m[2], m[3]
		u := [8]u32{u32(N * C * L), u32(C), u32(T), u32(L), 1, 0, 0, 0}
		f := [8]f32{f32(node.f64_metadata[0]), 0, 0, 0, 0, 0, 0, 0}
		xd, yd := _in_data(x), _in_data(node)
		_dispatch_rows(.Norm_Time, _p(u, f), {xd, yd, dy, _in_grad(x)}, N * C * L)
		t._grad_written_on_device(x)
	case .PermuteLOB:
		x := node.inputs[0]
		if !x.requires_grad {return true}
		m := node.int_metadata
		u := [8]u32{u32(m[0]), u32(m[1]), u32(m[2]), u32(m[3]), 1, 1, 0, 0}
		_dispatch_1d(.Permute_Mha, _p(u), {dy, _in_grad(x)}, n)
		t._grad_written_on_device(x)
	case .Reparameterize:
		mu, lv := node.inputs[0], node.inputs[1]
		if mu.requires_grad {
			g := _in_grad(mu)
			_dispatch_1d(
				.Unary_Bwd,
				_p({u32(n), U_COPY, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}),
				{dy, dy, dy, g},
				n,
			)
			t._grad_written_on_device(mu)
		}
		if lv.requires_grad {
			// ∂z/∂log_var = (z - mu) / 2
			zd, md := _in_data(node), _in_data(mu)
			_dispatch_1d(
				.Reparam,
				_p({u32(n), 1, 0, 0, 0, 0, 0, 0}, {0.5, 0, 0, 0, 0, 0, 0, 0}),
				{zd, md, dy, _in_grad(lv)},
				n,
			)
			t._grad_written_on_device(lv)
		}
	case:
		return false
	}
	return true
}
