package gpu

import t "../tensor"
import "core:fmt"
import "core:math"

// ============================================================================
// Device ops, part 2: softmax family, losses, normalisation, dropout,
// embeddings, head permutes and attention
// ============================================================================

// _dispatch_rows: one workgroup per row (row kernels index with row_of)
@(private)
_dispatch_rows :: proc(k: Kernel, p: Params, binds: []Bind, rows: int) {
	if rows <= 0 {return}
	gx := min(rows, 65535)
	gy := (rows + gx - 1) / gx
	_dispatch(k, p, binds, u32(gx), u32(gy))
}

@(private)
_dummy :: proc() -> Bind {
	return {G.dummy, 1}
}

// _upload_values: a pooled buffer holding `values` (release after recording)
@(private)
_upload_values :: proc(values: []f64) -> Bind {
	b := _acquire(len(values))
	_upload(b, values)
	return {b, len(values)}
}

// _upload_ints: integer metadata (class indices) as f32 (exact below 2^24)
@(private)
_upload_ints :: proc(values: []int) -> Bind {
	tmp := make([]f64, len(values), context.temp_allocator)
	for v, i in values {tmp[i] = f64(v)}
	return _upload_values(tmp)
}

// _aux: the op state saved by the forward (dropout or attention mask) on the
// device; uploaded from the host copy when the forward ran on the CPU
@(private)
_aux :: proc(node: ^t.Tensor, host: []f64) -> (b: Bind, temporary: bool) {
	if node.device != nil && node.device.aux_buf.ptr != nil {
		return _bind(node.device.aux_buf, len(host)), false
	}
	return _upload_values(host), true
}

// ----------------------------------------------------------------------------
// Attention: S = Q Kᵀ, P = softmax(scale · S + mask) per row, O = P V (batched)
// ----------------------------------------------------------------------------

@(private)
Attn :: struct {
	batch, sq, sk, dk, dv: int,
	scale:                 f32,
	mask:                  Bind, // dummy when mask_len == 0
	mask_len:              int,
	causal:                bool,
	// > 0: Q, K, V and the output are [B, S, heads·d] and batch is B·heads
	// (MultiHeadAttention); 0: contiguous [batch, S, d] blocks
	heads:                 int,
}

// _hl: the head layout of a matmul in attention (none for contiguous operands)
@(private)
_hl :: proc(a: Attn, on_a, on_b, on_c: bool) -> Heads {
	if a.heads == 0 {return {}}
	return {a.heads, on_a, on_b, on_c}
}

@(private)
_attn_dims :: proc(Q, K, V: ^t.Tensor) -> Attn {
	a := Attn {
		batch = Q.shape[0],
		sq    = Q.shape[1],
		sk    = K.shape[1],
		dk    = Q.shape[2],
		dv    = V.shape[2],
	}
	a.scale = f32(1.0 / math.sqrt(f64(a.dk)))
	if a.batch >
	   65535 {panic(fmt.tprintf("gpu attention: %d sequences exceed the device's 65535 batch limit", a.batch))}
	return a
}

// _attn_probs records P = softmax(scale · Q Kᵀ + mask) into a new buffer
@(private)
_attn_probs :: proc(a: Attn, q, k: Bind) -> (P: Bind) {
	n := a.batch * a.sq * a.sk
	S := Bind{_acquire(n), n}
	_matmul(
		q,
		k,
		S,
		a.sq,
		a.sk,
		a.dk,
		false,
		true,
		false,
		a.batch,
		heads = _hl(a, true, true, false),
	)
	P = {_acquire(n), n}
	u := [8]u32{u32(a.batch * a.sq), u32(a.sk), 0, u32(a.mask_len), u32(a.causal), u32(a.sq), 0, 0}
	_dispatch_rows(
		.Softmax_Rows,
		_p(u, {a.scale, 0, 0, 0, 0, 0, 0, 0}),
		{S, a.mask, P},
		a.batch * a.sq,
	)
	_release_buf(S.buf)
	return
}

@(private)
_attention_setup :: proc(
	node: ^t.Tensor,
	op: t.Op,
	Q, K, V: ^t.Tensor,
	ints: []int,
) -> (
	a: Attn,
	temp_mask: bool,
) {
	a = _attn_dims(Q, K, V)
	a.mask = _dummy()
	#partial switch op {
	case .MaskedScaledDotProductAttention:
		if mask := node.f64_metadata[:]; len(ints) > 0 && ints[0] > 0 {
			// forward: the mask was uploaded into the output's aux buffer
			a.mask, a.mask_len = _bind(node.device.aux_buf, ints[0]), ints[0]
		} else if len(mask) > 0 {
			a.mask, temp_mask = _aux(node, mask)
			a.mask_len = len(mask)
		}
	case .FlashAttention:
		if len(ints) > 1 {
			a.causal = ints[1] == 1
		} else {
			a.causal = len(node.int_metadata) > 0 && node.int_metadata[0] == 1
		}
	case:
	}
	return
}

// _mha_setup: the attention of a MultiHeadAttention node. m is its int_metadata
// [batch, seq_q, seq_k, heads, head_dim, kind, causal] (kind 1: masked, 2:
// flash); mask_len > 0 in the forward, whose mask is in the aux buffer.
@(private)
_mha_setup :: proc(node: ^t.Tensor, m: []int, mask_len: int) -> (a: Attn, temp_mask: bool) {
	B, H, hd := m[0], m[3], m[4]
	a = {
		batch = B * H,
		sq    = m[1],
		sk    = m[2],
		dk    = hd,
		dv    = hd,
		heads = H,
	}
	a.scale = f32(1.0 / math.sqrt(f64(hd)))
	if a.batch >
	   65535 {panic(fmt.tprintf("gpu attention: %d sequences x heads exceed the device's 65535 batch limit", a.batch))}
	a.mask = _dummy()
	if m[5] == 1 {
		if mask_len > 0 {
			a.mask, a.mask_len = _bind(node.device.aux_buf, mask_len), mask_len
		} else {
			a.mask, temp_mask = _aux(node, node.f64_metadata[:])
			a.mask_len = len(node.f64_metadata)
		}
	}
	a.causal = m[5] == 2 && m[6] == 1
	return
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

@(private)
_forward_p2 :: proc(op: t.Op, out: ^t.Tensor, inputs: []^t.Tensor, ints: []int, floats: []f64) {
	o := _data(out)
	n := t.tensor_numel(out)
	#partial switch op {
	case .Softmax:
		C := ints[0]
		u := [8]u32{u32(n / C), u32(C), 0, 0, 0, 1, 0, 0}
		_dispatch_rows(
			.Softmax_Rows,
			_p(u, {1, 0, 0, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), _dummy(), o},
			n / C,
		)
	case .LogSumExpDim1:
		batch, C := ints[0], ints[1]
		u := [8]u32{u32(batch), u32(C), 1, 0, 0, 1, 0, 0}
		_dispatch_rows(
			.Softmax_Rows,
			_p(u, {1, 0, 0, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), _dummy(), o},
			batch,
		)
	case .CrossEntropy:
		N, C := ints[0], ints[1]
		targets := _upload_ints(ints[2:])
		rows := Bind{_acquire(N), N}
		u := [8]u32{u32(N), u32(C), 2, 0, 0, 1, 0, 0}
		_dispatch_rows(
			.Softmax_Rows,
			_p(u, {1, 0, 0, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), targets, rows},
			N,
		)
		_reduce(rows, N, o, 0, f32(1.0 / f64(N)), false)
		_release_buf(rows.buf)
		_release_buf(targets.buf)
	case .BCELoss, .BinaryCrossEntropy:
		pred, target := inputs[0], inputs[1]
		m := t.tensor_numel(pred)
		loss := Bind{_acquire(m), m}
		_dispatch_1d(
			.Bce,
			_p({u32(m), 0, 0, 0, 0, 0, 0, 0}),
			{_data(pred), _data(target), _dummy(), loss},
			m,
		)
		_reduce(loss, m, o, 0, f32(1.0 / f64(m)), false)
		_release_buf(loss.buf)
	case .LayerNorm, .RMSNorm:
		N, D := ints[0], ints[1]
		rms := op == .RMSNorm
		gamma := _data(inputs[1])
		beta := gamma if rms else _data(inputs[2])
		// the statistics go to the aux buffer when the backward will need them
		keep := out.device.aux_buf.ptr != nil
		stats := _bind(out.device.aux_buf, 2 * N) if keep else _dummy()
		u := [8]u32{u32(N), u32(D), u32(rms), u32(keep), 0, 0, 0, 0}
		_dispatch_rows(
			.Norm_Fwd,
			_p(u, {f32(floats[0]), 0, 0, 0, 0, 0, 0, 0}),
			{_data(inputs[0]), gamma, beta, o, stats},
			N,
		)
	case .Dropout:
		if out.dropout_mask == nil {
			_copy(o.buf, _buf(inputs[0].device.data_buf), n)
		} else {
			mask := _bind(out.device.aux_buf, n)
			_dispatch_1d(
				.Binary,
				_p({u32(n), 2, 0, 0, 0, 0, 0, 0}),
				{_data(inputs[0]), mask, o},
				n,
			)
		}
	case .Embedding:
		rows, E, vocab := ints[0], ints[1], ints[2]
		u := [8]u32{u32(rows), u32(E), u32(vocab), 0, 0, 0, 0, 0}
		_dispatch_1d(.Gather_Rows, _p(u), {_data(inputs[0]), _data(inputs[1]), o}, rows * E)
	case .PermuteMHA, .PermuteMHAInverse:
		dir := u32(op == .PermuteMHAInverse)
		u := [8]u32{u32(ints[0]), u32(ints[1]), u32(ints[2]), u32(ints[3]), dir, 0, 0, 0}
		_dispatch_1d(.Permute_Mha, _p(u), {_data(inputs[0]), o}, n)
	case .ScaledDotProductAttention, .MaskedScaledDotProductAttention, .FlashAttention:
		Q, K, V := inputs[0], inputs[1], inputs[2]
		a, temp := _attention_setup(out, op, Q, K, V, ints)
		P := _attn_probs(a, _data(Q), _data(K))
		_matmul(P, _data(V), o, a.sq, a.dv, a.sk, false, false, false, a.batch)
		_release_buf(P.buf)
		if temp {_release_buf(a.mask.buf)}
	case .MultiHeadAttention:
		a, temp := _mha_setup(out, ints[:7], ints[7])
		P := _attn_probs(a, _data(inputs[0]), _data(inputs[1]))
		_matmul(
			P,
			_data(inputs[2]),
			o,
			a.sq,
			a.dv,
			a.sk,
			false,
			false,
			false,
			a.batch,
			heads = _hl(a, false, true, true),
		)
		_release_buf(P.buf)
		if temp {_release_buf(a.mask.buf)}
	case .GateMul:
		x := inputs[0]
		u := [8]u32{u32(n), u32(x.shape[1]), 0, 0, 0, 0, 0, 0}
		_dispatch_1d(.Row_Scale, _p(u), {_data(x), _data(inputs[1]), o}, n)
	case:
		_forward_p3(op, out, inputs, ints, floats)
	}
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------

@(private)
_backward_p2 :: proc(node: ^t.Tensor, dy: Bind, n: int) -> bool {
	#partial switch node.op {
	case .Softmax:
		a := node.inputs[0]
		if !a.requires_grad {return true}
		C := node.int_metadata[0]
		g, fresh := _out_grad(a)
		u := [8]u32{u32(n / C), u32(C), u32(!fresh), 0, 0, 0, 0, 0}
		_dispatch_rows(
			.Softmax_Bwd,
			_p(u, {1, 0, 0, 0, 0, 0, 0, 0}),
			{_in_data(node), dy, g},
			n / C,
		)
		t._grad_written_on_device(a)
	case .LogSumExpDim1:
		a := node.inputs[0]
		if !a.requires_grad {return true}
		C := node.int_metadata[0]
		batch := node.data.rows
		g, fresh := _out_grad(a)
		u := [8]u32{u32(batch), u32(C), 0, 0, 0, 0, 0, u32(fresh)}
		_dispatch_rows(.Row_Prob_Grad, _p(u), {_in_data(a), _dummy(), dy, g}, batch)
		t._grad_written_on_device(a)
	case .CrossEntropy:
		a := node.inputs[0]
		if !a.requires_grad {return true}
		N, C := node.int_metadata[0], node.int_metadata[1]
		targets := _upload_ints(node.int_metadata[2:])
		g, fresh := _out_grad(a)
		u := [8]u32{u32(N), u32(C), 1, 0, 0, 0, 0, u32(fresh)}
		f := [8]f32{f32(1.0 / f64(N)), 0, 0, 0, 0, 0, 0, 0}
		x := _in_data(a)
		_dispatch_rows(.Row_Prob_Grad, _p(u, f), {x, targets, dy, g}, N)
		_release_buf(targets.buf)
		t._grad_written_on_device(a)
	case .BCELoss, .BinaryCrossEntropy:
		pred, target := node.inputs[0], node.inputs[1]
		m := t.tensor_numel(pred)
		f := [8]f32{f32(1.0 / f64(m)), 0, 0, 0, 0, 0, 0, 0}
		for x, i in node.inputs {
			if !x.requires_grad {continue}
			pd, td := _in_data(pred), _in_data(target)
			_dispatch_1d(
				.Bce,
				_p({u32(m), u32(1 + i), 0, 0, 0, 0, 0, 0}, f),
				{pd, td, dy, _in_grad(x)},
				m,
			)
			t._grad_written_on_device(x)
		}
	case .LayerNorm, .RMSNorm:
		_norm_backward(node, dy)
	case .Dropout:
		a := node.inputs[0]
		if !a.requires_grad {return true}
		if node.dropout_mask == nil {
			g, fresh := _out_grad(a)
			_dispatch_1d(
				.Unary_Bwd,
				_p({u32(n), U_COPY, 0, 0, 0, 0, 0, u32(fresh)}, {1, 0, 0, 0, 0, 0, 0, 0}),
				{dy, dy, dy, g},
				n,
			)
		} else {
			mask, temp := _aux(node, node.dropout_mask)
			g, fresh := _out_grad(a)
			// dx += dy * mask (Binary_Bwd, gradient of a for a * mask)
			_dispatch_1d(
				.Binary_Bwd,
				_p({u32(n), 2, 0, 0, 0, 0, 0, u32(fresh)}),
				{mask, mask, dy, g},
				n,
			)
			if temp {_release_buf(mask.buf)}
		}
		t._grad_written_on_device(a)
	case .Embedding:
		ids, w := node.inputs[0], node.inputs[1]
		if !w.requires_grad {return true}
		rows := ids.shape[0] * ids.shape[1]
		E, vocab := w.shape[1], w.shape[0]
		u := [8]u32{u32(rows), u32(E), u32(vocab), 0, 0, 0, 0, 0}
		_dispatch_1d(.Scatter_Add_Rows, _p(u), {_in_data(ids), dy, _in_grad(w)}, rows * E)
		t._grad_written_on_device(w)
	case .PermuteMHA, .PermuteMHAInverse:
		a := node.inputs[0]
		if !a.requires_grad {return true}
		m := node.int_metadata
		// the backward of one direction is the other direction, accumulated
		dir := u32(node.op == .PermuteMHA)
		g, fresh := _out_grad(a)
		u := [8]u32{u32(m[0]), u32(m[1]), u32(m[2]), u32(m[3]), dir, u32(!fresh), 0, 0}
		_dispatch_1d(.Permute_Mha, _p(u), {dy, g}, n)
		t._grad_written_on_device(a)
	case .ScaledDotProductAttention,
	     .MaskedScaledDotProductAttention,
	     .FlashAttention,
	     .MultiHeadAttention:
		_attention_backward(node, dy)
	case .GateMul:
		x, gate := node.inputs[0], node.inputs[1]
		N, D := x.shape[0], x.shape[1]
		if x.requires_grad {
			gd := _in_data(gate)
			g, fresh := _out_grad(x)
			_dispatch_1d(
				.Row_Scale,
				_p({u32(N * D), u32(D), u32(!fresh), 0, 0, 0, 0, 0}),
				{dy, gd, g},
				N * D,
			)
			t._grad_written_on_device(x)
		}
		if gate.requires_grad {
			xd := _in_data(x)
			g, fresh := _out_grad(gate)
			_dispatch_1d(.Row_Dot, _p({u32(N), u32(D), 0, 0, 0, 0, 0, u32(fresh)}), {dy, xd, g}, N)
			t._grad_written_on_device(gate)
		}
	case:
		return _backward_p3(node, dy, n)
	}
	return true
}

// layer norm / RMS norm: per-row statistics, then the input and parameter gradients
@(private)
_norm_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x, gamma := node.inputs[0], node.inputs[1]
	rms := node.op == .RMSNorm
	beta: ^t.Tensor = nil if rms else node.inputs[2]
	N, D := node.int_metadata[0], node.int_metadata[1]
	eps := f32(node.f64_metadata[0])
	xd, gd := _in_data(x), _in_data(gamma)
	u := [8]u32{u32(N), u32(D), u32(rms), 0, 0, 0, 0, 0}
	f := [8]f32{eps, 0, 0, 0, 0, 0, 0, 0}
	// the statistics the device forward kept; recomputed when the forward ran
	// on the CPU
	stats: Bind
	kept := node.device != nil && node.device.aux_buf.ptr != nil
	if kept {
		stats = _bind(node.device.aux_buf, 2 * N)
	} else {
		stats = {_acquire(2 * N), 2 * N}
		_dispatch_rows(.Norm_Stats, _p(u, f), {xd, stats}, N)
	}
	if x.requires_grad {
		g, fresh := _out_grad(x)
		ux := u
		ux[7] = u32(fresh)
		_dispatch_rows(.Norm_Bwd_X, _p(ux, f), {xd, gd, dy, stats, g}, N)
		t._grad_written_on_device(x)
	}
	want_g := gamma.requires_grad
	want_b := beta != nil && beta.requires_grad
	if want_g || want_b {
		dg, db := _dummy(), _dummy()
		fresh_g, fresh_b: bool
		if want_g {dg, fresh_g = _out_grad(gamma)}
		if want_b {db, fresh_b = _out_grad(beta)}
		u2 := [8]u32 {
			u32(N),
			u32(D),
			u32(rms),
			u32(want_b),
			u32(want_g),
			0,
			u32(fresh_g),
			u32(fresh_b),
		}
		_dispatch_1d(.Norm_Bwd_Param, _p(u2), {xd, dy, stats, dg, db}, D)
		if want_g {t._grad_written_on_device(gamma)}
		if want_b {t._grad_written_on_device(beta)}
	}
	if !kept {_release_buf(stats.buf)}
}

// attention: recompute P, then dV += Pᵀ dO, dS = scale · P ⊙ (dO Vᵀ - rowsum),
// dQ += dS K, dK += dSᵀ Q (for MultiHeadAttention, Q, K, V, dO and the
// gradients in head layout)
@(private)
_attention_backward :: proc(node: ^t.Tensor, dO: Bind) {
	Q, K, V := node.inputs[0], node.inputs[1], node.inputs[2]
	a: Attn
	temp: bool
	if node.op == .MultiHeadAttention {
		a, temp = _mha_setup(node, node.int_metadata[:], 0)
	} else {
		a, temp = _attention_setup(node, node.op, Q, K, V, nil)
	}
	q, k, v := _in_data(Q), _in_data(K), _in_data(V)
	P := _attn_probs(a, q, k)
	if V.requires_grad {
		g, fresh := _out_grad(V)
		_matmul(
			P,
			dO,
			g,
			a.sk,
			a.dv,
			a.sq,
			true,
			false,
			!fresh,
			a.batch,
			heads = _hl(a, false, true, true),
		)
		t._grad_written_on_device(V)
	}
	if Q.requires_grad || K.requires_grad {
		m := a.batch * a.sq * a.sk
		dP := Bind{_acquire(m), m}
		_matmul(
			dO,
			v,
			dP,
			a.sq,
			a.sk,
			a.dv,
			false,
			true,
			false,
			a.batch,
			heads = _hl(a, true, true, false),
		)
		dS := Bind{_acquire(m), m}
		u := [8]u32{u32(a.batch * a.sq), u32(a.sk), 0, 0, 0, 0, 0, 0}
		_dispatch_rows(
			.Softmax_Bwd,
			_p(u, {a.scale, 0, 0, 0, 0, 0, 0, 0}),
			{P, dP, dS},
			a.batch * a.sq,
		)
		if Q.requires_grad {
			g, fresh := _out_grad(Q)
			_matmul(
				dS,
				k,
				g,
				a.sq,
				a.dk,
				a.sk,
				false,
				false,
				!fresh,
				a.batch,
				heads = _hl(a, false, true, true),
			)
			t._grad_written_on_device(Q)
		}
		if K.requires_grad {
			g, fresh := _out_grad(K)
			_matmul(
				dS,
				q,
				g,
				a.sk,
				a.dk,
				a.sq,
				true,
				false,
				!fresh,
				a.batch,
				heads = _hl(a, false, true, true),
			)
			t._grad_written_on_device(K)
		}
		_release_buf(dP.buf)
		_release_buf(dS.buf)
	}
	_release_buf(P.buf)
	if temp {_release_buf(a.mask.buf)}
}
