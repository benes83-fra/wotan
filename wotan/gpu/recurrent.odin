package gpu

import t "../tensor"

// Recurrent layers on the GPU (LSTM, GRU, tanh RNN). The input projection
// x @ W_ih of all steps is one matrix product; the recurrence runs with one
// workgroup per sequence (the sequences are independent) and a barrier between
// steps. The backward pass runs back through time and writes the gate
// gradients of every step; the weight, bias and input gradients are then
// products over all steps at once, as on the CPU. Rows of the per-step buffers
// are b·S + s, x's own layout.
//
// Both passes are serial in time, so a large layer keeps the GPU busy for a
// long time (about 1 s for the backward pass of 160 sequences of 160 steps
// with H = 768 on a laptop RTX 3060). Windows resets a GPU whose submission
// runs longer than about 2 s, so a pass runs as several launches of a block
// of steps each (REC_LAUNCH_WORK) and, when it needs more than one, every
// launch is submitted on its own. The states between blocks stay in the
// saved and carry buffers, so the values do not depend on the blocks.

// REC_LAUNCH_WORK: the multiply-adds of hidden-to-hidden products one
// recurrent launch may run (B · H · gate columns per step); about 35 ms for
// the backward pass on a laptop RTX 3060
@(private)
REC_LAUNCH_WORK :: 1 << 31

// _rec_steps: the steps per launch of a recurrence over B sequences with
// hidden size H and `gates` gate columns
@(private)
_rec_steps :: proc(B, S, H, gates: int) -> int {
	if G.rec_steps > 0 {return min(G.rec_steps, S)}
	return clamp(REC_LAUNCH_WORK / max(B * H * gates, 1), 1, S)
}

// _rec_launch: the recurrent kernel k over all S steps, in blocks of
// _rec_steps (u5 ..< u6 the block's steps, counted in the kernel's own
// direction); with more than one block every launch is submitted on its own
@(private)
_rec_launch :: proc(k: Kernel, p: Params, binds: []Bind, B, S, H, gates: int) {
	n := _rec_steps(B, S, H, gates)
	gx, gy := _seq_grid(B)
	for s0 := 0; s0 < S; s0 += n {
		q := p
		q.u[5], q.u[6] = u32(s0), u32(min(s0 + n, S))
		_dispatch(k, q, binds, gx, gy)
		if n < S {_flush()}
	}
}

// values the forward passes keep (the CPU's layouts, see tensor_lstm,
// tensor_gru, tensor_rnn): state rows (S + 1 each), then per-step blocks
@(private)
_lstm_saved :: proc(B, S, H: int) -> int {return 2 * (S + 1) * B * H + 5 * S * B * H}
@(private)
_gru_saved :: proc(B, S, H: int) -> int {return (S + 1) * B * H + 4 * S * B * H}
@(private)
_rnn_saved :: proc(B, S, H: int) -> int {return (S + 1) * B * H}

// _seq_grid: one workgroup per sequence b (b = wid.y · nw.x + wid.x)
@(private)
_seq_grid :: proc(B: int) -> (gx, gy: u32) {
	x := min(B, 65535)
	return u32(x), u32((B + x - 1) / x)
}

// _rec_forward: the input projection x @ W_ih [B·S, G], then the recurrence
// kernel k with these bindings after proj
@(private)
_rec_forward :: proc(
	k: Kernel,
	x, w_ih: ^t.Tensor,
	B, S, in_f, H, G: int,
	flags: u32,
	binds: []Bind,
) {
	proj := _acquire(B * S * G)
	_matmul(_data(x), _data(w_ih), {proj, B * S * G}, B * S, G, in_f, false, false, false)
	all: [8]Bind
	all[0] = {proj, B * S * G}
	copy(all[1:], binds)
	_rec_launch(
		k,
		_p({u32(B), u32(S), u32(H), flags, 0, 0, 0, 0}),
		all[:1 + len(binds)],
		B,
		S,
		H,
		G,
	)
	_release_buf(proj) // later reuse is recorded after the dispatches
}

// _w_hh_t: W_hhᵀ [G, H] in a pool buffer (the backward kernels read it so that
// neighbouring threads read neighbouring weights); the caller releases it
@(private)
_w_hh_t :: proc(w_hh: ^t.Tensor, H, G: int) -> Bind {
	wt := Bind{_acquire(H * G), H * G}
	_dispatch_1d(.Transpose, _p({u32(H), u32(G), 0, 0, 0, 0, 0, 0}), {_in_data(w_hh), wt}, H * G)
	return wt
}

// _h0_grad: the binding, wanted bit and overwrite bit of an initial state's
// gradient (the dummy buffer when it needs none)
@(private)
_h0_grad :: proc(x: ^t.Tensor, bit: uint) -> (g: Bind, wanted, overwrite: u32) {
	if !x.requires_grad {return _dummy(), 0, 0}
	fresh: bool
	g, fresh = _out_grad(x)
	return g, 1 << bit, u32(fresh) << bit
}

// _rec_param_grads: after backpropagation through time, over all steps at once
//   db += Σ rows of d_ih, dW_ih += xᵀ d_ih, dW_hh += hbᵀ d_hh, dx += d_ih W_ihᵀ
// (d_ih, d_hh [B·S, G]; hb [B·S, H] the states entering each step)
@(private)
_rec_param_grads :: proc(
	x, w_ih, w_hh, bias: ^t.Tensor,
	d_ih, d_hh, hb: Bind,
	R, in_f, H, G: int,
) {
	if bias.requires_grad {
		g, fresh := _out_grad(bias)
		_dispatch_1d(.Col_Sum, _p({u32(R), u32(G), 0, 0, 0, 0, 0, u32(fresh)}), {d_ih, g}, G)
	}
	if w_ih.requires_grad {
		xd := _in_data(x)
		g, fresh := _out_grad(w_ih)
		_matmul(xd, d_ih, g, in_f, G, R, true, false, !fresh)
	}
	if w_hh.requires_grad {
		g, fresh := _out_grad(w_hh)
		_matmul(hb, d_hh, g, H, G, R, true, false, !fresh)
	}
	if x.requires_grad {
		wd := _in_data(w_ih)
		g, fresh := _out_grad(x)
		_matmul(d_ih, wd, g, R, in_f, G, false, true, !fresh)
	}
}

// ----------------------------------------------------------------------------
// LSTM: gates [i, f, g, o]; inputs x, h_0, c_0, W_ih, W_hh, bias
// ----------------------------------------------------------------------------

// _lstm_forward: ints B, S, in, H
@(private)
_lstm_forward :: proc(out: ^t.Tensor, inputs: []^t.Tensor, ints: []int) {
	B, S, in_f, H := ints[0], ints[1], ints[2], ints[3]
	if B == 0 || S == 0 || H == 0 {return}
	x, h0, c0, w_ih, w_hh, bias := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
	_rec_forward(
		.Lstm_Fwd,
		x,
		w_ih,
		B,
		S,
		in_f,
		H,
		4 * H,
		0,
		{
			_data(w_hh),
			_data(bias),
			_data(h0),
			_data(c0),
			_data(out),
			_bind(out.device.aux_buf, _lstm_saved(B, S, H)),
		},
	)
}

// _lstm_backward: dy the node's gradient [B, S, H]
@(private)
_lstm_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x, h0, c0 := node.inputs[0], node.inputs[1], node.inputs[2]
	w_ih, w_hh, bias := node.inputs[3], node.inputs[4], node.inputs[5]
	B, S, in_f := x.shape[0], x.shape[1], x.shape[2]
	H := w_ih.shape[1] / 4
	if B == 0 || S == 0 || H == 0 {return}
	G := 4 * H
	R := B * S
	dG, hb, carry := _acquire(R * G), _acquire(R * H), _acquire(2 * B * H)
	wt := _w_hh_t(w_hh, H, G)
	defer {_release_buf(dG); _release_buf(hb); _release_buf(carry); _release_buf(wt.buf)}
	dg, hs := Bind{dG, R * G}, Bind{hb, R * H}

	// 1. back through time; dh_0 and dc_0 at the end
	dh0, w0, o0 := _h0_grad(h0, 0)
	dc0, w1, o1 := _h0_grad(c0, 1)
	_rec_launch(
		.Lstm_Bwd,
		_p({u32(B), u32(S), u32(H), w0 | w1, o0 | o1, 0, 0, 0}),
		{
			dy,
			_bind(node.device.aux_buf, _lstm_saved(B, S, H)),
			wt,
			dg,
			hs,
			{carry, 2 * B * H},
			dh0,
			dc0,
		},
		B,
		S,
		H,
		G,
	)
	// 2. all steps at once
	_rec_param_grads(x, w_ih, w_hh, bias, dg, dg, hs, R, in_f, H, G)
}

// ----------------------------------------------------------------------------
// GRU: gates [r, z, n]; inputs x, h_0, W_ih, W_hh, bias (, bias_hn)
// ----------------------------------------------------------------------------

// _gru_forward: ints B, S, in, H
@(private)
_gru_forward :: proc(out: ^t.Tensor, inputs: []^t.Tensor, ints: []int) {
	B, S, in_f, H := ints[0], ints[1], ints[2], ints[3]
	if B == 0 || S == 0 || H == 0 {return}
	x, h0, w_ih, w_hh, bias := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	has_hn := len(inputs) > 5
	_rec_forward(
		.Gru_Fwd,
		x,
		w_ih,
		B,
		S,
		in_f,
		H,
		3 * H,
		u32(has_hn),
		{
			_data(w_hh),
			_data(bias),
			_data(inputs[5]) if has_hn else _data(bias),
			_data(h0),
			_data(out),
			_bind(out.device.aux_buf, _gru_saved(B, S, H)),
		},
	)
}

@(private)
_gru_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x, h0, w_ih, w_hh, bias :=
		node.inputs[0], node.inputs[1], node.inputs[2], node.inputs[3], node.inputs[4]
	B, S, in_f := x.shape[0], x.shape[1], x.shape[2]
	H := w_ih.shape[1] / 3
	if B == 0 || S == 0 || H == 0 {return}
	G := 3 * H
	R := B * S
	dI, dH, hb, carry := _acquire(R * G), _acquire(R * G), _acquire(R * H), _acquire(2 * B * H)
	wt := _w_hh_t(w_hh, H, G)
	defer {
		_release_buf(dI); _release_buf(dH); _release_buf(hb); _release_buf(carry)
		_release_buf(wt.buf)
	}
	d_ih, d_hh, hs := Bind{dI, R * G}, Bind{dH, R * G}, Bind{hb, R * H}

	// 1. back through time; dh_0 at the end
	dh0, wanted, overwrite := _h0_grad(h0, 0)
	_rec_launch(
		.Gru_Bwd,
		_p({u32(B), u32(S), u32(H), wanted, overwrite, 0, 0, 0}),
		{
			dy,
			_bind(node.device.aux_buf, _gru_saved(B, S, H)),
			wt,
			d_ih,
			d_hh,
			hs,
			{carry, 2 * B * H},
			dh0,
		},
		B,
		S,
		H,
		G,
	)
	// 2. all steps at once; d b_hn = Σ rows of the n columns of d_hh
	_rec_param_grads(x, w_ih, w_hh, bias, d_ih, d_hh, hs, R, in_f, H, G)
	if len(node.inputs) > 5 && node.inputs[5].requires_grad {
		g, fresh := _out_grad(node.inputs[5])
		_dispatch_1d(
			.Col_Sum,
			_p({u32(R), u32(G), u32(2 * H), u32(H), 0, 0, 0, u32(fresh)}),
			{d_hh, g},
			H,
		)
	}
}

// ----------------------------------------------------------------------------
// tanh RNN: inputs x, h_0, W_ih, W_hh, bias
// ----------------------------------------------------------------------------

// _rnn_forward: ints B, S, in, H
@(private)
_rnn_forward :: proc(out: ^t.Tensor, inputs: []^t.Tensor, ints: []int) {
	B, S, in_f, H := ints[0], ints[1], ints[2], ints[3]
	if B == 0 || S == 0 || H == 0 {return}
	x, h0, w_ih, w_hh, bias := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	_rec_forward(
		.Rnn_Fwd,
		x,
		w_ih,
		B,
		S,
		in_f,
		H,
		H,
		0,
		{
			_data(w_hh),
			_data(bias),
			_data(h0),
			_data(out),
			_bind(out.device.aux_buf, _rnn_saved(B, S, H)),
		},
	)
}

@(private)
_rnn_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x, h0, w_ih, w_hh, bias :=
		node.inputs[0], node.inputs[1], node.inputs[2], node.inputs[3], node.inputs[4]
	B, S, in_f := x.shape[0], x.shape[1], x.shape[2]
	H := w_ih.shape[1]
	if B == 0 || S == 0 || H == 0 {return}
	R := B * S
	dG, hb, carry := _acquire(R * H), _acquire(R * H), _acquire(B * H)
	wt := _w_hh_t(w_hh, H, H)
	defer {_release_buf(dG); _release_buf(hb); _release_buf(carry); _release_buf(wt.buf)}
	dg, hs := Bind{dG, R * H}, Bind{hb, R * H}

	dh0, wanted, overwrite := _h0_grad(h0, 0)
	_rec_launch(
		.Rnn_Bwd,
		_p({u32(B), u32(S), u32(H), wanted, overwrite, 0, 0, 0}),
		{dy, _bind(node.device.aux_buf, _rnn_saved(B, S, H)), wt, dg, hs, {carry, B * H}, dh0},
		B,
		S,
		H,
		H,
	)
	_rec_param_grads(x, w_ih, w_hh, bias, dg, dg, hs, R, in_f, H, H)
}

// ----------------------------------------------------------------------------
// Mamba's selective scan (tensor_ssm): inputs x, h_0, A, B, C, Delta, D
// ----------------------------------------------------------------------------

// storage buffers Ssm_Bwd binds (devices that bind fewer use the CPU)
@(private)
SSM_BINDS :: 12

// _ssm_aux: the forward pass keeps exp(Δ·A) of every step (the CPU's saved
// layout, first) and the states entering every step: 2 · B·S·DN values
@(private)
_ssm_aux :: proc(B, S, DN: int) -> int {return 2 * B * S * DN}

// _ssm_grid: workgroups of 256 / N channels each (b = wid.y · nw.x + wid.x)
@(private)
_ssm_grid :: proc(B, Dm, N: int) -> (gx, gy: u32) {
	per := 256 / N
	groups := (B * Dm + per - 1) / per
	x := min(groups, 65535)
	return u32(x), u32((groups + x - 1) / x)
}

// _ssm_forward: ints B, S, d_model, d_state (d_state ≤ t.DEVICE_SSM_MAX_STATE)
@(private)
_ssm_forward :: proc(out: ^t.Tensor, inputs: []^t.Tensor, ints: []int) {
	B, S, Dm, N := ints[0], ints[1], ints[2], ints[3]
	if B == 0 || S == 0 || Dm == 0 || N == 0 {return}
	binds: [9]Bind
	for x, i in inputs {binds[i] = _data(x)}
	binds[7] = _data(out)
	binds[8] = _bind(out.device.aux_buf, _ssm_aux(B, S, Dm * N))
	gx, gy := _ssm_grid(B, Dm, N)
	_dispatch(.Ssm_Fwd, _p({u32(B), u32(S), u32(Dm), u32(N), 0, 0, 0, 0}), binds[:], gx, gy)
}

// _ssm_backward: dy the node's gradient [B, S, d_model]
@(private)
_ssm_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x, h0, A, Bt, C, Dl, D :=
		node.inputs[0],
		node.inputs[1],
		node.inputs[2],
		node.inputs[3],
		node.inputs[4],
		node.inputs[5],
		node.inputs[6]
	B, S, Dm := node.shape[0], node.shape[1], node.shape[2]
	N := A.shape[1]
	if B == 0 || S == 0 || Dm == 0 || N == 0 {return}
	DN := Dm * N
	TS := B * S * DN
	terms := _acquire(3 * TS)
	defer _release_buf(terms)
	tb := Bind{terms, 3 * TS}
	reads := [8]Bind {
		_in_data(x),
		_in_data(A),
		_in_data(Bt),
		_in_data(C),
		_in_data(Dl),
		_in_data(D),
		dy,
		_bind(node.device.aux_buf, _ssm_aux(B, S, DN)),
	}

	// 1. back through the scan: dx, dΔ, dh_0 and the terms
	wanted, overwrite: u32
	outs := [3]Bind{_dummy(), _dummy(), _dummy()}
	for src, k in ([3]^t.Tensor{x, Dl, h0}) {
		if !src.requires_grad {continue}
		fresh: bool
		outs[k], fresh = _out_grad(src)
		wanted |= 1 << uint(k)
		overwrite |= u32(fresh) << uint(k)
	}
	binds: [12]Bind
	copy(binds[:], reads[:])
	binds[8], binds[9], binds[10], binds[11] = tb, outs[0], outs[1], outs[2]
	gx, gy := _ssm_grid(B, Dm, N)
	p := _p({u32(B), u32(S), u32(Dm), u32(N), wanted, overwrite, 0, 0})
	_dispatch(.Ssm_Bwd, p, binds[:], gx, gy)

	// 2. dB, dC (over the channels) and dD (over steps and rows) in one launch
	wanted, overwrite = 0, 0
	grads := [3]Bind{_dummy(), _dummy(), _dummy()}
	for src, k in ([3]^t.Tensor{Bt, C, D}) {
		if !src.requires_grad {continue}
		fresh: bool
		grads[k], fresh = _out_grad(src)
		wanted |= 1 << uint(k)
		overwrite |= u32(fresh) << uint(k)
	}
	if wanted != 0 {
		p = _p({u32(B), u32(S), u32(Dm), u32(N), wanted, overwrite, 0, 0})
		_dispatch_1d(
			.Ssm_Reduce,
			p,
			{tb, dy, reads[0], grads[0], grads[1], grads[2]},
			max(B * S * N, Dm),
		)
	}

	// 3. dA: the dA terms (first block of terms) summed over steps and rows
	if A.requires_grad {
		g, fresh := _out_grad(A)
		_dispatch_1d(.Col_Sum, _p({u32(B * S), u32(DN), 0, 0, 0, 0, 0, u32(fresh)}), {tb, g}, DN)
	}
}
