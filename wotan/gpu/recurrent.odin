package gpu

import t "../tensor"

// Recurrent layers on the GPU. The input projection x @ W_ih of all steps is
// one matrix product; the recurrence runs in one launch, one workgroup per
// sequence (the sequences are independent), with a barrier between steps.
// The backward pass runs back through time in one launch, which writes the
// gate gradients of every step; the weight, bias and input gradients are then
// products over all steps at once, as on the CPU.

// _lstm_saved: values the LSTM forward pass keeps (the CPU's layout, see
// tensor_lstm): hs and cs with S + 1 rows, then i, f, g, o, tanh(c)
@(private)
_lstm_saved :: proc(B, S, H: int) -> int {
	return 2 * (S + 1) * B * H + 5 * S * B * H
}

// _seq_grid: one workgroup per sequence b (b = wid.y · nw.x + wid.x)
@(private)
_seq_grid :: proc(B: int) -> (gx, gy: u32) {
	x := min(B, 65535)
	return u32(x), u32((B + x - 1) / x)
}

// _lstm_forward: ints B, S, in, H; inputs x, h_0, c_0, W_ih, W_hh, bias
@(private)
_lstm_forward :: proc(out: ^t.Tensor, inputs: []^t.Tensor, ints: []int) {
	B, S, in_f, H := ints[0], ints[1], ints[2], ints[3]
	if B == 0 || S == 0 || H == 0 {return}
	x, h0, c0, w_ih, w_hh, bias := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
	G := 4 * H
	proj := _acquire(B * S * G)
	_matmul(_data(x), _data(w_ih), {proj, B * S * G}, B * S, G, in_f, false, false, false)
	gx, gy := _seq_grid(B)
	_dispatch(
		.Lstm_Fwd,
		_p({u32(B), u32(S), u32(H), 0, 0, 0, 0, 0}),
		{
			{proj, B * S * G},
			_data(w_hh),
			_data(bias),
			_data(h0),
			_data(c0),
			_data(out),
			_bind(out.device.aux_buf, _lstm_saved(B, S, H)),
		},
		gx,
		gy,
	)
	_release_buf(proj) // later reuse is recorded after the dispatches
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
	saved := _bind(node.device.aux_buf, _lstm_saved(B, S, H))
	dG, hb, carry := _acquire(R * G), _acquire(R * H), _acquire(2 * B * H)
	wt := _acquire(H * G)
	defer {_release_buf(dG); _release_buf(hb); _release_buf(carry); _release_buf(wt)}
	dg := Bind{dG, R * G}
	hs := Bind{hb, R * H}

	// 1. back through time; dh_0 and dc_0 at the end
	wanted, overwrite: u32
	dh0, dc0 := _dummy(), _dummy()
	// W_hhᵀ for the kernel's dh = dG @ W_hhᵀ
	whh := Bind{wt, H * G}
	_dispatch_1d(.Transpose, _p({u32(H), u32(G), 0, 0, 0, 0, 0, 0}), {_in_data(w_hh), whh}, H * G)
	if h0.requires_grad {
		fresh: bool
		dh0, fresh = _out_grad(h0)
		wanted |= 1
		overwrite |= u32(fresh)
	}
	if c0.requires_grad {
		fresh: bool
		dc0, fresh = _out_grad(c0)
		wanted |= 2
		overwrite |= u32(fresh) << 1
	}
	gx, gy := _seq_grid(B)
	_dispatch(
		.Lstm_Bwd,
		_p({u32(B), u32(S), u32(H), wanted, overwrite, 0, 0, 0}),
		{dy, saved, whh, dg, hs, {carry, 2 * B * H}, dh0, dc0},
		gx,
		gy,
	)

	// 2. all steps at once: db += Σ dG, dW_ih += xᵀ dG, dW_hh += hᵀ dG,
	// dx += dG W_ihᵀ (rows b·S + s: x's own layout)
	if bias.requires_grad {
		g, fresh := _out_grad(bias)
		_dispatch_1d(.Col_Sum, _p({u32(R), u32(G), 0, 0, 0, 0, 0, u32(fresh)}), {dg, g}, G)
	}
	if w_ih.requires_grad {
		xd := _in_data(x)
		g, fresh := _out_grad(w_ih)
		_matmul(xd, dg, g, in_f, G, R, true, false, !fresh)
	}
	if w_hh.requires_grad {
		g, fresh := _out_grad(w_hh)
		_matmul(hs, dg, g, H, G, R, true, false, !fresh)
	}
	if x.requires_grad {
		wd := _in_data(w_ih)
		g, fresh := _out_grad(x)
		_matmul(dg, wd, g, R, in_f, G, false, true, !fresh)
	}
}
