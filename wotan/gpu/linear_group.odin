package gpu

import t "../tensor"

// Grouped linear layers. Attention projects one input three times (queries,
// keys, values): three tensor_linear calls on the same x, recorded one after
// the other, and in the backward pass three nodes processed one after the
// other. Instead of launching each at once, a Linear forward or backward is
// held; the next one on the same input with the same sizes joins it, up to
// LINEAR_GROUP, and the group runs as one launch per kernel:
//   forward   out_m = x @ W_m + b_m                 (Matmul_Group, one z per m)
//   backward  db_m += Σ rows of dy_m                (Col_Sum_Group)
//             dx += dy_0 @ W_0ᵀ + dy_1 @ W_1ᵀ + ...  (Matmul_Group, sum mode)
//             dW_m += xᵀ @ dy_m                      (Matmul_Group, one z per m)
// The values are those of one launch per layer (the same kernels' arithmetic,
// dx summed in the same order). Every other command (a launch, upload,
// download, clear, copy or submit) first records what is held, so the
// program order of reads and writes does not change. A group needs distinct
// buffers for what it writes; otherwise the held layers run first.

// layers per group at most (kernels bind A, B, C and the bias of each)
@(private)
LINEAR_GROUP :: 3

@(private)
Held_Kind :: enum {
	None,
	Forward,
	Backward,
}

@(private)
Held_Layer :: struct {
	w, bias, out:              Bind, // forward
	dy, w_data, gx, gw, gb:    Bind, // backward
	fresh_x, fresh_w, fresh_b: bool,
}

@(private)
Held :: struct {
	kind:                   Held_Kind,
	x:                      Bind, // the shared input (data)
	rows, in_f, out_f:      int,
	need_x, need_w, need_b: bool, // backward: which gradients
	count:                  int,
	layers:                 [LINEAR_GROUP]Held_Layer,
}

@(private)
H: Held

// _grouping: whether linear layers are grouped (on unless switched off with
// set_linear_grouping, and the device binds the 12 buffers of Matmul_Group)
@(private)
_grouping :: proc() -> bool {
	return !G.linear_ungrouped && G.limits.maxStorageBuffersPerShaderStage >= 12
}

// _linear_forward: out = x @ W + bias (x [rows, in_f], W [in_f, out_f]), held
@(private)
_linear_forward :: proc(x, w, bias, out: Bind, rows, in_f, out_f: int) {
	if !_grouping() || rows == 0 || in_f == 0 || out_f == 0 {
		_matmul(x, w, out, rows, out_f, in_f, false, false, false, bias = bias)
		return
	}
	l := Held_Layer {
		w    = w,
		bias = bias,
		out  = out,
	}
	joins :=
		H.kind == .Forward &&
		H.x.buf == x.buf &&
		H.rows == rows &&
		H.in_f == in_f &&
		H.out_f == out_f &&
		!_conflict({l.w, l.bias, x}, {l.out})
	if !joins {
		_held_flush()
		H = {
			kind  = .Forward,
			x     = x,
			rows  = rows,
			in_f  = in_f,
			out_f = out_f,
		}
	}
	H.layers[H.count] = l
	H.count += 1
	if H.count == LINEAR_GROUP {_held_flush()}
}

// _linear_backward: the gradients of a Linear node (dy its gradient), held
@(private)
_linear_backward :: proc(node: ^t.Tensor, dy: Bind, rows, in_f, out_f: int) {
	a, w, bias := node.inputs[0], node.inputs[1], node.inputs[2]
	need_x, need_w, need_b := a.requires_grad, w.requires_grad, bias.requires_grad
	// what the layer reads first (an upload of them records what is held)
	l := Held_Layer {
		dy = dy,
	}
	xd: Bind
	if need_x {l.w_data = _in_data(w)}
	if need_w {xd = _in_data(a)}
	group := _grouping() && rows > 0 && in_f > 0 && out_f > 0 && a != w && a != bias && w != bias
	joins :=
		group &&
		H.kind == .Backward &&
		H.x.buf == xd.buf &&
		H.rows == rows &&
		H.in_f == in_f &&
		H.out_f == out_f &&
		H.need_x == need_x &&
		H.need_w == need_w &&
		H.need_b == need_b
	// then the gradients it writes (taking a gradient may clear it first)
	if need_b {l.gb, l.fresh_b = _out_grad(bias)}
	if need_x {l.gx, l.fresh_x = _out_grad(a)}
	if need_w {l.gw, l.fresh_w = _out_grad(w)}
	joins =
		joins &&
		H.kind == .Backward &&
		(!need_x || H.layers[0].gx.buf == l.gx.buf) &&
		!_conflict({l.dy, l.w_data, xd}, {l.gw, l.gb}, l.gx)
	if !joins {
		_held_flush()
		H = {
			kind   = .Backward,
			x      = xd,
			rows   = rows,
			in_f   = in_f,
			out_f  = out_f,
			need_x = need_x,
			need_w = need_w,
			need_b = need_b,
		}
	}
	H.layers[H.count] = l
	H.count += 1
	if H.count == LINEAR_GROUP || !group {_held_flush()}
}

// _held_writes / _held_reads: whether the held layers write / read buffer b
@(private)
_held_writes :: proc(b: Bind) -> bool {
	if b.buf == nil {return false}
	for l in H.layers[:H.count] {
		ws := [?]Bind{l.out, l.gx, l.gw, l.gb}
		for w in ws {if w.buf == b.buf {return true}}
	}
	return false
}

@(private)
_held_reads :: proc(b: Bind) -> bool {
	if b.buf == nil {return false}
	if H.x.buf == b.buf {return true}
	for l in H.layers[:H.count] {
		rs := [?]Bind{l.w, l.bias, l.dy, l.w_data}
		for r in rs {if r.buf == b.buf {return true}}
	}
	return false
}

// _conflict: whether a new layer that reads `reads` and writes `writes` (and
// adds into the held layers' dx, `gx`) can not join the held ones: a buffer
// it writes is used by them or read by itself, or one it reads is written by
// them
@(private)
_conflict :: proc(reads, writes: []Bind, gx := Bind{}) -> bool {
	for w in writes {
		if w.buf == nil {continue}
		if _held_writes(w) || _held_reads(w) {return true}
		for r in reads {if r.buf == w.buf {return true}}
	}
	if gx.buf != nil {
		if _held_reads(gx) {return true}
		for r in reads {if r.buf == gx.buf {return true}}
	}
	for r in reads {if _held_writes(r) {return true}}
	return false
}

// _held_flush records the held layers (one layer: the launches of an ungrouped
// layer)
@(private)
_held_flush :: proc() {
	if H.count == 0 {return}
	h := H
	H = {}
	ls := h.layers[:h.count]
	c := h.count
	switch h.kind {
	case .None:
	case .Forward:
		if c == 1 {
			l := ls[0]
			_matmul(h.x, l.w, l.out, h.rows, h.out_f, h.in_f, false, false, false, bias = l.bias)
			return
		}
		B, C, bias: [LINEAR_GROUP]Bind
		for l, m in ls {B[m], C[m], bias[m] = l.w, l.out, l.bias}
		_matmul_group(
			[]Bind{h.x},
			B[:c],
			C[:c],
			bias[:c],
			h.rows,
			h.out_f,
			h.in_f,
			false,
			false,
			0,
			false,
		)
	case .Backward:
		if c == 1 {
			l := ls[0]
			if h.need_b {
				_dispatch_1d(
					.Col_Sum,
					_p({u32(h.rows), u32(h.out_f), 0, 0, 0, 0, 0, u32(l.fresh_b)}),
					{l.dy, l.gb},
					h.out_f,
				)
			}
			if h.need_x {
				_matmul(l.dy, l.w_data, l.gx, h.rows, h.in_f, h.out_f, false, true, !l.fresh_x)
			}
			if h.need_w {
				_matmul(h.x, l.dy, l.gw, h.in_f, h.out_f, h.rows, true, false, !l.fresh_w)
			}
			return
		}
		dy, wd, gw, gb: [LINEAR_GROUP]Bind
		ow_b, acc_x, acc_w: u32
		for l, m in ls {
			dy[m], wd[m], gw[m], gb[m] = l.dy, l.w_data, l.gw, l.gb
			ow_b |= u32(l.fresh_b) << uint(m)
			acc_x |= u32(!l.fresh_x) << uint(m)
			acc_w |= u32(!l.fresh_w) << uint(m)
		}
		if h.need_b {
			binds: [2 * LINEAR_GROUP]Bind
			for m in 0 ..< LINEAR_GROUP {
				binds[m] = dy[m] if m < c else dy[0]
				binds[LINEAR_GROUP + m] = gb[m] if m < c else gb[0]
			}
			groups := (h.out_f + 255) / 256
			gx := min(groups, 65535)
			_dispatch(
				.Col_Sum_Group,
				_p({u32(h.rows), u32(h.out_f), 0, 0, 0, 0, 0, ow_b}),
				binds[:],
				u32(gx),
				u32((groups + gx - 1) / gx),
				u32(c),
			)
		}
		if h.need_x {
			_matmul_group(
				dy[:c],
				wd[:c],
				[]Bind{ls[0].gx},
				nil,
				h.rows,
				h.in_f,
				h.out_f,
				false,
				true,
				acc_x,
				true,
			)
		}
		if h.need_w {
			_matmul_group(
				[]Bind{h.x},
				dy[:c],
				gw[:c],
				nil,
				h.in_f,
				h.out_f,
				h.rows,
				true,
				false,
				acc_w,
				false,
			)
		}
	}
}

// _matmul_group records C_m (+)= op(A_m) @ op(B_m) (+ bias_m[column]) for the
// len(B) products m in one launch (a single A: shared; sum: all add into C[0];
// acc bit m: product m accumulates). Kernel choice and row chunks as _matmul.
@(private)
_matmul_group :: proc(
	A, B, C, bias: []Bind,
	M, N, K: int,
	trans_a, trans_b: bool,
	acc: u32,
	sum: bool,
) {
	c := len(B)
	binds: [4 * LINEAR_GROUP]Bind
	for m in 0 ..< LINEAR_GROUP {
		binds[m] = A[m] if m < len(A) else A[0]
		binds[LINEAR_GROUP + m] = B[m] if m < c else B[0]
		binds[2 * LINEAR_GROUP + m] = C[m] if m < len(C) else C[0]
		binds[3 * LINEAR_GROUP + m] = bias[m] if m < len(bias) else A[0]
	}
	flags := u32(len(bias) > 0) | u32(sum) << 1 | u32(len(A) > 1) << 2 | u32(c) << 4
	tiled :=
		G.matmul_kernel == .Tiled ||
		(G.matmul_kernel == .Auto && M >= MATMUL_TILED_MIN && N >= MATMUL_TILED_MIN)
	tile := 64 if tiled else 16
	k: Kernel = .Matmul_Tiled_Group if tiled else .Matmul_Group
	gx := u32((N + tile - 1) / tile)
	chunk := 65535 * tile
	for r0 := 0; r0 < M; r0 += chunk {
		rows := min(chunk, M - r0)
		p := _p({u32(M), u32(N), u32(K), u32(trans_a), u32(trans_b), acc, u32(r0), flags})
		_dispatch(k, p, binds[:], gx, u32((rows + tile - 1) / tile), 1 if sum else u32(c))
	}
}
