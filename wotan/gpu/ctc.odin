package gpu

import t "../tensor"

// ============================================================================
// Device CTC loss and log-softmax
// ============================================================================
//
// The CTC forward variables alpha run one workgroup per sample, stepping through
// time inside the kernel (a storage barrier per step); with a gradient they stay
// in the output's aux buffer (followed by log p of every sample) for the
// backward pass, which computes beta the same way and then one thread per
// log-probability. Metadata as the CPU op: [N, T, C, blank, reduction,
// zero_infinity], the input lengths, the target lengths, the targets.

@(private)
Ctc_Dims :: struct {
	N, T, C, blank, reduction, zero_inf, Lm: int,
}

@(private)
_ctc_dims :: proc(m: []int) -> (d: Ctc_Dims) {
	d.N, d.T, d.C, d.blank, d.reduction, d.zero_inf = m[0], m[1], m[2], m[3], m[4], m[5]
	S_max := 0
	for v in m[6 + d.N:][:d.N] {S_max = max(S_max, v)}
	d.Lm = 2 * S_max + 1
	return
}

// u0 N, u1 T, u2 C, u3 blank, u4 Lm, u5 reduction, u6 zero_infinity
@(private)
_ctc_params :: proc(d: Ctc_Dims) -> Params {
	return _p(
		{
			u32(d.N),
			u32(d.T),
			u32(d.C),
			u32(d.blank),
			u32(d.Lm),
			u32(d.reduction),
			u32(d.zero_inf),
			0,
		},
	)
}

// per sample [input length, target length, target offset], then the targets
@(private)
_ctc_info :: proc(m: []int, d: Ctc_Dims) -> Bind {
	targets := m[6 + 2 * d.N:]
	info := make([]int, 3 * d.N + len(targets), context.temp_allocator)
	offset := 0
	for b in 0 ..< d.N {
		info[3 * b], info[3 * b + 1], info[3 * b + 2] = m[6 + b], m[6 + d.N + b], offset
		offset += m[6 + d.N + b]
	}
	copy(info[3 * d.N:], targets)
	return _upload_ints(info)
}

@(private)
_ctc_forward :: proc(out: ^t.Tensor, lp: ^t.Tensor, ints: []int) {
	d := _ctc_dims(ints)
	n_alpha := d.N * d.T * d.Lm + d.N
	assert(n_alpha < 1 << 31, "gpu ctc_loss: the forward variables do not fit")
	kept := out.device.aux_buf.ptr != nil
	alpha := _bind(out.device.aux_buf, n_alpha) if kept else Bind{_acquire(n_alpha), n_alpha}
	info := _ctc_info(ints, d)
	pr := _ctc_params(d)
	_dispatch_rows(.Ctc_Alpha, pr, {_data(lp), info, alpha}, d.N)
	o := _data(out)
	if d.reduction == int(t.Loss_Reduction.None) {
		_dispatch_1d(.Ctc_Loss, pr, {alpha, info, o}, d.N)
	} else {
		losses := Bind{_acquire(d.N), d.N}
		_dispatch_1d(.Ctc_Loss, pr, {alpha, info, losses}, d.N)
		_reduce(losses, d.N, o, 0, 1, false)
		_release_buf(losses.buf)
	}
	_release_buf(info.buf)
	if !kept {_release_buf(alpha.buf)}
}

// false: the forward ran on the CPU (no alpha on the device), so the CPU case
// runs on host copies
@(private)
_ctc_backward :: proc(node: ^t.Tensor, dy: Bind) -> bool {
	lp := node.inputs[0]
	if !lp.requires_grad {return true}
	if node.device == nil || node.device.aux_buf.ptr == nil {return false}
	m := node.int_metadata[:]
	d := _ctc_dims(m)
	alpha := _bind(node.device.aux_buf, d.N * d.T * d.Lm + d.N)
	n_beta := d.N * d.T * d.Lm
	beta := Bind{_acquire(n_beta), n_beta}
	info := _ctc_info(m, d)
	pr := _ctc_params(d)
	lpd := _in_data(lp)
	_dispatch_rows(.Ctc_Beta, pr, {lpd, info, beta}, d.N)
	g, fresh := _out_grad(lp)
	pr.u[7] = u32(fresh)
	_dispatch_1d(.Ctc_Grad, pr, {lpd, info, alpha, beta, dy, g}, t.tensor_numel(lp))
	_release_buf(beta.buf)
	_release_buf(info.buf)
	t._grad_written_on_device(lp)
	return true
}

// log-softmax rows: Softmax_Rows in mode 3
@(private)
_log_softmax_forward :: proc(out, x: ^t.Tensor, C: int) {
	n := t.tensor_numel(out)
	u := [8]u32{u32(n / C), u32(C), 3, 0, 0, 1, 0, 0}
	_dispatch_rows(
		.Softmax_Rows,
		_p(u, {1, 0, 0, 0, 0, 0, 0, 0}),
		{_data(x), _dummy(), _data(out)},
		n / C,
	)
}

@(private)
_log_softmax_backward :: proc(node: ^t.Tensor, dy: Bind) {
	x := node.inputs[0]
	if !x.requires_grad {return}
	C := node.int_metadata[0]
	n := t.tensor_numel(node)
	g, fresh := _out_grad(x)
	u := [8]u32{u32(n / C), u32(C), 0, 0, 0, 0, 0, u32(fresh)}
	_dispatch_rows(.LogSoftmax_Bwd, _p(u), {_in_data(node), dy, g}, n / C)
	t._grad_written_on_device(x)
}
