package tensor

import l "../linalg"
import "core:fmt"
import "core:math"

// ============================================================================
// Log-softmax and the CTC loss
// ============================================================================

// tensor_log_softmax: log(softmax(x)) over the logical last dimension (as
// tensor_softmax), computed stably as x - max - log Σ exp(x - max), as
// torch.log_softmax(x, dim = -1). The output has the input's storage and shape.
tensor_log_softmax :: proc(x: ^Tensor) -> ^Tensor {
	_check_shape(x)
	C := _last_dim(x)
	if _device_use(.LogSoftmax, x) {
		out := _device_new_op(x.data.rows, x.data.cols, x.shape, x.requires_grad, _op_alloc(x))
		_device_forward(.LogSoftmax, out, {x}, {C})
		if out.requires_grad {
			out.op = .LogSoftmax
			append(&out.inputs, x)
			append(&out.int_metadata, C)
		}
		return out
	}
	_host_inputs(.LogSoftmax, x)
	out_data := l.matrix_new(f64, x.data.rows, x.data.cols, _op_alloc(x))
	for r in 0 ..< len(x.data.data) / C {
		row := x.data.data[r * C:][:C]
		y := out_data.data[r * C:][:C]
		m := -math.F64_MAX
		for v in row {m = max(m, v)}
		s := 0.0
		for v in row {s += math.exp(v - m)}
		lse := m + math.ln(s)
		for v, j in row {y[j] = v - lse}
	}
	out := _tensor_new_op(out_data, x.shape, x.requires_grad, _op_alloc(x))
	if out.requires_grad {
		out.op = .LogSoftmax
		append(&out.inputs, x)
		append(&out.int_metadata, C)
	}
	return out
}

// dx += g - exp(y) · Σ_j g_j per row (y = log_softmax(x))
_log_softmax_backward :: proc(node: ^Tensor) {
	x := node.inputs[0]
	if !x.requires_grad {return}
	tensor_ensure_grad(x)
	C := node.int_metadata[0]
	for r in 0 ..< len(node.data.data) / C {
		y := node.data.data[r * C:][:C]
		g := node.grad.data[r * C:][:C]
		dx := x.grad.data[r * C:][:C]
		s := 0.0
		for v in g {s += v}
		for j in 0 ..< C {dx[j] += g[j] - math.exp(y[j]) * s}
	}
}

// Loss_Reduction: how a loss over a batch is reduced (torch's `reduction`)
Loss_Reduction :: enum {
	Mean,
	Sum,
	None,
}

// tensor_ctc_loss: the connectionist temporal classification loss, as
// torch.nn.functional.ctc_loss, over log-probabilities log_probs [batch, T, C]
// (channels-last; usually tensor_log_softmax of the logits, torch's [T, batch, C]
// transposed). Sample b uses the first input_lengths[b] steps (1 ..= T) and the
// next target_lengths[b] labels of `targets`, the targets of all samples one
// after the other (torch's concatenated form). Labels are in [0, C) and not
// `blank`. The loss of a sample is -log Σ p(path) over all alignments of its
// labels with blanks; an impossible alignment (too few steps for the labels and
// the blanks needed between repeated labels) gives +inf, or 0 with
// zero_infinity, which also zeroes its gradient (without it, the gradient of
// that sample is NaN, as in torch).
// Reduction .Mean divides each sample's loss by its target length (at least 1)
// and averages over the batch; .Sum adds them; .None returns the losses [batch].
// The gradient is the derivative with respect to log_probs, -(posterior of class
// c at step t). torch's ctc_loss returns exp(log_probs) minus that posterior
// instead (on the steps a sample uses, scaled by the reduction like the rest);
// after a log_softmax both give the same gradient of the logits.
tensor_ctc_loss :: proc(
	log_probs: ^Tensor,
	targets: []int,
	input_lengths: []int,
	target_lengths: []int,
	blank := 0,
	reduction := Loss_Reduction.Mean,
	zero_infinity := false,
	loc := #caller_location,
) -> ^Tensor {
	_check_shape(log_probs, loc)
	B, T, C := log_probs.shape[0], log_probs.shape[1], log_probs.shape[2] * log_probs.shape[3]
	fmt.assertf(
		B >= 1 && T >= 1 && tensor_numel(log_probs) == B * T * C,
		"tensor_ctc_loss: log_probs must be [batch, T, C], got shape %v",
		log_probs.shape,
		loc = loc,
	)
	fmt.assertf(
		len(input_lengths) == B && len(target_lengths) == B,
		"tensor_ctc_loss: %d input and %d target lengths for %d samples",
		len(input_lengths),
		len(target_lengths),
		B,
		loc = loc,
	)
	fmt.assertf(
		blank >= 0 && blank < C,
		"tensor_ctc_loss: blank %d outside [0, %d)",
		blank,
		C,
		loc = loc,
	)
	n_targets := 0
	for b in 0 ..< B {
		fmt.assertf(
			input_lengths[b] >= 1 && input_lengths[b] <= T && target_lengths[b] >= 0,
			"tensor_ctc_loss: sample %d has input length %d (1 ..= %d) and target length %d",
			b,
			input_lengths[b],
			T,
			target_lengths[b],
			loc = loc,
		)
		n_targets += target_lengths[b]
	}
	fmt.assertf(
		len(targets) == n_targets,
		"tensor_ctc_loss: %d targets, the target lengths add up to %d",
		len(targets),
		n_targets,
		loc = loc,
	)
	for v in targets {
		fmt.assertf(
			v >= 0 && v < C && v != blank,
			"tensor_ctc_loss: label %d outside [0, %d) or the blank",
			v,
			C,
			loc = loc,
		)
	}
	// int_metadata: B, T, C, blank, reduction, zero_infinity, then the input
	// lengths, the target lengths and the targets
	meta := make([]int, 6 + 2 * B + len(targets), context.temp_allocator)
	copy(meta, []int{B, T, C, blank, int(reduction), int(zero_infinity)})
	copy(meta[6:], input_lengths)
	copy(meta[6 + B:], target_lengths)
	copy(meta[6 + 2 * B:], targets)
	out_shape := [4]int{B, 1, 1, 1} if reduction == .None else [4]int{1, 1, 1, 1}
	out_rows := B if reduction == .None else 1
	if _device_use(.CTCLoss, log_probs) {
		out := _device_new_op(
			out_rows,
			1,
			out_shape,
			log_probs.requires_grad,
			_op_alloc(log_probs),
		)
		if out.requires_grad {
			// the forward variables alpha [batch, T, 2·max S + 1] and log p of
			// every sample, kept on the device for the backward pass
			S_max := 0
			for v in target_lengths {S_max = max(S_max, v)}
			out.device.aux_buf = _device_alloc(out, B * T * (2 * S_max + 1) + B)
		}
		_device_forward(.CTCLoss, out, {log_probs}, meta)
		if out.requires_grad {
			out.op = .CTCLoss
			append(&out.inputs, log_probs)
			append(&out.int_metadata, ..meta)
		}
		return out
	}
	_host_inputs(.CTCLoss, log_probs)

	// the loss of every sample (the backward pass recomputes the forward variables)
	losses := make([]f64, B, context.temp_allocator)
	offset := 0
	for b in 0 ..< B {
		S := target_lengths[b]
		lp := log_probs.data.data[b * T * C:][:T * C]
		losses[b] = _ctc_sample(lp, targets[offset:][:S], input_lengths[b], C, blank, nil)
		offset += S
		if zero_infinity && math.is_inf(losses[b]) {losses[b] = 0}
	}
	out_data := l.matrix_new(f64, out_rows, 1, _op_alloc(log_probs))
	switch reduction {
	case .None:
		copy(out_data.data, losses)
	case .Sum:
		for v in losses {out_data.data[0] += v}
	case .Mean:
		for v, b in losses {out_data.data[0] += v / f64(max(target_lengths[b], 1))}
		out_data.data[0] /= f64(B)
	}
	out := _tensor_new_op(out_data, out_shape, log_probs.requires_grad, _op_alloc(log_probs))
	if out.requires_grad {
		out.op = .CTCLoss
		append(&out.inputs, log_probs)
		append(&out.int_metadata, ..meta)
	}
	return out
}

// log(exp(a) + exp(b)), with -inf for "no path"
@(private = "file")
_lse :: proc(a, b: f64) -> f64 {
	if a == math.inf_f64(-1) {return b}
	if b == math.inf_f64(-1) {return a}
	m := max(a, b)
	return m + math.ln(math.exp(a - m) + math.exp(b - m))
}

// _ctc_sample: the loss of one sample over the first Tn steps of lp [T, C], and
// (if grad != nil) its derivative with respect to lp in grad [T, C]. The
// extended labels are blank, l_1, blank, l_2, ..., blank (2S + 1 of them).
@(private = "file")
_ctc_sample :: proc(lp: []f64, labels: []int, Tn, C, blank: int, grad: []f64) -> f64 {
	NEG := math.inf_f64(-1)
	S := len(labels)
	L := 2 * S + 1
	ext :: proc(labels: []int, blank, s: int) -> int {return(
			labels[s / 2] if s % 2 == 1 else blank \
		)}
	// may step s - 2 -> s: s is a label different from the label two before
	skip :: proc(labels: []int, s: int) -> bool {
		return s % 2 == 1 && s >= 3 && labels[s / 2] != labels[s / 2 - 1]
	}
	alpha := make([]f64, Tn * L, context.temp_allocator)
	for &v in alpha {v = NEG}
	alpha[0] = lp[blank]
	if L > 1 {alpha[1] = lp[ext(labels, blank, 1)]}
	for t in 1 ..< Tn {
		prev, cur := alpha[(t - 1) * L:][:L], alpha[t * L:][:L]
		for s in 0 ..< L {
			a := prev[s]
			if s >= 1 {a = _lse(a, prev[s - 1])}
			if skip(labels, s) {a = _lse(a, prev[s - 2])}
			if a != NEG {cur[s] = a + lp[t * C + ext(labels, blank, s)]}
		}
	}
	last := alpha[(Tn - 1) * L:][:L]
	log_p := last[L - 1]
	if L > 1 {log_p = _lse(log_p, last[L - 2])}
	loss := -log_p
	if grad == nil {return loss}

	beta := make([]f64, Tn * L, context.temp_allocator)
	for &v in beta {v = NEG}
	end := beta[(Tn - 1) * L:][:L]
	end[L - 1] = lp[(Tn - 1) * C + ext(labels, blank, L - 1)]
	if L > 1 {end[L - 2] = lp[(Tn - 1) * C + ext(labels, blank, L - 2)]}
	for t := Tn - 2; t >= 0; t -= 1 {
		next, cur := beta[(t + 1) * L:][:L], beta[t * L:][:L]
		for s in 0 ..< L {
			v := next[s]
			if s + 1 < L {v = _lse(v, next[s + 1])}
			if s + 2 < L && skip(labels, s + 2) {v = _lse(v, next[s + 2])}
			if v != NEG {cur[s] = v + lp[t * C + ext(labels, blank, s)]}
		}
	}
	// d loss / d lp[t, c] = -Σ_{s: ext(s) = c} exp(alpha + beta - lp[t, c] - log p)
	// (alpha and beta both contain the emission of step t). Steps past Tn get 0.
	lcab := make([]f64, C, context.temp_allocator)
	for t in 0 ..< Tn {
		for &v in lcab {v = NEG}
		for s in 0 ..< L {
			c := ext(labels, blank, s)
			lcab[c] = _lse(lcab[c], alpha[t * L + s] + beta[t * L + s])
		}
		for c in 0 ..< C {
			if lcab[c] == NEG && log_p != NEG {continue}
			grad[t * C + c] = -math.exp(lcab[c] - lp[t * C + c] - log_p)
		}
	}
	return loss
}

// _ctc_backward: the derivative of every sample's loss (forward and backward
// variables recomputed from log_probs), scaled by the reduction and the
// upstream gradient
_ctc_backward :: proc(node: ^Tensor) {
	x := node.inputs[0]
	if !x.requires_grad {return}
	tensor_ensure_grad(x)
	m := node.int_metadata[:]
	B, T, C, blank, zero_infinity := m[0], m[1], m[2], m[3], m[5] != 0
	reduction := Loss_Reduction(m[4])
	input_lengths := m[6:][:B]
	target_lengths := m[6 + B:][:B]
	targets := m[6 + 2 * B:]
	g := make([]f64, T * C, context.temp_allocator)
	offset := 0
	for b in 0 ..< B {
		S := target_lengths[b]
		lp := x.data.data[b * T * C:][:T * C]
		for &v in g {v = 0}
		loss := _ctc_sample(lp, targets[offset:][:S], input_lengths[b], C, blank, g)
		offset += S
		if zero_infinity && math.is_inf(loss) {continue}
		f: f64
		switch reduction {
		case .None:
			f = node.grad.data[b]
		case .Sum:
			f = node.grad.data[0]
		case .Mean:
			f = node.grad.data[0] / (f64(B) * f64(max(target_lengths[b], 1)))
		}
		dx := x.grad.data[b * T * C:][:T * C]
		for v, i in g {dx[i] += f * v}
	}
}
