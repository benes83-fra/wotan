package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "../wotan/util"
import "core:fmt"
import "core:math"
import "core:mem"

// Recurrent cells and the selective SSM scan:
// - the forward pass equals a plain step-by-step reference exactly (the
//   input projection x @ W_ih is computed for all steps at once, the SSM
//   splits batch rows across threads; neither may change a single bit)
// - outputs and all input gradients are identical with 1 thread and with
//   all threads (shapes are large enough to be split)

@(private = "file")
_Cell :: enum {RNN, GRU, LSTM}

// Plain reference: out[b, s, :] for one cell, step by step, with naive matmuls
// that add their terms in the same order as the matmul kernels.
@(private = "file")
_ref_cell :: proc(cell: _Cell, x, h_0, c_0, w_ih, w_hh, bias: []f64, B, S, I, H: int, out: []f64) {
	k := 1 if cell == .RNN else (3 if cell == .GRU else 4)
	G := k * H
	h := make([]f64, B * H)
	c := make([]f64, B * H)
	gi := make([]f64, G)
	gh := make([]f64, G)
	defer {delete(h); delete(c); delete(gi); delete(gh)}
	copy(h, h_0)
	if cell == .LSTM {copy(c, c_0)}
	sig :: proc(v: f64) -> f64 {return 1.0 / (1.0 + math.exp(-v))}

	for s in 0 ..< S {
		h_new := make([]f64, B * H)
		defer delete(h_new)
		for b in 0 ..< B {
			for j in 0 ..< G {
				acc := 0.0
				for q in 0 ..< I {acc += x[(b * S + s) * I + q] * w_ih[q * G + j]}
				gi[j] = acc
				acc = 0.0
				for q in 0 ..< H {acc += h[b * H + q] * w_hh[q * G + j]}
				gh[j] = acc
			}
			for i in 0 ..< H {
				hp := h[b * H + i]
				switch cell {
				case .RNN:
					h_new[b * H + i] = math.tanh(gi[i] + gh[i] + bias[i])
				case .GRU:
					r := sig((gi[i] + bias[i]) + gh[i])
					z := sig((gi[H + i] + bias[H + i]) + gh[H + i])
					n := math.tanh((gi[2 * H + i] + bias[2 * H + i]) + r * gh[2 * H + i])
					h_new[b * H + i] = (1.0 - z) * n + z * hp
				case .LSTM:
					ig := sig((gi[i] + bias[i]) + gh[i])
					fg := sig((gi[H + i] + bias[H + i]) + gh[H + i])
					gg := math.tanh((gi[2 * H + i] + bias[2 * H + i]) + gh[2 * H + i])
					og := sig((gi[3 * H + i] + bias[3 * H + i]) + gh[3 * H + i])
					c[b * H + i] = fg * c[b * H + i] + ig * gg
					h_new[b * H + i] = og * math.tanh(c[b * H + i])
				}
				out[(b * S + s) * H + i] = h_new[b * H + i]
			}
		}
		copy(h, h_new)
	}
}

// Plain reference of the selective scan
@(private = "file")
_ref_ssm :: proc(x, h_0, A, Bm, C, Delta, D: []f64, B, S, Dm, N: int, out: []f64) {
	h := make([]f64, B * Dm * N)
	defer delete(h)
	copy(h, h_0)
	for s in 0 ..< S {
		for b in 0 ..< B {
			for d in 0 ..< Dm {
				ix := (b * S + s) * Dm + d
				y := x[ix] * D[d]
				for n in 0 ..< N {
					hi := (b * Dm + d) * N + n
					ib := (b * S + s) * N + n
					hv := math.exp(Delta[ix] * A[d * N + n]) * h[hi] + Delta[ix] * Bm[ib] * x[ix]
					h[hi] = hv
					y += C[ib] * hv
				}
				out[ix] = y
			}
		}
	}
}

@(private = "file")
_same :: proc(a, b: []f64) -> bool {
	if len(a) != len(b) {return false}
	for i in 0 ..< len(a) {
		if transmute(u64)a[i] != transmute(u64)b[i] {return false}
	}
	return true
}

@(private = "file")
_Run :: struct {
	out:   []f64,
	grads: [7][]f64,
}

@(private = "file")
_run_cell :: proc(cell: _Cell, ins: []^t.Tensor, allocator: mem.Allocator) -> _Run {
	for x in ins {t.tensor_zero_grad(x)}
	out: ^t.Tensor
	switch cell {
	case .RNN: out = t.tensor_rnn(ins[0], ins[1], ins[2], ins[3], ins[4], allocator)
	case .GRU: out = t.tensor_gru(ins[0], ins[1], ins[2], ins[3], ins[4], allocator)
	case .LSTM: out = t.tensor_lstm(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], allocator)
	}
	return _finish(out, ins, allocator)
}

@(private = "file")
_run_ssm :: proc(ins: []^t.Tensor, allocator: mem.Allocator) -> _Run {
	for x in ins {t.tensor_zero_grad(x)}
	out := t.tensor_ssm(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], allocator)
	return _finish(out, ins, allocator)
}

// Weighted-sum loss, backward, then keep copies of the output and gradients
@(private = "file")
_finish :: proc(out: ^t.Tensor, ins: []^t.Tensor, allocator: mem.Allocator) -> _Run {
	loss := gcl_weighted_sum(out)
	t.tensor_backward(loss)
	r: _Run
	r.out = make([]f64, len(out.data.data), allocator)
	copy(r.out, out.data.data)
	for x, i in ins {
		r.grads[i] = make([]f64, len(x.grad.data), allocator)
		copy(r.grads[i], x.grad.data)
	}
	t.tensor_free_graph(loss)
	return r
}

@(private = "file")
_same_run :: proc(a, b: _Run, n: int) -> bool {
	if !_same(a.out, b.out) {return false}
	for i in 0 ..< n {
		if !_same(a.grads[i], b.grads[i]) {return false}
	}
	return true
}

recurrent_scan_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Recurrent Cell and SSM Scan Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	saved_threads := util.parallel_max_threads
	saved_mm := l.matmul_max_threads
	defer {
		util.parallel_max_threads = saved_threads
		l.matmul_max_threads = saved_mm
	}
	all_threads :: proc() {util.parallel_max_threads = 0; l.matmul_max_threads = 0}
	one_thread :: proc() {util.parallel_max_threads = 1; l.matmul_max_threads = 1}

	// --- Recurrent cells ------------------------------------------------------
	// {B, S, I, H}: a small odd shape, and one whose input projection is threaded
	cell_shapes := [][4]int{{3, 5, 4, 6}, {16, 40, 64, 64}}
	names := [_Cell]string{.RNN = "rnn", .GRU = "gru", .LSTM = "lstm"}
	for cell in _Cell {
		k := 1 if cell == .RNN else (3 if cell == .GRU else 4)
		for sh, si in cell_shapes {
			B, S, I, H := sh[0], sh[1], sh[2], sh[3]
			seed := u64(400 + int(cell) * 10 + si)
			ins := [6]^t.Tensor {
				gcl_rand(1, B * S * I, {B, S, I, 1}, seed, true, allocator),
				gcl_rand(B, H, {B, H, 1, 1}, seed + 1, true, allocator, 0.5),
				gcl_rand(B, H, {B, H, 1, 1}, seed + 2, true, allocator, 0.5),
				gcl_rand(I, k * H, {I, k * H, 1, 1}, seed + 3, true, allocator, 0.3),
				gcl_rand(H, k * H, {H, k * H, 1, 1}, seed + 4, true, allocator, 0.3),
				gcl_rand(1, k * H, {1, k * H, 1, 1}, seed + 5, true, allocator, 0.3),
			}
			// RNN / GRU take (x, h_0, w_ih, w_hh, bias); LSTM also c_0
			args := ins[:] if cell == .LSTM else []^t.Tensor{ins[0], ins[1], ins[3], ins[4], ins[5]}
			n_in := len(args)

			all_threads()
			r_all := _run_cell(cell, args, allocator)
			one_thread()
			r_one := _run_cell(cell, args, allocator)
			all_threads()

			ref := make([]f64, B * S * H, allocator)
			_ref_cell(
				cell,
				ins[0].data.data,
				ins[1].data.data,
				ins[2].data.data,
				ins[3].data.data,
				ins[4].data.data,
				ins[5].data.data,
				B,
				S,
				I,
				H,
				ref,
			)
			expect(&passed, &total, _same(r_all.out, ref), fmt.tprintf("%s B%d S%d I%d H%d: forward == step-by-step reference", names[cell], B, S, I, H))
			expect(&passed, &total, _same_run(r_all, r_one, n_in), fmt.tprintf("%s B%d S%d I%d H%d: output and %d gradients identical on 1 and all threads", names[cell], B, S, I, H, n_in))
			for x in ins {t.tensor_free(x)}
		}
	}

	// --- Selective SSM ----------------------------------------------------------
	// {B, S, d_model, d_state}: small, and one split across many threads (odd batch)
	ssm_shapes := [][4]int{{2, 3, 2, 3}, {37, 9, 24, 16}}
	for sh, si in ssm_shapes {
		B, S, Dm, N := sh[0], sh[1], sh[2], sh[3]
		seed := u64(500 + si * 10)
		ins := [7]^t.Tensor {
			gcl_rand(1, B * S * Dm, {B, S, Dm, 1}, seed, true, allocator), // x
			gcl_rand(1, B * Dm * N, {B, Dm, N, 1}, seed + 1, true, allocator, 0.5), // h_0
			gcl_rand(Dm, N, {Dm, N, 1, 1}, seed + 2, true, allocator), // A
			gcl_rand(1, B * S * N, {B, S, N, 1}, seed + 3, true, allocator), // B
			gcl_rand(1, B * S * N, {B, S, N, 1}, seed + 4, true, allocator), // C
			gcl_rand(1, B * S * Dm, {B, S, Dm, 1}, seed + 5, true, allocator), // Delta
			gcl_rand(1, Dm, {1, Dm, 1, 1}, seed + 6, true, allocator), // D
		}
		gco_remap(ins[2], -1.5, -0.2)
		gco_remap(ins[5], 0.1, 1.0)

		all_threads()
		r_all := _run_ssm(ins[:], allocator)
		one_thread()
		r_one := _run_ssm(ins[:], allocator)
		all_threads()

		ref := make([]f64, B * S * Dm, allocator)
		_ref_ssm(
			ins[0].data.data,
			ins[1].data.data,
			ins[2].data.data,
			ins[3].data.data,
			ins[4].data.data,
			ins[5].data.data,
			ins[6].data.data,
			B,
			S,
			Dm,
			N,
			ref,
		)
		expect(&passed, &total, _same(r_all.out, ref), fmt.tprintf("ssm B%d S%d D%d N%d: forward == sequential reference", B, S, Dm, N))
		expect(&passed, &total, _same_run(r_all, r_one, 7), fmt.tprintf("ssm B%d S%d D%d N%d: output and 7 gradients identical on 1 and all threads", B, S, Dm, N))
		for x in ins {t.tensor_free(x)}
	}

	fmt.printf("Recurrent scan contracts: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
