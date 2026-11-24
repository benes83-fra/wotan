package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"

// The LSTM, GRU and RNN kernels run in launches of a block of steps each
// (gpu.set_recurrent_steps; large layers choose blocks themselves, so that no
// submission runs into Windows' GPU timeout): with blocks of 4 steps and of 1
// step the output and every gradient equal those of one launch bit for bit,
// and every block is submitted on its own
gpu_recurrent_steps_test :: proc() {
	fmt.println(
		"gpu_recurrent_steps_test: recurrent layers run in blocks of steps give the values of one launch",
	)
	previous := gpu.set_recurrent_steps(0)
	defer gpu.set_recurrent_steps(previous)
	B :: 3
	S :: 13
	I :: 4
	H :: 10
	Kind :: enum {
		LSTM,
		GRU,
		RNN,
	}
	// a tensor of `shape` stored [rows, cols], values 0.4 · sin(seed + 0.7 · i)
	make_in :: proc(rows, cols: int, shape: [4]int, seed: f64, scale := 1.0) -> ^t.Tensor {
		m := l.matrix_new(f64, rows, cols)
		for &v, i in m.data {v = f64(f32(scale * 0.4 * math.sin(seed + 0.7 * f64(i))))}
		x := t.tensor_new(m, true)
		x.shape = shape
		return x
	}
	// the output and the gradients of every input of sum(y²), one slice
	run :: proc(kind: Kind) -> (values: [dynamic]f64) {
		gates := [Kind]int {
			.LSTM = 4 * H,
			.GRU  = 3 * H,
			.RNN  = H,
		}
		G := gates[kind]
		ins := make([dynamic]^t.Tensor, context.temp_allocator)
		append(&ins, make_in(1, B * S * I, {B, S, I, 1}, 1))
		append(&ins, make_in(B, H, {B, H, 1, 1}, 2))
		if kind == .LSTM {append(&ins, make_in(B, H, {B, H, 1, 1}, 3))}
		append(&ins, make_in(I, G, {I, G, 1, 1}, 4))
		append(&ins, make_in(H, G, {H, G, 1, 1}, 5, 0.25))
		append(&ins, make_in(1, G, {1, G, 1, 1}, 6))
		if kind == .GRU {append(&ins, make_in(1, H, {1, H, 1, 1}, 7))}
		gpu.to_device(ins[:])
		y: ^t.Tensor
		switch kind {
		case .LSTM:
			y = t.tensor_lstm(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5])
		case .GRU:
			y = t.tensor_gru(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5])
		case .RNN:
			y = t.tensor_rnn(ins[0], ins[1], ins[2], ins[3], ins[4])
		}
		loss := t.tensor_sum(t.tensor_mul(y, y))
		t.tensor_backward(loss)
		append(&values, ..t.tensor_values(y))
		for x in ins {append(&values, ..t.tensor_grad_values(x))}
		t.tensor_free_graph(loss)
		for x in ins {t.tensor_free(x)}
		return
	}
	for kind in Kind {
		gpu.set_recurrent_steps(0)
		one := run(kind)
		defer delete(one)
		for steps in ([]int{4, 1}) {
			gpu.set_recurrent_steps(steps)
			flushes := gpu.stats().flushes
			got := run(kind)
			defer delete(got)
			submitted := gpu.stats().flushes - flushes
			same := len(got) == len(one)
			for v, i in got {same = same && v == one[i]}
			blocks := (S + steps - 1) / steps
			check(same, "%v in blocks of %d steps differs from one launch", kind, steps)
			check(
				submitted >= 2 * blocks,
				"%v in blocks of %d steps: %d submissions, expected at least %d",
				kind,
				steps,
				submitted,
				2 * blocks,
			)
			fmt.printfln(
				"    %v, %d steps in blocks of %d: %d values equal to one launch %v, %d submissions",
				kind,
				S,
				steps,
				len(got),
				same,
				submitted,
			)
		}
	}
}
