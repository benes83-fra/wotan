package main

import gpu "../wotan/gpu"
import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"

// The embedding backward on the GPU sums each vocabulary row's gradient over
// its positions in a fixed order: with ids that repeat and a different
// gradient at every position, two runs give the same weight gradient bit for
// bit, ids that live only on the device give it too, a second backward adds
// exactly the same values again, it matches the sum computed in f64, and
// ids that do not occur (or lie outside the vocabulary) give no gradient
gpu_embedding_order_test :: proc() {
	fmt.println("gpu_embedding_order_test: the embedding weight gradient is the same on every run")
	B :: 6
	S :: 100
	E :: 64
	Case :: struct {
		vocab, step, kinds: int,
	}
	// ids ((i mod kinds) · step) mod vocab, each about B·S / distinct times:
	// - 40 ids, 37 of them occur (the other rows stay zero)
	// - 37 ids, all of them occur
	// - 3000 ids, 97 of them spread over the vocabulary (a two-pass sort)
	// Two positions hold ids outside the vocabulary (-1 and vocab).
	for c in ([]Case{{40, 1, 37}, {37, 1, 37}, {3000, 31, 97}}) {
		V := c.vocab
		ids := t.tensor_new(l.matrix_new(f64, B, S), false)
		ids.shape = {B, S, 1, 1}
		for &v, i in ids.data.data {v = f64(((i * 7) % c.kinds * c.step) % V)}
		ids.data.data[5] = -1
		ids.data.data[B * S - 3] = f64(V)
		w := t.tensor_new(l.matrix_new(f64, V, E), true)
		w.shape = {V, E, 1, 1}
		for &v, i in w.data.data {v = f64(f32(0.3 * math.sin(0.37 * f64(i))))}
		// a different upstream gradient at every position (and element)
		g := t.tensor_new(l.matrix_new(f64, 1, B * S * E), false)
		g.shape = {B, S, E, 1}
		for &v, i in g.data.data {v = f64(f32(math.cos(0.01 * f64(i))))}
		zero := t.tensor_new(l.matrix_new(f64, B, S), false)
		zero.shape = {B, S, 1, 1}
		// the expected gradient: Σ g[r, e] over the positions r of id v, in f64
		want := make([]f64, V * E)
		occurs := make([]bool, V)
		defer {delete(want); delete(occurs)}
		for r in 0 ..< B * S {
			id := int(ids.data.data[r])
			if id < 0 || id >= V {continue}
			occurs[id] = true
			for e in 0 ..< E {want[id * E + e] += g.data.data[r * E + e]}
		}
		gpu.to_device([]^t.Tensor{ids, w, g, zero})
		defer {
			t.tensor_free(ids)
			t.tensor_free(w)
			t.tensor_free(g)
			t.tensor_free(zero)
		}
		// backward of sum(embedding(ids) · g) `times` times, the weight gradient
		run :: proc(ids, w, g, zero: ^t.Tensor, device_ids: bool, times := 1) -> []f64 {
			scope: t.Graph_Scope
			t.graph_begin(&scope)
			t.tensor_zero_grad(w)
			for _ in 0 ..< times {
				x := ids
				if device_ids {x = t.tensor_add(ids, zero)} 	// no host copy
				loss := t.tensor_sum(t.tensor_mul(t.tensor_embedding(x, w), g))
				t.tensor_backward(loss)
			}
			grad := t.tensor_grad_values(w)
			out := make([]f64, len(grad))
			copy(out, grad)
			t.graph_end(&scope)
			t.graph_scope_destroy(&scope)
			return out
		}
		first := run(ids, w, g, zero, false)
		second := run(ids, w, g, zero, false)
		device := run(ids, w, g, zero, true)
		twice := run(ids, w, g, zero, false, 2)
		defer {delete(first); delete(second); delete(device); delete(twice)}
		equal :: proc(a, b: []f64, factor := 1.0) -> (differ: int) {
			for v, i in a {if factor * v != b[i] {differ += 1}}
			return
		}
		worst, unused := 0.0, 0.0
		for v, i in first {
			worst = max(worst, abs(v - want[i]))
			if !occurs[i / E] {unused = max(unused, abs(v))}
		}
		d_runs, d_device, d_twice :=
			equal(first, second), equal(first, device), equal(first, twice, 2)
		check(d_runs == 0, "%d of %d gradient values differ between two runs", d_runs, len(first))
		check(d_device == 0, "%d values differ with ids only on the device", d_device)
		check(d_twice == 0, "%d values of two backwards differ from twice one", d_twice)
		check(worst < 1e-4, "largest difference from the f64 sums %.2e", worst)
		check(unused == 0, "ids that do not occur got gradient %.2e", unused)
		fmt.printfln(
			"    %d positions, %d ids: values differing between runs %d, with device ids %d, accumulated %d; largest error vs f64 %.1e",
			B * S,
			V,
			d_runs,
			d_device,
			d_twice,
			worst,
		)
	}
}
