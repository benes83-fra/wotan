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
// exactly the same values again, and it matches the sum computed in f64
gpu_embedding_order_test :: proc() {
	fmt.println("gpu_embedding_order_test: the embedding weight gradient is the same on every run")
	B :: 6
	S :: 50
	V :: 40
	E :: 64
	// ids (7·i) mod 37: every id below 37 about 8 times, 37..39 never
	ids := t.tensor_new(l.matrix_new(f64, B, S), false)
	ids.shape = {B, S, 1, 1}
	for &v, i in ids.data.data {v = f64((i * 7) % 37)}
	w := t.tensor_new(l.matrix_new(f64, V, E), true)
	w.shape = {V, E, 1, 1}
	for &v, i in w.data.data {v = f64(f32(0.3 * math.sin(0.37 * f64(i))))}
	// a different upstream gradient at every position (and element)
	c := t.tensor_new(l.matrix_new(f64, 1, B * S * E), false)
	c.shape = {B, S, E, 1}
	for &v, i in c.data.data {v = f64(f32(math.cos(0.01 * f64(i))))}
	zero := t.tensor_new(l.matrix_new(f64, B, S), false)
	zero.shape = {B, S, 1, 1}
	// the expected gradient: Σ c[r, e] over the positions r of id v, in f64
	want := make([]f64, V * E)
	defer delete(want)
	for r in 0 ..< B * S {
		id := int(ids.data.data[r])
		for e in 0 ..< E {want[id * E + e] += c.data.data[r * E + e]}
	}
	gpu.to_device([]^t.Tensor{ids, w, c, zero})
	defer {
		t.tensor_free(ids)
		t.tensor_free(w)
		t.tensor_free(c)
		t.tensor_free(zero)
	}
	// backward of sum(embedding(ids) · c) `times` times, the weight gradient
	run :: proc(ids, w, c, zero: ^t.Tensor, device_ids: bool, times := 1) -> []f64 {
		scope: t.Graph_Scope
		t.graph_begin(&scope)
		t.tensor_zero_grad(w)
		for _ in 0 ..< times {
			x := ids
			if device_ids {x = t.tensor_add(ids, zero)} 	// no host copy
			loss := t.tensor_sum(t.tensor_mul(t.tensor_embedding(x, w), c))
			t.tensor_backward(loss)
		}
		g := t.tensor_grad_values(w)
		out := make([]f64, len(g))
		copy(out, g)
		t.graph_end(&scope)
		t.graph_scope_destroy(&scope)
		return out
	}
	first := run(ids, w, c, zero, false)
	second := run(ids, w, c, zero, false)
	device := run(ids, w, c, zero, true)
	twice := run(ids, w, c, zero, false, 2)
	defer {delete(first); delete(second); delete(device); delete(twice)}
	equal :: proc(a, b: []f64, factor := 1.0) -> (differ: int) {
		for v, i in a {if factor * v != b[i] {differ += 1}}
		return
	}
	worst := 0.0
	for v, i in first {worst = max(worst, abs(v - want[i]))}
	unused := 0.0
	for v in first[37 * E:] {unused = max(unused, abs(v))}
	d_runs, d_device, d_twice := equal(first, second), equal(first, device), equal(first, twice, 2)
	check(d_runs == 0, "%d of %d gradient values differ between two runs", d_runs, len(first))
	check(d_device == 0, "%d values differ with ids only on the device", d_device)
	check(d_twice == 0, "%d values of two backwards differ from twice one", d_twice)
	check(worst < 1e-4, "largest difference from the f64 sums %.2e", worst)
	check(unused == 0, "unused ids got gradient %.2e", unused)
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
