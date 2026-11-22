package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:mem"

// tensor_multi_head_attention with key_padding (the mask kept as one row of
// seq_k values per sequence) against the same attention with the full additive
// mask [batch·heads, seq_q, seq_k] that padding used to build: identical output
// and gradients of q, k and v under a random upstream gradient, for padding
// alone, padding with causal masking, and cross-attention (seq_q != seq_k);
// the recorded mask has batch·seq_k values instead of batch·heads·seq_q·seq_k.
attention_padding_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== attention key padding vs the full mask ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	B, H, hd :: 3, 2, 4
	lengths := [B]int{7, 5, 3} // valid keys per sequence
	Case :: struct {
		name:         string,
		seq_q, seq_k: int,
		causal:       bool,
	}
	cases := [?]Case {
		{"padding", 7, 7, false},
		{"padding + causal", 7, 7, true},
		{"cross-attention 4 queries over 7 keys, padding", 4, 7, false},
	}
	for c in cases {
		n_q, n_k := B * c.seq_q * H * hd, B * c.seq_k * H * hd
		vals := [3][]f64 {
			make([]f64, n_q, context.temp_allocator),
			make([]f64, n_k, context.temp_allocator),
			make([]f64, n_k, context.temp_allocator),
		}
		for v in vals {for &x in v {x = rand.norm_float64()}}
		up := make([]f64, n_q, context.temp_allocator)
		for &x in up {x = rand.norm_float64()}
		kp := make([]f64, B * c.seq_k, context.temp_allocator)
		for b in 0 ..< B {for j in 0 ..< min(lengths[b], c.seq_k) {kp[b * c.seq_k + j] = 1}}
		// the full mask as padding used to build it: ATTN_MASKED once where a
		// key is padding or (causal) after the query
		full := make([]f64, B * H * c.seq_q * c.seq_k, context.temp_allocator)
		for blk in 0 ..< B * H {
			for i in 0 ..< c.seq_q {
				for j in 0 ..< c.seq_k {
					hidden := (c.causal && j > i) || kp[(blk / H) * c.seq_k + j] == 0
					if hidden {full[(blk * c.seq_q + i) * c.seq_k + j] = t.ATTN_MASKED}
				}
			}
		}
		run :: proc(
			vals: [3][]f64,
			up: []f64,
			seq_q, seq_k: int,
			causal: bool,
			kp, full: []f64,
		) -> (
			out: []f64,
			grads: [3][]f64,
			mask_values: int,
		) {
			xs: [3]^t.Tensor
			for &x, i in xs {
				m := l.matrix_new(f64, 1, len(vals[i]))
				copy(m.data, vals[i])
				x = t.tensor_new(m, true)
				x.shape = {B, seq_q if i == 0 else seq_k, H * hd, 1}
			}
			y: ^t.Tensor
			if kp != nil {
				y = t.tensor_multi_head_attention(
					xs[0],
					xs[1],
					xs[2],
					B,
					seq_q,
					seq_k,
					H,
					hd,
					causal = causal,
					key_padding = kp,
				)
			} else {
				y = t.tensor_multi_head_attention(
					xs[0],
					xs[1],
					xs[2],
					B,
					seq_q,
					seq_k,
					H,
					hd,
					mask = full,
				)
			}
			t.tensor_backward_with_grad(y, up)
			out = y.data.data
			for x, i in xs {grads[i] = x.grad.data}
			mask_values = len(y.inputs[0].f64_metadata) // the attention node under the inverse permute
			return
		}
		new_out, new_g, new_n := run(vals, up, c.seq_q, c.seq_k, c.causal, kp, nil)
		old_out, old_g, old_n := run(vals, up, c.seq_q, c.seq_k, c.causal, nil, full)
		same := len(new_out) == len(old_out)
		for v, i in new_out {same &&= v == old_out[i]}
		for g, k in new_g {for v, i in g {same &&= v == old_g[k][i]}}
		expect(
			&passed,
			&total,
			same && new_n == B * c.seq_k && old_n == len(full),
			fmt.tprintf(
				"%s (batch 3, 2 heads, lengths 7 5 3): output and q / k / v gradients identical to the full mask %v; mask values kept %d instead of %d",
				c.name,
				same,
				new_n,
				old_n,
			),
		)
	}
	fmt.printf("attention key padding: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
