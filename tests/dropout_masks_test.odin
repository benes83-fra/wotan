package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// tensor_dropout with set_dropout_masks(.Counter): philox4x32_10 against the
// Random123 known-answer vectors; masks follow the documented rule (word i of
// the Philox stream keyed by one rand.uint64() kept when >= ⌊p · 2³²⌋), the
// output and the gradient use them, the share kept is 1 - p, a call takes one
// draw from the random generator and is reproducible; p = 1 drops everything;
// .Stream gives the masks of rand.float64() as before.
dropout_masks_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Dropout mask generators ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	previous := t.set_dropout_masks(.Counter)
	defer t.set_dropout_masks(previous)

	// 1. Known-answer vectors of Random123 (kat_vectors, philox4x32_10)
	Kat :: struct {
		ctr:  [4]u32,
		key:  [2]u32,
		want: [4]u32,
	}
	kats := [?]Kat {
		{{0, 0, 0, 0}, {0, 0}, {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{
			{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
			{0xffffffff, 0xffffffff},
			{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
		},
		{
			{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
			{0xa4093822, 0x299f31d0},
			{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1},
		},
	}
	kat_ok := true
	for k in kats {kat_ok = kat_ok && t.philox4x32_10(k.ctr, k.key) == k.want}
	expect(&passed, &total, kat_ok, "philox4x32_10 matches the 3 Random123 known-answer vectors")

	// 2. The mask rule, the output and the gradient (n not a multiple of 4)
	n :: 100_003
	p :: 0.3
	x := t.tensor_new(l.matrix_new(f64, 1, n), true)
	defer t.tensor_free(x)
	for &v, i in x.data.data {v = math.sin(f64(i)) + 2}
	rand.reset(77)
	seed := rand.uint64()
	next := rand.uint64()
	rand.reset(77)
	y := t.tensor_dropout(x, p, true)
	one_draw := rand.uint64() == next
	key := [2]u32{u32(seed), u32(seed >> 32)}
	p_run: f64 = p
	threshold := u32(p_run * 4294967296.0)
	scale := 1.0 / (1.0 - p)
	rule_ok, out_ok := true, true
	kept := 0
	for i in 0 ..< n {
		w := t.philox4x32_10({u32(i / 4), 0, 0, 0}, key)[i % 4]
		want := scale if w >= threshold else 0
		if want != 0 {kept += 1}
		rule_ok = rule_ok && y.dropout_mask[i] == want
		out_ok = out_ok && y.data.data[i] == (x.data.data[i] * want if want != 0 else 0)
	}
	expect(
		&passed,
		&total,
		rule_ok,
		"Counter mask: word i of the stream kept when >= floor(p * 2^32)",
	)
	expect(&passed, &total, out_ok, "Counter output: x * mask")
	expect(&passed, &total, one_draw, "one draw from the random generator per call")
	share := f64(kept) / n
	sigma := math.sqrt(p * (1 - p) / n)
	expect(
		&passed,
		&total,
		abs(share - (1 - p)) < 4 * sigma,
		fmt.tprintf("share kept %.5f, expected %.5f (4 sigma %.5f)", share, 1 - p, 4 * sigma),
	)
	loss := t.tensor_sum(y)
	t.tensor_backward(loss)
	grad_ok := true
	for g, i in x.grad.data {grad_ok = grad_ok && g == y.dropout_mask[i]}
	expect(&passed, &total, grad_ok, "gradient of sum(dropout(x)): the mask")
	t.tensor_free_graph(loss)
	t.tensor_zero_grad(x)

	// 3. Reproducible after rand.reset; the next call draws another mask
	rand.reset(77)
	y1 := t.tensor_dropout(x, p, true)
	y2 := t.tensor_dropout(x, p, true)
	rand.reset(77)
	y3 := t.tensor_dropout(x, p, true)
	same13, same12 := true, true
	for v, i in y1.dropout_mask {
		same13 = same13 && v == y3.dropout_mask[i]
		same12 = same12 && v == y2.dropout_mask[i]
	}
	expect(
		&passed,
		&total,
		same13 && !same12,
		"the same mask after rand.reset, another one for the next call",
	)
	for z in ([]^t.Tensor{y1, y2, y3}) {t.tensor_free_graph(z)}

	// 4. p = 1 drops everything; not training is the identity
	all := t.tensor_dropout(x, 1, true)
	zero := true
	for v in all.data.data {zero = zero && v == 0}
	ident := t.tensor_dropout(x, p, false)
	same := ident.dropout_mask == nil
	for v, i in ident.data.data {same = same && v == x.data.data[i]}
	expect(
		&passed,
		&total,
		zero && same,
		"p = 1 drops every element; evaluation mode is the identity",
	)
	t.tensor_free_graph(all)
	t.tensor_free_graph(ident)

	// 5. .Stream: rand.float64() > p per element, as before
	t.set_dropout_masks(.Stream)
	rand.reset(78)
	draws := make([]f64, n)
	for &d in draws {d = rand.float64()}
	rand.reset(78)
	s := t.tensor_dropout(x, p, true)
	stream_ok := true
	for d, i in draws {stream_ok = stream_ok && s.dropout_mask[i] == (scale if d > p else 0)}
	expect(&passed, &total, stream_ok, "Stream mask: rand.float64() > p per element")
	t.tensor_free_graph(s)

	record_failures(total - passed)
	fmt.printf("Dropout masks: %d/%d passed\n", passed, total)
}
