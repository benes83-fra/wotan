package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

// Graph scopes (one arena per computation graph):
// - op outputs, gradient buffers, saved state and backward scratch go to the
//   scope's arena; parameters keep their own allocator and their gradients
//   survive graph_end
// - results inside a scope are bit-identical to running without one
// - a second iteration reuses the arena's memory (no new chunks) and allocates
//   nothing from the parameters' allocator
// - tensor_free_graph inside a scope is allowed; the global tensor counters
//   stay balanced across graph_end
// - scopes nest; without a scope op outputs keep their input's allocator

@(private = "file")
_Model :: struct {
	x, w, b, head:       ^t.Tensor, // dense: relu(x @ w + b) . head
	lx, h0, c0:          ^t.Tensor, // LSTM inputs (leaves without gradients)
	w_ih, w_hh, l_bias:  ^t.Tensor, // LSTM parameters
	sx, sh0, sA, sB, sC: ^t.Tensor, // SSM inputs
	sDelta, sD:          ^t.Tensor,
}

@(private = "file")
_params :: proc(m: ^_Model) -> [8]^t.Tensor {
	return {m.w, m.b, m.head, m.w_ih, m.w_hh, m.l_bias, m.sA, m.sD}
}

// Scalar loss over a dense layer, an LSTM and an SSM
@(private = "file")
_loss :: proc(m: ^_Model) -> ^t.Tensor {
	dense := t.tensor_sum(
		t.tensor_mul(t.tensor_relu(t.tensor_add_bias(t.tensor_matmul(m.x, m.w), m.b)), m.head),
	)
	lstm := t.tensor_sum(t.tensor_lstm(m.lx, m.h0, m.c0, m.w_ih, m.w_hh, m.l_bias))
	ssm := t.tensor_sum(t.tensor_ssm(m.sx, m.sh0, m.sA, m.sB, m.sC, m.sDelta, m.sD))
	return t.tensor_add(t.tensor_add(dense, lstm), ssm)
}

@(private = "file")
_Result :: struct {
	loss:  f64,
	grads: [8][]f64,
}

// Loss value and copies of the parameter gradients (gradients are zeroed first)
@(private = "file")
_step :: proc(m: ^_Model, keep: mem.Allocator) -> (r: _Result, root: ^t.Tensor) {
	ps := _params(m)
	for p in ps {t.tensor_zero_grad(p)}
	root = _loss(m)
	t.tensor_backward(root)
	r.loss = root.data.data[0]
	for p, i in ps {
		r.grads[i] = make([]f64, len(p.grad.data), keep)
		copy(r.grads[i], p.grad.data)
	}
	return
}

@(private = "file")
_bits_equal :: proc(a, b: _Result) -> bool {
	if transmute(u64)a.loss != transmute(u64)b.loss {return false}
	for i in 0 ..< len(a.grads) {
		if len(a.grads[i]) != len(b.grads[i]) {return false}
		for j in 0 ..< len(a.grads[i]) {
			if transmute(u64)a.grads[i][j] != transmute(u64)b.grads[i][j] {return false}
		}
	}
	return true
}

graph_scope_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Graph Scope Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// Model tensors live on a tracking allocator, so allocations from the
	// parameters' allocator during scoped iterations can be counted
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, allocator)
	defer mem.tracking_allocator_destroy(&track)
	pa := mem.tracking_allocator(&track)

	B, S, I, H, Dm, N := 4, 6, 5, 8, 6, 4
	m := _Model {
		x      = gcl_rand(B, I, {B, I, 1, 1}, 801, false, pa),
		w      = gcl_rand(I, H, {I, H, 1, 1}, 802, true, pa, 0.5),
		b      = gcl_rand(1, H, {1, H, 1, 1}, 803, true, pa, 0.1),
		head   = gcl_rand(B, H, {B, H, 1, 1}, 804, true, pa),
		lx     = gcl_rand(1, B * S * I, {B, S, I, 1}, 805, false, pa),
		h0     = gcl_rand(B, H, {B, H, 1, 1}, 806, false, pa, 0.5),
		c0     = gcl_rand(B, H, {B, H, 1, 1}, 807, false, pa, 0.5),
		w_ih   = gcl_rand(I, 4 * H, {I, 4 * H, 1, 1}, 808, true, pa, 0.3),
		w_hh   = gcl_rand(H, 4 * H, {H, 4 * H, 1, 1}, 809, true, pa, 0.3),
		l_bias = gcl_rand(1, 4 * H, {1, 4 * H, 1, 1}, 810, true, pa, 0.3),
		sx     = gcl_rand(1, B * S * Dm, {B, S, Dm, 1}, 811, false, pa),
		sh0    = gcl_rand(1, B * Dm * N, {B, Dm, N, 1}, 812, false, pa, 0.5),
		sA     = gcl_rand(Dm, N, {Dm, N, 1, 1}, 813, true, pa),
		sB     = gcl_rand(1, B * S * N, {B, S, N, 1}, 814, false, pa),
		sC     = gcl_rand(1, B * S * N, {B, S, N, 1}, 815, false, pa),
		sDelta = gcl_rand(1, B * S * Dm, {B, S, Dm, 1}, 816, false, pa),
		sD     = gcl_rand(1, Dm, {1, Dm, 1, 1}, 817, true, pa),
	}
	gco_remap(m.sA, -1.5, -0.2)
	gco_remap(m.sDelta, 0.1, 1.0)
	defer {
		for x in ([]^t.Tensor{m.x, m.w, m.b, m.head, m.lx, m.h0, m.c0, m.w_ih, m.w_hh, m.l_bias, m.sx, m.sh0, m.sA, m.sB, m.sC, m.sDelta, m.sD}) {
			t.tensor_free(x)
		}
	}

	// Reference: no scope (also allocates the parameter gradients, once)
	ref, ref_root := _step(&m, allocator)
	t.tensor_free_graph(ref_root)

	// Without a scope, op outputs keep their input's allocator
	{
		y := t.tensor_relu(m.w)
		expect(
			&passed,
			&total,
			y.allocator.procedure == m.w.allocator.procedure &&
			y.allocator.data == m.w.allocator.data,
			"without a scope, op outputs use their input's allocator",
		)
		t.tensor_free(y)
	}

	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	live :: proc() -> int {return t.global_tensors_created - t.global_tensors_freed}

	// Iteration 1
	live_before := live()
	t.graph_begin(&scope)
	r1, root1 := _step(&m, allocator)
	in_scope :=
		root1.allocator.data == rawptr(&scope) && root1.inputs[0].allocator.data == rawptr(&scope)
	param_owned := m.w.allocator.data == pa.data && m.w.grad.allocator.data == pa.data
	t.graph_end(&scope)
	_, reserved1 := t.graph_scope_bytes(&scope)
	expect(&passed, &total, in_scope, "op outputs inside a scope come from the scope's arena")
	expect(&passed, &total, param_owned, "parameters and their gradients keep their own allocator")
	expect(
		&passed,
		&total,
		_bits_equal(ref, r1),
		"loss and parameter gradients bit-identical to running without a scope",
	)
	expect(&passed, &total, live() == live_before, "tensor counters balanced after graph_end")

	// Iteration 2: same results, no new arena memory, nothing from the parameters' allocator;
	// tensor_free_graph inside the scope is allowed
	allocs_before := track.total_allocation_count
	t.graph_begin(&scope)
	r2, root2 := _step(&m, allocator)
	t.tensor_free_graph(root2)
	t.graph_end(&scope)
	_, reserved2 := t.graph_scope_bytes(&scope)
	expect(
		&passed,
		&total,
		_bits_equal(ref, r2),
		"second iteration bit-identical (memory reused after reset)",
	)
	expect(
		&passed,
		&total,
		reserved1 > 0 && reserved2 == reserved1,
		fmt.tprintf(
			"second iteration reuses the arena (reserved %d -> %d bytes)",
			reserved1,
			reserved2,
		),
	)
	expect(
		&passed,
		&total,
		track.total_allocation_count == allocs_before,
		fmt.tprintf(
			"no allocations from the parameters' allocator in a scoped iteration (%d)",
			track.total_allocation_count - allocs_before,
		),
	)
	expect(
		&passed,
		&total,
		live() == live_before,
		"tensor counters balanced with tensor_free_graph inside the scope",
	)

	// Nesting: the inner scope is released, the outer one keeps working
	{
		inner: t.Graph_Scope
		defer t.graph_scope_destroy(&inner)
		t.graph_begin(&scope)
		a := t.tensor_relu(m.w)
		t.graph_begin(&inner)
		b := t.tensor_relu(a)
		in_inner := b.allocator.data == rawptr(&inner)
		t.graph_end(&inner)
		c := t.tensor_mul(a, a)
		ok := in_inner && c.allocator.data == rawptr(&scope) && t.graph_active_scope() == &scope
		ok &&= c.data.data[0] == a.data.data[0] * a.data.data[0]
		t.graph_end(&scope)
		expect(
			&passed,
			&total,
			ok && t.graph_active_scope() == nil,
			"scopes nest: inner graph_end leaves the outer scope active",
		)
	}

	fmt.printf("Graph scope contracts: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
