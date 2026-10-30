package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

// Deep computation graphs (unrolled recurrences, long accumulation loops):
// - backward, validate and free_graph work on a chain far deeper than the call
//   stack allowed with the former recursive walks (crashed at ~5000 nodes)
// - the topological order puts every node after its inputs, each node once,
//   also when subexpressions are shared (diamonds)

autograd_deep_graph_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Autograd Deep Graph Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	scalar :: proc(v: f64, requires_grad: bool, allocator: mem.Allocator) -> ^t.Tensor {
		m := l.matrix_new(f64, 1, 1, allocator)
		m.data[0] = v
		x := t.tensor_new(m, requires_grad, allocator)
		if requires_grad {t.tensor_ensure_grad(x)}
		return x
	}

	// 1. chain of 50 000 ops: y = ((x + c) * k + c) * k ... ; dy/dx = k^N
	N := 50_000
	x := scalar(1.0, true, allocator)
	c := scalar(0.001, false, allocator)
	k := scalar(1.0, false, allocator) // keeps the derivative exactly 1
	y := x
	for i in 0 ..< N {
		y = t.tensor_add(y, c) if i % 2 == 0 else t.tensor_mul(y, k)
	}
	valid := t.tensor_validate_graph(y)
	t.tensor_backward(y)
	dx := x.grad.data[0]
	before := t.global_tensors_freed
	t.tensor_free_graph(y)
	freed := t.global_tensors_freed - before
	expect(
		&passed,
		&total,
		valid && dx == 1.0 && freed == N,
		fmt.tprintf(
			"chain of %d ops: validate %v, dy/dx = %v, freed %d intermediate nodes",
			N,
			valid,
			dx,
			freed,
		),
	)
	t.tensor_free(x)
	t.tensor_free(c)
	t.tensor_free(k)

	// 2. topological order on a graph with shared subexpressions
	a := scalar(2.0, true, allocator)
	b := scalar(3.0, true, allocator)
	ab := t.tensor_mul(a, b) // shared
	s1 := t.tensor_add(ab, a)
	s2 := t.tensor_mul(ab, b)
	out := t.tensor_add(t.tensor_mul(s1, s2), ab) // ab reached three ways
	topo := make([dynamic]^t.Tensor, 0, 16, allocator)
	defer delete(topo)
	visited := make(map[^t.Tensor]bool, allocator = allocator)
	defer delete(visited)
	t._build_topo(out, &topo, &visited)
	pos := make(map[^t.Tensor]int, allocator = allocator)
	defer delete(pos)
	for n, i in topo {pos[n] = i}
	ordered := len(pos) == len(topo) && topo[len(topo) - 1] == out
	for n in topo {
		for input in n.inputs {ordered &&= pos[input] < pos[n]}
	}
	t.tensor_backward(out)
	// out = (ab + a)(ab b) + ab with ab = a b:
	// d/da = (b + 1)(a b²) + (a b + a)(b²) + b,  d/db = a (a b²) + (a b + a)(2 a b) + a
	av, bv := 2.0, 3.0
	ga := (bv + 1) * (av * bv * bv) + (av * bv + av) * bv * bv + bv
	gb := av * (av * bv * bv) + (av * bv + av) * 2 * av * bv + av
	expect(
		&passed,
		&total,
		ordered && len(topo) == 7 && a.grad.data[0] == ga && b.grad.data[0] == gb,
		fmt.tprintf(
			"diamond graph: %d nodes in topological order, grads %v / %v (analytic %v / %v)",
			len(topo),
			a.grad.data[0],
			b.grad.data[0],
			ga,
			gb,
		),
	)
	t.tensor_free_graph(out)
	t.tensor_free(a)
	t.tensor_free(b)

	fmt.printf("Autograd deep graph summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
