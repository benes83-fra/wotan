package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

// Regression tests for autograd API contracts: logical shapes are always set
// and consistent with the data, and backward seeding is explicit/predictable.

api_passed, api_total: int

api_expect :: proc(cond: bool, name: string) {
	api_total += 1
	if cond {
		api_passed += 1
		fmt.printf("OK   %s\n", name)
	} else {
		fmt.printf("FAIL %s\n", name)
	}
}

autograd_api_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Autograd API Contract Tests ===")
	context.allocator = allocator
	api_passed, api_total = 0, 0

	// 1. Leaves get a default shape equal to their matrix dims
	m := gcl_rand(3, 4, {3, 4, 1, 1}, 201, false, allocator)
	defer t.tensor_free(m)
	raw := t.tensor_new(l.matrix_new(f64, 2, 5, allocator), false, allocator)
	defer t.tensor_free(raw)
	api_expect(raw.shape == [4]int{2, 5, 1, 1}, "tensor_new default shape = {rows, cols, 1, 1}")

	// 2. Reductions report a scalar shape (tensor_sum used to copy the input shape)
	s := t.tensor_sum(m)
	defer t.tensor_free(s)
	api_expect(s.shape == t.SCALAR_SHAPE, "tensor_sum output shape is scalar")

	// 3. Flattened matmul output shape
	W := gcl_rand(4, 5, {4, 5, 1, 1}, 202, false, allocator)
	defer t.tensor_free(W)
	x3 := gcl_rand(1, 2 * 3 * 4, {2, 3, 4, 1}, 203, false, allocator)
	defer t.tensor_free(x3)
	y3 := t.tensor_matmul(x3, W)
	defer t.tensor_free(y3)
	api_expect(y3.shape == [4]int{2, 3, 5, 1}, "flattened matmul [2,3,4]@[4,5] -> [2,3,5]")

	x_flat := gcl_rand(1, 6 * 4, {1, 24, 1, 1}, 204, false, allocator)
	defer t.tensor_free(x_flat)
	y_flat := t.tensor_matmul(x_flat, W)
	defer t.tensor_free(y_flat)
	api_expect(y_flat.shape == [4]int{6, 5, 1, 1}, "flattened matmul with plain [1,24] shape -> [6,5]")

	// Feature dim of size 1 (e.g. a scalar time series embedded by Linear(1, d))
	W1 := gcl_rand(1, 24, {1, 24, 1, 1}, 206, false, allocator)
	defer t.tensor_free(W1)
	x1 := gcl_rand(1, 32 * 16, {32, 16, 1, 1}, 207, false, allocator)
	defer t.tensor_free(x1)
	y1 := t.tensor_matmul(x1, W1)
	defer t.tensor_free(y1)
	api_expect(y1.shape == [4]int{32, 16, 24, 1}, "flattened matmul [32,16,1]@[1,24] -> [32,16,24]")

	// 4. Shape-preserving ops keep the logical shape
	sc := t.tensor_scale(x3, 0.5)
	defer t.tensor_free(sc)
	api_expect(sc.shape == x3.shape, "tensor_scale keeps shape")

	// 5. Scalar roots are always seeded with 1, even if root.grad holds a stale value
	p := gcl_rand(3, 4, {3, 4, 1, 1}, 205, true, allocator)
	defer t.tensor_free(p)
	loss := t.tensor_sum(t.tensor_mul(p, m))
	t.tensor_backward(loss, allocator)
	reference := make([]f64, len(p.grad.data), allocator)
	defer delete(reference, allocator)
	copy(reference, p.grad.data)
	t.tensor_free_graph(loss, allocator)

	t.tensor_zero_grad(p)
	loss2 := t.tensor_sum(t.tensor_mul(p, m))
	t.tensor_ensure_grad(loss2)
	loss2.grad.data[0] = 5.0 // stale value, e.g. left over from an earlier pass
	t.tensor_backward(loss2, allocator)
	same := true
	for i in 0 ..< len(reference) {same &&= abs(p.grad.data[i] - reference[i]) < 1e-12}
	api_expect(same, "scalar root ignores a stale root gradient")
	t.tensor_free_graph(loss2, allocator)

	// 6. tensor_backward_with_grad == legacy pre-filled root.grad
	upstream := []f64{0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 3.0, 0.1, -2.0, 0.8, 1.2, -0.3}
	t.tensor_zero_grad(p)
	out_a := t.tensor_mul(p, m)
	t.tensor_backward_with_grad(out_a, upstream, allocator)
	explicit := make([]f64, len(p.grad.data), allocator)
	defer delete(explicit, allocator)
	copy(explicit, p.grad.data)
	t.tensor_free_graph(out_a, allocator)

	t.tensor_zero_grad(p)
	out_b := t.tensor_mul(p, m)
	t.tensor_ensure_grad(out_b)
	copy(out_b.grad.data, upstream)
	t.tensor_backward(out_b, allocator)
	same = true
	for i in 0 ..< len(explicit) {
		same &&= abs(p.grad.data[i] - explicit[i]) < 1e-12
		same &&= abs(explicit[i] - upstream[i] * m.data.data[i]) < 1e-12
	}
	api_expect(same, "tensor_backward_with_grad matches legacy seeding and dL/dp = g * m")
	t.tensor_free_graph(out_b, allocator)

	fmt.printf("Autograd API summary: %d/%d passed\n", api_passed, api_total)
	record_failures(api_total - api_passed)
}
