package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

flash_attention_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== Testing FlashAttention (Kernel Fusion) ===")

	// Setup dimensions
	batch := 2
	seq_len := 128 // Large enough to show tiling benefits
	d_k := 64
	d_v := 64

	alloc := context.allocator

	// Create random Q, K, V
	Q_data := l.matrix_new(f64, batch * seq_len, d_k, alloc)
	K_data := l.matrix_new(f64, batch * seq_len, d_k, alloc)
	V_data := l.matrix_new(f64, batch * seq_len, d_v, alloc)

	for i in 0 ..< len(Q_data.data) {Q_data.data[i] = math.sin(f64(i) * 0.01)}
	for i in 0 ..< len(K_data.data) {K_data.data[i] = math.cos(f64(i) * 0.01)}
	for i in 0 ..< len(V_data.data) {V_data.data[i] = math.sin(f64(i) * 0.02)}

	Q := t.tensor_new(Q_data, true, alloc)
	K := t.tensor_new(K_data, true, alloc)
	V := t.tensor_new(V_data, true, alloc)

	Q.shape = [4]int{batch, seq_len, d_k, 1}
	K.shape = [4]int{batch, seq_len, d_k, 1}
	V.shape = [4]int{batch, seq_len, d_v, 1}

	// 1. Standard Attention (Materializes N x N matrix)
	fmt.println("Running Standard Scaled Dot-Product Attention...")
	Out_std := t.tensor_scaled_dot_product_attention(Q, K, V)

	// 2. FlashAttention (Fused, O(N) memory)
	fmt.println("Running FlashAttention (Fused Kernel)...")
	Out_flash := t.tensor_flash_attention(Q, K, V)

	// Verify Forward Pass Equivalence
	max_diff: f64 = 0.0
	for i in 0 ..< len(Out_std.data.data) {
		diff := math.abs(Out_std.data.data[i] - Out_flash.data.data[i])
		if diff > max_diff {max_diff = diff}
	}
	fmt.printf("Forward Pass Max Difference: %e (Should be ~1e-15)\n", max_diff)

	// 3. Test Backward Pass
	fmt.println("Running Backward Passes...")

	// Create a dummy scalar loss to backprop from
	loss_std := t.tensor_sum(Out_std)
	t.tensor_backward(loss_std)

	// ✅ CRITICAL: Save the standard gradients BEFORE zeroing them out
	dQ_std := make([]f64, len(Q.grad.data), alloc)
	dK_std := make([]f64, len(K.grad.data), alloc)
	dV_std := make([]f64, len(V.grad.data), alloc)
	defer {
		delete(dQ_std, alloc)
		delete(dK_std, alloc)
		delete(dV_std, alloc)
	}
	copy(dQ_std, Q.grad.data)
	copy(dK_std, K.grad.data)
	copy(dV_std, V.grad.data)

	// Zero out gradients to test Flash backward independently
	t.tensor_zero_grad(Q)
	t.tensor_zero_grad(K)
	t.tensor_zero_grad(V)

	loss_flash := t.tensor_sum(Out_flash)
	t.tensor_backward(loss_flash)

	// Verify Backward Pass Equivalence
	fmt.println("Verifying Gradients...")
	max_diff_dQ: f64 = 0.0
	max_diff_dK: f64 = 0.0
	max_diff_dV: f64 = 0.0

	for i in 0 ..< len(Q.grad.data) {
		diff := math.abs(Q.grad.data[i] - dQ_std[i])
		if diff > max_diff_dQ {max_diff_dQ = diff}
	}
	for i in 0 ..< len(K.grad.data) {
		diff := math.abs(K.grad.data[i] - dK_std[i])
		if diff > max_diff_dK {max_diff_dK = diff}
	}
	for i in 0 ..< len(V.grad.data) {
		diff := math.abs(V.grad.data[i] - dV_std[i])
		if diff > max_diff_dV {max_diff_dV = diff}
	}

	fmt.printf("dQ Max Diff: %e (Should be ~1e-15)\n", max_diff_dQ)
	fmt.printf("dK Max Diff: %e (Should be ~1e-15)\n", max_diff_dK)
	fmt.printf("dV Max Diff: %e (Should be ~1e-15)\n", max_diff_dV)

	if max_diff_dQ < 1e-10 && max_diff_dK < 1e-10 && max_diff_dV < 1e-10 {
		fmt.println("✅ FlashAttention Backward Pass Matches Standard Attention!")
	} else {
		fmt.println("❌ Gradient Mismatch Detected!")
	}
	fmt.println("✅ FlashAttention Test Complete!")

	// Cleanup
	t.tensor_free_graph(loss_std)
	t.tensor_free_graph(loss_flash)
	t.tensor_free(Q)
	t.tensor_free(K)
	t.tensor_free(V)
}
