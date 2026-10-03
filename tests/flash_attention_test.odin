package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

flash_attention_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== Testing FlashAttention (Kernel Fusion) ===")

	batch := 2
	seq_len := 128
	d_k := 64
	d_v := 64
	alloc := context.allocator

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

	fmt.println("Running Standard Scaled Dot-Product Attention...")
	Out_std := t.tensor_scaled_dot_product_attention(Q, K, V)

	fmt.println("Running FlashAttention (Fused Kernel)...")
	Out_flash := t.tensor_flash_attention(Q, K, V)

	max_diff: f64 = 0.0
	for i in 0 ..< len(Out_std.data.data) {
		diff := math.abs(Out_std.data.data[i] - Out_flash.data.data[i])
		if diff > max_diff {max_diff = diff}
	}
	fmt.printf("Forward Pass Max Difference: %e (Should be ~1e-15)\n", max_diff)

	fmt.println("Running Backward Passes...")
	loss_std := t.tensor_sum(Out_std)
	t.tensor_backward(loss_std)

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

	t.tensor_zero_grad(Q)
	t.tensor_zero_grad(K)
	t.tensor_zero_grad(V)

	loss_flash := t.tensor_sum(Out_flash)
	t.tensor_backward(loss_flash)

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

	t.tensor_free_graph(loss_std)
	t.tensor_free_graph(loss_flash)
	t.tensor_free(Q)
	t.tensor_free(K)
	t.tensor_free(V)
}

flash_attention_2_test :: proc(allocator: mem.Allocator) {
	fmt.println("=== Testing FlashAttention (Kernel Fusion & Causal Masking) ===")

	batch := 2
	seq_len := 128
	d_k := 64
	d_v := 64
	alloc := context.allocator

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

	// ---------------------------------------------------------
	// TEST 1: Standard Attention (No Mask)
	// ---------------------------------------------------------
	fmt.println("\n--- Test 1: Standard Attention (No Mask) ---")
	Out_std := t.tensor_scaled_dot_product_attention(Q, K, V)
	Out_flash := t.tensor_flash_attention(Q, K, V, causal = false)

	max_diff: f64 = 0.0
	for i in 0 ..< len(Out_std.data.data) {
		diff := math.abs(Out_std.data.data[i] - Out_flash.data.data[i])
		if diff > max_diff {max_diff = diff}
	}
	fmt.printf("Forward Pass Max Difference: %e (Should be ~1e-15)\n", max_diff)

	// ✅ FIX: Wrap unattached outputs in a dummy sum node to ensure they are properly freed
	loss_std_dummy := t.tensor_sum(Out_std)
	loss_flash_dummy := t.tensor_sum(Out_flash)
	t.tensor_free_graph(loss_std_dummy)
	t.tensor_free_graph(loss_flash_dummy)

	// ---------------------------------------------------------
	// TEST 2: Causal Attention (GPT Decoder Style)
	// ---------------------------------------------------------
	fmt.println("\n--- Test 2: Causal Attention (GPT Decoder Style) ---")

	mask := make([]f64, seq_len * seq_len, alloc)
	for i in 0 ..< seq_len {
		for j in 0 ..< seq_len {
			if j > i {
				mask[i * seq_len + j] = -1e9
			} else {
				mask[i * seq_len + j] = 0.0
			}
		}
	}

	Out_causal_std := t.tensor_masked_scaled_dot_product_attention(Q, K, V, mask)
	Out_causal_flash := t.tensor_flash_attention(Q, K, V, causal = true)

	max_diff_causal: f64 = 0.0
	for i in 0 ..< len(Out_causal_std.data.data) {
		diff := math.abs(Out_causal_std.data.data[i] - Out_causal_flash.data.data[i])
		if diff > max_diff_causal {max_diff_causal = diff}
	}
	fmt.printf("Causal Forward Pass Max Difference: %e (Should be ~1e-15)\n", max_diff_causal)

	// ---------------------------------------------------------
	// TEST 3: Causal Backward Pass Verification
	// ---------------------------------------------------------
	fmt.println("\n--- Test 3: Causal Backward Pass Verification ---")

	loss_causal_std := t.tensor_sum(Out_causal_std)
	t.tensor_zero_grad(Q)
	t.tensor_zero_grad(K)
	t.tensor_zero_grad(V)
	t.tensor_backward(loss_causal_std)

	dQ_std := make([]f64, len(Q.grad.data), alloc)
	copy(dQ_std, Q.grad.data)

	loss_causal_flash := t.tensor_sum(Out_causal_flash)
	t.tensor_zero_grad(Q)
	t.tensor_zero_grad(K)
	t.tensor_zero_grad(V)
	t.tensor_backward(loss_causal_flash)

	max_diff_dQ: f64 = 0.0
	for i in 0 ..< len(Q.grad.data) {
		diff := math.abs(dQ_std[i] - Q.grad.data[i])
		if diff > max_diff_dQ {max_diff_dQ = diff}
	}
	fmt.printf("Causal Backward Pass dQ Max Difference: %e\n", max_diff_dQ)

	if max_diff_dQ < 1e-10 {
		fmt.println("✅ Causal Backward Pass Matches!")
	} else {
		fmt.println("❌ Causal Gradient Mismatch!")
	}

	fmt.println("\n✅ FlashAttention Test Complete!")

	delete(mask, alloc)
	delete(dQ_std, alloc)

	t.tensor_free_graph(loss_causal_std)
	t.tensor_free_graph(loss_causal_flash)

	t.tensor_free(Q)
	t.tensor_free(K)
	t.tensor_free(V)
}
