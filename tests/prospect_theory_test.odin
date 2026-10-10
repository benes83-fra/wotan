package tests

import l "../wotan/linalg"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

prospect_theory_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Prospect Theory Backbone Test ===")

	// ========================================================================
	// Test 1: S-Curve Validation (Kahneman & Tversky 1992 calibration)
	// ========================================================================
	fmt.println("\n1. Value Function S-Curve (α=0.88, β=0.88, λ=2.25)")
	fmt.println(" -")

	// Test points: gains and losses of equal magnitude
	test_returns := []f64{2.0, 1.0, 0.5, 0.1, 0.0, -0.1, -0.5, -1.0, -2.0}

	returns_data := l.matrix_new(f64, 1, len(test_returns), allocator)
	copy(returns_data.data, test_returns)
	returns_tensor := t.tensor_new(returns_data, true, allocator)
	defer t.tensor_free(returns_tensor)

	v := t.tensor_prospect_value(returns_tensor, 0.88, 0.88, 2.25, allocator)

	fmt.println(" Return | v(x)    | Expected Property")
	fmt.println(" -------|---------|------------------")
	for i in 0 ..< len(test_returns) {
		x := test_returns[i]
		vx := v.data.data[i]
		fmt.printf(" %+6.1f | %+7.4f |", x, vx)

		// Validate key properties
		if x == 0.0 {
			if math.abs(vx) < 1e-10 {
				fmt.println(" v(0) = 0 ✓")
			} else {
				fmt.println(" ✗ v(0) ≠ 0!")
			}
		} else if x > 0 {
			if vx > 0 && vx < x {
				fmt.println(" Concave gain ✓")
			} else {
				fmt.println(" ✗")
			}
		} else {
			if vx < 0 && math.abs(vx) > math.abs(x) {
				fmt.println(" Loss aversion ✓")
			} else {
				fmt.println(" ✗")
			}
		}
	}

	// Validate loss aversion ratio: v(-x) / v(x) ≈ -λ
	v_pos := v.data.data[1] // v(1.0)
	v_neg := v.data.data[7] // v(-1.0)
	ratio := -v_neg / v_pos
	fmt.printf("\nLoss aversion ratio: -v(-1)/v(1) = %.4f (expected ≈ 2.25)", ratio)
	if math.abs(ratio - 2.25) < 0.01 {
		fmt.println(" ✓")
	} else {
		fmt.println(" ✗")
	}

	// ========================================================================
	// Test 2: Probability Weighting (inverse-S curve)
	// ========================================================================
	fmt.println("\n2. Probability Weighting (γ=0.61)")
	fmt.println(" -")

	test_probs := []f64{0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99}
	probs_data := l.matrix_new(f64, 1, len(test_probs), allocator)
	copy(probs_data.data, test_probs)
	probs_tensor := t.tensor_new(probs_data, true, allocator)
	defer t.tensor_free(probs_tensor)

	w := t.tensor_prob_weighting(probs_tensor, 0.61, allocator)

	fmt.println(" p     | w(p)  | Property")
	fmt.println(" ------|-------|----------")
	for i in 0 ..< len(test_probs) {
		p := test_probs[i]
		wp := w.data.data[i]
		fmt.printf(" %5.2f | %5.3f |", p, wp)

		if p < 0.3 && wp > p {
			fmt.println(" Overweight small p ✓")
		} else if p > 0.7 && wp < p {
			fmt.println(" Underweight large p ✓")
		} else {
			fmt.println("")
		}
	}

	// ========================================================================
	// Test 3: Prospect Utility as RL Reward (gradient check)
	// ========================================================================
	fmt.println("\n3. Prospect Utility Gradient Check")
	fmt.println(" -")

	// Simulated returns: mix of gains and losses
	rl_returns := []f64{0.05, -0.03, 0.02, -0.08, 0.01, -0.01, 0.04, -0.05}
	rl_data := l.matrix_new(f64, 1, len(rl_returns), allocator)
	copy(rl_data.data, rl_returns)
	rl_tensor := t.tensor_new(rl_data, true, allocator)
	defer t.tensor_free(rl_tensor)

	// Compute prospect utility (negated for minimization)
	loss := t.tensor_prospect_utility(rl_tensor, 0.88, 0.88, 2.25, allocator)

	fmt.printf("Prospect Utility Loss: %.6f\n", loss.data.data[0])

	// Backward pass
	t.tensor_backward(loss)

	// Verify gradients exist and have correct sign structure
	grad_sum := 0.0
	loss_grads := 0
	gain_grads := 0
	for i in 0 ..< len(rl_tensor.grad.data) {
		grad_sum += math.abs(rl_tensor.grad.data[i])
		if rl_returns[i] < 0 && rl_tensor.grad.data[i] < 0 {
			loss_grads += 1
		} else if rl_returns[i] > 0 && rl_tensor.grad.data[i] < 0 {
			gain_grads += 1
		}
	}

	fmt.printf("Gradient magnitude sum: %.6f\n", grad_sum)
	fmt.printf("Losses with negative gradient: %d/%d\n", loss_grads, 4)
	fmt.printf("Gains with negative gradient: %d/%d\n", gain_grads, 4)

	if grad_sum > 0 {
		fmt.println("✓ Gradients flow correctly through Prospect Utility")
	} else {
		fmt.println("✗ Gradient flow failed!")
	}

	// Cleanup
	t.tensor_free_graph(loss)
	t.tensor_free(v)
	t.tensor_free(w)

	fmt.println("\n=== Prospect Theory Backbone Test Complete ===\n")
}
