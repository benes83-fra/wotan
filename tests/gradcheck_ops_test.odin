package tests

import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:mem"

// Numerical gradient checks for the remaining autograd ops (element-wise math,
// losses, normalization, pooling, reshaping/permutation, routing, SSM, flash
// attention). Uses the gcl_* helpers from gradcheck_layers_test.odin; every
// differentiable input is checked.

gco_targets: []int
gco_idx, gco_rm, gco_rv: ^t.Tensor
gco_causal: bool

// Linearly rescale a tensor's values into [lo, hi], e.g. for log/sqrt/probability inputs
gco_remap :: proc(x: ^t.Tensor, lo, hi: f64) {
	mn, mx := x.data.data[0], x.data.data[0]
	for v in x.data.data {
		mn = min(mn, v)
		mx = max(mx, v)
	}
	span := mx - mn if mx > mn else 1.0
	for i in 0 ..< len(x.data.data) {
		x.data.data[i] = lo + (x.data.data[i] - mn) / span * (hi - lo)
	}
}

gradcheck_ops_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Autograd Op Gradient Checks ===")
	context.allocator = allocator
	gc_passed, gc_total = 0, 0

	// --- 1. Element-wise unary ops on a [3,4] input --------------------------
	gcl_in[0] = gcl_rand(3, 4, {3, 4, 1, 1}, 101, true, allocator, 2.0)
	gcl_check(
		"sigmoid",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_sigmoid(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"tanh",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_tanh(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"exp",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_exp(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"softplus",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_softplus(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"norm_cdf",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_norm_cdf(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"gelu",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_gelu(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"relu",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_relu(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"neg",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_neg(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"mean",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_mean(gcl_in[0]))},
		allocator,
	)
	gcl_check("reshape", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_reshape(gcl_in[0], {2, 6, 1, 1}, context.allocator))
		}, allocator)
	gcl_check(
		"dropout (training, fixed seed)",
		1,
		proc() -> ^t.Tensor {
			rand.reset(7) // same mask on every evaluation
			return gcl_weighted_sum(t.tensor_dropout(gcl_in[0], 0.3, true))
		},
		allocator,
	)
	gco_remap(gcl_in[0], 0.2, 3.0) // strictly positive domain
	gcl_check(
		"log",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_log(gcl_in[0]))},
		allocator,
	)
	gcl_check(
		"sqrt",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_sqrt(gcl_in[0]))},
		allocator,
	)
	gco_remap(gcl_in[0], 0.05, 0.3) // probability-like
	gcl_check(
		"entropy",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_entropy(gcl_in[0]))},
		allocator,
	)
	gcl_free_inputs(1)

	// --- 2. Element-wise binary ops ------------------------------------------
	gcl_in[0] = gcl_rand(3, 4, {3, 4, 1, 1}, 102, true, allocator)
	gcl_in[1] = gcl_rand(3, 4, {3, 4, 1, 1}, 103, true, allocator)
	gcl_check(
		"add",
		2,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_add(gcl_in[0], gcl_in[1]))},
		allocator,
	)
	gcl_check(
		"sub",
		2,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_sub(gcl_in[0], gcl_in[1]))},
		allocator,
	)
	gcl_check(
		"mul",
		2,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_mul(gcl_in[0], gcl_in[1]))},
		allocator,
	)
	gco_remap(gcl_in[1], 0.5, 2.0)
	gcl_check(
		"div",
		2,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_div(gcl_in[0], gcl_in[1]))},
		allocator,
	)
	gcl_free_inputs(2)

	// --- 3. Bias, reductions, routing ----------------------------------------
	gcl_in[0] = gcl_rand(4, 3, {4, 3, 1, 1}, 104, true, allocator)
	gcl_in[1] = gcl_rand(1, 3, {1, 3, 1, 1}, 105, true, allocator)
	gcl_check("add_bias 2D [4,3]+[1,3]", 2, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_add_bias(gcl_in[0], gcl_in[1]))
		}, allocator)
	gcl_free_inputs(2)
	gcl_in[0] = gcl_rand(1, 4 * 3, {1, 4, 3, 1}, 106, true, allocator)
	gcl_in[1] = gcl_rand(1, 3, {1, 3, 1, 1}, 107, true, allocator)
	gcl_check("add_bias flattened [1,4*3]+[1,3]", 2, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_add_bias(gcl_in[0], gcl_in[1]))
		}, allocator)
	gcl_free_inputs(2)

	gcl_in[0] = gcl_rand(3, 4, {3, 4, 1, 1}, 108, true, allocator)
	gcl_check(
		"sum_dim1 [3,4]",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_sum_dim1(gcl_in[0]))},
		allocator,
	)
	gcl_check("logsumexp_dim1 [3,4]", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_logsumexp_dim1(gcl_in[0], 4, context.allocator))
		}, allocator)
	gcl_check("slice_gate [3,4] expert 2", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_slice_gate(gcl_in[0], 2, context.allocator))
		}, allocator)
	gcl_free_inputs(1)

	gcl_in[0] = gcl_rand(3, 4, {3, 4, 1, 1}, 109, true, allocator)
	gcl_in[1] = gcl_rand(3, 1, {3, 1, 1, 1}, 110, true, allocator)
	gcl_check("gate_mul [3,4]*[3,1]", 2, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_gate_mul(gcl_in[0], gcl_in[1], context.allocator))
		}, allocator)
	gcl_free_inputs(2)

	// --- 4. Losses -----------------------------------------------------------
	gco_targets = []int{1, 0, 3}
	gcl_in[0] = gcl_rand(3, 4, {3, 4, 1, 1}, 111, true, allocator, 2.0)
	gcl_check("cross_entropy [3,4]", 1, proc() -> ^t.Tensor {
			return t.tensor_cross_entropy_loss(gcl_in[0], gco_targets)
		}, allocator)
	gcl_free_inputs(1)
	gcl_in[0] = gcl_rand(1, 3 * 4, {1, 3, 4, 1}, 112, true, allocator, 2.0)
	gcl_check("cross_entropy flattened [1,3*4]", 1, proc() -> ^t.Tensor {
			return t.tensor_cross_entropy_loss(gcl_in[0], gco_targets)
		}, allocator)
	gcl_free_inputs(1)

	gcl_in[0] = gcl_rand(3, 4, {3, 4, 1, 1}, 113, true, allocator)
	gcl_in[1] = gcl_rand(3, 4, {3, 4, 1, 1}, 114, false, allocator)
	gcl_check("mse_loss (pred)", 1, proc() -> ^t.Tensor {
			return t.tensor_mse_loss(gcl_in[0], gcl_in[1], context.allocator)
		}, allocator)
	gco_remap(gcl_in[0], 0.05, 0.95)
	gco_remap(gcl_in[1], 0.0, 1.0)
	gcl_check(
		"bce_loss (pred)",
		1,
		proc() -> ^t.Tensor {return t.tensor_bce_loss(gcl_in[0], gcl_in[1])},
		allocator,
	)
	gcl_check("binary_cross_entropy (pred)", 1, proc() -> ^t.Tensor {
			return t.tensor_binary_cross_entropy(gcl_in[0], gcl_in[1])
		}, allocator)
	gcl_free_inputs(2)

	// Losses with a target that requires grad (both inputs differentiable)
	gcl_in[0] = gcl_rand(3, 4, {3, 4, 1, 1}, 115, true, allocator)
	gcl_in[1] = gcl_rand(3, 4, {3, 4, 1, 1}, 116, true, allocator)
	gcl_check("mse_loss (pred + target)", 2, proc() -> ^t.Tensor {
			return t.tensor_mse_loss(gcl_in[0], gcl_in[1], context.allocator)
		}, allocator)
	gco_remap(gcl_in[0], 0.05, 0.95)
	gco_remap(gcl_in[1], 0.05, 0.95)
	gcl_check(
		"bce_loss (pred + target)",
		2,
		proc() -> ^t.Tensor {return t.tensor_bce_loss(gcl_in[0], gcl_in[1])},
		allocator,
	)
	gcl_check("binary_cross_entropy (pred + target)", 2, proc() -> ^t.Tensor {
			return t.tensor_binary_cross_entropy(gcl_in[0], gcl_in[1])
		}, allocator)
	gcl_free_inputs(2)

	// Shape propagation: deep-hedging cost path scale -> mul -> sum_dim1
	// (tensor_scale used to drop the shape, so sum_dim1 produced a 0-row tensor)
	gcl_in[0] = gcl_rand(4, 3, {4, 3, 1, 1}, 146, true, allocator)
	gcl_in[1] = gcl_rand(4, 3, {4, 3, 1, 1}, 147, true, allocator)
	gcl_check("scale -> mul -> sum_dim1 (hedging costs)", 2, proc() -> ^t.Tensor {
			cost_coeff := t.tensor_scale(gcl_in[0], 0.001)
			return gcl_weighted_sum(t.tensor_sum_dim1(t.tensor_mul(cost_coeff, gcl_in[1])))
		}, allocator)
	gcl_check("dropout -> sum_dim1 (shape kept)", 1, proc() -> ^t.Tensor {
			rand.reset(11)
			return gcl_weighted_sum(t.tensor_sum_dim1(t.tensor_dropout(gcl_in[0], 0.25, true)))
		}, allocator)
	gcl_free_inputs(2)

	gcl_in[0] = gcl_rand(2, 3, {2, 3, 1, 1}, 117, true, allocator)
	gcl_in[1] = gcl_rand(2, 3, {2, 3, 1, 1}, 118, true, allocator)
	gcl_check("kl_divergence (mu + log_var)", 2, proc() -> ^t.Tensor {
			return t.tensor_kl_divergence(gcl_in[0], gcl_in[1])
		}, allocator)
	gcl_free_inputs(2)

	// --- 5. Normalization ----------------------------------------------------
	gcl_in[0] = gcl_rand(3, 5, {3, 5, 1, 1}, 119, true, allocator)
	gcl_in[1] = gcl_rand(1, 5, {1, 5, 1, 1}, 120, true, allocator)
	gcl_in[2] = gcl_rand(1, 5, {1, 5, 1, 1}, 121, true, allocator)
	gcl_check("layer_norm [3,5]", 3, proc() -> ^t.Tensor {
			return gcl_weighted_sum(
				t.tensor_layer_norm(gcl_in[0], gcl_in[1], gcl_in[2], 1e-5, context.allocator),
			)
		}, allocator)
	gcl_free_inputs(3)

	gcl_in[0] = gcl_rand(1, 2 * 3 * 2 * 2, {2, 3, 2, 2}, 122, true, allocator)
	gcl_in[1] = gcl_rand(1, 3, {1, 3, 1, 1}, 123, true, allocator)
	gcl_in[2] = gcl_rand(1, 3, {1, 3, 1, 1}, 124, true, allocator)
	gco_rm = gcl_rand(1, 3, {1, 3, 1, 1}, 125, false, allocator)
	gco_rv = gcl_rand(1, 3, {1, 3, 1, 1}, 126, false, allocator)
	gcl_check("batch_norm_2d (training)", 3, proc() -> ^t.Tensor {
			return gcl_weighted_sum(
				t.tensor_batch_norm_2d(
					gcl_in[0],
					gcl_in[1],
					gcl_in[2],
					gco_rm,
					gco_rv,
					true,
					0.1,
					1e-5,
					context.allocator,
				),
			)
		}, allocator)
	gcl_free_inputs(3)
	t.tensor_free(gco_rm)
	t.tensor_free(gco_rv)

	gcl_in[0] = gcl_rand(1, 2 * 2 * 5 * 3, {2, 2, 5, 3}, 127, true, allocator)
	gcl_check("normalize_time [2,2,5,3]", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_normalize_time(gcl_in[0], 1e-8, context.allocator))
		}, allocator)
	gcl_free_inputs(1)

	// --- 6. Pooling, permutation, concat -------------------------------------
	gcl_in[0] = gcl_rand(1, 2 * 2 * 4 * 4, {2, 2, 4, 4}, 128, true, allocator)
	gcl_check(
		"max_pool2d k2 s2",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_max_pool2d(gcl_in[0], 2, 2, 2))},
		allocator,
	)
	gcl_check(
		"avg_pool2d k2 s2",
		1,
		proc() -> ^t.Tensor {return gcl_weighted_sum(t.tensor_avg_pool2d(gcl_in[0], 2, 2, 2))},
		allocator,
	)
	gcl_check("avg_pool2d k3 s1 (overlapping)", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_avg_pool2d(gcl_in[0], 3, 3, 1))
		}, allocator)
	gcl_check("permute_lob [2,2,4,4]", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_permute_lob(gcl_in[0], 2, 2, 4, 4, context.allocator))
		}, allocator)
	gcl_free_inputs(1)

	gcl_in[0] = gcl_rand(1, 2 * 3 * 4, {2, 3, 4, 1}, 129, true, allocator)
	gcl_check("permute_mha (2 heads)", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_permute_mha(gcl_in[0], 2, 3, 2, 2))
		}, allocator)
	gcl_check("permute_mha_inverse (2 heads)", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_permute_mha_inverse(gcl_in[0], 2, 3, 2, 2))
		}, allocator)
	gcl_free_inputs(1)

	gcl_in[0] = gcl_rand(1, 2 * 3 * 2 * 1, {2, 3, 2, 1}, 130, true, allocator)
	gcl_in[1] = gcl_rand(1, 2 * 1 * 2 * 1, {2, 1, 2, 1}, 131, true, allocator)
	gcl_check("concat dim=1", 2, proc() -> ^t.Tensor {
			ts := []^t.Tensor{gcl_in[0], gcl_in[1]}
			return gcl_weighted_sum(t.tensor_concat(ts, 1, context.allocator))
		}, allocator)
	gcl_free_inputs(2)
	gcl_in[0] = gcl_rand(1, 2 * 3 * 2 * 1, {2, 3, 2, 1}, 132, true, allocator)
	gcl_in[1] = gcl_rand(1, 2 * 3 * 4 * 1, {2, 3, 4, 1}, 133, true, allocator)
	gcl_check("concat dim=2", 2, proc() -> ^t.Tensor {
			ts := []^t.Tensor{gcl_in[0], gcl_in[1]}
			return gcl_weighted_sum(t.tensor_concat(ts, 2, context.allocator))
		}, allocator)
	gcl_free_inputs(2)

	// --- 7. Embedding (gradient wrt the weight table) ------------------------
	gco_idx = gcl_rand(2, 3, {2, 3, 1, 1}, 134, false, allocator)
	ids := []f64{0, 4, 2, 4, 1, 5} // index 4 repeats -> scatter-add
	copy(gco_idx.data.data, ids)
	gcl_in[0] = gcl_rand(6, 4, {6, 4, 1, 1}, 135, true, allocator)
	gcl_check("embedding (weight)", 1, proc() -> ^t.Tensor {
			return gcl_weighted_sum(t.tensor_embedding(gco_idx, gcl_in[0]))
		}, allocator)
	gcl_free_inputs(1)
	t.tensor_free(gco_idx)

	// --- 8. Flash attention (self-attention, same sequence length) -----------
	for causal in ([]bool{false, true}) {
		gco_causal = causal
		gcl_in[0] = gcl_rand(1, 2 * 5 * 4, {2, 5, 4, 1}, 136, true, allocator)
		gcl_in[1] = gcl_rand(1, 2 * 5 * 4, {2, 5, 4, 1}, 137, true, allocator)
		gcl_in[2] = gcl_rand(1, 2 * 5 * 3, {2, 5, 3, 1}, 138, true, allocator)
		name := causal ? "flash_attention (causal)" : "flash_attention"
		gcl_check(name, 3, proc() -> ^t.Tensor {
				return gcl_weighted_sum(
					t.tensor_flash_attention(
						gcl_in[0],
						gcl_in[1],
						gcl_in[2],
						gco_causal,
						context.allocator,
					),
				)
			}, allocator)
		gcl_free_inputs(3)
	}

	// --- 9. Selective SSM (Mamba) ---------------------------------------------
	{
		Bn, S, dm, ds := 2, 3, 2, 3
		gcl_in[0] = gcl_rand(1, Bn * S * dm, {Bn, S, dm, 1}, 139, true, allocator) // x
		gcl_in[1] = gcl_rand(1, Bn * dm * ds, {Bn, dm, ds, 1}, 140, true, allocator, 0.5) // h_0
		gcl_in[2] = gcl_rand(dm, ds, {dm, ds, 1, 1}, 141, true, allocator) // A (negative)
		gco_remap(gcl_in[2], -1.5, -0.2)
		gcl_in[3] = gcl_rand(1, Bn * S * ds, {Bn, S, ds, 1}, 142, true, allocator) // B
		gcl_in[4] = gcl_rand(1, Bn * S * ds, {Bn, S, ds, 1}, 143, true, allocator) // C
		gcl_in[5] = gcl_rand(1, Bn * S * dm, {Bn, S, dm, 1}, 144, true, allocator) // Delta (positive)
		gco_remap(gcl_in[5], 0.1, 1.0)
		gcl_in[6] = gcl_rand(1, dm, {1, dm, 1, 1}, 145, true, allocator) // D
		gcl_check("ssm (all 7 inputs)", 7, proc() -> ^t.Tensor {
				return gcl_weighted_sum(
					t.tensor_ssm(
						gcl_in[0],
						gcl_in[1],
						gcl_in[2],
						gcl_in[3],
						gcl_in[4],
						gcl_in[5],
						gcl_in[6],
						context.allocator,
					),
				)
			}, allocator)
		gcl_free_inputs(7)
	}

	fmt.printf("Op gradcheck summary: %d/%d passed\n", gc_passed, gc_total)
	record_failures(gc_total - gc_passed)
}
