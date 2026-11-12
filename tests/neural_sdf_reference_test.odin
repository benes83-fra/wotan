package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

// Neural SDF against numpy / scipy on 8 dates and 2 assets: the loss
// Σ_i (mean(M R_i) - 1)² + λ Var(M) and its gradient in M; the minimiser over every
// series M (neural_sdf_optimum, an (N+1) system) against the T x T Hessian system and
// BFGS; the network's parameter gradients against finite differences.

@(private = "file")
SDF_R := [8][2]f64 {
	{1.01, 0.98},
	{0.99, 1.03},
	{1.02, 1.00},
	{0.97, 0.96},
	{1.03, 1.05},
	{1.00, 0.99},
	{1.01, 1.04},
	{0.98, 1.01},
}
@(private = "file")
SDF_M := [8]f64{0.9, 1.1, 0.95, 1.2, 0.85, 1.0, 1.05, 0.97}
@(private = "file")
SDF_LAMBDA :: 0.05
@(private = "file")
SDF_LOSS_REF :: 0.0006362374999999984
@(private = "file")
SDF_LOSS0_REF :: 8.404999999999826e-05 // λ = 0
@(private = "file")
SDF_GRAD_REF := [8]f64 {
	0.0014547499999999732,
	0.0040549999999999736,
	0.002129749999999972,
	0.005138249999999973,
	0.0009964999999999712,
	0.002721499999999973,
	0.0034632499999999724,
	0.0023799999999999724,
}
@(private = "file")
SDF_OPT_REF := [8]f64 {
	0.9978873885969681,
	0.9935641198750348,
	0.9972747266951256,
	0.9966389745206008,
	0.9948125339514062,
	0.9966505196569392,
	0.9941883269132183,
	0.9941767817768851,
}
@(private = "file")
SDF_OPT_LOSS_REF :: 1.9241893902126993e-05

neural_sdf_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Neural SDF Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	R := t.tensor_new(l.matrix_new(f64, 8, 2), false)
	for row, i in SDF_R {R.data.data[2 * i], R.data.data[2 * i + 1] = row[0], row[1]}
	returns := R.data.data

	// 1. loss and dL/dM
	{
		scope: t.Graph_Scope
		defer t.graph_scope_destroy(&scope)
		t.graph_begin(&scope)
		alloc := t.graph_allocator(&scope)
		m := t.tensor_new(l.matrix_new(f64, 8, 1, alloc), true, alloc)
		copy(m.data.data, SDF_M[:])
		loss := ml_fin.neural_sdf_loss(m, R, SDF_LAMBDA, alloc)
		loss0 := ml_fin.neural_sdf_loss(m, R, 0, alloc).data.data[0]
		t.tensor_backward(loss)
		err_g := 0.0
		for v, i in SDF_GRAD_REF {err_g = max(err_g, abs(m.grad.data[i] - v))}
		err_l := abs(loss.data.data[0] - SDF_LOSS_REF)
		err_l0 := abs(loss0 - SDF_LOSS0_REF)
		t.graph_end(&scope)
		expect(
			&passed,
			&total,
			err_l < 1e-15 && err_l0 < 1e-15 && err_g < 1e-15,
			fmt.tprintf(
				"loss (λ = 0.05 and 0) and dL/dM vs numpy: errors %.1e, %.1e, gradient %.1e",
				err_l,
				err_l0,
				err_g,
			),
		)
	}

	// 2. the minimiser over every series
	{
		m, loss, min_m, ok := ml_fin.neural_sdf_optimum(returns, 2, SDF_LAMBDA, allocator)
		err := 0.0
		for v, i in SDF_OPT_REF {err = max(err, abs(m[i] - v))}
		// no nearby series does better
		better := 0
		for _ in 0 ..< 200 {
			p := make([]f64, 8, context.temp_allocator)
			for &v, i in p {v = m[i] + rand.float64_range(-1e-3, 1e-3)}
			errs := ml_fin.sdf_pricing_errors(p, returns, 2, context.temp_allocator)
			mean, sq := 0.0, 0.0
			for v in p {mean += v / 8; sq += v * v / 8}
			lp := errs[0] * errs[0] + errs[1] * errs[1] + SDF_LAMBDA * (sq - mean * mean)
			if lp < loss {better += 1}
		}
		expect(
			&passed,
			&total,
			ok &&
			err < 1e-12 &&
			abs(loss - SDF_OPT_LOSS_REF) < 1e-15 &&
			better == 0 &&
			min_m > 0.99,
			fmt.tprintf(
				"minimiser over all series vs numpy (T x T system, BFGS agrees to 2e-9): max err %.1e, loss %.6e (%.6e), 0/200 perturbations better (%d)",
				err,
				loss,
				SDF_OPT_LOSS_REF,
				better,
			),
		)
	}

	// 3. network gradients vs central finite differences
	{
		cfg := ml_fin.NeuralSDFConfig {
			input_size       = 3,
			hidden_size      = 5,
			num_layers       = 2,
			num_assets       = 2,
			epsilon          = 0.01,
			variance_penalty = SDF_LAMBDA,
		}
		model := ml_fin.neural_sdf_new(cfg)
		defer ml_fin.neural_sdf_free(model)
		opt := nn.adam_new(1e-3)
		defer nn.adam_free(&opt)
		nn.sequential_add_to_adam(model.network, &opt)
		// a generic point (the model starts with small output weights, near M = 1)
		for p in opt.parameters {for &v in p.data.data {v = rand.float64_range(-0.5, 0.5)}}
		z := t.tensor_new(l.matrix_new(f64, 8, 3), false)
		for &v in z.data.data {v = rand.float64_range(-1, 1)}
		value :: proc(model: ^ml_fin.NeuralSDF, z, R: ^t.Tensor) -> f64 {
			t.graph_begin(&model.scope, no_grad = true)
			defer t.graph_end(&model.scope)
			alloc := t.graph_allocator(&model.scope)
			return(
				ml_fin.neural_sdf_loss(ml_fin.neural_sdf_values(model, z, alloc), R, SDF_LAMBDA, alloc).data.data[0] \
			)
		}
		t.graph_begin(&model.scope)
		alloc := t.graph_allocator(&model.scope)
		nn.adam_zero_grad(&opt)
		t.tensor_backward(
			ml_fin.neural_sdf_loss(
				ml_fin.neural_sdf_values(model, z, alloc),
				R,
				SDF_LAMBDA,
				alloc,
			),
		)
		t.graph_end(&model.scope)
		max_err, checked := 0.0, 0
		h := 1e-6
		for p in opt.parameters {
			for i in 0 ..< len(p.data.data) {
				x0 := p.data.data[i]
				p.data.data[i] = x0 + h
				up := value(model, z, R)
				p.data.data[i] = x0 - h
				down := value(model, z, R)
				p.data.data[i] = x0
				fd := (up - down) / (2 * h)
				max_err = max(max_err, abs(fd - p.grad.data[i]) / max(1e-3, abs(fd)))
				checked += 1
			}
		}
		expect(
			&passed,
			&total,
			max_err < 1e-5,
			fmt.tprintf(
				"network gradient vs finite differences: %d parameters, max relative error %.1e",
				checked,
				max_err,
			),
		)
	}
	// 4. save / load / free under a tracking allocator: nothing live afterwards (the loaders
	// freed the checkpoint's optimizer contents but not the Adam struct itself)
	{
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, runtime.default_allocator())
		live: [2]int
		same: [2]bool
		{
			context.allocator = mem.tracking_allocator(&track)
			path := "model_checkpoints/neural_sdf_roundtrip.bin"
			defer os.remove(path)
			cfg := ml_fin.NeuralSDFConfig {
				input_size       = 3,
				hidden_size      = 4,
				num_layers       = 1,
				num_assets       = 2,
				epsilon          = 0.01,
				variance_penalty = SDF_LAMBDA,
			}
			model := ml_fin.neural_sdf_new(cfg)
			if !ml_fin.neural_sdf_save(model, path) {fmt.println("   neural_sdf_save failed")}
			loaded, ok := ml_fin.neural_sdf_load(path, cfg)
			if ok {
				same[0] = true
				for &layer, i in loaded.network.layers {
					a, _ := layer.(nn.LinearLayer)
					b, _ := model.network.layers[i].(nn.LinearLayer)
					if a.weights !=
					   nil {for v, j in a.weights.data.data {same[0] &&= v == b.weights.data.data[j]}}
				}
				ml_fin.neural_sdf_free(loaded)
			}
			ml_fin.neural_sdf_free(model)
			live[0] = len(track.allocation_map)

			hcfg := ml_fin.DeepHedgerConfig {
				state_size  = 3,
				hidden_size = 4,
				num_layers  = 1,
				num_assets  = 1,
			}
			hedger := ml_fin.deep_hedger_new(hcfg)
			if !ml_fin.deep_hedger_save(hedger, path) {fmt.println("   deep_hedger_save failed")}
			hl, hok := ml_fin.deep_hedger_load(path, context.allocator, hcfg)
			same[1] = hok
			if hok {ml_fin.deep_hedger_free(hl)}
			ml_fin.deep_hedger_free(hedger)
			live[1] = len(track.allocation_map) - live[0]
		}
		bad := len(track.bad_free_array)
		mem.tracking_allocator_destroy(&track)
		expect(
			&passed,
			&total,
			same[0] && same[1] && live == {0, 0} && bad == 0,
			fmt.tprintf(
				"save / load / free: weights reloaded %v, live allocations afterwards %v (neural SDF, deep hedger), bad frees %d",
				same,
				live,
				bad,
			),
		)
	}
	fmt.printf("Neural SDF reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
