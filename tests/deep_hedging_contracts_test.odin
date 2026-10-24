package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"

// Deep hedging memory and correctness contracts (small, deterministic data):
// - deep_hedging_evaluate equals a plain step-by-step reference bit for bit
// - training steps (single asset; multi asset with CVaR and transaction costs)
//   allocate nothing from the hedger's allocator once warmed up, and the
//   hedger's graph scope reuses its memory
// - the CVaR loss gradient matches finite differences, and freeing its graph
//   (with a real allocator, no scope) releases everything it allocated

@(private = "file")
_paths :: proc(n_paths, n_steps, state_size: int, seed: u64, allocator: mem.Allocator) -> (paths, payoffs: ^t.Tensor) {
	pd := l.matrix_new(f64, n_paths * (n_steps + 1), state_size, allocator)
	gcl_fill(pd.data, seed)
	// positive "spot" columns around 100, other features in [-1, 1]
	for i in 0 ..< n_paths * (n_steps + 1) {
		for a in 0 ..< state_size {
			if a < state_size - 1 {pd.data[i * state_size + a] = 100 + 5 * pd.data[i * state_size + a]}
		}
	}
	paths = t.tensor_new(pd, false, allocator)
	paths.shape = {n_paths, n_steps + 1, state_size, 1}
	yd := l.matrix_new(f64, n_paths, 1, allocator)
	gcl_fill(yd.data, seed + 1)
	payoffs = t.tensor_new(yd, false, allocator)
	payoffs.shape = {n_paths, 1, 1, 1}
	return
}

// Reference for deep_hedging_evaluate: forward the network at every step, no scope
@(private = "file")
_evaluate_ref :: proc(h: ^ml_fin.DeepHedger, paths, payoffs: ^t.Tensor, allocator: mem.Allocator) -> []f64 {
	B, S, F := paths.shape[0], paths.shape[1], paths.shape[2]
	pnl := make([]f64, B, allocator)
	prev := make([]f64, B, allocator)
	defer delete(prev, allocator)
	for s in 0 ..< S {
		sd := l.matrix_new(f64, B, F, allocator)
		for b in 0 ..< B {
			for f in 0 ..< F {sd.data[b * F + f] = paths.data.data[b * (S * F) + s * F + f]}
		}
		state := t.tensor_new(sd, false, allocator)
		delta := nn.sequential_forward(h.network, state)
		if s > 0 {
			for b in 0 ..< B {
				s_t := paths.data.data[b * (S * F) + s * F]
				s_prev := paths.data.data[b * (S * F) + (s - 1) * F]
				inc := delta.data.data[b] * (s_t - s_prev)
				if h.config.transaction_cost > 0 {
					inc -= h.config.transaction_cost * math.abs(delta.data.data[b] - prev[b]) * s_t
				}
				pnl[b] += inc
			}
		}
		for b in 0 ..< B {prev[b] = delta.data.data[b]}
		t.tensor_free_graph(delta)
		t.tensor_free(state)
	}
	for b in 0 ..< B {pnl[b] -= payoffs.data.data[b]}
	return pnl
}

deep_hedging_contracts_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Deep Hedging Contracts ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// The hedgers (network, optimizer) live on a tracking allocator so their
	// allocations during training can be counted
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, allocator)
	defer mem.tracking_allocator_destroy(&track)
	ha := mem.tracking_allocator(&track)

	// --- Single asset: evaluate vs reference, training steps -------------------
	{
		cfg := ml_fin.DeepHedgerConfig {
			state_size       = 3,
			hidden_size      = 8,
			num_layers       = 2,
			risk_measure     = .Variance,
			transaction_cost = 0.001,
			num_assets       = 1,
		}
		context.allocator = ha
		h := ml_fin.deep_hedger_new(cfg, ha)
		opt := nn.adam_new(0.001, allocator = ha)
		nn.sequential_add_to_adam(h.network, &opt)
		context.allocator = allocator
		defer {
			nn.adam_free(&opt)
			ml_fin.deep_hedger_free(h)
		}
		paths, payoffs := _paths(16, 5, 3, 901, allocator)

		pnl, _ := ml_fin.deep_hedging_evaluate(h, paths, payoffs, allocator)
		ref := _evaluate_ref(h, paths, payoffs, allocator)
		same := len(pnl) == len(ref)
		for i in 0 ..< len(pnl) {same &&= transmute(u64)pnl[i] == transmute(u64)ref[i]}
		expect(&passed, &total, same, "deep_hedging_evaluate equals the step-by-step reference")

		loss0 := ml_fin.deep_hedger_train_step(h, paths, payoffs, &opt) // warm-up: gradients, Adam state
		allocs := track.total_allocation_count
		_, reserved1 := t.graph_scope_bytes(&h.scope)
		loss1 := ml_fin.deep_hedger_train_step(h, paths, payoffs, &opt)
		loss2 := ml_fin.deep_hedger_train_step(h, paths, payoffs, &opt)
		_, reserved2 := t.graph_scope_bytes(&h.scope)
		finite := !math.is_nan(loss0) && !math.is_nan(loss1) && !math.is_nan(loss2)
		expect(&passed, &total, finite && track.total_allocation_count == allocs, fmt.tprintf("single-asset training steps allocate nothing from the hedger's allocator (%d)", track.total_allocation_count - allocs))
		expect(&passed, &total, reserved1 > 0 && reserved2 == reserved1, "single-asset training steps reuse the hedger's scope")
	}

	// --- Multi asset with CVaR and transaction costs ----------------------------
	{
		cfg := ml_fin.DeepHedgerConfig {
			state_size       = 4,
			hidden_size      = 8,
			num_layers       = 2,
			risk_measure     = .CVaR,
			cvar_alpha       = 0.1,
			transaction_cost = 0.001,
			num_assets       = 2,
		}
		context.allocator = ha
		h := ml_fin.deep_hedger_new(cfg, ha)
		opt := nn.adam_new(0.001, allocator = ha)
		nn.sequential_add_to_adam(h.network, &opt)
		context.allocator = allocator
		defer {
			nn.adam_free(&opt)
			ml_fin.deep_hedger_free(h)
		}
		paths, payoffs := _paths(16, 5, 4, 911, allocator)

		loss0 := ml_fin.deep_hedger_train_step_multi(h, paths, payoffs, &opt)
		allocs := track.total_allocation_count
		loss1 := ml_fin.deep_hedger_train_step_multi(h, paths, payoffs, &opt)
		loss2 := ml_fin.deep_hedger_train_step_multi(h, paths, payoffs, &opt)
		finite := !math.is_nan(loss0) && !math.is_nan(loss1) && !math.is_nan(loss2)
		expect(&passed, &total, finite && track.total_allocation_count == allocs, fmt.tprintf("multi-asset CVaR training steps allocate nothing from the hedger's allocator (%d)", track.total_allocation_count - allocs))
	}

	// --- CVaR loss: gradient and cleanup without a scope --------------------------
	{
		// Backed by the heap (frees are real), so leaks and bad frees are visible
		ct: mem.Tracking_Allocator
		mem.tracking_allocator_init(&ct, runtime.heap_allocator())
		defer mem.tracking_allocator_destroy(&ct)
		ca := mem.tracking_allocator(&ct)
		context.allocator = ca

		N := 12
		pd := l.matrix_new(f64, N, 1, ca)
		gcl_fill(pd.data, 921)
		for i in 0 ..< N {pd.data[i] *= 3}
		pnl := t.tensor_new(pd, true, ca)
		pnl.shape = {N, 1, 1, 1}

		loss := ml_fin.compute_hedging_loss(pnl, .CVaR, 0.2, ca)
		t.tensor_backward(loss)
		grad := make([]f64, N, allocator)
		copy(grad, pnl.grad.data)
		t.tensor_free_graph(loss)

		value :: proc(x: ^t.Tensor, alloc: mem.Allocator) -> f64 {
			loss := ml_fin.compute_hedging_loss(x, .CVaR, 0.2, alloc)
			v := loss.data.data[0]
			t.tensor_free_graph(loss)
			return v
		}
		max_err := 0.0
		h := 1e-6
		for i in 0 ..< N {
			x0 := pnl.data.data[i]
			pnl.data.data[i] = x0 + h
			up := value(pnl, ca)
			pnl.data.data[i] = x0 - h
			down := value(pnl, ca)
			pnl.data.data[i] = x0
			max_err = max(max_err, abs((up - down) / (2 * h) - grad[i]))
		}
		t.tensor_free(pnl)
		context.allocator = allocator
		expect(&passed, &total, max_err < 1e-6, fmt.tprintf("CVaR loss gradient matches finite differences (max err %.1e)", max_err))
		expect(&passed, &total, len(ct.allocation_map) == 0 && len(ct.bad_free_array) == 0, fmt.tprintf("CVaR loss graph frees everything it allocated (%d leaked, %d bad frees)", len(ct.allocation_map), len(ct.bad_free_array)))
	}

	fmt.printf("Deep hedging contracts: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
