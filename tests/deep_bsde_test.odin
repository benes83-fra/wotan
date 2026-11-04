package tests

import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import optim "../wotan/optimize"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// Deep BSDE prices of calls on the maximum of d independent GBMs (S_0 = K = 100, r = 5%,
// sigma = 20%, T = 1) against the exact value
//   e^{-rT} int_K^inf (1 - F(x)^d) dx,  F the log-normal CDF of one asset
// (scipy.integrate.quad; d = 1 is Black-Scholes, 10.4506; Monte Carlo with 200k paths
// agrees within its standard error). The loss is the variance left after the learned
// hedge Z; without a hedge it would be Var[payoff], so loss / Var[payoff] shows how much
// of the risk Z replicates. In 100 dimensions this plain network (Han et al. use one
// subnetwork per time step with batch normalisation) learns almost no hedge; Y_0 is still
// right, as the mean of Y_T over 5000 batches.

@(private = "file")
BSDE_Case :: struct {
	name:                  string,
	d, N, batch, epochs:   int,
	hidden, layers:        int,
	y0_init:               f64,
	reference, payoff_var: f64,
	tol:                   f64, // relative tolerance on Y_0
	net_lr:                f64, // 0: 1e-3
	max_loss_ratio:        f64, // loss / Var[payoff] allowed (1: the hedge is not checked)
}

@(private = "file")
_bsde_run :: proc(c: BSDE_Case, allocator: mem.Allocator) -> bool {
	S0, K, r, s, T :: 100.0, 100.0, 0.05, 0.2, 1.0
	S_0 := make([]f64, c.d, allocator)
	sigma := make([]f64, c.d, allocator)
	defer {delete(S_0, allocator); delete(sigma, allocator)}
	for i in 0 ..< c.d {S_0[i], sigma[i] = S0, s}

	model := ml_fin.deep_bsde_model_new(
		c.y0_init,
		c.d,
		T,
		c.N,
		c.hidden,
		c.layers,
		allocator = allocator,
	)
	defer ml_fin.deep_bsde_model_free(&model)
	net_lr := c.net_lr if c.net_lr > 0 else 1e-3
	opt := nn.adam_new(net_lr, allocator = allocator)
	defer nn.adam_free(&opt)
	nn.adam_add_param_with_lr(&opt, model.y0, 0.1) // Y_0 moves ~ lr per step: 100x the network's
	nn.sequential_add_to_adam(model.z_net, &opt)
	schedule := optim.lr_step_decay(net_lr, c.epochs / 3, 0.2) // both rates x0.2 at 1/3 and 2/3

	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	tail := 200 // loss averaged over the last epochs
	loss_tail := 0.0
	for epoch in 0 ..< c.epochs {
		t.graph_begin(&scope)
		nn.adam_zero_grad(&opt)
		loss := ml_fin.deep_bsde_bs_loss(&model, S_0, K, r, sigma, c.batch, c.N, allocator)
		t.tensor_backward(loss)
		nn.adam_step_scheduled(&opt, schedule, epoch)
		lv := loss.data.data[0]
		t.graph_end(&scope)
		if epoch >= c.epochs - tail {loss_tail += lv / f64(tail)}
		if epoch % (c.epochs / 5) == 0 {
			fmt.printf(
				"  epoch %5d | loss %9.4f | Y_0 %.4f\n",
				epoch,
				lv,
				ml_fin.deep_bsde_y0(&model),
			)
		}
	}
	y0 := ml_fin.deep_bsde_y0(&model)
	err := abs(y0 - c.reference) / c.reference
	ratio := loss_tail / c.payoff_var
	ok := err < c.tol && ratio < c.max_loss_ratio
	fmt.printf(
		"%s %s: Y_0 %.4f vs exact %.4f (%.2f%%, tolerance %.1f%%); loss %.2f = %.1f%% of Var[payoff] %.1f%s\n",
		"OK  " if ok else "FAIL",
		c.name,
		y0,
		c.reference,
		100 * err,
		100 * c.tol,
		loss_tail,
		100 * ratio,
		c.payoff_var,
		c.max_loss_ratio < 1 ? fmt.tprintf(" (the hedge Z removes the rest; at most %.0f%% allowed)", 100 * c.max_loss_ratio) : " (Z learns almost no hedge here: Y_0 is in effect a Monte Carlo mean)",
	)
	return ok
}

deep_bsde_european_call_test :: proc(allocator: mem.Allocator = context.allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Deep BSDE: 1D European call vs Black-Scholes ===")
	_bsde_run(
		{
			name = "1D European call",
			d = 1,
			N = 20,
			batch = 64,
			epochs = 3000,
			hidden = 32,
			layers = 2,
			y0_init = 5.0,
			reference = 10.450583572185565,
			payoff_var = 239.44727801854623,
			tol = 0.01,
			max_loss_ratio = 0.05,
		},
		allocator,
	)
}

deep_bsde_test :: proc(allocator: mem.Allocator = context.allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Deep BSDE: 10D max-call vs the exact integral ===")
	_bsde_run(
		{
			name = "10D max-call",
			d = 10,
			N = 20,
			batch = 64,
			epochs = 4000,
			hidden = 128,
			layers = 2,
			y0_init = 20.0,
			reference = 39.15672219310201,
			payoff_var = 290.4870969006911,
			tol = 0.01,
			net_lr = 3e-3,
			max_loss_ratio = 0.2,
		},
		allocator,
	)
}

deep_bsde_100d_test :: proc(allocator: mem.Allocator = context.allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Deep BSDE: 100D max-call vs the exact integral ===")
	_bsde_run(
		{
			name = "100D max-call",
			d = 100,
			N = 20,
			batch = 64,
			epochs = 5000,
			hidden = 64,
			layers = 3,
			y0_init = 40.0,
			reference = 67.33968211811577,
			payoff_var = 228.91318140790008,
			tol = 0.01,
			max_loss_ratio = 1,
		},
		allocator,
	)
}
