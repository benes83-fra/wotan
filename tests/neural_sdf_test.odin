package tests

import ts "../wotan/analytics"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Neural SDF M_t = f(Z_t)² + ε trained on the Euler-equation loss
// Σ_i (mean(M R_i) - 1)² + λ Var(M), with the standardised returns as Z_t. The loss is
// convex in the series M and neural_sdf_optimum gives its minimiser over every series
// (affine in R_t): the network must close most of the gap between the best constant SDF
// and that optimum (95% on synthetic data, 80% on live data). The old test fixed λ = 1
// (the learnt SDF was then a constant), started the network at M ~ ε and had no benchmark.

// Trains on (z, r); returns the final loss
@(private = "file")
_sdf_train :: proc(model: ^ml_fin.NeuralSDF, z, r: ^t.Tensor, epochs: int, lr: f64) -> f64 {
	opt := nn.adam_new(lr)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(model.network, &opt)
	for epoch in 0 ..< epochs { 	// cosine decay of the learning rate to lr / 100
		opt.learning_rate =
			lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.PI * f64(epoch) / f64(epochs))))
		loss := ml_fin.neural_sdf_train_step(model, z, r, &opt)
		if epoch % (epochs / 4) == 0 {fmt.printf("   epoch %5d  loss %.6e\n", epoch, loss)}
	}
	t.graph_begin(&model.scope, no_grad = true)
	defer t.graph_end(&model.scope)
	alloc := t.graph_allocator(&model.scope)
	return(
		ml_fin.neural_sdf_loss(ml_fin.neural_sdf_values(model, z, alloc), r, model.config.variance_penalty, alloc).data.data[0] \
	)
}

// The loss of the best constant SDF c = Σ μ_i / Σ μ_i² (μ = mean gross returns)
@(private = "file")
_sdf_constant_loss :: proc(returns: []f64, N: int) -> (c, loss: f64) {
	n := len(returns) / N
	mu := make([]f64, N, context.temp_allocator)
	for row in 0 ..< n {for i in 0 ..< N {mu[i] += returns[row * N + i] / f64(n)}}
	s, s2 := 0.0, 0.0
	for v in mu {s += v; s2 += v * v}
	c = s / s2
	for v in mu {loss += (c * v - 1) * (c * v - 1)}
	return
}

// Standardised returns (training means and std) as the state variables, and R as a tensor
@(private = "file")
_sdf_inputs :: proc(returns: []f64, N: int, mean, std: []f64) -> (z, r: ^t.Tensor) {
	n := len(returns) / N
	r = t.tensor_new(l.matrix_new(f64, n, N), false)
	copy(r.data.data, returns)
	z = t.tensor_new(l.matrix_new(f64, n, N), false)
	for v, k in returns {z.data.data[k] = (v - mean[k % N]) / std[k % N]}
	return
}

@(private = "file")
_sdf_moments :: proc(returns: []f64, N: int) -> (mean, std: []f64) {
	n := len(returns) / N
	mean, std = make([]f64, N), make([]f64, N)
	for v, k in returns {mean[k % N] += v / f64(n)}
	for v, k in returns {std[k % N] += (v - mean[k % N]) * (v - mean[k % N]) / f64(n)}
	for &v in std {v = math.sqrt(v)}
	return
}

// Fits the network and the benchmarks; returns whether the network closed the gap
@(private = "file")
_sdf_fit_report :: proc(
	returns: []f64,
	N: int,
	lambda: f64,
	epochs: int,
	min_closed: f64,
	save_path := "",
) -> bool {
	mean, std := _sdf_moments(returns, N)
	z, r := _sdf_inputs(returns, N, mean, std)
	c, loss_c := _sdf_constant_loss(returns, N)
	m_opt, loss_opt, min_opt, ok := ml_fin.neural_sdf_optimum(
		returns,
		N,
		lambda,
		context.allocator,
	)
	if !ok {fmt.println("FAIL optimum: singular system"); return false}
	cfg := ml_fin.NeuralSDFConfig {
		input_size       = N,
		hidden_size      = 32,
		num_layers       = 2,
		num_assets       = N,
		epsilon          = 0.01,
		variance_penalty = lambda,
	}
	model := ml_fin.neural_sdf_new(cfg)
	defer ml_fin.neural_sdf_free(model)
	loss_net := _sdf_train(model, z, r, epochs, 1e-3)
	m_net := ml_fin.neural_sdf_series(model, z, context.allocator)
	err_net := ml_fin.sdf_pricing_errors(m_net, returns, N, context.allocator)
	err_opt := ml_fin.sdf_pricing_errors(m_opt, returns, N, context.allocator)
	dist := 0.0
	for v, i in m_net {dist = max(dist, abs(v - m_opt[i]))}
	closed := (loss_c - loss_net) / (loss_c - loss_opt)
	fmt.printf("   best constant SDF %.4f: loss %.6e\n", c, loss_c)
	fmt.printf(
		"   optimum over all series:   loss %.6e (min M %.3f), pricing errors %.2e\n",
		loss_opt,
		min_opt,
		err_opt,
	)
	fmt.printf("   neural SDF:                loss %.6e, pricing errors %.2e\n", loss_net, err_net)
	fmt.printf("   max |M_net - M_opt| %.4f\n", dist)
	ok = closed > min_closed
	fmt.printf(
		"%s the network closes %.1f%% of the loss gap between the constant SDF and the optimum (at least %.0f%%)\n",
		"OK  " if ok else "FAIL",
		100 * closed,
		100 * min_closed,
	)
	if save_path != "" {
		saved := ml_fin.neural_sdf_save(model, save_path)
		loaded, load_ok := ml_fin.neural_sdf_load(save_path, cfg)
		same := saved && load_ok
		if load_ok {
			m2 := ml_fin.neural_sdf_series(loaded, z, context.allocator)
			for v, i in m2 {same &&= v == m_net[i]}
			ml_fin.neural_sdf_free(loaded)
		}
		fmt.printf(
			"%s saved to %s and reloaded: the same SDF series\n",
			"OK  " if same else "FAIL",
			save_path,
		)
		ok &&= same
	}
	return ok
}

// A synthetic one-factor market: 600 monthly-like returns of 3 assets with means 0.5%,
// 1%, 1.5% and betas 0.5, 1, 1.5 (λ = 1e-3, where the optimum halves the constant's loss)
neural_sdf_synthetic_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Neural SDF on a synthetic one-factor market ===")
	T, N :: 600, 3
	mu := [N]f64{0.005, 0.010, 0.015}
	beta := [N]f64{0.5, 1.0, 1.5}
	returns := make([]f64, T * N)
	for row in 0 ..< T {
		f := rand.float64_normal(0, 0.04)
		for i in 0 ..< N {returns[row * N + i] = 1 + mu[i] + beta[i] * f + rand.float64_normal(0, 0.02)}
	}
	ok := _sdf_fit_report(returns, N, 1e-3, 2000, 0.95)
	record_failures(int(!ok))
}

// SPY, QQQ, IWM, TLT daily gross returns (AdjClose, the last common two years): fitted on
// the first year, the pricing errors of the second year reported next to the constant
// SDF's (in sample the fit is a statement about the optimiser only)
neural_sdf_test :: proc(allocator: mem.Allocator = context.allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Neural SDF on SPY, QQQ, IWM, TLT daily returns ===")
	symbols := []string{"SPY", "QQQ", "IWM", "TLT"}
	N := len(symbols)
	// the four symbols matched by date (only the days all of them have)
	panel, panel_ok, failed := net.read_yahoo_panel(symbols, .Daily, .TwoYears, allocator)
	defer net.yahoo_panel_free(&panel)
	if !panel_ok {
		what := "Yahoo data shared by all symbols"
		if failed >= 0 {what = fmt.tprintf("Yahoo data for %s", symbols[failed])}
		fail_missing_data(#procedure, what)
		return
	}
	n := len(panel.dates)
	if n < 200 {
		fail_missing_data(#procedure, fmt.tprintf("200 days of Yahoo data (got %d)", n))
		return
	}
	// gross returns of the n common days, [day][symbol]
	returns := ts.simple_returns(panel.adj_close, N, allocator)
	defer delete(returns, allocator)
	for &r in returns {r += 1}
	n_train := (n - 1) / 2
	train, test := returns[:n_train * N], returns[n_train * N:]
	fmt.printf("%d daily returns: %d to fit, %d held out\n", n - 1, n_train, (n - 1) - n_train)
	LAMBDA :: 1e-5 // daily: squared pricing errors ~1e-7, Var(M) of a pricing SDF ~1e-2
	ok := _sdf_fit_report(train, N, LAMBDA, 20000, 0.8, "model_checkpoints/neural_sdf_model.bin")

	// held out: the SDF learnt as a function of the returns, applied to the second year
	mean, std := _sdf_moments(train, N)
	z_test, _ := _sdf_inputs(test, N, mean, std)
	z_train, r_train := _sdf_inputs(train, N, mean, std)
	cfg := ml_fin.NeuralSDFConfig {
		input_size       = N,
		hidden_size      = 32,
		num_layers       = 2,
		num_assets       = N,
		epsilon          = 0.01,
		variance_penalty = LAMBDA,
	}
	loaded, load_ok := ml_fin.neural_sdf_load("model_checkpoints/neural_sdf_model.bin", cfg)
	if load_ok {
		defer ml_fin.neural_sdf_free(loaded)
		c, _ := _sdf_constant_loss(train, N)
		const_m := make([]f64, len(test) / N)
		for &v in const_m {v = c}
		e_const := ml_fin.sdf_pricing_errors(const_m, test, N, allocator)
		e_net := ml_fin.sdf_pricing_errors(
			ml_fin.neural_sdf_series(loaded, z_test, allocator),
			test,
			N,
			allocator,
		)
		e_in := ml_fin.neural_sdf_evaluate(loaded, z_train, r_train, allocator)
		fmt.printf("   %v\n", symbols)
		fmt.printf("   pricing errors in sample (neural):  %.2e\n", e_in)
		fmt.printf("   held out, constant SDF:             %.2e\n", e_const)
		fmt.printf("   held out, neural SDF:               %.2e\n", e_net)
		fmt.println(
			"   (a year of daily returns pins each mean return down to about 1% / sqrt(250) = 6e-4",
		)
		fmt.println(
			"    a day: the in-sample fit prices that noise, and held out it does no better than",
		)
		fmt.println("    the constant)")
	}
	record_failures(int(!ok))
}
