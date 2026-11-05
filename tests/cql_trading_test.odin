package tests

import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

// Trains CQL (full batch, soft-updated target network) on a log of a noisy momentum rule
// trading synthetic AR(1) returns, then backtests the greedy policy against the logged
// rule, in sample and on a fresh path of the same process.
cql_train :: proc(
	q_net: ^ml_fin.CQLNetwork,
	config: ml_fin.CQLConfig,
	states, rewards, next_states, dones: ^t.Tensor,
	actions: []int,
	steps: int,
	lr: f64,
	allocator: mem.Allocator,
) {
	target := ml_fin.cql_network_new(config, allocator)
	defer ml_fin.cql_network_free(&target)
	ml_fin.cql_network_copy(q_net, &target)
	opt := nn.adam_new(lr, allocator = allocator)
	defer nn.adam_free(&opt)
	ml_fin.cql_network_add_to_adam(q_net, &opt)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	fmt.println(" step | total loss | TD loss  | CQL reg")
	for step in 0 ..< steps {
		t.graph_begin(&scope)
		nn.adam_zero_grad(&opt)
		q_values := ml_fin.cql_network_forward(q_net, states)
		next_q := ml_fin.cql_network_forward(&target, next_states)
		loss, td, reg := ml_fin.cql_loss(
			q_values,
			actions,
			rewards,
			next_q,
			dones,
			config,
			allocator,
		)
		t.tensor_backward(loss)
		nn.adam_step(&opt)
		lv := loss.data.data[0]
		t.graph_end(&scope)
		ml_fin.cql_soft_update(q_net, &target, config.tau)
		if step % (steps / 5) == 0 || step == steps - 1 {
			fmt.printf(" %4d | %10.5f | %8.5f | %8.5f\n", step + 1, lv, td, reg)
		}
	}
}

cql_trading_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Offline CQL on a synthetic momentum log ===")
	config := ml_fin.CQLConfig {
		state_dim   = ml_fin.CQL_STATE_DIM,
		num_actions = 3,
		hidden_dim  = 32,
		gamma       = 0.9,
		alpha       = 1.0,
		tau         = 0.01,
	}
	num_steps := 1000
	states, actions, rewards, next_states, dones, closes := ml_fin.generate_trading_dataset(
		num_steps,
		allocator,
	)
	fmt.printf(
		"Log: %d transitions of a noisy momentum rule on AR(1) returns (phi 0.3), 10 bp cost\n",
		num_steps,
	)
	q_net := ml_fin.cql_network_new(config, allocator)
	defer ml_fin.cql_network_free(&q_net)
	cql_train(&q_net, config, states, rewards, next_states, dones, actions, 1500, 3e-3, allocator)

	logged := 0.0
	for r in rewards.data.data {logged += r}
	pnl, counts := ml_fin.cql_backtest(&q_net, closes, 0.001)
	fmt.printf(
		"In sample:     greedy CQL policy %+.1f%% (short %d, flat %d, long %d), logged rule %+.1f%%\n",
		pnl,
		counts[0],
		counts[1],
		counts[2],
		logged,
	)
	fresh_s, fresh_a, fresh_r, fresh_ns, fresh_d, fresh := ml_fin.generate_trading_dataset(
		num_steps,
		allocator,
	)
	logged = 0
	for r in fresh_r.data.data {logged += r}
	pnl, counts = ml_fin.cql_backtest(&q_net, fresh, 0.001)
	fmt.printf(
		"Fresh path:    greedy CQL policy %+.1f%% (short %d, flat %d, long %d), logged rule %+.1f%%\n",
		pnl,
		counts[0],
		counts[1],
		counts[2],
		logged,
	)
	_, _, _, _ = fresh_s, fresh_a, fresh_ns, fresh_d
	fmt.println("(summed daily P&L in percent of notional; the temp allocator frees the data)")
}
