package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import optim "../wotan/optimize"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"

// CQL pieces against references: the loss and its gradient in Q vs PyTorch (CQL(H) with an
// MSE TD term, as in d3rlpy's DiscreteCQL), the gradient through the network vs finite
// differences, the trading dataset (no look-ahead: the state holds only past returns) vs
// numpy, and training on a one-state bandit to the minimiser found by scipy, where the
// regulariser turns the greedy action away from the rarely logged, high-reward action.
cql_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== CQL Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	mat :: proc(rows, cols: int, v: []f64, grad := false) -> ^t.Tensor {
		m := l.matrix_new(f64, rows, cols)
		copy(m.data, v)
		return t.tensor_new(m, grad)
	}

	{ 	// --- loss and dL/dQ vs PyTorch
		cfg := ml_fin.CQLConfig {
			num_actions = 3,
			gamma       = 0.9,
			alpha       = 0.7,
		}
		q := mat(
			5,
			3,
			{0.5, -0.2, 1.1, 0.0, 0.3, -0.7, 2.0, 1.5, 1.9, -1.0, -0.4, 0.2, 0.8, 0.8, -0.1},
			true,
		)
		nq := mat(
			5,
			3,
			{0.1, 0.9, -0.3, 1.2, 0.4, 0.0, -0.5, -0.6, -0.2, 0.3, 0.3, 0.7, 2.0, -1.0, 0.5},
		)
		r := mat(5, 1, {0.3, -0.1, 0.5, 0.0, 1.0})
		d := mat(5, 1, {0, 0, 1, 0, 1})
		acts := []int{2, 0, 1, 1, 0}
		scope: t.Graph_Scope
		defer t.graph_scope_destroy(&scope)
		t.graph_begin(&scope)
		loss, td, reg := ml_fin.cql_loss(q, acts, r, nq, d, cfg, allocator)
		t.tensor_backward(loss)
		lv := loss.data.data[0]
		t.graph_end(&scope)
		ref_grad := []f64 {
			0.04218514082758543,
			0.020948520973349902,
			-0.06713366180093534,
			-0.4828158268623989,
			0.06639168930313394,
			0.02442413755926494,
			0.055746507746291044,
			0.29381196612003335,
			0.050441526133675606,
			0.022793003445449826,
			-0.5104684399046803,
			0.0756754364592304,
			-0.16182590999045068,
			0.05817409000954928,
			0.023651819980901434,
		}
		err := max(
			abs(lv - 1.3346609212594642),
			abs(td - 0.6122799999999999),
			abs(reg - 1.0319727446563778),
		)
		for g, i in ref_grad {err = max(err, abs(q.grad.data[i] - g))}
		expect(
			&passed,
			&total,
			err < 1e-14,
			fmt.tprintf(
				"loss %.6f (TD %.5f + 0.7 x regulariser %.6f) and dL/dQ vs PyTorch: max err %.1e",
				lv,
				td,
				reg,
				err,
			),
		)
	}

	closes := []f64{100, 101, 99, 102, 103, 101, 104, 105}
	acts := []int{2, 2, 0, 1}
	states, rewards, next_states, dones := ml_fin.cql_trading_dataset(closes, acts, 0.001)

	{ 	// --- dataset vs numpy: decisions on days 3..6, the state only from closes up to the day
		S_ref := []f64 {
			3.0303030303030276,
			-1.980198019801982,
			1.0000000000000009,
			0.0,
			0.9803921568627416,
			3.0303030303030276,
			-1.980198019801982,
			1.0,
			-1.9417475728155331,
			0.9803921568627416,
			3.0303030303030276,
			1.0,
			2.970297029702973,
			-1.9417475728155331,
			0.9803921568627416,
			-1.0,
		}
		NS_ref := []f64 {
			0.9803921568627416,
			3.0303030303030276,
			-1.980198019801982,
			1.0,
			-1.9417475728155331,
			0.9803921568627416,
			3.0303030303030276,
			1.0,
			2.970297029702973,
			-1.9417475728155331,
			0.9803921568627416,
			-1.0,
			0.9615384615384581,
			2.970297029702973,
			-1.9417475728155331,
			0.0,
		}
		R_ref := []f64{0.8803921568627415, -1.9417475728155331, -3.170297029702973, -0.1}
		err := 0.0
		for v, i in S_ref {err = max(err, abs(states.data.data[i] - v))}
		for v, i in NS_ref {err = max(err, abs(next_states.data.data[i] - v))}
		for v, i in R_ref {err = max(err, abs(rewards.data.data[i] - v))}
		done_ok := dones.data.data[0] == 0 && dones.data.data[2] == 0 && dones.data.data[3] == 1
		expect(
			&passed,
			&total,
			err < 1e-13 && done_ok && states.data.rows == 4 && states.data.cols == 4,
			fmt.tprintf(
				"trading dataset vs numpy (past returns + held position; reward = next return - cost): max err %.1e, terminal last %v",
				err,
				done_ok,
			),
		)
	}

	{ 	// --- gradient through the network (target network fixed) vs central differences
		cfg := ml_fin.CQLConfig {
			state_dim   = 4,
			num_actions = 3,
			hidden_dim  = 6,
			gamma       = 0.95,
			alpha       = 0.5,
		}
		q_net := ml_fin.cql_network_new(cfg)
		defer ml_fin.cql_network_free(&q_net)
		target := ml_fin.cql_network_new(cfg)
		defer ml_fin.cql_network_free(&target)
		// keep the hidden units away from the ReLU kink for the differences
		for &b in q_net.fc1.bias.data.data {b = 0.5}
		scope: t.Graph_Scope
		defer t.graph_scope_destroy(&scope)
		objective :: proc(
			q_net, target: ^ml_fin.CQLNetwork,
			s, ns, r, d: ^t.Tensor,
			acts: []int,
			cfg: ml_fin.CQLConfig,
			scope: ^t.Graph_Scope,
			backward: bool,
		) -> f64 {
			t.graph_begin(scope, no_grad = !backward)
			defer t.graph_end(scope)
			loss, _, _ := ml_fin.cql_loss(
				ml_fin.cql_network_forward(q_net, s),
				acts,
				r,
				ml_fin.cql_network_forward(target, ns),
				d,
				cfg,
				context.allocator,
			)
			if backward {t.tensor_backward(loss)}
			return loss.data.data[0]
		}
		objective(&q_net, &target, states, next_states, rewards, dones, acts, cfg, &scope, true)
		params := []^t.Tensor{q_net.fc1.weights, q_net.fc1.bias, q_net.fc2.weights, q_net.fc2.bias}
		err, gmax := 0.0, 0.0
		h :: 1e-6
		for p in params {
			for j in 0 ..< len(p.data.data) {
				x0 := p.data.data[j]
				p.data.data[j] = x0 + h
				fp := objective(
					&q_net,
					&target,
					states,
					next_states,
					rewards,
					dones,
					acts,
					cfg,
					&scope,
					false,
				)
				p.data.data[j] = x0 - h
				fm := objective(
					&q_net,
					&target,
					states,
					next_states,
					rewards,
					dones,
					acts,
					cfg,
					&scope,
					false,
				)
				p.data.data[j] = x0
				fd := (fp - fm) / (2 * h)
				err = max(err, abs(fd - p.grad.data[j]))
				gmax = max(gmax, abs(fd))
			}
		}
		expect(
			&passed,
			&total,
			err < 1e-7 * max(gmax, 1) && gmax > 0,
			fmt.tprintf(
				"CQL loss gradient through the network vs finite differences: max err %.1e (largest grad %.2f)",
				err,
				gmax,
			),
		)
	}

	{ 	// --- one-state bandit (gamma 0): actions logged 6, 3, 1 times with rewards 1, 0, 2.
		// Minimiser (scipy): alpha 0 -> Q = mean rewards (1, 0, 2), greedy action 2;
		// alpha 1 -> Q = (1.108, 0.188, 0.789), greedy action 0 (the rare action is
		// pushed down).
		bandit :: proc(alpha: f64) -> (q: [3]f64) {
			cfg := ml_fin.CQLConfig {
				state_dim   = 1,
				num_actions = 3,
				hidden_dim  = 8,
				alpha       = alpha,
			}
			net := ml_fin.cql_network_new(cfg)
			defer ml_fin.cql_network_free(&net)
			acts := []int{0, 0, 0, 0, 0, 0, 1, 1, 1, 2}
			s := l.matrix_new(f64, 10, 1)
			r := l.matrix_new(f64, 10, 1)
			d := l.matrix_new(f64, 10, 1)
			for a, i in acts {
				s.data[i], d.data[i] = 1, 1
				r.data[i] = 1 if a == 0 else (0 if a == 1 else 2)
			}
			S, R, D := t.tensor_new(s), t.tensor_new(r), t.tensor_new(d)
			nq := t.tensor_new(l.matrix_new(f64, 10, 3)) // unused: every transition is terminal
			opt := nn.adam_new(0.01)
			defer nn.adam_free(&opt)
			ml_fin.cql_network_add_to_adam(&net, &opt)
			STEPS :: 6000
			schedule := optim.lr_step_decay(0.01, STEPS / 3, 0.1)
			scope: t.Graph_Scope
			defer t.graph_scope_destroy(&scope)
			for step in 0 ..< STEPS {
				t.graph_begin(&scope)
				nn.adam_zero_grad(&opt)
				loss, _, _ := ml_fin.cql_loss(
					ml_fin.cql_network_forward(&net, S),
					acts,
					R,
					nq,
					D,
					cfg,
					context.allocator,
				)
				t.tensor_backward(loss)
				nn.adam_step_scheduled(&opt, schedule, step)
				t.graph_end(&scope)
			}
			t.graph_begin(&scope, no_grad = true)
			out := ml_fin.cql_network_forward(&net, S)
			for a in 0 ..< 3 {q[a] = out.data.data[a]}
			t.graph_end(&scope)
			return
		}
		q0 := bandit(0)
		q1 := bandit(1)
		ref0 := [3]f64{1, 0, 2}
		ref1 := [3]f64{1.1079527593903897, 0.18762562449884998, 0.7894065701611124}
		err := 0.0
		for a in 0 ..< 3 {err = max(err, abs(q0[a] - ref0[a]), abs(q1[a] - ref1[a]))}
		argmax :: proc(q: [3]f64) -> int {
			b := 0
			for a in 1 ..< 3 {if q[a] > q[b] {b = a}}
			return b
		}
		expect(
			&passed,
			&total,
			err < 1e-8 && argmax(q0) == 2 && argmax(q1) == 0,
			fmt.tprintf(
				"bandit trained to the scipy minimiser: alpha 0 Q = (%.4f, %.4f, %.4f) greedy %d, alpha 1 Q = (%.4f, %.4f, %.4f) greedy %d; max err %.1e",
				q0[0],
				q0[1],
				q0[2],
				argmax(q0),
				q1[0],
				q1[1],
				q1[2],
				argmax(q1),
				err,
			),
		)
	}
	fmt.printf("CQL reference checks: %d/%d passed\n", passed, total)
}
