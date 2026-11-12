package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:mem"

// PPO pieces against references: GAE vs CleanRL's recursion (numpy), the minibatch loss
// gradient vs finite differences of the PPO objective (clipped and unclipped samples,
// value and entropy terms), the rollout buffer growing, and the trading environment on a
// scripted episode (observation layout, timing, average entry, closing at the end).
ppo_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== PPO Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	{ 	// --- GAE with two episode ends inside the rollout and a bootstrap value
		r := []f64{0.5, -0.2, 1.0, 0.3, 0.0, -1.0, 0.7}
		v := []f64{0.1, 0.4, -0.3, 0.2, 0.5, 0.0, -0.2}
		d := []bool{false, false, true, false, false, true, false}
		ref := []f64{1.0009247000000001, 0.2439, 1.3, -0.613629, -1.373, -1.0, 1.773}
		adv := ml_fin.compute_gae(r, v, d, 0.97, 0.9, allocator, 0.9)
		defer delete(adv)
		err := 0.0
		for x, i in adv {err = max(err, abs(x - ref[i]))}
		expect(
			&passed,
			&total,
			err < 1e-14,
			fmt.tprintf("GAE vs CleanRL (episode ends, bootstrap V = 0.9): max err %.1e", err),
		)
	}

	{ 	// --- minibatch loss: autograd gradient vs central differences of the objective
		agent := ml_fin.new_ppo_agent(
			4,
			3,
			5,
			value_coef = 0.5,
			entropy_coef = 0.05,
			alloc = allocator,
		)
		defer ml_fin.ppo_agent_free(agent)
		state: u64 = 51
		M :: 6
		actions := []int{0, 2, 1, 1, 0, 2}
		offsets := []f64{0.0, 0.5, -0.5, 0.1, -0.1, 0.3} // ratio = exp(-offset)
		adv := []f64{1.0, -0.8, 0.6, -0.4, 1.2, -1.0}
		ret := []f64{0.3, -0.2, 0.5, 0.1, -0.4, 0.2}
		for i in 0 ..< M {
			x := t.tensor_new(l.matrix_new(f64, 1, 4, allocator), false, allocator)
			x.shape = {1, 4, 1, 1}
			for &e in x.data.data {e = garch_ref_normal(&state)}
			t.graph_begin(&agent.scope, no_grad = true)
			p := t.tensor_softmax(nn.sequential_forward(agent.actor, x)).data.data[actions[i]]
			t.graph_end(&agent.scope)
			ml_fin.rollout_buffer_add(
				agent.buffer,
				x,
				actions[i],
				math.ln(p) + offsets[i],
				0,
				0,
				false,
			)
		}
		rows := []int{0, 1, 2, 3, 4, 5}
		objective :: proc(agent: ^ml_fin.PPOAgent, rows: []int, adv, ret: []f64) -> f64 {
			t.graph_begin(&agent.scope)
			defer t.graph_end(&agent.scope)
			_, obj, _ := ml_fin.ppo_minibatch_loss(agent, rows, adv, ret)
			return obj
		}
		nn.adam_zero_grad(&agent.optimizer)
		t.graph_begin(&agent.scope)
		loss, obj, _ := ml_fin.ppo_minibatch_loss(agent, rows, adv, ret)
		t.tensor_backward(loss)
		t.graph_end(&agent.scope)

		params := nn.sequential_parameters(agent.actor)
		defer delete(params)
		cp := nn.sequential_parameters(agent.critic)
		defer delete(cp)
		append(&params, ..cp[:])
		err, gmax := 0.0, 0.0
		h :: 1e-6
		for p in params {
			for j in 0 ..< len(p.data.data) {
				x0 := p.data.data[j]
				p.data.data[j] = x0 + h
				fp := objective(agent, rows, adv, ret)
				p.data.data[j] = x0 - h
				fm := objective(agent, rows, adv, ret)
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
				"PPO loss gradient (3 of 6 ratios outside the clip band, value_coef 0.5, entropy 0.05) vs finite differences: max err %.1e (largest grad %.2f, objective %.4f)",
				err,
				gmax,
				obj,
			),
		)
	}

	{ 	// --- the buffer grows instead of dropping what it holds
		buf := ml_fin.new_rollout_buffer(2, allocator)
		defer ml_fin.rollout_buffer_free(buf)
		for i in 0 ..< 5 {ml_fin.rollout_buffer_add(buf, nil, i, 0, f64(i), 0, false)}
		ok := buf.size == 5 && buf.capacity >= 5
		for i in 0 ..< 5 {ok = ok && buf.actions[i] == i && buf.rewards[i] == f64(i)}
		expect(
			&passed,
			&total,
			ok,
			"rollout buffer grows past its capacity, keeping every transition (it was cleared)",
		)
	}

	{ 	// --- trading environment on 13 days, window 3, 2 indicators
		N :: 13
		prices := make([]f64, N)
		volumes := make([]f64, N)
		ind := make([]f64, 2 * N)
		defer {delete(prices); delete(volumes); delete(ind)}
		for d in 0 ..< N {
			prices[d] = 100 + f64(d)
			volumes[d] = 1000
			ind[2 * d], ind[2 * d + 1] = 0.01 * f64(d), -0.01 * f64(d)
		}
		env := ml_fin.new_trading_env(prices, volumes, ind, 2, 3, 0.001, 20, allocator)
		defer ml_fin.trading_env_free(env)
		obs := ml_fin.env_reset(&env.env)
		// window days 0..2: prices, volumes, indicator 0, indicator 1
		want := []f64{0, 0.01, 0.02, 0, 0, 0, 0, 0.01, 0.02, 0, -0.01, -0.02}
		obs_ok := len(obs.data) == 12
		for v, i in want {obs_ok = obs_ok && abs(obs.data[i] - v) < 1e-15}
		delete(obs.data)
		// hold 4 steps (cooldown), buy on day 7, hold, buy again on day 12 (the last day)
		script := []int{0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		steps, done_at := 0, -1
		entry := 0.0
		for a, i in script {
			s := ml_fin.env_step(&env.env, a)
			delete(s.observation.data)
			steps += 1
			if i == 4 {entry = env.entry_price}
			if s.done && done_at < 0 {done_at = env.env.current_step}
		}
		cash := 100000.0 - 10 * 107 * 1.001 - 10 * 112 * 1.001 + 20 * 112 * 0.999
		ok :=
			obs_ok &&
			entry == 107 &&
			abs(env.entry_price - 109.5) < 1e-12 &&
			done_at == 12 &&
			env.inventory == 0 &&
			abs(env.cash - cash) < 1e-9
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"trading env: observation has both indicators %v, buy at day 7 = %.0f, average entry %.1f (109.5), done on day %d (12), closed at the last price: cash %.2f (%.2f)",
				obs_ok,
				entry,
				env.entry_price,
				done_at,
				env.cash,
				cash,
			),
		)
	}
	{ 	// --- multi-asset environment: 8 days, 2 assets, window 2, scripted episode replayed
		// by an independent numpy implementation of the documented rules
		D :: 8
		P := [D][2]f64 {
			{100, 50},
			{101, 49},
			{102, 48},
			{103, 50},
			{104, 52},
			{105, 51},
			{106, 53},
			{107, 55},
		}
		prices := make([]f64, 2 * D)
		volumes := make([]f64, 2 * D)
		ind := make([]f64, 2 * D)
		defer {delete(prices); delete(volumes); delete(ind)}
		for d in 0 ..< D {
			for a in 0 ..< 2 {
				prices[d * 2 + a] = P[d][a]
				volumes[d * 2 + a] = 1000
				ind[d * 2 + a] = 0.01 * f64(d + 10 * a)
			}
		}
		w0 := []f64{0, 0}
		w1 := []f64{1, 0}
		w2 := []f64{0.5, 0.5}
		acts := []ml_fin.ActionDef{{w0}, {w1}, {w2}}
		env := ml_fin.new_multi_asset_env(prices, volumes, ind, 2, 1, 2, acts, 0.001, allocator)
		defer ml_fin.multi_asset_env_free(env)
		obs := ml_fin.env_reset(&env.env)
		// per asset, per day of the window (days 0, 1): price / first - 1, volume / first - 1, indicator; then weights, cash
		want := []f64{0, 0, 0, 0.01, 0, 0.01, 0, 0, 0.1, -0.02, 0, 0.11, 0, 0, 1}
		obs_ok := len(obs.data) == len(want)
		for v, i in want {obs_ok = obs_ok && abs(obs.data[i] - v) < 1e-15}
		delete(obs.data)
		// the volatility penalty weighted by each asset's share of the portfolio value
		ref := []f64 {
			-0.0064803277525049855,
			-0.0014386938872473882,
			0.012410301491295864,
			-0.017211203935942873,
			0.015335722147566202,
			0.0238480627648493,
		}
		script := []int{1, 2, 2, 2, 2, 2}
		err := 0.0
		done_steps := 0
		for a, i in script {
			st := ml_fin.env_step(&env.env, a)
			delete(st.observation.data)
			err = max(err, abs(st.reward - ref[i]))
			if st.done {done_steps += 1}
		}
		flat := env.positions[0] == 0 && env.positions[1] == 0
		ok :=
			obs_ok &&
			err < 1e-15 &&
			done_steps == 1 &&
			flat &&
			abs(env.cash - 107664.58868252001) < 1e-8
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"multi-asset env: observation layout %v, 6 step rewards (market moves of the held positions included; volatility penalty weighted by position) max err %.1e, last day traded then sold: cash %.2f (107664.59)",
				obs_ok,
				err,
				env.cash,
			),
		)
	}
	fmt.printf("PPO reference checks: %d/%d passed\n", passed, total)
}
