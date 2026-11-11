package tests

import w "../wotan/core"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import "core:fmt"
import "core:mem"

// Offline CQL on AAPL (Yahoo, one year of adjusted closes): the log is a 3-day momentum
// rule (long above +1%, short below -1%, else flat, 10 bp cost). The agent trains on the
// first 70% of the days and is backtested against the rule on the last 30%.
cql_real_data_trading_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Yahoo Finance Offline CQL Trading Test ===")
	df, df_ok := net.read_yahoo("AAPL", .Daily, .OneYear, allocator)
	if !df_ok {fail_missing_data(#procedure, "Yahoo data for AAPL"); return}
	defer w.destroy_dataframe(&df)
	if df.rows < 60 {
		fail_missing_data(
			#procedure,
			fmt.tprintf("60 days of Yahoo data for AAPL (got %d)", df.rows),
		)
		return
	}
	col := w.column(&df, "AdjClose")
	closes := make([dynamic]f64, 0, df.rows, allocator)
	for i in 0 ..< df.rows {
		c, is_null := w.column_at_float(col, i)
		if !is_null && c > 0 {append(&closes, c)}
	}
	fmt.printf("Loaded %d adjusted closes\n", len(closes))
	if len(closes) < 60 {return}

	momentum_rule :: proc(closes: []f64, allocator: mem.Allocator) -> []int {
		acts := make([]int, len(closes) - 1 - ml_fin.CQL_LAGS, allocator)
		for &a, k in acts {
			i := ml_fin.CQL_LAGS + k
			m := closes[i] / closes[i - 3] - 1
			a = 2 if m > 0.01 else (0 if m < -0.01 else 1)
		}
		return acts
	}
	split := len(closes) * 7 / 10
	train := closes[:split]
	test := closes[split - ml_fin.CQL_LAGS:] // the first test decision needs CQL_LAGS earlier closes

	actions := momentum_rule(train, allocator)
	states, rewards, next_states, dones := ml_fin.cql_trading_dataset(
		train,
		actions,
		0.001,
		allocator,
	)
	config := ml_fin.CQLConfig {
		state_dim   = ml_fin.CQL_STATE_DIM,
		num_actions = 3,
		hidden_dim  = 32,
		gamma       = 0.9,
		alpha       = 1.0,
		tau         = 0.01,
	}
	q_net := ml_fin.cql_network_new(config, allocator)
	defer ml_fin.cql_network_free(&q_net)
	cql_train(&q_net, config, states, rewards, next_states, dones, actions, 1500, 3e-3, allocator)

	report :: proc(
		label: string,
		q_net: ^ml_fin.CQLNetwork,
		closes: []f64,
		allocator: mem.Allocator,
	) {
		acts := momentum_rule(closes, allocator)
		_, r, _, _ := ml_fin.cql_trading_dataset(closes, acts, 0.001, allocator)
		logged := 0.0
		for v in r.data.data {logged += v}
		pnl, counts := ml_fin.cql_backtest(q_net, closes, 0.001)
		fmt.printf(
			"%s (%d days): greedy CQL policy %+.1f%% (short %d, flat %d, long %d), momentum rule %+.1f%%\n",
			label,
			len(closes) - 1 - ml_fin.CQL_LAGS,
			pnl,
			counts[0],
			counts[1],
			counts[2],
			logged,
		)
	}
	report("Train", &q_net, train, allocator)
	report("Test ", &q_net, test, allocator)
	fmt.println("(summed daily P&L in percent of notional)")
}
