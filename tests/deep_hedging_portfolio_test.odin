package tests

import w "../wotan/core"
import fin "../wotan/finance"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import optim "../wotan/optimize"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Deep hedging of a call on a 50/50 SPY/QQQ basket with both ETFs as hedging instruments.
// The volatilities and the correlation are estimated from one year of daily log returns;
// the paths are correlated GBMs in units of today's prices, discounted. Compared by P&L
// variance with no hedge and with the best static hedge (the variance-minimising
// holdings of both assets, kept to maturity).
deep_hedging_portfolio_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Deep hedging: call on a SPY/QQQ basket ===")
	spy_df := net.read_yahoo("SPY", .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&spy_df)
	qqq_df := net.read_yahoo("QQQ", .Daily, .OneYear, allocator)
	defer w.destroy_dataframe(&qqq_df)
	n := min(spy_df.rows, qqq_df.rows)
	if n < 60 {
		fmt.println("Not enough SPY/QQQ data (offline?): skipped")
		return
	}
	// daily log returns of the last n common rows (the same trading days)
	log_returns :: proc(df: ^w.DataFrame, n: int) -> []f64 {
		col := w.column(df, "AdjClose")
		off := df.rows - n
		r := make([]f64, n - 1)
		// a missing (null or 0, as Yahoo sends for some rows) close counts as no move
		prev, _ := w.column_at_float(col, off)
		for i in 1 ..< n {
			b, null := w.column_at_float(col, off + i)
			ok := !null && b > 0
			r[i - 1] = math.ln(b / prev) if ok && prev > 0 else 0
			if ok {prev = b}
		}
		return r
	}
	ann_vol :: proc(r: []f64) -> f64 {
		m, m2 := 0.0, 0.0
		for x in r {m += x; m2 += x * x}
		k := f64(len(r))
		return math.sqrt((m2 - m * m / k) / (k - 1) * 252)
	}
	r_spy, r_qqq := log_returns(&spy_df, n), log_returns(&qqq_df, n)
	corr := fin.correlation(r_spy, r_qqq)
	sigma1, sigma2 := ann_vol(r_spy), ann_vol(r_qqq)
	fmt.printf(
		"From %d daily returns: SPY vol %.1f%%, QQQ vol %.1f%%, correlation %.3f\n",
		n - 1,
		100 * sigma1,
		100 * sigma2,
		corr,
	)

	N_PATHS, N_STEPS :: 2048, 50
	T, r := 1.0, 0.04
	dt := T / N_STEPS
	F :: 3 // discounted SPY / SPY_0, discounted QQQ / QQQ_0, time to maturity
	pd := l.matrix_new(f64, N_PATHS * (N_STEPS + 1), F)
	yd := l.matrix_new(f64, N_PATHS, 1)
	c2 := math.sqrt(1 - corr * corr)
	for p in 0 ..< N_PATHS {
		s1, s2 := 1.0, 1.0
		for k in 0 ..= N_STEPS {
			if k > 0 {
				z1 := rand.float64_normal(0, 1)
				z2 := corr * z1 + c2 * rand.float64_normal(0, 1)
				s1 *= math.exp((r - 0.5 * sigma1 * sigma1) * dt + sigma1 * math.sqrt(dt) * z1)
				s2 *= math.exp((r - 0.5 * sigma2 * sigma2) * dt + sigma2 * math.sqrt(dt) * z2)
			}
			disc := math.exp(-r * f64(k) * dt)
			row := pd.data[(p * (N_STEPS + 1) + k) * F:][:F]
			row[0], row[1], row[2] = s1 * disc, s2 * disc, T - f64(k) * dt
		}
		yd.data[p] = math.exp(-r * T) * max(0.5 * s1 + 0.5 * s2 - 1, 0) // at-the-money
	}
	paths := t.tensor_new(pd)
	paths.shape = {N_PATHS, N_STEPS + 1, F, 1}
	payoffs := t.tensor_new(yd)

	variance :: proc(x: []f64) -> f64 {
		m, m2 := 0.0, 0.0
		for v in x {m += v / f64(len(x)); m2 += v * v / f64(len(x))}
		return m2 - m * m
	}
	// best static holdings: regress the payoff on the two discounted price changes
	X := l.matrix_new(f64, N_PATHS, 3)
	for p in 0 ..< N_PATHS {
		last := pd.data[(p * (N_STEPS + 1) + N_STEPS) * F:][:F]
		X.data[p * 3 + 0] = 1
		X.data[p * 3 + 1] = last[0] - 1
		X.data[p * 3 + 2] = last[1] - 1
	}
	coef := l.lstsq(&X, yd.data).beta
	static_pnl := make([]f64, N_PATHS)
	for p in 0 ..< N_PATHS {
		static_pnl[p] = coef[1] * X.data[p * 3 + 1] + coef[2] * X.data[p * 3 + 2] - yd.data[p]
	}
	unhedged := variance(yd.data)
	static_var := variance(static_pnl)

	h := ml_fin.deep_hedger_new(
		{
			state_size = F,
			hidden_size = 64,
			num_layers = 2,
			risk_measure = .Variance,
			num_assets = 2,
		},
	)
	defer ml_fin.deep_hedger_free(h)
	EPOCHS :: 300
	opt := nn.adam_new(3e-3)
	defer nn.adam_free(&opt)
	nn.sequential_add_to_adam(h.network, &opt)
	schedule := optim.lr_cosine(3e-3, EPOCHS, warmup_steps = 10)
	for epoch in 0 ..< EPOCHS {
		opt.learning_rate = optim.lr_schedule_at(schedule, epoch)
		loss := ml_fin.deep_hedger_train_step(h, paths, payoffs, &opt)
		if epoch % 60 == 0 || epoch == EPOCHS - 1 {
			fmt.printf("  epoch %3d | P&L variance %.6f\n", epoch, loss)
		}
	}
	pnl, _ := ml_fin.deep_hedging_evaluate(h, paths, payoffs, allocator)
	deep_var := variance(pnl)
	ok := deep_var < static_var
	fmt.printf(
		"%s basket call (units of today's prices): P&L variance unhedged %.6f, best static holdings (%.3f SPY, %.3f QQQ) %.6f, deep hedge %.6f (%.0f%% below static)\n",
		"OK  " if ok else "FAIL",
		unhedged,
		coef[1],
		coef[2],
		static_var,
		deep_var,
		100 * (1 - deep_var / static_var),
	)
	record_failures(int(!ok))
}
