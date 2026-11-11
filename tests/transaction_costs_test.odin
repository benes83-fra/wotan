package tests

import w "../wotan/core"
import ml_fin "../wotan/ml_finance"
import yahoo "../wotan/net"
import "core:fmt"
import "core:math"
import "core:mem"

// Transaction costs on a real strategy: SPY over five years, a continuous 20-day
// momentum position clamp(r20 / (sigma sqrt 20), -1, 1) set at each close and held to
// the next, traded every day or only when the target moves more than a no-trade band of
// 0.25. Costs from the default model (1 bp half spread, square-root impact with Y = 1)
// with sigma and ADV from the trailing 20 days. Checks:
//   - the cost of the same trades grows with the capital, the impact term as its square
//     root (100x the capital: 10x the impact per unit traded)
//   - gross is the same at every capital, net falls with it; the break-even capital
//   - the band cuts turnover, and at large capital its net Sharpe beats daily trading

@(private = "file")
_tc_sharpe :: proc(r: []f64) -> (ann_ret, sharpe: f64) {
	s := ml_fin.sharpe_stats(r)
	m := 0.0
	for x in r {m += x / f64(len(r))}
	return 252 * m, s.sharpe * math.sqrt(252.0)
}

transaction_costs_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Transaction costs: SPY momentum, daily vs no-trade band, by capital ===")
	df, df_ok := yahoo.read_yahoo("SPY", .Daily, .FiveYears, allocator)
	if !df_ok {fail_missing_data(#procedure, "Yahoo data for SPY"); return}
	defer w.destroy_dataframe(&df)
	if df.rows < 600 {
		fmt.println("Not enough SPY data (offline?): skipped")
		return
	}
	pc, vc := w.column(&df, "AdjClose"), w.column(&df, "Volume")
	close := make([dynamic]f64)
	defer delete(close)
	volume := make([dynamic]f64)
	defer delete(volume)
	for i in 0 ..< df.rows {
		p, pn := w.column_at_float(pc, i)
		v, vn := w.column_at_int(vc, i) // Volume is an Int column
		if pn || vn || p <= 0 || v <= 0 {continue}
		append(&close, p)
		append(&volume, f64(v))
	}
	sigma, adv := ml_fin.liquidity_estimates(close[:], volume[:], 20)
	defer delete(sigma)
	defer delete(adv)

	// decisions at the close of t (t >= 20, sigma known), held over t -> t + 1
	BAND :: 0.25
	start := 20
	T := len(close) - 1 - start
	daily := make([]f64, T)
	defer delete(daily)
	banded := make([]f64, T)
	defer delete(banded)
	nxt := make([]f64, T)
	defer delete(nxt)
	sg := make([]f64, T)
	defer delete(sg)
	av := make([]f64, T)
	defer delete(av)
	prev := 0.0
	for k in 0 ..< T {
		t := start + k
		r20 := close[t] / close[t - 20] - 1
		target := clamp(r20 / (sigma[t] * math.sqrt(20.0)), -1, 1)
		daily[k] = target
		if abs(target - prev) > BAND {prev = target}
		banded[k] = prev
		nxt[k] = close[t + 1] / close[t] - 1
		sg[k], av[k] = sigma[t], adv[t]
	}
	gross_of :: proc(pos, nxt: []f64) -> []f64 {
		g := make([]f64, len(pos), context.temp_allocator)
		for p, k in pos {g[k] = p * nxt[k]}
		return g
	}
	turnover :: proc(pos: []f64) -> f64 {
		s, prev := 0.0, 0.0
		for p in pos {s += abs(p - prev); prev = p}
		return s * 252 / f64(len(pos))
	}
	g_daily, g_band := gross_of(daily, nxt), gross_of(banded, nxt)
	gr_d, gs_d := _tc_sharpe(g_daily)
	gr_b, gs_b := _tc_sharpe(g_band)
	fmt.printf(
		"   %d days; ADV now $%.1fbn, daily vol %.2f%%\n",
		T,
		adv[len(adv) - 1] / 1e9,
		100 * sigma[len(sigma) - 1],
	)
	fmt.printf(
		"   gross: daily %+.1f%% a year (Sharpe %.2f, turnover %.0f× a year); band %.2f %+.1f%% (Sharpe %.2f, turnover %.0f×)\n",
		100 * gr_d,
		gs_d,
		turnover(daily),
		BAND,
		100 * gr_b,
		gs_b,
		turnover(banded),
	)

	failures := 0
	check :: proc(failures: ^int, ok: bool, msg: string) {
		if !ok {failures^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", msg)
	}

	// impact scaling: 100x the capital, 10x the impact cost as a fraction of capital
	{
		imp := ml_fin.CostModel {
			impact          = 1,
			impact_exponent = 0.5,
		}
		c1 := ml_fin.position_costs(imp, daily, 1e7, sg, av, allocator = context.temp_allocator)
		c2 := ml_fin.position_costs(imp, daily, 1e9, sg, av, allocator = context.temp_allocator)
		s1, s2 := 0.0, 0.0
		for v, k in c1 {s1 += v; s2 += c2[k]}
		check(
			&failures,
			abs(s2 / s1 - 10) < 1e-9,
			fmt.tprintf(
				"square-root impact: the same trades at $1bn cost %.4f× the fraction of capital they cost at $10m (%.2f%% vs %.3f%% a year)",
				s2 / s1,
				100 * s2 * 252 / f64(T),
				100 * s1 * 252 / f64(T),
			),
		)
	}

	// by capital
	CAPS := []f64{1e6, 1e8, 1e10, 1e11}
	fmt.println("   capital     daily: return  Sharpe  cost      band: return  Sharpe  cost")
	prev_d, prev_b := math.INF_F64, math.INF_F64
	monotone := true
	last_sd, last_sb := 0.0, 0.0
	for cap in CAPS {
		cd := ml_fin.position_costs(
			ml_fin.DEFAULT_COST_MODEL,
			daily,
			cap,
			sg,
			av,
			allocator = context.temp_allocator,
		)
		cb := ml_fin.position_costs(
			ml_fin.DEFAULT_COST_MODEL,
			banded,
			cap,
			sg,
			av,
			allocator = context.temp_allocator,
		)
		nd := ml_fin.net_of_costs(g_daily, cd, context.temp_allocator)
		nb := ml_fin.net_of_costs(g_band, cb, context.temp_allocator)
		rd, sd := _tc_sharpe(nd)
		rb, sb := _tc_sharpe(nb)
		fmt.printf(
			"   $%-9.0e  %+7.1f%%  %6.2f  %5.2f%%       %+7.1f%%  %6.2f  %5.2f%%\n",
			cap,
			100 * rd,
			sd,
			100 * (gr_d - rd),
			100 * rb,
			sb,
			100 * (gr_b - rb),
		)
		monotone &&= rd < prev_d && rb < prev_b
		prev_d, prev_b = rd, rb
		last_sd, last_sb = sd, sb
	}
	check(&failures, monotone, "net return falls with the capital for both (gross is the same)")

	// break-even capital of daily trading: net annual return 0 (bisection on log10 capital)
	if gr_d > 0 {
		net_at :: proc(lc: f64, pos, gross, sg, av: []f64) -> f64 {
			c := ml_fin.position_costs(
				ml_fin.DEFAULT_COST_MODEL,
				pos,
				math.pow(10, lc),
				sg,
				av,
				allocator = context.temp_allocator,
			)
			m := 0.0
			for g, k in gross {m += g - c[k]}
			return m
		}
		lo, hi := 3.0, 15.0
		if net_at(lo, daily, g_daily, sg, av) > 0 && net_at(hi, daily, g_daily, sg, av) < 0 {
			for _ in 0 ..< 60 {
				mid := 0.5 * (lo + hi)
				if net_at(mid, daily, g_daily, sg, av) > 0 {lo = mid} else {hi = mid}
			}
			fmt.printf(
				"   daily trading breaks even at $%.0fm of capital\n",
				math.pow(10, lo) / 1e6,
			)
		} else {
			fmt.println("   daily trading: no break-even capital between $1k and $1000tn")
		}
	} else {
		fmt.println("   daily trading loses money before costs: no break-even capital")
	}
	check(
		&failures,
		turnover(banded) < turnover(daily) && last_sb > last_sd,
		fmt.tprintf(
			"the %.2f band trades %.0f%% of the daily turnover and at $%.0e has the higher net Sharpe (%.2f vs %.2f)",
			BAND,
			100 * turnover(banded) / turnover(daily),
			CAPS[len(CAPS) - 1],
			last_sb,
			last_sd,
		),
	)
	fmt.printf("Transaction costs: %d failure(s)\n", failures)
	record_failures(failures)
}
