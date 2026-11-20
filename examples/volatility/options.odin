package main

// Stage 5a: selling volatility. Implied volatility is on average above the
// volatility that follows (the volatility risk premium); option sellers earn it
// as insurers do, and lose heavily in crashes. Every HORIZON days, at the
// close, the strategy sells a one-month at-the-money straddle on SPY (a call
// and a put at strike = spot) at the Black-Scholes price for the VIX, and
// hedges its delta every day at that volatility until expiry. The P&L per
// straddle is in % of the spot at the sale.
//
// Simplifications: the VIX is a 30-day variance-swap rate on the S&P 500, a
// little above at-the-money implied volatility, so pricing at the VIX
// overstates the premium; no bid-ask spreads; zero interest and dividend rates.
// The break-even column measures how much the first one matters.
//
// The strategies differ only in when they sell:
// - every month;
// - when the VRP signal (mlf.compute_vrp: VIX minus a model's forecast,
//   z-scored against the training days) is positive;
// - when the regime model says calm.

import fin "../../wotan/finance"
import mlf "../../wotan/ml_finance"
import "core:fmt"
import "core:math"

Straddle :: struct {
	day:     int, // the sale
	pnl:     f64, // % of spot
	premium: f64, // % of spot
	vega:    f64, // % of spot per volatility point at the sale
}

// sell_straddles: one straddle every HORIZON days of the test period
sell_straddles :: proc(m: ^Market, f: ^Forecasts) -> []Straddle {
	out := make([dynamic]Straddle)
	TAU :: f64(HORIZON) / TRADING_DAYS
	for d := f.test_lo; d + HORIZON < m.n_days; d += HORIZON {
		S0 := m.spy[d]
		K := S0
		sigma := m.vix[d] / 100
		call := fin.compute_greeks(S0, K, TAU, 0, sigma, .Call)
		put := fin.compute_greeks(S0, K, TAU, 0, sigma, .Put)
		cash := call.price + put.price // received
		// daily delta hedge: the short straddle's delta is -(call + put delta),
		// so the hedge holds +(call + put delta) shares
		for j in 0 ..< HORIZON {
			S := m.spy[d + j]
			tau := f64(HORIZON - j) / TRADING_DAYS
			c := fin.compute_greeks(S, K, tau, 0, sigma, .Call)
			p := fin.compute_greeks(S, K, tau, 0, sigma, .Put)
			cash += (c.delta + p.delta) * (m.spy[d + j + 1] - S)
		}
		cash -= abs(m.spy[d + HORIZON] - K) // the payoff at expiry
		append(
			&out,
			Straddle {
				day = d,
				pnl = 100 * cash / S0,
				premium = 100 * (call.price + put.price) / S0,
				vega = 100 * (call.vega + put.vega) / S0,
			},
		)
	}
	return out[:]
}

Option_Strategy :: struct {
	name:    string,
	monthly: []f64, // P&L per period, % of spot; 0 in periods without a sale
	sold:    int,
}

Option_Stats :: struct {
	mean, sd, sharpe, psr, worst, hit_rate, total: f64,
	break_even:                                    f64, // vol points below the VIX at which the mean P&L is 0
}

// option_strategies: the straddles each rule sells
option_strategies :: proc(m: ^Market, f: ^Forecasts, trades: []Straddle) -> []Option_Strategy {
	Rule :: enum {
		Always,
		VRP_GARCH,
		VRP_LSTM,
		Calm,
	}
	names := [Rule]string {
		.Always    = "Sell every month",
		.VRP_GARCH = "Sell if VRP > avg (GARCH)",
		.VRP_LSTM  = "Sell if VRP > avg (LSTM)",
		.Calm      = "Sell if regime calm",
	}
	out := make([]Option_Strategy, len(Rule))
	for rule in Rule {
		s := &out[rule]
		s.name = names[rule]
		s.monthly = make([]f64, len(trades))
		for tr, i in trades {
			d := tr.day
			sell := true
			switch rule {
			case .Always:
			case .VRP_GARCH, .VRP_LSTM:
				fc := f.vol[.GARCH][d] if rule == .VRP_GARCH else f.vol[.LSTM][d]
				// compute_vrp takes the forecast as a daily volatility
				sig := mlf.compute_vrp(
					m.vix[d],
					fc / 100 / math.sqrt(f64(TRADING_DAYS)),
					f.vrp_mean[d],
					f.vrp_sd[d],
				)
				sell = sig.signal > 0
			case .Calm:
				sell = f.p_turbulent[d] < 0.5
			}
			if sell {s.monthly[i] = tr.pnl; s.sold += 1}
		}
	}
	return out
}

option_stats :: proc(s: ^Option_Strategy, trades: []Straddle) -> (st: Option_Stats) {
	n := len(s.monthly)
	vega, wins := 0.0, 0
	st.worst = math.INF_F64
	for v, i in s.monthly {
		st.total += v
		st.mean += v / f64(n)
		if v != 0 {
			vega += trades[i].vega
			st.worst = min(st.worst, v)
			if v > 0 {wins += 1}
		}
	}
	for v in s.monthly {st.sd += (v - st.mean) * (v - st.mean) / f64(n - 1)}
	st.sd = math.sqrt(st.sd)
	periods_per_year := f64(TRADING_DAYS) / HORIZON
	st.sharpe = st.mean / st.sd * math.sqrt(periods_per_year)
	st.psr = mlf.probabilistic_sharpe_ratio_returns(s.monthly)
	st.hit_rate = f64(wins) / f64(max(s.sold, 1))
	// P&L is linear in the sale price: a price lower by x vol points costs x · vega
	st.break_even = st.total / vega
	return
}

print_option_stats :: proc(strategies: []Option_Strategy, trades: []Straddle) {
	fmt.printfln(
		"  %-26s %5s %8s %7s %7s %6s %8s %6s %11s",
		"",
		"sold",
		"mean %",
		"sd %",
		"Sharpe",
		"PSR",
		"worst %",
		"hit",
		"break-even",
	)
	for &s in strategies {
		st := option_stats(&s, trades)
		fmt.printfln(
			"  %-26s %5d %8.3f %7.2f %7.2f %6.2f %8.2f %5.0f%% %8.1f pts",
			s.name,
			s.sold,
			st.mean,
			st.sd,
			st.sharpe,
			st.psr,
			st.worst,
			100 * st.hit_rate,
			st.break_even,
		)
	}
}
