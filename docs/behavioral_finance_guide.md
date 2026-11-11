# The `behavioral_finance` package

`wotan/behavioral_finance` implements models in which investors are not fully rational, and the measures used to test them. It covers:
- prospect theory: the value and probability weighting functions, the value of a prospect, and the prospect theory value of a stock;
- option prices under cumulative prospect theory (CPT), on log-normal and Heston distributions, and calibration to option prices;
- portfolio choice: the CPT-optimal portfolio, and behavioral portfolio theory with mental accounts;
- cross-sectional signals: the 52-week high, momentum and long-term reversal, the frog in the pan, lottery demand (MAX and salience) and the capital gains overhang, with the sorts and overlapping portfolios used to trade them;
- the disposition effect in trading records (Odean's PGR and PLR);
- herding tests on return dispersion;
- the Baker–Wurgler investor sentiment index;
- four equilibrium models: Barberis–Shleifer–Vishny, Hong–Stein, De Long–Shleifer–Summers–Waldmann and the X-CAPM.

Every example was compiled and run against the current code, and the output shown is the real output. A debug build with a tracking allocator reports no leaks for any of them. Most procedures reproduce the published tables and examples of their papers, and agree with NumPy, SciPy, pandas and statsmodels; section 11 lists the comparisons.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [Prospect theory](#2-prospect-theory)
3. [Option prices under prospect theory](#3-option-prices-under-prospect-theory)
4. [Portfolio choice](#4-portfolio-choice)
5. [Cross-sectional signals](#5-cross-sectional-signals)
6. [The disposition effect in trading records](#6-the-disposition-effect-in-trading-records)
7. [Herding](#7-herding)
8. [Investor sentiment](#8-investor-sentiment)
9. [Equilibrium models](#9-equilibrium-models)
10. [Heuristics](#10-heuristics)
11. [Validation](#11-validation)
12. [Quick reference](#12-quick-reference)

## 1. Setup and conventions

```odin
import bf "wotan/behavioral_finance"
import fin "wotan/finance"
import l "wotan/linalg"
```

The conventions in this package:
- **Panels.** Cross-sectional measures take and return `[][]f64` indexed `[asset][time]`. Free a panel with `signal_panel_free`; `compute_anchoring_signal_free` and `lottery_measure_free` are aliases of it. The exception is `overlapping_strategy_returns`, which takes its weights and returns as `[period][asset]`.
- **Monthly measures from daily data.** `months` holds day bounds: month k is days `months[k] ..< months[k+1]`, so `len(months)` is the number of months plus one. The value for month k predicts the return of month k + 1.
- **Missing data.** Inputs may hold NaN. Each procedure documents what it leaves out; a value that cannot be computed is NaN.
- **Results.** Procedures that return slices allocate them; free them with `delete`. Result structs have free procedures:
  - `prospect_precomputed_free`;
  - `bpt_result_free`;
  - `overlapping_returns_free`;
  - `herding_dispersion_free`;
  - `sentiment_index_free`;
  - `bsv_path_free`, `dssw_path_free`, `xcapm_path_free`.
- **Option types and prices** come from the `finance` package: `fin.OptionType`, `fin.VolSurfacePoint`, `fin.Heston_Params`.

## 2. Prospect theory

### Value and weighting functions

Tversky & Kahneman's (1992) cumulative prospect theory values outcomes relative to a reference point:
- the value function is v(x) = x^α for gains and −λ(−x)^β for losses;
- probabilities are replaced by decision weights from a weighting function w, applied to cumulative probabilities: P(an outcome at least as good) for gains, P(at least as bad) for losses.

`Prospect_Params` holds α, β, λ, γ (gains) and γ⁻ (losses; 0 means "the same as γ"), and the weighting function: `.Tversky_Kahneman`, p^γ / (p^γ + (1 − p)^γ)^(1/γ), or `.Prelec`, exp(−(−ln p)^γ). `TK92_PARAMS` holds Tversky & Kahneman's estimates (α = β = 0.88, λ = 2.25, γ = 0.61, γ⁻ = 0.69). `RATIONAL_PARAMS` (all 1) is a risk- and probability-neutral investor.

```odin
p := bf.TK92_PARAMS
for x in ([]f64{-100, -10, 10, 100}) {fmt.printf("v(%v) = %.4f\n", x, bf.prospect_value(x, p))}
for q in ([]f64{0.01, 0.1, 0.5, 0.9, 0.99}) {
	fmt.printf("w+(%.2f) = %.4f  w-(%.2f) = %.4f  Prelec(%.2f, 0.65) = %.4f\n", q,
		bf.prospect_weight(q, p.gamma), q, bf.prospect_weight(q, bf.prospect_gamma_minus(p)),
		q, bf.prospect_weight(q, 0.65, .Prelec))
}
```

```
v(-100) = -129.4740
v(-10) = -17.0680
v(10) = 7.5858
v(100) = 57.5440
w+(0.01) = 0.0553  w-(0.01) = 0.0397  Prelec(0.01, 0.65) = 0.0673
w+(0.10) = 0.1863  w-(0.10) = 0.1701  Prelec(0.10, 0.65) = 0.1791
w+(0.50) = 0.4206  w-(0.50) = 0.4540  Prelec(0.50, 0.65) = 0.4547
w+(0.90) = 0.7117  w-(0.90) = 0.7749  Prelec(0.90, 0.65) = 0.7933
w+(0.99) = 0.9116  w-(0.99) = 0.9451  Prelec(0.99, 0.65) = 0.9510
```

A loss of 100 weighs 2.25 times a gain of 100. Small probabilities are overweighted (w(0.01) > 0.01) and large ones underweighted.

### The value of a prospect

`cpt_value(outcomes, params)` is the CPT value of equally likely outcomes, already measured relative to the reference point. With γ = γ⁻ = 1 it is the plain average of v(x).

```odin
coin := []f64{-100, 100}
fmt.printf("50/50 lose or win 100: %.4f\n", bf.cpt_value(coin, p))
coin2 := []f64{-100, 200}
fmt.printf("50/50 lose 100 or win 200: %.4f\n", bf.cpt_value(coin2, p))
lottery := make([]f64, 1000)
defer delete(lottery)
for &v in lottery {v = -1}
lottery[0] = 999
fmt.printf("pay 1, win 1000 with probability 1/1000: %.4f\n", bf.cpt_value(lottery, p))
nw := p
nw.gamma, nw.gamma_minus = 1, 1
fmt.printf("  the same without probability weighting: %.4f\n", bf.cpt_value(lottery, nw))
```

```
50/50 lose or win 100: -34.5743
50/50 lose 100 or win 200: -14.2328
pay 1, win 1000 with probability 1/1000: 4.0805
  the same without probability weighting: -1.8116
```

The investor rejects both coin flips, even the one with twice the upside: that is loss aversion. They accept the fair lottery only because the 1/1000 chance is overweighted.

### The prospect theory value of a stock

Barberis, Mukherjee & Wang (2016) model investors who see a stock as the distribution of its past returns and value it with CPT. `tk_value(returns, benchmark, window, params)` computes their TK at every period t from the `window` returns before t (60 months in the paper), in excess of the benchmark (nil means raw returns). TK[t] is NaN for t < window and while the window holds a NaN. Stocks with a high TK are overpriced: TK predicts the next month's return with a negative sign.

```odin
rand.reset(1)
T := 120
skewed := make([]f64, T)
normal := make([]f64, T)
mkt := make([]f64, T)
defer {delete(skewed); delete(normal); delete(mkt)}
for t in 0 ..< T {
	mkt[t] = 0.008 + 0.04 * rand.float64_normal(0, 1)
	normal[t] = mkt[t] + 0.08 * rand.float64_normal(0, 1)
	// mean 0 and a similar sd (8.3%), right-skewed: a 36% gain one month in twenty
	u := rand.float64()
	jump := 0.36 if u < 0.05 else -0.36 * 0.05 / 0.95
	skewed[t] = mkt[t] + jump
}
tk1 := bf.tk_value(normal, mkt, 60)
tk2 := bf.tk_value(skewed, mkt, 60)
defer {delete(tk1); delete(tk2)}
```

```
TK[59] NaN  TK[60] normal -0.05529  skewed -0.03351  TK[119] normal -0.10020  skewed 0.00559
```

With a similar volatility, the right-skewed stock has the higher TK: probability weighting makes its rare large gains attractive.

## 3. Option prices under prospect theory

### The price

The CPT price of an option is the premium at which a CPT investor is indifferent between taking the position and not: the prospect value of the position is zero (Versluis, Lehnert & Wolff 2010; Nardon & Pianca 2012). The terminal price is drawn from its physical distribution, with drift μ, and the premium is carried to maturity at the risk-free rate. Two choices remain:
- **side:** `.Writer` receives the premium and pays the payoff (Versluis et al. take the writer's price as the market price); `.Holder` pays it;
- **frame:** `.Aggregated` nets premium and payoff into one outcome; `.Segregated` values them as two separate prospects.

`prospect_theory_price(S, K, T, r, mu, sigma, params, opt_type, side, frame, n_points)` prices a European option on a log-normal underlying. The distribution is cut into `n_points` cells (2000 by default) over ±10 standard deviations, with the strike on a cell edge.

```odin
S, K, Tm, r, mu, sig := 100.0, 100.0, 1.0, 0.05, 0.10, 0.20
rat := bf.prospect_theory_price(S, K, Tm, r, r, sig, bf.RATIONAL_PARAMS, .Call)
for ot in ([]fin.OptionType{.Call, .Put}) {
	for side in bf.CPT_Side {
		for frame in bf.CPT_Frame {
			fmt.printf("TK92 %v %v %v: %.4f\n", ot, side, frame,
				bf.prospect_theory_price(S, K, Tm, r, mu, sig, p, ot, side, frame))
		}
	}
}
```

```
Black-Scholes call 10.4506 put 5.5735
rational investor, mu = r: call 10.4507
TK92 Call Writer Aggregated: 26.9370
TK92 Call Writer Segregated: 36.9381
TK92 Call Holder Aggregated: 10.8255
TK92 Call Holder Segregated: 5.9544
TK92 Put Writer Aggregated: 10.2080
TK92 Put Writer Segregated: 12.1297
TK92 Put Holder Aggregated: 3.1549
TK92 Put Holder Segregated: 2.0623
```

The rational investor with μ = r recovers Black–Scholes. Loss-averse writers demand far more than the Black–Scholes price and holders pay less, so the holder and writer prices bracket it.

### Reusing the distribution

When many options share an expiry and an underlying, build the distribution once:
- `prospect_lognormal_grid(S, K, T, mu, sigma, n_points)` returns a `Prospect_Precomputed` (grid nodes and the CDF at each node);
- `prospect_price_precomputed(&pre, K, T, r, params, opt_type, side, frame)` prices on it;
- `prospect_option_value(&pre, K, T, r, premium, ...)` is the prospect value of a position taken at a given premium, zero at the CPT price.

```odin
pre := bf.prospect_lognormal_grid(S, K, Tm, mu, sig, 2000)
defer bf.prospect_precomputed_free(&pre)
c := bf.prospect_price_precomputed(&pre, K, Tm, r, p, .Call)
fmt.printf("writer call %.4f; V at that premium %.2e, at premium + 1 %.4f\n", c,
	bf.prospect_option_value(&pre, K, Tm, r, c, p, .Call),
	bf.prospect_option_value(&pre, K, Tm, r, c + 1, p, .Call))
```

```
writer call 26.9370; V at that premium -2.36e-15, at premium + 1 0.7847
```

The strike is a node of the grid, so pass the same K to the grid and the price.

### Calibration

`calibrate_prospect_theory(market_data, S, r, mu, sigma, setup, opt_types)` fits the CPT parameters to option prices by Adam on finite-difference gradients. Each option is described by:
- a `fin.VolSurfacePoint`: strike, expiry and `market_price` (or `implied_vol` when the price is 0);
- an entry in `opt_types` (nil means all calls).

`Prospect_Calibration_Setup` sets:
- the side and frame;
- the start values (zero means TK92);
- the parameters held fixed;
- the iteration limit (default 200);
- the grid size (default 1000).

α, β and λ trade off against each other, so fix some of them to identify the rest. In the example below, prices generated with γ⁻ = 0.55 are fitted with α, β and λ fixed:

```odin
truth := p
truth.lambda, truth.gamma_minus = 2.25, 0.55
strikes := []f64{80, 90, 100, 110, 120}
chain := make([]fin.VolSurfacePoint, 2 * len(strikes))
types := make([]fin.OptionType, 2 * len(strikes))
defer {delete(chain); delete(types)}
for k, i in strikes {
	for j in 0 ..< 2 {
		ot := fin.OptionType.Put if j == 0 else .Call
		price := bf.prospect_theory_price(S, k, 0.5, r, mu, sig, truth, ot, n_points = 1000)
		chain[2 * i + j] = {strike = k, expiry = 0.5, market_price = price}
		types[2 * i + j] = ot
	}
}
setup := bf.Prospect_Calibration_Setup {
	fixed    = {.Alpha, .Beta, .Lambda},
	max_iter = 300,
}
res := bf.calibrate_prospect_theory(chain, S, r, mu, sig, setup, types)
```

```
gamma 0.6101 (true 0.61) gamma_minus 0.5499 (true 0.55) rmse 2.27e-04 converged false after 300
```

`converged` is true only when the RMSE falls below 1e-6; here the iteration limit stopped Adam with both parameters recovered to four decimals.

### Behavioral Heston

`behavioral_heston_price` prices with the same CPT valuation but with S_T distributed as in the Heston model, with drift r. The Heston CDF comes from Gil-Pelaez inversion of the characteristic function (`extract_heston_cdf_grid`), on a log grid of 4000 cells by default. Heston parameters calibrated to option prices describe the risk-neutral distribution. CPT parameters calibrated on top of them, with `calibrate_behavioral_heston`, therefore measure what stochastic volatility leaves unexplained.

```odin
hp := fin.Heston_Params{v0 = 0.04, kappa = 2, theta = 0.04, sigma = 0.5, rho = -0.7}
hes := fin.heston_price(S, K, Tm, r, hp, .Call)
bh_rat := bf.behavioral_heston_price(S, K, Tm, r, hp, bf.RATIONAL_PARAMS, .Call)
bh_tk := bf.behavioral_heston_price(S, K, Tm, r, hp, p, .Call)
```

```
Heston call 10.154627, behavioral Heston rational 10.154665, TK92 writer 17.9960
```

With rational parameters it reproduces the Heston price; the price error falls as 1/n_grid², about 5e-6 relative at 4000 cells.

## 4. Portfolio choice

### The CPT portfolio

`prospect_portfolio_optimize(&returns, params, reference, constraints)` maximizes the CPT value of the portfolio's period returns relative to a reference return. Each row of `returns` is one equally likely scenario, for example the returns of past evaluation periods (Benartzi & Thaler 1995). The feasible set is a `fin.PortfolioConstraints`, long-only and fully invested by default.

The objective is not concave: v is convex over losses and the weights depend on rank. The optimizer therefore runs a projected gradient method (`fin.constrained_minimize`) from every vertex, every edge midpoint and the centre of the simplex, and keeps the best result. `prospect_metrics(weights, &returns, params, reference)` reports the CPT value, mean, volatility and probability of a loss.

Myopic loss aversion: the shorter the evaluation period, the more often stocks show a loss, and the less a loss-averse investor holds them.

```odin
// 1200 months: stocks 0.8% a month, sd 4.5%; bonds 0.3%, sd 1%
rand.reset(3)
M := 1200
stock := make([]f64, M)
bond := make([]f64, M)
defer {delete(stock); delete(bond)}
for t in 0 ..< M {
	stock[t] = 0.008 + 0.045 * rand.float64_normal(0, 1)
	bond[t] = 0.003 + 0.01 * rand.float64_normal(0, 1)
}
for h in ([]int{1, 12, 60}) {
	n := M / h
	R := l.matrix_new(f64, n, 2)
	defer l.matrix_free(&R)
	for k in 0 ..< n {
		gs, gb := 1.0, 1.0
		for t in k * h ..< (k + 1) * h {gs *= 1 + stock[t]; gb *= 1 + bond[t]}
		R.data[2 * k], R.data[2 * k + 1] = gs - 1, gb - 1
	}
	w := bf.prospect_portfolio_optimize(&R, p)
	defer delete(w)
	m := bf.prospect_metrics(w, &R, p)
	fmt.printf("%2d-month periods (%4d): stocks %.3f bonds %.3f  CPT value %.5f, mean %.4f, sd %.4f, P(loss) %.3f\n",
		h, n, w[0], w[1], m.cpt_value, m.expected_return, m.volatility, m.loss_probability)
}
```

```
 1-month periods (1200): stocks 0.120 bonds 0.880  CPT value -0.00296, mean 0.0034, sd 0.0105, P(loss) 0.387
12-month periods ( 100): stocks 1.000 bonds 0.000  CPT value 0.05844, mean 0.1070, sd 0.1634, P(loss) 0.240
60-month periods (  20): stocks 1.000 bonds 0.000  CPT value 0.57762, mean 0.6936, sd 0.6441, P(loss) 0.100
```

### Mental accounts

In behavioral portfolio theory (Shefrin & Statman 2000), investors split their wealth into mental accounts. Each account has a threshold return H and a maximum probability α of falling short of it, and maximizes its expected return subject to P(r < H) ≤ α. With normal returns, Das, Markowitz, Scheid & Statman (2010) show that the optimum is a mean-variance portfolio. Its implied risk aversion γ is the smallest one that meets the constraint.

`Mental_Account` holds:
- the threshold H;
- the probability α;
- the account's share of wealth;
- optionally a universe: the assets it may hold.

The procedures:
- `mental_account_portfolio(mu, &cov, account, long_only)` solves one account;
- `bpt_optimize(mu, &cov, accounts, long_only)` solves all of them and their wealth-weighted aggregate (the shares must sum to 1);
- `shortfall_probability(weights, mu, &cov, H)` is P(r < H) under normality.

The example is the paper's three-asset market with three accounts:

```odin
mu := []f64{0.05, 0.10, 0.25}
cov := l.matrix_new(f64, 3, 3)
defer l.matrix_free(&cov)
copy(cov.data, []f64{0.0025, 0, 0, 0, 0.04, 0.02, 0, 0.02, 0.25})
accounts := []bf.Mental_Account {
	{threshold = -0.10, probability = 0.05, share = 0.6}, // retirement
	{threshold = -0.05, probability = 0.15, share = 0.2}, // education
	{threshold = -0.15, probability = 0.20, share = 0.2}, // bequest
}
for long_only in ([]bool{false, true}) {
	res := bf.bpt_optimize(mu, &cov, accounts, long_only)
	defer bf.bpt_result_free(&res)
	// print each account and the aggregate
}
```

```
long only false
  account 0: Optimal gamma 3.7950 weights [0.5394, 0.2656, 0.1949] return 0.1023 sd 0.1230 P(r < H) 0.0500
  account 1: Optimal gamma 2.7063 weights [0.3787, 0.3499, 0.2714] return 0.1218 sd 0.1657 P(r < H) 0.1500
  account 2: Optimal gamma 0.8773 weights [-0.7890, 0.9620, 0.8270] return 0.2635 sd 0.4913 P(r < H) 0.2000
  aggregate [0.2416, 0.4217, 0.3367] return 0.1384 sd 0.2032
long only true
  account 0: Optimal gamma 3.7950 weights [0.5394, 0.2656, 0.1949] return 0.1023 sd 0.1230 P(r < H) 0.0500
  account 1: Optimal gamma 2.7063 weights [0.3787, 0.3499, 0.2714] return 0.1218 sd 0.1657 P(r < H) 0.1500
  account 2: Optimal gamma 0.7220 weights [0.0000, 0.0889, 0.9111] return 0.2367 sd 0.4594 P(r < H) 0.2000
  aggregate [0.3994, 0.2471, 0.3535] return 0.1330 sd 0.1938
```

These are the paper's Table 1 weights. Each constraint binds exactly at α. The status of an account says how its constraint was met:
- `.Optimal`: the constraint binds;
- `.Max_Return` (long-only): even the highest-return portfolio meets it;
- `.Unbounded` (with short sales): the expected return can grow without limit;
- `.Infeasible`: no portfolio reaches H with probability 1 − α. The weights are then the portfolio with the highest α-quantile wᵀμ + Φ⁻¹(α)σ, the best threshold any portfolio can guarantee. This is not the portfolio with the lowest P(r < H).

```odin
only_bonds_stocks := []bool{true, true, false}
a := bf.mental_account_portfolio(mu, &cov, {threshold = -0.05, probability = 0.1, universe = only_bonds_stocks})
defer delete(a.weights)
b := bf.mental_account_portfolio(mu, &cov, {threshold = 0.20, probability = 0.05})
defer delete(b.weights)
fmt.printf("H = 20%% at 5%%: %v, weights %.4f with the highest 5%% quantile %.4f\n", b.status,
	b.weights, b.expected_return + fin.norm_inv(0.05) * b.volatility)
```

```
universe bonds + stocks: Optimal weights [0.5393, 0.4607, 0.0000]
H = 20% at 5%: Infeasible, weights [0.8926, 0.0805, 0.0269] with the highest 5% quantile -0.0231
```

`mu` and `cov` must be per evaluation period, the same period as the thresholds.

## 5. Cross-sectional signals

The signals below share a workflow:
1. Compute a signal per asset and period.
2. Each period, sort the assets on it and go long the top and short the bottom (`long_short_portfolio`, `double_sort_portfolio`).
3. Hold the portfolios with overlapping cohorts (`overlapping_strategy_returns`).

### The 52-week high

George & Hwang (2004): investors anchor on the 52-week high, so stocks near it underreact to good news. `compute_anchoring_signal(prices, lookback)` is PTH = P / (the highest price over the `lookback` days ending today), 252 by default. Like pandas `p / p.rolling(lookback).max()`, it is NaN until a full window of valid prices exists and while the window holds an invalid one (NaN, zero, negative or infinite).

```odin
prices := [][]f64{{10, 11, 12, 11, 9, 10, 13, 12}, {20, 19, nan, 18, 17, 19, 18, 16}}
pth := bf.compute_anchoring_signal(prices, 3)
defer bf.signal_panel_free(pth)
```

```
asset 0: [NaN, NaN, 1.0000, 0.9167, 0.7500, 0.9091, 1.0000, 0.9231]
asset 1: [NaN, NaN, NaN, NaN, NaN, 1.0000, 0.9474, 0.8421]
```

### Long-short portfolios and overlapping holding

`long_short_portfolio(signal, fraction)` goes equally weighted long the top `fraction` of the assets with a finite signal and short the bottom `fraction`. It takes k = ⌊fraction·n⌋ assets on each side, breaking ties by asset index, so the weights sum to +1 and −1. `anchoring_portfolio` is an alias of it. George & Hwang use 30%, Jegadeesh & Titman deciles (0.1).

```odin
sig := []f64{0.91, 0.75, nan, 0.98, 0.62, 0.85, 0.70, 0.99, 0.80, 0.66}
wls := bf.long_short_portfolio(sig, 0.3)
defer delete(wls)
```

```
[0.0000, 0.0000, 0.0000, 0.5000, -0.5000, 0.0000, 0.0000, 0.5000, 0.0000, -0.5000]
```

Nine signals are finite, so k = 2.

`overlapping_strategy_returns(weights, returns, hold, skip)` holds the portfolios as Jegadeesh & Titman (1993) do:
- `weights[t]` is formed at the end of period t, and `returns[s]` is the return over period s;
- period s holds the cohorts formed at s − skip − hold … s − skip − 1, each with weight 1/hold;
- the result has the long basket's return, what the shorted assets earned, and the spread between them.

Periods before the first full set of cohorts are NaN, and an asset with a NaN return contributes zero.

```odin
// 2 assets, 5 periods: long asset 0 / short asset 1, then the reverse
W := [][]f64{{1, -1}, {1, -1}, {-1, 1}, {1, -1}, {1, -1}}
R := [][]f64{{0.01, 0.02}, {0.03, -0.01}, {0.02, 0.00}, {-0.01, 0.04}, {0.05, 0.01}}
ov := bf.overlapping_strategy_returns(W, R, hold = 2)
defer bf.overlapping_returns_free(&ov)
```

```
long [NaN, NaN, 0.0200, 0.0150, 0.0300]
short [NaN, NaN, 0.0000, 0.0150, 0.0300]
spread [NaN, NaN, 0.0200, 0.0000, 0.0000]
```

In period 3 the portfolio holds the cohorts of periods 1 and 2. Those point in opposite directions, so the spread is 0.

### Momentum and long-term reversal

`past_return(returns, lookback, skip, market, kind)` is the formation-period return. The value at t covers periods t − skip − lookback + 1 … t − skip, compounded (`.Compound`) or summed (`.Sum`). A market return, if given, is subtracted period by period first.

With it:
- **Jegadeesh & Titman's 12-1 momentum** is `past_return(r, 11, 1)`, then decile portfolios and overlapping holding;
- **De Bondt & Thaler's (1985) reversal** sorts on the summed market-adjusted return over 36 months: `past_return(r, 36, 0, market, .Sum)`.

```odin
r1 := [][]f64{{0.10, -0.05, 0.02, 0.04, -0.03, 0.06}}
mk := []f64{0.01, 0.01, 0.01, 0.01, 0.01, 0.01}
pc := bf.past_return(r1, 3, 1)
ps := bf.past_return(r1, 3, 1, mk, .Sum)
defer {bf.signal_panel_free(pc); bf.signal_panel_free(ps)}
```

```
compounded [NaN, NaN, NaN, 0.0659, 0.0078, 0.0290]
summed, market-adjusted [NaN, NaN, NaN, 0.0400, -0.0200, 0.0000]
```

### The frog in the pan

Da, Gurun & Warachka (2014) find that investors underreact more to information that arrives in many small pieces than to a few large ones. They measure this with information discreteness:
- **ID** = sgn(PRET) (%neg − %pos), where PRET is the formation-period return and %pos and %neg are the shares of days with positive and negative returns;
- **ID_Z** leaves out zero-return days: it divides by %neg + %pos.

Low ID (continuous information) means stronger momentum.

The procedures:
- `information_discreteness_value(daily, pret, kind)` computes ID for one formation period;
- `information_discreteness(daily, months, lookback, skip, kind)` computes ID and PRET at the end of every month. The defaults are DGW's: the formation period is months t−12 … t−2.

```odin
// a 10% formation return: one big day vs many small ones
big := []f64{0.10, -0.001, -0.001, -0.001, -0.001, 0, 0, 0, 0, 0}
small := []f64{0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0042}
```

```
discrete: PRET 0.0956 ID 0.30 ID_Z 0.6000
continuous: PRET 0.0983 ID -1.00 ID_Z -1.0000
```

DGW's test sorts sequentially, first on PRET quintiles, then on ID quintiles within each, and compares momentum across the ID groups:
- `quantile_groups(values, n_groups, out)` ranks the finite values (ties by index) and puts rank r in group ⌊r·n_groups/n⌋ (−1 for non-finite values);
- `double_sort(first, second, n1, n2, g1, g2, mode)` assigns both groups, `.Sequential` (DGW Table 2A) or `.Independent` (Table 2B);
- `double_sort_portfolio(first, second, g, n1, n2, mode)` is the winner-minus-loser portfolio on `first` within group g of `second`.

```odin
vals := []f64{0.3, -0.1, 0.5, nan, 0.2, 0.2, -0.4, 0.9}
groups := make([]int, len(vals))
defer delete(groups)
bf.quantile_groups(vals, 3, groups)
// 100 assets: PRET normal, ID uniform
wd := bf.double_sort_portfolio(first, second, 0)
```

```
terciles [1, 0, 2, -1, 0, 1, 0, 2]
PRET x ID 5 x 5, ID group 0: 4 long (sum 1.00), 4 short (sum -1.00)
```

The two values of 0.2 land in different terciles, because ties are broken by index.

### Lottery demand

Investors who like lottery-like payoffs overpay for stocks whose recent returns look like lotteries. Two measures predict low returns the next month:
- **MAX** (Bali, Cakici & Whitelaw 2011): the stock's highest daily return in the month. MAX(n) is the mean of its n highest.
- **The salience theory value ST** (Bordalo, Gennaioli & Shleifer 2012; measured by Cosemans & Frehen 2021). Each day is a state with probability 1/S. A day is salient when the stock's return differs from the market's, salience = |r − r̄| / (|r| + |r̄| + θ). The days are ranked by salience and weighted by δ^rank, normalized to mean 1. ST is the salience-weighted mean return minus the plain mean. θ = 0.1 and δ = 0.7, as in the paper.

The procedures:
- `max_daily_return(returns, months, n, min_obs)` computes MAX(n) for every asset and month;
- `salience_theory_value(returns, months, market, theta, delta, min_obs)` computes ST. With `market` nil, the market return is the equally weighted mean of each day's valid returns.

Both need `min_obs` valid days in the month (15 by default). `salience_weights` and `salience_value` compute the weights and ST of a single month.

```odin
rand.reset(9)
D := 20
daily := make([][]f64, 3)
defer {for d in daily {delete(d)}; delete(daily)}
for i in 0 ..< 3 {
	daily[i] = make([]f64, D)
	for t in 0 ..< D {daily[i][t] = 0.01 * rand.float64_normal(0, 1)}
}
daily[1][7] = 0.15 // a lottery-like jump
daily[2][12] = -0.15 // a crash
months := []int{0, D}
mx := bf.max_daily_return(daily, months)
mx3 := bf.max_daily_return(daily, months, n = 3)
st := bf.salience_theory_value(daily, months)
defer {bf.lottery_measure_free(mx); bf.lottery_measure_free(mx3); bf.lottery_measure_free(st)}
```

```
asset 0: MAX 0.0273 MAX(3) 0.0156 ST 0.00104
asset 1: MAX 0.1500 MAX(3) 0.0591 ST 0.02263
asset 2: MAX 0.0160 MAX(3) 0.0124 ST -0.02719
```

A salient upside gives ST > 0, a salient crash ST < 0.

### The capital gains overhang

Grinblatt & Han (2005) estimate holders' average cost basis from turnover. A share last traded n periods ago if it traded then and not since, which gives the reference price R (their eq. 25). Its inputs are the prices and the turnovers (volume / shares outstanding) of the `window` periods before t. Grinblatt & Han use weekly data and a window of 260 weeks.

The capital gains overhang g = (P − R) / P is the holders' average unrealized gain. It uses the price lagged one period. The disposition effect predicts that high-overhang stocks earn more.

- `reference_price(prices, turnover, window)` computes R;
- `capital_gains_overhang(prices, turnover, window)` computes g.

g at s predicts the return from s to s + 1. Both are NaN while the window holds a price ≤ 0 or a turnover outside [0, 1].

```odin
P := []f64{10, 11, 12, 13, 12, 11, 12, 14}
V := []f64{0.1, 0.2, 0.3, 0.1, 0.2, 0.4, 0.1, 0.2}
rp := bf.reference_price(P, V, 4)
g := bf.capital_gains_overhang(P, V, 4)
defer {delete(rp); delete(g)}
```

```
reference [NaN, NaN, NaN, NaN, 11.7679, 11.9651, 11.4954, 11.4817]
overhang [NaN, NaN, NaN, NaN, 0.0948, 0.0029, -0.0450, 0.0432]
```

At t = 4 the weights of the prices 13, 12, 11 and 10 are 0.1, 0.27, 0.126 and 0.0504, normalized: R = 6.43 / 0.5464 = 11.7679.

## 6. The disposition effect in trading records

Odean (1998) counts, on every day an account sells something, each position it held at the start of that day:
- a position sold that day above its average purchase price is a realized gain; below it, a realized loss;
- a position not sold, whose close is above or below the purchase price, is a paper gain or paper loss.

PGR = RG / (RG + PG) and PLR = RL / (RL + PL). PGR > PLR is the disposition effect: selling winners too early and riding losers.

`odean_disposition(trades, prices, sale_days_only)` takes the trades, where `quantity` is positive for buys and negative for sales (long positions only), and the closing prices `[asset][day]`. It returns the counts, PGR, PLR, their difference and its t-statistic. Some details:
- the sale price is the quantity-weighted mean of the day's sales of the asset;
- equal prices count as neither gain nor loss;
- positions bought the same day do not count;
- `sale_days_only = false` counts every day the account holds a position.

```odin
closes := [][]f64 {
	{10, 11, 12, 12, 13, 14}, // asset 0 rises
	{20, 19, 18, 17, 18, 16}, // asset 1 falls
	{30, 31, 29, 30, 32, 33},
}
trades := []bf.Trade {
	{account = 0, asset = 0, day = 0, quantity = 100, price = 10},
	{account = 0, asset = 1, day = 0, quantity = 50, price = 20},
	{account = 0, asset = 2, day = 1, quantity = 10, price = 31},
	{account = 0, asset = 0, day = 2, quantity = -100, price = 12}, // sells the winner
	{account = 0, asset = 2, day = 4, quantity = -10, price = 32},
	{account = 1, asset = 1, day = 0, quantity = 10, price = 20},
	{account = 1, asset = 0, day = 1, quantity = 10, price = 11},
	{account = 1, asset = 1, day = 3, quantity = -10, price = 17}, // sells the loser
}
dc := bf.odean_disposition(trades, closes)
da := bf.odean_disposition(trades, closes, sale_days_only = false)
```

```
sale days: RG 2 PG 1 RL 1 PL 3  PGR 0.6667 PLR 0.2500 difference 0.4167 t 1.20
every day: RG 2 PG 5 RL 1 PL 9  PGR 0.2857 PLR 0.1000
```

On account 0's day 2, for example:
- asset 0 is sold at 12, above its cost of 10: a realized gain;
- asset 1 closes at 18, below its cost of 20: a paper loss;
- asset 2 closes at 29, below 31: a paper loss.

## 7. Herding

When investors follow the market instead of their own information, individual returns cluster around the market return. Rational pricing does the opposite: with different betas, returns spread further apart the larger the market move. Herding therefore shows up as dispersion that rises less than linearly, or falls, in large moves.

`cross_sectional_dispersion(returns, market)` computes, per period:
- CSSD = √(Σ(Rᵢ − R_m)² / (N − 1));
- CSAD = Σ|Rᵢ − R_m| / N;
- the market return and the number of valid returns.

The market return is the equally weighted mean unless one is given; NaN returns are left out.

There are two tests:
- `cck_regression(csad, market, state, nw_lags)` (Chang, Cheng & Khorana 2000) fits CSAD = a + g₁|R_m| + g₂R_m². Rational pricing gives g₂ = 0; herding gives g₂ < 0. `state` restricts the fit to up or down markets.
- `christie_huang_regression(cssd, market, tail, nw_lags)` (Christie & Huang 1995) regresses CSSD on dummies for the `tail` share of periods with the lowest and highest market returns. Herding gives negative coefficients.

Both return the coefficients, standard errors, t-statistics and R². The standard errors are classical, or Newey–West with `nw_lags > 0`.

The example uses two simulated markets of 40 stocks over 2000 days. In the second, returns are pulled toward the market the more, the larger |R_m|:

```odin
rand.reset(11)
NA, T := 40, 2000
for herding in ([]bool{false, true}) {
	rets := make([][]f64, NA)
	defer {for x in rets {delete(x)}; delete(rets)}
	beta := make([]f64, NA)
	defer delete(beta)
	for i in 0 ..< NA {beta[i] = 0.5 + rand.float64(); rets[i] = make([]f64, T)}
	for t in 0 ..< T {
		m := 0.01 * rand.float64_normal(0, 1)
		// herding: returns pulled toward the market, the more so the larger |m|
		h := min(1, abs(m) / 0.03) if herding else 0
		for i in 0 ..< NA {
			rets[i][t] = ((1 - h) * beta[i] + h) * m + 0.005 * rand.float64_normal(0, 1)
		}
	}
	d := bf.cross_sectional_dispersion(rets)
	defer bf.herding_dispersion_free(&d)
	cck := bf.cck_regression(d.csad, d.market, nw_lags = 5)
	ch := bf.christie_huang_regression(d.cssd, d.market, 0.05)
}
```

```
herding false: CCK g1 0.0084 (t 1.5) g2 4.6334 (t 17.3); CH b_L 0.00206 (t 25.1) b_U 0.00229 (t 27.8)
herding true: CCK g1 0.0542 (t 10.7) g2 -1.5696 (t -7.1); CH b_L 0.00023 (t 3.8) b_U 0.00015 (t 2.4)
```

CCK detects this herding (g₂ < 0). Christie–Huang only shows smaller positive tail coefficients: it detects herding only when it is strong enough to reduce dispersion in the tails.

### Real data

`net.read_yahoo_panel(tickers, interval, range)` loads adjusted closes and dollar volumes of several tickers from Yahoo, on the dates all of them have. It returns `(panel, ok, failed)`:
- `ok` is false when a ticker cannot be loaded, and `failed` is then its index;
- if all tickers loaded but share no date, `failed` is −1.

Free the panel with `net.yahoo_panel_free`. The 24 large caps of the real-data tests, over ten years of daily data:

```odin
panel, ok, failed := net.read_yahoo_panel(tickers, .Daily, .TenYears)
defer net.yahoo_panel_free(&panel)
if !ok {
	fmt.printf("no data for %s\n", tickers[failed] if failed >= 0 else "a common date")
	return false
}
N, D := panel.n_tickers, len(panel.dates)
// daily returns [asset][day]
rets := make([][]f64, N)
defer {for r in rets {delete(r)}; delete(rets)}
for i in 0 ..< N {
	rets[i] = make([]f64, D - 1)
	for t in 1 ..< D {rets[i][t - 1] = panel.adj_close[t * N + i] / panel.adj_close[(t - 1) * N + i] - 1}
}
d := bf.cross_sectional_dispersion(rets)
defer bf.herding_dispersion_free(&d)
for state in bf.Herding_Market_State {
	c := bf.cck_regression(d.csad, d.market, state, nw_lags = 10)
}
ch := bf.christie_huang_regression(d.cssd, d.market, 0.01, nw_lags = 10)
```

```
24 tickers, 2512 common days from 2016-11-14 to 2026-11-11
CCK All  (2511 days): g1 0.2516 (t   9.8)  g2  0.0331 (t   0.1)  R2 0.196
CCK Up   (1413 days): g1 0.2761 (t   7.1)  g2  0.2228 (t   0.4)  R2 0.231
CCK Down (1098 days): g1 0.2286 (t   6.5)  g2 -0.1881 (t  -0.5)  R2 0.162
Christie-Huang 1%: b_L 0.0111 (t 3.9)  b_U 0.0111 (t 4.2)
```

The output is live data, so your numbers will differ. Neither test finds herding among these large caps: g₂ is insignificant and the tail coefficients are positive.

## 8. Investor sentiment

Sentiment is not observed directly. Baker & Wurgler (2006, 2007) take it as the common component of several imperfect proxies:
- `sentiment_index(proxies, use, sign_row)` computes the first principal component of the standardized proxies, scaled to mean 0 and unit variance: SENT = Σ cⱼ zⱼ (their eq. 2).
- The rows used are those with `use[t]` (all rows if nil) where every proxy is finite.
- The sign makes the index positive at `sign_row`; with −1, the loading largest in absolute value is positive instead.
- The result has the index, the coefficients cⱼ, the unit-length loadings, the share of variance explained and the number of rows.

```odin
rand.reset(13)
n := 400
latent := make([]f64, n)
defer delete(latent)
proxies := make([][]f64, 4)
defer {for x in proxies {delete(x)}; delete(proxies)}
load := []f64{1, 0.8, -0.6, 0.5} // the third proxy falls with sentiment
for &x in proxies {x = make([]f64, n)}
s := 0.0
for t in 0 ..< n {
	s = 0.9 * s + math.sqrt(1 - 0.81) * rand.float64_normal(0, 1)
	latent[t] = s
	for j in 0 ..< 4 {proxies[j][t] = 5 + load[j] * s + 0.7 * rand.float64_normal(0, 1)}
}
si := bf.sentiment_index(proxies)
defer bf.sentiment_index_free(&si)
```

```
loadings [0.543, 0.523, -0.479, 0.450], explained 0.615, n 400, index mean 1.2e-15 sd 1.0000, corr with latent 0.833
```

The other procedures:
- `orthogonalize(y, controls, use)`: the residuals of a proxy on macro controls. An index built from residuals is free of the business cycle (SENT⊥, eq. 3).
- `sentiment_select_lags(current, lagged, use)`: the 2006 first stage. It takes the first component of all proxies and their lags, and picks the version of each proxy that correlates more with it.
- Series helpers: `lag_series`, `trailing_sum`, `trailing_weighted_mean` and `log_ma_deviation`.
- `baker_wurgler_sentiment(data, first, last, sign_row)`: the maintained monthly index (SENT and SENT⊥), recomputed from the raw series of Wurgler's SENTIMENT.xlsx.

`sentiment_conditional_returns(returns, sentiment, nw_lags)` gives the mean returns after positive and after non-positive sentiment, their difference and the predictive slope r_t = a + d·S_t (2006, Tables 3 and 5), with t-statistics. Pass the sentiment known before each return, for example the previous month's index:

```odin
// returns that fall after high sentiment: r_t = 0.5 - 0.3 S_{t-1} + noise
rr := make([]f64, n)
lagged := bf.lag_series(si.index, 1)
defer {delete(rr); delete(lagged)}
for t in 0 ..< n {rr[t] = 0.5 - 0.3 * (lagged[t] if t > 0 else 0) + rand.float64_normal(0, 1)}
cr := bf.sentiment_conditional_returns(rr, lagged, 6)
```

```
after high 0.166 (193), after low 0.747 (206), difference -0.580 (t -6.3), slope -0.290 (t -6.4)
```

## 9. Equilibrium models

### Barberis, Shleifer & Vishny (1998)

Earnings follow a random walk, but the representative investor believes they switch between two regimes:
- a mean-reverting regime: the next shock has the same sign with probability π_L < ½;
- a trending regime: the same sign with probability π_H > ½.

q is the investor's probability of the mean-reverting regime. It falls after shocks of the same sign and rises after a reversal. The price is P = N/δ + y(p₁ − p₂q) (Proposition 1). Prices therefore underreact to a single shock and overreact to a series of shocks.

`BSV_1998` holds the paper's parameters (π_L = 1/3, π_H = 3/4, λ₁ = 0.1, λ₂ = 0.3). The paper does not state δ; the default is 0.1.

```odin
p := bf.BSV_1998
shocks := []f64{1, 1, 1, 1, -1, 1, -1, 1, 1, 1}
path := bf.bsv_simulate_path(shocks, N0 = 10, q0 = 0.5, p = p)
defer bf.bsv_path_free(&path)
p1, p2 := bf.bsv_price_coefficients(p)
ql, qh := bf.bsv_belief_bounds(p)
c := bf.bsv_sufficient_conditions(p)
rational := bf.BSV_Params{0.5, 0.5, 0.1, 0.3, 0.1}
r1, r2 := bf.bsv_price_coefficients(rational)
```

```
beliefs q [0.5000, 0.4000, 0.3429, 0.3126, 0.7173, 0.8784, 0.9273, 0.9408, 0.7393, 0.5631]
prices [111.043, 121.616, 131.944, 142.117, 130.203, 138.873, 131.408, 138.515, 149.671, 160.681]
p1 3.9100 p2 5.7346, q in [0.2821, 0.9458]
k_low 0.4921 k_high 0.8426: overreaction true underreaction true
pi_L = pi_H = 1/2: p1 0.00e+00 p2 -0.00e+00 (P = N / delta)
```

The belief bounds are BSV's 0.28 and 0.95. `bsv_sufficient_conditions` checks Proposition 2's sufficient conditions: p₁ > k_low·p₂ guarantees overreaction and p₁ < k_high·p₂ underreaction. The other procedures:
- `bsv_update_belief` and `bsv_beliefs`: the belief updates along a series of shocks;
- `bsv_price`: P from N, y and q;
- `bsv_reaction_gap`: the appendix's f(q).

### Hong & Stein (1999)

News about a risky asset diffuses slowly across z groups of newswatchers, who do not condition on prices. On their own, prices therefore underreact. Momentum traders enter every period, hold for j periods and buy φ times the last price change. Their orders turn the underreaction into an overshoot:

dP_t = (e_t + … + e_{t−z+1}) / z + φ dP_{t−1} − φ dP_{t−j−1}

In equilibrium, φ is the momentum traders' best response to the dynamics it creates (eq. 7). `hong_stein_equilibrium` returns the smallest stationary fixed point, as in the paper's tables. `HONG_STEIN_1999` is their base case: z = j = 12, risk tolerance γ = 1/3, news sd 0.5.

```odin
hs := bf.HONG_STEIN_1999
phi, ok := bf.hong_stein_equilibrium(hs)
h := bf.hong_stein_impulse(phi, hs, 36) // cumulative sums: the impulse response
defer delete(h)
ac := bf.hong_stein_autocovariances(phi, hs, 13)
defer delete(ac)
rn := hs
rn.gamma = math.inf_f64(1)
phi_rn, ok_rn := bf.hong_stein_equilibrium(rn)
dP := bf.hong_stein_simulate(e, phi, hs)
```

```
z = j = 12, gamma = 1/3: equilibrium phi 0.2605 (ok true), best response 0.2605, stationary true
cumulative impulse response: lag 11 1.3126, peak 1.3420 at lag 12, lag 35 1.0010
return autocorrelations lag 1 0.9466, lag 6 0.3846, lag 13 -0.3385; pricing error sd 0.9365 (newswatchers only 0.9373)
risk-neutral momentum traders: phi 0.4609 (ok true)
simulated dP [-0.0570, -0.0761, -0.0278, -0.0219, 0.0019] from news [-0.6845, -0.0507, 0.6391, -0.0802, 0.2676]
```

The price overshoots the fundamental (the cumulative response peaks above 1, at lag j, per Proposition 1) and then reverses: returns are positively autocorrelated at short lags and negatively at long ones. The other procedures:
- `hong_stein_best_response(phi, p)`: the momentum traders' best response when the market runs at φ;
- `hong_stein_stationary` and `hong_stein_ar_radius`: the stationarity check (the spectral radius of the AR part must be below 1);
- `hong_stein_pricing_error_sd`: the sd of P − D_{t+z−1}, the price's distance from the rational benchmark.

### Noise trader risk (De Long, Shleifer, Summers & Waldmann 1990)

Overlapping generations hold a safe asset and an unsafe one. A share μ of the young are noise traders who misjudge the unsafe asset's expected price by ρ_t, with mean ρ* and sd σ. Their misperception creates risk that sophisticated investors must bear, so the unsafe asset sells below its fundamental value of 1 unless the noise traders' bullishness outweighs that risk.

`Noise_Trader_Params` also covers:
- persistent misperceptions (AR(1) with coefficient φ, their Section III);
- dividend risk σ_ε (their Appendix I).

The procedures:
- prices: `dssw_price`, `dssw_mean_price`, `dssw_price_slope`, `dssw_price_variance`, `dssw_price_risk`;
- demands and returns: `dssw_demands`, `dssw_expected_excess_return`;
- the noise traders' excess return over the sophisticated investors: `dssw_conditional_return_gap`, `dssw_expected_return_gap`;
- the shares at which that excess return is zero: `dssw_critical_shares`.

```odin
nt := bf.Noise_Trader_Params{r = 0.05, gamma = 90, mu = 0.3, rho_star = 0.2, sigma = 0.05}
si, ni := bf.dssw_demands(nt.rho_star, nt)
lo, hi, n := bf.dssw_critical_shares(nt)
ft := nt
ft.sigma_eps = 0.008
lo, hi, n = bf.dssw_critical_shares(ft)
// imitation: 50,000 generations, alpha 0.002
sim := bf.dssw_simulate(rho, eps, q, alpha = 0.002)
defer bf.dssw_path_free(&sim)
```

```
mean price 1.4653, slope 0.2857, variance 2.04e-04, V 2.04e-04
at rho = rho*: price 1.4653, demands sophisticated -0.6333 noise 4.8111 (market 1.0000)
E dR -0.1471, E_t dR at rho = 0.3: -0.4350
critical shares: 2 roots 0.0000 0.5206 (mu* = 0.5206)
with dividend risk 0.008: 2 roots mu_L 0.0615 mu_H 0.4592
imitation from mu = 0.01: mean share over the last 10,000 generations 0.0618
imitation from mu = 0.30: mean share over the last 10,000 generations 0.0618
```

How imitation plays out:
- With `alpha > 0`, `dssw_simulate` lets the noise traders' share grow with their realized excess return (eq. 20).
- Without dividend risk, the share μ* = 0.5206 is unstable: noise traders die out below it and take over above it (eq. 21).
- With dividend risk, a stable share μ_L appears. Both starting shares settle there, as Appendix I predicts.

### The X-CAPM (Barberis, Greenwood, Jin & Shleifer 2015)

The X-CAPM is set in continuous time, with two kinds of traders:
- extrapolators expect the price to change at λ₀ + λ₁S, where sentiment S is an exponentially weighted average of past price changes;
- rational traders, a share μ, know the true process.

The equilibrium price is P = A + B·S + D/r (Proposition 1), and sentiment is an Ornstein–Uhlenbeck process. `xcapm_solve` solves the paper's Appendix A system. `XCAPM_2015` holds the parameters of Table 2 with β = 0.5 and μ = 0.25, the case of their eq. 28.

```odin
xp := bf.XCAPM_2015
s, sok := bf.xcapm_solve(xp)
res := bf.xcapm_residuals(s, xp) // all of Appendix A's conditions
v1 := bf.xcapm_price_change_variance(s, xp, 1)
cv := bf.xcapm_price_change_covariance(s, xp, 1, 1, 2)
rat := xp
rat.mu = 1
sr, _ := bf.xcapm_solve(rat)
xs := bf.xcapm_simulate(s, xp, D0 = 5, S0 = xp.g_D / xp.r, dt = 0.25, z = z) // z: 2 normals a step
defer bf.xcapm_path_free(&xs)
```

```
ok true: A -117.0377 B 0.9874 k 0.9876 sigma_P 19.7516 sigma_S 9.8758
demands: extrapolators 1.5375 + 0.5125 S, rational 15.3875 + -1.5375 S
largest Appendix A residual 1.3e-14
1-year expected price change at S = 0.3: rational 3.0534, extrapolators 0.3000
D/r - P predicts 1-year changes with slope 0.6275; P - D/r autocorrelation 0.3725; its sd 6.9386
sd of annual price changes 16.8627 (rational 10.0000), autocorrelation of annual changes -0.2034
mu = 1: A 30.0000 (g_D - gamma sigma_D^2 Q) / r^2 = 30.0000, B 0.0000
quarterly path: D [5.0000, 4.8520, 4.8317, 4.8623, 4.6090]
  S [2.0000, -3.7036, -3.6927, -1.5164, -10.1710]
  P [84.937, 73.387, 72.586, 75.958, 57.280]
```

These are the propositions' predictions:
- the dividend-price ratio predicts returns (`xcapm_dp_coefficient`);
- P − D/r is persistent (`xcapm_pd_autocorrelation`, `xcapm_pd_sd`);
- prices are excessively volatile (`xcapm_price_change_variance`);
- price changes are negatively autocorrelated (`xcapm_price_change_covariance`).

With μ = 1 the model is the rational benchmark of Corollary 1. `xcapm_expected_change_rational` and `xcapm_expected_change_extrapolator` give the two groups' expectations (eqs. 25, 26). `xcapm_simulate` draws an exact discretization of the joint process of D and S.

## 10. Heuristics

Two procedures are not from the literature and are labelled as such in the source. Use them as illustrations, not as estimated models.

- **`behavioral_kalman_filter(observations, anchor, alpha, process_var, measurement_var, anchors)`** is a local-level Kalman filter with anchoring. When an observation would push the estimate up while the estimate is still below the anchor, the gain is multiplied by 1 − α. The anchor is compared with the estimate, not with the observation, so an observation above the anchor is still damped while the estimate lags below it. α = 0 is the standard filter.

  ```odin
  obs := []f64{100, 101, 103, 106, 104, 108, 110, 109}
  for alpha in ([]f64{0, 0.8}) {
  	est, gains, vars := bf.behavioral_kalman_filter(obs, anchor = 107, alpha = alpha, process_var = 1, measurement_var = 2)
  	defer {delete(est); delete(gains); delete(vars)}
  }
  ```

  ```
  alpha 0.0: [100.00, 100.50, 101.75, 103.88, 103.94, 105.97, 107.98, 108.49]
    gains [0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500]
  alpha 0.8: [100.00, 100.10, 100.43, 101.11, 101.47, 102.32, 103.32, 104.07]
    gains [0.500, 0.100, 0.114, 0.122, 0.126, 0.129, 0.131, 0.132]
  ```

- **`generate_behavioral_signal(params, hist_vol, avg_implied_vol)`** maps calibrated CPT parameters and the gap between implied and historical volatility to a regime, an action and a conviction. It uses hand-set thresholds on γ⁻ (below 0.55: "panic", above 0.75: "complacency") and on the volatility gap.

  ```
  signal: Panic (Crashophobia) / SHORT VOLATILITY (Sell OTM Puts / Short Strangle), conviction 0.60, bvrp 0.09
  ```

## 11. Validation

The tests are registered in `main.odin` and listed in [the test catalog](test_catalog.md), section 11. The checks against papers and reference libraries:

| Area | Reference | Test |
|---|---|---|
| CPT option prices | 72 writer / holder, call / put, segregated / aggregated prices (TK92, moderate, Prelec) against a SciPy implementation that reproduces Nardon & Pianca's Tables 3–5; Table 6's holder–writer intervals; rational = Black–Scholes; behavioral Heston = Heston when rational | `prospect_reference_test` |
| CPT and mental-account portfolios | CPT value vs NumPy; the optimum vs the best point of a 1/600 simplex grid; Das et al. Tables 1–2 (implied γ, weights, 44 shortfall probabilities); the long-only case vs SciPy | `behavioral_portfolio_reference_test` |
| TK | Barberis et al.'s eq. (8), transcribed literally | `tk_value_reference_test` |
| 52-week high, overlapping returns | pandas rolling max (NaN, zero, negative prices, a delisting); NumPy | `anchoring_reference_test` |
| Momentum, reversal, frog in the pan | pandas / NumPy for past returns, ID, sorts; Da et al.'s Appendix A covariance and simulation table | `momentum_reference_test` |
| MAX, salience | Cosemans & Frehen's example; NumPy / SciPy on a panel with gaps and ties | `lottery_reference_test` |
| Capital gains overhang, PGR / PLR | NumPy (eq. 25); a separate Python implementation of Odean's counts on 84 generated trades | `disposition_reference_test` |
| Herding | CSSD / CSAD = NumPy; both regressions, classical and Newey–West = statsmodels | `herding_reference_test` |
| Sentiment index | the published SENT and SENT⊥ recomputed from Wurgler's raw series, to the single precision the file stores | `investor_sentiment_reference_test` |
| BSV | Table 1 beliefs, the belief bounds; p₁, p₂ = a direct series (NumPy); Proposition 2 on a grid | `bsv_reference_test` |
| Hong–Stein | Tables AI–AIII (18 parameter sets) to the printed decimals; autocovariances = statsmodels `arma_acovf` | `hong_stein_reference_test` |
| DSSW | the price, its slope and the expected return gap = a SymPy solution; μ*, μ_L, μ_H | `noise_trader_reference_test` |
| X-CAPM | eq. 28's 13 parameters as printed and = a separate Python solution; Tables 3–6 on 9 (μ, β) pairs | `xcapm_reference_test` |

The simulation tests then check each model's predictions on synthetic markets and a control market:
- the signal predicts returns where the model says it should, and not in the control;
- the real-data tests apply each measure to the 24 large caps.

Real-data tests load their data with the library loaders (`net.read_yahoo`, `net.read_yahoo_panel`, `net.download_file`). If the data cannot be loaded, the test reports a failure through `fail_missing_data` and the rest of the suite still runs. They never fall back to synthetic data.

For this guide, the values in sections 2–4 were also recomputed independently:
- the value function, weights and `cpt_value` examples with NumPy, to the printed decimals;
- the writer's segregated call (36.9381) by SciPy quadrature of Nardon & Pianca's eq. 16 (36.9372). The difference is the 2000-cell grid's error;
- the infeasible account's weights and 5% quantile with SciPy SLSQP.

## 12. Quick reference

### Prospect theory and option prices

| Procedure | Purpose |
|---|---|
| `prospect_value(x, p)` | value function v(x) |
| `prospect_weight(prob, gamma, kind)` | weighting function w(p), Tversky–Kahneman or Prelec |
| `prospect_gamma_minus(p)` | γ⁻ (γ when 0) |
| `cpt_value(outcomes, p)` | CPT value of equally likely outcomes |
| `tk_value(returns, benchmark, window, p)` | Barberis–Mukherjee–Wang TK per period |
| `prospect_theory_price(S, K, T, r, mu, sigma, p, type, side, frame, n_points)` | CPT option price, log-normal |
| `prospect_lognormal_grid` / `prospect_price_precomputed` / `prospect_option_value` | price on a precomputed distribution; prospect value at a premium |
| `calibrate_prospect_theory(chain, S, r, mu, sigma, setup, types)` | fit CPT parameters to option prices |
| `behavioral_heston_price` / `behavioral_heston_precompute` / `calibrate_behavioral_heston` | the same on a Heston distribution |
| `extract_heston_cdf_grid` | Heston CDF by Gil-Pelaez inversion |

### Portfolios

| Procedure | Purpose |
|---|---|
| `prospect_portfolio_optimize(&R, p, reference, constraints)` | CPT-optimal weights |
| `prospect_metrics(w, &R, p, reference)` | CPT value, mean, sd, P(loss) |
| `mental_account_portfolio(mu, &cov, account, long_only)` | one mental account |
| `bpt_optimize(mu, &cov, accounts, long_only)` | all accounts and the aggregate |
| `shortfall_probability(w, mu, &cov, H)` | P(r < H), normal returns |

### Signals and sorts

| Procedure | Purpose |
|---|---|
| `compute_anchoring_signal(prices, lookback)` | price / 52-week high |
| `past_return(returns, lookback, skip, market, kind)` | formation-period return |
| `information_discreteness(daily, months, lookback, skip, kind)` / `information_discreteness_value` | ID and PRET |
| `max_daily_return(returns, months, n, min_obs)` | MAX(n) |
| `salience_theory_value(returns, months, market, theta, delta, min_obs)` / `salience_value` / `salience_weights` | salience theory ST |
| `reference_price` / `capital_gains_overhang(prices, turnover, window)` | Grinblatt–Han |
| `long_short_portfolio(signal, fraction)` | top minus bottom fraction |
| `quantile_groups` / `double_sort` / `double_sort_portfolio` | quantile and double sorts |
| `overlapping_strategy_returns(weights, returns, hold, skip)` | Jegadeesh–Titman holding |

### Trading records, herding, sentiment

| Procedure | Purpose |
|---|---|
| `odean_disposition(trades, prices, sale_days_only)` | PGR, PLR, difference, t |
| `cross_sectional_dispersion(returns, market)` | CSSD, CSAD, market return |
| `cck_regression(csad, market, state, nw_lags)` | Chang–Cheng–Khorana |
| `christie_huang_regression(cssd, market, tail, nw_lags)` | Christie–Huang |
| `sentiment_index(proxies, use, sign_row)` | first principal component, unit variance |
| `sentiment_select_lags` / `orthogonalize` | 2006 first stage; residuals on controls |
| `baker_wurgler_sentiment(data, first, last, sign_row)` | the maintained SENT and SENT⊥ |
| `sentiment_conditional_returns(returns, sentiment, nw_lags)` | returns after high / low sentiment |
| `lag_series` / `trailing_sum` / `trailing_weighted_mean` / `log_ma_deviation` | series helpers |

### Equilibrium models

| Model | Procedures |
|---|---|
| BSV | `bsv_update_belief`, `bsv_beliefs`, `bsv_price_coefficients`, `bsv_price`, `bsv_simulate_path`, `bsv_belief_bounds`, `bsv_reaction_gap`, `bsv_sufficient_conditions` |
| Hong–Stein | `hong_stein_equilibrium`, `hong_stein_best_response`, `hong_stein_impulse`, `hong_stein_autocovariances`, `hong_stein_pricing_error_sd`, `hong_stein_ar_radius`, `hong_stein_stationary`, `hong_stein_simulate` |
| DSSW | `dssw_price`, `dssw_mean_price`, `dssw_price_slope`, `dssw_price_risk`, `dssw_price_variance`, `dssw_demands`, `dssw_expected_excess_return`, `dssw_conditional_return_gap`, `dssw_expected_return_gap`, `dssw_critical_shares`, `dssw_simulate` |
| X-CAPM | `xcapm_solve`, `xcapm_residuals`, `xcapm_price`, `xcapm_expected_change_rational`, `xcapm_expected_change_extrapolator`, `xcapm_dp_coefficient`, `xcapm_pd_autocorrelation`, `xcapm_pd_sd`, `xcapm_price_change_variance`, `xcapm_price_change_covariance`, `xcapm_simulate` |

### Heuristics

| Procedure | Purpose |
|---|---|
| `behavioral_kalman_filter` | Kalman filter with damped upward updates below an anchor |
| `generate_behavioral_signal` | regime and action from γ⁻ and the volatility gap (hand-set thresholds) |
