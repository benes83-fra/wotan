# The `ml_finance` package, part 4: derivatives, execution and synthetic data

This is the fourth part of the guide to `wotan/ml_finance`. [Part 1](ml_finance_methods_guide.md) covers labels, feature importance and backtest statistics, [part 2](ml_finance_asset_pricing_guide.md) asset-pricing models, [part 3](ml_finance_forecasting_guide.md) forecasting, and [part 5](ml_finance_rl_integration_guide.md) reinforcement learning and integration. This part covers:
- **optimal execution:** the Almgren–Chriss liquidation schedule, and Avellaneda–Stoikov market-making quotes;
- **the deep BSDE method:** option prices in many dimensions from a learned hedge;
- **deep calibration:** a network that maps a Heston implied-volatility surface to the model's parameters in one forward pass;
- **deep hedging:** a network that learns the hedge directly, with transaction costs and a risk measure;
- **TimeGAN:** synthetic time series that keep the dynamics of real ones.

Every example was compiled and run against the current code, and the output shown is the real output. A debug build with a tracking allocator reports no leaks for any of them. Section 7 lists the comparisons with independent solutions, QuantLib, SciPy and PyTorch.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [Optimal execution and market making](#2-optimal-execution-and-market-making)
3. [The deep BSDE method](#3-the-deep-bsde-method)
4. [Deep calibration of the Heston model](#4-deep-calibration-of-the-heston-model)
5. [Deep hedging](#5-deep-hedging)
6. [TimeGAN](#6-timegan)
7. [Validation](#7-validation)
8. [Quick reference](#8-quick-reference)

## 1. Setup and conventions

```odin
import fin "wotan/finance"
import l "wotan/linalg"
import mlf "wotan/ml_finance"
import nn "wotan/nn"
import optim "wotan/optimize"
import t "wotan/tensor"
```

Section 2 is closed-form. Sections 3–6 train networks, as in [part 3](ml_finance_forecasting_guide.md#1-setup-and-conventions):
- each training step runs inside a `t.Graph_Scope`;
- the models register their parameters with an `nn.Adam`;
- the inputs should be of order 1.

The deep hedger and TimeGAN own their scope and open it in their step procedures. The deep BSDE and calibration losses must be called between `t.graph_begin` and `t.graph_end`.

## 2. Optimal execution and market making

### Almgren–Chriss

A trader sells X shares over N periods of length τ = T/N. Selling n_k shares in period k moves the price in two ways:
- **permanently** by −γ n_k;
- **temporarily** by −(η/τ) n_k, on that trade only.

The implementation shortfall then has mean E = γX²/2 + (η̃/τ) Σ n_k², where η̃ = η − γτ/2, and variance V = σ²τ Σ x_k², where x_k are the holdings.

`almgren_chriss_solve` minimizes E + λV exactly (Almgren & Chriss 2000):
- **the solution:** x_j = X sinh(κ(T − t_j)) / sinh(κT), with cosh(κτ) = 1 + λσ²τ² / (2η̃);
- **λ = 0** gives TWAP, equal slices;
- **invalid input:** if η̃ ≤ 0 (the cost is not convex), the trajectory is empty and the costs are NaN.

The example is the paper's own: 10⁶ shares over 5 days in daily steps, σ = 0.95 $/share/day^½, η = 2.5e-6, γ = 2.5e-7 and λ = 1e-6.

```odin
p := mlf.AlmgrenChrissParams {
	initial_shares = 1e6,
	n_steps        = 5,
	T              = 5,
	sigma          = 0.95,
	eta            = 2.5e-6,
	gamma          = 2.5e-7,
	lambda         = 1e-6,
}
tr := mlf.almgren_chriss_solve(p)
defer mlf.almgren_chriss_free(&tr)
fmt.printf("holdings %.0f\ntrades   %.0f\nE %.0f, sd %.0f, E + lambda V %.0f\n", tr.shares_held, tr.trade_sizes,
	tr.expected_cost, math.sqrt(tr.variance), tr.expected_cost + p.lambda * tr.variance)
for lam in ([]f64{0, 1e-7, 1e-6, 1e-5}) { // the efficient frontier
	q := p
	q.lambda = lam
	x := mlf.almgren_chriss_solve(q)
	defer mlf.almgren_chriss_free(&x)
	fmt.printf("lambda %.0e: first trade %7.0f, E %7.0f, sd %7.0f\n", lam, x.trade_sizes[0], x.expected_cost,
		math.sqrt(x.variance))
}
bad := p
bad.eta = 1e-7 // eta - gamma tau / 2 < 0: not convex
b := mlf.almgren_chriss_solve(bad)
defer mlf.almgren_chriss_free(&b)
fmt.printf("eta too small: %d steps, E %v\n", len(b.trade_sizes), b.expected_cost)
```

```
holdings [1000000, 541956, 289854, 147897, 62142, 0]
trades   [458044, 252101, 141957, 85756, 62142]
E 848727, sd 603431, E + lambda V 1212856
lambda 0e+00: first trade  200000, E  600000, sd 1040673
lambda 1e-07: first trade  242118, E  607557, sd  961623
lambda 1e-06: first trade  458044, E  848727, sd  603431
lambda 1e-05: first trade  822132, E 1782711, sd  171712
eta too small: 0 steps, E NaN
```

A risk-neutral trader (λ = 0) sells evenly and pays the least impact, $600,000 in expectation. The cost of that is the largest uncertainty: a standard deviation of $1.04m. Higher risk aversion front-loads the sales. At λ = 1e-5, 82% goes on the first day, which cuts the standard deviation to $172,000 but triples the expected cost.

For impact that is not linear, see the square-root cost model in [part 1](ml_finance_methods_guide.md#9-transaction-costs).

### Avellaneda–Stoikov

A market maker with inventory q quotes around a *reservation price* r = s − qγσ²(T − t), not around the mid s (Avellaneda & Stoikov 2008):
- **the spread** is δ_a + δ_b = γσ²(T − t) + (2/γ) ln(1 + γ/k);
- **γ** is the market maker's risk aversion;
- **k** is how fast the probability of a fill falls with the distance from the mid.

```odin
as := mlf.AvellanedaStoikovParams{sigma = 2, gamma = 0.1, k = 1.5, T = 1}
for q in ([]f64{-5, 0, 5}) {
	for tt in ([]f64{0, 0.5, 1}) {
		qt := mlf.avellaneda_stoikov_quote(100, q, tt, as) // mid 100
		fmt.printf("q %+2.0f t %.1f: reservation %.4f, bid %.4f ask %.4f, spread %.4f\n", q, tt, qt.reservation_price,
			qt.bid_quote, qt.ask_quote, qt.optimal_spread)
	}
}
```

```
q -5 t 0.0: reservation 102.0000, bid 101.1546 ask 102.8454, spread 1.6908
q -5 t 0.5: reservation 101.0000, bid 100.2546 ask 101.7454, spread 1.4908
q -5 t 1.0: reservation 100.0000, bid 99.3546 ask 100.6454, spread 1.2908
q +0 t 0.0: reservation 100.0000, bid 99.1546 ask 100.8454, spread 1.6908
q +0 t 0.5: reservation 100.0000, bid 99.2546 ask 100.7454, spread 1.4908
q +0 t 1.0: reservation 100.0000, bid 99.3546 ask 100.6454, spread 1.2908
q +5 t 0.0: reservation 98.0000, bid 97.1546 ask 98.8454, spread 1.6908
q +5 t 0.5: reservation 99.0000, bid 98.2546 ask 99.7454, spread 1.4908
q +5 t 1.0: reservation 100.0000, bid 99.3546 ask 100.6454, spread 1.2908
```

How to read the quotes:
- **Inventory shifts both quotes.** Long 5 units, the market maker quotes 2 below the mid, so that it is more likely to sell than buy. Short 5, it quotes 2 above.
- **The skew and the risk part of the spread shrink toward the horizon.** At t = T only the liquidity term, (2/γ) ln(1 + γ/k) = 1.29, is left. Past T, the time left is taken as 0.
- **`ASQuotes`** also holds the bid and ask distances from the mid.

## 3. The deep BSDE method

Han, Jentzen & E (2018) price an option by learning its hedge. Under the pricing measure, d assets follow dX = rX dt + σX dW. The option value Y_t = u(t, X_t) then satisfies

  dY = rY dt + Z·dW, with Y_T = g(X_T), where Z = σX ∇u.

How the model trains:
- **Parameters.** The model learns the starting value Y_0, one scalar, and a network Z_t = z_net(t, ln(X_t / S_0)).
- **Simulation.** Each path is simulated forward, and the loss is E[(Y_T − g(X_T))²].
- **At the optimum,** Y_0 is the price and Z the hedge. The loss is then the variance the hedge leaves: the discretization error of rebalancing at N dates.

The payoff here is a call on the maximum of the assets, max(max_j X_T^j − K, 0); with d = 1 it is a European call.

The asset steps are simulated exactly (log-normal) and the linear driver is integrated exactly, so the time step affects only the hedge. That error has mean zero.

```odin
S_0 := []f64{100}
sigma := []f64{0.2}
model := mlf.deep_bsde_model_new(5, 1, 1.0, 20, 32, 2) // Y_0 start 5, d = 1, T = 1, 20 steps, 2 x 32 hidden
defer mlf.deep_bsde_model_free(&model)
opt := nn.adam_new(1e-3)
defer nn.adam_free(&opt)
nn.adam_add_param_with_lr(&opt, model.y0, 0.1) // Y_0 moves ~ the learning rate per step: give it 100x
nn.sequential_add_to_adam(model.z_net, &opt)
epochs := 3000
schedule := optim.lr_step_decay(1e-3, epochs / 3, 0.2) // both rates x 0.2 at 1/3 and 2/3
scope: t.Graph_Scope
defer t.graph_scope_destroy(&scope)
for e in 0 ..< epochs {
	t.graph_begin(&scope)
	nn.adam_zero_grad(&opt)
	loss := mlf.deep_bsde_bs_loss(&model, S_0, 100, 0.05, sigma, 64, 20) // K 100, r 5%, 64 paths, 20 steps
	t.tensor_backward(loss)
	nn.adam_step_scheduled(&opt, schedule, e)
	t.graph_end(&scope)
}
fmt.println(mlf.deep_bsde_y0(&model))
```

```
  epoch    0 loss  365.200 Y_0 5.1000
  epoch 1000 loss    6.250 Y_0 10.4679
  epoch 2000 loss    5.924 Y_0 10.4013
1 asset: Y_0 10.4615 (Black-Scholes 10.4506); loss 5.00 = 2.1% of Var[payoff] 239.45
  epoch    0 loss  345.700 Y_0 20.1000
  epoch 1000 loss   14.743 Y_0 29.7397
  epoch 2000 loss   17.662 Y_0 29.8049
max-call on 5 assets: Y_0 29.7541; loss 19.20
```

How to read the results:
- **One asset.** Y_0 lands within 0.1% of Black–Scholes. The loss shows the hedge removes 98% of the payoff's variance, with 20 rebalancing dates.
- **Five assets.** The second run uses 64 hidden units and a learning rate of 3e-3, starting from Y_0 = 20. Its price, 29.754, matches the exact value e^{−rT} ∫_K^∞ (1 − F(x)⁵) dx = 29.752, with F the log-normal distribution function of one asset. That is the same method in 5 dimensions at the cost of 1.
- **Run time.** Both runs together take 17 s.

`deep_bsde_bs_forward` returns Y_T and X_T, for other payoffs. Every intermediate tensor is allocated in the active graph scope, so call it inside one.

## 4. Deep calibration of the Heston model

Calibrating Heston to an implied-volatility surface is an iterative least-squares fit: about 20 ms per surface with Levenberg–Marquardt. Deep calibration learns the inverse map, surface → parameters, once, offline (Hernandez 2016; implied-volatility inputs as Horvath, Muguruza & Tomas 2021). After that, a calibration is one forward pass.

The setup:
- **The grid** is 5 forward moneyness levels K/F (0.8–1.2) × 4 expiries (0.25–2 years), stored expiry-major. In forward moneyness, Heston implied volatilities depend on neither the spot nor the rate, so one network serves any S and r.
- **Training data.** `generate_heston_calibration_dataset(n)` draws parameters uniformly from a box: v0 and θ in [0.01, 0.16], so volatilities of 10–40%; κ in [0.5, 5]; σ in [0.1, 1]; ρ in [−0.95, 0]. It computes their surfaces with `deep_heston_surface`.
- **Targets.** The parameters are scaled to [−1, 1] over the box (`deep_heston_encode`), so each counts equally in the loss.
- **The network** is an MLP with two hidden layers of `hidden_dim`. `deep_heston_calibrate(&model, ivs)` returns decoded parameters, clamped to the box.

```odin
X, Y := mlf.generate_heston_calibration_dataset(2100, context.allocator) // inputs [2100, 20], targets [2100, 5]
defer {t.tensor_free(X); t.tensor_free(Y)}
model := mlf.deep_heston_calibrator_new(128)
defer mlf.deep_heston_calibrator_free(&model)
opt := nn.adam_new(1e-3)
defer nn.adam_free(&opt)
mlf.deep_heston_calibrator_add_to_adam(&model, &opt)
// 100 epochs over the first 2000 rows in batches of 128, in a graph scope:
//   pred := mlf.deep_heston_calibrator_forward(&model, xb)
//   loss := t.tensor_mse_loss(pred, yb, alloc)
// then the 100 held-out rows:
p := mlf.deep_heston_calibrate(&model, ivs[:]) // ivs: 20 implied vols on the grid
```

```
2100 surfaces of 20 implied vols (1.508s)
  epoch   1 scaled-parameter MSE 0.23985
  epoch  26 scaled-parameter MSE 0.02400
  epoch  51 scaled-parameter MSE 0.01537
  epoch  76 scaled-parameter MSE 0.01347
  epoch 100 scaled-parameter MSE 0.01303
training 961ms
held out: scaled parameter RMSE v0 0.037 kappa 0.197 theta 0.073 sigma 0.093 rho 0.178
surface implied-vol RMSE: median 0.42 vol points, 90% 1.10; 310 us per calibration
```

v0 is identified best, and κ worst. κ and θ trade off against each other: surfaces with different (κ, θ) look almost alike. That matters less than it seems, because what pricing needs is the surface, and the predicted parameters reproduce it to a median of 0.42 volatility points. Each calibration takes 0.3 ms, about 60 times faster than Levenberg–Marquardt.

On one known surface, the network and Levenberg–Marquardt (`fin.calibrate_heston`) compare like this:

```odin
truth := fin.Heston_Params{v0 = 0.04, kappa = 2, theta = 0.06, sigma = 0.5, rho = -0.7}
ivs: [mlf.DEEP_HESTON_N_GRID]f64
mlf.deep_heston_surface(truth, ivs[:])
p := mlf.deep_heston_calibrate(&model, ivs[:])
// pts: the grid as quotes at S = 100, r = 3% (strikes m S e^{rT})
res := fin.calibrate_heston(pts, 100, 0.03)
```

```
grid implied vols, expiry 0.25: [0.2695, 0.2358, 0.2004, 0.1687, 0.1566]
network: v0 0.0413 kappa 2.380 theta 0.0629 sigma 0.549 rho -0.722
Levenberg-Marquardt: v0 0.0400 kappa 2.000 theta 0.0600 sigma 0.500 rho -0.700, iv RMSE 3.2e-13 (19ms)
```

The two methods complement each other. Use the network's parameters as an instant estimate, or as the starting point for an exact iterative fit.

Market quotes rarely lie on the grid. `deep_heston_grid_from_surface(quotes, S, r, out)` interpolates scattered `fin.VolSurfacePoint`s onto it:
- **in strike,** linearly at K = m S e^{rT} for each quoted expiry, flat beyond the quoted strikes;
- **in expiry,** linearly in total variance between the expiries on either side, with flat volatility beyond them.

The example uses three expiries (0.3, 0.8 and 1.5 years) × 7 strikes from the same Heston surface:

```
21 scattered quotes onto the grid (ok true): largest difference to the exact grid 0.73 vol points
network on it: v0 0.0418 kappa 1.828 theta 0.0617 sigma 0.493 rho -0.720
```

The interpolation error, up to 0.73 points at the short expiry, which has to be extrapolated from 0.3 years, moves κ more than the network's own error does.

## 5. Deep hedging

Deep hedging (Buehler, Gonon, Teichmann & Wood 2019) skips the model of the option price. A network sees the market state at each rebalancing date and sets the holdings. It is trained to minimize a risk measure of the hedger's terminal P&L, costs included.

The input layout:
- **`paths`** has shape [batch, N + 1, state_size]: the state at dates t_0 … t_N. The first `num_assets` columns are the prices of the hedging instruments in units of the numeraire: with a rate r, pass e^{−rt} S_t and the discounted payoff.
- **`payoffs`** has shape [batch, 1]: the liability paid at t_N.

The P&L of the hedger who is short the claim is

  Σ_k δ_k·(S_{k+1} − S_k) − c Σ_k |δ_k − δ_{k−1}|·S_k − Z.

Positions are opened at t_0 and closed at t_N, and every trade pays the proportional cost c. The premium is not included; it shifts the P&L by a constant. The risk measure (`risk_measure`) is one of:
- `.Variance`: the variance of the P&L;
- `.CVaR`: the mean of the worst `cvar_alpha` share of the losses.

`deep_hedger_train_step` takes one full-batch step and returns the loss. `deep_hedging_evaluate` returns the P&L per path and the terminal price of the first asset.

### Example: a call with 10 bp costs

The example hedges an at-the-money one-year call on a GBM, with S_0 = K = 1, r = 5% and σ = 20%:
- the hedge is rebalanced at 30 dates;
- the network sees (e^{−rt} S_t, time to maturity, σ);
- it is trained full-batch on 2048 paths for 1200 epochs with CVaR at 5%;
- it is tested on 10,000 new paths, against no hedge and against the Black–Scholes delta rebalanced at the same dates with the same costs.

```odin
// gbm_paths(n): paths [n, 31, 3] with rows (e^{-rt} S_t, T - t, sigma), payoffs e^{-rT} (S_T - K)^+
cfg := mlf.DeepHedgerConfig {
	state_size       = 3,
	hidden_size      = 32,
	num_layers       = 2,
	risk_measure     = .CVaR,
	cvar_alpha       = 0.05,
	transaction_cost = 0.001,
	num_assets       = 1,
}
h := mlf.deep_hedger_new(cfg)
defer mlf.deep_hedger_free(h)
opt := nn.adam_new(3e-3)
defer nn.adam_free(&opt)
nn.sequential_add_to_adam(h.network, &opt)
schedule := optim.lr_cosine(3e-3, 1200, warmup_steps = 10)
for e in 0 ..< 1200 {
	opt.learning_rate = optim.lr_schedule_at(schedule, e)
	loss := mlf.deep_hedger_train_step(h, paths, payoffs, &opt)
}
pnl, spot := mlf.deep_hedging_evaluate(h, test_paths, test_payoffs, context.allocator)
defer {delete(pnl); delete(spot)}
```

```
epoch   0 CVaR 5% of -P&L 0.3910
epoch 300 CVaR 5% of -P&L 0.1459
epoch 600 CVaR 5% of -P&L 0.1427
epoch 900 CVaR 5% of -P&L 0.1407
epoch 1199 CVaR 5% of -P&L 0.1404
10000 new paths, P&L of the short call with the Black-Scholes premium 0.1045 received:
  no hedge                     mean +0.0005  sd 0.1446  CVaR 5% 0.4123
  Black-Scholes delta, daily   mean -0.0029  sd 0.0119  CVaR 5% 0.0309
  deep hedge (CVaR)            mean -0.0032  sd 0.0214  CVaR 5% 0.0370
holdings at t = 0.5 for spots [0.8, 0.9, 1.0, 1.1, 1.2]: [0.151, 0.354, 0.634, 0.801, 0.927] (Black-Scholes [0.092, 0.309, 0.598, 0.822, 0.938])
```

How to read the results:
- **Against no hedge,** the deep hedge cuts the 5% CVaR from 0.41 to 0.037, about 11 times.
- **Without being told about Black–Scholes,** it learns holdings close to the Black–Scholes delta.
- **It does not beat the delta here,** and should not be expected to. In this complete market, with small costs, the delta rebalanced every step is close to optimal; after 1200 epochs the deep hedge's CVaR is 1.2 times the delta's. After 300 epochs it was 1.4 times, as in `deep_hedging_test`.
- **Where deep hedging pays** is where no formula exists: large costs, several instruments, path-dependent claims (`deep_hedging_exotic_test`), or real market data.
- **Run time.** This example and the variance run below take about 110 s together.

Trained on the variance objective without costs, the same network gets close to the delta hedge too:

```
no costs, variance objective:
  Black-Scholes delta, daily   mean -0.0000  sd 0.0118  CVaR 5% 0.0273
  deep hedge (variance)        mean -0.0001  sd 0.0143  CVaR 5% 0.0308
```

The checkpoint holds only the network. Pass the configuration to `deep_hedger_load`:

```odin
saved := mlf.deep_hedger_save(h, path)
h2, ok := mlf.deep_hedger_load(path, config = cfg)
defer if ok {mlf.deep_hedger_free(h2)}
// deep_hedging_evaluate(h2, ...) gives the same P&L
```

```
saved true, reloaded true, same P&L true
```

Three procedures write PNG charts:
- `deep_hedging_plot_pnl_distribution` draws a histogram of the P&L;
- `deep_hedging_plot_delta_trajectory` draws a series of holdings;
- `deep_hedging_plot_pnl_vs_spot` draws the P&L against the terminal spot.

`compute_hedging_loss(pnl, measure, alpha, alloc)` exposes the loss for your own training loops. For CVaR, its gradient is exactly the CVaR's: the tail weights are constants of the graph.

## 6. TimeGAN

TimeGAN (Yoon, Jarrett & van der Schaar 2019) generates synthetic sequences that keep the step-to-step dynamics of the data, not just their distribution. It has five recurrent networks, each an LSTM followed by a linear layer and a sigmoid:
- an **embedder** and a **recovery** network, an autoencoder between the features and a latent space;
- a **generator** that maps noise to latent sequences;
- a **supervisor** that predicts the next latent step;
- a **discriminator** that tells real latent sequences from generated ones.

Training has three phases, as in the authors' code:
1. **The autoencoder:** `train_embedder_recovery`.
2. **The supervisor,** on real embeddings: `train_supervisor`.
3. **Joint training:** `timegan_train_joint_step` runs two generator and embedder steps, then a discriminator step while the discriminator's loss is above 0.15. The generator loss combines the adversarial terms, the supervisor's next-step loss and a moment loss that matches the batch means and standard deviations.

Sequences are [n, T, F], flat and row-major, scaled to [0, 1] per feature with `timegan_scaler_fit` / `timegan_scale` (the sigmoid output needs it); `timegan_unscale` maps generated data back. For market data, `prepare_market_data(prices, volumes, indicators, n_indicators, seq_len, alloc)` builds overlapping windows of the clipped log return, the log volume change and the standardized indicators.

The example uses two series with known dynamics:
- x1 is an AR(1) with φ = 0.8;
- x2_t = 0.7 x1_{t−1} + noise, a lagged copy;
- the data are 3000 steps in windows of 12.

```odin
sc := mlf.timegan_scaler_fit(data, F, context.allocator)
defer mlf.timegan_scaler_free(&sc, context.allocator)
scaled := slice.clone(data)
defer delete(scaled)
mlf.timegan_scale(scaled, sc)
tg := mlf.new_timegan(12, 2, 8, 24, 1e-3) // seq_len, features, latent 8, hidden 24, learning rate
defer mlf.timegan_free(tg)
mlf.timegan_init_weights(tg) // Xavier-uniform, as the authors
for _ in 0 ..< 600 {mlf.train_embedder_recovery(tg, scaled, 64)}
for _ in 0 ..< 300 {mlf.train_supervisor(tg, scaled, 64)}
for _ in 0 ..< 300 {mlf.timegan_train_joint_step(tg, scaled, 64)}
synth := mlf.generate_synthetic_data(tg, 2000, context.allocator) // [2000, 12, 2], scaled units
defer delete(synth)
mlf.timegan_unscale(synth, sc)
```

```
embedding    0 reconstruction MSE 0.01703
embedding  300 reconstruction MSE 0.00483
embedding  599 reconstruction MSE 0.00297
supervisor   0 next-step MSE 0.021241
supervisor 150 next-step MSE 0.005212
supervisor 299 next-step MSE 0.003413
joint        0 G 27.5985 E 0.5308 D 2.0623
joint      100 G 18.9509 E 0.5205 D 1.9025
joint      200 G 14.0838 E 0.4747 D 1.9414
joint      299 G 12.3873 E 0.1268 D 1.8383
                     real     TimeGAN
mean x1, x2      +0.004 +0.002  -0.242 -0.161
sd   x1, x2       0.994  0.855   0.922  0.768
corr(x1[t], x1[t-1])  0.802     0.912
corr(x2[t], x1[t-1])  0.818     0.955
first synthetic window, x1: [0.17, -0.43, -0.41, -0.11, 0.41, 0.95]
```

With this short budget (about 18 s), the synthetic data have the right structure:
- persistent x1;
- x2 following x1 with a lag;
- volatilities about right.

But the dependence is too strong (0.91 and 0.96 against 0.80 and 0.82), and the means are off by a quarter of a standard deviation. GANs need long training and careful checking. `timegan_test` trains about 1.7 times longer, on twice the data, and checks:
- the lag-1 auto- and cross-correlations;
- the moments;
- whether a predictor trained on the synthetic data does well on real data.

Always run such checks before using synthetic data.

Each step procedure takes an `alloc` argument that it does not use; the model's own scope holds the graph.

## 7. Validation

The tests are registered in `main.odin` and listed in [the test catalog](test_catalog.md):

| Area | Reference | Test |
|---|---|---|
| Almgren–Chriss, Avellaneda–Stoikov | an independent minimizer of E + λV (the paper's example, TWAP, tiny λ, κT = 1820, invalid input); the 2008 closed form | `optimal_execution_test` |
| Deep BSDE | Black–Scholes; the 10- and 100-asset max-call vs the exact integral (SciPy `quad`) | `deep_bsde_test` |
| Deep calibration | the grid's Heston implied vols vs QuantLib; invariance to S and r; scaling round trip; scattered quotes onto the grid vs NumPy | `deep_calibration_reference_test` |
| Deep hedging | constant hedges vs the closed form; the variance-loss gradient vs finite differences; one period trained to the minimum-variance hedge; evaluation vs a step-by-step reference; no allocation per step once warm | `deep_hedging_reference_test`, `deep_hedging_contracts_test` |
| TimeGAN | supervised and moment losses vs PyTorch; generator and discriminator gradients vs finite differences for all five networks; data windows and scaler | `timegan_reference_test` |

The application tests:
- **`deep_hedging_test`:** five seeds against the static delta and the Black–Scholes delta hedge.
- **`deep_hedging_exotic_test`:** an Asian and an up-and-out call.
- **`deep_hedging_portfolio_test`:** a basket hedged with SPY and QQQ.
- **`deep_hedging_real_data_test`:** an SPY call at its implied volatility.
- **`deep_calibration_test`:** held-out surfaces against Levenberg–Marquardt, and AAPL quotes.
- **`timegan_test`:** the known process, and SPY's stylized facts.

For this guide, the values were also recomputed independently:
- the Almgren–Chriss trajectories and costs for the four λ values, by solving the quadratic problem as a linear system in NumPy;
- the Avellaneda–Stoikov quotes from the closed form;
- the 5-asset max-call value with SciPy's `quad`.

All agree to the printed decimals.

## 8. Quick reference

### Execution

| Procedure | Purpose |
|---|---|
| `almgren_chriss_solve(AlmgrenChrissParams{...})` / `almgren_chriss_free` | optimal liquidation schedule, E and V |
| `avellaneda_stoikov_quote(mid, inventory, t, AvellanedaStoikovParams{...})` | reservation price and quotes |

### Deep BSDE and calibration

| Procedure | Purpose |
|---|---|
| `deep_bsde_model_new(y0, d, T, N, hidden, layers)` / `deep_bsde_model_free` | the model |
| `deep_bsde_add_to_adam` / `deep_bsde_y0` | training, the price estimate |
| `deep_bsde_bs_loss(&m, S_0, K, r, sigma, n_paths, n_steps)` / `deep_bsde_bs_forward` | loss / Y_T and X_T (in a graph scope) |
| `deep_heston_calibrator_new(hidden)` / `_free` / `_add_to_adam` / `_forward` | the calibration network |
| `generate_heston_calibration_dataset(n, allocator)` | training surfaces and scaled parameters |
| `deep_heston_calibrate(&m, ivs)` | surface → `fin.Heston_Params` |
| `deep_heston_surface(params, out)` | Heston implied vols on the grid |
| `deep_heston_features` / `deep_heston_encode` / `deep_heston_decode` | input and target scaling |
| `deep_heston_grid_from_surface(quotes, S, r, out)` | scattered quotes onto the grid |
| `DEEP_HESTON_MONEYNESS` / `DEEP_HESTON_EXPIRIES` / `DEEP_HESTON_LO` / `DEEP_HESTON_HI` | grid and sampling box |

### Deep hedging

| Procedure | Purpose |
|---|---|
| `deep_hedger_new(DeepHedgerConfig{...})` / `deep_hedger_free` | the hedger |
| `deep_hedger_train_step(h, paths, payoffs, &opt)` | one full-batch step |
| `deep_hedging_evaluate(h, paths, payoffs, allocator)` | P&L per path, terminal spot |
| `compute_hedging_loss(pnl, measure, alpha, alloc)` | variance or CVaR of a P&L tensor |
| `deep_hedger_save(h, path)` / `deep_hedger_load(path, allocator, config)` | checkpoints |
| `deep_hedging_plot_pnl_distribution` / `_delta_trajectory` / `_pnl_vs_spot` | PNG charts |

### TimeGAN

| Procedure | Purpose |
|---|---|
| `new_timegan(seq_len, features, latent, hidden, lr)` / `timegan_free` / `timegan_init_weights` | the model |
| `timegan_scaler_fit` / `timegan_scale` / `timegan_unscale` / `timegan_scaler_free` | min-max scaling |
| `prepare_market_data(prices, volumes, indicators, n_ind, seq_len, alloc)` | market windows |
| `train_embedder_recovery` / `train_supervisor` / `timegan_train_joint_step` | the three phases |
| `train_generator` / `train_embedder_joint` / `train_discriminator` | the joint phase's single steps |
| `generate_synthetic_data(tg, n, alloc)` | synthetic sequences (scaled units) |
| `timegan_supervised_loss` / `timegan_moment_loss` / `timegan_generator_loss` / `timegan_discriminator_loss` | the losses |
