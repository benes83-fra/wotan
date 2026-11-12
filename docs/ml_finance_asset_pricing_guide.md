# The `ml_finance` package, part 2: machine-learning asset pricing

This is the second part of the guide to `wotan/ml_finance`. [Part 1](ml_finance_methods_guide.md) covers labels, feature importance and backtest statistics, and [part 3](ml_finance_forecasting_guide.md) covers forecasting. This part covers models of the cross-section of returns, and two tools for events and correlation:
- **instrumented PCA (IPCA):** latent factors whose betas are linear in the stocks' characteristics (Kelly, Pruitt & Su 2019);
- **the conditional autoencoder:** the same model with betas from a neural network (Gu, Kelly & Xiu 2021);
- **the neural stochastic discount factor:** a network that prices a set of assets through the Euler equations;
- **event studies:** cumulative abnormal returns around dated events, with a t-test;
- **implied correlation and dispersion:** the CBOE implied-correlation formula and a dispersion-trading signal.

Every example was compiled and run against the current code, and the output shown is the real output. A debug build with a tracking allocator reports no leaks for any of them. Section 7 lists the comparisons with NumPy, SciPy and torch.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [Instrumented PCA](#2-instrumented-pca)
3. [The conditional autoencoder](#3-the-conditional-autoencoder)
4. [The neural stochastic discount factor](#4-the-neural-stochastic-discount-factor)
5. [Event studies](#5-event-studies)
6. [Implied correlation and dispersion](#6-implied-correlation-and-dispersion)
7. [Validation](#7-validation)
8. [Quick reference](#8-quick-reference)

## 1. Setup and conventions

```odin
import l "wotan/linalg"
import mlf "wotan/ml_finance"
import nn "wotan/nn"
import t "wotan/tensor"
```

IPCA and the conditional autoencoder share one data layout, a **panel** of observations:
- `X: l.Matrix(f64)` is `[n, L]`: one row per observation (asset i at date t), one column per characteristic. The characteristics are known at t. Include a constant column if the betas should have a level.
- `y: []f64` holds the n returns from t to t + 1.
- `times: []int` holds each observation's date, numbered `0 ..< T`.

The panel may be unbalanced: assets can enter and leave, and the rows need no particular order. Every date needs observations, though, and the conditional autoencoder needs at least L of them per date (section 3).

Both models report Kelly, Pruitt & Su's R², which are not demeaned:
- **total R²:** 1 − Σ (r − r̂)² / Σ r², with r̂ = β(z)′ f_t from each date's realized factors;
- **predictive R²:** the same with the factors' time-series mean λ in place of f_t. This is the model's expected return β(z)′ λ.

## 2. Instrumented PCA

IPCA models the return of asset i from t to t + 1 as

  r_{i,t+1} = z_{i,t}′ Γ_α + z_{i,t}′ Γ_β f_{t+1} + ε

so the K betas are linear in the L characteristics (Γ_β is L × K). Γ_α gives alphas that depend on the characteristics, and is fitted only with `intercept = true`. Testing Γ_α = 0 is the paper's test of whether characteristics proxy for risk or for mispricing.

`ipca_fit` follows the authors' Python package (`ipca` 0.6.7, portfolio ALS). It works on the characteristic-managed portfolios Q_t = Z_t′ r_{t+1} / N_t and W_t = Z_t′ Z_t / N_t:
1. It starts Γ at the first K (+1) left singular vectors of Q.
2. It alternates two least-squares steps: the factors given Γ, one cross-sectional regression per date, then Γ given the factors.
3. After each step it normalizes:
   - Γ_β′ Γ_β = I;
   - F F′ is diagonal, with the factors ordered by variance;
   - every factor has a positive mean;
   - Γ_α is projected orthogonal to Γ_β.
4. It stops when no entry of Γ moves by more than `tol` (default 1e-5). Without an intercept the factors must also stop moving.

Because of the normalization, Γ_β is identified only up to the normalization's choices: compare its direction with a true Γ, not its scale. `ipca_portfolios` builds Q and W once. `ipca_fit_portfolios` fits from them, which saves the pass over the panel when several K are fitted.

### Example: one factor, an alpha on value

The panel has 200 assets over 60 dates, with 10% of the observations missing at random. There are four characteristics: a constant, size, value and momentum, the last three uniform on [−0.5, 0.5]. One factor with mean 1% and standard deviation 5% drives the returns. Its betas are z′ (1, −0.6, 0.4, 0), so small and value stocks have higher betas. There is also an alpha of 2% per unit of value, and noise with standard deviation 8%. The model is fitted on dates 0–39, and dates 40–59 are held out.

```odin
L :: 4
// a panel of N assets over T dates; return = z' (gb f + ga) + noise
panel :: proc(N, T: int, gb, ga: [L]f64) -> (X: l.Matrix(f64), y: []f64, times: []int) {
	Xd := make([dynamic]f64)
	yd := make([dynamic]f64)
	td := make([dynamic]int)
	for d in 0 ..< T {
		f := rand.float64_normal(0.01, 0.05)
		for _ in 0 ..< N {
			z := [L]f64{1, rand.float64() - 0.5, rand.float64() - 0.5, rand.float64() - 0.5}
			if rand.float64() < 0.1 {continue} // missing
			beta, alpha := 0.0, 0.0
			for c in 0 ..< L {beta += z[c] * gb[c]; alpha += z[c] * ga[c]}
			append(&Xd, ..z[:])
			append(&yd, alpha + beta * f + rand.float64_normal(0, 0.08))
			append(&td, d)
		}
	}
	return l.Matrix(f64){data = Xd[:], rows = len(yd), cols = L}, yd[:], td[:]
}

mean :: proc(x: []f64) -> f64 {
	s := 0.0
	for v in x {s += v}
	return s / f64(len(x))
}

rand.reset(11)
X, y, times := panel(200, 60, {1, -0.6, 0.4, 0}, {0, 0, 0.02, 0})
split := 0
for times[split] < 40 {split += 1}
Xa := l.Matrix(f64){data = X.data[:split * L], rows = split, cols = L}
fmt.printf("%d observations, %d to fit (dates 0-39)\n", X.rows, split)

m := mlf.ipca_fit(&Xa, y[:split], times[:split], {n_factors = 1})
defer mlf.ipca_free(&m)
nrm := math.sqrt(1 + 0.36 + 0.16)
fmt.printf("Gamma_beta %.4f (true direction %.4f); %d iterations, converged %v\n", m.gamma.data,
	[L]f64{1 / nrm, -0.6 / nrm, 0.4 / nrm, 0}, m.n_iter, m.converged)
fmt.printf("factor: first %.4f, mean %.4f\n", m.factors.data[:4], mean(m.factors.data))
tot, pred := mlf.ipca_r2(&m, &Xa, y[:split], times[:split])
fmt.printf("in sample: total R2 %.4f, predictive R2 %.4f\n", tot, pred)

ma := mlf.ipca_fit(&Xa, y[:split], times[:split], {n_factors = 1, intercept = true})
defer mlf.ipca_free(&ma)
// gamma is [L, K + 1]: column K is Gamma_alpha
ga_hat: [L]f64
for c in 0 ..< L {ga_hat[c] = ma.gamma.data[c * 2 + 1]}
fmt.printf("with intercept: Gamma_alpha %.4f; %d iterations\n", ga_hat, ma.n_iter)
tot_a, pred_a := mlf.ipca_r2(&ma, &Xa, y[:split], times[:split])
fmt.printf("in sample: total R2 %.4f, predictive R2 %.4f\n", tot_a, pred_a)
```

```
10753 observations, 7164 to fit (dates 0-39)
Gamma_beta [0.7426, -0.5370, 0.3987, -0.0349] (true direction [0.8111, -0.4867, 0.3244, 0.0000]); 4 iterations, converged true
factor: first [0.0582, -0.0255, 0.0620, 0.0062], mean 0.0292
in sample: total R2 0.3422, predictive R2 0.0539
with intercept: Gamma_alpha [-0.0062, 0.0015, 0.0186, 0.0017]; 5 iterations
in sample: total R2 0.3454, predictive R2 0.0571
```

How to read this output:
- **The beta direction is recovered.** The cosine between the estimated and the true Γ_β is 0.993. The scale moves into F: Γ_β has unit length, so the estimated factor is roughly the true one times |(1, −0.6, 0.4, 0)| = 1.23.
- **Γ_α is the alpha net of beta.** It is constrained orthogonal to Γ_β, so it estimates only the part of the true alpha that is not proportional to the beta: (0, 0, 0.02, 0) − 0.0065 × Γ_β = (−0.0053, 0.0032, 0.0179, 0). The fit, (−0.0062, 0.0015, 0.0186, 0.0017), is that vector up to noise. The component along Γ_β is indistinguishable from a higher factor mean.
- **R² levels.** The noise (8%) is larger than the factor (5%), so even the true model's expected total R² is only about 0.3. The predictive R² is lower because it uses only the expected returns.

### Fitted values and out of sample

`ipca_predict` returns the fitted returns z′ Γ g_t; with `mean_factor = true` it uses λ in place of f_t. For a new period, the factors are not known in advance. `ipca_factor_at` estimates one date's factors from that date's characteristics and returns, using the fitted Γ, the same cross-sectional regression the fit uses. `ipca_r2_out_of_sample` applies that to every date of a new panel, and uses the training λ for the predictive R². The new panel's dates are numbered from 0.

```odin
Xb := l.Matrix(f64){data = X.data[split * L:], rows = X.rows - split, cols = L}
tb := make([]int, X.rows - split)
for &d, i in tb {d = times[split + i] - 40}
fit := mlf.ipca_predict(&m, &Xa, times[:split])
fmt.printf("fitted returns of the first 3 observations %.4f (actual %.4f)\n", fit[:3], y[:3])
o1, o2 := mlf.ipca_r2_out_of_sample(&m, &Xb, y[split:], tb)
p1, p2 := mlf.ipca_r2_out_of_sample(&ma, &Xb, y[split:], tb)
// the first held-out date's factor from its own cross-section
n0 := 0
for tb[n0] == 0 {n0 += 1}
Z0 := l.Matrix(f64){data = Xb.data[:n0 * L], rows = n0, cols = L}
f0: [1]f64
mlf.ipca_factor_at(&m, &Z0, y[split:][:n0], f0[:])
fmt.printf("held out (dates 40-59): total R2 %.4f / %.4f, predictive R2 %.4f / %.4f (without / with alpha)\n",
	o1, p1, o2, p2)
fmt.printf("factor at date 40 from its %d observations: %.4f\n", n0, f0[0])
```

```
fitted returns of the first 3 observations [0.0507, 0.0334, 0.0196] (actual [0.1835, -0.2090, 0.0640])
held out (dates 40-59): total R2 0.4361 / 0.4378, predictive R2 0.1023 / 0.1030 (without / with alpha)
factor at date 40 from its 181 observations: 0.0773
```

Individual returns are mostly noise, so the fitted values are far from the actual ones. The held-out R² are higher than in sample because the factor happened to move more on dates 40–59: its mean square is 0.0085 there against 0.0055 in the fit. A total R² rises with the factor's share of the variance. Compare models on the same dates, not R² across periods. Here the alpha model is marginally better on both.

## 3. The conditional autoencoder

Gu, Kelly & Xiu replace IPCA's linear betas with a neural network:
- **the beta network** maps characteristics z [L] through ReLU hidden layers to K betas:
  - `hidden = nil` is CA0, linear betas as in IPCA;
  - `{32}` is CA1;
  - `{32, 16}` is CA2;
- **the factor network** is linear. It maps the L characteristic-managed portfolios x_t = (Z_t′ Z_t)⁻¹ Z_t′ r_{t+1} to K factors.

The fitted return is β(z_{i,t})′ f_{t+1}. Training minimizes the mean squared error over every observation plus `l1` × Σ |w| over the weight matrices. It uses full-batch Adam, with early stopping on a validation panel. GKX also use batch normalization and an ensemble of 10 seeds; neither is built in, and an ensemble is a loop over seeds.

`ca_panel_new` turns a panel into constant tensors:
- Z, the characteristics;
- r, the returns;
- the managed portfolios P [T, L], which require Z_t′ Z_t to be invertible, so each date needs at least L observations with full-rank characteristics;
- a date selector S [n, T], one-hot. S is dense, so it costs n × T values; keep panels to a size where that fits in memory.

### Example: nonlinear betas

There are 150 assets over 100 dates, with the same four characteristics as above and two factors. The betas are 1 + 6 (size² − 1/12) and 12 × value × momentum, which no linear model can represent. Dates 0–59 train, 60–79 validate and 80–99 test.

```odin
rand.reset(5)
N, T :: 150, 100
// ... X [N*T, 4], y, times as above, with
//     beta = {1 + 6 * (z[1] * z[1] - 1.0 / 12), 12 * z[2] * z[3]}
//     y = beta . f + N(0, 0.1), f = {N(0.02, 0.08), N(0.01, 0.06)}
train := mlf.ca_panel_new(&Xa, ya, ta) // dates 0-59, numbered 0 ..< 60
defer mlf.ca_panel_free(&train)
valid := mlf.ca_panel_new(&Xv, yv, tv) // dates 60-79, numbered 0 ..< 20
defer mlf.ca_panel_free(&valid)
test := mlf.ca_panel_new(&Xt, yt, tt) // dates 80-99
defer mlf.ca_panel_free(&test)
fmt.printf("managed portfolios P: %v; date selector S: %v\n", train.P.shape, train.S.shape)

ca := mlf.ca_new({n_chars = 4, n_factors = 2, hidden = []int{32}, l1 = 1e-5})
defer mlf.ca_free(ca)
res := mlf.ca_fit(ca, &train, &valid, epochs = 3000, learning_rate = 3e-3, patience = 300)
fmt.printf("%d epochs run, best %d, validation MSE %.5f, last loss %.5f\n", res.epochs_run, res.best_epoch,
	res.valid_mse, res.last_loss)
lam := mlf.ca_mean_factor(ca, &train) // the training risk premia
defer delete(lam)
fmt.printf("mean factor (training) %.4f
", lam)
tr1, tr2 := mlf.ca_r2(ca, &train)
te1, te2 := mlf.ca_r2(ca, &test, lam)
fmt.printf("CA1 R2: training total %.4f predictive %.4f; test total %.4f predictive %.4f
", tr1, tr2, te1, te2)
fac := mlf.ca_factors(ca, &test) // [T, K] row-major
defer delete(fac)
fmt.printf("test factors of the first date %.4f
", fac[:2])
```

```
managed portfolios P: [60, 4, 1, 1]; date selector S: [9000, 60, 1, 1]
513 epochs run, best 213, validation MSE 0.01477, last loss 0.01356
mean factor (training) [-0.0018, 0.0097]
CA1 R2: training total 0.4244 predictive 0.0013; test total 0.2556 predictive 0.0106
test factors of the first date [0.0515, 0.0355]
```

The validation MSE stopped improving at epoch 213. After 300 more epochs without improvement (`patience`), `ca_fit` stopped and restored the epoch-213 weights; `valid_mse` is their validation MSE. The training R² is well above the test R², which is the overfitting that early stopping limits.

`ca_r2` takes the training λ for the test panel. Without it, the panel's own mean factor is used, which looks ahead. The predictive R² is small here: the factor means are small next to their volatility, and 60 dates estimate them poorly. The network's factors are identified only up to rotation, so the individual factors and their means need not match the simulation's.

Against linear betas on the same test dates:

```odin
ca0 := mlf.ca_new({n_chars = 4, n_factors = 2}) // CA0: no hidden layer
defer mlf.ca_free(ca0)
res0 := mlf.ca_fit(ca0, &train, &valid, epochs = 3000, learning_rate = 3e-3, patience = 300)
lam0 := mlf.ca_mean_factor(ca0, &train)
defer delete(lam0)
c1, _ := mlf.ca_r2(ca0, &test, lam0)
ip := mlf.ipca_fit(&Xa, ya, ta, {n_factors = 2})
defer mlf.ipca_free(&ip)
i1, _ := mlf.ipca_r2_out_of_sample(&ip, &Xt, yt, tt)
fmt.printf("test total R2: CA0 (linear betas) %.4f (best epoch %d), IPCA %.4f
", c1, res0.best_epoch, i1)
```

```
test total R2: CA0 (linear betas) 0.2327 (best epoch 2375), IPCA 0.2330
```

CA0 and IPCA are the same model class, and gradient descent reaches IPCA's fit. CA1's hidden layer adds about 10% to the test R² (0.2556). Neither gets close to the true betas. The second beta, value × momentum, is uncorrelated with every linear characteristic, so the linear managed portfolios that feed the factor network carry almost nothing of the second factor. This is a property of GKX's design, not of the implementation; `asset_pricing_models_test` documents the same effect over 9 seeds.

### A training loop of your own

`ca_fit` is a convenience. `ca_train_step` performs one full-batch Adam step and returns the loss before the update. `ca_gradients` only fills the gradients. `ca_parameters` lists the trainable tensors for an optimizer.

```odin
ca2 := mlf.ca_new({n_chars = 4, n_factors = 2, hidden = []int{32, 16}})
defer mlf.ca_free(ca2)
opt := nn.adam_new(1e-3)
defer nn.adam_free(&opt)
params := mlf.ca_parameters(ca2)
defer delete(params)
for p in params {nn.adam_add_param(&opt, p)}
l0 := mlf.ca_train_step(ca2, &train, &opt)
for _ in 0 ..< 98 {mlf.ca_train_step(ca2, &train, &opt)}
l99 := mlf.ca_train_step(ca2, &train, &opt)
fmt.printf("CA2: %d parameter tensors; loss %.5f at step 1, %.5f at step 100
", len(params), l0, l99)
```

```
CA2: 8 parameter tensors; loss 0.02288 at step 1, 0.01588 at step 100
```

The 8 tensors are the weight and bias of the three beta layers and the factor layer. `ca_mse(m, &panel, lambda)` gives the MSE without the penalty, for monitoring.

## 4. The neural stochastic discount factor

A stochastic discount factor M prices gross returns through E[M_t R_{i,t}] = 1 for every asset i. The neural SDF is M_t = f_θ(Z_t)² + ε, a network of state variables Z_t observed with the returns; squaring keeps M positive, and ε is a floor. It is trained on

  Σ_i (mean_t(M_t R_{i,t}) − 1)² + λ Var_t(M)

With few assets, many SDFs price them exactly, and the variance penalty λ selects one. As λ → 0, the minimizer tends to the minimum-variance pricing SDF (Hansen–Jagannathan).

Pick λ on the scale of the squared pricing errors. A λ that is too large makes the constant SDF optimal. With daily returns, λ around 1e-5 is sensible (`neural_sdf_test` uses that); with monthly returns, around 1e-3.

`neural_sdf_optimum` solves for the minimizer over every series M_t: no network, no positivity constraint, an (N+1) × (N+1) linear system. No network can reach a lower loss. A network of the returns themselves can reach it, because the optimum is affine in R_t. That makes it the benchmark for training.

### Example: a one-factor market

The market has 600 periods and 3 assets, with mean returns 0.5%, 1% and 1.5%, betas 0.5, 1 and 1.5 on a factor with standard deviation 4%, and idiosyncratic noise of 2%. The state variables are the standardized returns.

```odin
rand.reset(9)
T, N :: 600, 3
mu := [N]f64{0.005, 0.010, 0.015}
beta := [N]f64{0.5, 1.0, 1.5}
R := t.tensor_new(l.matrix_new(f64, T, N)) // gross returns [T, N]
Z := t.tensor_new(l.matrix_new(f64, T, N)) // state variables [T, input_size]
defer {t.tensor_free(R); t.tensor_free(Z)}
for row in 0 ..< T {
	f := rand.float64_normal(0, 0.04)
	for i in 0 ..< N {R.data.data[row * N + i] = 1 + mu[i] + beta[i] * f + rand.float64_normal(0, 0.02)}
}
// ... Z = R standardised column by column

LAMBDA :: 1e-3
cfg := mlf.NeuralSDFConfig {
	input_size       = N,
	hidden_size      = 32,
	num_layers       = 2,
	num_assets       = N,
	epsilon          = 0.01,
	variance_penalty = LAMBDA,
}
model := mlf.neural_sdf_new(cfg)
defer mlf.neural_sdf_free(model)

m_opt, loss_opt, min_opt, ok := mlf.neural_sdf_optimum(R.data.data, N, LAMBDA, context.allocator)
defer delete(m_opt)
e_opt := mlf.sdf_pricing_errors(m_opt, R.data.data, N, context.allocator)
defer delete(e_opt)

opt := nn.adam_new(1e-3)
defer nn.adam_free(&opt)
nn.sequential_add_to_adam(model.network, &opt)
epochs := 2000
for e in 0 ..< epochs { // cosine decay of the learning rate to 1e-5
	opt.learning_rate = 1e-3 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.PI * f64(e) / f64(epochs))))
	loss := mlf.neural_sdf_train_step(model, Z, R, &opt)
	if e % 500 == 0 {fmt.printf("epoch %4d loss %.4e\n", e, loss)}
}
errs := mlf.neural_sdf_evaluate(model, Z, R, context.allocator) // |mean(M R_i) - 1|
defer delete(errs)
ms := mlf.neural_sdf_series(model, Z, context.allocator) // M_t
defer delete(ms)
```

The listing leaves out the printing and two benchmarks: M = 1, and the best constant SDF c = Σ μ_i / Σ μ_i², with μ the mean gross returns.


```
M = 1: pricing errors [0.0039, 0.0092, 0.0163]
optimum (ok true): loss 3.4427e-05, smallest M 0.557, pricing errors [2.6e-03, 4.4e-04, 3.0e-03]
epoch    0 loss 4.6214e-04
epoch  500 loss 3.6055e-05
epoch 1000 loss 3.4708e-05
epoch 1500 loss 3.4546e-05
trained: loss 3.4533e-05, pricing errors [2.6e-03, 4.7e-04, 3.0e-03], mean M 0.9947, sd 0.1364
largest |M - M_opt| 0.0472
best constant SDF 0.9903: loss 7.5743e-05; the network closes 99.7% of the gap to the optimum
```

How to read this output:
- **Starting point.** The network starts near M = 1, with small output weights and a bias of √(1 − ε). M = 1 leaves each asset's mean net return as its pricing error.
- **The optimum** does not price the assets exactly. With λ = 1e-3, a little pricing error is worth a lower variance. Its smallest value, 0.557, is above ε, so the network can reach it.
- **Training** gets within about 1e-7 of the optimal loss and closes 99.7% of the gap between the best constant SDF and the optimum. The learned series is within 0.05 of the optimal one everywhere.

The first output line uses `sdf_pricing_errors(m, returns, N, allocator)`, which gives the errors of any series, here M = 1. `neural_sdf_loss(m, r, λ, alloc)` builds the loss as a graph scalar, for your own training loops.

### Save and load

`neural_sdf_save(model, path)` writes the network as an `nn` checkpoint. The checkpoint does not hold the configuration, so pass it again to `neural_sdf_load(path, cfg)`:

```odin
saved := mlf.neural_sdf_save(model, path)
loaded, lok := mlf.neural_sdf_load(path, cfg)
if lok {
	defer mlf.neural_sdf_free(loaded)
	m2 := mlf.neural_sdf_series(loaded, Z, context.allocator) // the same series as ms
	defer delete(m2)
}
```

```
saved true, reloaded true, same series true
```

**Fixed while writing this guide:** `neural_sdf_load` and `deep_hedger_load` freed the contents of the optimizer that `nn.load_checkpoint` returns, but not the `Adam` struct itself, so every load leaked it. Both loaders now free it. `neural_sdf_reference_test` checks the save, load and free cycle of both models on a tracking allocator.

## 5. Event studies

`compute_car` implements the constant-mean-return model of MacKinlay (1997). For an event on row e:
- the normal return is the mean over the estimation window, rows [e + `est_start`, e + `est_end`), end exclusive;
- the CAR is the sum of r − normal over the event window, rows [e + `evt_start`, e + `evt_end`], inclusive.

Events are matched to `all_dates` by exact string comparison. An event is skipped if:
- its date is not in the data;
- either window falls outside the data;
- the estimation window has fewer than 2 rows;
- any return in either window is not finite.

The t-statistic is the cross-sectional mean(CAR) / (sd(CAR) / √N), with the N − 1 standard deviation, as `scipy.stats.ttest_1samp(cars, 0)`. It is 0 for fewer than 2 events.

### Example

The data are 400 business days from 2024-01-02, with returns N(0.05%, 1%). Six announcements each add +3% on the day and +1% the day after. A seventh event date falls on a Saturday. The estimation window is days −120 to −21, and the event window is days −1 to +2.

```odin
// dates: "2024-01-02", "2024-01-03", ... (weekdays); returns: N(0.0005, 0.01)
ann := []int{50, 130, 180, 250, 310, 395}
for e in ann {returns[e] += 0.03; returns[e + 1] += 0.01}
event_dates := make([]string, len(ann) + 1)
defer delete(event_dates)
for e, i in ann {event_dates[i] = dates[e]}
event_dates[len(ann)] = "2024-12-28" // a Saturday: not in the data
res := mlf.compute_car(dates, returns, event_dates, -120, -20, -1, 2, context.allocator)
defer mlf.event_study_result_free(&res, context.allocator)
fmt.printf("%d events used: %v\n", res.num_events, res.dates)
fmt.printf("CARs %.4f\nmean CAR %.4f, t %.3f\n", res.car_values, res.mean_car, res.t_statistic)
```

```
events ["2024-03-12", "2024-07-02", "2024-09-10", "2024-12-17", "2025-03-11", "2025-07-08", "2024-12-28"]
5 events used: ["2024-07-02", "2024-09-10", "2024-12-17", "2025-03-11", "2025-07-08"]
CARs [0.0778, 0.0505, 0.0570, 0.0739, 0.0402]
mean CAR 0.0599, t 8.472
```

The expected CAR is the injected 4%; the estimation-window mean removes the drift. The four days of 1% noise give each CAR a standard deviation of 2%, and this sample's mean came out at 6%. Two events were skipped:
- **2024-03-12**, on row 50, because its estimation window would start at row −70;
- **the Saturday**, because it is not in the data.

A NaN 100 days before the last event falls in that event's estimation window only, so only that event is skipped:

```
with a NaN 100 days before 2025-07-08: 4 events ["2024-07-02", "2024-09-10", "2024-12-17", "2025-03-11"], mean CAR 0.0648, t 9.868
```

`res.dates` borrows the strings from `event_dates`; `event_study_result_free(&res, allocator)` frees the two slices only. `event_study_test` runs the same calculation on AAPL around earnings dates, with FinBERT sentiment for the announcements.

## 6. Implied correlation and dispersion

An index's variance is its components' variances plus their covariances. Assume one common correlation ρ, with index weights w that sum to 1. Then

  σ_I² = Σ w_i² σ_i² + ρ [(Σ w_i σ_i)² − Σ w_i² σ_i²]

which solves for ρ: the CBOE implied-correlation formula. `calculate_pairwise_correlation(index_var, comp_vars, weights)` takes **variances** and returns ρ clamped to [−1, 1]. It normalizes the weights to sum 1, so market caps work as given. It returns NaN when ρ is undefined: fewer than two components with weight and variance.

`compute_dispersion_signal` takes **volatilities**. It computes ρ twice, from implied and from realized (or forecast) volatilities. The correlation premium is the difference. The premium is z-scored against its history (mean and standard deviation you pass), and the signal is tanh(z / 1.5), in [−1, 1]:
- above 0.5, implied correlation is expensive: sell index volatility and buy component volatility (short correlation);
- below −0.5, the reverse;
- between, neutral.

With a NaN premium the signal is NaN, and the recommendation is neutral.

```odin
w := []f64{30, 25, 20, 15, 10} // market caps
iv := []f64{0.28, 0.32, 0.25, 0.40, 0.35} // component implied vols
rv := []f64{0.24, 0.27, 0.22, 0.33, 0.30} // component realised (forecast) vols
index_iv, index_rv := 0.22, 0.17
vars := make([]f64, len(iv))
defer delete(vars)
for v, i in iv {vars[i] = v * v}
fmt.printf("implied correlation %.4f\n", mlf.calculate_pairwise_correlation(index_iv * index_iv, vars, w))
fmt.printf("one component: %v\n", mlf.calculate_pairwise_correlation(0.04, vars[:1], w[:1]))
// the premium's history: mean 0.05, sd 0.04
sig := mlf.compute_dispersion_signal(index_iv, index_rv, iv, rv, w, 0.05, 0.04)
fmt.printf("implied %.4f realised %.4f premium %.4f signal %.4f\n%s\n", sig.implied_corr, sig.realized_corr,
	sig.correlation_premium, sig.signal, sig.recommendation)
mlf.print_dispersion_signal(sig)
```

```
implied correlation 0.3697
one component: NaN
implied 0.3697 realised 0.2562 premium 0.1135 signal 0.7850
SHORT CORRELATION (Sell Index Vol / Buy Component Vol)

╔════════════════════════════════════════════════════════════╗
║          CROSS-ASSET DISPERSION TRADING DASHBOARD          ║
╠════════════════════════════════════════════════════════════╣
║  Index IV:               22.00%                            ║
║  Index forecast RV:      17.00%                            ║
╠════════════════════════════════════════════════════════════╣
║  Avg component IV:       32.00%                            ║
║  Avg component RV:       27.20%                            ║
╠════════════════════════════════════════════════════════════╣
║  IMPLIED CORRELATION:   +36.97% (market pricing)           ║
║  REALIZED CORRELATION:  +25.62% (model forecast)           ║
╠════════════════════════════════════════════════════════════╣
║  CORRELATION PREMIUM:   +11.35%                            ║
║  NORMALIZED SIGNAL:     +0.785                             ║
╠════════════════════════════════════════════════════════════╣
║  ACTION: SHORT CORRELATION                                 ║
║          (Sell Index Vol / Buy Component Vol)              ║
╚════════════════════════════════════════════════════════════╝
```

The premium of 11.35 points is 1.6 standard deviations above its history, so tanh(1.59 / 1.5) = 0.785. The component averages in the dashboard are unweighted. `normalize_weights(w)` returns the weights scaled to sum 1, or equal weights if they sum to 0.

**Fixed while writing this guide:** the dashboard's right border was misaligned on several rows, and the action line ran past the box. Every row is now padded to the box width, and a long recommendation wraps before its parenthesis.

## 7. Validation

The tests are registered in `main.odin` and listed in [the test catalog](test_catalog.md):

| Area | Reference | Test |
|---|---|---|
| IPCA | a NumPy transcription of the authors' `ipca` 0.6.7 portfolio ALS on an unbalanced panel, K = 2 without and with Γ_α: Γ, factors, iteration counts, total and predictive R², an out-of-sample factor | `ipca_reference_test` |
| Conditional autoencoder | torch (float64) on an unbalanced panel: the loss (MSE + L1) and every parameter gradient, the parameters and loss after 25 Adam steps, total and predictive R² | `conditional_autoencoder_reference_test` |
| Neural SDF | the loss and its gradient in M vs NumPy; the optimum vs NumPy's T × T system and BFGS; network gradients vs finite differences; save, load and free on a tracking allocator | `neural_sdf_reference_test` |
| CARs, implied correlation, dispersion signal | CARs and t vs `scipy.stats.ttest_1samp`; the CBOE formula and the signal vs NumPy | `ml_finance_metrics_reference_test` |

The application tests check the behaviour the models are for:
- **`asset_pricing_models_test`:**
  - IPCA recovers the span of a linear Γ and most of the true model's out-of-sample R²;
  - with nonlinear betas, the autoencoder beats IPCA;
  - on 24 large caps, the single IPCA factor is a market factor.
- **`neural_sdf_test`:** the network closes at least 95% of the loss gap on a synthetic market, and 80% on SPY, QQQ, IWM and TLT. It also compares held-out pricing errors with the constant SDF's.
- **`event_study_test`:** CARs around AAPL earnings, with FinBERT sentiment.
- **`dispersion_trading_test`:** `compute_dispersion_signal` on SPY, AAPL, MSFT and NVDA: the VIX as index implied volatility, GARCH–LSTM forecasts as realized volatilities, and the factor risk of the components. Single-stock implied volatilities are not on Yahoo, so the components' are assumed to be 1.15 × the forecast.

For this guide, the IPCA example was also refitted with an independent NumPy ALS, which reproduces every printed number. The CARs, t-statistics, implied correlations and signal were recomputed with SciPy and NumPy and agree to the printed decimals.

## 8. Quick reference

### IPCA

| Procedure | Purpose |
|---|---|
| `ipca_fit(&X, y, times, params)` | fit Γ and the factors; `IPCAParams{n_factors, intercept, max_iter, tol}` |
| `ipca_portfolios(&X, y, times)` / `ipca_fit_portfolios(&p, params)` | the managed portfolios once, then fits from them |
| `ipca_predict(&m, &X, times, mean_factor)` | fitted returns, with f_t or λ |
| `ipca_r2(&m, &X, y, times)` | total and predictive R² in sample |
| `ipca_factor_at(&m, &Z, r, f)` | one new date's factors from its cross-section |
| `ipca_r2_out_of_sample(&m, &X, y, times)` | R² on a new panel |
| `ipca_free` / `ipca_portfolios_free` | release |

### Conditional autoencoder

| Procedure | Purpose |
|---|---|
| `ca_new(CAConfig{n_chars, n_factors, hidden, l1})` | the model (`ca_free`) |
| `ca_panel_new(&X, y, times)` | a panel as tensors (`ca_panel_free`) |
| `ca_fit(m, &train, &valid, epochs, learning_rate, patience)` | Adam with early stopping; `CAFitResult` |
| `ca_train_step(m, &panel, &opt)` / `ca_gradients(m, &panel)` | one step / the gradients only |
| `ca_parameters(m)` | trainable tensors |
| `ca_fitted(m, &panel, alloc, lambda)` | fitted returns as a graph tensor |
| `ca_mse(m, &panel, lambda)` | MSE without the penalty |
| `ca_factors(m, &panel)` / `ca_mean_factor(m, &panel)` | factors [T, K], their mean |
| `ca_r2(m, &panel, lambda)` | total and predictive R² |

### Neural SDF

| Procedure | Purpose |
|---|---|
| `neural_sdf_new(NeuralSDFConfig{...})` | the model (`neural_sdf_free`) |
| `neural_sdf_train_step(model, z, r, &opt)` | one full-batch Adam step |
| `neural_sdf_values(model, z, alloc)` / `neural_sdf_loss(m, r, λ, alloc)` | M and the loss as graph tensors |
| `neural_sdf_series(model, z, allocator)` | M_t |
| `neural_sdf_evaluate(model, z, r, allocator)` / `sdf_pricing_errors(m, returns, N, allocator)` | \|mean(M R_i) − 1\| per asset |
| `neural_sdf_optimum(returns, N, λ, allocator)` | the minimizer over all series, its loss, its minimum |
| `neural_sdf_save(model, path)` / `neural_sdf_load(path, config)` | checkpoints |

### Events and dispersion

| Procedure | Purpose |
|---|---|
| `compute_car(all_dates, returns, event_dates, est_start, est_end, evt_start, evt_end, allocator)` | CARs, mean, t (`event_study_result_free`) |
| `calculate_pairwise_correlation(index_var, comp_vars, weights)` | CBOE implied (or realized) correlation |
| `compute_dispersion_signal(index_iv, index_rv, comp_ivs, comp_rvs, weights, hist_mean, hist_std)` | premium, signal, recommendation |
| `print_dispersion_signal(sig)` | the dashboard |
| `normalize_weights(weights)` | weights summing to 1 |
