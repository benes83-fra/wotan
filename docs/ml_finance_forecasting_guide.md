# The `ml_finance` package, part 3: forecasting

This is the third part of the guide to `wotan/ml_finance`. [Part 1](ml_finance_methods_guide.md) covers labels, feature importance and backtest statistics, [part 2](ml_finance_asset_pricing_guide.md) covers asset-pricing models, and [part 4](ml_finance_derivatives_guide.md) derivatives and execution. This part covers forecasting:
- **point forecasts of a series:** N-BEATS and PatchTST;
- **probabilistic forecasts:** a sequence encoder with a Gaussian, Student-t or quantile head, and scores for any forecast distribution;
- **token forecasting:** a small Chronos-style language model over quantized values, which forecasts by sampling paths;
- **volatility:**
  - GARCH one-step forecasts;
  - sequence-model forecasters and their ensemble with GARCH;
  - a volatility term structure for option prices;
  - the volatility risk premium signal and backtest metrics;
- **regimes:** a Gaussian hidden Markov model;
- **pairs trading:** a Kalman-filter hedge ratio with entry and exit rules, and a grid search.

Every example was compiled and run against the current code, and the output shown is the real output. A debug build with a tracking allocator reports no leaks for any of them. Section 8 lists the comparisons with PyTorch, SciPy, hmmlearn and pykalman.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [N-BEATS and PatchTST](#2-n-beats-and-patchtst)
3. [Probabilistic forecasts](#3-probabilistic-forecasts)
4. [Token forecasting](#4-token-forecasting)
5. [Volatility](#5-volatility)
6. [Regimes: the hidden Markov model](#6-regimes-the-hidden-markov-model)
7. [Kalman-filter pairs trading](#7-kalman-filter-pairs-trading)
8. [Validation](#8-validation)
9. [Quick reference](#9-quick-reference)

## 1. Setup and conventions

```odin
import ts "wotan/analytics"
import w "wotan/core"
import fin "wotan/finance"
import l "wotan/linalg"
import mlf "wotan/ml_finance"
import nn "wotan/nn"
import optim "wotan/optimize"
import t "wotan/tensor"
```

The models in sections 2–5 are built from `nn` layers and `tensor` autograd; the [nn guide](nn_guide.md) and the [tensor guide](tensor_guide.md) explain both. What matters here:
- **Two input layouts.** N-BEATS and PatchTST take windows as a plain matrix [batch, input_size]. The sequence models (`ProbForecaster`, the volatility forecasters) take a tensor whose `shape` is [batch, steps, features, 1]. Its data is row-major in that order.
- **Graph scopes.** A forward pass allocates its intermediate tensors in a `t.Graph_Scope`, which `graph_end` releases in one step:
  - Where a model owns a scope (N-BEATS, PatchTST, the neural SDF of part 2), its `*_train_step` and `*_predict` procedures open and close it for you.
  - Otherwise, wrap each step in `t.graph_begin(&scope)` … `t.graph_end(&scope)`, and use `no_grad = true` for predictions.
- **Optimizers.** Every model has a `*_parameters` or `*_add_to_adam` procedure that registers its trainable tensors with an `nn.Adam`.
- **Scale your data.** Neural networks train best on inputs and targets of order 1. The examples standardize returns or multiply them by 100. The Gaussian loss also clamps the variance at 1e-6, so a standard deviation below 1e-3 gets no gradient.

## 2. N-BEATS and PatchTST

Both models map a window of the last `input_size` values to the next `horizon` values in one step. `forecast_windows(series, input_size, horizon)` cuts a series into every such pair: X [n, input_size] and Y [n, horizon], with n = len − input_size − horizon + 1.

**N-BEATS** (Oreshkin et al. 2020) is a stack of fully connected blocks. Each block maps its input to a *backcast*, which is subtracted before the next block, and a *forecast*; the forecasts add up. The forecast starts at the last observed value.
- **Generic.** The blocks predict the backcast and forecast values directly.
- **Interpretable.** A trend stack, with polynomial bases of degree `degree`, comes before a seasonality stack, with Fourier bases. Each stack's blocks share one set of weights, as in the authors' code.
- **`relative = true`** (not in the paper) feeds the blocks the window minus its last value. A trending series then never shows the network input levels it has not seen in training.

**PatchTST** (Nie et al. 2023) cuts the window into patches of `patch_len` values every `stride` steps. It embeds each patch and runs a Transformer encoder over the patches. A linear head maps the flattened encodings to the horizon.
- **RevIN** standardizes each window by its own mean and standard deviation, then maps the forecast back. A model trained on one scale therefore forecasts any other.
- **Channel independence.** Each row of the input is one series; with several series, `channels[r]` selects row r's RevIN parameters.

### Example: trend and two seasons

The series is 600 steps of 0.02 t + 2 sin(2πt/12) + 0.5 sin(2πt/6) + N(0, 0.3²). The models see 36 steps and forecast 12, trained on windows inside the first 450 steps. They are tested on the 139 windows whose forecasts cover steps 450–599. The noise alone gives an MSE of 0.09.

```odin
rand.reset(1)
n := 600
s := make([]f64, n)
for i in 0 ..< n {
	x := f64(i)
	s[i] = 0.02 * x + 2 * math.sin(2 * math.PI * x / 12) + 0.5 * math.sin(2 * math.PI * x / 6) + rand.float64_normal(0, 0.3)
}
I, H :: 36, 12
Xa, Ya := mlf.forecast_windows(s[:450], I, H)
defer {l.matrix_free(&Xa); l.matrix_free(&Ya)}
Xt, Yt := mlf.forecast_windows(s[450 - I:], I, H)
defer {l.matrix_free(&Xt); l.matrix_free(&Yt)}
fmt.printf("training windows %d x %d -> %d; test windows %d\n", Xa.rows, Xa.cols, Ya.cols, Xt.rows)
// ... the baselines: the last value, and the last season (12 steps) repeated
```

```
training windows 403 x 36 -> 12; test windows 139
MSE naive 4.4358, seasonal naive 0.2343
```

Training is mini-batch Adam. `nbeats_train_step(m, x, y, &opt)` performs one step on a batch (zeroing the gradients first) and returns the loss before it:

```odin
m := mlf.nbeats_new({input_size = I, horizon = H, blocks = 3, layers = 2, layer_size = 64, relative = true})
defer mlf.nbeats_free(m)
opt := nn.adam_new(1e-3)
defer nn.adam_free(&opt)
params := mlf.nbeats_parameters(m)
defer delete(params)
for q in params {nn.adam_add_param(&opt, q)}
order := make([]int, Xa.rows)
defer delete(order)
for &o, i in order {o = i}
for _ in 0 ..< 60 { // epochs
	rand.shuffle(order)
	for s := 0; s < Xa.rows; s += 32 {
		b := min(32, Xa.rows - s)
		xb := t.tensor_new(l.matrix_new(f64, b, I))
		yb := t.tensor_new(l.matrix_new(f64, b, H))
		for k in 0 ..< b {
			copy(xb.data.data[k * I:][:I], Xa.data[order[s + k] * I:][:I])
			copy(yb.data.data[k * H:][:H], Ya.data[order[s + k] * H:][:H])
		}
		mlf.nbeats_train_step(m, xb, yb, &opt)
		t.tensor_free(xb)
		t.tensor_free(yb)
	}
}
xt := t.tensor_new(l.matrix_new(f64, Xt.rows, I))
defer t.tensor_free(xt)
copy(xt.data.data, Xt.data)
f := mlf.nbeats_predict(m, xt) // [139 * 12] row-major
defer delete(f)
```

The same loop trains the other two configurations. PatchTST uses the same loop with `patchtst_train_step(m, xb, yb, nil, &opt)`, where `nil` puts every row in channel 0:

```odin
// generic, without relative inputs
{input_size = I, horizon = H, blocks = 3, layers = 2, layer_size = 64}
// interpretable: 2 trend blocks and 2 seasonality blocks; 200 epochs at 3e-3
{input_size = I, horizon = H, interpretable = true, blocks = 2, layers = 2, layer_size = 32,
	seasonality_layer_size = 64, degree = 2, harmonics = 1, relative = true}

pt := mlf.patchtst_new({input_size = I, horizon = H, patch_len = 6, stride = 3, d_model = 16, n_heads = 2,
	d_ff = 32, n_layers = 2, channels = 1, revin = true})
defer mlf.patchtst_free(pt)
// ... patchtst_parameters, Adam, 60 epochs at 1e-3
fp := mlf.patchtst_predict(pt, xt, nil)
```

```
generic, relative: 3 blocks, 28944 parameters, test MSE 0.1126 (553ms)
first test forecast [9.14, 8.29, 7.62, 6.92, 7.05, 8.13, 9.14, 10.73, 11.20, 11.29, 10.32, 9.48]
actual              [8.57, 8.51, 7.76, 6.75, 6.91, 7.62, 8.95, 10.22, 10.98, 11.20, 10.74, 9.92]
generic, absolute: 3 blocks, 28944 parameters, test MSE 0.1774 (537ms)
interpretable, relative: 4 blocks, 10526 parameters, test MSE 0.1711 (1.477s)
== PatchTST
12 patches, 39 parameter tensors, 7070 parameters
test MSE 0.1021 (2.193s)
```

How to read the results:
- **Every model beats both baselines.** The seasonal naive forecast (0.234) misses the trend, and the naive forecast misses everything else.
- **PatchTST comes closest to the noise floor (0.09),** with the generic N-BEATS on relative inputs close behind.
- **Relative inputs help on a trending series.** The test windows lie above every training level; without relative inputs, the generic model's error is 58% higher.
- **The interpretable model** has fewer parameters and trains longer. It is the choice when the trend and seasonal parts themselves are of interest.

RevIN makes PatchTST's forecasts follow the scale of their input:

```odin
x100 := t.tensor_new(l.matrix_new(f64, Xt.rows, I))
defer t.tensor_free(x100)
for v, i in Xt.data {x100.data.data[i] = 100 * v}
f100 := mlf.patchtst_predict(pt, x100, nil) // compare f100 / 100 with fp
```

```
windows x 100: forecasts / 100 differ by at most 9.10e-06
```

The difference is not exactly zero because RevIN adds 1e-5 to each window's variance. On series whose values are themselves of order 1e-3, such as raw returns, that term matters: scale them first.

## 3. Probabilistic forecasts

`ProbForecaster` is a DeepAR-style model. A sequence encoder reads the window: any `nn.SeqBackend`, that is LSTM, GRU, Mamba or TCN. Its last step feeds a head that predicts a distribution for each of the next `horizon` values:

| `output` | Prediction | Loss |
|---|---|---|
| `.Gaussian` | mean μ, sd σ = softplus(·) + `min_sigma` | Gaussian negative log-likelihood |
| `.StudentT` | location μ, scale σ, degrees of freedom ν = 2 + exp(·), starting at 10 | Student-t negative log-likelihood |
| `.Quantile` | the quantiles at `quantiles` (ascending, in (0, 1)) | pinball loss |

Each horizon gets its own distribution from the same encoding (direct forecasting), so errors do not build up across steps. `prob_forecaster_loss(model, x, y)` takes y as [batch · horizon], row-major. `prob_forecaster_forward` returns a `Prob_Forecast` of graph tensors:
- `mu`, `sigma` and `nu`, each [batch, horizon];
- or `quantiles` [batch, horizon · Q], where column h·Q + j holds level j of step h + 1.

### Scores and risk measures

These procedures work on any forecast, not just this model's.

```odin
mu, sd, nu := 0.0005, 0.012, 5.0 // a daily return forecast; t(5) with the same location and scale
v, es := mlf.prob_forecast_gaussian_var_es(mu, sd, 0.05)
v2, es2 := mlf.prob_forecast_student_t_var_es(mu, sd, nu, 0.05)
// for outcomes 0 and -3%:
mlf.prob_forecast_gaussian_nll(mu, sd, y)
mlf.prob_forecast_student_t_nll(mu, sd, nu, y)
mlf.prob_forecast_gaussian_crps(mu, sd, y)
mlf.prob_forecast_student_t_crps(mu, sd, nu, y)
// 2000 samples of N(mu, sd²): empirical quantiles and CRPS
q: [3]f64
mlf.prob_forecast_sample_quantiles(smp, {0.05, 0.5, 0.95}, q[:])
mlf.prob_forecast_sample_crps(smp, -0.03)
// quantile predictions for 4 outcomes at levels 0.1, 0.5, 0.9; the last row has crossed quantiles
pred := []f64{-1, 0, 1, -1, 0, 1, -2, 0, 2, 0.5, -0.5, 1}
y := []f64{0.2, 1.5, -2.5, 0}
levels := []f64{0.1, 0.5, 0.9}
mlf.prob_forecast_pinball(pred, y, levels)
crossed := mlf.prob_forecast_sort_quantiles(pred, 3) // sorts each row in place
cov := mlf.prob_forecast_coverage(pred, y, 3)
```

```
Gaussian: 5% quantile -0.01924, VaR / ES at 5%: 0.01924 / 0.02425
Student-t(5): 5% quantile -0.02368, VaR / ES at 5%: 0.02368 / 0.03418
outcome +0.000: NLL Gaussian -3.5030, t -3.4532; CRPS Gaussian 0.002813, t 0.003092
outcome -0.030: NLL Gaussian -0.2739, t -0.9659; CRPS Gaussian 0.023772, t 0.022784
2000 samples: quantiles 5/50/95% [-0.01975, 0.00059, 0.02037], CRPS at -0.03 0.023644
pinball 0.3917, 1 row crossed; after sorting 0.3583
coverage [0.25, 0.50, 0.75]
```

How to read the scores:
- **Tails.** With the same scale, the t(5) puts more mass in the tails. Its VaR and especially its ES are larger.
- **NLL and CRPS rank forecasts differently.** For a −3% day (2.5 σ), the t's NLL is much lower than the Gaussian's, while the CRPS barely tells them apart. The NLL weighs tail events heavily; the CRPS is in the units of the outcome and is more forgiving.
- **The sample CRPS** of 2000 Gaussian draws, 0.023644, approximates the closed form, 0.023772.
- **`prob_forecast_sort_quantiles`** applies the monotone rearrangement (Chernozhukov et al. 2010). It removes crossed quantiles and never increases the pinball loss: 0.3917 fell to 0.3583.
- **`prob_forecast_coverage`** reports, per level, the share of outcomes at or below the predicted quantile. A calibrated forecast's coverage equals the levels.

### Example: GARCH returns

The returns follow a GARCH(1,1) with ω = 0.02, α = 0.1 and β = 0.88, so their unconditional variance is 1. The model sees the last 20 values of (r, |r|), is trained on 2000 windows for 10 epochs, and is tested on the next 500:

```odin
W :: 20
batch_x :: proc(r: []f64, idx: []int, alloc: mem.Allocator) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, len(idx) * W * 2, alloc)
	for i, b in idx {
		for s in 0 ..< W {
			v := r[i - W + s]
			m.data[(b * W + s) * 2] = v
			m.data[(b * W + s) * 2 + 1] = abs(v)
		}
	}
	x := t.tensor_new(m, false, alloc)
	x.shape = {len(idx), W, 2, 1} // [batch, steps, features, 1]
	return x
}

train :: proc(model: ^mlf.ProbForecaster, r: []f64, idx: []int, epochs: int) {
	BATCH :: 50
	H := model.config.horizon
	steps := epochs * (len(idx) / BATCH)
	schedule := optim.lr_cosine(3e-3, steps, warmup_steps = steps / 20)
	opt := nn.adam_new(3e-3)
	defer nn.adam_free(&opt)
	mlf.prob_forecaster_add_to_adam(model, &opt)
	order := slice.clone(idx)
	defer delete(order)
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	step := 0
	for _ in 0 ..< epochs {
		rand.shuffle(order)
		for b := 0; b + BATCH <= len(order); b += BATCH {
			batch := order[b:b + BATCH]
			t.graph_begin(&scope)
			alloc := t.graph_allocator(&scope)
			y := make([]f64, BATCH * H, alloc) // y[k H + h]: step h + 1 after window k
			for i, k in batch {copy(y[k * H:][:H], r[i:i + H])}
			loss := mlf.prob_forecaster_loss(model, batch_x(r, batch, alloc), y)
			nn.adam_zero_grad(&opt)
			t.tensor_backward(loss)
			opt.learning_rate = optim.lr_schedule_at(schedule, step)
			nn.adam_step(&opt)
			t.graph_end(&scope)
			step += 1
		}
	}
}

model := mlf.prob_forecaster_new({input_size = 2, hidden_size = 16, seq_backend = .LSTM, output = .Gaussian})
defer mlf.prob_forecaster_free(model)
train(model, r, train_idx, 10)
t.graph_begin(&scope, no_grad = true)
f := mlf.prob_forecaster_forward(model, batch_x(r, test_idx, t.graph_allocator(&scope)))
// f.mu.data.data[k], f.sigma.data.data[k]: window k's forecast; score with the procedures above
t.graph_end(&scope)
```

```
Gaussian (LSTM): test NLL 1.3830 (true sigma 1.3772, unconditional N(0, 1) 1.3870), CRPS 0.5494 (true 0.5457)
  90% intervals cover 89.8%; corr(predicted sigma, true sigma) 0.838
```

The predicted σ tracks the true conditional volatility with a correlation of 0.84, and the 90% intervals cover 89.8%.

The scores move little because most of a return's variation is unpredictable even with the true σ. That gives the scale for judging the NLL:
- the gap between the unconditional N(0, 1) and the true model is 0.0098 nats;
- the model closes 41% of that gap.

The quantile head forecasts 5 steps at once:

```odin
levels := []f64{0.05, 0.5, 0.95}
model := mlf.prob_forecaster_new({input_size = 2, hidden_size = 16, seq_backend = .GRU, output = .Quantile,
	quantiles = levels, horizon = 5})
// ... train as above; pred := f.quantiles.data.data, [500, 5 * 3]
crossed := mlf.prob_forecast_sort_quantiles(pred, 3)
cov := mlf.prob_forecast_coverage(pred, y, 3) // y [500 * 5]: the next 5 returns of each window
```

```
quantile head (GRU), 5 steps: pinball 0.1970, coverage [0.050, 0.504, 0.938], 0 crossed rows
  first window: 5% quantiles of steps 1..5 [-1.397, -1.457, -1.473, -1.605, -1.442]
```

## 4. Token forecasting

`TokenForecaster` is a decoder-only Chronos (Ansari et al. 2024), trained from scratch:
- **Tokens.** Each window of `context_length + prediction_length` values is scaled by its context's mean |value| and quantized into `n_tokens` bins (default 512, over [−15, 15] in scaled units).
- **Training.** A GPT learns to predict every next token, with a cross-entropy loss.
- **Forecasts.** The model samples `prediction_length` tokens after a context, path by path with a KV cache, and decodes them back to values.

The paths are draws from the predicted *joint* distribution of the future. Any statistic of them gets a forecast: quantiles per step, the sum over the horizon, the minimum along the path. Score them with `prob_forecast_sample_quantiles` and `prob_forecast_sample_crps`. Missing values (NaN) are not supported.

The example trains on an AR(1), x_t = 0.9 x_{t−1} + N(0, 1), whose true forecast distribution is known: x_{T+h} ~ N(0.9^h x_T, (1 − 0.9^{2h}) / (1 − 0.81)).

```odin
rand.reset(2000)
PHI :: 0.9
C, H :: 32, 8
N, TRAIN_END :: 12000, 10000
x := make([]f64, N)
defer delete(x)
for i in 1 ..< N {x[i] = PHI * x[i - 1] + rand.float64_normal(0, 1)}
model := mlf.token_forecaster_new({context_length = C, prediction_length = H, d_model = 32, num_heads = 2, d_ff = 64,
	num_layers = 2})
defer mlf.token_forecaster_free(model)
STEPS, B :: 400, 32
opt := nn.adam_new(3e-3)
defer nn.adam_free(&opt)
mlf.token_forecaster_add_to_adam(model, &opt)
schedule := optim.lr_cosine(3e-3, STEPS, warmup_steps = STEPS / 20)
scope: t.Graph_Scope
defer t.graph_scope_destroy(&scope)
windows := make([]f64, B * (C + H))
defer delete(windows)
for s in 0 ..< STEPS {
	for b in 0 ..< B { // random windows from the training part
		i := rand.int_max(TRAIN_END - C - H)
		copy(windows[b * (C + H):][:C + H], x[i:i + C + H])
	}
	t.graph_begin(&scope) // the loss must be computed inside a scope
	loss := mlf.token_forecaster_loss(model, windows, B)
	nn.adam_zero_grad(&opt)
	t.tensor_backward(loss)
	opt.learning_rate = optim.lr_schedule_at(schedule, s)
	nn.adam_step(&opt)
	t.graph_end(&scope)
}
// 40 held-out contexts, 100 sample paths each
paths := mlf.token_forecaster_sample(model, x[i:i + C], 100) // [100, 8] row-major
defer delete(paths)
```

```
vocabulary 512 tokens, bins over [-15, 15]
step   0 loss 6.338
step 100 loss 4.237
step 200 loss 3.846
step 300 loss 3.834
step 399 loss 3.760
first context ends at -2.571: step-1 sample quantiles 10/50/90% [-4.407, -2.878, -0.303] (true -3.595, -2.314, -1.032)
40 contexts: CRPS / true CRPS 1.116 at step 1, 1.102 at step 8; 80% intervals cover 75% / 82%
```

After 400 steps (about 15 s), the sampled forecasts score within about 12% of the true distribution's CRPS. The first context's interval is wider than the true one. A longer run narrows it: `token_forecast_test` trains 600 steps and gets ratios of 1.03–1.13 over 5 seeds. `temperature` and `top_k` in `token_forecaster_sample` sharpen or broaden the sampling.

## 5. Volatility

### GARCH one-step forecasts

`garch_recursive_forecast(returns, omega, alpha, beta)` runs the GARCH(1,1) recursion with fitted parameters: σ²_{t+1} = ω + α r_t² + β σ²_t. The first variance is the mean of r², the start `ts.garch_fit` uses, and out[t] uses only returns before t. Pass demeaned returns.

The data are 1500 daily returns of a GARCH(1,1) with ω = 2e-6, α = 0.08 and β = 0.9. The model is fitted on the first 1000:

```odin
g := ts.garch_fit_spec(r[:1000], {p = 1, q = 1}) // ts.garch_fit also prints its iterations
defer ts.garch_result_free(&g)
om, al, be := g.params.omega, g.params.alpha[0], g.params.beta[0]
g1000 := mlf.garch_recursive_forecast(r[:1000], om, al, be) // equals sqrt(g.conditional_var)
defer delete(g1000)
gv := mlf.garch_recursive_forecast(r, om, al, be) // all 1500 days: out of sample from day 1000
defer delete(gv)
```

```
GARCH fit: omega 2.011e-06 alpha 0.0584 beta 0.9175
on the fitted returns: max relative |sigma^2 - conditional_var| 2.2e-16
one-step vols on all 1500 days: day 1000 0.01233 (true 0.01295)
```

On a longer series, the starting variance is the mean square over all of it, which reaches into the future for the first few days. Its effect decays as β^t.

### Sequence-model forecasters and the ensemble

There are two neural volatility forecasters. Both output one unconstrained number per sample, which may come out negative.
- **`VolatilityForecaster`:** an MLP with two ReLU hidden layers, on a flat feature vector [batch, flat_input].
- **`LSTMVolatilityForecaster`:** a sequence block (`seq_backend`: LSTM, GRU or Mamba) over [batch, seq_len, input_size, 1], then flatten, FC + ReLU, FC.

`EnsembleVolatilityForecaster` holds an `LSTMVolatilityForecaster`, GARCH parameters and a weight w. `ensemble_predict(&e, garch_vol, model_vol)` returns w · garch + (1 − w) · model, with w clamped to [0, 1]. `compute_optimal_weight(garch, model, actual)` gives the w that minimizes the squared error, in closed form.

The example forecasts the realized volatility of the next 5 days from the last 20 days of (r, |r|), scaled by 100. The model is trained on days 20–994 for 30 epochs. The weight is chosen on days 1000–1249 and tested on days 1250–1494:

```odin
SEQ, F, H :: 20, 2, 5
e := mlf.ensemble_volatility_new(F, 16, SEQ) // input size, hidden size, sequence length; LSTM
defer mlf.ensemble_volatility_free(&e)
e.garch_omega, e.garch_alpha, e.garch_beta = om, al, be
opt := nn.adam_new(1e-3)
defer nn.adam_free(&opt)
mlf.ensemble_add_to_optimizer(&e, &opt)
// per batch, in a graph scope:
//   pred := mlf.lstm_volatility_forecaster_forward(&e.lstm, x) // x [B, 20, 2, 1] -> [B, 1]
//   loss := t.tensor_mse_loss(pred, y, alloc)                 // y: 100 x the 5-day realised vol
// ... lv, lt: the model's forecasts / 100 for days 1000-1249 and 1250-1494
e.ensemble_weight = mlf.compute_optimal_weight(gv[1000:1250], lv, act[1000:1250])
ens := mlf.ensemble_predict(&e, gv[i], lt[i - 1250])
```

```
epoch  0 MSE 0.2633
epoch 10 MSE 0.0815
epoch 20 MSE 0.0750
epoch 29 MSE 0.0728
GARCH weight chosen on days 1000-1249: 1.000
test RMSE (x 1e4): GARCH 28.52, LSTM 28.14, ensemble 28.52
```

On GARCH data, GARCH is the right model, and the weight goes all the way to it. On days 1250–1494 the LSTM happened to do slightly better. With 250 validation days, a choice between forecasts this close is mostly noise. The weight pays off when the two forecasts make different errors, as on real data in `ensemble_volatility_test`.

### The volatility term structure and option prices

`ensemble_term_structure_vol(model_vol, omega, alpha, beta, current_var, horizon_days)` gives the annualized volatility over the next `horizon_days`:
- **the GARCH path** follows the expected variance, v_{h+1} = ω + (α + β) v_h, which reverts to the long-run level;
- **each day's blend** is w_h × the model's variance + (1 − w_h) × GARCH's, with w_h = exp(−h/20), so the model dominates the first weeks and GARCH the long end;
- **the result** is √(252 × the mean of the blend).

`ensemble_price_and_greeks` prices a European option at the volatility for its life, rounded to trading days, using `fin.price_and_greeks`.

```odin
cur := om + al * r[n - 1] * r[n - 1] + be * gv[n - 1] * gv[n - 1] // next-day variance
model_vol := 0.20 // the sequence model's annualised forecast
for h in ([]int{0, 1, 5, 21, 63, 252}) {
	fmt.printf("  %3d days: %.4f\n", h, mlf.ensemble_term_structure_vol(model_vol, om, al, be, cur, h))
}
price, gr := mlf.ensemble_price_and_greeks(100, 100, 0.25, 0.03, model_vol, om, al, be, cur, .Call)
```

```
next-day GARCH vol 0.1212 annualised; long-run 0.1451
    0 days: 0.1212
    1 days: 0.1969
    5 days: 0.1913
   21 days: 0.1750
   63 days: 0.1572
  252 days: 0.1477
3-month ATM call: price 3.5104, delta 0.5536, vega 0.1977 (sigma 0.1572)
```

A horizon of 0 returns GARCH's next-day volatility. From 1 day on, the model's 20% dominates, then fades toward the long-run 14.5%. The 3-month call is priced at the 63-day volatility; vega is per volatility point.

### The volatility risk premium

`compute_vrp(vix, forecast_rv_daily, hist_mean, hist_std)` compares implied volatility (VIX, in %) with a forecast of realized volatility (daily, as a decimal). It annualizes the forecast, takes the spread IV − RV in volatility points and z-scores it against its history. The signal is tanh(z / 1.5):
- above z = 1.5, the recommendation is to sell premium;
- below −1.5, to buy it;
- between 0.5 and 1.5 in either direction, to lean that way.

VIX is a 30-day volatility, so the forecast should cover about the next 21 trading days.

```odin
sg := mlf.compute_vrp(18.5, 0.009, 3.0, 2.5) // VIX 18.5; daily RV forecast 0.9%; history mean 3, sd 2.5
mlf.print_vrp_signal(sg)
```

```
RV 14.29%, VRP 4.21 points, z 0.485, signal 0.3126: NEUTRAL (Fairly priced)

========================================
   VOLATILITY RISK PREMIUM (VRP) SIGNAL   
========================================
Implied Volatility (VIX):    18.50%
Forecast Realized Vol (RV):  14.29%
----------------------------------------
Variance Risk Premium (VRP): +4.21%
Historical VRP Z-Score:      +0.49
Normalized Signal:          +0.313  (Range: -1.0 to +1.0)
----------------------------------------
RECOMMENDATION: NEUTRAL (Fairly priced)
  -> Strategy: Delta-neutral theta decay or stay on the sidelines.
========================================
```

`VRPBacktestResult` holds the series of a strategy's backtest. Fill in `daily_pnl`, `equity_curve` and `positions`, then call `compute_backtest_metrics` to compute:
- the total P&L;
- the annualized Sharpe ratio;
- the maximum drawdown, from the running peak in P&L units;
- the number of entries;
- the share of days with nonzero P&L that were positive.

```odin
bt := mlf.VRPBacktestResult {
	daily_pnl    = []f64{0.5, -0.2, 0, 0.3, -0.6, 0.4, 0, 0.2},
	positions    = []f64{1, 1, 0, -1, -1, 1, 0, 1},
	equity_curve = make([]f64, 8, context.temp_allocator),
}
acc := 0.0
for p, i in bt.daily_pnl {acc += p; bt.equity_curve[i] = acc}
mlf.compute_backtest_metrics(&bt)
mlf.print_backtest_result(&bt)
```

```
total 0.60, Sharpe 3.330, max drawdown 0.60, trades 4, win rate 0.667

╔══════════════════════════════════════════════════╗
║     WALK-FORWARD VRP BACKTEST RESULTS            ║
╠══════════════════════════════════════════════════╣
║  Trading days:             8                     ║
║  Trades (entries):         4                     ║
║  Total P&L:            +0.6000 (daily vol units) ║
║  Sharpe ratio:           +3.33 (annualized)      ║
║  Max drawdown:          0.6000                   ║
║  Win rate (days):       66.7%                    ║
╠══════════════════════════════════════════════════╣
║  STRONG EDGE: Sharpe ratio above 1.5             ║
╚══════════════════════════════════════════════════╝
```

A Sharpe ratio of 3.3 from 8 days means nothing. The edge line only restates the Sharpe ratio; test it with `probabilistic_sharpe_ratio` ([part 1](ml_finance_methods_guide.md#6-backtest-statistics)). `vrp_backtest_result_free` frees the series when they were allocated with `r.allocator`. Here they are literals and temporary memory, so there is nothing to free.

**Fixed while writing this guide:** `print_backtest_result`'s right border was misaligned on several rows. Every row is now padded to the box. The last line used to read "Strategy shows significant α"; it now states the Sharpe threshold it actually checks.

## 6. Regimes: the hidden Markov model

`HMM` is a Gaussian hidden Markov model with N states:
- `transition` is A[i·N + j] = P(next state j | state i);
- `emission_mu` and `emission_var` hold each state's normal distribution;
- `initial` holds the first state's probabilities.

The procedures:
- **`hmm_init_simple(&h, obs)`** starts every state at the sample mean. State i gets the mean of the i-th quantile bucket of squared deviations as its variance, so the states are ordered from calm (0) to turbulent (N − 1). Each state stays with probability 0.95.
- **`hmm_fit(&h, obs, max_iter, tol)`** runs Baum–Welch (EM) with scaled forward–backward. The variances are floored at 1e-3 × the sample variance, so a state cannot collapse onto one observation. It returns the log-likelihood of the parameters before the last update, as hmmlearn does.
- **`hmm_decode`** returns the Viterbi path, the most likely state sequence.
- **`hmm_filter`** returns P(state at t | observations up to t) [T, N]: what was known on day t. The Viterbi path and smoothed posteriors use the whole sample, so they look ahead. Trade on the filtered probabilities.
- **`hmm_log_likelihood`** gives ln P(obs).

The example has 2000 days in two regimes:
- calm: mean +0.05% and sd 0.7%, staying with probability 0.98;
- turbulent: mean −0.1% and sd 2%, staying with probability 0.95.

```odin
h := mlf.hmm_new(2)
defer mlf.hmm_free(&h)
mlf.hmm_init_simple(&h, obs)
ll, conv := mlf.hmm_fit(&h, obs)
path := mlf.hmm_decode(&h, obs)
defer delete(path)
filt := mlf.hmm_filter(&h, obs) // [T, 2]: filt[t * 2 + 1] = P(turbulent at t | data up to t)
defer delete(filt)
```

```
start: variances [1.110e-05, 2.773e-04], transition [0.950, 0.050, 0.050, 0.950]
fit: log-likelihood 6393.70, converged true
means [0.00046, -0.00119], sds [0.00686, 0.01928]
transition [0.9842, 0.0158, 0.0390, 0.9610], initial [0.000, 1.000]
log-likelihood of the fitted model 6393.70
regime accuracy: Viterbi 0.934, filtered (P > 0.5) 0.902
last day: P(turbulent | data so far) 0.0241, Viterbi 0, true 0
```

The fit recovers the regimes' means, volatilities and persistence. The initial distribution is fitted from a single day, so it is always close to 0 or 1. The filtered probabilities classify 90% of the days correctly without looking ahead; the Viterbi path, which uses the whole sample, gets 93%. The orchestrator, covered in part 5 of this guide, uses `hmm_filter` for this reason.

hmmlearn 0.3.3, started from the same parameters, reaches the same fit: log-likelihood 6393.70, the same means, volatilities and transitions, and 93.4% Viterbi accuracy. That needs `covars_prior = 0`, though. hmmlearn's default prior, 1e-2, adds a pseudo-variance far larger than the variance of daily returns (about 5e-5). With it, the fit stops after 6 iterations at a lower likelihood. On raw returns, either turn the prior off or standardize the data first.

## 7. Kalman-filter pairs trading

`kalman_pairs_strategy(&df, col_x, col_y, config)` trades the spread of two price columns of a DataFrame:
- **Filter.** The regression y_t = α_t + β_t x_t + e_t is filtered online, with (α, β) a random walk with process noise Q = `process_noise` × I.
- **Warmup.** The first `warmup_window` valid rows (default 60) give the start by OLS: the coefficients, their covariance and the residual variance as R. These rows are not traded.
- **Signal.** The z-score is the one-step-ahead innovation of y_t, divided by its predicted standard deviation √S. It uses only rows before t.
- **Entry.** If z > `entry_threshold`, the strategy goes short y and long β x; if z < −`entry_threshold`, the reverse. β is fixed at entry.
- **Exit.** The position is closed when z comes back inside ±`exit_threshold`, after at least `min_hold_days`. At ±`stop_loss_threshold` it is closed at once, followed by `cooldown_days` without entries.
- **Returns.** The position decided on row t earns from t to the next valid row, per unit of gross notional |y| + |β x|, less `transaction_cost` on each change.
- **Missing data.** Rows with a missing or non-positive price are skipped: no update, no trade, a return of 0.

The data are 750 days:
- x is a log random walk with 2% daily volatility;
- y = 5 + β_t x + an AR(1) spread (φ = 0.8, shock sd 0.5);
- β_t starts at 1.5 and drifts;
- y is missing on row 400.

```odin
df := w.dataframe_new()
defer w.destroy_dataframe(&df)
cx := w.column_new("x", .Float, 0)
cy := w.column_new("y", .Float, 0)
// ... w.append_float(&cx, px); w.append_float(&cy, 5 + beta * px + spread) (w.append_null on row 400)
w.add_column(&df, cx)
w.add_column(&df, cy)
cfg := mlf.kalman_pairs_default_config() // entry 2, exit 0.5, stop 3.5, noise 1e-5, cost 0.1%
res := mlf.kalman_pairs_strategy(&df, "x", "y", cfg)
defer mlf.pairs_trading_result_free(&res)
```

```
750 rows, 10 position changes, first at rows [480, 485, 542, 546]
hedge ratio at rows 60 / 400 / 749: 1.531 / 1.504 / 1.467 (true 1.482 / 1.479 / 1.479)
z after the warmup: mean -0.013, sd 0.737, |z| > 2 on 6 rows
process noise 1e-06: z sd 0.934, 30 position changes, beta at row 749 1.459, Sharpe 1.886
process noise 1e-07: z sd 1.122, 40 position changes, beta at row 749 1.477, Sharpe 2.215
process noise 1e-08: z sd 1.275, 36 position changes, beta at row 749 1.488, Sharpe 1.105
row 400 (missing y): return 0, z 0, spread NaN
sum of returns 0.0188, annualised Sharpe 0.518
```

The hedge ratio is tracked well, but with the default process noise the strategy hardly trades: z has a standard deviation of only 0.74, and passes ±2 on 6 days. The process noise sets the trade-off:
- **Large Q** lets the coefficients follow the spread itself. Part of every deviation is absorbed as a change in α or β, the innovations shrink, and the signal goes quiet.
- **Small Q** keeps the deviation in the spread. The z-scores grow, eventually above 1 as the coefficients become too rigid to follow β.

Here 1e-7 trades most and has the best Sharpe ratio. Tune Q on the data, and judge the result with the statistics of part 1.

`kalman_pairs_grid_search` tries every combination of entry, exit, process noise, minimum hold and cooldown, and returns the configuration with the best in-sample Sharpe ratio. That maximum is biased upward. Pass `trials` to get every grid point's returns, the columns of a [T, N] matrix, for the deflated Sharpe ratio or the probability of backtest overfitting:

```odin
trials: l.Matrix(f64)
best, best_sr := mlf.kalman_pairs_grid_search(&df, "x", "y", {1.5, 2.0, 2.5}, {0.0, 0.5}, {1e-6, 1e-5}, {1, 3},
	{0, 5}, trials = &trials)
defer l.matrix_free(&trials)
ds := mlf.deflated_sharpe_trials(&trials)
```

```
grid of 48: best entry 1.5 exit 0.0 noise 1e-06 hold 3 cooldown 0, Sharpe 2.525
deflated: PSR 1.000, DSR 0.982
```

The DSR of 0.982 says the best of the 48 configurations is still significant after the selection. That is expected here, because the data are a mean-reverting pair by construction. On KO / PEP, `kalman_pairs_test` reports the same statistics.

## 8. Validation

The tests are registered in `main.odin` and listed in [the test catalog](test_catalog.md):

| Area | Reference | Test |
|---|---|---|
| N-BEATS | a torch transcription of the authors' code: generic and interpretable forecasts, MSE and every parameter gradient, then 10 Adam steps | `nbeats_reference_test` |
| PatchTST | a torch transcription of the official model (RevIN, padding and unfolding, LayerNorm encoder, flatten head): forecasts, MSE and all 39 parameter gradients, then 8 Adam steps | `patchtst_reference_test` |
| Probabilistic heads and scores | the Gaussian, Student-t and pinball losses vs PyTorch; quantiles, VaR / ES, NLL and CRPS vs SciPy | `prob_forecast_test` |
| Token forecasting | Chronos' scaling and binning; sample quantiles and ensemble CRPS vs NumPy | `token_forecast_test` |
| GARCH recursion, ensemble weight, VRP metrics | `garch_fit`'s conditional variances; SciPy's `minimize_scalar`; the backtest metrics by hand | `ml_finance_metrics_reference_test` |
| HMM | hmmlearn 0.3.3: forward log-likelihood, one Baum–Welch step, the fit and its Viterbi path, a bad tick, rescaled returns, one state | `hmm_regime_test` |
| `hmm_filter` | NumPy's forward recursion | `orchestrator_reference_test` |
| Kalman pairs | statsmodels OLS for the warmup and pykalman for the filter on a synthetic pair; trades, hedged returns, missing rows, the grid and its trial matrix | `kalman_pairs_test` |

The application tests:
- **`forecasting_models_test`:** every model beats the naive forecasts on trend plus seasonality, and RevIN handles channels at scales 1 and 100. On SPY log prices, no model beats the random walk.
- **`prob_forecast_test`:** the LSTM forecaster on GARCH data beats the unconditional baseline, with calibrated intervals.
- **`token_forecast_test`:** the GPT's CRPS stays within 20% of the true distribution's.
- **`ensemble_volatility_test`:** the ensemble, the VRP signal and the backtest on SPY and VIX.
- **`kalman_pairs_test`:** KO / PEP and semiconductor pairs, with the DSR and PBO of the grid searches.

For this guide, the values were also recomputed independently:
- the score values (quantiles, VaR, ES, NLL, CRPS by numerical integration, pinball, coverage) with SciPy and NumPy;
- the volatility term structure, the option price, the VRP and the backtest metrics with NumPy and SciPy;
- the HMM fit with hmmlearn.

All agree to the printed decimals.

## 9. Quick reference

### N-BEATS and PatchTST

| Procedure | Purpose |
|---|---|
| `forecast_windows(series, input_size, horizon)` | X [n, input_size], Y [n, horizon] |
| `nbeats_new(NBeatsConfig{...})` / `nbeats_free` | the model |
| `nbeats_parameters` / `nbeats_train_step(m, x, y, &opt)` / `nbeats_predict(m, x)` | training and forecasts |
| `nbeats_forward(m, x, alloc)` | forecasts as a graph tensor |
| `patchtst_new(PatchTSTConfig{...})` / `patchtst_free` | the model |
| `patchtst_parameters` / `patchtst_train_step(m, x, y, channels, &opt)` / `patchtst_predict(m, x, channels)` | training and forecasts |
| `patchtst_forward` / `patchtst_loss` | graph versions |

### Probabilistic and token forecasts

| Procedure | Purpose |
|---|---|
| `prob_forecaster_new(ProbForecasterConfig{...})` / `prob_forecaster_free` | the model |
| `prob_forecaster_add_to_adam` / `prob_forecaster_parameters` / `prob_forecaster_train(m, training)` | training |
| `prob_forecaster_forward(m, x)` / `prob_forecaster_loss(m, x, y)` | distributions / loss |
| `prob_forecast_gaussian_quantile` / `_var_es` / `_nll` / `_crps` | Gaussian forecasts |
| `prob_forecast_student_t_quantile` / `_var_es` / `_nll` / `_crps` | Student-t forecasts |
| `prob_forecast_sample_quantiles` / `prob_forecast_sample_crps` | sample forecasts |
| `prob_forecast_pinball` / `prob_forecast_coverage` / `prob_forecast_sort_quantiles` | quantile forecasts |
| `token_forecaster_new(TokenForecasterConfig{...})` / `token_forecaster_free` | Chronos-style model |
| `token_forecaster_add_to_adam` / `token_forecaster_loss(m, windows, batch)` | training (in a graph scope) |
| `token_forecaster_sample(m, context, n_samples, temperature, top_k)` | sample paths |

### Volatility

| Procedure | Purpose |
|---|---|
| `garch_recursive_forecast(returns, omega, alpha, beta)` | one-step GARCH volatilities |
| `volatility_forecaster_new` / `_forward` / `_free` | MLP forecaster |
| `lstm_volatility_forecaster_new(LSTMVolaConfig{...})` / `_forward` / `_free` / `lstm_volatility_add_to_optimizer` | sequence forecaster |
| `ensemble_volatility_new(input, hidden, seq_len, allocator, backend)` / `_free` / `ensemble_add_to_optimizer` | GARCH + model ensemble |
| `ensemble_predict(&e, garch_vol, model_vol)` / `compute_optimal_weight(garch, model, actual)` | the blend and its weight |
| `ensemble_term_structure_vol(model_vol, ω, α, β, current_var, days)` | volatility over a horizon |
| `ensemble_price_and_greeks(S, K, T, r, model_vol, ω, α, β, current_var, type)` | option price and Greeks |
| `compute_vrp(vix, rv_daily, mean, std)` / `print_vrp_signal` | VRP signal |
| `compute_backtest_metrics(&r)` / `print_backtest_result(&r)` / `vrp_backtest_result_free` | backtest summary |

### Regimes and pairs

| Procedure | Purpose |
|---|---|
| `hmm_new(n)` / `hmm_free` / `hmm_init_simple(&h, obs)` | the model and its start |
| `hmm_fit(&h, obs, max_iter, tol)` | Baum–Welch |
| `hmm_decode` / `hmm_filter` / `hmm_log_likelihood` | Viterbi path, filtered probabilities, ln P |
| `kalman_pairs_strategy(&df, x, y, config)` / `pairs_trading_result_free` | the strategy |
| `kalman_pairs_default_config()` / `DEFAULT_KALMAN_PAIRS_CONFIG` | defaults |
| `kalman_pairs_grid_search(&df, x, y, entries, exits, noises, holds, cooldowns, cost, allocator, trials)` | grid search, optional trial matrix |
| `extract_float_col(&df, name, allocator)` | a column as `[]f64` (nulls as NaN) |
