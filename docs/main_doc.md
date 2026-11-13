# Wotan documentation

Wotan is a library for data analysis, statistics, machine learning and quantitative finance, written in [Odin](https://odin-lang.org/). This page is the entry point to its documentation. It gives an overview of the packages, says which guide covers what, and lists the conventions that all the guides share. For installation and building, see the [README](../README.md).

## Contents

1. [The packages at a glance](#1-the-packages-at-a-glance)
2. [The guides](#2-the-guides)
3. [Where to start](#3-where-to-start)
4. [Conventions used throughout](#4-conventions-used-throughout)
5. [Tests and validation](#5-tests-and-validation)
6. [Packages without a guide](#6-packages-without-a-guide)

## 1. The packages at a glance

The packages build on each other in layers. Lower layers do not depend on higher ones.

```
 applications   finance  ·  ml_finance  ·  behavioral_finance
                     │             │
 models         analytics (+ analytics/ML)  ·  nn (+ tokenizers)  ·  optimize
                     │                         │
 numerics       linalg  ─────────────────────  tensor (autograd)
                     │
 data           core (DataFrame)  ·  importer  ·  exporter  ·  net  ·  plot
```

| Package | Import as | What it does |
|---|---|---|
| `core` | `w "wotan/core"` | typed columns and the `DataFrame`: selection, masks, computed columns, sorting, indexes, groupby, joins, dates |
| `importer` / `exporter` | `io "wotan/importer"` | CSV, JSON, JSONL, HTML and Excel in and out |
| `linalg` | `l "wotan/linalg"` | the `Matrix` type, products, LU, Cholesky, QR, least squares, eigenvalues, SVD, SIMD kernels |
| `tensor` | `t "wotan/tensor"` | reverse-mode autograd, graph scopes, convolutions, recurrent cells, attention |
| `nn` | `nn "wotan/nn"` | layers, sequential models, losses, optimizers, GPT, BERT, GAN, VAE, checkpoints; `nn/tokenizers` for BPE and WordPiece |
| `optimize` | `optim "wotan/optimize"` | SGD, Adam, RMSProp, L-BFGS, least squares, learning-rate schedules |
| `analytics` | `ts "wotan/analytics"` | rolling and EW statistics, tests, ARIMA, GARCH, VAR, cointegration, Kalman filters |
| `analytics/ML` | `ml "wotan/analytics/ML"` | regression, classification, SVMs, trees and ensembles, k-means, scalers, metrics, cross-validation, pipelines |
| `finance` | `fin "wotan/finance"` | option pricing and simulation, volatility models, curves and bonds, swaps and credit, portfolios, risk, backtests |
| `ml_finance` | `mlf "wotan/ml_finance"` | machine learning for finance: labels and backtest statistics, asset pricing, forecasting, deep hedging, RL |
| `behavioral_finance` | `bf "wotan/behavioral_finance"` | prospect theory, behavioural portfolios, sentiment, herding, heuristics |
| `net` | `net "wotan/net"` | HTTP through libcurl, Yahoo Finance prices and option chains |
| `plot` | `plot "wotan/plot"` | PNG charts: lines, scatter, histograms, bars, heatmaps |

The aliases are the ones the guides use; any alias works.

## 2. The guides

Each guide is a user guide for one package, or one part of a large package. They all follow the same plan:
- **Setup** with the imports and the conventions of the package;
- **sections** that introduce one topic each, with runnable examples;
- **validation**: what the results were compared against;
- **quick reference**: a table of the procedures.

Every example in the guides was compiled and run against the current code, and the output shown is the real output. Where writing a guide turned up a bug, the bug was fixed, and the guide says so.

### Data and numerics

| Guide | Covers |
|---|---|
| [core](core_guide.md) | the data model, building and loading DataFrames, printing, selecting and filtering, computed columns, sorting, indexes, grouping, joins, dates and times, memory |
| [data in and out](data_io_guide.md) (`importer`, `exporter`, `net`) | reading and writing CSV, JSON, JSON Lines, HTML and Excel; HTTP; Yahoo Finance histories as a DataFrame, as arrays per ticker or as a date-matched panel; prices to returns |
| [linalg](linalg_guide.md) | the `Matrix` type, matrices from DataFrames, products, covariance and correlation, LU, Cholesky, QR, least squares, eigenvalues and SVD, vector kernels |
| [tensor](tensor_guide.md) | tensors and gradients, element-wise and matrix operations, reductions and losses, normalisation, convolution, recurrent cells and attention, the backward pass, graph scopes |
| [plot](plot_guide.md) | configuration, line charts, several series, scatter plots and histograms, bars, heatmaps, themes, axes, drawing your own charts |

### Statistics and machine learning

| Guide | Covers |
|---|---|
| [analytics](analytics_guide.md) | rolling and exponentially weighted statistics, covariance and PCA, autocorrelation and stationarity tests, decomposition and smoothing, ARIMA, SARIMA and ARIMAX, GARCH, VAR and Granger causality, cointegration, Kalman filters, regression diagnostics |
| [machine learning](ml_guide.md) (`analytics/ML`) | from DataFrames to model inputs, linear and regularised regression, Fama–MacBeth, classification, sample weights, SVMs, trees and ensembles, the MLP, k-means, preprocessing, metrics, cross-validation, hyper-parameter search, pipelines, saving models |
| [neural networks](nn_guide.md) | layers, MLPs, sequential models, losses, optimizers and training loops, GPT, BERT, GAN and VAE, saving and loading, transfer learning, tokenizers |

### Finance

The `finance` package has six parts:

| Guide | Covers |
|---|---|
| [1. options and simulation](finance_options_guide.md) | Black–Scholes, implied volatility, Greeks by autograd, Monte Carlo exotics, American options (trees, finite differences, Longstaff–Schwartz), COS pricing, quasi-Monte Carlo, rough volatility |
| [2. volatility models, exotics and FX](finance_volatility_guide.md) | SABR, Heston pricing and calibration, exotics under Heston and Merton, local volatility, variance swaps, FX options |
| [3. dates, curves, bonds and inflation](finance_fixed_income_guide.md) | calendars, day counters, schedules, yield curves, bonds, callable and convertible bonds, inflation |
| [4. swaps, rate options and credit](finance_rates_credit_guide.md) | swaps, Hull–White caps, floors and swaptions, G2++ and Bermudans, CDS, CDO tranches, copulas |
| [5. portfolio construction and risk analytics](finance_portfolio_guide.md) | mean–variance and constrained portfolios, Black–Litterman, CVaR, risk parity, factor models, rolling analytics |
| [6. risk measurement, backtesting and market data](finance_risk_guide.md) | VaR and expected shortfall, extreme value theory, GARCH risk, performance metrics, backtesting, Yahoo Finance data |

The `ml_finance` package has five parts:

| Guide | Covers |
|---|---|
| [1. methods](ml_finance_methods_guide.md) | triple-barrier labels, sample weights, fractional differencing, feature importance, deflated Sharpe ratio and backtest overfitting, rank IC, HRP, transaction costs, conformal intervals |
| [2. asset pricing](ml_finance_asset_pricing_guide.md) | instrumented PCA, the conditional autoencoder, the neural SDF, event studies, implied correlation and dispersion |
| [3. forecasting](ml_finance_forecasting_guide.md) | N-BEATS and PatchTST, probabilistic and token forecasts, volatility ensembles, hidden Markov regimes, Kalman pairs trading |
| [4. derivatives, execution and synthetic data](ml_finance_derivatives_guide.md) | optimal execution and market making, deep BSDE, deep calibration, deep hedging, TimeGAN |
| [5. RL, order books, text and integration](ml_finance_rl_integration_guide.md) | trading environments, PPO, offline CQL, DeepLOB, graph attention, FinBERT sentiment and the Sharpe loss, the orchestrator |

And one more guide:

| Guide | Covers |
|---|---|
| [behavioural finance](behavioral_finance_guide.md) | prospect theory and option prices under it, portfolio choice, cross-sectional signals, the disposition effect, herding, investor sentiment, equilibrium models, heuristics |

The [test catalog](test_catalog.md) lists every test file and what it checks.

## 3. Where to start

Every path starts with the [core guide](core_guide.md), since most of Wotan reads and writes DataFrames. To load your own files or market data, read the [data in and out guide](data_io_guide.md) next. After that:

- **Data analysis and statistics:** [analytics](analytics_guide.md), then [machine learning](ml_guide.md); [plot](plot_guide.md) for charts.
- **Deep learning:** [tensor](tensor_guide.md) for autograd and graph scopes, then [neural networks](nn_guide.md).
- **Derivatives pricing:** finance parts [1](finance_options_guide.md) and [2](finance_volatility_guide.md); for rates and credit, parts [3](finance_fixed_income_guide.md) and [4](finance_rates_credit_guide.md).
- **Portfolios and risk:** finance parts [5](finance_portfolio_guide.md) and [6](finance_risk_guide.md), then [ml_finance part 1](ml_finance_methods_guide.md) for the statistics that keep backtests honest.
- **Machine learning for finance:** the [tensor](tensor_guide.md) and [neural network](nn_guide.md) guides first, then the five `ml_finance` parts. Part 1 is the foundation for the others.

The linear algebra guide is for when you need matrices directly. Most packages take and return slices or DataFrames.

## 4. Conventions used throughout

**Numbers.** Values are `f64` throughout. Rates, volatilities and returns are decimals (0.05 for 5%), and time is in years unless a guide says otherwise.

**Memory.** Ownership is explicit, as usual in Odin: whoever creates something frees it.
- A procedure that returns a DataFrame, a model or a result struct gives you something to free, with the matching `destroy_*` or `*_free` procedure or `delete` for plain slices. Each guide has a section on the rules of its package.
- Procedures that allocate take an `allocator` parameter, which defaults to `context.allocator`. Some use `context.temp_allocator` for scratch space. In a loop, clear it with `free_all(context.temp_allocator)`.
- Put a `defer` right after each allocation, as the examples do.
- Build with `-debug` and wrap `context.allocator` in a `mem.Tracking_Allocator`, and every leak is reported with the line that allocated it. The [core guide](core_guide.md#1-setup) shows the setup.

**Graph scopes.** Tensor operations allocate their outputs in the active `Graph_Scope`: one arena per forward and backward pass. `graph_end` frees the whole graph at once. Parameters live outside the scope and survive it. Copy out any value you need before `graph_end`. The [tensor guide](tensor_guide.md#11-memory-management-and-graph-scopes) explains the details. Some models own a scope and use it in their training procedures; the guides show which.

**Failure.** Network calls such as `net.read_yahoo` return `(value, ok)`, with `ok` false when a request fails or returns no data. The file loaders panic when a file cannot be read, except `xlsx_load`, which returns an empty DataFrame. Numerical procedures return NaN when a result is undefined, for example a correlation from one component, and the guides say where that happens. Many procedures check their inputs, such as matching lengths, with an `assert` that names the procedure.

**Randomness.** Procedures that sample use Odin's global generator. Call `rand.reset(seed)` first for reproducible results. The guides' examples do.

## 5. Tests and validation

`main.odin` is the test driver. It calls the test procedures in `tests/`, about 440 of them in about 250 files. Run it with `odin run . -o:speed -debug` from the repository root. The full run takes a while. To run only some tests, comment the others out in `main.odin`, or call them from your own small `main`.

There are three kinds of test:
- **Reference tests** compare results with established libraries: QuantLib, statsmodels, SciPy, scikit-learn, PyTorch, Hugging Face, hmmlearn, pykalman, TA-Lib, NumPy and pandas. The reference values are hard-coded in the tests.
- **Contract tests** check properties: gradients against finite differences, identical results across thread counts, save/load round trips, memory fully released.
- **Examples and demos** print results, often next to published values. Some write charts to `plots/`.

Checks print `OK` or `FAIL`. Failed checks are counted, and the program exits with code 1 if any failed. Tests that need real market data report an unreachable source as a failure. They never substitute made-up data. Almost every test seeds the random generator from its own name, so results do not depend on which tests ran before.

Each guide's validation section lists what its examples and tests were compared against. The [test catalog](test_catalog.md) has the details per file; tests marked 🌐 need network access.

## 6. Packages without a guide

A few small packages support the others and have no guide of their own:

| Package | Contents | Documented in |
|---|---|---|
| `optimize` | optimizers and learning-rate schedules | [neural networks, section 6](nn_guide.md#6-optimisers-and-training-loops); L-BFGS and least squares in the [ML](ml_guide.md) and [portfolio](finance_portfolio_guide.md) guides |
| `data` | the MNIST loader | the CNN tests; `scripts/` downloads the data |
| `util` | the worker pool for parallel loops, quicksort | used internally |
| `zip_min` | a minimal ZIP reader for the Excel importer | used internally |
