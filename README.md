<p align="center">
  <img src="https://raw.githubusercontent.com/benes83-fra/wotan/main/wotan%20logo.png" width="240" alt="Wotan logo">
</p>

<h1 align="center">Wotan</h1>

<p align="center">
  DataFrames, statistics, machine learning and quantitative finance, written in <a href="https://odin-lang.org/">Odin</a>.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/language-Odin-1e90ff.svg" alt="Odin">
  <img src="https://img.shields.io/badge/status-active%20development-orange.svg" alt="Status: active development">
</p>

---

Wotan is a library for data analysis and quantitative finance in Odin. It started as a typed DataFrame and grew into a toolkit covering time series, classical ML, a small autograd and deep learning stack, and a large set of finance models. It has no Python runtime dependency. The only external library is libcurl, used for HTTP requests.

> **Status:** Wotan covers a lot of ground, but it is still a work in progress. APIs can change between commits, documentation is thin in places, and some models have only been checked against the reference values in the test suite. Bug reports and feedback are very welcome.

## Contents

- [Features](#features)
- [Getting started](#getting-started)
- [Examples](#examples)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Optional: pretrained weights](#optional-pretrained-weights)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Features

**Data handling** (`core`, `importer`, `exporter`)
- Typed columns: `Int`, `Float`, `String`, `Bool`, `Date`, `Time`, `Datetime`
- Slicing, column selection, sorting, boolean masks and filtering
- Expression-based `select` (arithmetic, type conversion, custom procs)
- GroupBy with aggregations; inner, left, right and outer joins on one or more keys
- Read and write CSV, JSON, JSONL, HTML tables and Excel (`.xlsx`)
- Date, time and datetime arithmetic

**Statistics and time series** (`analytics`, `linalg`)
- Linear algebra: LU, QR, Cholesky, SVD and eigen decompositions, with SIMD kernels
- Rolling and exponentially weighted statistics, rolling covariance and PCA
- ARMA, ARIMA, SARIMA; GARCH and regime-switching GARCH
- Diagnostics: ACF/PACF, Ljung–Box, Jarque–Bera, ADF, KPSS
- Kalman filter and RTS smoother, EKF, UKF

**Machine learning** (`analytics/ML`)
- OLS, WLS, GLS, Ridge, Lasso (with cross-validation)
- Logistic regression, naive Bayes, KNN, k-means
- Decision trees, random forests, gradient boosting
- SVM, kernel SVM, SVR
- Pipelines, grid and random search, metrics, model serialization

**Deep learning** (`tensor`, `nn`, `optimize`)
- Reverse-mode autograd on tensors
- Dense, Conv2D, pooling, dropout, batch norm, RNN/GRU/LSTM
- Transformers (encoder/decoder, multi-head and flash attention), GPT and BERT style models, mixture-of-experts, Mamba
- GAN, WGAN, VAE; BPE and WordPiece tokenizers
- SGD, Adam, RMSProp, L-BFGS and learning-rate schedulers

**Quantitative finance** (`finance`, `ml_finance`, `behavioral_finance`)
- European, American (LSM) and exotic options; Greeks computed with autograd; implied volatility and volatility surfaces
- Bonds, yield curves, swaps, interest-rate, FX and credit derivatives, CDS bootstrapping
- Portfolio optimisation with constraints, HRP, risk decomposition, VaR, factor models, backtesting
- Monte Carlo simulation, copulas, fractional Brownian motion
- ML methods for finance: deep hedging, deep BSDE, DeepLOB, TimeGAN, PPO and CQL agents, Kalman pairs trading, HMM regimes, neural SDF, event studies
- Behavioural finance: prospect theory, anchoring, behavioural portfolio theory

**Data sources and output** (`net`, `plot`)
- Yahoo Finance prices, dividends, splits and option chains
- PNG charts: line, multi-line, scatter, histogram, bar and heatmap

## Getting started

### Requirements

- A recent [Odin](https://odin-lang.org/docs/install/) compiler
- libcurl, used through Odin's `vendor:curl` package. It is only needed for the `net` package (Yahoo Finance and HTTP).

### Build and run

```sh
git clone https://github.com/benes83-fra/wotan
cd wotan

odin run . -debug          # runs main.odin; debug builds report memory leaks
```

`main.odin` is the demo and test driver. Comment calls in or out there to choose which examples run.

To build Wotan as a library:

```sh
odin build . -o:speed -build-mode:static
odin build . -o:speed -build-mode:dynamic
```

`makeall.sh` and `makeall.bat` build all three variants.

### Using Wotan in your project

Wotan is a set of ordinary Odin packages. Copy or submodule the `wotan/` directory and import the packages you need:

```odin
import w     "wotan/core"
import io    "wotan/importer"
import ts    "wotan/analytics"
import fin   "wotan/finance"
import net   "wotan/net"
```

## Examples

The examples below are shortened. Every one of them has a complete, runnable version in [`tests/`](tests/).

### DataFrames: load, filter, group, join

```odin
df := io.csv_load("people_dates.csv")            // column types are inferred
defer w.destroy_dataframe(&df)

// Filter: age < 31 AND active
m1   := w.mask_lt(w.column(&df, "age"), 31)
m2   := w.column_mask(w.column(&df, "active"))
mask := w.and(m1, m2)
young_active := w.wobei(&df, mask)
w.dataframe_pretty_print(&young_active, 10)

// Group by age and aggregate salary
gdf := w.groupby(&df, []string{"age"})
out := w.agg(&gdf, []w.Agg_Expr{
    w.count("n"),
    w.sum_agg("total_salary", w.column(&df, "salary")),
    w.avg_agg("avg_salary",   w.column(&df, "salary")),
})
w.dataframe_pretty_print(&out, 10)
```

Joins work on one or more keys:

```odin
people := w.df_from(
    w.column_from_ints("id",      []int{1, 2, 3}),
    w.column_from_strings("name", []string{"Alice", "Bob", "Charlie"}),
)
salary := w.df_from(
    w.column_from_ints("id",       []int{1, 2, 4}),
    w.column_from_floats("salary", []f64{50000, 42000, 90000}),
)

joined := w.join(&people, &salary, []string{"id"}, .Outer)
w.dataframe_pretty_print(&joined, 10)
```

### Option pricing with Greeks from autograd

Greeks are computed by differentiating the pricing graph, not by finite differences.

```odin
S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.20

price, g := fin.price_and_greeks(S, K, T, r, sigma, .Call)
fmt.printf("price=%.4f  delta=%.4f  gamma=%.4f  vega=%.4f\n",
    price, g.delta, g.gamma, g.vega)
// price=10.4506  delta=0.6368  gamma=0.0188  vega=0.3752

iv, converged, iters := fin.implied_volatility(10.45, S, K, T, r, .Call)
fmt.printf("implied vol=%.4f (converged=%v after %d iterations)\n", iv, converged, iters)
```

See `tests/american_options_test.odin`, `tests/exotic_pricing_test.odin` and `tests/vol_surface_test.odin` for more.

### Market data and volatility forecasting

```odin
df := net.read_yahoo("AAPL", .Daily, .FiveYears)
defer w.destroy_dataframe(&df)

// ...compute log returns from the "Close" column...

fit := ts.garch_fit(returns, .GARCH, p = 1, q = 1)
fmt.printf("omega=%.6f alpha=%.4f beta=%.4f\n",
    fit.params.omega, fit.params.alpha[0], fit.params.beta[0])

fc := ts.garch_forecast(&fit, returns, horizon = 5)
fmt.printf("5-day vol forecast: %.4f\n", fc.std_forecast)
```

ARIMA works the same way and returns prediction intervals:

```odin
fit := ts.arima_fit(y, 1, 0, 1)
fc  := ts.arima_forecast(y, fit, 1, 0, 1, h = 20, alpha = 0.05)
// fc.mean, fc.lower, fc.upper
```

### Kalman filter pairs trading

```odin
import ml_fin "wotan/ml_finance"

// aligned_df holds two date-aligned price columns
config := ml_fin.kalman_pairs_default_config()
config.entry_threshold = 2.0
config.exit_threshold  = 0.5

result := ml_fin.kalman_pairs_strategy(&aligned_df, "Asset_A", "Asset_B", config)
// result.betas    – dynamic hedge ratio
// result.z_scores – spread z-score
// result.returns  – strategy returns
```

The full version, including date alignment and a parameter grid search, is in `tests/kalman_pairs_test.odin`.

### Autograd

```odin
import l "wotan/linalg"
import t "wotan/tensor"

a := t.tensor_new(l.matrix_new(f64, 2, 2), true)   // requires_grad = true
b := t.tensor_new(l.matrix_new(f64, 2, 2), true)
// ...fill a.data.data and b.data.data...

loss := t.tensor_sum(t.tensor_relu(t.tensor_matmul(a, b)))
t.tensor_backward(loss)
// a.grad and b.grad now hold d(loss)/da and d(loss)/db
```

The same engine runs the neural network layers in `nn`, the deep hedging and deep BSDE solvers, and the Greeks calculation shown above.

### Plotting

```odin
import plot "wotan/plot"

plot.line_png(xs, ys, "equity_curve.png")
```

## Project layout

```
wotan/
├── core/               DataFrame, columns, masks, select, groupby, join, dates
├── importer/           CSV, JSON, JSONL, HTML, Excel readers (with type inference)
├── exporter/           CSV, JSON, HTML, Excel writers
├── linalg/             matrices and decompositions (LU, QR, SVD, eigen, Cholesky)
├── analytics/          time series, statistical tests, Kalman filters, GARCH
│   └── ML/             classical machine learning
├── tensor/             autograd engine
├── nn/                 neural network layers and models, tokenizers
├── optimize/           optimizers and learning-rate schedulers
├── finance/            pricing, fixed income, portfolio, risk, backtesting
├── ml_finance/         ML methods for finance
├── behavioral_finance/ prospect theory, anchoring, behavioural portfolios
├── net/                HTTP via libcurl, Yahoo Finance
├── plot/               PNG chart rendering
└── data/               dataset loaders (MNIST)
tests/                  runnable examples and regression tests
scripts/                scripts to download model Data or Models (ie. MNIST, finBert, distilgpt2)
demo_file/              files used in tests and demo
docs/                   Markdown Documentation
plots/                  plots that get generate during the demo
model_checkpoints/      Neural Network Checkpoints that are either imported from HuggingFaces (so far finBERT and distilgpt2) or are generate by the demo
main.odin               demo and test driver
```

## Tests

`tests/` contains over 100 test procedures, one per area: DataFrame operations, I/O, linear algebra, every ML model, the deep learning stack, and the finance modules. Each is a plain `proc(allocator: mem.Allocator)` called from `main.odin`. Most print results next to known reference values. Some write PNG charts to the plots/ directory.

Tests whose names contain `real_data` fetch live data from Yahoo Finance and need network access. [`test_catalog.md`](docs/test_catalog.md) gives an overview of what each test covers.

Run with `-debug` to turn on the tracking allocator. It reports any leaked allocation when the program exits.

## Optional: pretrained weights

The FinBERT and GPT-2 examples can load Hugging Face weights. The conversion scripts need Python with `transformers` installed:

```sh
python convert_bert.py ProsusAI/finbert finbert_checkpoint.bin
python convert_gpt2.py
```

Without these checkpoints the examples still run, but with randomly initialised weights.

## Limitations

These are areas where Wotan is known to be incomplete:

- **API stability:** names and signatures are still changing. There are no versioned releases yet.
- **Documentation:** apart from this README, the tests are the main documentation.
- **Performance:** everything runs on the CPU. Some hot paths use SIMD, but most code has not been profiled or tuned, and large deep learning models are slow.
- **Validation:** models are tested against textbook and reference values. They have not been validated for production trading or risk use.
- **Memory management:** ownership is explicit, as is usual in Odin. Most objects need a matching `destroy_*` or `*_free` call.

## Contributing

Contributions, bug reports and feature requests are welcome. Please open an issue or a pull request. When you add a feature, please also add a test in `tests/` that shows how to use it.

## License

[MIT](LICENSE) © 2026 Benjamin Eßer
