# `main.odin` — Analysis & Summary

## Overview

`main.odin` is the CLI entry point / demo app for **Wotan**, a large-scale data analytics, machine learning, and finance library written in [Odin](https://odin-lang.org/). It exercises nearly every public API by constructing small DataFrames from scratch and running regression tests against each subsystem.

---

## 1. Core Framework (`wotan/core`)

| Feature | Description |
|---------|-------------|
| **DataFrames** | Mutable, typed-tabular data structure (`ww.dataframe_new`, `w.column_new`, `w.add_column`, etc.) with columns of type Int, String, Float64, Boolean, Date, Datetime, Time. |
| **Column slicing** | `df_series` extracts a single column as a Series; indexing returns value + null flag. |
| **Row/col slicing** | `dataframe_slice_rows` / `dataframe_select_columns` — positional and named projection. |
| **Filtering** | Boolean-mask filtering (`dataframe_filter_bool_column`, `filter`) and mask combinators (`mask_and`, `mask_less-than`). `wobei` applies a mask inline. |
| **Select / expressions** | Eager expression builder supports: column reference, add/subtract/divide/round operations, boolean masking, apply-proc lambdas, type coercion (int→float, etc.). |
| **Date/Datetime arithmetic** | `Date`, `Time`, `Datetime` structs with month/day/hour/second increment and diff helpers; `now()` constructor. |
| **Memory tracking** | In `ODIN_DEBUG` builds, a `mem.Tracking_Allocator` wraps the default allocator; on exit it prints per-call-site leak info (`fmt.eprintf("%v leaked %v bytes\n")`). |

---

## 2. Import / Export (`wotan/importer`, `wotan/exporter`)

| Format | CSV / JSON / JSONL / HTML / Excel (`xlsx`) / ZIP — all exercised through tests. |
|--------|-----------------------------------------------------------------------------|| Date parsing from CSV is validated via dedicated date export tests. HTTP GET support is present (`http_get_test`). Yahoo Finance JSON/Events endpoints are included, enabling market-data pull workflows.

## 3. Linear Algebra & Statistics (`wotan/linalg`, `wotan/tensor`)

- **SIMD-accelerated** BLAS-like ops (vector sub, rotation).
- **LU** decomposition: basic → singular, determinant, solve, inverse, numeric dump.
- **SVD / Thin-SVD**: full, rank-deficient, nearly-rank-deficient, 1×1, wide/narrow matrix variants. Golub-Reinsch implementation path.
- **EIG** diagonal & symmetric eigendecomposition; condition-number helpers (`cond2`).
- **QR** decomposition, Kronecker product.
- **Correlation** matrix computation.

## 4. Classical ML (`wotan/ml`)

| Subsystem | Key tests exercised |
|-----------|---------------------|
| **OLS (Ordinary Least Squares)** | Basic + full pipeline regression. |
| **WLS / GLS / Ridge / Lasso** | Weighted, generalized least-squares; regularised linear (RidgeCV, LassoCV cross-validation). |
| **Decision Tree / Random Forest / Gradient Boosting** | `dt_test`, `rf_test`, `gb_test`; with tree serialization. |
| **Logistic Regression** | Binary + real-world + multiclass variants. |
| **KNN** — k-closest-neighbor classification. |
| **PCR (Principal Component Regression)** | PCA + regression two-step. |
| **Gaussian Naive Bayes** | `gnb_test`. |
| **K-Means clustering** | `km_test`, `km` module. |
| **SVM / Kernel SVM / SVR** | Linear and kernelised support-vector machines; regression variant. |
| **MLP (Multilayer Perceptron)** — feedforward neural net with autograd, MSE training, classification head, Dropout. |
| **Metrics & Model Selection** | Classification metrics, train/test splits, GridSearchCV, RandomizedSearchCV, Pipelines (serial + comprehensive). |

## 5. Deep Learning (`wotan/nn`)

- **Layers**: Conv2D, Pooling, Dense with activation helpers (`ReLU`), Bias, Dropout.
- **Network API**: `sequential`, flexible network builder.
- **Activations / Optimizers**: ReLU, Adam optimizer (basic + classification variant).
- **CNN demo**: MNIST CNN loader + augmentation pipeline (training / test models, augmented model path).
- **BatchNorm + RNN family**: Simple RNN, GRU, LSTM; Embedding layer.
- **Transformer architecture**: Positional encoding → multi-head attention → Layer Norm → FFN → encoder block → full encoder → decoder; reversal test present.
- **Language Models**: Character-level LM, GPT (full + Shakespeare fine-tune path), BERT model.
- **Generative models**: GAN v2, WGAN, VAE.
- **Transfer Learning**: Generic transfer learning utilities.

## 6. Time Series & Econometrics (`wotan/analytics`)

| Category | Subsystems / tests |
|----------|--------------------|
| **Autocorrelation** | ACF/PACF helpers; Ljung-Box test for randomness. |
| **Normality** | Jarque-Bera normality test. |
| **Stationarity** | ADF (Dickey-Fuller), KPSS, composite stationarity block. |
| **ARIMA / SARIMA** | `arima_test`, fitting, auto-ARIMA, seasonal (SARIMA), MC simulation paths (ARMA(1,1), ARMA(2,2), ARIMA(p,d,q)). Residual diagnostics included. |
| **Rolling stats** | Rolling windows / rolling matrices. |
| **Resamplers** `residuals`, `ewm` (exponentially-weighted mean), PCA / covariance via EWMA. |

## 7. Filtering & State Estimation (`wotan/filter`)

- **Kalman Filter** — standard, time-varying control variant.
- **UKF (Unscented Kalman Filter)** — basic, control mode, RTS smoother.
- **EKF (Extended Kalman Filter)** — tiny test harness and RTS smoother.

## 8. Optimisation & Portfolio (`wotan/optimalz/portfolio`, `risk`)

- **Portfolio** construction, constraint optimisation, risk analytics.
- Monte Carlo / constraint-based solvers via the optimizer module.

## 9. Plotting / Viz (`wotan/plot`)

| Chart type | Tests exercised |
|------------|-----------------|
| Line plot (basic, multi-line) | `line_plot_test`, `multi_line_test` |
| Bar chart | `bar_chart_test` |
| Heatmap | `heatmap_test` |

## 10. Auto-diff Engine (`wotan/autograd`)

- Back-prop through basic ops: multiplication, matmul, summation, ReLU.
- Bias-layer gradient tracking.

---

## How `main.odin` Works (Flow)

1. **Debug allocator setup** — wraps default allocator with a leak tracker; prints leaks on exit inside `ODIN_DEBUG`.
2. **Manual DataFrame construction** — builds tiny tables row-by-row, exercises head/print/slice/filter ops.
3. **CSV import** — loads `people.csv` and `people_dates.csv` to validate the full ingestion pipeline (typing columns, dates).
4. **Row slicing & boolean filter demos** — demonstrates positional slice, mask creation (`mask_lt`, `column_mask`), mask combinators (`and` via `mask_and`), and the inline `wobei`.
5. **Select expressions** — builds derived columns with math ops, lambda applies, type coercions, boolean masks.
6. **Date/time arithmetic demos** — increment/decrement loops for Date, Time, Datetime.
7. **GroupBy + Aggregation** — groups by column(s), counts/sums/average via `Agg_Expr`.
8. **Join demos** — single-key join (`join_single`) and multi-key join (`join`) with different inner/outer semantics. Plus the builder DSL `df_from`.
9. **Test suite invocation** — calls `test.*_test()` for ~100+ individual tests across every module listed above.
10. **Cleanup / memory drain** — explicit `destroy_dataframe` + `free_select_exprs` (guard-rails under debug allocator).

---

## Key Takeaways

- **Scope**: Wotan is a monolith covering data wrangling, classical ML, deep learning, time-series econometrics, portfolio optimization, and auto-diff.
- **Design philosophy**: Imperative, explicit-memory model (`destroy_*` calls) with an optional debug allocator for leak detection.
- **Column-first architecture**: DataFrames are collections of typed `Column`s; operations (filter, select, join) manipulate these columns with mask objects rather than in-place row iteration.
- **Deep-learning stack is comprehensive**: From basic MLP/CNN → RNN/GRU/LSTM → Transformer/BERT/GPT to GAN/WGAN/VAE. The library even ships augmentation pipelines and MNIST training code.
- **Finance focus**: ARIMA/SARIMA, Kalman/UKF/EKF state estimation, portfolio+risk analytics, Yahoo Finance import — strongly positioned for quantitative analysis workflows.
- **No implicit memory management beyond `defer free_all(context.temp_allocator)`**; the `when ODIN_DEBUG` block is a leak-detection net that will print every orphaned allocation on exit.
