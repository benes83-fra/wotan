# The `analytics` package: a user guide

`wotan/analytics` is the statistics and time-series layer of Wotan. It provides rolling and exponentially weighted statistics, covariance matrices and PCA, the usual diagnostic and unit-root tests, decomposition and exponential smoothing, ARIMA-type models, GARCH models, vector autoregressions, cointegration tests and Kalman filters. Most procedures work on plain `[]f64` slices; the rolling procedures and a few `df_*` wrappers work on `core` DataFrames. Where a procedure follows a reference library (pandas, statsmodels, SciPy, arch), its doc comment says so, and the reference tests in [`tests/`](../tests/) check it against hard-coded values from that library.

This guide walks through the package with small examples. It assumes you know basic Odin and have read the [core user guide](core_guide.md). Every example was compiled and run against the current code, and the output shown is the real output. The examples also free everything they allocate: a debug build with a tracking allocator reports no leaks.

The machine-learning subpackage `wotan/analytics/ML` (regressions, classifiers, trees and so on) is not covered here.

## Contents

1. [Setup](#1-setup)
2. [Series, DataFrames and matrices](#2-series-dataframes-and-matrices)
3. [Rolling statistics](#3-rolling-statistics)
4. [Exponentially weighted statistics](#4-exponentially-weighted-statistics)
5. [Covariance matrices and PCA](#5-covariance-matrices-and-pca)
6. [Autocorrelation and residual tests](#6-autocorrelation-and-residual-tests)
7. [Stationarity tests](#7-stationarity-tests)
8. [Decomposition and exponential smoothing](#8-decomposition-and-exponential-smoothing)
9. [ARIMA](#9-arima)
10. [SARIMA and ARIMAX](#10-sarima-and-arimax)
11. [GARCH and volatility](#11-garch-and-volatility)
12. [VAR and Granger causality](#12-var-and-granger-causality)
13. [Cointegration](#13-cointegration)
14. [Kalman filters](#14-kalman-filters)
15. [Regression diagnostics, bootstrap and optimisation](#15-regression-diagnostics-bootstrap-and-optimisation)
16. [Memory management](#16-memory-management)
17. [A complete example](#17-a-complete-example)
18. [Quick reference](#18-quick-reference)

## 1. Setup

Import the package under a short name. This guide uses `ts`, as the README and most tests do. You will usually also need `core` for DataFrames and `linalg` for the matrix type used by VAR, ARIMAX, cointegration and the regression tests:

```odin
import ts "wotan/analytics"
import w  "wotan/core"
import l  "wotan/linalg"   // only for VAR, ARIMAX, Johansen and the regression tests
import "core:fmt"
```

Build with `-debug` while you learn the library and install a tracking allocator, as described in the [core guide](core_guide.md#1-setup):

```odin
import "core:mem"

main :: proc() {
    track: mem.Tracking_Allocator
    mem.tracking_allocator_init(&track, context.allocator)
    context.allocator = mem.tracking_allocator(&track)
    defer {
        for _, e in track.allocation_map {fmt.printf("leak: %v bytes at %v\n", e.size, e.location)}
        mem.tracking_allocator_destroy(&track)
    }
    // ... your code ...
}
```

Several examples use the same 80-point series. It is a simulated ARMA(1,1) with φ = 0.85 and θ = 0.5, taken from [`stationarity_reference_test.odin`](../tests/stationarity_reference_test.odin), which also holds the statsmodels values for it:

```odin
Y := []f64{0, -0.069, 0.516, 0.863, 0.251, 0.307, 1.746, 3.083, 2.39, 0.414, -0.904, -1.039, -3.187, -4.09, -4.832, -5.463, -5.554, -5.309, -4.259, -2.372, -1.623, -0.078, -0.048, -0.022, 1.061, 1.447, 0.534, -0.84, -1.632, -1.396, -2.086, -2.487, -2.378, -1.56, -0.841, -0.252, -0.69, -1.043, -0.168, 1.743, 0.969, 1.708, 3.555, 4.476, 4.459, 3.609, 4.369, 6.403, 8.224, 9.206, 8.84, 6.485, 4.903, 4.822, 3.139, 2.419, 2.683, 3.192, 1.877, 0.342, -0.477, -1.793, -0.37, 0.059, 0.131, 0.018, 1.469, 3.361, 4.15, 1.641, 0.345, 1.003, 2.198, 1.753, 3.003, 2.143, 0.5, 1.029, 1.391, 3.21}
```

## 2. Series, DataFrames and matrices

The package uses three kinds of input:

| Input | Used by |
|---|---|
| `[]f64` | Tests, models and filters: ACF, ADF, ARIMA, GARCH, decomposition, bootstrap, ... |
| `^w.DataFrame` + column names | Rolling and EWM statistics, rolling covariance and PCA, `df_acf`, `df_pacf`, the volatility helpers |
| `^l.Matrix(f64)` (n × k, row-major) | VAR, Johansen, ARIMAX regressors, the regression tests |

To go from a DataFrame to the other two, use `linalg`:

```odin
df := w.df_from(
    w.column_from_floats("close", []f64{100, 101.5, 99.8, 102.2, 103.0, 101.9}),
    w.column_from_ints("volume", []int{1200, 900, 1500, 1100, 1300, 1000}),
)
defer w.destroy_dataframe(&df)

px := l.vector_from_df(&df, "close") // a []f64 copy (Int or Float column)
defer delete(px)

jb := ts.df_jarque_bera(px) // a one-row DataFrame
defer w.destroy_dataframe(&jb)
w.dataframe_pretty_print(&jb)

ac := ts.df_acf(&df, "volume", 2) // a Float Column
defer w.destroy_column(&ac)
v, _ := w.column_at_float(&ac, 1)
fmt.printf("%s has %d values, lag-1 = %.4f\n", ac.name, ac.len, v)

M := l.matrix_from_df(&df, []string{"close", "volume"}) // n x 2, row-major
defer l.matrix_free(&M)
fmt.println(M.rows, M.cols, l.matrix_at(&M, 2, 1)^) // 6 2 1500
```

```
    JB                  p_value
  0 0.5138051088095734  0.7734435787632925
acf has 3 values, lag-1 = -0.6476
```

`vector_from_df` and `matrix_from_df` read the raw column storage, so a null cell comes back as `0`. Filter out null rows first (see the core guide, section 9) if your column has any.

Many tests have a `df_*` twin that returns its results as a one-row DataFrame, which is handy for printing or for collecting results over several series: `df_adf`, `df_kpss`, `df_stationarity`, `df_ljung_box`, `df_jarque_bera`, `df_residual_diagnostics`. `df_acf` and `df_pacf` take a DataFrame column and return a `Column`.

## 3. Rolling statistics

A `RollingWindow` names a DataFrame column, a window length in rows and a minimum number of non-null values. `rolling_apply` computes one statistic for every row and returns a new `Column` of the same length. The statistic is a `core` `Aggregator`, the same type `groupby2_agg` uses:

```odin
quotes := []f64{100, 102, 101, 105, 107, 0, 110, 108}
price := w.column_new("close", .Float, len(quotes))
for v, i in quotes {
    if i == 5 {w.append_null(&price); continue} // a missing quote
    w.append_float(&price, v)
}
df := w.df_from(price)
defer w.destroy_dataframe(&df)

rw := ts.rolling_window(&df, "close", 3, 2) // 3 rows, at least 2 values
ma := ts.rolling_apply(rw, w.Aggregator{name = "ma3", kind = .Mean})
w.add_column(&df, ma) // df now owns ma

sd := ts.rolling_apply(rw, w.Aggregator{name = "var3", kind = .Var})
w.add_column(&df, sd)

rw.centered = true
cma := ts.rolling_apply(rw, w.Aggregator{name = "centered", kind = .Mean})
w.add_column(&df, cma)

w.dataframe_pretty_print(&df)
```

```
    close  ma3                 var3               centered
  0 100    NULL                NULL               101
  1 102    101                 2                  101
  2 101    101                 1                  102.66666666666667
  3 105    102.66666666666667  4.333333333333334  104.33333333333333
  4 107    104.33333333333333  9.333333333333332  106
  5 NULL   106                 2                  108.5
  6 110    108.5               4.5                109
  7 108    109                 2                  109
```

The rules follow pandas' `Series.rolling`:

- **The window counts rows, nulls included.** Row `i` uses rows `i - window + 1 .. i`, or a window centred on `i` when `centered` is set. Nulls and NaN inside the window are skipped, so row 5 above (a null) still gets the mean of 105 and 107.
- **A row is null when its window has fewer than `max(min_periods, 1)` values.** Row 0 has only one value, so it is null.
- **Variance and covariance use `n - 1`**, and need at least two values.
- **Types:** the mean, median and quantile of an `Int` column are `Float`; the sum, min, max and count stay `Int`.

`rolling_apply_many` applies several aggregators to the same window and returns a DataFrame. The window's column is the input; the aggregator's own `column` field is ignored. For `Cov` and `Corr`, the second column goes in `other`:

```odin
df := w.df_from(
    w.column_from_floats("a", []f64{1, 3, 2, 5, 4, 6}),
    w.column_from_floats("b", []f64{2, 2, 3, 4, 6, 5}),
)
defer w.destroy_dataframe(&df)

rw := ts.rolling_window(&df, "a", 4, 4)
stats := ts.rolling_apply_many(rw, []w.Aggregator {
    {name = "max", kind = .Max},
    w.make_median_agg("median", "a"),
    w.make_quantile_agg("q75", "a", 0.75),
    w.make_corr("corr_ab", "a", "b"), // second column in `other`
    w.make_cov("cov_ab", "a", "b"),
})
defer w.destroy_dataframe(&stats)
w.dataframe_pretty_print(&stats)
```

```
    max   median  q75   corr_ab             cov_ab
  0 NULL  NULL    NULL  NULL                NULL
  1 NULL  NULL    NULL  NULL                NULL
  2 NULL  NULL    NULL  NULL                NULL
  3 5     2.5     3.5   0.7644707871564383  1.25
  4 5     3.5     4.25  0.5291502622129182  1.1666666666666667
  5 6     4.5     5.25  0.5291502622129182  1.1666666666666667
```

The aggregation kinds for rolling windows:

| `kind` | Result |
|---|---|
| `.Sum`, `.Mean`, `.Min`, `.Max`, `.Count` | As named; `Count` counts non-null values |
| `.Median`, `.Quantile` | Linear interpolation between ranks; set `quantile` (or use `make_median_agg`, `make_quantile_agg`) |
| `.Var`, `.Cov`, `.Corr` | Sample variance and covariance (`n - 1`), Pearson correlation; `Cov` and `Corr` use only rows where both columns are non-null |
| `.EWM_Mean`, `.EWM_Var`, `.EWM_Std`, `.EWM_Cov`, `.EWM_Corr` | Exponentially weighted, see section 4 |

There is no rolling standard deviation kind: take the square root of `.Var`. Rolling `Min` and `Max` also work on `Date`, `Time`, `Datetime` and `String` columns, and on `Bool` columns `Sum` counts the `true` values.

## 4. Exponentially weighted statistics

The `ewm_*_column` procedures compute exponentially weighted moments with smoothing weight `alpha` on the newest value. They match pandas' `ewm(alpha=...)`. The `window` of the `RollingWindow` is ignored; `min_periods` still applies:

```odin
df := w.df_from(w.column_from_floats("r", []f64{0.01, -0.02, 0.015, 0.03, -0.01, 0.005}))
defer w.destroy_dataframe(&df)

rw := ts.rolling_window(&df, "r", 0, 1) // EWM ignores the window size

m_adj := ts.ewm_mean_column(rw, 0.5, adjust = true, name = "ewm_adj")
w.add_column(&df, m_adj)
m_rec := ts.ewm_mean_column(rw, 0.5, adjust = false, name = "ewm_rec")
w.add_column(&df, m_rec)
sd := ts.ewm_std_column(rw, 0.5, adjust = false, bias = false, name = "ewm_std")
w.add_column(&df, sd)
// the Aggregator route: always adjust = false
ag := ts.rolling_apply(rw, ts.make_ewm_mean("agg_mean", "r", 0.5))
w.add_column(&df, ag)

w.dataframe_pretty_print(&df)
```

```
    r      ewm_adj               ewm_rec                ewm_std               agg_mean
  0 0.01   0.01                  0.01                   NULL                  0.01
  1 -0.02  -0.01                 -0.005                 0.021213203435596427  -0.005
  2 0.015  0.004285714285714286  0.004999999999999999   0.018439088914585774  0.004999999999999999
  3 0.03   0.018000000000000002  0.017499999999999998   0.02                  0.017499999999999998
  4 -0.01  0.003548387096774194  0.003749999999999999   0.021962534944228788  0.003749999999999999
  5 0.005  0.004285714285714286  0.0043749999999999995  0.015525969163793715  0.0043749999999999995
```

- **`adjust = true`** weights the k-th most recent value by (1 − α)^k and divides by the sum of the weights (pandas' default). Row 1: (−0.02 + 0.5 · 0.01) / 1.5 = −0.01.
- **`adjust = false`** is the recursion m_t = (1 − α) m_{t−1} + α x_t. Row 1: 0.5 · 0.01 + 0.5 · (−0.02) = −0.005.
- **`bias = false`** applies the unbiased correction to variances, so the first variance is null.
- **Through `rolling_apply`** (`make_ewm_mean`, `make_ewm_var`, `make_ewm_cov`, `make_ewm_corr`, or `kind = .EWM_Std`) the EWM kinds always use `adjust = false`. Call the `ewm_*_column` procedures directly when you need `adjust = true` or `ignore_na`.

| Procedure | pandas equivalent |
|---|---|
| `ewm_mean_column(rw, alpha, adjust, ignore_na, name)` | `ewm(...).mean()` |
| `ewm_var_column(rw, alpha, adjust, bias, ...)`, `ewm_std_column` | `ewm(...).var(bias)`, `.std(bias)` |
| `ewm_cov_column(rw, other, alpha, adjust, bias, ...)` | `ewm(...).cov(other, bias)` |
| `ewm_corr_column(rw, other, alpha, adjust, ...)` | `ewm(...).corr(other)` |

`other` is a `^w.Column`, for example `w.column(&df, "b")`.

## 5. Covariance matrices and PCA

`pca_dataframe` runs a principal component analysis on DataFrame columns. Eigenvalues come in descending order and use the `n - 1` covariance, like scikit-learn's `explained_variance_`. Each row of `eigenvectors` is one component, with its largest loading made positive. The example uses the data of Lindsay Smith's well-known PCA tutorial, whose eigenvalues are 1.28402771 and 0.0490833989:

```odin
df := w.df_from(
    w.column_from_floats("x", []f64{2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1}),
    w.column_from_floats("y", []f64{2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9}),
)
defer w.destroy_dataframe(&df)
cols := []string{"x", "y"}

pca := ts.pca_dataframe(&df, cols, context.allocator)
defer ts.destroy_pca_result(pca)
fmt.println("eigenvalues:", pca.eigenvalues)
fmt.println("component 1:", pca.eigenvectors[0])
fmt.println("mean:", pca.mean)

ratio := ts.pca_explained_variance_ratio(pca)
defer delete(ratio)
fmt.printf("explained: %.4f %.4f\n", ratio[0], ratio[1])

data := ts.extract_numeric_matrix(&df, cols)
defer ts.destroy_matrix(data)
scores := ts.pca_transform(data[:3], pca) // first three rows
defer ts.destroy_matrix(scores)
ts.print_matrix(scores)
```

```
eigenvalues: [1.2840277121727837, 0.04908339893832736]
component 1: [0.6778733985280118, 0.7351786555444081]
mean: [1.81, 1.9099999999999997]
explained: 0.9632 0.0368
[   0.8280   0.1751 ]
[  -1.7776  -0.1429 ]
[   0.9922  -0.3844 ]
```

`extract_numeric_matrix` returns a `[][]f64` (one slice per row), the format that `pca_transform`, `pca_inverse_transform` and `covariance_matrix` use. Free it with `destroy_matrix`. `pca_transform` subtracts the fitted mean; `pca_inverse_transform` adds it back and also accepts scores that keep only the first k components.

`pca_dataframe` and `extract_numeric_matrix` accept `Float`, `Int` and `Bool` columns (`Bool` as 0 and 1). A null cell is read as 0, which biases the mean and covariance, so drop rows with nulls first.

**Rolling and EWM versions.** `rolling_cov_matrix` computes the covariance of every pair of columns over a rolling window. It returns a long DataFrame with the columns `row`, `col_i`, `col_j` and `cov`; `extract_cov_matrix_row` turns one row back into a k × k matrix. `rolling_pca` runs a PCA for every row and returns one `PCAResult` per row, empty while the window is not full. `ewm_pca_last` runs a PCA on the EWM covariance at the last row:

```odin
cov := ts.rolling_cov_matrix(&df, cols, 5, 5)
defer w.destroy_dataframe(&cov)
last := ts.extract_cov_matrix_row(&cov, cols, 9) // the window ending at row 9
defer ts.destroy_matrix(last)
ts.print_matrix(last)

rp := ts.rolling_pca(&df, cols, 5, 5)
defer {
    for r in rp {ts.destroy_pca_result(r)}
    delete(rp)
}
fmt.println(len(rp[3].eigenvalues), rp[9].eigenvalues)

e := ts.ewm_pca_last(&df, cols, alpha = 0.3, min_periods = 1, bias = false, adjust = true)
defer ts.destroy_pca_result(e)
fmt.println(e.eigenvalues)
```

```
[   0.3170   0.3545 ]
[   0.3545   0.4870 ]
0 [0.7665480078124141, 0.037451992187585825]
[0.9420474700178806, 0.032998596216736426]
```

`rolling_corr_matrix` returns the same long format with a `corr` column; `rolling_var_matrix` has the columns `row`, `col` and `var`. `ewm_cov_matrix` gives the EWM covariances in the same long format, and `ewm_pca` gives one EWM PCA per row.

## 6. Autocorrelation and residual tests

`acf` and `pacf` return `max_lag + 1` values; index 0 is lag 0 (always 1). The ACF uses the biased estimator and the PACF the Durbin–Levinson recursion, as statsmodels' `acf` and `pacf(method="ldb")`:

```odin
y := Y
ac := ts.acf(y, 5) // lags 0..5; ac[0] = 1
defer delete(ac)
fmt.printf("acf : %.4f\n", ac[1:])

pc := ts.pacf(y, 5)
defer delete(pc)
fmt.printf("pacf: %.4f\n", pc[1:])

Q, dof, p_lb := ts.ljung_box(y, 10, 0)
fmt.printf("Ljung-Box Q = %.2f, df = %d, p = %.3g\n", Q, dof, p_lb)

jb, p_jb := ts.jarque_bera(y)
fmt.printf("Jarque-Bera = %.4f, p = %.4f\n", jb, p_jb)

mean, variance, skew, kurt := ts.sample_moments(y)
fmt.printf("mean %.4f var %.4f skew %.4f kurt %.4f\n", mean, variance, skew, kurt)

diag := ts.df_residual_diagnostics(y, 10, 0)
defer w.destroy_dataframe(&diag)
w.dataframe_pretty_print(&diag)
```

```
acf : [0.9278, 0.8083, 0.6965, 0.5987, 0.4905]
pacf: [0.9278, -0.3774, 0.1151, -0.0387, -0.1891]
Ljung-Box Q = 242.46, df = 10, p = 2.09e-46
Jarque-Bera = 2.7296, p = 0.2554
mean 0.8822 var 8.9612 skew 0.3348 kurt 3.6088
    mean                variance           skewness            kurtosis           JB                  JB_p                 LB_Q               LB_df  LB_p
  0 0.8822124999999998  8.961238992343752  0.3347646339655465  3.608786294269308  2.7296339756883388  0.25542741803319857  242.4579016099503  10     2.087358347252862e-46
```

The ACF, PACF, Ljung–Box Q and Jarque–Bera values agree with the statsmodels and SciPy values in the reference test. The huge Q says what the ACF already shows: the series is strongly autocorrelated, as an ARMA(1,1) with φ = 0.85 should be.

- **`ljung_box(v, max_lag, dof_adj)`**: `dof_adj` is the number of fitted ARMA parameters (p + q) when you test model residuals. The degrees of freedom are `max_lag - dof_adj`, at least 1.
- **`jarque_bera`** and **`sample_moments`** report the kurtosis, not the excess kurtosis: a normal distribution has 3.
- **`df_residual_diagnostics(residuals, max_lag, dof_adj)`** collects the moments, Jarque–Bera and Ljung–Box in one row.

## 7. Stationarity tests

The package has three unit-root and stationarity tests:

| Test | Null hypothesis | Procedure |
|---|---|---|
| Augmented Dickey–Fuller | Unit root | `adf_test(y, max_lags, reg_type, lag_sel)` |
| Phillips–Perron | Unit root | `phillips_perron_test(y, lags, trend, test_type)` |
| KPSS | Stationary (around a level or a trend) | `kpss_test(y, kind, nw_lags)` |

A small ADF or PP p-value rejects the unit root; a small KPSS p-value rejects stationarity. `stationarity_test` runs ADF and KPSS together and combines them at the 5 % level:

```odin
y := Y
stat, p, lags, nobs, c1, c5, c10 := ts.adf_test(y, 2, .Constant, .Fixed)
fmt.printf("ADF  stat %.4f  p %.4f  lags %d  nobs %d  crit %.3f %.3f %.3f\n", stat, p, lags, nobs, c1, c5, c10)

stat, p, lags, nobs, _, c5, _ = ts.adf_test(y, 6, .ConstantTrend, .AIC)
fmt.printf("ADF ct/AIC stat %.4f  p %.4f  lags %d  nobs %d\n", stat, p, lags, nobs)

k, kp, k10, k5, k25, k1 := ts.kpss_test(y, .Level, -1)
fmt.printf("KPSS stat %.4f  p %.4f  crit %.3f %.3f %.3f %.3f\n", k, kp, k10, k5, k25, k1)

pp := ts.phillips_perron_test(y)
fmt.printf("PP   stat %.4f  p %.4f  lags %d  rho %.4f\n", pp.statistic, pp.p_value, pp.lags, pp.rho)

decision, a_stat, a_p, k_stat, k_p := ts.stationarity_test(y)
fmt.println(decision, a_stat, a_p, k_stat, k_p)

dy := ts.difference(y, 1)
defer delete(dy)
decision, _, a_p, _, k_p = ts.stationarity_test(dy)
fmt.printf("%v  adf p %.4f  kpss p %.4f\n", decision, a_p, k_p)

table := ts.df_adf(y, 2)
defer w.destroy_dataframe(&table)
w.dataframe_pretty_print(&table)

st := ts.df_stationarity(y)
defer w.destroy_dataframe(&st)
w.dataframe_pretty_print(&st)
```

```
ADF  stat -1.9364  p 0.3151  lags 2  nobs 77  crit -3.518 -2.900 -2.587
ADF ct/AIC stat -2.9076  p 0.1598  lags 1  nobs 78
KPSS stat 0.4459  p 0.0574  crit 0.347 0.463 0.574 0.739
PP   stat -1.8497  p 0.3561  lags 12  rho 0.9350
Inconclusive -2.4797290305791857 0.12048469407822571 0.44593939358729084 0.05735370966065051
Stationary  adf p 0.0000  kpss p 0.1000
    adf_stat             p_value             lags_used  n_obs  crit_1pct           crit_5pct           crit_10pct
  0 -1.9364019882978807  0.3151349179078138  2          77     -3.518281134660583  -2.899878185191432  -2.5872229937594873
    decision      adf_stat             adf_p                kpss_stat            kpss_p
  0 Inconclusive  -2.4797290305791857  0.12048469407822571  0.44593939358729084  0.05735370966065051
```

The two ADF lines and the KPSS line reproduce statsmodels' `adfuller` and `kpss` (the values are in the reference test). With φ = 0.85 and only 80 points, ADF cannot reject a unit root and KPSS only just fails to reject stationarity, so the combined verdict is `Inconclusive`. The differenced series is clearly stationary.

**ADF options.** `reg_type` is `.None`, `.Constant`, `.Trend` or `.ConstantTrend` (statsmodels' `"n"`, `"c"`, `"ct"`). With `lag_sel = .Fixed` the regression uses exactly `max_lags` lagged differences; with `.AIC` or `.BIC` every lag up to `max_lags` is tried on a common sample and the best is refitted, as statsmodels' `autolag` does. P-values are MacKinnon (1994), critical values MacKinnon (2010).

**KPSS options.** `kind` is `.Level` or `.Trend`. `nw_lags = -1` selects the Newey–West bandwidth automatically (statsmodels' `nlags="auto"`). The p-value is interpolated in the KPSS table and clamped to [0.01, 0.10], which is why the differenced series shows exactly 0.1000.

**Phillips–Perron** returns a `PPResult`. `trend` is a `CointTrend` (`.None`, `.Constant`, `.ConstantTrend`); `lags = -1` uses ⌈12 (n/100)^¼⌉. With `test_type = .Rho` you get the Z_ρ statistic without a p-value.

**`stationarity_test`** returns a `StationarityDecision`:

| ADF rejects? | KPSS rejects? | Decision |
|---|---|---|
| yes | no | `.Stationary` |
| no | yes | `.DifferenceStationary` |
| yes | yes | `.TrendStationary` |
| no | no | `.Inconclusive` |

The last two labels are this library's convention. The statsmodels documentation reads these two cases the other way round. `auto_arima_d_from_tests` uses the decision to choose the differencing order: `d = 0` for `.Stationary` and `.TrendStationary`, otherwise `d = 1`.

The tests return plain values; only their `df_*` versions allocate (a DataFrame to destroy).

## 8. Decomposition and exponential smoothing

The examples in this section use six years of quarterly sales:

```odin
SALES := []f64{
    13.2, 9.8, 11.8, 17.2,
    15.1, 11.7, 13.4, 19.1,
    16.3, 13.6, 15.3, 19.1,
    17.6, 13.6, 14.9, 19.5,
    18.2, 14.7, 16.7, 22.2,
    19.8, 16.0, 18.1, 22.8,
}
```

`seasonal_decompose` is the classical moving-average decomposition, `stl_decompose` the STL method, and `hp_filter` the Hodrick–Prescott filter. They match statsmodels' `seasonal_decompose`, `STL` and `hpfilter`:

```odin
x := SALES

cls := ts.seasonal_decompose(x, 4, .Additive)
defer ts.decompose_free(&cls)
fmt.printf("seasonal: %.3f\n", cls.seasonal[:4])
fmt.printf("trend   : %.3f\n", cls.trend[:6])

stl := ts.stl_decompose(x, ts.STLParams{period = 4})
defer ts.decompose_free(&stl)
fmt.printf("stl seasonal: %.3f\n", stl.seasonal[:4])
fmt.printf("stl trend   : %.3f\n", stl.trend[:6])

cycle, trend := ts.hp_filter(x, 1600)
defer delete(cycle)
defer delete(trend)
fmt.printf("hp trend: %.3f\n", trend[:6])
```

```
seasonal: [0.948, -2.829, -1.399, 3.281]
trend   : [NaN, NaN, 13.237, 13.713, 14.150, 14.587]
stl seasonal: [0.766, -2.978, -1.477, 3.572]
stl trend   : [12.396, 12.831, 13.266, 13.699, 14.135, 14.576]
hp trend: [12.564, 12.900, 13.237, 13.572, 13.904, 14.232]
```

- **`seasonal_decompose(x, period, model)`** returns a `DecomposeResult` with `trend`, `seasonal` and `resid`. The trend is a centred moving average, so it is NaN for the first and last `period / 2` points. Use `.Multiplicative` for positive data whose seasonal swings grow with the level.
- **`stl_decompose(x, params)`** needs only `period`; every other field of `STLParams` has the statsmodels default when left at 0. Set `robust = true` for data with outliers; the robustness weights are then in `weights`.
- **`hp_filter(x, lambda)`** returns `(cycle, trend)`. Use λ = 1600 for quarterly data, 129600 for monthly data and 6.25 for annual data.

**Holt–Winters.** `holt_winters_fit` estimates the smoothing parameters and initial states by least squares, as statsmodels' `ExponentialSmoothing(...).fit()`. The spec chooses the model:

```odin
spec := ts.HoltWintersSpec {
    trend    = .Additive,
    seasonal = .Additive,
    period   = 4,
}
hw := ts.holt_winters_fit(SALES, spec)
defer ts.holt_winters_free(&hw)
p := hw.params
fmt.printf("alpha %.4f  beta %.4f  gamma %.4f  sse %.4f  aic %.3f\n", p.alpha, p.beta, p.gamma, hw.sse, hw.aic)
fc := ts.holt_winters_forecast(&hw, 4)
defer delete(fc)
fmt.printf("forecast: %.3f\n", fc)
```

```
alpha 0.6314  beta 0.0000  gamma 0.0000  sse 4.7913  aic -22.670
forecast: [20.850, 17.383, 19.183, 24.133]
```

`trend` is `.None` or `.Additive` (with `damped = true` for a damped trend), `seasonal` is `.None`, `.Additive` or `.Multiplicative`. Here β and γ are 0: the slope and the seasonal pattern are estimated once from the start of the data and not updated. The result also holds the one-step-ahead `fitted` values, the `resid`, the `level`, `trend` and `season` states, and `aic`, `aicc` and `bic`. To run the recursions with parameters you choose yourself, use `holt_winters_filter(y, spec, params)`.

## 9. ARIMA

`arima_fit(y, p, d, q)` estimates an ARIMA(p, d, q) by exact Gaussian maximum likelihood. The conventions are statsmodels' `ARIMA` with `trend="n"`: there is **no constant term**, and MA coefficients enter with a plus sign, z_t = Σ φ_i z_{t−i} + e_t + Σ θ_j e_{t−j}. Demean the series first if its mean is not zero, or use ARIMAX with a column of ones (section 10).

```odin
y := Y
fit := ts.arima_fit(y, 1, 0, 1) // ARIMA(p = 1, d = 0, q = 1)
defer {
    delete(fit.phi)
    delete(fit.theta)
}
fmt.printf("phi %.4f  theta %.4f  sigma2 %.4f\n", fit.phi[0], fit.theta[0], fit.sigma2)
fmt.printf("loglik %.4f  aic %.4f  bic %.4f  converged %v\n", fit.loglik, fit.aic, fit.bic, fit.converged)

fc := ts.arima_forecast(y, fit, 1, 0, 1, h = 4, alpha = 0.05)
defer {
    delete(fc.mean)
    delete(fc.lower)
    delete(fc.upper)
}
for i in 0 ..< len(fc.mean) {
    fmt.printf("h=%d  %7.4f  [%7.4f, %7.4f]\n", i + 1, fc.mean[i], fc.lower[i], fc.upper[i])
}

resid := ts.sarima_residuals(y, fit.phi, fit.theta, {}, {}, 0, 0, 0, fit.sigma2)
defer delete(resid)
Q, dof, p := ts.ljung_box(resid, 10, 2) // dof_adj = p + q
fmt.printf("residuals: n %d  Ljung-Box Q %.3f df %d p %.3f\n", len(resid), Q, dof, p)

ll := ts.arima_loglik(y, {0.85}, {0.5}, 0, 1.0) // at fixed parameters
fmt.printf("loglik at (0.85, 0.5, 1): %.4f\n", ll)
```

```
phi 0.8893  theta 0.5065  sigma2 0.9303
loglik -111.9259  aic 229.8518  bic 236.9979  converged true
h=1   3.8523  [ 1.9619,  5.7427]
h=2   3.4257  [ 0.1799,  6.6715]
h=3   3.0463  [-0.9588,  7.0513]
h=4   2.7089  [-1.8070,  7.2248]
residuals: n 80  Ljung-Box Q 2.895 df 8 p 0.941
loglik at (0.85, 0.5, 1): -112.3322
```

The estimates are close to the true φ = 0.85 and θ = 0.5, and the residuals show no autocorrelation left.

- **The fit** returns an `ArimaFitResult` with `phi`, `theta`, `sigma2`, `loglik`, `aic`, `bic` and `converged`. AIC and BIC count σ² as a parameter and use the number of observations after differencing, as statsmodels does. There is no free procedure: delete `phi` and `theta`.
- **The forecast** gives exactly `h` values on the original scale of `y`, with (1 − `alpha`) prediction intervals, as statsmodels' `get_forecast(h).conf_int(alpha)`. Pass the same `p, d, q` you fitted.
- **Residuals.** `sarima_residuals` with empty seasonal parts returns the one-step-ahead innovations of the (differenced) series, one per differenced observation. `df_residuals(y, fit, p, d, q)` returns them, with their variances, as a DataFrame.
- **`arima_loglik(y, phi, theta, d, sigma2)`** evaluates the exact log-likelihood at parameters you choose.

**Differencing.** `difference(y, d)` returns the d-th difference, which is `d` values shorter. `difference_with_history` also returns the first `d` values, which `inverse_difference` needs to undo it. `seasonal_difference(y, D, s)` takes lag-`s` differences:

```odin
x := []f64{10, 12, 15, 19, 24}
d1, hist := ts.difference_with_history(x, 1)
defer delete(d1)
defer delete(hist)
fmt.println(d1, hist)
back := ts.inverse_difference(d1, hist, 1)
defer delete(back)
fmt.println(back)

d2 := ts.difference(x, 2)
defer delete(d2)
fmt.println(d2)

q := []f64{1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6}
sd := ts.seasonal_difference(q, 1, 4)
defer delete(sd)
fmt.println(sd)
```

```
[2, 3, 4, 5] [10]
[10, 12, 15, 19, 24]
[1, 1, 1]
[1, 1, 1, 1, 1, 1, 1, 1]
```

**Automatic order selection.** `arima_auto_with_tests(y, max_p, max_q, criterion)` first chooses `d` with `auto_arima_d_from_tests` (section 7), then fits every (p, q) up to the limits and keeps the best by `"aic"` or `"bic"`. Comparing information criteria is only valid between models of the same differenced series, which is why `d` is fixed first. `arima_auto(y, max_p, max_d, max_q, criterion)` does the same with `d` capped at `max_d`.

The result is an `ArimaAutoResult` with the chosen order and its fit; free the fit's coefficients as usual:

```odin
y := Y
best := ts.arima_auto_with_tests(y, max_p = 2, max_q = 2, criterion = "aic")
defer {
    delete(best.fit.phi)
    delete(best.fit.theta)
}
fmt.printf("ARIMA(%d,%d,%d)  aic %.4f  theta %.4f\n", best.p, best.d, best.q, best.fit.aic, best.fit.theta)
```

```
ARIMA(0,1,1)  aic 227.2629  theta [0.4855]
```

The tests were inconclusive for this series (section 7), so `d = 1` was chosen and the search picked an IMA(1,1). Its AIC is not comparable with the AIC of the ARMA(1,1) fitted above, which models the undifferenced series. As always with automatic selection, check the result against what you know about the data.

## 10. SARIMA and ARIMAX

**SARIMA.** `sarima_fit(y, p, d, q, P, D, Q, s)` adds seasonal AR and MA terms of season length `s` and seasonal differencing of order `D`. The example simulates 120 quarters of the "airline" model with θ = −0.4 and Θ = −0.5 (minus signs because MA terms enter with a plus sign) and σ² = 0.25, and fits it back:

```odin
// quarterly "airline" data: (1-L)(1-L^4) y = (1 - 0.4L)(1 - 0.5L^4) e
rand.reset(7)
n := 120
e := make([]f64, n)
defer delete(e)
y := make([]f64, n)
defer delete(y)
for t in 0 ..< n {
    e[t] = rand.float64_normal(0, 0.5)
    z := e[t]
    if t >= 1 {z -= 0.4 * e[t - 1]}
    if t >= 4 {z -= 0.5 * e[t - 4]}
    if t >= 5 {z += 0.2 * e[t - 5]}
    y[t] = 100 + z
    if t >= 1 {y[t] += y[t - 1] - 100}
    if t >= 4 {y[t] += y[t - 4] - 100}
    if t >= 5 {y[t] -= y[t - 5] - 100}
}

// SARIMA(0,1,1)(0,1,1)_4, the "airline" model
fit := ts.sarima_fit(y, 0, 1, 1, 0, 1, 1, 4)
defer {
    delete(fit.phi)
    delete(fit.theta)
    delete(fit.Phi)
    delete(fit.Theta)
}
fmt.printf("theta %.4f  Theta %.4f  sigma2 %.4f  aic %.3f\n", fit.theta[0], fit.Theta[0], fit.sigma2, fit.aic)

fc := ts.sarima_forecast(y, fit.phi, fit.theta, fit.Phi, fit.Theta, 1, 1, 4, h = 4, sigma2 = fit.sigma2)
defer {
    delete(fc.mean)
    delete(fc.lower)
    delete(fc.upper)
}
for i in 0 ..< 4 {fmt.printf("h=%d  %.3f  [%.3f, %.3f]\n", i + 1, fc.mean[i], fc.lower[i], fc.upper[i])}

diag := ts.sarima_residual_diagnostics(y, fit.phi, fit.theta, fit.Phi, fit.Theta, 1, 1, 4, fit.sigma2, 8)
defer {
    delete(diag.acf_vals)
    delete(diag.pacf_vals)
}
fmt.printf("Ljung-Box p %.3f  Jarque-Bera p %.3f\n", diag.p_lb, diag.p_jb)
```

```
theta -0.5104  Theta -0.4240  sigma2 0.2251  aic 162.022
h=1  137.102  [136.172, 138.032]
h=2  139.359  [138.324, 140.394]
h=3  139.513  [138.382, 140.645]
h=4  139.156  [137.937, 140.375]
Ljung-Box p 0.275  Jarque-Bera p 0.311
```

`sarima_forecast` takes the polynomials and orders separately rather than the fit result; its default `alpha` is 0.05. `sarima_residuals` returns the innovations, `sarima_residual_diagnostics` adds their ACF, PACF, Ljung–Box and Jarque–Bera (in a `SarimaDiagResult`, whose `acf_vals` and `pacf_vals` you delete), and `sarima_auto_with_tests` searches over (p, q, P, Q) like `arima_auto_with_tests`.

**ARIMAX.** `arimax_fit(y, X, p, d, q)` fits a regression with ARIMA errors, y_t = x_tᵀβ + u_t with u ~ ARIMA(p, d, q), as statsmodels' `SARIMAX(y, exog, order)`. `X` is an n × k `l.Matrix(f64)`. Add a column of ones for a constant:

```odin
rand.reset(11)
n := 100
y := make([]f64, n)
defer delete(y)
X := l.matrix_new(f64, n, 2) // columns: constant, x
defer l.matrix_free(&X)
u := 0.0
for t in 0 ..< n {
    x := math.sin(f64(t) / 5)
    u = 0.6 * u + rand.float64_normal(0, 0.3)
    X.data[t * 2 + 0] = 1
    X.data[t * 2 + 1] = x
    y[t] = 2 + 1.5 * x + u
}
fit := ts.arimax_fit(y, &X, 1, 0, 0)
defer ts.arimax_free(&fit)
fmt.printf("beta %.4f  phi %.4f  sigma2 %.4f\n", fit.beta, fit.phi[0], fit.sigma2)

Xf := l.matrix_new(f64, 3, 2) // regressors for the next 3 periods
defer l.matrix_free(&Xf)
for i in 0 ..< 3 {
    Xf.data[i * 2 + 0] = 1
    Xf.data[i * 2 + 1] = math.sin(f64(n + i) / 5)
}
fc := ts.arimax_forecast(y, &X, fit, 0, &Xf)
defer {
    delete(fc.mean)
    delete(fc.lower)
    delete(fc.upper)
}
fmt.printf("%.4f\n", fc.mean)
```

```
beta [1.9491, 1.4761]  phi 0.5072  sigma2 0.0971
[3.2406, 3.3611, 3.4104]
```

The true values are β = (2, 1.5), φ = 0.6 and σ² = 0.09. `arimax_forecast` needs the future regressors, one row per step. With `d > 0` a constant column vanishes after differencing and is rejected; use a time trend column for a drift instead.

## 11. GARCH and volatility

`garch_fit_spec(residuals, spec)` fits a GARCH-type model to a series of zero-mean residuals by conditional maximum likelihood. `GARCH_Spec` combines a variance equation and an error distribution:

| Field | Values |
|---|---|
| `variance` | `.GARCH`, `.GJR` (leverage term γ), `.EGARCH` |
| `dist` | `.Normal`, `.StudentT` (degrees of freedom ν), `.GED` (shape) |
| `p`, `q` | ARCH and GARCH orders |

**Always set `q`.** `q = 0` means no GARCH term, so `GARCH_Spec{variance = .GARCH}` without `q = 1` fits an ARCH(1) model. A `p` of 0 is changed to 1.

The example simulates 1500 returns (in percent) from a GARCH(1,1) with ω = 0.05, α = 0.10 and β = 0.85, whose long-run variance is 0.05 / (1 − 0.95) = 1:

```odin
// GARCH(1,1) returns in percent: omega 0.05, alpha 0.10, beta 0.85
simulate_returns :: proc(n: int) -> []f64 {
    rand.reset(2024)
    r := make([]f64, n)
    s2, prev := 1.0, 0.0
    for t in 0 ..< n {
        s2 = 0.05 + 0.10 * prev * prev + 0.85 * s2
        prev = math.sqrt(s2) * rand.float64_normal(0, 1)
        r[t] = prev
    }
    return r
}
```

```odin
r := simulate_returns(1500)
defer delete(r)

lm := ts.arch_lm_test(r, 5)
fmt.printf("ARCH-LM(5) = %.2f, p = %.2g\n", lm.lm, lm.lm_pvalue)

resid := ts.extract_residuals(r) // r - mean(r)
defer delete(resid)

spec := ts.GARCH_Spec{variance = .GARCH, dist = .Normal, p = 1, q = 1}
g := ts.garch_fit_spec(resid, spec)
defer ts.garch_result_free(&g)
fmt.printf(
    "omega %.4f  alpha %.4f  beta %.4f  persistence %.4f\n",
    g.params.omega, g.params.alpha[0], g.params.beta[0], g.persistence,
)
fmt.printf("loglik %.2f  aic %.2f  bic %.2f  converged %v\n", g.log_likelihood, g.aic, g.bic, g.converged)
fmt.printf("long-run variance %.4f\n", g.params.omega / (1 - g.persistence))

for h in ([]int{1, 10, 100}) {
    f := ts.garch_forecast(&g, resid, h)
    fmt.printf("h=%3d  var %.4f  std %.4f  95%% band [%.3f, %.3f]\n", h, f.variance_forecast, f.std_forecast, f.confidence_95[0], f.confidence_95[1])
}

v95, v99 := ts.garch_var_series(&g, resid)
defer delete(v95)
defer delete(v99)
n := len(resid)
fmt.printf("last VaR 95%% %.4f  99%% %.4f\n", v95[n - 1], v99[n - 1])

// GJR with Student-t errors
gjr := ts.garch_fit_spec(resid, ts.GARCH_Spec{variance = .GJR, dist = .StudentT, p = 1, q = 1})
defer ts.garch_result_free(&gjr)
fmt.printf("GJR-t: gamma %.4f  nu %.2f  aic %.2f\n", gjr.params.gamma, gjr.params.nu, gjr.aic)
fmt.printf("t quantile 0.99: %.4f (normal 2.3263)\n", ts.garch_std_quantile(&gjr.params, 0.99))
```

```
ARCH-LM(5) = 81.39, p = 4.3e-16
omega 0.0414  alpha 0.0757  beta 0.8819  persistence 0.9576
loglik -2064.66  aic 4135.32  bic 4151.26  converged true
long-run variance 0.9755
h=  1  var 0.7032  std 0.8386  95% band [-1.644, 1.644]
h= 10  var 0.7911  std 0.8895  95% band [-1.743, 1.743]
h=100  var 0.9718  std 0.9858  95% band [-1.932, 1.932]
last VaR 95% 1.3597  99% 1.9230
GJR-t: gamma 0.0160  nu 94.81  aic 4138.54
t quantile 0.99: 2.3412 (normal 2.3263)
```

- **Test first.** A small ARCH-LM p-value (section 15) says the variance is not constant and a GARCH model is worth fitting.
- **The result** is a `GARCH_Result`: `params` (with `omega`, `alpha`, `beta`, `gamma`, `egarch_gamma`, `nu`, `shape`), `log_likelihood`, `aic`, `bic`, the in-sample `conditional_var` and `standardized_resid`, and `persistence` (α + β, plus γ/2 for GJR). Free it with `garch_result_free`.
- **Forecasts.** `garch_forecast(&g, resid, h)` returns the conditional variance `h` steps ahead. As `h` grows it reverts to the long-run variance ω / (1 − persistence); the 1-step forecast is low here because the last observations were calm. `confidence_95` is the 95 % interval of the return at that horizon under the fitted error distribution.
- **Value at risk.** `garch_var_series` returns the 95 % and 99 % VaR for every observation (as positive numbers, assuming a zero mean). `garch_std_quantile` gives the quantile of the unit-variance error distribution, which is wider than the normal for Student-t errors.
- **Model choice.** The data are normal and symmetric, so the GJR-t model finds almost no leverage (γ ≈ 0), a large ν, and a worse AIC than the plain GARCH. The fitted ν has no upper bound: on normal data it can become very large.

The parameters are constrained so that the persistence stays below 0.999. `garch_fit(residuals, model_type, p, q)` is an older entry point that takes a single `GARCH_Model_Type` (`.GARCH`, `.GJR_GARCH`, `.StudentT`, `.GED`); it **prints its progress** to standard output, so prefer `garch_fit_spec`.

**Volatility helpers.** The `create_*_volatility` procedures return a `VolatilityModel` whose `conditional_var` holds one variance per row, NaN during the warm-up:

```odin
df := w.df_from(w.column_from_floats("r", r))
defer w.destroy_dataframe(&df)
rv := ts.create_rolling_volatility(&df, "r", 20)
defer delete(rv.conditional_var)
ev := ts.create_ewma_volatility(&df, "r", 0.06, 20)
defer delete(ev.conditional_var)
gv := ts.create_garch_volatility(r)
defer ts.destroy_garch_volatility(&gv)
fmt.printf("last variance: rolling %.4f  ewma %.4f  garch %.4f\n", rv.conditional_var[n - 1], ev.conditional_var[n - 1], gv.conditional_var[n - 1])
```

```
GARCH iter 0: loss=2396.5757, ω=0.000000
GARCH iter 100: loss=2084.9530, ω=0.000795
GARCH iter 200: loss=2066.7968, ω=0.022240
GARCH iter 300: loss=2064.6869, ω=0.039064
GARCH iter 400: loss=2064.6636, ω=0.040633
GARCH iter 500: loss=2064.6611, ω=0.041181
GARCH converged at iteration 559 (loss change: 0.00000098)
last variance: rolling 0.4080  ewma 0.6330  garch 0.6833
```

`create_garch_volatility` demeans the series and calls `garch_fit`, hence the progress lines. Free the rolling and EWMA models by deleting `conditional_var`, and the GARCH model with `destroy_garch_volatility`. The EWMA `alpha` is the weight on the newest observation (RiskMetrics' λ = 1 − alpha).

**Regime-switching GARCH.** `rs_garch_fit(returns)` fits a two-regime GARCH(1,1) with a Hamilton filter. It needs at least 50 observations and, because it starts from `garch_fit`, it also prints progress lines. The example alternates calm (σ = 0.5) and turbulent (σ = 2) blocks of 100 returns:

```odin
rand.reset(3)
n := 400
r := make([]f64, n)
defer delete(r)
for t in 0 ..< n {
    sd := 0.5 if (t / 100) % 2 == 0 else 2.0 // calm and turbulent blocks
    r[t] = rand.float64_normal(0, sd)
}
res := ts.rs_garch_fit(r)
defer {
    for row in res.filtered_probs {delete(row)}
    delete(res.filtered_probs)
    delete(res.most_likely_path)
}
p := res.params
fmt.printf("p11 %.3f  p22 %.3f  loglik %.2f\n", p.p_11, p.p_22, res.log_likelihood)
fmt.println("regime at t = 50, 150, 250, 350:", res.most_likely_path[50], res.most_likely_path[150], res.most_likely_path[250], res.most_likely_path[350])
```

```
GARCH iter 0: loss=684.3867, ω=0.000000
GARCH iter 100: loss=623.5683, ω=0.000760
GARCH converged at iteration 184 (loss change: 0.00002818)
p11 0.992  p22 0.990  loglik -596.18
regime at t = 50, 150, 250, 350: 0 1 0 1
```

The `RS_GARCH_Result` holds the regime parameters, `filtered_probs` (one `[2]`-slice per observation) and the Viterbi `most_likely_path` (0 or 1). It has no free procedure; delete the fields as shown.

## 12. VAR and Granger causality

`var_fit(Y, p)` fits a vector autoregression with an intercept, equation by equation by OLS, as statsmodels' `VAR(endog).fit(p)`. `Y` is an n × k matrix with one column per variable:

```odin
// two series: y1 depends on its own lag and on y2's lag; y2 is an AR(1)
rand.reset(5)
n := 200
Y := l.matrix_new(f64, n, 2) // n x k, one column per variable
defer l.matrix_free(&Y)
a, b := 0.0, 0.0
for t in 0 ..< n {
    a, b = 0.5 * a + 0.3 * b + rand.float64_normal(0, 1), 0.4 * b + rand.float64_normal(0, 1)
    Y.data[t * 2 + 0] = a
    Y.data[t * 2 + 1] = b
}

sel := ts.var_select_order(&Y, 4)
defer ts.var_lag_selection_free(&sel)
fmt.println("lag by aic/bic/hqic/fpe:", sel.aic_lag, sel.bic_lag, sel.hqic_lag, sel.fpe_lag)

res := ts.var_fit(&Y, 1)
defer ts.var_free(&res)
A := res.coefs[0] // A_1, k x k, row = equation
fmt.printf("intercept %.4f\n", res.intercept)
fmt.printf("A1 = [%.4f %.4f; %.4f %.4f]\n", A.data[0], A.data[1], A.data[2], A.data[3])
fmt.printf("aic %.4f  bic %.4f  nobs %d\n", res.aic, res.bic, res.nobs)
stable, modulus := ts.var_is_stable(&res)
fmt.printf("stable %v (max modulus %.4f)\n", stable, modulus)

point, lower, upper := ts.var_forecast_interval(&res, 3, alpha = 0.05)
defer {
    l.matrix_free(&point)
    l.matrix_free(&lower)
    l.matrix_free(&upper)
}
for h in 0 ..< 3 {
    fmt.printf("h=%d  y1 %.4f [%.4f, %.4f]  y2 %.4f\n", h + 1, point.data[h * 2], lower.data[h * 2], upper.data[h * 2], point.data[h * 2 + 1])
}

irf := ts.var_irf(&res, 3)
defer ts.var_irf_free(irf)
for h in 0 ..= 3 {
    // [i, m] = response of variable i to a unit shock in variable m
    fmt.printf("h=%d  y1<-y2 %.4f  y2<-y1 %.4f\n", h, irf[h].data[0 * 2 + 1], irf[h].data[1 * 2 + 0])
}

c1 := ts.var_test_causality(&res, caused = {0}, causing = {1})
c2 := ts.var_test_causality(&res, caused = {1}, causing = {0})
fmt.printf("y2 -> y1: F %.3f p %.3g\n", c1.statistic, c1.p_value)
fmt.printf("y1 -> y2: F %.3f p %.3g\n", c2.statistic, c2.p_value)
```

```
lag by aic/bic/hqic/fpe: 1 1 1 1
intercept [0.0724, 0.0836]
A1 = [0.4800 0.3603; -0.0208 0.3514]
aic -0.0800  bic 0.0193  nobs 199
stable true (max modulus 0.4197)
h=1  y1 0.2168 [-1.7532, 2.1868]  y2 -0.0391
h=2  y1 0.1624 [-2.1067, 2.4314]  y2 0.0654
h=3  y1 0.1739 [-2.1970, 2.5448]  y2 0.1032
h=0  y1<-y2 0.0000  y2<-y1 0.0000
h=1  y1<-y2 0.3603  y2<-y1 -0.0208
h=2  y1<-y2 0.2996  y2<-y1 -0.0173
h=3  y1<-y2 0.1856  y2<-y1 -0.0107
y2 -> y1: F 25.614 p 6.42e-07
y1 -> y2: F 0.145 p 0.704
```

All four criteria pick one lag, the estimated A₁ is close to the true [0.5 0.3; 0 0.4], y2 Granger-causes y1 and not the other way round.

- **`VARResult`** holds `intercept`, the lag matrices `coefs[j]` (A_{j+1}, k × k, entry [i, m] = effect of variable m on equation i), the statsmodels-layout `params` and `stderr`, the residual covariance `sigma_u`, the residuals, and `llf`, `aic`, `bic`, `hqic` and `fpe`. Free it with `var_free`.
- **`var_select_order(Y, max_lags)`** fits every lag order on the same sample, as statsmodels' `select_order`.
- **`var_forecast(&res, steps)`** returns the point forecasts (steps × k); `var_forecast_interval` adds the bounds.
- **`var_irf(&res, periods, orthogonalized)`** returns `periods + 1` matrices. With `orthogonalized = true` the shocks are orthogonalised with the Cholesky factor of Σ_u (statsmodels' `orth_irfs`). Free the slice with `var_irf_free`.
- **`var_test_causality(&res, caused, causing, kind)`** is a Wald test that the lags of `causing` have zero coefficients in the equations of `caused`. `kind` is `.F` (default) or `.Wald`.

`granger_causality_test(y, x, max_lag)` is the bivariate test of statsmodels' `grangercausalitytests`: does `x` help to predict `y`? It returns one `GrangerLagResult` per lag order, with the F, χ² and likelihood-ratio versions. Its lag-1 F statistic equals the VAR(1) test above; the p-values differ because the degrees of freedom differ:

```odin
y1 := make([]f64, n)
defer delete(y1)
y2 := make([]f64, n)
defer delete(y2)
for t in 0 ..< n {y1[t], y2[t] = Y.data[t * 2], Y.data[t * 2 + 1]}
g := ts.granger_causality_test(y1, y2, 2) // does y2 help predict y1?
defer delete(g)
for r in g {fmt.printf("lag %d  F %.3f  p %.3g\n", r.lag, r.f_stat, r.f_pvalue)}
```

```
lag 1  F 25.614  p 9.57e-07
lag 2  F 11.946  p 1.29e-05
```

## 13. Cointegration

Two non-stationary series are cointegrated when a linear combination of them is stationary. `engle_granger(y0, y1, trend)` is the two-step test of statsmodels' `coint`: regress `y0` on the series in `y1`, then run an ADF test on the residuals. `johansen_test(X, det_order, k_ar_diff)` is statsmodels' `coint_johansen`:

```odin
// x is a random walk; y = 2 x + stationary noise, so (y, x) are cointegrated
rand.reset(9)
n := 250
x := make([]f64, n)
defer delete(x)
y := make([]f64, n)
defer delete(y)
walk := 0.0
for t in 0 ..< n {
    walk += rand.float64_normal(0, 1)
    x[t] = walk
    y[t] = 1 + 2 * walk + rand.float64_normal(0, 1)
}

eg := ts.engle_granger(y, {x}, .Constant)
defer ts.engle_granger_free(&eg)
fmt.printf("EG stat %.4f  p %.3g  crit 5%% %.4f  beta %.4f\n", eg.statistic, eg.p_value, eg.crit[1], eg.beta)

X := l.matrix_new(f64, n, 2)
defer l.matrix_free(&X)
for t in 0 ..< n {X.data[t * 2], X.data[t * 2 + 1] = y[t], x[t]}
jo := ts.johansen_test(&X, det_order = 0, k_ar_diff = 1)
defer ts.johansen_free(&jo)
for r in 0 ..< 2 {
    fmt.printf("r<=%d  trace %.3f  95%% crit %.3f\n", r, jo.trace_stat[r], jo.trace_crit[r][1])
}
fmt.println("rank (trace, max-eig):", jo.rank_trace, jo.rank_max_eig)
v0, v1 := l.matrix_at(&jo.evec, 0, 0)^, l.matrix_at(&jo.evec, 1, 0)^
fmt.printf("cointegrating vector (normalised): [1, %.4f]\n", v1 / v0)
```

```
EG stat -17.2725  p 4.15e-29  crit 5% -3.3608  beta [2.0035, 1.0469]
r<=0  trace 98.081  95% crit 15.494
r<=1  trace 1.595  95% crit 3.841
rank (trace, max-eig): 1 1
cointegrating vector (normalised): [1, -2.0035]
```

- **Engle–Granger.** `y1` is a `[][]f64`, so you can pass several regressors. `beta` holds their coefficients followed by the deterministic terms (here the constant 1.0469), and `residuals` is the spread y0 − fitted. A statistic below the critical value (`crit` holds 1 %, 5 % and 10 %) rejects "no cointegration". `trend` is a `CointTrend`: `.None`, `.Constant`, `.ConstantTrend` or `.ConstantTrendSq`. Free the result with `engle_granger_free`.
- **Johansen.** `det_order` is −1 (no deterministic terms), 0 (constant) or 1 (linear trend); `k_ar_diff` is the number of lagged differences in the VECM. The trace and maximum-eigenvalue statistics come with 90 %, 95 % and 99 % critical values (`trace_crit[r]`, `max_eig_crit[r]`), and `rank_trace` and `rank_max_eig` give the rank chosen by the sequential tests at 5 %. Column i of `evec` is the cointegrating vector of eigenvalue i. Here the rank is 1 and the vector recovers y − 2x. Free the result with `johansen_free`.

`mackinnon_pvalue` and `mackinnon_critical_values` give the MacKinnon p-values and critical values on their own.

## 14. Kalman filters

The Kalman filters use Odin's fixed-size `matrix` types, so the state size `N` and the measurement size `M` are compile-time constants (at most 8, because an Odin matrix holds at most 64 elements). Matrix literals are written **row by row**. The filter state lives on the stack; only the smoothers allocate.

**Linear filter and RTS smoother.** `kalman_init` builds a `KalmanFilter(N, M)`; `kalman_predict` and `kalman_update` advance it in place. For smoothing, store the predicted state (after `kalman_predict`, before `kalman_update`) and the filtered state at every step, then call `rts_smooth`:

```odin
// state = (position, velocity); only the position is measured
x0 := [2]f64{0, 1}
P0 := matrix[2, 2]f64{
    1, 0,
    0, 1,
}
F := matrix[2, 2]f64{ // matrix literals are row-major
    1, 1,
    0, 1,
}
H := matrix[1, 2]f64{1, 0}
Q := matrix[2, 2]f64{
    0.01, 0,
    0, 0.01,
}
R := matrix[1, 1]f64{1}

kf := ts.kalman_init(x0, P0, F, H, Q, R)
z := []f64{1.2, 2.1, 2.9, 4.2, 5.1, 6.05}

xf := make([]ts.KalmanState(2), len(z)) // filtered
defer delete(xf)
xp := make([]ts.KalmanState(2), len(z)) // predicted
defer delete(xp)

for t in 0 ..< len(z) {
    ts.kalman_predict(&kf)
    xp[t] = {kf.x, kf.P} // store x_{t|t-1} for the smoother
    ts.kalman_update(&kf, [1]f64{z[t]})
    xf[t] = {kf.x, kf.P}
}

xs := ts.rts_smooth(F, xf, xp)
defer delete(xs)
for t in 0 ..< len(z) {
    fmt.printf("t=%d  z %.2f  filtered %.4f %.4f  smoothed %.4f %.4f\n", t, z[t], xf[t].x[0], xf[t].x[1], xs[t].x[0], xs[t].x[1])
}
```

```
t=0  z 1.20  filtered 1.1336 1.0664  smoothed 1.0759 1.0017
t=1  z 2.10  filtered 2.1331 1.0330  smoothed 2.0771 1.0015
t=2  z 2.90  filtered 2.9986 0.9655  smoothed 3.0778 1.0020
t=3  z 4.20  filtered 4.0989 1.0100  smoothed 4.0809 1.0015
t=4  z 5.10  filtered 5.1043 1.0087  smoothed 5.0823 1.0012
t=5  z 6.05  filtered 6.0832 1.0012  smoothed 6.0832 1.0012
```

The smoothed estimates use all six measurements, so the velocity is close to 1 at every step, while the filtered velocity moves with each new measurement. The last smoothed state equals the last filtered state.

**Nonlinear filters.** The extended (EKF) and unscented (UKF) Kalman filters take the state and covariance as values and return the new ones. You pass the transition `f` and the measurement function `h` as procedures; the EKF also needs their Jacobians. The example tracks a log price that follows a random walk from noisy observations of the price itself:

```odin
// log price follows a random walk; we observe the price exp(x) with noise
f :: proc(x: [1]f64) -> [1]f64 {return x}
F_jac :: proc(x: [1]f64) -> matrix[1, 1]f64 {return {1}}
h :: proc(x: [1]f64) -> [1]f64 {return {math.exp(x[0])}}
H_jac :: proc(x: [1]f64) -> matrix[1, 1]f64 {return {math.exp(x[0])}}
```

```odin
prices := []f64{101.0, 99.5, 102.3, 104.1, 103.0}
Q := matrix[1, 1]f64{0.0004}
R := matrix[1, 1]f64{1.0}

// EKF
x := [1]f64{math.ln(f64(100))}
P := matrix[1, 1]f64{0.01}
for p in prices {
    x, P = ts.ekf_predict(x, P, f, F_jac, Q)
    x, P = ts.ekf_update(x, P, [1]f64{p}, h, H_jac, R)
}
fmt.printf("EKF: price %.4f  var(log price) %.6f\n", math.exp(x[0]), P[0, 0])

// UKF: no Jacobians; the allocator holds each step's sigma points
params := ts.UKF_Params{alpha = 1e-3, beta = 2, kappa = 0}
x = {math.ln(f64(100))}
P = {0.01}
for p in prices {
    x, P = ts.ukf_predict(x, P, f, Q, params, context.allocator)
    x, P = ts.ukf_update(x, P, [1]f64{p}, h, R, params, context.allocator)
}
fmt.printf("UKF: price %.4f  var(log price) %.6f\n", math.exp(x[0]), P[0, 0])
```

```
EKF: price 103.1232  var(log price) 0.000078
UKF: price 103.0980  var(log price) 0.000078
```

`ukf_predict` and `ukf_update` use the allocator you pass for the sigma points and weights of the step and free them before returning, so there is nothing for you to free. Only `ukf_sigma_points`, when you call it yourself (for the unscented smoother), returns slices you delete.

| Procedure | Purpose |
|---|---|
| `kalman_init`, `kalman_predict`, `kalman_update` | Linear filter |
| `kalman_predict_control(&kf, B, u)`, `rts_smooth_control` | With a control input x' = F x + B u |
| `kalman_forward_tv_control`, `rts_smooth_tv_control` | Time-varying F, H, B, Q, R sequences |
| `rts_smooth(F, xf, xp)` | Rauch–Tung–Striebel smoother; returns a slice to delete |
| `ekf_predict`, `ekf_update` (+ `_control` variants), `ekf_rts_smooth` | Extended Kalman filter and smoother |
| `ukf_predict`, `ukf_update` (+ `_control` variants), `ukf_sigma_points`, `ukf_rts_smooth` | Unscented Kalman filter and smoother |
| `kalman_loglik_scalar`, `kalman_filter_residuals` | Scalar observations with flat row-major arrays and any state size (used by ARIMA) |

The control and smoother conventions (which `u[t]` or `F_seq[t]` belongs to which step) are spelled out in the doc comments in [`kalman.odin`](../wotan/analytics/kalman.odin) and checked in [`kalman_test.odin`](../tests/kalman_test.odin).

## 15. Regression diagnostics, bootstrap and optimisation

**Regression diagnostics.** These tests take the residuals of a regression and, where needed, the regressor matrix (an `l.Matrix(f64)` that includes a constant column). They match statsmodels' `durbin_watson`, `het_breuschpagan`, `het_white`, `acorr_breusch_godfrey` and `het_arch`, and return an `LMTestResult` with the LM statistic, its χ² p-value and the F version. The example fits a straight line to data whose error spread grows with x:

```odin
// y = 2 + 0.5 x + e, with an error spread that grows with x
rand.reset(3)
n := 100
X := l.matrix_new(f64, n, 2) // constant, x
defer l.matrix_free(&X)
y := make([]f64, n)
defer delete(y)
for t in 0 ..< n {
    x := f64(t + 1) / 10
    X.data[t * 2], X.data[t * 2 + 1] = 1, x
    y[t] = 2 + 0.5 * x + rand.float64_normal(0, 0.2 * x)
}
// simple OLS by hand: slope = cov(x, y) / var(x)
mx, my := 0.0, 0.0
for t in 0 ..< n {mx += X.data[t * 2 + 1]; my += y[t]}
mx /= f64(n); my /= f64(n)
sxy, sxx := 0.0, 0.0
for t in 0 ..< n {
    dx := X.data[t * 2 + 1] - mx
    sxy += dx * (y[t] - my)
    sxx += dx * dx
}
b := sxy / sxx
a := my - b * mx
resid := make([]f64, n)
defer delete(resid)
for t in 0 ..< n {resid[t] = y[t] - a - b * X.data[t * 2 + 1]}

fmt.printf("Durbin-Watson %.4f\n", ts.durbin_watson(resid))
bp := ts.breusch_pagan_test(resid, &X)
fmt.printf("Breusch-Pagan LM %.3f p %.3g\n", bp.lm, bp.lm_pvalue)
wh := ts.white_test(resid, &X)
fmt.printf("White LM %.3f p %.3g df %.0f\n", wh.lm, wh.lm_pvalue, wh.df)
bg := ts.breusch_godfrey_test(resid, &X, 2)
fmt.printf("Breusch-Godfrey LM %.3f p %.3g\n", bg.lm, bg.lm_pvalue)
al := ts.arch_lm_test(resid, 2)
fmt.printf("ARCH-LM %.3f p %.3g\n", al.lm, al.lm_pvalue)
```

```
Durbin-Watson 1.5557
Breusch-Pagan LM 26.946 p 2.09e-07
White LM 30.748 p 2.1e-07 df 2
Breusch-Godfrey LM 5.042 p 0.0804
ARCH-LM 12.065 p 0.0024
```

Breusch–Pagan and White clearly detect the heteroskedasticity. (In practice you would get the residuals from the OLS in `wotan/analytics/ML`.)

| Test | Null hypothesis | Procedure |
|---|---|---|
| Durbin–Watson | No first-order autocorrelation (statistic ≈ 2) | `durbin_watson(resid)` |
| Breusch–Godfrey | No autocorrelation up to lag q | `breusch_godfrey_test(resid, exog, nlags)` (`exog` may be `nil`) |
| Breusch–Pagan | Homoskedastic errors | `breusch_pagan_test(resid, exog_het, robust = true)` |
| White | Homoskedastic errors | `white_test(resid, exog)` |
| ARCH-LM | No ARCH effects | `arch_lm_test(resid, nlags)` |

**Bootstrap.** `bootstrap_ci(data, statistic, ...)` computes a bootstrap confidence interval for any statistic, as `scipy.stats.bootstrap`. The statistic is a `proc(x: []f64, user_data: rawptr) -> f64`. Resampling can be iid, or by moving or stationary blocks for dependent data:

```odin
mean_stat :: proc(x: []f64, _: rawptr) -> f64 {
    s := 0.0
    for v in x {s += v}
    return s / f64(len(x))
}
```

```odin
x := SALES
r := ts.bootstrap_ci(x, mean_stat, n_resamples = 2000, seed = 42) // BCa, 95%
fmt.printf("mean %.4f  95%% CI [%.4f, %.4f]  se %.4f\n", r.theta_hat, r.low, r.high, r.standard_error)
rp := ts.bootstrap_ci(x, mean_stat, n_resamples = 2000, method = .Percentile, seed = 42)
fmt.printf("percentile CI [%.4f, %.4f]\n", rp.low, rp.high)

// dependent data: moving blocks of 8 observations
rb := ts.bootstrap_ci(Y, mean_stat, n_resamples = 2000, method = .Percentile, resample = .MovingBlock, block_size = 8, seed = 42)
ri := ts.bootstrap_ci(Y, mean_stat, n_resamples = 2000, method = .Percentile, seed = 42)
fmt.printf("ARMA series mean: iid se %.4f  block se %.4f\n", ri.standard_error, rb.standard_error)
```

```
mean 16.2375  95% CI [15.0076, 17.4304]  se 0.6284
percentile CI [15.0208, 17.4501]
ARMA series mean: iid se 0.3386  block se 0.8568
```

The iid bootstrap ignores the strong autocorrelation of the ARMA series and understates the standard error of its mean; blocks keep the dependence. `method` is `.BCa` (default; iid only), `.Percentile` or `.Basic`; `resample` is `.IID`, `.MovingBlock` or `.Stationary`. A `seed` of 0 uses the context's random generator. `bootstrap_ci` allocates nothing you need to free.

**Nelder–Mead.** `nelder_mead(f, ctx, x0, max_iter, tol, allocator, xtol)` minimises a function of several variables without derivatives; the ARIMA and Holt–Winters fits use it. The objective receives a `rawptr` for your own data. It returns the best point (allocated in `allocator`), the minimum and whether it converged:

```odin
rosenbrock :: proc(x: []f64, ctx: rawptr) -> f64 {
    a := (cast(^f64)ctx)^ // ctx carries user data
    return (a - x[0]) * (a - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0])
}
```

```odin
a := 1.0
x0 := []f64{-1.2, 1.0}
best, fmin, ok := ts.nelder_mead(rosenbrock, &a, x0, max_iter = 5000, tol = 1e-12, allocator = context.allocator, xtol = 1e-8)
defer delete(best)
fmt.printf("minimum at [%.6f, %.6f], f = %.3g, converged %v\n", best[0], best[1], fmin, ok)
```

```
minimum at [1.000000, 1.000000], f = 6.27e-18, converged true
```

## 16. Memory management

The usual Odin rule holds: whoever creates something frees it. For this package that comes down to a few rules.

1. **Slices are freed with `delete`.** This covers `acf`, `difference`, `extract_residuals`, `holt_winters_forecast`, `garch_var_series`, `granger_causality_test`, `rts_smooth`, `nelder_mead`'s point, and so on.
2. **Result structs with a free procedure:**

   | Result | Free with |
   |---|---|
   | `GARCH_Result` | `garch_result_free` |
   | `VARResult`, `VARLagSelection`, IRF slice | `var_free`, `var_lag_selection_free`, `var_irf_free` |
   | `EngleGrangerResult`, `JohansenResult` | `engle_granger_free`, `johansen_free` |
   | `DecomposeResult` (classical, STL) | `decompose_free` |
   | `HoltWintersResult` | `holt_winters_free` |
   | `ArimaxFitResult` | `arimax_free` |
   | `PCAResult` | `destroy_pca_result` (for `[]PCAResult`: each element, then `delete`) |
   | `[][]f64` matrices | `destroy_matrix` |
   | `l.Matrix(f64)` (VAR forecasts, input matrices) | `l.matrix_free` |
   | GARCH `VolatilityModel` | `destroy_garch_volatility` |

3. **Result structs without a free procedure:** delete their slice fields. `ArimaFitResult` (`phi`, `theta`), `SarimaFitResult` (`phi`, `theta`, `Phi`, `Theta`), `ArimaForecastResult` and `SarimaForecastResult` (`mean`, `lower`, `upper`), `SarimaDiagResult` (`acf_vals`, `pacf_vals`), rolling and EWMA `VolatilityModel` (`conditional_var`), `RS_GARCH_Result` (each row of `filtered_probs`, then `filtered_probs` and `most_likely_path`).
4. **Columns and DataFrames follow the core rules.** A `Column` from `rolling_apply`, `ewm_*_column`, `df_acf` or `df_pacf` is either added to a DataFrame with `w.add_column` (which then owns it) or freed with `w.destroy_column`. DataFrames from `rolling_apply_many`, `rolling_cov_matrix` and the `df_*` procedures are freed with `w.destroy_dataframe`.
5. **The fixed-size Kalman filters allocate nothing you keep.** `KalmanFilter`, `KalmanState` and the EKF and UKF states are plain values; only the smoothers return a slice.
6. **The `allocator` parameters** say where results go. The procedures in this guide free their internal scratch before returning (checked with a tracking allocator in [`analytics_memory_test.odin`](../tests/analytics_memory_test.odin)), so you can use the default allocator everywhere; pass `context.temp_allocator` only if you want the results themselves in the temporary arena.

Put a `defer` right after each allocation, as the examples do, and run with `-debug` and a tracking allocator to catch anything you missed.

## 17. A complete example

This example simulates 750 daily closing prices, then runs a typical volatility workflow: compute log returns, check that prices have a unit root and returns do not, check the returns for autocorrelation and ARCH effects, fit a GARCH(1,1) with Student-t errors, forecast the volatility, and put the GARCH variance next to a 20-day rolling variance in a DataFrame. The data come from a GARCH(1,1) with ω = 0.04, α = 0.08, β = 0.88 and t(6) shocks.

```odin
package main

import ts "wotan/analytics"
import w  "wotan/core"
import "core:fmt"
import "core:math"
import "core:math/rand"

// unit-variance Student-t shock with 6 degrees of freedom
t6 :: proc() -> f64 {
    chi2 := 0.0
    for _ in 0 ..< 6 {z := rand.float64_normal(0, 1); chi2 += z * z}
    return rand.float64_normal(0, 1) / math.sqrt(chi2 / 6) * math.sqrt(4.0 / 6.0)
}

// 750 daily closes from a GARCH(1,1) with t(6) shocks, returns in percent
simulate_prices :: proc(n: int) -> []f64 {
    rand.reset(2)
    prices := make([]f64, n)
    p, s2, r := 100.0, 1.0, 0.0
    for t in 0 ..< n {
        s2 = 0.04 + 0.08 * r * r + 0.88 * s2
        r = 0.03 + math.sqrt(s2) * t6()
        p *= math.exp(r / 100)
        prices[t] = p
    }
    return prices
}

main :: proc() {
    prices := simulate_prices(750)
    defer delete(prices)

    // 1. log returns in percent
    n := len(prices) - 1
    rets := make([]f64, n)
    defer delete(rets)
    for t in 0 ..< n {rets[t] = 100 * math.ln(prices[t + 1] / prices[t])}

    // 2. prices have a unit root, returns do not
    d_px, _, p_px, _, _ := ts.stationarity_test(prices)
    d_rt, _, p_rt, _, _ := ts.stationarity_test(rets)
    fmt.printf("prices : %v (ADF p %.3f)\n", d_px, p_px)
    fmt.printf("returns: %v (ADF p %.3f)\n", d_rt, p_rt)

    // 3. the mean: no autocorrelation left to model?
    Q, _, p_lb := ts.ljung_box(rets, 10, 0)
    fmt.printf("Ljung-Box(10) on returns: Q %.2f p %.3f\n", Q, p_lb)

    // 4. the variance: ARCH effects, then a GARCH(1,1) with Student-t errors
    resid := ts.extract_residuals(rets)
    defer delete(resid)
    lm := ts.arch_lm_test(resid, 5)
    fmt.printf("ARCH-LM(5): %.2f p %.2g\n", lm.lm, lm.lm_pvalue)

    g := ts.garch_fit_spec(resid, ts.GARCH_Spec{variance = .GARCH, dist = .StudentT, p = 1, q = 1})
    defer ts.garch_result_free(&g)
    fmt.printf(
        "GARCH-t: omega %.4f alpha %.4f beta %.4f nu %.1f\n",
        g.params.omega, g.params.alpha[0], g.params.beta[0], g.params.nu,
    )
    f1 := ts.garch_forecast(&g, resid, 1)
    f10 := ts.garch_forecast(&g, resid, 10)
    fmt.printf("vol forecast: 1 day %.3f%%, 10 days ahead %.3f%%\n", f1.std_forecast, f10.std_forecast)

    // 5. GARCH variance next to a 20-day rolling variance
    df := w.df_from(
        w.column_from_floats("ret", rets),
        w.column_from_floats("garch_var", g.conditional_var), // copied
    )
    defer w.destroy_dataframe(&df)
    rw := ts.rolling_window(&df, "ret", 20, 20)
    roll := ts.rolling_apply(rw, w.Aggregator{name = "roll_var20", kind = .Var})
    w.add_column(&df, roll)

    tail := w.iloc_range(&df, df.rows - 5, df.rows)
    defer w.destroy_dataframe(&tail)
    w.dataframe_pretty_print(&tail)
}
```

```
prices : DifferenceStationary (ADF p 0.327)
returns: Stationary (ADF p 0.000)
Ljung-Box(10) on returns: Q 17.17 p 0.071
ARCH-LM(5): 55.95 p 8.3e-11
GARCH-t: omega 0.0353 alpha 0.0935 beta 0.8820 nu 5.3
vol forecast: 1 day 0.825%, 10 days ahead 0.913%
    ret                  garch_var           roll_var20
744 0.23926323467689062  0.5431696047357624  0.3009897462080833
745 0.28141253565294555  0.5195266406662717  0.2693558069343477
746 0.5788576584842149   0.500677795878805   0.2745650619575246
747 -1.142030911975355   0.5076488494060953  0.3290535841832463
748 -1.0804335020265206  0.6063751676618233  0.3169544665025402
```

The fit recovers the simulated parameters reasonably well (persistence 0.976 against 0.96, ν = 5.3 against 6). The 10-day-ahead volatility is higher than the 1-day one because the current variance is below its long-run level. Everything the program allocates is freed by its own `defer`.

## 18. Quick reference

| Task | Procedures |
|---|---|
| Data conversion | `l.vector_from_df`, `l.matrix_from_df`, `extract_numeric_matrix`, `extract_residuals` |
| Rolling windows | `rolling_window`, `rolling_apply`, `rolling_apply_many` with `w.Aggregator` (`make_median_agg`, `make_quantile_agg`, `make_var`, `make_cov`, `make_corr`) |
| Exponential weighting | `ewm_mean_column`, `ewm_var_column`, `ewm_std_column`, `ewm_cov_column`, `ewm_corr_column`; `make_ewm_mean`, `make_ewm_var`, `make_ewm_cov`, `make_ewm_corr` |
| Covariance and PCA | `rolling_cov_matrix`, `rolling_corr_matrix`, `rolling_var_matrix`, `ewm_cov_matrix`, `extract_cov_matrix_row`, `covariance_matrix`, `pca_dataframe`, `pca_from_cov`, `pca_transform`, `pca_inverse_transform`, `pca_explained_variance_ratio`, `rolling_pca`, `ewm_pca`, `ewm_pca_last` |
| Autocorrelation | `acf`, `pacf`, `df_acf`, `df_pacf`, `ljung_box`, `df_ljung_box` |
| Distribution | `sample_moments`, `jarque_bera`, `df_jarque_bera`, `chi2_sf` |
| Unit roots and stationarity | `adf_test`, `df_adf`, `kpss_test`, `df_kpss`, `phillips_perron_test`, `stationarity_test`, `df_stationarity` |
| Decomposition | `hp_filter`, `seasonal_decompose`, `stl_decompose` (`STLParams`) |
| Exponential smoothing | `holt_winters_fit`, `holt_winters_filter`, `holt_winters_forecast` (`HoltWintersSpec`) |
| Differencing | `difference`, `difference_with_history`, `inverse_difference`, `seasonal_difference`, `sarima_difference` |
| ARIMA | `arima_fit`, `arima_forecast`, `arima_loglik`, `arima_auto`, `arima_auto_with_tests`, `auto_arima_d_from_tests`, `simulate_arima_pdq` |
| SARIMA | `sarima_fit`, `sarima_forecast`, `sarima_residuals`, `sarima_residual_diagnostics`, `sarima_auto_with_tests` |
| ARIMAX | `arimax_fit`, `arimax_forecast`, `arimax_loglik` |
| Residual summaries | `df_residual_diagnostics`, `df_residual_acf`, `df_residual_pacf` |
| GARCH | `garch_fit_spec` (`GARCH_Spec`), `garch_forecast`, `garch_var_series`, `garch_std_quantile`, `garch_fit` (legacy) |
| Volatility models | `create_rolling_volatility`, `create_ewma_volatility`, `create_garch_volatility`, `rs_garch_fit` |
| VAR | `var_fit`, `var_select_order`, `var_forecast`, `var_forecast_interval`, `var_irf`, `var_is_stable`, `var_test_causality` |
| Granger causality | `granger_causality_test`, `var_test_causality` |
| Cointegration | `engle_granger`, `johansen_test`, `mackinnon_pvalue`, `mackinnon_critical_values` |
| Kalman | `kalman_init`, `kalman_predict`, `kalman_update`, `rts_smooth`, `ekf_predict`, `ekf_update`, `ekf_rts_smooth`, `ukf_predict`, `ukf_update`, `ukf_rts_smooth` (and the `_control` variants) |
| Regression diagnostics | `durbin_watson`, `breusch_godfrey_test`, `breusch_pagan_test`, `white_test`, `arch_lm_test` |
| Resampling and optimisation | `bootstrap_ci`, `bootstrap_interval`, `nelder_mead` |
| Free | `garch_result_free`, `var_free`, `var_lag_selection_free`, `var_irf_free`, `engle_granger_free`, `johansen_free`, `decompose_free`, `holt_winters_free`, `arimax_free`, `destroy_pca_result`, `destroy_matrix`, `destroy_garch_volatility`, `delete` (slices) |

For more examples, see the tests in [`tests/`](../tests/); [`test_catalog.md`](test_catalog.md) maps them to topics. The reference tests, in particular [`stationarity_reference_test.odin`](../tests/stationarity_reference_test.odin), [`arima_contracts_test.odin`](../tests/arima_contracts_test.odin), [`coint_var_reference_test.odin`](../tests/coint_var_reference_test.odin) and [`step5_reference_test.odin`](../tests/step5_reference_test.odin), compare the procedures with statsmodels, SciPy and pandas.
