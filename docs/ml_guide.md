# The `analytics/ML` package: a user guide

`wotan/analytics/ML` is the classical machine-learning and regression package of Wotan. It fits linear models with full statistical inference (OLS, robust and clustered standard errors, WLS, GLS, ridge, lasso, elastic net, Fama-MacBeth), classifiers and regressors (logistic regression, naive Bayes, k-nearest neighbours, support vector machines, decision trees, random forests, gradient boosting, a small neural network), and k-means clustering. Around the models it provides scalers and encoders, evaluation metrics, train/test splits, cross-validation, grid and random search, pipelines, and saving and loading of fitted models. A bridge turns a `core` DataFrame into model inputs.

The models follow scikit-learn and statsmodels closely, and most of them are tested against those libraries (see [`docs/test_catalog.md`](test_catalog.md)). Where an example below uses the same data as such a reference test, the guide says so, and the printed values agree with the library's.

This guide walks through the package with small examples. It assumes you know basic Odin and the `core` package ([`core_guide.md`](core_guide.md)). Statistics and time-series tools outside this package, such as stand-alone PCA, regression diagnostics or ARIMA, are covered in the companion guide to the parent package, [`analytics_guide.md`](analytics_guide.md). Every example was compiled and run against the current code, and the output shown is the real output. The examples also free everything they allocate: a debug build with a tracking allocator reports no leaks and no bad frees.

## Contents

1. [Setup](#1-setup)
2. [Data, matrices and labels](#2-data-matrices-and-labels)
3. [From a DataFrame to model inputs](#3-from-a-dataframe-to-model-inputs)
4. [Linear regression](#4-linear-regression)
5. [Regularised regression](#5-regularised-regression)
6. [Fama-MacBeth regressions](#6-fama-macbeth-regressions)
7. [Classification: logistic regression, naive Bayes, KNN](#7-classification-logistic-regression-naive-bayes-knn)
8. [Sample and class weights](#8-sample-and-class-weights)
9. [Support vector machines](#9-support-vector-machines)
10. [Trees and ensembles](#10-trees-and-ensembles)
11. [Neural networks (MLP)](#11-neural-networks-mlp)
12. [Clustering with k-means](#12-clustering-with-k-means)
13. [Preprocessing: scalers and encoders](#13-preprocessing-scalers-and-encoders)
14. [Metrics](#14-metrics)
15. [Splitting data and cross-validation](#15-splitting-data-and-cross-validation)
16. [Hyper-parameter search](#16-hyper-parameter-search)
17. [Pipelines](#17-pipelines)
18. [Saving and loading models](#18-saving-and-loading-models)
19. [Memory management](#19-memory-management)
20. [A complete example](#20-a-complete-example)
21. [Quick reference](#21-quick-reference)

## 1. Setup

Import the package under a short name. This guide uses `ml`, as the Wotan tests do. Models take their data as `linalg` matrices, so you need that package too, and `core` if your data starts in a DataFrame:

```odin
import ml "wotan/analytics/ML"
import l  "wotan/linalg"
import w  "wotan/core"        // only for DataFrames (section 3)
import "core:fmt"
import "core:math/rand"       // to seed the models that use random numbers
```

Build with `-debug` while you learn the library, and install a tracking allocator as described in section 1 of the [`core` guide](core_guide.md#1-setup). It reports every allocation you forgot to free when the program exits.

Some models draw random numbers from Odin's global generator: random forests, gradient boosting with `subsample < 1`, decision trees with `max_features` set, and the MLP. Call `rand.reset(seed)` before fitting them to get the same model on every run. Everything else in the package is deterministic, or has its own `random_state` field that does not touch the global generator (k-means, splits, cross-validation, early stopping).

## 2. Data, matrices and labels

Every model takes its features as an `l.Matrix(f64)`: `rows` observations by `cols` features, stored row-major in the slice `data`. Cell `(i, j)` is `X.data[i * X.cols + j]`.

```odin
X := l.matrix_from_flat([]f64{
    1.0, 0.5,
    2.0, -1.2,
    3.0, 0.3,
}, 3, 2) // 3 rows (observations), 2 columns (features), row-major
defer l.matrix_free(&X)

fmt.println(X.rows, X.cols)        // 3 2
fmt.println(X.data[2 * X.cols + 1]) // row 2, column 1: 0.3
l.matrix_at(&X, 0, 1)^ = 0.7        // matrix_at returns a pointer to the cell
fmt.println(X.data)                 // [1, 0.7, 2, -1.2, 3, 0.3]

Z := l.matrix_new(f64, 2, 3)        // zero-filled 2 x 3
defer l.matrix_free(&Z)
```

`matrix_from_flat` copies the slice, so the matrix owns its data. Free every matrix with `l.matrix_free`.

Targets are always a `[]f64` with one entry per row. What the labels must look like depends on the model:

| Model | Target `y` |
|---|---|
| Regressions (OLS, ridge, SVR, regression trees, ...) | any real numbers |
| Logistic regression, MLP binary classification | `0` or `1` (other values panic) |
| `svm_fit_linear`, `kernel_svm_fit` | `-1` or `+1` (other values panic) |
| Naive Bayes, KNN, classification trees, forests and boosting, one-vs-rest wrappers | any class labels, for example `0, 1, 2` |

Prediction procedures return a new `[]f64` (one value per row), or an `l.Matrix(f64)` of class probabilities with one column per class. The probability columns are in the order of the model's sorted `classes` field.

The examples in the rest of the guide use two small datasets. The regression data has 15 observations of two regressors. It is the data of [`ml_linear_reference_test.odin`](../tests/ml_linear_reference_test.odin), which checks the results against statsmodels and scikit-learn:

```odin
x1 := []f64{0.5, -1.2, 0.3, 1.8, -0.7, 2.1, -0.4, 0.9, -1.6, 1.1, 0.0, -0.9, 1.4, -0.2, 0.6}
x2 := []f64{1.0, 0.4, -0.8, 0.2, 1.5, -1.1, 0.7, -0.3, 0.9, -1.4, 0.5, -0.6, 1.2, 0.1, -0.9}
y  := []f64{2.9, -0.8, 1.4, 5.0, 0.2, 4.1, 0.9, 2.8, -1.7, 3.6, 1.2, -0.1, 3.9, 1.6, 2.5}
```

The classification data has 12 points with two features, a 0/1 label `YL`, a continuous target `YR`, and three query points. It is the data of [`ml_svm_cluster_reference_test.odin`](../tests/ml_svm_cluster_reference_test.odin), which checks SVMs, k-means, KNN and naive Bayes against scikit-learn:

```odin
XD := []f64 {
    0.2, 1.1,    1.5, -0.3,   -0.7, 0.4,    2.1, 1.8,
    -1.2, -1.5,  0.9, 0.6,    -0.4, 2.2,    1.7, -1.4,
    0.1, -0.8,   -1.8, 0.9,   2.4, 0.2,     -0.3, -0.2,
}
YL := []f64{1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0}
YR := []f64{0.5, 1.2, -0.3, 2.0, -1.1, 0.9, 0.4, 0.6, -0.5, -0.9, 1.6, -0.1}
QD := []f64{0.5, 0.5, -1.0, 1.0, 1.0, -1.0}

X := l.matrix_from_flat(XD, 12, 2) // the 12 points
defer l.matrix_free(&X)
T := l.matrix_from_flat(QD, 3, 2)  // the 3 query points
defer l.matrix_free(&T)
```

From section 7 on, `X` and `T` are these two matrices.

## 3. From a DataFrame to model inputs

`prepare_dataset` turns a DataFrame into an `MLDataset`: a feature matrix `X`, the target `y`, the feature names, and an encoder for every string column. The target is named; every other `Int`, `Float`, `Bool` or `String` column becomes a feature, in column order (other types are skipped with a warning).

```odin
train := w.df_from(
    w.column_from_floats("ret_1m",  []f64{0.02, -0.01, 0.03, 0.00, -0.02, 0.04}),
    w.column_from_ints("volume",    []int{120, 80, 150, 90, 70, 160}),
    w.column_from_strings("sector", []string{"Tech", "Energy", "Tech", "Banks", "Energy", "Tech"}),
    w.column_from_bools("up",       []bool{true, false, true, false, false, true}),
)
defer w.destroy_dataframe(&train)

ds, ok := ml.prepare_dataset(&train, "up") // "up" is the target, every other column a feature
defer ml.dataset_free(&ds)

fmt.println(ok, ds.feature_names) // true ["ret_1m", "volume", "sector"]
fmt.println(ds.y)                 // [1, 0, 1, 0, 0, 1]
for i in 0 ..< ds.X.rows {
    fmt.println(ds.X.data[i * ds.X.cols:(i + 1) * ds.X.cols])
}
fmt.println(ds.encoders[0].col_name, ds.encoders[0].classes)
```

```
true ["ret_1m", "volume", "sector"]
[1, 0, 1, 0, 0, 1]
[0.02, 120, 0]
[-0.01, 80, 1]
[0.03, 150, 0]
[0, 90, 2]
[-0.02, 70, 1]
[0.04, 160, 0]
sector ["Tech", "Energy", "Banks"]
```

`Bool` values become `0` and `1`. A string column is **label-encoded**: each distinct value gets an integer code in order of first appearance (`Tech` = 0, `Energy` = 1, `Banks` = 2). The codes are ordinal numbers, which suits tree models. For linear models, KNN or SVMs, one-hot encode the column instead (section 13). The target must be numeric or `Bool`: with a string target `prepare_dataset` prints an error and returns `ok = false`, so encode it first with `le_fit` (section 13).

Call `prepare_dataset` on the training data only. For test or live data, `transform_dataset` reuses the training encoders, so the same category gets the same code. Categories it has never seen get `-1`:

```odin
test := w.df_from(
    w.column_from_floats("ret_1m",  []f64{0.01, -0.03}),
    w.column_from_ints("volume",    []int{110, 60}),
    w.column_from_strings("sector", []string{"Banks", "Utilities"}), // "Utilities" was never seen
)
defer w.destroy_dataframe(&test)

X_test, y_test, ok2 := ml.transform_dataset(&test, &ds) // no target column: y_test is all 0
defer l.matrix_free(&X_test) // safe even when ok2 is false
defer delete(y_test)
if !ok2 do return
fmt.println(X_test.data) // [0.01, 110, 2, -0.03, 60, -1]
```

Pass the target name as a third argument (`ml.transform_dataset(&test, &ds, "up")`) if the frame has one. `transform_dataset` fails (prints an error and returns `ok = false`) when a feature column of the training data is missing from the frame, or when the named target column does not exist or is not numeric. On failure it returns an empty matrix and a `nil` slice, so the frees above are harmless either way.

When you only need some numeric columns, `l.matrix_from_df` and `l.vector_from_df` read them directly:

```odin
df := w.df_from(
    w.column_from_floats("x1", []f64{0.5, -1.2, 0.3}),
    w.column_from_ints("x2",   []int{1, 0, 2}),
    w.column_from_floats("y",  []f64{2.9, -0.8, 1.4}),
)
defer w.destroy_dataframe(&df)

X := l.matrix_from_df(&df, []string{"x1", "x2"}) // Int or Float columns, in this order
defer l.matrix_free(&X)
yv := l.vector_from_df(&df, "y")
defer delete(yv)
fmt.println(X.data, yv) // [0.5, 1, -1.2, 0, 0.3, 2] [2.9, -0.8, 1.4]
```

**Nulls become 0.** Neither the bridge nor `matrix_from_df` (nor `ols_from_df`, section 4) skips or reports null cells; they enter the model as zeros:

```odin
x := w.column_new("x", .Float, 4)
w.append_float(&x, 1)
w.append_float(&x, 2)
w.append_null(&x)
w.append_float(&x, 4)
df := w.df_from(x, w.column_from_floats("y", []f64{2, 4, 6, 8}))
defer w.destroy_dataframe(&df)
d2, _ := ml.prepare_dataset(&df, "y")
defer ml.dataset_free(&d2)
fmt.println(d2.X.data) // [1, 2, 0, 4]: the null became 0
```

Drop or fill missing values in the DataFrame before you convert it, for example with a mask and `w.filter` (see the [`core` guide](core_guide.md#9-filtering-with-masks)).

## 4. Linear regression

`ols_fit` returns an `OLSResult` with the coefficients and the usual statsmodels inference. There is no automatic intercept: add a column of ones to `X` yourself.

```odin
n := len(y)
X := l.matrix_new(f64, n, 3) // columns: constant, x1, x2
defer l.matrix_free(&X)
for i in 0 ..< n {
    X.data[i * 3 + 0] = 1 // the intercept is an explicit column of ones
    X.data[i * 3 + 1] = x1[i]
    X.data[i * 3 + 2] = x2[i]
}

res := ml.ols_fit(&X, y)
defer ml._ols_result_free(&res, context.allocator)

names := []string{"const", "x1", "x2"}
fmt.printf("%-6s %8s %8s %8s %8s\n", "", "coef", "std err", "t", "P>|t|")
for name, j in names {
    fmt.printf("%-6s %8.4f %8.4f %8.3f %8.4f\n", name, res.beta[j], res.stderr[j], res.tvalues[j], res.pvalues[j])
}
fmt.printf("R² %.4f  adj. R² %.4f  F %.2f (p %.2g)  σ² %.4f  df_resid %d\n",
    res.r2, res.r2_adj, res.fstat, res.f_pvalue, res.sigma2, res.df_resid)
fmt.println("95% CI for x1:", res.ci_low[1], res.ci_high[1])
```

```
           coef  std err        t    P>|t|
const    1.3824   0.1152   12.001   0.0000
x1       1.7693   0.1124   15.744   0.0000
x2       0.1554   0.1370    1.135   0.2787
R² 0.9586  adj. R² 0.9517  F 138.88 (p 5e-09)  σ² 0.1809  df_resid 12
95% CI for x1: 1.5244238902876992 2.014114520498181
```

These are statsmodels' values for the same data (`ml_linear_reference_test` checks them to 10 digits). The fields of `OLSResult`:

| Field | Content |
|---|---|
| `beta`, `stderr`, `tvalues`, `pvalues` | One entry per column of `X`; p-values are two-sided from t(`df_resid`) |
| `ci_low`, `ci_high` | 95% confidence interval per coefficient |
| `vcov` | Coefficient covariance matrix (`l.Matrix`, k x k) |
| `fitted`, `residuals` | One entry per row |
| `r2`, `r2_adj`, `fstat`, `f_pvalue` | Fit statistics; centred when `X` has a constant column, uncentred otherwise (as statsmodels) |
| `sigma2`, `df_resid`, `rank` | Residual variance, `n - rank`, rank of `X` |
| `cov_type` | `.NonRobust`, or the robust type (see below) |

Free the result with `ml._ols_result_free(&res, allocator)`, passing the allocator the fit used (`context.allocator` unless you passed another). Despite the leading underscore it is the public way to free every `OLSResult` in the package, including those of WLS, GLS, ridge, lasso and elastic net.

`ols_fit(&X, y, .QR)` solves by QR decomposition instead of the default Cholesky; the result is the same to rounding. A rank-deficient `X` does not panic: the fit uses the pseudo-inverse and reports the lower `rank`.

To predict, multiply new rows (with the same columns, including the constant) by `beta`:

```odin
X_new := l.matrix_from_flat([]f64{1, 0.0, 0.0, 1, 1.0, -1.0}, 2, 3) // same columns as X
defer l.matrix_free(&X_new)
pred := l.matvec_dyn(&X_new, res.beta) // X_new · β
defer delete(pred)
fmt.println(pred) // [1.382409605664371, 2.996278914571205]
```

`ols_from_df` fits directly from DataFrame columns and adds the constant itself (`add_intercept = true` by default). Remember that nulls become zeros (section 3):

```odin
df := w.df_from(
    w.column_from_floats("x1", x1),
    w.column_from_floats("x2", x2),
    w.column_from_floats("y", y),
)
defer w.destroy_dataframe(&df)
r := ml.ols_from_df(&df, "y", []string{"x1", "x2"}) // adds the constant itself
defer ml._ols_result_free(&r, context.allocator)
fmt.println(r.beta) // [1.382409605664371, 1.7692692053929402, 0.1553998964861063]
```

### Robust standard errors

`ols_fit_robust` fits OLS and replaces the covariance, standard errors, p-values, confidence intervals and F statistic by a robust version. The coefficients do not change. The `RobustCovSpec` selects the estimator:

```odin
hc1 := ml.ols_fit_robust(&X, y, {type = .HC1})
defer ml._ols_result_free(&hc1, context.allocator)
hc3 := ml.ols_fit_robust(&X, y, {type = .HC3})
defer ml._ols_result_free(&hc3, context.allocator)
nw := ml.ols_fit_robust(&X, y, {type = .HAC, maxlags = 2, use_t = true})
defer ml._ols_result_free(&nw, context.allocator)
groups := []int{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5}
cl := ml.ols_fit_robust(&X, y, {type = .Cluster, groups = groups, use_t = true})
defer ml._ols_result_free(&cl, context.allocator)

fmt.printf("classical %.4f\n", res.stderr)
fmt.printf("HC1       %.4f\n", hc1.stderr)
fmt.printf("HC3       %.4f\n", hc3.stderr)
fmt.printf("HAC(2)    %.4f\n", nw.stderr)
fmt.printf("cluster   %.4f  p = %.4f\n", cl.stderr, cl.pvalues)
```

```
classical [0.1152, 0.1124, 0.1370]
HC1       [0.1066, 0.1170, 0.1260]
HC3       [0.1224, 0.1453, 0.1502]
HAC(2)    [0.0837, 0.1084, 0.1346]
cluster   [0.1065, 0.0871, 0.1774]  p = [0.0002, 0.0000, 0.4305]
```

| `RobustCovSpec` field | Meaning |
|---|---|
| `type` | `.HC0`, `.HC1`, `.HC2`, `.HC3` (White and its small-sample variants), `.HAC` (Newey-West), `.Cluster` |
| `maxlags` | HAC lags; a negative value picks statsmodels' default ⌊4 (n/100)^(2/9)⌋ |
| `hac_correction` | HAC: multiply by n/(n-k) |
| `groups` | Cluster: one integer group label per row (need not be sorted) |
| `cluster_no_correction` | Cluster: skip the G/(G-1)·(n-1)/(n-k) correction |
| `use_t` | t and F tests with finite degrees of freedom (n - k, or G - 1 for clusters) instead of normal and chi-squared |

As in statsmodels, robust inference uses the normal distribution unless `use_t` is set. The conventions match `OLS(...).fit(cov_type=...)` of statsmodels, and [`ols_robust_reference_test.odin`](../tests/ols_robust_reference_test.odin) checks every type against it. If you already have an OLS result, `ml.ols_set_robust_cov(&res, &X, {type = .HC1})` converts it in place (`res.cov_type` becomes `.HC1`).

### Weighted and generalised least squares

`wls_fit` takes one positive weight per row (proportional to the inverse error variance). `gls_fit` takes the full n x n error covariance matrix Ω. Both return an `OLSResult`; residuals and fitted values are on the original scale, as in statsmodels:

```odin
wts := []f64{1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.8, 1.2, 2.5, 0.6, 1.0, 1.8, 0.9, 1.3, 2.2}
wr := ml.wls_fit(&X, y, wts) // weights ∝ 1 / error variance
defer ml._ols_result_free(&wr, context.allocator)

Omega := l.matrix_new(f64, n, n) // AR(1)-type error covariance: 0.5^|i-j|
defer l.matrix_free(&Omega)
for i in 0 ..< n do for j in 0 ..< n do Omega.data[i * n + j] = math.pow(0.5, f64(abs(i - j)))
gr := ml.gls_fit(&X, y, &Omega)
defer ml._ols_result_free(&gr, context.allocator)

fmt.printf("WLS β %.6f  R² %.6f\n", wr.beta, wr.r2)
fmt.printf("GLS β %.6f  se %.6f\n", gr.beta, gr.stderr)
```

```
WLS β [1.314465, 1.728119, 0.197414]  R² 0.961858
GLS β [1.427033, 1.760005, 0.128271]  se [0.251409, 0.104919, 0.132150]
```

Both agree with statsmodels' `WLS` and `GLS`. For a covariance with Kronecker structure Ω = A ⊗ B, `gls_fit_kron(&X, y, &A, &B)` avoids building the full matrix.

The t and F distribution functions used for the p-values are public too:

```odin
fmt.println(ml.student_t_ppf(0.975, 12))   // 2.178812829667227
fmt.println(ml.student_t_sf2(2.0, 10))     // P(|T| > 2), T ~ t(10): 0.07338803477074027
fmt.println(ml.f_dist_sf(3.0, 2, 12))      // P(F > 3), F ~ F(2, 12): 0.08779149519890261
```

`student_t_cdf(t, df)` is the CDF.

## 5. Regularised regression

`ridge_fit`, `lasso_fit` and `elastic_net_fit` take the same `X` as OLS, including the column of ones. A constant column is recognised and **not penalised**, so with `X = [1, features]` they equal scikit-learn's `Ridge(alpha=λ)`, `Lasso(alpha=λ/n)` and `ElasticNet(alpha=λ/n, l1_ratio=ρ)` with `fit_intercept=True`. They return an `OLSResult`:

```odin
rr := ml.ridge_fit(&X, y, 2.0)                                // λ = 2
defer ml._ols_result_free(&rr, context.allocator)
la := ml.lasso_fit(&X, y, 3.0, max_iter = 100000, tol = 1e-13) // λ = 3
defer ml._ols_result_free(&la, context.allocator)
en := ml.elastic_net_fit(&X, y, 3.0, 0.5)                      // λ = 3, l1_ratio = 0.5
defer ml._ols_result_free(&en, context.allocator)

fmt.printf("ridge   β %.4f  se %.4f\n", rr.beta, rr.stderr)
fmt.printf("lasso   β %.4f  se %.4f\n", la.beta, la.stderr)
fmt.printf("enet    β %.4f  R² %.4f\n", en.beta, en.r2)
```

```
ridge   β [1.4470, 1.5500, 0.0432]  se [0.1287, 0.1095, 0.1268]
lasso   β [1.4530, 1.5420, 0.0000]  se [NaN, NaN, NaN]
enet    β [1.4639, 1.4979, 0.0000]  R² 0.9383
```

The ridge and lasso coefficients are scikit-learn's (checked in `ml_linear_reference_test`). Ridge reports sandwich standard errors. The lasso and elastic net have no valid closed-form covariance, so their standard errors, t, p and confidence intervals are NaN; the coefficients, fitted values, residuals and R² are filled in. The lasso and elastic net use coordinate descent: `max_iter` (default 1000) and `tol` (default 1e-4, on the largest coefficient change) control it.

`ridge_cv` and `lasso_cv` choose λ by k-fold cross-validation (lowest mean squared error) and refit on all the data. They return the best λ, its fit, and the CV error of every candidate:

```odin
lambdas := []f64{0.01, 0.1, 1, 10, 100}
best, fit, cv_mse := ml.ridge_cv(&X, y, lambdas, k = 5) // 5 folds, seed 42
defer ml._ols_result_free(&fit, context.allocator)
defer delete(cv_mse)
fmt.printf("best λ %v  β %.4f\n", best, fit.beta)
fmt.printf("CV MSE per λ %.4f\n", cv_mse)
```

```
best λ 0.1  β [1.3862, 1.7565, 0.1480]
CV MSE per λ [0.2499, 0.2478, 0.2630, 1.1003, 3.2734]
```

### Principal component regression

`pcr_fit` centres the features, keeps the leading principal components and regresses `y` on them. Give it the features only, without the column of ones; the intercept is the mean of `y`. Choose the number of components with `n_components`, or set `n_components = 0` and a `variance_threshold`:

```odin
XF := l.matrix_new(f64, n, 2) // features only: PCR centres the data itself
defer l.matrix_free(&XF)
for i in 0 ..< n {XF.data[i * 2], XF.data[i * 2 + 1] = x1[i], x2[i]}

pcr := ml.pcr_fit(&XF, y, {n_components = 1})
defer ml.pcr_free(&pcr)
fmt.printf("component %.4f  β %.4f  intercept %.4f\n", pcr.components.data, pcr.betas, pcr.bias)

Xq := l.matrix_from_flat([]f64{0, 0, 1, -1}, 2, 2)
defer l.matrix_free(&Xq)
pred := ml.pcr_predict(&pcr, &Xq)
defer delete(pred)
fmt.printf("%.4f\n", pred)

all := ml.pcr_fit(&XF, y, {variance_threshold = 0.95}) // keep 95% of the variance
defer ml.pcr_free(&all)
fmt.println(all.n_components) // 2
```

```
component [0.8524, -0.5228]  β [1.4269]  intercept 1.8333
[1.6029, 3.5653]
2
```

With all components kept, PCR gives the OLS predictions. For a stand-alone PCA (loadings, explained variance, transforms) see the `analytics` package ([`analytics_guide.md`](analytics_guide.md)).

## 6. Fama-MacBeth regressions

`fama_macbeth` runs one cross-sectional OLS per period and averages the coefficients over time. Pass one design matrix and one return vector per period; include a column of ones for the intercept. The standard error is the standard deviation of the period coefficients over √T, or with `nw_lags > 0` a Newey-West standard error:

```odin
// 4 months, 5 stocks: each month's returns and the stocks' market betas
betas := [4][5]f64 {
    {0.8, 1.0, 1.2, 0.9, 1.5},
    {0.7, 1.1, 1.3, 0.9, 1.4},
    {0.8, 1.0, 1.1, 1.0, 1.6},
    {0.9, 1.2, 1.2, 0.8, 1.5},
}
rets := [4][5]f64 {
    {0.010, 0.014, 0.019, 0.012, 0.025},
    {-0.004, 0.001, 0.003, -0.002, 0.006},
    {0.020, 0.024, 0.026, 0.025, 0.036},
    {0.006, 0.011, 0.010, 0.004, 0.015},
}
Xs: [4]l.Matrix(f64)
ys: [4][]f64
for t in 0 ..< 4 {
    Xs[t] = l.matrix_new(f64, 5, 2) // columns: constant, beta
    for i in 0 ..< 5 {Xs[t].data[i * 2], Xs[t].data[i * 2 + 1] = 1, betas[t][i]}
    ys[t] = rets[t][:]
}
defer for &m in Xs do l.matrix_free(&m)

fm := ml.fama_macbeth(Xs[:], ys[:])
defer ml.fama_macbeth_result_free(&fm)
fmt.printf("λ %.5f  se %.5f  t %.2f  periods %d\n", fm.coef, fm.std_err, fm.t_stat, fm.n_periods)

nw := ml.fama_macbeth(Xs[:], ys[:], nw_lags = 1) // Newey-West standard errors
defer ml.fama_macbeth_result_free(&nw)
fmt.printf("NW se %.5f\n", nw.std_err)
```

```
λ [-0.00628, 0.01766]  se [0.00388, 0.00186]  t [-1.62, 9.49]  periods 4
NW se [0.00232, 0.00089]
```

The market risk premium here is about 1.8% per month with a t-statistic of 9.5. Periods with fewer rows than columns, or whose least squares fails, are skipped; `n_periods` counts the periods used.

## 7. Classification: logistic regression, naive Bayes, KNN

From here on, `X` is the 12 x 2 classification matrix and `T` the 3 query points (section 2).

### Logistic regression

`logistic_fit` takes 0/1 labels and a `LogisticParams`. Set every field: the zero value is not a usable default (`C = 0` panics, and the zero optimizer is plain SGD with a learning rate of 0).

```odin
params := ml.LogisticParams {
    C              = 1.0,    // inverse penalty strength, as in scikit-learn
    max_iter       = 1000,
    tol            = 1e-8,   // stop when the gradient norm is below tol
    learning_rate  = 1.0,
    fit_intercept  = true,
    optimizer_type = .LBFGS,
}
model := ml.logistic_fit(&X, YL, params) // labels must be 0 or 1
defer ml.logistic_free(&model)

proba := ml.logistic_predict_proba(&model, &T) // P(y = 1)
defer delete(proba)
labels := ml.logistic_predict(&model, &T)      // 1 where P >= 0.5
defer delete(labels)

fmt.printf("w %.4f  b %.4f  converged %v\n", model.weights, model.bias, model.converged)
fmt.printf("%.4f %v\n", proba, labels)
```

```
w [1.5151, 0.1967]  b -0.5795  converged true
[0.5687, 0.1303, 0.6768] [1, 0, 1]
```

The objective is scikit-learn's: an L2 penalty ½‖w‖² plus C times the log loss, with an unpenalised intercept. `.LBFGS` is the reliable choice; `.Adam`, `.RMSProp` and `.SGD` are gradient methods that need a suitable `learning_rate` and more iterations. Use `C = math.INF_F64` for no penalty. For more than two classes, use `ovr_logistic_fit` (section 9).

### Naive Bayes and k-nearest neighbours

`gnb_fit` is scikit-learn's `GaussianNB` (the optional third argument is `var_smoothing`, default 1e-9). `knn_fit` stores a copy of the training data; `k` is the number of neighbours and `.Distance` weights them by 1/distance (`.Uniform` is a plain vote). Both accept any class labels and have a `predict_proba`:

```odin
nb := ml.gnb_fit(&X, YL)
defer ml.gnb_free(&nb)
nb_pred := ml.gnb_predict(&nb, &T)
defer delete(nb_pred)
nb_proba := ml.gnb_predict_proba(&nb, &T) // n x n_classes, columns in nb.classes order
defer l.matrix_free(&nb_proba)

knn := ml.knn_fit(&X, YL, {k = 3, weights = .Distance})
defer ml.knn_free(&knn)
knn_pred := ml.knn_predict(&knn, &T)
defer delete(knn_pred)
knn_proba := ml.knn_predict_proba(&knn, &T)
defer l.matrix_free(&knn_proba)

fmt.println("classes", nb.classes, knn.classes)
fmt.println("NB ", nb_pred, "KNN", knn_pred)
for i in 0 ..< T.rows {
    fmt.printf("P(row %d) NB %.4f  KNN %.4f\n", i,
        nb_proba.data[i * 2:(i + 1) * 2], knn_proba.data[i * 2:(i + 1) * 2])
}
```

```
classes [0, 1] [0, 1]
NB  [1, 0, 1] KNN [1, 0, 1]
P(row 0) NB [0.2590, 0.7410]  KNN [0.1937, 0.8063]
P(row 1) NB [0.9959, 0.0041]  KNN [0.7668, 0.2332]
P(row 2) NB [0.0399, 0.9601]  KNN [0.3110, 0.6890]
```

Both predictions match scikit-learn's (`ml_svm_cluster_reference_test`); the probabilities are checked in [`ml_step6_reference_test.odin`](../tests/ml_step6_reference_test.odin). KNN uses Euclidean distance, so scale the features first (section 13) when their units differ.

## 8. Sample and class weights

Most classifiers and the tree models accept weights, as scikit-learn does:

- **`sample_weight`** is an optional last argument of the fit procedure: one non-negative weight per row. A row with weight 3 counts like three copies of it. Pass it by name.
- **`class_weight`** is a field of the parameter struct. `.Balanced` gives class c the weight n / (n_classes · n_c), so that every class carries the same total weight; `.None` (the default) gives every class 1.

The effective weight of a row is the product of the two.

```odin
params := ml.LogisticParams{C = 1.0, max_iter = 1000, tol = 1e-8, learning_rate = 1.0, fit_intercept = true, optimizer_type = .LBFGS}
y_rare := []f64{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0} // 3 positives, 9 negatives

plain := ml.logistic_fit(&X, y_rare, params)
defer ml.logistic_free(&plain)

params.class_weight = .Balanced // each class gets the same total weight
balanced := ml.logistic_fit(&X, y_rare, params)
defer ml.logistic_free(&balanced)

params.class_weight = .None
sw := []f64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1} // row 10 counts three times
weighted := ml.logistic_fit(&X, y_rare, params, sample_weight = sw)
defer ml.logistic_free(&weighted)

p1 := ml.logistic_predict_proba(&plain, &T)
defer delete(p1)
p2 := ml.logistic_predict_proba(&balanced, &T)
defer delete(p2)
p3 := ml.logistic_predict_proba(&weighted, &T)
defer delete(p3)
fmt.printf("plain    %.4f\nbalanced %.4f\nweighted %.4f\n", p1, p2, p3)
```

```
plain    [0.2351, 0.1029, 0.2801]
balanced [0.4656, 0.2278, 0.4968]
weighted [0.2751, 0.0874, 0.3551]
```

Balancing raises the probabilities of the rare class. Where weights are supported:

| Model | `sample_weight` | `class_weight` |
|---|---|---|
| `logistic_fit`, `svm_fit_linear`, `kernel_svm_fit` | yes | yes (in the params) |
| `dt_fit`, `rf_fit` | yes | yes, classification criteria only |
| `gb_fit`, `svr_fit` | yes | no |
| `pipeline_fit` | yes, passed to the model (logistic, SVMs, SVR, trees, forests, boosting; others panic) | through the model params |
| `metrics_accuracy`, `metrics_log_loss` | yes | |

For SVMs a weight scales the per-sample C. [`ml_weights_reference_test.odin`](../tests/ml_weights_reference_test.odin) checks the weighted fits against scikit-learn.

## 9. Support vector machines

The SVMs are solved exactly with SMO, like LIBSVM, so they match scikit-learn's `SVC` and `SVR`. The classifiers need **-1/+1 labels**, and their predict procedures return **decision values**, not labels: the predicted class is the sign.

```odin
ys := make([]f64, len(YL)) // SVMs need -1 / +1 labels
defer delete(ys)
for v, i in YL do ys[i] = 1 if v == 1 else -1

lin := ml.svm_fit_linear(&X, ys, {C = 1.0, fit_intercept = true})
defer ml.svm_free(&lin)
dec := ml.svm_predict_linear(&lin, &T) // decision values w·x + b
defer delete(dec)
fmt.printf("w %.4f  b %.4f  %d support vectors\n", lin.weights, lin.bias, len(lin.support_vectors))
fmt.printf("decision %.4f\n", dec)

rbf := ml.kernel_svm_fit(&X, ys, {C = 3.0, kernel_type = .RBF, gamma = 0.7})
defer ml.kernel_svm_free(&rbf)
d2 := ml.kernel_svm_predict(&rbf, &T)
defer delete(d2)
fmt.printf("RBF decision %.4f  (%d support vectors)\n", d2, len(rbf.support_vectors))

labels := make([]f64, len(d2)) // the class is the sign of the decision value
defer delete(labels)
for v, i in d2 do labels[i] = 1 if v >= 0 else -1
fmt.println(labels) // [1, -1, 1]
```

```
w [1.4000, 0.3000]  b -0.7100  4 support vectors
decision [0.1400, -1.8100, 0.3900]
RBF decision [0.8767, -0.9768, 0.3473]  (10 support vectors)
[1, -1, 1]
```

scikit-learn gives `[0.14, -1.81, 0.39]` and `[0.8765, -0.9767, 0.3475]`. The small difference in the RBF values comes from the stopping tolerance: `tol` (on the KKT conditions) defaults to scikit-learn's 1e-3, and with `tol = 1e-10` the results agree to six digits, as the reference test shows. `max_iter` limits the passes over the data (0 = no limit). `learning_rate` and `optimizer_type` exist in the parameter structs but the exact solver ignores them.

`KernelSVMParams.kernel_type` is `.Linear`, `.RBF` (exp(-γ‖x - z‖²)) or `.Polynomial` ((γ x·z + coef0)^degree). A `gamma` of 0 or less means scikit-learn's `"scale"`: 1 / (n_features · Var(X)).

`svr_fit` is ε-insensitive support vector regression with the same kernels; `epsilon` is the width of the tube in which errors cost nothing:

```odin
svr := ml.svr_fit(&X, YR, {C = 2.0, epsilon = 0.2, kernel_type = .RBF, gamma = 0.5})
defer ml.svr_free(&svr)
pred := ml.svr_predict(&svr, &T)
defer delete(pred)
fmt.printf("%.4f  (%d support vectors)\n", pred, len(svr.support_vectors))
```

```
[0.5111, -0.4206, 0.3360]  (8 support vectors)
```

scikit-learn's `SVR` predicts `[0.5112, -0.4207, 0.3361]` (again equal to six digits with a tighter `tol`).

### More than two classes

The one-vs-rest wrappers fit one binary model per class (that class against all others) and predict the class whose model scores highest. They accept any class labels and take the parameters of the underlying model:

```odin
y3 := []f64{2, 1, 0, 2, 0, 1, 0, 1, 0, 0, 2, 0} // three classes
lp := ml.LogisticParams{C = 1.0, max_iter = 500, tol = 1e-6, learning_rate = 1.0, fit_intercept = true, optimizer_type = .LBFGS}
ovr := ml.ovr_logistic_fit(&X, y3, lp) // one 0/1 model per class
defer ml.ovr_free(&ovr)
p1 := ml.ovr_predict(&ovr, &X)
defer delete(p1)

osvm := ml.ovr_kernel_svm_fit(&X, y3, {C = 3.0, kernel_type = .RBF, gamma = 0.7})
defer ml.ovr_free(&osvm)
p2 := ml.ovr_predict(&osvm, &X)
defer delete(p2)

fmt.println(ovr.classes)
fmt.println("logistic", p1, ml.metrics_accuracy(y3, p1))
fmt.println("RBF SVM ", p2, ml.metrics_accuracy(y3, p2))
```

```
[0, 1, 2]
logistic [0, 1, 0, 2, 0, 2, 0, 1, 0, 0, 2, 0] 0.8333333333333334
RBF SVM  [2, 1, 0, 2, 0, 1, 0, 1, 0, 0, 2, 0] 1
```

`ovr_linear_svm_fit` works the same way. (These are in-sample accuracies, to show the API; section 15 evaluates models properly.) Naive Bayes, KNN, classification trees, forests and gradient boosting handle several classes natively and need no wrapper. The MLP classifies two classes only.

## 10. Trees and ensembles

### Decision trees

`dt_fit` grows a CART tree. The `criterion` in `TreeParams` decides the kind of tree: `.MSE` (the zero value) gives a regression tree whose leaves predict the mean, `.Gini` or `.Entropy` a classification tree whose leaves predict the majority class and store the class fractions:

```odin
tree := ml.dt_fit(&X, YL, {max_depth = 3, min_samples = 2, criterion = .Gini})
defer ml.dt_free(&tree)
pred := ml.dt_predict(&tree, &T)
defer delete(pred)
proba := ml.dt_predict_proba(&tree, &T)
defer l.matrix_free(&proba)

root := tree.nodes[tree.root_idx]
fmt.printf("%d nodes; root splits feature %d at %.2f\n", len(tree.nodes), root.feature_idx, root.threshold)
fmt.println(pred, proba.data)
```

```
3 nodes; root splits feature 0 at 0.15
[1, 0, 1] [0, 1, 1, 0, 0, 1]
```

One split on the first feature separates the two classes perfectly, so the tree stops there.

| `TreeParams` field | Meaning |
|---|---|
| `max_depth` | Nodes at this depth become leaves (0 = a single leaf) |
| `min_samples` | Minimum number of samples to split a node (scikit-learn `min_samples_split`) |
| `max_features` | Features tried per split (0 = all); uses the global RNG |
| `criterion` | `.MSE`, `.Gini`, `.Entropy` |
| `class_weight` | `.None` or `.Balanced` (classification only) |

The same tree on the continuous target `YR`, with its in-sample statistics:

```odin
tree := ml.dt_fit(&X, YR, {max_depth = 3, min_samples = 2}) // criterion .MSE (the default)
defer ml.dt_free(&tree)
pred := ml.dt_predict(&tree, &T)
defer delete(pred)
fmt.println(pred) // [0.5, -1.1, 0.9]

stats := ml.dt_compute_stats(&tree, &X, YR) // in-sample fit, structure, importances
defer {
    delete(stats.residuals)
    delete(stats.fitted)
    delete(stats.feature_importance)
    delete(stats.splits_per_feature)
}
fmt.printf("%d nodes, %d leaves, depth %d, R² %.4f\n", stats.n_nodes, stats.n_leaves, stats.max_depth, stats.r2)
fmt.printf("importance %.4f, splits per feature %v\n", stats.feature_importance, stats.splits_per_feature)
```

```
15 nodes, 8 leaves, depth 3, R² 0.9750
importance [0.9638, 0.0362], splits per feature [6, 1]
```

There is no free procedure for `TreeStats`: delete its four slices yourself, as above. `tree_stats_print(&stats)` prints a summary. If you only need the importances (scikit-learn's impurity-based `feature_importances_`), call `ml.tree_feature_importance(&tree, X.cols)`, which returns a `[]f64` that sums to 1.

### Random forests

`rf_fit` averages many trees grown on bootstrap samples with random feature subsets (`max_features = 0` uses √n_features). With a classification criterion it is scikit-learn's `RandomForestClassifier` (soft voting); with `.MSE` a `RandomForestRegressor`. Seed the global RNG first:

```odin
rand.reset(42) // bootstrap samples and feature draws use the global RNG
rf := ml.rf_fit(&X, YL, {
    n_trees     = 50,
    max_depth   = 3,
    min_samples = 2,
    bootstrap   = true,
    oob_score   = true,
    criterion   = .Gini, // .MSE for a regression forest
})
defer ml.rf_free(&rf)
pred := ml.rf_predict(&rf, &T)
defer delete(pred)
proba := ml.rf_predict_proba(&rf, &T)
defer l.matrix_free(&proba)
fmt.printf("OOB accuracy %.4f  predictions %v  P %.2f\n", rf.oob_score, pred, proba.data)
```

```
OOB accuracy 0.8333  predictions [1, 0, 1]  P [0.43, 0.57, 0.93, 0.07, 0.16, 0.84]
```

`oob_score` is the out-of-bag accuracy for a classification forest and the out-of-bag R² for a regression forest (NaN unless `oob_score` and `bootstrap` are set). `rf_compute_stats(&rf, &X, y)` returns a `TreeStats` with the forest's averaged feature importances; free it like the tree stats above.

### Gradient boosting

`gb_fit` is scikit-learn's `GradientBoostingRegressor` with `loss = .Squared_Error` (the zero value) and `GradientBoostingClassifier` with `loss = .Log_Loss`:

```odin
reg := ml.gb_fit(&X, YR, {n_estimators = 50, learning_rate = 0.1, max_depth = 2, min_samples = 2})
defer ml.gb_free(&reg)
pr := ml.gb_predict(&reg, &T)
defer delete(pr)
fmt.printf("regression %.4f (F0 = %.4f)\n", pr, reg.initial_pred)

clf := ml.gb_fit(&X, YL, {n_estimators = 50, learning_rate = 0.1, max_depth = 2, min_samples = 2, loss = .Log_Loss})
defer ml.gb_free(&clf)
pc := ml.gb_predict(&clf, &T)
defer delete(pc)
proba := ml.gb_predict_proba(&clf, &T)
defer l.matrix_free(&proba)
fmt.printf("classification %v  P %.4f\n", pc, proba.data)
```

```
regression [0.2817, -0.8712, 0.4327] (F0 = 0.3583)
classification [1, 0, 1]  P [0.0033, 0.9967, 0.9967, 0.0033, 0.0033, 0.9967]
```

The regressor starts from the mean of `y` (`initial_pred`) and adds `learning_rate` times each tree. `gb_decision_function` returns the raw scores (log-odds) of a classifier. `subsample` below 1 fits each tree on a random part of the rows (stochastic boosting, global RNG); 0 and 1 both mean all rows.

**Early stopping.** With `n_iter_no_change > 0`, `gb_fit` holds out `validation_fraction` of the rows (stratified for classification) and stops when the validation loss has not improved by `tol` within the last `n_iter_no_change` stages:

```odin
rand.reset(7)
n := 200
Xb := l.matrix_new(f64, n, 2)
defer l.matrix_free(&Xb)
yb := make([]f64, n)
defer delete(yb)
for i in 0 ..< n {
    a, b := rand.float64_range(-2, 2), rand.float64_range(-2, 2)
    Xb.data[i * 2], Xb.data[i * 2 + 1] = a, b
    yb[i] = a * a + b + rand.float64_normal(0, 0.3)
}

gb := ml.gb_fit(&Xb, yb, {
    n_estimators        = 500,
    learning_rate       = 0.1,
    max_depth           = 3,
    min_samples         = 2,
    n_iter_no_change    = 5,   // stop after 5 stages without improvement
    validation_fraction = 0.2, // held out from the 200 rows
    random_state        = 1,   // seed of the hold-out split
})
defer ml.gb_free(&gb)
fmt.println("stages fitted:", gb.n_estimators) // fewer than 500
```

```
stages fitted: 80
```

The tests [`ml_tree_reference_test.odin`](../tests/ml_tree_reference_test.odin), [`ml_tree_classifier_test.odin`](../tests/ml_tree_classifier_test.odin) and `ml_step6_reference_test` compare trees, forests and boosting (including early stopping) with scikit-learn.

## 11. Neural networks (MLP)

`mlp_fit` trains a fully connected network for regression or binary classification. `hidden_layers` lists the layer widths. `max_iter` is the number of epochs; `batch_size` 0 trains on the full batch. `.Adam` uses Adam, any other `optimizer_type` plain SGD. Seed the global RNG, which sets the initial weights and the batch order:

```odin
rand.reset(1) // initial weights and mini-batch order use the global RNG
hidden := []int{8}
net := ml.mlp_fit(&X, YL, {
    hidden_layers     = hidden,
    activation        = .Tanh,
    output_activation = .Sigmoid,
    task              = .BinaryClassification,
    learning_rate     = 0.05,
    max_iter          = 500, // epochs
    batch_size        = 0,   // 0 = full batch
    optimizer_type    = .Adam,
})
defer ml.mlp_free(&net)
p := ml.mlp_predict(&net, &T) // the output layer: here P(y = 1)
defer delete(p)
fmt.printf("%d layers, %d epochs, P %.4f\n", net.n_layers, net.n_iter, p)
```

```
2 layers, 500 epochs, P [0.9999, 0.0000, 0.9999]
```

`mlp_predict` returns the output of the last layer: a probability for binary classification (threshold it at 0.5 for labels), the prediction for regression with a `.Linear` output. The activations are `.ReLU`, `.Sigmoid`, `.Tanh` and `.Linear`. The MLP does not scale its inputs; put it in a pipeline with a scaler (section 17) for real data.

With `early_stopping = true` the network holds out `validation_fraction` (default 0.1) of the rows, scores them after every epoch (R² or accuracy), stops after `n_iter_no_change` epochs without improvement and returns the weights of the best epoch:

```odin
// Xb, yb: the 200 rows of the early-stopping example in section 10
rand.reset(1)
net := ml.mlp_fit(&Xb, yb, {
    hidden_layers       = []int{16, 16},
    activation          = .ReLU,
    output_activation   = .Linear,
    task                = .Regression,
    learning_rate       = 0.01,
    max_iter            = 2000,
    batch_size          = 32,
    optimizer_type      = .Adam,
    early_stopping      = true, // hold out 10%, keep the best epoch
    n_iter_no_change    = 20,
    random_state        = 1,
})
defer ml.mlp_free(&net)
fmt.printf("stopped after %d epochs, best validation R² %.4f\n", net.n_iter, net.best_validation_score)
```

```
stopped after 116 epochs, best validation R² 0.9685
```

The network only reads `hidden_layers` during `mlp_fit`, so a slice literal is fine here. For larger networks, autograd, other layer types and optimisers, see the deep-learning packages of Wotan.

## 12. Clustering with k-means

`km_fit` runs Lloyd's algorithm. `.KMeansPlusPlus` initialisation with several starts (`n_init`) keeps the run with the lowest inertia. A non-zero `random_state` makes the result reproducible without touching the global RNG:

```odin
km := ml.km_fit(&X, {
    n_clusters   = 2,
    max_iter     = 300,
    tol          = 1e-4,
    init         = .KMeansPlusPlus,
    n_init       = 10, // keep the best of 10 starts
    random_state = 1,  // != 0: reproducible, independent of the global RNG
})
defer ml.km_free(&km)
fmt.printf("centres %.4f\n", km.centroids.data)
fmt.println("labels ", km.labels, "inertia", km.inertia)

new_labels := ml.km_predict(&km, &T) // nearest centre
defer delete(new_labels)
fmt.println("query  ", new_labels)
fmt.printf("silhouette %.4f\n", ml.metrics_silhouette(&X, km.labels))

stats := ml.km_compute_stats(&km, &X)
defer ml.km_stats_free(&stats, context.allocator)
ml.km_stats_print(&stats)
```

```
centres [-0.5857, 0.3000, 1.7200, 0.1800]
labels  [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] inertia 19.244571428571433
query   [0, 0, 1]
silhouette 0.3228
=== K-MEANS STATISTICS ===
Samples: 12
Features: 2
Clusters: 2
Iterations: 3
Converged: true
Inertia: 19.244571

Cluster Sizes:
  Cluster  0:    7 points, inertia=12.3886
  Cluster  1:    5 points, inertia=6.8560
=== END K-MEANS STATS ===
```

The centres and inertia equal scikit-learn's `KMeans` on this data. `centroids` is a `n_clusters x n_features` matrix and `labels` a `[]int`. To start from given centres (scikit-learn's `init=array`), pass them flat in `init_centers`, for example `init_centers = {0.2, 1.1, 1.5, -0.3}`. With `random_state = 0` the random initialisation uses the global RNG. `metrics_silhouette_samples` gives the silhouette of every point.

## 13. Preprocessing: scalers and encoders

### Scalers

There are three scalers with the same four procedures: `*_fit`, `*_transform`, `*_inverse_transform` and `*_free`. Fit them on the training data and apply the same fitted scaler to the test data:

```odin
A := l.matrix_from_flat([]f64{
    1, 100,
    2, 110,
    3, 90,
    10, 300, // an outlier row
}, 4, 2)
defer l.matrix_free(&A)

ss := ml.standard_scaler_fit(&A) // fit on training data only
defer ml.standard_scaler_free(&ss)
Z := ml.standard_scaler_transform(&ss, &A)
defer l.matrix_free(&Z)
back := ml.standard_scaler_inverse_transform(&ss, &Z)
defer l.matrix_free(&back)
fmt.printf("mean %v  std %.4f\n", ss.mean, ss.std)
fmt.printf("standard %.4f\n", Z.data)
fmt.println("inverse ", back.data)

mm := ml.minmax_scaler_fit(&A) // feature_range defaults to {0, 1}
defer ml.minmax_scaler_free(&mm)
M := ml.minmax_scaler_transform(&mm, &A)
defer l.matrix_free(&M)
fmt.printf("min-max  %.4f\n", M.data)

rs := ml.robust_scaler_fit(&A) // median and interquartile range
defer ml.robust_scaler_free(&rs)
R := ml.robust_scaler_transform(&rs, &A)
defer l.matrix_free(&R)
fmt.printf("robust   %.4f  (median %v, IQR %v)\n", R.data, rs.center, rs.scale)
```

```
mean [4, 150]  std [3.5355, 86.8907]
standard [-0.8485, -0.5754, -0.5657, -0.4603, -0.2828, -0.6905, 1.6971, 1.7263]
inverse  [1, 100, 2, 110, 3, 90, 10, 300]
min-max  [0.0000, 0.0476, 0.1111, 0.0952, 0.2222, 0.0000, 1.0000, 1.0000]
robust   [-0.5000, -0.0833, -0.1667, 0.0833, 0.1667, -0.2500, 2.5000, 3.2500]  (median [2.5, 105], IQR [3, 60])
```

| Scaler | Transform | Notes |
|---|---|---|
| `StandardScaler` | (x - mean) / std | Population std (ddof 0), as scikit-learn; a constant feature is only centred |
| `MinMaxScaler` | maps [min, max] to `feature_range` | `minmax_scaler_fit(&X, {-1, 1})` for another range |
| `RobustScaler` | (x - median) / IQR | Quantile range (25, 75) by default: `robust_scaler_fit(&X, {10, 90})`; NaNs are ignored when fitting. The outlier barely moves the median and IQR, unlike the mean and std |

Each transform returns a new matrix to free with `l.matrix_free`.

### Encoders

`ohe_fit` learns the sorted categories of a `[]string`; `ohe_transform` returns a flat `[]f64` with one row of indicators per input. `le_fit` and `le_transform` map categories to integer codes:

```odin
sectors := []string{"Tech", "Energy", "Tech", "Banks"}

ohe := ml.ohe_fit(sectors) // categories sorted: Banks, Energy, Tech
defer ml.ohe_free(&ohe)
flat := ml.ohe_transform(&ohe, []string{"Banks", "Tech", "Utilities"})
defer delete(flat)
fmt.println(ohe.categories, flat) // an unseen category is all zeros

D := l.matrix_from_flat(flat, 3, len(ohe.categories)) // as a 3 x 3 matrix
defer l.matrix_free(&D)

le := ml.le_fit(sectors)
defer ml.le_free(&le)
codes := ml.le_transform(&le, []string{"Tech", "Banks", "Utilities"})
defer delete(codes)
fmt.println(codes) // [2, 0, -1]
```

```
["Banks", "Energy", "Tech"] [1, 0, 0, 0, 0, 1, 0, 0, 0]
[2, 0, -1]
```

`ohe_fit(data, drop_first = true)` drops the first category's column, which avoids perfect collinearity with an intercept in OLS. Unseen categories give a row of zeros (one-hot) or `-1` (label encoder). To use one-hot columns from a DataFrame, read the string column with `w.column_at_string` into a `[]string`, encode it, and combine the result with the numeric features into one matrix.

## 14. Metrics

All metrics take `[]f64` slices of equal length. They follow scikit-learn's definitions and are tested against it in [`ml_metrics_reference_test.odin`](../tests/ml_metrics_reference_test.odin).

**Regression** (the values are those of the scikit-learn documentation examples):

```odin
y_true := []f64{3.0, -0.5, 2.0, 7.0}
y_pred := []f64{2.5, 0.0, 2.0, 8.0}
fmt.println("MSE ", ml.metrics_mse(y_true, y_pred), " RMSE", ml.metrics_rmse(y_true, y_pred), " MAE", ml.metrics_mae(y_true, y_pred))
fmt.println("R²  ", ml.metrics_r2(y_true, y_pred), " explained variance", ml.metrics_explained_variance(y_true, y_pred))
fmt.println("MAPE", ml.metrics_mape(y_true, y_pred))
```

```
MSE  0.375  RMSE 0.6123724356957945  MAE 0.5
R²   0.9486081370449679  explained variance 0.9571734475374732
MAPE 0.3273809523809524
```

**Classification from labels.** Precision, recall and F1 are for the positive class (`pos_label`, default 1):

```odin
y_true := []f64{0, 0, 1, 1, 1, 0, 1, 0}
y_pred := []f64{0, 1, 1, 1, 0, 0, 1, 0}
fmt.println(ml.metrics_accuracy(y_true, y_pred), ml.metrics_precision(y_true, y_pred),
    ml.metrics_recall(y_true, y_pred), ml.metrics_f1(y_true, y_pred))

cm := ml.metrics_confusion_matrix(y_true, y_pred, []f64{0, 1}) // rows = true, columns = predicted
defer delete(cm)
fmt.println(cm) // [3, 1, 1, 3]

rep := ml.metrics_classification_report(y_true, y_pred, []f64{0, 1})
defer ml.metrics_free_classification_report(&rep)
fmt.println(rep.precision, rep.recall, rep.f1, rep.macro_f1)
```

```
0.75 0.75 0.75 0.75
[3, 1, 1, 3]
[0.75, 0.75] [0.75, 0.75] [0.75, 0.75] 0.75
```

The confusion matrix is a flat `[]int` of `n_classes x n_classes`, row-major, in the order of the `classes` you pass. The report has per-class and macro-averaged precision, recall and F1, plus the accuracy.

**Classification from scores or probabilities:**

```odin
y_true := []f64{0, 0, 1, 1, 1, 0, 1, 0}
p_pos  := []f64{0.1, 0.6, 0.8, 0.7, 0.4, 0.2, 0.9, 0.3} // P(y = 1)

fmt.println("ROC AUC  ", ml.metrics_roc_auc(y_true, p_pos))
fmt.println("avg prec.", ml.metrics_average_precision(y_true, p_pos))
fmt.println("log loss ", ml.metrics_log_loss_binary(y_true, p_pos))
fmt.println("Brier    ", ml.metrics_brier_score(y_true, p_pos))

roc := ml.metrics_roc_curve(y_true, p_pos)
defer ml.metrics_roc_curve_free(&roc)
fmt.println(roc.fpr, roc.tpr, roc.thresholds)

P := l.matrix_new(f64, len(p_pos), 2) // the same as an n x 2 probability matrix
defer l.matrix_free(&P)
for p, i in p_pos {P.data[i * 2], P.data[i * 2 + 1] = 1 - p, p}
fmt.println(ml.metrics_log_loss(y_true, &P, []f64{0, 1}), ml.metrics_roc_auc_ovr(y_true, &P, []f64{0, 1}))
```

```
ROC AUC   0.9375
avg prec. 0.95
log loss  0.40036743569623084
Brier     0.125
[0, 0, 0, 0.25, 0.25, 1] [0, 0.25, 0.75, 0.75, 1, 1] [+Inf, 0.9, 0.7, 0.6, 0.4, 0.1]
0.40036743569623084 0.9375
```

The scores for ROC AUC and average precision can be probabilities or any decision value where higher means more positive, such as an SVM's decision values. `metrics_log_loss` and `metrics_roc_auc_ovr` (macro one-vs-rest AUC) take a probability matrix as the `*_predict_proba` procedures return it, with its `classes`. The ROC AUC of a single-class `y_true` is NaN. `metrics_roc_curve(..., drop_intermediate = false)` keeps every threshold.

## 15. Splitting data and cross-validation

### Train/test split

`train_test_split` returns copies of the rows in a `TrainTestSplit` (`X_train`, `X_test`, `y_train`, `y_test`). The test size is `ceil(test_size · n)` rows; the rows are shuffled with `random_state` unless `shuffle = false`:

```odin
split := ml.train_test_split(&X, YL, 0.25, random_state = 7) // 9 train, 3 test rows
defer ml.train_test_split_free(&split, context.allocator)
fmt.println(split.X_train.rows, split.X_test.rows, split.y_test)

ordered := ml.train_test_split(&X, YL, 0.25, shuffle = false) // the last 3 rows
defer ml.train_test_split_free(&ordered, context.allocator)
fmt.println(ordered.y_test)
```

```
9 3 [1, 0, 0]
[0, 1, 0]
```

Use `shuffle = false` for time series, so that the test rows come after the training rows. `subset_matrix(&X, indices)` and `subset_slice(y, indices)` copy arbitrary rows.

### Cross-validation splitters

A `CVSplitter` describes how to split, and `cv_make_folds` returns the train and test indices of every split as a `[]CVFold`. Free them with `cv_folds_free`. The kinds follow scikit-learn's `KFold`, `StratifiedKFold` and `TimeSeriesSplit` (checked in [`ml_cv_splitter_test.odin`](../tests/ml_cv_splitter_test.odin)):

```odin
folds := ml.cv_make_folds({kind = .KFold, n_splits = 3}, 10) // 10 samples
defer ml.cv_folds_free(folds)
for f in folds do fmt.println(f.train, f.test)
```

```
[4, 5, 6, 7, 8, 9] [0, 1, 2, 3]
[0, 1, 2, 3, 7, 8, 9] [4, 5, 6]
[0, 1, 2, 3, 4, 5, 6] [7, 8, 9]
```

Stratified folds keep the class proportions of `y` in every fold:

```odin
labels := []f64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1}
strat := ml.cv_make_folds({kind = .Stratified_KFold, n_splits = 2}, 10, labels)
defer ml.cv_folds_free(strat)
for f in strat do fmt.println(f.train, f.test)
```

```
[3, 4, 5, 8, 9] [0, 1, 2, 6, 7]
[0, 1, 2, 6, 7] [3, 4, 5, 8, 9]
```

Time-series splits train on the past and test on the next block. `gap` drops samples between the two (against look-ahead from overlapping labels), and `max_train_size` turns the expanding window into a rolling one:

```odin
walk := ml.cv_make_folds({kind = .Time_Series, n_splits = 3, gap = 1}, 10)
defer ml.cv_folds_free(walk)
for f in walk do fmt.println(f.train, f.test)

rolling := ml.cv_make_folds({kind = .Time_Series, n_splits = 3, test_size = 2, max_train_size = 3}, 10)
defer ml.cv_folds_free(rolling)
for f in rolling do fmt.println(f.train, f.test)
```

```
[0, 1, 2] [4, 5]
[0, 1, 2, 3, 4] [6, 7]
[0, 1, 2, 3, 4, 5, 6] [8, 9]
[1, 2, 3] [4, 5]
[3, 4, 5] [6, 7]
[5, 6, 7] [8, 9]
```

For financial labels that span several periods there are López de Prado's purged K-fold and combinatorial purged CV. Sample i's label is assumed to use observations i to i + `label_horizon` (or `label_end[i]`); training samples whose labels overlap a test block are purged, and `embargo` more samples after each test block are dropped:

```odin
purged := ml.cv_make_folds({kind = .Purged_KFold, n_splits = 3, label_horizon = 1, embargo = 1}, 10)
defer ml.cv_folds_free(purged)
for f in purged do fmt.println(f.train, f.test)
```

```
[6, 7, 8, 9] [0, 1, 2, 3]
[0, 1, 2, 9] [4, 5, 6]
[0, 1, 2, 3, 4, 5] [7, 8, 9]
```

`.Combinatorial_Purged` with `n_test_groups` returns one fold per combination of test groups; `cv_cpcv_paths` and `cv_cpcv_path` assemble their out-of-sample results into backtest paths. See [`purged_cv_reference_test.odin`](../tests/purged_cv_reference_test.odin).

### Cross-validated scores

`cross_val_score` (shuffled K-fold) and `cross_val_score_cv` (any splitter) call an **evaluator** on every split and return one score per split. An evaluator fits a model on the training indices and scores it on the test indices. The package has one per model, and each takes its parameter struct through a `rawptr`:

```odin
params := ml.KNNParams{k = 3}
scores := ml.cross_val_score(&X, YL, 4, ml.knn_cv_evaluator, &params) // shuffled 4-fold
defer delete(scores)
fmt.println(scores)

cv := ml.CVSplitter{kind = .Stratified_KFold, n_splits = 3, shuffle = true, random_state = 1}
s2 := ml.cross_val_score_cv(&X, YL, cv, ml.knn_cv_evaluator, &params)
defer delete(s2)
fmt.println(s2)
```

```
[1, 0.6666666666666666, 1, 1]
[1, 0.75, 1]
```

| Evaluator | Parameter type | Score |
|---|---|---|
| `logistic_cv_evaluator` | `LogisticParams` | accuracy |
| `knn_cv_evaluator` | `KNNParams` | accuracy |
| `linear_svm_cv_evaluator`, `kernel_svm_cv_evaluator` | `SVMParams`, `KernelSVMParams` | accuracy (one-vs-rest, so any labels) |
| `dt_cv_evaluator`, `rf_cv_evaluator`, `gb_cv_evaluator` | `TreeParams`, `RFParams`, `GBParams` | R² for regressors (`criterion = .MSE`, `loss = .Squared_Error`), accuracy for classifiers |
| `ridge_cv_evaluator`, `lasso_cv_evaluator` | `RidgeParams`, `LassoParams` | R²; fits an intercept (see below) |
| `elastic_net_cv_evaluator` | `ElasticNetParams` | R²; fits an intercept (see below) |
| `svr_cv_evaluator` | `SVRParams` | R² |

The linear evaluators fit an unpenalised intercept, as the pipelines do: they add a column of ones unless `X` already has a constant column, which then serves as the intercept. To score something else, write your own evaluator with the `ModelEvaluator` signature. This one scores Gaussian naive Bayes by its log loss (negated, because higher scores are better):

```odin
nb_log_loss_eval :: proc(
    X: ^l.Matrix(f64),
    y: []f64,
    train_idx, val_idx: []int,
    user_data: rawptr,
    allocator: mem.Allocator,
) -> f64 {
    X_tr := ml.subset_matrix(X, train_idx, allocator)
    defer l.matrix_free(&X_tr)
    y_tr := ml.subset_slice(y, train_idx, allocator)
    defer delete(y_tr, allocator)
    X_va := ml.subset_matrix(X, val_idx, allocator)
    defer l.matrix_free(&X_va)
    y_va := ml.subset_slice(y, val_idx, allocator)
    defer delete(y_va, allocator)

    model := ml.gnb_fit(&X_tr, y_tr, allocator = allocator)
    defer ml.gnb_free(&model)
    P := ml.gnb_predict_proba(&model, &X_va, allocator)
    defer l.matrix_free(&P)
    return -ml.metrics_log_loss(y_va, &P, model.classes)
}
```

```odin
s3 := ml.cross_val_score_cv(&X, YL, cv, nb_log_loss_eval, nil)
defer delete(s3)
fmt.printf("%.4f\n", s3)
```

```
[-0.2341, -0.5738, -0.1275]
```

(`mem` is `import "core:mem"`.) The evaluator must free what it allocates; `user_data` carries whatever your evaluator needs, or `nil`.

## 16. Hyper-parameter search

`grid_search_cv` tries every parameter set of a slice with a splitter and an evaluator, and returns the set with the highest mean score. `grid_search` does the same with a shuffled K-fold (seed 42), and there is a shortcut per model (`grid_search_knn`, `grid_search_logistic`, `grid_search_ridge`, ...):

```odin
grid := []ml.KNNParams{{k = 1}, {k = 3}, {k = 5}, {k = 3, weights = .Distance}}
best, score := ml.grid_search_knn(&X, YL, 4, grid) // shuffled 4-fold, mean accuracy
fmt.println(best, score)

cv := ml.CVSplitter{kind = .Stratified_KFold, n_splits = 3}
trees := []ml.TreeParams {
    {max_depth = 1, min_samples = 2, criterion = .Gini},
    {max_depth = 3, min_samples = 2, criterion = .Gini},
}
bt, st := ml.grid_search_cv(&X, YL, cv, ml.dt_cv_evaluator, trees)
fmt.println(bt.max_depth, st)
```

```
KNNParams{k = 1, weights = "Uniform"} 0.9166666666666666
1 0.8333333333333334
```

Ties go to the first parameter set. Use a `.Time_Series` or purged splitter for financial models, so that no test fold lies before its training data.

`random_search` and `random_search_cv` evaluate `n_iter` parameter sets drawn by a **sampler**, a procedure from a seed to a parameter struct. The search calls it with `seed`, `seed + 1`, ... so the draws are reproducible, and all draws use the same folds:

```odin
// random search draws parameter sets from a sampler: any deterministic function of the seed
svr_sampler :: proc(seed: u64) -> ml.SVRParams {
    u := f64((seed * 2654435761) % 1000) / 1000 // a number in [0, 1)
    return ml.SVRParams{C = math.pow(10, -1 + 3 * u), epsilon = 0.1, kernel_type = .RBF, gamma = 0.5}
}
```

```odin
best, score := ml.random_search_svr(&X, YR, 3, 8, svr_sampler, seed = 1) // 8 draws, 3 folds, R²
fmt.printf("C = %.4f, mean R² = %.4f\n", best.C, score)
```

```
C = 19.1867, mean R² = 0.3298
```

The searches return parameters, not a fitted model: fit the final model with the best parameters on the whole training set (as the example in section 20 does). An empty grid returns the zero parameters and a score of `-math.F64_MAX`.

In `LassoParams` and `ElasticNetParams`, a `max_iter` or `tol` of 0 means the defaults 1000 and 1e-4, in the searches and in pipelines alike.

## 17. Pipelines

A `Pipeline` chains scalers and one model. `pipeline_fit` fits each scaler on the output of the previous one and the model on the result; `pipeline_predict` sends new data through the fitted scalers and the model. This guarantees that the test data is scaled with the training statistics.

```odin
pipe := ml.pipeline_new()
defer ml.pipeline_free(&pipe)
ml.pipeline_add_standard_scaler(&pipe)
ml.pipeline_set_logistic(&pipe, {C = 1.0, max_iter = 1000, tol = 1e-8, learning_rate = 1.0, fit_intercept = true, optimizer_type = .LBFGS})

ml.pipeline_fit(&pipe, &X, YL)           // fits the scaler, then the model on scaled X
pred := ml.pipeline_predict(&pipe, &T)   // scales T with the fitted scaler, then predicts
defer delete(pred)
fmt.println(pred) // [1, 0, 1]

// the fitted parts are ordinary fields
fmt.printf("scaler mean %.4f, weights %.4f\n", pipe.steps[0].standard_scaler.mean, pipe.model.logistic.weights)
```

```
[1, 0, 1]
scaler mean [0.3750, 0.2500], weights [1.5711, 0.1706]
```

The steps are `pipeline_add_standard_scaler`, `pipeline_add_minmax_scaler` (range 0 to 1) and `pipeline_add_robust_scaler`. The models are set with `pipeline_set_ols`, `_ridge`, `_lasso`, `_elastic_net`, `_logistic`, `_linear_svm`, `_kernel_svm`, `_svr`, `_knn`, `_gnb`, `_decision_tree`, `_random_forest`, `_gradient_boosting`, `_mlp`, `_ovr_logistic`, `_ovr_linear_svm` and `_ovr_kernel_svm`, each with the model's parameter struct (`pipeline_set_ols` takes an `OLSMethod`, `pipeline_set_gnb` the smoothing `epsilon`; both have defaults). The pipeline stores the parameters until `pipeline_fit`, so slices inside them, such as the MLP's `hidden_layers`, must still be alive then.

For the linear regressions (OLS, ridge, lasso, elastic net) the pipeline adds the column of ones itself; `beta[0]` is the intercept:

```odin
pipe := ml.pipeline_new()
defer ml.pipeline_free(&pipe)
ml.pipeline_add_robust_scaler(&pipe)
ml.pipeline_set_ridge(&pipe, {lambda = 1.0}) // the pipeline adds the intercept column
ml.pipeline_fit(&pipe, &X, YR)
pred := ml.pipeline_predict(&pipe, &T)
defer delete(pred)
fmt.printf("β %.4f  predictions %.4f\n", pipe.model.ridge.beta, pred)
```

```
β [0.2495, 1.1059, 0.3867]  predictions [0.4969, -0.1817, 0.3481]
```

What `pipeline_predict` returns depends on the model: class labels for the classifiers (logistic regression gives 0/1, the binary SVMs give -1/+1 from the sign of the decision value), predictions for the regressions, and the raw output of the last layer for the MLP (a probability for binary classification). For probabilities of other models, apply the fitted parts yourself, as section 20 does.

`pipeline_fit` takes an optional `sample_weight`, which goes to the model (the scalers are unweighted). You can refit a pipeline; the previous fit is released:

```odin
pipe := ml.pipeline_new()
defer ml.pipeline_free(&pipe)
ml.pipeline_set_decision_tree(&pipe, {max_depth = 2, min_samples = 2, criterion = .Gini})
sw := []f64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}
ml.pipeline_fit(&pipe, &X, YL, sample_weight = sw) // weights go to the model only
pred := ml.pipeline_predict(&pipe, &T)
defer delete(pred)
fmt.println(pred) // [1, 0, 1]

ml.pipeline_fit(&pipe, &X, YL) // refitting releases the previous fit
```

## 18. Saving and loading models

Fitted models are saved through pipelines. A pipeline with no scaler is just a wrapped model. `pipeline_save` writes a binary file and `pipeline_load` reads it back; both return `false` on failure. Free a loaded pipeline with `pipeline_free`, **even when loading failed**, because a partly read file may already have allocated parts:

```odin
pipe := ml.pipeline_new()
defer ml.pipeline_free(&pipe)
ml.pipeline_add_minmax_scaler(&pipe)
ml.pipeline_set_random_forest(&pipe, {n_trees = 20, max_depth = 3, min_samples = 2, bootstrap = true, criterion = .Gini})
rand.reset(3)
ml.pipeline_fit(&pipe, &X, YL)

path := "forest.wotan"
saved := ml.pipeline_save(&pipe, path)
defer os.remove(path) // import "core:os"; only to clean up after the example

loaded, ok := ml.pipeline_load(path)
defer ml.pipeline_free(&loaded) // free it even when ok is false
before := ml.pipeline_predict(&pipe, &T)
defer delete(before)
after := ml.pipeline_predict(&loaded, &T)
defer delete(after)
fmt.println(saved, ok, loaded.model.type, len(loaded.model.rf.trees))
fmt.println(before, after)

missing, mok := ml.pipeline_load("no_such_file.wotan")
defer ml.pipeline_free(&missing)
fmt.println(mok) // false
```

```
true true RandomForest 20
[1, 0, 1] [1, 0, 1]
false
```

The file stores the fitted scalers, the fitted model, including the inference statistics of linear models (standard errors, p-values, R²), bit-exactly, and the model's hyper-parameters ([`ml_selection_pipeline_test.odin`](../tests/ml_selection_pipeline_test.odin) and [`step6_cleanup_test.odin`](../tests/step6_cleanup_test.odin) check this, and that truncated or corrupt files are rejected; [`ml_fixes_test.odin`](../tests/ml_fixes_test.odin) checks the stored parameters). A loaded pipeline can therefore be refitted directly, for example on new data:

```odin
// path: a saved logistic-regression pipeline (C = 1, L-BFGS)
loaded, _ := ml.pipeline_load(path)
defer ml.pipeline_free(&loaded)
fmt.println(loaded.model.logistic_params.C, loaded.model.logistic_params.optimizer_type)
ml.pipeline_fit(&loaded, &X, YL) // refit with the stored parameters
p := ml.pipeline_predict(&loaded, &T)
defer delete(p)
fmt.println(p)
```

```
1 LBFGS
[1, 0, 1]
```

Files written before the hyper-parameters were stored (format version 5 and older) still load and predict, but their parameters are zero: set the model again (`pipeline_set_logistic(&loaded, params)` and so on) before you refit such a pipeline.

For your own model files, `model_writer_open`, `write_f64`, `write_slice_f64`, `write_matrix`, `model_writer_close` and their `read_*` counterparts (`model_reader_open`, `read_slice_f64`, ...) are the building blocks the pipeline format uses; `write_decision_tree` and `read_decision_tree` save single trees.

## 19. Memory management

The package follows the usual Odin rule: whoever creates something frees it. Every fit and predict procedure takes an optional `allocator` (default `context.allocator`) and allocates its result with it.

1. **Every fitted model has a free procedure**, named after the model: `logistic_free`, `gnb_free`, `knn_free`, `svm_free`, `kernel_svm_free`, `svr_free`, `ovr_free`, `dt_free`, `rf_free`, `gb_free`, `mlp_free`, `km_free`, `pcr_free`, the scalers' `*_scaler_free`, `ohe_free`, `le_free`, `pipeline_free`, `dataset_free`, `fama_macbeth_result_free`.
2. **Every `OLSResult` is freed with `ml._ols_result_free(&res, allocator)`**, with the allocator the fit used. This covers OLS, robust OLS, WLS, GLS, ridge, lasso, elastic net, and the `best_result` of `ridge_cv` and `lasso_cv`.
3. **Predictions are plain slices or matrices.** Free a `[]f64` (or the `[]int` of `km_predict`) with `delete`, and a matrix (from `*_predict_proba`, the scaler transforms, `subset_matrix`) with `l.matrix_free`.
4. **Some results are structs of slices with their own free:** `train_test_split_free(&split, allocator)`, `cv_folds_free(folds)`, `metrics_roc_curve_free`, `metrics_free_classification_report`, `km_stats_free(&stats, allocator)`. `TreeStats` (from `dt_compute_stats` and `rf_compute_stats`) has none: delete its `residuals`, `fitted`, `feature_importance` and `splits_per_feature`. The `cv_errors` of `ridge_cv` and `lasso_cv`, the scores of `cross_val_score`, and `metrics_confusion_matrix` are slices to `delete`.
5. **Models own copies of what they keep.** KNN copies its training data, the SVMs copy their support vectors, and `prepare_dataset` copies the column names and categories. You can free your inputs (matrices, DataFrames) as soon as the fit returns.
6. **A pipeline owns its fitted parts.** `pipeline_free` frees the scalers and the model, so do not free `pipe.model.logistic` or a scaler inside a pipeline yourself. Free a loaded pipeline even if `pipeline_load` failed.

The usual pattern is a `defer` right after each allocation, as in the examples. Run with `-debug` and a tracking allocator, and any leak is reported with the line that allocated it.

## 20. A complete example

This example builds a table of stock-days with two numeric features and a sector, converts it with the DataFrame bridge, holds out a test set, chooses the penalty of a logistic regression by stratified cross-validation, fits a scaler-plus-model pipeline, evaluates it on the test set, and saves and reloads it.

```odin
package main

import w  "wotan/core"
import ml "wotan/analytics/ML"
import l  "wotan/linalg"
import "core:fmt"
import "core:math/rand"
import "core:os"

N :: 200

// A synthetic table of stock-days: did the stock go up the next day?
make_data :: proc() -> w.DataFrame {
	rand.reset(42)
	mom    := w.column_new("momentum", .Float, N)
	vol    := w.column_new("volatility", .Float, N)
	sector := w.column_new("sector", .String, N)
	up     := w.column_new("up", .Bool, N)
	names  := []string{"Tech", "Energy", "Banks"}
	for _ in 0 ..< N {
		m := rand.float64_normal(0, 1)
		v := rand.float64_range(0.1, 0.5)
		s := names[rand.int_max(3)]
		z := 1.5 * m - 4 * (v - 0.3) + (0.5 if s == "Tech" else 0) + rand.float64_normal(0, 1)
		w.append_float(&mom, m)
		w.append_float(&vol, v)
		w.append_string(&sector, s)
		w.append_bool(&up, z > 0)
	}
	return w.df_from(mom, vol, sector, up)
}

main :: proc() {
	df := make_data()
	defer w.destroy_dataframe(&df)

	// 1. DataFrame -> feature matrix and labels ("sector" is label-encoded)
	ds, ok := ml.prepare_dataset(&df, "up")
	defer ml.dataset_free(&ds)
	if !ok do return
	fmt.println("features:", ds.feature_names)

	// 2. Hold out 25% for the final test
	split := ml.train_test_split(&ds.X, ds.y, 0.25, random_state = 1)
	defer ml.train_test_split_free(&split, context.allocator)

	// 3. Choose C by stratified 5-fold cross-validation on the training rows
	base := ml.LogisticParams {
		max_iter = 500, tol = 1e-6, learning_rate = 1, fit_intercept = true, optimizer_type = .LBFGS,
	}
	grid: [4]ml.LogisticParams
	for C, i in ([]f64{0.01, 0.1, 1, 10}) {
		grid[i] = base
		grid[i].C = C
	}
	cv := ml.CVSplitter{kind = .Stratified_KFold, n_splits = 5, shuffle = true, random_state = 1}
	best, cv_acc := ml.grid_search_cv(&split.X_train, split.y_train, cv, ml.logistic_cv_evaluator, grid[:])
	fmt.printf("best C = %v (CV accuracy %.3f)\n", best.C, cv_acc)

	// 4. Scaler + logistic regression, fitted on the training rows
	pipe := ml.pipeline_new()
	defer ml.pipeline_free(&pipe)
	ml.pipeline_add_standard_scaler(&pipe)
	ml.pipeline_set_logistic(&pipe, best)
	ml.pipeline_fit(&pipe, &split.X_train, split.y_train)

	// 5. Evaluate on the test rows
	pred := ml.pipeline_predict(&pipe, &split.X_test)
	defer delete(pred)
	cm := ml.metrics_confusion_matrix(split.y_test, pred, []f64{0, 1})
	defer delete(cm)

	Z := ml.standard_scaler_transform(&pipe.steps[0].standard_scaler, &split.X_test)
	defer l.matrix_free(&Z)
	proba := ml.logistic_predict_proba(&pipe.model.logistic, &Z)
	defer delete(proba)

	fmt.printf("test accuracy %.3f, ROC AUC %.3f, log loss %.3f\n",
		ml.metrics_accuracy(split.y_test, pred),
		ml.metrics_roc_auc(split.y_test, proba),
		ml.metrics_log_loss_binary(split.y_test, proba))
	fmt.println("confusion matrix [tn fp fn tp]:", cm)
	fmt.printf("coefficients (scaled features): %.3f, bias %.3f\n", pipe.model.logistic.weights, pipe.model.logistic.bias)

	// 6. Save, reload and check that the reloaded model predicts the same
	path := "up_model.wotan"
	if !ml.pipeline_save(&pipe, path) do return
	defer os.remove(path)
	loaded, lok := ml.pipeline_load(path)
	defer ml.pipeline_free(&loaded)
	again := ml.pipeline_predict(&loaded, &split.X_test)
	defer delete(again)
	same := lok
	for v, i in again do same = same && v == pred[i]
	fmt.println("reloaded model gives the same predictions:", same)
}
```

```
features: ["momentum", "volatility", "sector"]
best C = 10 (CV accuracy 0.793)
test accuracy 0.780, ROC AUC 0.868, log loss 0.436
confusion matrix [tn fp fn tp]: [24, 8, 3, 15]
coefficients (scaled features): [2.220, -0.905, 0.383], bias -0.101
reloaded model gives the same predictions: true
```

The momentum coefficient is positive and the volatility coefficient negative, as in the model that generated the data. The cross-validation in step 3 runs on unscaled features, while the final model is scaled; with L-BFGS and these small, similar-scale features that makes no difference to the choice, but for features of very different scales write an evaluator that fits the whole pipeline. The sector enters as an ordinal code (section 3), which is crude for a linear model; one-hot encoding it would be the better choice. For probabilities, step 5 applies the pipeline's fitted scaler and logistic model directly, because `pipeline_predict` returns labels.

## 21. Quick reference

| Task | Procedures |
|---|---|
| Matrices | `l.matrix_from_flat`, `l.matrix_new`, `l.matrix_at`, `l.matrix_free`, `l.matrix_from_df`, `l.vector_from_df`, `l.matvec_dyn` |
| DataFrame bridge | `prepare_dataset`, `transform_dataset`, `dataset_free` |
| OLS | `ols_fit` (`.Cholesky`, `.QR`), `ols_from_df`, `_ols_result_free` |
| Robust errors | `ols_fit_robust`, `ols_set_robust_cov` with `RobustCovSpec` (`.HC0`-`.HC3`, `.HAC`, `.Cluster`) |
| WLS / GLS | `wls_fit`, `gls_fit`, `gls_fit_kron` |
| Penalised | `ridge_fit`, `lasso_fit`, `elastic_net_fit`, `ridge_cv`, `lasso_cv` |
| PCR | `pcr_fit`, `pcr_predict`, `pcr_free` |
| Fama-MacBeth | `fama_macbeth`, `fama_macbeth_result_free` |
| Distributions | `student_t_cdf`, `student_t_sf2`, `student_t_ppf`, `f_dist_sf` |
| Logistic | `logistic_fit`, `logistic_predict`, `logistic_predict_proba`, `logistic_free` |
| Naive Bayes | `gnb_fit`, `gnb_predict`, `gnb_predict_proba`, `gnb_free` |
| KNN | `knn_fit`, `knn_predict`, `knn_predict_proba`, `knn_free` |
| SVM | `svm_fit_linear`, `svm_predict_linear`, `svm_free`, `kernel_svm_fit`, `kernel_svm_predict`, `kernel_svm_free` |
| SVR | `svr_fit`, `svr_predict`, `svr_free` |
| Multiclass | `ovr_logistic_fit`, `ovr_linear_svm_fit`, `ovr_kernel_svm_fit`, `ovr_predict`, `ovr_free` |
| Trees | `dt_fit`, `dt_predict`, `dt_predict_proba`, `dt_free`, `tree_feature_importance`, `dt_compute_stats`, `tree_stats_print` |
| Forests | `rf_fit`, `rf_predict`, `rf_predict_proba`, `rf_compute_stats`, `rf_free` |
| Boosting | `gb_fit`, `gb_predict`, `gb_predict_proba`, `gb_decision_function`, `gb_free` |
| MLP | `mlp_fit`, `mlp_predict`, `mlp_free` |
| k-means | `km_fit`, `km_predict`, `km_compute_stats`, `km_stats_print`, `km_stats_free`, `km_free` |
| Weights | `sample_weight =` argument; `class_weight = .Balanced` in the params |
| Scalers | `standard_scaler_*`, `minmax_scaler_*`, `robust_scaler_*` (`_fit`, `_transform`, `_inverse_transform`, `_free`) |
| Encoders | `ohe_fit`, `ohe_transform`, `ohe_free`, `le_fit`, `le_transform`, `le_free` |
| Regression metrics | `metrics_mse`, `metrics_rmse`, `metrics_mae`, `metrics_r2`, `metrics_explained_variance`, `metrics_mape` |
| Classification metrics | `metrics_accuracy`, `metrics_precision`, `metrics_recall`, `metrics_f1`, `metrics_confusion_matrix`, `metrics_classification_report` |
| Score metrics | `metrics_roc_auc`, `metrics_roc_auc_ovr`, `metrics_roc_curve`, `metrics_average_precision`, `metrics_log_loss`, `metrics_log_loss_binary`, `metrics_brier_score` |
| Clustering metrics | `metrics_silhouette`, `metrics_silhouette_samples` |
| Splits | `train_test_split`, `train_test_split_free`, `holdout_split`, `subset_matrix`, `subset_slice` |
| Cross-validation | `CVSplitter` (`.KFold`, `.Stratified_KFold`, `.Time_Series`, `.Purged_KFold`, `.Combinatorial_Purged`), `cv_make_folds`, `cv_folds_free`, `cross_val_score`, `cross_val_score_cv`, `*_cv_evaluator`, `cv_cpcv_paths`, `cv_cpcv_path` |
| Search | `grid_search`, `grid_search_cv`, `grid_search_<model>`, `random_search`, `random_search_cv`, `random_search_<model>` |
| Pipelines | `pipeline_new`, `pipeline_add_*_scaler`, `pipeline_set_<model>`, `pipeline_fit`, `pipeline_predict`, `pipeline_free` |
| Save / load | `pipeline_save`, `pipeline_load`; `model_writer_*`, `write_*`, `model_reader_*`, `read_*` |

For more examples, the tests in [`tests/`](../tests/) exercise every procedure; [`docs/test_catalog.md`](test_catalog.md) lists which test covers which model. The reference tests (`ml_*_reference_test.odin`, `ols_robust_reference_test.odin`) show the exact correspondence with scikit-learn and statsmodels parameters. The sources are in [`wotan/analytics/ML/`](../wotan/analytics/ML/); the doc comment above each procedure states its conventions.
