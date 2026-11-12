# The `ml_finance` package, part 1: methods for financial machine learning

This is the first part of the guide to `wotan/ml_finance` ([part 2](ml_finance_asset_pricing_guide.md) covers the asset-pricing models). It covers the tools that surround a model rather than the models themselves, mostly after López de Prado's *Advances in Financial Machine Learning* (AFML, 2018) and *Machine Learning for Asset Managers* (MLAM, 2020):
- events and labels: the CUSUM filter, triple-barrier labels, meta-labels and bet sizes;
- sample weights for overlapping labels, and the sequential bootstrap;
- fractional differencing: stationary features that keep memory;
- feature importance: MDI, MDA and SFI, single and clustered;
- backtest statistics: the probabilistic and deflated Sharpe ratios, the minimum track record length and the probability of backtest overfitting;
- signal evaluation: the cross-sectional rank IC, and a small set of trading features;
- hierarchical risk parity;
- transaction costs with square-root market impact;
- split conformal prediction intervals.

Every example was compiled and run against the current code, and the output shown is the real output. A debug build with a tracking allocator reports no leaks for any of them. The results agree with López de Prado's code, pandas, statsmodels, scikit-learn and SciPy; section 11 lists the comparisons.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [Events and labels](#2-events-and-labels)
3. [Sample weights](#3-sample-weights)
4. [Fractional differencing](#4-fractional-differencing)
5. [Feature importance](#5-feature-importance)
6. [Backtest statistics](#6-backtest-statistics)
7. [Signal evaluation](#7-signal-evaluation)
8. [Hierarchical risk parity](#8-hierarchical-risk-parity)
9. [Transaction costs](#9-transaction-costs)
10. [Conformal prediction intervals](#10-conformal-prediction-intervals)
11. [Validation](#11-validation)
12. [Quick reference](#12-quick-reference)

## 1. Setup and conventions

```odin
import ml "wotan/analytics/ML"
import l "wotan/linalg"
import mlf "wotan/ml_finance"
import t "wotan/tensor"
```

The conventions in this part:
- **Bars.** A price series is `[]f64` indexed by bar. Events and label spans are bar indices.
- **Matrices.** Feature matrices and trial returns are `l.Matrix(f64)`, one row per sample or period.
- **Sharpe ratios** are per period: mean / sample standard deviation of the period returns. Multiply by √(periods per year) to annualize.
- **Results.** Procedures that return slices allocate them; free them with `delete`. Result structs have free procedures: `importance_result_free`, `feature_clusters_free`, `pbo_result_free`, `hrp_result_free` and `conformal_free`.
- **Models and cross-validation** come from `analytics/ML`: random forests and trees (`ml.RFParams`, `ml.TreeParams`), and `ml.CVSplitter`, including purged and combinatorial purged K-fold. The [ML guide](ml_guide.md) covers them.

## 2. Events and labels

### Events

Labelling every bar gives heavily overlapping, mostly uninformative samples. AFML samples bars where something happened instead:
- `cusum_filter(x, threshold)` is the symmetric CUSUM filter (snippet 2.4), usually on log prices. It sums the increments into a positive and a negative sum, each floored at 0. An event fires when either sum passes ±threshold, which resets that sum.
- `daily_volatility(close, span)` is the EWM standard deviation of the simple returns, as pandas `close.pct_change().ewm(span=span).std()`. It is the usual barrier width.

```odin
// 1000 bars of a random walk in log prices, 1% daily volatility
rand.reset(42)
n := 1000
close := make([]f64, n)
logp := make([]f64, n)
defer {delete(close); delete(logp)}
logp[0] = math.ln(100.0)
for i in 1 ..< n {logp[i] = logp[i - 1] + 0.01 * rand.float64_normal(0, 1)}
for i in 0 ..< n {close[i] = math.exp(logp[i])}

events := mlf.cusum_filter(logp, 0.02)
defer delete(events)
vol := mlf.daily_volatility(close, 50)
defer delete(vol)
```

```
184 CUSUM events, first [4, 5, 9, 14, 16, 21]; daily vol at bar 500 0.00944
```

### Triple-barrier labels

`triple_barrier_labels(close, events, target, params, side)` labels each event by which of three barriers the price path touches first (AFML ch. 3):
- the upper barrier at +`profit_taking` × target;
- the lower barrier at −`stop_loss` × target;
- the vertical barrier after `max_holding` bars, or at the end of the data.

`target` is per bar, read at the event bar; usually it is `daily_volatility`. A barrier multiple of 0 disables that barrier. Each `TripleBarrierLabel` holds:
- the span: `start` (the event bar) and `end` (the bar of the first touch);
- the return over the span, and the label: the sign of the return (−1, 0, 1);
- the barrier touched, and the target.

With `zero_on_vertical`, a vertical touch gives label 0. Events with a NaN target, a target ≤ `min_target`, or no bar after them are dropped.

```odin
params := mlf.TripleBarrierParams {
	profit_taking = 2,
	stop_loss     = 2,
	max_holding   = 10,
}
labels := mlf.triple_barrier_labels(close, events, vol, params)
defer delete(labels)
```

```
184 labels: [.Profit_Taking = 63, .Stop_Loss = 87, .Vertical = 34]; label 1: 80
  start 4 end 5 ret -0.0211 label -1 Stop_Loss target 0.0092
  start 5 end 15 ret +0.0158 label 1 Vertical target 0.0140
  start 9 end 19 ret -0.0073 label -1 Vertical target 0.0133
```

**Meta-labels.** Passing `side`, one per event, gives meta-labels. `side` is the bet of a primary model (+1 long, −1 short). The return is then measured in the bet's direction, and the label is 1 if the bet made money, else 0. A secondary model trained on these labels learns when to trust the primary one.

```odin
side := make([]f64, len(events))
defer delete(side)
for &s in side {s = 1} // a primary model that always bets long
meta := mlf.triple_barrier_labels(close, events, vol, params, side)
defer delete(meta)
```

```
meta-labels: the long bet paid in 80 of 184
```

**Label spans for purged cross-validation.** Labels that overlap in time leak information between training and test folds. `ml.CVSplitter` with `.Purged_KFold` or `.Combinatorial_Purged` removes the overlapping training samples, given each sample's label end in sample units. `triple_barrier_sample_ends(labels)` converts the bar spans: `out[i]` is the index of the last label starting at or before label i's end bar. An embargo is then counted in labels, not bars.

```
sample ends of the first labels [1, 3, 4, 4, 5, 6]
```

**Bet sizes.** `bet_size_from_probability(p, side)` turns a classifier's probability p that a bet pays into a size in [−1, 1] (snippet 10.1): z = (p − ½)/√(p(1 − p)) and size = side · (2Φ(z) − 1).

```
bet size at p = 0.50: 0.0000
bet size at p = 0.55: 0.0801
bet size at p = 0.60: 0.1617
bet size at p = 0.75: 0.4363
bet size at p = 0.90: 0.8176
```

## 3. Sample weights

Overlapping labels are not independent. AFML ch. 4 weights them by how much they share. All of the procedures below take the labels from section 2:
- `label_concurrency(n_bars, labels)` counts the labels alive at each bar; a span includes both ends.
- `average_uniqueness(labels, concurrency)` is the mean of 1/concurrency over each label's span. It is 1 for a label that overlaps nothing.
- `return_attribution_weights(close, labels, concurrency)` is |Σ log-return / concurrency| over the bars the label earns, scaled to sum to the number of labels (snippet 4.10). Unlike the book's code, it leaves out the event bar's own return, which happened before the label existed.
- `time_decay_weights(weights, uniqueness, c)` multiplies the weights by a linear decay along the cumulative uniqueness: the newest label keeps 1 and the oldest gets c. A negative c zeroes the oldest fraction −c.
- `sequential_bootstrap(labels, n_bars, n_draws)` draws labels with probabilities that fall as a label overlaps the labels already drawn (snippets 4.5, 4.6). `sequential_bootstrap_probabilities` gives the probabilities of one draw. It uses `context.random_generator`.

```odin
conc := mlf.label_concurrency(n, labels)
u := mlf.average_uniqueness(labels, conc)
ra := mlf.return_attribution_weights(close, labels, conc)
defer {delete(conc); delete(u); delete(ra)}
decayed := make([]f64, len(u))
defer delete(decayed)
for &v in decayed {v = 1}
mlf.time_decay_weights(decayed, u, 0.5)
rand.reset(7)
draws := mlf.sequential_bootstrap(labels, n)
defer delete(draws)
```

```
mean uniqueness 0.7191; first uniqueness [0.7500, 0.6061, 0.4091, 0.3333]; first return-attribution weights [0.7872, 0.3004, 0.1494, 0.6509]
time decay c = 0.5: oldest 0.5028, middle 0.7478, newest 1.0000
uniqueness of a sequential bootstrap sample 0.6136, of a standard one 0.5246
```

The sequential bootstrap draws a sample whose labels overlap less than a standard bootstrap's.

## 4. Fractional differencing

Prices are not stationary. Returns are, but they discard the price level's memory. Fractional differencing (AFML ch. 5) applies (1 − B)^d with 0 < d < 1, which yields a stationary series that still correlates with the price:
- `frac_diff_weights(d, threshold)` gives the weights w₀ = 1, w_k = −w_{k−1}(d − k + 1)/k, up to the last one with |w_k| ≥ threshold.
- `frac_diff_ffd(x, d, threshold)` applies them over a fixed-width window (snippet 5.3). The first len(weights) − 1 bars are NaN.
- `frac_diff_min_d(x, step, max_d, threshold, significance, adf_lags)` finds the smallest d on a grid (0, 0.05, …) whose series rejects a unit root. The test is ADF with a constant and one fixed lag, as the book's `adfuller(maxlag=1, regression='c', autolag=None)`. The result also gives the series' correlation with the original.

```odin
wts := mlf.frac_diff_weights(0.4, 1e-3)
defer delete(wts)
ffd := mlf.frac_diff_ffd(logp, 0.4, 1e-3)
defer delete(ffd)
res := mlf.frac_diff_min_d(logp)
```

```
d = 0.4: 55 weights, first [1.0000, -0.4000, -0.1200, -0.0640, -0.0416]
FFD series: NaN for the first 54 bars, then [0.6065, 0.6165, 0.6191]
smallest d 0.20 (found true): ADF -3.401 p 0.0109, correlation with the log price 0.8848
```

At d = 0.2 the series is stationary at 5% and still correlates 0.88 with the log price.

## 5. Feature importance

The three methods of AFML ch. 8 answer different questions:

| Method | What it measures | Data |
|---|---|---|
| MDI (mean decrease impurity) | how much a forest's splits on the feature reduce impurity | in sample |
| MDA (mean decrease accuracy) | how much the out-of-sample score drops when the feature's test values are permuted | cross-validated |
| SFI (single feature importance) | the cross-validated score of a model on the feature alone | cross-validated |

- `mdi_importance(&forest, n_features, groups)` takes a fitted forest. Fit it with `max_features = 1`, so that every feature is offered at some split and none is masked by a stronger one. A feature a tree never used is left out of that tree's mean. The means are normalized to sum to 1.
- `mda_importance(model, &X, y, cv, sample_weight, scoring, groups, seed)` refits the model on each fold. It reports the score lost by permuting a feature as a share of the remaining loss: (base − permuted) / −permuted for the negative log loss, / (1 − permuted) for accuracy. It also returns the mean unpermuted out-of-sample score. `seed ≠ 0` makes the permutations reproducible.
- `sfi_importance(model, &X, y, cv, sample_weight, scoring)` gives the mean and standard error of the scores, not normalized.

The model is an `ImportanceModel`: fit, predict-probability and free procedures. `rf_importance_model(&params)` and `dt_importance_model(&params)` wrap a classification forest or tree; the params must outlive the model. Use a purged `cv` when the labels overlap. Every result has a mean and a standard error per feature: the standard deviation over trees or folds, times n^−½, as in AFML.

In the example, features 0 and 1 drive the label, 2 and 3 are noisy copies of 0, and 4 and 5 are noise:

```odin
rand.reset(1)
n, F := 600, 6
X := l.matrix_new(f64, n, F)
defer l.matrix_free(&X)
y := make([]f64, n)
defer delete(y)
for i in 0 ..< n {
	a, b := rand.float64_normal(0, 1), rand.float64_normal(0, 1)
	row := X.data[i * F:][:F]
	row[0], row[1] = a, b
	row[2], row[3] = a + 0.3 * rand.float64_normal(0, 1), a + 0.3 * rand.float64_normal(0, 1)
	row[4], row[5] = rand.float64_normal(0, 1), rand.float64_normal(0, 1)
	y[i] = 1 if a + b + 0.5 * rand.float64_normal(0, 1) > 0 else 0
}
params := ml.RFParams {
	n_trees      = 100,
	max_features = 1,
	min_samples  = 2,
	max_depth    = 6,
	bootstrap    = true,
	criterion    = .Entropy,
}
forest := ml.rf_fit(&X, y, params)
defer ml.rf_free(&forest)
mdi := mlf.mdi_importance(&forest, F)
defer mlf.importance_result_free(&mdi)

model := mlf.rf_importance_model(&params)
cv := ml.CVSplitter{kind = .KFold, n_splits = 5}
mda, oos := mlf.mda_importance(model, &X, y, cv, seed = 1)
defer mlf.importance_result_free(&mda)
tree := ml.TreeParams{max_depth = 3, min_samples = 2, criterion = .Entropy}
sfi := mlf.sfi_importance(mlf.dt_importance_model(&tree), &X, y, cv, scoring = .Accuracy)
defer mlf.importance_result_free(&sfi)
```

```
MDI [0.186, 0.348, 0.159, 0.166, 0.071, 0.070]
    se [0.013, 0.012, 0.013, 0.013, 0.004, 0.004]
MDA [0.203, 0.451, 0.118, 0.082, 0.010, 0.019]
    se [0.028, 0.030, 0.010, 0.011, 0.011, 0.004]; out-of-sample score (-log loss) -0.3549
SFI (accuracy) [0.713, 0.753, 0.707, 0.652, 0.533, 0.475]
```

Feature 0 shares its importance with its two copies: MDI and MDA split it between the substitutes, while SFI, which sees one feature at a time, scores each copy close to feature 0 itself.

**Clustered importance** (MLAM 6.4, 6.5) removes that dilution by scoring groups of features together:
- `cluster_features(&X, max_clusters, n_init)` clusters the features by k-means on their correlation distance √((1 − ρ)/2). It keeps the number of clusters with the highest silhouette t-statistic (MLAM 4.1).
- Pass its `groups` to `mdi_importance` or `mda_importance`. A group's MDI is the sum over its features. MDA permutes all of a group's features at once.

```odin
clusters := mlf.cluster_features(&X)
defer mlf.feature_clusters_free(&clusters)
cmda, _ := mlf.mda_importance(model, &X, y, cv, groups = clusters.groups, seed = 1)
defer mlf.importance_result_free(&cmda)
cmdi := mlf.mdi_importance(&forest, F, clusters.groups)
defer mlf.importance_result_free(&cmdi)
```

```
clusters [[0, 2, 3], [1, 4, 5]], labels [0, 1, 0, 0, 1, 1]
clustered MDA [0.387, 0.478], clustered MDI [0.509, 0.491]
```

The three copies of feature 0 form one cluster. The noise features are uncorrelated with everything, so they land with feature 1; with more noise features they would form clusters of their own.

`mdi_from_tree_importances` and `mda_from_scores` are the aggregation steps alone, for importances computed elsewhere.

## 6. Backtest statistics

A high Sharpe ratio may be luck, especially when it is the best of many trials. These procedures quantify that.

**The probabilistic Sharpe ratio** (Bailey & López de Prado 2012) is the probability that the true Sharpe ratio exceeds a benchmark SR*, given the observed SR, the number of returns n, and their skewness γ₃ and kurtosis γ₄:

PSR = Φ((SR − SR*)·√(n − 1) / √(1 − γ₃·SR + (γ₄ − 1)/4 · SR²))

- `sharpe_stats(returns)` gives the SR, the skewness, the kurtosis (3 for a normal, not excess) and n.
- `probabilistic_sharpe_ratio(sharpe, benchmark, n, skewness, kurtosis)` computes the PSR; `probabilistic_sharpe_ratio_returns(returns, benchmark)` computes it from a return series.
- `min_track_record_length(sharpe, benchmark, skewness, kurtosis, probability)` is the number of returns needed for the PSR to reach `probability`.

```odin
// 500 daily returns: mean 0.06%, sd 1.1% (annual SR about 0.9), a 3% loss on 2% of the days
rand.reset(3)
T := 500
r := make([]f64, T)
defer delete(r)
for &x in r {
	x = 0.0012 + 0.01 * rand.float64_normal(0, 1)
	if rand.float64() < 0.02 {x -= 0.03}
}
s := mlf.sharpe_stats(r)
```

```
SR 0.0827 (annual 1.31), skewness -0.359, kurtosis 3.832, n 500
PSR(0) 0.9654; with normal moments 0.9675
minimum track record at 95%: 410 returns
```

**The deflated Sharpe ratio** (Bailey & López de Prado 2014) corrects for selection. The best of N strategies without skill, whose Sharpe ratios vary with variance V across the trials, is expected to reach

SR₀ = √V · ((1 − γ)Φ⁻¹(1 − 1/N) + γΦ⁻¹(1 − 1/(N·e))), where γ is Euler's constant.

The DSR is the PSR against SR₀:
- `expected_max_sharpe(n_trials, sharpe_variance)` computes SR₀;
- `deflated_sharpe_ratio(sharpe, n, skewness, kurtosis, n_trials, sharpe_variance)` computes the DSR.

The paper's worked example (annual SR 2.5 over 1250 days, 100 trials, annual variance 0.5, skewness −3, kurtosis 10):

```odin
sr := 2.5 / math.sqrt(250.0)
V := 0.5 / 250
fmt.printf("BLdP 2014 example: SR0 %.4f, DSR %.4f\n", mlf.expected_max_sharpe(100, V),
	mlf.deflated_sharpe_ratio(sr, 1250, -3, 10, 100, V))
```

```
BLdP 2014 example: SR0 0.1132, DSR 0.9004
```

These are the paper's numbers.

`deflated_sharpe_trials(&returns)` takes the returns of all trials, one column each. It picks the best column and computes its PSR and DSR, with N the number of columns and V the variance of their Sharpe ratios. Trials that were run but are not in the matrix make N too small and the DSR too optimistic.

**The probability of backtest overfitting** (Bailey, Borwein, López de Prado & Zhu 2017) uses combinatorially symmetric cross-validation (CSCV):
1. `backtest_overfitting(&returns, n_blocks, metric)` cuts the periods into S blocks (S even).
2. For each of the C(S, S/2) ways to choose half of the blocks as the in-sample set, it finds the trial with the best in-sample metric.
3. It ranks that trial's out-of-sample metric among all trials.

The PBO is the share of combinations in which the in-sample winner is no better than the out-of-sample median. The result also has:
- the probability that the winner loses out of sample;
- the regression of its out-of-sample metric on its in-sample one;
- the per-combination logits and metrics.

The metric is the per-period Sharpe ratio unless one is given.

```odin
N := 50
trials := l.matrix_new(f64, 1000, N)
defer l.matrix_free(&trials)
for &x in trials.data {x = 0.01 * rand.float64_normal(0, 1)} // no skill
d := mlf.deflated_sharpe_trials(&trials)
p := mlf.backtest_overfitting(&trials, 16)
defer mlf.pbo_result_free(&p)
```

```
best of 50 unskilled: trial 9, annual SR 1.24, PSR 0.993, SR0 1.23, DSR 0.507
PBO 0.379 over 12870 combinations, P[OOS loss] 0.392, slope -0.341
```

The best of 50 unskilled trials looks significant on its own (PSR 0.993). Against the maximum that 50 trials of noise reach, it is not (DSR 0.507).

`print_selection_stats(&trials, periods_per_year, n_blocks)` prints both reports, with annualized Sharpe ratios. Here, trial 7 is given a real edge:

```odin
for t in 0 ..< 1000 {trials.data[t * N + 7] += 0.002} // trial 7 has skill
mlf.print_selection_stats(&trials)
```

```
   selection over 50 trials of 1000 periods: best trial 7, annual SR 2.91, PSR 1.000; the best of 50 without skill would reach SR 1.54 -> DSR 0.997
   CSCV over 16 blocks: PBO 0.01, P[the in-sample best loses out of sample] 0.01, OOS SR = 5.50 -0.90 × IS SR (annual)
```

The skilled trial survives the deflation and is almost never overfit. The negative slope is expected here: the same trial wins every combination, and its in-sample and out-of-sample halves partition one fixed history, so a better in-sample half means a worse out-of-sample one.

## 7. Signal evaluation

### Cross-sectional rank IC

`cross_sectional_rank_ic(preds, targets, allocator)` computes, for each cross-section (for example each day across all assets), the Spearman correlation between predictions and realized returns. It then summarizes the series: mean IC, its standard deviation, ICIR = mean / sd, and the share of positive periods. Both tensors have the shape [periods, assets, 1, 1]. As in alphalens and Qlib:
- assets with a NaN prediction or target are dropped from their cross-section;
- a cross-section with fewer than 2 assets or constant values is left out, rather than counted as an IC of 0.

```odin
// 60 days x 30 assets; the prediction is the return plus noise
rand.reset(2)
B, A := 60, 30
pd := l.matrix_new(f64, 1, B * A)
yd := l.matrix_new(f64, 1, B * A)
for i in 0 ..< B * A {
	yd.data[i] = rand.float64_normal(0, 1)
	pd.data[i] = yd.data[i] + 4 * rand.float64_normal(0, 1)
}
pd.data[5] = math.nan_f64() // a missing prediction: dropped from day 0
P := t.tensor_new(pd)
Y := t.tensor_new(yd)
defer {t.tensor_free(P); t.tensor_free(Y)}
P.shape, Y.shape = {B, A, 1, 1}, {B, A, 1, 1}
ic := mlf.cross_sectional_rank_ic(P, Y, context.allocator)
```

```
mean IC 0.1986, sd 0.1746, ICIR 1.1377, positive 0.833, periods 60
```

### Trading features

`compute_trading_features(prices, volumes, allocator)` returns 4 features per day, row-major, each computed from data up to and including that day:
- the log return;
- the RSI(14), as (RSI − 50)/50. It follows TA-Lib's Wilder smoothing and is 0 until day 14;
- the volume z-score against the 20 previous days, clipped to ±3 and scaled to ±1;
- the volume ratio to the 20-day mean, minus 1, clipped to ±2 and scaled to ±1.

The volume features are 0 for the first 20 days.

```
day  1: [0.0033, 0.0000, 0.0000, 0.0000]
day 14: [-0.0100, 0.2691, 0.0000, 0.0000]
day 15: [-0.0096, 0.1439, 0.0000, 0.0000]
day 30: [-0.0055, 0.3379, 1.0000, 1.0000]
```

Day 30 has a volume spike, so both volume features are at their caps.

## 8. Hierarchical risk parity

HRP (López de Prado 2016) allocates without inverting the covariance matrix, which makes it stable when assets are highly correlated:
1. It computes the correlation distance d = √(½(1 − ρ)).
2. It clusters by single linkage on the distance between the columns of d (the paper's step). With `linkage = .Correlation_Distance`, it clusters on d itself, as PyPortfolioOpt does.
3. It orders the assets by the dendrogram's leaves.
4. It bisects the ordered list recursively. Each half gets a share inversely proportional to the variance of its inverse-variance portfolio.

There are two entry points:
- `hrp_allocate_cov(&cov, allocator, linkage)` works from a covariance matrix;
- `hrp_allocate(&returns, allocator, limit, linkage)` uses the sample covariance of a [T, N] return matrix. With `limit`, it caps every weight at 30%; this is a heuristic outside HRP and off by default.

```odin
// 4 assets: two correlated pairs
vol := []f64{0.10, 0.12, 0.20, 0.25}
corr := []f64{1, 0.8, 0.2, 0.1, 0.8, 1, 0.1, 0.2, 0.2, 0.1, 1, 0.7, 0.1, 0.2, 0.7, 1}
cov := l.matrix_new(f64, 4, 4)
defer l.matrix_free(&cov)
for i in 0 ..< 4 {for j in 0 ..< 4 {cov.data[i * 4 + j] = corr[i * 4 + j] * vol[i] * vol[j]}}
h := mlf.hrp_allocate_cov(&cov)
defer mlf.hrp_result_free(&h)
```

```
weights [0.4695, 0.3261, 0.1246, 0.0798], order [0, 1, 2, 3]
clustering on d itself: weights [0.4695, 0.3261, 0.1246, 0.0798]
```

SciPy's single linkage with López de Prado's bisection gives the same order and weights.

## 9. Transaction costs

A trade of notional q in an asset with daily return volatility σ and average daily traded value ADV costs

cost(q) = |q| · (commission + half_spread + Y·σ·(|q| / ADV)^b)

With b = 0.5 this is the square-root law of market impact (Torre 1997; Almgren et al. 2005). With b = 1 the impact is linear, and equals the Almgren–Chriss temporary impact with η = Y·σ·S²/ADV (`almgren_chriss_eta`). The impact term is skipped when σ or ADV is not positive. `DEFAULT_COST_MODEL` is an assumption for liquid US ETFs and large caps: no commission, a 1 bp half spread, and square-root impact with Y = 1.

The procedures:
- `trade_cost_rate(model, notional, sigma, adv)` and `trade_cost` give the cost of one trade, per unit of notional or in currency;
- `rebalance_cost(model, before, after, capital, sigma, adv)` gives the cost of moving between weight vectors, as a fraction of capital;
- `position_costs(model, positions, capital, sigma, adv, initial)` gives the cost of each period's position change, for a backtest;
- `liquidity_estimates(close, volume, window)` gives σ and ADV known at each day's close, from the trailing `window` days, without look-ahead;
- `net_of_costs(gross, cost)` gives net returns.

```odin
m := mlf.DEFAULT_COST_MODEL
for q in ([]f64{1e5, 1e7, 1e9}) {
	fmt.printf("trade %.0e of an asset with 1%% daily vol, ADV 1e9: %.2f bp, cost %.0f\n", q,
		1e4 * mlf.trade_cost_rate(m, q, 0.01, 1e9), mlf.trade_cost(m, q, 0.01, 1e9))
}
before := []f64{0.25, 0.25, 0.25, 0.25}
after := []f64{0.40, 0.30, 0.20, 0.10}
sig := []f64{0.01, 0.012, 0.02, 0.025}
adv := []f64{5e9, 2e9, 1e9, 2e8}
fmt.printf("rebalance of 1e8: %.2f bp of capital\n", 1e4 * mlf.rebalance_cost(m, before, after, 1e8, sig, adv))
pos := []f64{0, 1, 1, -1, 0}
pc := mlf.position_costs(m, pos, 1e6)
defer delete(pc)
```

```
trade 1e+05 of an asset with 1% daily vol, ADV 1e9: 2.00 bp, cost 20
trade 1e+07 of an asset with 1% daily vol, ADV 1e9: 11.00 bp, cost 11000
trade 1e+09 of an asset with 1% daily vol, ADV 1e9: 101.00 bp, cost 10100000
rebalance of 1e8: 12.50 bp of capital
position costs (spread only) [0.0000, 0.0001, 0.0000, 0.0002, 0.0001]
Almgren-Chriss eta for price 50, vol 2%, ADV 1e8: 5.000e-07
liquidity: sigma[19] NaN sigma[20] 0.01390 adv[19] 1.0095e+08 adv[29] 1.0105e+08
net of costs [0.0100, -0.0201, 0.0150, -0.0002, 0.0049]
```

A hundred times the size costs ten times as much per unit. The position series pays the spread on each change, twice for the reversal from +1 to −1. σ needs `window + 1` closes, so it starts a day after ADV.

## 10. Conformal prediction intervals

Split conformal prediction (Lei et al. 2018) turns any point forecast into an interval with guaranteed coverage. `conformal_calibrate(&cp, actual, predicted, alpha)` takes a calibration set of forecasts and outcomes:
- the scores are the absolute errors |y − ŷ|;
- Q is the k-th smallest of the n scores, with k = ⌈(n + 1)(1 − α)⌉;
- for a new point exchangeable with the calibration data, P(|y − ŷ| ≤ Q) ≥ 1 − α.

`conformal_predict_interval(&cp, y_hat)` returns [ŷ − Q, ŷ + Q], and `print_conformal_stats` prints a summary. Pairs with a NaN are left out. When n < 1/α − 1, no finite Q has the guarantee, and Q is +Inf.

```odin
rand.reset(4)
nc := 99
actual := make([]f64, nc)
pred := make([]f64, nc)
defer {delete(actual); delete(pred)}
for i in 0 ..< nc {pred[i] = rand.float64_normal(0, 1); actual[i] = pred[i] + 0.5 * rand.float64_normal(0, 1)}
cp := mlf.conformal_new()
defer mlf.conformal_free(&cp)
mlf.conformal_calibrate(&cp, actual, pred, 0.1)
lo, hi := mlf.conformal_predict_interval(&cp, 0.3)
```

```
Q 0.8083 (the 90th of 99 scores); interval for 0.3: [-0.5083, 1.1083]
coverage on 10,000 new points 0.8953 (target 0.90)
10 calibration points at alpha 0.05: Q +Inf
```

The guarantee holds on average over calibration sets; a single calibration set can cover slightly less or more. Exchangeability rules out using a calibration set from a different volatility regime than the forecasts.

## 11. Validation

The tests are registered in `main.odin` and listed in [the test catalog](test_catalog.md):

| Area | Reference | Test |
|---|---|---|
| Events, labels, weights, fractional differencing | López de Prado's snippets in pandas / statsmodels on one price path: CUSUM, `ewm(span).std()`, triple-barrier labels and meta-labels, concurrency, uniqueness, return attribution, time decay, sequential-bootstrap probabilities, FFD weights and series, the smallest d passing `adfuller`; bet sizes vs SciPy | `labeling_reference_test` |
| Feature importance | MDI and MDA aggregation vs pandas; SFI and MDA end to end with scikit-learn's `DecisionTreeClassifier`, KFold, sample weights and reproducible permutations; `cluster_features` vs MLAM's `clusterKMeansBase` | `feature_importance_reference_test` |
| Backtest statistics | PSR, DSR, MinTRL and SR₀ vs SciPy and the worked example of Bailey & López de Prado (2014); PBO by CSCV vs an itertools / `array_split` / `rankdata` reference | `backtest_stats_reference_test` |
| Purged cross-validation | `Purged_KFold` and `Combinatorial_Purged` vs a pairwise-purging Python reference | `purged_cv_reference_test` |
| Rank IC, trading features, conformal | rank IC vs `scipy.stats.spearmanr` + pandas; features vs TA-Lib RSI(14) and pandas rolling volume; the split-conformal quantile | `ml_finance_metrics_reference_test` |
| HRP | López de Prado's reference code | `black_litterman_hrp_reference_test` |
| Transaction costs | NumPy for trades, a rebalance and a position series; trailing σ and ADV vs pandas `rolling(20)`; with b = 1 the costs equal `almgren_chriss_solve`'s expected shortfall | `transaction_costs_reference_test` |

The application tests check the behaviour the methods are for:
- the sequential bootstrap's higher uniqueness, and meta-labeling raising precision (`labeling_test`);
- MLAM's synthetic experiment for clustered importance (`feature_importance_test`);
- PSR calibration under skew, and DSR / PBO separating skill from selection (`backtest_stats_test`);
- leakage under shuffled versus purged K-fold (`purged_cv_test`);
- the square-root scaling of costs with capital (`transaction_costs_test`).

For this guide, the bet sizes, the fractional-differencing weights, the HRP weights and the rebalance cost were also recomputed independently with SciPy and NumPy, and agree to the printed decimals.

## 12. Quick reference

### Labels and weights

| Procedure | Purpose |
|---|---|
| `cusum_filter(x, threshold)` | event bars |
| `daily_volatility(close, span)` | EWM sd of returns (barrier width) |
| `triple_barrier_labels(close, events, target, params, side)` | labels or meta-labels |
| `triple_barrier_sample_ends(labels)` | label ends for purged CV |
| `bet_size_from_probability(p, side)` | size in [−1, 1] |
| `label_concurrency` / `average_uniqueness` | overlap of labels |
| `return_attribution_weights` / `time_decay_weights` | sample weights |
| `sequential_bootstrap` / `sequential_bootstrap_probabilities` | bootstrap of overlapping labels |
| `frac_diff_weights` / `frac_diff_ffd` / `frac_diff_min_d` | fractional differencing |

### Feature importance

| Procedure | Purpose |
|---|---|
| `mdi_importance(&forest, n_features, groups)` | MDI of a fitted forest |
| `mda_importance(model, &X, y, cv, weights, scoring, groups, seed)` | permutation importance, cross-validated |
| `sfi_importance(model, &X, y, cv, weights, scoring)` | single-feature CV scores |
| `rf_importance_model` / `dt_importance_model` | `ImportanceModel` for a forest or tree |
| `cluster_features(&X, max_clusters, n_init)` | feature clusters for clustered MDI / MDA |
| `mdi_from_tree_importances` / `mda_from_scores` | aggregation steps |

### Backtest statistics

| Procedure | Purpose |
|---|---|
| `sharpe_stats(returns)` | SR, skewness, kurtosis, n |
| `probabilistic_sharpe_ratio` / `probabilistic_sharpe_ratio_returns` | PSR |
| `min_track_record_length` | returns needed for a PSR level |
| `expected_max_sharpe(n_trials, V)` | SR₀ |
| `deflated_sharpe_ratio` / `deflated_sharpe_trials` | DSR |
| `sharpe_ratios(&returns)` | per-column SR |
| `backtest_overfitting(&returns, n_blocks, metric)` | PBO by CSCV |
| `print_selection_stats(&trials, periods_per_year, n_blocks)` | DSR and PBO report |

### Signals, portfolios, costs, intervals

| Procedure | Purpose |
|---|---|
| `cross_sectional_rank_ic(preds, targets, allocator)` | rank IC summary |
| `compute_trading_features(prices, volumes, allocator)` | return, RSI and volume features |
| `hrp_allocate(&returns, …)` / `hrp_allocate_cov(&cov, …)` | HRP weights and leaf order |
| `trade_cost_rate` / `trade_cost` | cost of one trade |
| `rebalance_cost` / `position_costs` / `net_of_costs` | costs in a backtest |
| `liquidity_estimates(close, volume, window)` | trailing σ and ADV |
| `almgren_chriss_eta` | linear-impact η |
| `conformal_new` / `conformal_calibrate` / `conformal_predict_interval` / `print_conformal_stats` | split conformal intervals |

`gat_cross_val_score`, also in `metrics.odin`, is covered with the graph models in a later part.
