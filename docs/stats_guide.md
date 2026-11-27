# Statistics guide (`stats`)

The `stats` package holds classical statistics:
- **Probability distributions** and the special functions behind them: densities, cumulative probabilities and their tails, quantiles, moments and random draws for eleven distributions.
- **Hypothesis tests:** t- and z-tests, tests and intervals for proportions, chi-squared and Fisher's exact test, rank tests, Kolmogorov-Smirnov and normality tests, tests for equal variances, correlation tests, and corrections for multiple testing.

Both match scipy.stats and statsmodels, at their accuracy or better.

## Contents

1. [Setup](#1-setup)
2. [Distributions as values](#2-distributions-as-values)
3. [Continuous distributions](#3-continuous-distributions)
4. [Discrete distributions](#4-discrete-distributions)
5. [Tails and quantiles](#5-tails-and-quantiles)
6. [Random draws](#6-random-draws)
7. [Special functions](#7-special-functions)
8. [Hypothesis tests: conventions](#8-hypothesis-tests-conventions)
9. [Means](#9-means)
10. [Proportions and counts](#10-proportions-and-counts)
11. [Rank tests](#11-rank-tests)
12. [Distribution and normality tests](#12-distribution-and-normality-tests)
13. [Variances and correlation](#13-variances-and-correlation)
14. [Multiple testing](#14-multiple-testing)
15. [Tests on DataFrame columns](#15-tests-on-dataframe-columns)
16. [Accuracy and validation](#16-accuracy-and-validation)
17. [Quick reference](#17-quick-reference)

## 1. Setup

```odin
import st "wotan/stats"
import "core:fmt"
```

The distributions and special functions allocate nothing except `sample`, which returns a slice to `delete`. The tests take slices and return their results by value; their scratch memory comes from `context.temp_allocator` and is released before they return. Only `p_adjust` and the DataFrame helpers of section 15 return memory to free.

## 2. Distributions as values

A distribution is a struct of its parameters, such as `st.Student_T{df = 24}` or `st.Binomial{n = 50, p = 0.1}`. The procedures take it as their first argument. Each one exists under the distribution's name (`st.student_t_cdf(d, x)`) and in a group that takes any distribution (`st.cdf(d, x)`):

```odin
t := st.Student_T{df = 24}
fmt.printf("t(24): 97.5%% quantile %.6f, P(T > 2.8) = %.6g\n", st.ppf(t, 0.975), st.sf(t, 2.8))
chi := st.Chi_Squared{df = 10}
fmt.printf("chi-squared(10): P(X > 35) = %.6g, 95%% quantile %.6f\n", st.sf(chi, 35), st.ppf(chi, 0.95))

z := st.STANDARD_NORMAL
fmt.printf("normal: P(Z > 9) = %.6g, 1 - cdf(9) = %.6g\n", st.sf(z, 9), 1 - st.cdf(z, 9))
fmt.printf("normal: isf(1e-12) = %.6f\n", st.isf(z, 1e-12))

b := st.Binomial{n = 50, p = 0.1}
fmt.printf(
    "binomial(50, 0.1): P(X = 5) = %.6f, P(X <= 2) = %.6f, P(X > 10) = %.6g, median %v\n",
    st.pmf(b, 5), st.cdf(b, 2), st.sf(b, 10), st.ppf(b, 0.5),
)

r := rand.create(42)    // import "core:math/rand"
context.random_generator = rand.default_random_generator(&r)
draws := st.sample(st.Gamma{shape = 2, scale = 3}, 4)
defer delete(draws)
fmt.printf("gamma(2, 3): mean %v, variance %v, draws %.4f\n", st.mean(st.Gamma{2, 3}), st.variance(st.Gamma{2, 3}), draws)
```

```
t(24): 97.5% quantile 2.063899, P(T > 2.8) = 0.00496459
chi-squared(10): P(X > 35) = 0.000124865, 95% quantile 18.307038
normal: P(Z > 9) = 1.12859e-19, 1 - cdf(9) = 0
normal: isf(1e-12) = 7.034484
binomial(50, 0.1): P(X = 5) = 0.184925, P(X <= 2) = 0.111729, P(X > 10) = 0.0093546, median 5
gamma(2, 3): mean 6, variance 18, draws [2.9354, 5.5070, 0.4553, 3.7636]
```

scipy gives the same values for every line but the draws, which depend on the generator.

The procedures of every distribution:

| Procedure | Result |
|---|---|
| `pdf(d, x)`, `logpdf(d, x)` | the density and its logarithm (continuous) |
| `pmf(d, k)`, `logpmf(d, k)` | the probability of the value k and its logarithm (discrete) |
| `cdf(d, x)` | P(X ≤ x) |
| `sf(d, x)` | P(X > x), the survival function, computed directly, not as 1 - cdf |
| `ppf(d, p)` | the quantile: the x with cdf(x) = p |
| `isf(d, q)` | the inverse of sf: the x with sf(x) = q |
| `mean(d)`, `variance(d)` | the moments |
| `random(d)`, `sample(d, n)` | one draw, or a slice of n draws |

Invalid parameters, such as a standard deviation of 0, and probabilities outside [0, 1] give NaN, as in scipy. The log densities are `-Inf` outside the support.

## 3. Continuous distributions

| Struct | Parameters | Support | scipy.stats equivalent |
|---|---|---|---|
| `Normal` | `mu`, `sigma` > 0; `STANDARD_NORMAL` is {0, 1} | all reals | `norm(mu, sigma)` |
| `Log_Normal` | `mu`, `sigma` > 0: the distribution of e^X for X normal | x > 0 | `lognorm(sigma, scale = e^mu)` |
| `Student_T` | `df` > 0 (`+Inf` gives the standard normal) | all reals | `t(df)` |
| `Chi_Squared` | `df` > 0 | x ≥ 0 | `chi2(df)` |
| `Fisher_F` | `d1`, `d2` > 0 | x ≥ 0 | `f(d1, d2)` |
| `Gamma` | `shape` > 0, `scale` > 0 (the rate is 1 / scale) | x ≥ 0 | `gamma(shape, scale = scale)` |
| `Beta` | `a`, `b` > 0 | 0 ≤ x ≤ 1 | `beta(a, b)` |
| `Exponential` | `rate` > 0 | x ≥ 0 | `expon(scale = 1 / rate)` |

Degrees of freedom and shapes may be any positive real, not only integers. A density that is infinite at the edge of its support, such as Gamma with shape below 1 at 0, is `+Inf` there.

Where a moment does not exist, `mean` and `variance` follow scipy: `+Inf` where the integral diverges (the mean of t with df ≤ 1, the variance of t with 1 < df ≤ 2, F's mean for d2 ≤ 2 and variance for d2 ≤ 4) and NaN where it is undefined (the variance of t with df ≤ 1).

## 4. Discrete distributions

| Struct | Parameters | Values | scipy.stats equivalent |
|---|---|---|---|
| `Binomial` | `n` ≥ 0 trials (int), success probability `p` | 0 .. n | `binom(n, p)` |
| `Poisson` | mean `lambda` ≥ 0 | 0, 1, 2, ... | `poisson(lambda)` |
| `Negative_Binomial` | `n` > 0 successes (any real), success probability `p` | the failures before the n-th success | `nbinom(n, p)` |

The value is passed as an `f64`. `pmf` of a non-integer is 0, and `cdf` and `sf` round the value down, so `cdf(d, 2.5)` is `cdf(d, 2)`. The probabilities use Loader's saddle-point form, as R does. It keeps full precision where the factorials of the textbook formula overflow or cancel, for example for a binomial with ten million trials.

`ppf(d, p)` is the smallest k with `cdf(k) ≥ p`, and `isf(d, q)` the smallest k with `sf(k) ≤ q`, both as in scipy. So `ppf(d, 0)` is -1, and `ppf(d, 1)` is n for the binomial and `+Inf` for the unbounded distributions. Quantiles are returned as `f64` for that reason.

## 5. Tails and quantiles

`1 - cdf(x)` loses its digits once cdf(x) is close to 1. In the example above, P(Z > 9) is 1.1e-19, while `1 - cdf(9)` gives 0. `sf` computes the upper tail directly, and `isf` inverts it, so p-values and critical values keep their full relative precision down to 1e-300:

```odin
st.sf(st.Chi_Squared{df = 5}, 100)      // 5.4e-20: a test statistic far out in the tail
st.isf(st.Student_T{df = 30}, 1e-300)   // 5.0e10
st.ppf(st.Normal{0, 1}, 1e-300)         // -37.047096
```

The same applies to quantiles near 1. Pass the upper-tail probability to `isf` rather than `ppf(d, 1 - q)`, because `1 - q` rounds when q is small.

A quantile beyond the range of `f64` is `±Inf`. An example is the 1e-300 quantile of t with 0.3 degrees of freedom, about -10^1000.

## 6. Random draws

`random(d)` draws one value and `sample(d, n)` a slice of n values: `f64` for the continuous distributions and `int` for the discrete ones. They use `context.random_generator`, so a seeded generator (`rand.create(seed)`, as in the example) gives the same draws on every run. Invalid parameters give NaN, or -1 for the discrete distributions.

| Distribution | Method |
|---|---|
| normal, log-normal | `core:math/rand`'s normal |
| gamma | Marsaglia and Tsang's method; shapes below 1 by multiplying with U^(1/shape) |
| chi-squared, beta, t, F | from gammas and normals |
| exponential | inversion |
| Poisson | multiplying uniforms for a mean below 10, Hörmann's PTRS above |
| binomial | inversion for n·min(p, 1 - p) below 10, Hörmann's BTRS above |
| negative binomial | a Poisson draw with a gamma-distributed mean |

A draw takes 12 to 130 ns.

## 7. Special functions

The distributions are built on these functions, which are public:

| Procedure | Function | scipy.special |
|---|---|---|
| `gamma_p(a, x)`, `gamma_q(a, x)`, `gamma_pq(a, x)` | the regularized incomplete gamma functions P and Q = 1 - P, each computed directly | `gammainc`, `gammaincc` |
| `gamma_p_inv(a, p)`, `gamma_q_inv(a, q)` | their inverses in x | `gammaincinv`, `gammainccinv` |
| `beta_inc(a, b, x)`, `beta_inc_c(a, b, x)` | the regularized incomplete beta function I_x(a, b) and 1 - I_x(a, b) | `betainc` |
| `beta_inc_pair(a, b, x, y)` | both at once, with y = 1 - x given exactly | |
| `beta_inc_inv(a, b, p)`, `beta_inc_inv_pair(a, b, p, q)` | the x with I_x(a, b) = p; the pair form also returns 1 - x | `betaincinv` |
| `log_beta(a, b)`, `log_gamma(x)` | log B(a, b) and log Γ(x) | `betaln`, `gammaln` |
| `expm1(x)`, `log1pmx(x)` | e^x - 1 and log(1 + x) - x, accurate near 0 | `expm1` |
| `erf_inv(y)`, `erfc_inv(y)` | the inverse error functions | `erfinv`, `erfcinv` |

The error function and the gamma function are Odin's (`math.erf`, `math.erfc`, `math.lgamma`).

The incomplete gamma function uses its series below x = a + 1 and a continued fraction above it. For small a and x, it uses the series for Q directly. The incomplete beta function uses a power series where b·x ≤ 1 and otherwise the continued fraction of Didonato and Morris, which Boost also uses. Each computes the tail that is below the mean directly and the other as its complement. For large parameters, the factors x^a e^-x / Γ(a) and x^a (1-x)^b / B(a, b) go through Stirling's series and log1p, so they keep their precision for parameters in the millions. The inverses are Newton iterations on the logarithm of the smaller tail probability, with bisection as a fallback.

The normal quantile is Acklam's approximation with one Halley step, accurate to the last bits.

## 8. Hypothesis tests: conventions

Every test returns a result struct and `ok`:

```odin
drug1 := []f64{0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0}
drug2 := []f64{1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4}

r, ok := st.t_test_paired(drug1, drug2)
if ok {
    fmt.printf("paired t: t = %.4f, df = %v, p = %.4g, mean difference %.2f, 95%% CI [%.4f, %.4f]\n",
        r.statistic, r.df, r.p_value, r.estimate, r.ci_low, r.ci_high)
}
welch, _ := st.t_test_two_sample(drug1, drug2)
fmt.printf("Welch t:  t = %.4f, df = %.3f, p = %.4g\n", welch.statistic, welch.df, welch.p_value)
wil, _ := st.wilcoxon_signed_rank(drug1, drug2)
fmt.printf("Wilcoxon: V = %v, p = %.4g (normal approximation: a zero difference and ties)\n", wil.statistic, wil.p_value)
sw, _ := st.shapiro_wilk(drug1)
fmt.printf("Shapiro-Wilk on drug 1: W = %.4f, p = %.4f\n", sw.statistic, sw.p_value)
```

```
paired t: t = -4.0621, df = 9, p = 0.002833, mean difference -1.58, 95% CI [-2.4599, -0.7001]
Welch t:  t = -1.8608, df = 17.776, p = 0.07939
Wilcoxon: V = 0, p = 0.007632 (normal approximation: a zero difference and ties)
Shapiro-Wilk on drug 1: W = 0.9258, p = 0.4079
```

The data are Student's 1908 sleep data: extra hours of sleep of ten patients under two drugs. R and scipy give the same values.

**`ok` is false** when the input cannot be tested at all: too few values, slices of different lengths where pairs are needed, a NaN or infinite value, a confidence level outside (0, 1), counts that do not fit (k > n). The result is then zero and must not be used.

**Undefined statistics** give `ok = true` and NaN or infinity, as in scipy. Examples are a correlation with a constant variable and a t-test on samples without spread. A few such cases, where scipy raises an error instead, give `ok = false`: Kruskal-Wallis and Shapiro-Wilk with all values equal, and an exact Kendall p-value with ties.

**Common parameters:**

| Parameter | Values | Meaning |
|---|---|---|
| `alternative` | `.Two_Sided` (default), `.Less`, `.Greater` | the alternative hypothesis. For two samples, `.Less` means the first tends to be smaller than the second; for one sample, smaller than the hypothesised value. |
| `confidence` | 0.95 by default | level of the confidence interval. One-sided alternatives give one-sided intervals, with one end at ±Inf (or 0 or 1 for a proportion). |
| `method` | `.Auto` (default), `.Exact`, `.Asymptotic` | for tests with a discrete null distribution: the exact distribution or the normal (or limiting) approximation. `.Auto` follows each test's rule, given with the test. |

**Result structs:**

| Struct | Fields | Returned by |
|---|---|---|
| `Test_Result` | `statistic`, `p_value`, `df`, `df2` (0 when unused) | most tests |
| `Mean_Test_Result` | `statistic`, `p_value`, `df`, `estimate`, `std_error`, `ci_low`, `ci_high` | t- and z-tests |
| `Binomial_Test_Result` | `estimate`, `p_value`, `ci_low`, `ci_high` | `binomial_test` |
| `Fisher_Result` | `odds_ratio`, `p_value` | `fisher_exact` |
| `Correlation_Result` | `coefficient`, `p_value`, `ci_low`, `ci_high` | `pearson`, `spearman`, `kendall` |
| `Anderson_Result` | `statistic`, `p_value`, `critical_values` | `anderson_darling` |

## 9. Means

| Procedure | Tests | Like |
|---|---|---|
| `t_test_one_sample(x, mu)` | mean of x = mu | scipy `ttest_1samp` |
| `t_test_paired(x, y, mu)` | mean of x[i] - y[i] = mu | scipy `ttest_rel` |
| `t_test_two_sample(x, y, mu, equal_var)` | mean(x) - mean(y) = mu | statsmodels `ttest_ind`, R `t.test` |
| `z_test_one_sample(x, mu)`, `z_test_two_sample(x, y, mu)` | the same with the normal distribution as reference | statsmodels `ztest` |

`t_test_two_sample` is Welch's test by default, which does not assume equal variances (R's default). `equal_var = true` gives Student's test with the pooled variance, scipy's default. The z-tests use the sample variance, pooled for two samples; they suit large samples. Each test needs two values per sample.

`estimate` is the mean or the difference of means, and the interval is around it. For data with a large common offset, such as prices near 10^6 that differ in the decimals, the means are taken relative to a value of the sample, so their difference keeps its digits.

## 10. Proportions and counts

```odin
party := []f64{762, 327, 468, 484, 239, 477} // rows: women, men; columns: three parties
expected: [6]f64
chi, _ := st.chi_square_contingency(party, cols = 3, expected = expected[:])
fmt.printf("independence: X2 = %.3f, df = %v, p = %.3g; expected %.1f\n", chi.statistic, chi.df, chi.p_value, expected)
tea, _ := st.fisher_exact({{3, 1}, {1, 3}}, .Greater)
fmt.printf("tea tasting: odds ratio %v, one-sided p = %.4f\n", tea.odds_ratio, tea.p_value)
b, _ := st.binomial_test(682, 925, p = 0.75)
fmt.printf("Mendel: 682 of 925, p = %.4f, 95%% CI [%.4f, %.4f]\n", b.p_value, b.ci_low, b.ci_high)
```

```
independence: X2 = 30.070, df = 2, p = 2.95e-07; expected [703.7, 319.6, 533.7, 542.3, 246.4, 411.3]
tea tasting: odds ratio 9, one-sided p = 0.2429
Mendel: 682 of 925, p = 0.3825, 95% CI [0.7077, 0.7654]
```

| Procedure | What it does | Like |
|---|---|---|
| `binomial_test(k, n, p)` | exact test of a success probability, with the Clopper-Pearson interval | scipy `binomtest`, R `binom.test` |
| `proportion_ci(k, n, confidence, method)` | confidence interval for a proportion | scipy `proportion_ci`, statsmodels `proportion_confint` |
| `proportion_z_test_one_sample(k, n, p0)` | z-test of one proportion | statsmodels `proportions_ztest` |
| `proportion_z_test_two_sample(k1, n1, k2, n2)` | z-test of two proportions, pooled | statsmodels `proportions_ztest` |
| `chi_square_test(observed, expected, ddof)` | goodness of fit | scipy `chisquare`, `power_divergence` |
| `chi_square_contingency(observed, cols)` | independence of rows and columns | scipy `chi2_contingency` |
| `fisher_exact({{a, b}, {c, d}})` | exact test for a 2 x 2 table | scipy `fisher_exact` |

**The two-sided binomial and Fisher p-values** add the probabilities of all outcomes no more likely than the observed one, as scipy and R do.

**`proportion_ci` methods:**
- `.Wilson`: the score interval, the default; good coverage for any n.
- `.Wilson_CC`: Wilson with continuity correction.
- `.Clopper_Pearson`: exact, from beta quantiles; conservative.
- `.Jeffreys`: beta quantiles of the Jeffreys posterior.
- `.Agresti_Coull`: the Wald interval on k + z²/2 successes in n + z² trials.
- `.Normal`: the Wald interval p ± z √(p(1 - p)/n), clipped to [0, 1]; poor for small n or p near 0 or 1.

**Chi-squared tests:**
- `chi_square_test` with `expected = nil` tests equal frequencies. Given expected counts must have the same total as the observed ones. `ddof` lowers the degrees of freedom k - 1 for parameters estimated from the data.
- `chi_square_contingency` takes the table row-major with its number of columns. `crosstab` (section 15) builds one from two DataFrame columns. A 2 x 2 table gets Yates' continuity correction unless `correction = false`. The optional `expected` slice receives the expected counts. Every row and column must have a positive total.
- Both take `divergence` for the Cressie-Read family: `.Pearson` (the usual statistic), `.Log_Likelihood` (the G-test), `.Freeman_Tukey`, `.Mod_Log_Likelihood`, `.Neyman`, `.Cressie_Read`.

**Fisher's test** reports the sample odds ratio ad / (bc), as scipy does; R reports the conditional maximum-likelihood estimate instead. `.Less` tests for an odds ratio below 1. A table with a zero row or column total gives p = 1 and a NaN odds ratio.

## 11. Rank tests

| Procedure | Tests | Auto is exact when |
|---|---|---|
| `mann_whitney_u(x, y)` | x and y come from the same distribution (Wilcoxon rank-sum) | a sample has at most 8 values and there are no ties |
| `wilcoxon_signed_rank(x, y)` | the differences x - y (or x alone, with `y = nil`) are symmetric about 0 | at most 50 differences, without zeros or ties |
| `kruskal_wallis(samples)` | k samples come from the same distribution | (chi-squared approximation) |

**The statistics:**
- `mann_whitney_u` reports U for x: the number of pairs with x[i] > y[j], ties counting 1/2. `.Greater` tests for x tending to be larger.
- `wilcoxon_signed_rank` reports the smaller rank sum for a two-sided test and the positive rank sum for a one-sided one.

The asymptotic p-values correct for ties. The Mann-Whitney approximation applies a continuity correction unless `continuity = false`; the Wilcoxon one applies it only with `correction = true`. This follows scipy's defaults; R's `wilcox.test` corrects by default, so its p-values differ slightly.

**Zeros in `wilcoxon_signed_rank`** are handled by `zero_method`:
- `.Wilcox` (default): drop them.
- `.Pratt`: rank them, then drop them.
- `.Zsplit`: split their ranks between the two sums.

**Exact distributions.** The exact Mann-Whitney distribution is counted in 128-bit integers while the number of arrangements fits (up to about 60 x 60 values), and in floating point beyond. The Wilcoxon and Kendall distributions are built in floating point for any size. All three agree with integer counts to about 1e-14. Asking for `method = .Exact` on data with ties ignores the ties.

## 12. Distribution and normality tests

| Procedure | Tests | Like |
|---|---|---|
| `ks_test_one_sample(x, dist)` | x comes from `dist`, any distribution of section 3 with fixed parameters | scipy `kstest` |
| `ks_test_one_sample_fn(x, cdf)` | the same, for a cdf procedure of your own | |
| `ks_test_two_sample(x, y)` | x and y come from the same distribution | scipy `ks_2samp` |
| `shapiro_wilk(x)` | normality, 3 or more values | R `shapiro.test`, scipy `shapiro` |
| `anderson_darling(x)` | normality, with critical values | scipy `anderson`, statsmodels `normal_ad` |
| `dagostino_pearson(x)` | normality from skewness and kurtosis together, 8 or more values | scipy `normaltest` |
| `skew_test(x)`, `kurtosis_test(x)` | normal skewness, normal kurtosis | scipy `skewtest`, `kurtosistest` |
| `jarque_bera(x)` | normality from skewness and kurtosis, for large samples | scipy `jarque_bera` |

**Kolmogorov-Smirnov.**
- The one-sample test uses the exact distribution of D for any n, by the algorithms scipy uses: Durbin's matrix, Pomeranz's recursion and the Pelz-Good series. `.Asymptotic` uses Kolmogorov's limit.
- The one-sided statistics are D+ = sup(F_n - F) for `.Greater` and D- for `.Less`. They have exact p-values from the Birnbaum-Tingey formula.
- The two-sample test is exact by lattice-path counting when both samples have at most 10,000 values (`.Auto`); larger samples use the asymptotic distribution. An exact p-value reports D rounded to the lattice of the sample sizes, as scipy does.
- The one-sided two-sample test stays exact for sizes where scipy's exact method overflows and falls back to an approximation (1000 x 1500, say).

**Shapiro-Wilk** uses Royston's algorithm (AS R94), computed in double precision throughout as R does. scipy's version runs in single precision, so its W differs from R's and this one by up to about 1e-6 (6e-5 for 5000 values), and its p-values by up to a few percent for large samples. Samples above 5000 are accepted; the p-value approximation was fitted up to 5000.

**Anderson-Darling** estimates the mean and standard deviation from the data. `critical_values[i]` belongs to the significance level `ANDERSON_LEVELS[i]` (15, 10, 5, 2.5 and 1 %): A² above it rejects normality at that level. `p_value` is D'Agostino and Stephens' approximation, as statsmodels' `normal_ad`.

## 13. Variances and correlation

| Procedure | Tests | Like |
|---|---|---|
| `levene(samples, center)` | equal variances; F statistic with `df` and `df2` | scipy `levene` |
| `bartlett(samples)` | equal variances, assuming normal data | scipy `bartlett` |
| `pearson(x, y)` | zero linear correlation; r with a Fisher-z interval | scipy `pearsonr` |
| `spearman(x, y)` | zero rank correlation (rho); t approximation | scipy `spearmanr` |
| `kendall(x, y)` | zero rank correlation (tau-b) | scipy `kendalltau` |

**Levene.** `center` is `.Median` by default (the Brown-Forsythe test, robust to non-normal data). `.Mean` gives Levene's original test. `.Trimmed` uses the trimmed mean and trims the samples by `proportion_to_cut` at each end. Bartlett's test is more powerful for normal data but misleading otherwise.

**Pearson** tests with the exact distribution of r for normal data. Its interval uses Fisher's z and needs 4 pairs. scipy 1.10.0's one-sided p-values are wrong for negative r; these are correct.

**Kendall** handles ties (tau-b). `.Auto` is exact without ties for up to 33 pairs; otherwise it uses the normal approximation with the variance corrected for ties. Its run time is O(n log n). Spearman and Kendall leave the interval NaN.

## 14. Multiple testing

`p_adjust(p, method)` returns the adjusted p-values in the original order; delete the slice. Hypothesis i is rejected at level α when its adjusted value is at most α. This is statsmodels' `multipletests` and R's `p.adjust`.

```odin
p := []f64{0.01, 0.04, 0.03, 0.005, 0.2}
holm, _ := st.p_adjust(p, .Holm)
defer delete(holm)
bh, _ := st.p_adjust(p, .BH)
defer delete(bh)
fmt.printf("Holm %.4g\nBH   %.4g\n", holm, bh)
```

```
Holm [0.04, 0.09, 0.09, 0.025, 0.2]
BH   [0.025, 0.05, 0.05, 0.025, 0.2]
```

| Method | Controls | Note |
|---|---|---|
| `.Bonferroni` | family-wise error | min(1, m p) |
| `.Sidak` | family-wise error | 1 - (1 - p)^m; exact for independent tests |
| `.Holm` (default) | family-wise error | step-down Bonferroni; never worse than Bonferroni |
| `.Holm_Sidak` | family-wise error | step-down Sidak |
| `.Hochberg` | family-wise error | step-up; independent or positively dependent tests |
| `.Hommel` | family-wise error | more powerful than Hochberg, same conditions |
| `.BH` | false discovery rate | Benjamini-Hochberg |
| `.BY` | false discovery rate | Benjamini-Yekutieli; any dependence |

## 15. Tests on DataFrame columns

The tests take slices. These helpers take them from a DataFrame:

| Procedure | Returns |
|---|---|
| `column_values(&df, name)` | the numbers of an Int, Float or Bool column, without nulls and NaN |
| `paired_values(&df, a, b)` | the rows where both columns hold a number, as two slices |
| `grouped_values(&df, value, group)` | `Groups{labels, samples}`: the values split by the labels of a String, Int or Bool column; free with `groups_destroy` |
| `crosstab(&df, row, col)` | `Table{counts, rows, cols, row_labels, col_labels}`: counts of label pairs; free with `table_destroy` |

Labels are sorted, numerically for Int columns. Each returns `ok = false` for a missing column or one of the wrong type. Free the slices of `column_values` and `paired_values` with `delete`.

```odin
df := w.dataframe_new()      // import w "wotan/core"
defer w.destroy_dataframe(&df)
w.add_column(&df, w.column_from_floats("fev", []f64{2.9, 3.0, 2.5, 2.6, 3.2, 3.8, 2.7, 4.0, 2.4, 2.8, 3.4, 3.7, 2.2, 2.0}))
w.add_column(&df, w.column_from_strings("group", []string{"normal", "normal", "normal", "normal", "normal",
    "airway", "airway", "airway", "airway", "asbestosis", "asbestosis", "asbestosis", "asbestosis", "asbestosis"}))
g, _ := st.grouped_values(&df, "fev", "group")
defer st.groups_destroy(&g)
kw, _ := st.kruskal_wallis(g.samples)
fmt.printf("groups %v: Kruskal-Wallis H = %.4f, df = %v, p = %.4f\n", g.labels, kw.statistic, kw.df, kw.p_value)
lev, _ := st.levene(g.samples)
fmt.printf("Levene (median): W = %.4f, p = %.4f\n", lev.statistic, lev.p_value)
```

```
groups ["airway", "asbestosis", "normal"]: Kruskal-Wallis H = 0.7714, df = 2, p = 0.6800
Levene (median): W = 4.4769, p = 0.0378
```

For a contingency table, pass `t.counts` and `t.cols` of a `crosstab` result to `chi_square_contingency`.

## 16. Accuracy and validation

### Distributions

5,281 values were compared with 60-digit references computed with mpmath:
- **What was compared:** every procedure of every distribution and the special functions.
- **Where:** in the centre and in the tails down to 1e-300.
- **Parameters:** t with 0.3 to a million degrees of freedom, Gamma with shapes from 0.01 to 10^5, Beta(10^5, 10^5), Binomial(10^7, 0.3) and Poisson(10^6).

The quantiles were checked by substituting them into the high-precision cdf. The largest relative errors, with scipy 1.10 scored the same way:

| | wotan | scipy 1.10 |
|---|---|---|
| normal, log-normal, exponential | 1.2e-13 | 5.7e-14 |
| t, chi-squared, gamma | 1.2e-13 | 7.3e-10 |
| F, beta | 2.9e-13 | 1.6e-10 |
| binomial, Poisson, negative binomial | 6.1e-13 | 6.6e-7 |
| quantiles (continuous) | 6.6e-14 | wrong in the far tails of beta and t; `Inf` for finite F quantiles |
| special functions | 4.7e-13 | 7.8e-7 for `betainc` with both parameters at 10^6; `betaincinv` wrong far in the tails |

The discrete quantiles were checked exactly: cdf(k - 1) < p ≤ cdf(k), computed at 60 digits. scipy fails this check far in the tails, for example the 1e-300 quantile of Poisson(10^6).

The random draws were tested against the cdf:
- **Continuous:** a Kolmogorov–Smirnov test on 200,000 draws.
- **Discrete:** a chi-squared test.
- **Coverage:** 36 parameter sets, including both sides of each method switch.

`stats_distributions_test` checks 122 of the reference values with a tolerance of 1e-12, the conventions above, the draws, and the helpers in other packages that are built on this one. `chi2_sf` in `analytics`, the t and F helpers of `analytics/ML`'s regression output and `norm_inv` in `finance` give the same values as `stats`.

### Hypothesis tests

5,051 results were compared on generated data: normal, skewed, heavy-tailed, with ties, and with a large common offset (values near 10^6 that differ in the decimals). Sample sizes ran from 2 to 20,000, and every alternative and option was covered. The references were:
- **40-digit computations (mpmath)** from exact rational means and moments, for the t- and z-tests, the moment-based normality tests, Pearson's r and the beta quantiles of the exact intervals.
- **A 40-digit port of R's Shapiro-Wilk algorithm.**
- **scipy 1.10 and statsmodels 0.13.5** where their computation is exact, for the rank tests, the exact Kolmogorov-Smirnov distributions, chi-squared, Fisher, Levene, Bartlett and `p_adjust`.

The largest relative errors, over every statistic, p-value, degrees of freedom and interval end:

| Tests | wotan | scipy / statsmodels on the same cases |
|---|---|---|
| t- and z-tests | 7.3e-13 | 1.2e-8 (the t quantile is off by 2e-11 at df = 1; offset data lose digits) |
| binomial test, proportion intervals and z-tests | 5.3e-14 | 6.4e-10 for scipy's exact interval |
| chi-squared, Fisher | 3.0e-14 | |
| Mann-Whitney, Wilcoxon, Kruskal-Wallis | 1.1e-14 | |
| Kolmogorov-Smirnov | 5.5e-13 | |
| Shapiro-Wilk | 1.3e-12 | 6e-5 in W and 8% in p for 5000 values (single precision) |
| Anderson-Darling | 3.5e-13 | statsmodels: 3e-8, and A² = Inf where it is 138.5 |
| skewness, kurtosis, D'Agostino-Pearson, Jarque-Bera | 3.5e-12 | 1.4e-8 |
| Levene, Bartlett | 2.9e-14 | |
| Pearson, Spearman, Kendall | 4.0e-13 | 4.7e-12; one-sided Pearson wrong for negative r |
| `p_adjust` | 5.1e-16 | |

The p-values were compared down to 1e-296. Below about 1e-200, a p-value from a z-score inherits about z² times the rounding of the statistic: skew_test's p-value of 2e-296 is off by 1.2e-11.

The exact null distributions were also checked against integer counts of the arrangements, permutations and lattice paths, at sizes beyond scipy's exact methods: Mann-Whitney 60 x 70, Wilcoxon with 80 differences, Kendall with 50 pairs and the one-sided two-sample Kolmogorov-Smirnov test with 1000 x 1500 values. They agree to 2.3e-14.

`stats_tests_test` checks the tests on published data with known answers (Student's sleep data, Mendel's peas, the tea-tasting and twin-conviction tables, the Hollander & Wolfe examples of R's documentation), the exact distributions above, the conventions of section 8 and the DataFrame helpers.

## 17. Quick reference

### Distributions

| Distribution | Struct | Procedures (`<prefix>_pdf`, `_cdf`, ...) |
|---|---|---|
| normal | `Normal{mu, sigma}`, `STANDARD_NORMAL` | `normal_` |
| log-normal | `Log_Normal{mu, sigma}` | `log_normal_` |
| Student's t | `Student_T{df}` | `student_t_` |
| chi-squared | `Chi_Squared{df}` | `chi_squared_` |
| F | `Fisher_F{d1, d2}` | `fisher_f_` |
| gamma | `Gamma{shape, scale}` | `gamma_` |
| beta | `Beta{a, b}` | `beta_` |
| exponential | `Exponential{rate}` | `exponential_` |
| binomial | `Binomial{n, p}` | `binomial_` |
| Poisson | `Poisson{lambda}` | `poisson_` |
| negative binomial | `Negative_Binomial{n, p}` | `negative_binomial_` |

| Group | Takes |
|---|---|
| `pdf`, `logpdf` | the continuous distributions |
| `pmf`, `logpmf` | the discrete distributions |
| `cdf`, `sf`, `ppf`, `isf`, `mean`, `variance`, `random`, `sample` | all |

Special functions: `gamma_p`, `gamma_q`, `gamma_pq`, `gamma_p_inv`, `gamma_q_inv`, `beta_inc`, `beta_inc_c`, `beta_inc_pair`, `beta_inc_inv`, `beta_inc_inv_pair`, `log_beta`, `log_gamma`, `expm1`, `log1pmx`, `erf_inv`, `erfc_inv`.

### Hypothesis tests

| Area | Procedures |
|---|---|
| means | `t_test_one_sample`, `t_test_paired`, `t_test_two_sample`, `z_test_one_sample`, `z_test_two_sample` |
| proportions | `binomial_test`, `proportion_ci`, `proportion_z_test_one_sample`, `proportion_z_test_two_sample` |
| counts | `chi_square_test`, `chi_square_contingency`, `fisher_exact` |
| ranks | `mann_whitney_u`, `wilcoxon_signed_rank`, `kruskal_wallis` |
| distributions | `ks_test_one_sample`, `ks_test_one_sample_fn`, `ks_test_two_sample` |
| normality | `shapiro_wilk`, `anderson_darling`, `dagostino_pearson`, `skew_test`, `kurtosis_test`, `jarque_bera` |
| variances | `levene`, `bartlett` |
| correlation | `pearson`, `spearman`, `kendall` |
| multiple testing | `p_adjust` |
| DataFrame input | `column_values`, `paired_values`, `grouped_values` / `groups_destroy`, `crosstab` / `table_destroy` |

Types: `Alternative`, `Method`, `Test_Result`, `Mean_Test_Result`, `Binomial_Test_Result`, `Fisher_Result`, `Correlation_Result`, `Anderson_Result` and `ANDERSON_LEVELS`, `CI_Method`, `Divergence`, `Zero_Method`, `Levene_Center`, `P_Adjust`, `Groups`, `Table`.
