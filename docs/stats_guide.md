# Statistics guide (`stats`)

The `stats` package holds classical statistics. This guide covers its probability distributions and the special functions behind them: densities, cumulative probabilities and their tails, quantiles, moments and random draws for eleven distributions, at the accuracy of scipy.stats or better.

## Contents

1. [Setup](#1-setup)
2. [Distributions as values](#2-distributions-as-values)
3. [Continuous distributions](#3-continuous-distributions)
4. [Discrete distributions](#4-discrete-distributions)
5. [Tails and quantiles](#5-tails-and-quantiles)
6. [Random draws](#6-random-draws)
7. [Special functions](#7-special-functions)
8. [Accuracy and validation](#8-accuracy-and-validation)
9. [Quick reference](#9-quick-reference)

## 1. Setup

```odin
import st "wotan/stats"
import "core:fmt"
```

The package depends only on Odin's `core:math`. Nothing in it allocates except `sample`, which returns a slice to `delete`.

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

## 8. Accuracy and validation

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

## 9. Quick reference

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
