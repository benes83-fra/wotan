# The `finance` package, part 1: options and simulation

`wotan/finance` is the quantitative finance package of Wotan. It is large, so its guide comes in parts. This first part covers equity option pricing and the simulation tools behind it:
- Black–Scholes prices and Greeks in closed form, implied volatility, and a differentiable Black–Scholes price for the [`tensor`](tensor_guide.md) autograd engine;
- a GARCH volatility term structure for option pricing;
- Monte Carlo pricers for Asian, barrier, lookback and basket options with pathwise Greeks;
- American options with binomial and trinomial trees, finite differences and Longstaff–Schwartz Monte Carlo;
- Fourier (COS) pricing for Heston, Merton jump diffusion, Bates and any model with a characteristic function;
- Monte Carlo building blocks: path generators for GBM, Heston, Merton and Hull–White, payoff helpers, a regression step, control variates and common-random-number Greeks;
- quasi-Monte Carlo with Sobol' sequences and the Brownian bridge;
- fractional Brownian motion, fractional Black–Scholes and the rough Bergomi model.

Every example was compiled and run against the current code, and the output shown is the real output. The results agree with QuantLib 1.44, SciPy and NumPy; section 12 lists the comparisons. A debug build with a tracking allocator reports no leaks for any of the examples.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [Black–Scholes in closed form](#2-blackscholes-in-closed-form)
3. [Implied volatility](#3-implied-volatility)
4. [Differentiable Black–Scholes and GARCH volatility](#4-differentiable-blackscholes-and-garch-volatility)
5. [Monte Carlo exotics with pathwise Greeks](#5-monte-carlo-exotics-with-pathwise-greeks)
6. [American options: trees and finite differences](#6-american-options-trees-and-finite-differences)
7. [American options: Longstaff–Schwartz](#7-american-options-longstaffschwartz)
8. [Fourier pricing (COS)](#8-fourier-pricing-cos)
9. [Monte Carlo building blocks](#9-monte-carlo-building-blocks)
10. [Quasi-Monte Carlo](#10-quasi-monte-carlo)
11. [Fractional Brownian motion and rough volatility](#11-fractional-brownian-motion-and-rough-volatility)
12. [Validation](#12-validation)
13. [Quick reference](#13-quick-reference)

## 1. Setup and conventions

```odin
import fin "wotan/finance"
import l "wotan/linalg"   // matrices, for fBm factors and autograd inputs
import t "wotan/tensor"   // only for the differentiable Black-Scholes price
import "core:math/rand"
```

Inside the Wotan repository the path is relative, for example `import fin "../wotan/finance"` from `tests/`.

**Inputs.** Prices take the spot `S`, strike `K`, time to expiry `T` in years, the continuously compounded risk-free rate `r` and the volatility `sigma`, all as `f64`. An `OptionType` is `.Call` or `.Put`. Only the trees (section 6) take a continuous dividend yield `q`; everything else in this part assumes no dividends.

**Greek units.** There are two conventions; check the result type before you compare numbers:

| Result type | Returned by | vega | theta | rho |
|---|---|---|---|---|
| `Greeks` | `compute_greeks`, `price_and_greeks`, `garch_price_and_greeks` | per 1 % of volatility | per calendar day (÷ 365) | per 1 % of rate |
| `AmericanOptionResult`, `FiniteDifferenceResult`, `LSM_Result`, `VR_LSM_Result` | sections 6 and 7 | per unit of volatility | per year | — |
| Monte Carlo exotics | section 5 | per unit of volatility | — | — |

Theta is always the change of value as time passes (negative for most long options). Delta and gamma are always ∂V/∂S and ∂²V/∂S².

**Random numbers.** The Monte Carlo procedures draw from Odin's `core:math/rand`, so `rand.reset(seed)` makes a run reproducible. The examples reset the generator before each Monte Carlo call.

**Memory.** Procedures that return slices or matrices take an `allocator`, and the caller frees the result. The `mc_generate_*` path generators (section 9) have no default allocator; pass one explicitly. A few procedures use `context.temp_allocator` for scratch space, so call `free_all(context.temp_allocator)` now and then in long loops.

## 2. Black–Scholes in closed form

`compute_greeks` returns the price and all first-order Greeks plus gamma:

```odin
g := fin.compute_greeks(100, 110, 0.5, 0.03, 0.25, .Call) // S, K, T, r, sigma, type
fmt.printf("price %.6f delta %.6f gamma %.6f\n", g.price, g.delta, g.gamma)
fmt.printf("vega %.6f (per 1%%) theta %.6f (per day) rho %.6f (per 1%%)\n", g.vega, g.theta, g.rho)

// put-call parity: C − P = S − K e^{−rT}
p := fin.compute_greeks(100, 110, 0.5, 0.03, 0.25, .Put)
fmt.printf("C - P = %.12f, S - K e^-rT = %.12f\n", g.price - p.price, 100 - 110 * math.exp(-0.03 * 0.5))

price, greeks := fin.price_and_greeks(100, 110, 0.5, 0.03, 0.25, .Put)
fmt.printf("put %.6f delta %.6f\n", price, greeks.delta)

fmt.println(fin.compute_greeks(100, 90, 0, 0.03, 0.25, .Call)) // expired
fmt.println(fin.compute_greeks(100, 90, 1, 0.03, 0, .Put))     // no volatility
```

```
price 3.898551 delta 0.357214 gamma 0.021106
vega 0.263828 (per 1%) theta -0.020686 (per day) rho 0.159114 (per 1%)
C - P = -8.362313356337, S - K e^-rT = -8.362313356337
put 12.260865 delta -0.642786
Greeks{price = 10, delta = 1, gamma = 0, vega = 0, theta = 0, rho = 0}
Greeks{price = 0, delta = 0, gamma = 0, vega = 0, theta = 0, rho = 0}
```

`price_and_greeks` is the same calculation, with the price returned separately.

The edge cases are well defined:
- **`T ≤ 0`:** the option is worth its payoff.
- **`sigma ≤ 0`:** the forward is certain, so the option is worth its discounted payoff at the forward. The put above is out of the money at the forward (100 > 90·e^−0.03), so it is worth 0.

The normal CDF is computed with `erfc`, so prices stay accurate far out of the money. A call struck at 300 with the spot at 100 is worth 6.0e-28, and its Greeks are just as small.

## 3. Implied volatility

`implied_volatility(market_price, S, K, T, r, type)` returns `(sigma, converged, iterations)`:

```odin
market := 4.20
sigma, ok, iters := fin.implied_volatility(market, 100, 110, 0.5, 0.03, .Call)
fmt.printf("sigma %.10f ok %v after %d iterations\n", sigma, ok, iters)
fmt.printf("check: %.10f\n", fin.compute_greeks(100, 110, 0.5, 0.03, sigma, .Call).price)

// a far out-of-the-money call priced at 6e-28 still inverts
tiny := fin.compute_greeks(100, 300, 0.25, 0.02, 0.2, .Call).price
s2, ok2, _ := fin.implied_volatility(tiny, 100, 300, 0.25, 0.02, .Call)
fmt.printf("price %.3e -> sigma %.12f %v\n", tiny, s2, ok2)

// below intrinsic value: no implied volatility
s3, ok3, _ := fin.implied_volatility(9.0, 100, 90, 0.5, 0.03, .Call)
fmt.println(s3, ok3)
```

```
sigma 0.2613767831 ok true after 8 iterations
check: 4.1999999999
price 6.012e-28 -> sigma 0.200000000000 true
0 false
```

How the solver works:
- It takes Newton steps on the logarithm of the price, safeguarded by bisection inside a bracket.
- It solves on the out-of-the-money option of the pair, whose price is the time value of the in-the-money one (put–call parity). Deep in-the-money options therefore don't lose precision to their intrinsic value.
- It recovers volatilities from 1 % to more than 500 %. It handles maturities from days to 30 years, and negative rates.

It returns `(0, false, 0)` for prices outside the no-arbitrage bounds:
- below the intrinsic value `max(S − K·e^−rT, 0)` for a call (`max(K·e^−rT − S, 0)` for a put);
- at or above `S` for a call (`K·e^−rT` for a put).

It also returns `(0, false, 0)` when the time value is lost to rounding. An example is a call struck at 60 with σ = 1 %, whose price equals S − K·e^−rT in double precision. A NaN price also gives `false`.

The optional `max_iter` (50) and `tol` (1e-8, the price error) come after an unused `allocator` parameter, so pass them by name: `implied_volatility(p, S, K, T, r, .Call, max_iter = 100)`.

## 4. Differentiable Black–Scholes and GARCH volatility

`black_scholes_price` builds the Black–Scholes formula from tensor operations, so `tensor_backward` gives the derivatives with respect to every input that requires a gradient:

```odin
scalar :: proc(v: f64, grad: bool) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, 1)
	m.data[0] = v
	return t.tensor_new(m, grad)
}
S := scalar(100, true)
K := scalar(110, false)
T := scalar(0.5, true)
r := scalar(0.03, true)
sigma := scalar(0.25, true)
defer for x in ([]^t.Tensor{S, K, T, r, sigma}) {t.tensor_free(x)}

price := fin.black_scholes_price(S, K, T, r, sigma, .Call, context.allocator)
t.tensor_backward(price)
fmt.printf("price %.6f\n", price.data.data[0])
fmt.printf("dV/dS %.6f  dV/dT %.6f  dV/dr %.6f  dV/dsigma %.6f\n",
	S.grad.data[0], T.grad.data[0], r.grad.data[0], sigma.grad.data[0])
t.tensor_free_graph(price)
```

```
price 3.898551
dV/dS 0.357214  dV/dT 7.550380  dV/dr 15.911444  dV/dsigma 26.382774
```

These are raw derivatives, so they differ in units from section 2:
- vega here is 100 × the per-1 % vega;
- ∂V/∂T is −365 × the per-day theta.

The inputs may hold many options at once. Pass tensors of equal shape, for example n × 1, and every option is priced elementwise. To get per-option Greeks from one backward pass, sum the prices first with `tensor_sum` and call `tensor_backward` on the sum. Since each price depends only on its own inputs, S.grad then holds each option's delta.

`tensor_free_graph(price)` frees every intermediate, including those computed only from inputs without a gradient (such as e^−rT when r is fixed). With no gradient at all, the price is marked `owned_by_graph`, so either `tensor_free` or `tensor_free_graph` frees it. The inputs belong to you.

`tensor_smooth_max(a, b, allocator)` is a differentiable approximation of `max(a, b)`: (a + b + √((a − b)² + 10⁻⁶))/2. It is at most 5·10⁻⁴ above the true maximum. It is meant for payoffs such as `max(S − K, 0)` inside autograd code.

**GARCH volatility.** `garch_term_structure_vol(omega, alpha, beta, current_var, horizon_days)` averages the GARCH(1,1) variance forecast over the next `horizon_days` trading days and annualises it with 252 days:
- `current_var` is the one-step-ahead daily variance σ²ₜ₊₁;
- `omega`, `alpha` and `beta` are the daily GARCH parameters, for example from wotan's GARCH fit.

`garch_price_and_greeks` turns the maturity into trading days (`round(T·252)`), takes that volatility and prices with `compute_greeks`:

```odin
omega, alpha, beta := 2e-6, 0.08, 0.90
var_today := 3e-4 // tomorrow's variance forecast, daily
for days in ([]int{1, 21, 63, 252}) {
	fmt.printf("%3d days: %.4f\n", days, fin.garch_term_structure_vol(omega, alpha, beta, var_today, days))
}
price, g := fin.garch_price_and_greeks(100, 100, 0.25, 0.03, omega, alpha, beta, var_today, .Call)
fmt.printf("3m ATM call %.4f delta %.4f\n", price, g.delta)
```

```
  1 days: 0.2750
 21 days: 0.2582
 63 days: 0.2324
252 days: 0.1875
3m ATM call 4.9985 delta 0.5488
```

Today's variance (27.5 % annualised) is above the long-run level √(252·ω/(1 − α − β)) = 15.9 %, so longer maturities get lower volatilities. The forecast is the exact recursion vₕ₊₁ = ω + (α + β)·vₕ, so it also works for integrated GARCH (α + β = 1).

## 5. Monte Carlo exotics with pathwise Greeks

Five pricers simulate geometric Brownian motion on the grid tᵢ = i·T/n_steps. Each returns the price, delta (∂V/∂S₀) and vega (∂V/∂σ, per unit):

| Procedure | Payoff |
|---|---|
| `monte_carlo_asian_option` | max(mean(S₁..Sₙ) − K, 0) |
| `monte_carlo_barrier_call_bs` | max(Sₙ − K, 0), zero if any Sᵢ reaches `barrier` (up-and-out) |
| `monte_carlo_barrier_asian_option` | the Asian payoff, zero if any Sᵢ exceeds `barrier` |
| `monte_carlo_lookback_call_option` | max(max(S₀, S₁..Sₙ) − K, 0) |
| `monte_carlo_basket_option` | max(w₁·mean(S¹) + w₂·mean(S²) − K, 0), two correlated assets; returns price, delta₁, delta₂, vega₁, vega₂ |

```odin
rand.reset(1)
p, d, v := fin.monte_carlo_asian_option(100, 100, 1, 0.05, 0.2, 100_000, 12) // S0, K, T, r, sigma, paths, steps
fmt.printf("asian         %.4f delta %.4f vega %.4f\n", p, d, v)
p, d, v = fin.monte_carlo_barrier_call_bs(100, 100, 1, 0.05, 0.2, 130, 100_000, 52) // barrier 130
fmt.printf("up-and-out    %.4f delta %.4f vega %.4f\n", p, d, v)
p, d, v = fin.monte_carlo_barrier_asian_option(100, 100, 1, 0.05, 0.2, 130, 100_000, 52)
fmt.printf("barrier asian %.4f delta %.4f vega %.4f\n", p, d, v)
p, d, v = fin.monte_carlo_lookback_call_option(100, 100, 1, 0.05, 0.2, 100_000, 52)
fmt.printf("lookback      %.4f delta %.4f vega %.4f\n", p, d, v)
// S1, S2, K, T, r, sigma1, sigma2, correlation, w1, w2, paths, steps
bp, d1, d2, v1, v2 := fin.monte_carlo_basket_option(100, 50, 75, 1, 0.05, 0.2, 0.35, 0.6, 0.5, 1.0, 100_000, 12)
fmt.printf("basket        %.4f delta %.4f %.4f vega %.4f %.4f\n", bp, d1, d2, v1, v2)
```

```
asian         6.1478 delta 0.5923 vega 23.0288
up-and-out    3.8014 delta 0.0643 vega -25.4092
barrier asian 2.1088 delta 0.1611 vega -16.0612
lookback      17.4366 delta 1.0943 vega 77.1741
basket        26.4925 delta 0.4815 0.9655 vega 1.0939 0.8010
```

The same options priced by NumPy with 2 million paths:

| Option | wotan (100k paths) | NumPy (2M paths) |
|---|---|---|
| Asian | 6.1478 | 6.1636 ± 0.0060 |
| up-and-out | 3.8014 | 3.8225 ± 0.0047 |
| barrier Asian | 2.1088 | 2.1158 ± 0.0026 |
| lookback | 17.4366 | 17.4271 ± 0.0108 |
| basket | 26.4925 | 26.4646 ± 0.0106 |

All five agree within about one standard error of wotan's 100k paths.

**How the Greeks are computed.** They are exact pathwise derivatives of each simulated payoff, accumulated path by path, so memory use doesn't grow with the number of paths or steps.

The barrier options use *one-step survival* (Glasserman & Staum 2001). Every step is drawn conditional on staying below the barrier, and the path carries the product of the survival probabilities as its weight. This has two advantages over checking a 0/1 knock-out indicator:
- **Lower variance:** the estimate is unbiased for the discretely monitored barrier, and no path is wasted on a knock-out.
- **Correct Greeks:** the weighted payoff is smooth in S₀ and σ, so the delta and vega include the effect of knock-outs, which pathwise derivatives of an indicator miss. That is why the up-and-out vega above is negative.

Monitoring is discrete, on the simulation grid (S₁..Sₙ), so the price depends on `n_steps`. Use the number of real monitoring dates. A barrier at or below the strike makes an up-and-out call worthless, and the pricers return 0.

## 6. American options: trees and finite differences

Four lattice methods price American options. Each returns the price, delta, gamma and theta (per year):

| Procedure | Method | Dividend yield | Extra field |
|---|---|---|---|
| `american_option_binomial(S, K, T, r, sigma, q, type, n_steps = 1000)` | Cox–Ross–Rubinstein tree | yes | `early_exercise_premium` |
| `american_option_trinomial(…)` | trinomial tree in log price | yes | `early_exercise_premium` |
| `finite_difference_american(S, K, T, r, sigma, type, n_space = 200, n_time = 200)` | Crank–Nicolson, Brennan–Schwartz | no | — |
| `crank_nicolson_european(…)` | the same grid without early exercise | no | — |

There are wrappers that fix the type:
- `american_call_binomial`, `american_put_binomial`, `american_call_trinomial` and `american_put_trinomial`, with `q = 0` as the default;
- `crank_nicolson_call`, `crank_nicolson_put`, `fd_american_call` and `fd_american_put`.

```odin
S, K, T, r, sigma, q := 100.0, 105.0, 1.0, 0.05, 0.3, 0.0
b := fin.american_option_binomial(S, K, T, r, sigma, q, .Put, 2000)
tr := fin.american_option_trinomial(S, K, T, r, sigma, q, .Put, 2000)
fd := fin.finite_difference_american(S, K, T, r, sigma, .Put, 400, 400)
eu := fin.crank_nicolson_european(S, K, T, r, sigma, .Put, 400, 400)
fmt.printf("binomial  %.6f delta %.6f gamma %.6f theta %.6f premium %.6f\n", b.price, b.delta, b.gamma, b.theta, b.early_exercise_premium)
fmt.printf("trinomial %.6f delta %.6f gamma %.6f theta %.6f premium %.6f\n", tr.price, tr.delta, tr.gamma, tr.theta, tr.early_exercise_premium)
fmt.printf("FD        %.6f delta %.6f gamma %.6f theta %.6f\n", fd.price, fd.delta, fd.gamma, fd.theta)
fmt.printf("CN euro   %.6f   closed form %.6f\n", eu.price, fin.compute_greeks(S, K, T, r, sigma, .Put).price)

// a call on a dividend payer can be worth exercising early
c := fin.american_call_binomial(100, 95, 1, 0.03, 0.25, q = 0.06)
fmt.printf("american call with q = 6%%: %.6f premium %.6f\n", c.price, c.early_exercise_premium)
```

```
binomial  12.571428 delta -0.478326 gamma 0.015326 theta -3.876714 premium 0.714291
trinomial 12.569905 delta -0.478317 gamma 0.015326 theta -3.877548 premium 0.713998
FD        12.569357 delta -0.478312 gamma 0.015326 theta -3.880065
CN euro   11.855762   closed form 11.855971
american call with q = 6%: 10.818777 premium 0.522659
```

QuantLib's finite-difference engine on an 800 × 800 grid gives 12.569353 for the same put.

The early exercise premium is the American price minus the European price on the same tree, so the tree's discretisation error mostly cancels.

**Choosing a method:**
- **Binomial tree:** the price oscillates as `n_steps` changes. With 2000 steps the put above is 0.002 off.
- **Trinomial tree:** smoother. With 2000 steps it is 0.0006 off.
- **Finite differences:** the most accurate for the same work. With a 400 × 400 grid the put is 4e-6 off. They use a log-price grid centred on the spot, so the price needs no interpolation, and two implicit half steps at the start (Rannacher) damp the oscillations Crank–Nicolson produces from the payoff kink. Use them when there are no dividends.

All the trees run in O(n_steps) memory.

The Greeks come from the nodes next to the spot at the first two levels (trees) or from the grid (finite differences). They need at least 2 steps; fewer are raised to 2. Expired options (`T ≤ 0`) return the payoff.

## 7. American options: Longstaff–Schwartz

Longstaff–Schwartz Monte Carlo decides exercise by regressing the continuation value on the spot. Exercise is checked at `n_exercise_dates` equally spaced dates; the last date is expiry. With few dates the option is Bermudan, and with as many dates as steps it approximates the American option.

| Procedure | Option | Result |
|---|---|---|
| `lsm_american_vanilla(S_0, K, T, r, sigma, type, n_paths = 10000, n_steps = 50, n_exercise_dates = 10, poly_degree = 3)` | call or put | `LSM_Result`: price, delta, gamma, vega, theta |
| `lsm_american_call`, `lsm_american_put` | the same with the type fixed | `LSM_Result` |
| `lsm_american_asian_call(…)` | exercise pays the running average (S₀ included) minus K | `LSM_Result` |
| `lsm_american_vanilla_reduced(…)` | antithetic paths plus the European option as control variate | `VR_LSM_Result`, with `variance_reduction` |

```odin
rand.reset(1)
a := fin.lsm_american_put(100, 105, 1, 0.05, 0.3, n_paths = 50_000, n_steps = 50, n_exercise_dates = 50)
fmt.printf("plain   %.4f delta %.4f gamma %.4f vega %.4f theta %.4f\n", a.price, a.delta, a.gamma, a.vega, a.theta)
rand.reset(1)
vr := fin.lsm_american_vanilla_reduced(100, 105, 1, 0.05, 0.3, .Put, n_paths = 50_000, n_steps = 50, n_exercise_dates = 50)
fmt.printf("reduced %.4f delta %.4f variance reduction %.1fx\n", vr.price, vr.delta, vr.variance_reduction)
rand.reset(1)
as := fin.lsm_american_asian_call(100, 100, 1, 0.05, 0.3, n_paths = 50_000, n_exercise_dates = 10, poly_degree = 2)
fmt.printf("asian   %.4f delta %.4f\n", as.price, as.delta)
```

```
plain   12.5982 delta -0.4868 gamma 0.0134 vega 39.0070 theta -4.1678
reduced 12.5455 delta -0.4722 variance reduction 5.2x
asian   8.5992 delta 0.6607
```

The reference is QuantLib's finite-difference price of the Bermudan put with the same 50 dates, 12.5550. Over 12 seeds, plain LSM averages 12.561 with a standard deviation of 0.062; the reduced version has about half that standard deviation. `variance_reduction` is the variance of plain Monte Carlo over that of the reduced estimator, for the same number of paths.

How it works:
- **Regression:** only in-the-money paths are used. The basis is powers of S/K − 1 up to `poly_degree`; the Asian version uses all monomials in S/K − 1 and A/K − 1, because its continuation value depends on both the spot and the running average. The least-squares problem is solved by QR.
- **Exercise at t = 0:** the price is never below the intrinsic value.
- **Greeks:** central differences on the same normals, with the exercise rule of the first run held fixed. By the envelope theorem the first derivatives are the same, and the fixed rule avoids the noise of re-fitting every bumped run. Delta uses ±1 % of spot and gamma ±5 %. Gamma remains the noisiest number.

The LSM pricers take no dividend yield.

## 8. Fourier pricing (COS)

The COS method of Fang & Oosterlee (2008) prices European options from the characteristic function of ln S_T. It prices many strikes from one evaluation of the characteristic function per term, at close to machine precision.

The model procedures write one price per strike into `out`:

| Procedure | Model | Parameters |
|---|---|---|
| `heston_prices_cos(S, strikes, T, r, params, type, out)` | Heston | `Heston_Params{v0, kappa, theta, sigma, rho}` |
| `merton_prices_cos(…)` | Merton jump diffusion (λ = 0 is Black–Scholes) | `MJD_Params{sigma, lambda, mu_j, sigma_j}` |
| `bates_prices_cos(…)` | Heston plus Merton jumps | `Bates_Params{heston, lambda, mu_j, sigma_j}` |

```odin
strikes := []f64{80, 90, 100, 110, 120}
prices: [5]f64
hp := fin.Heston_Params{v0 = 0.04, kappa = 1.5, theta = 0.06, sigma = 0.7, rho = -0.7}
fin.heston_prices_cos(100, strikes, 1, 0.03, hp, .Call, prices[:])
fmt.printf("heston %.6f\n", prices)
fin.merton_prices_cos(100, strikes, 1, 0.03, fin.MJD_Params{sigma = 0.2, lambda = 0.5, mu_j = -0.1, sigma_j = 0.15}, .Call, prices[:])
fmt.printf("merton %.6f\n", prices)
fin.bates_prices_cos(100, strikes, 1, 0.03, fin.Bates_Params{heston = hp, lambda = 0.5, mu_j = -0.1, sigma_j = 0.15}, .Call, prices[:])
fmt.printf("bates  %.6f\n", prices)

// implied volatilities of the Heston smile
fin.heston_prices_cos(100, strikes, 1, 0.03, hp, .Call, prices[:])
for K, k in strikes {
	iv, _, _ := fin.implied_volatility(prices[k], 100, K, 1, 0.03, .Call)
	fmt.printf("K %.0f: %.4f  ", K, iv)
}
```

```
heston [24.368262, 16.279692, 9.291294, 4.111774, 1.359743]
merton [23.997020, 16.530187, 10.626904, 6.392518, 3.623728]
bates  [24.917240, 17.255752, 10.727417, 5.682036, 2.460124]
K 80: 0.2604  K 90: 0.2280  K 100: 0.1968  K 110: 0.1693  K 120: 0.1520
```

The Heston prices agree with QuantLib's `AnalyticHestonEngine` to 12 significant digits from 7 days to 10 years, and the Bates prices with its `BatesEngine`. The negative ρ produces the downward-sloping skew.

**Your own model.** `cos_option_prices(cf, data, strikes, discount, type, out)` prices any model whose characteristic function you supply. The function is E[exp(iu·ln S_T)] under the pricing measure; `data` is passed through to it. It must accept complex `u` with −1 ≤ Im(u) ≤ 0, because the forward and the cumulants that set the truncation range are evaluated there.

As a check, the Black–Scholes characteristic function reproduces the closed form:

```odin
BS_CF :: struct {S, r, T, sigma: f64}
bs_cf :: proc(u: complex128, data: rawptr) -> complex128 {
	c := (^BS_CF)(data)
	i := complex(0.0, 1.0)
	m := math.ln(c.S) + (c.r - 0.5 * c.sigma * c.sigma) * c.T
	return fin.cmath_exp(i * u * complex(m, 0) - complex(0.5 * c.sigma * c.sigma * c.T, 0) * u * u)
}

c := BS_CF{S = 100, r = 0.03, T = 1, sigma = 0.25}
fin.cos_option_prices(bs_cf, &c, strikes, math.exp(-0.03), .Put, prices[:])
fmt.printf("custom %.12f\nexact  %.12f\n", prices[2], fin.compute_greeks(100, 100, 1, 0.03, 0.25, .Put).price)
```

```
custom 8.393030179994
exact  8.393030179994
```

The optional parameters are:
- `L = 20`: the half-width of the truncation range in standard deviations;
- `min_terms = 64` and `max_terms = 65536`: the number of terms doubles until the characteristic function has decayed below 1e-16.

Puts are computed directly and calls by put–call parity with the model's forward φ(−i), so the two always satisfy parity. Very far out-of-the-money prices are accurate to about 1e-13 in absolute terms, not relative.

## 9. Monte Carlo building blocks

For payoffs that the ready-made pricers don't cover, `monte_carlo.odin` provides the pieces to write your own. Paths are flat slices, path-major: `paths[p · (n_steps + 1) + step]`, with step 0 the initial value. Every generator takes its random numbers from the caller, so you control seeding, antithetic pairs and quasi-random points (section 10).

| Procedure | Produces | Random numbers needed |
|---|---|---|
| `mc_generate_gbm_paths(S_0, r, sigma, T, n_paths, n_steps, z, allocator)` | GBM, exact on the grid | n_paths · n_steps normals |
| `mc_generate_gbm_paths_antithetic(…)` | paths ⌈n/2⌉.. use the negated normals of paths 0.. | ⌈n_paths/2⌉ · n_steps normals |
| `mc_generate_heston_paths(S_0, r, T, params, n_paths, n_steps, z, allocator)` | Heston S and v, full-truncation Euler | 2 · n_paths · n_steps normals |
| `mc_generate_mjd_paths(S_0, r, T, params, n_paths, n_steps, z, u, allocator)` | Merton jump diffusion, exact on the grid | 2 · n_paths · n_steps normals, n_paths · n_steps uniforms |
| `mc_generate_hw1f_paths(r0, T, HW_Params{a, sigma}, n_paths, n_steps, z, allocator)` | Hull–White short rate fitted to the flat curve r0, exact | n_paths · n_steps normals |
| `mc_generate_normals(n, allocator)`, `mc_generate_uniforms(n, allocator)` | i.i.d. draws from `core:math/rand` | — |

Payoff helpers read one path:
- `mc_compute_asian_average(paths, p, n_steps)` averages S₁..Sₙ, and `mc_compute_asian_average_to_step(paths, p, step, n_steps)` averages S₀..S_step.
- `mc_check_barrier_breached(paths, p, n_steps, barrier, is_up)` and `mc_check_double_barrier_breached(paths, p, n_steps, lower, upper)` check whether a barrier is reached at any of S₁..Sₙ.
- `mc_compute_lookback_max` and `mc_compute_lookback_min` include S₀.

Three tools reduce noise or give Greeks:
- **`mc_apply_control_variate(payoffs, controls, control_exact, n)`** returns the adjusted price and the variance reduction factor, using the optimal coefficient β = Cov/Var.
- **`mc_compute_crn_delta_gamma(S_0, price_fn)`** returns central-difference delta and gamma with a bump of 1 % of spot. `mc_compute_crn_vega(S_0, sigma, price_fn)` returns vega the same way. The `price_fn` must reuse the same random numbers on every call (common random numbers). Odin procedures don't capture variables, so keep those numbers in a global, as below.
- **`mc_lsm_regression_step`** is one Longstaff–Schwartz backward step on spot values at the exercise dates, for writing your own early-exercise pricer.

```odin
rand.reset(11)
n_paths, n_steps := 50_000, 12
z := fin.mc_generate_normals(n_paths * n_steps, context.allocator)
defer delete(z)
paths := fin.mc_generate_gbm_paths(100, 0.05, 0.2, 1, n_paths, n_steps, z, context.allocator)
defer delete(paths)

// arithmetic Asian call, with the European call on the same paths as control variate
payoff := make([]f64, n_paths)
control := make([]f64, n_paths)
defer {delete(payoff); delete(control)}
disc := math.exp(-0.05)
for p in 0 ..< n_paths {
	payoff[p] = max(fin.mc_compute_asian_average(paths, p, n_steps) - 100, 0) * disc
	control[p] = max(paths[p * (n_steps + 1) + n_steps] - 100, 0) * disc
}
exact := fin.compute_greeks(100, 100, 1, 0.05, 0.2, .Call).price
adj, ratio := fin.mc_apply_control_variate(payoff, control, exact, n_paths)
plain := 0.0
for v in payoff {plain += v}
fmt.printf("asian plain %.4f, with control variate %.4f (variance / %.1f)\n", plain / f64(n_paths), adj, ratio)

up := 0
for p in 0 ..< n_paths {if fin.mc_check_barrier_breached(paths, p, n_steps, 120, true) {up += 1}}
fmt.printf("paths touching 120: %.2f%%\n", 100 * f64(up) / f64(n_paths))
```

```
asian plain 6.1106, with control variate 6.1265 (variance / 3.8)
paths touching 120: 33.17%
```

Common-random-number Greeks, with the normals in a global:

```odin
crn_z: []f64
crn_price :: proc(S_0: f64) -> f64 {
	paths := fin.mc_generate_gbm_paths(S_0, 0.05, 0.2, 1, len(crn_z), 1, crn_z, context.temp_allocator)
	sum := 0.0
	for p in 0 ..< len(crn_z) {sum += max(paths[p * 2 + 1] - 100, 0)}
	return sum / f64(len(crn_z)) * math.exp(-0.05)
}

crn_z = fin.mc_generate_normals(200_000, context.allocator)
defer delete(crn_z)
delta, gamma := fin.mc_compute_crn_delta_gamma(100, crn_price)
g := fin.compute_greeks(100, 100, 1, 0.05, 0.2, .Call)
fmt.printf("CRN delta %.4f gamma %.5f (exact %.4f %.5f)\n", delta, gamma, g.delta, g.gamma)
free_all(context.temp_allocator)
```

```
CRN delta 0.6368 gamma 0.01845 (exact 0.6368 0.01876)
```

The Heston generator next to the COS price of the same option:

```odin
hp := fin.Heston_Params{v0 = 0.04, kappa = 1.5, theta = 0.06, sigma = 0.7, rho = -0.7}
zh := fin.mc_generate_normals(n_paths * 50 * 2, context.allocator)
defer delete(zh)
S, V := fin.mc_generate_heston_paths(100, 0.03, 1, hp, n_paths, 50, zh, context.allocator)
defer {delete(S); delete(V)}
mc := 0.0
for p in 0 ..< n_paths {mc += max(S[p * 51 + 50] - 100, 0)}
cos: [1]f64
fin.heston_prices_cos(100, {100}, 1, 0.03, hp, .Call, cos[:])
fmt.printf("heston euler MC %.4f, COS %.4f\n", mc / f64(n_paths) * math.exp(-0.03), cos[0])
```

```
heston euler MC 9.2878, COS 9.2913
```

Model details:
- **Heston:** full truncation (Lord, Koekkoek & van Dijk 2010) uses v⁺ = max(v, 0) in the drift and the diffusion, which keeps the discounted stock a martingale step by step. `V_paths` holds v⁺. The Euler discretisation has a small bias, so use enough steps (50 per year here).
- **Merton:** the number of jumps per step is Poisson(λΔt), drawn from the uniforms by inverting the CDF. The drift is compensated, so E[S_T] = S₀e^{rT}.
- **Hull–White:** the paths are sampled exactly from the Ornstein–Uhlenbeck transition and fitted to a flat curve at r0, so E[r(t)] = r0 + σ²/(2a²)(1 − e^{−at})².

## 10. Quasi-Monte Carlo

`make_sobol(dims, skip = 0, shift_seed = 0)` creates a Sobol' sequence with up to 2048 dimensions. It uses Joe–Kuo direction numbers and 32 bits, in Gray-code order, and matches `scipy.stats.qmc.Sobol(scramble = false, bits = 32)` point for point. `sobol_next(&s, out)` writes the next point in [0, 1)^dims, and `sobol_skip_to(&s, n)` jumps to point n. Free the sequence with `free_sobol`.

```odin
s := fin.make_sobol(3)
defer fin.free_sobol(&s)
pt: [3]f64
for _ in 0 ..< 4 {
	fin.sobol_next(&s, pt[:])
	fmt.println(pt)
}
```

```
[0, 0, 0]
[0.5, 0.5, 0.5]
[0.75, 0.25, 0.25]
[0.25, 0.75, 0.75]
```

`mc_generate_sobol_normals(n_paths, n_steps, n_factors = 1, brownian_bridge = true, shift_seed = 0, allocator)` turns Sobol' points into standard normals laid out for the `mc_generate_*` generators. For Heston, use `n_factors = 2`. `n_steps · n_factors` may be at most 2048.

**Shift the sequence.** Without a shift, path 0 is the origin of the sequence. It maps to z ≈ −6.4 in every dimension, which is far outside the 1/n_paths tail it stands for. That biases payoffs that are large in the tail, such as puts. A random digital shift (`shift_seed ≠ 0`) moves the origin. Averaging several shifts removes the bias and gives an error estimate.

A one-step European put, three ways:

```odin
exact := fin.compute_greeks(100, 100, 1, 0.05, 0.2, .Put).price
price_put :: proc(z: []f64, n: int) -> f64 {
	paths := fin.mc_generate_gbm_paths(100, 0.05, 0.2, 1, n, 1, z, context.temp_allocator)
	sum := 0.0
	for p in 0 ..< n {sum += max(100 - paths[p * 2 + 1], 0)}
	return sum / f64(n) * math.exp(-0.05)
}
for n in ([]int{1024, 16384}) {
	rand.reset(3)
	zr := fin.mc_generate_normals(n, context.temp_allocator)
	zs := fin.mc_generate_sobol_normals(n, 1, allocator = context.temp_allocator)
	shifted := 0.0
	for seed in 1 ..= 8 {
		zd := fin.mc_generate_sobol_normals(n, 1, shift_seed = u64(seed), allocator = context.temp_allocator)
		shifted += price_put(zd, n) / 8
	}
	fmt.printf("n %5d  error: random %+.5f  sobol %+.5f  sobol, 8 shifts %+.5f\n",
		n, price_put(zr, n) - exact, price_put(zs, n) - exact, shifted - exact)
	free_all(context.temp_allocator)
}
```

```
n  1024  error: random +0.10138  sobol +0.04280  sobol, 8 shifts +0.00787
n 16384  error: random +0.04625  sobol +0.00249  sobol, 8 shifts -0.00035
```

**The Brownian bridge.** By default (`brownian_bridge = true`), each path is built terminal value first, then the successive midpoints. The first and best-distributed Sobol' dimensions then carry the largest-scale features of the path. The output is still standardised increments, i.i.d. N(0, 1), so any generator accepts it.

On a 64-step Asian call (4096 paths, eight shifts), the bridge reduces the spread between shifts from 0.0091 to 0.0051. The reference is 5.83689 from a control-variate Monte Carlo with 8 million paths:

```
bridge false: error +0.0062, spread over shifts 0.0091
bridge true: error +0.0019, spread over shifts 0.0051
```

The bridge is also available on its own:
- `make_brownian_bridge(n_steps)` for equal steps;
- `make_brownian_bridge_times(times)` for any increasing times;
- `brownian_bridge_transform(&b, z, out)` maps variates to standardised increments;
- `free_brownian_bridge` frees it.

It matches QuantLib's `BrownianBridge` to 15 digits.

## 11. Fractional Brownian motion and rough volatility

**Fractional Brownian motion** B^H with Hurst exponent 0 < H < 1 has covariance ½(t^{2H} + s^{2H} − |t − s|^{2H}). H = 0.5 is ordinary Brownian motion; H < 0.5 gives rough paths and H > 0.5 smooth, trending paths.

wotan simulates fBm exactly on any time grid through the Cholesky factor of that covariance:
- `generate_fbm_path(times, H)` returns one path.
- For many paths, factor once with `fbm_cholesky(times, H)` and call `fbm_path_from_cholesky(&L)` per path. The factorisation is O(n³); a path is O(n²).

Times may include 0 or repeat; those points get the value they must have, without failing the factorisation.

```odin
rand.reset(2)
times := []f64{0.25, 0.5, 0.75, 1.0}
path := fin.generate_fbm_path(times, 0.3)
defer delete(path)
fmt.printf("%.4f\n", path)

L := fin.fbm_cholesky(times, 0.3)
defer l.matrix_free(&L)
for _ in 0 ..< 2 {
	p := fin.fbm_path_from_cholesky(&L)
	fmt.printf("%.4f\n", p)
	delete(p)
}
```

```
[0.4942, 0.1631, -0.2472, -1.1844]
[0.1339, 0.4433, -0.1678, -1.0010]
[-0.2576, 0.4291, 0.8602, 0.5669]
```

**Fractional Black–Scholes** (Wick–Itô, Necula 2002) uses S_T = S·exp(rT − ½σ²T^{2H} + σB^H_T). The call is the Black–Scholes price with total variance σ²T^{2H}:
- `fbm_european_call_closed_form(S, K, T, r, sigma, H)` is the closed form;
- `fbm_european_call_price(…, n_paths, n_steps)` is its Monte Carlo. The payoff depends only on B^H_T, which is sampled directly, so `n_steps` has no effect.

The model admits arbitrage under pathwise trading, because fBm is not a semimartingale. Use it for empirical fitting and as a baseline.

```odin
for H in ([]f64{0.3, 0.5, 0.7}) {
	fmt.printf("H %.1f: closed form %.4f  MC %.4f\n", H,
		fin.fbm_european_call_closed_form(100, 100, 2, 0.03, 0.2, H),
		fin.fbm_european_call_price(100, 100, 2, 0.03, 0.2, H, 200_000, 0))
}
```

```
H 0.3: closed form 12.7043  MC 12.7458
H 0.5: closed form 14.0736  MC 14.0617
H 0.7: closed form 15.6520  MC 15.7070
```

The Monte Carlo prices are within one standard error (about 0.05).

**Rough Bergomi** (Bayer, Friz & Gatheral 2016) is an arbitrage-free rough volatility model:
- V_t = ξ₀·exp(η·Ṽ_t − ½η²t^{2H}), with the Volterra process Ṽ_t = √(2H)∫₀ᵗ (t − s)^{H−½} dW_s;
- the stock is driven by ρ·dW + √(1 − ρ²)·dW⊥.

The same W drives the volatility and the stock, which produces the steep short-maturity skew that the model is known for. Its parameters are:

| Field | Meaning | Typical |
|---|---|---|
| `xi_0` | initial forward variance | 0.04 for 20 % volatility |
| `eta` | volatility of volatility | 1.5–2.5 for index options |
| `H` | Hurst exponent | 0.05–0.15 |
| `rho` | correlation of stock and volatility shocks | −0.7 to −0.9 |

How it is simulated:
- `rbergomi_mc_calls(S_0, strikes, T, r, params, n_paths, n_steps)` prices calls at several strikes from one set of paths, so the smile has no strike-to-strike noise. `rbergomi_mc_call` prices one strike.
- The Gaussian pair (W, Ṽ) is sampled exactly on the grid, from the Cholesky factor of their joint covariance (`rbergomi_covariance`, computed by quadrature). ln S then follows a log-Euler step.

```odin
rand.reset(4)
params := fin.rBergomi_Params{xi_0 = 0.04, eta = 1.9, H = 0.1, rho = -0.9}
strikes := []f64{80, 90, 100, 110, 120}
calls := fin.rbergomi_mc_calls(100, strikes, 0.25, 0.0, params, 50_000, 64)
defer delete(calls)
for K, k in strikes {
	iv, _, _ := fin.implied_volatility(calls[k], 100, K, 0.25, 0.0, .Call)
	fmt.printf("K %.0f: %.4f (iv %.4f)\n", K, calls[k], iv)
}
```

```
K 80: 20.3491 (iv 0.2904)
K 90: 11.0686 (iv 0.2305)
K 100: 3.4292 (iv 0.1720)
K 110: 0.1877 (iv 0.1260)
K 120: 0.0045 (iv 0.1279)
```

At three months, the implied volatility falls from 29 % to 13 % between the 80 and 110 strikes. The 120 strike is worth 0.0045, priced from a handful of paths, so its implied volatility is noisy.

With one step the variance stays at ξ₀ and the model reduces to Black–Scholes. With 400,000 paths, `rbergomi_mc_call(100, 100, 0.5, 0, params, 400_000, 1)` gives 5.6345, against the Black–Scholes 5.6372 at σ = 20 %.

## 12. Validation

**Reference tests in the suite:**

| Test | Compares |
|---|---|
| `european_options_reference_test` | Black–Scholes, implied volatility and the Monte Carlo exotics against SciPy closed forms and a 1M-path NumPy Monte Carlo |
| `american_options_reference_test` | trees, finite differences and Longstaff–Schwartz against QuantLib prices and Greeks |
| `fourier_qmc_reference_test` | COS for Heston, Bates and Merton against QuantLib; Sobol' points against SciPy; the Brownian bridge |
| `monte_carlo_reference_test` | the path generators, Hull–White moments, the regression step and control variates against NumPy |
| `fbm_reference_test` | fBm covariances, rBergomi and fractional Black–Scholes against QUADPACK and NumPy |

**Additional checks while writing this guide:**

| What | Reference | Agreement |
|---|---|---|
| `compute_greeks`: price and all Greeks, negative rates | QuantLib `AnalyticEuropeanEngine` | every printed digit (15) |
| `black_scholes_price` gradients | finite differences of the closed form | 9 digits (the accuracy of the differences) |
| `implied_volatility` round trips: price 6e-28, σ = 5, T = 30, negative rates | the input σ | 1e-14 |
| binomial tree, 2000 steps, q = 4 % | QuantLib `BinomialCRRVanillaEngine` | 5e-6 |
| American finite differences, q = 0 | QuantLib `FdBlackScholesVanillaEngine` 800 × 800 | 4e-6 |
| Longstaff–Schwartz, 50 dates | QuantLib Bermudan, finite differences | 12.561 ± 0.018 vs 12.555 |
| Heston COS, 7 days to 10 years, calls and puts, strikes 50–200 | QuantLib `AnalyticHestonEngine` | 12 digits; 3e-14 absolute far out of the money |
| Bates COS | QuantLib `BatesEngine` | 12 digits |
| Merton COS | Merton's series of Black–Scholes prices | 12 digits |
| Sobol' points with skip | SciPy `qmc.Sobol` | exact |
| Brownian bridge, equal and unequal steps | QuantLib `BrownianBridge` | 15 digits |
| Monte Carlo exotics (section 5) | NumPy, 2M paths | within one standard error |
| GBM, Heston and Merton paths | E[e^{−rT}S_T] = S₀ | within Monte Carlo error |
| fBm covariance (H = 0.3, with t = 0 and repeated times) | the exact covariance | within Monte Carlo error |

**Fixed while writing this guide:**
- **Tree Greeks with 2 steps.** The binomial and trinomial trees read the second-level node values for gamma and theta inside the backward loop, which starts at level `n_steps − 1`. With `n_steps ≤ 2` that level is the expiry level, which the loop never visits. Gamma came out as 0 and theta as −price/(2Δt). Both trees now take level 2 from the payoffs, and the 2-step binomial gamma matches QuantLib's 2-step CRR tree.
- **Batches in `black_scholes_price`.** It multiplied σ² by a 1 × 1 constant tensor, and `tensor_mul` doesn't broadcast, so n × 1 inputs panicked. It now scales by ½ with `tensor_scale`.
- **Leaks in `black_scholes_price`.** The tensor engine doesn't link operations whose inputs all lack a gradient into the graph. Intermediates such as −rT, e^−rT, K·e^−rT and √T for a fixed r and T were therefore never freed. When only σ had a gradient that was 7 tensors per call; without any gradient it was all 21, the price included. They are now handed to the graph or freed at once, the same way the `nn` losses do it.
- **Two comments in `monte_carlo.odin`:** the file header now describes the building blocks, and `mc_compute_asian_average` no longer claims continuous monitoring (it averages the grid values S₁..Sₙ).

`finance_fixes_test` checks the three code fixes. Each check fails on the previous code.

**Known limitations:**
- Dividends are supported only by the trees.
- The `mc_generate_*` procedures require an allocator argument.
- The unshifted Sobol' normals start at the origin (section 10).

## 13. Quick reference

| Task | Procedure | Section |
|---|---|---|
| Black–Scholes price and Greeks | `compute_greeks`, `price_and_greeks` | 2 |
| Implied volatility | `implied_volatility` | 3 |
| Black–Scholes with autograd | `black_scholes_price`, `tensor_smooth_max` | 4 |
| GARCH volatility for a maturity | `garch_term_structure_vol`, `garch_price_and_greeks` | 4 |
| Asian, barrier, lookback, basket by Monte Carlo | `monte_carlo_asian_option`, `monte_carlo_barrier_call_bs`, `monte_carlo_barrier_asian_option`, `monte_carlo_lookback_call_option`, `monte_carlo_basket_option` | 5 |
| American option on a tree | `american_option_binomial`, `american_option_trinomial` and wrappers | 6 |
| American or European option by finite differences | `finite_difference_american`, `crank_nicolson_european` and wrappers | 6 |
| American option by Monte Carlo | `lsm_american_vanilla`, `lsm_american_put`, `lsm_american_call`, `lsm_american_asian_call`, `lsm_american_vanilla_reduced` | 7 |
| Heston, Merton, Bates prices | `heston_prices_cos`, `merton_prices_cos`, `bates_prices_cos` | 8 |
| Prices from any characteristic function | `cos_option_prices` | 8 |
| Simulated paths | `mc_generate_gbm_paths`, `mc_generate_gbm_paths_antithetic`, `mc_generate_heston_paths`, `mc_generate_mjd_paths`, `mc_generate_hw1f_paths` | 9 |
| Payoff helpers | `mc_compute_asian_average`, `mc_compute_asian_average_to_step`, `mc_check_barrier_breached`, `mc_check_double_barrier_breached`, `mc_compute_lookback_max`, `mc_compute_lookback_min` | 9 |
| Variance reduction and Greeks | `mc_apply_control_variate`, `mc_compute_crn_delta_gamma`, `mc_compute_crn_vega`, `mc_lsm_regression_step` | 9 |
| Random numbers | `mc_generate_normals`, `mc_generate_uniforms`, `mc_generate_sobol_normals` | 9, 10 |
| Sobol' sequence | `make_sobol`, `sobol_next`, `sobol_skip_to`, `free_sobol` | 10 |
| Brownian bridge | `make_brownian_bridge`, `make_brownian_bridge_times`, `brownian_bridge_transform`, `free_brownian_bridge` | 10 |
| Fractional Brownian motion | `generate_fbm_path`, `fbm_cholesky`, `fbm_path_from_cholesky` | 11 |
| Fractional Black–Scholes | `fbm_european_call_closed_form`, `fbm_european_call_price` | 11 |
| Rough Bergomi | `rbergomi_mc_call`, `rbergomi_mc_calls`, `rbergomi_covariance` | 11 |
