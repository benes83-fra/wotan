# The `finance` package, part 2: volatility models, exotics and FX

This is the second part of the guide to `wotan/finance`. [Part 1](finance_options_guide.md) covers Black–Scholes, implied volatility, American options, COS pricing and the Monte Carlo tools; this part builds on it. It covers:
- the SABR smile and the Heston model: prices, characteristic function and calibration to a volatility surface;
- Monte Carlo exotics (barrier, Asian, lookback, basket) under Heston, Merton jump diffusion and local volatility, with common-random-number Greeks;
- a Dupire local-volatility surface built from a Heston model;
- variance and volatility swaps: replication from options, Heston fair strikes and mark-to-market;
- FX options: Garman–Kohlhagen, quantos, delta-quoted strikes and smiles, vanna-volga, digitals, barrier digitals and double-no-touch options;
- default probabilities and the risk-neutral density implied by option prices (Breeden–Litzenberger).

Every example was compiled and run against the current code, and the output shown is the real output. The results agree with QuantLib 1.44, SciPy and NumPy; section 9 lists the comparisons. A debug build with a tracking allocator reports no leaks for any of the examples.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [SABR](#2-sabr)
3. [Heston: prices and calibration](#3-heston-prices-and-calibration)
4. [Exotics under Heston and Merton](#4-exotics-under-heston-and-merton)
5. [Local volatility](#5-local-volatility)
6. [Variance and volatility swaps](#6-variance-and-volatility-swaps)
7. [FX options](#7-fx-options)
8. [Implied default probability](#8-implied-default-probability)
9. [Validation](#9-validation)
10. [Quick reference](#10-quick-reference)

## 1. Setup and conventions

```odin
import fin "wotan/finance"
import "core:math/rand"
```

The conventions of part 1 apply:
- Times are in years, rates continuously compounded, and volatilities annual.
- `OptionType` is `.Call` or `.Put`.
- The Monte Carlo pricers use `core:math/rand`, so `rand.reset(seed)` makes a run reproducible.

Three model parameter structs appear throughout:

| Struct | Fields |
|---|---|
| `SABR_Params` | `alpha` (initial volatility, in units of F^(1−β)), `beta` (CEV exponent), `rho` (spot–volatility correlation), `nu` (volatility of volatility) |
| `Heston_Params` | `v0` (initial variance), `kappa` (mean reversion speed), `theta` (long-run variance), `sigma` (volatility of variance), `rho` (correlation) |
| `MJD_Params` | `sigma` (diffusion volatility), `lambda` (jumps per year), `mu_j` and `sigma_j` (mean and volatility of the log jump size) |

Quotes for calibration are `VolSurfacePoint{strike, expiry, implied_vol, market_price}`. The calibrations read `strike`, `expiry` and `implied_vol`; `market_price` is for your own bookkeeping. The `OptionQuote` struct (bid/ask for calls and puts) is a container for raw market data; no procedure in this part reads it.

## 2. SABR

`sabr_implied_vol(F, K, T, params)` is Hagan's lognormal implied volatility for the forward `F`. It is written so that it stays accurate at and near the money.

```odin
p := fin.SABR_Params{alpha = 1.3, beta = 0.6, rho = -0.35, nu = 0.5}
F := 100 * math.exp(0.03 * 1.0) // forward for S = 100, r = 3%, T = 1
for K in ([]f64{70, 85, 100, 115, 130}) {
	fmt.printf("K %3.0f: %.4f\n", K, fin.sabr_implied_vol(F, K, 1, p))
}
```

```
K  70: 0.2712
K  85: 0.2358
K 100: 0.2100
K 115: 0.1941
K 130: 0.1875
```

α is not a volatility unless β = 1. The at-the-money volatility is roughly α/F^(1−β), here 1.3/103^0.4 ≈ 0.20.

**Calibration.** `calibrate_sabr(quotes, S, r)` fits one parameter set to all quotes by Levenberg–Marquardt on the implied-volatility errors. Each quote's forward is S·e^(r·expiry).

β and ρ both tilt the smile, so they are hard to tell apart from noisy quotes. In practice β is usually fixed (0.5 or 1 are common) and only α, ρ and ν are fitted; pass `fixed_beta` for that. Here the quotes come from the parameters above, with 0.1 % noise added:

```odin
rand.reset(1)
quotes := make([dynamic]fin.VolSurfacePoint)
defer delete(quotes)
for K in ([]f64{70, 80, 90, 100, 110, 120, 130}) {
	iv := fin.sabr_implied_vol(F, K, 1, p) + 0.001 * rand.float64_normal(0, 1)
	append(&quotes, fin.VolSurfacePoint{strike = K, expiry = 1, implied_vol = iv})
}
free := fin.calibrate_sabr(quotes[:], 100, 0.03)
fixed := fin.calibrate_sabr(quotes[:], 100, 0.03, fixed_beta = 0.6)
for r, name in ([]fin.SABR_CalibrationResult{free, fixed}) {
	fmt.printf("%s alpha %.4f beta %.4f rho %.4f nu %.4f  rmse %.5f  %d iterations\n",
		"free: " if name == 0 else "fixed:", r.params.alpha, r.params.beta, r.params.rho, r.params.nu, r.rmse, r.iterations)
}
fmt.printf("true params on the noisy quotes: rmse %.5f\n", fin.compute_sabr_rmse(quotes[:], 100, 0.03, p))
```

```
free:  alpha 1.9159 beta 0.5166 rho -0.3327 nu 0.4896  rmse 0.00067  101 iterations
fixed: alpha 1.3030 beta 0.6000 rho -0.3618 nu 0.5026  rmse 0.00067  5 iterations
true params on the noisy quotes: rmse 0.00106
```

Both fits match the quotes equally well, but the free fit moves β and α far from their true values. With β fixed, α, ρ and ν come back close to the truth.

The result is a `SABR_CalibrationResult{params, rmse, converged, iterations}`. `rmse` is in volatility units. `compute_sabr_rmse` computes the same error for any parameter set.

The fit starts from β = 0.7 (or the fixed β), ρ = −0.3, ν = 0.4, and the α that reproduces the at-the-money volatility of the shortest expiry. One parameter set across several expiries is a compromise: SABR is usually calibrated per expiry, so pass one expiry's quotes at a time if you need a surface.

## 3. Heston: prices and calibration

`heston_price(S, K, T, r, params, type)` prices one European option and `heston_prices_same_expiry(S, strikes, T, r, params, type, out)` prices several strikes of one expiry. Both use the COS method from part 1 (`heston_prices_cos`), so they are accurate to about 12 digits. `heston_char_func(u, S, r, T, params)` is the characteristic function of ln S_T. It is written in the "little Heston trap" form, which stays stable for long maturities and small σ.

```odin
HP :: fin.Heston_Params{v0 = 0.04, kappa = 1.5, theta = 0.06, sigma = 0.7, rho = -0.7}

fmt.printf("call %.6f put %.6f\n", fin.heston_price(100, 100, 1, 0.03, HP, .Call), fin.heston_price(100, 100, 1, 0.03, HP, .Put))
strikes := []f64{80, 100, 120}
out: [3]f64
fin.heston_prices_same_expiry(100, strikes, 1, 0.03, HP, .Call, out[:])
fmt.printf("%.6f\n", out)
```

```
call 9.291294 put 6.335848
[24.368262, 9.291294, 1.359743]
```

**Calibration.** `calibrate_heston(quotes, S, r)` fits the five parameters to a surface. It minimises implied-volatility errors, as QuantLib's `ImpliedVolError` does, so cheap wings and short expiries count as much as expensive options. The method is Levenberg–Marquardt, with the Jacobian computed in parallel.

The fit starts from v0 and θ equal to the at-the-money implied variance of the shortest and the longest expiry, κ = 2, σ = 0.5 and ρ = −0.5. `true_sigma` (0.16) is only used for v0 and θ when there is no at-the-money quote.

The Feller condition 2κθ > σ² keeps the variance away from zero. Fourier pricing doesn't need it, but a Monte Carlo simulation of the variance does. `feller_weight > 0` enforces it exactly: σ is fitted as a fraction of √(2κθ).

```odin
quotes := make([dynamic]fin.VolSurfacePoint)
defer delete(quotes)
for T in ([]f64{0.25, 0.5, 1, 2}) {
	for K in ([]f64{80, 90, 100, 110, 120}) {
		c := fin.heston_price(100, K, T, 0.03, HP, .Call)
		iv, _, _ := fin.implied_volatility(c, 100, K, T, 0.03, .Call)
		append(&quotes, fin.VolSurfacePoint{strike = K, expiry = T, implied_vol = iv, market_price = c})
	}
}
r := fin.calibrate_heston(quotes[:], 100, 0.03)
fmt.printf("free:   v0 %.4f kappa %.4f theta %.4f sigma %.4f rho %.4f  iv rmse %.2e  feller %.4f\n",
	r.params.v0, r.params.kappa, r.params.theta, r.params.sigma, r.params.rho, r.iv_rmse, r.feller_margin)
rf := fin.calibrate_heston(quotes[:], 100, 0.03, feller_weight = 1)
fmt.printf("feller: v0 %.4f kappa %.4f theta %.4f sigma %.4f rho %.4f  iv rmse %.2e  feller %.4f\n",
	rf.params.v0, rf.params.kappa, rf.params.theta, rf.params.sigma, rf.params.rho, rf.iv_rmse, rf.feller_margin)
fmt.printf("price rmse %.2e\n", fin.compute_heston_rmse(quotes[:], 100, 0.03, rf.params))
```

```
free:   v0 0.0400 kappa 1.5000 theta 0.0600 sigma 0.7000 rho -0.7000  iv rmse 3.23e-12  feller -0.3100
feller: v0 0.0421 kappa 2.3848 theta 0.0480 sigma 0.4779 rho -0.8347  iv rmse 9.89e-03  feller 0.0005
price rmse 2.48e-01
```

The free fit recovers the parameters to 10 digits, in 9 iterations. These parameters violate the Feller condition (`feller_margin` < 0), so the constrained fit has to compromise and misses the surface by 1 % in volatility. The result type, `Heston_CalibrationResult`, holds:
- `params`, `converged` and `iterations`;
- `iv_rmse`, the minimised implied-volatility error;
- `rmse`, the price error;
- `feller_margin`, which is 2κθ − σ².

`compute_heston_rmse` computes the price error for any parameter set.

The objective functions `sabr_objective_fn` and `heston_objective_fn`, with their context structs, are public, so you can plug them into the optimisers of `wotan/optimize` yourself.

## 4. Exotics under Heston and Merton

These Monte Carlo pricers return the price, delta and vega. For baskets they return two deltas and two vegas.

| Procedure | Model | Payoff |
|---|---|---|
| `heston_mc_barrier_option(S_0, K, T, r, barrier, is_up, type, params, n_paths, n_steps)` | Heston | knock-out call or put, up or down |
| `heston_mc_asian_option(S_0, K, T, r, type, params, n_paths, n_steps)` | Heston | arithmetic average of S₁..Sₙ |
| `heston_mc_lookback_option(…)` | Heston | fixed strike: call on the maximum, put on the minimum (S₀ included) |
| `heston_mc_basket_option(S1_0, S2_0, K, T, r, w1, w2, type, params1, params2, corr_12, n_paths, n_steps)` | two Heston assets | w₁S¹_T + w₂S²_T against K, at expiry |
| `mjd_mc_barrier_option(…, params: MJD_Params, …)` | Merton jump diffusion | knock-out call or put, up or down |
| `mjd_mc_asian_option(…)` | Merton jump diffusion | arithmetic average |

There are also wrappers fixed to calls: `heston_mc_barrier_call` (up-and-out), `heston_mc_asian_call`, `heston_mc_lookback_call` and `heston_mc_basket_call`.

```odin
rand.reset(2)
p, d, v := fin.heston_mc_barrier_option(100, 100, 1, 0.03, 130, true, .Call, HP, 50_000, 100)
fmt.printf("heston up-and-out call  %.4f delta %.4f vega %.4f\n", p, d, v)
rand.reset(2)
p, d, v = fin.heston_mc_barrier_option(100, 100, 1, 0.03, 80, false, .Put, HP, 50_000, 100)
fmt.printf("heston down-and-out put %.4f delta %.4f vega %.4f\n", p, d, v)
rand.reset(2)
p, d, v = fin.heston_mc_asian_option(100, 100, 1, 0.03, .Call, HP, 50_000, 12)
fmt.printf("heston asian call       %.4f delta %.4f vega %.4f\n", p, d, v)
rand.reset(2)
p, d, v = fin.heston_mc_lookback_option(100, 100, 1, 0.03, .Put, HP, 50_000, 52)
fmt.printf("heston lookback put     %.4f delta %.4f vega %.4f\n", p, d, v)
rand.reset(2)
bp, d1, d2, v1, v2 := fin.heston_mc_basket_option(100, 50, 150, 1, 0.03, 1, 1, .Call, HP, HP, 0.5, 50_000, 50)
fmt.printf("heston basket call      %.4f delta %.4f %.4f vega %.4f %.4f\n", bp, d1, d2, v1, v2)

mjd := fin.MJD_Params{sigma = 0.2, lambda = 0.5, mu_j = -0.1, sigma_j = 0.15}
rand.reset(2)
p, d, v = fin.mjd_mc_barrier_option(100, 100, 1, 0.03, 130, true, .Call, mjd, 50_000, 100)
fmt.printf("merton up-and-out call  %.4f delta %.4f vega %.4f\n", p, d, v)
rand.reset(2)
p, d, v = fin.mjd_mc_asian_option(100, 100, 1, 0.03, .Call, mjd, 50_000, 12)
fmt.printf("merton asian call       %.4f delta %.4f vega %.4f\n", p, d, v)
```

```
heston up-and-out call  7.0919 delta 0.3543 vega -9.5104
heston down-and-out put 0.8183 delta -0.0403 vega -1.7234
heston asian call       5.7207 delta 0.6554 vega 22.1186
heston lookback put     13.1764 delta -0.7639 vega 64.9753
heston basket call      12.2487 delta 0.6879 0.6707 vega 35.4795 13.0765
merton up-and-out call  3.3100 delta 0.0290 vega -23.2810
merton asian call       6.3522 delta 0.5805 vega 21.3231
```

How these pricers work:
- **Time grid and monitoring:** the grid is equidistant. Barriers are monitored on it (not at t = 0), averages use S₁..Sₙ, and the lookback extremes include S₀.
- **Barriers:** they use one-step survival, as in part 1. Each step is drawn conditional on staying alive, and the path is weighted by the survival probabilities. A barrier at or below 0 is never hit as a down barrier and always hit as an up barrier.
- **Heston dynamics:** full-truncation Euler, so use enough steps (50 to 100 per year). Its small bias shows in the Asian price.
- **Merton dynamics:** exact on the grid, with the number of jumps per step drawn from a Poisson distribution.
- **Greeks:** central differences on common random numbers. Every path draws its normals once and is re-simulated for the base case and each bump. Delta bumps S₀ by ±1 %.
- **What vega means:** it is per unit of volatility, but the volatility depends on the model:

| Model | Volatility bumped for vega |
|---|---|
| Heston | √v0 and √θ together (a parallel shift of the instantaneous and long-run volatility) |
| Merton | the diffusion volatility σ |
| local volatility (section 5) | the whole surface, shifted in parallel |

Where a model reduces to Black–Scholes, all three agree with part 1's pricer: Heston with σ = 0 and v0 = θ, Merton with λ = 0, and a flat local-volatility surface. Here is a 20 % up-and-out call, priced three ways:

```odin
flat := fin.Heston_Params{v0 = 0.04, kappa = 1, theta = 0.04, sigma = 0, rho = 0}
rand.reset(3)
h, _, hv := fin.heston_mc_barrier_option(100, 100, 1, 0.03, 130, true, .Call, flat, 100_000, 100)
rand.reset(3)
m, _, mv := fin.mjd_mc_barrier_option(100, 100, 1, 0.03, 130, true, .Call, fin.MJD_Params{sigma = 0.2}, 100_000, 100)
rand.reset(3)
b, _, bv := fin.monte_carlo_barrier_call_bs(100, 100, 1, 0.03, 0.2, 130, 100_000, 100)
fmt.printf("flat 20%%: heston %.4f (vega %.2f)  merton %.4f (%.2f)  black-scholes %.4f (%.2f)\n", h, hv, m, mv, b, bv)
```

```
flat 20%: heston 3.5270 (vega -22.84)  merton 3.5197 (-23.14)  black-scholes 3.5371 (-22.99)
```

**The basket's correlation.** In the basket, `corr_12` is the correlation of the two stocks' Brownian motions. Each variance process is correlated only with its own stock. The stocks' independent shocks are given the correlation ρ* = corr₁₂/(√(1−ρ₁²)·√(1−ρ₂²)), so that the stocks end up with corr₁₂. ρ* is clamped to [−1, 1] when corr₁₂ can't be reached with the stocks' ρ.

This basket pays on the terminal prices. Part 1's `monte_carlo_basket_option` is an Asian basket on the average prices.

## 5. Local volatility

`build_lv_surface_from_heston(S_0, r, params)` builds a Dupire local-volatility surface from a Heston model:

σ²_loc(K, T) = (∂C/∂T + r·K·∂C/∂K) / (½·K²·∂²C/∂K²)

The derivatives are finite differences of COS prices. Puts are used below the forward, so deep in-the-money calls don't lose the derivatives to cancellation. The grid is fixed:
- 21 strikes from 0.6·S₀ to 1.4·S₀;
- 10 expiries from 0.1 to 1 year;
- the edge values are copied from the nearest interior point.

A `LocalVolSurface` holds `strikes`, `expiries` and the flattened `lv_data[expiry · num_strikes + strike]`. You can also fill one yourself from another source. Free it with `lv_surface_free`.

Two procedures use the surface:
- **`get_local_vol(lv, S, t)`** interpolates bilinearly. Outside the grid it uses the edge values.
- **`lv_mc_barrier_option(S_0, K, T, r, barrier, is_up, type, lv, n_paths, n_steps)`** prices barrier options. Each step uses the local volatility at its start.

```odin
lv := fin.build_lv_surface_from_heston(100, 0.03, HP)
defer fin.lv_surface_free(&lv)
for S in ([]f64{70, 85, 100, 115, 130}) {
	fmt.printf("sigma_loc(S = %.0f, t = 0.5) = %.4f\n", S, fin.get_local_vol(lv, S, 0.5))
}
rand.reset(4)
for K in ([]f64{80, 100, 120}) {
	p, _, _ := fin.lv_mc_barrier_option(100, K, 1, 0.03, 1e9, true, .Call, lv, 100_000, 100) // barrier never reached
	fmt.printf("K %3.0f: local vol MC %.4f  heston %.4f\n", K, p, fin.heston_price(100, K, 1, 0.03, HP, .Call))
}
rand.reset(5)
p, d, v := fin.lv_mc_barrier_option(100, 100, 1, 0.03, 130, true, .Call, lv, 50_000, 100)
fmt.printf("local vol up-and-out call %.4f delta %.4f vega %.4f\n", p, d, v)
```

```
sigma_loc(S = 70, t = 0.5) = 0.4374
sigma_loc(S = 85, t = 0.5) = 0.3240
sigma_loc(S = 100, t = 0.5) = 0.1948
sigma_loc(S = 115, t = 0.5) = 0.1288
sigma_loc(S = 130, t = 0.5) = 0.1660
K  80: local vol MC 24.3297  heston 24.3683
K 100: local vol MC 9.3085  heston 9.2913
K 120: local vol MC 1.3722  heston 1.3597
local vol up-and-out call 6.5537 delta 0.1930 vega -29.3171
```

By construction, the local-volatility model reprices the Heston vanillas. The Monte Carlo prices are within 0.04 of the COS prices. The rest of the difference comes from the coarse grid and the fixed volatility in the first 0.1 years.

The barrier prices differ between the models: 6.55 under local volatility against 7.09 under Heston. Both match the same vanillas, but they differ in their forward volatility. This model risk is the reason to price barriers under more than one model.

## 6. Variance and volatility swaps

A variance swap pays N_var·(σ²_realised − K_var) at maturity, where variances are annualised. Its fair strike is the risk-neutral expected variance. That can be replicated statically by out-of-the-money options weighted by 1/K² (Demeterfi, Derman, Kamal & Zou 1999). Both replication procedures take option prices from a callback, `OptionPriceFn :: proc(K, type, data) -> f64`, so any model or interpolated market smile can supply them.

| Procedure | Computes |
|---|---|
| `variance_swap_fair_variance(S, T, r, q, price, data, n_panels = 40)` | the exact integral over a continuum of strikes, by Gauss–Legendre quadrature in log-strike |
| `variance_swap_replication(S, T, r, q, call_strikes, put_strikes, price, data, dk = 5)` | QuantLib's `ReplicatingVarianceSwapEngine` on a finite strip of quoted strikes |
| `heston_fair_variance(T, params)` | the Heston closed form θ + (v0 − θ)(1 − e^−κT)/(κT) |
| `heston_fair_volatility(T, params)` | E[σ_realised], the fair volatility-swap strike under Heston |
| `realized_variance(prices, periods_per_year = 252)` | (A/n)·Σ ln²(Sᵢ/Sᵢ₋₁), the zero-mean term-sheet convention |
| `variance_swap_value(N_var, K_var, realised_var, elapsed, T, remaining_fair_var, discount)` | mark-to-market for the long side |
| `variance_notional_from_vega(vega_notional, strike_vol)` | N_var = N_vega/(2·K_vol) |
| `heston_integrated_variance_laplace(s, T, params)` | E[exp(−s∫v dt)], used for the volatility swap |

In `variance_swap_replication`, the strip's boundary is the lowest call strike (or the highest put strike if there are no calls). One extra strike `dk` is added beyond each end.

```odin
Swap_Data :: struct {S, T, r: f64}
heston_option :: proc(K: f64, opt: fin.OptionType, data: rawptr) -> f64 {
	d := (^Swap_Data)(data)
	return fin.heston_price(d.S, K, d.T, d.r, HP, opt)
}

d := Swap_Data{S = 100, T = 1, r = 0.03}
fair := fin.variance_swap_fair_variance(100, 1, 0.03, 0, heston_option, &d)
fmt.printf("fair variance: from options %.10f, Heston formula %.10f (vol %.4f)\n", fair, fin.heston_fair_variance(1, HP), math.sqrt(fair))
calls := []f64{100, 110, 120, 130, 140, 150}
puts := []f64{100, 90, 80, 70, 60, 50}
strip := fin.variance_swap_replication(100, 1, 0.03, 0, calls, puts, heston_option, &d, dk = 10)
fmt.printf("11-strike strip: %.6f\n", strip)
fmt.printf("fair volatility %.4f (convexity adjustment %.4f)\n", fin.heston_fair_volatility(1, HP), math.sqrt(fair) - fin.heston_fair_volatility(1, HP))
```

```
fair variance: from options 0.0496417355, Heston formula 0.0496417355 (vol 0.2228)
11-strike strip: 0.049800
fair volatility 0.1995 (convexity adjustment 0.0233)
```

The integral over Heston's option prices reproduces the closed form to 10 digits. The 11-strike strip is off by 0.0002 in variance. The fair volatility is below √(fair variance), because E[σ] < √E[σ²]. The difference is the convexity adjustment, here 2.3 volatility points with Heston's high volatility of variance.

**Mark-to-market.** `variance_swap_value` combines the realised variance so far with the fair variance of the rest of the period: N_var·DF·[(t/T)·σ²_realised + ((T − t)/T)·K_var(t, T) − K]. Here a swap struck at the fair variance is valued half a year in, on a simulated price path with 25 % volatility:

```odin
rand.reset(6)
prices := make([]f64, 127)
defer delete(prices)
prices[0] = 100
for i in 1 ..< len(prices) {prices[i] = prices[i - 1] * math.exp(0.25 / math.sqrt(252.0) * rand.float64_normal(0, 1))}
realized := fin.realized_variance(prices)
remaining := fin.heston_fair_variance(0.5, HP) // from today's Heston state, simplified
n_var := fin.variance_notional_from_vega(100_000, math.sqrt(fair))
value := fin.variance_swap_value(n_var, fair, realized, 0.5, 1, remaining, math.exp(-0.03 * 0.5))
fmt.printf("realized vol %.4f, variance notional %.0f, value %.0f\n", math.sqrt(realized), n_var, value)
```

```
realized vol 0.2463, variance notional 224412, value 810
```

The volatility-swap strike uses E[√I] = (1/(2√π))·∫₀^∞ (1 − E[e^(−sI)])·s^(−3/2) ds, with the Laplace transform of the integrated variance. That transform is written without the 1/σ² cancellation, so σ → 0 gives the deterministic limit exactly.

## 7. FX options

**Garman–Kohlhagen.** `fx_gk_price(S, K, T, r_d, r_f, sigma, type)` and `fx_gk_greeks(…)` price FX options. The spot `S` is the price of one unit of foreign currency in domestic currency, for example USD per EUR; `r_d` and `r_f` are the domestic and foreign rates. `fx_gk_call` and `fx_gk_put` are the same with the type fixed.

The FX procedures take an `FXOptionType` (`.Call`, `.Put`). It is separate from `OptionType`, with the same values.

The Greeks in `FXGKResult` are raw derivatives, not scaled per 1 % or per day:
- `delta` is the spot delta, and `delta_f` the forward delta (not premium-adjusted);
- `gamma`;
- `vega` is ∂P/∂σ;
- `theta` is −∂P/∂T, per year;
- `rho_d` and `rho_f` are the derivatives with respect to each rate.

```odin
S, T, rd, rf := 1.10, 0.5, 0.03, 0.01 // EURUSD, USD rate, EUR rate
g := fin.fx_gk_call(S, 1.15, T, rd, rf, 0.09)
fmt.printf("call %.6f delta %.4f fwd delta %.4f gamma %.4f vega %.4f theta %.4f rho_d %.4f rho_f %.4f\n",
	g.price, g.delta, g.delta_f, g.gamma, g.vega, g.theta, g.rho_d, g.rho_f)
```

```
call 0.013130 delta 0.3037 fwd delta 0.3052 gamma 4.9801 vega 0.2712 theta -0.0307 rho_d 0.1605 rho_f -0.1670
```

**Smiles quoted in delta.** FX smiles are quoted as an at-the-money volatility, a 25-delta risk reversal and a 25-delta butterfly. `fx_smile_to_vols` turns these into the 25Δ put, ATM and 25Δ call volatilities, using the smile-strangle convention. `fx_strike_from_delta(S, delta, T, r_d, r_f, sigma, type)` finds the strike with a given spot delta, not premium-adjusted.

```odin
smile := fin.FXSmile{atm_vol = 0.085, risk_reversal_25d = -0.015, butterfly_25d = 0.004}
vols := fin.fx_smile_to_vols(smile)
fmt.printf("25d put %.4f  atm %.4f  25d call %.4f\n", vols.vol_25d_put, vols.vol_atm, vols.vol_25d_call)
k_put := fin.fx_strike_from_delta(S, -0.25, T, rd, rf, vols.vol_25d_put, .Put)
k_call := fin.fx_strike_from_delta(S, 0.25, T, rd, rf, vols.vol_25d_call, .Call)
fmt.printf("strikes: 25d put %.5f  atm-fwd %.5f  25d call %.5f\n", k_put, S * math.exp((rd - rf) * T), k_call)
```

```
25d put 0.0965  atm 0.0850  25d call 0.0815
strikes: 25d put 1.06384  atm-fwd 1.11106  25d call 1.15675
```

**Vanna-volga.** `fx_vanna_volga_adjust` prices an exotic on the smile (Castagna & Mercurio 2007). It uses three pillars: the 25Δ put, an at-the-money-forward call and the 25Δ call. It finds the weights of a pillar portfolio that has the exotic's vega, vanna and volga at the ATM volatility, then adds that portfolio's smile cost (market price minus ATM-volatility price) to the exotic's ATM-volatility price.

Each exotic below has a `…_sensitivities` procedure for its vanna, volga and vega, by central differences. Pass the vega as `exotic_vega`. With the default of 0, only vanna and volga are hedged.

Here is a down-and-out digital that pays 1 USD if S_T > 1.12, knocked out below 1.05:

```odin
sa := vols.vol_atm
mp := fin.fx_gk_price(S, k_put, T, rd, rf, vols.vol_25d_put, .Put)
ma := fin.fx_gk_price(S, S * math.exp((rd - rf) * T), T, rd, rf, sa, .Call)
mc := fin.fx_gk_price(S, k_call, T, rd, rf, vols.vol_25d_call, .Call)
bs := fin.fx_down_and_out_digital_call_price(S, 1.12, 1.05, T, rd, rf, sa)
vanna, volga, vega := fin.fx_down_and_out_digital_call_sensitivities(S, 1.12, 1.05, T, rd, rf, sa)
vv := fin.fx_vanna_volga_adjust(S, T, rd, rf, smile, bs, vanna, volga, mp, ma, mc, exotic_vega = vega)
fmt.printf("digital DO: at ATM vol %.5f, vanna-volga %.5f (weights %.3f %.3f %.3f)\n",
	vv.price_bs, vv.price_vv, vv.weight_25d_put, vv.weight_atm, vv.weight_25d_call)
```

```
digital DO: at ATM vol 0.39390, vanna-volga 0.39113 (weights -7.609 18.994 -20.914)
```

The weights are the notionals of the three vanillas per unit of the digital. They are large because a digital's sensitivities are large next to those of a vanilla on one unit.

**FX exotics:**

| Procedure | Pays 1 domestic unit at T if | Method |
|---|---|---|
| `fx_digital_call_price(S, K, T, r_d, r_f, sigma)` | S_T > K | closed form |
| `fx_down_and_out_digital_call_price(S, K, L, T, r_d, r_f, sigma)` | S_T > K and S never touched L (continuous) | method of images (Reiner & Rubinstein 1991) |
| `fx_dnt_pde_price(S, L, U, T, r_d, r_f, sigma, n_space = 200, n_time = 1000)` | L < S_t < U throughout (double-no-touch) | Crank–Nicolson with a Rannacher start |

```odin
fmt.printf("digital %.5f  double-no-touch 1.05-1.15: %.5f\n",
	fin.fx_digital_call_price(S, 1.12, T, rd, rf, sa), fin.fx_dnt_pde_price(S, 1.05, 1.15, T, rd, rf, sa))
```

```
digital 0.42860  double-no-touch 1.05-1.15: 0.14388
```

**Quantos.** `quanto_option(S, K, T, r_d, r_f, q, sigma, sigma_fx, rho, type, fixed_fx = 1)` prices an option on a foreign asset whose payoff is paid in domestic currency at a fixed exchange rate:
- `r_f` is the asset's currency rate and `q` its dividend yield;
- `sigma_fx` is the volatility of the exchange rate, quoted as domestic per foreign;
- `rho` is the correlation between the asset and the exchange rate.

Under the domestic measure the asset drifts at r_f − q − ρσσ_X; `quanto_forward` returns that forward. The result follows QuantLib's `QuantoEngine`: price, delta, gamma, vega, theta, rho, `dividend_rho`, and the quanto sensitivities `qvega` (σ_X), `qrho` (r_f) and `qlambda` (ρ).

```odin
q := fin.quanto_option(100, 100, 1, 0.03, 0.01, 0.02, 0.25, 0.10, -0.3, .Call, fixed_fx = 1.0)
fmt.printf("quanto call %.4f delta %.4f vega %.4f qvega %.4f qlambda %.4f\n", q.price, q.delta, q.vega, q.qvega, q.qlambda)
fmt.printf("quanto forward %.4f\n", fin.quanto_forward(100, 1, 0.01, 0.02, 0.25, 0.10, -0.3))
```

```
quanto call 9.5209 delta 0.5283 vega 39.9490 qvega 3.9624 qlambda -1.3208
quanto forward 99.7503
```

## 8. Implied default probability

The option market prices a firm's equity tail, and with it a default probability. With C(K) the call prices at one maturity (Breeden & Litzenberger 1978):
- the risk-neutral density is e^(rT)·∂²C/∂K²;
- P(S_T < K) = 1 + e^(rT)·∂C/∂K.

`implied_default_probability(strikes, call_prices, r, T, K_default, allocator, spot = 0)` reads both from a strip of call prices, with strictly ascending strikes. It returns a `DefaultProbResult` with these fields:

| Field | Meaning |
|---|---|
| `pd_risk_neutral` | P(S_T < K_default), from the slope of a quadratic through the three strikes nearest `K_default` |
| `expected_loss` | the put at `K_default` (Merton's expected loss to the debt), by put–call parity |
| `rnd_strikes`, `rnd_values` | the strikes and the density at each, clamped at 0 (free both) |

The probability includes all the mass below the lowest quoted strike, which integrating the density over the grid would miss.

Pass the dividend-free spot S·e^(−qT) as `spot` for an exact expected loss. Without it, the put at the lowest strike is assumed worthless, which makes the expected loss a lower bound. `K_default` outside the strikes gives NaN.

```odin
n := 41
strikes := make([]f64, n)
calls := make([]f64, n)
defer {delete(strikes); delete(calls)}
for i in 0 ..< n {
	K := 20 + 4 * f64(i)
	vol := 0.30 + 0.25 * max(1 - K / 100, 0) // higher vol below the spot
	strikes[i] = K
	calls[i] = fin.compute_greeks(100, K, 1, 0.03, vol, .Call).price
}
res := fin.implied_default_probability(strikes, calls, 0.03, 1, 50, spot = 100)
defer {delete(res.rnd_strikes); delete(res.rnd_values)}
flat := fin.compute_greeks(100, 50, 1, 0.03, 0.30, .Put)
fmt.printf("P(S_T < 50) %.4f, expected loss %.4f (flat 30%% vol: put %.4f)\n", res.pd_risk_neutral, res.expected_loss, flat.price)
for i := 0; i < n; i += 10 {fmt.printf("density at %.0f: %.5f\n", res.rnd_strikes[i], res.rnd_values[i])}
```

```
P(S_T < 50) 0.0519, expected loss 0.5309 (flat 30% vol: put 0.0546)
density at 20: 0.00054
density at 60: 0.00492
density at 100: 0.03713
density at 140: 0.00479
density at 180: 0.00117
```

The exact values for this smile are a probability of 0.05178 and an expected loss of 0.5303. The skew makes the expected loss ten times larger than a flat 30 % volatility would suggest.

The derivatives are finite differences, so noise in the prices is amplified. Take Monte Carlo prices from one set of paths for all strikes, for example `rbergomi_mc_calls` from part 1.

## 9. Validation

**Reference tests in the suite:**

| Test | Compares |
|---|---|
| `vol_surface_reference_test` | Heston and SABR prices and calibration round trips against QuantLib; Levenberg–Marquardt against SciPy |
| `calib_exotics_reference_test` | the Heston, Merton and local-volatility exotics against a 1M-path NumPy Monte Carlo and closed forms |
| `vol_products_reference_test` | variance-swap replication, Heston fair variance and volatility, and quanto options against QuantLib and SciPy |
| `fx_derivatives_reference_test` | Garman–Kohlhagen, cash-or-nothing, barrier digital and double-no-touch against QuantLib; vanna-volga weights |
| `implied_default_prob_reference_test` | default probability on a two-lognormal mixture against SciPy |

**Additional checks while writing this guide:**

| What | Reference | Agreement |
|---|---|---|
| `sabr_implied_vol` with β = 0, 0.5, 0.6, 1 and ρ = ±0.99, 6 strikes | QuantLib `sabrVolatility` | 1e-14 |
| Garman–Kohlhagen prices down to 1e-49 | mpmath, 40 digits | 1e-9 relative (after the fix below) |
| `calibrate_heston` on a model surface (4 × 5 quotes) | the true parameters | 10 digits |
| `calibrate_sabr`, free and with `fixed_beta`, on noise-free quotes | the true parameters | 1e-10 |
| variance-swap integral on Heston COS prices | `heston_fair_variance` | 10 digits |
| `heston_fair_volatility` as σ → 0 | √(fair variance) | 12 digits |
| local-volatility MC vanillas | Heston COS | within 0.04 |
| Heston and Merton exotics at σ = 0 or λ = 0 | Black–Scholes MC | within Monte Carlo error |
| default probability and expected loss on a skewed smile | exact derivatives of the smile | 9e-5 and 5e-4 |
| `fx_strike_from_delta` round trip | the input delta | 1e-12 |

**Fixed while writing this guide:**
- **FX prices far out of the money.** The FX code computed N(x) as ½(1 + erf(x/√2)), which cancels in the left tail. Far out-of-the-money Garman–Kohlhagen prices came out as exactly 0, or negative (−4.9e-17 for a call worth 5.7e-19). It now uses `erfc` like the rest of the package.
- **FX digital at σ = 0.** `fx_digital_call_price` divided by σ, so σ = 0 with the forward exactly at the strike gave NaN. With σ = 0 the digital now pays if the forward is above the strike.
- **Fixing β in SABR.** `calibrate_sabr`'s comment said β could be fixed by setting equal bounds, but the bounds are internal, so there was no way to do it. `calibrate_sabr` now takes `fixed_beta`.
- **Comments:** stale comments in `vol_surface.odin` and `calib_exotics.odin` were tidied ("CRITICAL FIX … identity function before", "Legacy Wrapper: Keeps your existing code working").

`finance_fixes_test` checks the three code changes. The two FX checks fail on the previous code; `fixed_beta` is new.

**Known limitations:**
- `build_lv_surface_from_heston` has a fixed grid (0.6–1.4·S₀, 0.1–1 year), so longer barriers use the 1-year volatilities.
- `fx_down_and_out_digital_call_price` and `fx_dnt_pde_price` need σ > 0.
- `debug_heston()` prints a self-check of the Heston pricer. It is used by `vol_surface_test` and is not needed otherwise.

## 10. Quick reference

| Task | Procedure | Section |
|---|---|---|
| SABR implied volatility | `sabr_implied_vol` | 2 |
| Calibrate SABR | `calibrate_sabr` (`fixed_beta`), `compute_sabr_rmse` | 2 |
| Heston prices | `heston_price`, `heston_prices_same_expiry`, `heston_char_func` | 3 |
| Calibrate Heston | `calibrate_heston` (`feller_weight`), `compute_heston_rmse` | 3 |
| Exotics under Heston | `heston_mc_barrier_option`, `heston_mc_asian_option`, `heston_mc_lookback_option`, `heston_mc_basket_option` and the `…_call` wrappers | 4 |
| Exotics under Merton | `mjd_mc_barrier_option`, `mjd_mc_asian_option` | 4 |
| Local volatility | `build_lv_surface_from_heston`, `get_local_vol`, `lv_mc_barrier_option`, `lv_surface_free` | 5 |
| Variance swap strike | `variance_swap_fair_variance`, `variance_swap_replication`, `heston_fair_variance` | 6 |
| Volatility swap strike | `heston_fair_volatility`, `heston_integrated_variance_laplace` | 6 |
| Realised variance and swap value | `realized_variance`, `variance_swap_value`, `variance_notional_from_vega` | 6 |
| FX vanillas | `fx_gk_price`, `fx_gk_greeks`, `fx_gk_call`, `fx_gk_put` | 7 |
| FX smile and strikes | `fx_smile_to_vols`, `fx_strike_from_delta` | 7 |
| Vanna-volga | `fx_vanna_volga_adjust` | 7 |
| FX digitals and barriers | `fx_digital_call_price`, `fx_down_and_out_digital_call_price`, `fx_dnt_pde_price` and their `…_sensitivities` | 7 |
| Quantos | `quanto_option`, `quanto_forward` | 7 |
| Default probability and density | `implied_default_probability` | 8 |
