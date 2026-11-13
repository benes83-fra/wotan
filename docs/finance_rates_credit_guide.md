# The `finance` package, part 4: swaps, rate options and credit

This is the fourth part of the guide to `wotan/finance`. [Part 3](finance_fixed_income_guide.md) covers dates, calendars, yield curves and bonds, and this part builds on it. [Part 1](finance_options_guide.md) and [part 2](finance_volatility_guide.md) cover equity and FX options. This part covers:
- interest rate swaps on a time axis and on real dates: single-curve, OIS-discounted multi-curve, overnight-indexed, and seasoned swaps with their current fixing;
- caps, floors and swaptions under Hull-White and G2++ (two-factor Gaussian): closed forms, tensors with automatic differentiation, Monte Carlo with Greeks, Bermudan swaptions and calibration;
- credit default swaps: a hazard-rate model on a time axis, and the standard ISDA contract on real dates with upfronts and indices;
- CDO tranches under one-factor Gaussian and Student-t copulas;
- bivariate copulas: fitting five families, rank correlations and tail dependence.

Every example was compiled and run against the current code, and the output shown is the real output. The results agree with QuantLib 1.44, SciPy, and independent Python references; section 10 lists the comparisons. A debug build with a tracking allocator reports no leaks for any of the examples.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [Swaps on a time axis](#2-swaps-on-a-time-axis)
3. [Swaps on real dates](#3-swaps-on-real-dates)
4. [Hull-White: caps, floors and swaptions](#4-hull-white-caps-floors-and-swaptions)
5. [G2++ and Bermudan swaptions](#5-g2-and-bermudan-swaptions)
6. [Credit default swaps on a time axis](#6-credit-default-swaps-on-a-time-axis)
7. [Standard CDS on real dates](#7-standard-cds-on-real-dates)
8. [CDO tranches](#8-cdo-tranches)
9. [Copulas](#9-copulas)
10. [Validation](#10-validation)
11. [Quick reference](#11-quick-reference)

## 1. Setup and conventions

```odin
import w "wotan/core"
import fin "wotan/finance"
import l "wotan/linalg"   // only for the tensor examples
import t "wotan/tensor"   // only for the tensor examples
import "core:math/rand"
```

The part 3 conventions carry over:
- `YieldCurve` holds continuously compounded zero rates. Dated curves measure time in years from their `reference_date` under ACT/365F.
- `Calendar`, `Period`, `Schedule` and `DayCountConvention` work as described there.

The rate models in sections 4 and 5 take a flat initial curve: `r0` is a continuously compounded rate, and P(0, t) = e^(−r0 t). All times are in years.

The Monte Carlo pricers use `core:math/rand`, so `rand.reset(seed)` makes a run reproducible. Several pricers accept an `n_steps` argument and ignore it: they sample the Gaussian factors exactly at the dates that matter, so there is no time-stepping error.

## 2. Swaps on a time axis

An `InterestRateSwap` has:
- a `notional`;
- an `effective_date` and a `maturity`, in years from today;
- a `payer_leg` and a `receiver_leg`. Each `SwapLeg` is `.Fixed` (with a `fixed_rate`) or `.Floating` (with a `floating_spread`), has a `payment_frequency` (years between payments) and a `day_count`.

Coupon dates step back from maturity. On a time axis, `.ACT_360` scales year fractions by 365/360; the other conventions use them as they are. `create_payer_swap` and `create_receiver_swap(notional, fixed_rate, maturity, payment_frequency = 0.25, day_count = .ACT_360)` build spot-starting swaps.

`price_swap(swap, &curve)` returns a `SwapValuationResult`:
- `pv_fixed_leg` and `pv_floating_leg`;
- `npv`, for whoever receives the receiver leg;
- `par_swap_rate`;
- `pv01` = notional × annuity × 1 bp;
- `modified_duration` and `convexity` of the fixed coupons. For continuous compounding these are Σ t·cf·DF / P and Σ t²·cf·DF / P.

The same curve projects and discounts.

```odin
curve := fin.bootstrap_yield_curve_from_swaps([]f64{0.25, 0.5, 1, 2, 3, 5, 7, 10},
	[]f64{0.0300, 0.0310, 0.0320, 0.0335, 0.0345, 0.0360, 0.0370, 0.0380})
defer fin.free_yield_curve(&curve)
payer := fin.create_payer_swap(10e6, 0.035, 5)
r := fin.price_swap(payer, &curve)
fmt.printf("5Y payer at 3.5%%: npv %.2f fixed %.2f float %.2f par %.10f pv01 %.2f dur %.6f\n",
	r.npv, r.pv_fixed_leg, r.pv_floating_leg, r.par_swap_rate, r.pv01, r.modified_duration)
fmt.printf("compute_par_swap_rate %.10f  pv01 %.2f  duration %.6f\n", fin.compute_par_swap_rate(5, &curve),
	fin.compute_swap_pv01(5, 10e6, &curve), fin.compute_swap_duration(5, &curve))
rec := fin.create_receiver_swap(10e6, 0.035, 5)
fmt.printf("receiver npv %.2f\n", fin.price_swap(rec, &curve).npv)
// seasoned: started 0.1 years ago, current quarterly coupon fixed at 3.1%
old := payer
old.effective_date, old.maturity = -0.1, 4.9
fmt.printf("seasoned without fixing: %v\n", fin.price_swap(old, &curve).npv)
old.current_fixing = 0.031
fmt.printf("seasoned with fixing 3.1%%: npv %.2f\n", fin.price_swap(old, &curve).npv)
```

```
5Y payer at 3.5%: npv 46300.48 fixed 1620516.63 float 1666817.10 par 0.0360000000 pv01 4630.05 dur 2.548440
compute_par_swap_rate 0.0360000000  pv01 4630.05  duration 4.591370
receiver npv -46300.48
seasoned without fixing: NaN
seasoned with fixing 3.1%: npv 39562.69
```

The curve was bootstrapped from these very swaps, so the 5-year par rate is the 3.6% quote.

The two durations measure different things:
- the result's `modified_duration` is that of the fixed coupons alone;
- `compute_swap_duration` treats the fixed side of a par swap as a par bond, coupons plus the notional at maturity, and gives its parallel-shift duration.

`compute_par_swap_rate`, `compute_swap_pv01` and `compute_swap_duration` work directly from a maturity.

**Seasoned swaps.** A swap with `effective_date < 0` started in the past. Coupons paid at or before t = 0 are left out. The floating coupon in progress was fixed at its start, so it needs `current_fixing`; without it the NPV is NaN rather than a number projected from the wrong period.

## 3. Swaps on real dates

`SwapConventions` describes a market's vanilla swap:
- `calendar` and `settlement_days` (the spot lag);
- the fixed leg's `fixed_tenor`, `fixed_convention` and `fixed_day_count`;
- the floating leg's `float_tenor`, `float_convention` and `float_day_count`;
- `end_of_month`;
- `payment_lag`, in business days after each accrual end;
- `fixing_days` and `indexed_coupons` for Ibor legs, and the overnight options described below.

`make_dated_swap(conv, start, tenor, fixed_rate, notional = 1, direction = .Payer, float_spread = 0)` builds both schedules, and `free_dated_swap` frees them. `swap_spot_date(conv, trade_date)` gives the usual start. `price_dated_swap(swap, &discount, &forecast = nil)` discounts on `discount` and projects floating coupons on `forecast`, or on `discount` when that is nil. It returns the same `SwapValuationResult` as section 2, and `npv` is for the side given by `direction`.

There are three kinds of floating coupon:

| Floating leg | Setting | Projected rate |
|---|---|---|
| Ibor, indexed | `indexed_coupons = true`, `fixing_days = 2` | the index rate over the index period: from the value date `fixing_days` after the fixing to value date + `float_tenor`, as QuantLib's `IborCoupon.createIndexedCoupons()` |
| Ibor, par | `indexed_coupons = false` | the forward over the accrual period itself, as QuantLib's default par coupons |
| Overnight (OIS) | `indexed_coupons = false` | the daily-compounded overnight rate over the accrual period, which telescopes to one forward; `lookback_days`, `lockout_days` and `observation_shift` give the variants of QuantLib's `OvernightIndexedCoupon` |

Below, a €STR curve is built from OIS quotes, and a 6M Euribor projection curve from a deposit and swaps, discounted on €STR (`discount = &ois`, see part 3). A 10-year swap is then priced on them.

```odin
today := w.Date{2026, 11, 11}
ois_conv := fin.SwapConventions {
	calendar         = {.TARGET},
	settlement_days  = 2,
	fixed_tenor      = {1, .Years},
	fixed_convention = .ModifiedFollowing,
	fixed_day_count  = .ACT_360,
	float_tenor      = {1, .Years},
	float_convention = .ModifiedFollowing,
	float_day_count  = .ACT_360,
	payment_lag      = 1,
}
ois_quotes := []fin.CurveQuote {
	{.Swap, {1, .Months}, 0.0190}, {.Swap, {3, .Months}, 0.0192}, {.Swap, {6, .Months}, 0.0195},
	{.Swap, {1, .Years}, 0.0198}, {.Swap, {2, .Years}, 0.0205}, {.Swap, {3, .Years}, 0.0212},
	{.Swap, {5, .Years}, 0.0225}, {.Swap, {10, .Years}, 0.0248}, {.Swap, {20, .Years}, 0.0265},
	{.Swap, {30, .Years}, 0.0260},
}
ois := fin.bootstrap_yield_curve_dated(today, ois_quotes, ois_conv)
defer fin.free_yield_curve(&ois)
e6_conv := fin.SwapConventions {
	calendar         = {.TARGET},
	settlement_days  = 2,
	fixed_tenor      = {1, .Years},
	fixed_convention = .ModifiedFollowing,
	fixed_day_count  = .THIRTY_360,
	float_tenor      = {6, .Months},
	float_convention = .ModifiedFollowing,
	float_day_count  = .ACT_360,
	fixing_days      = 2,
	indexed_coupons  = true,
}
e6_quotes := []fin.CurveQuote {
	{.Deposit, {6, .Months}, 0.0210}, {.Swap, {1, .Years}, 0.0215}, {.Swap, {2, .Years}, 0.0222},
	{.Swap, {3, .Years}, 0.0230}, {.Swap, {5, .Years}, 0.0245}, {.Swap, {10, .Years}, 0.0270},
	{.Swap, {20, .Years}, 0.0290}, {.Swap, {30, .Years}, 0.0285},
}
e6 := fin.bootstrap_yield_curve_dated(today, e6_quotes, e6_conv, &ois)
defer fin.free_yield_curve(&e6)
spot := fin.swap_spot_date(e6_conv, today)
sw := fin.make_dated_swap(e6_conv, spot, {10, .Years}, 0.0260, 10e6)
defer fin.free_dated_swap(&sw)
m := fin.price_dated_swap(sw, &ois, &e6)
fmt.printf("10Y payer 2.6%% vs 6M Euribor, OIS discounting: npv %.4f par %.12f pv01 %.4f\n", m.npv, m.par_swap_rate, m.pv01)
single := fin.price_dated_swap(sw, &e6)
fmt.printf("  discounted on the Euribor curve instead: npv %.4f par %.12f\n", single.npv, single.par_swap_rate)
o := fin.make_dated_swap(ois_conv, spot, {5, .Years}, 0.0225, 10e6, .Receiver)
defer fin.free_dated_swap(&o)
fmt.printf("5Y OIS receiver at the quote: npv %.6f par %.12f\n", fin.price_dated_swap(o, &ois).npv, fin.price_dated_swap(o, &ois).par_swap_rate)
```

```
10Y payer 2.6% vs 6M Euribor, OIS discounting: npv 88087.6696 par 0.027000000000 pv01 8808.7670
  discounted on the Euribor curve instead: npv 86865.6101 par 0.026994974442
5Y OIS receiver at the quote: npv 0.000000 par 0.022500000000
```

With OIS discounting, the 10-year swap's par rate is its 2.70% quote, and the NPV is 10 bp × PV01. Discounting the same swap on the Euribor curve gives a slightly different par rate and NPV, because the curve was built for OIS discounting.

QuantLib (`VanillaSwap`, `OvernightIndexedSwap`, `DiscountingSwapEngine`) gives the same numbers to all printed digits, with indexed coupons switched on.

**Seasoned swaps.** On real dates, flows paid on or before the curves' reference date are left out. The floating coupon in progress needs `current_fixing`:
- **Ibor coupon:** the fixing that has already been made. That applies when its fixing date is before the reference date; a fixing on the reference date itself is projected.
- **Overnight coupon:** the rate compounded from the accrual start to the reference date, simple in the float day count. The rest of the period is projected from the curve. This works for plain OIS legs only; with lookback, lockout or observation shift, a coupon in progress gives NaN.

Without `current_fixing`, the NPV is NaN, as with QuantLib's "missing fixing".

```odin
// seasoned Euribor swap: 5Y from 13 Nov 2024; the coupon from 13 May 2026
// fixed on 11 May 2026 at 2.15%
s5 := fin.make_dated_swap(e6_conv, {2024, 11, 13}, {5, .Years}, 0.0250, 10e6)
defer fin.free_dated_swap(&s5)
fmt.printf("seasoned Euribor swap without the fixing: %v\n", fin.price_dated_swap(s5, &ois, &e6).npv)
s5.current_fixing = 0.0215
q := fin.price_dated_swap(s5, &ois, &e6)
fmt.printf("  with it: npv %.4f fixed leg %.4f float leg %.4f\n", q.npv, q.pv_fixed_leg, q.pv_floating_leg)

// seasoned OIS: 3Y from 2 Mar 2026, €STR at 2% every day so far
s3 := fin.make_dated_swap(ois_conv, {2026, 3, 2}, {3, .Years}, 0.0200, 10e6)
defer fin.free_dated_swap(&s3)
growth := 1.0
for d := (w.Date{2026, 3, 2}); w.date_less(d, today); {
	next := fin.calendar_advance({.TARGET}, d, 1, .Days)
	growth *= 1 + 0.02 * fin.year_fraction(.ACT_360, d, next)
	d = next
}
s3.current_fixing = (growth - 1) / fin.year_fraction(.ACT_360, {2026, 3, 2}, today)
z := fin.price_dated_swap(s3, &ois)
fmt.printf("seasoned OIS (compounded so far %.10f): npv %.4f float leg %.4f\n", s3.current_fixing.?, z.npv, z.pv_floating_leg)
```

```
seasoned Euribor swap without the fixing: NaN
  with it: npv -197650.9544 fixed leg 969406.3834 float leg 771755.4290
seasoned OIS (compounded so far 0.0201406903): npv 17365.1854 float leg 610239.4357
```

QuantLib, given the same historical fixings, gives the same NPVs and leg values.

`dated_swap_last_relevant_date` gives the last date a swap depends on, which is the bootstrap's pillar.

## 4. Hull-White: caps, floors and swaptions

Hull-White one-factor (`HW_Params{a, sigma}`) is fitted to the flat curve e^(−r0 t).

**Closed forms.** A caplet on [T_s, T_e] is (1 + δK) puts on the bond P(T_s, T_e), with strike 1/(1 + δK). In a Gaussian model it has a Black-type formula on ln P. The procedures:
- `hw_price_cap_floor_strip(r0, K, T_start, T_end, delta, params, is_cap)` prices a strip. It returns `total_price` and `leg_prices`; free the latter.
- `hull_white_caplet_price_tensor` and `hull_white_floorlet_price_tensor(a, sigma, T_start, T_end, F, K, P, allocator)` take n × 1 tensors: forward rates F, strikes K and discount factors P(0, T_e). They price on any curve you supply F and P from, and give sensitivities by automatic differentiation. Free the result with `tensor_free_graph`.

**Monte Carlo.** `hw_mc_cap_option`, `hw_mc_floor_option` and `hw_mc_swaption_option` return the price and two Greeks on common random numbers:
- `delta_r`, the derivative with respect to r0 (a parallel curve shift);
- `vega`, with respect to σ.

The swaption is European, payer or receiver, on a swap paying K·δ at T_exp + jδ up to T_maturity.

```odin
col :: proc(v: []f64, grad := false) -> ^t.Tensor {
	m := l.matrix_new(f64, len(v), 1)
	copy(m.data, v)
	return t.tensor_new(m, grad)
}

rand.reset(7)
r0 := 0.03
hw := fin.HW_Params{a = 0.1, sigma = 0.01}
Ts := []f64{1, 2, 3, 4}
Te := []f64{2, 3, 4, 5}
d := []f64{1, 1, 1, 1}
cap := fin.hw_price_cap_floor_strip(r0, 0.032, Ts, Te, d, hw, true)
defer delete(cap.leg_prices)
flr := fin.hw_price_cap_floor_strip(r0, 0.032, Ts, Te, d, hw, false)
defer delete(flr.leg_prices)
fmt.printf("cap %.12f  caplets %.10f\n", cap.total_price, cap.leg_prices)
fmt.printf("floor %.12f  floorlets %.10f\n", flr.total_price, flr.leg_prices)
swap := 0.0
for i in 0 ..< 4 {swap += math.exp(-r0 * Ts[i]) - (1 + 0.032) * math.exp(-r0 * Te[i])}
fmt.printf("cap - floor %.15f, forward swap %.15f\n", cap.total_price - flr.total_price, swap)
p, dr, vega := fin.hw_mc_cap_option(r0, 0.032, Ts, Te, d, 4, hw, 200000, 0)
fmt.printf("MC cap %.6f  d/dr0 %.4f  vega %.4f\n", p, dr, vega)

// tensor caplets with autograd: sensitivities to the forwards
F := make([]f64, 4)
P := make([]f64, 4)
defer {delete(F); delete(P)}
for i in 0 ..< 4 {F[i] = math.exp(r0) - 1; P[i] = math.exp(-r0 * Te[i])}
tS, tE, tF, tK, tP := col(Ts), col(Te), col(F, true), col([]f64{0.032, 0.032, 0.032, 0.032}), col(P)
c := fin.hull_white_caplet_price_tensor(hw.a, hw.sigma, tS, tE, tF, tK, tP, context.allocator)
s := t.tensor_sum(c)
t.tensor_backward(s)
fmt.printf("caplets %.10f  dV/dF %.8f\n", c.data.data, tF.grad.data)
t.tensor_free_graph(s)
for x in ([]^t.Tensor{tS, tE, tF, tK, tP}) {t.tensor_free(x)}

// European swaption 2Y into 5Y, annual, K = 3%
sp, sdr, sv := fin.hw_mc_swaption_option(r0, 0.03, 2, 7, 1, true, hw, 200000, 0)
fmt.printf("HW payer swaption 2x5 MC %.6f  d/dr0 %.4f  vega %.4f\n", sp, sdr, sv)
rp, _, _ := fin.hw_mc_swaption_option(r0, 0.03, 2, 7, 1, false, hw, 200000, 0)
fmt.printf("receiver %.6f\n", rp)
```

```
cap 0.016262286758  caplets [0.0028302761, 0.0039220972, 0.0045585439, 0.0049513696]
floor 0.021831101859  floorlets [0.0042857412, 0.0053345468, 0.0059292493, 0.0062815646]
cap - floor -0.005568815100476, forward swap -0.005568815100476
MC cap 0.016313  d/dr0 1.6241  vega 1.8821
caplets [0.0028302761, 0.0039220972, 0.0045585439, 0.0049513696]  dV/dF [0.41069351, 0.41456594, 0.40894419, 0.40051347]
HW payer swaption 2x5 MC 0.018962  d/dr0 2.2122  vega 1.7945
receiver 0.017054
```

The caplets are QuantLib's `HullWhite.discountBondOption` values exactly, and cap − floor is the forward swap to 1e-15.

The Monte Carlo values have a standard deviation of about 2.5e-5 at 200 000 paths. Against the exact values the deviations are:
- cap: 0.016313 against 0.016262;
- payer swaption: 0.018962 against Jamshidian's 0.018986;
- receiver swaption: 0.017054 against 0.017028.

Averaged over 20 such runs, all three are within one standard error.

`dV/dF` comes from one backward pass through the tensor graph. It is the sensitivity of each caplet to its own forward rate, holding the discount factor fixed.

**Calibration.** `calibrate_hull_white(market_prices, T_start, T_end, F, K, P, n_caplets)` fits a and σ to caplet prices. It minimises the RMSE with Adam on log-parameters (a in [0.01, 0.5], σ in [0.001, 0.1]) and returns an `HW_CalibrationResult` (`params`, `rmse`, `converged`, `iterations`). Fitting the strip above back:

```odin
mkt := make([]f64, 4)
defer delete(mkt)
copy(mkt, cap.leg_prices)
K := []f64{0.032, 0.032, 0.032, 0.032}
cal := fin.calibrate_hull_white(mkt, Ts, Te, F, K, P, 4)
fmt.printf("calibrated a %.6f sigma %.6f  rmse %.2e  %d iterations\n", cal.params.a, cal.params.sigma, cal.rmse, cal.iterations)
```

```
calibrated a 0.099812 sigma 0.009996  rmse 5.55e-07  144 iterations
```

The optimizer stops when the RMSE drops below 1e-6, so a lies within 2e-4 of the true 0.1. Caplets alone pin a down only loosely, because a and σ trade off against each other.

## 5. G2++ and Bermudan swaptions

G2++ adds a second Gaussian factor (`HW2_Params{a, b, sigma, eta, rho}`): r = x + y + φ(t), with dx = −a x dt + σ dW1, dy = −b y dt + η dW2 and correlation ρ, fitted to e^(−r0 t). Hull-White is the case η = 0. The procedures:
- **Caplets.** `hull_white_2f_caplet_price_tensor(params, T_start, T_end, F, K, P, allocator)` is the closed form, and `hw2_mc_cap_option` prices caps by Monte Carlo, with Greeks.
- **European swaptions.** `hw2f_swaption_price_strip(params, specs, n, r0, allocator)` uses a frozen-weights approximation of the swap rate's normal variance with the Bachelier formula. Each `SwaptionSpec{T_exp, swap_length, F, K, annuity}` describes a swap with annual payments; you supply the forward swap rate F and the annuity. `compute_hw2f_swap_variance` gives the variance alone.
- **Bermudan swaptions.** `hw2f_mc_bermudan_option(r0, K, exercise_dates, maturity, delta, params, n_paths, n_steps)` prices a Bermudan payer swaption by Longstaff–Schwartz on exact G2++ paths. Each exercise enters a swap paying K·δ up to `maturity`. With a single exercise date it prices the European.
- **Bonds.** `hw2f_zero_coupon_bond(t, T, r_t, x_t, params, r0)` gives the bond price, given the short rate and the second factor.

The example continues the program of section 4, reusing `r0`, the caplet dates, F, K and P, and the `col` helper.

```odin
// G2++
g2 := fin.HW2_Params{a = 0.3, b = 0.05, sigma = 0.008, eta = 0.006, rho = -0.6}
tS, tE, tF, tK, tP = col(Ts), col(Te), col(F), col(K), col(P)
c2 := fin.hull_white_2f_caplet_price_tensor(g2, tS, tE, tF, tK, tP, context.allocator)
fmt.printf("G2 caplets %.10f\n", c2.data.data)
g2cap := 0.0
for v in c2.data.data {g2cap += v}
t.tensor_free_graph(c2)
for x in ([]^t.Tensor{tS, tE, tF, tK, tP}) {t.tensor_free(x)}
p2, _, _ := fin.hw2_mc_cap_option(r0, 0.032, Ts, Te, d, 4, g2, 200000, 0)
fmt.printf("G2 cap analytic %.6f  MC %.6f\n", g2cap, p2)
// swaption 2Y x 5Y: frozen-weights Bachelier vs a one-date Bermudan (= European) by MC
A := 0.0
for k in 1 ..= 5 {A += math.exp(-r0 * f64(2 + k))}
fwd := (math.exp(-r0 * 2) - math.exp(-r0 * 7)) / A
spec := []fin.SwaptionSpec{{T_exp = 2, swap_length = 5, F = fwd, K = 0.03, annuity = A}}
sw := fin.hw2f_swaption_price_strip(g2, spec, 1, r0, context.allocator)
defer delete(sw)
eu := fin.hw2f_mc_bermudan_option(r0, 0.03, []f64{2}, 7, 1, g2, 200000, 0)
fmt.printf("G2 swaption: forward %.10f annuity %.10f  frozen weights %.6f  MC %.6f\n", fwd, A, sw[0], eu)
berm := fin.hw2f_mc_bermudan_option(r0, 0.03, []f64{2, 3, 4, 5, 6}, 7, 1, g2, 200000, 0)
fmt.printf("G2 Bermudan 2..6 into 7Y: %.6f\n", berm)
hw1 := fin.HW2_Params{a = 0.1, b = 0.1, sigma = 0.01}
b1 := fin.hw2f_mc_bermudan_option(r0, 0.03, []f64{2, 3, 4, 5, 6}, 7, 1, hw1, 200000, 0)
e1 := fin.hw2f_mc_bermudan_option(r0, 0.03, []f64{2}, 7, 1, hw1, 200000, 0)
fmt.printf("HW Bermudan %.6f  European %.6f\n", b1, e1)
fmt.printf("G2 zero bond P(2, 7 | r = 4%%, y = 0.002) %.10f\n", fin.hw2f_zero_coupon_bond(2, 7, 0.04, 0.002, g2, r0))
```

```
G2 caplets [0.0013885978, 0.0019569639, 0.0023087371, 0.0025624881]
G2 cap analytic 0.008217  MC 0.008212
G2 swaption: forward 0.0304545340 annuity 4.3074140558  frozen weights 0.011234  MC 0.011185
G2 Bermudan 2..6 into 7Y: 0.014352
HW Bermudan 0.024550  European 0.019005
G2 zero bond P(2, 7 | r = 4%, y = 0.002) 0.8352566893
```

The reference values come from QuantLib, on a flat curve with times in exact years:

| Instrument | Here | QuantLib |
|---|---|---|
| G2 caplets | as printed | `G2.discountBondOption`, exactly |
| G2 European swaption | frozen weights 0.011234; Monte Carlo 0.011185 | `G2SwaptionEngine` 0.011231 |
| G2 Bermudan, exercises 2–6 into 7Y | 0.014352 | FD 0.014373; tree 0.014384 |
| Hull-White Bermudan | 0.024550 | FD 0.024610 |
| Hull-White European (one-date Bermudan) | 0.019005 | Jamshidian 0.018986 |

The frozen-weights approximation is within 0.03% of the exact G2 price. The single Monte Carlo runs are within two standard errors (about 2.4e-5). Longstaff–Schwartz Bermudans sit slightly below the exact value, because the regression gives a near-optimal exercise policy rather than the optimal one.

**Calibration.**
- `calibrate_hull_white_2f(market_prices, T_start, T_end, F, K, P, n)` fits all five parameters to caplets alone. Caplets cannot separate the two factors well, so expect several parameter sets with the same fit.
- `calibrate_hull_white_2f_joint(caps…, swaption_specs, swaption_market, n_swaptions, r0)` fits caplets and swaptions together. It minimises the sum of the relative RMSEs of both, keeps b < a − 0.01, and returns an `HW2F_JointResult`.

## 6. Credit default swaps on a time axis

The time-axis CDS engine uses piecewise-constant hazard rates and a flat continuous rate r. Per unit notional:
- premium annuity = Σ Δ_k S(t_k) D(t_k) + the premium accrued up to default;
- protection = (1 − R) ∫ D(u) dF(u).

Both integrals are exact on each piece where the hazard is constant. Coupon dates step back from maturity every `payment_freq` years. The procedures:
- **Flat hazard.** `price_cds(notional, spread, maturity, CDS_Params{lambda, recovery, r}, payment_freq)` gives the buyer's PV for a flat hazard.
- **Bootstrap.** `bootstrap_cds_curve(tenors, spreads, recovery, r, payment_freq)` returns a `CDS_Curve` whose hazards reprice each par spread exactly. Free its `tenors` and `hazards`.
- **Survival.** `get_survival_prob(t, curve)` gives the survival probability; the last hazard extends beyond the last tenor.
- **Pricing on the curve.** `price_cds_full(notional, coupon, maturity, curve, payment_freq)` returns:
  - `upfront_pct`: protection − coupon × annuity, per unit notional;
  - `rpv01`: per 1M notional, whatever the notional argument;
  - `premium_leg_pv` and `protection_pv`.

```odin
p := fin.CDS_Params{lambda = 0.02, recovery = 0.4, r = 0.03}
fmt.printf("5Y CDS at 100bp, hazard 2%%: buyer PV %.6f per 10M\n", fin.price_cds(10e6, 0.01, 5, p, 0.25))
fmt.printf("  at 120bp: %.6f\n", fin.price_cds(10e6, 0.012, 5, p, 0.25))
tenors := []f64{1, 3, 5, 7, 10}
spreads := []f64{0.0060, 0.0085, 0.0110, 0.0125, 0.0135}
curve := fin.bootstrap_cds_curve(tenors, spreads, 0.4, 0.03, 0.25)
defer {delete(curve.tenors); delete(curve.hazards)}
fmt.printf("hazards %.10f\n", curve.hazards)
fmt.printf("survival 1Y %.10f 5Y %.10f 12Y %.10f\n", fin.get_survival_prob(1, curve), fin.get_survival_prob(5, curve), fin.get_survival_prob(12, curve))
for T, i in tenors {
	r := fin.price_cds_full(10e6, spreads[i], T, curve, 0.25)
	fmt.printf("  %2.0fY at its par spread: upfront %.2e\n", T, r.upfront_pct)
}
r := fin.price_cds_full(10e6, 0.01, 5, curve, 0.25)
fmt.printf("5Y with a 100bp coupon: upfront %.8f rpv01 %.4f premium %.4f protection %.4f\n", r.upfront_pct, r.rpv01, r.premium_leg_pv, r.protection_pv)
```

```
5Y CDS at 100bp, hazard 2%: buyer PV 90135.224670 per 10M
  at 120bp: 1986.645478
hazards [0.0099625624, 0.0163308313, 0.0253132646, 0.0282831214, 0.0275577865]
survival 1Y 0.9900868995 5Y 0.9109650433 12Y 0.7500571094
   1Y at its par spread: upfront -8.67e-19
   3Y at its par spread: upfront 2.78e-17
   5Y at its par spread: upfront -5.55e-17
   7Y at its par spread: upfront -4.16e-17
  10Y at its par spread: upfront -1.39e-17
5Y with a 100bp coupon: upfront 0.00445702 rpv01 445.7024 premium 445702.4158 protection 490272.6574
```

With a 2% hazard and 40% recovery, the fair spread is close to λ(1 − R) = 120 bp; at 120 bp the buyer's PV is almost zero. Python numerical integration of the same legs gives the prices, the hazards and the 5-year legs above to all printed digits.

## 7. Standard CDS on real dates

`DatedCDS` is the post-2009 standard contract:
- quarterly coupons on the IMM 20th, with a CDS2015 schedule (part 3);
- ACT/360 accrual, with the last period including its end date;
- a full first coupon, with the accrued part rebated to the buyer;
- protection from the step-in date (trade + 1);
- an upfront paid three business days after the trade.

`price_dated_cds(cds, &hazard, &discount)` values it as QuantLib's `IsdaCdsEngine` does (the ISDA CDS Standard Model). It returns a `DatedCDSResult`:
- `npv`, for the CDS's `side`;
- the leg values, `accrual_rebate` and `upfront_pv`;
- `fair_spread` and `fair_upfront`;
- `risky_annuity`.

The other procedures:
- **Building the contract.** `make_standard_cds(trade_date, tenor, coupon, notional = 1, side = .Buyer, upfront = 0)` builds it; free it with `free_dated_cds`.
- **Hazard bootstrap.** `bootstrap_cds_curve_dated(tenors, spreads, recovery, &discount)` bootstraps ISDA hazards from par spreads, and `bootstrap_cds_curve_dated_upfront` from upfront quotes at a common coupon.
- **Quote conversion.** `cds_upfront_from_quoted_spread` and `cds_quoted_spread_from_upfront` convert quotes using the ISDA flat-hazard convention.
- **Indices.** `price_cds_index_intrinsic(index, constituents, weights, &discount)` values a CDX or iTraxx position name by name.

The example continues section 6, reusing its `spreads`. The discount curve is a flat-forward deposit-and-swap curve, as the ISDA model uses:

```odin
today := w.Date{2026, 11, 11}
conv := fin.SwapConventions{calendar = {.TARGET}, settlement_days = 2, fixed_tenor = {1, .Years}, fixed_convention = .ModifiedFollowing,
	fixed_day_count = .THIRTY_360, float_tenor = {6, .Months}, float_convention = .ModifiedFollowing, float_day_count = .ACT_360}
quotes := []fin.CurveQuote{{.Deposit, {1, .Months}, 0.0195}, {.Deposit, {3, .Months}, 0.0200}, {.Deposit, {6, .Months}, 0.0210},
	{.Swap, {1, .Years}, 0.0215}, {.Swap, {2, .Years}, 0.0222}, {.Swap, {3, .Years}, 0.0230}, {.Swap, {5, .Years}, 0.0245},
	{.Swap, {7, .Years}, 0.0257}, {.Swap, {10, .Years}, 0.0270}}
disc := fin.bootstrap_yield_curve_dated(today, quotes, conv, interpolation = .FlatForward)
defer fin.free_yield_curve(&disc)
cds_tenors := []fin.Period{{1, .Years}, {3, .Years}, {5, .Years}, {7, .Years}, {10, .Years}}
hz := fin.bootstrap_cds_curve_dated(cds_tenors, spreads, 0.4, &disc)
defer {delete(hz.tenors); delete(hz.hazards)}
fmt.printf("dated hazards %.10f\n", hz.hazards)
cds := fin.make_standard_cds(today, {5, .Years}, 0.01, 10e6)
defer fin.free_dated_cds(&cds)
fmt.println("maturity", cds.schedule.dates[len(cds.schedule.dates) - 1], "first coupon", cds.schedule.dates[1], "upfront date", cds.upfront_date)
res := fin.price_dated_cds(cds, &hz, &disc)
fmt.printf("5Y 100bp buyer: npv %.4f protection %.4f premium %.4f rebate %.4f\n  fair spread %.10f fair upfront %.10f risky annuity %.8f\n",
	res.npv, res.protection_leg, res.premium_leg, res.accrual_rebate, res.fair_spread, res.fair_upfront, res.risky_annuity)
up := fin.cds_upfront_from_quoted_spread({5, .Years}, 0.01, 0.0110, 0.4, &disc)
back := fin.cds_quoted_spread_from_upfront({5, .Years}, 0.01, up, 0.4, &disc)
fmt.printf("quoted 110bp -> upfront %.10f -> spread %.12f\n", up, back)
// paying that upfront makes the trade fair on a flat 110bp curve
traded := fin.make_standard_cds(today, {5, .Years}, 0.01, 10e6, upfront = up)
defer fin.free_dated_cds(&traded)
flat := fin.bootstrap_cds_curve_dated([]fin.Period{{5, .Years}}, []f64{0.0110}, 0.4, &disc)
defer {delete(flat.tenors); delete(flat.hazards)}
fmt.printf("with the upfront on the flat curve: npv %.6f\n", fin.price_dated_cds(traded, &flat, &disc).npv)
// a 3-name index, one name with a wider curve
wide := fin.bootstrap_cds_curve_dated(cds_tenors, []f64{0.02, 0.03, 0.04, 0.045, 0.05}, 0.4, &disc)
defer {delete(wide.tenors); delete(wide.hazards)}
idx := fin.make_standard_cds(today, {5, .Years}, 0.01, 30e6)
defer fin.free_dated_cds(&idx)
ir := fin.price_cds_index_intrinsic(idx, []^fin.CDS_Curve{&hz, &hz, &wide}, []f64{1.0 / 3, 1.0 / 3, 1.0 / 3}, &disc)
fmt.printf("index: npv %.4f fair spread %.10f fair upfront %.10f\n", ir.npv, ir.fair_spread, ir.fair_upfront)
```

```
dated hazards [0.0101102470, 0.0167951888, 0.0258783862, 0.0287842000, 0.0279883437]
maturity Date{year = 2031, month = 12, day = 20} first coupon Date{year = 2026, month = 12, day = 21} upfront date Date{year = 2026, month = 11, day = 16}
5Y 100bp buyer: npv 46875.9152 protection 515635.0676 premium 483199.6886 rebate 14440.5362
  fair spread 0.0110000000 fair upfront 0.0046888602 risky annuity 4.83199689
quoted 110bp -> upfront 0.0046535613 -> spread 0.011000000000
with the upfront on the flat curve: npv 0.000000
index: npv 1370397.1029 fair spread 0.0200537785 fair upfront 0.0456922664
```

A 5-year CDS traded on 11 November 2026 matures on 20 December 2031 (part 3). Its first coupon date, 20 December 2026, is a Sunday, so it rolls to Monday the 21st.

The quoted-spread upfront (0.4654%) differs from the fair upfront on the term-structured curve (0.4689%). The ISDA quote convention values the CDS on a flat hazard curve, not on the full term structure.

The dated hazards differ from the time-axis ones of section 6 for the same spreads. Real coupon dates, ACT/360 accruals, the actual discount curve and the step-in date all enter.

## 8. CDO tranches

`price_cdo_tranche_mc(tranche, params, spread, n_paths)` and `price_cdo_tranche_t_copula_mc(tranche, params, nu, spread, n_paths)` price a tranche of a homogeneous pool (`CDO_Params{N, lambda, recovery, r, rho, maturity, payment_freq}`):
- Each name defaults when its latent variable crosses its threshold, with exponential default times.
- The latent variables follow a one-factor Gaussian or Student-t copula with correlation `rho`.
- The tranche loss (min(L, d) − min(L, a)) / (d − a) is tracked at each coupon date. Protection is paid mid-period, and premium is paid on the outstanding tranche notional.

Both pricers return the PV to the protection buyer per unit tranche notional, and the expected tranche loss at maturity, `expected_loss_pct`.

```odin
rand.reset(11)
cdo := fin.CDO_Params{N = 125, lambda = 0.02, recovery = 0.4, r = 0.03, rho = 0.3, maturity = 5, payment_freq = 0.25}
for tr in ([]fin.CDO_Tranche{{"equity", 0, 0.03}, {"mezzanine", 0.03, 0.07}, {"senior", 0.07, 0.15}}) {
	pv, el := fin.price_cdo_tranche_mc(tr, cdo, 0.05, 100000)
	pvt, elt := fin.price_cdo_tranche_t_copula_mc(tr, cdo, 4, 0.05, 100000)
	fmt.printf("%-9s gaussian pv %.6f EL %.6f   t(4) pv %.6f EL %.6f\n", tr.name, pv, el, pvt, elt)
}
```

```
equity    gaussian pv 0.560767 EL 0.722239   t(4) pv 0.418268 EL 0.606020
mezzanine gaussian pv 0.180190 EL 0.397600   t(4) pv 0.136169 EL 0.351943
senior    gaussian pv -0.060930 EL 0.169914   t(4) pv -0.045955 EL 0.179892
```

Fat tails in the t copula make joint defaults more likely. That moves expected loss from the equity tranche to the senior tranche, while the pool's total expected loss stays the same.

At a 500 bp running spread, the equity buyer gains, because its expected loss far exceeds the premium; the senior buyer overpays.

The exact values (a conditional-binomial expectation over the common factor) are:

| Tranche | Gaussian | t(4) |
|---|---|---|
| equity | 0.559764 | 0.416017 |
| mezzanine | 0.181906 | 0.134365 |
| senior | −0.061531 | −0.044943 |

One run of 100 000 paths has a standard deviation of 1–2e-3, and the means of ten runs are within two standard errors of these values.

## 9. Copulas

The copula functions work on pseudo-observations u ∈ (0, 1):
- `pit_empirical(data)` gives ranks / (n + 1), averaging ranks over ties.
- `pit_normal(data)` gives Φ(x), for data that is already standard normal.

| Family | Fit | Log-likelihood | Parameters |
|---|---|---|---|
| Gaussian | `fit_gaussian_copula` | `gaussian_copula_loglik`, `gaussian_copula_pdf` | ρ |
| Student-t | `fit_student_t_copula` | `student_t_copula_loglik`, `student_t_copula_log_pdf` | ρ, ν ∈ [1, 200] |
| Clayton (lower tail) | `fit_clayton` | `clayton_copula_loglik` | θ ∈ (0, 200] |
| Gumbel (upper tail) | `fit_gumbel` | `gumbel_copula_loglik` | θ ∈ [1, 200] |
| Frank (no tail) | `fit_frank` | `frank_copula_loglik` | θ ∈ [−200, 200] |

All fits are maximum likelihood. Each returns a `CopulaResult` with `parameters` (free it), `log_likelihood`, `aic`, `bic`, `converged` and `n_obs`, so the families can be compared directly.

Rank dependence and tail dependence:
- `kendall_tau` is τ_b, as in SciPy;
- `spearman_rho(x, y, allocator)` is Spearman's ρ;
- `lower_tail_dependence` and `upper_tail_dependence` estimate P(U2 < q | U1 < q) and P(U2 > q | U1 > q) at a finite threshold (0.05 and 0.95 by default).

```odin
// 2000 draws from a t copula with ρ = 0.6, ν = 4, on t(4) and normal margins
rand.reset(3)
n := 2000
x := make([]f64, n)
y := make([]f64, n)
defer {delete(x); delete(y)}
for i in 0 ..< n {
	z1 := rand.float64_normal(0, 1)
	z2 := 0.6 * z1 + math.sqrt(1 - 0.36) * rand.float64_normal(0, 1)
	w := 0.0
	for _ in 0 ..< 4 {g := rand.float64_normal(0, 1); w += g * g}
	s := math.sqrt(w / 4)
	x[i] = z1 / s // t(4) margin
	y[i] = 0.01 * math.exp(z2 / s) // some other monotone margin
}
u1 := fin.pit_empirical(x)
u2 := fin.pit_empirical(y)
defer {delete(u1); delete(u2)}
fmt.printf("Kendall tau %.10f  Spearman rho %.10f\n", fin.kendall_tau(x, y), fin.spearman_rho(x, y, context.allocator))
fmt.printf("empirical tail dependence: lower %.4f upper %.4f\n", fin.lower_tail_dependence(u1, u2), fin.upper_tail_dependence(u1, u2))
fits := [5]fin.CopulaResult {
	fin.fit_gaussian_copula(u1, u2),
	fin.fit_student_t_copula(u1, u2),
	fin.fit_clayton(u1, u2),
	fin.fit_gumbel(u1, u2),
	fin.fit_frank(u1, u2),
}
for &f in fits {
	fmt.printf("%-9v params %.6f  loglik %.4f  AIC %.4f  BIC %.4f\n", f.copula_type, f.parameters, f.log_likelihood, f.aic, f.bic)
	delete(f.parameters)
}
fmt.printf("Gaussian copula density at (0.9, 0.9), rho 0.6: %.10f\n", fin.gaussian_copula_pdf(0.9, 0.9, 0.6))
fmt.printf("t copula log density at (0.9, 0.9), rho 0.6, nu 4: %.10f\n", fin.student_t_copula_log_pdf(0.9, 0.9, 0.6, 4))
```

```
Kendall tau 0.4086223112  Spearman rho 0.5715250189
empirical tail dependence: lower 0.3900 upper 0.3500
Gaussian  params [0.603645]  loglik 449.6236  AIC -897.2472  BIC -891.6463
StudentT  params [0.603262, 6.326931]  loglik 473.9689  AIC -943.9378  BIC -932.7360
Clayton   params [1.020972]  loglik 380.6026  AIC -759.2052  BIC -753.6043
Gumbel    params [1.633330]  loglik 423.0056  AIC -844.0113  BIC -838.4104
Frank     params [4.330716]  loglik 405.8133  AIC -809.6266  BIC -804.0257
Gaussian copula density at (0.9, 0.9), rho 0.6: 2.3141303472
t copula log density at (0.9, 0.9), rho 0.6, nu 4: 1.0059639941
```

Copulas don't depend on the margins: the rank measures and fits would be the same for any monotone transformation of x and y.

The t copula wins on AIC and BIC, and recovers ρ = 0.60. Its ν estimate (6.3) is noisy at n = 2000, since ν is set by a few joint extremes.

Both elliptical fits give ρ ≈ 0.60, close to sin(πτ/2) = 0.599 from Kendall's τ. That relation holds for every elliptical copula.

SciPy (`kendalltau`, `spearmanr`, and maximum likelihood on the same log-densities) gives every number above. The t density agrees to 3e-10.

## 10. Validation

**Reference tests in the suite:**

| Test | Compares |
|---|---|
| `swaps_yield_curve_reference_test`, `swaps_test` | time-axis bootstrap and swap valuation |
| `dated_instruments_reference_test` | dated bootstraps, OIS and Euribor multi-curve, `VanillaSwap`, overnight coupons with lookback, lockout and observation shift, against QuantLib |
| `ir_derivatives_reference_test` and the other `ir_derivatives_…`, `hull_white_contracts_test`, `bermudan_swaption_test` | Hull-White and G2++ caplets, floorlets, swaptions and Bermudans against QuantLib and exact Gaussian integration |
| `dated_cds_reference_test` | flat-forward curves, ISDA hazard bootstraps from spreads and upfronts, standard CDS values, quote conversions and an index, against `IsdaCdsEngine` |
| `credit_derivatives_reference_test`, `cds_bootstrapping_test` | time-axis CDS legs and bootstraps, CDO tranches |
| `copula_reference_test` | all five families, rank correlations and tail measures, against SciPy |

**Additional checks:**

| What | Reference | Agreement |
|---|---|---|
| 10Y swap on €STR and 6M Euribor curves (OIS and single-curve discounting); 5Y OIS | QuantLib `VanillaSwap` (indexed coupons), `OvernightIndexedSwap` | all printed digits |
| seasoned Euribor swap with its fixing; seasoned OIS with past €STR fixings | QuantLib with the same fixings | all printed digits |
| HW and G2++ caplets and floorlets | QuantLib `HullWhite` and `G2.discountBondOption` | 1e-10 (all printed digits) |
| cap − floor | the forward swap | 1e-15 |
| HW cap and European swaptions by Monte Carlo, 20 × 200 000 paths | closed form, Jamshidian | within one standard error |
| G2++ frozen-weights swaption | `G2SwaptionEngine` | 3e-6 (0.03%) |
| HW and G2++ Bermudans by Longstaff–Schwartz | QuantLib FD and tree engines | 2–6e-5 below (the low bias of the method) |
| time-axis CDS prices, hazard bootstrap, legs | Python numerical integration | all printed digits |
| CDO tranches, Gaussian and t(4) | exact conditional-binomial expectation | within two standard errors |
| copula fits, log-likelihoods, Kendall, Spearman, tail estimates, densities | SciPy | all printed digits (t density 3e-10) |

**Known limitations:**
- **Flat rate-model curve.** The Hull-White and G2++ Monte Carlo pricers, `hw_price_cap_floor_strip`, `compute_hw2f_swap_variance` and `hw2f_mc_bermudan_option` take a flat initial curve r0. The tensor caplet pricers accept any curve through F and P.
- **Swaption scope.** `SwaptionSpec` and the Bermudan pricer assume annual (or `delta`-spaced) fixed payments; the Bermudan is payer only.
- **Overnight coupons in progress.** Seasoned overnight coupons with lookback, lockout or observation shift give NaN.
- **Ignored inputs.** `n_steps` is ignored throughout (sampling is exact), and `CDS_Price_Result.rpv01` is per 1M notional whatever the notional argument.
- **CDO pools.** The CDO pricers assume a homogeneous pool.

## 11. Quick reference

| Task | Procedure | Section |
|---|---|---|
| Time-axis swaps | `create_payer_swap`, `create_receiver_swap`, `price_swap`, `compute_par_swap_rate`, `compute_swap_pv01`, `compute_swap_duration`, `day_count_fraction` | 2 |
| Dated swaps | `make_dated_swap`, `price_dated_swap`, `swap_spot_date`, `swap_maturity_date`, `dated_swap_last_relevant_date`, `free_dated_swap` | 3 |
| Hull-White caps and floors | `hw_price_cap_floor_strip`, `hull_white_caplet_price_tensor`, `hull_white_floorlet_price_tensor`, `hw_mc_cap_option`, `hw_mc_floor_option` | 4 |
| Hull-White swaptions | `hw_mc_swaption_option` | 4 |
| Hull-White calibration | `calibrate_hull_white` | 4 |
| G2++ | `hull_white_2f_caplet_price_tensor`, `hw2_mc_cap_option`, `hw2f_swaption_price_strip`, `compute_hw2f_swap_variance`, `hw2f_mc_bermudan_option`, `hw2f_zero_coupon_bond` | 5 |
| G2++ calibration | `calibrate_hull_white_2f`, `calibrate_hull_white_2f_joint` | 5 |
| Time-axis CDS | `price_cds`, `bootstrap_cds_curve`, `get_survival_prob`, `price_cds_full` | 6 |
| Standard CDS | `make_standard_cds`, `price_dated_cds`, `bootstrap_cds_curve_dated`, `bootstrap_cds_curve_dated_upfront`, `free_dated_cds` | 7 |
| CDS quotes and indices | `cds_upfront_from_quoted_spread`, `cds_quoted_spread_from_upfront`, `price_cds_index_intrinsic` | 7 |
| CDO tranches | `price_cdo_tranche_mc`, `price_cdo_tranche_t_copula_mc`, `student_t_cdf` | 8 |
| Copula fits | `fit_gaussian_copula`, `fit_student_t_copula`, `fit_clayton`, `fit_gumbel`, `fit_frank` and the `…_loglik` functions | 9 |
| Dependence measures | `pit_empirical`, `pit_normal`, `kendall_tau`, `spearman_rho`, `lower_tail_dependence`, `upper_tail_dependence` | 9 |
