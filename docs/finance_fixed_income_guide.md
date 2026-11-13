# The `finance` package, part 3: dates, curves, bonds and inflation

This is the third part of the guide to `wotan/finance`. [Part 1](finance_options_guide.md) covers options and Monte Carlo, and [part 2](finance_volatility_guide.md) covers volatility models, exotics and FX. This part covers the fixed-income building blocks:
- holiday calendars, business-day conventions, day counters, coupon schedules, IMM and CDS dates;
- yield curves bootstrapped from deposits and swaps, on a time axis or on real dates;
- fixed-rate bonds: price, yield, accrued interest, duration and convexity, on a time axis or on real dates;
- callable and puttable bonds under Hull-White, with effective duration and option-adjusted spread;
- convertible bonds on a Tsiveriotis-Fernandes lattice;
- inflation: CPI fixings and forecasts, zero-inflation curves, zero-coupon inflation swaps and inflation-linked bonds.

Swaps, caps, swaptions and credit default swaps are in part 4. This part only uses `SwapConventions` (section 5) to describe the swaps a curve is bootstrapped from.

Every example was compiled and run against the current code, and the output shown is the real output. The results agree with QuantLib 1.44 and an independent Python implementation; section 11 lists the comparisons. A debug build with a tracking allocator reports no leaks for any of the examples.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [Calendars and business days](#2-calendars-and-business-days)
3. [Day counters](#3-day-counters)
4. [Schedules, IMM and CDS dates](#4-schedules-imm-and-cds-dates)
5. [Yield curves](#5-yield-curves)
6. [Bonds on a time axis](#6-bonds-on-a-time-axis)
7. [Bonds on real dates](#7-bonds-on-real-dates)
8. [Callable and puttable bonds](#8-callable-and-puttable-bonds)
9. [Convertible bonds](#9-convertible-bonds)
10. [Inflation](#10-inflation)
11. [Validation](#11-validation)
12. [Quick reference](#12-quick-reference)

## 1. Setup and conventions

```odin
import w "wotan/core"
import fin "wotan/finance"
```

Dates are the core package's `w.Date{year, month, day}`. A zero `w.Date{}` means "no date" wherever a date parameter is optional. A length of time is a `Period{n, unit}` with `unit` one of `.Days`, `.Weeks`, `.Months` and `.Years`, so `{6, .Months}` is six months.

The package has two kinds of fixed-income objects:
- **Time-axis objects** (`YieldCurve` from `bootstrap_yield_curve_from_swaps`, `Bond`, `CallableBond`, `PuttableBond`) measure everything in years from today. They are quick to set up and good for models and teaching.
- **Dated objects** (`Schedule`, `DatedBond`, `ConvertibleBond`, `ZeroCouponInflationSwap`, `InflationLinkedBond`, and curves from `bootstrap_yield_curve_dated`) work on real dates with calendars and day counters, following QuantLib's conventions. Use them to price real instruments.

Dated curves still store times, in years from their `reference_date` under ACT/365 Fixed (`date_to_time`). `yield_curve_discount_factor_date` converts a date for you.

Rates on curves are continuously compounded. Bond yields are compounded at the coupon frequency, as markets quote them.

## 2. Calendars and business days

A `Calendar` is a `bit_set` of markets: `.Weekends`, `.TARGET` (euro area), `.US_Settlement`, `.US_Government_Bond` (SIFMA) and `.UK_Settlement`. A day is a business day when it is one in every market of the set, so `{.TARGET, .UK_Settlement}` is the joint calendar. The empty set `{}` is the null calendar, where every day is a business day.

```odin
cals := [4]fin.Calendar{{.TARGET}, {.US_Settlement}, {.US_Government_Bond}, {.UK_Settlement}}
names := [4]string{"TARGET", "US settlement", "US government bond", "UK settlement"}
for cal, k in cals {
	fmt.printf("%-19s", names[k])
	for d := (w.Date{2026, 1, 1}); d.year == 2026; d = w.add_day_date(d, 1) {
		wd := w.date_weekday(d)
		if wd != .Saturday && wd != .Sunday && fin.is_holiday(cal, d) {
			fmt.printf(" %02d-%02d", d.month, d.day)
		}
	}
	fmt.println()
}
```

```
TARGET              01-01 04-03 04-06 05-01 12-25
US settlement       01-01 01-19 02-16 05-25 06-19 07-03 09-07 10-12 11-11 11-26 12-25
US government bond  01-01 01-19 02-16 05-25 06-19 07-03 09-07 10-12 11-11 11-26 12-25
UK settlement       01-01 04-03 04-06 05-04 05-25 08-31 12-25 12-28
```

The two US calendars agree in 2026 because Good Friday (3 April) is the first Friday of April. SIFMA keeps the bond market open on that day because the payroll report comes out; in other years the government bond calendar also closes on Good Friday. Independence Day falls on a Saturday and is observed on Friday 3 July, and the UK moves Boxing Day from Saturday 26 December to Monday 28 December.

`calendar_adjust(cal, d, bdc)` moves a non-business day according to a `BusinessDayConvention`. Sunday 31 May 2026 is the last day of its month:

```odin
sun := w.Date{2026, 5, 31}
for bdc in fin.BusinessDayConvention {
	fmt.printf("%-27v %v\n", bdc, fin.calendar_adjust({.TARGET}, sun, bdc))
}
```

```
Following                   Date{year = 2026, month = 6, day = 1}
ModifiedFollowing           Date{year = 2026, month = 5, day = 29}
Preceding                   Date{year = 2026, month = 5, day = 29}
ModifiedPreceding           Date{year = 2026, month = 5, day = 29}
Unadjusted                  Date{year = 2026, month = 5, day = 31}
HalfMonthModifiedFollowing  Date{year = 2026, month = 5, day = 29}
Nearest                     Date{year = 2026, month = 6, day = 1}
```

Modified following moves back to Friday 29 May, because following would cross into June. Nearest picks the closer business day, and following on a tie.

`calendar_advance(cal, d, n, unit, bdc, end_of_month)` works like QuantLib's `Calendar::advance`. Days count business days. Weeks, months and years are calendar arithmetic followed by the adjustment. With `end_of_month`, a date on the last business day of its month maps to the last business day of the target month. `business_days_between(cal, from, to, include_first = true, include_last = false)` counts business days.

```odin
feb := w.Date{2026, 2, 27} // last business day of February
fmt.println("Feb 27 + 1M:", fin.calendar_advance({.TARGET}, feb, 1, .Months),
	"with end_of_month:", fin.calendar_advance({.TARGET}, feb, 1, .Months, end_of_month = true))
fmt.println("Dec 23 + 3 business days (TARGET):", fin.calendar_advance({.TARGET}, w.Date{2026, 12, 23}, 3, .Days))
fmt.println("Dec 23 + 3 business days (UK):", fin.calendar_advance({.UK_Settlement}, w.Date{2026, 12, 23}, 3, .Days))
fmt.println("TARGET business days in 2026:",
	fin.business_days_between({.TARGET}, w.Date{2026, 1, 1}, w.Date{2027, 1, 1}))
fmt.println("TARGET + UK business days in 2026:",
	fin.business_days_between({.TARGET, .UK_Settlement}, w.Date{2026, 1, 1}, w.Date{2027, 1, 1}))
```

```
Feb 27 + 1M: Date{year = 2026, month = 3, day = 27} with end_of_month: Date{year = 2026, month = 3, day = 31}
Dec 23 + 3 business days (TARGET): Date{year = 2026, month = 12, day = 29}
Dec 23 + 3 business days (UK): Date{year = 2026, month = 12, day = 30}
TARGET business days in 2026: 256
TARGET + UK business days in 2026: 252
```

Other helpers:
- `is_business_day` is the opposite of `is_holiday`.
- `calendar_end_of_month` and `calendar_is_end_of_month` give the last business day of a month and test for it.
- `calendar_advance_period` takes a `Period`.
- `date_add_period` is plain calendar arithmetic with no adjustment.
- `easter_sunday(year)` gives Easter Sunday; the holiday rules use it.

## 3. Day counters

`year_fraction(dc, d1, d2, ref_start, ref_end, termination)` is QuantLib's `DayCounter::yearFraction`, and `day_count_days(dc, d1, d2, termination)` gives the day count. The conventions are:

| `DayCountConvention` | Meaning |
|---|---|
| `.ACT_360`, `.ACT_365` | Actual/360 and Actual/365 Fixed |
| `.ACT_ACT` | Actual/Actual ISDA: days in each calendar year over that year's length |
| `.ACT_ACT_ICMA` | Actual/Actual ICMA: actual days over the coupon's reference period, in units of the coupon period |
| `.THIRTY_360` | 30/360 Bond Basis |
| `.THIRTY_360_US` | 30/360 US, with the end-of-February rules |
| `.THIRTY_E_360` | 30E/360 (Eurobond basis) |
| `.THIRTY_E_360_ISDA` | 30E/360 ISDA (German); `termination`, the maturity, keeps its 28/29 February |

```odin
a, b := w.Date{2026, 2, 28}, w.Date{2026, 8, 31}
for dc in fin.DayCountConvention {
	fmt.printf("%-18v %4d days  %.10f\n", dc, fin.day_count_days(dc, a, b), fin.year_fraction(dc, a, b))
}
// ICMA: a short first coupon 15 Mar 2026 - 15 Jul 2026 of a semi-annual bond
fmt.printf("ACT/ACT ICMA stub %.10f, ISDA %.10f\n",
	fin.year_fraction(.ACT_ACT_ICMA, w.Date{2026, 3, 15}, w.Date{2026, 7, 15}, w.Date{2026, 1, 15}, w.Date{2026, 7, 15}),
	fin.year_fraction(.ACT_ACT, w.Date{2026, 3, 15}, w.Date{2026, 7, 15}))
```

```
ACT_360             184 days  0.5111111111
ACT_365             184 days  0.5041095890
ACT_ACT             184 days  0.5041095890
THIRTY_360          183 days  0.5083333333
ACT_ACT_ICMA        184 days  0.5000000000
THIRTY_360_US       180 days  0.5000000000
THIRTY_E_360        182 days  0.5055555556
THIRTY_E_360_ISDA   180 days  0.5000000000
ACT/ACT ICMA stub 0.3370165746, ISDA 0.3342465753
```

28 February is the last day of February in 2026. The US and German 30/360 conventions treat it as the 30th, so they give exactly half a year; Bond Basis and 30E/360 don't.

Without a reference period, ICMA uses [d1, d2] itself and rounds it to whole months, which is why it gives exactly 0.5 here. Bond code passes each coupon's reference period from the schedule (`schedule_reference_period`, section 4). For the stub, ICMA counts the 122 days against the 181-day regular period 15 Jan – 15 Jul, giving ½ · 122/181 = 0.3370.

The time-axis code (`Bond`, `bootstrap_yield_curve_from_swaps`) has no dates. It uses `day_count_fraction(t1, t2, dc)`, which is `Δt · 365/360` for `.ACT_360` and `Δt` for every other convention.

## 4. Schedules, IMM and CDS dates

`make_schedule(effective, termination, tenor, calendar, convention, termination_convention, rule, end_of_month, first_date, next_to_last_date)` generates coupon dates as QuantLib's `Schedule` does:
- `rule` is a `DateGenerationRule`:
  - `.Backward` (the default) steps back from the termination date, which leaves any stub at the front;
  - `.Forward` steps forward from the effective date, leaving any stub at the back;
  - `.Zero` gives the two end dates only;
  - `.CDS` and `.CDS2015` are the standard CDS rules.
- `first_date` and `next_to_last_date` set explicit stub dates.
- `convention` adjusts the dates in between; it defaults to `.ModifiedFollowing`. `termination_convention` adjusts the termination date.

The result holds `dates` (n + 1 adjusted dates) and `is_regular` (n flags; false marks a stub). Free it with `free_schedule`. `schedule_reference_period(s, k)` gives the reference period that ICMA uses for period k. For a stub, that is the regular period ending (or starting) at the stub's regular end.

```odin
s := fin.make_schedule(w.Date{2026, 3, 15}, w.Date{2029, 1, 15}, {6, .Months}, {.TARGET})
defer fin.free_schedule(&s)
for k in 0 ..< len(s.dates) - 1 {
	rs, re := fin.schedule_reference_period(s, k)
	fmt.printf("%v -> %v  regular %-5v reference %v -> %v\n", s.dates[k], s.dates[k + 1], s.is_regular[k], rs, re)
}
f := fin.make_schedule(w.Date{2026, 3, 15}, w.Date{2029, 1, 15}, {6, .Months}, {.TARGET}, rule = .Forward)
defer fin.free_schedule(&f)
fmt.println("forward:", f.dates, f.is_regular)
e := fin.make_schedule(w.Date{2026, 2, 27}, w.Date{2027, 2, 26}, {3, .Months}, {.TARGET}, end_of_month = true)
defer fin.free_schedule(&e)
fmt.println("end of month:", e.dates)
```

```
Date{year = 2026, month = 3, day = 16} -> Date{year = 2026, month = 7, day = 15}  regular false reference Date{year = 2026, month = 1, day = 15} -> Date{year = 2026, month = 7, day = 15}
Date{year = 2026, month = 7, day = 15} -> Date{year = 2027, month = 1, day = 15}  regular true  reference Date{year = 2026, month = 7, day = 15} -> Date{year = 2027, month = 1, day = 15}
Date{year = 2027, month = 1, day = 15} -> Date{year = 2027, month = 7, day = 15}  regular true  reference Date{year = 2027, month = 1, day = 15} -> Date{year = 2027, month = 7, day = 15}
Date{year = 2027, month = 7, day = 15} -> Date{year = 2028, month = 1, day = 17}  regular true  reference Date{year = 2027, month = 7, day = 15} -> Date{year = 2028, month = 1, day = 17}
Date{year = 2028, month = 1, day = 17} -> Date{year = 2028, month = 7, day = 17}  regular true  reference Date{year = 2028, month = 1, day = 17} -> Date{year = 2028, month = 7, day = 17}
Date{year = 2028, month = 7, day = 17} -> Date{year = 2029, month = 1, day = 15}  regular true  reference Date{year = 2028, month = 7, day = 17} -> Date{year = 2029, month = 1, day = 15}
forward: [Date{year = 2026, month = 3, day = 16}, Date{year = 2026, month = 9, day = 15}, Date{year = 2027, month = 3, day = 15}, Date{year = 2027, month = 9, day = 15}, Date{year = 2028, month = 3, day = 15}, Date{year = 2028, month = 9, day = 15}, Date{year = 2029, month = 1, day = 15}] [true, true, true, true, true, false]
end of month: [Date{year = 2026, month = 2, day = 27}, Date{year = 2026, month = 5, day = 29}, Date{year = 2026, month = 8, day = 31}, Date{year = 2026, month = 11, day = 30}, Date{year = 2027, month = 2, day = 26}]
```

The 15 March 2026 start falls on a Sunday and moves to the 16th. The backward schedule has a short first period with the reference period 15 Jan – 15 Jul. The forward schedule puts the stub at the end instead. The end-of-month schedule starts on the last business day of February, so every date is the last business day of its month.

IMM dates are the third Wednesdays of March, June, September and December (`is_imm_date`, `imm_next_date`; pass `main_cycle = false` for every month). `cds_maturity_date(trade_date, tenor, rule = .CDS2015)` gives the scheduled end of a standard CDS. Under CDS2015 maturities are 20 June and 20 December, and they roll on 20 March and 20 September. So a 5-year trade made between 20 September 2026 and 19 March 2027 matures on 20 December 2031.

```odin
fmt.println("next IMM dates:", fin.imm_next_date(w.Date{2026, 11, 11}), fin.imm_next_date(w.Date{2026, 12, 16}),
	fin.is_imm_date(w.Date{2026, 12, 16}))
fmt.println("5Y CDS traded 2026-11-11 matures", fin.cds_maturity_date(w.Date{2026, 11, 11}, {5, .Years}),
	"; traded 2026-12-21:", fin.cds_maturity_date(w.Date{2026, 12, 21}, {5, .Years}))
c := fin.make_schedule(w.Date{2026, 11, 12}, fin.cds_maturity_date(w.Date{2026, 11, 11}, {1, .Years}),
	{3, .Months}, {.Weekends}, .Following, .Unadjusted, .CDS2015)
defer fin.free_schedule(&c)
fmt.println("1Y CDS schedule:", c.dates)
fmt.println("Easter 2026, 2027:", fin.easter_sunday(2026), fin.easter_sunday(2027))
```

```
next IMM dates: Date{year = 2026, month = 12, day = 16} Date{year = 2027, month = 3, day = 17} true
5Y CDS traded 2026-11-11 matures Date{year = 2031, month = 12, day = 20} ; traded 2026-12-21: Date{year = 2031, month = 12, day = 20}
1Y CDS schedule: [Date{year = 2026, month = 9, day = 21}, Date{year = 2026, month = 12, day = 21}, Date{year = 2027, month = 3, day = 22}, Date{year = 2027, month = 6, day = 21}, Date{year = 2027, month = 9, day = 20}, Date{year = 2027, month = 12, day = 20}]
Easter 2026, 2027: Date{year = 2026, month = 4, day = 5} Date{year = 2027, month = 3, day = 28}
```

The CDS schedule starts on the 20th on or before the trade date, rolls quarterly on the 20th, and leaves the termination date unadjusted. `period_frequency(p)` gives the payments per year of a tenor, or 0 if it doesn't divide a year.

## 5. Yield curves

A `YieldCurve` holds increasing pillar `tenors` (years) and continuously compounded `zero_rates`, and an `interpolation`:
- `.LinearZero` (the default) interpolates zero rates linearly in time, and holds them flat before the first pillar and after the last.
- `.FlatForward` gives piecewise-constant forward rates, so discount factors are log-linear. The first forward runs back to t = 0, and the last is carried beyond the last pillar.

You can fill the fields yourself (`YieldCurve{tenors = {1.0}, zero_rates = {0.025}}` is a flat 2.5% curve) or bootstrap one. Free a bootstrapped curve with `free_yield_curve`.

**On a time axis.** `bootstrap_yield_curve_from_swaps(tenors, par_swap_rates, payment_frequency = 0.25, day_count = .ACT_360)` solves one zero rate per tenor so that each spot-starting par swap reprices exactly on the final curve. The swap pays fixed every `payment_frequency` years, with coupon dates stepped back from the tenor. A tenor shorter than one period is a deposit.

```odin
tenors := []f64{0.25, 0.5, 1, 2, 3, 5, 7, 10}
rates := []f64{0.0300, 0.0310, 0.0320, 0.0335, 0.0345, 0.0360, 0.0370, 0.0380}
curve := fin.bootstrap_yield_curve_from_swaps(tenors, rates)
defer fin.free_yield_curve(&curve)
for t, i in curve.tenors {
	fmt.printf("%5.2fY  zero %.10f  DF %.10f\n", t, curve.zero_rates[i], fin.yield_curve_discount_factor(&curve, t))
}
fmt.printf("z(4) %.10f  DF(4) %.10f  fwd(4, 5) %.10f\n", fin.yield_curve_zero_rate(&curve, 4),
	fin.yield_curve_discount_factor(&curve, 4), fin.yield_curve_forward_rate(&curve, 4, 5))
ff := curve
ff.interpolation = .FlatForward
fmt.printf("same pillars, flat forward: z(4) %.10f  fwd(4, 5) %.10f\n", fin.yield_curve_zero_rate(&ff, 4),
	fin.yield_curve_forward_rate(&ff, 4, 5))
```

```
 0.25Y  zero 0.0303016029  DF 0.9924532203
 0.50Y  zero 0.0313116670  DF 0.9844660820
 1.00Y  zero 0.0323248298  DF 0.9681920333
 2.00Y  zero 0.0338578011  DF 0.9345262130
 3.00Y  zero 0.0348897778  DF 0.9006222791
 5.00Y  zero 0.0364679218  DF 0.8333182898
 7.00Y  zero 0.0375428382  DF 0.7688957632
10.00Y  zero 0.0386488177  DF 0.6794387581
z(4) 0.0356788498  DF(4) 0.8670007829  fwd(4, 5) 0.0396242100
same pillars, flat forward: z(4) 0.0358761178  fwd(4, 5) 0.0388351379
```

The queries are `yield_curve_discount_factor(curve, t)`, `yield_curve_zero_rate(curve, t)` and `yield_curve_forward_rate(curve, t1, t2)`. The forward is continuously compounded over [t1, t2].

Switching the interpolation only changes the curve between pillars; it doesn't re-solve them. To bootstrap with flat forwards, use the dated bootstrap below. The time-axis bootstrap interpolates linear zeros while it solves.

**On real dates.** `bootstrap_yield_curve_dated(reference_date, quotes, conv, discount = nil, interpolation = .LinearZero)` reproduces QuantLib's `PiecewiseLinearZero` and `PiecewiseFlatForward`. The inputs are:
- `quotes`: `CurveQuote{kind, tenor, rate}` with `kind` `.Deposit` or `.Swap`, in increasing maturity. Each instrument starts on the spot date.
- `conv`: a `SwapConventions` that describes the swaps and the spot lag. Part 4 covers it in full; a fixed-vs-6M-Euribor setup needs only the fields below.
- `discount`: optional. With it, the result is a projection curve, with swaps discounted on the given curve (for example OIS). Without it, the curve discounts itself.

Each pillar sits on the last date its instrument depends on, so the tenors come out as ACT/365F times of real dates.

```odin
today := w.Date{2026, 11, 11}
conv := fin.SwapConventions {
	calendar         = {.TARGET},
	settlement_days  = 2,
	fixed_tenor      = {1, .Years},
	fixed_convention = .ModifiedFollowing,
	fixed_day_count  = .THIRTY_360,
	float_tenor      = {6, .Months},
	float_convention = .ModifiedFollowing,
	float_day_count  = .ACT_360,
}
quotes := []fin.CurveQuote {
	{.Deposit, {1, .Months}, 0.0195},
	{.Deposit, {3, .Months}, 0.0200},
	{.Deposit, {6, .Months}, 0.0210},
	{.Swap, {1, .Years}, 0.0215},
	{.Swap, {2, .Years}, 0.0222},
	{.Swap, {3, .Years}, 0.0230},
	{.Swap, {5, .Years}, 0.0245},
	{.Swap, {10, .Years}, 0.0270},
	{.Swap, {20, .Years}, 0.0290},
	{.Swap, {30, .Years}, 0.0285},
}
curve := fin.bootstrap_yield_curve_dated(today, quotes, conv)
defer fin.free_yield_curve(&curve)
ffc := fin.bootstrap_yield_curve_dated(today, quotes, conv, interpolation = .FlatForward)
defer fin.free_yield_curve(&ffc)
for t, i in curve.tenors {
	fmt.printf("%8.5f  linear zero %.10f   flat forward %.10f\n", t, curve.zero_rates[i], ffc.zero_rates[i])
}
probe := []w.Date{{2027, 11, 15}, {2031, 6, 30}, {2040, 1, 2}, {2060, 12, 1}}
for d in probe {
	fmt.printf("DF %v: %.12f  %.12f\n", d, fin.yield_curve_discount_factor_date(&curve, d), fin.yield_curve_discount_factor_date(&ffc, d))
}
```

```
 0.09041  linear zero 0.0197542526   flat forward 0.0197542526
 0.26301  linear zero 0.0202152063   flat forward 0.0202152063
 0.50137  linear zero 0.0211644669   flat forward 0.0211644669
 1.01096  linear zero 0.0212642683   flat forward 0.0212642683
 2.00822  linear zero 0.0219288531   flat forward 0.0219288531
 3.00822  linear zero 0.0227367342   flat forward 0.0227367342
 5.00822  linear zero 0.0242602761   flat forward 0.0242565847
10.01370  linear zero 0.0268598417   flat forward 0.0268298281
20.01918  linear zero 0.0290785423   flat forward 0.0290088453
30.02740  linear zero 0.0282230430   flat forward 0.0281912748
DF Date{year = 2027, month = 11, day = 15}: 0.978732118711  0.978732118711
DF Date{year = 2031, month = 6, day = 30}: 0.894808481341  0.894408703949
DF Date{year = 2040, month = 1, day = 2}: 0.696022547761  0.693151058826
DF Date{year = 2060, month = 12, day = 1}: 0.382194421968  0.385152234122
```

Up to 3 years the two interpolations give the same pillars. Every date those instruments need is the spot date (before the first pillar) or the annual fixed coupon date of an earlier swap (a pillar itself). The floating legs of a self-discounting curve telescope to DF(spot) − DF(end). From 5 years on, coupons fall between pillars (the 4-year coupon of the 5-year swap), so the interpolation matters.

Every number above matches QuantLib to the digits shown, with one exception: the linear-zero discount factor in 2060, beyond the last pillar. QuantLib extrapolates the last zero-rate segment linearly and gets 0.386190; here the zero rate is held flat. The flat-forward curve carries its last forward as QuantLib does, and matches.

## 6. Bonds on a time axis

A `Bond` describes a bond in years from today:
- `maturity` counts years from issuance.
- `issue_date` is the issue time; it is negative for a bond issued in the past.
- `settlement_date` is when you buy it.

Coupons are paid every 1/f years, stepped back from `issue_date + maturity`. If the maturity is not a whole number of periods, the first period is a short stub with a pro-rata coupon. Accrued interest is linear in time within a period. The `day_count` field is informational.

`create_fixed_bond(face, coupon_rate, maturity, frequency = .SemiAnnual)` and `create_zero_coupon_bond(face, maturity)` build bonds issued and settled today.

Pricing gives a `BondPriceResult`:
- `clean_price`, `dirty_price` and `accrued_interest`;
- `yield_to_maturity`, compounded at the coupon frequency (annual for zeros);
- `modified_duration` = −P′(y)/P and `convexity` = P″(y)/P, using the street convention P(y) = Σ cf (1 + y/f)^(−f τ), where τ is the time from settlement.

Prices are values at the settlement date. There are two ways to price:
- `price_bond_from_yield_curve(bond, &curve)` discounts on a `YieldCurve`.
- `price_bond_from_curve(bond, df, user_data)` takes a callback `df(t, user_data)` that returns the discount factor to t.

```odin
b := fin.create_fixed_bond(100, 0.05, 5)
r := fin.price_bond_from_yield_curve(b, &curve) // the swap curve of section 5
fmt.printf("5Y 5%% semi-annual: clean %.10f dirty %.10f accrued %.4f ytm %.10f dur %.10f conv %.10f\n",
	r.clean_price, r.dirty_price, r.accrued_interest, r.yield_to_maturity, r.modified_duration, r.convexity)
seasoned := b
seasoned.coupon_rate = 0.04
seasoned.issue_date = -0.3
r2 := fin.price_bond_from_yield_curve(seasoned, &curve)
fmt.printf("seasoned 4%%: clean %.10f dirty %.10f accrued %.4f ytm %.10f dur %.10f conv %.10f\n",
	r2.clean_price, r2.dirty_price, r2.accrued_interest, r2.yield_to_maturity, r2.modified_duration, r2.convexity)
dates := fin.get_coupon_dates(seasoned, context.allocator)
fmt.println("coupon dates", dates)
delete(dates)
stub := fin.create_fixed_bond(100, 0.05, 4.8)
r3 := fin.price_bond_from_yield_curve(stub, &curve)
fmt.printf("4.8Y (short first coupon): clean %.10f ytm %.10f\n", r3.clean_price, r3.yield_to_maturity)
z := fin.price_bond_from_yield_curve(fin.create_zero_coupon_bond(100, 7), &curve)
fmt.printf("7Y zero: %.10f ytm %.10f dur %.10f\n", z.clean_price, z.yield_to_maturity, z.modified_duration)
```

```
5Y 5% semi-annual: clean 106.0601820474 dirty 106.0601820474 accrued 0.0000 ytm 0.0366256543 dur 4.4223990114 conv 23.0287676143
seasoned 4%: clean 101.5258185116 dirty 102.7258185116 accrued 1.2000 ytm 0.0364282136 dur 4.2085475132 conv 20.8969484626
coupon dates [0.20000000000000018, 0.7000000000000002, 1.2000000000000002, 1.7000000000000002, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7]
4.8Y (short first coupon): clean 105.9107076748 ytm 0.0364771701
7Y zero: 76.8895763226 ytm 0.0382564731 dur 6.7420720999
```

The seasoned bond was issued 0.3 years ago, so it has accrued 0.3/0.5 of a 2.00 coupon, and its next coupon is in 0.2 years. The zero's modified duration is T/(1 + y) = 7/1.03826.

The yield tools work on their own as well:
- `solve_ytm(bond, clean_price, allocator)` solves the yield, using a safeguarded Newton step on a bracket. It returns NaN when no flows remain or the price isn't positive.
- `calculate_duration_convexity(bond, ytm, allocator)` gives the two risk measures at a given yield.
- `calculate_accrued_interest` and `get_coupon_dates` expose the pieces.

On a flat continuous curve the semi-annual yield on a coupon date is exactly 2(e^(r/2) − 1):

```odin
flat_r :: 0.04
flat_df :: proc(t: f64, _: rawptr) -> f64 {return math.exp(-flat_r * t)}

f := fin.price_bond_from_curve(b, flat_df, nil)
fmt.printf("flat curve: ytm %.15f vs %.15f\n", f.yield_to_maturity, 2 * (math.exp(flat_r / 2) - 1))
fmt.printf("solve_ytm(98) %.10f\n", fin.solve_ytm(b, 98, context.allocator))
d, c := fin.calculate_duration_convexity(b, 0.05, context.allocator)
fmt.printf("at 5%%: dur %.10f conv %.10f\n", d, c)
```

```
flat curve: ytm 0.040402680053512 vs 0.040402680053512
solve_ytm(98) 0.0546251339
at 5%: dur 4.3760319655 conv 22.6123221851
```

## 7. Bonds on real dates

A `DatedBond{face_value, coupon_rate, schedule, day_count, payment_convention, settlement_days}` follows QuantLib's `FixedRateBond`:
- Each coupon is face × rate × `year_fraction` over its accrual period, with the stub reference periods from the schedule.
- It is paid on the accrual end, adjusted with `payment_convention`.
- The face value is repaid with the last coupon.
- A flow paid on the settlement date belongs to the seller.

The bond doesn't own its schedule.

The procedures, all at a settlement date:

| Procedure | Gives |
|---|---|
| `dated_bond_settlement_date(bond, trade_date)` | `settlement_days` business days after the trade date |
| `dated_bond_accrued_interest(bond, settlement)` | accrued interest |
| `dated_bond_clean_price`, `dated_bond_dirty_price(bond, yield, settlement)` | price from a yield |
| `dated_bond_yield(bond, clean, settlement)` | yield from a clean price |
| `dated_bond_duration_convexity(bond, yield, settlement)` | modified duration and convexity |
| `price_dated_bond(bond, &curve, settlement)` | a `BondPriceResult` on a dated curve |
| `dated_bond_cashflows(bond)` | the `DatedCashFlow`s (payment date, accrual and reference periods, amount) |

Yields are compounded at the coupon frequency in the bond's day count, period by period as in QuantLib's `CashFlows::npv`.

A Bund-like bond on the curve of section 5, and a Treasury-like bond from a yield:

```odin
// a Bund-like bond: 2.5% annual ACT/ACT ICMA, 15 Feb 2026 - 15 Feb 2036, T+2
s := fin.make_schedule(w.Date{2026, 2, 15}, w.Date{2036, 2, 15}, {1, .Years}, {.TARGET},
	.Unadjusted, .Unadjusted)
defer fin.free_schedule(&s)
bund := fin.DatedBond{100, 0.025, s, .ACT_ACT_ICMA, .Following, 2}
settle := fin.dated_bond_settlement_date(bund, today)
r := fin.price_dated_bond(bund, &curve, settle)
fmt.printf("bund settle %v: clean %.10f dirty %.10f accrued %.10f yield %.12f dur %.10f conv %.10f\n",
	settle, r.clean_price, r.dirty_price, r.accrued_interest, r.yield_to_maturity, r.modified_duration, r.convexity)

// a Treasury-like bond: 4.25% semi-annual ACT/ACT ICMA, 15 May 2025 - 15 May 2035, T+1
ts := fin.make_schedule(w.Date{2025, 5, 15}, w.Date{2035, 5, 15}, {6, .Months}, {.US_Government_Bond},
	.Unadjusted, .Unadjusted)
defer fin.free_schedule(&ts)
ust := fin.DatedBond{100, 0.0425, ts, .ACT_ACT_ICMA, .Following, 1}
us := fin.dated_bond_settlement_date(ust, today)
clean := fin.dated_bond_clean_price(ust, 0.041, us)
acc := fin.dated_bond_accrued_interest(ust, us)
y := fin.dated_bond_yield(ust, clean, us)
dur, cv := fin.dated_bond_duration_convexity(ust, 0.041, us)
fmt.printf("ust settle %v: clean %.10f accrued %.10f yield back %.14f dur %.10f conv %.10f\n", us, clean, acc, y, dur, cv)
fmt.printf("ust at 99-16: yield %.12f\n", fin.dated_bond_yield(ust, 99.5, us))
flows := fin.dated_bond_cashflows(ust)
defer delete(flows)
for f in flows[:3] {fmt.printf("%v %.6f\n", f.payment_date, f.amount)}
```

```
bund settle Date{year = 2026, month = 11, day = 13}: clean 98.6616054012 dirty 100.5177697847 accrued 1.8561643836 yield 0.026642067637 dur 8.0063216979 conv 77.2383189118
ust settle Date{year = 2026, month = 11, day = 12}: clean 101.0662957827 accrued 2.0903532609 yield back 0.04100000000000 dur 6.9475919768 conv 57.9113364132
ust at 99-16: yield 0.043205597285
Date{year = 2025, month = 11, day = 17} 2.125000
Date{year = 2026, month = 5, day = 15} 2.125000
Date{year = 2026, month = 11, day = 16} 2.125000
```

The schedules are unadjusted, as is usual for government bonds, so the accrual periods are exact half or whole years. Payments still move: 15 November 2025 was a Saturday, so that coupon is paid on Monday the 17th.

The Treasury settles on 12 November, three days before its accrual period ends. The buyer receives the coupon paid on the 16th, and pays the seller 181/184 of it as accrued interest. QuantLib gives the same numbers to 1e-10.

## 8. Callable and puttable bonds

A `CallableBond` has a face value, a coupon rate, `coupon_frequency` payments a year, a `maturity` in years, and a `call_schedule` of `CallDate{time, price}`. `PuttableBond` is the same with a `put_schedule` of `PutDate`s.

Coupons are paid at maturity − k/f. A bond whose maturity isn't a whole number of periods gets a short first coupon from t = 0, so these are bonds issued today; there is no accrued interest. Call and put prices are clean. On an exercise date that is also a coupon date, the coupon is paid as well. Exercise dates outside (0, maturity) are ignored.

The model is one-factor Hull-White fitted to a discount curve `P0T_func(t)`: r(t) = α(t) + x(t), dx = −a x dt + σ dW. With r0 = f(0, 0), the instantaneous forward at 0, the model reprices the curve exactly. The steps:
- The paths of x and its integral are sampled exactly at the exercise dates, so there is no time-stepping error, and they come in antithetic pairs.
- The straight bond on each exercise date is the closed-form Hull-White bond price.
- Longstaff–Schwartz regresses only the option's continuation value, on powers of the standardised state.

`callable_bond_lsm_hw1f(bond, a, sigma, r0, P0T_func, n_paths = 10000, n_steps, poly_degree = 3)` returns a `CallableBondResult`:
- `price`, `straight_bond_price` and `embedded_call_value`;
- `call_probability`;
- `effective_duration` and `effective_convexity`, from ±50 bp parallel shifts of the curve on the same paths.

`callable_bond_oas(…, market_price, …)` finds the constant spread over the short rate (a parallel shift of the curve) at which the model price equals the market price. It returns NaN if no spread in [−100%, 100%] does. The puttable versions are `puttable_bond_lsm_hw1f` and `puttable_bond_oas`.

`n_steps` is accepted but unused, because the sampling is exact. The results' `oas` field is always 0; use the OAS procedures.

```odin
// market curve z(t) = 3% + 1.2% (1 − e^{−t/2.5}) − 0.4% t e^{−t/2.5}
p0 :: proc(t: f64) -> f64 {
	z := 0.03 + 0.012 * (1 - math.exp(-t / 2.5)) - 0.004 * t * math.exp(-t / 2.5)
	return math.exp(-z * t)
}

rand.reset(42)
bond := fin.CallableBond {
	face_value       = 100,
	coupon_rate      = 0.05,
	coupon_frequency = 2,
	maturity         = 10,
	call_schedule    = []fin.CallDate{{5, 102}, {6, 101.5}, {7, 101}, {8, 100.5}, {9, 100}},
}
r := fin.callable_bond_lsm_hw1f(bond, 0.1, 0.01, 0.03, p0, 65536)
fmt.printf("price %.4f straight %.6f call value %.4f P(call) %.4f eff dur %.4f eff conv %.2f\n",
	r.price, r.straight_bond_price, r.embedded_call_value, r.call_probability, r.effective_duration, r.effective_convexity)
prices: [8]f64
for &p in prices {p = fin.callable_bond_lsm_hw1f(bond, 0.1, 0.01, 0.03, p0, 10000).price}
mean, sd := 0.0, 0.0
for p in prices {mean += p / 8}
for p in prices {sd += (p - mean) * (p - mean) / 7}
fmt.printf("8 runs of 10000 paths: mean %.4f sd %.4f\n", mean, math.sqrt(sd))

// σ → 0: the cheapest deterministic call (or none)
det := r.straight_bond_price
for c in bond.call_schedule {
	v := c.price * p0(c.time)
	for k in 1 ..= int(c.time * 2 + 1e-9) {v += 2.5 * p0(f64(k) / 2)}
	det = min(det, v)
}
r0 := fin.callable_bond_lsm_hw1f(bond, 0.1, 1e-7, 0.03, p0, 2000)
fmt.printf("sigma 1e-7: %.10f, cheapest deterministic call %.10f, P(call) %.3f\n", r0.price, det, r0.call_probability)

oas := fin.callable_bond_oas(bond, 0.1, 0.01, 0.03, p0, 102.0, 20000)
fmt.printf("OAS at 102.00: %.2f bp\n", 1e4 * oas)

put := fin.PuttableBond {
	face_value       = 100,
	coupon_rate      = 0.03,
	coupon_frequency = 2,
	maturity         = 10,
	put_schedule     = []fin.PutDate{{3, 100}, {5, 100}, {7, 100}},
}
pr := fin.puttable_bond_lsm_hw1f(put, 0.05, 0.012, 0.03, p0, 65536)
fmt.printf("puttable %.4f straight %.4f put value %.4f P(put) %.4f eff dur %.4f\n",
	pr.price, pr.straight_bond_price, pr.embedded_put_value, pr.put_probability, pr.effective_duration)
fmt.printf("puttable OAS at 99.00: %.2f bp\n", 1e4 * fin.puttable_bond_oas(put, 0.05, 0.012, 0.03, p0, 99, 20000))
```

```
price 104.7504 straight 107.412198 call value 2.6618 P(call) 0.6605 eff dur 6.2201 eff conv -37.86
8 runs of 10000 paths: mean 104.7515 sd 0.0263
sigma 1e-7: 107.0616913676, cheapest deterministic call 107.0616913642, P(call) 1.000
OAS at 102.00: 41.91 bp
puttable 100.1832 straight 90.9807 put value 9.2025 P(put) 0.8600 eff dur 4.3638
puttable OAS at 99.00: 28.57 bp
```

An exact Hull-White backward induction gives 104.7516 for this callable and 100.1807 for the puttable (`bonds_reference_test`). The Monte Carlo prices are within their standard error of 0.026 per 10 000 paths.

The negative effective convexity is the call at work: when rates fall, the issuer calls and the price stops rising.

With σ → 0 the rates follow the forwards. The model then calls on the cheapest date, and matches that value to 3e-9.

The OAS solver reuses one set of paths across all its trial spreads. A market price below the model price means a positive spread.

`P0T_func` is a plain `proc(f64) -> f64` with no user data. To price on a bootstrapped curve, keep the curve in a package-level variable. Take r0 from the curve's short end.

```odin
g_curve: fin.YieldCurve
curve_df :: proc(t: f64) -> f64 {return fin.yield_curve_discount_factor(&g_curve, t)}

g_curve = fin.bootstrap_yield_curve_from_swaps([]f64{1, 2, 3, 5, 7, 10}, []f64{0.030, 0.032, 0.034, 0.036, 0.037, 0.038})
defer fin.free_yield_curve(&g_curve)
f00 := fin.yield_curve_zero_rate(&g_curve, 1e-6)
rc := fin.callable_bond_lsm_hw1f(bond, 0.1, 0.01, f00, curve_df, 65536)
fmt.printf("on the swap curve (r0 = %.6f): price %.4f straight %.4f P(call) %.4f\n", f00, rc.price, rc.straight_bond_price, rc.call_probability)
// edge: call price far above any value -> straight bond
never := bond
never.call_schedule = []fin.CallDate{{5, 1000}}
rn := fin.callable_bond_lsm_hw1f(never, 0.1, 0.01, 0.03, p0, 4000)
fmt.printf("unreachable call: %.10f vs straight %.10f, P(call) %.4f\n", rn.price, rn.straight_bond_price, rn.call_probability)
// edge: a 4.75Y bond (short first coupon)
short := bond
short.maturity = 4.75
short.call_schedule = []fin.CallDate{{2, 100}}
rs := fin.callable_bond_lsm_hw1f(short, 0.1, 0.01, 0.03, p0, 20000)
fmt.printf("4.75Y callable: %.4f straight %.4f\n", rs.price, rs.straight_bond_price)
```

```
on the swap curve (r0 = 0.030302): price 105.9187 straight 109.3507 P(call) 0.7236
unreachable call: 107.4121977995 vs straight 107.4121977995, P(call) 0.0000
4.75Y callable: 101.9051 straight 105.4260
```

## 9. Convertible bonds

A `ConvertibleBond{bond, conversion_ratio, conversion_start, conversion_end, callability}` wraps a `DatedBond`. The holder may convert it into `conversion_ratio` shares on any date in [conversion_start, conversion_end]. Each `ConvertibleCallability{date, type, price, clean, trigger}` is one of:
- an issuer call (`.Call`), during which the holder may still convert;
- a holder put (`.Put`).

`price` is per face. With `clean` the accrued interest is added to it. A `trigger` above 0 makes a soft call: the issuer can call only while the share is at least trigger × the conversion price, which is face / ratio.

`price_convertible_bond(cb, valuation, mkt, n_steps = 801)` gives the dirty value per bond on the valuation date. It follows QuantLib's `BinomialCRRConvertibleEngine`, using a Cox–Ross–Rubinstein tree with Tsiveriotis–Fernandes credit:
- each node carries the probability of ending up converted;
- the node is discounted at r + (1 − p) · credit spread, where p is that probability.

`ConvertibleMarket` holds:
- `spot`;
- `rate`, continuous ACT/365F;
- `dividend`, a continuous yield;
- `volatility`;
- `credit_spread`;
- optional `dividends`, a list of `CashDividend{date, amount}`. Their present value is taken out of the spot and added back wherever the share is compared with a level.

Event dates snap to the nearest tree step.

```odin
today := w.Date{2026, 11, 11}
sch := fin.make_schedule({2023, 3, 15}, {2030, 3, 15}, {6, .Months}, {.TARGET}, .Following, .Following)
defer fin.free_schedule(&sch)
bond := fin.DatedBond{100, 0.025, sch, .ACT_ACT_ICMA, .Following, 0}
mkt := fin.ConvertibleMarket{spot = 25, rate = 0.03, dividend = 0.01, volatility = 0.3, credit_spread = 0.02}
cb := fin.ConvertibleBond{bond, 3.5, {2023, 3, 15}, {2030, 3, 15}, nil}
fmt.printf("plain: %.6f (parity %.2f)\n", fin.price_convertible_bond(cb, today, mkt), 3.5 * mkt.spot)
calls := []fin.ConvertibleCallability {
	{date = {2028, 3, 15}, type = .Call, price = 102, clean = true, trigger = 1.3},
	{date = {2027, 3, 15}, type = .Put, price = 100, clean = true},
}
cb.callability = calls
fmt.printf("soft call + put: %.6f\n", fin.price_convertible_bond(cb, today, mkt))
mkt.dividends = []fin.CashDividend{{{2027, 6, 1}, 0.5}, {{2028, 6, 1}, 0.5}}
fmt.printf("+ dividends: %.6f\n", fin.price_convertible_bond(cb, today, mkt))
mkt.dividends = nil
cb.callability = nil
for v in ([]f64{1e-12, 3.5}) {
	cb.conversion_ratio = v
	for s in ([]f64{5, 15, 25, 35, 50}) {
		mkt.spot = s
		fmt.printf("ratio %.1f S %2.0f: %.6f\n", v, s, fin.price_convertible_bond(cb, today, mkt))
	}
}
// straight bond (no conversion) vs discounting at r + spread on the coupon flows
flows := fin.dated_bond_cashflows(bond)
defer delete(flows)
pv := 0.0
for f in flows {
	if w.date_less(today, f.payment_date) {
		t := fin.date_to_time(today, f.payment_date)
		pv += f.amount * math.exp(-0.05 * t)
	}
}
fmt.printf("straight bond at r + s, continuous: %.6f\n", pv)
// conversion window ending before maturity
cb.conversion_ratio = 3.5
mkt.spot = 25
cb.conversion_end = {2028, 3, 15}
fmt.printf("conversion until 2028-03-15: %.6f\n", fin.price_convertible_bond(cb, today, mkt))
mkt.spot = 1e-6
fmt.printf("  same, S ~ 0: %.6f\n", fin.price_convertible_bond(cb, today, mkt))
```

```
plain: 110.162149 (parity 87.50)
soft call + put: 109.905790
+ dividends: 108.188519
ratio 0.0 S  5: 92.600493
ratio 0.0 S 15: 92.600493
ratio 0.0 S 25: 92.600493
ratio 0.0 S 35: 92.600493
ratio 0.0 S 50: 92.600493
ratio 3.5 S  5: 92.608705
ratio 3.5 S 15: 95.684508
ratio 3.5 S 25: 110.162149
ratio 3.5 S 35: 134.538125
ratio 3.5 S 50: 179.941973
straight bond at r + s, continuous: 92.598776
conversion until 2028-03-15: 102.926894
  same, S ~ 0: 92.600493
```

Without conversion (ratio ≈ 0) the lattice gives the bond floor, 92.6005. That is the coupons discounted at r + spread, compounded simply per step; continuous discounting gives 92.5988. As the share rises, the value moves from the floor toward parity: at S = 50 the parity is 175.

The soft call costs the holder a little more than the put adds, so the bond with both is worth slightly less than the plain one. Cash dividends lower the value, because the holder doesn't receive them.

Every value up to "straight bond" matches QuantLib's `BinomialCRRConvertibleEngine` to the six decimals shown.

The last two lines show a conversion right that ends before maturity. The tree runs on to maturity, with conversion allowed only inside the window, so with a worthless share the bond is worth exactly its straight value. Here wotan and QuantLib differ: QuantLib ends its tree at the last conversion date. That pays the redemption and every later coupon early, and values the S ≈ 0 bond at 101.78, above its straight value. When the window runs to maturity, the two agree.

A soft call still lets the holder convert on the call date even outside the conversion window, because a call notice forces conversion. QuantLib does the same.

## 10. Inflation

The conventions are QuantLib's zero-inflation framework, verified against version 1.44:
- **Index.** A `CPIIndex{months, values, curve}` holds one fixing per month, keyed by the first of the month. The months must be consecutive. After the last published month, fixings are forecast from the optional `curve`. Before the first, `cpi_fixing` returns NaN.
- **Zero-inflation curve.** A `ZeroInflationCurve{base_date, day_count, months, rates}` forecasts I(m) = I(base) · (1 + z(m))^t(base, m), with annually compounded zero inflation rates z. The rates are linear in time between pillar months, flat back to the base month and linear beyond the last pillar.
- **Lagged observation.** `cpi_observation(index, d, lag_months, interp)` reads the index for date d with an L-month lag:
  - `.Flat` uses the fixing of the month of d − L;
  - `.Linear`, the TIPS and linker convention, interpolates between that month and the next by the day of d within its own month.
- **Zero-coupon inflation swap.** A `ZeroCouponInflationSwap` pays N (I(T − L)/I(S − L) − 1) against N ((1 + K)^τ − 1) at maturity. The procedures are:
  - `zcis_cashflows` gives both legs and the payment date;
  - `zcis_fair_rate` gives the K at which the swap is worth zero;
  - `zcis_npv(swap, index, &nominal)` values the swap for the inflation receiver (the fixed payer) on a nominal curve. QuantLib's `ZeroCouponInflationSwap` with `Swap.Payer` pays the inflation leg, so its NPV has the opposite sign.
- **Bootstrap.** `bootstrap_zero_inflation_curve(index, start, base_date, quotes, lag_months, interp, calendar)` solves one pillar per `ZCISQuote{tenor, rate}`, so that each swap from `start` reprices at its quote. Under `.Linear` a quote also reads the month after its pillar, so the pillars are re-solved together until they settle, as in QuantLib's iterative bootstrap.

```odin
today := w.Date{2026, 11, 11}
// monthly fixings Jan 2024 - Sep 2026 (33 months), about 2.2% a year
months: [33]w.Date
values: [33]f64
for i in 0 ..< 33 {
	months[i] = w.add_month_date({2024, 1, 1}, i32(i))
	values[i] = math.round(1000 * 120 * math.pow(1.022, f64(i) / 12) * (1 + 0.002 * math.sin(f64(i)))) / 1000
}
index := fin.CPIIndex{months[:], values[:], nil}
fmt.println("last fixings:", values[30:])
quotes := []fin.ZCISQuote{{{1, .Years}, 0.0210}, {{2, .Years}, 0.0205}, {{3, .Years}, 0.0204}, {{5, .Years}, 0.0210}, {{10, .Years}, 0.0222}}
for interp in fin.CPIInterpolation {
	curve := fin.bootstrap_zero_inflation_curve(&index, today, {2026, 8, 1}, quotes, 3, interp, {.TARGET})
	defer fin.free_zero_inflation_curve(&curve)
	index.curve = &curve
	fmt.printf("%v: pillars", interp)
	for m, i in curve.months {fmt.printf(" %d-%02d %.10f", m.year, m.month, curve.rates[i])}
	fmt.println()
	fmt.printf("  CPI forecasts: Dec 2026 %.6f, Sep 2031 %.6f\n", fin.cpi_fixing(&index, {2026, 12, 1}), fin.cpi_fixing(&index, {2031, 9, 1}))
	fmt.printf("  observation for 11 Nov 2026 with 3M lag: %.6f\n", fin.cpi_observation(&index, today, 3, interp))
	sw := fin.ZeroCouponInflationSwap{1e6, 0.0210, today, {2031, 11, 11}, 3, interp, {.TARGET}, .ModifiedFollowing, .ACT_365}
	nominal := fin.YieldCurve{tenors = {1.0}, zero_rates = {0.025}, reference_date = today}
	inf, fix, pay := fin.zcis_cashflows(sw, &index)
	fmt.printf("  5Y swap: fair %.12f (quote 0.021), legs %.4f / %.4f paid %v, NPV %.6f\n", fin.zcis_fair_rate(sw, &index), inf, fix, pay, fin.zcis_npv(sw, &index, &nominal))
	old := sw
	old.start, old.maturity, old.fixed_rate = {2025, 3, 3}, {2028, 3, 3}, 0.025
	fmt.printf("  seasoned 3Y from 3 Mar 2025 at 2.5%%: fair %.10f NPV %.4f\n", fin.zcis_fair_rate(old, &index), fin.zcis_npv(old, &index, &nominal))
	if interp == .Linear {
		sch := fin.make_schedule({2025, 4, 15}, {2035, 4, 15}, {6, .Months}, {.TARGET}, .Unadjusted, .Unadjusted)
		defer fin.free_schedule(&sch)
		base := fin.cpi_observation(&index, {2025, 4, 15}, 3, .Linear)
		b := fin.InflationLinkedBond{100, 0.0125, base, sch, .ACT_ACT_ICMA, .Following, 3, .Linear}
		flows := fin.inflation_linked_bond_cashflows(b, &index)
		defer delete(flows)
		fmt.printf("  ILB base CPI %.6f; first flows %v %.6f, %v %.6f; redemption %v %.6f\n", base, flows[0].payment_date, flows[0].amount,
			flows[1].payment_date, flows[1].amount, flows[len(flows) - 1].payment_date, flows[len(flows) - 1].amount)
		fmt.printf("  ILB dirty at 13 Nov 2026: %.10f\n", fin.inflation_linked_bond_dirty_price(b, &index, &nominal, {2026, 11, 13}))
	}
	fmt.printf("  before the first fixing (Jun 2023): %v\n", fin.cpi_fixing(&index, {2023, 6, 1}))
	index.curve = nil
}
```

```
last fixings: [126.459, 126.837, 127.31]
Flat: pillars 2027-08 0.0210000000 2028-08 0.0205566601 2029-08 0.0204188019 2031-08 0.0210000000 2036-08 0.0222000000
  CPI forecasts: Dec 2026 127.721140, Sep 2031 140.997052
  observation for 11 Nov 2026 with 3M lag: 126.837000
  5Y swap: fair 0.021000000000 (quote 0.021), legs 109566.7617 / 109566.7617 paid Date{year = 2031, month = 11, day = 11}, NPV 0.000000
  seasoned 3Y from 3 Mar 2025 at 2.5%: fair 0.0218831036 NPV -9488.0151
  before the first fixing (Jun 2023): NaN
Linear: pillars 2027-08 0.0216739808 2028-08 0.0208404107 2029-08 0.0206167417 2031-08 0.0211261716 2036-08 0.0222567380
  CPI forecasts: Dec 2026 127.749314, Sep 2031 141.084892
  observation for 11 Nov 2026 with 3M lag: 126.994667
  5Y swap: fair 0.021000000000 (quote 0.021), legs 109566.7617 / 109566.7617 paid Date{year = 2031, month = 11, day = 11}, NPV 0.000000
  seasoned 3Y from 3 Mar 2025 at 2.5%: fair 0.0221021419 NPV -8823.1394
  ILB base CPI 122.721733; first flows Date{year = 2025, month = 10, day = 15} 0.631499, Date{year = 2026, month = 4, day = 15} 0.638169; redemption Date{year = 2035, month = 4, day = 16} 124.162129
  ILB dirty at 13 Nov 2026: 111.3783513680
  before the first fixing (Jun 2023): NaN
```

Each pillar is the month of its lagged maturity. A swap from 11 November 2026 for 1 year observes August 2027.

Under `.Flat` the 1-year pillar is the quote itself: the swap observes August 2026 to August 2027, exactly the curve's base and first pillar. Under `.Linear` the observation also reads the following months, so every pillar moves.

The seasoned swap starts from published fixings: December 2024, and also January 2025 under `.Linear`. The fixed rate of 2.5% is above the fair rate, so the inflation receiver loses.

The bootstrap restores the index's own `curve` field when it returns. That is why the example sets `index.curve = &curve` afterwards.

**Inflation-linked bonds.** An `InflationLinkedBond{face_value, real_coupon, base_cpi, schedule, day_count, payment_convention, lag_months, interpolation}` follows QuantLib's `CPIBond`:
- each real coupon is scaled by I(accrual end, lagged) / base_cpi;
- the face value is scaled by I(maturity, lagged) / base_cpi.

The procedures:
- `inflation_linked_bond_cashflows` gives the flows.
- `inflation_linked_bond_dirty_price(b, index, &nominal, settlement)` discounts them on a nominal curve.

The `.Linear` branch of the example above prices a 10-year linker issued 15 April 2025, with a 1.25% real coupon. Its base CPI is the lagged observation on the issue date.

The first coupon is the real coupon 0.625 scaled by the index ratio, 1.0104. The forecast redemption is the face value scaled by 1.2416. QuantLib's `CPIBond` gives the same flows, and the same dirty value to 1e-10.

## 11. Validation

**Reference tests in the suite:**

| Test | Compares |
|---|---|
| `dates_reference_test` | holiday lists, adjust and advance, business-day counts, every day counter and schedule generation on seeded random inputs, IMM dates and CDS maturities, against QuantLib 1.44 |
| `dated_instruments_reference_test` | `PiecewiseLinearZero` and flat-forward bootstraps, OIS and Euribor multi-curve setups, swaps and `FixedRateBond` prices, yields and risk measures, against QuantLib |
| `swaps_yield_curve_reference_test` | the time-axis bootstrap and swap valuation |
| `bonds_reference_test` | `Bond` yields, durations and convexities against QuantLib; the Hull-White bond price; callable and puttable bonds against exact backward induction and QuantLib's Hull-White tree; effective duration and OAS round trips |
| `convertible_bonds_reference_test` | five convertibles (soft and hard calls, puts, cash dividends, a late conversion start) against `BinomialCRRConvertibleEngine` at 801 and 2000 steps, to 1e-9 |
| `inflation_reference_test` | Flat and Linear zero-inflation curves, HICP forecasts to 2046, zero-coupon swaps and a `CPIBond`, against QuantLib |
| `bonds_test`, `yield_curve_test`, `callable_bond_test` | usage and consistency checks |

**Additional checks:**

| What | Reference | Agreement |
|---|---|---|
| 2026 holidays of all four calendars; the 7 adjustments; advance with and without end of month; business days on single and joint calendars | QuantLib | exact |
| the 8 day counters across end of February; an ICMA stub | QuantLib | all 10 printed digits |
| backward, forward and end-of-month schedules; IMM dates; CDS2015 maturities for 6 trade dates around the rolls; a CDS schedule | QuantLib | exact |
| the time-axis bootstrap, and the prices, yields, durations and convexities of 4 bonds | an independent Python version (SciPy `brentq`, finite differences) | 10 digits |
| linear-zero and flat-forward bootstraps from 3 deposits and 7 swaps: zero rates at the pillars, and discount factors up to 30Y | `PiecewiseLinearZero`, `PiecewiseFlatForward` | all 10 printed digits; 2e-12 |
| a Bund on that curve: price, accrued, yield, duration, convexity; a Treasury: price from yield and yield from price | QuantLib `FixedRateBond`, `BondFunctions` | 1e-9 |
| a callable bond as σ → 0 | the cheapest deterministic call | 3e-9 |
| a callable bond with an unreachable call | the straight bond | exact |
| convertible prices across 10 spot and ratio points, soft call plus put, and cash dividends | `BinomialCRRConvertibleEngine` | all 6 printed decimals |
| a convertible whose conversion ends before maturity, with no credit spread | QuantLib pricing of the equivalent shorter convertible | 2e-3 at 4001 steps |
| inflation curves, forecasts, lagged observations, swap fair rates and NPVs, inflation-linked bond flows and dirty price | QuantLib | all printed digits |

**Known limitations:**
- **Linear-zero extrapolation.** Linear-zero curves hold the zero rate flat beyond the last pillar, while QuantLib extrapolates it linearly. In the section 5 example this gives a 2060 discount factor of 0.3822 against 0.3862. Flat-forward curves match QuantLib.
- **Hull-White curve input.** `P0T_func` has no user-data pointer, so a bootstrapped curve has to go through a package-level variable.
- **Callable and puttable bonds:**
  - they model newly issued bonds only, with no seasoned bonds and no accrued interest;
  - `n_steps`, the `settlement` field and the results' `oas` field are unused;
  - the private `_hw_bermudan_price_fixed` is not called anywhere.
- **Time-axis bonds.** Accrued interest is linear in time, and `Bond.day_count` is informational.
- **Inflation-linked bonds.** They have a dirty price only, with no accrued interest, clean price or real yield.

## 12. Quick reference

| Task | Procedure | Section |
|---|---|---|
| Business days | `is_business_day`, `is_holiday`, `business_days_between` | 2 |
| Adjust and move dates | `calendar_adjust`, `calendar_advance`, `calendar_advance_period`, `date_add_period`, `calendar_end_of_month`, `calendar_is_end_of_month` | 2 |
| Day counts | `year_fraction`, `day_count_days`, `day_count_fraction` (time axis), `date_to_time` | 3 |
| Coupon schedules | `make_schedule`, `schedule_reference_period`, `free_schedule`, `period_frequency` | 4 |
| IMM and CDS dates | `is_imm_date`, `imm_next_date`, `cds_maturity_date`, `easter_sunday` | 4 |
| Curves | `bootstrap_yield_curve_from_swaps`, `bootstrap_yield_curve_dated`, `free_yield_curve` | 5 |
| Curve queries | `yield_curve_discount_factor`, `yield_curve_discount_factor_date`, `yield_curve_zero_rate`, `yield_curve_forward_rate` | 5 |
| Time-axis bonds | `create_fixed_bond`, `create_zero_coupon_bond`, `price_bond_from_yield_curve`, `price_bond_from_curve`, `solve_ytm`, `calculate_duration_convexity`, `calculate_accrued_interest`, `get_coupon_dates` | 6 |
| Dated bonds | `price_dated_bond`, `dated_bond_clean_price`, `dated_bond_dirty_price`, `dated_bond_yield`, `dated_bond_duration_convexity`, `dated_bond_accrued_interest`, `dated_bond_settlement_date`, `dated_bond_cashflows` | 7 |
| Callable and puttable bonds | `callable_bond_lsm_hw1f`, `callable_bond_oas`, `puttable_bond_lsm_hw1f`, `puttable_bond_oas` | 8 |
| Convertibles | `price_convertible_bond` | 9 |
| CPI | `cpi_fixing`, `cpi_observation`, `zero_inflation_rate`, `bootstrap_zero_inflation_curve`, `free_zero_inflation_curve` | 10 |
| Inflation swaps and bonds | `zcis_cashflows`, `zcis_fair_rate`, `zcis_npv`, `inflation_linked_bond_cashflows`, `inflation_linked_bond_dirty_price` | 10 |
