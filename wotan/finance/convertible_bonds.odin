package finance

import w "../core"
import "core:math"
import "core:mem"

// ============================================================================
// CONVERTIBLE BONDS (TSIVERIOTIS-FERNANDES LATTICE)
// ============================================================================
// A fixed-coupon bond (DatedBond) convertible into conversion_ratio shares
// within [conversion_start, conversion_end], with optional issuer calls (hard,
// or soft: only while the share price is at least trigger × the conversion
// price face/ratio) and holder puts. Valued as QuantLib's
// BinomialConvertibleEngine with a Cox-Ross-Rubinstein tree:
// - constant r (continuous zero rate to maturity), dividend yield q, volatility
//   σ; cash dividends are taken out of the spot (their present value) and added
//   back, discounted, wherever the share price is compared with a level;
// - up probability ½ + ½(r − q − σ²/2)Δt/(σ√Δt), steps of ±σ√Δt in ln S;
// - Tsiveriotis-Fernandes credit: each node carries the probability p of ending
//   up converted, rolled back as the risk-neutral average, and its value is
//   discounted at r + (1 − p)·credit_spread (simple per step);
// - at each node: calls / puts, then coupons, then conversion if it pays more.
// Event dates are snapped to the nearest time step.

ConvertibleCallType :: enum u8 {
	Call, // issuer redeems at the price (the holder may convert instead)
	Put, // holder sells back at the price
}

ConvertibleCallability :: struct {
	date:    w.Date,
	type:    ConvertibleCallType,
	price:   f64, // per face
	clean:   bool, // price excludes accrued interest (it is added)
	trigger: f64, // soft call: callable only if S ≥ trigger · face/ratio; 0 = hard call
}

ConvertibleBond :: struct {
	bond:             DatedBond,
	conversion_ratio: f64, // shares per bond
	conversion_start: w.Date,
	conversion_end:   w.Date, // usually the maturity
	callability:      []ConvertibleCallability,
}

CashDividend :: struct {
	date:   w.Date,
	amount: f64,
}

ConvertibleMarket :: struct {
	spot:          f64,
	rate:          f64, // continuously compounded, ACT/365F
	dividend:      f64, // continuous dividend yield
	volatility:    f64,
	credit_spread: f64,
	dividends:     []CashDividend,
}

// Value of the convertible at the valuation date (the dirty price per bond)
price_convertible_bond :: proc(
	cb: ConvertibleBond,
	valuation: w.Date,
	mkt: ConvertibleMarket,
	n_steps: int = 801,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	b := cb.bond
	sd := b.schedule.dates
	maturity := sd[len(sd) - 1]
	T := date_to_time(valuation, cb.conversion_end) // the exercise's last date ends the tree
	time :: proc(valuation, d: w.Date) -> f64 {return date_to_time(valuation, d)}
	r, q, sg := mkt.rate, mkt.dividend, mkt.volatility
	disc :: proc(r, t: f64) -> f64 {return math.exp(-r * t)}

	// spot net of the dividends still to be paid
	s0 := mkt.spot
	for dv in mkt.dividends {
		if !w.date_less(dv.date, valuation) {s0 -= dv.amount * disc(r, time(valuation, dv.date))}
	}

	n := n_steps
	dt := T / f64(n)
	dx := sg * math.sqrt(dt)
	pu := 0.5 + 0.5 * (r - q - 0.5 * sg * sg) * dt / dx
	pd := 1.0 - pu
	snap :: proc(t, dt: f64, n: int) -> int {return clamp(int(math.round(t / dt)), 0, n)}

	// events on the grid: coupons (amount per step), calls / puts, dividends
	flows := dated_bond_cashflows(b, context.temp_allocator)
	defer delete(flows, context.temp_allocator)
	coupon := make([]f64, n + 1, context.temp_allocator)
	defer delete(coupon, context.temp_allocator)
	for f in flows[:len(flows) - 1] {
		if w.date_less(
			valuation,
			f.payment_date,
		) {coupon[snap(time(valuation, f.payment_date), dt, n)] += f.amount}
	}
	redemption := flows[len(flows) - 1].amount
	Call :: struct {
		step:           int,
		type:           ConvertibleCallType,
		price, trigger: f64,
	}
	calls := make([dynamic]Call, 0, len(cb.callability), context.temp_allocator)
	defer delete(calls)
	for c in cb.callability {
		if !w.date_less(valuation, c.date) {continue}
		price := c.price
		if c.clean {price += dated_bond_accrued_interest(b, c.date)}
		append(&calls, Call{snap(time(valuation, c.date), dt, n), c.type, price, c.trigger})
	}
	div_step := make([]int, len(mkt.dividends), context.temp_allocator)
	defer delete(div_step, context.temp_allocator)
	for dv, k in mkt.dividends {div_step[k] = snap(time(valuation, dv.date), dt, n)}
	conv_lo :=
		snap(time(valuation, cb.conversion_start), dt, n) if w.date_less(valuation, cb.conversion_start) else -1
	conv_hi := snap(T, dt, n)

	V := make([]f64, n + 1, allocator)
	P := make([]f64, n + 1, allocator) // conversion probability
	R := make([]f64, n + 1, allocator) // blended discount rate
	S := make([]f64, n + 1, allocator) // share price on the grid incl. future dividends
	defer {delete(V, allocator); delete(P, allocator); delete(R, allocator); delete(S, allocator)}

	// share prices at step i (dividends still to come added back, discounted to step i)
	grid :: proc(S: []f64, i: int, s0, dx, dt, r: f64, divs: []CashDividend, div_step: []int) {
		for j in 0 ..= i {
			S[j] = s0 * math.exp(f64(2 * j - i) * dx)
			for dv, k in divs {
				if div_step[k] >= i {S[j] += dv.amount * math.exp(-r * f64(div_step[k] - i) * dt)}
			}
		}
	}
	// calls / puts, coupons and conversion at step i
	adjust :: proc(
		V, P, S: []f64,
		i: int,
		calls: []Call,
		coupon: []f64,
		ratio, face: f64,
		convertible: bool,
	) {
		m := i + 1
		for c in calls {
			if c.step != i {continue}
			switch c.type {
			case .Call:
				if c.trigger > 0 {
					level := face / ratio * c.trigger
					for j in 0 ..< m {
						if S[j] >= level {V[j] = min(max(c.price, ratio * S[j]), V[j])}
					}
				} else if convertible {
					for j in 0 ..< m {V[j] = min(max(c.price, ratio * S[j]), V[j])}
				} else {
					for j in 0 ..< m {V[j] = min(c.price, V[j])}
				}
			case .Put:
				for j in 0 ..< m {V[j] = max(V[j], c.price)}
			}
		}
		if coupon[i] != 0 {for j in 0 ..< m {V[j] += coupon[i]}}
		if convertible {
			for j in 0 ..< m {
				if V[j] <= ratio * S[j] {
					V[j] = ratio * S[j]
					P[j] = 1
				}
			}
		}
	}
	convertible :: proc(i, lo, hi: int) -> bool {return i >= lo && i <= hi}

	// maturity
	for j in 0 ..= n {V[j], P[j] = redemption, 0}
	grid(S, n, s0, dx, dt, r, mkt.dividends, div_step)
	adjust(
		V,
		P,
		S,
		n,
		calls[:],
		coupon,
		cb.conversion_ratio,
		b.face_value,
		convertible(n, conv_lo, conv_hi),
	)
	for j in 0 ..= n {R[j] = r + (1.0 - P[j]) * mkt.credit_spread}

	// roll back; each node discounts its children at their blended rates
	for i := n - 1; i >= 0; i -= 1 {
		for j in 0 ..= i {
			p := pd * P[j] + pu * P[j + 1]
			v := pd * V[j] / (1.0 + R[j] * dt) + pu * V[j + 1] / (1.0 + R[j + 1] * dt)
			V[j], P[j] = v, p
			R[j] = r + (1.0 - p) * mkt.credit_spread
		}
		if i > 0 {
			grid(S, i, s0, dx, dt, r, mkt.dividends, div_step)
			adjust(
				V,
				P,
				S,
				i,
				calls[:],
				coupon,
				cb.conversion_ratio,
				b.face_value,
				convertible(i, conv_lo, conv_hi),
			)
		}
	}
	// the last adjustment (at the valuation date) is applied once more as in QuantLib's rollback
	grid(S, 0, s0, dx, dt, r, mkt.dividends, div_step)
	adjust(
		V,
		P,
		S,
		0,
		calls[:],
		coupon,
		cb.conversion_ratio,
		b.face_value,
		convertible(0, conv_lo, conv_hi),
	)
	_ = maturity
	return V[0]
}
