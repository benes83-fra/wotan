package finance

import "core:mem"

// ============================================================================
// PUTTABLE BOND STRUCTURES
// ============================================================================

// Same schedule conventions as CallableBond; put prices are clean.
PuttableBond :: struct {
	face_value:       f64,
	coupon_rate:      f64, // Annual coupon rate
	coupon_frequency: int, // Payments per year (e.g., 2 for semi-annual)
	maturity:         f64, // Bond maturity in years
	put_schedule:     []PutDate, // Put dates and prices (dates outside (0, maturity) are ignored)
	settlement:       f64, // Settlement date (usually 0; not used by the pricer)
}

PutDate :: struct {
	time:  f64, // Time from today (in years)
	price: f64, // Put price (usually par or par + premium)
}

PuttableBondResult :: struct {
	price:               f64,
	straight_bond_price: f64,
	embedded_put_value:  f64, // Puttable bonds are worth MORE than straight bonds
	oas:                 f64, // Always 0 here (the price is at zero spread); see puttable_bond_oas
	effective_duration:  f64, // −∂P/∂s / P for a parallel shift s of the initial curve
	effective_convexity: f64, // ∂²P/∂s² / P
	put_probability:     f64, // Probability of being put
}

@(private = "file")
_hw_puttable_build :: proc(
	bond: PuttableBond,
	a, sigma, r0: f64,
	P0T: proc(_: f64) -> f64,
	n_paths, degree: int,
	allocator: mem.Allocator,
) -> _HW_Bermudan {
	m := len(bond.put_schedule)
	t := make([]f64, m, allocator)
	k := make([]f64, m, allocator)
	defer {delete(t, allocator); delete(k, allocator)}
	for c, i in bond.put_schedule {
		t[i] = c.time
		k[i] = c.price
	}
	return _hw_bermudan_build(bond.face_value, bond.coupon_rate, bond.coupon_frequency, bond.maturity, t, k, false, a, sigma, r0, P0T, n_paths, degree, allocator)
}

// ============================================================================
// OAS SOLVER FOR PUTTABLE BONDS
// ============================================================================
// Constant spread over the Hull-White short rate (a parallel shift of the
// initial curve) at which the model price equals market_price; NaN if no
// spread in [−100%, 100%] matches. n_steps is unused (exact sampling).
puttable_bond_oas :: proc(
	bond: PuttableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	market_price: f64,
	n_paths: int = 5000,
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> f64 {
	e := _hw_puttable_build(bond, a, sigma, r0, P0T_func, n_paths, poly_degree, allocator)
	defer _hw_bermudan_free(&e)
	return _hw_bermudan_oas(&e, market_price)
}

// ============================================================================
// MAIN API: LSM FOR PUTTABLE BONDS (HULL-WHITE 1F)
// ============================================================================
// See callable_bond_lsm_hw1f; the investor puts when the put price exceeds the
// continuation value.
puttable_bond_lsm_hw1f :: proc(
	bond: PuttableBond,
	a: f64,
	sigma: f64,
	r0: f64,
	P0T_func: proc(_: f64) -> f64,
	n_paths: int = 10000,
	n_steps: int = 100,
	poly_degree: int = 3,
	allocator: mem.Allocator = context.allocator,
) -> PuttableBondResult {
	e := _hw_puttable_build(bond, a, sigma, r0, P0T_func, n_paths, poly_degree, allocator)
	defer _hw_bermudan_free(&e)

	price, put_prob := _hw_bermudan_price(&e, 0.0)
	straight_price := _hw_bermudan_straight(&e, 0.0)
	h := 0.005
	price_up, _ := _hw_bermudan_price(&e, h)
	price_dn, _ := _hw_bermudan_price(&e, -h)

	return PuttableBondResult {
		price               = price,
		straight_bond_price = straight_price,
		embedded_put_value  = price - straight_price,
		oas                 = 0.0,
		effective_duration  = -(price_up - price_dn) / (2.0 * h * price),
		effective_convexity = (price_up + price_dn - 2.0 * price) / (h * h * price),
		put_probability     = put_prob,
	}
}
