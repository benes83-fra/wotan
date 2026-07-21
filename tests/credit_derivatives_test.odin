package tests

import fin "../wotan/finance"
import "core:fmt"
import "core:math"
import "core:mem"

credit_derivatives_2008_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n======================================================================")
	fmt.println("         CREDIT DERIVATIVES: THE 2008 GAUSSIAN COPULA ENGINE")
	fmt.println("======================================================================\n")

	// 1. Single Name CDS Calibration
	fmt.println("1. Calibrating Single-Name CDS...")
	cds_params := fin.CDS_Params {
		lambda   = 0.0150, // ~1.5% hazard rate (roughly BBB rating)
		recovery = 0.40,
		r        = 0.04,
	}

	// Market spread for this name is ~100 bps (0.0100)
	market_spread := 0.0100
	cds_pv := fin.price_cds(10_000_000.0, market_spread, 5.0, cds_params, 0.25)

	fmt.printf("   5Y CDS Fair Spread: ~100 bps\n")
	fmt.printf("   PV of Protection (Notional $10M): $%.2f\n\n", cds_pv)

	// 2. CDO Setup (The 2008 Machine)
	fmt.println("2. Pricing a 5Y Synthetic CDO (100 Names, 40% Recovery)...")
	cdo_params := fin.CDO_Params {
		N        = 100,
		lambda   = 0.0150,
		recovery = 0.40,
		r        = 0.04,
		rho      = 0.15, // The "magic" correlation number in 2005-2006
		maturity = 5.0,
	}

	tranches := [3]fin.CDO_Tranche {
		{name = "Equity (0-3%)", attach = 0.00, detach = 0.03},
		{name = "Mezzanine (3-7%)", attach = 0.03, detach = 0.07},
		{name = "Senior (7-100%)", attach = 0.07, detach = 1.00},
	}

	fmt.println("   Base Case: Correlation (rho) = 15%")
	fmt.println("   ----------------------------------------------------------------------")
	fmt.printf("   %-20s | %-12s | %-15s\n", "Tranche", "Exp. Loss %", "Tranche PV")
	fmt.println("   ----------------------------------------------------------------------")

	for t in tranches {
		// Spread is roughly proportional to risk (Equity pays ~1000bps, Mezz ~150bps, Senior ~20bps)
		spread: f64
		if t.attach ==
		   0.0 {spread = 0.1000} else if t.attach == 0.03 {spread = 0.0150} else {spread = 0.0020}

		pv, exp_loss := fin.price_cdo_tranche_mc(t, cdo_params, spread, 50000, allocator)
		fmt.printf("   %-20s | %11.4f%% | $%12.4f\n", t.name, exp_loss * 100.0, pv)
	}

	// 3. The Crisis Scenario: Correlation spikes to 35%
	fmt.println("\n   CRISIS SCENARIO: Correlation (rho) spikes to 35%")
	fmt.println("   ----------------------------------------------------------------------")
	cdo_params.rho = 0.35

	for t in tranches {
		spread: f64
		if t.attach ==
		   0.0 {spread = 0.1000} else if t.attach == 0.03 {spread = 0.0150} else {spread = 0.0020}

		pv, exp_loss := fin.price_cdo_tranche_mc(t, cdo_params, spread, 50000, allocator)
		fmt.printf("   %-20s | %11.4f%% | $%12.4f\n", t.name, exp_loss * 100.0, pv)
	}

	fmt.println("\n💡 The 2008 Flaw: The Gaussian Copula assumes joint defaults follow a")
	fmt.println("   normal distribution. It drastically underestimates 'tail dependence'")
	fmt.println("   (the probability that *many* names default together). When housing")
	fmt.println("   prices fell, correlation spiked from 15% to >35%, instantly wiping")
	fmt.println("   out the Mezzanine tranche and bleeding into Senior tranches.")
	fmt.println("======================================================================\n")
	// =========================================================================
	// 4. THE FIX: Student-t Copula (Captures Tail Dependence)
	// =========================================================================
	fmt.println("4. THE FIX: Pricing with Student-t Copula (nu = 4.0)")
	fmt.println("   ----------------------------------------------------------------------")

	// nu = 4.0 provides strong tail dependence, closely matching empirical default data
	cdo_t_params := fin.CDO_Params {
		N        = 100,
		lambda   = 0.0150,
		recovery = 0.40,
		r        = 0.04,
		rho      = 0.15, // Same base correlation as the "safe" Gaussian scenario
		maturity = 5.0,
	}
	nu := 4.0

	fmt.printf("   %-20s | %-12s | %-15s\n", "Tranche", "Exp. Loss %", "Tranche PV")
	fmt.println("   ----------------------------------------------------------------------")
	for t in tranches {
		spread: f64
		if t.attach ==
		   0.0 {spread = 0.1000} else if t.attach == 0.03 {spread = 0.0150} else {spread = 0.0020}

		pv, exp_loss := fin.price_cdo_tranche_t_copula_mc(
			t,
			cdo_t_params,
			nu,
			spread,
			50000,
			allocator,
		)
		fmt.printf("   %-20s | %11.4f%% | $%12.4f\n", t.name, exp_loss * 100.0, pv)
	}

	fmt.println("\n💡 The Correction: By sharing a single Chi-Squared variable (V) across all")
	fmt.println("   names, the Student-t Copula introduces 'tail dependence'. Even with a")
	fmt.println("   modest correlation of 15%, a small V value causes *all* default times")
	fmt.println("   to compress simultaneously. This correctly prices the Mezzanine tranche")
	fmt.println("   as highly risky, exposing the fatal flaw of the Gaussian Copula.")
	fmt.println("======================================================================\n")
}
