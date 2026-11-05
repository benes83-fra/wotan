package tests

import fin "../wotan/finance"
import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Deep Heston calibration pieces against references: the grid of implied vols vs
// QuantLib (AnalyticHestonEngine + impliedVolatility, S = 1, r = 0, 30/360 so the
// expiries are exact), invariance of the forward-moneyness grid to S and r, the parameter
// scaling round trip, and the interpolation of scattered quotes onto the grid vs numpy.
deep_calibration_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Deep Heston Calibration Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	G :: ml_fin.DEEP_HESTON_N_GRID

	{ 	// --- grid implied vols vs QuantLib
		cases := [2]fin.Heston_Params {
			{v0 = 0.04, kappa = 2.0, theta = 0.04, sigma = 0.3, rho = -0.5},
			{v0 = 0.02, kappa = 0.8, theta = 0.12, sigma = 0.9, rho = -0.85}, // Feller violated, steep skew
		}
		refs := [2][G]f64 {
			{
				0.23387684452787347,
				0.21402705845470338,
				0.19600230811111705,
				0.1825122297070647,
				0.17629464960531127,
				0.2269042739396036,
				0.2093928960448125,
				0.1939226503687459,
				0.18210786332488949,
				0.17538521310857924,
				0.2174702653725036,
				0.20399001201156072,
				0.1924688793875165,
				0.1833347891600909,
				0.1768913271675757,
				0.20854972637664307,
				0.19988911198267892,
				0.1925365836518057,
				0.18642175522821783,
				0.18147380466252547,
			},
			{
				0.27771601039210564,
				0.21637291052057392,
				0.13749317406535758,
				0.09495003024147015,
				0.11088539991467496,
				0.2655363738151689,
				0.21089339436697876,
				0.14839187649882815,
				0.10072165038371218,
				0.10786662949453797,
				0.2574882635735268,
				0.21364647656277402,
				0.16806663831134142,
				0.1255422490619258,
				0.11132934620237228,
				0.2562340759600049,
				0.2243972013777015,
				0.19340027246181268,
				0.16396346793621994,
				0.1398227089944188,
			},
		}
		err := 0.0
		ok := true
		for c, k in cases {
			ivs: [G]f64
			ok = ok && ml_fin.deep_heston_surface(c, ivs[:])
			for v, i in ivs {err = max(err, abs(v - refs[k][i]))}
		}
		expect(
			&passed,
			&total,
			ok && err < 1e-9,
			fmt.tprintf(
				"grid implied vols (5 forward moneyness x 4 expiries, 2 parameter sets) vs QuantLib: max err %.1e",
				err,
			),
		)
	}

	{ 	// --- the same forward moneyness gives the same implied vols for any S and r
		p := fin.Heston_Params {
			v0    = 0.05,
			kappa = 1.5,
			theta = 0.08,
			sigma = 0.6,
			rho   = -0.7,
		}
		ivs: [G]f64
		ml_fin.deep_heston_surface(p, ivs[:])
		S, r := 180.0, 0.045
		M := ml_fin.DEEP_HESTON_MONEYNESS
		E := ml_fin.DEEP_HESTON_EXPIRIES
		err := 0.0
		for T, j in E {
			for m, i in M {
				K := m * S * math.exp(r * T)
				price := fin.heston_price(S, K, T, r, p, .Call)
				iv, _, _ := fin.implied_volatility(price, S, K, T, r, .Call, tol = 1e-12)
				err = max(err, abs(iv - ivs[j * len(M) + i]))
			}
		}
		expect(
			&passed,
			&total,
			err < 1e-9,
			fmt.tprintf(
				"grid vols are the same at S = 180, r = 4.5%% (K = m S exp(rT)): max diff %.1e",
				err,
			),
		)
	}

	{ 	// --- scaling round trip and the box corners
		err := 0.0
		for _ in 0 ..< 100 {
			p := fin.Heston_Params {
				v0    = 0.01 + 0.15 * rand.float64(),
				kappa = 0.5 + 4.5 * rand.float64(),
				theta = 0.01 + 0.15 * rand.float64(),
				sigma = 0.1 + 0.9 * rand.float64(),
				rho   = -0.95 + 0.95 * rand.float64(),
			}
			x := ml_fin.deep_heston_encode(p)
			for v in x {if abs(v) > 1 {err = max(err, 1)}}
			q := ml_fin.deep_heston_decode(x)
			err = max(
				err,
				abs(q.v0 - p.v0),
				abs(q.kappa - p.kappa),
				abs(q.theta - p.theta),
				abs(q.sigma - p.sigma),
				abs(q.rho - p.rho),
			)
		}
		lo := ml_fin.deep_heston_decode({-5, -1, -1, -1, -1}) // clamped to the box
		hi := ml_fin.deep_heston_decode({1, 1, 1, 1, 9})
		corners :=
			lo.v0 == 0.01 &&
			lo.kappa == 0.5 &&
			lo.rho == -0.95 &&
			hi.v0 == 0.16 &&
			hi.sigma == 1.0 &&
			hi.rho == 0.0
		expect(
			&passed,
			&total,
			err < 1e-14 && corners,
			fmt.tprintf(
				"parameters scaled to [-1, 1] over the box and back: max err %.1e, clamped at the corners %v",
				err,
				corners,
			),
		)
	}

	{ 	// --- interpolation of scattered quotes vs numpy (np.interp in strike, linear total
		// variance in expiry, flat outside; a one-quote expiry is ignored)
		quotes := []fin.VolSurfacePoint {
			{strike = 120, expiry = 1.5, implied_vol = 0.21},
			{strike = 100, expiry = 0.3, implied_vol = 0.25},
			{strike = 100, expiry = 0.1, implied_vol = 0.9},
			{strike = 80, expiry = 1.5, implied_vol = 0.28},
			{strike = 110, expiry = 0.3, implied_vol = 0.22},
			{strike = 100, expiry = 1.5, implied_vol = 0.24},
			{strike = 90, expiry = 0.3, implied_vol = 0.30},
		}
		ref := [G]f64 {
			0.3,
			0.2959317202017594,
			0.24728781346783962,
			0.22,
			0.22,
			0.2866445001981601,
			0.27471441243961015,
			0.2402966130262268,
			0.21870654917448457,
			0.21505813167606566,
			0.27620440187340023,
			0.25765738382082337,
			0.23491671266324277,
			0.21773141858875045,
			0.21127588598796598,
			0.2726355424146053,
			0.251714985216431,
			0.23309582101369244,
			0.21740540311506168,
			0.21,
		}
		grid: [G]f64
		ok := ml_fin.deep_heston_grid_from_surface(quotes, 100, 0.03, grid[:])
		err := 0.0
		for v, i in grid {err = max(err, abs(v - ref[i]))}
		// quotes exactly on the grid come back unchanged
		p := fin.Heston_Params {
			v0    = 0.03,
			kappa = 2.5,
			theta = 0.06,
			sigma = 0.5,
			rho   = -0.6,
		}
		ivs: [G]f64
		ml_fin.deep_heston_surface(p, ivs[:])
		on_grid := make([]fin.VolSurfacePoint, G)
		M := ml_fin.DEEP_HESTON_MONEYNESS
		E := ml_fin.DEEP_HESTON_EXPIRIES
		for T, j in E {
			for m, i in M {
				// stored in reverse to check the sorting
				on_grid[G - 1 - (j * len(M) + i)] = {
					strike      = m * 150 * math.exp(0.02 * T),
					expiry      = T,
					implied_vol = ivs[j * len(M) + i],
				}
			}
		}
		back: [G]f64
		ok = ok && ml_fin.deep_heston_grid_from_surface(on_grid, 150, 0.02, back[:])
		err2 := 0.0
		for v, i in back {err2 = max(err2, abs(v - ivs[i]))}
		expect(
			&passed,
			&total,
			ok && err < 1e-15 && err2 < 1e-15,
			fmt.tprintf(
				"quotes onto the grid vs numpy: max err %.1e; quotes on the grid points recovered: max err %.1e",
				err,
				err2,
			),
		)
	}
	fmt.printf("Deep calibration reference checks: %d/%d passed\n", passed, total)
}
