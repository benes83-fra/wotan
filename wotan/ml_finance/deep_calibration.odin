package ml_finance

import fin "../finance"
import l "../linalg"
import nn "../nn"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

// ============================================================================
// Deep Heston Calibrator (inverse map, Hernandez 2016; implied-vol inputs as in
// Horvath, Muguruza & Tomas 2021)
// ============================================================================
// An MLP maps the implied volatilities of a fixed grid of options to the Heston
// parameters, replacing an iterative solver by one forward pass. The grid is in forward
// moneyness K / F (F = S e^{rT}) and expiry, where Heston implied vols do not depend on S
// or r, so one network serves any spot and rate. The parameters are learnt scaled to
// [-1, 1] over the sampling box, so each counts the same in the loss.

DEEP_HESTON_MONEYNESS :: [5]f64{0.8, 0.9, 1.0, 1.1, 1.2} // K / F
DEEP_HESTON_EXPIRIES :: [4]f64{0.25, 0.5, 1.0, 2.0} // years
DEEP_HESTON_N_GRID :: len(DEEP_HESTON_MONEYNESS) * len(DEEP_HESTON_EXPIRIES)

// Sampling box of the training set (v0, kappa, theta, sigma, rho): volatilities 10%-40%;
// the Feller condition is not imposed (pricing does not need it)
DEEP_HESTON_LO :: [5]f64{0.01, 0.5, 0.01, 0.1, -0.95}
DEEP_HESTON_HI :: [5]f64{0.16, 5.0, 0.16, 1.0, 0.0}

DeepHestonCalibrator :: struct {
	mlp:        ^nn.Sequential,
	input_dim:  int, // DEEP_HESTON_N_GRID implied vols, expiry-major
	output_dim: int, // 5 scaled parameters (v0, kappa, theta, sigma, rho)
	allocator:  mem.Allocator,
}

deep_heston_calibrator_new :: proc(
	hidden_dim: int,
	allocator: mem.Allocator = context.allocator,
) -> DeepHestonCalibrator {
	model := DeepHestonCalibrator {
		input_dim  = DEEP_HESTON_N_GRID,
		output_dim = 5,
		allocator  = allocator,
		mlp        = nn.sequential_new(allocator),
	}
	nn.sequential_add(model.mlp, nn.linear_layer_new(model.input_dim, hidden_dim, allocator))
	nn.sequential_add(model.mlp, nn.Activation.ReLU)
	nn.sequential_add(model.mlp, nn.linear_layer_new(hidden_dim, hidden_dim, allocator))
	nn.sequential_add(model.mlp, nn.Activation.ReLU)
	nn.sequential_add(model.mlp, nn.linear_layer_new(hidden_dim, 5, allocator))
	return model
}

deep_heston_calibrator_free :: proc(model: ^DeepHestonCalibrator) {
	if model.mlp != nil {nn.sequential_free(model.mlp)}
}

// features [batch, DEEP_HESTON_N_GRID] (see deep_heston_features) -> scaled parameters [batch, 5]
deep_heston_calibrator_forward :: proc(
	model: ^DeepHestonCalibrator,
	features: ^t.Tensor,
) -> ^t.Tensor {
	return nn.sequential_forward(model.mlp, features)
}

deep_heston_calibrator_add_to_adam :: proc(model: ^DeepHestonCalibrator, opt: ^nn.Adam) {
	nn.sequential_add_to_adam(model.mlp, opt)
}

// Network input of a grid of implied vols: (iv - 0.25) / 0.1
deep_heston_features :: proc(ivs: []f64, out: []f64) {
	assert(len(ivs) == DEEP_HESTON_N_GRID && len(out) == DEEP_HESTON_N_GRID)
	for v, i in ivs {out[i] = (v - 0.25) / 0.1}
}

// Parameters -> [-1, 1] over the sampling box, and back (clamped to the box)
deep_heston_encode :: proc(p: fin.Heston_Params) -> (x: [5]f64) {
	v := [5]f64{p.v0, p.kappa, p.theta, p.sigma, p.rho}
	lo, hi := DEEP_HESTON_LO, DEEP_HESTON_HI
	for k in 0 ..< 5 {x[k] = 2 * (v[k] - lo[k]) / (hi[k] - lo[k]) - 1}
	return
}

deep_heston_decode :: proc(x: [5]f64) -> fin.Heston_Params {
	lo, hi := DEEP_HESTON_LO, DEEP_HESTON_HI
	v: [5]f64
	for k in 0 ..< 5 {v[k] = lo[k] + (clamp(x[k], -1, 1) + 1) / 2 * (hi[k] - lo[k])}
	return {v0 = v[0], kappa = v[1], theta = v[2], sigma = v[3], rho = v[4]}
}

// Implied vols of the grid under Heston (expiry-major: out[j * 5 + i] for expiry j,
// moneyness i); false when a price has no implied vol (outside the no-arbitrage bounds)
deep_heston_surface :: proc(p: fin.Heston_Params, out: []f64) -> bool {
	assert(
		len(out) == DEEP_HESTON_N_GRID,
		"deep_heston_surface: out needs DEEP_HESTON_N_GRID slots",
	)
	M := DEEP_HESTON_MONEYNESS
	NM :: len(DEEP_HESTON_MONEYNESS)
	// S = 1, r = 0: K = K / F
	prices: [NM]f64
	for T, j in DEEP_HESTON_EXPIRIES {
		fin.heston_prices_same_expiry(1, M[:], T, 0, p, .Call, prices[:])
		for i in 0 ..< NM {
			iv, ok, _ := fin.implied_volatility(prices[i], 1, M[i], T, 0, .Call, tol = 1e-12)
			if !ok {return false}
			out[j * NM + i] = iv
		}
	}
	return true
}

// ============================================================================
// Dataset
// ============================================================================
// num_samples parameter sets drawn uniformly from the box (redrawn when an implied vol
// does not exist); inputs: deep_heston_features of their surfaces [N, DEEP_HESTON_N_GRID];
// targets: deep_heston_encode [N, 5]
generate_heston_calibration_dataset :: proc(
	num_samples: int,
	allocator: mem.Allocator,
) -> (
	inputs: ^t.Tensor,
	targets: ^t.Tensor,
) {
	G :: DEEP_HESTON_N_GRID
	inputs_data := l.matrix_new(f64, num_samples, G, allocator)
	targets_data := l.matrix_new(f64, num_samples, 5, allocator)
	lo, hi := DEEP_HESTON_LO, DEEP_HESTON_HI
	ivs: [G]f64
	for s in 0 ..< num_samples {
		p: fin.Heston_Params
		for {
			v: [5]f64
			for k in 0 ..< 5 {v[k] = lo[k] + rand.float64() * (hi[k] - lo[k])}
			p = {
				v0    = v[0],
				kappa = v[1],
				theta = v[2],
				sigma = v[3],
				rho   = v[4],
			}
			if deep_heston_surface(p, ivs[:]) {break}
		}
		deep_heston_features(ivs[:], inputs_data.data[s * G:][:G])
		x := deep_heston_encode(p)
		copy(targets_data.data[s * 5:][:5], x[:])
	}
	return t.tensor_new(
		inputs_data,
		false,
		allocator,
	), t.tensor_new(targets_data, false, allocator)
}

// One calibration: the grid's implied vols -> Heston parameters (no graph is kept)
deep_heston_calibrate :: proc(model: ^DeepHestonCalibrator, ivs: []f64) -> fin.Heston_Params {
	scope: t.Graph_Scope
	t.graph_begin(&scope, no_grad = true)
	defer {t.graph_end(&scope); t.graph_scope_destroy(&scope)}
	alloc := t._scope_or(context.allocator)
	x := l.matrix_new(f64, 1, DEEP_HESTON_N_GRID, alloc)
	deep_heston_features(ivs, x.data)
	out := deep_heston_calibrator_forward(model, t.tensor_new(x, false, alloc))
	return deep_heston_decode(
		{out.data.data[0], out.data.data[1], out.data.data[2], out.data.data[3], out.data.data[4]},
	)
}

// Market implied vols on the grid from scattered quotes: per quoted expiry linear in
// strike at K = m S e^{r T_e} (flat beyond the quoted strikes), then total variance
// linear in expiry between the two quoted expiries around the grid expiry (flat implied
// vol beyond). Expiries with fewer than two quotes are ignored. False when no expiry is
// left.
deep_heston_grid_from_surface :: proc(
	points: []fin.VolSurfacePoint,
	S, r: f64,
	out: []f64,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	assert(len(out) == DEEP_HESTON_N_GRID)
	pts := slice.clone(points, allocator)
	defer delete(pts, allocator)
	slice.sort_by(pts, proc(a, b: fin.VolSurfacePoint) -> bool {
		return a.expiry < b.expiry || (a.expiry == b.expiry && a.strike < b.strike)
	})
	// runs of equal expiry with at least two strikes
	Slice :: struct {
		T:     f64,
		quote: []fin.VolSurfacePoint,
	}
	slices := make([dynamic]Slice, 0, allocator)
	defer delete(slices)
	for start := 0; start < len(pts); {
		end := start
		for end < len(pts) && pts[end].expiry == pts[start].expiry {end += 1}
		if end - start >= 2 {append(&slices, Slice{pts[start].expiry, pts[start:end]})}
		start = end
	}
	if len(slices) == 0 {return false}
	iv_at :: proc(s: Slice, K: f64) -> f64 {
		q := s.quote
		if K <= q[0].strike {return q[0].implied_vol}
		for k in 1 ..< len(q) {
			if K <= q[k].strike {
				w := (K - q[k - 1].strike) / (q[k].strike - q[k - 1].strike)
				return (1 - w) * q[k - 1].implied_vol + w * q[k].implied_vol
			}
		}
		return q[len(q) - 1].implied_vol
	}
	M := DEEP_HESTON_MONEYNESS
	for T, j in DEEP_HESTON_EXPIRIES {
		hi := 0
		for hi < len(slices) && slices[hi].T < T {hi += 1}
		for m, i in M {
			v: f64
			if hi == 0 || hi == len(slices) {
				s := slices[0] if hi == 0 else slices[len(slices) - 1]
				v = iv_at(s, m * S * math.exp(r * s.T))
			} else {
				a, b := slices[hi - 1], slices[hi]
				va := iv_at(a, m * S * math.exp(r * a.T))
				vb := iv_at(b, m * S * math.exp(r * b.T))
				w := (T - a.T) / (b.T - a.T)
				v = math.sqrt(((1 - w) * va * va * a.T + w * vb * vb * b.T) / T)
			}
			out[j * len(M) + i] = v
		}
	}
	return true
}
