package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// TimeGAN against PyTorch and finite differences: the supervised loss
// MSE(H[:, 1:], S(H)[:, :-1]) and the moment loss of the authors' code (values and
// gradients); the gradients of the generator and discriminator losses for every
// parameter of the five networks; the data windows and the min-max scaler.

@(private = "file")
TG_LS_REF :: 0.16055555555555556
@(private = "file")
TG_LS_GH := [?]f64 {
	0.0,
	0.0,
	0.03333333333333334,
	0.08333333333333333,
	0.01111111111111112,
	0.011111111111111112,
	0.061111111111111116,
	-0.01111111111111112,
	0.0,
	0.0,
	0.03888888888888889,
	-0.03333333333333334,
	0.01666666666666667,
	0.01666666666666667,
	0.06666666666666667,
	0.06666666666666668,
	0.0,
	0.0,
	0.044444444444444446,
	-0.027777777777777776,
	0.022222222222222216,
	0.022222222222222223,
	-0.049999999999999996,
	0.07222222222222222,
}
@(private = "file")
TG_LS_GHS := [?]f64 {
	-0.03333333333333334,
	-0.08333333333333333,
	-0.01111111111111112,
	-0.011111111111111112,
	-0.061111111111111116,
	0.01111111111111112,
	0.0,
	0.0,
	-0.03888888888888889,
	0.03333333333333334,
	-0.01666666666666667,
	-0.01666666666666667,
	-0.06666666666666667,
	-0.06666666666666668,
	0.0,
	0.0,
	-0.044444444444444446,
	0.027777777777777776,
	-0.022222222222222216,
	-0.022222222222222223,
	0.049999999999999996,
	-0.07222222222222222,
	0.0,
	0.0,
}
@(private = "file")
TG_LV_REF :: 0.17606807079169448
@(private = "file")
TG_LV_G := [?]f64 {
	-0.007416385980427674,
	-0.08635130551299328,
	-0.07927733659101523,
	-0.0695784731952056,
	0.02627849184702572,
	0.041572869933729925,
	0.061635152992145045,
	-0.0063436661749802226,
	0.006416831888918131,
	0.016457895160307705,
	0.10636900513392791,
	0.10787111695093587,
	0.09930938337201808,
	0.005770287926541613,
	-0.007135001726724252,
	-0.022637391060009598,
	0.04255650860802776,
	0.022323973128027957,
	0.013138516282931199,
	-0.07974198290523479,
	-0.08667116023784531,
	-0.09090869757175916,
	-0.008537338922314737,
	-0.005101293346027103,
}

@(private = "file")
_tg_leaf :: proc(shape: [4]int, grad: bool, alloc: mem.Allocator) -> ^t.Tensor {
	x := t.tensor_new(l.matrix_new(f64, 1, shape[0] * shape[1] * shape[2], alloc), grad, alloc)
	x.shape = shape
	return x
}

timegan_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== TimeGAN Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)

	// 1. supervised loss, B = 3, T = 4, L = 2
	{
		t.graph_begin(&scope)
		defer t.graph_end(&scope)
		a := t.graph_allocator(&scope)
		h, hs := _tg_leaf({3, 4, 2, 1}, true, a), _tg_leaf({3, 4, 2, 1}, true, a)
		for i in 0 ..< 24 {
			h.data.data[i] = 0.1 * f64((i * 7) % 11)
			hs.data.data[i] = 0.05 * f64((i * 5) % 13)
		}
		loss := ml_fin.timegan_supervised_loss(h, hs, 3)
		t.tensor_backward(loss)
		err := 0.0
		for v, i in TG_LS_GH {err = max(err, abs(h.grad.data[i] - v))}
		for v, i in TG_LS_GHS {err = max(err, abs(hs.grad.data[i] - v))}
		err_l := abs(loss.data.data[0] - TG_LS_REF)
		expect(
			&passed,
			&total,
			err_l < 1e-15 && err < 1e-15,
			fmt.tprintf(
				"supervised loss (next-step latent) vs PyTorch: value err %.1e, gradients %.1e",
				err_l,
				err,
			),
		)
	}

	// 2. moment loss, B = 4, T = 3, F = 2
	{
		t.graph_begin(&scope)
		defer t.graph_end(&scope)
		a := t.graph_allocator(&scope)
		x, xh := _tg_leaf({4, 3, 2, 1}, false, a), _tg_leaf({4, 3, 2, 1}, true, a)
		for i in 0 ..< 24 {
			x.data.data[i] = math.sin(0.7 * f64(i))
			xh.data.data[i] = 0.8 * math.cos(0.3 * f64(i))
		}
		loss := ml_fin.timegan_moment_loss(xh, x, 4, a)
		t.tensor_backward(loss)
		err := 0.0
		for v, i in TG_LV_G {err = max(err, abs(xh.grad.data[i] - v))}
		err_l := abs(loss.data.data[0] - TG_LV_REF)
		expect(
			&passed,
			&total,
			err_l < 1e-14 && err < 1e-14,
			fmt.tprintf(
				"moment loss (batch std and mean) vs PyTorch: value err %.1e, gradient %.1e",
				err_l,
				err,
			),
		)
	}

	// 3. generator and discriminator losses: every parameter vs central differences
	{
		B, T, F, Z :: 6, 4, 2, 3
		tg := ml_fin.new_timegan(T, F, Z, 4, 1e-3, allocator)
		defer ml_fin.timegan_free(tg)
		ml_fin.timegan_init_weights(tg)
		xd := make([]f64, B * T * F)
		zd := make([]f64, B * T * Z)
		for &v in xd {v = rand.float64()}
		for &v in zd {v = rand.float64()}
		value :: proc(tg: ^ml_fin.TimeGAN, xd, zd: []f64, gen: bool, backward: bool) -> f64 {
			t.graph_begin(&tg.scope, no_grad = !backward)
			defer t.graph_end(&tg.scope)
			a := t.graph_allocator(&tg.scope)
			x, z := _tg_leaf({B, T, F, 1}, false, a), _tg_leaf({B, T, Z, 1}, false, a)
			copy(x.data.data, xd)
			copy(z.data.data, zd)
			loss: ^t.Tensor
			if gen {loss, _ = ml_fin.timegan_generator_loss(tg, x, z, a)} else {loss = ml_fin.timegan_discriminator_loss(tg, x, z, a)}
			if backward {t.tensor_backward(loss)}
			return loss.data.data[0]
		}
		for gen in ([]bool{true, false}) {
			params := make([dynamic]^t.Tensor, context.temp_allocator)
			for o in ([]^nn.Adam{&tg.optimizer_e, &tg.optimizer_r, &tg.optimizer_g, &tg.optimizer_s, &tg.optimizer_d}) {
				for p in o.parameters {
					for &g in p.grad.data {g = 0}
					append(&params, p)
				}
			}
			value(tg, xd, zd, gen, true)
			max_err, n, with_grad := 0.0, 0, 0
			h := 1e-4 // the generator loss is ill-conditioned (100 sqrt(L_S), sqrt of batch variances): smaller steps measure round-off (error ~ 1/h)
			for p in params {
				nonzero := false
				for i in 0 ..< len(p.data.data) {
					x0 := p.data.data[i]
					p.data.data[i] = x0 + h
					up := value(tg, xd, zd, gen, false)
					p.data.data[i] = x0 - h
					down := value(tg, xd, zd, gen, false)
					p.data.data[i] = x0
					fd := (up - down) / (2 * h)
					max_err = max(max_err, abs(fd - p.grad.data[i]) / max(1e-2, abs(fd)))
					nonzero ||= p.grad.data[i] != 0
					n += 1
				}
				if nonzero {with_grad += 1}
			}
			// the generator loss reaches every network; the discriminator's all but the recovery
			want := len(params) if gen else len(params) - len(tg.optimizer_r.parameters)
			expect(
				&passed,
				&total,
				max_err < 1e-5 && with_grad == want,
				fmt.tprintf(
					"%s loss gradient vs finite differences: %d values, max error %.1e; %d/%d parameter tensors get a gradient",
					"generator" if gen else "discriminator",
					n,
					max_err,
					with_grad,
					want,
				),
			)
		}
	}

	// 4. data windows and scaler
	{
		n_days, T :: 30, 5
		prices, volumes := make([]f64, n_days), make([]f64, n_days)
		ind := make([]f64, n_days)
		for i in 0 ..< n_days {
			prices[i] = 100 * math.exp(0.01 * math.sin(f64(i)))
			volumes[i] = 1e6 * (1.5 + math.cos(0.5 * f64(i)))
			ind[i] = f64(i)
		}
		ind_copy := make([]f64, n_days)
		copy(ind_copy, ind)
		seqs, n, fdim := ml_fin.prepare_market_data(prices, volumes, ind, 1, T, allocator)
		mean, sd := 14.5, 0.0
		for v in ind {sd += (v - mean) * (v - mean) / n_days}
		sd = math.sqrt(sd)
		err := 0.0
		for s in 0 ..< n {
			for k in 0 ..< T {
				day := 1 + s + k
				row := seqs[(s * T + k) * 3:][:3]
				err = max(err, abs(row[0] - math.ln(prices[day] / prices[day - 1])))
				err = max(err, abs(row[1] - math.ln(volumes[day] / volumes[day - 1])))
				err = max(err, abs(row[2] - clamp((ind[day] - mean) / sd, -3, 3)))
			}
		}
		same := true
		for v, i in ind {same &&= v == ind_copy[i]}
		sc := ml_fin.timegan_scaler_fit(seqs, fdim, allocator)
		scaled := make([]f64, len(seqs))
		copy(scaled, seqs)
		ml_fin.timegan_scale(scaled, sc)
		lo, hi := 1.0, 0.0
		for v in scaled {lo, hi = min(lo, v), max(hi, v)}
		ml_fin.timegan_unscale(scaled, sc)
		rt := 0.0
		for v, i in scaled {rt = max(rt, abs(v - seqs[i]))}
		expect(
			&passed,
			&total,
			n == n_days - T &&
			fdim == 3 &&
			err < 1e-15 &&
			same &&
			lo >= 0 &&
			hi <= 1 &&
			hi > 0.999 &&
			rt < 1e-12,
			fmt.tprintf(
				"%d windows of %d days from day 1 (max err %.1e), indicators left unchanged %v; scaled to [%.3f, %.7f], round trip %.1e",
				n,
				T,
				err,
				same,
				lo,
				hi,
				rt,
			),
		)
	}
	fmt.printf("TimeGAN reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
