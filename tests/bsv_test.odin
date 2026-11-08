package tests

import bfin "../wotan/behavioral_finance"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"

// Barberis, Shleifer & Vishny (1998) investor sentiment (behavioral_finance/bsv.odin)
//   bsv_reference_test: the beliefs of their Table 1 (21 shocks, q0 = 0.5, pi_L = 1/3,
//     pi_H = 3/4, lambda = 0.1 / 0.3) to the printed two decimals; the belief bounds
//     0.28 / 0.95 of the appendix; p1, p2 (matrix inverse) = a direct series of the
//     investor's expected discounted shocks (numpy, 3000 periods, delta 0.1 and 0.05);
//     pi_L = pi_H = 1/2 (the investor believes the random walk) gives p1 = p2 = 0; the
//     appendix's expected price change f(q) = enumeration of the two next shocks;
//     Proposition 2: both conditions hold for the paper's parameters, and on a grid of
//     (pi_L, pi_H) every pair passing the underreaction bound has a positive expected
//     gap E[f(q)] over the beliefs' stationary distribution (simulated); overreaction
//     only for high pi_L and pi_H, underreaction only for low ones (their Fig. 1)
//   bsv_test: their Tables 2 and 3 on simulated firms (2000 firms, 6 years, beliefs
//     after 30 years of shocks, N_1 = 1, y = 0.1, delta = 0.1, 100 samples; BSV state
//     neither N_1, y nor delta): earnings sorts r_n+ - r_n- fall from positive (n = 1,
//     underreaction) to negative (n = 3, 4, overreaction), winner - loser deciles on past
//     returns likewise, high - low E/P positive; with pi_L = pi_H = 1/2 (rational) no
//     sort earns anything (|t| < 3.5 over the samples)
//     On 9 seeds: earnings sorts 0.0219 - 0.0228, 0.0053 - 0.0063, -0.0070 to -0.0091,
//     -0.0162 to -0.0222; return sorts 0.0154 - 0.0178, -0.0011 - 0.0021, -0.0070 to
//     -0.0113, -0.0094 to -0.0145; E/P 0.032 - 0.035; rational |t| <= 2.3. Smaller than
//     the paper's numbers: they depend on y / N_1, which BSV do not report.

@(private = "file")
_bsv_rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}

bsv_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== BSV Investor Sentiment Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	p := bfin.BSV_1998

	// 1. Table 1
	signs := []f64{1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1}
	paper := []f64 {
		0.50,
		0.80,
		0.90,
		0.93,
		0.94,
		0.74,
		0.89,
		0.69,
		0.87,
		0.92,
		0.94,
		0.74,
		0.56,
		0.44,
		0.36,
		0.74,
		0.89,
		0.69,
		0.87,
		0.92,
		0.72,
	}
	q := bfin.bsv_beliefs(signs, 0.5, p)
	defer delete(q)
	e1 := 0.0
	for v, t in q {e1 = max(e1, abs(v - paper[t]))}
	ql, qh := bfin.bsv_belief_bounds(p)
	expect(
		&passed,
		&total,
		e1 <= 0.005 && abs(ql - 0.28) < 0.005 && abs(qh - 0.95) < 0.005,
		fmt.tprintf(
			"Table 1 beliefs over 21 shocks (largest deviation from the printed values %.4f); bounds q_low %.4f, q_high %.4f (paper 0.28, 0.95)",
			e1,
			ql,
			qh,
		),
	)

	// 2. p1, p2 = direct series; the rational investor
	p1, p2 := bfin.bsv_price_coefficients(p)
	p.delta = 0.05
	p1b, p2b := bfin.bsv_price_coefficients(p)
	p.delta = 0.1
	e2 := max(
		_bsv_rel(p1, 3.9099526066350707),
		_bsv_rel(p2, 5.734597156398107),
		_bsv_rel(p1b, 8.013157894736842),
		_bsv_rel(p2b, 11.60526315789474),
	)
	r1, r2 := bfin.bsv_price_coefficients({0.5, 0.5, 0.1, 0.3, 0.1})
	expect(
		&passed,
		&total,
		e2 < 1e-12 && abs(r1) < 1e-12 && abs(r2) < 1e-12,
		fmt.tprintf(
			"Proposition 1: p1 %.6f, p2 %.6f (delta 0.1) = the direct series of expected discounted shocks (relative error %.1e, also delta 0.05); pi_L = pi_H = 1/2: p1, p2 = %.0e, %.0e",
			p1,
			p2,
			e2,
			r1,
			r2,
		),
	)

	// 3. f(q) = enumeration: E[P_{t+1} - P_t | y_t = +y, q] with y_{t+1} = +-y equally likely,
	// minus the same after y_t = -y
	e3 := 0.0
	for qq in ([]f64{0.3, 0.5, 0.7, 0.9}) {
		N, y := 2.0, 0.1
		P0 := bfin.bsv_price(N, y, qq, p1, p2, p.delta)
		up :=
			0.5 *
			(bfin.bsv_price(N + y, y, bfin.bsv_update_belief(qq, true, p), p1, p2, p.delta) - P0)
		up +=
			0.5 *
			(bfin.bsv_price(N - y, -y, bfin.bsv_update_belief(qq, false, p), p1, p2, p.delta) - P0)
		M0 := bfin.bsv_price(N, -y, qq, p1, p2, p.delta)
		dn :=
			0.5 *
			(bfin.bsv_price(N - y, -y, bfin.bsv_update_belief(qq, true, p), p1, p2, p.delta) - M0)
		dn +=
			0.5 *
			(bfin.bsv_price(N + y, y, bfin.bsv_update_belief(qq, false, p), p1, p2, p.delta) - M0)
		e3 = max(e3, abs((up - dn) / y - bfin.bsv_reaction_gap(qq, p1, p2, p)))
	}
	expect(
		&passed,
		&total,
		e3 < 1e-12,
		fmt.tprintf(
			"appendix f(q) = enumeration of the next shock at q = 0.3, 0.5, 0.7, 0.9 (error %.1e)",
			e3,
		),
	)

	// 4. Proposition 2
	c := bfin.bsv_sufficient_conditions(p)
	n_under, n_bad, n_over_only, n_under_only, n_pairs := 0, 0, 0, 0, 0
	worst := math.inf_f64(1)
	for i in 1 ..= 9 {
		for j in 1 ..= 9 {
			g := bfin.BSV_Params{0.05 * f64(i), 0.5 + 0.05 * f64(j), 0.1, 0.3, 0.1}
			cg := bfin.bsv_sufficient_conditions(g)
			n_pairs += 1
			if cg.overreaction && !cg.underreaction {n_over_only += 1}
			if cg.underreaction && !cg.overreaction {n_under_only += 1}
			if !cg.underreaction {continue}
			n_under += 1
			// E[f(q)] over the stationary beliefs: simulate the true random walk's signs
			qq, s := 0.5, 0.0
			for t in 0 ..< 200_000 {
				qq = bfin.bsv_update_belief(qq, rand.float64() < 0.5, g)
				if t >= 1000 {s += bfin.bsv_reaction_gap(qq, cg.p1, cg.p2, g)}
			}
			worst = min(worst, s / 199_000)
			if s <= 0 {n_bad += 1}
		}
	}
	hi := bfin.bsv_sufficient_conditions({0.45, 0.95, 0.1, 0.3, 0.1})
	lo := bfin.bsv_sufficient_conditions({0.05, 0.55, 0.1, 0.3, 0.1})
	expect(
		&passed,
		&total,
		c.overreaction &&
		c.underreaction &&
		c.k_low * c.p2 < c.p1 &&
		c.p1 < c.k_high * c.p2 &&
		n_bad == 0 &&
		n_under > 10 &&
		hi.overreaction &&
		!hi.underreaction &&
		!lo.overreaction &&
		lo.underreaction,
		fmt.tprintf(
			"Proposition 2: %.4f p2 < p1 < %.4f p2 for the paper's parameters; on %d (pi_L, pi_H) pairs %d pass the underreaction bound, all with E[f(q)] > 0 (smallest %.4f); (0.45, 0.95) only overreaction, (0.05, 0.55) only underreaction (Fig. 1)",
			c.k_low,
			c.k_high,
			n_pairs,
			n_under,
			worst,
		),
	)

	fmt.printf("BSV reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

// Tables 2 and 3: per sample the sort spreads; returns the means and t-statistics over
// samples. [0..3] earnings sorts n = 1..4, [4..7] return sorts, [8] E/P
@(private = "file")
_bsv_tables :: proc(p: bfin.BSV_Params, samples: int) -> (mean, t: [9]f64) {
	F, Y, BURN :: 2000, 6, 30
	N1, Y_SHOCK :: 1.0, 0.1
	p1, p2 := bfin.bsv_price_coefficients(p)
	S := make([]f64, F * Y, context.temp_allocator) // shocks
	P := make([]f64, F * Y, context.temp_allocator)
	N := make([]f64, F * Y, context.temp_allocator)
	R := make([]f64, F * Y, context.temp_allocator) // R[f*Y + t]: return over t -> t + 1
	key := make([]f64, F, context.temp_allocator)
	idx := make([]int, F, context.temp_allocator)
	defer {
		delete(S, context.temp_allocator)
		delete(P, context.temp_allocator)
		delete(N, context.temp_allocator)
		delete(R, context.temp_allocator)
		delete(key, context.temp_allocator)
		delete(idx, context.temp_allocator)
	}
	sum, sum2: [9]f64
	for _ in 0 ..< samples {
		for f in 0 ..< F {
			q, last := 0.5, 1.0 if rand.float64() < 0.5 else -1.0
			for _ in 0 ..< BURN {
				y := 1.0 if rand.float64() < 0.5 else -1.0
				q = bfin.bsv_update_belief(q, y == last, p)
				last = y
			}
			n := N1
			for t in 0 ..< Y {
				y := 1.0 if rand.float64() < 0.5 else -1.0
				q = bfin.bsv_update_belief(q, y == last, p)
				last = y
				n += Y_SHOCK * y
				S[f * Y + t], N[f * Y + t] = y, n
				P[f * Y + t] = bfin.bsv_price(n, Y_SHOCK * y, q, p1, p2, p.delta)
			}
			for t in 0 ..< Y - 1 {
				R[f * Y + t] = (P[f * Y + t + 1] + N[f * Y + t + 1]) / P[f * Y + t] - 1
			}
		}
		spread: [9]f64
		count: [9]int
		for n in 1 ..= 4 {
			// earnings: n shocks of one sign ending at t, return over t -> t + 1
			for t in n - 1 ..< Y - 1 {
				sp, np, sn, nn := 0.0, 0, 0.0, 0
				for f in 0 ..< F {
					pos, neg := true, true
					for k in t - n + 1 ..= t {
						pos &&= S[f * Y + k] > 0
						neg &&= S[f * Y + k] < 0
					}
					if pos {sp += R[f * Y + t]; np += 1}
					if neg {sn += R[f * Y + t]; nn += 1}
				}
				if np > 0 &&
				   nn > 0 {spread[n - 1] += sp / f64(np) - sn / f64(nn); count[n - 1] += 1}
			}
			// returns: deciles on the cumulative return over t - n .. t, return over t -> t + 1
			for t in n ..< Y - 1 {
				for f in 0 ..< F {
					g := 1.0
					for k in t - n ..< t {g *= 1 + R[f * Y + k]}
					key[f], idx[f] = g, f
				}
				_bsv_sort(idx, key)
				spread[3 + n] += _bsv_decile_spread(idx, R, Y, t)
				count[3 + n] += 1
			}
		}
		for t in 0 ..< Y - 1 {
			for f in 0 ..< F {key[f], idx[f] = N[f * Y + t] / P[f * Y + t], f}
			_bsv_sort(idx, key)
			spread[8] += _bsv_decile_spread(idx, R, Y, t)
			count[8] += 1
		}
		for k in 0 ..< 9 {
			v := spread[k] / f64(count[k])
			sum[k] += v
			sum2[k] += v * v
		}
	}
	for k in 0 ..< 9 {
		m := sum[k] / f64(samples)
		sd := math.sqrt((sum2[k] - f64(samples) * m * m) / f64(samples - 1))
		mean[k], t[k] = m, m / (sd / math.sqrt(f64(samples)))
	}
	return
}

@(private = "file")
_bsv_sort :: proc(idx: []int, key: []f64) {
	context.user_ptr = raw_data(key)
	slice.sort_by(idx, proc(a, b: int) -> bool {
		k := ([^]f64)(context.user_ptr)
		return k[a] < k[b] || (k[a] == k[b] && a < b)
	})
}

// top minus bottom decile of idx (sorted ascending) in the return over t -> t + 1
@(private = "file")
_bsv_decile_spread :: proc(idx: []int, R: []f64, Y, t: int) -> f64 {
	k := len(idx) / 10
	hi, lo := 0.0, 0.0
	for i in 0 ..< k {
		hi += R[idx[len(idx) - 1 - i] * Y + t]
		lo += R[idx[i] * Y + t]
	}
	return (hi - lo) / f64(k)
}

bsv_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== BSV Investor Sentiment: Under- and Overreaction ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	SAMPLES :: 100
	m, t := _bsv_tables(bfin.BSV_1998, SAMPLES)
	expect(
		&passed,
		&total,
		m[0] > 0 && m[0] > m[1] && m[1] > m[2] && m[2] > m[3] && m[2] < 0 && t[0] > 3 && t[3] < -3,
		fmt.tprintf(
			"Table 2, earnings sorts r_n+ - r_n-: %.4f, %.4f, %.4f, %.4f for n = 1..4 (t %.1f .. %.1f; paper 0.0391, 0.0131, -0.0072, -0.0309)",
			m[0],
			m[1],
			m[2],
			m[3],
			t[0],
			t[3],
		),
	)
	expect(
		&passed,
		&total,
		m[4] > 0 &&
		m[4] > m[5] &&
		m[5] > m[6] &&
		m[6] > m[7] &&
		m[7] < 0 &&
		t[4] > 3 &&
		t[7] < -3 &&
		m[8] > 0 &&
		t[8] > 3,
		fmt.tprintf(
			"Table 3, winner - loser deciles on past returns: %.4f, %.4f, %.4f, %.4f (paper 0.0280, 0.0102, -0.0094, -0.0181); high - low E/P %.4f (t %.1f; paper 0.0435)",
			m[4],
			m[5],
			m[6],
			m[7],
			m[8],
			t[8],
		),
	)
	rm, rt := _bsv_tables({0.5, 0.5, 0.1, 0.3, 0.1}, SAMPLES)
	worst := 0.0
	for k in 0 ..< 8 {worst = max(worst, abs(rt[k]))}
	expect(
		&passed,
		&total,
		worst < 3.5,
		fmt.tprintf(
			"rational investor (pi_L = pi_H = 1/2): earnings sorts %.4f .. %.4f, return sorts %.4f .. %.4f, largest |t| %.1f (E/P is the same for every firm)",
			rm[0],
			rm[3],
			rm[4],
			rm[7],
			worst,
		),
	)
	fmt.printf("BSV: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
