package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:mem"

// IPCA (ml_finance/ipca.odin) against a numpy transcription of the authors' package
// (ipca 0.6.7, portfolio ALS: start, factor and Gamma steps, normalisation, stopping
// rule, predict_panel) on an unbalanced LCG panel: 50 assets, 36 dates, 5
// characteristics (constant + 4), 1637 observations, K = 2, without and with the
// intercept (Gamma_alpha), tolerance 1e-13. Gamma, factors, the Kelly-Pruitt-Su total /
// predictive R² and an out-of-sample factor from Gamma.

@(private = "file")
_IP_GAMMA0 := [10]f64 {
	0.1137616687179472,
	-0.42991757487667576,
	0.91693117669399749,
	-0.20415464049188281,
	0.014490917546882845,
	-0.47233563163431103,
	-0.37955649955735171,
	-0.67623604362676648,
	0.04497084488296943,
	-0.30511575830563359,
}
@(private = "file")
_IP_GAMMA1 := [15]f64 {
	0.40098549943214573,
	-0.019427648315224925,
	-0.0012692877119620811,
	0.87176309108664185,
	0.08934088259216566,
	0.0015529963078333381,
	0.02198316638276801,
	-0.92154234494343967,
	0.00011983172637629991,
	0.22628994525615401,
	0.053738020664299876,
	-0.0034818522518845038,
	0.16597994859612081,
	-0.37351399544036262,
	-0.00035910976279232912,
}
// factors [:, :3] (row-major), sum, sum of squares, total R², predictive R², f at date 5
@(private = "file")
_IP_REST0 := [12]f64 {
	0.0023308080824438995,
	0.0059338062899968527,
	0.010369685317559187,
	0.0032966388790454113,
	-0.00039011744368757121,
	-0.0017904815920953708,
	0.21670493484645151,
	0.0014291187908277805,
	0.30588020024169593,
	0.2218869520744835,
	0.003350760850462842,
	0.0032773300066678639,
}
@(private = "file")
_IP_REST1 := [12]f64 {
	0.00037740702143879991,
	0.0051138308812568502,
	0.0086271623826111882,
	0.003074795167417091,
	-0.0026834793592092321,
	-0.00075491636884339778,
	0.1432619817984935,
	0.00097396941513528941,
	0.32677067291407924,
	0.22130008946740543,
	0.00041229055260544344,
	-0.00062672019795527868,
}

@(private = "file")
_ip_u :: proc(x: ^u64) -> f64 {
	x^ = 6364136223846793005 * x^ + 1442695040888963407
	return f64(x^ >> 11) * 1.1102230246251565e-16
}

@(private = "file")
_ip_err :: proc(got, want: []f64) -> f64 {
	e := 0.0
	for v, i in want {
		d := abs(got[i] - v)
		e = max(e, d / abs(v) if abs(v) > 1e-3 else d / 1e-3)
	}
	return e
}

ipca_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== IPCA Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	N, T, L :: 50, 36, 5
	x: u64 = 31
	Gb: [L][2]f64
	for a in 0 ..< L {for k in 0 ..< 2 {Gb[a][k] = _ip_u(&x) - 0.5}}
	Ga := [L]f64{0, 0.004, 0, -0.003, 0}
	Ft: [2][T]f64
	for k in 0 ..< 2 {for t in 0 ..< T {Ft[k][t] = 0.01 * (_ip_u(&x) - 0.4)}}
	rows := make([dynamic]f64)
	defer delete(rows)
	y := make([dynamic]f64)
	defer delete(y)
	times := make([dynamic]int)
	defer delete(times)
	for t in 0 ..< T {
		for i in 0 ..< N {
			z: [L]f64
			z[0] = 1
			for a in 1 ..< L {z[a] = 2 * (_ip_u(&x) - 0.5)}
			e := 0.02 * (_ip_u(&x) - 0.5)
			if (i * 7 + t * 3) % 11 == 0 {continue}
			r := e
			for a in 0 ..< L {
				r += z[a] * Ga[a]
				for k in 0 ..< 2 {r += z[a] * Gb[a][k] * Ft[k][t]}
			}
			append(&rows, ..z[:])
			append(&y, r)
			append(&times, t)
		}
	}
	X := l.Matrix(f64) {
		data = rows[:],
		rows = len(y),
		cols = L,
	}
	ysum := 0.0
	for v in y {ysum += v}
	expect(
		&passed,
		&total,
		len(y) == 1637 && abs(ysum - 0.61682600940368937) < 1e-13,
		fmt.tprintf(
			"the panel: %d observations (unbalanced), the same returns as the reference",
			len(y),
		),
	)

	for intercept in ([]bool{false, true}) {
		m := ml_fin.ipca_fit(
			&X,
			y[:],
			times[:],
			{n_factors = 2, intercept = intercept, tol = 1e-13},
		)
		defer ml_fin.ipca_free(&m)
		want_g := _IP_GAMMA1[:] if intercept else _IP_GAMMA0[:]
		want_r := _IP_REST1[:] if intercept else _IP_REST0[:]
		got: [12]f64
		for k in 0 ..< 2 {for t in 0 ..< 3 {got[k * 3 + t] = m.factors.data[k * T + t]}}
		for v in m.factors.data {got[6] += v; got[7] += v * v}
		got[8], got[9] = ml_fin.ipca_r2(&m, &X, y[:], times[:])
		// out-of-sample factor at date 5
		z5 := make([dynamic]f64, context.temp_allocator)
		r5 := make([dynamic]f64, context.temp_allocator)
		for t, k in times {
			if t == 5 {append(&z5, ..rows[k * L:][:L]); append(&r5, y[k])}
		}
		Z5 := l.Matrix(f64) {
			data = z5[:],
			rows = len(r5),
			cols = L,
		}
		ml_fin.ipca_factor_at(&m, &Z5, r5[:], got[10:12])
		eg := _ip_err(m.gamma.data, want_g)
		er := _ip_err(got[:], want_r)
		expect(
			&passed,
			&total,
			m.converged && eg < 1e-9 && er < 1e-9,
			fmt.tprintf(
				"K 2, intercept %v: converged in %d iterations (reference %d); Gamma, factors, total / predictive R² %.4f / %.4f and an out-of-sample factor = reference (relative error %.1e, %.1e)",
				intercept,
				m.n_iter,
				44 if intercept else 61,
				got[8],
				got[9],
				eg,
				er,
			),
		)
	}
	fmt.printf("IPCA reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
