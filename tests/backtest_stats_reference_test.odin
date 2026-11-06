package tests

import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:math"
import "core:mem"

// Backtest statistics against scipy / numpy (the reference script recomputes each
// formula independently; CSCV with itertools.combinations, numpy.array_split and
// scipy.stats.rankdata) and against the worked example of Bailey & López de Prado (2014),
// "The deflated Sharpe ratio": annual SR 2.5 over 5 years of daily returns, skewness -3,
// kurtosis 10, the best of 100 trials whose annual SRs have variance 0.5 -> SR0 0.1132
// per day, DSR 0.9004.

@(private = "file")
_bs_lcg_matrix :: proc(T, N: int) -> l.Matrix(f64) {
	m := l.matrix_new(f64, T, N)
	x: u64 = 12345
	for t in 0 ..< T {
		for j in 0 ..< N {
			x = 6364136223846793005 * x + 1442695040888963407
			m.data[t * N + j] =
				0.01 * (f64(x >> 11) * 1.1102230246251565e-16 - 0.5) + 0.0002 * f64(j % 3)
		}
	}
	return m
}

@(private = "file")
_bs_sharpe_metric :: proc(r: []f64) -> f64 {return ml_fin.sharpe_stats(r).sharpe}

backtest_stats_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Backtest Statistics Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}

	// A. the paper's example
	{
		sr := 2.5 / math.sqrt(250.0)
		sr0 := ml_fin.expected_max_sharpe(100, 0.5 / 250)
		dsr := ml_fin.deflated_sharpe_ratio(sr, 1250, -3, 10, 100, 0.5 / 250)
		expect(
			&passed,
			&total,
			abs(sr0 - 0.11317200186513214) < 1e-15 &&
			abs(dsr - 0.90039683444939034) < 1e-14 &&
			abs(sr0 - 0.1132) < 5e-5 &&
			abs(dsr - 0.9004) < 5e-5,
			fmt.tprintf(
				"paper example: SR0 %.4f (paper 0.1132), DSR %.4f (paper 0.9004); scipy errors %.1e, %.1e",
				sr0,
				dsr,
				abs(sr0 - 0.11317200186513214),
				abs(dsr - 0.90039683444939034),
			),
		)
	}
	// B. PSR, minimum track record length, expected maximum vs scipy
	{
		Case :: struct {
			sr, b: f64,
			n:     int,
			sk, k: f64,
			psr:   f64,
			trl:   f64,
		}
		cases := []Case {
			{0.1, 0.0, 500, -0.5, 6.0, 0.98488768811843119, 288.46399199763766},
			{0.05, 0.02, 252, 0.3, 4.0, 0.68383129282330779, 2967.703551400456},
			{-0.02, 0.0, 1000, -1.2, 9.0, 0.26121564248078227, math.INF_F64},
			{0.2, 0.1, 60, 0.0, 3.0, 0.77653608476998881, 276.96543231773217},
		}
		err_psr, err_trl := 0.0, 0.0
		inf_ok := true
		for c in cases {
			err_psr = max(
				err_psr,
				abs(ml_fin.probabilistic_sharpe_ratio(c.sr, c.b, c.n, c.sk, c.k) - c.psr),
			)
			trl := ml_fin.min_track_record_length(c.sr, c.b, c.sk, c.k)
			if math.is_inf(
				c.trl,
			) {inf_ok &&= math.is_inf(trl)} else {err_trl = max(err_trl, rel(trl, c.trl))}
		}
		emax := [3][3]f64 {
			{2, 0.01, 0.05197553442805939},
			{10, 0.003, 0.086244300865638446},
			{1000, 0.0005, 0.072786729795494984},
		}
		err_e := 0.0
		for e in emax {err_e = max(err_e, rel(ml_fin.expected_max_sharpe(int(e[0]), e[1]), e[2]))}
		expect(
			&passed,
			&total,
			err_psr < 1e-14 &&
			err_trl < 1e-12 &&
			inf_ok &&
			err_e < 1e-13 &&
			ml_fin.expected_max_sharpe(1, 1) == 0,
			fmt.tprintf(
				"PSR (4 cases) max err %.1e; minimum track record length rel err %.1e (+Inf when SR <= benchmark %v); expected max SR rel err %.1e",
				err_psr,
				err_trl,
				inf_ok,
				err_e,
			),
		)
	}
	// C. Sharpe ratio, skewness, kurtosis and PSR of a series vs numpy / scipy.stats
	{
		r := make([]f64, 500)
		for &v, t in r {
			ft := f64(t)
			c := math.cos(0.37 * ft)
			v = 0.01 * math.sin(1.7 * ft + 0.3) + 0.002 * c * c * c + 0.0004
		}
		s := ml_fin.sharpe_stats(r)
		psr := ml_fin.probabilistic_sharpe_ratio_returns(r)
		err := max(
			rel(s.sharpe, 0.057774903479593222),
			rel(s.skewness, -0.0019754972276219626),
			rel(s.kurtosis, 1.5768809610586714),
			rel(psr, 0.9015110286465492),
		)
		expect(
			&passed,
			&total,
			err < 1e-9 && s.n == 500,
			fmt.tprintf(
				"sharpe_stats: SR %.5f, skewness %.5f, kurtosis %.4f, PSR %.4f; max rel err %.1e",
				s.sharpe,
				s.skewness,
				s.kurtosis,
				psr,
				err,
			),
		)
	}
	// D. DSR of the best of 7 trials vs numpy / scipy
	M := _bs_lcg_matrix(301, 7)
	defer l.matrix_free(&M)
	{
		d := ml_fin.deflated_sharpe_trials(&M)
		err := max(
			rel(d.sharpe_variance, 0.0056520038780483841),
			rel(d.expected_max, 0.10425744094552096),
			rel(d.psr, 0.99962472493817178),
			rel(d.dsr, 0.94384989266857844),
		)
		expect(
			&passed,
			&total,
			d.best == 2 && err < 1e-12,
			fmt.tprintf(
				"deflated_sharpe_trials: best trial %d, SR %.4f, PSR %.4f, SR0 %.4f, DSR %.4f; max rel err %.1e",
				d.best,
				d.stats.sharpe,
				d.psr,
				d.expected_max,
				d.dsr,
				err,
			),
		)
	}
	// E. CSCV on the same 301 x 7 matrix, 8 blocks (uneven: 38 38 38 38 38 37 37 37 rows),
	// vs the itertools / array_split / rankdata reference; a custom metric (the same Sharpe
	// ratio, computed from the rows instead of block sums) gives the same result
	{
		p := ml_fin.backtest_overfitting(&M, 8)
		defer ml_fin.pbo_result_free(&p)
		q := ml_fin.backtest_overfitting(&M, 8, _bs_sharpe_metric)
		defer ml_fin.pbo_result_free(&q)
		sum_abs, sum_is, sum_oos := 0.0, 0.0, 0.0
		for c in 0 ..< len(p.logits) {
			sum_abs += abs(p.logits[c])
			sum_is += p.is_metric[c]
			sum_oos += p.oos_metric[c]
		}
		err := max(
			abs(p.logits[0] - 0),
			abs(p.logits[37] - 1.9459101490553132),
			abs(p.logits[69] - 0),
			rel(sum_abs, 83.029698781936901),
			rel(sum_is, 14.703117829990004),
			rel(sum_oos, 10.760895783209891),
			rel(p.slope, -0.5447806016672303),
			rel(p.intercept, 0.26815527372880033),
		)
		same := p.pbo == q.pbo
		for c in 0 ..< len(p.logits) {
			same &&=
				p.logits[c] == q.logits[c] &&
				abs(p.is_metric[c] - q.is_metric[c]) < 1e-12 &&
				abs(p.oos_metric[c] - q.oos_metric[c]) < 1e-12
		}
		expect(
			&passed,
			&total,
			len(p.logits) == 70 &&
			abs(p.pbo - 11.0 / 70) < 1e-15 &&
			p.prob_oos_loss == 0 &&
			err < 1e-10 &&
			same,
			fmt.tprintf(
				"CSCV: %d combinations, PBO %.4f (11/70), P[OOS loss] %.2f, degradation slope %.4f, intercept %.4f; max err %.1e; custom metric identical %v",
				len(p.logits),
				p.pbo,
				p.prob_oos_loss,
				p.slope,
				p.intercept,
				err,
				same,
			),
		)
	}
	fmt.printf("Backtest statistics reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
