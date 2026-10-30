package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import ml_finance "../wotan/ml_finance"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

// Clean-up checks:
// - pipelines with OLS / Ridge / ElasticNet keep their inference (stderr, t, p,
//   CI, vcov, σ², R², F, df, rank, cov_type) through save → load (format 5)
// - TimeGAN trains on exactly one sequence (int_range(0, 0) used to assert and the
//   last sequence was never drawn) and returns 0 without a complete sequence
step6_cleanup_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Clean-ups: OLS inference serialization, TimeGAN sampling ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	bits_equal :: proc(a, b: []f64) -> bool {
		if len(a) != len(b) do return false
		for v, i in a {if transmute(u64)v != transmute(u64)b[i] do return false}
		return true
	}
	same_f :: proc(a, b: f64) -> bool {return transmute(u64)a == transmute(u64)b}

	X := l.matrix_new(f64, 30, 2)
	y := make([]f64, 30)
	for i in 0 ..< 30 {
		t := f64(i)
		X.data[i * 2] = math.sin(0.7 * t) * 3 + t * 0.1
		X.data[i * 2 + 1] = math.cos(0.3 * t) + 0.05 * t * t
		y[i] = 1.5 + 0.8 * X.data[i * 2] - 0.4 * X.data[i * 2 + 1] + 0.3 * math.sin(2.1 * t)
	}
	path := "step6_cleanup_test_model.wotan"
	defer os.remove(path)
	check :: proc(
		pipe: ^ml.Pipeline,
		X: ^l.Matrix(f64),
		y: []f64,
		path: string,
		get: proc(p: ^ml.Pipeline) -> ^ml.OLSResult,
	) -> bool {
		ml.pipeline_fit(pipe, X, y)
		if !ml.pipeline_save(pipe, path) do return false
		ld, ok := ml.pipeline_load(path)
		defer ml.pipeline_free(&ld)
		if !ok do return false
		a, b := get(pipe), get(&ld)
		return(
			bits_equal(a.beta, b.beta) &&
			bits_equal(a.stderr, b.stderr) &&
			bits_equal(a.tvalues, b.tvalues) &&
			bits_equal(a.pvalues, b.pvalues) &&
			bits_equal(a.ci_low, b.ci_low) &&
			bits_equal(a.ci_high, b.ci_high) &&
			bits_equal(a.vcov.data, b.vcov.data) &&
			same_f(a.sigma2, b.sigma2) &&
			same_f(a.r2, b.r2) &&
			same_f(a.r2_adj, b.r2_adj) &&
			same_f(a.fstat, b.fstat) &&
			same_f(a.f_pvalue, b.f_pvalue) &&
			a.df_resid == b.df_resid &&
			a.rank == b.rank &&
			a.cov_type == b.cov_type &&
			len(a.stderr) == len(a.beta) \
		)
	}
	{
		p := ml.pipeline_new()
		ml.pipeline_add_standard_scaler(&p)
		ml.pipeline_set_ols(&p)
		expect(
			&passed,
			&total,
			check(&p, &X, y, path, proc(p: ^ml.Pipeline) -> ^ml.OLSResult {return &p.model.ols}),
			"OLS pipeline: stderr / t / p / CI / vcov / σ² / R² / F / df survive save → load",
		)
		ml.pipeline_free(&p)
	}
	{
		p := ml.pipeline_new()
		ml.pipeline_set_ridge(&p, {lambda = 2})
		expect(
			&passed,
			&total,
			check(&p, &X, y, path, proc(p: ^ml.Pipeline) -> ^ml.OLSResult {return &p.model.ridge}),
			"Ridge pipeline: inference survives save → load",
		)
		ml.pipeline_free(&p)
	}
	{
		p := ml.pipeline_new()
		ml.pipeline_add_robust_scaler(&p)
		ml.pipeline_set_elastic_net(&p, {lambda = 1, l1_ratio = 0.5})
		expect(
			&passed,
			&total,
			check(
				&p,
				&X,
				y,
				path,
				proc(p: ^ml.Pipeline) -> ^ml.OLSResult {return &p.model.elastic_net},
			),
			"ElasticNet pipeline: inference survives save → load",
		)
		ml.pipeline_free(&p)
	}

	// --- TimeGAN: one sequence ---
	{
		seq_len, feat := 4, 2
		one := []f64{0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.0, 0.15} // exactly one sequence
		tg := ml_finance.new_timegan(seq_len, feat, 3, 4, 1e-3, allocator)
		defer ml_finance.timegan_free(tg)
		ml_finance.timegan_init_weights(tg)
		le := ml_finance.train_embedder_recovery(tg, one, 2, allocator)
		ls := ml_finance.train_supervisor(tg, one, 2, allocator)
		ld := ml_finance.train_discriminator(tg, one, 2, allocator)
		lg := ml_finance.train_generator(tg, one, 2, allocator)
		finite :: proc(v: f64) -> bool {return !math.is_nan(v) && !math.is_inf(v)}
		expect(
			&passed,
			&total,
			finite(le) && finite(ls) && finite(ld) && finite(lg) && ls > 0 && ld > 0,
			fmt.tprintf(
				"TimeGAN trains on a single sequence (losses %.3g %.3g %.3g %.3g)",
				le,
				ls,
				ld,
				lg,
			),
		)
		short := one[:5] // no complete sequence
		z :=
			ml_finance.train_supervisor(tg, short, 2, allocator) +
			ml_finance.train_discriminator(tg, short, 2, allocator) +
			ml_finance.train_generator(tg, short, 2, allocator)
		expect(
			&passed,
			&total,
			z == 0,
			"TimeGAN training steps return 0 without a complete sequence",
		)
	}

	l.matrix_free(&X)
	delete(y)
	fmt.printf("Clean-up summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
