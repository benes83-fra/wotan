package tests

import ml "../wotan/analytics/ML"
import l "../wotan/linalg"
import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// López de Prado's synthetic experiment (MLAM ch. 6): 40 features, 5 informative, 6
// noisy copies of each (30 redundant: x_i + N(0, 0.3^2)) and 5 noise; the label is
// sign(sum of the informative + N(0, 1)). A forest with max_features = 1; SFI with
// shallow trees (a deep forest on one weak feature overfits below a constant forecast).
//   - cluster_features puts every informative feature with its copies and keeps the
//     noise apart
//   - MDI and SFI rank every noise feature below every informative / redundant one
//   - substitution: MDA of single features splits each family's importance between its
//     7 members (each small), clustered MDA gives the whole family; the noise clusters
//     get under a tenth of a family's MDA, MDI below 1 / n_clusters and below every family

@(private = "file")
_FI_INF, _FI_COPIES, _FI_NOISE :: 5, 6, 5

// family of a feature: 0..4 informative (with copies), -1 noise
@(private = "file")
_fi_family :: proc(f: int) -> int {
	if f < _FI_INF {return f}
	if f < _FI_INF * (1 + _FI_COPIES) {return (f - _FI_INF) / _FI_COPIES}
	return -1
}

feature_importance_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	context.allocator = allocator
	fmt.println("\n=== Feature importance: informative, redundant and noise features ===")
	N :: 1000
	F :: _FI_INF * (1 + _FI_COPIES) + _FI_NOISE
	X := l.matrix_new(f64, N, F)
	defer l.matrix_free(&X)
	y := make([]f64, N)
	defer delete(y)
	for i in 0 ..< N {
		row := X.data[i * F:][:F]
		s := 0.0
		for j in 0 ..< _FI_INF {
			row[j] = rand.float64_normal(0, 1)
			s += row[j]
		}
		for c in 0 ..< _FI_INF * _FI_COPIES {
			row[_FI_INF + c] = row[c / _FI_COPIES] + rand.float64_normal(0, 0.3)
		}
		for k in 0 ..< _FI_NOISE {row[_FI_INF * (1 + _FI_COPIES) + k] = rand.float64_normal(0, 1)}
		y[i] = 1 if s + rand.float64_normal(0, 1) > 0 else 0
	}
	failures := 0
	check :: proc(failures: ^int, ok: bool, msg: string) {
		if !ok {failures^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", msg)
	}

	// clusters
	clusters := ml_fin.cluster_features(&X)
	defer ml_fin.feature_clusters_free(&clusters)
	pure := true
	n_family_clusters := 0
	for g in clusters.groups {
		fam := _fi_family(g[0])
		for f in g {pure &&= _fi_family(f) == fam}
		if fam >= 0 {n_family_clusters += 1}
	}
	check(
		&failures,
		pure && n_family_clusters == _FI_INF,
		fmt.tprintf(
			"cluster_features: %d clusters, each informative feature with its %d copies (%d family clusters), noise never mixed in",
			len(clusters.groups),
			_FI_COPIES,
			n_family_clusters,
		),
	)

	params := ml.RFParams {
		n_trees      = 200,
		max_features = 1,
		min_samples  = 20,
		max_depth    = 8,
		bootstrap    = true,
		criterion    = .Gini,
	}
	forest := ml.rf_fit(&X, y, params)
	defer ml.rf_free(&forest)
	model := ml_fin.rf_importance_model(&params)
	cv := ml.CVSplitter {
		kind     = .KFold,
		n_splits = 5,
	}
	mdi := ml_fin.mdi_importance(&forest, F)
	defer ml_fin.importance_result_free(&mdi)
	cmdi := ml_fin.mdi_importance(&forest, F, clusters.groups)
	defer ml_fin.importance_result_free(&cmdi)
	mda, oos := ml_fin.mda_importance(model, &X, y, cv)
	defer ml_fin.importance_result_free(&mda)
	cmda, _ := ml_fin.mda_importance(model, &X, y, cv, groups = clusters.groups)
	defer ml_fin.importance_result_free(&cmda)
	sfi_params := params
	sfi_params.n_trees = 50
	sfi_params.max_depth = 3
	sfi_params.min_samples = 100
	sfi := ml_fin.sfi_importance(ml_fin.rf_importance_model(&sfi_params), &X, y, cv)
	defer ml_fin.importance_result_free(&sfi)

	// per family: mean of the single-feature results
	fmt.printf(
		"   out-of-sample -log loss of the forest: %.3f (constant prediction: %.3f)\n",
		oos,
		-math.ln(2.0),
	)
	fmt.println("   family          MDI (each)  MDA (each)  SFI (each)   cluster: MDI   MDA")
	min_sig_mdi, max_noise_mdi := math.INF_F64, 0.0
	min_sig_sfi, max_noise_sfi := math.INF_F64, -math.INF_F64
	max_single_mda := 0.0
	for fam in -1 ..< _FI_INF {
		m1, m2, m3, n := 0.0, 0.0, 0.0, 0.0
		for f in 0 ..< F {
			if _fi_family(f) != fam {continue}
			m1 += mdi.mean[f]; m2 += mda.mean[f]; m3 += sfi.mean[f]; n += 1
			if fam < 0 {
				max_noise_mdi = max(max_noise_mdi, mdi.mean[f])
				max_noise_sfi = max(max_noise_sfi, sfi.mean[f])
			} else {
				min_sig_mdi = min(min_sig_mdi, mdi.mean[f])
				min_sig_sfi = min(min_sig_sfi, sfi.mean[f])
				max_single_mda = max(max_single_mda, mda.mean[f])
			}
		}
		cm, ca := 0.0, 0.0
		for g, k in clusters.groups {
			if _fi_family(g[0]) == fam {cm += cmdi.mean[k]; ca += cmda.mean[k]}
		}
		fmt.printf(
			"   %-14s  %9.4f  %10.4f  %10.4f   %12.3f  %5.3f\n",
			fmt.tprintf("informative %d", fam) if fam >= 0 else "noise (all)",
			m1 / n,
			m2 / n,
			m3 / n,
			cm,
			ca,
		)
	}
	check(
		&failures,
		max_noise_mdi < min_sig_mdi && max_noise_sfi < min_sig_sfi,
		fmt.tprintf(
			"MDI and SFI rank every noise feature below every informative / redundant one (MDI %.4f < %.4f, SFI %.3f < %.3f)",
			max_noise_mdi,
			min_sig_mdi,
			max_noise_sfi,
			min_sig_sfi,
		),
	)
	K := len(clusters.groups)
	fam_ok, noise_ok := true, true
	min_fam_cmda, min_fam_cmdi, max_noise_cmdi, max_noise_cmda := math.INF_F64, math.INF_F64, 0.0, -math.INF_F64
	for g, k in clusters.groups {
		if _fi_family(g[0]) >= 0 {
			fam_ok &&= cmda.mean[k] > 2 * cmda.std_err[k]
			min_fam_cmda = min(min_fam_cmda, cmda.mean[k])
			min_fam_cmdi = min(min_fam_cmdi, cmdi.mean[k])
		} else {
			noise_ok &&= cmdi.mean[k] < 1 / f64(K)
			max_noise_cmda = max(max_noise_cmda, cmda.mean[k])
			max_noise_cmdi = max(max_noise_cmdi, cmdi.mean[k])
		}
	}
	check(
		&failures,
		fam_ok && noise_ok && min_fam_cmdi > max_noise_cmdi && max_noise_cmda < 0.1 * min_fam_cmda,
		fmt.tprintf(
			"clustered: every family's MDA significant (> 2 standard errors) and MDI above every noise cluster's (%.3f > %.3f); noise clusters: MDI below 1/%d, MDA %.4f under a tenth of the smallest family's %.4f",
			min_fam_cmdi,
			max_noise_cmdi,
			K,
			max_noise_cmda,
			min_fam_cmda,
		),
	)
	check(
		&failures,
		min_fam_cmda > 2 * max_single_mda,
		fmt.tprintf(
			"substitution: permuting one feature of a family costs at most %.4f, permuting the whole family at least %.4f",
			max_single_mda,
			min_fam_cmda,
		),
	)
	fmt.printf("Feature importance: %d failure(s)\n", failures)
	record_failures(failures)
}
