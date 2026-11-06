package tests

import ml "../wotan/analytics/ML"
import "core:fmt"
import "core:mem"
import "core:slice"

// Purged K-fold and combinatorial purged cross-validation (ml.CVSplitter kinds
// Purged_KFold / Combinatorial_Purged) against a Python reference that purges pairwise
// (every training label against every test label, closed intervals) instead of by block
// interval, with numpy.array_split groups and itertools.combinations splits; the zero-
// horizon case against sklearn's KFold; the CPCV paths; and the GAT's former purging rule
// (now this splitter).

@(private = "file")
_PCV_SUMMARY_C := [15][3]int {
	{17, 7905, 10},
	{13, 6834, 10},
	{13, 5439, 10},
	{13, 3594, 10},
	{16, 4184, 10},
	{16, 6784, 10},
	{12, 4943, 10},
	{12, 3098, 10},
	{15, 3688, 10},
	{16, 4964, 10},
	{12, 2153, 10},
	{15, 2743, 10},
	{16, 2444, 10},
	{15, 1348, 10},
	{19, 2109, 10},
}

@(private = "file")
_PCV_SUMMARY_D := [10][3]int {
	{11, 4081, 10},
	{6, 2791, 10},
	{5, 1047, 10},
	{9, 1616, 10},
	{9, 2801, 10},
	{4, 586, 10},
	{7, 1011, 10},
	{9, 731, 10},
	{8, 155, 10},
	{13, 675, 10},
}

@(private = "file")
_PCV_PATHS := [30]int {
	0,
	0,
	1,
	2,
	3,
	4,
	1,
	5,
	5,
	6,
	7,
	8,
	2,
	6,
	9,
	9,
	10,
	11,
	3,
	7,
	10,
	12,
	12,
	13,
	4,
	8,
	11,
	13,
	14,
	14,
}

// (train size, sum of squared train indices, test size) per fold
@(private = "file")
_pcv_summary_ok :: proc(folds: []ml.CVFold, want: [][3]int) -> bool {
	if len(folds) != len(want) {return false}
	for f, i in folds {
		ss := 0
		for j in f.train {ss += j * j}
		if len(f.train) != want[i][0] ||
		   ss != want[i][1] ||
		   len(f.test) != want[i][2] {return false}
	}
	return true
}

purged_cv_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Purged / Combinatorial Purged CV Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// A. purged K-fold: 20 samples, 4 folds, labels 2 ahead, embargo 1
	{
		folds := ml.cv_make_folds(
			ml.CVSplitter{kind = .Purged_KFold, n_splits = 4, label_horizon = 2, embargo = 1},
			20,
		)
		defer ml.cv_folds_free(folds)
		want := [4][]int {
			{8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
			{0, 1, 2, 13, 14, 15, 16, 17, 18, 19},
			{0, 1, 2, 3, 4, 5, 6, 7, 18, 19},
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		}
		ok := len(folds) == 4
		for f, i in folds {ok &&= slice.equal(f.train, want[i]) && len(f.test) == 5 && f.test[0] == 5 * i}
		expect(
			&passed,
			&total,
			ok,
			"purged K-fold (20 samples, 4 folds, label horizon 2, embargo 1): training sets equal the pairwise reference (fold 0 trains on 8.. : 5, 6 purged, 7 embargoed)",
		)
	}
	// B. horizon 0, embargo 0: KFold without shuffling (sklearn KFold(5) on 23 samples)
	{
		p := ml.cv_make_folds(ml.CVSplitter{kind = .Purged_KFold, n_splits = 5}, 23)
		defer ml.cv_folds_free(p)
		k := ml.cv_make_folds(ml.CVSplitter{kind = .KFold, n_splits = 5}, 23)
		defer ml.cv_folds_free(k)
		ok := len(p) == len(k)
		for f, i in p {ok &&= slice.equal(f.train, k[i].train) && slice.equal(f.test, k[i].test)}
		expect(
			&passed,
			&total,
			ok,
			"without label overlap or embargo it is KFold (= sklearn KFold(5), 23 samples)",
		)
	}
	// C. CPCV: 30 samples, 6 groups, 2 test groups, horizon 1, embargo 2
	{
		folds := ml.cv_make_folds(
			ml.CVSplitter {
				kind = .Combinatorial_Purged,
				n_splits = 6,
				n_test_groups = 2,
				label_horizon = 1,
				embargo = 2,
			},
			30,
		)
		defer ml.cv_folds_free(folds)
		ok := _pcv_summary_ok(folds, _PCV_SUMMARY_C[:])
		// split 1 tests groups 0 and 2 (two runs), split 5 groups 1 and 2 (one run)
		ok &&= slice.equal(
			folds[1].train,
			[]int{8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
		)
		ok &&= slice.equal(
			folds[5].train,
			[]int{0, 1, 2, 3, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
		)
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"CPCV (30 samples, N 6, k 2, horizon 1, embargo 2): %d splits = C(6, 2); every training set equals the reference, incl. separate and adjacent test groups",
				len(folds),
			),
		)
	}
	// D. irregular label ends (event-driven labels, e_i = i + 7i mod 5), N 5, k 2, embargo 1
	{
		ends := make([]int, 25)
		for &e, i in ends {e = i + (i * 7) % 5}
		folds := ml.cv_make_folds(
			ml.CVSplitter {
				kind = .Combinatorial_Purged,
				n_splits = 5,
				n_test_groups = 2,
				label_end = ends,
				embargo = 1,
			},
			25,
		)
		defer ml.cv_folds_free(folds)
		expect(
			&passed,
			&total,
			_pcv_summary_ok(folds, _PCV_SUMMARY_D[:]),
			"CPCV with per-sample label ends (N 5, k 2, embargo 1): all 10 training sets equal the reference",
		)
	}
	// E. paths: N 6, k 2 -> 5 paths, each group once per path; stitching
	{
		paths, n_paths := ml.cv_cpcv_paths(6, 2)
		defer delete(paths)
		vals := make([][]f64, 15)
		for &v, s in vals {
			v = make([]f64, 30)
			for &x, i in v {x = f64(1000 * s + i)}
		}
		stitched := true
		for p in 0 ..< n_paths {
			path := ml.cv_cpcv_path(vals, paths, 6, p)
			defer delete(path)
			for x, i in path {stitched &&= x == f64(1000 * paths[p * 6 + i / 5] + i)}
		}
		expect(
			&passed,
			&total,
			n_paths == 5 && slice.equal(paths, _PCV_PATHS[:]) && stitched,
			"CPCV paths (N 6, k 2): 5 = C(5, 1) paths, the split of every group equals the reference; cv_cpcv_path takes each group from its split",
		)
	}
	// F. the GAT's former rule (train i < val_start - gap or i >= val_end + gap) is
	// Purged_KFold with label_horizon = gap
	{
		ok := true
		for c in ([][3]int{{30, 4, 5}, {101, 5, 12}, {64, 8, 3}, {17, 3, 0}}) {
			batch, n_splits, gap := c[0], c[1], c[2]
			folds := ml.cv_make_folds(
				ml.CVSplitter{kind = .Purged_KFold, n_splits = n_splits, label_horizon = gap},
				batch,
			)
			defer ml.cv_folds_free(folds)
			fs, rem := batch / n_splits, batch % n_splits
			for f, k in folds {
				vs := k * fs + min(k, rem)
				ve := vs + fs + (1 if k < rem else 0)
				old := make([dynamic]int)
				defer delete(old)
				for i in 0 ..< batch {
					if i < vs - gap || i >= ve + gap {append(&old, i)}
				}
				ok &&= slice.equal(f.train, old[:])
			}
		}
		expect(
			&passed,
			&total,
			ok,
			"the GAT's former purging rule equals Purged_KFold with label_horizon = window length (4 configurations)",
		)
	}
	fmt.printf("Purged CV reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
