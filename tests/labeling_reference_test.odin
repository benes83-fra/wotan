package tests

import ml "../wotan/analytics/ML"
import ml_fin "../wotan/ml_finance"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"

// Labels, sample weights and fractional differencing against López de Prado's snippets
// run in pandas (2.4 CUSUM, 3.1 daily volatility = pct_change().ewm(span).std(),
// 3.2 - 3.5 triple barrier and meta-labels, 4.1 / 4.2 concurrency and uniqueness, 4.10
// return attribution (over the bars the label is alive, see return_attribution_weights),
// 4.11 time decay, 4.5 sequential-bootstrap probabilities, 5.3 FFD weights and series)
// and statsmodels adfuller for the minimum d, on a 400-bar LCG price path; bet sizes vs
// scipy's normal CDF.

@(private = "file")
_LAB_EVENTS := []int {
	15,
	20,
	25,
	33,
	43,
	51,
	70,
	90,
	97,
	107,
	110,
	116,
	132,
	138,
	149,
	166,
	170,
	177,
	191,
	199,
	209,
	215,
	223,
	232,
	244,
	261,
	267,
	272,
	282,
	285,
	300,
	327,
	331,
	353,
	360,
	372,
	385,
	391,
	395,
}
@(private = "file")
_LAB_ENDS := []int {
	16,
	25,
	28,
	38,
	44,
	61,
	73,
	93,
	105,
	108,
	112,
	124,
	134,
	141,
	150,
	168,
	171,
	182,
	196,
	202,
	215,
	216,
	225,
	233,
	246,
	265,
	268,
	274,
	283,
	292,
	301,
	331,
	336,
	356,
	370,
	374,
	391,
	394,
	398,
}
@(private = "file")
_LAB_LABELS := []int {
	1,
	-1,
	1,
	1,
	1,
	-1,
	1,
	-1,
	-1,
	1,
	1,
	-1,
	1,
	1,
	1,
	-1,
	1,
	1,
	-1,
	1,
	1,
	1,
	1,
	1,
	1,
	-1,
	1,
	1,
	1,
	-1,
	1,
	1,
	1,
	1,
	-1,
	1,
	1,
	-1,
	1,
}
@(private = "file")
_LAB_BARRIERS := []int {
	0,
	1,
	0,
	0,
	0,
	1,
	0,
	1,
	1,
	0,
	0,
	1,
	0,
	0,
	0,
	1,
	0,
	0,
	1,
	0,
	0,
	0,
	0,
	0,
	0,
	1,
	0,
	0,
	0,
	1,
	0,
	0,
	0,
	0,
	2,
	0,
	0,
	1,
	0,
}
@(private = "file")
_LAB_META_ENDS := []int {
	16,
	24,
	28,
	38,
	44,
	56,
	73,
	93,
	98,
	108,
	112,
	120,
	134,
	141,
	150,
	167,
	171,
	178,
	193,
	202,
	210,
	216,
	225,
	233,
	246,
	262,
	268,
	274,
	283,
	286,
	301,
	328,
	336,
	356,
	362,
	374,
	387,
	393,
	398,
}
@(private = "file")
_LAB_META_LABELS := []int {
	1,
	1,
	0,
	0,
	1,
	0,
	0,
	1,
	1,
	0,
	1,
	1,
	1,
	1,
	1,
	0,
	0,
	0,
	1,
	1,
	0,
	1,
	0,
	0,
	0,
	0,
	0,
	1,
	1,
	0,
	0,
	0,
	1,
	1,
	0,
	0,
	1,
	0,
	0,
}
@(private = "file")
_LAB_U := []f64 {
	1,
	0.91666666666666663,
	0.875,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	0.9285714285714286,
	0.75,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	0.90000000000000002,
	0.91666666666666663,
	1,
	1,
	1,
	0.9285714285714286,
	0.875,
	1,
}
@(private = "file")
_LAB_WR := []f64 {
	0.80178928513414893,
	1.348073589322941,
	1.0254946900731936,
	0.81975104989997183,
	0.93626270901492592,
	1.4623368458873933,
	0.98981416572701109,
	1.2621610381232258,
	1.4319600569604432,
	0.87477087933377984,
	1.0291523842831591,
	1.4111857584253031,
	0.80773217254686025,
	0.57183271692388127,
	0.8733936102452774,
	1.6571138618903907,
	0.6943996496252739,
	0.83550446370018405,
	1.7162283053827398,
	0.79608282889360593,
	0.71193473725965084,
	0.73324909308248121,
	0.711954491563902,
	0.83380990503770469,
	1.118673601715328,
	1.3040557653408924,
	0.61974045622632046,
	0.69437104822813411,
	0.80314476143109947,
	1.0994479462719466,
	0.66020041017873099,
	0.72836017564963762,
	1.0837674809381388,
	1.1563141010243556,
	0.89675019331161265,
	0.67928530459081393,
	1.033395803794553,
	1.880984860986926,
	0.90551980197405924,
}
@(private = "file")
_LAB_DECAY05 := []f64 {
	0.51312664083010373,
	0.52515939492436559,
	0.53664520565070628,
	0.54977184648081012,
	0.56289848731091385,
	0.57602512814101758,
	0.58915176897112143,
	0.60227840980122516,
	0.61540505063132889,
	0.62853169146143273,
	0.64165833229153646,
	0.65478497312164019,
	0.66791161395174403,
	0.68103825478184776,
	0.69416489561195149,
	0.70729153644205522,
	0.72041817727215895,
	0.7335448181022628,
	0.74667145893236653,
	0.75979809976247026,
	0.77198712339042375,
	0.78183210401300163,
	0.79495874484310536,
	0.80808538567320909,
	0.82121202650331293,
	0.83433866733341666,
	0.84746530816352039,
	0.86059194899362423,
	0.87371858982372785,
	0.8868452306538317,
	0.89997187148393554,
	0.91178584823102882,
	0.92381860232529056,
	0.9369452431553944,
	0.95007188398549813,
	0.96319852481560186,
	0.97538754844355546,
	0.98687335916989616,
	1,
}
@(private = "file")
_LAB_DECAYM := []f64 {
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0.009417843897153888,
	0.044422219444097244,
	0.0794265949910406,
	0.11443097053798401,
	0.14943534608492737,
	0.18443972163187072,
	0.21944409717881408,
	0.25444847272575744,
	0.28945284827270079,
	0.32445722381964415,
	0.3594615993665875,
	0.3919656623744634,
	0.41821894403467097,
	0.45322331958161433,
	0.48822769512855768,
	0.52323207067550104,
	0.5582364462224445,
	0.59324082176938786,
	0.62824519731633122,
	0.66324957286327457,
	0.69825394841021793,
	0.73325832395716128,
	0.76476226194941033,
	0.79684960620077505,
	0.83185398174771841,
	0.86685835729466176,
	0.90186273284160512,
	0.93436679584948124,
	0.96499562445305664,
	1,
}

@(private = "file")
_lab_max_err :: proc(a, b: []f64) -> f64 {
	if len(a) != len(b) {return math.inf_f64(1)}
	e := 0.0
	for v, i in a {e = max(e, abs(v - b[i]))}
	return e
}

labeling_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Labeling Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	rel :: proc(a, b: f64) -> f64 {return abs(a - b) / max(abs(b), 1e-300)}

	// the LCG price path
	N :: 400
	close := make([]f64, N)
	logp := make([]f64, N)
	{
		x: u64 = 2024
		logp[0] = math.ln(100.0)
		for i in 1 ..< N {
			x = 6364136223846793005 * x + 1442695040888963407
			u := f64(x >> 11) * 1.1102230246251565e-16
			logp[i] = logp[i - 1] + 0.02 * (u - 0.5) + 0.0002
		}
		for v, i in logp {close[i] = math.exp(v)}
	}

	// A. CUSUM events and daily volatility
	events := ml_fin.cusum_filter(logp, 0.02)
	vol := ml_fin.daily_volatility(close, 20)
	{
		nan_ok := math.is_nan(vol[0]) && math.is_nan(vol[1]) && !math.is_nan(vol[2])
		err := max(
			rel(vol[5], 0.0060261215283215382),
			rel(vol[200], 0.0068023841044135896),
			rel(vol[399], 0.0062137391744766504),
		)
		expect(
			&passed,
			&total,
			slice.equal(events, _LAB_EVENTS) && nan_ok && err < 1e-12,
			fmt.tprintf(
				"CUSUM filter (h 0.02 on log prices): %d events as snippet 2.4; daily volatility = pandas pct_change().ewm(span=20).std() (rel err %.1e, NaN on bars 0 and 1)",
				len(events),
				err,
			),
		)
	}

	// B. triple barrier, pt 1, sl 2, 10 bars
	labels := ml_fin.triple_barrier_labels(
		close,
		events,
		vol,
		{profit_taking = 1, stop_loss = 2, max_holding = 10},
	)
	{
		ok := len(labels) == len(_LAB_ENDS)
		ret_sum := 0.0
		for lab, i in labels {
			if !ok {break}
			ok &&=
				lab.start == _LAB_EVENTS[i] &&
				lab.end == _LAB_ENDS[i] &&
				lab.label == _LAB_LABELS[i] &&
				int(lab.barrier) == _LAB_BARRIERS[i]
			ret_sum += lab.ret
		}
		counts: [ml_fin.Barrier]int
		for lab in labels {counts[lab.barrier] += 1}
		expect(
			&passed,
			&total,
			ok && rel(ret_sum, 0.091450538490653988) < 1e-12,
			fmt.tprintf(
				"triple barrier (pt 1, sl 2 x vol, 10 bars): %d labels, ends, labels and barriers as snippets 3.2 - 3.5 (profit taking %d, stop loss %d, vertical %d); returns rel err %.1e",
				len(labels),
				counts[.Profit_Taking],
				counts[.Stop_Loss],
				counts[.Vertical],
				rel(ret_sum, 0.091450538490653988),
			),
		)
	}
	// C. meta-labels: side from 5-bar momentum, symmetric barriers
	{
		side := make([]f64, len(events))
		for t, e in events {side[e] = 1 if close[t] > close[max(t - 5, 0)] else -1}
		meta := ml_fin.triple_barrier_labels(
			close,
			events,
			vol,
			{profit_taking = 1, stop_loss = 1, max_holding = 10},
			side,
		)
		ok := len(meta) == len(_LAB_META_ENDS)
		for lab, i in meta {
			if !ok {break}
			ok &&=
				lab.end == _LAB_META_ENDS[i] &&
				lab.label == _LAB_META_LABELS[i] &&
				lab.side == side[i]
		}
		expect(
			&passed,
			&total,
			ok,
			"meta-labels (side from 5-bar momentum): ends and 0 / 1 labels as snippet 3.7",
		)
	}
	// D. concurrency, uniqueness, return attribution, time decay
	conc := ml_fin.label_concurrency(N, labels)
	u := ml_fin.average_uniqueness(labels, conc)
	{
		csum, cmax := 0, 0
		for c in conc {csum += c; cmax = max(cmax, c)}
		wr := ml_fin.return_attribution_weights(close, labels, conc)
		d05 := make([]f64, len(u))
		dm := make([]f64, len(u))
		for i in 0 ..< len(u) {d05[i], dm[i] = 1, 1}
		ml_fin.time_decay_weights(d05, u, 0.5)
		ml_fin.time_decay_weights(dm, u, -0.25)
		e1, e2, e3, e4 :=
			_lab_max_err(u, _LAB_U),
			_lab_max_err(wr, _LAB_WR),
			_lab_max_err(d05, _LAB_DECAY05),
			_lab_max_err(dm, _LAB_DECAYM)
		expect(
			&passed,
			&total,
			csum == 175 && cmax == 2 && e1 < 1e-15 && e2 < 1e-12 && e3 < 1e-14 && e4 < 1e-14,
			fmt.tprintf(
				"weights: concurrency (sum %d, max %d), uniqueness err %.1e, return attribution err %.1e, time decay c 0.5 / -0.25 err %.1e / %.1e",
				csum,
				cmax,
				e1,
				e2,
				e3,
				e4,
			),
		)
	}
	// E. sequential bootstrap probabilities after drawing labels 0, 3, 3
	{
		count := make([]int, N)
		for i in ([]int{0, 3, 3}) {
			for t in labels[i].start ..= labels[i].end {count[t] += 1}
		}
		p := ml_fin.sequential_bootstrap_probabilities(labels, count)
		want_hi := 0.026431718061674013
		err := max(
			abs(p[0] - 0.013215859030837006),
			abs(p[3] - 0.0088105726872246704),
			abs(p[1] - want_hi),
			abs(p[38] - want_hi),
		)
		sum := 0.0
		for v in p {sum += v}
		expect(
			&passed,
			&total,
			err < 1e-16 && abs(sum - 1) < 1e-14,
			fmt.tprintf(
				"sequential bootstrap probabilities (snippet 4.5) after drawing labels 0, 3, 3: a label drawn twice 1/3 as likely, drawn once 1/2; err %.1e",
				err,
			),
		)
	}
	// F. fractional differencing
	{
		wts := ml_fin.frac_diff_weights(0.4, 1e-4)
		y := ml_fin.frac_diff_ffd(logp, 0.4, 1e-4)
		K := len(wts) - 1
		s := 0.0
		for v in y[K:] {s += v}
		nan_ok := math.is_nan(y[K - 1]) && !math.is_nan(y[K])
		err := max(
			rel(wts[1], -0.4),
			rel(wts[5], -0.029952000000000006),
			rel(wts[K], -0.00010031216133609022),
			rel(y[K], 0.34122893164627488),
			rel(y[N - 1], 0.33089532200338656),
			rel(s, 40.273123405002714),
		)
		md := ml_fin.frac_diff_min_d(logp, 0.05, 1, 1e-4)
		err_d := max(
			rel(md.adf_stat, -3.3182920027944309),
			rel(md.p_value, 0.014080360335222121),
			rel(md.correlation, 0.46546049296208075),
		)
		expect(
			&passed,
			&total,
			len(wts) == 282 &&
			nan_ok &&
			err < 1e-10 &&
			md.found &&
			abs(md.d - 0.40) < 1e-12 &&
			err_d < 1e-8,
			fmt.tprintf(
				"FFD (d 0.4, threshold 1e-4): %d weights, series as snippet 5.3 (rel err %.1e); smallest d passing ADF %.2f (stat %.3f, p %.4f, corr %.3f) as statsmodels adfuller (rel err %.1e)",
				len(wts),
				err,
				md.d,
				md.adf_stat,
				md.p_value,
				md.correlation,
				err_d,
			),
		)
	}
	// G. bet sizes; label ends in sample units for the purged splitter
	{
		b := [3]f64 {
			ml_fin.bet_size_from_probability(0.55),
			ml_fin.bet_size_from_probability(0.8, -1),
			ml_fin.bet_size_from_probability(0.3),
		}
		err := max(
			abs(b[0] - 0.080055619182992643),
			abs(b[1] + 0.54674529524626392),
			abs(b[2] + 0.33747941645994262),
		)
		// sample ends: brute force over the bar spans
		ends := ml_fin.triple_barrier_sample_ends(labels)
		ok := true
		for lab, i in labels {
			last := i
			for j in i + 1 ..< len(labels) {if labels[j].start <= lab.end {last = j}}
			ok &&= ends[i] == last
		}
		// the purged splitter on sample ends drops exactly the labels whose bar spans meet a test label's
		folds := ml.cv_make_folds(
			ml.CVSplitter{kind = .Purged_KFold, n_splits = 5, label_end = ends},
			len(labels),
		)
		defer ml.cv_folds_free(folds)
		for f in folds {
			in_train := make([]bool, len(labels), context.temp_allocator)
			for j in f.train {in_train[j] = true}
			for j in 0 ..< len(labels) {
				if slice.contains(f.test, j) {continue}
				overlap := false
				for i in f.test {overlap ||= labels[j].start <= labels[i].end && labels[i].start <= labels[j].end}
				ok &&= in_train[j] == !overlap
			}
		}
		expect(
			&passed,
			&total,
			err < 1e-15 && ok,
			fmt.tprintf(
				"bet sizes 2 Phi(z) - 1 vs scipy (err %.1e); triple_barrier_sample_ends + Purged_KFold purge exactly the labels whose bar spans meet a test label's %v",
				err,
				ok,
			),
		)
	}
	fmt.printf("Labeling reference checks: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
