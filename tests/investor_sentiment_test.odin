package tests

import bfin "../wotan/behavioral_finance"
import w "../wotan/core"
import importer "../wotan/importer"
import net "../wotan/net"
import zm "../wotan/zip_min"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"

// Wurgler's maintained monthly sentiment data (raw proxies, macro controls and the
// published SENT / SENT_ORTH) and Ken French's size portfolios
@(private = "file")
BW_URL :: "https://pages.stern.nyu.edu/~jwurgler/data/SENTIMENT.xlsx"
@(private = "file")
FF_SIZE_URL :: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_ME_CSV.zip"

// Download url to path; false when offline
@(private = "file")
_bw_download :: proc(url, path: string) -> bool {
	body, ok := net.http_get(url, context.temp_allocator)
	if !ok || len(body) < 1000 {return false}
	return os.write_entire_file(path, transmute([]u8)body) == nil
}

// Value-weighted monthly returns (%) of the smallest minus the largest size decile,
// keyed by yyyymm
@(private = "file")
_ff_small_minus_big :: proc(zip_path: string) -> map[int]f64 {
	out := make(map[int]f64)
	bytes, ok := zm.zip_read_file(zip_path, "Portfolios_Formed_on_ME.csv", context.temp_allocator)
	if !ok {return out}
	text := string(bytes)
	in_block := false
	for line in strings.split_lines_iterator(&text) {
		if !in_block {
			in_block = strings.contains(line, "Value Weight Returns -- Monthly")
			continue
		}
		fields := strings.split(line, ",", context.temp_allocator)
		if len(fields) < 20 {
			if len(out) > 0 {break}
			continue // the column header
		}
		ym, ok_ym := strconv.parse_int(strings.trim_space(fields[0]))
		if !ok_ym {continue}
		lo, _ := strconv.parse_f64(strings.trim_space(fields[10])) // Lo 10
		hi, _ := strconv.parse_f64(strings.trim_space(fields[19])) // Hi 10
		out[ym] = lo - hi
	}
	return out
}

investor_sentiment_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Baker-Wurgler Sentiment: Published Index ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	XLSX :: "bw_sentiment_test.xlsx"
	if !_bw_download(BW_URL, XLSX) {
		fmt.println("No sentiment data (offline?): skipped")
		return
	}
	defer os.remove(XLSX)
	df, found := importer.xlsx_load_sheet(XLSX, "DATA", allocator)
	defer w.destroy_dataframe(&df)
	if !found || df.rows < 600 {
		fmt.println("Unexpected sentiment workbook: skipped")
		return
	}
	n := df.rows
	col :: proc(df: ^w.DataFrame, name: string) -> []f64 {
		c := w.column(df, name)
		out := make([]f64, df.rows)
		for t in 0 ..< df.rows {
			v, null := w.column_at_numeric(c, t)
			out[t] = math.nan_f64() if null else v
		}
		return out
	}
	ym := col(&df, "yearmo")
	pub, pub_orth := col(&df, "SENT"), col(&df, "SENT_ORTH")
	data := bfin.BW_Sentiment_Data {
		cefd     = col(&df, "cefd"),
		nipo     = col(&df, "nipo"),
		ripo     = col(&df, "ripo"),
		pdnd     = col(&df, "pdnd"),
		s        = col(&df, "s"),
		indpro   = col(&df, "indpro"),
		consdur  = col(&df, "consdur"),
		consnon  = col(&df, "consnon"),
		consserv = col(&df, "consserv"),
		employ   = col(&df, "employ"),
		cpi      = col(&df, "cpi"),
		recess   = col(&df, "recess"),
	}
	defer {
		delete(ym); delete(pub); delete(pub_orth)
		for x in ([][]f64{data.cefd, data.nipo, data.ripo, data.pdnd, data.s, data.indpro, data.consdur, data.consnon, data.consserv, data.employ, data.cpi, data.recess}) {delete(x)}
	}
	row :: proc(ym: []f64, v: f64) -> int {
		for x, t in ym {if x == v {return t}}
		return -1
	}
	// the maintained index: July 1965 to the last month, positive in December 2000
	first, dec2000 := row(ym, 196507), row(ym, 200012)
	sent, sent_orth := bfin.baker_wurgler_sentiment(data, first, n - 1, dec2000)
	defer {bfin.sentiment_index_free(&sent); bfin.sentiment_index_free(&sent_orth)}

	// 1. both indexes against the published ones (stored by Stata in single
	// precision, hence ~1e-5), over every month either is published
	cmp :: proc(mine, published: []f64) -> (max_diff: f64, n, n_mismatch: int) {
		for t in 0 ..< len(mine) {
			a, b := mine[t], published[t]
			if math.is_nan(a) && math.is_nan(b) {continue}
			if math.is_nan(a) != math.is_nan(b) {n_mismatch += 1; continue}
			max_diff = max(max_diff, abs(a - b))
			n += 1
		}
		return
	}
	d1, n1, m1 := cmp(sent.index, pub)
	d2, n2, m2 := cmp(sent_orth.index, pub_orth)
	expect(
		&passed,
		&total,
		n1 > 600 && n2 > 600 && m1 == 0 && m2 == 0 && d1 < 1e-4 && d2 < 1e-4,
		fmt.tprintf(
			"published SENT reproduced over %d months (max diff %.1e) and SENT_ORTH over %d (%.1e); first component explains %.0f%% / %.0f%%, SENT = %.3f cefd + %.3f nipo + %.3f ripo(t-12) + %.3f pdnd(t-12) + %.3f s",
			n1,
			d1,
			n2,
			d2,
			100 * sent.explained,
			100 * sent_orth.explained,
			sent.coefficients[0],
			sent.coefficients[1],
			sent.coefficients[2],
			sent.coefficients[3],
			sent.coefficients[4],
		),
	)

	// 2. the 2006 first stage applied to the five proxies and their 12-month lags
	// picks the lags the maintained index uses (ripo and pdnd lagged, the rest current)
	{
		ta := context.temp_allocator
		rz := make([]f64, n, ta)
		for t in 0 ..< n {rz[t] = 0 if data.nipo[t] == 0 || math.is_nan(data.ripo[t]) else data.ripo[t]}
		cur := [5][]f64 {
			data.cefd,
			bfin.trailing_sum(data.nipo, 12, ta),
			bfin.trailing_weighted_mean(rz, data.nipo, 12, ta),
			data.pdnd,
			data.s,
		}
		lagged: [5][]f64
		for x, j in cur {lagged[j] = bfin.lag_series(x, 12, ta)}
		use := make([]bool, n, ta)
		for t in first ..< n {use[t] = true}
		pick := bfin.sentiment_select_lags(cur[:], lagged[:], use, ta)
		expect(
			&passed,
			&total,
			!pick[0] && !pick[1] && pick[2] && pick[3] && !pick[4],
			fmt.tprintf(
				"first stage (2006): lagged version chosen for cefd %v, nipo %v, ripo %v, pdnd %v, s %v",
				pick[0],
				pick[1],
				pick[2],
				pick[3],
				pick[4],
			),
		)
	}

	// 3. the cross-sectional prediction on Ken French's size deciles: small minus big
	// (value-weighted) earns less after high sentiment (previous month's SENT_ORTH)
	ZIP :: "ff_size_test.zip"
	if !_bw_download(FF_SIZE_URL, ZIP) {
		fmt.println("No size portfolios (offline?): size test skipped")
	} else {
		defer os.remove(ZIP)
		smb := _ff_small_minus_big(ZIP)
		defer delete(smb)
		r := make([dynamic]f64, context.temp_allocator)
		s := make([dynamic]f64, context.temp_allocator)
		for t in first + 1 ..< n {
			v, ok := smb[int(ym[t])]
			if ok {append(&r, v); append(&s, sent_orth.index[t - 1])}
		}
		c := bfin.sentiment_conditional_returns(r[:], s[:], 12)
		expect(
			&passed,
			&total,
			len(r) > 600 && c.slope < 0 && c.t_slope < -2 && c.difference < 0,
			fmt.tprintf(
				"size deciles, %d months: small minus big %.2f%%/month after high and %.2f%% after low SENT_ORTH (difference %.2f, NW t %.1f); predictive slope %.2f (NW t %.1f)",
				len(r),
				c.mean_high,
				c.mean_low,
				c.difference,
				c.t_difference,
				c.slope,
				c.t_slope,
			),
		)
	}

	fmt.printf("Investor sentiment reference: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

investor_sentiment_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Baker-Wurgler Sentiment: Construction and Prediction ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	ta := context.temp_allocator
	nan := math.nan_f64()

	// 1. the series helpers on hand-checked values
	{
		x := []f64{1, 2, 3, 4, 5}
		lg := bfin.lag_series(x, 2, ta)
		ts := bfin.trailing_sum([]f64{1, 2, nan, 4, 5, 6}, 2, ta)
		// weights 0 ignore their (missing) x; all-zero weights give NaN
		wm := bfin.trailing_weighted_mean(
			[]f64{10, nan, 30, 0, 0, 0},
			[]f64{1, 0, 3, 0, 0, 0},
			3,
			ta,
		)
		// trend of the first two logs: exp(ln 4 - (ln 1 + ln 4) / 2) - 1 = 1
		dv := bfin.log_ma_deviation([]f64{1, 4, 4, nan, 8}, 2, ta)
		ok := math.is_nan(lg[1]) && lg[2] == 1 && lg[4] == 3
		ok &&=
			math.is_nan(ts[0]) &&
			ts[1] == 3 &&
			math.is_nan(ts[2]) &&
			math.is_nan(ts[3]) &&
			ts[4] == 9 &&
			ts[5] == 11
		ok &&=
			math.is_nan(wm[1]) &&
			abs(wm[2] - 25) < 1e-15 &&
			wm[3] == 30 &&
			wm[4] == 30 &&
			math.is_nan(wm[5])
		ok &&=
			math.is_nan(dv[0]) &&
			abs(dv[1] - 3) < 1e-14 &&
			abs(dv[2] - 1) < 1e-14 &&
			math.is_nan(dv[3]) &&
			abs(dv[4] - 1) < 1e-14
		expect(
			&passed,
			&total,
			ok,
			"lag, trailing sum, trailing weighted mean and log moving-average deviation on hand-checked values (NaN rules included)",
		)
	}

	// A simulated economy: latent sentiment S (AR(1), 0.95, unit variance) and a
	// macro factor M (AR(1), 0.9) over 1200 months; five proxies b_j S + c_j M + noise,
	// two of which (the IPO-like ones) lead the others by 12 months
	T :: 1200
	LEAD :: 12
	b := [5]f64{0.8, -0.7, 0.6, -0.7, 0.5}
	cm := [5]f64{0.6, 0.5, -0.5, 0.4, 0.6}
	leads := [5]bool{false, false, true, true, false}
	S := make([]f64, T + LEAD, ta)
	M := make([]f64, T + LEAD, ta)
	S[0], M[0] = rand.float64_normal(0, 1), rand.float64_normal(0, 1)
	for t in 1 ..< T + LEAD {
		S[t] = 0.95 * S[t - 1] + math.sqrt(1 - 0.95 * 0.95) * rand.float64_normal(0, 1)
		M[t] = 0.9 * M[t - 1] + math.sqrt(1 - 0.81) * rand.float64_normal(0, 1)
	}
	obs: [5][]f64 // as observed at t
	aligned: [5][]f64 // proxies at the lag that matches S_t (leaders lagged)
	for j in 0 ..< 5 {
		obs[j] = make([]f64, T, ta)
		L := LEAD if leads[j] else 0
		for t in 0 ..< T {obs[j][t] = b[j] * S[t + L] + cm[j] * M[t + L] + rand.float64_normal(0, 0.6)}
		aligned[j] = bfin.lag_series(obs[j], LEAD, ta) if leads[j] else obs[j]
	}
	corr :: proc(a, b: []f64) -> f64 {
		n, ma, mb := 0.0, 0.0, 0.0
		for t in 0 ..< len(a) {if !math.is_nan(a[t]) && !math.is_nan(b[t]) {n += 1; ma += a[t]; mb += b[t]}}
		ma /= n
		mb /= n
		sab, saa, sbb := 0.0, 0.0, 0.0
		for t in 0 ..< len(a) {
			if math.is_nan(a[t]) || math.is_nan(b[t]) {continue}
			sab += (a[t] - ma) * (b[t] - mb)
			saa += (a[t] - ma) * (a[t] - ma)
			sbb += (b[t] - mb) * (b[t] - mb)
		}
		return sab / math.sqrt(saa * sbb)
	}

	// 2. the first stage finds the two leaders when the proxies share only sentiment
	// (b_j S + noise). With the macro factor in them too it does so for only 45% of
	// seeds (Python sweep, 300 seeds): a second common factor confuses the selection.
	{
		clean, lagged: [5][]f64
		for j in 0 ..< 5 {
			clean[j] = make([]f64, T, ta)
			L := LEAD if leads[j] else 0
			for t in 0 ..< T {clean[j][t] = b[j] * S[t + L] + rand.float64_normal(0, 0.6)}
			lagged[j] = bfin.lag_series(clean[j], LEAD, ta)
		}
		pick := bfin.sentiment_select_lags(clean[:], lagged[:], nil, ta)
		ok := true
		for j in 0 ..< 5 {ok &&= pick[j] == leads[j]}
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"first stage picks the 12-month lag exactly for the two leading proxies: %v",
				pick,
			),
		)
	}

	// 3. the index and its macro-orthogonalized version (controls: M at the
	// proxy's own date): unit variance, eq. (2) coefficients reproduce it, residuals
	// orthogonal to the controls. Its correlation with S and the share explained
	// against the population values for b_j S + noise (0.9255, 0.6306); bounds 4 sd of
	// a 300-seed Python sweep of this design (sd 0.016 and 0.037; sample means 0.915,
	// 0.617, below the population values as the series are persistent). The raw index
	// (population correlation 0.886) is printed: it was below the orthogonalized one
	// in 97% of seeds, too few for a check.
	{
		raw := bfin.sentiment_index(aligned[:], nil, LEAD, ta)
		Mt := M[:T]
		e: [5][]f64
		for j in 0 ..< 5 {e[j] = bfin.orthogonalize(aligned[j], [][]f64{Mt}, nil, ta)}
		orth := bfin.sentiment_index(e[:], nil, LEAD, ta)
		St := S[:T]
		mean, ss, rebuilt := 0.0, 0.0, 0.0
		for t in LEAD ..< T {mean += orth.index[t] / f64(T - LEAD)}
		for t in LEAD ..< T {ss += (orth.index[t] - mean) * (orth.index[t] - mean) / f64(T - LEAD - 1)}
		// rebuild from standardized residuals with the reported coefficients
		zbar, zsd: [5]f64
		for j in 0 ..< 5 {
			for t in LEAD ..< T {zbar[j] += e[j][t] / f64(T - LEAD)}
			for t in LEAD ..< T {zsd[j] += (e[j][t] - zbar[j]) * (e[j][t] - zbar[j]) / f64(T - LEAD - 1)}
			zsd[j] = math.sqrt(zsd[j])
		}
		for t in LEAD ..< T {
			v := 0.0
			for j in 0 ..< 5 {v += orth.coefficients[j] * (e[j][t] - zbar[j]) / zsd[j]}
			rebuilt = max(rebuilt, abs(v - orth.index[t]))
		}
		max_ortho := 0.0
		for j in 0 ..< 5 {max_ortho = max(max_ortho, abs(corr(e[j], Mt)))}
		// the sign convention (positive at row LEAD) is not tied to S
		c_raw, c_orth := abs(corr(raw.index, St)), abs(corr(orth.index, St))
		ok := abs(mean) < 1e-12 && abs(ss - 1) < 1e-12 && rebuilt < 1e-12 && max_ortho < 1e-10
		ok &&= raw.n_obs == T - LEAD && orth.index[LEAD] > 0 && raw.index[LEAD] > 0
		ok &&= abs(c_orth - 0.9255) < 4 * 0.016 && abs(orth.explained - 0.6306) < 4 * 0.037
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"index over %d months: mean %.0e, variance 1 %+.0e, coefficients rebuild it to %.0e, residuals orthogonal to M (|corr| %.0e); correlation with latent sentiment %.3f orthogonalized (population 0.926; raw %.3f, 0.886), share explained %.3f (0.631)",
				raw.n_obs,
				mean,
				ss - 1,
				rebuilt,
				max_ortho,
				c_orth,
				c_raw,
				orth.explained,
			),
		)
	}

	// 4. conditional returns: r_t = 0.5 - 0.6 S_{t-1} + noise (sd 4). The slope is
	// recovered, and the high-minus-low difference is the difference of the means.
	{
		r := make([]f64, T, ta)
		s := make([]f64, T, ta)
		for t in 0 ..< T {
			s[t] = S[t]
			r[t] = 0.5 - 0.6 * S[t] + rand.float64_normal(0, 4)
		}
		s[3], r[7] = nan, nan // skipped rows
		c := bfin.sentiment_conditional_returns(r, s)
		cn := bfin.sentiment_conditional_returns(r, s, 6)
		se := c.slope / c.t_slope
		ok :=
			abs(c.slope + 0.6) < 4 * se &&
			c.t_slope < -2 &&
			abs(c.difference - (c.mean_high - c.mean_low)) < 1e-12
		ok &&=
			c.n_high + c.n_low == T - 2 &&
			cn.slope == c.slope &&
			cn.difference == c.difference &&
			cn.t_slope != c.t_slope
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"conditional returns: slope %.3f (se %.3f; true -0.6), %.2f after high vs %.2f after low sentiment (n %d / %d), NW(6) t %.1f vs OLS t %.1f",
				c.slope,
				se,
				c.mean_high,
				c.mean_low,
				c.n_high,
				c.n_low,
				cn.t_slope,
				c.t_slope,
			),
		)
	}

	fmt.printf("Investor sentiment: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
