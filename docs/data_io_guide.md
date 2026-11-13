# Data in and out: `importer`, `exporter` and `net`

Three packages move data into and out of Wotan:

- **`importer`** reads CSV, JSON, JSON Lines, HTML tables and Excel workbooks into `core` DataFrames.
- **`exporter`** writes DataFrames back to the same formats.
- **`net`** fetches data over HTTP. It also loads price histories from Yahoo Finance, as a DataFrame, as plain arrays for one ticker, or as a panel of several tickers matched by date.

This guide covers all three. It assumes you know the `DataFrame` from the [core guide](core_guide.md). Every example was compiled and run against the current code, and the output shown is the real output. The examples free everything they allocate: a run under a tracking allocator reports no leaks. The Yahoo examples ran on 2026-11-13, so their numbers will differ when you run them.

## Contents

1. [Setup](#1-setup)
2. [Overview](#2-overview)
3. [CSV](#3-csv)
4. [JSON and JSON Lines](#4-json-and-json-lines)
5. [HTML tables](#5-html-tables)
6. [Excel](#6-excel)
7. [Writing files](#7-writing-files)
8. [HTTP](#8-http)
9. [Yahoo Finance](#9-yahoo-finance)
10. [Fixed while writing this guide](#10-fixed-while-writing-this-guide)
11. [Tests](#11-tests)
12. [Quick reference](#12-quick-reference)

## 1. Setup

```odin
import ts  "wotan/analytics" // log_returns / simple_returns (section 9)
import w   "wotan/core"
import ex  "wotan/exporter"
import io  "wotan/importer"
import net "wotan/net"
import "core:fmt"
```

`net` uses libcurl through `vendor:curl`, which ships with Odin.

## 2. Overview

| Format | Read | Write |
|---|---|---|
| CSV | `io.csv_load(path)`, `io.csv_load_from_string(text)` | `ex.csv_save(&df, path)` |
| JSON (array of objects) | `io.json_load(path)` | `ex.json_write(&df, path, alloc)` |
| JSON Lines | `io.jsonl_load(path)` | `ex.jsonl_write(&df, path, alloc)` |
| HTML `<table>` | `io.html_load(path)`, `_table_index`, `_table_id`, `_all`, `io.html_load_from_string(text)` | `ex.html_write(&df, path, alloc)` |
| Excel `.xlsx` | `io.xlsx_load(path)`, `io.xlsx_load_sheet(path, name)`, `io.xlsx_sheet_names(path)` | `ex.xlsx_save(&df, path, alloc)` |
| URL | `net.http_get(url)`, `net.download_file(url, path)`, `net.read_csv_from_url`, `read_html_from_url`, `read_table_from_url` | |
| Yahoo Finance | `net.read_yahoo`, `net.read_yahoo_series`, `net.read_yahoo_panel` | |

**Types.** Every importer infers each column's type from its values:

- whole numbers become `Int`, and decimals become `Float` (a mix of the two becomes `Float`);
- `true`/`false` become `Bool`;
- ISO dates, times and datetimes become `Date`, `Time` and `Datetime`;
- anything else, or a mix that fits none of these, becomes `String`.

Empty fields and the tokens `NA`, `null` and `None` become nulls.

**Ownership.** A loader returns a DataFrame that you free with `w.destroy_dataframe`. The DataFrame owns its column names and string values; it doesn't point into the file buffer or the temp allocator.

**Failure.** File loaders panic when a file cannot be read or doesn't parse. The exception is `xlsx_load`, which returns an empty DataFrame. Everything that goes over the network returns `(value, ok)` instead, with `ok` false when the request fails or brings no data. Network failures are normal (rate limits, outages), so they are reported, not fatal.

## 3. CSV

`csv_load_from_string(text, allocator, types, sep, null_tokens)` and `csv_load(path, types, sep, null_tokens, allocator)` parse CSV with quoted fields: a separator, quote or line break inside quotes is part of the field. The null semantics are designed so that a save followed by a load is lossless:

- an unquoted field that is empty or a null token is null;
- in a String column, a quoted field is always the literal text, so `""` is the empty string and `"NA"` the text NA;
- in typed columns, quoted content is parsed like unquoted content.

```odin
text := `ticker;price;shares;note
AAPL;185.64;100;"buy; core"
MSFT;370.87;NA;
"NVDA";495.22;20;""
`
df := io.csv_load_from_string(text, sep = ';')
defer w.destroy_dataframe(&df)
w.dataframe_pretty_print(&df)
note := w.column(&df, "note")
_, n1 := w.column_at_string(note, 1)
s2, n2 := w.column_at_string(note, 2)
fmt.printf("note[1] null=%v, note[2] = %q null=%v\n", n1, s2, n2)

// explicit types: keep the share count as a Float
df2 := io.csv_load_from_string(text, sep = ';', types = {.String, .Float, .Float, .String})
defer w.destroy_dataframe(&df2)
fmt.println(w.column(&df2, "shares").type)
```

```
    ticker  price   shares  note
  0 AAPL    185.64  100     buy; core
  1 MSFT    370.87  NULL    NULL
  2 NVDA    495.22  20
note[1] null=true, note[2] = "" null=false
Float
```

The `NA` share count and MSFT's empty note are nulls, while NVDA's quoted `""` is an empty string. Rows with fewer fields than the header get nulls in the missing columns, and extra fields are ignored. A UTF-8 byte order mark and CRLF line ends are handled.

## 4. JSON and JSON Lines

`json_load(path, null_tokens, allocator)` reads a top-level array of objects. `jsonl_load(path, null_tokens, allocator)` reads one object per line and skips blank lines.

The columns are the union of all keys, in order of first appearance, and a record without a key gets a null there. JSON `null` becomes a null, and nested objects or arrays are kept as their JSON text. A JSON string in a String column is always the literal text, so `""` and `"NA"` stay strings.

```odin
df := io.json_load("demo_files/example_data.json", allocator = context.allocator)
defer w.destroy_dataframe(&df)
w.dataframe_pretty_print(&df)

dl := io.jsonl_load("demo_files/example_data2.jsonl", allocator = context.allocator)
defer w.destroy_dataframe(&dl)
w.dataframe_print(&dl)
```

```
    name     age
  0 Alice    30
  1 Bob      20
  2 Charlie  NULL
DataFrame:
  rows: 4
  columns: 3
    name (String)
    age (Int)
    active (Bool)
```

## 5. HTML tables

The HTML loaders find `<table>` elements, take the first row's `<th>`/`<td>` cells as the header, and read the rest as data. `<thead>` and `<tbody>` are allowed. Tags inside cells are stripped and entities decoded (`&amp;`, `&#233;`, ...). Rows with fewer cells than the header are padded with nulls.

| Procedure | Table |
|---|---|
| `html_load(path, alloc)` | the first one |
| `html_load_table_index(path, i, alloc)` | the i-th (from 0) |
| `html_load_table_id(path, id, alloc)` | the one with `id="..."` |
| `html_load_all(path, alloc)` | all, as `[]DataFrame` |
| `html_load_from_string(text, alloc)`, `html_load_all_from_string(text, alloc)` | the same for HTML in memory |

```odin
all := io.html_load_all("demo_files/test_multi.html", context.allocator)
defer {for &d in all {w.destroy_dataframe(&d)}; delete(all)}
fmt.println("tables:", len(all))

second := io.html_load_table_id("demo_files/test_multi.html", "second", context.allocator)
defer w.destroy_dataframe(&second)
w.dataframe_pretty_print(&second)

scores := io.html_load("demo_files/tag_table.html", context.allocator)
defer w.destroy_dataframe(&scores)
w.dataframe_pretty_print(&scores)
```

```
tables: 3
    X    Y
  0 foo  bar
    Name     Age   Score
  0 Alice    30    88.5
  1 Bob      20    91
  2 Charlie  NULL  77.25
  3 Dora     40    NULL
```

## 6. Excel

`xlsx_load(path, alloc)` reads the first worksheet of an `.xlsx` workbook, and `xlsx_load_sheet(path, name, alloc)` reads a worksheet by name and returns `(df, found)`. `xlsx_sheet_names(path, alloc)` lists the worksheets in workbook order; free each name and the slice.

The first row is the header. Cells are read with their Excel types: numbers, booleans, shared and inline strings, rich text, and errors (read as nulls). A number whose cell style is a date or time format becomes a `Date`, `Time` or `Datetime`. A custom format such as `#,##0 "USD"` is not mistaken for a date.

```odin
names := io.xlsx_sheet_names("demo_files/example.xlsx", context.allocator)
defer {for n in names {delete(n)}; delete(names)}
fmt.println("sheets:", names)

dt := io.xlsx_load("demo_files/datetime.xlsx", context.allocator)
defer w.destroy_dataframe(&dt)
w.dataframe_pretty_print(&dt)
```

```
sheets: ["Tabelle1"]
    Date        Time      DateTime
  0 2024-01-15  14:30:00  2024-01-15 14:30:00
```

## 7. Writing files

All writers produce text that the matching importer reads back to the same DataFrame. `io_roundtrip_test` checks this for every column type, with nulls, integers beyond 2^53, tiny and huge floats, and strings with separators, quotes, line breaks, outer spaces, Unicode, markup and the text `NA`.

| Writer | Output |
|---|---|
| `csv_save(&df, path, sep = ',', allocator) -> bool` | Header plus one line per row. Nulls are empty unquoted fields. A string is quoted when it would otherwise read back differently: empty, with outer spaces, a null token, or containing the separator, a quote or a line break. |
| `json_write(&df, path, alloc)` | One array of objects. Nulls and non-finite floats are `null`, and dates are ISO strings. |
| `jsonl_write(&df, path, alloc)` | One object per line, otherwise as `json_write`. |
| `html_write(&df, path, alloc)` | A `<table>` with `<thead>` and `<tbody>`; text is escaped, and nulls are empty cells. |
| `xlsx_save(&df, path, alloc) -> bool` | One worksheet, `Sheet1`. Numbers are exact; dates and times are serial numbers with a date style. |

Floats are written in their shortest exact form and always keep a `.` or an exponent, so a float column stays a float column.

```odin
ok := ex.csv_save(&df, "trades.csv") // the DataFrame from section 3
back := io.csv_load("trades.csv")
defer w.destroy_dataframe(&back)

ex.json_write(&dl, "people.json", context.allocator) // the JSON Lines frame from section 4
ex.jsonl_write(&dl, "people.jsonl", context.allocator)
ex.html_write(&second, "second.html", context.allocator) // the HTML table "second"

people := io.csv_load("demo_files/people_dates.csv")
defer w.destroy_dataframe(&people)
ex.xlsx_save(&people, "people.xlsx", context.allocator)
xl := io.xlsx_load("people.xlsx", context.allocator)
defer w.destroy_dataframe(&xl)
w.dataframe_print(&xl)
```

`trades.csv` (MSFT's null note is an empty field, NVDA's empty string is `""`):

```
ticker,price,shares,note
AAPL,185.64,100,buy; core
MSFT,370.87,,
NVDA,495.22,20,""
```

`people.json` and `people.jsonl`:

```
[{"name":"Alice","age":30,"active":true},{"name":"Bob","age":20,"active":false},{"name":"Charlie","age":null,"active":true},{"name":"Dora","age":40,"active":null}]
```

```
{"name":"Alice","age":30,"active":true}
{"name":"Bob","age":20,"active":false}
{"name":"Charlie","age":null,"active":true}
{"name":"Dora","age":40,"active":null}
```

`second.html`:

```
<meta charset="utf-8">
<table>
  <thead>
    <tr><th>X</th><th>Y</th></tr>
  </thead>
  <tbody>
    <tr><td>foo</td><td>bar</td></tr>
  </tbody>
</table>
```

The workbook read back has the CSV's types, including the `Datetime` column:

```
DataFrame:
  rows: 5
  columns: 5
    age (Int)
    name (String)
    salary (Float)
    active (Bool)
    birthday (Datetime)
```

## 8. HTTP

| Procedure | Returns | `ok` is false when |
|---|---|---|
| `http_get(url, alloc)` | `(body, ok)` | the request fails; any HTTP status counts as an answer |
| `download_file(url, path)` | `ok` | the request fails, the status is outside 200–299, or the file cannot be written |
| `read_csv_from_url(url, alloc)` | `(df, ok)` | the request fails or the status is outside 200–299 |
| `read_html_from_url(url, alloc)` | `(df, ok)` | as above; reads the first `<table>` |
| `read_table_from_url(url, alloc)` | `(df, ok)` | as above; HTML if the body contains `<table>`, CSV otherwise |
| `http_get_with_cookie(url, cookie, alloc)` | `(body, ok)` | as `http_get`, with a `Cookie` header |

All requests follow redirects and send a browser user agent. On failure the DataFrame is empty, so it can still be destroyed.

```odin
body, ok := net.http_get("https://example.com", context.allocator)
defer delete(body)
fmt.println(ok, len(body) > 0)

df, got := net.read_table_from_url("https://example.com/missing.csv")
defer w.destroy_dataframe(&df)
fmt.println("missing.csv:", got, df.rows)
```

```
true true
missing.csv: false 0
```

## 9. Yahoo Finance

Yahoo's chart API gives daily, weekly or monthly histories of stocks, ETFs and indices, with dividends and splits. It needs no key, but it rate-limits heavy use, so every Yahoo procedure returns `ok`. The interval is `.Daily`, `.Weekly` or `.Monthly`. The range is `.OneMonth`, `.ThreeMonths`, `.SixMonths`, `.OneYear`, `.TwoYears`, `.FiveYears`, `.TenYears`, `.YTD` or `.Max`.

There are three levels:

| Procedure | Gives | Use it for |
|---|---|---|
| `read_yahoo(symbol, interval, range, alloc)` | `(DataFrame, ok)`: Date, Open, High, Low, Close, AdjClose, Volume, Dividend, Split | inspecting the raw data, dividends and splits |
| `read_yahoo_series(symbol, interval, range, alloc)` | `(Yahoo_Series, ok)`: arrays `dates`, `open`, `high`, `low`, `close`, `adj_close`, `volume` | one ticker in numeric code |
| `read_yahoo_panel(tickers, interval, range, alloc)` | `(Yahoo_Panel, ok, failed)`: `dates`, `adj_close` and `dollar_volume`, row-major `[day * n_tickers + ticker]` | several tickers on the same days |

**Close or AdjClose.** `Close` is adjusted for splits only. `AdjClose` is also adjusted for dividends, so its returns are total returns. Use `AdjClose` for returns. The latest adjusted close equals the latest close, since adjustment rescales only the earlier days.

### 9.1 The DataFrame and Yahoo's missing days

`read_yahoo` keeps every row Yahoo sends. Some symbols have rows without any quote, and their cells are nulls. The VIX is the common case: Yahoo lists it on most US market holidays, when no index value exists.

```odin
df, ok := net.read_yahoo("^VIX", .Daily, .OneYear)
defer w.destroy_dataframe(&df)
if !ok {fmt.println("Yahoo is unreachable or rate-limiting"); return}
fmt.println(df.rows, "rows")
closes, dates := w.column(&df, "Close"), w.column(&df, "Date")
for i in 0 ..< df.rows {
	if _, null := w.column_at_float(closes, i); null {
		d, _ := w.column_at_date(dates, i)
		fmt.print(w.date_to_string(d, context.temp_allocator), "")
	}
}
fmt.println()
```

```
262 rows
2025-11-27 2025-12-25 2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-06-19 2026-07-03
```

Thanksgiving, Christmas, New Year, Martin Luther King Day, Presidents' Day, Good Friday, Juneteenth and Independence Day (observed). Two consequences follow:

- **Never pair two tickers by row index.** SPY has no rows on these days, so row *i* of SPY and row *i* of the VIX drift apart with every holiday. Over two years the gap grows to 21 rows: SPY's last day, 2026-11-13, sits next to the VIX of 2026-10-15. Use `read_yahoo_panel`, which matches by date.
- **Check the null flag.** A null cell reads as `0` from `column_at_float`, and `ln(x / 0)` is infinite.

`read_yahoo_series` and `read_yahoo_panel` drop these rows for you.

### 9.2 One ticker: `read_yahoo_series`

`read_yahoo_series` returns the history as plain `[]f64` arrays of equal length, oldest first. It drops the days with no close, no adjusted close or a price ≤ 0, so `close[i] > 0` holds. On a rare kept day where only open, high, low or volume is missing, that value is NaN. Free the series with `yahoo_series_free`.

`ts.log_returns(prices)` and `ts.simple_returns(prices)` turn prices into the `T - 1` returns from day *t* to *t + 1*.

```odin
s, ok := net.read_yahoo_series("AAPL", .Daily, .FiveYears)
defer net.yahoo_series_free(&s)
if !ok {fmt.println("Yahoo is unreachable or rate-limiting"); return}
n := len(s.dates)
fmt.printf("%s: %d days, %s .. %s\n", s.symbol, n,
	w.date_to_string(s.dates[0], context.temp_allocator),
	w.date_to_string(s.dates[n - 1], context.temp_allocator))
fmt.printf("first day: close %.2f, adjusted %.2f, volume %.0f\n", s.close[0], s.adj_close[0], s.volume[0])

r := ts.log_returns(s.adj_close) // n - 1 daily log returns
defer delete(r)
mean, m2 := 0.0, 0.0
for x in r {mean += x}
mean /= f64(len(r))
for x in r {m2 += (x - mean) * (x - mean)}
fmt.printf("%d log returns: annualised mean %.1f%%, volatility %.1f%%\n",
	len(r), 252 * mean * 100, math.sqrt(252 * m2 / f64(len(r) - 1)) * 100)
```

```
AAPL: 1255 days, 2021-11-15 .. 2026-11-13
first day: close 150.00, adjusted 146.41, volume 59222800
1254 log returns: annualised mean 18.0%, volatility 28.1%
```

The adjusted close of the first day is below the close by the dividends paid since.

### 9.3 Several tickers: `read_yahoo_panel`

`read_yahoo_panel` loads each ticker and keeps the dates on which all of them have a positive adjusted close. It returns:

- the adjusted closes and the dollar volumes (adjusted close × volume), row-major `[day * n_tickers + ticker]`;
- `ok` false and `failed` = the index of the ticker that could not be loaded, or −1 when the tickers share no date.

Free the panel with `yahoo_panel_free`. The returns helpers take the number of series as a second argument and keep the same layout.

```odin
tickers := []string{"SPY", "^VIX"}
p, ok, failed := net.read_yahoo_panel(tickers, .Daily, .TwoYears)
defer net.yahoo_panel_free(&p)
if !ok {fmt.println("could not load", tickers[failed] if failed >= 0 else "a common date"); return}

T, N := len(p.dates), p.n_tickers
fmt.printf("%d common days, last %s: SPY %.2f, VIX %.2f\n", T,
	w.date_to_string(p.dates[T - 1], context.temp_allocator),
	p.adj_close[(T - 1) * N + 0], p.adj_close[(T - 1) * N + 1])

r := ts.log_returns(p.adj_close, N) // [day][ticker], T - 1 rows
defer delete(r)
spy := make([]f64, T - 1)
vix := make([]f64, T - 1)
defer {delete(spy); delete(vix)}
for t in 0 ..< T - 1 {spy[t], vix[t] = r[t * N], r[t * N + 1]}
fmt.printf("correlation of daily log returns: %.3f\n", corr(spy, vix)) // Pearson correlation
```

```
502 common days, last 2026-11-13: SPY 751.81, VIX 20.43
correlation of daily log returns: -0.842
```

SPY and the VIX move strongly against each other on the same day. Paired by row index, as several tests did before this guide, the same two years give a correlation of −0.09, because most pairs then mix different days.

### 9.4 More Yahoo procedures

- `read_yahoo_json(symbol, alloc)` returns the daily history over the full range. Unlike `read_yahoo(symbol, .Daily, .Max)`, its request asks for no events, so the Dividend and Split columns are 0.
- `yahoo_json_to_dataframe(text, alloc)` parses a chart answer you fetched yourself.
- Option chains are in `finance`: `fin.fetch_yahoo_options` and `fin.fetch_yahoo_option_surface` (see the [risk guide, section 8](finance_risk_guide.md#8-market-data)).
- `yahoo_download_csv` uses Yahoo's old CSV download endpoint. Yahoo now answers it with HTTP 401 (*User is not logged in*), so it no longer works.

## 10. Fixed while writing this guide

- **CSV column names were in the temp allocator.** A DataFrame from `csv_load` or `csv_load_from_string` took its column names from the parser's temporary buffers. After `free_all(context.temp_allocator)` the names were garbage and `w.column(&df, "name")` panicked. The HTML, Excel and JSON importers cloned their names, but nothing freed the copies. Columns can now own their name (`w.column_own_name`), and `destroy_column` frees it. The importers use it, and so do the procedures that copy a column (`column_take` and everything built on it, joins, reindexing), so a copy no longer borrows its name from the source. A join also leaked the names it made for clashing columns (`x_right`). On the core and I/O tests, the leak count under a tracking allocator went from 266 to 0 (the last two were a view that `io_roundtrip_test` never freed).
- **Yahoo's adjusted close on the first day was 0.** Yahoo nests the adjusted closes as `"adjclose":[{"adjclose":[...]}]`, and the array extractor stopped at the first key. The first value it parsed was `{"adjclose":[146.41`, which failed and became 0. `read_yahoo_panel` hid this by dropping prices ≤ 0, but a test that took returns from AdjClose got an infinite first return. The extractor now skips keys whose value is not a flat array.
- **Only one dividend and one split were read.** The events parser cut the dividends object at the first `}`, which closes the first entry. On five years of SPY it found 1 of 20 dividends; the dividend and split columns now match Yahoo's data.
- **Missing Yahoo values were 0.** JSON `null` was parsed as 0, so the VIX had a level of 0 on holidays, with the null flag false. Missing values are now nulls in the DataFrame. Several tests paired SPY and the VIX by row index and are now matched by date (section 9.1).
- **The exporters leaked the text of every number and date cell**, and `xlsx_save` leaked a small offsets table. The writers now take the cell texts from a scratch arena that is reset after each row.
- `yahoo_json_to_dataframe` ignored its allocator, and `read_csv_from_url`, `read_html_from_url` and `read_table_from_url` panicked when a request failed. These three now return `(df, ok)`, and `csv_load` takes an allocator like the other loaders.

The DataFrames `yahoo_json_to_dataframe` builds were checked against an independent pandas parse of the same JSON text. Over five years of SPY, the VIX and AAPL and ten years of NVDA, the dates, the positions of the nulls and every value, including 40 dividends and 2 splits, agree to 1e-13.

## 11. Tests

| Test | What it checks |
|---|---|
| `io_roundtrip_test` | exact CSV, JSON, JSONL and XLSX round trips for all column types, nulls and edge cases; HTML up to blank strings; CSV corner cases (BOM, CRLF, short and long rows); Excel-style workbooks (rich text, styles, error cells) |
| `csv_export_test`, `json_basic_test`, `jsonl_basic_test`, `json_export_test` | CSV and JSON loading and writing |
| `html_basic_test`, `html_extended_test`, `html_tags_test`, `html_export_test` | HTML tables by index, id and all, tags inside cells, writing |
| `excel_basic_test`, `excel_date_test`, `excel_export_test`, `zip_smoke_test` | reading sheets and dates, writing workbooks, the ZIP container |
| `http_get_test`, `yahoo_finance_json_test`, `yahoo_finance_events_test` | live HTTP and Yahoo requests, including dividends and splits |

Most of the finance tests load their data with `read_yahoo`, `read_yahoo_series` or `read_yahoo_panel`. When Yahoo cannot be reached they count a failure and go on; they never fall back to made-up data.

## 12. Quick reference

| Procedure | Purpose |
|---|---|
| `io.csv_load(path, types, sep, null_tokens, alloc)` / `io.csv_load_from_string(text, alloc, types, sep, null_tokens)` | CSV file / text → DataFrame |
| `io.json_load(path, null_tokens, alloc)` / `io.jsonl_load(path, null_tokens, alloc)` | JSON array / JSON Lines → DataFrame |
| `io.html_load(path, alloc)`, `_table_index`, `_table_id`, `_all`, `io.html_load_from_string`, `io.html_load_all_from_string` | HTML tables → DataFrame(s) |
| `io.xlsx_load(path, alloc)`, `io.xlsx_load_sheet(path, name, alloc) -> (df, found)`, `io.xlsx_sheet_names(path, alloc)` | Excel worksheets |
| `ex.csv_save(&df, path, sep, alloc) -> bool`, `ex.json_write`, `ex.jsonl_write`, `ex.html_write`, `ex.xlsx_save -> bool` | DataFrame → file |
| `net.http_get(url, alloc) -> (body, ok)`, `net.http_get_with_cookie`, `net.download_file(url, path) -> ok` | HTTP |
| `net.read_csv_from_url`, `net.read_html_from_url`, `net.read_table_from_url` `-> (df, ok)` | a table from a URL |
| `net.read_yahoo(symbol, interval, range, alloc) -> (df, ok)` | Yahoo history as a DataFrame |
| `net.read_yahoo_series(symbol, interval, range, alloc) -> (Yahoo_Series, ok)`, `net.yahoo_series_free` | one ticker as arrays, days without a price dropped |
| `net.read_yahoo_panel(tickers, interval, range, alloc) -> (Yahoo_Panel, ok, failed)`, `net.yahoo_panel_free` | several tickers on their common dates |
| `ts.log_returns(prices, n_series = 1, alloc)`, `ts.simple_returns(prices, n_series = 1, alloc)` | prices → returns, single series or row-major panel |
| `w.column_own_name(&col, name)` | give a column its own copy of its name |
