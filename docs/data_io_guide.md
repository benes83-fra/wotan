# Data in and out: `importer`, `exporter` and `net`

Three packages move data into and out of Wotan:

- **`importer`** reads CSV, JSON, JSON Lines, HTML tables, Excel workbooks and Parquet files into `core` DataFrames.
- **`exporter`** writes DataFrames to CSV, JSON, JSON Lines, HTML and Excel.
- **`net`** fetches data over HTTP. It also loads price histories from Yahoo Finance, as a DataFrame, as plain arrays for one ticker, or as a panel of several tickers matched by date.

This guide covers all three. It assumes you know the `DataFrame` from the [core guide](core_guide.md). Every example was compiled and run against the current code, and the output shown is the real output. The examples free everything they allocate: a run under a tracking allocator reports no leaks. The Yahoo examples ran on 2026-11-13, so their numbers will differ when you run them.

## Contents

1. [Setup](#1-setup)
2. [Overview](#2-overview)
3. [CSV](#3-csv)
4. [JSON and JSON Lines](#4-json-and-json-lines)
5. [HTML tables](#5-html-tables)
6. [Excel](#6-excel)
7. [Parquet](#7-parquet)
8. [Writing files](#8-writing-files)
9. [HTTP](#9-http)
10. [Yahoo Finance](#10-yahoo-finance)
11. [CBOE index histories](#11-cboe-index-histories)
12. [Tests](#12-tests)
13. [Quick reference](#13-quick-reference)

## 1. Setup

```odin
import ts  "wotan/analytics" // log_returns / simple_returns (section 10)
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
| Parquet | `io.parquet_load(path, columns)`, `io.parquet_load_from_bytes(data, columns)` | |
| URL | `net.http_get(url)`, `net.download_file(url, path)`, `net.read_csv_from_url`, `read_html_from_url`, `read_table_from_url` | |
| Yahoo Finance | `net.read_yahoo`, `net.read_yahoo_series`, `net.read_yahoo_panel` | |

**Types.** The text-based importers (all but Parquet, whose files declare their types) infer each column's type from its values:

- whole numbers become `Int`, and decimals become `Float` (a mix of the two becomes `Float`);
- `true`/`false` become `Bool`;
- ISO dates, times and datetimes become `Date`, `Time` and `Datetime`;
- anything else, or a mix that fits none of these, becomes `String`.

Empty fields and the tokens `NA`, `null` and `None` become nulls.

**Ownership.** A loader returns a DataFrame that you free with `w.destroy_dataframe`. The DataFrame owns its column names and string values; it doesn't point into the file buffer or the temp allocator.

**Failure.** The CSV, JSON, HTML and Excel loaders panic when a file cannot be read or doesn't parse; the panic message names the loader and the file. `parquet_load` returns `(df, ok)` instead (section 7). A sheet name that the workbook doesn't have is not an error: `xlsx_load_sheet` returns `found = false`. Everything that goes over the network returns `(value, ok)` instead, with `ok` false when the request fails or brings no data. Network failures are normal (rate limits, outages), so they are reported, not fatal.

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

`xlsx_load(path, alloc)` reads the first worksheet of an `.xlsx` workbook, and `xlsx_load_sheet(path, name, alloc)` reads a worksheet by name and returns `(df, found)`. `xlsx_sheet_names(path, alloc)` lists the worksheets in workbook order; free each name and the slice. All three panic when the file is missing or is not an `.xlsx` workbook.

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

## 7. Parquet

`parquet_load(path, columns = nil, alloc) -> (df, ok)` reads an Apache Parquet file, the columnar format that pandas (`to_parquet`), pyarrow, Spark, DuckDB and Polars write. `columns` names the columns to load, in the order you want them. Only those columns' data is read from disk, which makes loading a few columns of a wide file much faster than loading all of it. `parquet_load_from_bytes(data, columns, alloc)` reads a file already in memory.

```odin
px, ok := io.parquet_load("demo_files/parquet/pandas_index.parquet")
if !ok {return}
defer w.destroy_dataframe(&px)
w.dataframe_pretty_print(&px)

// only three columns, in this order; the others are not read
df, ok2 := io.parquet_load("demo_files/parquet/types_dict_snappy.parquet", {"ts_us", "cat", "dec"})
if !ok2 {return}
defer w.destroy_dataframe(&df)
w.dataframe_pretty_print(&df, max_rows = 4)

_, ok3 := io.parquet_load("demo_files/parquet/nested.parquet")
fmt.println("nested:", ok3)
```

```
    close   volume  date
  0 101.5   1200    2024-01-02 00:00:00
  1 102.25  900     2024-01-03 00:00:00
  2 NULL    0       2024-01-04 00:00:00
  3 99      4500    2024-01-05 00:00:00
    ts_us                cat  dec
  0 2024-03-01 09:30:15  EU   123.45
  1 1969-12-31 23:59:59  US   -0.01
  2 NULL                 EU   NULL
  3 1970-01-01 00:00:00  US   9.999999999e+07
... (8 more rows)
parquet: column "tags" is nested (a list, map or struct), which is not supported; pass `columns` to load the others
nested: false
```

The first file was written by pandas from a frame indexed by date. pandas stores its index as an ordinary column (here `date`), and the loader keeps it as one; `w.set_index(&px, "date")` makes it the index again. pyarrow writes pandas' NaN as null, so the missing close is a null.

**Types.** A Parquet file declares its column types, so nothing is inferred:

| Parquet | DataFrame |
|---|---|
| `BOOLEAN` | `Bool` |
| `INT32`, `INT64`, and the 8-, 16-, 32- and 64-bit signed and unsigned integer types | `Int` |
| `FLOAT`, `DOUBLE` | `Float` |
| `DECIMAL`, however it is stored | `Float` (the nearest double) |
| `STRING`, `ENUM`, `JSON`, binary and fixed-length binary | `String` (binary as its raw bytes) |
| `DATE` | `Date` |
| `TIME` in milli-, micro- or nanoseconds | `Time` |
| `TIMESTAMP` in milli-, micro- or nanoseconds, and the legacy `INT96` timestamps of Spark and Impala | `Datetime` |

`Datetime` and `Time` hold whole seconds, so sub-second parts are dropped: `1969-12-31 23:59:59.5` becomes `23:59:59`, the second it falls in. Timestamps stored as instants (pandas' time-zone-aware columns) are read as UTC. Unsigned 64-bit values above 2^63 − 1 do not fit `Int` and wrap to negative numbers.

**What it reads.** Files whose columns are all flat (no lists, maps or structs), in any number of row groups. Values may be stored plain or dictionary-encoded, in data pages of version 1 or 2, uncompressed or compressed with Snappy (pandas' default), gzip or LZ4.

**Failure.** `ok` is false, with the reason on stderr, when the file is missing or isn't a Parquet file, when a requested column doesn't exist or is named twice, and when a column needs something the reader doesn't support: a nested type, the Zstandard, Brotli or LZO codec, one of the delta or byte-stream-split encodings, or encryption. Each case names the column, and `columns` lets you load the others. The loader never returns partial or guessed data. Damaged files fail cleanly too: the metadata is checked as it is read, so a corrupt or truncated file gives `ok = false`, not a crash or a frame of the wrong shape.

## 8. Writing files

All writers produce text that the matching importer reads back to the same DataFrame. `io_roundtrip_test` checks this for every column type, with nulls, integers beyond 2^53, tiny and huge floats, and strings with separators, quotes, line breaks, outer spaces, Unicode, markup and the text `NA`.

| Writer | Output |
|---|---|
| `csv_save(&df, path, sep = ',', allocator) -> bool` | Header plus one line per row. Nulls are empty unquoted fields. A string is quoted when it would otherwise read back differently: empty, with outer spaces, a null token, or containing the separator, a quote or a line break. |
| `csv_string(&df, sep = ',', allocator) -> string` | The text `csv_save` writes, as a string you free. |
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

## 9. HTTP

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

## 10. Yahoo Finance

Yahoo's chart API gives daily, weekly or monthly histories of stocks, ETFs and indices, with dividends and splits. It needs no key, but it rate-limits heavy use, so every Yahoo procedure returns `ok`. The interval is `.Daily`, `.Weekly` or `.Monthly`. The range is `.OneMonth`, `.ThreeMonths`, `.SixMonths`, `.OneYear`, `.TwoYears`, `.FiveYears`, `.TenYears`, `.YTD` or `.Max`, which is the whole history at the requested interval.

There are three levels:

| Procedure | Gives | Use it for |
|---|---|---|
| `read_yahoo(symbol, interval, range, alloc)` | `(DataFrame, ok)`: Date, Open, High, Low, Close, AdjClose, Volume, Dividend, Split | inspecting the raw data, dividends and splits |
| `read_yahoo_series(symbol, interval, range, alloc)` | `(Yahoo_Series, ok)`: arrays `dates`, `open`, `high`, `low`, `close`, `adj_close`, `volume` | one ticker in numeric code |
| `read_yahoo_panel(tickers, interval, range, alloc)` | `(Yahoo_Panel, ok, failed)`: `dates`, `adj_close`, `volume` and `dollar_volume`, row-major `[day * n_tickers + ticker]` | several tickers on the same days |

**Close or AdjClose.** `Close` is adjusted for splits only. `AdjClose` is also adjusted for dividends, so its returns are total returns. Use `AdjClose` for returns. The latest adjusted close equals the latest close, since adjustment rescales only the earlier days.

### 10.1 The DataFrame and Yahoo's missing days

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

### 10.2 One ticker: `read_yahoo_series`

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

### 10.3 Several tickers: `read_yahoo_panel`

`read_yahoo_panel` loads each ticker and keeps the dates on which all of them have a positive adjusted close. It returns:

- the adjusted closes, the volumes (shares traded) and the dollar volumes (adjusted close × volume), row-major `[day * n_tickers + ticker]`;
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

SPY and the VIX move strongly against each other on the same day. Paired by row index, the same two years give a correlation of −0.09, because most pairs then mix different days.

Code that reads a DataFrame (a strategy or backtest that looks columns up by name) gets one from `yahoo_panel_to_dataframe`: a Date column, then one Float column of adjusted closes per ticker, named by the slice you pass.

```odin
tickers := []string{"KO", "PEP"}
p, ok, _ := net.read_yahoo_panel(tickers, .Daily, .TwoYears)
defer net.yahoo_panel_free(&p)
if !ok {fmt.println("could not load the panel"); return}

T, N := len(p.dates), p.n_tickers
last := (T - 1) * N
fmt.printf("KO volume %.0f shares, dollar volume %.0f\n", p.volume[last], p.dollar_volume[last])

df := net.yahoo_panel_to_dataframe(&p, tickers)
defer w.destroy_dataframe(&df)
w.dataframe_pretty_print(&df, 4)
```

```
KO volume 10004198 shares, dollar volume 992316424
    Date        KO                  PEP                 
  0 2024-11-13  59.576717376708984  152.43211364746094  
  1 2024-11-14  59.15117263793945   152.8114776611328   
  2 2024-11-15  58.38518142700195   146.76934814453125  
  3 2024-11-18  58.4986686706543    146.5010223388672   
... (498 more rows)
```

### 10.4 Today's row while the market is open

During the exchange's regular session, Yahoo's last row is the session so far: its close is the current price and its volume the shares traded until now. Both change until the close. `read_yahoo_series` and `read_yahoo_panel` mark this case with `last_is_live`. A backtest or a model fitted on closing prices should leave that row out:

```odin
s, ok := net.read_yahoo_series("SPY", .Daily, .OneMonth)
defer net.yahoo_series_free(&s)
if !ok {fmt.println("Yahoo is unreachable or rate-limiting"); return}
n := len(s.dates)
fmt.printf("last row: %s, %.2f, live %v\n",
	w.date_to_string(s.dates[n - 1], context.temp_allocator), s.adj_close[n - 1], s.last_is_live)
if s.last_is_live {n -= 1} // closes only
fmt.printf("last close: %s, %.2f\n",
	w.date_to_string(s.dates[n - 1], context.temp_allocator), s.adj_close[n - 1])
```

Run at 15:27 New York time:

```
last row: 2026-11-20, 693.07, live true
last close: 2026-11-19, 715.87
```

Before the open and after the close `last_is_live` is false, and every row is a close. For a panel, the first `len(p.dates) - 1` dates are the closes: the arrays are row-major, so their first `(len(p.dates) - 1) * p.n_tickers` values. With weekly and monthly intervals the last row also covers a period that has not ended, even after the close; `last_is_live` only reports the running session.

`read_yahoo` returns the DataFrame alone. To test a chart answer you fetched yourself, pass its text and the current Unix time to `yahoo_session_live(text, time.time_to_unix(time.now()))`. It reads the regular session's start and end and the time of the last trade from Yahoo's `meta` block.

### 10.5 More Yahoo procedures

- `read_yahoo_json(symbol, alloc)` returns the daily history over the full range. Unlike `read_yahoo(symbol, .Daily, .Max)`, its request asks for no events, so the Dividend and Split columns are 0.
- `yahoo_json_to_dataframe(text, alloc)` parses a chart answer you fetched yourself.
- Option chains are in `finance`: `fin.fetch_yahoo_options` and `fin.fetch_yahoo_option_surface` (see the [risk guide, section 8](finance_risk_guide.md#8-market-data)).
- `yahoo_download_csv(symbol, alloc, interval = .Daily, range = .Max)` returns the history as CSV text, with the columns of `read_yahoo`. `ok` is false when the request fails, as with `read_yahoo`. The text is in `csv_save` format, so `io.csv_load_from_string` reads it back to the same DataFrame. It is built from the chart API: Yahoo's own CSV download requires a paid login.

  ```odin
  text, ok := net.yahoo_download_csv("MSFT", context.allocator, .Daily, .OneMonth)
  defer delete(text)
  ```

  The first lines:

  ```
  Date,Open,High,Low,Close,AdjClose,Volume,Dividend,Split
  2026-10-14,545.27001953125,545.75,532.5399780273438,533.6900024414062,533.6900024414062,22324300,0.0,0.0
  2026-10-15,533.3200073242188,541.1300048828125,531.219970703125,539.4400024414062,539.4400024414062,24151200,0.0,0.0
  ```

The DataFrames `yahoo_json_to_dataframe` builds are checked against an independent pandas parse of the same JSON text. Over five years of SPY, the VIX and AAPL and ten years of NVDA, the dates, the positions of the nulls and every value, including 40 dividends and 2 splits, agree to 1e-13.

## 11. CBOE index histories

CBOE publishes the daily history of its indices as CSV files: the volatility indices (VIX, VXN, VVIX, the single-stock VXAPL, VXAZN, VXGOG, VXGS and VXIBM), the implied-correlation indices (COR1M, COR3M and others) and indices such as SKEW. Yahoo quotes several of them (`^COR1M`, `^VXAPL`) but serves only the latest value, without history. `net.read_cboe_index(symbol)` loads the history from CBOE:

```odin
cor, ok := net.read_cboe_index("COR1M") // a leading ^ is ignored
if !ok {return} // CBOE unreachable, or an unknown symbol
defer net.cboe_series_free(&cor)
n := len(cor.dates)
ta := context.temp_allocator
fmt.printf("%s: %d days, %s %.2f .. %s %.2f
", cor.symbol, n,
    w.date_to_string(cor.dates[0], ta), cor.close[0], w.date_to_string(cor.dates[n - 1], ta), cor.close[n - 1])
```

```
COR1M: 5255 days, 2006-01-03 23.50 .. 2026-11-20 36.05
```

- **The result** is a `Cboe_Series`: `symbol`, `dates` (ascending) and `close`, the index's closing value of each day. Free it with `cboe_series_free`.
- **The files** come in two formats, `DATE,OPEN,HIGH,LOW,CLOSE` and `DATE,<value>`. The loader takes the CLOSE column, or the last column. Rows whose date or value does not parse, or whose value is not positive, are left out.
- **Failures:** `ok` is false when CBOE cannot be reached, when it does not know the symbol (it answers with HTTP 403) or when the file has no valid row.
- **Your own files:** `cboe_history_parse(text, symbol)` parses a file you fetched or saved yourself.

The dates are CBOE's trading days. To combine an index with Yahoo prices, put its values in a `map[w.Date]f64` and keep the dates both have, as `dispersion_trading_test` does with VXAPL, VXAZN, VXGOG and COR1M.

## 12. Tests

| Test | What it checks |
|---|---|
| `io_roundtrip_test` | exact CSV, JSON, JSONL and XLSX round trips for all column types, nulls and edge cases; HTML up to blank strings; CSV corner cases (BOM, CRLF, short and long rows); Excel-style workbooks (rich text, styles, error cells) |
| `csv_export_test`, `json_basic_test`, `jsonl_basic_test`, `json_export_test` | CSV and JSON loading and writing |
| `html_basic_test`, `html_extended_test`, `html_tags_test`, `html_export_test` | HTML tables by index, id and all, tags inside cells, writing |
| `excel_basic_test`, `excel_date_test`, `excel_export_test`, `zip_smoke_test` | reading sheets and dates, writing workbooks, the ZIP container |
| `parquet_test` | every column type, with nulls and extreme values, against pyarrow 14.0.2 in five layouts (plain and dictionary values, page versions 1 and 2, no compression, Snappy, gzip, LZ4, several row groups and many small pages); INT96 timestamps; decimals stored as integers (DuckDB); a pandas file; column selection; `ok = false` for unsupported files and for truncated or corrupted copies. The fixtures in `demo_files/parquet` are written by `scripts/make_parquet_fixtures.py` |
| `http_get_test`, `yahoo_finance_json_test`, `yahoo_finance_events_test` | live HTTP and Yahoo requests, including dividends and splits |
| `yahoo_max_range_test` | `.Max` gives the whole daily history (SPY from 1993-01-29, no gap over a week) and weekly rows a week apart |
| `cboe_index_test` | offline: both CBOE file formats and the rows left out; live: COR1M's history from 2006-01-03, ascending, and an unknown symbol |
| `yahoo_live_row_test` | `last_is_live` of a series and a panel agrees with the trading hours in Yahoo's answer; a live row is dated today |
| `data_io_fixes_test` | offline: the chart parser, `Yahoo_Series`, and `yahoo_session_live` on a recorded answer during, after and before the session |

Most of the finance tests load their data with `read_yahoo`, `read_yahoo_series` or `read_yahoo_panel`. When Yahoo cannot be reached they count a failure and go on; they never fall back to made-up data.

## 13. Quick reference

| Procedure | Purpose |
|---|---|
| `io.csv_load(path, types, sep, null_tokens, alloc)` / `io.csv_load_from_string(text, alloc, types, sep, null_tokens)` | CSV file / text → DataFrame |
| `io.json_load(path, null_tokens, alloc)` / `io.jsonl_load(path, null_tokens, alloc)` | JSON array / JSON Lines → DataFrame |
| `io.html_load(path, alloc)`, `_table_index`, `_table_id`, `_all`, `io.html_load_from_string`, `io.html_load_all_from_string` | HTML tables → DataFrame(s) |
| `io.xlsx_load(path, alloc)`, `io.xlsx_load_sheet(path, name, alloc) -> (df, found)`, `io.xlsx_sheet_names(path, alloc)` | Excel worksheets |
| `io.parquet_load(path, columns, alloc) -> (df, ok)`, `io.parquet_load_from_bytes(data, columns, alloc) -> (df, ok)` | Parquet file / bytes → DataFrame |
| `ex.csv_save(&df, path, sep, alloc) -> bool`, `ex.csv_string(&df, sep, alloc) -> string`, `ex.json_write`, `ex.jsonl_write`, `ex.html_write`, `ex.xlsx_save -> bool` | DataFrame → file |
| `net.http_get(url, alloc) -> (body, ok)`, `net.http_get_with_cookie`, `net.download_file(url, path) -> ok` | HTTP |
| `net.read_csv_from_url`, `net.read_html_from_url`, `net.read_table_from_url` `-> (df, ok)` | a table from a URL |
| `net.read_yahoo(symbol, interval, range, alloc) -> (df, ok)` | Yahoo history as a DataFrame |
| `net.yahoo_download_csv(symbol, alloc, interval, range) -> (text, ok)` | Yahoo history as CSV text |
| `net.read_yahoo_series(symbol, interval, range, alloc) -> (Yahoo_Series, ok)`, `net.yahoo_series_free` | one ticker as arrays, days without a price dropped; `last_is_live` marks a running session |
| `net.read_yahoo_panel(tickers, interval, range, alloc) -> (Yahoo_Panel, ok, failed)`, `net.yahoo_panel_free` | several tickers on their common dates; `last_is_live` as for a series |
| `net.yahoo_session_live(text, now_unix) -> bool` | whether a chart answer's last row is the running session |
| `net.read_cboe_index(symbol, alloc) -> (Cboe_Series, ok)`, `net.cboe_history_parse(text, symbol, alloc)`, `net.cboe_series_free` | a CBOE index's daily history (VIX, COR1M, VXAPL, SKEW, ...) |
| `net.yahoo_panel_to_dataframe(&panel, names, alloc)` | a panel's adjusted closes as a DataFrame (Date plus one column per ticker) |
| `ts.log_returns(prices, n_series = 1, alloc)`, `ts.simple_returns(prices, n_series = 1, alloc)` | prices → returns, single series or row-major panel |
| `w.column_own_name(&col, name)` | give a column its own copy of its name |
