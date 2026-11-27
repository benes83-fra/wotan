# Data in and out: `importer`, `exporter` and `net`

Three packages move data into and out of Wotan:

- **`importer`** reads CSV, JSON, JSON Lines, HTML tables, Excel workbooks and Parquet files into `core` DataFrames.
- **`exporter`** writes DataFrames to CSV, JSON, JSON Lines, HTML, Excel and Parquet.
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
| CSV | `io.csv_load(path)`, `io.csv_load_from_string(text)`; `try_` variants | `ex.csv_save(&df, path)` |
| JSON (array of objects) | `io.json_load(path)`, `io.try_json_load(path)` | `ex.json_write(&df, path, alloc)` |
| JSON Lines | `io.jsonl_load(path)`, `io.try_jsonl_load(path)` | `ex.jsonl_write(&df, path, alloc)` |
| HTML `<table>` | `io.html_load(path)`, `_table_index`, `_table_id`, `_all`, `io.html_load_from_string(text)`; `try_` variants | `ex.html_write(&df, path, alloc)` |
| Excel `.xlsx` | `io.xlsx_load(path)`, `io.xlsx_load_sheet(path, name)`, `io.xlsx_sheet_names(path)`; `try_` variants | `ex.xlsx_save(&df, path, alloc)` |
| Parquet | `io.parquet_load(path, columns, filters = ...)`, `io.parquet_load_from_bytes(data, columns)`, `io.parquet_info(path)` | `ex.parquet_save(&df, path)`, `ex.parquet_bytes(&df)` |
| URL | `net.http_get(url)`, `net.download_file(url, path)`, `net.read_csv_from_url`, `read_html_from_url`, `read_table_from_url` | |
| Yahoo Finance | `net.read_yahoo`, `net.read_yahoo_series`, `net.read_yahoo_panel` | |

**Types.** The text-based importers (all but Parquet, whose files declare their types) infer each column's type from its values:

- whole numbers become `Int`, and decimals become `Float` (a mix of the two becomes `Float`). Numbers are read as pandas reads them: decimal digits with an optional sign, a decimal point and an exponent, plus `inf`, `infinity` and `nan` in any case. Hexadecimal (`0x10`), digit separators (`1_000`) and floats beyond the `f64` range (`1e400`) are text. A whole number beyond the `i64` range makes its column `Float`, where pandas uses an unsigned type;
- `true`/`false` become `Bool`;
- ISO dates, times and datetimes become `Date`, `Time` and `Datetime`, with fractions of a second (`09:30:15.250`) to the nanosecond;
- durations in the form pandas writes them, with their days (`0 days 00:05:00`, `-1 days +23:55:00`), become `Duration`. pandas itself leaves such a column as text; `pd.to_timedelta` converts it. A bare `00:05:00` is a `Time`;
- datetimes with a UTC offset, as pandas writes a column with a time zone (`2024-03-10 03:30:00-04:00`, also `...Z` and `...+0530`), become a `Datetime` column with a time zone, holding the instants. When every value has the same offset, the zone is that offset (`-04:00`); when the offsets differ, as New York's do between winter and summer, it is `UTC`. Text cannot say which named zone it came from; `tz_convert` puts the column back into one. A column that mixes values with and without an offset stays `String`;
- anything else, or a mix that fits none of these, becomes `String`.

Empty fields and the tokens `NA`, `null` and `None` become nulls. So does `NaN` text in a column that is not `Float` (pandas would make an integer column with `NaN` a float column); in a `Float` column it is NaN.

**Column names.** The first row (the header) names the columns. As in pandas, a repeated name is numbered (`a`, `a.1`, `a.2`, skipping names the header already has) and an empty one becomes `Unnamed: <position>`, so every column can be found by name.

**Ownership.** A loader returns a DataFrame that you free with `w.destroy_dataframe`. The DataFrame owns its column names and string values; it doesn't point into the file buffer or the temp allocator.

**Failure.** Every CSV, JSON, HTML and Excel loader has a checked twin whose name starts with `try_`. It returns `(df, ok)`, and when `ok` is false the reason is on stderr, naming the loader and the file. Use the `try_` loaders where the file comes from outside the program:

```odin
df, ok := io.try_csv_load("demo_files/no_such.csv")
if !ok {return} // stderr: try_csv_load: cannot read "demo_files/no_such.csv"
defer w.destroy_dataframe(&df)
```

The loaders without `try_` panic with the same message instead; they suit scripts where a missing file is a bug. `parquet_load` always returns `(df, ok)` (section 7). Everything that goes over the network returns `(value, ok)` as well, with `ok` false when the request fails or brings no data. Network failures are normal (rate limits, outages), so they are reported, not fatal.

| Checked loader | `ok = false` when |
|---|---|
| `try_csv_load(path, types, sep, null_tokens, alloc)`, `try_csv_load_from_string(text, alloc, types, sep, null_tokens)` | the file cannot be read; the text has no header; `types` has the wrong length or a cell does not parse as its given type |
| `try_json_load(path, null_tokens, alloc)`, `try_jsonl_load(path, null_tokens, alloc)` | the file cannot be read; the JSON is invalid, not an array, empty, or holds something other than objects |
| `try_html_load(path, alloc)`, `try_html_load_table_index(path, i, alloc)`, `try_html_load_table_id(path, id, alloc)`, `try_html_load_from_string(text, alloc)` | the file cannot be read; there is no such table (the panicking loaders give an empty frame then) |
| `try_html_load_all(path, alloc) -> ([]df, ok)` | the file cannot be read |
| `try_xlsx_load(path, alloc)`, `try_xlsx_load_sheet(path, name, alloc)`, `try_xlsx_sheet_names(path, alloc) -> ([]string, ok)` | the file cannot be read or is not an `.xlsx` workbook; there is no worksheet, or none of that name |

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

The `NA` share count and MSFT's empty note are nulls, while NVDA's quoted `""` is an empty string. Rows with fewer fields than the header get nulls in the missing columns, and extra fields are ignored. A UTF-8 byte order mark, CRLF, LF and CR line ends and blank lines are handled.

**Speed.** The reader cuts the text into chunks at line ends and reads them on all cores: it finds every chunk's fields without copying them, infers the column types chunk by chunk, and converts each chunk's columns straight into the DataFrame. A quoted field whose line break falls on a chunk boundary is noticed and that part is read again in order, so the frame is the same for any number of threads (`util.parallel_max_threads` caps them; 1 reads on the calling thread). A file is mapped into memory rather than copied. Reading 1 million rows of seven columns (an 82 MB file with datetimes, tickers, prices with gaps, volumes, unique text, flags and dates), best of three runs on a 16-thread machine:

| Reader | Time |
|---|---|
| `io.csv_load`, all threads | 86 ms |
| `io.csv_load`, 1 thread | 328 ms |
| pyarrow 14 `csv.read_csv`, all threads / 1 thread | 49 ms / 314 ms |
| pandas 1.5.3 `read_csv` (dates left as text) | 1259 ms |
| pandas 1.5.3 `read_csv` with `parse_dates` (as `csv_load` reads them) | 1644 ms |

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

`xlsx_load(path, alloc)` reads the first worksheet of an `.xlsx` workbook, and `xlsx_load_sheet(path, name, alloc)` reads a worksheet by name and returns `(df, found)`. `xlsx_sheet_names(path, alloc)` lists the worksheets in workbook order; free each name and the slice. All three panic when the file is missing or is not an `.xlsx` workbook; their `try_` twins return `ok = false` (section 2).

The first row is the header. Cells are read with their Excel types: numbers, booleans, shared and inline strings, rich text, and errors (read as nulls). A number whose cell style is a date or time format becomes a `Date`, `Time` or `Datetime`, to the millisecond (Excel's own resolution; `xlsx_save` writes fractions of a second the same way). A number with an elapsed-time format such as `[h]:mm:ss` is a length of time and becomes a `Duration`, to the microsecond; `xlsx_save` writes `Duration` columns that way. (openpyxl, which pandas uses for `.xlsx`, reads such cells as times of day.) Excel has no time zones: `xlsx_save` writes a `Datetime` column with a time zone in its local time, which reads back as plain wall-clock times. A custom format such as `#,##0 "USD"` is not mistaken for a date.

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

Apache Parquet is the columnar file format of the data world: compact, typed, and fast to read one column at a time.

### 7.1 Reading

`parquet_load(path, columns = nil, alloc) -> (df, ok)` reads an Apache Parquet file, the columnar format that pandas (`to_parquet`), pyarrow, Spark, DuckDB and Polars write. `columns` names the columns to load, in the order you want them. Only those columns' data is read from disk, which makes loading a few columns of a wide file much faster than loading all of it. A fourth parameter, `filters`, loads only the rows in a range of values (section 7.4). `parquet_load_from_bytes(data, columns, alloc)` reads a file already in memory.

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

// lists, structs and maps arrive as JSON text
nest, ok3 := io.parquet_load("demo_files/parquet/nested_duckdb.parquet", {"id", "ints", "st", "m"})
if !ok3 {return}
defer w.destroy_dataframe(&nest)
w.dataframe_pretty_print(&nest, max_rows = 6)
st, _ := w.column_at_string(w.column(&nest, "st"), 5)
fmt.println("st of row 5:", st)
```

```
    close   volume  date
  0 101.5   1200    2024-01-02 00:00:00
  1 102.25  900     2024-01-03 00:00:00
  2 NULL    0       2024-01-04 00:00:00
  3 99      4500    2024-01-05 00:00:00
    ts_us                       cat  dec
  0 2024-03-01 09:30:15.999999  EU   123.45
  1 1969-12-31 23:59:59.500     US   -0.01
  2 NULL                        EU   NULL
  3 1970-01-01 00:00:00         US   9.999999999e+07
... (8 more rows)
    id  ints        st                               m
  0 0   NULL        NULL                             {"k0":0}
  1 1   [1,null,2]  {"a":1,"b":"r1","l":[0.5,null]}  {"k1":1}
  2 2   [2,null,4]  {"a":2,"b":"r2","l":[1.0,null]}  {"k2":2}
  3 3   [3,null,6]  {"a":3,"b":"r3","l":[1.5,null]}  {"k3":3}
  4 4   [4,null,8]  NULL                             {"k4":4}
  5 5   []          {"a":5,"b":"r5","l":[2.5,null]}  {"k5":5}
... (114 more rows)
st of row 5: {"a":5,"b":"r5","l":[2.5,null]}
```

The first file was written by pandas from a frame indexed by date. pandas stores its index as an ordinary column (here `date`), and the loader keeps it as one; `w.set_index(&px, "date")` makes it the index again. pyarrow writes pandas' NaN as null, so the missing close is a null.

**Types.** A Parquet file declares its column types, so nothing is inferred:

| Parquet | DataFrame |
|---|---|
| `BOOLEAN` | `Bool` |
| `INT32`, `INT64`, and the 8-, 16-, 32- and 64-bit signed and unsigned integer types | `Int` |
| `FLOAT`, `DOUBLE`, `FLOAT16` | `Float` |
| `DECIMAL`, however it is stored | `Float` (the nearest double) |
| `STRING`, `ENUM`, `JSON`, binary and fixed-length binary | `String` (binary as its raw bytes) |
| `UUID` | `String`, in the usual `8-4-4-4-12` hexadecimal form |
| `DATE` | `Date` |
| `TIME` in milli-, micro- or nanoseconds | `Time` |
| `TIMESTAMP` in milli-, micro- or nanoseconds, and the legacy `INT96` timestamps of Spark and Impala | `Datetime` |
| `INT64` that the file's Arrow schema calls a duration (pandas' `timedelta64`, pyarrow's `duration`, in any unit) | `Duration` |
| lists, maps and structs, nested to any depth | `String` holding JSON text (see below) |

Times and timestamps keep every unit down to the nanosecond. Timestamps stored as instants (`isAdjustedToUTC`, as pandas, pyarrow, Spark and DuckDB store a column with a time zone) become a `Datetime` column with a time zone: the zone the file's Arrow schema names, such as `America/New_York` or `+05:30`, or `UTC` when it names none. Timestamps stored as wall-clock times, and `INT96` timestamps, have no zone. Unsigned 64-bit values above 2^63 − 1 do not fit `Int` and wrap to negative numbers.

**Nested columns.** A list, map or struct column becomes a `String` column with one JSON text per row. This is how the JSON loader keeps nested values (section 4), and the text can be parsed with `core:encoding/json`:

- a list is `[1,2,3]`, and an empty list is `[]`;
- a struct is an object with its fields in schema order, `{"a":1,"b":"x"}`;
- a map is an object, `{"key":value}`. Keys that are not strings become their text, so the integer key 7 is `"7"`;
- a missing value inside a list, struct or map is `null`. A missing value of the whole column is a null cell, not the text `null`;
- the values inside are written as `ex.json_write` writes cells (section 8). Numbers are JSON numbers, NaN and infinities are `null`, and decimals are their nearest double. Unsigned 64-bit integers are written exactly, also above 2^63 − 1. Dates, times and timestamps are strings in wotan's text form, such as `"2024-03-01 09:30:00.5"`. Binary values without a text annotation are strings with each byte from 0x80 up written as `\u00XX`.

The text is the same as Python's `json` module would produce from pyarrow's values (`to_pylist`) under these rules. `parquet_nested_test` checks this for files from pyarrow and DuckDB. On a million rows with a list column (3.5 million integers) and a two-field struct, best of three runs on a 16-thread machine, wotan takes 133 ms (389 ms on one thread). `pandas.read_parquet`, which builds Python lists and dicts, takes 724 ms, and pyarrow's `read_table`, which keeps Arrow arrays and builds no text, takes 64 ms.

**What it reads.** Files with flat and nested columns, in any number of row groups, in data pages of version 1 or 2. Values may use any of the format's encodings: plain, dictionary, run-length (booleans), the three delta encodings, and byte stream split. Pages may be uncompressed or compressed with Snappy (pandas' and pyarrow's default), Zstandard (the default of Polars, DuckDB and newer Spark), gzip or LZ4.

**Speed.** The columns are decoded in parallel, one thread per core. `util.parallel_max_threads` (package `wotan/util`) caps the threads, and 1 decodes on the calling thread; the frame is the same either way. Reading 1 million rows of seven columns (dates, tickers, prices with NaN, volumes, unique text, flags) written by pyarrow, best of five runs on a 16-thread machine:

| Compression | wotan, all threads | wotan, 1 thread | pyarrow 14, all threads | pyarrow 14, 1 thread | two of the columns, wotan / pyarrow |
|---|---|---|---|---|---|
| none | 26 ms | 56 ms | 43 ms | 82 ms | 8 / 9 ms |
| Snappy | 38 ms | 87 ms | 53 ms | 124 ms | 15 / 17 ms |
| Zstandard | 42 ms | 110 ms | 56 ms | 128 ms | 21 / 22 ms |
| gzip | 61 ms | 190 ms | 65 ms | 173 ms | 46 / 36 ms |

pyarrow's times are for its own Arrow tables; `pandas.read_parquet`, which converts them to a pandas frame, takes about 360 ms on the same file. gzip is the one codec where a single wotan thread is slower than pyarrow's, because Odin's inflate is slower than zlib.

Zstandard is a package of its own, `wotan/zstd`, usable on any data:

- `zstd.decompress(dst, src) -> (written, ok)` decompresses any Zstandard data (several frames, skippable frames, checksums verified) into a buffer of known size, and `zstd.decompressed_size(src)` gives the size the frames declare. Frames that need a dictionary are refused.
- `zstd.compress(src, level, alloc) -> (bytes, ok)` compresses into one frame that declares its size; free the bytes with `delete`. `zstd.compress_to_dynamic(&buf, src, level) -> ok` appends the frame to a `[dynamic]u8` instead. libzstd and every other Zstandard decoder read the frames. `ok` is false only for input of 2 GiB or more.

The `level` runs from 1 (fastest) to `zstd.MAX_LEVEL` (12, smallest); the default, `zstd.DEFAULT_LEVEL`, is 3, as for libzstd and the `zstd` command line. Level 0 means the default, levels below 1 compress as 1, and levels above 12 compress as 12. At each level, a 26 MB set of text, CSV, numeric arrays and random bytes comes out within about 1% of libzstd's total at that level; single files vary more, in either direction. libzstd's levels 13 to 22 use slower match finders that `wotan/zstd` does not have. On two of the larger files, one thread, best of two runs:

| Level | wotan's source code, 3.8 MB | Time | minute prices as CSV, 11.6 MB | Time | libzstd's sizes | libzstd's times |
|---|---|---|---|---|---|---|
| 1 | 1.09 MB | 17 ms | 3.03 MB | 52 ms | 1.13 / 2.89 MB | 10 / 27 ms |
| 3 | 1.01 MB | 22 ms | 3.02 MB | 81 ms | 1.02 / 3.05 MB | 16 / 58 ms |
| 5 | 0.96 MB | 42 ms | 2.69 MB | 128 ms | 0.95 / 2.79 MB | 30 / 128 ms |
| 7 | 0.91 MB | 138 ms | 2.53 MB | 443 ms | 0.90 / 2.54 MB | 65 / 223 ms |
| 9 | 0.90 MB | 110 ms | 2.50 MB | 527 ms | 0.89 / 2.44 MB | 73 / 323 ms |
| 12 | 0.89 MB | 437 ms | 2.50 MB | 2849 ms | 0.88 / 2.45 MB | 166 / 752 ms |

wotan takes 1 to 2 times as long as libzstd up to level 9 and 2.5 to 4 times as long at level 12; libzstd's higher levels search with SIMD instructions.

**Failure.** `ok` is false, with the reason on stderr, when the file is missing or isn't a Parquet file, when a requested column doesn't exist or is named twice, and when a column needs something the reader doesn't support: the Brotli or LZO codec, or encryption. Each case names the column, and `columns` lets you load the others. The loader never returns partial or guessed data. Damaged files fail cleanly too: the metadata is checked as it is read, so a corrupt or truncated file gives `ok = false`, not a crash or a frame of the wrong shape.

### 7.2 Writing

`ex.parquet_save(&df, path, compression = .Snappy, row_group_rows = 1 << 20, dictionary = true, compression_level = 0) -> bool` writes a DataFrame as a Parquet file. `ex.parquet_bytes(&df, ...)` returns the file's bytes instead, as `([]u8, ok)`; free them with `delete`.

```odin
trades := io.csv_load_from_string(`ticker,date,price,shares
AAPL,2024-03-01,179.66,100
MSFT,2024-03-01,415.50,
NVDA,2024-03-04,852.37,20
`)
defer w.destroy_dataframe(&trades)
if !ex.parquet_save(&trades, "trades.parquet") {return}

back, ok := io.parquet_load("trades.parquet")
if !ok {return}
defer w.destroy_dataframe(&back)
w.dataframe_pretty_print(&back)
for c in back.columns {fmt.printf("%s: %v  ", c.name, c.type)}
fmt.println()

// gzip: smaller files, slower to write
small := ex.parquet_save(&trades, "trades_gzip.parquet", .Gzip)
fmt.println("gzip:", small)
```

```
    ticker  date        price   shares
  0 AAPL    2024-03-01  179.66  100
  1 MSFT    2024-03-01  415.5   NULL
  2 NVDA    2024-03-04  852.37  20
ticker: String  date: Date  price: Float  shares: Int
gzip: true
```

Every column is written with the standard Parquet type for its values, so pandas, pyarrow, Spark, DuckDB and Polars read it with the same meaning, and `parquet_load` gives back the same DataFrame:

| DataFrame | Parquet |
|---|---|
| `Int` | `INT64` |
| `Float` | `DOUBLE` (NaN is written as NaN, not as null) |
| `Bool` | `BOOLEAN` |
| `String` | `BYTE_ARRAY` with the `STRING` type |
| `Date` | `INT32` with the `DATE` type |
| `Datetime` | `INT64` `TIMESTAMP` in microseconds, as wall-clock time (no time zone) |
| `Datetime` with a time zone | `INT64` `TIMESTAMP` in microseconds as UTC instants, the zone named in the file's Arrow schema |
| `Time` | `INT64` `TIME` in microseconds |
| `Duration` | `INT64` nanoseconds, named a duration in the file's Arrow schema |

pandas reads a zoned column back as `datetime64[ns, zone]` and DuckDB as `TIMESTAMP WITH TIME ZONE`. A file gets an Arrow schema only when it has a `Duration` or zoned `Datetime` column.

A `Datetime` or `Time` column whose values have digits below the microsecond is written in nanoseconds instead, so nothing is lost, as long as every value fits that unit. For `Datetime`, the nanosecond range ends at 1677 and 2262; a column with a date outside it stays in microseconds and drops the extra digits.

Every column may hold nulls. pandas reads an integer column with nulls as `float64`, as it does for any integer data with missing values. A frame indexed with `set_index(&df, name, drop = true)` keeps its index outside its columns; the writer stores it as the first column, under its name, the way pandas stores its index.

The options:

- **`compression`**: `.Snappy` (the default, as in pandas) is fast with moderate compression. `.Zstd` (Zstandard) gives files about a third smaller than Snappy's, a little smaller than gzip's, and takes about 1.5 times as long to write as Snappy. `.Gzip` gives files about as small and writes several times slower. `.Lz4` is about as fast as Snappy. `.None` stores the pages as they are.
- **`compression_level`**: the level of `.Zstd`, 1 to 12, or of `.Gzip`, 1 to 9. 0, the default, takes the codec's own default, 3 for Zstandard and 6 for gzip; levels out of range are clamped, and the other codecs ignore the level. Higher levels pay off when the values repeat: text and codes more than random numbers. On the frame below, whose prices, volumes and IDs are random, every Zstandard level gives about 20 MB: level 1 writes in 156 ms, level 3 in 188 ms, level 12 in 1.8 s. gzip's level 9 takes 7.8 s for 20.7 MB.
- **`row_group_rows`**: the rows per row group (a million by default). Each row group records the minimum, maximum and null count of every column. Readers can then skip the row groups that cannot match a filter; `parquet_load` with `filters` does this (section 7.4), as do DuckDB and pyarrow.
- **`dictionary`**: String columns are stored as a dictionary of their distinct values plus small indices, which suits columns that repeat values, such as tickers or sectors. A column whose dictionary would pass 1 MiB in a row group is stored plainly. Pass `false` to store every String column plainly.

The column chunks are encoded and compressed in parallel, and `util.parallel_max_threads` caps the threads as for reading; the file's bytes are the same for any number of threads. On the 1-million-row frame from section 7.1, best of three runs, compared with pyarrow 14 writing the same data:

| Compression | wotan, all threads | wotan, 1 thread | Size | pyarrow 14 | pyarrow size |
|---|---|---|---|---|---|
| Snappy | 111 ms | 231 ms | 29.0 MB | 304 ms | 30.9 MB |
| Zstandard | 192 ms | 413 ms | 20.1 MB | 273 ms | 23.8 MB |
| gzip | 461 ms | 1192 ms | 20.9 MB | 2360 ms | 22.9 MB |
| LZ4 | 76 ms | 180 ms | 29.0 MB | 257 ms | 32.0 MB |
| none | 57 ms | 150 ms | 44.4 MB | 250 ms | 41.4 MB |

pyarrow compresses Zstandard at libzstd's level 1 by default; wotan's Zstandard pages come out about the size of level 3.

`parquet_save` returns false, with the reason on stderr, when the file cannot be written (for example, a folder that doesn't exist) or when a hidden index has the name of one of the columns.

### 7.3 What a file holds: `parquet_info`

`parquet_info(path, alloc) -> (info, ok)` reads only the footer of a Parquet file, the part that describes it, and loads no data. Even for a large file it returns in well under a millisecond. Free the result with `destroy_parquet_info`. `parquet_info_from_bytes(data, alloc)` does the same for a file in memory. `print_parquet_info(&info)` prints a summary:

```odin
prices := io.csv_load("data/spy_prices.csv")
defer w.destroy_dataframe(&prices)
if !ex.parquet_save(&prices, "spy.parquet", row_group_rows = 10) {return}

info, ok := io.parquet_info("spy.parquet")
if !ok {return}
defer io.destroy_parquet_info(&info)
io.print_parquet_info(&info)
for g, i in info.row_groups {
	s := g.stats[0] // the Date column
	fmt.printfln(
		"row group %d: %d rows, %s .. %s",
		i,
		g.rows,
		w.date_to_string(s.min.(w.Date), context.temp_allocator),
		w.date_to_string(s.max.(w.Date), context.temp_allocator),
	)
}
```

```
30 rows in 3 row groups, written by wotan
  column  type   parquet type  nulls  range
  Date    Date   INT32 DATE        0  2023-01-03 .. 2023-02-14
  Open    Float  DOUBLE            0  384.58 .. 411.26
  High    Float  DOUBLE            0  387.16 .. 413.43
  Low     Float  DOUBLE            0  381.04 .. 410.58
  Close   Float  DOUBLE            0  383.6 .. 412.85
  Volume  Int    INT64             0  54321800 .. 88078700
row group 0: 10 rows, 2023-01-03 .. 2023-01-17
row group 1: 10 rows, 2023-01-18 .. 2023-01-31
row group 2: 10 rows, 2023-02-01 .. 2023-02-14
```

A `Parquet_Info` has these fields:

| Field | Holds |
|---|---|
| `rows` | the number of rows in the file |
| `created_by` | the library that wrote the file, if it says |
| `columns` | one `Parquet_Column_Info` per top-level column: `name`; `type`, the DataFrame type it loads as; `parquet_type`, the type as stored (for example `INT64 TIMESTAMP(MICROS, UTC)` or `FIXED_LEN_BYTE_ARRAY(16) DECIMAL(38, 6)`); `nullable`; and `reason`, which explains why wotan cannot load the column (for example one compressed with Brotli), or is `""` when it can. A column wotan cannot load has `type = .Invalid`; a nested column has `type = .String` (JSON text). |
| `row_groups` | one `Parquet_Row_Group_Info` per row group: `rows`, `offset` (where its data starts in the file), `compressed_bytes`, and `stats`, with one `Parquet_Column_Stats` per entry of `columns` |

A `Parquet_Column_Stats` holds what the writer recorded for one column in one row group:
- `null_count`, which is -1 if the file doesn't say.
- `min` and `max`, as a `Parquet_Value`: a union of `int`, `f64`, `bool`, `string`, `Date`, `Time`, `Datetime` and `Duration`. Each value has the column's DataFrame type, converted as the cells are: a timestamp becomes a `Datetime` and a decimal a `Float`.

`min` and `max` are `nil` in these cases:
- the writer recorded no statistics;
- a bound is NaN;
- the column is an `INT96` timestamp, a `UUID`, or nested. These are stored in an order that differs from the order of the loaded values.

Writers may shorten long strings in the statistics. The stored `min` is then a prefix of the smallest value, and the stored `max` sorts after the largest value. Both are still valid bounds. pyarrow, DuckDB, Spark and wotan record statistics for every row group.

### 7.4 Loading only the rows you need: filters

`parquet_load(path, columns, alloc, filters)` and `parquet_load_from_bytes` take a list of `Parquet_Filter`s and keep only the rows that match all of them:

```odin
Parquet_Filter :: struct {
	column:       string,
	lower, upper: Parquet_Value, // nil leaves that end open
	inclusive:    w.Inclusive,   // which ends belong to the range: .Both (default), .Neither, .Left, .Right
}
```

A row matches when its cell in `column` lies between `lower` and `upper`. Set `lower = upper` to pick one value. A missing cell or NaN never matches.

The bounds must suit the column's DataFrame type:
- an `Int` or `Float` column takes `int` or `f64` bounds;
- a `Bool`, `String`, `Date`, `Time`, `Datetime` or `Duration` column takes a value of that type. Inside the filter literal a duration needs its type spelled out: `upper = w.Duration(4 * w.DAYS)`. A bound for a column with a time zone is in UTC; `w.datetime_from_local` turns a local time into one.

Strings compare byte by byte, which for plain text is alphabetical order. A filter column does not have to be in `columns`.

```odin
// February only: row groups 0 and 1 end in January and are not read
feb, fok := io.parquet_load(
	"spy.parquet",
	{"Date", "Close"},
	filters = {{column = "Date", lower = w.Date{2023, 2, 1}, upper = w.Date{2023, 2, 28}}},
)
if !fok {return}
defer w.destroy_dataframe(&feb)
w.dataframe_pretty_print(&feb, max_rows = 4)

// two filters: both must hold
high, hok := io.parquet_load(
	"spy.parquet",
	{"Date", "Close", "Volume"},
	filters = {
		{column = "Close", lower = 405},
		{column = "Volume", upper = 70_000_000, inclusive = .Neither},
	},
)
if !hok {return}
defer w.destroy_dataframe(&high)
w.dataframe_pretty_print(&high)

_, bad := io.parquet_load("spy.parquet", filters = {{column = "Date", lower = "2023-02-01"}})
fmt.println("string bound for a Date column:", bad)
```

```
    Date        Close
  0 2023-02-01  402.43
  1 2023-02-02  404.26
  2 2023-02-03  405.58
  3 2023-02-06  406.43
... (6 more rows)
    Date        Close   Volume
  0 2023-02-03  405.58  68765400
  1 2023-02-06  406.43  59876500
  2 2023-02-07  407.85  63456700
  3 2023-02-08  408.12  57654300
  4 2023-02-09  409.58  61234500
  5 2023-02-10  410.43  58765400
  6 2023-02-13  411.26  55432100
  7 2023-02-14  412.85  59876500
parquet: filter on "Date": a bound of type string for a Date column
string bound for a Date column: false
```

The rows are the ones that pyarrow's `read_table(filters=...)` and `pandas.read_parquet(filters=...)` return, in file order.

**Skipped row groups.** Before reading any data, the loader compares each filter with the `min`, `max` and null count of every row group (section 7.3). It does not read a row group that cannot hold a match. In the remaining row groups it checks each row.

The saving therefore depends on how the file is laid out. A history sorted by date, written in row groups of a fixed number of rows, keeps each date range in a few row groups, so a date filter reads only those. A filter on a column whose values are spread over every row group, such as one ticker in a date-sorted file, reads the whole file and then drops rows. Writing with `row_group_rows` of 100 000 to a million rows (section 7.2) keeps the row groups fine enough to skip.

Reading a file of 10 million rows (daily bars of 4000 tickers over ten years, sorted by date, in 101 row groups), best of five runs on a 16-thread machine:

| Read | Rows | wotan | pyarrow 14 `read_table` |
|---|---|---|---|
| the whole file | 10 080 000 | 121 ms | 88 ms |
| one month (`Date` filter) | 84 000 | 3 ms | 9 ms |
| one ticker (`String` filter, no row group skipped) | 2 520 | 152 ms | 92 ms |

`pandas.read_parquet` with the same one-month filter takes 17 ms.

When nothing can be skipped, wotan first loads the row groups and then removes rows, so a filter adds about a quarter to the time of a full load. pyarrow filters while it reads.

**Failure.** `ok` is false, with the reason on stderr, when:
- a filter names a column the file doesn't have, one wotan cannot load, or a nested column (JSON text cannot be filtered);
- a bound has a type that doesn't suit the column (for example a string for a `Date` column);
- a bound is NaN.

## 8. Writing files

All writers produce files that the matching importer reads back to the same DataFrame. `io_roundtrip_test` checks this for every column type, with nulls, integers beyond 2^53, tiny and huge floats, and strings with separators, quotes, line breaks, outer spaces, Unicode, markup and the text `NA`.

| Writer | Output |
|---|---|
| `csv_save(&df, path, sep = ',', allocator) -> bool` | Header plus one line per row. Nulls are empty unquoted fields. A string is quoted when it would otherwise read back differently: empty, with outer spaces, a null token, or containing the separator, a quote or a line break. |
| `csv_string(&df, sep = ',', allocator) -> string` | The text `csv_save` writes, as a string you free. |
| `json_write(&df, path, alloc)` | One array of objects. Nulls and non-finite floats are `null`, and dates are ISO strings. |
| `jsonl_write(&df, path, alloc)` | One object per line, otherwise as `json_write`. |
| `html_write(&df, path, alloc)` | A `<table>` with `<thead>` and `<tbody>`; text is escaped, and nulls are empty cells. |
| `xlsx_save(&df, path, alloc) -> bool` | One worksheet, `Sheet1`. Numbers are exact; dates and times are serial numbers with a date style (local time for a column with a time zone). |
| `parquet_save(&df, path, compression, row_group_rows, dictionary) -> bool` | A Parquet file with the standard type for every column (section 7.2). |
| `parquet_bytes(&df, compression, row_group_rows, dictionary, alloc) -> ([]u8, bool)` | The bytes `parquet_save` writes. |

Floats are written in their shortest exact form and always keep a `.` or an exponent, so a float column stays a float column. The CSV, JSON and HTML writers write a `Datetime` column with a time zone as local time with its UTC offset, as pandas does (`2024-03-10 03:30:00-04:00`); the importers read that back as the same instants (section 2).

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
| `loaders_checked_test` | every `try_` loader gives the frame of its panicking twin, and `ok = false` in 20 failure cases (missing files, empty or non-CSV text, wrong types, bad JSON and JSON Lines, no table, no sheet, not a workbook); header names and numbers as pandas 1.5.3 reads them; a 1.7 MB CSV with quoted line breaks read identically on one thread and on all; the finance loaders `try_load_market_data` and `try_load_option_chain` against pandas 1.5.3 and on files that do not fit |
| `parquet_filter_test` | `parquet_info` against pyarrow 14's metadata (rows, row groups, column types, and null count, min and max per row group for files written by pyarrow and DuckDB; none for UUID, INT96 and nested columns); 16 filters on a 600-row file in 10 row groups, each giving the rows pyarrow's `read_table(filters=...)` gives; a corrupt row group outside the filter's range is never read; `ok = false` for bad filters and unreadable files; the same rows from one thread and from all |
| `parquet_nested_test` | nested columns as JSON text: files from pyarrow 14 (small version 1 pages and row groups; version 2 pages with Zstandard) and DuckDB 1.1, every cell of every nested column equal to the JSON text of pyarrow's value (lists, lists of lists, structs with lists and structs, maps with string and integer keys, lists of structs, decimals, binary, times, timestamps, bools, nulls and empty lists at every level, a 6000-element list across pages, a required struct); one thread and all; `parquet_info`; column selection and a filter beside nested columns; 300 damaged copies |
| `duration_test` | durations against pandas 1.5.3: the text forms both ways; `diff`, `duration_between`, `add_duration(s)`, `datetime_floor`, `duration_total` on a CSV written by pandas; groupby and `resample_every` bars; pyarrow's durations in Parquet; a frame with durations through Parquet, CSV, JSON, HTML and Excel |
| `timezone_test` | time zones against pandas 1.5.3 and Python's zoneinfo: zone lookups and UTC offsets; skipped and repeated wall-clock times; `tz_localize`, `tz_convert`, local parts, floors and resampling on a CSV written by pandas; text with offsets read as zoned columns; pandas' zoned Parquet files; zoned frames through Parquet, CSV, JSON, HTML and Excel; the zone kept by frame operations |
| `parquet_test` | reading: every column type, with nulls and extreme values, against pyarrow in eight layouts (plain, dictionary, delta and byte-stream-split values, page versions 1 and 2, no compression, Snappy, gzip, Zstandard, LZ4, several row groups and many small pages); FLOAT16 and byte stream split on integers (pyarrow 17); Zstandard pages of several blocks; DuckDB's Zstandard file with UUIDs; the Zstandard decoder and the compressor at every level on their own; the same frame from one thread and from all; INT96 timestamps; decimals stored as integers (DuckDB); a pandas file; column selection; `ok = false` for unsupported files (Brotli) and for truncated or corrupted copies. Writing: the same table with every codec, with and without dictionaries, in one or three row groups, read back cell for cell; Zstandard and gzip at their lowest, default and highest levels; the same bytes from one thread and from all; views, an empty frame, a hidden index, 70 000 rows with an all-null column. The fixtures in `demo_files/parquet` were written by pyarrow 14 and 17 and by DuckDB 1.1 |
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
| `io.try_csv_load`, `io.try_csv_load_from_string`, `io.try_json_load`, `io.try_jsonl_load`, `io.try_html_load`, `_table_index`, `_table_id`, `_all`, `_from_string`, `io.try_xlsx_load`, `io.try_xlsx_load_sheet`, `io.try_xlsx_sheet_names` | the same, returning `(df, ok)` (reason on stderr) instead of panicking |
| `io.html_load(path, alloc)`, `_table_index`, `_table_id`, `_all`, `io.html_load_from_string`, `io.html_load_all_from_string` | HTML tables → DataFrame(s) |
| `io.xlsx_load(path, alloc)`, `io.xlsx_load_sheet(path, name, alloc) -> (df, found)`, `io.xlsx_sheet_names(path, alloc)` | Excel worksheets |
| `io.parquet_load(path, columns, alloc, filters) -> (df, ok)`, `io.parquet_load_from_bytes(data, columns, alloc, filters) -> (df, ok)` | Parquet file / bytes → DataFrame, optionally only the rows in value ranges (`Parquet_Filter`) |
| `io.parquet_info(path, alloc) -> (info, ok)`, `io.parquet_info_from_bytes`, `io.print_parquet_info(&info)`, `io.destroy_parquet_info(&info)` | a Parquet file's rows, columns and per-row-group statistics, from its footer |
| `ex.parquet_save(&df, path, compression, row_group_rows, dictionary, compression_level) -> bool`, `ex.parquet_bytes(&df, ...) -> ([]u8, ok)` | DataFrame → Parquet file / bytes |
| `zstd.decompress(dst, src) -> (written, ok)`, `zstd.decompressed_size(src) -> (size, ok)` | Zstandard data → bytes (package `wotan/zstd`) |
| `zstd.compress(src, level, alloc) -> (bytes, ok)`, `zstd.compress_to_dynamic(&buf, src, level) -> ok` | bytes → one Zstandard frame, level 1 to 12 (default 3) |
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
