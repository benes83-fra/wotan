# The `core` package: a user guide

`wotan/core` is the data layer of Wotan. It provides typed columns, the `DataFrame` that holds them, and the operations you need to work with tables: selecting, filtering, handling missing data, cleaning and combining tables, column calculations such as returns and ranks, reshaping, date and text parts, computing new columns, sorting, indexing, grouping, joining and date arithmetic. The rest of Wotan (the importers and exporters, the statistics and finance packages) reads and writes these DataFrames.

This guide walks through the package with small examples. It assumes you know basic Odin. Every example was compiled and run against the current code, and the output shown is the real output. The examples also free everything they allocate: a debug build with a tracking allocator reports no leaks.

## Contents

1. [Setup](#1-setup)
2. [The data model](#2-the-data-model)
3. [Building a DataFrame](#3-building-a-dataframe)
4. [Reading values](#4-reading-values)
5. [Loading data from files](#5-loading-data-from-files)
6. [Printing](#6-printing)
7. [Selecting columns](#7-selecting-columns)
8. [Selecting rows](#8-selecting-rows)
9. [Filtering with masks](#9-filtering-with-masks)
10. [Missing data](#10-missing-data)
11. [Cleaning and combining tables](#11-cleaning-and-combining-tables)
12. [Column calculations](#12-column-calculations)
13. [Reshaping and sampling](#13-reshaping-and-sampling)
14. [Date and text parts](#14-date-and-text-parts)
15. [Computed columns](#15-computed-columns)
16. [Sorting](#16-sorting)
17. [Indexes and label-based access](#17-indexes-and-label-based-access)
18. [Grouping and aggregation](#18-grouping-and-aggregation)
19. [Joins](#19-joins)
20. [Dates and times](#20-dates-and-times)
21. [Memory management](#21-memory-management)
22. [Checked entry points](#22-checked-entry-points)
23. [A complete example](#23-a-complete-example)
24. [Quick reference](#24-quick-reference)

## 1. Setup

Import the package under a short name. This guide uses `w`, as the Wotan sources and tests do:

```odin
import w  "wotan/core"
import io "wotan/importer"   // only needed to read files (section 5)
import "core:fmt"
```

Build with `-debug` while you learn the library. `main.odin` installs a tracking allocator in debug builds, which reports every allocation you forgot to free when the program exits. To get the same check in your own program:

```odin
import "core:mem"

main :: proc() {
    track: mem.Tracking_Allocator
    mem.tracking_allocator_init(&track, context.allocator)
    context.allocator = mem.tracking_allocator(&track)
    defer {
        for _, e in track.allocation_map {fmt.printf("leak: %v bytes at %v\n", e.size, e.location)}
        mem.tracking_allocator_destroy(&track)
    }
    // ... your code ...
}
```

## 2. The data model

A `DataFrame` is an ordered set of named columns of equal length. Each column has a single type:

| `ColumnType` | Odin value type | Example |
|---|---|---|
| `.Int` | `int` | `42` |
| `.Float` | `f64` | `3.14` |
| `.Bool` | `bool` | `true` |
| `.String` | `string` | `"AAPL"` |
| `.Date` | `w.Date{year, month, day}` | `w.Date{2024, 1, 31}` |
| `.Time` | `w.Time{hour, minute, second}` | `w.Time{9, 30, 0}` |
| `.Datetime` | `w.Datetime{year, month, day, hour, minute, second}` | `w.Datetime{2024, 1, 31, 9, 30, 0}` |

Three ideas run through the whole package:

- **Any cell can be null.** Missing values are tracked separately from the data. Every read returns the value and an `is_null` flag.
- **Results are new DataFrames.** Operations do not modify their input. They return a new `DataFrame`, which you must free with `w.destroy_dataframe`. The exceptions modify the frame in place and say so: `add_column`, `set_index` and `reset_index`.
- **Some results are views.** A view reads the data of another DataFrame instead of copying it. Views are cheap, but they must not outlive their source. Section 8 lists which operations return views and section 21 sums up the rules.

## 3. Building a DataFrame

The quickest way to build a DataFrame is from slices, with one `column_from_*` constructor per type and `df_from` to combine them:

```odin
df := w.df_from(
    w.column_from_strings("ticker",  []string{"AAPL", "MSFT", "SAP", "ASML"}),
    w.column_from_strings("country", []string{"US", "US", "DE", "NL"}),
    w.column_from_floats("price",    []f64{189.5, 410.2, 182.0, 905.1}),
    w.column_from_ints("shares",     []int{100, 40, 75, 10}),
)
defer w.destroy_dataframe(&df)

w.dataframe_pretty_print(&df)
fmt.println(df.rows, len(df.columns)) // 4 4
```

```
    ticker  country  price  shares
  0 AAPL    US       189.5  100
  1 MSFT    US       410.2  40
  2 SAP     DE       182    75
  3 ASML    NL       905.1  10
```

The constructors are `column_from_ints`, `column_from_floats`, `column_from_bools`, `column_from_strings`, `column_from_dates`, `column_from_times` and `column_from_datetimes`. The DataFrame takes ownership of the columns you pass to `df_from`, so you free only the DataFrame.

To build a column value by value, for example while parsing, create it with `column_new` and append to it. `append_null` adds a missing value. `add_column` attaches the column to an existing DataFrame; its length must match `df.rows`:

```odin
vol := w.column_new("volume", .Float, 4) // the capacity is a hint; columns grow as needed
w.append_float(&vol, 1.2e6)
w.append_float(&vol, 9.8e5)
w.append_null(&vol) // missing value
w.append_float(&vol, 4.1e5)
w.add_column(&df, vol) // df now owns vol
```

There is one `append_*` per type: `append_int`, `append_float`, `append_bool`, `append_string`, `append_date`, `append_time` and `append_datetime`. Appending a value of the wrong type panics. `append_string` copies the string into the column, so the source string can be freed afterwards.

Column **names** are not copied. A string literal is fine. If you build a name at run time, it must stay alive as long as the DataFrame does.

## 4. Reading values

`w.column(&df, name)` returns a `^Column`. It panics if the name does not exist; `try_column` returns `ok = false` instead (section 22). The typed readers `column_at_int`, `column_at_float`, `column_at_bool`, `column_at_string`, `column_at_date`, `column_at_time` and `column_at_datetime` return the value and whether it is null:

```odin
price := w.column(&df, "price")
p, null := w.column_at_float(price, 2)
fmt.println(p, null) // 182 false

v, missing := w.column_at_float(w.column(&df, "volume"), 2)
fmt.println(v, missing) // 0 true

// any numeric column as f64; nulls (and NaN) come back as is_null = true
sh, _ := w.column_at_numeric(w.column(&df, "shares"), 0)
fmt.println(sh) // 100

name, _ := w.column_at_string(w.column(&df, "ticker"), 3)
fmt.println(name, w.is_null(w.column(&df, "volume"), 2)) // ASML true
```

A null cell reads as the zero value of its type, so always check the flag before you use the value.

Use the reader that matches the column's type. In particular, `column_at_float` on an `Int` column reinterprets the bits and gives garbage. When you do not know whether a column is `Int` or `Float` (CSV type inference decides that for you), use `column_at_numeric`, which accepts `Int`, `Float` and `Bool`.

To loop over rows, look the columns up once and index them:

```odin
total := 0.0
shares := w.column(&df, "shares")
for i in 0 ..< df.rows {
    px, px_null := w.column_at_float(price, i)
    n, n_null := w.column_at_int(shares, i)
    if px_null || n_null {continue}
    total += px * f64(n)
}
fmt.printf("position value: %.1f\n", total) // 58059.0
```

For lower-level access, `df_series` returns a `Series`, a typed window onto a column's storage that does not copy anything:

```odin
s := w.df_series(&df, "price") // no copy: points into the column
x, _ := w.series_at_float(&s, 1)
fmt.println(s.len, x) // 4 410.2
```

## 5. Loading data from files

Reading files is the job of the `importer` package, which returns `core` DataFrames. It reads CSV, JSON, JSONL, HTML tables and Excel files. Column types are inferred from the data: integers become `Int`, decimals `Float`, `true`/`false` `Bool`, and ISO dates and times become `Date`, `Time` or `Datetime`. Empty fields and the tokens `NA`, `null` and `None` become nulls.

```odin
text := `date,ticker,close,volume
2024-01-02,AAPL,185.64,82488700
2024-01-02,MSFT,370.87,25258600
2024-01-03,AAPL,184.25,
2024-01-03,MSFT,370.60,23083500
`
df := io.csv_load_from_string(text)
defer w.destroy_dataframe(&df)

w.dataframe_print(&df) // column names and inferred types
v, null := w.column_at_int(w.column(&df, "volume"), 2)
fmt.println(v, null) // 0 true (the empty field is null)
```

```
DataFrame:
  rows: 4
  columns: 4
    date (Date)
    ticker (String)
    close (Float)
    volume (Int)
```

For files on disk use `io.csv_load("path/to/file.csv")`. You can pass your own column types and separator: `io.csv_load(path, types = ..., sep = ';')`. For the other formats see `io.xlsx_load`, `io.xlsx_load_sheet` and the JSON and HTML loaders. The `exporter` package writes DataFrames back out, for example with `exporter.csv_save`. The [data in and out guide](data_io_guide.md) covers all formats, HTTP and Yahoo Finance.

## 6. Printing

| Procedure | Prints |
|---|---|
| `w.dataframe_pretty_print(&df, max_rows = 20, precision = -1)` | An aligned table with row labels; nulls show as `NULL`. Float cells show the shortest text that reads back exactly; `precision = 4` shows four decimals instead |
| `w.df_head(&df, n)` | The first `n` rows, tab-separated |
| `w.dataframe_print(&df)` | The shape and each column's name and type |

The numbers in the left margin of `dataframe_pretty_print` are row labels. A selection keeps the labels of the rows it came from (see the `iloc_many` example in section 8), so you can trace rows back to the source table.

## 7. Selecting columns

```odin
view := w.select(&df, []string{"ticker", "price"}) // view: reads df's data
defer w.destroy_dataframe(&view)

owned := w.dataframe_select_columns(&df, []string{"ticker", "price"}, true) // copy
defer w.destroy_dataframe(&owned)
```

`select` with a list of names returns a view. `dataframe_select_columns(..., copy = true)` returns an independent copy. The same `select` procedure also accepts expressions that compute new columns (section 15).

To remove columns, `drop_columns` returns a copy without them. `rename_column` renames a column in place. It returns `false` if the old name does not exist, the new name is already taken, or the frame is a view:

```odin
slim, ok := w.drop_columns(&df, {"shares"}) // copy without "shares"
defer w.destroy_dataframe(&slim)

renamed := w.rename_column(&slim, "price", "close") // true
```

## 8. Selecting rows

The examples in sections 8 and 9, and in the sections on computed columns, sorting and grouping, use this table:

```odin
df := w.df_from(
    w.column_from_strings("ticker",  []string{"AAPL", "MSFT", "SAP", "ASML", "NVDA", "SIE"}),
    w.column_from_strings("country", []string{"US", "US", "DE", "NL", "US", "DE"}),
    w.column_from_floats("price",    []f64{189.5, 410.2, 182.0, 905.1, 875.3, 172.4}),
    w.column_from_ints("shares",     []int{100, 40, 75, 10, 20, 60}),
)
defer w.destroy_dataframe(&df)
```

Rows by position:

```odin
first3 := w.iloc_range(&df, 0, 3) // rows 0, 1, 2 as a view
defer w.destroy_dataframe(&first3)

picked := w.iloc_many(&df, []int{4, 0}) // rows 4 and 0, copied, in that order
defer w.destroy_dataframe(&picked)

row := w.df_row(&df, 2) // a single row as a one-row DataFrame (copy)
defer w.destroy_dataframe(&row)

w.dataframe_pretty_print(&picked)
```

```
    ticker  country  price  shares
  4 NVDA    US       875.3  20
  0 AAPL    US       189.5  100
```

`w.iloc(&df, i)` and `w.iloc(&df, i0, i1)` are shorthands for `iloc_single` and `iloc_range`. Ranges are half-open: `[i0, i1)`.

`head(&df, n)` and `tail(&df, n)` return the first or last `n` rows as a view. A negative `n` counts from the other end: `tail(&df, -2)` is every row but the first two. `df_head` prints rows; `head` and `tail` instead return a DataFrame, which you can print, pass on or materialize.

Whether you get a view or a copy depends only on the kind of selection:

| Returns a **view** | Returns a **copy** |
|---|---|
| `iloc_range`, `head`, `tail`, `df_slice`, `loc_range`, `loc_from`, `loc_until`, `select` (by names), `dataframe_slice_rows(..., copy = false)` | `iloc_single`, `iloc_many`, `df_row`, `loc_single`, `loc_many`, `loc_mask`, `filter`, `wobei`, sorting, `select` (expressions), `agg`, joins, `materialize` |

Contiguous ranges are views because they can be read without copying. Selections that pick rows out of order have to copy them. To turn a view into an independent DataFrame, call `w.materialize(&view)`.

## 9. Filtering with masks

A mask is a `[]bool` with one entry per row. The `mask_gt`, `mask_lt` and `mask_eq` procedures compare a column with a value. They work on `Int`, `Float`, `Date`, `Time` and `Datetime` columns, and `mask_eq` also works on strings. `mask_contains_string` tests for a substring. Null cells always give `false`. Combine masks with `and`, `or` and `not`, then pass the result to `filter`:

```odin
expensive := w.mask_gt(w.column(&df, "price"), 200.0) // []bool, one per row
defer delete(expensive)
us := w.mask_eq(w.column(&df, "country"), "US")
defer delete(us)

both := w.and(expensive, us)
defer delete(both)

us_expensive := w.filter(&df, both) // owned copy of the matching rows
defer w.destroy_dataframe(&us_expensive)
w.dataframe_pretty_print(&us_expensive) // MSFT, NVDA
```

```
    ticker  country  price  shares
  0 MSFT    US       410.2  40
  1 NVDA    US       875.3  20
```

Every mask is a new allocation, including those returned by `and`, `or` and `not`, so free each one with `delete`. The comparison value must match the column's type: `200.0` for a `Float` column, `50` for an `Int` column, a `w.Date{...}` for a `Date` column.

```odin
not_us := w.not(us)
defer delete(not_us)
either := w.or(expensive, not_us)
defer delete(either)
fmt.println(either) // [false, true, true, true, true, true]
```

If the condition is already stored in a `Bool` column, filter on it directly. `wobei` ("where" in German) accepts a mask, a `^Column` or a column name:

```odin
flags := w.df_from(
    w.column_from_strings("id", []string{"a", "b", "c"}),
    w.column_from_bools("active", []bool{true, false, true}),
)
defer w.destroy_dataframe(&flags)

active := w.wobei(&flags, "active") // rows where the Bool column is true
defer w.destroy_dataframe(&active)
fmt.println(active.rows) // 2
```

The `and`, `or` and `not` overloads also accept `Bool` columns, with nulls counted as `false`.

Two more procedures build masks from a column. `isin` tests membership in a list of values. `between` tests a range: both ends are included by default, and `.Neither`, `.Left` or `.Right` changes that. Both work on numbers, strings and dates, and give `false` for missing cells. They return `(mask, ok)`, with `ok = false` when the values do not match the column's type:

```odin
eu, ok := w.isin(w.column(&df, "country"), []string{"DE", "NL"})
defer delete(eu)
mid, _ := w.between(w.column(&df, "price"), 180.0, 500.0)
defer delete(mid)

both := w.and(eu, mid)
defer delete(both)
picked := w.filter(&df, both)
defer w.destroy_dataframe(&picked)
w.dataframe_pretty_print(&picked) // SAP
```

```
    ticker  country  price  shares
  0 SAP     DE       182    75
```

A NaN among the `isin` values matches missing cells, as in pandas.

`compare(col, op, value)` is the checked form of `mask_gt`, `mask_lt` and `mask_eq` (section 22), and `str_contains`, `str_startswith` and `str_endswith` (section 14) test text.

## 10. Missing data

A cell is missing when it is null, or when it is NaN in a `Float` column. The procedures in this section treat both kinds the same way, as pandas does with `None` and `NaN`. The example below uses this table of daily quotes, with gaps in `close` and `volume`:

```odin
close := w.column_new("close", .Float, 6)
w.append_float(&close, 101.5)
w.append_null(&close)                   // no quote
w.append_float(&close, math.nan_f64())  // NaN counts as missing too
w.append_float(&close, 103.0)
w.append_null(&close)
w.append_float(&close, 104.5)

volume := w.column_new("volume", .Int, 6)
w.append_int(&volume, 900)
w.append_int(&volume, 1200)
w.append_null(&volume)
w.append_int(&volume, 1100)
w.append_null(&volume)
w.append_null(&volume)

df := w.df_from(w.column_from_strings("day", []string{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat"}), close, volume)
defer w.destroy_dataframe(&df)
```

**Finding gaps.** `count_na` on a DataFrame returns a small table with one row per column. On a single column it returns the count. `isna` and `notna` return a mask (a `[]bool`) that works with everything in section 9. `na_at(col, row)` tests a single cell.

```odin
counts := w.count_na(&df)
defer w.destroy_dataframe(&counts)
w.dataframe_pretty_print(&counts)
fmt.println(w.count_na(w.column(&df, "close"))) // 3

gaps := w.isna(w.column(&df, "close"))
defer delete(gaps)
fmt.println(gaps) // [false, true, true, false, true, false]
```

```
    column  na
  0 day     0
  1 close   3
  2 volume  3
```

**Dropping rows.** `dropna` returns the rows that have no missing cell. Pass column names to check only those columns. Pass `how = .All` to drop only the rows whose checked cells are all missing. Pass `thresh = n` to keep the rows that have at least `n` present cells; it replaces `how`. The rows keep their labels:

```odin
clean, ok := w.dropna(&df)
defer w.destroy_dataframe(&clean)
w.dataframe_pretty_print(&clean)

priced, _ := w.dropna(&df, {"close"})
defer w.destroy_dataframe(&priced)
w.dataframe_pretty_print(&priced)
```

```
    day  close  volume
  0 Mon  101.5  900
  3 Thu  103    1100
    day  close  volume
  0 Mon  101.5  900
  3 Thu  103    1100
  5 Sat  104.5  NULL
```

**Filling with a constant.** `fillna` replaces missing cells with a value. On a DataFrame it fills the named columns, or, with no names, every column whose type takes the value. The value must match the column type: `0.0` for `Float`, `"?"` for `String`, a `w.Date{...}` for `Date`. An `int` value also fills `Float` columns. `fillna` also takes a single `^Column` and then returns a new column.

```odin
zeros, _ := w.fillna(&df, 0, {"volume"})
defer w.destroy_dataframe(&zeros)
w.dataframe_pretty_print(&zeros)
```

```
    day  close  volume
  0 Mon  101.5  900
  1 Tue  NULL   1200
  2 Wed  NaN    0
  3 Thu  103    1100
  4 Fri  NULL   0
  5 Sat  104.5  0
```

**Carrying values.** `ffill` fills each gap with the last present value above it, which is the usual way to bring a price series onto a common calendar. `bfill` uses the next present value below the gap. Cells before the first value (for `ffill`) or after the last value (for `bfill`) stay missing. With `limit = n`, at most `n` cells of each gap are filled. Both take an optional list of column names, and both also work on a single `^Column`. In a long table with several tickers, pass `by` (key column names for a frame, key columns for a single column) so that a value is carried only within its own ticker; section 12 shows an example.

```odin
carried, _ := w.ffill(&df, {"close"})
defer w.destroy_dataframe(&carried)
w.dataframe_pretty_print(&carried)

one, _ := w.ffill(&df, limit = 1)
defer w.destroy_dataframe(&one)
w.dataframe_pretty_print(&one)
```

```
    day  close  volume
  0 Mon  101.5  900
  1 Tue  101.5  1200
  2 Wed  101.5  NULL
  3 Thu  103    1100
  4 Fri  103    NULL
  5 Sat  104.5  NULL
    day  close  volume
  0 Mon  101.5  900
  1 Tue  101.5  1200
  2 Wed  NaN    1200
  3 Thu  103    1100
  4 Fri  103    1100
  5 Sat  104.5  NULL
```

**Interpolating.** `interpolate` fills each gap between two present values of an `Int` or `Float` column linearly and returns a `Float` column. By default the rows count as equally spaced. Pass a second column (`Int`, `Float`, `Date` or `Datetime`) to space them by its values instead, so that uneven dates are weighted by their distance. Cells before the first value and after the last stay missing (call `ffill` afterwards to carry the last value on). `limit` works as for `ffill`.

```odin
days := w.column_from_dates("day", {{2024, 1, 1}, {2024, 1, 2}, {2024, 1, 5}, {2024, 1, 7}})
defer w.destroy_column(&days)
rate := w.column_new("rate", .Float, 4)
w.append_float(&rate, 1.0)
w.append_null(&rate)
w.append_null(&rate)
w.append_float(&rate, 4.0)
defer w.destroy_column(&rate)

by_row, _ := w.interpolate(&rate)        // 1, 2, 3, 4
defer w.destroy_column(&by_row)
by_day, _ := w.interpolate(&rate, &days) // 1, 1.5, 3, 4: Jan 2 is 1 of 6 days along
defer w.destroy_column(&by_day)
```

None of these procedures changes its input. Each result is a new owned DataFrame or column; free it with `destroy_dataframe` or `destroy_column`, and free masks with `delete`. A result keeps the index and row labels of its source, so `loc` still works on it. The input may also be a view, such as a slice from `iloc_range`.

The DataFrame and column procedures return `(result, ok)`. `ok` is `false` for an unknown column name, a fill value of the wrong type for a named column, or a negative `limit`:

```odin
_, bad := w.fillna(&df, "n/a", {"close"}) // a String value for a Float column
fmt.println(ok, bad)                      // true false
```

## 11. Cleaning and combining tables

This section uses two monthly trade lists, read as text the way a messy CSV often arrives:

```odin
jan := w.df_from(
    w.column_from_strings("ticker", []string{"AAPL", "MSFT", "AAPL"}),
    w.column_from_strings("price", []string{"189.5", "410.2", "189.5"}),
    w.column_from_ints("qty", []int{10, 5, 10}),
)
defer w.destroy_dataframe(&jan)
feb := w.df_from(
    w.column_from_strings("ticker", []string{"NVDA", "MSFT"}),
    w.column_from_strings("price", []string{"875.3", "n/a"}),
    w.column_from_ints("qty", []int{2, 7}),
)
defer w.destroy_dataframe(&feb)
```

**Stacking frames.** `concat_rows` puts the rows of several frames one after another. Every frame needs the same column names with the same types, in any order. The result uses the column order of the first frame and is numbered from 0. If all frames share the same index, the result keeps it.

```odin
trades, ok := w.concat_rows({&jan, &feb})
defer w.destroy_dataframe(&trades)
w.dataframe_pretty_print(&trades)
```

```
    ticker  price  qty
  0 AAPL    189.5  10
  1 MSFT    410.2  5
  2 AAPL    189.5  10
  3 NVDA    875.3  2
  4 MSFT    n/a    7
```

**Changing a column's type.** `cast_column` converts one column. `cast_columns` converts the named columns of a frame and copies the rest. Text is parsed after trimming spaces, and blank text becomes null. By default the cast is strict: a single cell that cannot be converted (here `"n/a"`) makes it fail with `ok = false`. With `strict = false`, such cells become null instead:

```odin
_, strict_ok := w.cast_columns(&trades, {"price"}, .Float)
fmt.println(strict_ok) // false: "n/a" is not a number

typed, _ := w.cast_columns(&trades, {"price"}, .Float, strict = false)
defer w.destroy_dataframe(&typed)
w.dataframe_pretty_print(&typed)
```

```
    ticker  price  qty
  0 AAPL    189.5  10
  1 MSFT    410.2  5
  2 AAPL    189.5  10
  3 NVDA    875.3  2
  4 MSFT    NULL   7
```

| From | To | Rule |
|---|---|---|
| `Int`, `Float`, `Bool` | each other | `Float` to `Int` truncates toward zero, and fails for infinite values or values beyond the `int` range. Non-zero becomes `true`; `true` becomes 1 |
| any type | `String` | The text of `value_as_string`, which the importers read back without loss |
| `String` | any type | Integers, floats, `true` / `false` / `1` / `0` in any case, `YYYY-MM-DD`, `HH:MM:SS`, `YYYY-MM-DD HH:MM:SS` (also with a `T`; a date alone means midnight) |
| `Date` | `Datetime` | Midnight of that day |
| `Datetime` | `Date`, `Time` | The date, or the time of day |

Missing cells stay missing. Other pairs, such as `Date` to `Int`, return `ok = false`.

**Repeated rows.** `duplicated` returns a mask that marks the rows repeating an earlier row, and `drop_duplicates` returns the rows it does not mark. Both compare all columns, or only the ones you name. `keep = .Last` keeps the last occurrence instead of the first, and `keep = .None` marks every row that occurs more than once. Two missing cells count as equal, and so do `-0.0` and `0.0`:

```odin
dups, _ := w.duplicated(&typed)
defer delete(dups)
fmt.println(dups) // [false, false, true, false, false]

clean, _ := w.drop_duplicates(&typed)
defer w.destroy_dataframe(&clean)

latest, _ := w.drop_duplicates(&typed, {"ticker"}, .Last) // last trade per ticker
defer w.destroy_dataframe(&latest)
w.dataframe_pretty_print(&latest)
```

```
    ticker  price  qty
  2 AAPL    189.5  10
  3 NVDA    875.3  2
  4 MSFT    NULL   7
```

**Distinct values.** `unique` returns a new column with the distinct values in order of first appearance, and `nunique` counts them. `value_counts` returns a frame with each value and how often it occurs, most frequent first; equal counts keep the order of first appearance. `nunique` and `value_counts` ignore missing cells unless you pass `dropna = false`. `unique` keeps one missing entry if the column has any.

```odin
tickers := w.unique(w.column(&typed, "ticker")) // AAPL, MSFT, NVDA
defer w.destroy_column(&tickers)
fmt.println(w.nunique(w.column(&typed, "ticker"))) // 3

counts := w.value_counts(w.column(&typed, "ticker"))
defer w.destroy_dataframe(&counts)
w.dataframe_pretty_print(&counts)
```

```
    ticker  count
  0 AAPL    2
  1 MSFT    2
  2 NVDA    1
```

Every result in this section is a new owned frame, column or mask; free it as usual. The inputs may be views and are never changed. Frame results keep the row labels and the index of their source. The procedures that take column names return `ok = false` for an unknown name.

## 12. Column calculations

These procedures compute a new column from an existing one, in the manner of pandas Series methods. Each takes an optional `name` for the result; without it, the result is named like its source. A name lets you add the result to the same frame with `add_column`. A missing cell in the input gives a missing (null) cell in the result, and the input is never changed.

**Lags and changes.** `shift(col, n)` moves the values down by `n` rows (up for a negative `n`) and works on every column type. `diff(col, n)` is the difference to the value `n` rows earlier; it stays `Int` for an `Int` column. `pct_change(col, n)` is the relative change `x[i] / x[i-n] - 1` as a `Float` column. It does not fill gaps first, so a gap gives a missing change; call `ffill` beforehand if you want to carry prices over gaps.

**Running values.** `cumsum` and `cumprod` work on `Int` and `Float` columns. `cummax` and `cummin` work on every type that can be ordered, including strings and dates. A missing cell is missing in the result, and the running value carries on past it. With `skipna = false`, every row from the first missing cell on is missing instead.

```odin
df := w.df_from(
    w.column_from_strings("day", []string{"Mon", "Tue", "Wed", "Thu", "Fri"}),
    w.column_from_floats("close", []f64{100.0, 104.0, 98.8, 101.764, 96.0}),
)
defer w.destroy_dataframe(&df)
close := w.column(&df, "close")

ret, _ := w.pct_change(close, name = "ret")
w.add_column(&df, ret) // the frame owns the new columns
prev, _ := w.shift(close, 1, name = "prev")
w.add_column(&df, prev)
peak, _ := w.cummax(close, name = "peak") // running high, e.g. for drawdowns
w.add_column(&df, peak)
move, _ := w.diff(close, name = "move")
w.add_column(&df, move)
w.dataframe_pretty_print(&df, precision = 4) // four decimals
```

```
    day  close     ret      prev      peak      move
  0 Mon  100.0000  NULL     NULL      100.0000  NULL
  1 Tue  104.0000  0.0400   100.0000  104.0000  4.0000
  2 Wed  98.8000   -0.0500  104.0000  104.0000  -5.2000
  3 Thu  101.7640  0.0300   98.8000   104.0000  2.9640
  4 Fri  96.0000   -0.0566  101.7640  104.0000  -5.7640
```

Without `precision`, floats print in full, for example `0.040000000000000036` for the first return: the shortest text that reads back as exactly the stored value.

These procedures return `(column, ok)`. `ok` is `false` for a column type they do not support (`shift`, `cummax`, `cummin` and `rank` take every type) or for group keys of the wrong length.

**Per group.** In a long table with one row per date and ticker, a plain `pct_change` would compute the first return of each ticker from the previous ticker's last price. `shift`, `diff`, `pct_change`, the four running procedures, `rank`, `ffill`, `bfill` and `interpolate` therefore take an optional `by`: one or more key columns that split the rows into groups. Each group is then computed on its own, in row order, like pandas' `groupby(...).pct_change()`. Rows with a missing key get a missing result. With the date as key, `rank` gives cross-sectional ranks:

```odin
px := w.df_from(
    w.column_from_strings("date", {"01-02", "01-02", "01-03", "01-03", "01-04", "01-04"}),
    w.column_from_strings("ticker", {"AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"}),
    w.column_from_floats("close", {185.0, 410.0, 187.0, math.nan_f64(), 183.0, 418.2}),
)
defer w.destroy_dataframe(&px)
tickers := []^w.Column{w.column(&px, "ticker")}

filled, _ := w.ffill(w.column(&px, "close"), by = tickers) // carry MSFT's 410 over its gap
defer w.destroy_column(&filled)
ret, _ := w.pct_change(&filled, by = tickers, name = "ret") // returns per ticker
w.add_column(&px, ret)
r, _ := w.rank(w.column(&px, "ret"), ascending = false, name = "rank", by = {w.column(&px, "date")})
w.add_column(&px, r) // 1 = best return of the day
w.dataframe_pretty_print(&px, precision = 4)
```

```
    date   ticker  close     ret      rank
  0 01-02  AAPL    185.0000  NULL     NULL
  1 01-02  MSFT    410.0000  NULL     NULL
  2 01-03  AAPL    187.0000  0.0108   1.0000
  3 01-03  MSFT    NaN       0.0000   2.0000
  4 01-04  AAPL    183.0000  -0.0214  2.0000
  5 01-04  MSFT    418.2000  0.0200   1.0000
```

**Bounding values.** `clip(col, lower, upper)` raises values below `lower` and lowers values above `upper`. Leave out a bound, or pass NaN, to skip it. An `Int` column stays `Int`, so its bounds must be whole numbers:

```odin
capped, _ := w.clip(w.column(&df, "ret"), lower = -0.03, upper = 0.03)
defer w.destroy_column(&capped) // NULL, 0.03, -0.03, 0.03, -0.03
```

**Ranking.** `rank` numbers the values from 1 (the smallest, or the largest with `ascending = false`) and returns a `Float` column. Ties get the mean of their positions by default. Pass `method = .Min`, `.Max`, `.First` (order of appearance) or `.Dense` (no gaps after ties) for the other conventions.

**Top and bottom rows.** `nlargest(&df, n, column)` and `nsmallest` return the `n` rows with the largest or smallest values, in that order. Ties keep their order of appearance, missing values are skipped, and the rows keep their labels:

```odin
r, _ := w.rank(w.column(&df, "ret"), ascending = false, name = "ret_rank")
w.add_column(&df, r)

best, _ := w.nlargest(&df, 2, "ret")
defer w.destroy_dataframe(&best)
w.dataframe_pretty_print(&best, precision = 4)
```

```
    day  close     ret     prev      peak      move    ret_rank
  1 Tue  104.0000  0.0400  100.0000  104.0000  4.0000  1.0000
  3 Thu  101.7640  0.0300  98.8000   104.0000  2.9640  2.0000
```

**Summary statistics.** `describe` summarizes every `Int` and `Float` column of a frame, as pandas `describe()` does. The result has a `stat` column with the labels count, mean, std (the sample standard deviation), min, 25%, 50%, 75% and max, and one `Float` column per numeric column. Missing cells are left out of every statistic:

```odin
prices := w.select(&df, []string{"close"})
defer w.destroy_dataframe(&prices)
stats := w.describe(&prices)
defer w.destroy_dataframe(&stats)
w.dataframe_pretty_print(&stats, precision = 4)
```

```
    stat   close
  0 count  5.0000
  1 mean   100.1128
  2 std    3.0203
  3 min    96.0000
  4 25%    98.8000
  5 50%    100.0000
  6 75%    101.7640
  7 max    104.0000
```

For moving windows (rolling means, rolling volatility, exponentially weighted statistics), see [Rolling statistics](analytics_guide.md#3-rolling-statistics) in the `analytics` guide.

## 13. Reshaping and sampling

**Long to wide.** Data often arrives in long form, with one row per date and ticker. `pivot(&df, index, columns, values)` turns it into wide form: one row per distinct `index` key, and one column per distinct `columns` key, named by the key's text. Both sets of keys are sorted ascending. A pair that does not occur gives null, and rows with a missing key are left out.

```odin
long := w.df_from(
    w.column_from_strings("date", {"2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02", "2024-01-04"}),
    w.column_from_strings("ticker", {"MSFT", "MSFT", "AAPL", "AAPL", "AAPL"}),
    w.column_from_floats("close", {411.0, 410.0, 186.25, 185.5, 187.0}),
)
defer w.destroy_dataframe(&long)

wide, ok := w.pivot(&long, "date", "ticker", "close")
defer w.destroy_dataframe(&wide)
w.dataframe_pretty_print(&wide)
```

```
    date        AAPL    MSFT
  0 2024-01-02  185.5   410
  1 2024-01-03  186.25  411
  2 2024-01-04  187     NULL
```

`pivot` returns `ok = false` when a pair occurs twice. To combine repeated pairs, use `pivot_table` with an aggregation: `.Mean` (the default), `.Sum`, `.Count`, `.Min`, `.Max`, `.First` or `.Last`. Missing values are left out before aggregating.

```odin
trades := w.df_from(
    w.column_from_ints("day", {2, 1, 2, 1, 2}),
    w.column_from_strings("side", {"buy", "sell", "buy", "buy", "sell"}),
    w.column_from_ints("qty", {5, 3, 7, 4, 2}),
)
defer w.destroy_dataframe(&trades)

_, unique_pairs := w.pivot(&trades, "day", "side", "qty") // false: (2, buy) occurs twice
volume, _ := w.pivot_table(&trades, "day", "side", "qty", .Sum)
defer w.destroy_dataframe(&volume)
w.dataframe_pretty_print(&volume)
```

```
    day  buy  sell
  0 1    4    3
  1 2    12   2
```

**Wide to long.** `melt` goes the other way. You name the id columns that identify a row, and optionally the value columns (by default, every other column). For each value column and each row, you get one row with the id columns, the value column's name and its value. The value columns must share one type; `Int` and `Float` mix into `Float`.

```odin
back, _ := w.melt(&wide, {"date"}, var_name = "ticker", value_name = "close")
defer w.destroy_dataframe(&back)
w.dataframe_pretty_print(&back)
```

```
    date        ticker  close
  0 2024-01-02  AAPL    185.5
  1 2024-01-03  AAPL    186.25
  2 2024-01-04  AAPL    187
  3 2024-01-02  MSFT    410
  4 2024-01-03  MSFT    411
  5 2024-01-04  MSFT    NULL
```

**Random rows.** `sample(&df, n)` draws `n` rows at random without replacement, in the order drawn; pass `replace = true` to draw with replacement. The rows keep their labels. It uses `context.random_generator`, so seeding the generator (for example with `rand.reset(seed)` from `core:math/rand`) makes the draw repeatable:

```odin
rand.reset(7)
some, _ := w.sample(&long, 3)
defer w.destroy_dataframe(&some)
w.dataframe_pretty_print(&some)
```

```
    date        ticker  close
  1 2024-01-02  MSFT    410
  0 2024-01-03  MSFT    411
  3 2024-01-02  AAPL    185.5
```

**Calendar periods.** `resample` turns rows with a `Date` or `Datetime` column into one row per day, week, month, quarter or year, as pandas' `resample` does. Each `Resample_Spec` names a column, an aggregation (the same seven as `pivot_table`) and an output name. `.First` and `.Last` are the earliest and latest values of a period by date, so `.First`, `.Max`, `.Min` and `.Last` of a price give open, high, low and close bars, even if the rows are not sorted. Each period is labelled by its last day. Weeks end on Sunday unless you pass `week_end`:

```odin
daily := w.df_from(
    w.column_from_dates("date", {{2024, 3, 4}, {2024, 3, 5}, {2024, 3, 7}, {2024, 3, 8}, {2024, 3, 11}, {2024, 3, 13}, {2024, 3, 15}}),
    w.column_from_floats("close", {101.0, 103.5, 102.0, 104.25, 103.0, 106.5, 105.0}),
    w.column_from_ints("volume", {1200, 900, 1500, 1100, 800, 2000, 1300}),
)
defer w.destroy_dataframe(&daily)

weekly, ok := w.resample(&daily, "date", .Week, {
    {"close", .First, "open"},
    {"close", .Max, "high"},
    {"close", .Min, "low"},
    {"close", .Last, "close"},
    {"volume", .Sum, ""}, // "" keeps the column's name
}, week_end = .Friday)
defer w.destroy_dataframe(&weekly)
w.dataframe_pretty_print(&weekly)
```

```
    date        open  high    low  close   volume
  0 2024-03-08  101   104.25  101  104.25  4700
  1 2024-03-15  103   106.5   103  105     4100
```

Every period from the first to the last one in the data appears, so a month without rows shows up with a `.Sum` and `.Count` of 0 and nulls otherwise. With `by = {"ticker"}`, each ticker gets its own run of periods. The result starts with the key columns and the period labels.

`melt`, `pivot`, `pivot_table`, `resample` and `sample` return `ok = false` for an unknown column name or an impossible request, such as a sample larger than the table without replacement.

## 14. Date and text parts

`date_part` extracts one part of every cell of a `Date`, `Datetime` or `Time` column as a new column, as pandas' `.dt` accessors do. The numeric parts give an `Int` column: `.Year`, `.Quarter`, `.Month`, `.Day`, `.Weekday` (0 = Monday), `.Day_Of_Year`, `.Week` (the ISO 8601 week), `.Hour`, `.Minute` and `.Second`. The flags give a `Bool` column: `.Is_Month_Start`, `.Is_Month_End`, `.Is_Quarter_Start`, `.Is_Quarter_End`, `.Is_Year_Start`, `.Is_Year_End` and `.Is_Leap_Year`. A `Date` column has no time parts, and a `Time` column has only the time parts.

The `str_` procedures work on `String` columns, as pandas' `.str` accessors do:

| Procedure | Result |
|---|---|
| `str_upper`, `str_lower` | Upper or lower case, letter by letter (Unicode letters included; unlike Python, `"ß"` stays `"ß"`) |
| `str_strip(col, chars = "")` | Without leading and trailing whitespace, or without the characters in `chars` |
| `str_replace(col, old, new)` | Every `old` replaced by `new` (plain text, not a regular expression) |
| `str_slice(col, start, stop)` | Characters `start ..< stop`, with Python's rules: negative positions count from the end |
| `str_split_part(col, sep, i)` | Part `i` after splitting at `sep` (negative `i` counts from the end; an empty `sep` splits at whitespace); null if there are too few parts |
| `str_len` | The number of characters, as an `Int` column |
| `str_contains`, `str_startswith`, `str_endswith` | A mask; pass `case_sensitive = false` to ignore case |

Positions and lengths count characters (Unicode code points), not bytes. Like the other column procedures, these return `(result, ok)`, take an optional `name`, and keep missing cells missing (`false` in masks):

```odin
df := w.df_from(
    w.column_from_dates("date", {{2024, 3, 28}, {2024, 3, 29}, {2024, 4, 1}, {2024, 6, 28}}),
    w.column_from_strings("name", {"  apple inc ", "Microsoft Corp", "NVIDIA Corp ", "Alphabet Inc"}),
    w.column_from_strings("isin", {"US0378331005", "US5949181045", "US67066G1040", "US02079K3059"}),
)
defer w.destroy_dataframe(&df)
d := w.column(&df, "date")

wd, ok := w.date_part(d, .Weekday, name = "weekday")
w.add_column(&df, wd)
q, _ := w.date_part(d, .Quarter, name = "quarter")
w.add_column(&df, q)

clean, _ := w.str_strip(w.column(&df, "name")) // a helper column, not added
defer w.destroy_column(&clean)
upper, _ := w.str_upper(&clean, name = "name_uc")
w.add_column(&df, upper)
country, _ := w.str_slice(w.column(&df, "isin"), 0, 2, name = "country")
w.add_column(&df, country)
first, _ := w.str_split_part(&clean, " ", 0, name = "first_word")
w.add_column(&df, first)
w.dataframe_pretty_print(&df)

corp, _ := w.str_endswith(w.column(&df, "name_uc"), "CORP")
defer delete(corp)
fmt.println(corp) // [false, true, true, false]
```

```
    date        name            isin          weekday  quarter  name_uc         country  first_word
  0 2024-03-28    apple inc     US0378331005  3        1        APPLE INC       US       apple
  1 2024-03-29  Microsoft Corp  US5949181045  4        1        MICROSOFT CORP  US       Microsoft
  2 2024-04-01  NVIDIA Corp     US67066G1040  0        2        NVIDIA CORP     US       NVIDIA
  3 2024-06-28  Alphabet Inc    US02079K3059  4        2        ALPHABET INC    US       Alphabet
```

## 15. Computed columns

`select` with a list of `Select_Expr` builds a new DataFrame whose columns are computed from existing ones:

```odin
double_it :: proc(x: f64) -> f64 {return 2 * x}

price  := w.column(&df, "price")
shares := w.column(&df, "shares")
big    := w.mask_gt(shares, 50)

exprs := []w.Select_Expr {
    w.col_expr("ticker", w.column(&df, "ticker")),
    w.mult_expr("price_eur", price, 0.92),           // Float * f64
    w.add_expr("shares_plus_10", shares, 10),        // Int + int stays Int
    w.apply_expr("double_price", price, double_it),  // any proc(f64) -> f64
    w.conv_expr("shares_text", shares, "string"),    // type conversion
    w.mask_expr("big_position", big),                // a []bool as a Bool column
}
out := w.select(&df, exprs)
defer w.destroy_dataframe(&out)
w.free_select_exprs(exprs) // frees the masks given to mask_expr

w.dataframe_pretty_print(&out)
```

```
    ticker  price_eur  shares_plus_10  double_price  shares_text  big_position
  0 AAPL    174.34     110             379           100          true
  1 MSFT    377.384    50              820.4         40           false
  2 SAP     167.44     85              364           75           true
  3 ASML    832.692    20              1810.2        10           false
  4 NVDA    805.276    30              1750.6        20           false
  5 SIE     158.608    70              344.8         60           true
```

The available expressions:

| Expression | Result |
|---|---|
| `col_expr(name, col)` | A copy of `col` under a new name |
| `add_expr`, `sub_expr`, `mult_expr`, `div_expr` `(name, col, value)` | Arithmetic with a constant. An `int` constant on an `Int` column gives `Int` (integer division for `div_expr`). Otherwise the result is `Float`. Dividing by an `int` 0 panics. |
| `apply_expr(name, col, fn)` | `fn` applied to every non-null cell. `fn` can be a `proc(x: T) -> T` for `int`, `f64`, `string`, `bool`, `Date`, `Time` or `Datetime`. |
| `conv_int_to_f64_expr`, `conv_f64_to_int_expr` | Numeric conversions |
| `conv_expr(name, col, "type")` | Conversion to `"int"`, `"float"`, `"bool"`, `"string"`, `"date"`, `"time"` or `"datetime"` |
| `mask_expr(name, mask)` | A `[]bool` as a `Bool` column |

Nulls stay null in every expression. `conv_expr` can convert any type to `"string"` and parse strings into any type. It also converts dates to and from `YYYYMMDD` integers, and datetimes to and from Unix seconds. A conversion that is not supported panics instead of producing a wrong column.

`free_select_exprs` frees the masks you passed to `mask_expr`. Call it after `select` and do not `delete` those masks yourself.

## 16. Sorting

```odin
by_price := w.dataframe_sort(&df, "price", descending = true)
defer w.destroy_dataframe(&by_price)

// country ascending, then price descending within each country
by_country := w.dataframe_sort_by(&df, []string{"country", "price"}, []bool{false, true})
defer w.destroy_dataframe(&by_country)
w.dataframe_pretty_print(&by_country)
```

```
    ticker  country  price  shares
  0 SAP     DE       182    75
  1 SIE     DE       172.4  60
  2 ASML    NL       905.1  10
  3 NVDA    US       875.3  20
  4 MSFT    US       410.2  40
  5 AAPL    US       189.5  100
```

Sorting is stable, so rows with equal keys keep their original order. Nulls and NaN sort last in both directions. If you only need the order, `column_sort_order(col, descending)` returns the row permutation as a `[]int` (free it with `delete`).

## 17. Indexes and label-based access

`set_index` makes a column the index. It sorts the rows by that column (in place) so that the `loc_*` procedures can find keys by binary search:

```odin
df := w.df_from(
    w.column_from_dates("date", []w.Date{{2024, 1, 5}, {2024, 1, 2}, {2024, 1, 4}, {2024, 1, 3}}),
    w.column_from_floats("close", []f64{181.2, 185.6, 181.9, 184.3}),
)
defer w.destroy_dataframe(&df)

w.set_index(&df, "date") // sorts the rows by date

day := w.loc_single(&df, w.Date{2024, 1, 3}) // one row (copy)
defer w.destroy_dataframe(&day)

window := w.loc_range(&df, w.Date{2024, 1, 3}, w.Date{2024, 1, 4}) // inclusive, view
defer w.destroy_dataframe(&window)

w.dataframe_pretty_print(&window)
```

```
    date        close
  1 2024-01-03  184.3
  2 2024-01-04  181.9
```

| Procedure | Selects |
|---|---|
| `loc_single(&df, key)` | The first row with that key; panics if there is none (`try_loc` returns `ok = false` instead) |
| `loc_range(&df, start, stop)` | All rows with `start <= key <= stop` (both ends included) |
| `loc_from(&df, start)`, `loc_until(&df, stop)` | Open-ended ranges |
| `loc_many(&df, keys)` | All rows matching each key, in the order of `keys` |
| `loc_mask(&df, mask)` | Rows where the mask is `true` |

The key must have the index column's type: a `w.Date` for a `Date` index, an `int` for an `Int` index, a `string` for a `String` index (such as tickers), and so on. The overload `w.loc(&df, key)` or `w.loc(&df, start, stop)` works for a single key or a range. For lists and masks, call `loc_many` or `loc_mask` by name, because `w.loc` with a slice is ambiguous.

`set_index(&df, name, drop = true)` removes the column from the visible data but still uses it as the index.

`reindex` aligns a Date-indexed frame to a list of dates. Dates that are missing from the frame get a row of nulls, which is useful for putting time series on a common calendar:

```odin
calendar := []w.Date{{2024, 1, 1}, {2024, 1, 2}, {2024, 1, 3}}
aligned := w.reindex(&df, calendar) // 2024-01-01 is not in df: a row of nulls
defer w.destroy_dataframe(&aligned)
w.dataframe_pretty_print(&aligned)

w.reset_index(&df)
fmt.println(df.has_index) // false
```

```
    date        close
  0 2024-01-01  NULL
  1 2024-01-02  185.6
  2 2024-01-03  184.3
```

`reset_index(&df)` clears the index. If the index column was dropped, it comes back as the first column. `reset_index(&df, drop = true)` discards the index values instead.

## 18. Grouping and aggregation

There are two group-by APIs.

**`groupby` and `agg`** cover the common aggregations. You pass the aggregated column as a `^Column`:

```odin
gdf := w.groupby(&df, []string{"country"})
defer w.destroy_grouped_dataframe(&gdf)

summary := w.agg(
    &gdf,
    []w.Agg_Expr {
        w.count("n"),
        w.avg_agg("avg_price", w.column(&df, "price")),
        w.max_agg("max_price", w.column(&df, "price")),
        w.sum_agg("total_shares", w.column(&df, "shares")),
    },
)
defer w.destroy_dataframe(&summary)
w.dataframe_pretty_print(&summary)
```

```
    country  n  avg_price          max_price  total_shares
  0 US       3  491.6666666666667  875.3      160
  1 DE       2  177.2              182        135
  2 NL       1  905.1              905.1      10
```

The aggregations are `count`, `sum_agg`, `avg_agg`, `min_agg` and `max_agg`. `min_agg` and `max_agg` work on every type, including strings and dates. Groups appear in the order of their first row, nulls and NaN are skipped, and you can group by several columns by passing several names.

**`groupby2` and `groupby2_agg`** add statistical aggregations and name columns by string:

```odin
gb := w.groupby2(&df, []string{"country"})
defer w.destroy_groupby2(&gb)

stats := w.groupby2_agg(
    &gb,
    []w.Aggregator {
        {name = "mean_price", column = "price", kind = .Mean},
        w.make_median_agg("median_price", "price"),
        w.make_quantile_agg("p90_price", "price", 0.9),
        w.make_var("var_price", "price"),
        w.make_corr("corr_price_shares", "price", "shares"),
    },
)
defer w.destroy_dataframe(&stats)
w.dataframe_pretty_print(&stats)
```

```
    country  mean_price         median_price  p90_price  var_price           corr_price_shares
  0 US       491.6666666666667  410.2         782.28     122558.02333333332  -0.8851618298407099
  1 DE       177.2              177.2         181.04     46.07999999999994   1
  2 NL       905.1              905.1         905.1      NULL                NULL
```

The `AggregationKind` values are `Sum`, `Mean`, `Min`, `Max`, `Count`, `Median`, `Quantile`, `Var`, `Cov` and `Corr`, plus the exponentially weighted `EWM_Mean`, `EWM_Var`, `EWM_Std`, `EWM_Cov` and `EWM_Corr`. Set `alpha` on the `Aggregator` for the EWM kinds. The pairwise kinds (`Cov`, `Corr`, `EWM_Cov`, `EWM_Corr`) take the second column in `other`. Variances and covariances use `n - 1`, quantiles interpolate linearly, and EWM uses pandas' `adjust=True` weights. A statistic that is undefined for a group is null, as for the variance of NL's single row above.

## 19. Joins

`join` matches rows on one or more key columns, with SQL semantics:

```odin
holdings := w.df_from(
    w.column_from_strings("ticker", []string{"AAPL", "SAP", "TSLA"}),
    w.column_from_ints("shares", []int{100, 75, 5}),
)
defer w.destroy_dataframe(&holdings)

quotes := w.df_from(
    w.column_from_strings("ticker", []string{"AAPL", "MSFT", "SAP"}),
    w.column_from_floats("price", []f64{189.5, 410.2, 182.0}),
)
defer w.destroy_dataframe(&quotes)

inner := w.join(&holdings, &quotes, []string{"ticker"}) // .Inner by default
defer w.destroy_dataframe(&inner)

left := w.join(&holdings, &quotes, []string{"ticker"}, .Left)
defer w.destroy_dataframe(&left)
w.dataframe_pretty_print(&left) // TSLA has a null price
```

```
    ticker  shares  price
  0 AAPL    100     189.5
  1 SAP     75      182
  2 TSLA    5       NULL
```

The join kinds are `.Inner`, `.Left`, `.Right` and `.Outer`:
- **Columns:** the output has all left columns, then the right columns except the keys. If a right column's name is already taken, it gets the suffix `_right`.
- **Duplicate keys:** every matching pair of rows is produced.
- **Null keys** never match.

When the key columns have different names, use `join_on`:

```odin
sectors := w.df_from(
    w.column_from_strings("symbol", []string{"AAPL", "SAP"}),
    w.column_from_strings("sector", []string{"Tech", "Software"}),
)
defer w.destroy_dataframe(&sectors)

with_sector := w.join_on(&holdings, &sectors, []string{"ticker"}, []string{"symbol"}, .Left)
defer w.destroy_dataframe(&with_sector)
```

```
    ticker  shares  sector
  0 AAPL    100     Tech
  1 SAP     75      Software
  2 TSLA    5       NULL
```

### As-of joins

`merge_asof` joins each row of the left table to the nearest row of the right table by an ordered key, such as a time or a date, instead of requiring equal keys. The typical use is the last quote at or before each trade, or the last published value at or before each date. `by` names key columns that must match exactly, such as the ticker:

```odin
trades := w.df_from(
    w.column_from_datetimes("time", {{2024, 1, 2, 9, 30, 5}, {2024, 1, 2, 9, 30, 9}, {2024, 1, 2, 9, 31, 0}}),
    w.column_from_strings("ticker", {"AAPL", "MSFT", "AAPL"}),
    w.column_from_ints("qty", {100, 50, 200}),
)
defer w.destroy_dataframe(&trades)
quotes := w.df_from(
    w.column_from_datetimes("time", {{2024, 1, 2, 9, 30, 0}, {2024, 1, 2, 9, 30, 0}, {2024, 1, 2, 9, 30, 8}, {2024, 1, 2, 9, 30, 40}}),
    w.column_from_strings("ticker", {"AAPL", "MSFT", "AAPL", "AAPL"}),
    w.column_from_floats("bid", {184.9, 409.8, 185.1, 185.3}),
)
defer w.destroy_dataframe(&quotes)

latest, ok := w.merge_asof(&trades, &quotes, "time", by = {"ticker"})
defer w.destroy_dataframe(&latest)
w.dataframe_pretty_print(&latest)
```

```
    time                 ticker  qty  bid
  0 2024-01-02 09:30:05  AAPL    100  184.9
  1 2024-01-02 09:30:09  MSFT    50   409.8
  2 2024-01-02 09:31:00  AAPL    200  185.3
```

The result has one row per left row, in left order and with the left row labels: all left columns, then the right columns except the keys. A left row without a match gets nulls. The options:

| Option | Effect |
|---|---|
| `direction = .Backward` (default) | The last right row whose key is at or before the left key |
| `direction = .Forward` | The first right row at or after the left key |
| `direction = .Nearest` | The closer of the two; the earlier one on a tie |
| `tolerance = d` | Match only within distance `d`, in the key's units: days for `Date`, seconds for `Time` and `Datetime`. `tolerance = 10.0` above would leave the last trade without a bid, because its quote is 20 seconds old |
| `allow_exact_matches = false` | A right key equal to the left key does not count |
| `right_on = "name"` | The right key column, when it is named differently |

The key can be an `Int`, `Float`, `Date`, `Time` or `Datetime` column of the same type in both tables. Unlike pandas, neither table needs to be sorted. Among right rows with the same key, `.Backward` takes the last and `.Forward` the first.

## 20. Dates and times

`Date`, `Time` and `Datetime` are plain structs, so you can write them as literals. The procedures below do calendar arithmetic on them:

```odin
d := w.Date{2024, 1, 31}

fmt.println(w.add_day_date(d, 1))                     // 2024-02-01
fmt.println(w.add_month_date(d, 1))                   // 2024-02-29: clamped to the month's end
fmt.println(w.date_diff(d, w.Date{2024, 3, 1}))       // 30 days (b - a)
fmt.println(w.date_weekday(d))                        // Wednesday
fmt.println(w.date_end_of_month(w.Date{2023, 2, 10})) // 2023-02-28

parsed, ok := w.parse_date("2024-07-15")              // 2024-07-15 true

s := w.date_to_string(parsed) // "2024-07-15", allocated
defer delete(s)

dt := w.Datetime{2024, 7, 15, 9, 30, 0}
fmt.println(w.datetime_to_unix(dt))                   // 1721035800 (seconds since 1970-01-01)
fmt.println(w.get_Date_from_Datetime(dt), w.get_Time_from_Datetime(dt))
```

(`fmt.println` prints dates as structs, for example `Date{year = 2024, month = 2, day = 1}`. The comments show the values in ISO form for readability.)

| Task | Procedures |
|---|---|
| Arithmetic | `add_day_date`, `add_month_date`, `add_year_date`, and the `_datetime` variants; `add_seconds_time`, `add_minutes_time`, `add_hour_time` |
| Differences and comparison | `date_diff` (days), `date_compare`, `date_less`, `time_compare`, `datetime_compare` |
| Calendar facts | `date_weekday`, `date_end_of_month`, `date_is_end_of_month`, `get_days_in_month`, `get_days_of_the_year` |
| Parsing | `parse_date` (`YYYY-MM-DD`), `parse_time` (`HH:MM:SS`), `parse_datetime` (`YYYY-MM-DD HH:MM:SS` or with `T`) |
| Formatting | `date_to_string`, `time_to_string`, `datetime_to_string` (allocate; free with `delete`) |
| Conversion | `datetime_to_unix`, `datetime_from_unix`, `date_from_unix`, `date_to_int` and `int_to_date` (`YYYYMMDD`), `date_to_f64` and `f64_to_date` (Julian day) |
| Now | `now()` gives a `Datetime`; `now_unix()` gives seconds |

`add_month_date` clamps the day to the target month's length, as in the example: 31 January plus one month is 29 February 2024. For business-day calendars, day-count conventions and payment schedules, see the `finance` package.

## 21. Memory management

Wotan follows the usual Odin rule: whoever creates something frees it. In `core` that comes down to a few rules.

1. **Every procedure that returns a `DataFrame` gives you something to free** with `w.destroy_dataframe(&df)`, and that includes views. Destroying a view frees only the view's own bookkeeping, never the source's data.
2. **A view must not outlive its source, and you must not add columns to the source while views of it exist.** A view points at the source's columns, so after `destroy_dataframe(&source)` the view reads freed memory. Adding a column can also move the source's columns in memory. If a result has to outlive its source, `materialize` it or use the copying variant.
3. **Masks are plain slices.** Free every mask with `delete`, including the results of `and`, `or` and `not`. Masks passed to `mask_expr` are the exception: `free_select_exprs` frees them.
4. **A grouping is separate from its result.** Free a `GroupedDataFrame` with `destroy_grouped_dataframe` and a `GroupBy2` with `destroy_groupby2`. The DataFrame returned by `agg` or `groupby2_agg` is independent of the grouping, so free it with `destroy_dataframe`.
5. **Columns passed to `df_from` or `add_column` belong to the DataFrame.** Do not destroy them separately. A standalone column that you never added to a DataFrame is freed with `destroy_column`.
6. **Strings returned by the date formatting procedures are allocated.** `date_to_string`, `time_to_string` and `datetime_to_string` return strings to free with `delete`. Strings read from a column (`column_at_string`) belong to the column and must not be freed.
7. **A column borrows the name you give `column_new`,** so pass a string that lives at least as long as the column, such as a literal. Columns made by the importers and by the copying procedures (`column_take`, `column_clone`, `materialize`, the copying row selections, joins and reindexing) own their names, so a copy stays valid after its source is destroyed. `w.column_own_name(&col, name)` gives any column its own copy, freed by `destroy_column`.

The usual pattern is to put a `defer` right after each allocation, as the examples in this guide do. Run with `-debug` and the tracking allocator from section 1, and any leak is reported with the line that allocated it.

**Temporary scratch.** `w.TEMP_GUARD()` marks the current position of `context.temp_allocator` and releases everything allocated on it after the mark when the enclosing procedure (or block) ends. Wotan's procedures that use temporary scratch, such as the chart procedures, the ARIMA fits and the optimizers, start with it, so a loop that calls them does not grow the temporary allocator. It rewinds:
- `runtime.Default_Temp_Allocator`, Odin's default or one you set up yourself;
- `runtime.Arena`;
- `mem.Arena` from `core:mem`;
- `virtual.Arena` from `core:mem/virtual`.

Any other allocator, such as a tracking allocator, cannot be rewound to a mark; the guard then does nothing, and the memory stays until `free_all(context.temp_allocator)`. Odin's own `runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()` rewinds only the global default. `w.TEMP_GUARD(ignore = true)` does nothing, for code paths whose result itself goes on the temporary allocator. `w.temp_begin()` and `w.temp_end(mark)` are the two halves without the `defer`.

```odin
// a step that needs 64 KB of scratch space
step :: proc() -> int {
	w.TEMP_GUARD() // the scratch below is released when step returns
	buf := make([]u8, 64 * 1024, context.temp_allocator)
	buf[0] = 1
	return len(buf)
}

main :: proc() {
	backing := make([]u8, 1 << 20)
	defer delete(backing)
	arena: mem.Arena
	mem.arena_init(&arena, backing)
	context.temp_allocator = mem.arena_allocator(&arena)

	for _ in 0 ..< 100 {_ = step()}
	fmt.println("arena bytes in use after 100 steps:", arena.offset)
}
```

```
arena bytes in use after 100 steps: 0
```

Without the guard, the 1 MB arena would be full after 16 steps.

## 22. Checked entry points

Some of the basic procedures panic when they get a name, key or position that does not exist: `column`, `add_column`, `set_index`, `loc`, `iloc`, `select`, `dataframe_sort_by`, `join`, `groupby2` and the `mask_*` comparisons. That is convenient while you explore data, and fatal in a program that reads column names from a file header or a configuration. Each of them has a checked variant that returns `ok = false` instead and otherwise gives exactly the same result:

| Checked variant | Instead of | `ok = false` when |
|---|---|---|
| `try_column(&df, name)` | `column` | the name does not exist |
| `try_add_column(&df, col)` | `add_column` | the length differs or the name is taken (the column then stays yours to free) |
| `try_set_index(&df, name, drop)` | `set_index` | the name does not exist |
| `try_loc(&df, key)` | `loc` | there is no index, the key's type does not match it, or no row has that key |
| `try_iloc(&df, i)`, `try_iloc_many(&df, idxs)` | `iloc` | a position is out of range |
| `try_select(&df, names)` | `select` | a name does not exist |
| `try_sort(&df, columns, descending)` | `dataframe_sort_by` | a name does not exist |
| `try_join`, `try_join_on` | `join`, `join_on` | a key does not exist, or the key lists differ in length |
| `try_groupby(&df, keys, aggs)` | `groupby2` + `groupby2_agg` | a key or column does not exist, a correlation lacks its second column, a quantile or EWM alpha is out of range |
| `compare(col, op, value)` | `mask_gt`, `mask_lt`, `mask_eq` | the value's type does not match the column |

`compare` takes `.Eq`, `.Ne`, `.Lt`, `.Le`, `.Gt` or `.Ge`, and gives `false` for missing cells under every operator (pandas' `!=` gives `true` there). `try_groupby` runs the grouping and the aggregation in one call and frees the grouping itself.

```odin
price, found := w.try_column(&df, "price") // nil, false: there is no such column
w.set_index(&df, "isin")
row, hit := w.try_loc(&df, "US5949181045") // the Microsoft row, true
defer w.destroy_dataframe(&row)
_, miss := w.try_loc(&df, "XX0000000000") // false
```

The procedures of sections 10 to 14, `merge_asof` and `resample` report problems through `ok` themselves and need no checked variants.

## 23. A complete example

This example loads trades and sector data, computes a signed notional per trade, keeps the trades from 4 March onwards, adds the sector to each trade, and ranks sectors by net flow. It uses most of the package.

```odin
package main

import w  "wotan/core"
import io "wotan/importer"
import "core:fmt"

TRADES :: `date,ticker,side,qty,price
2024-03-01,AAPL,buy,100,179.66
2024-03-01,SAP,buy,50,172.10
2024-03-04,AAPL,sell,40,175.10
2024-03-05,MSFT,buy,20,402.65
2024-03-05,SAP,sell,20,174.30
2024-03-06,MSFT,buy,10,402.09
`

SECTORS :: `ticker,sector
AAPL,Hardware
MSFT,Software
SAP,Software
`

main :: proc() {
    trades := io.csv_load_from_string(TRADES)
    defer w.destroy_dataframe(&trades)
    sectors := io.csv_load_from_string(SECTORS)
    defer w.destroy_dataframe(&sectors)

    // 1. notional = qty * price, with the sign of the trade
    qty   := w.column(&trades, "qty")   // inferred as Int
    price := w.column(&trades, "price") // inferred as Float
    side  := w.column(&trades, "side")
    signed := w.column_new("signed_notional", .Float, trades.rows)
    for i in 0 ..< trades.rows {
        q, _ := w.column_at_numeric(qty, i)
        p, _ := w.column_at_float(price, i)
        s, _ := w.column_at_string(side, i)
        w.append_float(&signed, q * p if s == "buy" else -q * p)
    }
    w.add_column(&trades, signed)

    // 2. only March 4 onwards
    from := w.mask_gt(w.column(&trades, "date"), w.Date{2024, 3, 3})
    defer delete(from)
    recent := w.filter(&trades, from)
    defer w.destroy_dataframe(&recent)

    // 3. add the sector, then net flow per sector
    enriched := w.join(&recent, &sectors, []string{"ticker"}, .Left)
    defer w.destroy_dataframe(&enriched)

    gdf := w.groupby(&enriched, []string{"sector"})
    defer w.destroy_grouped_dataframe(&gdf)
    flows := w.agg(&gdf, []w.Agg_Expr {
        w.count("trades"),
        w.sum_agg("net_flow", w.column(&enriched, "signed_notional")),
    })
    defer w.destroy_dataframe(&flows)

    // 4. largest inflow first
    ranked := w.dataframe_sort(&flows, "net_flow", descending = true)
    defer w.destroy_dataframe(&ranked)
    w.dataframe_pretty_print(&ranked)
}
```

```
    sector    trades  net_flow
  0 Software  3       8587.9
  1 Hardware  1       -7004
```

The signed column is added to `trades` (step 1) before `recent` filters it. `filter` copies the rows, so `recent` and everything built from it stay valid even if `trades` is freed first.

## 24. Quick reference

| Task | Procedures |
|---|---|
| Create | `df_from`, `dataframe_new`, `add_column`, `column_new`, `column_from_*`, `append_*`, `append_null` |
| Inspect | `df.rows`, `len(df.columns)`, `column`, `dataframe_column_name`, `dataframe_column_type` |
| Read | `column_at_*`, `column_at_numeric`, `is_null`, `df_series`, `series_at_*`, `value_as_string` |
| Print | `dataframe_pretty_print`, `df_head`, `dataframe_print` |
| Columns | `select` (names), `dataframe_select_columns`, `drop_columns`, `rename_column` |
| Rows by position | `iloc`, `iloc_range`, `iloc_many`, `head`, `tail`, `df_row`, `df_slice`, `df_take` |
| Masks | `mask_gt`, `mask_lt`, `mask_eq`, `compare`, `mask_contains_string`, `isin`, `between`, `str_contains`, `str_startswith`, `str_endswith`, `and`, `or`, `not`, `column_mask` |
| Filter | `filter`, `wobei`, `loc_mask` |
| Missing data | `na_at`, `count_na`, `isna`, `notna`, `dropna`, `fillna`, `ffill`, `bfill`, `interpolate` |
| Clean and combine | `concat_rows`, `cast_column`, `cast_columns`, `duplicated`, `drop_duplicates`, `unique`, `nunique`, `value_counts` |
| Column calculations | `shift`, `diff`, `pct_change`, `cumsum`, `cumprod`, `cummax`, `cummin`, `clip`, `rank`, `nlargest`, `nsmallest`, `describe`; per group with `by` |
| Reshape and sample | `pivot`, `pivot_table`, `melt`, `resample`, `sample` |
| Date and text parts | `date_part`, `str_upper`, `str_lower`, `str_strip`, `str_replace`, `str_slice`, `str_split_part`, `str_len` |
| Checked variants | `try_column`, `try_add_column`, `try_set_index`, `try_loc`, `try_iloc`, `try_iloc_many`, `try_select`, `try_sort`, `try_join`, `try_join_on`, `try_groupby`, `compare` |
| Compute | `select` (expressions), `col_expr`, `add_expr`, `sub_expr`, `mult_expr`, `div_expr`, `apply_expr`, `conv_expr`, `mask_expr`, `free_select_exprs` |
| Sort | `dataframe_sort`, `dataframe_sort_by`, `column_sort_order` |
| Index | `set_index`, `reset_index`, `loc`, `loc_single`, `loc_range`, `loc_from`, `loc_until`, `loc_many`, `reindex` |
| Group | `groupby` + `agg` (`count`, `sum_agg`, `avg_agg`, `min_agg`, `max_agg`); `groupby2` + `groupby2_agg` (`Aggregator`) |
| Join | `join`, `join_on` with `.Inner`, `.Left`, `.Right`, `.Outer`; `merge_asof` |
| Copy | `materialize`, `column_clone`, `column_take` |
| Free | `destroy_dataframe`, `destroy_column`, `destroy_grouped_dataframe`, `destroy_groupby2`, `delete` (masks, strings) |
| Temporary scratch | `TEMP_GUARD`, `temp_begin`, `temp_end` |

For more examples, the tests in [`tests/`](../tests/) exercise every procedure. [`core_contracts_test.odin`](../tests/core_contracts_test.odin) in particular pins down the exact semantics of nulls, views, sorting, grouping and joins, and [`missing_data_test.odin`](../tests/missing_data_test.odin), [`frame_tools_test.odin`](../tests/frame_tools_test.odin), [`column_ops_test.odin`](../tests/column_ops_test.odin), [`reshape_test.odin`](../tests/reshape_test.odin), [`resample_test.odin`](../tests/resample_test.odin), [`group_asof_test.odin`](../tests/group_asof_test.odin) and [`parts_checked_test.odin`](../tests/parts_checked_test.odin) check the procedures of sections 9 to 14 and `merge_asof` against pandas, and the checked variants against the originals.
