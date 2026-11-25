# The `core` package: a user guide

`wotan/core` is the data layer of Wotan. It provides typed columns, the `DataFrame` that holds them, and the operations you need to work with tables: selecting, filtering, handling missing data, cleaning and combining tables, computing new columns, sorting, indexing, grouping, joining and date arithmetic. The rest of Wotan (the importers and exporters, the statistics and finance packages) reads and writes these DataFrames.

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
12. [Computed columns](#12-computed-columns)
13. [Sorting](#13-sorting)
14. [Indexes and label-based access](#14-indexes-and-label-based-access)
15. [Grouping and aggregation](#15-grouping-and-aggregation)
16. [Joins](#16-joins)
17. [Dates and times](#17-dates-and-times)
18. [Memory management](#18-memory-management)
19. [A complete example](#19-a-complete-example)
20. [Quick reference](#20-quick-reference)

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
- **Some results are views.** A view reads the data of another DataFrame instead of copying it. Views are cheap, but they must not outlive their source. Section 8 lists which operations return views and section 18 sums up the rules.

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

`w.column(&df, name)` returns a `^Column`. It panics if the name does not exist. The typed readers `column_at_int`, `column_at_float`, `column_at_bool`, `column_at_string`, `column_at_date`, `column_at_time` and `column_at_datetime` return the value and whether it is null:

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
| `w.dataframe_pretty_print(&df, max_rows = 20)` | An aligned table with row labels; nulls show as `NULL` |
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

`select` with a list of names returns a view. `dataframe_select_columns(..., copy = true)` returns an independent copy. The same `select` procedure also accepts expressions that compute new columns (section 12).

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

**Carrying values.** `ffill` fills each gap with the last present value above it, which is the usual way to bring a price series onto a common calendar. `bfill` uses the next present value below the gap. Cells before the first value (for `ffill`) or after the last value (for `bfill`) stay missing. With `limit = n`, at most `n` cells of each gap are filled. Both take an optional list of column names, and both also work on a single `^Column`.

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

## 12. Computed columns

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

## 13. Sorting

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

## 14. Indexes and label-based access

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
| `loc_single(&df, key)` | The first row with that key; panics if there is none |
| `loc_range(&df, start, stop)` | All rows with `start <= key <= stop` (both ends included) |
| `loc_from(&df, start)`, `loc_until(&df, stop)` | Open-ended ranges |
| `loc_many(&df, keys)` | All rows matching each key, in the order of `keys` |
| `loc_mask(&df, mask)` | Rows where the mask is `true` |

The key must have the index column's type: a `w.Date` for a `Date` index, an `int` for an `Int` index, and so on. The overload `w.loc(&df, key)` or `w.loc(&df, start, stop)` works for a single key or a range. For lists and masks, call `loc_many` or `loc_mask` by name, because `w.loc` with a slice is ambiguous.

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

## 15. Grouping and aggregation

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

## 16. Joins

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

## 17. Dates and times

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

## 18. Memory management

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

## 19. A complete example

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

## 20. Quick reference

| Task | Procedures |
|---|---|
| Create | `df_from`, `dataframe_new`, `add_column`, `column_new`, `column_from_*`, `append_*`, `append_null` |
| Inspect | `df.rows`, `len(df.columns)`, `column`, `dataframe_column_name`, `dataframe_column_type` |
| Read | `column_at_*`, `column_at_numeric`, `is_null`, `df_series`, `series_at_*`, `value_as_string` |
| Print | `dataframe_pretty_print`, `df_head`, `dataframe_print` |
| Columns | `select` (names), `dataframe_select_columns`, `drop_columns`, `rename_column` |
| Rows by position | `iloc`, `iloc_range`, `iloc_many`, `head`, `tail`, `df_row`, `df_slice`, `df_take` |
| Masks | `mask_gt`, `mask_lt`, `mask_eq`, `mask_contains_string`, `and`, `or`, `not`, `column_mask` |
| Filter | `filter`, `wobei`, `loc_mask` |
| Missing data | `na_at`, `count_na`, `isna`, `notna`, `dropna`, `fillna`, `ffill`, `bfill` |
| Clean and combine | `concat_rows`, `cast_column`, `cast_columns`, `duplicated`, `drop_duplicates`, `unique`, `nunique`, `value_counts` |
| Compute | `select` (expressions), `col_expr`, `add_expr`, `sub_expr`, `mult_expr`, `div_expr`, `apply_expr`, `conv_expr`, `mask_expr`, `free_select_exprs` |
| Sort | `dataframe_sort`, `dataframe_sort_by`, `column_sort_order` |
| Index | `set_index`, `reset_index`, `loc`, `loc_single`, `loc_range`, `loc_from`, `loc_until`, `loc_many`, `reindex` |
| Group | `groupby` + `agg` (`count`, `sum_agg`, `avg_agg`, `min_agg`, `max_agg`); `groupby2` + `groupby2_agg` (`Aggregator`) |
| Join | `join`, `join_on` with `.Inner`, `.Left`, `.Right`, `.Outer` |
| Copy | `materialize`, `column_clone`, `column_take` |
| Free | `destroy_dataframe`, `destroy_column`, `destroy_grouped_dataframe`, `destroy_groupby2`, `delete` (masks, strings) |
| Temporary scratch | `TEMP_GUARD`, `temp_begin`, `temp_end` |

For more examples, the tests in [`tests/`](../tests/) exercise every procedure. [`core_contracts_test.odin`](../tests/core_contracts_test.odin) in particular pins down the exact semantics of nulls, views, sorting, grouping and joins, and [`missing_data_test.odin`](../tests/missing_data_test.odin) and [`frame_tools_test.odin`](../tests/frame_tools_test.odin) check the procedures of sections 10 and 11 against pandas.
