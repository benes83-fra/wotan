<p align="center">
  <img src="https://raw.githubusercontent.com/benes83-fra/wotan/main/wotan%20logo.png" width="280" alt="Wotan Logo">
</p>

<h1 align="center">Wotan — High‑Performance DataFrame Engine for Odin</h1>

<p align="center">
A fast, expressive, allocator‑safe DataFrame and time‑series analytics engine written in Odin.
</p>

---

## 🚀 Features

### **DataFrame Engine**
- Strongly typed columns (`Int`, `Float`, `String`, `Bool`, `Date`, `Time`, `Datetime`)
- Row slicing (copy + view)
- Column selection
- Boolean filtering & mask algebra
- Sorting (all types)
- Expression engine (`select`, `apply`, `add`, `conv`, `mask_expr`)
- GroupBy + aggregations
- Joins (single‑key & multi‑key)

### **Classical Machine Learning**
- Linear regression (OLS, WLS, GLS, Ridge, Lasso with CV)
- Decision Trees, Random Forests, Gradient Boosting
- Logistic Regression (binary + multiclass PCR)
- KNN clustering, Gaussian Naive Bayes
- Support Vector Machines (SVM, Kernel SVM SVR)
- Metrics, Pipeline API, Grid/Random SearchCV

### **Deep Learning**
- Neural net primitives: Conv2D, Pooling, Dense, Dropout, BatchNorm
- RNN family: Simple RNN, GRU, LSTM
- Transformer architecture: multi-head attention, encoder/decoder, positional encoding
- Pretrained-style models: GPT, BERT, character-level language model
- Generative models: GAN, WGAN, VAE
- Transfer learning utilities
- Autograd engine (back-prop through matmul, mul, sum, ReLU, bias ops)

### **Portfolio & Risk Analytics**
- Portfolio construction with constraints
- Risk decomposition and financial analytics
- Monte Carlo simulation paths
- Factor analysis

### **State‑Space Models**
- Kalman filter
- Kalman smoother (RTS)
- Control models
- Time‑varying Kalman
- EKF, UKF + smoothing

### **Time-Series Analytics**
- Rolling windows and rolling correlation/covariance matrices
- Exponentially weighted statistics (EWM)
- PCA & rolling PCA
- ARIMA(p,d,q), ARMA(p,q), SARIMA
- Residual diagnostics (ACF, PACF, Ljung‑Box, JB test)
- Stationarity tests (ADF, KPSS)

### **Plotters**
- Line plots (single, multi-line, dashed/dotted)
- Bar charts
- Heatmaps
- Confusion matrices

### **Importers / Exporters**
- CSV, JSON, JSONL
- HTML tables
- Excel (.xlsx)
- ZIP utilities

### **Yahoo Finance Ingestion**
- Historical prices
- Dividends
- Splits
- Event alignment

---

Wotan is very lean so far, apart from libcurl for webrequest there are no external dependencies

## 📦 Installation

```sh
git clone https://github.com/benes83-fra/wotan


```

## 📖 Examples

### Create a DataFrame

```odin
import w "../wotan/core"

df := w.dataframe_new()

col_age  := w.column_new("age", .Int, 4)
col_name := w.column_new("name", .String, 4)

w.append_int(&col_age, 10)
w.append_string(&col_name, "Hubert")

w.append_int(&col_age, 20)
w.append_string(&col_name, "Anna")

w.append_int(&col_age, 30)
w.append_string(&col_name, "Markus")

w.append_int(&col_age, 40)
w.append_string(&col_name, "Julia")

w.add_column(&df, col_age)
w.add_column(&df, col_name)

w.dataframe_pretty_print(&df, 10)
```

### Filter rows with boolean masks

```odin
// Show only rows where age < 31 and active == true
df := csv.csv_load("people_dates.csv")

m1 := w.mask_lt(w.column(&df, "age"), 31)
m2 := w.column_mask(w.column(&df, "active"))
mask := w.and(m1, m2)

young_active := w.wobei(&df, mask)
w.dataframe_pretty_print(&young_active, 10)

delete(mask); delete(m1); delete(m2)
```

### Expressions: add, convert, and compute

```odin
exprs := []w.Select_Expr {
    w.col_expr("age",           w.column(&df, "age")),
    w.add_expr("age_plus_10",  w.column(&df, "age"), 10),
    w.apply_expr("upper_name", w.column(&df, "name"), proc(s: string) -> string {
        return strings.to_upper(s, context.temp_allocator)
    }),
    w.div_expr("salary_k",     w.column(&df, "salary"), 1000),
    w.conv_int_to_f64_expr("age_f64", w.column(&df, "age")),
}

result := w.select(&df, exprs)
w.dataframe_pretty_print(&result, 10)

w.free_select_exprs(exprs) // or defer on the slice
```

### GroupBy + aggregation

```odin
gdf := w.groupby(&df, []string{"age"})
agg := []w.Agg_Expr {
    w.count("n"),
    w.sum_agg("total_salary", w.column(&df, "salary")),
    w.avg_agg("avg_salary",   w.column(&df, "salary")),
}

out := w.agg(&gdf, agg)
w.dataframe_pretty_print(&out, 10)

w.destroy_grouped_dataframe(&gdf)
w.destroy_dataframe(&out)
```

### Join two dataframes

```odin
people := df_from(
    column_from_ints("id",      []int{1, 2, 3}),
    column_from_strings("name", []string{"Alice", "Bob", "Charlie"}),
    column_from_ints("age",     []int{30, 20, 40}),
)

salary := df_from(
    column_from_ints("id",     []int{1, 2, 4}),
    column_from_floats("salary", []f64{50000.0, 42000.0, 90000.0}),
)

joined := w.join(&people, &salary, []string{"id"}, .Outer, context.temp_allocator)
w.dataframe_pretty_print(&joined, 10)
```

### Multi-key join

```odin
left := df_from(
    column_from_ints("id",     []int{1, 1, 2}),
    column_from_strings("dept", []string{"10", "20", "10"}),
    column_from_strings("name", []string{"Alice", "Bob", "Carol"}),
)

right := df_from(
    column_from_ints("id",      []int{1, 2}),
    column_from_strings("dept",  []string{"10", "10"}),
    column_from_floats("salary",[]f64{50000.0, 60000.0}),
)

joined := w.join(&left, &right, []string{"id", "dept"}, .Inner, context.temp_allocator)
w.dataframe_pretty_print(&joined, 10)
```

### Load CSV / JSON / Excel

```odin
// From CSV (type list required)
types := []w.ColumnType{.Int, .String, .Float}
df := csv.csv_load("data.csv", types)

// From JSON / JSONL
jdf := json.load("data.json")
jldf := jsonl.load("data.jsonl")

// From Excel
edf := excel.read("data.xlsx")
```

### Time Series: ARIMA fit & forecast

```odin
model := w.arima_fit(df, "Close", .P(1), .D(0), .Q(1))
forecast := w.arima_forecast(&model, 5)
for i, v in forecast {
    fmt.printf("Step %d: %f\n", i, v)
}
```

### Date & datetime math

```odin
date1 := w.Date{2020, 2, 7}
date2 := w.Date{2024, 2, 6}
days_between := w.get_date_day_diffs(date1, date2)

// Add months / days
d := w.add_month_date(date1, -7)
d = w.add_day_date(d, 30)

// Time of day
t := w.Time{16, 2, 58}
t = w.add_seconds_time(t, -600)

// Full datetime
dt := w.Datetime{1983, 7, 20, 13, 13, 13}
dt = w.add_hours_datetime(dt, -49)
```

### Yahoo Finance ingestion

```odin
df := w.yahoo_load("AAPL", .Daily, .TenYears)
sorted := w.dataframe_sort(&df, "Close", true)
w.df_head(&sorted, 10)
```

### 🧪 Tests
- GroupBy
- Rolling windows
- PCA
- EWM
- Kalman filters
- ARIMA / SARIMA
- Stationarity tests
- JSON / HTML / Excel importers
- Yahoo Finance ingestion

## Run
```sh
odin run . -debug 

```

## Build - Executeable and Libraries
```sh
odin build . 


odin build . -build_mode:static

odin build . -build_mode:dynamic

```



### 🤝 Contributing
Contributions, bug reports, and feature requests are welcome.
Wotan is evolving rapidly — feedback is highly appreciated.

### 🛠 Roadmap
- For pandas like Data Querring support
- Support for traditional ML - Regression, Classification, Trees, SVMs, Forrests and all that fun stuff
- A graphic Plotter, either via Raylib (this is an Odin Project after all) or as plain PNG files.
- to maybe once net/http drops, be crazy and try to implement some Wotan Notebook... It might at least be interesting to try