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

### **Time‑Series Analytics**
- Rolling windows
- Rolling correlation/covariance matrices
- Exponential weighted statistics (EWM)
- PCA & rolling PCA
- ARIMA(p,d,q), ARMA(p,q), SARIMA
- Residual diagnostics (ACF, PACF, Ljung‑Box, JB test)
- Stationarity tests (ADF, KPSS)

### **State‑Space Models**
- Kalman filter
- Kalman smoother (RTS)
- Control models
- Time‑varying Kalman
- EKF, UKF + smoothing

### **Importers / Exporters**
- CSV
- JSON / JSONL
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

## Create a DataFrame
```sh
odin

import w "../wotan/core"
df := w.dataframe_new()

col_age  := w.column_new("age", .Int, 4)
col_name := w.column_new("name", .String, 4)

w.append_int(&col_age, 10)
w.append_string(&col_name, "Hubert")

w.append_int(&col_age, 20)
w.append_string(&col_name, "Anna")

w.add_column(&df, col_age)
w.add_column(&df, col_name)

w.dataframe_pretty_print(&df)

```
## Filtering & Slicing
```sh
young := w.filter(&df, w.mask_lt(w.column(&df, "age"), 30))
w.dataframe_pretty_print(&young)

```

## Sorting
```sh
sorted := w.dataframe_sort(&df, "age", false)
w.dataframe_pretty_print(&sorted)


```
## Web request i.e. Yahoo Finance
```sh
aapl := w.yahoo_load("AAPL", .Daily, .TenYears)
sorted := w.dataframe_sort(&aapl, "Close", true)
w.df_head(&sorted, 10)


```
## Time Series Analysis like ARIMA
```sh
model := w.arima_fit(df, "y", p=1, d=0, q=1)
forecast := w.arima_forecast(&model, 5)

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