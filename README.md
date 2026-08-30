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
model := ana.arima_fit(df, "Close", .P(1), .D(0), .Q(1))
forecast := ana.arima_forecast(&model, 5)
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
df := net.yahoo_load("AAPL", .Daily, .TenYears)
sorted := w.dataframe_sort(&df, "Close", true)
w.df_head(&sorted, 10)
```

## GARCH (1,1)
```odin
// Generate synthetic GARCH(1,1) returns
n := 1000
returns := make([]f64, n, context.allocator)
true_omega := 0.00001
true_alpha := 0.1
true_beta := 0.85

// Initialize variance
cond_var := make([]f64, n, context.temp_allocator)
cond_var[0] = true_omega / (1.0 - true_alpha - true_beta)

// Generate returns with Box-Muller transform
for i in 1 ..< n {
    cond_var[i] = true_omega + true_alpha * returns[i-1]^2 + true_beta * cond_var[i-1]
    std_dev := math.sqrt(cond_var[i])
    z := math.sqrt(-2.0 * math.ln(rand.float64())) * math.cos(2.0 * math.PI * rand.float64())
    returns[i] = z * std_dev
}

// Fit GARCH(1,1) model
result := ts.garch_fit(returns, .GARCH, 1, 1, 1000, 1e-6, context.allocator)
fmt.printf("GARCH(1,1): ω=%.6f, α=%.4f, β=%.4f (α+β=%.4f)\n",
    result.params.omega, result.params.alpha[0], result.params.beta[0],
    result.params.alpha[0] + result.params.beta[0])

// Forecast volatility
forecast := ts.garch_forecast(&result, returns, 5, context.allocator)
fmt.printf("5-day volatility forecast: σ²=%.6f, σ=%.6f\n",
    forecast.variance_forecast, forecast.std_forecast)

```
##  Automatic Differentiation (Autograd)
```odin
// Create tensors
A := t.tensor_new(l.matrix_from_f64([[1, 2], [3, 4]]), true, context.allocator)
B := t.tensor_new(l.matrix_from_f64([[5, 6], [7, 8]]), true, context.allocator)

// Build computation graph: C = A + B
C := t.tensor_add(A, B)

// Backward pass
t.tensor_backward(C)

// Check gradients (dC/dA = 1, dC/dB = 1)
fmt.println("Gradient of A (should be all 1.0s):")
for i in 0 ..< 2 {
    for j in 0 ..< 2 {
        fmt.printf("%.1f ", A.grad.data[i * 2 + j]) // Output: 1.0 1.0
    }
    fmt.println()
}

// Matrix multiplication: D = A * B
D := t.tensor_matmul(A, B)
t.tensor_backward(D) // Gradients automatically computed!
```

##  BERT Model (Transformer from Scratch)
```odin
// Build vocabulary
vocab := make(map[u8]int, context.allocator)
vocab['['] = 0  // CLS_TOKEN
vocab[']'] = 1  // SEP_TOKEN
vocab['_'] = 2  // MASK_TOKEN
// Add characters from training text...

// Create BERT model
model := nn.bert_model_new(
    vocab_size=256,
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=2,
    max_seq_len=32,
    context.allocator,
)

// Training loop
for epoch in 0 ..< 2500 {
    nn.adam_zero_grad(&optimizer)
    loss := nn.bert_forward(&model, input_ids, segment_ids)
    t.tensor_backward(loss)
    nn.adam_step(&optimizer)
    if epoch % 100 == 0 {
        fmt.printf("Epoch %d: Loss = %.4f\n", epoch, loss.data[0])
    }
}
```
##   ARIMA Time Series Modeling
```odin
// Generate synthetic ARMA(1,1) data
phi := []f64{0.7}
theta := []f64{0.4}
sigma2 := 0.1
T := 400

y := make([]f64, T, context.allocator)
for t in 1 ..< T {
    e := rand.float64_normal(0.0, math.sqrt(sigma2))
    y[t] = phi[0] * y[t-1] + e + theta[0] * (y[t-1] - phi[0] * y[t-2])
}

// Fit ARIMA(1,0,1)
fit := analytic.arima_fit(y, 1, 0, 1, context.allocator)
fmt.printf("ARIMA(1,0,1): φ=%.3f, θ=%.3f, σ²=%.3f\n",
    fit.phi[0], fit.theta[0], fit.sigma2)

// Forecast 20 steps ahead
forecast := analytic.arima_forecast(y, fit, 1, 0, 1, 20, 0.05, context.allocator)
for i in 0 ..< 5 {
    fmt.printf("t+%d: mean=%.3f [%.3f, %.3f]\n", i+1,
        forecast.mean[i], forecast.lower[i], forecast.upper[i])
}
```
##  Support Vector Machines (SVM)
```odin
n := 100
X := l.matrix_new(f64, n, 2, allocator)
y := make([]f64, n, allocator)

for i in 0 ..< n {
    x0 := rand.float64_normal(0, 1)
    x1 := rand.float64_normal(0, 1)
    X.data[i * 2 + 0] = x0
    X.data[i * 2 + 1] = x1
    // XOR pattern: +1 if x0*x1 > 0, else -1
    tmp: f64
    if x0 + x1 > 0 {
        tmp = 1.0
    } else {
        tmp = -1.0
    }
    y[i] = tmp
}

params := ml.KernelSVMParams {
    C             = 10.0,
    gamma         = 1.0, // RBF width
    kernel_type   = .RBF,
    max_iter      = 500,
    tol           = 1e-3,
    learning_rate = 0.01,
}

model := ml.kernel_svm_fit(&X, y, params, allocator)
defer ml.kernel_svm_free(&model)

fmt.printf(
    "Kernel SVM: %v support vectors, bias=%.3f\n",
    len(model.support_vectors),
    model.bias,
)

// Predict on training data
preds := ml.kernel_svm_predict(&model, &X, allocator)
defer delete(preds, allocator)

// Compute accuracy
correct := 0
for i in 0 ..< n {
    tmp: f64
    if preds[i] > 0 {
        tmp = 1.0
    } else {
        tmp = -1.0
    }
    pred_class := tmp
    if pred_class == y[i] {correct += 1}
}
accuracy := f64(correct) / f64(n)
fmt.printf("Training accuracy (RBF kernel): %.2f%%\n", accuracy * 100)
```

## Q-Learning
```odin
df := net.read_yahoo("AAPL", .Daily, .OneYear, allocator)
defer w.destroy_dataframe(&df)

if df.rows == 0 {
    fmt.println("Aborting test due to empty DataFrame.")
    return
}

fmt.printf("✓ Successfully loaded %d days of OHLCV data.\n", df.rows)

fmt.println("Building CQL dataset from OHLCV features...")
states, actions, rewards, next_states, dones := build_cql_dataset_from_ohlcv(&df, allocator)
defer t.tensor_free(states)
defer t.tensor_free(next_states)
defer t.tensor_free(rewards)
defer t.tensor_free(dones)
defer delete(actions, allocator)

if states == nil {
    fmt.println("Error: Failed to build CQL dataset.")
    return
}

batch_size := states.shape[0]
fmt.printf("Dataset size: %d steps\n", batch_size)

// 1. CQL Configuration
config := ml_fin.CQLConfig {
    state_dim   = 3,
    num_actions = 3,
    hidden_dim  = 32,
    gamma       = 0.95,
    alpha       = 1.0, // Conservative penalty weight
}

// 2. Initialize Q-Networks
q_net := ml_fin.cql_network_new(config, allocator)
defer ml_fin.cql_network_free(&q_net)

// 3. Optimizer
opt := nn.adam_new(0.003, allocator = allocator)
defer nn.adam_free(&opt)

nn.adam_add_param(&opt, q_net.fc1.weights)
nn.adam_add_param(&opt, q_net.fc1.bias)
nn.adam_add_param(&opt, q_net.fc2.weights)
nn.adam_add_param(&opt, q_net.fc2.bias)

fmt.println("\nStarting Offline CQL Training on Real AAPL Data...")
fmt.println("Epoch | Total Loss | TD Loss  | CQL Reg  | Status")
fmt.println("------|------------|----------|----------|-------------------------")

epochs := 50

for epoch := 0; epoch < epochs; epoch += 1 {
    nn.adam_zero_grad(&opt)

    q_values := ml_fin.cql_network_forward(&q_net, states)
    next_q_values := ml_fin.cql_network_forward(&q_net, next_states)

    loss, td_val, cql_val := ml_fin.cql_loss(
        q_values,
        actions,
        rewards,
        next_q_values,
        dones,
        config,
        allocator,
    )

    t.tensor_backward(loss)
    nn.adam_step(&opt)

    total_loss_val := loss.data.data[0]

    status := ""
    if epoch == 0 {
        status = "(Initial)"
    } else if total_loss_val < 2.0 {
        status = "(Converging)"
    }

    fmt.printf(
        " %3d  | %.5f    | %.5f | %.5f | %s\n",
        epoch + 1,
        total_loss_val,
        td_val,
        cql_val,
        status,
    )

    t.tensor_free_graph(loss)
}
```

## Alternative Data
```odin
symbol := "AAPL"
fmt.printf("Target Asset: %s\n", symbol)

// 1. Get Finnhub API Key
api_key := os.get_env("FINNHUB_API_KEY", allocator)
text: string

if api_key != "" {
    now := w.now()
    to_date_str := fmt.tprintf("%04d-%02d-%02d", now.year, int(now.month), now.day)

    // Subtract 30 days safely
    thirty_days_ago := w.add_day_datetime(now, -30)
    from_date_str := fmt.tprintf(
        "%04d-%02d-%02d",
        thirty_days_ago.year,
        int(thirty_days_ago.month),
        thirty_days_ago.day,
    )

    url := fmt.aprintf(
        "https://finnhub.io/api/v1/company-news?symbol=%s&from=%s&to=%s&token=%s",
        symbol,
        from_date_str,
        to_date_str,
        api_key,
        allocator = context.temp_allocator,
    )

    response, ok := net.http_get(url, context.temp_allocator)
    if ok {
        text = response
    }

    // ✅ CRITICAL FIX: Strip UTF-8 BOM if present.
    // This is the #1 cause of "Unexpected_Token" errors in Odin's JSON parser.
    if len(text) >= 3 && text[0] == 0xEF && text[1] == 0xBB && text[2] == 0xBF {
        text = text[3:]
    }

    trimmed_text := strings.trim_space(text)


} 

// 2. Parse the JSON data
root, err := json.parse_string(text, json.DEFAULT_SPECIFICATION, true, allocator)
if err != .None {
    fmt.printf("ERROR: Failed to parse JSON: %v\n", err)
    fmt.println(
        "Hint: If the API returned an error message, the pipeline will fall back to mock data next time.",
    )
    return
}

// Extract sentiment scores and timestamps
NewsItem :: struct {
    datetime:  i64,
    sentiment: f64,
}
items := make([dynamic]NewsItem, 0, allocator)

#partial switch val in root {
case json.Array:
    for item in val {
        #partial switch v in item {
        case json.Object:
            dt: i64 = 0
            sent: f64 = 0.0
            for key, v2 in v {
                if key == "datetime" {
                    #partial switch v3 in v2 {
                    case json.Integer:
                        dt = v3
                    }
                } else if key == "sentiment" {
                    #partial switch v3 in v2 {
                    case json.Float:
                        sent = v3
                    case json.Integer:
                        sent = f64(v3)
                    }
                }
            }
            if dt > 0 {
                append(&items, NewsItem{datetime = dt, sentiment = sent})
            }
        }
    }
}


// 3. Fetch Yahoo Finance Price Data
fmt.println("\n2. Fetching Yahoo Finance Price Data...")
df_yahoo := net.read_yahoo(symbol, .Daily, .OneYear, allocator)
defer w.destroy_dataframe(&df_yahoo)
fmt.printf("Fetched %d rows of price data\n", df_yahoo.rows)

// 4. Align Data (Group sentiment by day and align with daily returns)
fmt.println("\n3. Aligning Sentiment with Daily Returns...")

// Create a map of date -> average sentiment
sentiment_by_date := make(map[i64]f64, allocator)
count_by_date := make(map[i64]int, allocator)

for item in items {
    sentiment_by_date[item.datetime] += item.sentiment
    count_by_date[item.datetime] += 1
}

// For this mock/demo, we will just use the parsed items directly aligned with mock returns
// to prove the Random Forest pipeline works end-to-end.
n_samples := len(items)
X_data := l.matrix_new(f64, n_samples, 1, allocator)
y_data := make([]f64, n_samples, allocator)

for item, i in items {
    avg_sent := sentiment_by_date[item.datetime] / f64(count_by_date[item.datetime])
    X_data.data[i] = avg_sent

    // Mock return: positive sentiment slightly correlates with positive return
    // plus some random noise
    y_data[i] = (avg_sent * 0.01) + (rand.float64_normal(0.0, 0.02))
}

// 5. Train Random Forest Model
fmt.println("\n4. Training Random Forest Model...")
split_idx := max(10, int(f64(n_samples) * 0.8))

X_train := l.matrix_new(f64, split_idx, 1, allocator)
y_train := make([]f64, split_idx, allocator)

test_size := n_samples - split_idx
X_test := l.matrix_new(f64, test_size, 1, allocator)
y_test := make([]f64, test_size, allocator)

for i in 0 ..< split_idx {
    X_train.data[i] = X_data.data[i]
    y_train[i] = y_data[i]
}
for i in 0 ..< test_size {
    X_test.data[i] = X_data.data[split_idx + i]
    y_test[i] = y_data[split_idx + i] // Corrected indexing
}

rf_params := ml.RFParams {
    n_trees     = 50,
    max_depth   = 5,
    min_samples = 5,
    bootstrap   = true,
}

model := ml.rf_fit(&X_train, y_train, rf_params, allocator)
defer ml.rf_free(&model)
fmt.println("   Model trained successfully")

// 6. Inference & Backtest
fmt.println("\n5. Running Inference & Backtest...")
predictions := ml.rf_predict(&model, &X_test, allocator)
defer delete(predictions, allocator)

strategy_returns := make([]f64, test_size, allocator)
defer delete(strategy_returns, allocator)

correct_direction := 0
for i in 0 ..< test_size {
    if predictions[i] > 0.0 {
        strategy_returns[i] = y_test[i]
        if y_test[i] > 0.0 {
            correct_direction += 1
        }
    } else {
        strategy_returns[i] = 0.0
    }
}

sharpe := fin.sharpe_ratio_from_returns(strategy_returns, 0.0, 252.0)
accuracy := f64(correct_direction) / f64(test_size) * 100.0

fmt.printf("   Directional Accuracy: %.2f%%\n", accuracy)
fmt.printf("   Annualized Sharpe Ratio: %.3f\n", sharpe)

// Cleanup
l.matrix_free(&X_data)
delete(y_data, allocator)
l.matrix_free(&X_train)
delete(y_train, allocator)
l.matrix_free(&X_test)
delete(y_test, allocator)
delete(strategy_returns, allocator)
```

## Analytical Option Pricing with Autograd
```odin
fmt.println("\n=== Derivatives Pricing Test ===\n")

// ====================================================================
// Test 1: Black-Scholes Pricing (known values)
// ====================================================================
fmt.println("--- Test 1: Black-Scholes Pricing ---")

// Standard test case: S=100, K=100, T=1, r=5%, σ=20%
S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.20

call_price, call_greeks := fin.price_and_greeks(S, K, T, r, sigma, .Call, allocator)
put_price, put_greeks := fin.price_and_greeks(S, K, T, r, sigma, .Put, allocator)

fmt.printf("ATM Call (S=100, K=100, T=1, r=5%%, σ=20%%):\n")
fmt.printf("  Price: %.4f  (expected ~10.4506)\n", call_price)
fmt.printf("  Delta: %.4f  (expected ~0.6368)\n", call_greeks.delta)
fmt.printf("  Gamma: %.4f  (expected ~0.0187)\n", call_greeks.gamma)
fmt.printf("  Vega:  %.4f  (expected ~0.1870 per 1%%)\n", call_greeks.vega)
fmt.printf("  Theta: %.4f  (expected ~-0.0163 per day)\n", call_greeks.theta)
fmt.printf("  Rho:   %.4f  (expected ~0.0532 per 1%%)\n", call_greeks.rho)

fmt.printf("\nATM Put:\n")
fmt.printf("  Price: %.4f  (expected ~5.5735)\n", put_price)
fmt.printf("  Delta: %.4f  (expected ~-0.3632)\n", put_greeks.delta)

// Put-Call Parity check: C - P = S - K*exp(-r*T)
pcp_rhs := S - K * math.exp(-r * T)
pcp_lhs := call_price - put_price
fmt.printf("\nPut-Call Parity:\n")
fmt.printf("  C - P = %.4f\n", pcp_lhs)
fmt.printf("  S - K*exp(-rT) = %.4f\n", pcp_rhs)
fmt.printf("  Error: %.2e\n", math.abs(pcp_lhs - pcp_rhs))

// ====================================================================
// Test 2: Moneyness (ITM, ATM, OTM)
// ====================================================================
fmt.println("\n--- Test 2: Moneyness ---")

strikes := []f64{80.0, 90.0, 100.0, 110.0, 120.0}
labels := []string{"ITM", "ITM", "ATM", "OTM", "OTM"}

fmt.printf(
    "  %-5s  K=%-6s  Call=%-10s  Put=%-10s  CallΔ=%-8s  PutΔ=%-8s\n",
    "Type",
    "Strike",
    "Price",
    "Price",
    "Delta",
    "Delta",
)

for i in 0 ..< len(strikes) {
    cp, cg := fin.price_and_greeks(S, strikes[i], T, r, sigma, .Call, allocator)
    pp, pg := fin.price_and_greeks(S, strikes[i], T, r, sigma, .Put, allocator)
    fmt.printf(
        "  %-5s  K=%-6.0f  Call=%-10.4f  Put=%-10.4f  CallΔ=%-8.4f  PutΔ=%-8.4f\n",
        labels[i],
        strikes[i],
        cp,
        pp,
        cg.delta,
        pg.delta,
    )
}

// ====================================================================
// Test 3: Implied Volatility
// ====================================================================
fmt.println("\n--- Test 3: Implied Volatility ---")

// Given a market price, find the implied vol
market_call := 10.45 // approximately ATM call price
iv, converged, iters := fin.implied_volatility(market_call, S, K, T, r, .Call, allocator)

fmt.printf("  Market Call Price: %.2f\n", market_call)
fmt.printf("  Implied Vol:       %.4f (%.2f%%)\n", iv, iv * 100)
fmt.printf("  Converged:         %v in %d iterations\n", converged, iters)

// Verify: price at implied vol should match market price
verify_price, _ := fin.price_and_greeks(S, K, T, r, iv, .Call, allocator)
fmt.printf(
    "  Verify Price:      %.4f (error: %.2e)\n",
    verify_price,
    math.abs(verify_price - market_call),
)

// Test with different market prices
fmt.println("\n  Vol Surface Scan:")
test_prices := []f64{5.0, 8.0, 10.45, 13.0, 16.0}
for mp in test_prices {
    iv2, conv2, it2 := fin.implied_volatility(mp, S, K, T, r, .Call, allocator)
    fmt.printf(
        "    Market=%.2f  →  IV=%.4f (%.2f%%)  [%v in %d iter]\n",
        mp,
        iv2,
        iv2 * 100,
        conv2,
        it2,
    )
}

// ====================================================================
// Test 4: Greeks Sanity Checks
// ====================================================================
// ====================================================================
// Test 4: Greeks Sanity Checks (with CORRECT expectations)
// ====================================================================
fmt.println("\n--- Test 4: Greeks Sanity Checks ---")

cg := fin.compute_greeks(S, K, T, r, sigma, .Call, allocator)
pg := fin.compute_greeks(S, K, T, r, sigma, .Put, allocator)

// Call delta should be in (0, 1)
fmt.printf("  Call delta in (0,1): %v (%.4f)\n", cg.delta > 0 && cg.delta < 1, cg.delta)

// Put delta should be in (-1, 0)
fmt.printf("  Put delta in (-1,0): %v (%.4f)\n", pg.delta < 0 && pg.delta > -1, pg.delta)

// Call delta - Put delta ≈ 1 (put-call parity for deltas)
delta_diff := cg.delta - pg.delta
fmt.printf(
    "  CallΔ - PutΔ ≈ 1:  %v (%.4f)\n",
    math.abs(delta_diff - 1.0) < 0.01,
    delta_diff,
)

// Gamma should be positive and same for call/put
fmt.printf("  Gamma > 0:          %v (%.4f)\n", cg.gamma > 0, cg.gamma)
fmt.printf(
    "  Call γ ≈ Put γ:     %v (diff: %.2e)\n",
    math.abs(cg.gamma - pg.gamma) < 1e-6,
    math.abs(cg.gamma - pg.gamma),
)

// Vega should be positive and same for call/put
fmt.printf("  Vega > 0:           %v (%.4f)\n", cg.vega > 0, cg.vega)
fmt.printf(
    "  Call ν ≈ Put ν:     %v (diff: %.2e)\n",
    math.abs(cg.vega - pg.vega) < 1e-6,
    math.abs(cg.vega - pg.vega),
)

// Theta should be NEGATIVE for long calls (time decay)
fmt.printf("  Call theta < 0:     %v (%.4f per day)\n", cg.theta < 0, cg.theta)
fmt.printf("  Put theta < 0:      %v (%.4f per day)\n", pg.theta < 0, pg.theta)

// Verify against closed-form BS Greeks
fmt.println("\n--- Verification Against Closed-Form BS ---")
inv_sqrt_2pi := 0.3989422804014327
sqrt_T := math.sqrt(T)
d1 := (math.ln_f64(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
d2 := d1 - sigma * sqrt_T
phi_d1 := math.exp_f64(-0.5 * d1 * d1) * inv_sqrt_2pi

bs_delta := norm_cdf(d1)
bs_gamma := phi_d1 / (S * sigma * sqrt_T)
bs_vega := S * phi_d1 * sqrt_T / 100.0 // per 1% move
bs_rho := K * T * math.exp(-r * T) * norm_cdf(d2) / 100.0 // per 1% move

fmt.printf("  %-8s  %-12s  %-12s  %-10s\n", "Greek", "Autograd", "Closed-Form", "Error")
fmt.printf(
    "  %-8s  %-12.6f  %-12.6f  %.2e\n",
    "Delta",
    cg.delta,
    bs_delta,
    math.abs(cg.delta - bs_delta),
)
fmt.printf(
    "  %-8s  %-12.6f  %-12.6f  %.2e\n",
    "Gamma",
    cg.gamma,
    bs_gamma,
    math.abs(cg.gamma - bs_gamma),
)
fmt.printf(
    "  %-8s  %-12.6f  %-12.6f  %.2e\n",
    "Vega",
    cg.vega,
    bs_vega,
    math.abs(cg.vega - bs_vega),
)
fmt.printf(
    "  %-8s  %-12.6f  %-12.6f  %.2e\n",
    "Rho",
    cg.rho,
    bs_rho,
    math.abs(cg.rho - bs_rho),
)
fmt.println("\n✓ Derivatives test completed!")
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