package core

import "core:mem"
import "core:math"
arima_state_space :: proc(
    phi, theta: []f64,
    d: int,
    sigma2: f64,
    allocator := context.allocator,
) -> (F, Q, P0: []f64, H: []f64, R: []f64, x0: []f64, N: int) {
    p := len(phi)
    q := len(theta)
    N = max(p, q + 1)

    // Allocate flat arrays
    F = make([]f64, N * N, allocator)
    Q = make([]f64, N * N, allocator)
    P0 = make([]f64, N * N, allocator)
    H = make([]f64, N, allocator)     
    R = make([]f64, 1, allocator)    
    x0 = make([]f64, N, allocator)

    // --- Build F ---
    // Access F[row, col] as F[row * N + col]
    for i in 0 ..< p {
        F[0 * N + i] = phi[i]
    }
    for j in 0 ..< q {
        // Corrected indexing: first row, columns starting after p
        F[0 * N + (p + j)] = theta[j]
    }

    // Shift AR lags
    for i in 1 ..< p {
        F[i * N + (i - 1)] = 1.0
    }

    // Shift MA lags
    for j in 1 ..< q + 1 {
        row := p + j - 1
        col := p + j - 2
        if row < N && col < N {
            F[row * N + col] = 1.0
        }
    }

    // --- Observation matrix ---
    H[0] = 1.0 // H is 1xN, so it's just a vector

    // --- Noise matrices ---
    Q[0 * N + 0] = sigma2
    R[0] = 0.0

    // --- Initial covariance ---
    for i in 0 ..< N {
        P0[i * N + i] = 1e6
    }

    return
}
// difference(series, d) -> differenced series
difference :: proc(series: []f64, d: int, allocator := context.allocator) -> []f64 {
    if d <= 0 {
        // No differencing
        out := make([]f64, len(series), allocator)
        for i in 0 ..< len(series) {
            out[i] = series[i]
        }
        return out
    }

    // First-order difference
    diff := make([]f64, len(series)-1, allocator)
    for i in 1 ..< len(series) {
        diff[i-1] = series[i] - series[i-1]
    }

    // Higher-order differencing: recurse
    for k in 1 ..< d {
        if len(diff) <= 1 {
            break
        }
        next := make([]f64, len(diff)-1, allocator)
        for i in 1 ..< len(diff) {
            next[i-1] = diff[i] - diff[i-1]
        }
        diff = next
    }

    return diff
}
difference_with_history :: proc(
    series: []f64,
    d: int,
    allocator := context.allocator,
) -> (diff: []f64, history: []f64) {

    if d <= 0 {
        diff = make([]f64, len(series), allocator)
        for i in 0 ..< len(series) {
            diff[i] = series[i]
        }
        history = make([]f64, 0, allocator)
        return
    }

    // Save last d values for inverse differencing
    history = make([]f64, d, allocator)
    for i in 0 ..< d {
        history[i] = series[i]
    }

    diff = difference(series, d, allocator)
    return
}
inverse_difference :: proc(
    diff: []f64,
    history: []f64,
    d: int,
    allocator := context.allocator,
) -> []f64 {

    if d <= 0 {
        out := make([]f64, len(diff), allocator)
        for i in 0 ..< len(diff) {
            out[i] = diff[i]
        }
        return out
    }

    // Start with the first d original values
    out := make([]f64, len(diff) + d, allocator)
    for i in 0 ..< d {
        out[i] = history[i]
    }

    // First integration
    for i in 0 ..< len(diff) {
        out[d + i] = out[d + i - 1] + diff[i]
    }

    // Higher-order integration
    for k in 1 ..< d {
        for i in d ..< len(out) {
            out[i] = out[i] + out[i-1]
        }
    }

    return out
}
arima_loglik :: proc(
    y: []f64,
    phi, theta: []f64,
    d: int,
    sigma2: f64,
    allocator := context.allocator,
) -> f64 {
    // TODO: apply differencing if d > 0 before calling this
    F, Q, P0, H, R, x0, N := arima_state_space(phi, theta, d, sigma2, allocator)
    return kalman_loglik_scalar(y, F, Q, P0, H, R, x0, N)
}
ArimaObjectiveCtx :: struct {
    y: []f64,
    d: int,
    allocator: mem.Allocator,
}

arima_neg_loglik_obj :: proc(params: []f64, ctx: rawptr) -> f64 {
    data := (^ArimaObjectiveCtx)(ctx)

    phi_val   := params[0]
    theta_val := params[1]
    log_sig2  := params[2]

    // enforce some crude stability / bounds
    if math.abs(phi_val) > 0.99 {
        return 1e9
    }
    if math.abs(theta_val) > 0.99 {
        return 1e9
    }

    sigma2 := math.exp(log_sig2)
    if sigma2 <= 0.0 || sigma2 > 1000.0 {
        return 1e9
    }

    phi   := []f64{phi_val}
    theta := []f64{theta_val}

    // differencing if needed
    y_eff := data.y
    if data.d > 0 {
        y_eff = difference(data.y, data.d, data.allocator)
        if len(y_eff) < 5 {
            return 1e9
        }
    }

    ll := arima_loglik(y_eff, phi, theta, data.d, sigma2, data.allocator)

    // Nelder–Mead minimizes, so return negative log-likelihood
    return -ll
}

ArimaFitResult :: struct {
    phi: f64,
    theta: f64,
    sigma2: f64,
    loglik: f64,
    converged: bool,
}

arima_fit_arma11 :: proc(
    y: []f64,
    d: int,
    allocator := context.allocator,
) -> ArimaFitResult {
    ctx := ArimaObjectiveCtx{
        y = y,
        d = d,
        allocator = allocator,
    }

    // initial guess
    x0 := []f64{0.5, 0.2, math.ln_f64(0.1)}

    max_iter := 500
    tol := 1e-6

    best_x, best_f := nelder_mead(
        arima_neg_loglik_obj,
        rawptr(&ctx),
        x0,
        max_iter,
        tol,
        allocator,
    )

    res: ArimaFitResult
    res.phi = best_x[0]
    res.theta = best_x[1]
    res.sigma2 = math.exp_f64(best_x[2])
    res.loglik = -best_f
    res.converged = true // crude; you can refine with iteration checks

    return res
}
