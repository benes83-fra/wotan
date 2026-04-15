package main


import "core:mem"
import "core:fmt"
import "core:math"
import "core:math/rand"
import w "./wotan/core"
arima_test :: proc(allocator: mem.Allocator) {
    fmt.println("=== ARIMA TEST ===")

    // Synthetic ARMA(1,1) data
    // y_t = 0.7 y_{t-1} + e_t + 0.4 e_{t-1}
    phi := []f64{0.7}
    theta := []f64{0.4}
    sigma2 := 0.1
    d := 0

    // Generate synthetic data
    T := 200
    y := make([]f64, T, allocator)
    e_prev := 0.0
    y_prev := 0.0

    for t in 0 ..< T {
        e := rand.float64() * math.sqrt_f64(sigma2)
        y[t] = 0.7 * y_prev + e + 0.4 * e_prev
        y_prev = y[t]
        e_prev = e
    }

    // Compute log-likelihood
    ll := w.arima_loglik(y, phi, theta, d, sigma2, allocator)

    fmt.printf("Log-likelihood = %f\n", ll)
    fmt.println("=== END ARIMA TEST ===")
}
arima_fit_test :: proc(allocator: mem.Allocator) {
    fmt.println("=== ARIMA(1,0,1) FIT TEST ===")

    phi_true := 0.7
    theta_true := 0.4
    sigma2_true := 0.1
    d := 0

    T := 300
    y := make([]f64, T, allocator)
    e_prev := 0.0
    y_prev := 0.0

    for t in 0 ..< T {
        e := rand.float64() * math.sqrt_f64(sigma2_true)
        y[t] = phi_true * y_prev + e + theta_true * e_prev
        y_prev = y[t]
        e_prev = e
    }

    fit := w.arima_fit_arma11(y, d, allocator)

    fmt.printf("True phi=%.3f, theta=%.3f, sigma2=%.3f\n", phi_true, theta_true, sigma2_true)
    fmt.printf("Fit  phi=%.3f, theta=%.3f, sigma2=%.3f, loglik=%.3f\n",
        fit.phi, fit.theta, fit.sigma2, fit.loglik)

    fmt.println("=== END ARIMA(1,0,1) FIT TEST ===")
}
