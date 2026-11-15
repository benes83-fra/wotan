// Wotan: data analysis, statistics, machine learning and quantitative finance.
//
// This root package has no declarations of its own. Odin does not re-export
// packages, so import the subpackages you need directly, for example:
//
//     import w "wotan/core"
//     import io "wotan/importer"
//     import fin "wotan/finance"
//
// The packages:
//     core                 the DataFrame and typed columns
//     importer, exporter   CSV, JSON, JSONL, HTML and Excel in and out
//     net                  HTTP, Yahoo Finance prices and option chains
//     plot                 PNG charts
//     linalg               matrices, factorizations, eigenvalues, SVD
//     tensor               reverse-mode autograd
//     nn                   layers, models, losses, optimizers, checkpoints
//     optimize             SGD, Adam, RMSProp, L-BFGS, least squares
//     analytics            time series, statistical tests, GARCH, Kalman filters
//     analytics/ML         classical machine learning
//     finance              pricing, curves, credit, portfolios, risk, backtests
//     ml_finance           machine learning for finance
//     behavioral_finance   prospect theory, sentiment, herding
//
// docs/main_doc.md is the entry point to the guides.
package wotan
