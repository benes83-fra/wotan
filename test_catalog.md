# Test Catalog

This is an overview of the test and example procedures in [`tests/`](tests/). Each file has one or more procedures with the signature `proc(allocator: mem.Allocator)`. They are called from `main.odin` as `test.<name>(allocator)`. To choose which tests run, comment the calls in `main.odin` in or out.

Most tests print their results next to known reference values. They are meant as runnable examples as much as regression tests, and most of them do not assert.

**Markers**

| Marker | Meaning |
|--------|---------|
| 🌐 | Needs network access (Yahoo Finance, Finnhub, GDELT or other HTTP sources) |
| 🖼 | Writes PNG charts to the working directory |
| 📁 | Reads local data or checkpoint files (see [Required files](#required-files)) |

---

## 1. DataFrame core

| File | Procedures | Covers |
|------|------------|--------|
| `slicing_test` | `loc_slice_test`, `loc_many_test`, `loc_from_test`, `loc_until_test`, `loc_mask_test`, `iloc_test`, `materialize_test` | Label- and position-based row selection, mask selection, turning views into copies |
| `indexing_test` | `indexing_test`, `reset_index_test`, `reindex_test`, `set_index_drop_test` | Setting, resetting and reindexing the index |
| `sort_test` | `sort_test` | Sorting by column, ascending and descending |
| `groupby2_test` | `groupby2_test` | GroupBy with single and multiple keys, aggregations |
| `join_test` | `join_test` | Inner, left, right and outer joins; no-match cases; multi-key joins |
| `dataset_test` 📁 | `dataset_test` | Bridge from a DataFrame to ML matrices |

## 2. Import / export

| File | Procedures | Covers |
|------|------------|--------|
| `csv_export_test` 📁 | `csv_export_test` | CSV write and read-back round trip |
| `json_test` 📁 | `json_basic_test`, `jsonl_basic_test`, `json_export_test` | JSON and JSONL loading and export |
| `html_test` 📁 | `html_basic_test`, `html_extended_test`, `html_tags_test`, `html_export_test` (plus helpers `test_extract_tables`, `test_index`, `test_id`, `test_all`) | Extracting HTML tables by index or id, HTML export |
| `excel_test` 📁 | `excel_basic_test`, `zip_smoke_test`, `excel_date_test`, `excel_export_test` | `.xlsx` reading and writing, date cells, the ZIP layer |
| `net_test` 🌐 | `http_get_test`, `yahoo_finance_csv_test`, `yahoo_finance_json_test`, `yahoo_finance_events_test` | HTTP GET via libcurl; Yahoo prices, dividends and splits |

## 3. Linear algebra

| File | Procedures | Covers |
|------|------------|--------|
| `linag_test` | `matrix_test`, `simd_linalg_test`, `rotate_simd_test`, `ols_test`, `ols_full_test`, `correlation_test`, `qr_test`, `svd_test`, `svd_golub_reinsch_test`, `thin_svd_test`, `svd_zero_matrix_test`, `svd_1x1_test`, `svd_rank1_tall_test`, `svd_nearly_rank_deficient_test`, `svd_numeric_dump_test` | Matrix basics, SIMD kernels, OLS, correlation, QR, SVD including edge cases |
| `linalg2_test` | `lu_*` (basic, solve, det, identity, singular, random, inverse, numeric dump), `eigh_diagonal_test`, `eigh_symmetric_test`, `cond2_*`, `eigh_cond_full_test`, `wls_test`, `gls_test`, `ridge_test`, `lasso_test`, `kron_test`, `gls_kron_test` | LU decomposition and inverse, symmetric eigendecomposition, condition numbers, WLS/GLS/Ridge/Lasso, Kronecker products |

## 4. Time series and statistics

| File | Procedures | Covers |
|------|------------|--------|
| `arima_test` | `arima_test`, `arima_fit_test`, `arima_dataframe_test`, `arima_fit_arma22_test`, `mc_arima_pdq_test`, `arima_auto_test`, `auto_arima_stationarity_test`, `sarima_mc_test`, `sarima_resid_diagnostics_quick_test` | ARMA/ARIMA/SARIMA fitting, forecasting and Monte Carlo parameter recovery; auto-ARIMA |
| `arima_test` (diagnostics) | `autocorrelation_test`, `ljung_box_test`, `jarque_bera_test`, `residual_diagnostics_test`, `residuals_test`, `adf_test_block`, `kpss_test_block`, `stationarity_test_block` | ACF/PACF, Ljung–Box, Jarque–Bera, ADF, KPSS |
| `rolling_test` | `rolling_test` | Rolling and EWM correlation |
| `rolling_matrix_test` | `rolling_matrix_test`, `ewm_cov_test`, `ewm_pca_test` | Rolling covariance matrices, EWM covariance, EWM PCA |
| `pca_test` | `pca_test`, `pcr_test` | PCA, rolling PCA, principal component regression |
| `factor_test` | `factor_analysis_test` | Factor analysis, varimax rotation, factor scores |
| `kalman_test` | `kalman_test`, `kalman_control_test`, `kalman_tv_control_test`, `ekf_tiny_test`, `ekf_tiny_rts_test`, `ukf_tiny_test`, `ukf_tiny_rts_test`, `ukf_tiny_control_test`, `ukf_tiny_control_rts_test` | Linear Kalman filter (with control and time-varying inputs), EKF, UKF, RTS smoothing |
| `garch_test` | `garch_test` | GARCH(1,1) parameter recovery on synthetic data, volatility forecast |
| `garch_real_data_test` 🌐🖼 | `garch_real_data_test` | GARCH on SPY with Student-t and GED errors |
| `rs_garch_test` 🌐🖼 | `rs_garch_test` | Regime-switching GARCH on SPY |

## 5. Classical machine learning

| File | Procedures | Covers |
|------|------------|--------|
| `cv_test` | `ridge_cv_test`, `lasso_cv_test` | Ridge and Lasso with cross-validated regularisation |
| `logistic_test` | `logistic_test`, `real_world_logistic_test`, `multiclass_test` | Binary and multiclass logistic regression |
| `naive_bayes_test` | `gnb_test` | Gaussian naive Bayes |
| `kmean_test` | `km_test`, `knn_test` | k-means clustering, k-nearest neighbours |
| `tree_test` | `dt_test`, `rf_test`, `gb_test`, `tree_stats_test`, `vec_sub_simd_test` | Decision trees, random forests, gradient boosting, tree statistics |
| `svm_test` | `svm_test`, `kernel_svm_test`, `svr_test` | Linear SVM, kernel SVM, support vector regression |
| `metrics_test` | `metrics_test` | Classification and regression metrics |
| `model_selection_test` | `model_selection_test` | Train/test splits, cross-validation utilities |
| `grid_search_test` | `grid_search_test`, `extended_grid_search_test`, `random_search_test` | Grid search and random search across models |
| `pipeline_test` | `pipeline_test`, `pipeline_comprehensive_test` | Preprocessing and model pipelines |
| `serialization_test` | `serialization_test`, `tree_serialization_test` | Saving and loading models |

## 6. Autograd and neural networks

| File | Procedures | Covers |
|------|------------|--------|
| `autograd_test` | `autograd_test`, `autograd_mul_test`, `autograd_matmul_test`, `autograd_sum_test`, `autograd_relu_test`, `autograd_bias_test`, `conv2d_test`, `pooling_test` | Gradients through basic ops, Conv2D, pooling |
| `nn_test` 📁 | `nn_test`, `optim_test`, `mse_train_test`, `xor_test`, `classification_test`, `adam_test`, `adam_classification_test`, `dropout_test`, `flexible_network_test`, `sequential_test`, `persistence_test` | Layers, losses, SGD/Adam, XOR, dropout, `Sequential` API, saving weights |
| `optim_test` | `run_optimizer_tests` (runs `test_sgd_slice`, `test_sgd_matrix`, `test_adam_slice`, `test_rmsprop_slice`, `test_schedulers`) | Optimizers in `optimize/` and learning-rate schedulers |
| `mlp_test` | `mlp_test` | Multi-layer perceptron |
| `batchnorm_test` 📁 | `run_all_batchnorm_tests` (forward, backward, sequential, MNIST, train/eval) | BatchNorm2d |
| `mnist_test` 📁 | `mnist_loader_test`, `mnist_cnn_test`, `augmentation_test`, `test_both_models`, `test_augmented_model` | MNIST loader, CNN training, data augmentation |
| `rnn_test` | `rnn_simple_test`, `gru_simple_test`, `lstm_simple_test`, `embedding_simple_test`, `positional_encoding_test`, `attention_simple_test`, `multi_head_attention_test`, `layer_norm_test`, `ffn_simple_test` | Recurrent layers and transformer building blocks |
| `transformer_test` | `transformer_encoder_block_test`, `transformer_decoder_test`, `transformer_reversal_test`, `char_lm_test` | Encoder and decoder stacks, sequence reversal, character-level language model |
| `flash_attention_test` | `flash_attention_test`, `flash_attention_2_test` | Fused attention, causal masking, backward-pass check |
| `gpt_test` | `gpt_full_test`, `gpt_shakespeare_test` | GPT training from scratch |
| `gpt_moe_flash_test` | `gpt_moe_flash_test` | GPT with mixture-of-experts and flash attention (associative recall) |
| `gpt_completion_test` 🌐📁 | `gpt_completion_test` | Text completion with converted DistilGPT-2 weights |
| `bert_test` | `bert_test` | Character-level BERT, masked language modelling |
| `mamba_test` | `mamba_test`, `mamba_learning_test` | Mamba state-space model: gradient check, sequence forecasting |
| `tokenizer_bpe_test` 📁 | `tokenizer_bpe_test` | BPE tokenizer |
| `tokenizer_wordpiece_test` | `tokenizer_wordpiece_test` | WordPiece tokenizer |
| `transfer_learning_test` 📁 | `transfer_learning_test` | Pretraining on one task, fine-tuning on another |
| `gan_test` | `gan_test`, `gan_test_v2` | GAN on sine waves |
| `wgan_test` | `wgan_test` | Wasserstein GAN |
| `vae_test` | `vae_test` | VAE, latent-space interpolation |

## 7. Derivatives pricing

| File | Procedures | Covers |
|------|------------|--------|
| `finance_test` | `derivatives_test`, `portfolio_test`, `constraints_test` | Black–Scholes prices, Greeks from autograd vs closed form, implied volatility; portfolio construction and constraints |
| `american_options_test` | `american_options_test`, `trinomial_tree_test`, `finite_differences_test`, `american_finite_difference_test` | Binomial and trinomial trees, finite-difference methods, early-exercise premium |
| `american_lsm_test` | `american_lsm_test`, `american_asian_lsm_test`, `variance_reduction_test` | Longstaff–Schwartz Monte Carlo, variance reduction |
| `advanced_derivatives_test` | `advanced_derivatives_test`, `exotic_options_test` | Differentiable Monte Carlo; barrier, Asian, basket and lookback options |
| `exotic_pricing_test` 🌐 | `exotic_pricing_test` | Exotics on live spot data with autograd Greeks; skew effects |
| `mc_and_swaps_test` | `unified_mc_test`, `swaps_test` | Shared Monte Carlo engine; swap valuation on a bootstrapped curve |
| `vol_surface_test` | `vol_surface_test` | SABR and Heston calibration to a volatility surface |
| `live_options_calibration` 🌐 | `live_options_calibration_test` | SABR and Heston calibrated to a live option chain |
| `garch_option_test` 🌐🖼 | `garch_options_test` | Option pricing with a GARCH volatility term structure |
| `fx_derivatives_test` | `fx_derivatives_test` | Garman–Kohlhagen, FX smile, digital and barrier digital options |
| `implied_pd_and_fbm_test` | `implied_pd_and_fbm_test`, `rough_volatility_test`, `rough_volatility_pd_test` | Breeden–Litzenberger implied default probability, fractional Brownian motion, rough volatility |

## 8. Fixed income and credit

| File | Procedures | Covers |
|------|------------|--------|
| `yield_curve_test` | `yield_curve_test` | Bootstrapping a zero curve from par swap rates, forward rates |
| `bonds_test` | `bonds_test` | Bond pricing on a bootstrapped curve, sensitivity to curve shifts |
| `callable_bond_test` | `callable_bond_test`, `puttable_bond_test` | Callable and puttable bonds via LSM, OAS solver |
| `ir_derivatives_test` | `ir_derivatives_test`, `ir_derivatives_floor_test`, `ir_derivatives_swaption_strip_test` | Hull–White calibration to caps; floors; swaptions |
| `ir_derivatives_2f_test` | `ir_derivatives_2f_test`, `ir_derivatives_2f_swaption_test`, `bermudan_swaption_test` | Two-factor Hull–White, joint cap and swaption calibration, Bermudan swaptions |
| `cds_bootstrapping_test` | `cds_bootstrapping_test` | Bootstrapping a hazard-rate curve from CDS spreads |
| `credit_derivatives_test` | `credit_derivatives_2008_test`, `credit_derivatives_pipeline_test` | CDO tranches with Gaussian vs Student-t copula, copula fitting |

## 9. Portfolio, risk and backtesting

| File | Procedures | Covers |
|------|------------|--------|
| `risk_test` | `risk_test` | VaR, CVaR, volatility, maximum drawdown, Sharpe ratio |
| `finance_analytics_test` | `finance_analytics_test` | Risk decomposition, rolling analytics, charts |
| `factor_model_test` 🌐 | `factor_model_test` | Fama–French style factor model, covariance reconstruction |
| `hrp_test` 🌐 | `hrp_test` | Hierarchical risk parity |
| `backtest_test` | `backtest_test`, `backtest_portfolio_test` | SMA crossover, mean reversion, breakout and pairs strategies; portfolio backtest |
| `kalman_pairs_test` 🌐🖼 | `kalman_pairs_real_data_test`, `kalman_pairs_real_data_grid_test`, `kalman_pairs_semiconductor_test`, `kalman_pairs_semiconductor_adf_test`, `kalman_pairs_auto_select_test` | Kalman-filter pairs trading, grid search, cointegration screening, automatic pair selection |
| `optimal_execution_test` | `optimal_execution_test` | Almgren–Chriss liquidation, Avellaneda–Stoikov market making |
| `pdpm_test` 🖼 | `pdpm_test`, `pdpm2_test` | Dybvig payoff distribution pricing, stochastic discount factors |
| `pdpm_real_data_test` 🖼📁 | `pdpm_real_data_test` | PDPM on stored SPY prices and options |
| `pdpm_multifactor_test` 🌐🖼 | `pdpm_multifactor_test` | Two-factor SDF calibrated to a live option chain |

## 10. Machine learning for finance

| File | Procedures | Covers |
|------|------------|--------|
| `deep_hedging_test` 📁 | `deep_hedging_test` | Deep hedging of a European call vs the Black–Scholes delta; model save/load |
| `deep_hedging_exotic_test` 📁 | `deep_hedging_exotic_test` | Deep hedging of Asian and barrier options |
| `deep_hedging_portfolio_test` 🌐 | `deep_hedging_portfolio_test` | Multi-asset deep hedging with correlations from real data |
| `deep_hedging_real_data_test` 🌐🖼 | `deep_hedging_real_data_test` | Deep hedging trained on a live option chain |
| `deep_bsde_test` | `deep_bsde_test`, `deep_bsde_european_call_test`, `deep_bsde_100d_test` | Deep BSDE solver: 1-D validation, 100-D max-call benchmark |
| `deep_calibration_test` 🌐 | `deep_calibration_test` | Neural network Heston calibration, fast inference, AAPL data |
| `deeplob_test` | `deeplob_test`, `deeplob_talkative_test` | DeepLOB on a synthetic limit order book |
| `deeplob_real_data_test` 🌐📁 | `deeplob_real_data_test`, `deeplob_real_data_sharpe_test` | DeepLOB on real data, Sharpe-ratio loss |
| `timegan_test` 🌐 | `timegan_test` | Synthetic market data with TimeGAN |
| `neural_sdf_test` 🌐📁 | `neural_sdf_test` | Neural stochastic discount factor (Euler equation loss) |
| `gat_cross_sectional_test` 🌐 | `gat_cross_sectional_test`, `gat_real_world_test`, `gat_real_world_cv_test` | Graph attention networks for cross-sectional returns, rank IC |
| `hmm_regime_test` 🌐 | `hmm_regime_test` | Hidden Markov model regimes (Baum–Welch, Viterbi) |
| `volatility_forecasting_test` 🌐 | `volatility_forecasting_test`, `lstm_volatility_test`, `lstm_volatility_real_data_test` | Volatility forecasting with neural models and LSTMs |
| `ensemble_volatility_test` 🌐 | `ensemble_volatility_test`, `vrp_signal_test`, `vrp_backtest_test` | GARCH plus sequence-model ensemble, variance risk premium signal and backtest |
| `volatility_arbitrage_test` 🌐 | `volatility_arbitrage_test` | Implied vs forecast realised volatility |
| `dispersion_trading_test` 🌐 | `dispersion_trading_test` | Dispersion trading: realised correlation, factor risk |
| `ppo_test` 🌐 | `ppo_trading_test`, `ppo_trading_real_data_test`, `walk_forward_validation_test` | PPO trading agent on synthetic and real data, walk-forward validation |
| `ppo_multi_asset_test` 🌐 | `ppo_multi_asset_test`, `ppo_multi_asset_wf_test` | Multi-asset PPO (SPY vs QQQ) |
| `cql_trading_test` | `cql_trading_test` | Offline conservative Q-learning |
| `cql_real_data_test` 🌐 | `cql_real_data_trading_test` | CQL on Yahoo OHLCV data |
| `orchestrator_test` 🌐 | `orchestrator_test` | Closed-loop trading orchestrator combining several models |
| `event_study_test` 🌐🖼📁 | `event_study_test`, `event_study_nlp_test`, `event_study_tokenizer_nlp_test` | Cumulative abnormal returns, FinBERT sentiment events |
| `alt_data_gdelt_pipeline_test` 🌐 | `alt_data_gdelt_pipeline_test`, `alt_data_finnhub_pipeline_test` | News sentiment (GDELT, Finnhub) to random forest to backtest; falls back to mock data |

## 11. Behavioural finance

| File | Procedures | Covers |
|------|------------|--------|
| `prospect_theory_test` | `prospect_theory_test` | Prospect theory value and weighting functions as differentiable tensor ops |
| `prospect_vol_smile_test` | `prospect_vol_smile_test` | Volatility smile produced by prospect theory preferences |
| `prospect_portfolio_test` 🌐 | `prospect_portfolio_test` | Portfolio optimisation under prospect theory |
| `prospect_ppo_test` 🌐 | `prospect_ppo_test` | PPO agent with prospect theory rewards (disposition effect) |
| `market_implied_psychology_test` 🌐🖼 | `market_implied_psychology_test`, `real_market_psychology_test` | Calibrating prospect theory parameters to option prices, including live SPY options |
| `behavioral_heston_test` 🌐🖼 | `behavioral_heston_test` | Heston model with a behavioural residual layer |
| `behavioral_backtester_test` 🌐🖼 | `behavioral_backtester_test`, `historical_gamma_backtest_test` | Backtesting the market-implied psychology signal |
| `anchoring_backtest_test` 🌐🖼 | `anchoring_backtest_test` | 52-week-high anchoring anomaly (George & Hwang, 2004) |
| `bpt_test` 🌐 | `bpt_test`, `bpt_v2_test` | Behavioural portfolio theory: mental accounting layers, sleeve mandates |

## 12. Plotting

| File | Procedures | Covers |
|------|------------|--------|
| `plot_test` 🖼 | `plot_test`, `line_plot_test`, `bar_chart_test`, `multi_line_test`, `heatmap_test` | Scatter, histogram, line (solid, dashed, dotted), bar, multi-line and heatmap PNGs |

---

## Required files

Tests marked 📁 read some of these files from the working directory:

| File(s) | Used by | How to get it |
|---------|---------|---------------|
| `people.csv`, `people_dates.csv`, `test.csv`, `train.csv` | DataFrame and ML examples | Included in the repository |
| `example_data.json`, `example_data2.jsonl`, `example.xlsx`, `datetime.xlsx`, `example_table.html`, `tag_table.html`, `test_multi.html` | Import tests | Included in the repository |
| `data/spy_prices.csv`, `data/spy_options.csv` | `pdpm_real_data_test` | Included in the repository |
| `data/mnist/*` | `mnist_test`, `batchnorm_test` | Included in the repository (or run `download_mnist.sh`) |
| `finbert_checkpoint.bin`, `bert_vocab.txt` | `event_study_test` (FinBERT) | `python convert_bert.py ProsusAI/finbert finbert_checkpoint.bin` |
| `distilgpt2_checkpoint.bin` | `gpt_completion_test` | `python convert_gpt2.py` |
| `vocab.json`, `merges.txt` | `gpt_completion_test` | Downloaded from Hugging Face automatically if missing |
| `mnist_98_91.bin`, `mnist_augmented.bin`, `task_a_checkpoint.bin`, `neural_sdf_model.bin` | MNIST, transfer learning and neural SDF tests | Written by earlier runs of the same tests |

Without the converted checkpoints, the FinBERT and GPT-2 tests still run, but with randomly initialised weights.

Export tests write `export_test.*`, `out.json` and `out.jsonl`. Persistence tests write temporary `.bin` files and remove them afterwards.
