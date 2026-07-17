package main

import test "./tests"
import w "./wotan/core"
import csv "./wotan/importer"
import "core:fmt"
import "core:mem"
import "core:strings"

main :: proc() {
	when ODIN_DEBUG {
		default_allocator := context.allocator
		tracking_allocator: mem.Tracking_Allocator
		mem.tracking_allocator_init(&tracking_allocator, default_allocator)
		context.allocator = mem.tracking_allocator(&tracking_allocator)

		defer {
			if len(tracking_allocator.allocation_map) > 0 {
				for _, entry in tracking_allocator.allocation_map {
					fmt.eprintf("%v leaked %v bytes\n", entry.location, entry.size)

				}
			}
			mem.tracking_allocator_destroy(&tracking_allocator)
		}

	}
	defer free_all(context.temp_allocator)
	df := w.dataframe_new()

	col_age := w.column_new("age", .Int, 4)
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

	w.df_head(&df, 5)


	w.dataframe_print(&df)

	s := w.df_series(&df, "age")
	v, null := w.series_at_int(&s, 3)
	fmt.printf("age[2] = %d (null=%v)\n", v, null)

	w.destroy_dataframe(&df)
	types := []w.ColumnType{.Int, .String}
	df2 := csv.csv_load("people.csv", types)

	fmt.println("Here not here yet")
	w.dataframe_print(&df2)
	w.df_head(&df2, 5)
	w.destroy_dataframe(&df2)

	df3 := csv.csv_load("people_dates.csv")
	w.dataframe_print(&df3)
	w.df_head(&df3, 10)
	w.dataframe_pretty_print(&df3)


	df4 := w.dataframe_new()
	c1 := w.column_new("age", .Int, 4)
	w.append_int(&c1, 10)
	w.append_int(&c1, 20)
	w.append_int(&c1, 30)
	w.append_int(&c1, 40)
	w.add_column(&df4, c1)

	slice := w.dataframe_slice_rows_copy(&df4, 1, 3)
	w.df_head(&slice, 10) // should print rows 1 and 2 (20, 30)
	w.destroy_dataframe(&slice)

	fmt.println("Showing column selection:")
	df5 := w.dataframe_slice_rows(&df4, 1, 2, false)
	w.df_head(&df5, 5)
	w.destroy_dataframe(&df4)
	w.destroy_dataframe(&df5)
	df_age := w.dataframe_select_columns(&df3, []string{"age"}, false)
	w.df_head(&df_age, 5)

	w.destroy_dataframe(&df3)
	w.destroy_dataframe(&df_age)


	fmt.println("Showing Boolean filtering:")
	fmt.println("Without filter:")
	df6 := csv.csv_load("people_dates.csv")
	w.dataframe_pretty_print(&df6, 20)
	fmt.println("With filter:")
	// assume "active" is Bool
	df_active := w.dataframe_filter_bool_column(&df6, "active")
	w.dataframe_pretty_print(&df_active, 20)

	w.destroy_dataframe(&df6)
	w.destroy_dataframe(&df_active)
	fmt.println("Showing column filtering:")
	df7 := csv.csv_load("people_dates.csv")

	df_active2 := w.filter(&df7, "active")
	w.dataframe_pretty_print(&df_active2, 20)

	w.destroy_dataframe(&df7)
	w.destroy_dataframe(&df_active2)

	fmt.println("Filtering more complex booleans")
	df8 := csv.csv_load("people_dates.csv")
	c_mask := (w.column_lt(w.column(&df8, "age"), 31))
	bmask := w.column_mask(&c_mask)
	mask2 := w.column_mask(w.column(&df8, "active"))
	mask := w.mask_and(mask2, bmask)
	df_active3 := w.filter(&df8, mask)
	w.dataframe_pretty_print(&df_active3, 20)
	delete(mask)

	fmt.println("Using wobei/where with masks")
	//memory safe implementation. The syntax is more flexible otherwise, but there is some risk of leaks
	m1 := w.mask_lt(w.column(&df8, "age"), 31)
	m2 := w.column_mask(w.column(&df8, "active"))
	mask = w.and(m1, m2)
	delete(m1)
	delete(m2)

	df9 := w.wobei(&df8, mask)
	w.dataframe_pretty_print(&df9, 20)
	delete(mask)
	delete(mask2)
	delete(bmask)
	defer w.destroy_column(&c_mask)

	fmt.println("A simple select example")
	exprs := []w.Select_Expr {
		w.col_expr("age", w.column(&df8, "age")),
		w.add_expr("age_plus_10", w.column(&df8, "age"), 10),
		w.mask_expr("is_young", w.mask_lt(w.column(&df8, "age"), 30)),
	}

	df10 := w.select(&df8, exprs)
	w.dataframe_pretty_print(&df10, 20)

	fmt.println("Apply expressions on select:")
	exprs2 := []w.Select_Expr {
		w.col_expr("age", w.column(&df8, "age")),
		w.apply_expr("age_plus_5", w.column(&df8, "age"), proc(x: int) -> int {
			return x + 5
		}),
		w.apply_expr("upper_name", w.column(&df8, "name"), proc(s: string) -> string {
			return strings.to_upper(s, context.temp_allocator)
		}),
		w.apply_expr("is_even", w.column(&df8, "age"), proc(x: bool) -> bool {
			return x
		}),
		w.div_expr("Breaking Salaries", w.column(&df8, "salary"), 10),
		w.conv_int_to_f64_expr("Floating Salaries", w.column(&df8, "salary")),
		w.conv_expr("Conv Age", w.column(&df8, "age"), "float"),
		w.conv_expr("Birthday madness", w.column(&df8, "age"), "float"),
		w.col_expr("datetime", w.column(&df8, "birthday")),
	}
	df11 := w.select(&df8, exprs2)
	w.dataframe_pretty_print(&df11, 20)

	fmt.println("Some test with Dates")
	date1 := w.Date{2020, 2, 7}
	date2 := w.Date{2024, 2, 6}
	days := w.get_date_day_diffs(date1, date2)
	fmt.println("A lot of days:", days)
	for i in 0 ..= 5 {
		date1 = w.add_month_date(date1, -7)
		fmt.printf("Date incremeneted to %v\n", date1)
	}

	for i in 0 ..= 20 {
		date2 = w.add_day_date(date2, -100)
		fmt.printf("Date incremeneted to %v\n", date2)
	}
	fmt.println("Now we test some times")
	time1 := w.Time{16, 2, 58}

	for i in 0 ..= 20 {
		time1 = w.add_seconds_time(time1, -600)
		fmt.printf("Time increased to %v\n", time1)
	}
	dt := w.Datetime{1983, 7, 20, 13, 13, 13}
	for i in 0 ..= 20 {
		dt = w.add_hours_datetime(dt, -49)
		fmt.printf("Time elapsed %v \n", dt)
	}
	fmt.println(w.now())
	gdf := w.groupby(&df8, []string{"age"})
	exprs3 := []w.Agg_Expr {
		w.count("n"),
		w.sum_agg("avg_age", w.column(&df8, "age")),
		w.avg_agg("total_salary", w.column(&df8, "salary")),
	}
	out := w.agg(&gdf, exprs3)
	w.dataframe_pretty_print(&out, 20)


	w.destroy_grouped_dataframe(&gdf)
	w.destroy_dataframe(&out)

	// --- Insert this right before calling groupby(&df8, ...) ---

	people := w.dataframe_new()
	w.add_column(&people, w.column_from_ints("id", []int{1, 2, 3}))
	w.add_column(&people, w.column_from_strings("name", []string{"Alice", "Bob", "Charlie"}))
	w.add_column(&people, w.column_from_ints("age", []int{30, 20, 40}))
	people.rows = 3
	w.dataframe_pretty_print(&people, 20)
	salary := w.dataframe_new()
	w.add_column(&salary, w.column_from_ints("id", []int{1, 2, 4}))
	w.add_column(&salary, w.column_from_floats("salary", []f64{50000.00, 42000.00, 90000.00}))
	salary.rows = 3
	w.dataframe_pretty_print(&salary, 20)

	joined := w.join_single(int, &people, &salary, "id", .Outer, context.temp_allocator)

	fmt.println("Joined DF1:")

	w.dataframe_pretty_print(&joined, 20)


	left := w.dataframe_new()
	w.add_column(&left, w.column_from_ints("id", []int{1, 1, 2}))
	w.add_column(&left, w.column_from_ints("dept", []int{10, 20, 10}))
	w.add_column(&left, w.column_from_strings("name", []string{"Alice", "Bob", "Carol"}))
	left.rows = 3

	right := w.dataframe_new()
	w.add_column(&right, w.column_from_ints("id", []int{1, 2, 1}))
	w.add_column(&right, w.column_from_ints("dept", []int{10, 10, 30}))
	w.add_column(&right, w.column_from_floats("salary", []f64{50000, 60000, 70000}))
	right.rows = 3

	joined2 := w.join(&left, &right, []string{"id", "dept"}, .Inner, context.temp_allocator)

	fmt.println("Joined DF2:")
	w.dataframe_pretty_print(&joined2, 20)

	left2 := w.dataframe_new()
	w.add_column(&left2, w.column_from_ints("id", []int{1, 1, 2}))
	w.add_column(&left2, w.column_from_strings("country", []string{"DE", "US", "DE"}))
	w.add_column(&left2, w.column_from_strings("name", []string{"Alice", "Bob", "Carol"}))
	left2.rows = 3

	right2 := w.dataframe_new()
	w.add_column(&right2, w.column_from_ints("id", []int{1, 2, 1}))
	w.add_column(&right2, w.column_from_strings("country", []string{"DE", "DE", "FR"}))
	w.add_column(&right2, w.column_from_floats("salary", []f64{50000, 60000, 70000}))
	right2.rows = 3
	w.dataframe_pretty_print(&left2, 20)
	w.dataframe_pretty_print(&right2, 20)

	joined3 := w.join(
		&left2,
		&right2,
		[]string{"id", "country"},
		.Inner,
		allocator = context.temp_allocator,
	)
	test.join_test()


	w.dataframe_pretty_print(&joined3, 20)
	dfx := w.df_from(
		w.column_from_ints("id", []int{1, 2, 3}),
		w.column_from_strings("name", []string{"A", "B", "C"}),
		w.column_from_floats("salary", []f64{10, 20, 30}),
	)
	w.dataframe_pretty_print(&dfx, 20)

	test.groupby2_test()


	test.pca_test(context.temp_allocator)
	test.ewm_pca_test(context.temp_allocator)
	test.ewm_cov_test(context.temp_allocator)
	test.rolling_test(context.temp_allocator)
	test.rolling_matrix_test(context.temp_allocator)
	test.kalman_test(context.temp_allocator)
	test.kalman_control_test(context.temp_allocator)
	test.kalman_tv_control_test(context.temp_allocator)
	test.ekf_tiny_test(context.temp_allocator)
	test.ekf_tiny_rts_test(context.temp_allocator)
	test.ukf_tiny_test(context.temp_allocator)
	test.ukf_tiny_rts_test(context.temp_allocator)
	test.ukf_tiny_control_test(context.temp_allocator)
	test.ukf_tiny_control_rts_test(context.temp_allocator)
	test.arima_test(context.temp_allocator)
	test.arima_fit_test(context.temp_allocator)
	test.arima_dataframe_test(context.temp_allocator)
	//mc_arima_arma11(context.temp_allocator, 200, 300)
	test.arima_fit_arma22_test(context.temp_allocator)
	// mc_arima_arma22(context.temp_allocator, 200, 300)
	// mc_arima_arma_pq([]f64{0.6, -0.1, 0.2}, []f64{0.5}, 0.1, 200, 300, context.temp_allocator)
	//mc_arima_pdq_test(context.temp_allocator)
	//arima_auto_test(context.temp_allocator)
	test.autocorrelation_test(context.temp_allocator)
	test.ljung_box_test(context.temp_allocator)
	test.jarque_bera_test(context.temp_allocator)
	test.residual_diagnostics_test(context.temp_allocator)
	test.residuals_test(context.temp_allocator)
	test.adf_test_block(context.temp_allocator)
	test.kpss_test_block(context.temp_allocator)
	test.stationarity_test_block(context.temp_allocator)
	//auto_arima_stationarity_test(context.temp_allocator)
	//mc_arima_pdq_test(context.temp_allocator)
	// sarima_mc_test(
	// 	200,
	// 	300,
	// 	[]f64{0.5},
	// 	1,
	// 	[]f64{0.4},
	// 	[]f64{0.3},
	// 	1,
	// 	[]f64{0.2},
	// 	12,
	// 	0.1,
	// 	context.temp_allocator,
	// )
	test.sarima_light_demo(context.temp_allocator)
	test.sarima_resid_diagnostics_quick_test(context.temp_allocator)

	test.json_basic_test(context.temp_allocator)
	test.jsonl_basic_test(context.temp_allocator)
	test.json_export_test(context.temp_allocator)
	test.html_basic_test(context.temp_allocator)
	test.html_extended_test(context.temp_allocator)
	test.html_tags_test(context.temp_allocator)
	test.html_export_test(context.temp_allocator)
	test.zip_smoke_test(context.temp_allocator)
	test.excel_basic_test(context.temp_allocator)
	test.excel_date_test(context.temp_allocator)
	test.excel_export_test(context.temp_allocator)
	test.csv_export_test(context.temp_allocator)
	test.http_get_test(context.temp_allocator)
	test.yahoo_finance_json_test(context.temp_allocator)
	test.yahoo_finance_events_test(context.temp_allocator)
	test.sort_test()
	test.loc_slice_test(context.temp_allocator)
	test.loc_many_test(context.temp_allocator)
	test.loc_from_test(context.temp_allocator)
	test.loc_until_test((context.temp_allocator))
	test.loc_mask_test(context.temp_allocator)
	test.iloc_test(context.temp_allocator)
	test.materialize_test(context.temp_allocator)
	test.indexing_test(context.temp_allocator)
	test.reset_index_test(context.temp_allocator)
	test.reindex_test(context.temp_allocator)
	test.set_index_drop_test(context.temp_allocator)
	test.matrix_test(context.temp_allocator)
	test.simd_linalg_test(context.temp_allocator)
	test.ols_test(context.temp_allocator)
	test.ols_full_test(context.temp_allocator)
	test.correlation_test(context.temp_allocator)
	test.qr_test(context.temp_allocator)
	test.rotate_simd_test(context.temp_allocator)
	test.svd_test(context.temp_allocator)
	test.svd_golub_reinsch_test(context.temp_allocator)
	test.thin_svd_test(context.temp_allocator)
	test.svd_zero_matrix_test(context.temp_allocator)
	test.svd_wide_matrix_test(context.temp_allocator)
	test.svd_rank1_tall_test(context.temp_allocator)
	test.svd_1x1_test(context.temp_allocator)
	test.svd_nearly_rank_deficient_test(context.temp_allocator)
	test.svd_numeric_dump_test(context.temp_allocator)
	test.lu_basic_test(context.temp_allocator)
	test.lu_identity_test(context.temp_allocator)
	test.lu_singular_test(context.temp_allocator)
	test.lu_det_test(context.temp_allocator)
	test.lu_numeric_dump_test(context.temp_allocator)
	test.lu_solve_test(context.temp_allocator)
	test.lu_inverse_basic_test(context.temp_allocator)
	test.lu_inverse_identity_test(context.temp_allocator)
	test.lu_inverse_singular_test(context.temp_allocator)
	test.eigh_diagonal_test(context.temp_allocator)
	test.eigh_symmetric_test(context.temp_allocator)
	test.eigh_cond_full_test(context.temp_allocator)
	test.cond2_sym_identity_test(context.temp_allocator)
	test.cond2_svd_rank_deficient_test(context.temp_allocator)
	test.cond2_svd_identity_test(context.temp_allocator)
	test.cond2_sym_spd_test(context.temp_allocator)
	test.wls_test(context.temp_allocator)
	test.gls_test(context.temp_allocator)
	test.ridge_test(context.temp_allocator)
	test.lasso_test(context.temp_allocator)
	test.kron_test(context.temp_allocator)
	test.gls_kron_test(context.temp_allocator)
	test.ridge_cv_test(context.temp_allocator)
	test.lasso_cv_test(context.temp_allocator)
	test.dt_test(context.temp_allocator)
	test.rf_test(context.temp_allocator)
	test.gb_test(context.temp_allocator)
	test.vec_sub_simd_test(context.temp_allocator)
	test.tree_stats_test(context.temp_allocator)
	test.km_test(context.temp_allocator)
	test.svm_test(context.temp_allocator)
	test.kernel_svm_test(context.temp_allocator)
	test.svr_test(context.temp_allocator)
	test.run_optimizer_tests(context.temp_allocator)
	test.logistic_test(context.temp_allocator)
	test.real_world_logistic_test(context.temp_allocator)
	test.multiclass_test(context.temp_allocator)
	test.knn_test(context.temp_allocator)
	test.pcr_test(context.temp_allocator)
	test.gnb_test(context.temp_allocator)
	test.metrics_test(context.temp_allocator)
	test.model_selection_test(context.temp_allocator)
	test.grid_search_test(context.temp_allocator)
	test.extended_grid_search_test(context.temp_allocator)
	test.pipeline_test(context.temp_allocator)
	test.pipeline_comprehensive_test(context.temp_allocator)
	test.serialization_test(context.temp_allocator)
	test.mlp_test(context.temp_allocator)
	test.dataset_test(context.temp_allocator)
	test.tree_serialization_test(context.temp_allocator)
	test.random_search_test(context.temp_allocator)
	test.plot_test(context.temp_allocator)
	test.line_plot_test(context.temp_allocator)
	test.bar_chart_test(context.temp_allocator)
	test.multi_line_test(context.temp_allocator)
	test.heatmap_test(context.temp_allocator)
	test.autograd_test(context.temp_allocator)
	test.autograd_mul_test(context.temp_allocator)
	test.autograd_matmul_test(context.temp_allocator)
	test.autograd_sum_test(context.temp_allocator)
	test.autograd_relu_test(context.temp_allocator)
	test.autograd_bias_test(context.temp_allocator)
	test.nn_test(context.temp_allocator)
	test.optim_test(context.temp_allocator)
	test.mse_train_test(context.temp_allocator)
	test.xor_test(context.temp_allocator)
	test.classification_test(context.temp_allocator)
	test.adam_test(context.temp_allocator)
	test.adam_classification_test(context.temp_allocator)
	test.dropout_test(context.temp_allocator)
	test.conv2d_test(context.temp_allocator)
	test.pooling_test(context.temp_allocator)
	test.flexible_network_test(context.temp_allocator)
	test.sequential_test(context.temp_allocator)
	test.mnist_loader_test(context.temp_allocator)
	test.persistence_test(context.temp_allocator)
	test.augmentation_test(context.temp_allocator)
	// test.mnist_cnn_test(context.temp_allocator)
	// test.test_both_models(context.temp_allocator)
	// test.test_augmented_model(context.temp_allocator)
	test.run_all_batchnorm_tests(context.temp_allocator)
	test.rnn_simple_test(context.temp_allocator)
	test.gru_simple_test(context.temp_allocator)
	test.lstm_simple_test(context.temp_allocator)
	test.embedding_simple_test(context.temp_allocator)
	test.positional_encoding_test(context.temp_allocator)
	test.attention_simple_test(context.temp_allocator)
	test.multi_head_attention_test(context.temp_allocator)
	test.layer_norm_test(context.temp_allocator)
	test.ffn_simple_test(context.temp_allocator)
	test.transformer_encoder_block_test(context.temp_allocator)
	test.transformer_encoder_test(context.temp_allocator)
	test.transformer_decoder_test(context.temp_allocator)
	test.transformer_reversal_test(context.temp_allocator)
	// test.char_lm_test(context.temp_allocator)
	// test.gpt_full_test(context.temp_allocator)
	// test.gpt_shakespeare_test(context.temp_allocator)
	// test.bert_test(context.temp_allocator)
	// test.gan_test(context.temp_allocator)
	// test.gan_test_v2(context.temp_allocator)
	// test.wgan_test(context.temp_allocator)
	// test.vae_test(context.temp_allocator)
	// test.transfer_learning_test(context.temp_allocator)
	test.derivatives_test(context.temp_allocator)
	test.portfolio_test(context.temp_allocator)
	test.constraints_test(context.temp_allocator)
	test.risk_test(context.temp_allocator)
	test.finance_analytics_test(context.temp_allocator)
	test.factor_analysis_test(context.temp_allocator)
	test.backtest_test(context.temp_allocator)
	test.backtest_portfolio_test(context.temp_allocator)
	test.pdpm_test(context.temp_allocator)
	test.pdpm2_test(context.temp_allocator)
	test.pdpm_real_data_test(context.temp_allocator)
	test.pdpm_multifactor_test(context.temp_allocator)
	test.garch_test(context.temp_allocator)
	test.garch_real_data_test(context.temp_allocator)
	test.factor_model_test(context.temp_allocator)
	test.rs_garch_test(context.temp_allocator)
	test.garch_options_test(context.temp_allocator)
	test.advanced_derivatives_test(context.temp_allocator)
	test.exotic_options_test(context.temp_allocator)
	test.vol_surface_test(context.temp_allocator)
	test.live_options_calibration_test(context.temp_allocator)
	test.exotic_pricing_test(context.temp_allocator)
	fmt.println("Demo End")
	w.destroy_dataframe(&dfx)
	w.destroy_dataframe(&left2)
	w.destroy_dataframe(&right2)
	w.destroy_dataframe(&joined3)
	w.destroy_dataframe(&left)
	w.destroy_dataframe(&right)
	w.destroy_dataframe(&joined2)

	w.destroy_dataframe(&joined)
	w.destroy_dataframe(&salary)
	w.destroy_dataframe(&people)
	w.destroy_dataframe(&df_active3)
	w.destroy_dataframe(&df8)
	w.destroy_dataframe(&df9)
	w.destroy_dataframe(&df10)
	w.destroy_dataframe(&df11)
	defer w.free_select_exprs(exprs)
	defer w.free_select_exprs(exprs2)

}
