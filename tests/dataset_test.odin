package tests

import ml "../wotan/analytics/ML"
import w "../wotan/core"
import importer "../wotan/importer"
import l "../wotan/linalg"
import "core:fmt"
import "core:mem"

dataset_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing DataFrame Bridge ===")

	// 1. Load your CSVs into your DataFrames (using your existing CSV reader)
	df_train := importer.csv_load("train.csv")
	defer w.destroy_dataframe(&df_train)
	df_test := importer.csv_load("test.csv")
	defer w.destroy_dataframe(&df_test)


	// 2. Prepare Training Data (Fits encoders automatically!)
	train_data, ok := ml.prepare_dataset(&df_train, "target", allocator)
	defer ml.dataset_free(&train_data)

	if !ok {
		fmt.println("Failed to prepare dataset")
		return
	}

	fmt.printf("Training Data: %v samples, %v features\n", train_data.X.rows, train_data.X.cols)
	fmt.printf("Found %v categorical columns to encode\n", len(train_data.encoders))

	// 3. Build and Fit Pipeline
	pipe := ml.pipeline_new(allocator)
	defer ml.pipeline_free(&pipe)

	ml.pipeline_add_standard_scaler(&pipe)
	ml.pipeline_set_logistic(
		&pipe,
		{C = 1.0, max_iter = 100, learning_rate = 1.0, optimizer_type = .LBFGS},
	)

	// Fit on the prepared Matrix and Target
	ml.pipeline_fit(&pipe, &train_data.X, train_data.y)

	// 4. Prepare Test Data (Uses saved encoders, NO LEAKAGE!)
	test_X, test_y, ok2 := ml.transform_dataset(&df_test, &train_data, "target_column", allocator)
	defer l.matrix_free(&test_X)
	defer delete(test_y, allocator)

	if !ok2 {
		fmt.println("Failed to transform test dataset")
		return
	}

	// 5. Predict and Evaluate
	preds := ml.pipeline_predict(&pipe, &test_X, allocator)
	defer delete(preds, allocator)

	acc := ml.metrics_accuracy(test_y, preds)
	fmt.printf("Test Accuracy: %.2f%%\n", acc * 100)
}
