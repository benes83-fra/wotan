package ML

import w "../../core"
import l "../../linalg"
import "core:fmt"
import "core:mem"

// ============================================================================
// Dataset Structures
// ============================================================================

// ✅ RENAMED to avoid conflict with the existing LabelEncoder in preprocessing.odin
ColumnLabelEncoder :: struct {
	col_name:  string,
	col_index: int,
	classes:   []string, // Unique classes found during fit
	n_classes: int,
	allocator: mem.Allocator,
}

MLDataset :: struct {
	X:             l.Matrix(f64),
	y:             []f64,
	feature_names: []string, // Ordered list of feature column names (crucial for test alignment)
	encoders:      []ColumnLabelEncoder, // ✅ Updated type
	allocator:     mem.Allocator,
}

// ============================================================================
// Public API: Prepare Dataset (Fit)
// Call this ONLY on your Training Data!
// ============================================================================

prepare_dataset :: proc(
	df: ^w.DataFrame,
	target_col: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	MLDataset,
	bool,
) {
	n_rows := df.rows
	n_cols := len(df.columns)

	target_idx := -1
	n_features := 0
	for i in 0 ..< n_cols {
		col := &df.columns[i]
		if col.name == target_col {
			target_idx = i
			continue
		}
		if col.type == .Float || col.type == .Int || col.type == .Bool || col.type == .String {
			n_features += 1
		} else {
			fmt.printf(
				"Warning: Column '%s' has unsupported type %v, skipping.\n",
				col.name,
				col.type,
			)
		}
	}

	if target_idx == -1 {
		fmt.printf("Error: Target column '%s' not found\n", target_col)
		return MLDataset{}, false
	}

	X := l.matrix_new(f64, n_rows, n_features, allocator)
	y := make([]f64, n_rows, allocator)
	encoders := make([dynamic]ColumnLabelEncoder, 0, allocator)
	feature_names := make([]string, n_features, allocator)

	target_col_ptr := &df.columns[target_idx]
	for i in 0 ..< n_rows {
		#partial switch target_col_ptr.type {
		case .Float:
			v, is_null := w.column_at_float(target_col_ptr, i)
			if is_null {
				y[i] = 0.0
			} else {
				y[i] = v
			}
		case .Int:
			v, is_null := w.column_at_int(target_col_ptr, i)
			if is_null {
				y[i] = 0.0
			} else {
				y[i] = f64(v)
			}
		case .Bool:
			v, is_null := w.column_at_bool(target_col_ptr, i)
			if is_null {
				y[i] = 0.0
			} else if v {
				y[i] = 1.0
			} else {
				y[i] = 0.0
			}
		case:
			fmt.printf(
				"Warning: Target column '%s' has type %v. Please encode it first.\n",
				target_col,
				target_col_ptr.type,
			)
			y[i] = 0.0
		}
	}

	feat_idx := 0
	for col_idx in 0 ..< n_cols {
		if col_idx == target_idx {continue}
		col := &df.columns[col_idx]

		if col.type != .Float && col.type != .Int && col.type != .Bool && col.type != .String {
			continue
		}

		feature_names[feat_idx] = col.name

		#partial switch col.type {
		case .Float:
			for i in 0 ..< n_rows {
				v, is_null := w.column_at_float(col, i)
				if is_null {
					X.data[i * n_features + feat_idx] = 0.0
				} else {
					X.data[i * n_features + feat_idx] = v
				}
			}
		case .Int:
			for i in 0 ..< n_rows {
				v, is_null := w.column_at_int(col, i)
				if is_null {
					X.data[i * n_features + feat_idx] = 0.0
				} else {
					X.data[i * n_features + feat_idx] = f64(v)
				}
			}
		case .Bool:
			for i in 0 ..< n_rows {
				v, is_null := w.column_at_bool(col, i)
				if is_null {
					X.data[i * n_features + feat_idx] = 0.0
				} else if v {
					X.data[i * n_features + feat_idx] = 1.0
				} else {
					X.data[i * n_features + feat_idx] = 0.0
				}
			}
		case .String:
			classes := make([dynamic]string, 0, allocator)
			class_map := make(map[string]int, allocator)

			for i in 0 ..< n_rows {
				v, is_null := w.column_at_string(col, i)
				if is_null {v = "NULL"}

				if _, ok := class_map[v]; !ok {
					class_map[v] = len(classes)
					append(&classes, v)
				}
			}

			// ✅ Use the renamed struct
			encoder := ColumnLabelEncoder {
				col_name  = col.name,
				col_index = feat_idx,
				classes   = classes[:],
				n_classes = len(classes),
				allocator = allocator,
			}
			append(&encoders, encoder)

			for i in 0 ..< n_rows {
				v, is_null := w.column_at_string(col, i)
				if is_null {v = "NULL"}
				X.data[i * n_features + feat_idx] = f64(class_map[v])
			}

			delete(class_map)
		}
		feat_idx += 1
	}

	return MLDataset {
			X = X,
			y = y,
			feature_names = feature_names,
			encoders = encoders[:],
			allocator = allocator,
		},
		true
}

// ============================================================================
// Public API: Transform Dataset (For Test Data)
// Uses the encoders fitted on the Training Data to prevent leakage!
// ============================================================================

transform_dataset :: proc(
	df: ^w.DataFrame,
	dataset: ^MLDataset,
	target_col: string = "",
	allocator: mem.Allocator = context.allocator,
) -> (
	l.Matrix(f64),
	[]f64,
	bool,
) {
	n_rows := df.rows
	n_features := len(dataset.feature_names)

	X := l.matrix_new(f64, n_rows, n_features, allocator)
	y := make([]f64, n_rows, allocator)

	if target_col != "" {
		target_idx, ok := df.name_to_index[target_col]
		if ok {
			target_col_ptr := &df.columns[target_idx]
			for i in 0 ..< n_rows {
				#partial switch target_col_ptr.type {
				case .Float:
					v, is_null := w.column_at_float(target_col_ptr, i)
					if is_null {
						y[i] = 0.0
					} else {
						y[i] = v
					}
				case .Int:
					v, is_null := w.column_at_int(target_col_ptr, i)
					if is_null {
						y[i] = 0.0
					} else {
						y[i] = f64(v)
					}
				case .Bool:
					v, is_null := w.column_at_bool(target_col_ptr, i)
					if is_null {
						y[i] = 0.0
					} else if v {
						y[i] = 1.0
					} else {
						y[i] = 0.0
					}
				}
			}
		}
	}

	for feat_name, feat_idx in dataset.feature_names {
		col_idx, ok := df.name_to_index[feat_name]
		if !ok {
			fmt.printf("Error: Feature column '%s' not found in test DataFrame\n", feat_name)
			l.matrix_free(&X)
			delete(y, allocator)
			return X, y, false
		}
		col := &df.columns[col_idx]

		encoder_idx := -1
		for enc, i in dataset.encoders {
			if enc.col_name == feat_name {
				encoder_idx = i
				break
			}
		}

		if encoder_idx != -1 {
			enc := &dataset.encoders[encoder_idx]
			for i in 0 ..< n_rows {
				v, is_null := w.column_at_string(col, i)
				if is_null {v = "NULL"}

				class_val := 0.0
				for c_name, c_idx in enc.classes {
					if c_name == v {
						class_val = f64(c_idx)
						break
					}
				}
				X.data[i * n_features + feat_idx] = class_val
			}
		} else {
			#partial switch col.type {
			case .Float:
				for i in 0 ..< n_rows {
					v, is_null := w.column_at_float(col, i)
					if is_null {
						X.data[i * n_features + feat_idx] = 0.0
					} else {
						X.data[i * n_features + feat_idx] = v
					}
				}
			case .Int:
				for i in 0 ..< n_rows {
					v, is_null := w.column_at_int(col, i)
					if is_null {
						X.data[i * n_features + feat_idx] = 0.0
					} else {
						X.data[i * n_features + feat_idx] = f64(v)
					}
				}
			case .Bool:
				for i in 0 ..< n_rows {
					v, is_null := w.column_at_bool(col, i)
					if is_null {
						X.data[i * n_features + feat_idx] = 0.0
					} else if v {
						X.data[i * n_features + feat_idx] = 1.0
					} else {
						X.data[i * n_features + feat_idx] = 0.0
					}
				}
			case:
				fmt.printf("Warning: Test column '%s' type mismatch or unsupported.\n", feat_name)
			}
		}
	}

	return X, y, true
}

// ============================================================================
// Public API: Free
// ============================================================================

dataset_free :: proc(ds: ^MLDataset) {
	l.matrix_free(&ds.X)
	delete(ds.y, ds.allocator)
	delete(ds.feature_names, ds.allocator)
	for enc in ds.encoders {
		delete(enc.classes, enc.allocator)
	}
	delete(ds.encoders, ds.allocator)
}
