package ML

import l "../../linalg"
import "core:mem"
import "core:os"


// ============================================================================
// Binary Writer
// ============================================================================

_MODEL_FORMAT_VERSION :: 3

ModelWriter :: struct {
	file: ^os.File,
	err:  bool,
}

model_writer_open :: proc(path: string) -> (ModelWriter, bool) {
	f, err := os.create(path)
	if err != nil {return ModelWriter{}, false}

	w := ModelWriter {
		file = f,
	}
	// Write magic number "WOTAN" (5 bytes)
	write_bytes(&w, []u8{'W', 'O', 'T', 'A', 'N'})
	// Write version (2: gradient boosting also stores initial_pred / learning_rate;
	// 3: trees store classes + class fractions, forests / boosting their classes)
	write_u32(&w, _MODEL_FORMAT_VERSION)
	return w, true
}

write_bytes :: proc(w: ^ModelWriter, data: []u8) {
	if w.err {return}
	n, err := os.write(w.file, data)
	if err != nil || n != len(data) {w.err = true}
}

write_u32 :: proc(w: ^ModelWriter, val: u32) {
	if w.err {return}
	b := transmute([4]u8)val
	write_bytes(w, b[:])
}

write_i64 :: proc(w: ^ModelWriter, val: i64) {
	if w.err {return}
	b := transmute([8]u8)val
	write_bytes(w, b[:])
}

write_f64 :: proc(w: ^ModelWriter, val: f64) {
	if w.err {return}
	b := transmute([8]u8)val
	write_bytes(w, b[:])
}

write_bool :: proc(w: ^ModelWriter, val: bool) {
	write_u32(w, u32(val))
}

write_slice_f64 :: proc(w: ^ModelWriter, data: []f64) {
	write_i64(w, i64(len(data)))
	if len(data) > 0 {
		byte_len := len(data) * size_of(f64)
		raw_ptr := cast([^]u8)(raw_data(data))
		write_bytes(w, raw_ptr[:byte_len])
	}
}

write_slice_int :: proc(w: ^ModelWriter, data: []int) {
	write_i64(w, i64(len(data)))
	for val in data {
		write_i64(w, i64(val))
	}
}

write_matrix :: proc(w: ^ModelWriter, mat: ^l.Matrix(f64)) {
	write_i64(w, i64(mat.rows))
	write_i64(w, i64(mat.cols))
	if mat.rows > 0 && mat.cols > 0 && mat.data != nil {
		byte_len := len(mat.data) * size_of(f64)
		raw_ptr := cast([^]u8)(raw_data(mat.data))
		write_bytes(w, raw_ptr[:byte_len])
	}
}

model_writer_close :: proc(w: ^ModelWriter) {
	if w.file != nil {
		os.close(w.file)
	}
}

// ============================================================================
// Binary Reader
// ============================================================================

ModelReader :: struct {
	file:      ^os.File,
	err:       bool,
	allocator: mem.Allocator,
	remaining: i64, // bytes left in the file: bounds every length read from it
	version:   u32,
}

model_reader_open :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	ModelReader,
	bool,
) {
	f, err := os.open(path)
	if err != nil {return ModelReader{}, false}

	r := ModelReader {
		file      = f,
		allocator = allocator,
	}
	size, size_err := os.file_size(f)
	if size_err != nil {
		os.close(f)
		return ModelReader{}, false
	}
	r.remaining = size

	// Read magic number
	magic := make([]u8, 5, allocator)
	defer delete(magic, allocator)
	read_bytes(&r, magic)

	if magic[0] != 'W' ||
	   magic[1] != 'O' ||
	   magic[2] != 'T' ||
	   magic[3] != 'A' ||
	   magic[4] != 'N' {
		r.err = true
		os.close(f)
		return r, false
	}

	// Read version
	version := read_u32(&r)
	r.version = version
	if version < 1 || version > _MODEL_FORMAT_VERSION {
		r.err = true
		os.close(f)
		return r, false
	}

	return r, true
}

read_bytes :: proc(r: ^ModelReader, data: []u8) {
	if r.err {return}
	if i64(len(data)) > r.remaining {r.err = true; return}
	n, err := os.read(r.file, data)
	if err != nil || n != len(data) {r.err = true; return}
	r.remaining -= i64(n)
}

// A length read from the file is only accepted if that many elements of
// elem_size bytes can still follow (corrupt / truncated files fail cleanly)
_read_len :: proc(r: ^ModelReader, elem_size: int) -> int {
	n := read_i64(r)
	if r.err do return 0
	if n < 0 || n > r.remaining / i64(max(elem_size, 1)) {r.err = true; return 0}
	return int(n)
}

// Enum value read from the file, checked against the enum's range
_read_enum :: proc(r: ^ModelReader, $E: typeid) -> E {
	v := read_u32(r)
	if r.err do return E(0)
	if int(v) < int(min(E)) || int(v) > int(max(E)) {r.err = true; return E(0)}
	return E(v)
}

read_u32 :: proc(r: ^ModelReader) -> u32 {
	if r.err {return 0}
	b: [4]u8
	read_bytes(r, b[:])
	return transmute(u32)b
}

read_i64 :: proc(r: ^ModelReader) -> i64 {
	if r.err {return 0}
	b: [8]u8
	read_bytes(r, b[:])
	return transmute(i64)b
}

read_f64 :: proc(r: ^ModelReader) -> f64 {
	if r.err {return 0.0}
	b: [8]u8
	read_bytes(r, b[:])
	return transmute(f64)b
}

read_bool :: proc(r: ^ModelReader) -> bool {
	return read_u32(r) != 0
}

read_slice_f64 :: proc(r: ^ModelReader) -> []f64 {
	n := _read_len(r, size_of(f64))
	if n <= 0 {return nil}

	data := make([]f64, n, r.allocator)
	byte_len := n * size_of(f64)
	raw_ptr := cast([^]u8)(raw_data(data))
	read_bytes(r, raw_ptr[:byte_len])
	return data
}

read_slice_int :: proc(r: ^ModelReader) -> []int {
	n := _read_len(r, size_of(i64))
	if n <= 0 {return nil}

	data := make([]int, n, r.allocator)
	for i in 0 ..< n {
		data[i] = int(read_i64(r))
	}
	return data
}

read_matrix :: proc(r: ^ModelReader) -> l.Matrix(f64) {
	rows := _read_len(r, 1)
	cols := _read_len(r, 1)
	if rows <= 0 || cols <= 0 {return l.Matrix(f64){}}
	if i64(rows) > r.remaining / i64(cols * size_of(f64)) {r.err = true; return l.Matrix(f64){}}

	mat := l.matrix_new(f64, rows, cols, r.allocator)
	byte_len := len(mat.data) * size_of(f64)
	raw_ptr := cast([^]u8)(raw_data(mat.data))
	read_bytes(r, raw_ptr[:byte_len])
	return mat
}

model_reader_close :: proc(r: ^ModelReader) {
	if r.file != nil {
		os.close(r.file)
	}
}

// ============================================================================
// Public API: Save and Load Pipeline
// ============================================================================

pipeline_save :: proc(pipe: ^Pipeline, path: string) -> bool {
	w, ok := model_writer_open(path)
	if !ok {return false}
	defer model_writer_close(&w)

	// 1. Write Pipeline Type / Model Type
	write_u32(&w, u32(pipe.model.type))
	write_bool(&w, pipe.is_fitted)

	if !pipe.is_fitted {
		return !w.err
	}

	// 2. Write Steps
	write_i64(&w, i64(len(pipe.steps)))
	for i in 0 ..< len(pipe.steps) {
		step := &pipe.steps[i]
		write_u32(&w, u32(step.type))
		switch step.type {
		case .StandardScaler:
			write_slice_f64(&w, step.standard_scaler.mean)
			write_slice_f64(&w, step.standard_scaler.std)
			write_slice_f64(&w, step.standard_scaler.inv_std)
			write_i64(&w, i64(step.standard_scaler.n_features))
		case .MinMaxScaler:
			write_slice_f64(&w, step.minmax_scaler.data_min)
			write_slice_f64(&w, step.minmax_scaler.data_max)
			write_slice_f64(&w, step.minmax_scaler.scale)
			write_slice_f64(&w, step.minmax_scaler.min_val)
			write_i64(&w, i64(step.minmax_scaler.n_features))
		}
	}

	// 3. Write Model (Using #partial switch for unhandled tree models)
	#partial switch pipe.model.type {
	case .Logistic:
		write_slice_f64(&w, pipe.model.logistic.weights)
		write_f64(&w, pipe.model.logistic.bias)
	case .LinearSVM:
		write_slice_f64(&w, pipe.model.linear_svm.weights)
		write_f64(&w, pipe.model.linear_svm.bias)
	case .KernelSVM:
		write_slice_int(&w, pipe.model.kernel_svm.support_vectors)
		write_matrix(&w, &pipe.model.kernel_svm.sv_data)
		write_slice_f64(&w, pipe.model.kernel_svm.alpha)
		write_slice_f64(&w, pipe.model.kernel_svm.sv_labels)
		write_f64(&w, pipe.model.kernel_svm.bias)
		write_u32(&w, u32(pipe.model.kernel_svm.kernel_type))
		write_f64(&w, pipe.model.kernel_svm.gamma)
		write_i64(&w, i64(pipe.model.kernel_svm.degree))
		write_f64(&w, pipe.model.kernel_svm.coef0)
		write_f64(&w, pipe.model.kernel_svm.C)
	case .KNN:
		write_matrix(&w, &pipe.model.knn.X_train)
		write_slice_f64(&w, pipe.model.knn.y_train)
		write_i64(&w, i64(pipe.model.knn.k))
		write_u32(&w, u32(pipe.model.knn.weights))
	case .Ridge, .Lasso, .OLS:
		res: OLSResult
		if pipe.model.type ==
		   .Ridge {res = pipe.model.ridge} else if pipe.model.type == .Lasso {res = pipe.model.lasso} else {res = pipe.model.ols}

		write_slice_f64(&w, res.beta)
	case .SVR:
		write_slice_int(&w, pipe.model.svr.support_vectors)
		write_matrix(&w, &pipe.model.svr.sv_data)
		write_slice_f64(&w, pipe.model.svr.alpha_diff)
		write_f64(&w, pipe.model.svr.bias)
		write_u32(&w, u32(pipe.model.svr.kernel_type))
		write_f64(&w, pipe.model.svr.gamma)
		write_i64(&w, i64(pipe.model.svr.degree))
		write_f64(&w, pipe.model.svr.coef0)
		write_f64(&w, pipe.model.svr.C)
		write_f64(&w, pipe.model.svr.epsilon)
	case .DecisionTree:
		write_decision_tree(&w, &pipe.model.dt)

	case .RandomForest:
		write_i64(&w, i64(len(pipe.model.rf.trees)))
		for &tree in pipe.model.rf.trees {
			write_decision_tree(&w, &tree)
		}
		write_slice_f64(&w, pipe.model.rf.classes)

	case .GradientBoosting:
		// ⚠️ ADJUST: If your GB uses a different field name for trees (like 'estimators'), change it here
		write_f64(&w, pipe.model.gb.initial_pred)
		write_f64(&w, pipe.model.gb.learning_rate)
		write_i64(&w, i64(len(pipe.model.gb.trees)))
		for &tree in pipe.model.gb.trees {
			write_decision_tree(&w, &tree)
		}
		write_u32(&w, u32(pipe.model.gb.loss))
		write_slice_f64(&w, pipe.model.gb.classes)
		write_slice_f64(&w, pipe.model.gb.init_raw)
	case .MLP:
		write_i64(&w, i64(pipe.model.mlp.n_layers))
		for i in 0 ..< pipe.model.mlp.n_layers {
			write_matrix(&w, &pipe.model.mlp.weights[i])
			write_slice_f64(&w, pipe.model.mlp.biases[i])
			write_u32(&w, u32(pipe.model.mlp.activations[i]))
		}

	case .GaussianNB:
		write_slice_f64(&w, pipe.model.gnb.classes)
		write_matrix(&w, &pipe.model.gnb.theta)
		write_matrix(&w, &pipe.model.gnb.inv_sigma)
		write_slice_f64(&w, pipe.model.gnb.joint_bias)

	case .OvR_Logistic, .OvR_LinearSVM, .OvR_KernelSVM:
		ovr_ptr: ^OvRClassifier
		if pipe.model.type ==
		   .OvR_Logistic {ovr_ptr = &pipe.model.ovr_logistic} else if pipe.model.type == .OvR_LinearSVM {ovr_ptr = &pipe.model.ovr_linear_svm} else {ovr_ptr = &pipe.model.ovr_kernel_svm}

		write_slice_f64(&w, ovr_ptr.classes)
		write_i64(&w, i64(len(ovr_ptr.models)))
		for &model in ovr_ptr.models {
			write_binary_classifier(&w, &model)
		}

	}

	return !w.err
}

pipeline_load :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	Pipeline,
	bool,
) {
	r, ok := model_reader_open(path, allocator)
	if !ok {return Pipeline{}, false}
	defer model_reader_close(&r)

	pipe := pipeline_new(allocator)
	pipe.model.type = _read_enum(&r, PipelineModelType)
	pipe.is_fitted = read_bool(&r)

	if !pipe.is_fitted {
		return pipe, !r.err
	}

	// 1. Read Steps
	n_steps := _read_len(&r, 4)
	for i in 0 ..< n_steps {
		step_type := _read_enum(&r, PipelineStepType)
		if r.err do break
		switch step_type {
		case .StandardScaler:
			step := PipelineStep {
				type = .StandardScaler,
			}
			step.standard_scaler.mean = read_slice_f64(&r)
			step.standard_scaler.std = read_slice_f64(&r)
			step.standard_scaler.inv_std = read_slice_f64(&r)
			step.standard_scaler.n_features = int(read_i64(&r))
			step.standard_scaler.allocator = allocator
			append(&pipe.steps, step)
		case .MinMaxScaler:
			step := PipelineStep {
				type = .MinMaxScaler,
			}
			step.minmax_scaler.data_min = read_slice_f64(&r)
			step.minmax_scaler.data_max = read_slice_f64(&r)
			step.minmax_scaler.scale = read_slice_f64(&r)
			step.minmax_scaler.min_val = read_slice_f64(&r)
			step.minmax_scaler.n_features = int(read_i64(&r))
			step.minmax_scaler.allocator = allocator
			append(&pipe.steps, step)
		}
	}

	// 2. Read Model
	#partial switch pipe.model.type {
	case .Logistic:
		pipe.model.logistic.weights = read_slice_f64(&r)
		pipe.model.logistic.bias = read_f64(&r)
		pipe.model.logistic.allocator = allocator
	case .LinearSVM:
		pipe.model.linear_svm.weights = read_slice_f64(&r)
		pipe.model.linear_svm.bias = read_f64(&r)
		pipe.model.linear_svm.allocator = allocator
	case .KernelSVM:
		pipe.model.kernel_svm.support_vectors = read_slice_int(&r)
		pipe.model.kernel_svm.sv_data = read_matrix(&r)
		pipe.model.kernel_svm.alpha = read_slice_f64(&r)
		pipe.model.kernel_svm.sv_labels = read_slice_f64(&r)
		pipe.model.kernel_svm.bias = read_f64(&r)
		pipe.model.kernel_svm.kernel_type = _read_enum(&r, SVMKernelType)
		pipe.model.kernel_svm.gamma = read_f64(&r)
		pipe.model.kernel_svm.degree = int(read_i64(&r))
		pipe.model.kernel_svm.coef0 = read_f64(&r)
		pipe.model.kernel_svm.C = read_f64(&r)
		pipe.model.kernel_svm.allocator = allocator
	case .KNN:
		pipe.model.knn.X_train = read_matrix(&r)
		pipe.model.knn.y_train = read_slice_f64(&r)
		pipe.model.knn.k = int(read_i64(&r))
		pipe.model.knn.weights = _read_enum(&r, KNNWeights)
		pipe.model.knn.allocator = allocator
	case .Ridge, .Lasso, .OLS:
		beta := read_slice_f64(&r)
		// ✅ FIX: Removed allocator field from OLSResult literal
		res := OLSResult {
			beta = beta,
		}
		if pipe.model.type ==
		   .Ridge {pipe.model.ridge = res} else if pipe.model.type == .Lasso {pipe.model.lasso = res} else {pipe.model.ols = res}
	case .SVR:
		pipe.model.svr.support_vectors = read_slice_int(&r)
		pipe.model.svr.sv_data = read_matrix(&r)
		pipe.model.svr.alpha_diff = read_slice_f64(&r)
		pipe.model.svr.bias = read_f64(&r)
		pipe.model.svr.kernel_type = _read_enum(&r, SVMKernelType)
		pipe.model.svr.gamma = read_f64(&r)
		pipe.model.svr.degree = int(read_i64(&r))
		pipe.model.svr.coef0 = read_f64(&r)
		pipe.model.svr.C = read_f64(&r)
		pipe.model.svr.epsilon = read_f64(&r)
		pipe.model.svr.allocator = allocator
	case .DecisionTree:
		pipe.model.dt = read_decision_tree(&r, allocator)

	case .RandomForest:
		n_trees := _read_len(&r, 8)
		pipe.model.rf.trees = make([dynamic]DecisionTree, 0, n_trees, allocator)
		pipe.model.rf.allocator = allocator
		for t in 0 ..< n_trees {
			tree := read_decision_tree(&r, allocator)
			append(&pipe.model.rf.trees, tree)
		}
		if r.version >= 3 do pipe.model.rf.classes = read_slice_f64(&r)
		// every tree must be a classifier over the forest's classes, or all regressors
		for &tree in pipe.model.rf.trees {
			if len(tree.classes) != len(pipe.model.rf.classes) do r.err = true
		}

	case .GradientBoosting:
		// version 1 files did not store F₀ and the learning rate: unusable
		if r.version < 2 {r.err = true; break}
		pipe.model.gb.initial_pred = read_f64(&r)
		pipe.model.gb.learning_rate = read_f64(&r)
		n_trees := _read_len(&r, 8)
		// ⚠️ ADJUST: If your GB uses a different field name for trees, change it here
		pipe.model.gb.trees = make([dynamic]DecisionTree, 0, n_trees, allocator)
		pipe.model.gb.allocator = allocator
		for t in 0 ..< n_trees {
			tree := read_decision_tree(&r, allocator)
			append(&pipe.model.gb.trees, tree)
		}
		pipe.model.gb.trees_per_stage = 1
		if r.version >= 3 {
			gb := &pipe.model.gb
			gb.loss = _read_enum(&r, GBLoss)
			gb.classes = read_slice_f64(&r)
			gb.init_raw = read_slice_f64(&r)
			if gb.loss == .Log_Loss {
				K := len(gb.classes)
				gb.trees_per_stage = 1 if K == 2 else K
				if K < 2 || len(gb.init_raw) != gb.trees_per_stage || n_trees % gb.trees_per_stage != 0 do r.err = true
			}
		}
		for &tree in pipe.model.gb.trees {
			if len(tree.classes) != 0 do r.err = true // boosting trees are regression trees
		}
	case .MLP:
		n_layers := _read_len(&r, 16)
		pipe.model.mlp.n_layers = n_layers
		pipe.model.mlp.weights = make([]l.Matrix(f64), n_layers, allocator)
		pipe.model.mlp.biases = make([][]f64, n_layers, allocator)
		pipe.model.mlp.activations = make([]Activation, n_layers, allocator)
		pipe.model.mlp.allocator = allocator

		for i in 0 ..< n_layers {
			pipe.model.mlp.weights[i] = read_matrix(&r)
			pipe.model.mlp.biases[i] = read_slice_f64(&r)
			pipe.model.mlp.activations[i] = _read_enum(&r, Activation)
		}

	case .GaussianNB:
		pipe.model.gnb.classes = read_slice_f64(&r)
		pipe.model.gnb.theta = read_matrix(&r)
		pipe.model.gnb.inv_sigma = read_matrix(&r)
		pipe.model.gnb.joint_bias = read_slice_f64(&r)
		pipe.model.gnb.allocator = allocator

	case .OvR_Logistic, .OvR_LinearSVM, .OvR_KernelSVM:
		ovr_ptr: ^OvRClassifier
		if pipe.model.type ==
		   .OvR_Logistic {ovr_ptr = &pipe.model.ovr_logistic} else if pipe.model.type == .OvR_LinearSVM {ovr_ptr = &pipe.model.ovr_linear_svm} else {ovr_ptr = &pipe.model.ovr_kernel_svm}

		ovr_ptr.classes = read_slice_f64(&r)
		ovr_ptr.allocator = allocator
		n_models := _read_len(&r, 4)
		ovr_ptr.models = make([]BinaryClassifier, n_models, allocator)
		for i in 0 ..< n_models {
			ovr_ptr.models[i] = read_binary_classifier(&r, allocator)
		}

	}

	return pipe, !r.err
}


// Write a single DecisionTree (which is just a flat array of Nodes and a root index)
write_decision_tree :: proc(w: ^ModelWriter, tree: ^DecisionTree) {
	write_i64(w, i64(tree.root_idx))
	write_i64(w, i64(len(tree.nodes)))

	for node in tree.nodes {
		write_i64(w, i64(node.feature_idx))
		write_f64(w, node.threshold)
		write_f64(w, node.value)
		write_i64(w, i64(node.left_idx))
		write_i64(w, i64(node.right_idx))
		write_f64(w, node.impurity)
		write_i64(w, i64(node.sample_count))
	}
	write_slice_f64(w, tree.classes)
	write_slice_f64(w, tree.proba[:])
}

// Read a single DecisionTree
read_decision_tree :: proc(r: ^ModelReader, allocator: mem.Allocator) -> DecisionTree {
	tree: DecisionTree
	tree.allocator = allocator
	tree.root_idx = int(read_i64(r))

	n_nodes := _read_len(r, 7 * 8)
	tree.nodes = make([dynamic]Node, 0, n_nodes, allocator)

	for i in 0 ..< n_nodes {
		node: Node
		node.feature_idx = int(read_i64(r))
		node.threshold = read_f64(r)
		node.value = read_f64(r)
		node.left_idx = int(read_i64(r))
		node.right_idx = int(read_i64(r))
		node.impurity = read_f64(r)
		node.sample_count = int(read_i64(r))
		append(&tree.nodes, node)
	}
	if r.version >= 3 {
		tree.classes = read_slice_f64(r)
		proba := read_slice_f64(r)
		defer delete(proba, r.allocator)
		if len(proba) != n_nodes * len(tree.classes) do r.err = true
		tree.proba = make([dynamic]f64, 0, len(proba), allocator)
		append(&tree.proba, ..proba)
	}

	// Structural check: children must be valid nodes (or both -1 for a leaf),
	// otherwise prediction would index out of bounds
	if tree.root_idx < 0 || tree.root_idx >= max(n_nodes, 1) do r.err = true
	for node in tree.nodes {
		leaf := node.left_idx < 0 && node.right_idx < 0
		inner := node.left_idx > 0 && node.left_idx < n_nodes && node.right_idx > 0 && node.right_idx < n_nodes && node.feature_idx >= 0
		if !leaf && !inner do r.err = true
	}
	if r.err {
		// keep the result safe to traverse (as a regression leaf)
		delete(tree.classes, r.allocator)
		tree.classes = nil
		clear(&tree.proba)
		clear(&tree.nodes)
		append(&tree.nodes, Node{feature_idx = -1, left_idx = -1, right_idx = -1})
		tree.root_idx = 0
	}

	return tree
}


// ============================================================================
// Binary Classifier Serialization Helpers (for OvR)
// ============================================================================

write_binary_classifier :: proc(w: ^ModelWriter, clf: ^BinaryClassifier) {
	write_u32(w, u32(clf.type))
	switch clf.type {
	case .Logistic:
		write_slice_f64(w, clf.logistic.weights)
		write_f64(w, clf.logistic.bias)
	case .LinearSVM:
		write_slice_f64(w, clf.linear_svm.weights)
		write_f64(w, clf.linear_svm.bias)
	case .KernelSVM:
		write_slice_int(w, clf.kernel_svm.support_vectors)
		write_matrix(w, &clf.kernel_svm.sv_data)
		write_slice_f64(w, clf.kernel_svm.alpha)
		write_slice_f64(w, clf.kernel_svm.sv_labels)
		write_f64(w, clf.kernel_svm.bias)
		write_u32(w, u32(clf.kernel_svm.kernel_type))
		write_f64(w, clf.kernel_svm.gamma)
		write_i64(w, i64(clf.kernel_svm.degree))
		write_f64(w, clf.kernel_svm.coef0)
		write_f64(w, clf.kernel_svm.C)
	}
}

read_binary_classifier :: proc(r: ^ModelReader, allocator: mem.Allocator) -> BinaryClassifier {
	clf: BinaryClassifier
	clf.allocator = allocator
	clf.type = _read_enum(r, BinaryClassifierType)
	switch clf.type {
	case .Logistic:
		clf.logistic.weights = read_slice_f64(r)
		clf.logistic.bias = read_f64(r)
		clf.logistic.allocator = allocator
	case .LinearSVM:
		clf.linear_svm.weights = read_slice_f64(r)
		clf.linear_svm.bias = read_f64(r)
		clf.linear_svm.allocator = allocator
	case .KernelSVM:
		clf.kernel_svm.support_vectors = read_slice_int(r)
		clf.kernel_svm.sv_data = read_matrix(r)
		clf.kernel_svm.alpha = read_slice_f64(r)
		clf.kernel_svm.sv_labels = read_slice_f64(r)
		clf.kernel_svm.bias = read_f64(r)
		clf.kernel_svm.kernel_type = _read_enum(r, SVMKernelType)
		clf.kernel_svm.gamma = read_f64(r)
		clf.kernel_svm.degree = int(read_i64(r))
		clf.kernel_svm.coef0 = read_f64(r)
		clf.kernel_svm.C = read_f64(r)
		clf.kernel_svm.allocator = allocator
	}
	return clf
}
