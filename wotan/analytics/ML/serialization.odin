package ML

import l "../../linalg"
import "core:mem"
import "core:os"

// ============================================================================
// Binary Writer
// ============================================================================

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
	// Write version
	write_u32(&w, 1)
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
	if version != 1 {
		r.err = true
		os.close(f)
		return r, false
	}

	return r, true
}

read_bytes :: proc(r: ^ModelReader, data: []u8) {
	if r.err {return}
	n, err := os.read(r.file, data)
	if err != nil || n != len(data) {r.err = true}
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
	n := int(read_i64(r))
	if n <= 0 {return nil}

	data := make([]f64, n, r.allocator)
	byte_len := n * size_of(f64)
	raw_ptr := cast([^]u8)(raw_data(data))
	read_bytes(r, raw_ptr[:byte_len])
	return data
}

read_slice_int :: proc(r: ^ModelReader) -> []int {
	n := int(read_i64(r))
	if n <= 0 {return nil}

	data := make([]int, n, r.allocator)
	for i in 0 ..< n {
		data[i] = int(read_i64(r))
	}
	return data
}

read_matrix :: proc(r: ^ModelReader) -> l.Matrix(f64) {
	rows := int(read_i64(r))
	cols := int(read_i64(r))
	if rows <= 0 || cols <= 0 {return l.Matrix(f64){}}

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
	pipe.model.type = cast(PipelineModelType)read_u32(&r)
	pipe.is_fitted = read_bool(&r)

	if !pipe.is_fitted {
		return pipe, !r.err
	}

	// 1. Read Steps
	n_steps := int(read_i64(&r))
	for i in 0 ..< n_steps {
		step_type := cast(PipelineStepType)read_u32(&r)
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
		pipe.model.kernel_svm.kernel_type = cast(SVMKernelType)read_u32(&r)
		pipe.model.kernel_svm.gamma = read_f64(&r)
		pipe.model.kernel_svm.degree = int(read_i64(&r))
		pipe.model.kernel_svm.coef0 = read_f64(&r)
		pipe.model.kernel_svm.C = read_f64(&r)
		pipe.model.kernel_svm.allocator = allocator
	case .KNN:
		pipe.model.knn.X_train = read_matrix(&r)
		pipe.model.knn.y_train = read_slice_f64(&r)
		pipe.model.knn.k = int(read_i64(&r))
		pipe.model.knn.weights = cast(KNNWeights)read_u32(&r)
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
		pipe.model.svr.kernel_type = cast(SVMKernelType)read_u32(&r)
		pipe.model.svr.gamma = read_f64(&r)
		pipe.model.svr.degree = int(read_i64(&r))
		pipe.model.svr.coef0 = read_f64(&r)
		pipe.model.svr.C = read_f64(&r)
		pipe.model.svr.epsilon = read_f64(&r)
		pipe.model.svr.allocator = allocator
	}

	return pipe, !r.err
}
