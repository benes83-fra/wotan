package nn

import l "../linalg"
import t "../tensor"
import "core:encoding/endian"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

// ============================================================================
// Model files
// ============================================================================
// Little-endian binary files that begin with CHECKPOINT_MAGIC. A tensor is
// stored as rows, cols (i32) and rows·cols f64 values, a nil tensor as 0, 0.
//   Sequential checkpoint: magic, epoch, layer count, per layer its type id,
//     configuration and tensors, then the optimizer type and the Adam state.
//     Type 3: the count of all parameters and a frozen flag (i32) for each
//     (order of sequential_parameters(all = true)), then the timestep,
//     hyperparameters, moments of the trainable parameters and their step
//     counts. Older types are still read (all parameters trainable): 2
//     without the frozen flags, 1 also with one shared step count, 0 also
//     without the hyperparameters.
//   GPT (38; older: 34 without the frozen flags, 33 also without the GELU
//     flag (exact GELU), 31 also without the norm and weight-tying flags, 28
//     also without the dropout rate, 27 also without the attention flags, 17
//     also without MoE), BERT (39; older: 32 without the frozen flags, 29 also
//     without the dropout rate, 18 also without the attention flags),
//     Generator (19), Discriminator (20), VAE encoder (21) and decoder (22):
//     magic, type id, configuration, tensors. GPT and BERT files end with the
//     count of all parameters and a frozen flag (i32) for each, in the order of
//     gpt_model_parameters / bert_model_parameters (all = true); older files
//     load with every parameter trainable. Files without the dropout rate load
//     with 0.1.
//   Parameter list (40): magic, type id, the metadata (byte count i32 and the
//     bytes), the parameter count, per parameter its frozen flag (i32) and
//     tensor, then an Adam flag (i32). With Adam state: the timestep (i64),
//     learning rate, beta_1, beta_2, epsilon, weight decay, and per parameter
//     a registered flag (i32), for a registered one followed by its
//     learning-rate scale, step count (i64) and both moments (count i32 and
//     values).
// Attention layers store use_flash and is_causal before their projections.
// A file is built in memory and written at once. Reading checks every access,
// so a truncated or foreign file fails instead of panicking. Loading builds
// each layer from its configuration and reads the stored values into its
// tensors (keeping their shapes); a stored tensor of another size replaces
// the layer's tensor (e.g. a classification head of another size). The
// random initialization of these layers uses a private generator, so loading
// leaves the caller's random sequence unchanged.

CHECKPOINT_MAGIC :: []u8{'W', 'O', 'T', 'A', 'N', '_', 'C', 'K', 'P', 'T'}

@(private = "file")
_Layer_Id :: enum {
	Linear         = 0,
	Conv2d         = 1, // the same kernel size, stride and padding on both axes
	Conv2d_Rect    = 38, // with separate width values (written only when one differs)
	MaxPool2d      = 2,
	Activation     = 3,
	Flatten        = 4,
	BatchNorm2d    = 5,
	AvgPool2d      = 6,
	Dropout        = 7,
	RNN            = 8,
	GRU            = 35, // with bias_hn (PyTorch's n part of bias_hh)
	LSTM           = 10,
	Embedding      = 11,
	LayerNorm      = 13,
	RMSNorm        = 31,
	Conv1d         = 32, // stride 1, groups 1
	Conv1d_SG      = 37, // with stride and groups (written only when one is not 1)
	TCNBlock       = 33,
	FFN            = 36, // with the tanh-GELU flag
	Mamba          = 34, // RMSNorm weight stored after the SSM tensors
	MoE            = 26,
	// layers with attention: use_flash and is_causal of every attention
	// sublayer are stored with its projections
	MHA            = 27,
	EncoderBlock   = 28,
	Encoder        = 29,
	GAT            = 30,
	// the same without the attention flags (read only)
	MHA_0          = 12,
	EncoderBlock_0 = 15,
	Encoder_0      = 16,
	GAT_0          = 24,
	// Mamba without the pre-norm and the residual (a plain layer, MambaLayer.plain)
	Mamba_0        = 25,
	// GRU without bias_hn: loads with bias_hn = nil (the same function and the
	// same parameter list); written for such layers
	GRU_0          = 9,
	// FFN without the GELU flag (read only; exact GELU)
	FFN_0          = 14,
}

@(private = "file")
_File_Id :: enum {
	GPT_Dense     = 17, // no MoE or attention flags (read only)
	BERT_0        = 18, // no attention flags (read only)
	Generator     = 19,
	Discriminator = 20,
	VAE_Encoder   = 21,
	VAE_Decoder   = 22,
	GPT_MoE       = 27, // MoE flags, no attention flags (read only)
	GPT_1         = 28, // no dropout rate (read only)
	BERT_1        = 29, // no dropout rate (read only)
	GPT_2         = 31, // no norm or weight-tying flags (read only)
	GPT_3         = 33, // no GELU flag (read only; exact GELU)
	GPT_4         = 34, // no frozen flags (read only)
	BERT_2        = 32, // no frozen flags (read only)
	GPT           = 38, // dropout rate, RMSNorm, tied-embedding and tanh-GELU flags; per block: MoE, use_flash and is_causal flags; frozen flags
	BERT          = 39, // dropout rate; per block: use_flash and is_causal flags; frozen flags
	Parameters    = 40, // a parameter list with metadata and optional Adam state
}

// ============================================================================
// Reader / writer
// ============================================================================

@(private = "file")
_Io :: struct {
	writing: bool,
	buf:     [dynamic]u8, // writing
	data:    []u8, // reading
	pos:     int,
	ok:      bool,
	alloc:   mem.Allocator,
}

@(private = "file")
_writer :: proc(alloc: mem.Allocator) -> _Io {
	io := _Io {
		writing = true,
		buf     = make([dynamic]u8, 0, 4096, alloc),
		ok      = true,
		alloc   = alloc,
	}
	append(&io.buf, ..CHECKPOINT_MAGIC)
	return io
}

// Writes the file and releases the buffer (`quiet`: no message on failure)
@(private = "file")
_save :: proc(io: ^_Io, path: string, quiet := false) -> bool {
	defer delete(io.buf)
	if err := os.write_entire_file(path, io.buf[:]); err != nil {
		if !quiet {fmt.printf("Failed to write %s: %v\n", path, err)}
		return false
	}
	return true
}

// Reads a file and checks the magic bytes (`quiet`: no message on failure)
@(private = "file")
_reader :: proc(path: string, alloc: mem.Allocator, quiet := false) -> (io: _Io, ok: bool) {
	data, err := os.read_entire_file(path, alloc)
	if err != nil {
		if !quiet {fmt.printf("Failed to read %s: %v\n", path, err)}
		return {}, false
	}
	io = _Io {
		data  = data,
		ok    = true,
		alloc = alloc,
	}
	magic := _take(&io, len(CHECKPOINT_MAGIC))
	if magic == nil || mem.compare(magic, CHECKPOINT_MAGIC) != 0 {
		if !quiet {fmt.printf("%s is not a wotan model file\n", path)}
		delete(data, alloc)
		return {}, false
	}
	return io, true
}

// The next n bytes, or nil (and !ok) past the end
@(private = "file")
_take :: proc(io: ^_Io, n: int) -> []u8 {
	if !io.ok || n < 0 || n > len(io.data) - io.pos {
		io.ok = false
		return nil
	}
	defer io.pos += n
	return io.data[io.pos:io.pos + n]
}

@(private = "file")
_put_i32 :: proc(io: ^_Io, v: int) {
	b: [4]u8
	endian.put_i32(b[:], .Little, i32(v))
	append(&io.buf, ..b[:])
}

@(private = "file")
_put_i64 :: proc(io: ^_Io, v: int) {
	b: [8]u8
	endian.put_i64(b[:], .Little, i64(v))
	append(&io.buf, ..b[:])
}

@(private = "file")
_put_f64 :: proc(io: ^_Io, v: f64) {
	b: [8]u8
	endian.put_f64(b[:], .Little, v)
	append(&io.buf, ..b[:])
}

@(private = "file")
_get_i32 :: proc(io: ^_Io) -> int {
	b := _take(io, 4)
	if b == nil {return 0}
	v, _ := endian.get_i32(b, .Little)
	return int(v)
}

@(private = "file")
_get_i64 :: proc(io: ^_Io) -> int {
	b := _take(io, 8)
	if b == nil {return 0}
	v, _ := endian.get_i64(b, .Little)
	return int(v)
}

@(private = "file")
_get_f64 :: proc(io: ^_Io) -> f64 {
	b := _take(io, 8)
	if b == nil {return 0}
	v, _ := endian.get_f64(b, .Little)
	return v
}

// A size or count: non-negative and no larger than the rest of the file
@(private = "file")
_get_size :: proc(io: ^_Io) -> int {
	v := _get_i32(io)
	if v < 0 || v > len(io.data) - io.pos {io.ok = false}
	return v if io.ok else 0
}

// A flag stored as i32 (written from `on`, or read)
@(private = "file")
_io_flag :: proc(io: ^_Io, on: bool) -> bool {
	if io.writing {
		_put_i32(io, int(on))
		return on
	}
	return _get_i32(io) != 0
}

// A tensor (written from x^, or read into it)
@(private = "file")
_io_tensor :: proc(io: ^_Io, x: ^^t.Tensor) {
	if io.writing {
		if x^ == nil {
			_put_i32(io, 0)
			_put_i32(io, 0)
			return
		}
		t.tensor_values(x^) // a device tensor is saved from its host copy
		m := x^.data
		_put_i32(io, m.rows)
		_put_i32(io, m.cols)
		start := len(io.buf)
		resize(&io.buf, start + 8 * len(m.data))
		for v, i in m.data {endian.put_f64(io.buf[start + 8 * i:], .Little, v)}
		return
	}
	rows, cols := _get_size(io), _get_size(io)
	if !io.ok {return}
	if rows == 0 && cols == 0 {
		if x^ != nil {t.tensor_free(x^)}
		x^ = nil
		return
	}
	bytes := _take(io, 8 * rows * cols) if rows <= len(io.data) / max(cols, 1) else nil
	if bytes == nil {
		io.ok = false
		return
	}
	n := rows * cols
	if x^ == nil || t.tensor_numel(x^) != n {
		requires_grad := x^ == nil || x^.requires_grad
		if x^ != nil {t.tensor_free(x^)}
		x^ = t.tensor_new(l.matrix_new(f64, rows, cols, io.alloc), requires_grad, io.alloc)
	}
	t.tensor_values(x^)
	for &v, i in x^.data.data {v, _ = endian.get_f64(bytes[8 * i:], .Little)}
	t.tensor_mark_host_dirty(x^) // a device tensor uploads the loaded values
}

// ============================================================================
// Layers
// ============================================================================

// Weights and bias (a nil bias is stored as an empty tensor)
@(private = "file")
_io_linear :: proc(io: ^_Io, lin: ^LinearLayer) {
	_io_tensor(io, &lin.weights)
	_io_tensor(io, &lin.bias)
	_sync_linear(lin)
}

// Bias flag, weights and the bias if present
@(private = "file")
_io_linear_flagged :: proc(io: ^_Io, lin: ^LinearLayer) {
	if !_io_flag(io, lin.bias != nil) && lin.bias != nil {
		t.tensor_free(lin.bias)
		lin.bias = nil
	}
	_io_tensor(io, &lin.weights)
	if lin.bias != nil {_io_tensor(io, &lin.bias)}
	_sync_linear(lin)
}

// in/out features follow the weights (which may have been replaced)
@(private = "file")
_sync_linear :: proc(lin: ^LinearLayer) {
	if lin.weights !=
	   nil {lin.in_features, lin.out_features = lin.weights.data.rows, lin.weights.data.cols}
}

// The flags (if stored) and the projections
@(private = "file")
_io_mha :: proc(io: ^_Io, mha: ^MultiHeadAttentionLayer, flags: bool) {
	if flags {
		mha.use_flash = _io_flag(io, mha.use_flash)
		mha.is_causal = _io_flag(io, mha.is_causal)
	}
	_io_linear(io, &mha.q_proj)
	_io_linear(io, &mha.k_proj)
	_io_linear(io, &mha.v_proj)
	_io_linear(io, &mha.out_proj)
}

@(private = "file")
_io_norm :: proc(io: ^_Io, ln: ^LayerNormLayer) {
	_io_tensor(io, &ln.gamma)
	_io_tensor(io, &ln.beta)
}

// Encoder block in file order: attention, feed-forward, norms
@(private = "file")
_io_block :: proc(
	io: ^_Io,
	mha: ^MultiHeadAttentionLayer,
	ffn: ^FFNLayer,
	ln1, ln2: ^LayerNormLayer,
	flags: bool,
) {
	_io_mha(io, mha, flags)
	_io_linear(io, &ffn.fc1)
	_io_linear(io, &ffn.fc2)
	_io_norm(io, ln1)
	_io_norm(io, ln2)
}

@(private = "file")
_io_moe :: proc(io: ^_Io, moe: ^MoELayer) {
	_io_linear_flagged(io, &moe.router)
	for &e in moe.experts {
		_io_linear_flagged(io, &e.fc1)
		_io_linear_flagged(io, &e.fc2)
	}
}

// The tensors of a layer in file order (written, or read into the layer);
// flags: the layer has its current id, not an older one (those lack the
// attention flags, or Mamba's norm weight)
@(private = "file")
_io_layer :: proc(io: ^_Io, layer: ^Layer, flags: bool) {
	switch &l in layer {
	case LinearLayer:
		_io_tensor(io, &l.weights)
		if l.bias != nil {_io_tensor(io, &l.bias)}
		_sync_linear(&l)
	case Conv2dLayer:
		_io_tensor(io, &l.weight)
		if l.bias != nil {_io_tensor(io, &l.bias)}
	case Conv1dLayer:
		_io_tensor(io, &l.weight)
		if l.bias != nil {_io_tensor(io, &l.bias)}
	case TCNBlock:
		for c in ([]^Conv1dLayer{&l.conv1, &l.conv2, &l.downsample}) {
			if c.weight == nil {continue}
			_io_tensor(io, &c.weight)
			_io_tensor(io, &c.bias)
		}
	case BatchNorm2dLayer:
		_io_tensor(io, &l.weight)
		_io_tensor(io, &l.bias)
		_io_tensor(io, &l.running_mean)
		_io_tensor(io, &l.running_var)
	case RNNLayer:
		_io_tensor(io, &l.w_ih)
		_io_tensor(io, &l.w_hh)
		_io_tensor(io, &l.bias)
	case GRULayer:
		_io_tensor(io, &l.w_ih)
		_io_tensor(io, &l.w_hh)
		_io_tensor(io, &l.bias)
		if l.bias_hn != nil {_io_tensor(io, &l.bias_hn)}
	case LSTMLayer:
		_io_tensor(io, &l.w_ih)
		_io_tensor(io, &l.w_hh)
		_io_tensor(io, &l.bias)
	case EmbeddingLayer:
		_io_tensor(io, &l.weight)
	case MultiHeadAttentionLayer:
		_io_mha(io, &l, flags)
	case LayerNormLayer:
		_io_norm(io, &l)
	case FFNLayer:
		_io_linear(io, &l.fc1)
		_io_linear(io, &l.fc2)
	case TransformerEncoderBlock:
		_io_block(io, &l.mha, &l.ffn, &l.ln1, &l.ln2, flags)
	case TransformerEncoder:
		for &b in l.blocks {_io_block(io, &b.mha, &b.ffn, &b.ln1, &b.ln2, flags)}
	case GATLayer:
		_io_tensor(io, &l.linear.weights)
		if l.linear.bias != nil {_io_tensor(io, &l.linear.bias)}
		_sync_linear(&l.linear)
		_io_mha(io, &l.mha, flags)
		_io_tensor(io, &l.adjacency)
		if l.adjacency != nil {l.adjacency.requires_grad = false}
	case MambaLayer:
		_io_linear(io, &l.proj_x)
		_io_linear(io, &l.proj_B)
		_io_linear(io, &l.proj_C)
		_io_linear(io, &l.proj_Delta)
		_io_linear(io, &l.proj_out)
		_io_tensor(io, &l.A)
		_io_tensor(io, &l.D)
		if flags && !l.plain {_io_tensor(io, &l.norm.gamma)}
	case MoELayer:
		_io_moe(io, &l)
	case MaxPool2dLayer, AvgPool2dLayer, DropoutLayer, Activation, FlattenLayer:
	}
}

// Type id and configuration of a layer
@(private = "file")
_write_layer_header :: proc(io: ^_Io, layer: ^Layer) {
	ints :: proc(io: ^_Io, id: _Layer_Id, vals: ..int) {
		_put_i32(io, int(id))
		for v in vals {_put_i32(io, v)}
	}
	switch l in layer {
	case LinearLayer:
		ints(io, .Linear, l.in_features, l.out_features, int(l.bias != nil))
	case Conv2dLayer:
		square := l.kernel_w == l.kernel_size && l.stride_w == l.stride && l.padding_w == l.padding
		ints(
			io,
			.Conv2d if square else .Conv2d_Rect,
			l.in_channels,
			l.out_channels,
			l.kernel_size,
			l.stride,
			l.padding,
			int(l.bias != nil),
		)
		if !square {
			_put_i32(io, l.kernel_w)
			_put_i32(io, l.stride_w)
			_put_i32(io, l.padding_w)
		}
	case MaxPool2dLayer:
		ints(io, .MaxPool2d, l.kernel_size, l.stride)
	case Activation:
		ints(io, .Activation, int(l))
	case FlattenLayer:
		ints(io, .Flatten)
	case BatchNorm2dLayer:
		ints(io, .BatchNorm2d, l.num_features)
		_put_f64(io, l.eps)
		_put_f64(io, l.momentum)
	case AvgPool2dLayer:
		ints(io, .AvgPool2d, l.kernel_size, l.stride)
	case DropoutLayer:
		ints(io, .Dropout)
		_put_f64(io, l.drop_prob)
	case RNNLayer:
		ints(io, .RNN, l.input_size, l.hidden_size)
	case GRULayer:
		ints(io, .GRU if l.bias_hn != nil else .GRU_0, l.input_size, l.hidden_size)
	case LSTMLayer:
		ints(io, .LSTM, l.input_size, l.hidden_size)
	case EmbeddingLayer:
		ints(io, .Embedding, l.num_embeddings, l.embedding_dim)
	case MultiHeadAttentionLayer:
		ints(io, .MHA, l.d_model, l.num_heads)
	case LayerNormLayer:
		ints(io, .RMSNorm if l.rms else .LayerNorm, l.d_model)
		_put_f64(io, l.eps)
	case FFNLayer:
		ints(io, .FFN, l.d_model, l.d_ff, int(l.gelu_tanh))
	case TransformerEncoderBlock:
		ints(io, .EncoderBlock, l.d_model, l.num_heads, l.d_ff)
	case TransformerEncoder:
		ints(io, .Encoder, l.num_layers, l.d_model, l.num_heads, l.d_ff)
	case GATLayer:
		ints(
			io,
			.GAT,
			l.linear.in_features,
			l.linear.out_features,
			l.num_heads,
			int(l.linear.bias != nil),
		)
	case MambaLayer:
		ints(io, .Mamba_0 if l.plain else .Mamba, l.d_model, l.d_state)
	case Conv1dLayer:
		stride, groups := max(l.stride, 1), max(l.groups, 1)
		plain := stride == 1 && groups == 1
		ints(
			io,
			.Conv1d if plain else .Conv1d_SG,
			l.in_channels,
			l.out_channels,
			l.kernel_size,
			l.dilation,
			l.pad_left,
			l.pad_right,
			int(l.bias != nil),
		)
		if !plain {
			_put_i32(io, stride)
			_put_i32(io, groups)
		}
	case TCNBlock:
		ints(
			io,
			.TCNBlock,
			l.conv1.in_channels,
			l.conv1.out_channels,
			l.conv1.kernel_size,
			l.conv1.dilation,
		)
		_put_f64(io, l.dropout)
	case MoELayer:
		ints(io, .MoE, l.num_experts, l.top_k, l.d_model, l.d_ff)
	}
}

// Reads a type id and configuration and builds the layer (its tensors are
// read next, by _io_layer, with the attention flags if `flags`)
@(private = "file")
_read_layer_header :: proc(
	io: ^_Io,
	alloc: mem.Allocator,
) -> (
	layer: Layer,
	flags: bool,
	ok: bool,
) {
	id := _get_i32(io)
	#partial switch _Layer_Id(id) {
	case .MHA, .EncoderBlock, .Encoder, .GAT, .Mamba:
		flags = true
	}
	switch _Layer_Id(id) {
	case .Linear:
		in_f, out_f, has_bias := _get_size(io), _get_size(io), _get_i32(io) != 0
		if !io.ok {return}
		lin := linear_layer_new(in_f, out_f, alloc)
		if !has_bias {
			t.tensor_free(lin.bias)
			lin.bias = nil
		}
		layer = lin
	case .Conv2d, .Conv2d_Rect:
		c_in, c_out, k, stride, pad :=
			_get_size(io), _get_size(io), _get_size(io), _get_size(io), _get_size(io)
		has_bias := _get_i32(io) != 0
		k_w, stride_w, pad_w := k, stride, pad
		if _Layer_Id(id) == .Conv2d_Rect {
			k_w, stride_w, pad_w = _get_size(io), _get_size(io), _get_size(io)
		}
		if !io.ok || k == 0 || k_w == 0 || stride == 0 || stride_w == 0 {
			io.ok = false
			return
		}
		layer = conv2d_layer_new(
			c_in,
			c_out,
			k,
			stride,
			pad,
			has_bias,
			alloc,
			kernel_w = k_w,
			stride_w = stride_w,
			padding_w = pad_w,
		)
	case .Conv1d, .Conv1d_SG:
		c_in, c_out, k, dilation, pad_left, pad_right :=
			_get_size(io),
			_get_size(io),
			_get_size(io),
			_get_size(io),
			_get_size(io),
			_get_size(io)
		has_bias := _get_i32(io) != 0
		stride, groups := 1, 1
		if _Layer_Id(id) == .Conv1d_SG {stride, groups = _get_size(io), _get_size(io)}
		if !io.ok || k == 0 || dilation == 0 || stride == 0 || groups == 0 {return}
		if c_in % groups != 0 || c_out % groups != 0 {
			io.ok = false
			return
		}
		conv := conv1d_layer_new(
			c_in,
			c_out,
			k,
			dilation,
			use_bias = has_bias,
			allocator = alloc,
			stride = stride,
			groups = groups,
		)
		conv.pad_left, conv.pad_right = pad_left, pad_right
		layer = conv
	case .TCNBlock:
		c_in, c_out, k, dilation := _get_size(io), _get_size(io), _get_size(io), _get_size(io)
		dropout := _get_f64(io)
		if !io.ok || k == 0 || dilation == 0 {return}
		layer = tcn_block_new(c_in, c_out, k, dilation, dropout, alloc)
	case .MaxPool2d:
		k, s := _get_size(io), _get_size(io)
		layer = maxpool2d_layer_new(k, s)
	case .Activation:
		act := _get_i32(io)
		if act < 0 || act > int(max(Activation)) {io.ok = false}
		layer = Activation(act)
	case .Flatten:
		layer = FlattenLayer{}
	case .BatchNorm2d:
		n := _get_size(io)
		eps, momentum := _get_f64(io), _get_f64(io)
		if !io.ok {return}
		layer = batch_norm_2d_layer_new(n, eps, momentum, alloc)
	case .AvgPool2d:
		k, s := _get_size(io), _get_size(io)
		layer = avgpool2d_layer_new(k, s)
	case .Dropout:
		layer = dropout_layer_new(_get_f64(io))
	case .RNN, .GRU, .GRU_0, .LSTM:
		in_size, hidden := _get_size(io), _get_size(io)
		if !io.ok {return}
		#partial switch _Layer_Id(id) {
		case .RNN:
			layer = rnn_layer_new(in_size, hidden, alloc)
		case .GRU, .GRU_0:
			gru := gru_layer_new(in_size, hidden, alloc)
			if _Layer_Id(id) == .GRU_0 {
				t.tensor_free(gru.bias_hn)
				gru.bias_hn = nil
			}
			layer = gru
		case .LSTM:
			layer = lstm_layer_new(in_size, hidden, alloc)
		}
	case .Embedding:
		n, dim := _get_size(io), _get_size(io)
		if !io.ok {return}
		layer = embedding_layer_new(n, dim, alloc)
	case .MHA, .MHA_0:
		d, heads := _get_size(io), _get_size(io)
		if !io.ok || heads == 0 {return}
		layer = multi_head_attention_layer_new(d, heads, allocator = alloc)
	case .LayerNorm, .RMSNorm:
		d := _get_size(io)
		eps := _get_f64(io)
		if !io.ok {return}
		layer = (rms_norm_layer_new if _Layer_Id(id) == .RMSNorm else layer_norm_layer_new)(
			d,
			eps,
			alloc,
		)
	case .FFN, .FFN_0:
		d, d_ff := _get_size(io), _get_size(io)
		gelu_tanh := _Layer_Id(id) == .FFN && _get_i32(io) != 0
		if !io.ok {return}
		layer = ffn_layer_new(d, d_ff, gelu_tanh, alloc)
	case .EncoderBlock, .EncoderBlock_0:
		d, heads, d_ff := _get_size(io), _get_size(io), _get_size(io)
		if !io.ok || heads == 0 {return}
		layer = transformer_encoder_block_new(d, heads, d_ff, alloc)
	case .Encoder, .Encoder_0:
		n, d, heads, d_ff := _get_size(io), _get_size(io), _get_size(io), _get_size(io)
		if !io.ok || heads == 0 {return}
		layer = transformer_encoder_new(n, d, heads, d_ff, alloc)
	case .GAT, .GAT_0:
		in_f, hidden, heads := _get_size(io), _get_size(io), _get_size(io)
		has_bias := _get_i32(io) != 0
		if !io.ok || heads == 0 {return}
		gat := gat_layer_new(in_f, hidden, heads, nil, alloc) // the adjacency is read with the tensors
		if !has_bias {
			t.tensor_free(gat.linear.bias)
			gat.linear.bias = nil
		}
		layer = gat
	case .Mamba, .Mamba_0:
		d, n := _get_size(io), _get_size(io)
		if !io.ok || n < 2 {return}
		m := mamba_layer_new(d, n, alloc)
		m.plain = _Layer_Id(id) == .Mamba_0 // saved before the block form: its own function
		layer = m
	case .MoE:
		e, k, d, d_ff := _get_size(io), _get_size(io), _get_size(io), _get_size(io)
		if !io.ok {return}
		layer = moe_layer_new(d, d_ff, e, k, alloc)
	case:
		fmt.printf("Unknown layer type %d\n", id)
		io.ok = false
	}
	return layer, flags, io.ok
}

// ============================================================================
// Sequential checkpoints
// ============================================================================

// Saves a Sequential model with its Adam state and the epoch
save_checkpoint :: proc(
	model: ^Sequential,
	opt: ^Adam,
	path: string,
	epoch: int,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	io := _writer(allocator)
	_put_i32(&io, epoch)
	_put_i32(&io, len(model.layers))
	for &layer in model.layers {
		_write_layer_header(&io, &layer)
		_io_layer(&io, &layer, true)
	}

	_put_i32(&io, 3) // optimizer type: frozen flags, Adam with hyperparameters and per-parameter steps
	all := sequential_parameters(model, context.temp_allocator, all = true)
	_put_i32(&io, len(all))
	for p in all {_put_i32(&io, int(!p.requires_grad))}
	delete(all)
	host_copies := _adam_state_to_host(opt)
	defer for i in host_copies {_adam_free_host_moments(opt, i)} 	// their moments stay on the device
	_put_i64(&io, opt.timestep)
	for v in ([]f64{opt.learning_rate, opt.beta_1, opt.beta_2, opt.epsilon, opt.weight_decay}) {_put_f64(&io, v)}
	_put_i32(&io, len(opt.parameters))
	for i in 0 ..< len(opt.parameters) {
		for m in ([][]f64{opt.moment_1[i], opt.moment_2[i]}) {
			_put_i32(&io, len(m))
			for v in m {_put_f64(&io, v)}
		}
	}
	for n in opt.steps {_put_i64(&io, n)}
	return _save(&io, path)
}

// Loads a Sequential model (in evaluation mode, frozen parameters frozen
// again), its Adam optimizer (with the trainable parameters attached and
// their moments restored) and the epoch. Free with sequential_free, adam_free
// and free(opt).
load_checkpoint :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	model: ^Sequential,
	opt: ^Adam,
	epoch: int,
	ok: bool,
) {
	rng: rand.Default_Random_State
	context.random_generator = rand.default_random_generator(&rng)
	io := _reader(path, allocator) or_return
	defer delete(io.data, allocator)

	epoch = _get_i32(&io)
	n_layers := _get_size(&io)
	model = sequential_new(allocator)
	model.training = false // loaded models start in evaluation mode (sequential_train to resume)
	for _ in 0 ..< n_layers {
		layer, flags, layer_ok := _read_layer_header(&io, allocator)
		if !layer_ok {
			io.ok = false // also an invalid configuration (e.g. 0 heads): a corrupt file
			break
		}
		append(&model.layers, layer)
		_io_layer(&io, &model.layers[len(model.layers) - 1], flags)
	}

	// Optimizer
	opt_type := _get_i32(&io)
	if io.ok && (opt_type < 0 || opt_type > 3) {
		fmt.println("Unsupported optimizer type")
		io.ok = false
	}
	if opt_type == 3 {
		all := sequential_parameters(model, context.temp_allocator, all = true)
		defer delete(all)
		n_flags := _get_size(&io)
		if io.ok && n_flags != len(all) {
			io.ok = false // the parameters don't match the layers: a corrupt file
		}
		for i in 0 ..< n_flags {
			frozen := _get_i32(&io) != 0
			if io.ok && frozen {tensor_set_requires_grad(all[i], false)}
		}
	}
	opt = new(Adam, allocator)
	opt^ = adam_new(allocator = allocator)
	opt.timestep = _get_i64(&io)
	if opt_type >= 1 {
		opt.learning_rate, opt.beta_1, opt.beta_2 = _get_f64(&io), _get_f64(&io), _get_f64(&io)
		opt.epsilon, opt.weight_decay = _get_f64(&io), _get_f64(&io)
	}

	// Moments belong to the model's trainable parameters in the order of
	// sequential_parameters; if they don't fit, training restarts from zero
	// moments.
	params := sequential_parameters(model, context.temp_allocator)
	defer delete(params)
	for p in params {adam_add_param(opt, p)}
	n_saved := _get_size(&io)
	fits := n_saved == len(params)
	for i in 0 ..< 2 * n_saved {
		count := _get_size(&io)
		fits = fits && count == t.tensor_numel(params[i / 2])
		values := _take(&io, 8 * count)
		if !fits || values == nil {continue}
		m := opt.moment_1[i / 2] if i % 2 == 0 else opt.moment_2[i / 2]
		for &v, j in m {v, _ = endian.get_f64(values[8 * j:], .Little)}
	}
	// per-parameter steps (type 2 and 3); older checkpoints had one shared count
	for i in 0 ..< n_saved {
		n := _get_i64(&io) if opt_type >= 2 else opt.timestep
		if fits {opt.steps[i] = n}
	}
	if !fits {
		if n_saved > 0 {
			fmt.println("load_checkpoint: optimizer state does not match the model; moments reset")
		}
		for i in 0 ..< len(params) {
			mem.zero_slice(opt.moment_1[i])
			mem.zero_slice(opt.moment_2[i])
			opt.steps[i] = 0
		}
		opt.timestep = 0
	}

	if !io.ok {
		fmt.printf("%s: corrupt or truncated checkpoint\n", path)
		sequential_free(model)
		adam_free(opt)
		free(opt, allocator)
		return nil, nil, 0, false
	}
	return model, opt, epoch, true
}

// Loads the layers of a checkpoint into `seq` where they match: a layer is
// loaded when its type and configuration equal the checkpoint's at the same
// position, and skipped otherwise (e.g. a new output head).
sequential_load_partial :: proc(
	seq: ^Sequential,
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	loaded: int,
	skipped: int,
	ok: bool,
) {
	rng: rand.Default_Random_State
	context.random_generator = rand.default_random_generator(&rng)
	io := _reader(path, allocator) or_return
	defer delete(io.data, allocator)

	_ = _get_i32(&io) // epoch
	n_layers := _get_size(&io)
	for i in 0 ..< min(n_layers, len(seq.layers)) {
		stored, flags, layer_ok := _read_layer_header(&io, allocator)
		if !layer_ok {
			io.ok = false // also an invalid configuration (e.g. 0 heads): a corrupt file
			break
		}
		defer layer_free(&stored)

		// same type and configuration: the headers written for both layers
		// have the same bytes (also for files of an older layer id)
		headers := _writer(context.temp_allocator)
		_write_layer_header(&headers, &stored)
		mid := len(headers.buf)
		_write_layer_header(&headers, &seq.layers[i])
		same := string(headers.buf[len(CHECKPOINT_MAGIC):mid]) == string(headers.buf[mid:])
		delete(headers.buf)

		// a plain Mamba layer (an older file) loads its weights into a Mamba
		// block of the same size (norm weight 1): an initialization, since the
		// block computes another function
		old_m, is_old_m := stored.(MambaLayer)
		new_m, is_new_m := &seq.layers[i].(MambaLayer)
		if !same &&
		   is_old_m &&
		   is_new_m &&
		   old_m.plain &&
		   !new_m.plain &&
		   old_m.d_model == new_m.d_model &&
		   old_m.d_state == new_m.d_state {
			_io_layer(&io, &stored, flags)
			src := stored.(MambaLayer)
			for pair in ([][2]^LinearLayer{{&new_m.proj_x, &src.proj_x}, {&new_m.proj_B, &src.proj_B}, {&new_m.proj_C, &src.proj_C}, {&new_m.proj_Delta, &src.proj_Delta}, {&new_m.proj_out, &src.proj_out}}) {
				copy(pair[0].weights.data.data, pair[1].weights.data.data)
				if pair[0].bias != nil &&
				   pair[1].bias != nil {copy(pair[0].bias.data.data, pair[1].bias.data.data)}
			}
			copy(new_m.A.data.data, src.A.data.data)
			copy(new_m.D.data.data, src.D.data.data)
			loaded += 1
			continue
		}

		// a GRU stored without bias_hn loads into a GRU of the same size with
		// bias_hn = 0 (the same function)
		old_gru, is_old_gru := stored.(GRULayer)
		is_old_gru &&= old_gru.bias_hn == nil
		new_gru, is_new_gru := &seq.layers[i].(GRULayer)
		if !same &&
		   is_old_gru &&
		   is_new_gru &&
		   new_gru.bias_hn != nil &&
		   new_gru.input_size == old_gru.input_size &&
		   new_gru.hidden_size == old_gru.hidden_size {
			_io_layer(&io, &stored, flags)
			copy(new_gru.w_ih.data.data, stored.(GRULayer).w_ih.data.data)
			copy(new_gru.w_hh.data.data, stored.(GRULayer).w_hh.data.data)
			copy(new_gru.bias.data.data, stored.(GRULayer).bias.data.data)
			mem.zero_slice(new_gru.bias_hn.data.data)
			loaded += 1
		} else if same {
			_io_layer(&io, &seq.layers[i], flags)
			loaded += 1
		} else {
			_io_layer(&io, &stored, flags)
			skipped += 1
		}
	}
	if !io.ok {
		fmt.printf("%s: corrupt or truncated checkpoint\n", path)
		return loaded, skipped, false
	}
	return loaded, skipped, true
}

// The count of the parameters and a frozen flag (i32) for each
@(private = "file")
_put_frozen :: proc(io: ^_Io, ps: []^t.Tensor) {
	_put_i32(io, len(ps))
	for p in ps {_put_i32(io, int(!p.requires_grad))}
}

// Reads _put_frozen's flags and freezes the flagged parameters; a count that
// doesn't match the model marks the file as corrupt
@(private = "file")
_get_frozen :: proc(io: ^_Io, ps: []^t.Tensor) {
	n := _get_size(io)
	if io.ok && n != len(ps) {io.ok = false}
	for i in 0 ..< n {
		frozen := _get_i32(io) != 0
		if !io.ok {return}
		if frozen {tensor_set_requires_grad(ps[i], false)}
	}
}

// ============================================================================
// Parameter lists (models of any structure)
// ============================================================================
// For models that are not Sequential, GPT or BERT: the caller lists the
// parameters (always in the same order) and rebuilds the model itself, e.g.
// from a configuration stored as metadata. With an Adam optimizer, a restarted
// run continues exactly as the uninterrupted one would.

// Saves the values of `params` in order with each one's frozen flag and, when
// `opt` is not nil, the Adam state of those registered in it: the
// hyperparameters and timestep, and per parameter its moments, step count and
// learning-rate scale. Parameters registered in `opt` but not in `params` are
// not saved. `metadata` is stored as it is (e.g. the model configuration and
// the training step). Parameters and moments may be on the host or the GPU;
// they stay where they are. Returns false (without a message) if the file can't
// be written or a parameter is nil.
save_parameters :: proc(
	path: string,
	params: []^t.Tensor,
	opt: ^Adam = nil,
	metadata: []u8 = nil,
) -> bool {
	for p in params {if p == nil {return false}}
	io := _writer(context.allocator)
	_put_i32(&io, int(_File_Id.Parameters))
	_put_i32(&io, len(metadata))
	append(&io.buf, ..metadata)
	_put_i32(&io, len(params))
	for &p in params {
		_put_i32(&io, int(!p.requires_grad))
		_io_tensor(&io, &p)
	}
	_put_i32(&io, int(opt != nil))
	if opt != nil {
		host_copies := _adam_state_to_host(opt)
		defer for i in host_copies {_adam_free_host_moments(opt, i)} 	// their moments stay on the device
		_put_i64(&io, opt.timestep)
		for v in ([]f64{opt.learning_rate, opt.beta_1, opt.beta_2, opt.epsilon, opt.weight_decay}) {_put_f64(&io, v)}
		for p in params {
			i := _adam_index(opt, p)
			_put_i32(&io, int(i >= 0))
			if i < 0 {continue}
			_put_f64(&io, opt.lr_scales[i])
			_put_i64(&io, opt.steps[i])
			for m in ([][]f64{opt.moment_1[i], opt.moment_2[i]}) {
				_put_i32(&io, len(m))
				start := len(io.buf)
				resize(&io.buf, start + 8 * len(m))
				for v, j in m {endian.put_f64(io.buf[start + 8 * j:], .Little, v)}
			}
		}
	}
	return _save(&io, path, quiet = true)
}

// Loads a file of save_parameters into `params`, which must have the stored
// count and shapes (host or GPU tensors; GPU tensors upload the values on
// their next use), and restores their frozen flags. When `opt` is not nil and
// the file holds Adam state, the parameters registered in `opt` get their
// moments, step counts and learning-rate scales back, and `opt` its
// hyperparameters and timestep; the parameters registered must be the ones
// that were registered when saving (matched by position in `params`). Returns
// the metadata, allocated with `allocator` (nil if none was stored). A missing,
// truncated or foreign file, or a count, shape or registration that doesn't
// match, returns ok = false and changes nothing.
load_parameters :: proc(
	path: string,
	params: []^t.Tensor,
	opt: ^Adam = nil,
	allocator := context.allocator,
) -> (
	metadata: []u8,
	ok: bool,
) {
	io, meta := _read_parameters_header(path) or_return
	defer delete(io.data)

	// read and check everything before changing anything
	Stored :: struct {
		frozen:     bool,
		values:     []u8,
		registered: bool,
		lr_scale:   f64,
		steps:      int,
		moments:    [2][]u8,
	}
	stored := make([]Stored, len(params))
	defer delete(stored)
	if _get_size(&io) != len(params) {return nil, false}
	for p, j in params {
		s := &stored[j]
		s.frozen = _get_i32(&io) != 0
		rows, cols := _get_size(&io), _get_size(&io)
		if !io.ok || p == nil || rows != p.data.rows || cols != p.data.cols {return nil, false}
		s.values = _take(&io, 8 * rows * cols)
	}
	has_adam := _get_i32(&io) != 0
	timestep: int
	hyper: [5]f64 // learning rate, beta_1, beta_2, epsilon, weight decay
	if has_adam {
		timestep = _get_i64(&io)
		for &v in hyper {v = _get_f64(&io)}
		for p, j in params {
			s := &stored[j]
			s.registered = _get_i32(&io) != 0
			if !s.registered {continue}
			s.lr_scale, s.steps = _get_f64(&io), _get_i64(&io)
			for &m in s.moments {
				if _get_size(&io) != t.tensor_numel(p) {io.ok = false}
				m = _take(&io, 8 * t.tensor_numel(p))
			}
		}
	}
	if !io.ok || io.pos != len(io.data) {return nil, false}
	restore_adam := opt != nil && has_adam
	if restore_adam {
		for p, j in params {
			if (_adam_index(opt, p) >= 0) != stored[j].registered {return nil, false}
		}
	}

	for p, j in params {
		s := stored[j]
		for &v, k in t.tensor_values(p) {v, _ = endian.get_f64(s.values[8 * k:], .Little)}
		t.tensor_mark_host_dirty(p)
		if p.requires_grad == s.frozen {tensor_set_requires_grad(p, !s.frozen)}
	}
	if restore_adam {
		opt.timestep = timestep
		opt.learning_rate, opt.beta_1, opt.beta_2 = hyper[0], hyper[1], hyper[2]
		opt.epsilon, opt.weight_decay = hyper[3], hyper[4]
		for p, j in params {
			s := stored[j]
			if !s.registered {continue}
			i := _adam_index(opt, p)
			opt.lr_scales[i], opt.steps[i] = s.lr_scale, s.steps
			// host moments: a device parameter uploads them at its next step
			if t._device_hooks.optim_release !=
			   nil {t._device_hooks.optim_release(opt.device_keys[i])}
			if opt.moment_1[i] == nil {
				opt.moment_1[i] = make([]f64, len(s.moments[0]) / 8, opt.allocator)
				opt.moment_2[i] = make([]f64, len(s.moments[1]) / 8, opt.allocator)
			}
			for m, k in ([][]f64{opt.moment_1[i], opt.moment_2[i]}) {
				for &v, n in m {v, _ = endian.get_f64(s.moments[k][8 * n:], .Little)}
			}
		}
	}
	if len(meta) > 0 {
		metadata = make([]u8, len(meta), allocator)
		copy(metadata, meta)
	}
	return metadata, true
}

// The metadata of a file of save_parameters (allocated with `allocator`, nil
// if none was stored), without loading the parameters: e.g. the configuration
// to build the model before load_parameters
read_parameters_metadata :: proc(
	path: string,
	allocator := context.allocator,
) -> (
	metadata: []u8,
	ok: bool,
) {
	io, meta := _read_parameters_header(path) or_return
	defer delete(io.data)
	if len(meta) > 0 {
		metadata = make([]u8, len(meta), allocator)
		copy(metadata, meta)
	}
	return metadata, true
}

// Reads a parameter-list file (context.allocator) up to its parameters and
// returns the metadata bytes (a slice of io.data)
@(private = "file")
_read_parameters_header :: proc(path: string) -> (io: _Io, metadata: []u8, ok: bool) {
	io = _reader(path, context.allocator, quiet = true) or_return
	if _File_Id(_get_i32(&io)) != .Parameters {
		delete(io.data)
		return {}, nil, false
	}
	metadata = _take(&io, _get_size(&io))
	if !io.ok {
		delete(io.data)
		return {}, nil, false
	}
	return io, metadata, true
}

// The position of `p` among the parameters registered in `opt`, or -1
@(private = "file")
_adam_index :: proc(opt: ^Adam, p: ^t.Tensor) -> int {
	for q, i in opt.parameters {if q == p {return i}}
	return -1
}

// ============================================================================
// GPT
// ============================================================================

save_gpt_model :: proc(model: ^GPTModel, path: string) -> bool {
	io := _writer(context.allocator)
	_put_i32(&io, int(_File_Id.GPT))
	for v in ([]int{model.vocab_size, model.d_model, model.num_heads, model.d_ff, model.num_layers, model.max_seq_len}) {
		_put_i32(&io, v)
	}
	_put_f64(&io, model.dropout)
	_put_i32(&io, int(model.rms_norm))
	_put_i32(&io, int(model.tie_embeddings))
	_put_i32(&io, int(model.gelu_tanh))
	_io_gpt(&io, model, .GPT)
	ps := gpt_model_parameters(model, context.temp_allocator, all = true)
	_put_frozen(&io, ps[:])
	delete(ps)
	return _save(&io, path)
}

// Loads a GPT model with its blocks as stored (MoE or dense, FlashAttention,
// causal). use_flash applies to older files that don't store it.
load_gpt_model :: proc(
	path: string,
	use_flash: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> (
	model: ^GPTModel,
	ok: bool,
) {
	rng: rand.Default_Random_State
	context.random_generator = rand.default_random_generator(&rng)
	io := _reader(path, allocator) or_return
	defer delete(io.data, allocator)

	file_id := _File_Id(_get_i32(&io))
	if file_id != .GPT &&
	   file_id != .GPT_4 &&
	   file_id != .GPT_3 &&
	   file_id != .GPT_2 &&
	   file_id != .GPT_1 &&
	   file_id != .GPT_MoE &&
	   file_id != .GPT_Dense {
		fmt.printf("%s is not a GPT model\n", path)
		return nil, false
	}
	cfg: [6]int // vocab, d_model, heads, d_ff, layers, max_seq_len
	for &v in cfg {v = _get_size(&io)}
	current := file_id == .GPT || file_id == .GPT_4 || file_id == .GPT_3
	dropout := _get_f64(&io) if current || file_id == .GPT_2 else 0.1
	rms_norm := current && _get_i32(&io) != 0
	tie_embeddings := current && _get_i32(&io) != 0
	gelu_tanh := (file_id == .GPT || file_id == .GPT_4) && _get_i32(&io) != 0
	if !io.ok || cfg[2] == 0 {
		fmt.printf("%s: corrupt or truncated model\n", path)
		return nil, false
	}
	model = new(GPTModel, allocator)
	model^ = gpt_model_new(
		cfg[0],
		cfg[1],
		cfg[2],
		cfg[3],
		cfg[4],
		cfg[5],
		use_flash = use_flash,
		dropout = dropout,
		rms_norm = rms_norm,
		tie_embeddings = tie_embeddings,
		gelu_tanh = gelu_tanh,
		allocator = allocator,
	)
	_io_gpt(&io, model, file_id)
	if file_id == .GPT {
		ps := gpt_model_parameters(model, context.temp_allocator, all = true)
		_get_frozen(&io, ps[:])
		delete(ps)
	}
	if !io.ok {
		fmt.printf("%s: corrupt or truncated model\n", path)
		gpt_model_free(model)
		free(model, allocator)
		return nil, false
	}
	return model, true
}

// GPT tensors in file order; each block starts with its flags (none in the
// oldest version)
@(private = "file")
_io_gpt :: proc(io: ^_Io, model: ^GPTModel, version: _File_Id) {
	_io_tensor(io, &model.token_emb.weight)
	_io_tensor(io, &model.pos_emb.weight)
	for &block in model.blocks {
		is_moe := version != .GPT_Dense && _io_flag(io, block.use_moe)
		_io_mha(
			io,
			&block.mha,
			version == .GPT ||
			version == .GPT_4 ||
			version == .GPT_3 ||
			version == .GPT_2 ||
			version == .GPT_1,
		)
		if is_moe {
			num_experts, top_k := block.moe.num_experts, block.moe.top_k
			if io.writing {
				_put_i32(io, num_experts)
				_put_i32(io, top_k)
			} else {
				num_experts, top_k = _get_size(io), _get_size(io)
				if !io.ok {return}
				ffn_layer_free(&block.ffn)
				block.ffn = {}
				block.moe = moe_layer_new(model.d_model, model.d_ff, num_experts, top_k, io.alloc)
				for &e in block.moe.experts {e.gelu_tanh = model.gelu_tanh}
				block.use_moe = true
			}
			_io_moe(io, &block.moe)
		} else {
			_io_linear(io, &block.ffn.fc1)
			_io_linear(io, &block.ffn.fc2)
		}
		_io_norm(io, &block.ln1)
		_io_norm(io, &block.ln2)
	}
	_io_norm(io, &model.final_ln)
	_io_linear(io, &model.output_proj)
	model.vocab_size = model.output_proj.out_features
}

// ============================================================================
// BERT
// ============================================================================

save_bert_model :: proc(model: ^BERTModel, path: string) -> bool {
	io := _writer(context.allocator)
	_put_i32(&io, int(_File_Id.BERT))
	for v in ([]int{model.vocab_size, model.d_model, model.num_heads, model.d_ff, model.num_layers, model.max_seq_len}) {
		_put_i32(&io, v)
	}
	_put_f64(&io, model.dropout)
	_io_bert(&io, model, .BERT)
	ps := bert_model_parameters(model, context.temp_allocator, all = true)
	_put_frozen(&io, ps[:])
	delete(ps)
	return _save(&io, path)
}

// Loads a BERT model (heads of another size, e.g. after bert_replace_nsp_head,
// and the attention flags of every block are restored). use_flash applies to
// older files that don't store it.
load_bert_model :: proc(
	path: string,
	use_flash: bool = false,
	allocator: mem.Allocator = context.allocator,
) -> (
	model: ^BERTModel,
	ok: bool,
) {
	rng: rand.Default_Random_State
	context.random_generator = rand.default_random_generator(&rng)
	io := _reader(path, allocator) or_return
	defer delete(io.data, allocator)

	file_id := _File_Id(_get_i32(&io))
	if file_id != .BERT && file_id != .BERT_2 && file_id != .BERT_1 && file_id != .BERT_0 {
		fmt.printf("%s is not a BERT model\n", path)
		return nil, false
	}
	cfg: [6]int // vocab, d_model, heads, d_ff, layers, max_seq_len
	for &v in cfg {v = _get_size(&io)}
	dropout := _get_f64(&io) if file_id == .BERT || file_id == .BERT_2 else 0.1
	if !io.ok || cfg[2] == 0 {
		fmt.printf("%s: corrupt or truncated model\n", path)
		return nil, false
	}
	model = new(BERTModel, allocator)
	model^ = bert_model_new(
		cfg[0],
		cfg[1],
		cfg[2],
		cfg[3],
		cfg[4],
		cfg[5],
		use_flash,
		dropout,
		allocator,
	)
	_io_bert(&io, model, file_id)
	if file_id == .BERT {
		ps := bert_model_parameters(model, context.temp_allocator, all = true)
		_get_frozen(&io, ps[:])
		delete(ps)
	}
	if !io.ok {
		fmt.printf("%s: corrupt or truncated model\n", path)
		bert_model_free(model)
		free(model, allocator)
		return nil, false
	}
	return model, true
}

@(private = "file")
_io_bert :: proc(io: ^_Io, model: ^BERTModel, version: _File_Id) {
	_io_tensor(io, &model.token_emb.weight)
	_io_tensor(io, &model.pos_emb.weight)
	_io_tensor(io, &model.segment_emb.weight)
	_io_norm(io, &model.emb_ln)
	for &b in model.encoder_blocks {
		_io_block(io, &b.mha, &b.ffn, &b.ln1, &b.ln2, version != .BERT_0)
	}
	_io_linear(io, &model.pooler)
	_io_linear(io, &model.mlm_head)
	_io_linear(io, &model.nsp_head)
	model.vocab_size = model.mlm_head.out_features
}

// ============================================================================
// GAN and VAE networks (three linear layers each)
// ============================================================================

@(private = "file")
_save_linears :: proc(path: string, id: _File_Id, layers: ..^LinearLayer) -> bool {
	io := _writer(context.allocator)
	_put_i32(&io, int(id))
	for lin in layers {_io_linear(&io, lin)}
	return _save(&io, path)
}

@(private = "file")
_load_linears :: proc(
	path: string,
	id: _File_Id,
	alloc: mem.Allocator,
	layers: ..^LinearLayer,
) -> bool {
	io := _reader(path, alloc) or_return
	defer delete(io.data, alloc)
	if _File_Id(_get_i32(&io)) != id {
		fmt.printf("%s is not a %v\n", path, id)
		return false
	}
	for lin in layers {_io_linear(&io, lin)}
	if !io.ok {
		fmt.printf("%s: corrupt or truncated model\n", path)
		for lin in layers {linear_layer_free(lin)}
	}
	return io.ok
}

save_generator :: proc(gen: ^Generator, path: string) -> bool {
	return _save_linears(path, .Generator, &gen.fc1, &gen.fc2, &gen.fc3)
}

load_generator :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^Generator,
	bool,
) {
	gen := new(Generator, allocator)
	if !_load_linears(path, .Generator, allocator, &gen.fc1, &gen.fc2, &gen.fc3) {
		free(gen, allocator)
		return nil, false
	}
	return gen, true
}

save_discriminator :: proc(disc: ^Discriminator, path: string) -> bool {
	return _save_linears(path, .Discriminator, &disc.fc1, &disc.fc2, &disc.fc3)
}

load_discriminator :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^Discriminator,
	bool,
) {
	disc := new(Discriminator, allocator)
	if !_load_linears(path, .Discriminator, allocator, &disc.fc1, &disc.fc2, &disc.fc3) {
		free(disc, allocator)
		return nil, false
	}
	return disc, true
}

save_vae_encoder :: proc(enc: ^VAEEncoder, path: string) -> bool {
	return _save_linears(path, .VAE_Encoder, &enc.fc1, &enc.fc_mu, &enc.fc_logvar)
}

load_vae_encoder :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^VAEEncoder,
	bool,
) {
	enc := new(VAEEncoder, allocator)
	if !_load_linears(path, .VAE_Encoder, allocator, &enc.fc1, &enc.fc_mu, &enc.fc_logvar) {
		free(enc, allocator)
		return nil, false
	}
	return enc, true
}

save_vae_decoder :: proc(dec: ^VAEDecoder, path: string) -> bool {
	return _save_linears(path, .VAE_Decoder, &dec.fc1, &dec.fc2, &dec.fc3)
}

load_vae_decoder :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	^VAEDecoder,
	bool,
) {
	dec := new(VAEDecoder, allocator)
	if !_load_linears(path, .VAE_Decoder, allocator, &dec.fc1, &dec.fc2, &dec.fc3) {
		free(dec, allocator)
		return nil, false
	}
	return dec, true
}
