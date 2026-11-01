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
//     configuration and tensors, then the Adam state (type 1 with the
//     hyperparameters; type 0 without them is still read).
//   GPT (31; older: 28 without the dropout rate, 27 also without the
//     attention flags, 17 also without MoE), BERT (32; older: 29 without the
//     dropout rate, 18 also without the attention flags), Generator (19),
//     Discriminator (20), VAE encoder (21) and decoder (22): magic, type id,
//     configuration, tensors. Files without the dropout rate load with 0.1.
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
	Conv2d         = 1,
	MaxPool2d      = 2,
	Activation     = 3,
	Flatten        = 4,
	BatchNorm2d    = 5,
	AvgPool2d      = 6,
	Dropout        = 7,
	RNN            = 8,
	GRU            = 9,
	LSTM           = 10,
	Embedding      = 11,
	LayerNorm      = 13,
	FFN            = 14,
	Mamba          = 25,
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
	GPT           = 31, // dropout rate; per block: MoE, use_flash and is_causal flags
	BERT          = 32, // dropout rate; per block: use_flash and is_causal flags
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

// Writes the file and releases the buffer
@(private = "file")
_save :: proc(io: ^_Io, path: string) -> bool {
	defer delete(io.buf)
	if err := os.write_entire_file(path, io.buf[:]); err != nil {
		fmt.printf("Failed to write %s: %v\n", path, err)
		return false
	}
	return true
}

// Reads a file and checks the magic bytes
@(private = "file")
_reader :: proc(path: string, alloc: mem.Allocator) -> (io: _Io, ok: bool) {
	data, err := os.read_entire_file(path, alloc)
	if err != nil {
		fmt.printf("Failed to read %s: %v\n", path, err)
		return {}, false
	}
	io = _Io {
		data  = data,
		ok    = true,
		alloc = alloc,
	}
	magic := _take(&io, len(CHECKPOINT_MAGIC))
	if magic == nil || mem.compare(magic, CHECKPOINT_MAGIC) != 0 {
		fmt.printf("%s is not a wotan model file\n", path)
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
	if x^ == nil || len(x^.data.data) != n {
		requires_grad := x^ == nil || x^.requires_grad
		if x^ != nil {t.tensor_free(x^)}
		x^ = t.tensor_new(l.matrix_new(f64, rows, cols, io.alloc), requires_grad, io.alloc)
	}
	for &v, i in x^.data.data {v, _ = endian.get_f64(bytes[8 * i:], .Little)}
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
// flags: the attention flags are stored (not in the older layer ids)
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
		ints(
			io,
			.Conv2d,
			l.in_channels,
			l.out_channels,
			l.kernel_size,
			l.stride,
			l.padding,
			int(l.bias != nil),
		)
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
		ints(io, .GRU, l.input_size, l.hidden_size)
	case LSTMLayer:
		ints(io, .LSTM, l.input_size, l.hidden_size)
	case EmbeddingLayer:
		ints(io, .Embedding, l.num_embeddings, l.embedding_dim)
	case MultiHeadAttentionLayer:
		ints(io, .MHA, l.d_model, l.num_heads)
	case LayerNormLayer:
		ints(io, .LayerNorm, l.d_model)
		_put_f64(io, l.eps)
	case FFNLayer:
		ints(io, .FFN, l.d_model, l.d_ff)
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
		ints(io, .Mamba, l.d_model, l.d_state)
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
	case .MHA, .EncoderBlock, .Encoder, .GAT:
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
	case .Conv2d:
		c_in, c_out, k, stride, pad :=
			_get_size(io), _get_size(io), _get_size(io), _get_size(io), _get_size(io)
		has_bias := _get_i32(io) != 0
		if !io.ok {return}
		layer = conv2d_layer_new(c_in, c_out, k, stride, pad, has_bias, alloc)
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
	case .RNN, .GRU, .LSTM:
		in_size, hidden := _get_size(io), _get_size(io)
		if !io.ok {return}
		#partial switch _Layer_Id(id) {
		case .RNN:
			layer = rnn_layer_new(in_size, hidden, alloc)
		case .GRU:
			layer = gru_layer_new(in_size, hidden, alloc)
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
	case .LayerNorm:
		d := _get_size(io)
		eps := _get_f64(io)
		if !io.ok {return}
		layer = layer_norm_layer_new(d, eps, alloc)
	case .FFN:
		d, d_ff := _get_size(io), _get_size(io)
		if !io.ok {return}
		layer = ffn_layer_new(d, d_ff, alloc)
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
	case .Mamba:
		d, n := _get_size(io), _get_size(io)
		if !io.ok || n < 2 {return}
		layer = mamba_layer_new(d, n, alloc)
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

	_put_i32(&io, 1) // optimizer type: Adam with hyperparameters
	_put_i64(&io, opt.timestep)
	for v in ([]f64{opt.learning_rate, opt.beta_1, opt.beta_2, opt.epsilon, opt.weight_decay}) {_put_f64(&io, v)}
	_put_i32(&io, len(opt.parameters))
	for i in 0 ..< len(opt.parameters) {
		for m in ([][]f64{opt.moment_1[i], opt.moment_2[i]}) {
			_put_i32(&io, len(m))
			for v in m {_put_f64(&io, v)}
		}
	}
	return _save(&io, path)
}

// Loads a Sequential model (in evaluation mode), its Adam optimizer (with the
// parameters attached and their moments restored) and the epoch. Free with
// sequential_free, adam_free and free(opt).
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
		if !layer_ok {break}
		append(&model.layers, layer)
		_io_layer(&io, &model.layers[len(model.layers) - 1], flags)
	}

	// Optimizer
	opt_type := _get_i32(&io)
	if io.ok && opt_type != 0 && opt_type != 1 {
		fmt.println("Unsupported optimizer type")
		io.ok = false
	}
	opt = new(Adam, allocator)
	opt^ = adam_new(allocator = allocator)
	opt.timestep = _get_i64(&io)
	if opt_type == 1 {
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
		fits = fits && count == len(params[i / 2].data.data)
		values := _take(&io, 8 * count)
		if !fits || values == nil {continue}
		m := opt.moment_1[i / 2] if i % 2 == 0 else opt.moment_2[i / 2]
		for &v, j in m {v, _ = endian.get_f64(values[8 * j:], .Little)}
	}
	if !fits {
		if n_saved > 0 {
			fmt.println("load_checkpoint: optimizer state does not match the model; moments reset")
		}
		for i in 0 ..< len(params) {
			mem.zero_slice(opt.moment_1[i])
			mem.zero_slice(opt.moment_2[i])
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
		if !layer_ok {break}
		defer layer_free(&stored)

		// same type and configuration: the headers written for both layers
		// have the same bytes (also for files of an older layer id)
		headers := _writer(context.temp_allocator)
		_write_layer_header(&headers, &stored)
		mid := len(headers.buf)
		_write_layer_header(&headers, &seq.layers[i])
		same := string(headers.buf[len(CHECKPOINT_MAGIC):mid]) == string(headers.buf[mid:])
		delete(headers.buf)

		if same {
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
	_io_gpt(&io, model, .GPT)
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
	if file_id != .GPT && file_id != .GPT_1 && file_id != .GPT_MoE && file_id != .GPT_Dense {
		fmt.printf("%s is not a GPT model\n", path)
		return nil, false
	}
	cfg: [6]int // vocab, d_model, heads, d_ff, layers, max_seq_len
	for &v in cfg {v = _get_size(&io)}
	dropout := _get_f64(&io) if file_id == .GPT else 0.1
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
		allocator = allocator,
	)
	_io_gpt(&io, model, file_id)
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
		_io_mha(io, &block.mha, version == .GPT || version == .GPT_1)
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
	return _save(&io, path)
}

// Loads a BERT model (heads of another size, e.g. after bert_replace_nsp_head,
// and the attention flags of every block are restored). use_flash applies to
// older files that don't store it.
load_bert_model :: proc(
	path: string,
	allocator: mem.Allocator = context.allocator,
	use_flash: bool = false,
) -> (
	model: ^BERTModel,
	ok: bool,
) {
	rng: rand.Default_Random_State
	context.random_generator = rand.default_random_generator(&rng)
	io := _reader(path, allocator) or_return
	defer delete(io.data, allocator)

	file_id := _File_Id(_get_i32(&io))
	if file_id != .BERT && file_id != .BERT_1 && file_id != .BERT_0 {
		fmt.printf("%s is not a BERT model\n", path)
		return nil, false
	}
	cfg: [6]int // vocab, d_model, heads, d_ff, layers, max_seq_len
	for &v in cfg {v = _get_size(&io)}
	dropout := _get_f64(&io) if file_id == .BERT else 0.1
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
