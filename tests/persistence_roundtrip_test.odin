package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

// Save / load of every layer type and model: bytes after a load and re-save,
// outputs before and after, partial loading, corrupt files, freezing.

@(private = "file")
PR_DIR :: "model_checkpoints/"

@(private = "file")
_pr_file :: proc(name: string) -> string {
	return fmt.tprintf("%spersistence_test_%s.bin", PR_DIR, name)
}

// The files a and b have the same bytes
@(private = "file")
_pr_same_file :: proc(a, b: string) -> bool {
	x, _ := os.read_entire_file(a, context.allocator)
	y, _ := os.read_entire_file(b, context.allocator)
	defer delete(x)
	defer delete(y)
	return len(x) > 0 && mem.compare(x, y) == 0
}

@(private = "file")
_pr_randomize :: proc(ps: []^t.Tensor) {
	for p in ps {for &v in p.data.data {v = rand.float64() - 0.5}}
}

@(private = "file")
_pr_input :: proc(shape: [4]int, ints: int = 0) -> ^t.Tensor {
	n := shape[0] * shape[1] * shape[2] * shape[3]
	x := t.tensor_new(l.matrix_new(f64, 1, n), false)
	x.shape = shape
	for &v in x.data.data {v = f64(rand.int_max(ints)) if ints > 0 else rand.float64() - 0.5}
	return x
}

@(private = "file")
_pr_max_diff :: proc(a, b: ^t.Tensor) -> f64 {
	worst := 0.0
	for v, i in a.data.data {worst = max(worst, abs(v - b.data.data[i]))}
	return worst
}

persistence_roundtrip_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Persistence (save / load round trips) ===")
	context.allocator = allocator
	os.make_directory(PR_DIR)
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	files := make([dynamic]string)
	defer {
		for f in files {_ = os.remove(f)}
		delete(files)
	}
	file :: proc(files: ^[dynamic]string, name: string) -> string {
		f := _pr_file(name)
		append(files, f)
		return f
	}

	// 1. Every layer type with Adam state: load + re-save gives the same bytes;
	// attention flags of the MHA, encoder and GAT layers are restored
	{
		adj := t.tensor_new(l.matrix_new(f64, 4, 4), false)
		for &v in adj.data.data {v = rand.float64()}
		bn := nn.batch_norm_2d_layer_new(3)
		for &v in bn.running_var.data.data {v = 1 + rand.float64()}
		seq := nn.sequential_new()
		nn.sequential_add(
			seq,
			nn.linear_layer_new(5, 4),
			nn.conv2d_layer_new(2, 3, 3, stride = 2, padding = 1),
			nn.conv2d_layer_new(2, 3, 3, use_bias = false),
			nn.maxpool2d_layer_new(2),
			nn.Activation.Tanh,
			nn.FlattenLayer{},
			bn,
			nn.avgpool2d_layer_new(2, 1),
			nn.dropout_layer_new(0.3),
			nn.rnn_layer_new(3, 4),
			nn.gru_layer_new(3, 4),
			nn.lstm_layer_new(3, 4),
			nn.embedding_layer_new(7, 4),
			nn.multi_head_attention_layer_new(4, 2),
			nn.layer_norm_layer_new(4, 1e-6),
			nn.ffn_layer_new(4, 6),
			nn.transformer_encoder_block_new(4, 2, 6),
			nn.transformer_encoder_new(2, 4, 2, 6),
			nn.gat_layer_new(3, 4, 2, adj),
			nn.mamba_layer_new(4, 3),
			nn.moe_layer_new(4, 6, 3, 2),
			nn.rms_norm_layer_new(4, 1e-6),
			nn.conv1d_layer_new(4, 4, 3, dilation = 2, causal = true),
			nn.tcn_block_new(4, 6, 3, 2, dropout = 0.1),
		)
		mha := &seq.layers[13].(nn.MultiHeadAttentionLayer)
		mha.use_flash, mha.is_causal = true, true
		(&seq.layers[16].(nn.TransformerEncoderBlock)).mha.is_causal = true
		(&seq.layers[17].(nn.TransformerEncoder)).blocks[1].mha.use_flash = true
		(&seq.layers[18].(nn.GATLayer)).mha.use_flash = true
		ps := nn.sequential_parameters(seq)
		_pr_randomize(ps[:])
		opt := nn.adam_new(0.01, allocator = allocator, weight_decay = 0.1)
		nn.sequential_add_to_adam(seq, &opt)
		opt.timestep = 7
		for &n, i in opt.steps {n = 1 + i % 7} 	// per-parameter counts differ from timestep
		for m in opt.moment_1 {for &v in m {v = rand.float64()}}
		a, b := file(&files, "all_a"), file(&files, "all_b")
		ok := nn.save_checkpoint(seq, &opt, a, 3)
		seq2, opt2, epoch, loaded := nn.load_checkpoint(a)
		ok &&= loaded && nn.save_checkpoint(seq2, opt2, b, epoch) && _pr_same_file(a, b)
		ok &&=
			len(seq2.layers) == 24 &&
			epoch == 3 &&
			opt2.timestep == 7 &&
			len(opt2.parameters) == len(ps)
		steps_ok := len(opt2.steps) == len(opt.steps)
		for n, i in opt2.steps {steps_ok &&= n == opt.steps[i]}
		// a checkpoint of the former format (optimizer type 1, one shared step
		// count): every parameter continues from timestep
		old_ok := false
		{
			data, _ := os.read_entire_file(a, context.allocator)
			defer delete(data)
			tail := 4 + 8 + 5 * 8 + 4 + 8 * len(ps) // type, timestep, hyperparameters, count, steps
			for p in ps {tail += 2 * (4 + 8 * len(p.data.data))}
			old := data[:len(data) - 8 * len(ps)]
			type_at := len(data) - tail
			old_type := old[type_at]
			old[type_at] = 1
			c := file(&files, "all_old")
			_ = os.write_entire_file(c, old)
			seq3, opt3, _, loaded3 := nn.load_checkpoint(c)
			if loaded3 {
				old_ok = old_type == 2 && opt3.timestep == 7 && len(opt3.steps) == len(ps)
				for n in opt3.steps {old_ok &&= n == 7}
				for m, i in opt3.moment_1 {old_ok &&= mem.compare_ptrs(raw_data(m), raw_data(opt2.moment_1[i]), 8 * len(m)) == 0}
				nn.sequential_free(seq3)
				nn.adam_free(opt3)
				free(opt3)
			}
		}
		ok &&= steps_ok && old_ok
		conv := seq2.layers[1].(nn.Conv2dLayer)
		bn2 := seq2.layers[6].(nn.BatchNorm2dLayer)
		ok &&= conv.weight.shape == [4]int{3, 2, 3, 3} && !bn2.running_var.requires_grad
		mha2 := seq2.layers[13].(nn.MultiHeadAttentionLayer)
		enc2 := seq2.layers[17].(nn.TransformerEncoder)
		flags_ok :=
			mha2.use_flash &&
			mha2.is_causal &&
			seq2.layers[16].(nn.TransformerEncoderBlock).mha.is_causal &&
			!enc2.blocks[0].mha.use_flash &&
			enc2.blocks[1].mha.use_flash &&
			seq2.layers[18].(nn.GATLayer).mha.use_flash
		rms2 := seq2.layers[21].(nn.LayerNormLayer)
		flags_ok &&= rms2.rms && rms2.beta == nil && rms2.eps == 1e-6
		conv1d2 := seq2.layers[22].(nn.Conv1dLayer)
		tcn2 := seq2.layers[23].(nn.TCNBlock)
		flags_ok &&=
			conv1d2.dilation == 2 &&
			conv1d2.pad_left == 4 &&
			conv1d2.pad_right == 0 &&
			tcn2.dropout == 0.1 &&
			tcn2.conv2.dilation == 2 &&
			tcn2.downsample.weight != nil
		expect(
			&passed,
			&total,
			ok && flags_ok,
			fmt.tprintf(
				"checkpoint with all 21 layer types and Adam state (%d parameters): load + re-save gives the same bytes, per-parameter step counts restored %v, former format (one shared count) still loads %v; Conv2d weights keep shape %v, BatchNorm running statistics stay non-trainable; attention flags restored: %v",
				len(ps),
				steps_ok,
				old_ok,
				conv.weight.shape,
				flags_ok,
			),
		)
		nn.sequential_free(seq2)
		nn.adam_free(opt2)
		free(opt2)
	}

	// 2. Same outputs after loading: CNN (eval), GAT, Mamba
	{
		seq := nn.sequential_new()
		nn.sequential_add(
			seq,
			nn.conv2d_layer_new(2, 3, 3, padding = 1),
			nn.batch_norm_2d_layer_new(3),
			nn.maxpool2d_layer_new(2),
			nn.FlattenLayer{},
			nn.linear_layer_new(12, 2),
		)
		bn := seq.layers[1].(nn.BatchNorm2dLayer)
		for &v in bn.running_mean.data.data {v = rand.float64()}
		for &v in bn.running_var.data.data {v = 1 + rand.float64()}
		adj := t.tensor_new(l.matrix_new(f64, 5, 5), false)
		for &v in adj.data.data {v = rand.float64()}
		seq_g := nn.sequential_new()
		nn.sequential_add(seq_g, nn.gat_layer_new(3, 4, 2, adj), nn.mamba_layer_new(4, 3))
		for s in ([]^nn.Sequential{seq, seq_g}) {
			ps := nn.sequential_parameters(s)
			_pr_randomize(ps[:])
			nn.sequential_eval(s)
		}
		x := _pr_input({2, 2, 4, 4})
		xg := _pr_input({2, 5, 3, 1})
		worst := 0.0
		for s, k in ([]^nn.Sequential{seq, seq_g}) {
			opt := nn.adam_new()
			f := file(&files, fmt.tprintf("out%d", k))
			nn.save_checkpoint(s, &opt, f, 0)
			s2, opt2, _, _ := nn.load_checkpoint(f)
			input := x if k == 0 else xg
			worst = max(
				worst,
				_pr_max_diff(nn.sequential_forward(s, input), nn.sequential_forward(s2, input)),
			)
			nn.adam_free(&opt)
			nn.sequential_free(s2)
			nn.adam_free(opt2)
			free(opt2)
		}
		expect(
			&passed,
			&total,
			worst == 0,
			fmt.tprintf(
				"outputs after loading: Conv2d → BatchNorm2d (eval) → MaxPool → Linear and GAT → Mamba, max diff %.1e",
				worst,
			),
		)
	}

	// 3. GPT: dense / MoE × standard / FlashAttention, a model with mixed blocks
	// (dense + FlashAttention, MoE + non-causal); logits with the causal mask
	// and without it (then is_causal decides). BERT: standard, FlashAttention,
	// mixed blocks; with a replaced head
	{
		ids := _pr_input({2, 5, 1, 1}, 13)
		segs := _pr_input({2, 5, 1, 1}, 2)
		mask := nn.create_causal_mask(5, allocator)
		d_gpt := 0.0
		flags_ok := true
		names := [6]string {
			"dense",
			"MoE",
			"dense + flash",
			"MoE + flash",
			"mixed",
			"RMSNorm + tied embeddings",
		}
		for v in 0 ..< 6 {
			gpt := nn.gpt_model_new(
				13,
				8,
				2,
				16,
				2,
				8,
				use_flash = v == 2 || v == 3,
				use_moe = v % 2 == 1 || v == 4,
				num_experts = 3,
				top_k = 2,
				dropout = 0.05 * f64(v),
				rms_norm = v == 5,
				tie_embeddings = v == 5,
			)
			if v == 4 {
				b0 := &gpt.blocks[0] // dense with FlashAttention
				nn.moe_layer_free(&b0.moe)
				b0.moe = {}
				b0.ffn = nn.ffn_layer_new(8, 16)
				b0.use_moe = false
				b0.mha.use_flash = true
				gpt.blocks[1].mha.is_causal = false
			}
			ps := nn.gpt_model_parameters(&gpt)
			_pr_randomize(ps[:])
			f := file(&files, fmt.tprintf("gpt%d", v))
			saved := nn.save_gpt_model(&gpt, f)
			gpt2, loaded := nn.load_gpt_model(f)
			flags_ok &&= saved && loaded
			if !loaded {continue}
			flags_ok &&= gpt2.dropout == gpt.dropout
			flags_ok &&=
				gpt2.rms_norm == gpt.rms_norm &&
				gpt2.final_ln.rms == gpt.rms_norm &&
				gpt2.tie_embeddings == gpt.tie_embeddings &&
				(gpt2.output_proj.weights == nil) == gpt.tie_embeddings
			for &b, i in gpt.blocks {
				b2 := &gpt2.blocks[i]
				flags_ok &&=
					b.use_moe == b2.use_moe &&
					b.mha.use_flash == b2.mha.use_flash &&
					b.mha.is_causal == b2.mha.is_causal &&
					b2.ln1.rms == gpt.rms_norm &&
					b2.ln2.rms == gpt.rms_norm
				if b.use_moe {flags_ok &&= b2.moe.num_experts == b.moe.num_experts && b2.moe.top_k == b.moe.top_k}
			}
			for m in ([][]f64{mask, nil}) {
				d_gpt = max(
					d_gpt,
					_pr_max_diff(
						nn.gpt_model_forward(&gpt, ids, m, false),
						nn.gpt_model_forward(gpt2, ids, m, false),
					),
				)
			}
			nn.gpt_model_free(gpt2)
		}
		expect(
			&passed,
			&total,
			flags_ok && d_gpt == 0,
			fmt.tprintf(
				"GPT %v: dropout rate, RMSNorm / tied embeddings and MoE / use_flash / is_causal of every block restored: %v, logits after loading max diff %.1e",
				names,
				flags_ok,
				d_gpt,
			),
		)

		d_bert := 0.0
		bert_ok := true
		for v in 0 ..< 3 {
			bert := nn.bert_model_new(
				13,
				8,
				2,
				16,
				2,
				8,
				use_flash = v == 1,
				dropout = 0.15 * f64(v),
			)
			nn.bert_replace_nsp_head(&bert, 3, allocator)
			if v == 2 {
				bert.encoder_blocks[0].mha.use_flash = true
				bert.encoder_blocks[1].mha.is_causal = true
			}
			bps := nn.bert_model_parameters(&bert)
			_pr_randomize(bps[:])
			fb := file(&files, fmt.tprintf("bert%d", v))
			saved := nn.save_bert_model(&bert, fb)
			bert2, loaded := nn.load_bert_model(fb)
			bert_ok &&= saved && loaded
			if !loaded {continue}
			bert_ok &&= bert2.nsp_head.out_features == 3 && bert2.dropout == bert.dropout
			for &b, i in bert.encoder_blocks {
				b2 := &bert2.encoder_blocks[i]
				bert_ok &&=
					b.mha.use_flash == b2.mha.use_flash && b.mha.is_causal == b2.mha.is_causal
			}
			mlm, nsp := nn.bert_model_forward(&bert, ids, segs, false)
			mlm2, nsp2 := nn.bert_model_forward(bert2, ids, segs, false)
			d_bert = max(d_bert, _pr_max_diff(mlm, mlm2), _pr_max_diff(nsp, nsp2))
			nn.bert_model_free(bert2)
		}
		expect(
			&passed,
			&total,
			bert_ok && d_bert == 0,
			fmt.tprintf(
				"BERT [standard, flash, mixed] with a 3-class head: dropout rate and use_flash / is_causal of every block restored: %v, MLM and NSP logits after loading max diff %.1e",
				bert_ok,
				d_bert,
			),
		)
	}

	// 4. Partial loading: matching layers load, others are skipped, also after
	// a skipped TransformerEncoder
	{
		src := nn.sequential_new()
		nn.sequential_add(
			src,
			nn.transformer_encoder_new(2, 4, 2, 6),
			nn.linear_layer_new(4, 8),
			nn.Activation.ReLU,
			nn.linear_layer_new(8, 3),
		)
		ps := nn.sequential_parameters(src)
		_pr_randomize(ps[:])
		dst := nn.sequential_new()
		nn.sequential_add(
			dst,
			nn.transformer_encoder_new(2, 4, 2, 10),
			nn.linear_layer_new(4, 8),
			nn.Activation.ReLU,
			nn.linear_layer_new(8, 5),
		)
		opt := nn.adam_new()
		f := file(&files, "partial")
		nn.save_checkpoint(src, &opt, f, 0)
		loaded, skipped, ok := nn.sequential_load_partial(dst, f)
		a := src.layers[1].(nn.LinearLayer)
		b := dst.layers[1].(nn.LinearLayer)
		same := _pr_max_diff(a.weights, b.weights) == 0 && _pr_max_diff(a.bias, b.bias) == 0
		expect(
			&passed,
			&total,
			ok && loaded == 2 && skipped == 2 && same,
			fmt.tprintf(
				"partial loading (encoder with another d_ff and a 5-class head skipped): loaded %d, skipped %d, Linear after the encoder restored exactly: %v",
				loaded,
				skipped,
				same,
			),
		)
		nn.adam_free(&opt)
	}

	// 5. Corrupt files fail without crashing
	{
		src := nn.sequential_new()
		nn.sequential_add(src, nn.linear_layer_new(6, 5), nn.lstm_layer_new(5, 4))
		opt := nn.adam_new()
		nn.sequential_add_to_adam(src, &opt)
		f, cut := file(&files, "corrupt"), file(&files, "cut")
		nn.save_checkpoint(src, &opt, f, 0)
		data, _ := os.read_entire_file(f, allocator)
		rejected := 0
		for n in ([]int{0, 7, 30, len(data) / 2, len(data) - 1}) {
			_ = os.write_entire_file(cut, data[:n])
			m, o, _, ok := nn.load_checkpoint(cut)
			if !ok && m == nil && o == nil {rejected += 1}
		}
		_, gpt_ok := nn.load_gpt_model(f)
		expect(
			&passed,
			&total,
			rejected == 5 && !gpt_ok,
			fmt.tprintf(
				"truncated checkpoints rejected %d/5, a Sequential checkpoint read as GPT rejected: %v",
				rejected,
				!gpt_ok,
			),
		)
		nn.adam_free(&opt)
	}

	// 6. Freezing covers every layer type
	{
		seq := nn.sequential_new()
		nn.sequential_add(
			seq,
			nn.linear_layer_new(3, 4),
			nn.gat_layer_new(4, 4, 2, nil),
			nn.mamba_layer_new(4, 3),
			nn.moe_layer_new(4, 6, 3, 2),
			nn.transformer_encoder_new(1, 4, 2, 6),
		)
		n_all := len(nn.sequential_parameters(seq, all = true))
		nn.sequential_freeze_range(seq, 1, 4)
		after_range := len(nn.sequential_parameters(seq))
		nn.sequential_freeze_all(seq)
		frozen := len(nn.sequential_parameters(seq))
		nn.sequential_unfreeze_all(seq)
		thawed := len(nn.sequential_parameters(seq))
		gpt := nn.gpt_model_new(13, 8, 2, 16, 3, 8, use_moe = true, num_experts = 2, top_k = 1)
		g_all := len(nn.gpt_model_parameters(&gpt))
		nn.gpt_freeze_blocks(&gpt, 0, 2)
		g_part := len(nn.gpt_model_parameters(&gpt))
		per_block := (g_all - 6) / 3 // without embeddings, final norm and output head
		ok :=
			frozen == 0 &&
			thawed == n_all &&
			after_range == 2 + 16 &&
			g_part == g_all - 2 * per_block
		expect(
			&passed,
			&total,
			ok,
			fmt.tprintf(
				"freezing: Sequential with GAT, Mamba, MoE, encoder (%d tensors): range 1..4 leaves %d trainable, all → %d, unfreeze → %d; GPT (MoE) blocks 0..2 frozen: %d of %d trainable",
				n_all,
				after_range,
				frozen,
				thawed,
				g_part,
				g_all,
			),
		)
	}

	fmt.printf("Persistence summary: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
