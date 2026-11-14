package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:slice"

// Regression tests for wotan/nn issues found while writing the nn guide:
// constant op inputs made by forward passes (position ids, zero RNN / Mamba
// states, the [CLS] gather ids, the positional encoding, the MoE top-k mask)
// were allocated on the heap even inside a graph scope, so graph_end left one
// tensor per constant per step behind; MoE without aux_loss leaked its router
// softmax, and as the first layer of a model (input without gradient) its
// token matrix; the probabilistic losses leaked the intermediates computed
// from data inputs (a fixed variance, scale or ν); a checkpoint whose layer
// header had an invalid configuration (e.g. 0 attention heads) stopped
// reading and could load as ok. Found with the tokenizers' part of the guide:
// the WordPiece cleaner kept private-use and unassigned characters (as [UNK])
// that Hugging Face's BertTokenizer removes with the other C* categories.
// Later: bpe_tokenize put [CLS] / [SEP] ids (0 = "!") into GPT-2 input; GPT
// with use_flash ignored an explicit attention mask; adam / sgd_zero_grad
// zeroed the gradients instead of dropping them, so parameters without a
// gradient still moved (momentum, weight decay); checkpoints lost the frozen
// flags (and with them the Adam moments); the Mamba layer had no input
// normalisation (outputs exploded on large inputs).
// Every check runs under a Tracking_Allocator on the default heap allocator.

@(private = "file")
nf_ids :: proc(vals: []f64, batch, seq: int) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, batch * seq)
	copy(m.data, vals)
	x := t.tensor_new(m, false)
	x.shape = {batch, seq, 1, 1}
	return x
}

@(private = "file")
nf_seq :: proc(batch, seq, d: int) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, batch * seq * d)
	for &v, i in m.data {v = 0.1 * f64(i % 7) - 0.3}
	x := t.tensor_new(m, false)
	x.shape = {batch, seq, d, 1}
	return x
}

@(private = "file")
NF_TARGETS :: []int{4, 2, 7, 0, 0, 1, 5, 9, 2, 1, 6, 3}

@(private = "file")
nf_gpt: nn.GPTModel
@(private = "file")
nf_bert: nn.BERTModel
@(private = "file")
nf_model: ^nn.Sequential
@(private = "file")
nf_ids_t, nf_seg_t, nf_x: ^t.Tensor

// Live heap allocations after each of 4 training steps (scoped: inside a
// graph scope, else freed with tensor_free_graph); all equal = no leak
@(private = "file")
nf_steps :: proc(
	track: ^mem.Tracking_Allocator,
	scoped: bool,
	loss_of: proc() -> ^t.Tensor,
) -> [4]int {
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	counts: [4]int
	for i in 0 ..< 4 {
		if scoped {t.graph_begin(&scope)}
		loss := loss_of()
		t.tensor_backward(loss)
		if scoped {t.graph_end(&scope)} else {t.tensor_free_graph(loss)}
		counts[i] = len(track.allocation_map)
	}
	return counts
}

// Reference values (torch 2.1.0, see the Mamba and optimizer checks)
@(private = "file")
NF_MAMBA_OUT_1 :: [?]f64 {
	0.058479590481694654,
	1.2131926293919633,
	1.3306647354119105,
	0.5033410494374555,
	-1.0223847683855922,
	-1.1538665614081305,
	-0.48098887233075094,
	0.42589315682269346,
	0.53793765468160348,
	0.58050135238531708,
	0.30260814275638426,
	-0.048218360214594136,
	-0.77803426276933352,
	-0.46192351625995376,
	0.27410377527059049,
	1.0638606147510858,
	1.0551541760204872,
	0.43862124006560066,
	-0.60169624695115087,
	-1.1646832725515495,
	-1.1148183723436376,
	0.28447010950717899,
	1.4035980746007528,
	1.4423540305006179,
	-0.080423741227698131,
	-0.91150082614093264,
	-0.9734723794103558,
	-0.2732489166885399,
	0.12650434450836873,
	0.59409301733441777,
	0.58253914382156502,
	0.26233943421471789,
	-0.65302071562931396,
	-0.68095198483472252,
	-0.29093254482759906,
	0.49004556600114768,
	0.83999557684671822,
	1.0168035196417196,
	0.37084984696721146,
	-0.65385301353946157,
}
@(private = "file")
NF_MAMBA_DGAMMA_1 :: [?]f64 {
	0.10176530788438443,
	0.063847510902573956,
	-0.0055170008687449121,
	0.089746736651219211,
}
@(private = "file")
NF_MAMBA_DTHETA_A_1 :: [?]f64 {
	-0.0010988518826532943,
	-0.0035420160855249051,
	-0.0044965139966910992,
	0.0011782591184675069,
	0.0015141376614623758,
	0.00047044333589200537,
	-0.0051749730856642728,
	-0.0016009843119260873,
	0.0030872432432427774,
	-0.0040056093046908796,
	-0.0037552050251448833,
	-0.00071373593820793231,
}
@(private = "file")
NF_MAMBA_OUT_10 :: [?]f64 {
	2.7405158526930591,
	13.065372114516812,
	12.891481534611831,
	3.3258237808022328,
	-7.5268071138625645,
	-10.471468486481983,
	-5.5169751628747363,
	1.873262393996141,
	5.6441647762005935,
	5.02973051328373,
	1.315658095503959,
	-2.8962494816605338,
	-5.7425732274782701,
	-4.0058918643025736,
	1.8469418109837652,
	8.3580826570880333,
	9.460603219879717,
	2.6480154130874292,
	-8.1671610963067902,
	-13.91801369443848,
	-9.1716544809743006,
	3.6300746191445343,
	13.523901929561463,
	12.683316613576952,
	2.1495600153209664,
	-7.7855710199387556,
	-10.209579832721365,
	-4.9473689484150469,
	1.8861870918234451,
	5.772885989519712,
	4.8909034157767053,
	1.0489410510493764,
	-3.6851016198326119,
	-5.6720687716046152,
	-3.6334774984641851,
	2.4127110657346127,
	8.3654455255735805,
	9.2656855976025785,
	2.0604383000227102,
	-8.7092436633165118,
}
@(private = "file")
NF_MAMBA_DGAMMA_10 :: [?]f64 {
	0.8033363737897502,
	0.49628357344345131,
	-0.061080010938055818,
	0.6848195691385276,
}
@(private = "file")
NF_MAMBA_DTHETA_A_10 :: [?]f64 {
	-0.011938790296169996,
	-0.030976931633118297,
	-0.036860713845250012,
	0.018513459966433311,
	0.01597206780997534,
	-0.0022128593378983722,
	-0.029522337975121934,
	-0.0011678858630216268,
	0.028226203852116584,
	-0.054066287019770981,
	-0.044491077236726961,
	-0.0051236932602707532,
}
@(private = "file")
NF_OPT_REF :: [3][5]f64 {
	{
		0.11291540668959373,
		-0.60393906162792688,
		1.6015049004399815,
		1.3002390661235703,
		-0.50069442073052517,
	},
	{
		0.10021054778864248,
		-0.57104759816241801,
		1.529032716375093,
		1.2714370012167395,
		-0.48784063905080438,
	},
	{
		0.2100404701427469,
		0.0044957115663580849,
		1.3461376805709877,
		1.3495874999999999,
		-0.45718749999999997,
	},
}

@(private = "file")
NF_TOK_TEXTS :: [3]string {
	"Hello world!",
	"The Fed raised rates by 25bp.",
	"Stocks rallied; bonds fell.",
}
// Hugging Face transformers 4.24, add_special_tokens = True: GPT2Tokenizer
// (adds no special tokens) and BertTokenizer (bert-base-uncased, max_length 12)
@(private = "file")
NF_TOK_GPT2 :: [3][]int {
	{15496, 995, 0},
	{464, 10169, 4376, 3965, 416, 1679, 46583, 13},
	{1273, 3320, 32537, 26, 13100, 3214, 13},
}
@(private = "file")
NF_TOK_BERT :: [3][]int {
	{101, 7592, 2088, 999, 102, 0, 0, 0, 0, 0, 0, 0},
	{101, 1996, 7349, 2992, 6165, 2011, 2423, 2497, 2361, 1012, 102, 0},
	{101, 15768, 24356, 1025, 9547, 3062, 1012, 102, 0, 0, 0, 0},
}

// Deterministic parameter values (also in the torch reference): element j of
// parameter k is 0.5·sin(1 + 0.37 j + 1.1 k)
@(private = "file")
nf_fill_params :: proc(params: []^t.Tensor) {
	for p, k in params {
		for &v, j in p.data.data {v = 0.5 * math.sin(1.0 + 0.37 * f64(j) + 1.1 * f64(k))}
	}
}

// Mamba input [2, 5, 4]: scale · sin(0.9 i + 0.2) · (1 + 0.5 cos(0.3 i))
@(private = "file")
nf_mamba_input :: proc(scale: f64) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, 40)
	for &v, i in m.data {
		v = scale * math.sin(0.9 * f64(i) + 0.2) * (1 + 0.5 * math.cos(0.3 * f64(i)))
	}
	x := t.tensor_new(m, false)
	x.shape = {2, 5, 4, 1}
	return x
}

@(private = "file")
nf_max_diff :: proc(a, b: []f64) -> f64 {
	if len(a) != len(b) {return math.INF_F64}
	d := 0.0
	for v, i in a {d = max(d, abs(v - b[i]))}
	return d
}

// Little-endian i32 at data[at:]
@(private = "file")
nf_i32 :: proc(data: []u8, at: int) -> int {
	x := u32(data[at]) | u32(data[at + 1]) << 8 | u32(data[at + 2]) << 16 | u32(data[at + 3]) << 24
	return int(i32(x))
}

@(private = "file")
nf_put_i32 :: proc(buf: ^[dynamic]u8, v: int) {
	x := u32(i32(v))
	append(buf, u8(x), u8(x >> 8), u8(x >> 16), u8(x >> 24))
}

// Bytes of save_checkpoint's optimizer block after the frozen flags
// (timestep, hyperparameters, moment count, moments, steps)
@(private = "file")
nf_adam_tail :: proc(opt: ^nn.Adam) -> int {
	n := 8 + 5 * 8 + 4
	for m in opt.moment_1 {n += 2 * (4 + 8 * len(m))}
	return n + 8 * len(opt.parameters)
}

// The file as written before the frozen flags existed (optimizer type 2): the
// type and flags of a type-3 file with n_all parameters replaced; `cut` bytes
// before the type are left out as well (an older layer layout)
@(private = "file")
nf_as_type2 :: proc(data: []u8, opt: ^nn.Adam, n_all, cut: int) -> (old: [dynamic]u8, ok: bool) {
	flags := 4 + 4 * n_all
	type_at := len(data) - nf_adam_tail(opt) - flags - 4
	if type_at < cut || nf_i32(data, type_at) != 3 || nf_i32(data, type_at + 4) != n_all {return}
	append(&old, ..data[:type_at - cut])
	nf_put_i32(&old, 2)
	append(&old, ..data[type_at + 4 + flags:])
	return old, true
}

// One Adam step on mean(out²) (the frozen-model checkpoint check), in a graph
// scope: the frozen first layer's outputs are not part of the graph
@(private = "file")
nf_frozen_step :: proc(model: ^nn.Sequential, opt: ^nn.Adam, x: ^t.Tensor) {
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	nn.adam_zero_grad(opt)
	t.graph_begin(&scope)
	out := nn.sequential_forward(model, x)
	loss := t.tensor_mean(t.tensor_mul(out, out))
	t.tensor_backward(loss)
	t.graph_end(&scope)
	nn.adam_step(opt)
}

nn_fixes_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== nn regression tests ===")
	passed, total := 0, 0
	report :: proc(name: string, ok: bool, passed, total: ^int) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	flat :: proc(c: [4]int) -> bool {return c[0] == c[1] && c[1] == c[2] && c[2] == c[3]}

	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, runtime.default_allocator())
	defer mem.tracking_allocator_destroy(&track)
	context.allocator = mem.tracking_allocator(&track)

	nf_ids_t = nf_ids({1, 4, 2, 7, 0, 0, 3, 5, 9, 2, 1, 6}, 2, 6)
	nf_seg_t = nf_ids({0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1}, 2, 6)
	nf_x = nf_seq(2, 5, 4)

	{ 	// GPT and BERT training steps in a graph scope (position ids, [CLS] ids)
		nf_gpt = nn.gpt_model_new(11, 8, 2, 16, 2, 6)
		nf_bert = nn.bert_model_new(11, 8, 2, 16, 2, 6)
		g := nf_steps(&track, true, proc() -> ^t.Tensor {
			logits := nn.gpt_model_forward(&nf_gpt, nf_ids_t, nil, true)
			return t.tensor_cross_entropy_loss(nn.as_matrix(logits, 12, 11), NF_TARGETS)
		})
		b := nf_steps(&track, true, proc() -> ^t.Tensor {
			mlm, nsp := nn.bert_model_forward(&nf_bert, nf_ids_t, nf_seg_t, true)
			return t.tensor_add(
				t.tensor_cross_entropy_loss(nn.as_matrix(mlm, 12, 11), NF_TARGETS),
				t.tensor_cross_entropy_loss(nsp, {0, 1}),
			)
		})
		report(
			fmt.tprintf("GPT / BERT steps in a graph scope: no heap growth %v %v", g, b),
			flat(g) && flat(b),
			&passed,
			&total,
		)
		nn.gpt_model_free(&nf_gpt)
		nn.bert_model_free(&nf_bert)
	}
	{ 	// Sequential LSTM / GRU / Mamba zero states in a graph scope
		nf_model = nn.sequential_new()
		nn.sequential_add(
			nf_model,
			nn.lstm_layer_new(4, 3),
			nn.gru_layer_new(3, 3),
			nn.linear_layer_new(3, 4),
			nn.mamba_layer_new(4, 3),
		)
		c := nf_steps(&track, true, proc() -> ^t.Tensor {
				out := nn.sequential_forward(nf_model, nf_x)
				return t.tensor_mean(t.tensor_mul(out, out))
			})
		report(
			fmt.tprintf("Sequential LSTM, GRU, Mamba in a graph scope: no heap growth %v", c),
			flat(c),
			&passed,
			&total,
		)
		nn.sequential_free(nf_model)
	}
	{ 	// MoE first layer, no aux loss, tensor_free_graph (no scope)
		nf_model = nn.sequential_new()
		nn.sequential_add(nf_model, nn.moe_layer_new(4, 6, 3, 2))
		c := nf_steps(&track, false, proc() -> ^t.Tensor {
			out := nn.sequential_forward(nf_model, nf_x)
			return t.tensor_mean(t.tensor_mul(out, out))
		})
		report(
			fmt.tprintf("MoE without aux loss, tensor_free_graph: no heap growth %v", c),
			flat(c),
			&passed,
			&total,
		)
		nn.sequential_free(nf_model)
	}
	{ 	// losses with data inputs: fixed variance, fixed scale and ν, data predictions
		m := l.matrix_new(f64, 4, 1) // mu is learned, the rest is data
		copy(m.data, []f64{0.1, -0.2, 0.3, 0.0})
		mu := t.tensor_new(m, true)
		nf_const := [3]^t.Tensor{}
		for &c, k in nf_const {
			cm := l.matrix_new(f64, 4, 1)
			for &v in cm.data {v = 1.5 + f64(k)}
			c = t.tensor_new(cm, false)
		}
		counts: [4]int
		for i in 0 ..< 4 {
			g := nn.gaussian_nll_loss(mu, nf_const[0], nf_const[1])
			t.tensor_backward(g)
			t.tensor_free_graph(g)
			s := nn.student_t_nll_loss(mu, nf_const[0], nf_const[2], nf_const[1])
			t.tensor_backward(s)
			t.tensor_free_graph(s)
			q := nn.quantile_loss(nf_const[0], {0.5, 1.0, 2.0, 0.0}, {0.5}) // no gradient at all
			t.tensor_free(q)
			counts[i] = len(track.allocation_map)
		}
		report(
			fmt.tprintf("losses with data var / sigma / nu / pred: no heap growth %v", counts),
			flat(counts),
			&passed,
			&total,
		)
		t.tensor_free(mu)
		for c in nf_const {t.tensor_free(c)}
	}
	{ 	// a layer header with 0 attention heads is a corrupt file
		path := "model_checkpoints/nn_fixes_heads0.ckpt"
		os.make_directory("model_checkpoints")
		s := nn.sequential_new()
		nn.sequential_add(s, nn.multi_head_attention_layer_new(4, 2), nn.linear_layer_new(4, 2))
		opt := nn.adam_new()
		nn.sequential_add_to_opt(s, &opt)
		saved := nn.save_checkpoint(s, &opt, path, 1)
		data, _ := os.read_entire_file(path, context.allocator)
		// magic (10 bytes), epoch, layer count, layer id, d_model, then num_heads
		heads_at := len(nn.CHECKPOINT_MAGIC) + 4 * 4
		good := saved && data[heads_at] == 2
		for i in 0 ..< 4 {data[heads_at + i] = 0}
		_ = os.write_entire_file(path, data)
		before := len(track.allocation_map)
		m, o, _, ok := nn.load_checkpoint(path)
		good = good && !ok && m == nil && o == nil && len(track.allocation_map) == before
		_, _, pok := nn.sequential_load_partial(s, path)
		good = good && !pok
		report(
			"checkpoint with 0 heads: load_checkpoint and load_partial fail, no leak",
			good,
			&passed,
			&total,
		)
		delete(data)
		_ = os.remove(path)
		nn.adam_free(&opt)
		nn.sequential_free(s)
	}

	{ 	// WordPiece drops private-use and unassigned characters, as BertTokenizer
		path := "model_checkpoints/nn_fixes_vocab.txt"
		_ = os.write_entire_file(
			path,
			transmute([]u8)string("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nprivate\nuse\n"),
		)
		wp, ok := tok.wordpiece_tokenizer_new(path, 8)
		ids := tok.wordpiece_encode(&wp, "private  use ͸") if ok else nil
		good := ok && len(ids) == 2 && ids[0] == 5 && ids[1] == 6 // was 5, [UNK], 6, [UNK]
		report(
			fmt.tprintf("WordPiece: private-use and unassigned characters dropped %v", ids),
			good,
			&passed,
			&total,
		)
		delete(ids)
		if ok {tok.wordpiece_tokenizer_free(&wp)}
		_ = os.remove(path)
	}

	{ 	// bpe_tokenize adds [CLS] / [SEP] only where the vocabulary has them
		vocab, merges :=
			"model_checkpoints/nn_fixes_bpe.json", "model_checkpoints/nn_fixes_bpe.txt"
		_ = os.write_entire_file(
			vocab,
			transmute([]u8)string(`{"!": 0, "a": 1, "b": 2, "ab": 3, "<|endoftext|>": 4}`),
		)
		_ = os.write_entire_file(merges, transmute([]u8)string("#version: 0.2\na b\n"))
		bt, ok := tok.bpe_tokenizer_new(vocab, merges, 6)
		good := ok && bt.cls_id == -1 && bt.sep_id == -1
		if ok {
			ids, mask := tok.bpe_tokenize(&bt, "aba") // default add_special_tokens = true
			good &&=
				slice.equal(ids, []int{3, 1, 0, 0, 0, 0}) &&
				slice.equal(mask, []int{1, 1, 0, 0, 0, 0})
			delete(ids)
			delete(mask)
			tok.bpe_tokenizer_free(&bt)
		}
		_ = os.remove(vocab)
		_ = os.remove(merges)
		report(
			"bpe_tokenize on a vocabulary without [CLS] / [SEP] adds no special tokens",
			good,
			&passed,
			&total,
		)

		// the real vocabularies vs Hugging Face (where downloaded, as tokenizer_reference_test)
		GPT2_VOCAB, GPT2_MERGES, BERT_VOCAB ::
			"model_checkpoints/vocab.json",
			"model_checkpoints/merges.txt",
			"model_checkpoints/bert_vocab.txt"
		if !os.exists(GPT2_VOCAB) || !os.exists(GPT2_MERGES) || !os.exists(BERT_VOCAB) {
			fmt.println(
				"SKIP GPT-2 / BERT vocabularies not downloaded (see tokenizer_reference_test)",
			)
		} else {
			gpt2, g_ok := tok.bpe_tokenizer_new(GPT2_VOCAB, GPT2_MERGES, 12)
			bert, b_ok := tok.wordpiece_tokenizer_new(BERT_VOCAB, 12)
			same := g_ok && b_ok
			ref_gpt2, ref_bert := NF_TOK_GPT2, NF_TOK_BERT
			for text, i in NF_TOK_TEXTS {
				if !same {break}
				ids, mask := tok.bpe_tokenize(&gpt2, text)
				want := ref_gpt2[i]
				n := 0
				for m in mask {n += m}
				same &&= n == len(want) && slice.equal(ids[:n], want)
				delete(ids)
				delete(mask)
				ids, mask = tok.tokenize(&bert, text)
				same &&= slice.equal(ids, ref_bert[i])
				delete(ids)
				delete(mask)
			}
			if g_ok {tok.bpe_tokenizer_free(&gpt2)}
			if b_ok {tok.wordpiece_tokenizer_free(&bert)}
			report(
				"default tokenization of 3 texts as Hugging Face: GPT-2 without special tokens, BERT [CLS] … [SEP]",
				same,
				&passed,
				&total,
			)
		}
	}

	{ 	// FlashAttention GPT honours an explicit mask (a causal one keeps the kernel)
		nf_gpt = nn.gpt_model_new(11, 8, 2, 16, 2, 6, use_flash = true)
		causal := nn.create_causal_mask(6, context.allocator)
		custom := nn.create_causal_mask(6, context.allocator)
		for i in 1 ..< 6 {custom[i * 6] = -1e9} 	// the first token hidden from the others
		logits :: proc(mask: []f64, flash: bool) -> []f64 {
			for &b in nf_gpt.blocks {b.mha.use_flash = flash}
			out := nn.gpt_model_forward(&nf_gpt, nf_ids_t, mask, false)
			vals := slice.clone(out.data.data)
			t.tensor_free_graph(out)
			return vals
		}
		flash_custom, exact_custom := logits(custom, true), logits(custom, false)
		flash_causal, exact_causal := logits(causal, true), logits(causal, false)
		flash_nil := logits(nil, true)
		d_custom := nf_max_diff(flash_custom, exact_custom)
		d_effect := nf_max_diff(flash_custom, flash_causal)
		d_causal := nf_max_diff(flash_causal, exact_causal)
		d_nil := nf_max_diff(flash_causal, flash_nil)
		report(
			fmt.tprintf(
				"GPT use_flash with a custom mask = exact path %.1e (the mask moves the logits by %.2f); causal mask: flash vs exact %.1e, vs no mask %.1e",
				d_custom,
				d_effect,
				d_causal,
				d_nil,
			),
			d_custom < 1e-12 && d_effect > 1e-3 && d_causal < 1e-12 && d_nil == 0,
			&passed,
			&total,
		)
		for v in ([][]f64{flash_custom, exact_custom, flash_causal, exact_causal, flash_nil, causal, custom}) {delete(v)}
		nn.gpt_model_free(&nf_gpt)
	}

	{ 	// zero_grad drops the gradients and a parameter without one is skipped:
		// torch 2.1 Adam(lr 0.1), AdamW(lr 0.1, wd 0.1), SGD(lr 0.1, momentum
		// 0.9, wd 0.05), the second parameter not in the loss of steps 1 and 2
		worst := 0.0
		dropped := true
		refs := NF_OPT_REF
		tensor :: proc(vals: []f64, rg: bool) -> ^t.Tensor {
			m := l.matrix_new(f64, 1, len(vals))
			copy(m.data, vals)
			return t.tensor_new(m, rg)
		}
		for kind in 0 ..< 3 {
			p1, p2 := tensor({0.5, -1.0, 2.0}, true), tensor({1.5, -0.7}, true)
			c1, c2 := tensor({1.0, 2.0, 0.5}, false), tensor({0.3, 1.2}, false)
			adam: nn.Adam
			sgd: nn.SGD
			if kind < 2 {
				adam = nn.adam_new(0.1, weight_decay = 0.1 if kind == 1 else 0)
				nn.adam_add_param(&adam, p1)
				nn.adam_add_param(&adam, p2)
			} else {
				sgd = nn.sgd_new(0.1, momentum = 0.9, weight_decay = 0.05)
				nn.sgd_add_param(&sgd, p1)
				nn.sgd_add_param(&sgd, p2)
			}
			buf1: [^]f64
			for k in 0 ..< 4 {
				if kind < 2 {nn.adam_zero_grad(&adam)} else {nn.sgd_zero_grad(&sgd)}
				dropped &&= !t.tensor_has_grad(p1) && !t.tensor_has_grad(p2)
				loss := t.tensor_mean(t.tensor_mul(t.tensor_mul(p1, p1), c1))
				if k != 1 && k != 2 {
					loss = t.tensor_add(
						loss,
						t.tensor_mean(t.tensor_mul(t.tensor_mul(p2, p2), c2)),
					)
				}
				t.tensor_backward(loss)
				dropped &&= t.tensor_has_grad(p1) && !t.tensor_has_grad(p2) == (k == 1 || k == 2)
				// the buffers are reused, not reallocated, from step to step
				if k ==
				   0 {buf1 = raw_data(p1.grad.data)} else {dropped &&= raw_data(p1.grad.data) == buf1}
				if kind < 2 {nn.adam_step(&adam)} else {nn.sgd_step(&sgd)}
				t.tensor_free_graph(loss)
			}
			got := [5]f64 {
				p1.data.data[0],
				p1.data.data[1],
				p1.data.data[2],
				p2.data.data[0],
				p2.data.data[1],
			}
			ref := refs[kind]
			worst = max(worst, nf_max_diff(got[:], ref[:]))
			if kind < 2 {nn.adam_free(&adam)} else {nn.sgd_free(&sgd)}
			for x in ([]^t.Tensor{p1, p2, c1, c2}) {t.tensor_free(x)}
		}
		report(
			fmt.tprintf(
				"adam / sgd_zero_grad drop the gradients %v; Adam, AdamW, SGD momentum skip a parameter without one, vs torch 2.1: %.1e",
				dropped,
				worst,
			),
			dropped && worst < 1e-14,
			&passed,
			&total,
		)
	}

	{ 	// checkpoints keep frozen parameters frozen, with the Adam state intact
		path := "model_checkpoints/nn_fixes_frozen.ckpt"
		model := nn.sequential_new()
		nn.sequential_add(
			model,
			nn.linear_layer_new(3, 4),
			nn.Activation.Tanh,
			nn.linear_layer_new(4, 2),
		)
		nn.sequential_freeze_range(model, 0, 1)
		opt := nn.adam_new(0.05, weight_decay = 0.01)
		nn.sequential_add_to_opt(model, &opt)
		xm := l.matrix_new(f64, 5, 3)
		for &v, i in xm.data {v = math.sin(0.7 * f64(i))}
		x := t.tensor_new(xm, false)
		for _ in 0 ..< 3 {nf_frozen_step(model, &opt, x)}
		saved := nn.save_checkpoint(model, &opt, path, 3)
		m2, o2, epoch, ok := nn.load_checkpoint(path)
		good := saved && ok && epoch == 3
		if ok {
			all1 := nn.sequential_parameters(model, all = true)
			all2 := nn.sequential_parameters(m2, all = true)
			good &&= len(all1) == len(all2) && len(o2.parameters) == len(opt.parameters)
			for p, i in all1 {good &&= good && p.requires_grad == all2[i].requires_grad}
			for i in 0 ..< len(opt.parameters) {
				if !good {break}
				good &&=
					slice.equal(opt.moment_1[i], o2.moment_1[i]) &&
					slice.equal(opt.moment_2[i], o2.moment_2[i]) &&
					opt.steps[i] == o2.steps[i]
			}
			good &&= opt.timestep == o2.timestep
			m2.training = model.training
			nf_frozen_step(model, &opt, x) // the next step: the same in both
			nf_frozen_step(m2, o2, x)
			for p, i in all1 {good &&= good && slice.equal(p.data.data, all2[i].data.data)}
			delete(all1)
			delete(all2)
			nn.sequential_free(m2)
			nn.adam_free(o2)
			free(o2)
		}
		// a checkpoint without the frozen flags (optimizer type 2) still loads, all trainable
		data, _ := os.read_entire_file(path, context.allocator)
		old, old_ok := nf_as_type2(data, &opt, 4, 0)
		_ = os.write_entire_file(path, old[:])
		m3, o3, _, ok3 := nn.load_checkpoint(path)
		old_ok &&= ok3
		if ok3 {
			all3 := nn.sequential_parameters(m3, all = true)
			for p in all3 {old_ok &&= p.requires_grad}
			delete(all3)
			nn.sequential_free(m3)
			nn.adam_free(o3)
			free(o3)
		}
		delete(old)
		delete(data)
		_ = os.remove(path)
		report(
			fmt.tprintf(
				"partly frozen model with Adam state: save → load keeps the frozen flags, moments and steps, identical next step %v; a file without the flags loads %v",
				good,
				old_ok,
			),
			good && old_ok,
			&passed,
			&total,
		)
		t.tensor_free(x)
		nn.adam_free(&opt)
		nn.sequential_free(model)
	}

	{ 	// Mamba: pre-norm residual block vs a torch reimplementation (the
		// state-spaces/mamba Block, RMSNorm + residual, around wotan's mixer)
		mamba := nn.mamba_layer_new(4, 3)
		ps := make([dynamic]^t.Tensor)
		layer := nn.Layer(mamba)
		nn.layer_parameters(&layer, &ps, all = true)
		nf_fill_params(ps[:])
		d_out, d_grad := 0.0, 0.0
		mixer: [2][]f64
		for scale, s in ([]f64{1, 10}) {
			x := nf_mamba_input(scale)
			h0 := t.tensor_new(l.matrix_new(f64, 1, 2 * 4 * 3), false)
			h0.shape = {2, 4, 3, 1}
			out := nn.mamba_layer_forward(&mamba, x, h0)
			for p in ps {t.tensor_zero_grad(p)}
			loss := t.tensor_mean(t.tensor_mul(out, out))
			t.tensor_backward(loss)
			ref_out := NF_MAMBA_OUT_1 if s == 0 else NF_MAMBA_OUT_10
			ref_dg := NF_MAMBA_DGAMMA_1 if s == 0 else NF_MAMBA_DGAMMA_10
			ref_da := NF_MAMBA_DTHETA_A_1 if s == 0 else NF_MAMBA_DTHETA_A_10
			d_out = max(d_out, nf_max_diff(out.data.data, ref_out[:]) / scale)
			d_grad = max(
				d_grad,
				nf_max_diff(mamba.norm.gamma.grad.data, ref_dg[:]),
				nf_max_diff(mamba.A.grad.data, ref_da[:]),
			)
			mixer[s] = make([]f64, 40)
			for v, i in out.data.data {mixer[s][i] = v - x.data.data[i]}
			t.tensor_free_graph(loss)
			t.tensor_free(x)
			t.tensor_free(h0)
		}
		d_scale := nf_max_diff(mixer[0], mixer[1])
		report(
			fmt.tprintf(
				"Mamba block (RMSNorm, residual) vs torch: output %.1e, grads of norm weight / θ_A %.1e; 10σ input: SSM part within %.1e of the 1σ one",
				d_out,
				d_grad,
				d_scale,
			),
			d_out < 1e-12 && d_grad < 1e-12 && d_scale < 1e-4,
			&passed,
			&total,
		)
		delete(mixer[0])
		delete(mixer[1])

		// checkpoints: the norm weight round-trips; a Mamba layer of an older
		// file (no norm weight) loads with the weight 1
		path := "model_checkpoints/nn_fixes_mamba.ckpt"
		s := nn.sequential_new()
		nn.sequential_add(s, mamba)
		opt := nn.adam_new()
		saved := nn.save_checkpoint(s, &opt, path, 0)
		m2, o2, _, ok := nn.load_checkpoint(path)
		good := saved && ok
		if ok {
			p2 := nn.sequential_parameters(m2, all = true)
			good &&= len(p2) == len(ps)
			for p, i in ps {good &&= good && slice.equal(p.data.data, p2[i].data.data)}
			delete(p2)
			nn.sequential_free(m2)
			nn.adam_free(o2)
			free(o2)
		}
		data, _ := os.read_entire_file(path, context.allocator)
		id_at := len(nn.CHECKPOINT_MAGIC) + 8
		good &&= nf_i32(data, id_at) == 34
		data[id_at] = 25 // the older layer id, without the norm weight (8 + 8·4 bytes)
		old, old_ok := nf_as_type2(data, &opt, len(ps), 8 + 8 * 4)
		_ = os.write_entire_file(path, old[:])
		m3, o3, _, ok3 := nn.load_checkpoint(path)
		old_ok &&= ok3
		if ok3 {
			p3 := nn.sequential_parameters(m3, all = true)
			old_ok &&= len(p3) == len(ps)
			if old_ok {
				for v in p3[0].data.data {old_ok &&= v == 1}
				for p, i in ps[1:] {old_ok &&= slice.equal(p.data.data, p3[i + 1].data.data)}
			}
			delete(p3)
			nn.sequential_free(m3)
			nn.adam_free(o3)
			free(o3)
		}
		delete(old)
		delete(data)
		_ = os.remove(path)
		report(
			fmt.tprintf(
				"Mamba checkpoint: norm weight saved and loaded %v; an older Mamba layer loads with norm weight 1 %v",
				good,
				old_ok,
			),
			good && old_ok,
			&passed,
			&total,
		)
		delete(ps)
		nn.adam_free(&opt)
		nn.sequential_free(s) // frees the Mamba layer
	}

	t.tensor_free(nf_ids_t)
	t.tensor_free(nf_seg_t)
	t.tensor_free(nf_x)
	report(
		fmt.tprintf("all models freed: %d live allocations", len(track.allocation_map)),
		len(track.allocation_map) == 0,
		&passed,
		&total,
	)

	fmt.printf("nn regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
