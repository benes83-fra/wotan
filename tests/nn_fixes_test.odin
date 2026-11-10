package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:mem"
import "core:os"

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
