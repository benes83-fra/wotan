package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:slice"

// No-grad (inference) scopes: graph_begin(&scope, no_grad = true).
// For every model family the forward pass is run once in a normal scope and
// once in a no-grad scope, on the same weights and inputs:
// - the outputs are bit-identical
// - the no-grad outputs require no gradient and record no inputs, so backward
//   through them is a no-op that leaves the parameter gradients at zero
// - nothing outlives graph_end (the models live on a heap tracking allocator)
// - the arena holds less than with graph recording
// and a normal scope inside a no-grad scope records a graph again.

// One forward pass: its outputs, the root tensors of the pass and the parameters
@(private = "file")
_Ng_Case :: struct {
	name:    string,
	forward: proc(c: ^_Ng_Case) -> []^t.Tensor,
	model:   rawptr,
	inputs:  []^t.Tensor,
	params:  []^t.Tensor,
}

@(private = "file")
_ng_mask: []f64 // the causal mask of the GPT cases

@(private = "file")
// Random input; stored flat ([1, n]) like sequence data, or as a [rows, cols] matrix
_ng_input :: proc(n: int, shape: [4]int, ids_below := 0, as_matrix := false) -> ^t.Tensor {
	m := l.matrix_new(f64, shape[0], shape[1]) if as_matrix else l.matrix_new(f64, 1, n)
	for &v in m.data {v = f64(rand.int_max(ids_below)) if ids_below > 0 else rand.float64_range(-1, 1)}
	x := t.tensor_new(m, false)
	x.shape = shape
	return x
}

no_grad_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== No-grad (inference) scopes ===")
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// Models and inputs on the heap, so anything left behind is visible
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, runtime.heap_allocator())
	defer mem.tracking_allocator_destroy(&track)
	context.allocator = mem.tracking_allocator(&track)

	mask := nn.create_causal_mask(5, context.allocator)
	defer delete(mask)

	gpt := nn.gpt_model_new(11, 8, 2, 16, 2, 8)
	gpt_moe := nn.gpt_model_new(11, 8, 2, 16, 1, 8, use_moe = true, num_experts = 4, top_k = 2)
	bert := nn.bert_model_new(11, 8, 2, 16, 2, 8)
	mamba := nn.mamba_layer_new(4, 3)
	mlp := nn.sequential_new()
	nn.sequential_add(
		mlp,
		nn.linear_layer_new(3, 8),
		nn.Activation.ReLU,
		nn.dropout_layer_new(0.3),
		nn.linear_layer_new(8, 2),
	)
	mlp.training = false
	rnn := nn.sequential_new()
	nn.sequential_add(rnn, nn.lstm_layer_new(3, 4), nn.gru_layer_new(4, 4))
	rnn.training = false
	enc := nn.vae_encoder_new(4, 6, 3)
	dec := nn.vae_decoder_new(3, 6, 4)
	_VAE :: struct {
		enc: ^nn.VAEEncoder,
		dec: ^nn.VAEDecoder,
	}
	vae := _VAE{&enc, &dec}
	defer {
		nn.gpt_model_free(&gpt)
		nn.gpt_model_free(&gpt_moe)
		nn.bert_model_free(&bert)
		nn.mamba_layer_free(&mamba)
		nn.sequential_free(mlp)
		nn.sequential_free(rnn)
		nn.vae_encoder_free(&enc)
		nn.vae_decoder_free(&dec)
	}
	_ng_mask = mask

	gpt_ps, moe_ps, bert_ps :=
		nn.gpt_model_parameters(&gpt),
		nn.gpt_model_parameters(&gpt_moe),
		nn.bert_model_parameters(&bert)
	mamba_ps, mlp_ps, rnn_ps, vae_ps: [dynamic]^t.Tensor
	mlp_ps = nn.sequential_parameters(mlp)
	rnn_ps = nn.sequential_parameters(rnn)
	for lin in ([]^nn.LinearLayer{&enc.fc1, &enc.fc_mu, &enc.fc_logvar, &dec.fc1, &dec.fc2, &dec.fc3}) {
		append(&vae_ps, lin.weights, lin.bias)
	}
	append(&mamba_ps, mamba.A, mamba.D)
	defer {
		delete(gpt_ps)
		delete(moe_ps)
		delete(bert_ps)
		delete(mamba_ps)
		delete(mlp_ps)
		delete(rnn_ps)
		delete(vae_ps)
	}

	ids := _ng_input(10, {2, 5, 1, 1}, 11)
	segs := _ng_input(12, {2, 6, 1, 1}, 2)
	bert_ids := _ng_input(12, {2, 6, 1, 1}, 11)
	x_mamba := _ng_input(48, {2, 6, 4, 1})
	h0_mamba := t.tensor_new(l.matrix_new(f64, 2, 12), false)
	h0_mamba.shape = {2, 4, 3, 1}
	x_mlp := _ng_input(15, {5, 3, 1, 1}, as_matrix = true)
	x_rnn := _ng_input(24, {2, 4, 3, 1})
	x_vae := _ng_input(20, {5, 4, 1, 1}, as_matrix = true)
	defer for x in ([]^t.Tensor{ids, segs, bert_ids, x_mamba, h0_mamba, x_mlp, x_rnn, x_vae}) {t.tensor_free(x)}

	cases := [?]_Ng_Case {
		{"GPT (2 blocks)", proc(c: ^_Ng_Case) -> []^t.Tensor {
				return slice.clone(
					[]^t.Tensor {
						nn.gpt_model_forward(
							(^nn.GPTModel)(c.model),
							c.inputs[0],
							_ng_mask,
							false,
						),
					},
					context.temp_allocator,
				)
			}, &gpt, {ids}, gpt_ps[:]},
		{"GPT with top-2-of-4 MoE", proc(c: ^_Ng_Case) -> []^t.Tensor {
				return slice.clone(
					[]^t.Tensor {
						nn.gpt_model_forward(
							(^nn.GPTModel)(c.model),
							c.inputs[0],
							_ng_mask,
							false,
						),
					},
					context.temp_allocator,
				)
			}, &gpt_moe, {ids}, moe_ps[:]},
		{"BERT (MLM + NSP)", proc(c: ^_Ng_Case) -> []^t.Tensor {
				mlm, nsp := nn.bert_model_forward(
					(^nn.BERTModel)(c.model),
					c.inputs[0],
					c.inputs[1],
					false,
				)
				return slice.clone([]^t.Tensor{mlm, nsp}, context.temp_allocator)
			}, &bert, {bert_ids, segs}, bert_ps[:]},
		{"Mamba layer", proc(c: ^_Ng_Case) -> []^t.Tensor {
				return slice.clone(
					[]^t.Tensor {
						nn.mamba_layer_forward(
							(^nn.MambaLayer)(c.model),
							c.inputs[0],
							c.inputs[1],
						),
					},
					context.temp_allocator,
				)
			}, &mamba, {x_mamba, h0_mamba}, mamba_ps[:]},
		{"Sequential MLP with dropout (eval)", proc(c: ^_Ng_Case) -> []^t.Tensor {
				return slice.clone(
					[]^t.Tensor{nn.sequential_forward((^nn.Sequential)(c.model), c.inputs[0])},
					context.temp_allocator,
				)
			}, mlp, {x_mlp}, mlp_ps[:]},
		{"Sequential LSTM → GRU", proc(c: ^_Ng_Case) -> []^t.Tensor {
				return slice.clone(
					[]^t.Tensor{nn.sequential_forward((^nn.Sequential)(c.model), c.inputs[0])},
					context.temp_allocator,
				)
			}, rnn, {x_rnn}, rnn_ps[:]},
		{
			"VAE encoder → reparameterize → decoder",
			proc(c: ^_Ng_Case) -> []^t.Tensor {
				m := (^_VAE)(c.model)
				rand.reset(7) // the same ε in both runs
				mu, log_var := nn.vae_encoder_forward(m.enc, c.inputs[0])
				z := nn.reparameterize(mu, log_var, context.allocator)
				return slice.clone(
					[]^t.Tensor{nn.vae_decoder_forward(m.dec, z), mu, log_var},
					context.temp_allocator,
				)
			},
			&vae,
			{x_vae},
			vae_ps[:],
		},
	}

	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	for &c in cases {
		// With graph recording
		t.graph_begin(&scope)
		outs := c.forward(&c)
		ref := make([][]f64, len(outs), context.temp_allocator)
		for o, i in outs {ref[i] = slice.clone(o.data.data, context.temp_allocator)}
		used_grad, _ := t.graph_scope_bytes(&scope)
		recorded := outs[0].requires_grad && len(outs[0].inputs) > 0
		t.graph_end(&scope)

		// Without
		live := len(track.allocation_map)
		t.graph_begin(&scope, no_grad = true)
		outs = c.forward(&c)
		same, no_graph := true, true
		for o, i in outs {
			same &&= slice.equal(transmute([]u64)o.data.data, transmute([]u64)ref[i])
			no_graph &&= !o.requires_grad && len(o.inputs) == 0
		}
		for o in outs {t.tensor_backward(o)}
		grads_zero := true
		for p in c.params {
			if p.grad.data != nil {for g in p.grad.data {grads_zero &&= g == 0}}
		}
		used_nograd, _ := t.graph_scope_bytes(&scope)
		t.graph_end(&scope)
		left := len(track.allocation_map) - live

		expect(
			&passed,
			&total,
			recorded && same && no_graph && grads_zero && left == 0 && used_nograd < used_grad,
			fmt.tprintf(
				"%s: outputs bit-identical %v, no graph %v (recorded with grad %v), backward a no-op %v, left after graph_end %d, arena %d → %d bytes (%.0f%%)",
				c.name,
				same,
				no_graph,
				recorded,
				grads_zero,
				left,
				used_grad,
				used_nograd,
				100 * f64(used_nograd) / f64(used_grad),
			),
		)
	}

	// A normal scope inside a no-grad scope records again; leaving it restores no-grad
	{
		inner: t.Graph_Scope
		defer t.graph_scope_destroy(&inner)
		t.graph_begin(&scope, no_grad = true)
		a := nn.sequential_forward(mlp, x_mlp)
		t.graph_begin(&inner)
		b := nn.sequential_forward(mlp, x_mlp)
		inner_on := t.grad_enabled()
		t.graph_end(&inner)
		after := t.grad_enabled()
		t.graph_end(&scope)
		expect(
			&passed,
			&total,
			!a.requires_grad && b.requires_grad && inner_on && !after && t.grad_enabled(),
			"a normal scope inside a no-grad scope records a graph; graph_end returns to no-grad, and to recording after the outer scope",
		)
	}

	fmt.printf("No-grad scopes: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
