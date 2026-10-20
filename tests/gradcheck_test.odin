package gradcheck

import "core:fmt"
import "core:math"
import "core:mem"
import l "wotan:linalg"
import t "wotan:tensor"

mk :: proc(vals: []f64, rows, cols: int, rg: bool) -> ^t.Tensor {
	m := l.matrix_new(f64, rows, cols)
	copy(m.data, vals)
	return t.tensor_new(m, rg)
}

// Compare analytic grad of x vs central finite difference of f(x) (scalar loss)
check :: proc(name: string, x: ^t.Tensor, f: proc(x: ^t.Tensor) -> ^t.Tensor) {
	t.tensor_zero_grad(x)
	loss := f(x)
	t.tensor_backward(loss)
	analytic := make([]f64, len(x.grad.data))
	copy(analytic, x.grad.data)
	t.tensor_free_graph(loss)
	h := 1e-6
	max_err := 0.0
	for i in 0 ..< len(x.data.data) {
		orig := x.data.data[i]
		x.data.data[i] = orig + h
		lp := f(x); fp := lp.data.data[0]; t.tensor_free_graph(lp)
		x.data.data[i] = orig - h
		lm := f(x); fm := lm.data.data[0]; t.tensor_free_graph(lm)
		x.data.data[i] = orig
		num := (fp - fm) / (2 * h)
		err := abs(num - analytic[i]) / max(1.0, abs(num))
		if err > max_err {max_err = err}
		if i < 3 {fmt.printf("    [%d] analytic=% .6f numeric=% .6f\n", i, analytic[i], num)}
	}
	status := "OK  " if max_err < 1e-4 else "FAIL"
	fmt.printf("%s %-40s max_rel_err=%.2e\n\n", status, name, max_err)
}

W1, W2: ^t.Tensor

gradcheck_test :: proc(allocator: mem.Allocator) {
	W1 = mk([]f64{0.1, -0.2, 0.3, 0.4, 0.5, -0.6}, 2, 3, false)
	W2 = mk([]f64{0.7, 0.1, -0.3, 0.2, 0.9, -0.5}, 2, 3, false)

	// 1. Flattened [1, N*D] input feeding TWO matmuls (like Q/K/V projections)
	x := mk([]f64{1, 2, 3, 4}, 1, 4, true) // N=2, D=2
	check("matmul flattened, fan-out (QKV pattern)", x, proc(x: ^t.Tensor) -> ^t.Tensor {
		a := t.tensor_matmul(x, W1)
		b := t.tensor_matmul(x, W2)
		return t.tensor_sum(t.tensor_add(a, b))
	})
	// 1b. flattened + residual
	check("matmul flattened + residual add", x, proc(x: ^t.Tensor) -> ^t.Tensor {
		W := mk([]f64{0.1, -0.2, 0.3, 0.4}, 2, 2, false)
		a := t.tensor_matmul(x, W)
		return t.tensor_sum(t.tensor_mul(t.tensor_add(a, x), t.tensor_add(a, x)))
	})

	// 2. KL divergence wrt log_var
	lv := mk([]f64{0.5, -0.3, 1.2}, 1, 3, true)
	check("kl_divergence d/dlog_var", lv, proc(lv: ^t.Tensor) -> ^t.Tensor {
		mu := mk([]f64{0.1, 0.2, 0.3}, 1, 3, false)
		mu.op = .Constant; mu.owned_by_graph = true
		return t.tensor_kl_divergence(mu, lv)
	})

	// 3. Div with constant numerator
	d := mk([]f64{1.5, 2.0, 3.0}, 1, 3, true)
	check("div(const, x)", d, proc(x: ^t.Tensor) -> ^t.Tensor {
		one := mk([]f64{1, 1, 1}, 1, 3, false)
		one.owned_by_graph = true
		return t.tensor_sum(t.tensor_div(one, x))
	})

	// 4. Scale with small scalar
	s := mk([]f64{1, 2, 3}, 1, 3, true)
	check(
		"scale(x, 3.3e-7)",
		s,
		proc(x: ^t.Tensor) -> ^t.Tensor {
			return t.tensor_sum(t.tensor_scale(x, 3.3e-7 * 1e6)) // control: large ok
		},
	)
	check("scale(x, 1/(252*10000))", s, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_scale(t.tensor_sum(x), 1.0 / (252.0 * 10000.0))
	})

	// 5. LeakyReLU alpha != 0.01
	lr := mk([]f64{-1, 2, -3}, 1, 3, true)
	check("leaky_relu alpha=0.2", lr, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sum(t.tensor_leaky_relu(x, 0.2))
	})

	// 6. Sharpe loss with realistic daily returns
	r := mk([]f64{0.0012, -0.0007, 0.00031, 0.0009, -0.00045}, 1, 5, true)
	check("sharpe_loss daily returns", r, proc(x: ^t.Tensor) -> ^t.Tensor {
		return t.tensor_sharpe_loss(x, 0.0000159)
	})

	// 8. Flatten with fan-out (flatten output + original 4D input both used)
	f := mk([]f64{1, 2, 3, 4}, 1, 4, true)
	f.shape = [4]int{1, 1, 2, 2}
	check("flatten fan-out", f, proc(x: ^t.Tensor) -> ^t.Tensor {
		fl := t.tensor_flatten(x)
		y := t.tensor_mul(fl, fl)
		return t.tensor_add(t.tensor_sum(y), t.tensor_sum(t.tensor_mul(x, x)))
	})

	// 7. Softmax on [2,3] batch: rows should each sum to 1
	lg := mk([]f64{1, 2, 3, 1, 1, 1}, 2, 3, false)
	p := t.tensor_softmax(lg)
	fmt.printf(
		"softmax [2,3] row sums: %.4f %.4f  (expected 1, 1)\n",
		p.data.data[0] + p.data.data[1] + p.data.data[2],
		p.data.data[3] + p.data.data[4] + p.data.data[5],
	)
}
