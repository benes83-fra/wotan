
package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "core:fmt"
import "core:mem"
nn_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Testing Neural Network Linear Layer ===")

	// Create a Linear Layer: 2 inputs -> 3 outputs
	layer := nn.linear_layer_new(2, 3, allocator)

	// Create a batch of 2 samples, each with 2 features
	// X = [[1.0, 2.0],
	//      [3.0, 4.0]]
	x_data := l.matrix_new(f64, 2, 2, allocator)
	x_data.data[0] = 1.0; x_data.data[1] = 2.0
	x_data.data[2] = 3.0; x_data.data[3] = 4.0
	x := t.tensor_new(x_data, true, allocator)

	defer t.tensor_free(x) // User manages input lifecycle

	// 3. Forward Pass (Clean API!)
	out := nn.linear_forward(&layer, x)
	fmt.printf("Input shape:  %dx%d\n", x.data.rows, x.data.cols)
	fmt.printf("Output shape: %dx%d\n", out.data.rows, out.data.cols)

	fmt.println("\nOutput values (Y = X @ W + b):")
	for i in 0 ..< out.data.rows {
		for j in 0 ..< out.data.cols {
			fmt.printf("%.4f ", out.data.data[i * out.data.cols + j])
		}
		fmt.println("")
	}

	// Cleanup
	t.tensor_free_graph(out)
}
