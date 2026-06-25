// wotan/nn/gan.odin
package nn

import l "../linalg"
import t "../tensor"
import "core:mem"

// ============================================================================
// Generator: noise → fake data
// ============================================================================

Generator :: struct {
	fc1: LinearLayer, // noise_dim → hidden_dim
	fc2: LinearLayer, // hidden_dim → hidden_dim
	fc3: LinearLayer, // hidden_dim → data_dim
}

generator_new :: proc(
	noise_dim: int,
	hidden_dim: int,
	data_dim: int,
	allocator: mem.Allocator = context.allocator,
) -> Generator {
	g: Generator
	g.fc1 = linear_layer_new(noise_dim, hidden_dim, allocator)
	g.fc2 = linear_layer_new(hidden_dim, hidden_dim, allocator)
	g.fc3 = linear_layer_new(hidden_dim, data_dim, allocator)
	return g
}

generator_free :: proc(g: ^Generator) {
	linear_layer_free(&g.fc1)
	linear_layer_free(&g.fc2)
	linear_layer_free(&g.fc3)
}

generator_forward :: proc(g: ^Generator, noise: ^t.Tensor) -> ^t.Tensor {
	x := linear_forward(&g.fc1, noise)
	x = t.tensor_relu(x)
	x = linear_forward(&g.fc2, x)
	x = t.tensor_relu(x)
	x = linear_forward(&g.fc3, x)
	x = t.tensor_tanh(x) // Output in [-1, 1]
	return x
}

generator_add_to_optimizer :: proc(g: ^Generator, opt: ^Adam) {
	adam_add_param(opt, g.fc1.weights)
	adam_add_param(opt, g.fc1.bias)
	adam_add_param(opt, g.fc2.weights)
	adam_add_param(opt, g.fc2.bias)
	adam_add_param(opt, g.fc3.weights)
	adam_add_param(opt, g.fc3.bias)
}

// ============================================================================
// Discriminator: data → real/fake probability
// ============================================================================

Discriminator :: struct {
	fc1: LinearLayer, // data_dim → hidden_dim
	fc2: LinearLayer, // hidden_dim → hidden_dim
	fc3: LinearLayer, // hidden_dim → 1
}

discriminator_new :: proc(
	data_dim: int,
	hidden_dim: int,
	allocator: mem.Allocator = context.allocator,
) -> Discriminator {
	d: Discriminator
	d.fc1 = linear_layer_new(data_dim, hidden_dim, allocator)
	d.fc2 = linear_layer_new(hidden_dim, hidden_dim, allocator)
	d.fc3 = linear_layer_new(hidden_dim, 1, allocator)
	return d
}

discriminator_free :: proc(d: ^Discriminator) {
	linear_layer_free(&d.fc1)
	linear_layer_free(&d.fc2)
	linear_layer_free(&d.fc3)
}

discriminator_forward :: proc(d: ^Discriminator, data: ^t.Tensor) -> ^t.Tensor {
	x := linear_forward(&d.fc1, data)
	x = t.tensor_relu(x)
	x = linear_forward(&d.fc2, x)
	x = t.tensor_relu(x)
	x = linear_forward(&d.fc3, x)
	x = t.tensor_sigmoid(x) // Output probability [0, 1]
	return x
}

discriminator_add_to_optimizer :: proc(d: ^Discriminator, opt: ^Adam) {
	adam_add_param(opt, d.fc1.weights)
	adam_add_param(opt, d.fc1.bias)
	adam_add_param(opt, d.fc2.weights)
	adam_add_param(opt, d.fc2.bias)
	adam_add_param(opt, d.fc3.weights)
	adam_add_param(opt, d.fc3.bias)
}


// Add to nn.odin
discriminator_forward_no_sigmoid :: proc(d: ^Discriminator, data: ^t.Tensor) -> ^t.Tensor {
	x := linear_forward(&d.fc1, data)
	x = t.tensor_relu(x)
	x = linear_forward(&d.fc2, x)
	x = t.tensor_relu(x)
	x = linear_forward(&d.fc3, x)
	// No sigmoid - raw output for WGAN critic
	return x
}
