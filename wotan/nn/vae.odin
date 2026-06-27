package nn

import l "../linalg"
import t "../tensor"
import "core:math"
import "core:math/rand"
import "core:mem"

// ============================================================================
// VAE Encoder: x → (mu, log_var)
// ============================================================================

VAEEncoder :: struct {
	fc1:       LinearLayer, // input_dim → hidden_dim
	fc_mu:     LinearLayer, // hidden_dim → latent_dim
	fc_logvar: LinearLayer, // hidden_dim → latent_dim
}

vae_encoder_new :: proc(
	input_dim: int,
	hidden_dim: int,
	latent_dim: int,
	allocator: mem.Allocator = context.allocator,
) -> VAEEncoder {
	enc: VAEEncoder
	enc.fc1 = linear_layer_new(input_dim, hidden_dim, allocator)
	enc.fc_mu = linear_layer_new(hidden_dim, latent_dim, allocator)
	enc.fc_logvar = linear_layer_new(hidden_dim, latent_dim, allocator)
	return enc
}

vae_encoder_free :: proc(enc: ^VAEEncoder) {
	linear_layer_free(&enc.fc1)
	linear_layer_free(&enc.fc_mu)
	linear_layer_free(&enc.fc_logvar)
}

vae_encoder_forward :: proc(
	enc: ^VAEEncoder,
	x: ^t.Tensor,
) -> (
	mu: ^t.Tensor,
	log_var: ^t.Tensor,
) {
	h := linear_forward(&enc.fc1, x)
	h = t.tensor_relu(h)

	mu = linear_forward(&enc.fc_mu, h)
	log_var = linear_forward(&enc.fc_logvar, h)

	return mu, log_var
}

vae_encoder_add_to_optimizer :: proc(enc: ^VAEEncoder, opt: ^Adam) {
	adam_add_param(opt, enc.fc1.weights)
	adam_add_param(opt, enc.fc1.bias)
	adam_add_param(opt, enc.fc_mu.weights)
	adam_add_param(opt, enc.fc_mu.bias)
	adam_add_param(opt, enc.fc_logvar.weights)
	adam_add_param(opt, enc.fc_logvar.bias)
}

// ============================================================================
// VAE Decoder: z → x_reconstructed
// ============================================================================

VAEDecoder :: struct {
	fc1: LinearLayer, // latent_dim → hidden_dim
	fc2: LinearLayer, // hidden_dim → hidden_dim
	fc3: LinearLayer, // hidden_dim → input_dim
}

vae_decoder_new :: proc(
	latent_dim: int,
	hidden_dim: int,
	input_dim: int,
	allocator: mem.Allocator = context.allocator,
) -> VAEDecoder {
	dec: VAEDecoder
	dec.fc1 = linear_layer_new(latent_dim, hidden_dim, allocator)
	dec.fc2 = linear_layer_new(hidden_dim, hidden_dim, allocator)
	dec.fc3 = linear_layer_new(hidden_dim, input_dim, allocator)
	return dec
}

vae_decoder_free :: proc(dec: ^VAEDecoder) {
	linear_layer_free(&dec.fc1)
	linear_layer_free(&dec.fc2)
	linear_layer_free(&dec.fc3)
}

vae_decoder_forward :: proc(dec: ^VAEDecoder, z: ^t.Tensor) -> ^t.Tensor {
	h := linear_forward(&dec.fc1, z)
	h = t.tensor_relu(h)
	h = linear_forward(&dec.fc2, h)
	h = t.tensor_relu(h)
	x_recon := linear_forward(&dec.fc3, h)
	x_recon = t.tensor_tanh(x_recon) // Output in [-1, 1]
	return x_recon
}

vae_decoder_add_to_optimizer :: proc(dec: ^VAEDecoder, opt: ^Adam) {
	adam_add_param(opt, dec.fc1.weights)
	adam_add_param(opt, dec.fc1.bias)
	adam_add_param(opt, dec.fc2.weights)
	adam_add_param(opt, dec.fc2.bias)
	adam_add_param(opt, dec.fc3.weights)
	adam_add_param(opt, dec.fc3.bias)
}

// ============================================================================
// Reparameterization Trick: sample z from N(mu, sigma)
// ============================================================================

reparameterize :: proc(mu: ^t.Tensor, log_var: ^t.Tensor, allocator: mem.Allocator) -> ^t.Tensor {
	// z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
	// sigma = exp(0.5 * log_var)

	batch_size := mu.data.rows
	latent_dim := mu.data.cols

	z_data := l.matrix_new(f64, batch_size, latent_dim, allocator)

	for b in 0 ..< batch_size {
		for d in 0 ..< latent_dim {
			mu_val := mu.data.data[b * latent_dim + d]
			log_var_val := log_var.data.data[b * latent_dim + d]
			std := math.exp(0.5 * log_var_val)

			// Sample from standard normal
			epsilon := rand.norm_float64()

			z_data.data[b * latent_dim + d] = mu_val + std * epsilon
		}
	}

	z := t.tensor_new(z_data, true, allocator)
	return z
}
