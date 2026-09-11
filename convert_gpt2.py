import torch
import struct
import os
from transformers import GPT2LMHeadModel

def write_i32(f, val):
    f.write(struct.pack('<i', int(val)))

def write_f64(f, val):
    f.write(struct.pack('<d', float(val)))

def write_tensor(f, tensor, transpose=False):
    if tensor is None:
        write_i32(f, 0)
        write_i32(f, 0)
        return
    tensor = tensor.cpu().detach()
    if transpose:
        tensor = tensor.t()
    
    if len(tensor.shape) == 1:
        rows = 1
        cols = tensor.shape[0]
        data = tensor.to(torch.float64).numpy().flatten()
    else:
        rows = tensor.shape[0]
        cols = tensor.shape[1]
        data = tensor.to(torch.float64).numpy().flatten()
        
    write_i32(f, rows)
    write_i32(f, cols)
    for val in data:
        write_f64(f, float(val))

def convert_distilgpt2(output_path):
    print("Loading distilgpt2...")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    state_dict = model.state_dict()
    config = model.config
    
    vocab_size = config.vocab_size
    d_model = config.n_embd
    num_heads = config.n_head
    d_ff = config.n_inner if config.n_inner is not None else 4 * d_model
    num_layers = config.n_layer
    max_seq_len = config.n_positions

    print(f"✅ CONFIRMED: vocab_size = {vocab_size} (Should be 50257)")
    print(f"Writing checkpoint to {output_path}...")
    
    with open(output_path, "wb") as f:
        f.write(b"WOTAN_CKPT")
        write_i32(f, 17) # GPT type ID
        write_i32(f, vocab_size)
        write_i32(f, d_model)
        write_i32(f, num_heads)
        write_i32(f, d_ff)
        write_i32(f, num_layers)
        write_i32(f, max_seq_len)

        # Embeddings
        write_tensor(f, state_dict["transformer.wte.weight"])
        write_tensor(f, state_dict["transformer.wpe.weight"])

        # Blocks
        for i in range(num_layers):
            prefix = f"transformer.h.{i}."
            c_attn_w = state_dict[f"{prefix}attn.c_attn.weight"]
            c_attn_b = state_dict[f"{prefix}attn.c_attn.bias"]
            
            write_tensor(f, c_attn_w[:, :d_model])
            write_tensor(f, c_attn_b[:d_model])
            write_tensor(f, c_attn_w[:, d_model:2*d_model])
            write_tensor(f, c_attn_b[d_model:2*d_model])
            write_tensor(f, c_attn_w[:, 2*d_model:])
            write_tensor(f, c_attn_b[2*d_model:])
            
            write_tensor(f, state_dict[f"{prefix}attn.c_proj.weight"])
            write_tensor(f, state_dict[f"{prefix}attn.c_proj.bias"])
            write_tensor(f, state_dict[f"{prefix}mlp.c_fc.weight"])
            write_tensor(f, state_dict[f"{prefix}mlp.c_fc.bias"])
            write_tensor(f, state_dict[f"{prefix}mlp.c_proj.weight"])
            write_tensor(f, state_dict[f"{prefix}mlp.c_proj.bias"])
            
            write_tensor(f, state_dict[f"{prefix}ln_1.weight"].unsqueeze(0))
            write_tensor(f, state_dict[f"{prefix}ln_1.bias"].unsqueeze(0))
            write_tensor(f, state_dict[f"{prefix}ln_2.weight"].unsqueeze(0))
            write_tensor(f, state_dict[f"{prefix}ln_2.bias"].unsqueeze(0))

        # Final LN
        write_tensor(f, state_dict["transformer.ln_f.weight"].unsqueeze(0))
        write_tensor(f, state_dict["transformer.ln_f.bias"].unsqueeze(0))

        # Output Proj (MUST transpose)
        write_tensor(f, state_dict["lm_head.weight"], transpose=True)
        write_i32(f, 1)
        write_i32(f, vocab_size)
        for _ in range(vocab_size):
            write_f64(f, 0.0)

    print(f"✓ Successfully wrote {output_path}")

if __name__ == "__main__":
    convert_distilgpt2("distilgpt2_checkpoint.bin")