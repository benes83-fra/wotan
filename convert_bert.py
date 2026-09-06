import struct
import sys
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    print("Please install: pip install torch transformers")
    sys.exit(1)


def write_i32(f, val):
    f.write(struct.pack('<i', int(val)))


def write_f64(f, val):
    f.write(struct.pack('<d', float(val)))


def write_tensor(f, tensor, name="unknown"):
    """
    Writes a PyTorch tensor to Wotan's binary format.
    
    Wotan's conventions:
      - 1D tensors (biases, gamma, beta) -> [1, N] row vectors
      - Embedding weights               -> [vocab, dim] (unchanged)
      - Linear layer weights            -> [in_features, out_features] (TRANSPOSED from PyTorch)
    """
    if tensor is None:
        print(f"  ❌ {name}: MISSING - writing 0x0 tensor!")
        write_i32(f, 0)
        write_i32(f, 0)
        return False

    if tensor.dim() == 1:
        # 1D tensor -> write as 1 x N row vector (for biases, LayerNorm gamma/beta)
        rows = 1
        cols = tensor.shape[0]
        data = tensor.cpu().detach().to(torch.float64).numpy()
    else:
        # 2D tensor
        if "embedding" in name.lower() or name in ("token_emb", "pos_emb", "segment_emb"):
            # Embedding weights: keep as [vocab, dim]
            rows = tensor.shape[0]
            cols = tensor.shape[1]
            data = tensor.cpu().detach().to(torch.float64).numpy().flatten()
        else:
            # Linear layer weights: PyTorch stores [out, in], Wotan expects [in, out]
            # Transpose first, then flatten
            rows = tensor.shape[1]  # in_features
            cols = tensor.shape[0]  # out_features
            data = tensor.cpu().detach().t().to(torch.float64).numpy().flatten()

    write_i32(f, rows)
    write_i32(f, cols)
    for val in data:
        write_f64(f, float(val))

    print(f"  ✅ {name}: {rows}x{cols}")
    return True


def convert_hf_bert_to_wotan(hf_model_path, output_path):
    print(f"Loading model from '{hf_model_path}'...")

    from transformers import AutoModelForSequenceClassification, AutoModel, AutoConfig

    config = AutoConfig.from_pretrained(hf_model_path)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
        print("✅ Loaded as SequenceClassification model")
    except Exception as e:
        print(f"Failed to load as classification model: {e}")
        model = AutoModel.from_pretrained(hf_model_path)
        print("✅ Loaded as base BERT model")

    state_dict = model.state_dict()

    print(f"\n{'=' * 60}")
    print(f"State dict has {len(state_dict)} keys")
    print(f"{'=' * 60}")
    for key in sorted(state_dict.keys()):
        print(f"  {key}: {state_dict[key].shape}")
    print(f"{'=' * 60}\n")

    vocab_size = config.vocab_size
    d_model = config.hidden_size
    num_heads = config.num_attention_heads
    d_ff = config.intermediate_size
    num_layers = config.num_hidden_layers
    max_seq_len = config.max_position_embeddings

    print(f"Config: vocab={vocab_size}, d_model={d_model}, heads={num_heads}, layers={num_layers}")

    # Detect prefix
    prefix = ""
    for test_key in ["bert.embeddings.word_embeddings.weight",
                     "roberta.embeddings.word_embeddings.weight",
                     "embeddings.word_embeddings.weight"]:
        if test_key in state_dict:
            prefix = test_key.split("embeddings")[0]
            print(f"Detected prefix: '{prefix}'")
            break

    with open(output_path, "wb") as f:
        f.write(b"WOTAN_CKPT")
        write_i32(f, 18)  # BERT type ID

        write_i32(f, vocab_size)
        write_i32(f, d_model)
        write_i32(f, num_heads)
        write_i32(f, d_ff)
        write_i32(f, num_layers)
        write_i32(f, max_seq_len)

        print("\nWriting tensors:")

        # Embeddings
        emb_key = f"{prefix}embeddings.word_embeddings.weight"
        if emb_key not in state_dict:
            print(f"❌ CRITICAL: Key '{emb_key}' not found!")
            print(f"   Available keys with 'embedding':")
            for k in state_dict.keys():
                if "embedding" in k.lower():
                    print(f"     {k}")
        write_tensor(f, state_dict.get(emb_key), "token_emb")

        pos_key = f"{prefix}embeddings.position_embeddings.weight"
        write_tensor(f, state_dict.get(pos_key), "pos_emb")

        seg_key = f"{prefix}embeddings.token_type_embeddings.weight"
        write_tensor(f, state_dict.get(seg_key), "segment_emb")

        # Embedding LayerNorm
        ln_w_key = f"{prefix}embeddings.LayerNorm.weight"
        ln_b_key = f"{prefix}embeddings.LayerNorm.bias"
        write_tensor(f, state_dict.get(ln_w_key), "emb_ln.weight")
        write_tensor(f, state_dict.get(ln_b_key), "emb_ln.bias")

        # Encoder blocks
        for i in range(num_layers):
            p = f"{prefix}encoder.layer.{i}."
            write_tensor(f, state_dict.get(f"{p}attention.self.query.weight"), f"block{i}.q.weight")
            write_tensor(f, state_dict.get(f"{p}attention.self.query.bias"), f"block{i}.q.bias")
            write_tensor(f, state_dict.get(f"{p}attention.self.key.weight"), f"block{i}.k.weight")
            write_tensor(f, state_dict.get(f"{p}attention.self.key.bias"), f"block{i}.k.bias")
            write_tensor(f, state_dict.get(f"{p}attention.self.value.weight"), f"block{i}.v.weight")
            write_tensor(f, state_dict.get(f"{p}attention.self.value.bias"), f"block{i}.v.bias")
            write_tensor(f, state_dict.get(f"{p}attention.output.dense.weight"), f"block{i}.out_proj.weight")
            write_tensor(f, state_dict.get(f"{p}attention.output.dense.bias"), f"block{i}.out_proj.bias")

            write_tensor(f, state_dict.get(f"{p}intermediate.dense.weight"), f"block{i}.ffn.fc1.weight")
            write_tensor(f, state_dict.get(f"{p}intermediate.dense.bias"), f"block{i}.ffn.fc1.bias")
            write_tensor(f, state_dict.get(f"{p}output.dense.weight"), f"block{i}.ffn.fc2.weight")
            write_tensor(f, state_dict.get(f"{p}output.dense.bias"), f"block{i}.ffn.fc2.bias")

            write_tensor(f, state_dict.get(f"{p}attention.output.LayerNorm.weight"), f"block{i}.ln1.weight")
            write_tensor(f, state_dict.get(f"{p}attention.output.LayerNorm.bias"), f"block{i}.ln1.bias")
            write_tensor(f, state_dict.get(f"{p}output.LayerNorm.weight"), f"block{i}.ln2.weight")
            write_tensor(f, state_dict.get(f"{p}output.LayerNorm.bias"), f"block{i}.ln2.bias")

        # Pooler
        pool_w_key = f"{prefix}pooler.dense.weight"
        pool_b_key = f"{prefix}pooler.dense.bias"
        if pool_w_key in state_dict:
            write_tensor(f, state_dict.get(pool_w_key), "pooler.weight")
            write_tensor(f, state_dict.get(pool_b_key), "pooler.bias")
        else:
            print(f"  ⚠ No pooler found, writing identity")
            write_tensor(f, torch.eye(d_model, dtype=torch.float64), "pooler.weight (identity)")
            write_tensor(f, torch.zeros(d_model, dtype=torch.float64), "pooler.bias (zeros)")

        # MLM Head (tied to embeddings)
        write_tensor(f, state_dict.get(emb_key), "mlm_head.weight (tied)")
        write_tensor(f, torch.zeros(vocab_size, dtype=torch.float64), "mlm_head.bias")

        # Classification Head
        cls_w = state_dict.get("classifier.weight")
        cls_b = state_dict.get("classifier.bias")

        if cls_w is None:
            cls_w = state_dict.get("cls.seq_relationship.weight")
            cls_b = state_dict.get("cls.seq_relationship.bias")

        if cls_w is not None:
            write_tensor(f, cls_w, f"classifier.weight ({cls_w.shape[0]} classes)")
            write_tensor(f, cls_b, "classifier.bias")
        else:
            print(f"  ⚠ No classifier found, writing dummy 3-class")
            write_tensor(f, torch.randn(3, d_model, dtype=torch.float64) * 0.02, "classifier.weight (dummy)")
            write_tensor(f, torch.zeros(3, dtype=torch.float64), "classifier.bias (zeros)")

    file_size = os.path.getsize(output_path)
    print(f"\n🎉 Wrote {output_path} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_bert.py <hf_model_name_or_path> <output_wotan.bin>")
        sys.exit(1)

    convert_hf_bert_to_wotan(sys.argv[1], sys.argv[2])