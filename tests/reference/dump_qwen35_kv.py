#!/usr/bin/env python3
"""
Dump real KV cache from Qwen3.5-0.8B for TurboQuant validation.

Model: https://huggingface.co/Qwen/Qwen3.5-0.8B
Architecture: Hybrid (6 × (3× DeltaNet + 1× Gated Attention))
- 24 layers total, 6 layers with standard KV cache (Gated Attention layers)
- KV heads: 2 (GQA), Query heads: 8, Head dim: 256
- Context: 262K tokens

Usage:
    pip install transformers torch
    python3 tests/reference/dump_qwen35_kv.py
"""

import struct
import os
import sys
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../../spec/test_vectors/qwen35_kv")

MAGIC = 0x544B5651  # "QVKT" in little-endian


def dump_with_transformers():
    """Dump real KV cache using HuggingFace transformers."""
    print("Loading Qwen3.5-0.8B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    prompt = (
        "The key to efficient LLM inference is quantizing the key-value cache, "
        "which reduces memory usage while preserving attention accuracy. "
        "Recent research from Google has shown that TurboQuant can achieve "
        "6x KV cache compression at 3-bit with zero quality loss. "
        "The algorithm works by first applying a Random Hadamard Transform "
        "to decorrelate the vector coordinates, then using optimal scalar "
        "quantizers on each dimension independently."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    print(f"Prompt tokens: {seq_len}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Qwen3.5-0.8B: hybrid architecture
    # past_key_values has entries for each layer, but DeltaNet layers
    # may have different cache format. We extract what's available.
    layers_saved = 0
    for layer_idx in range(len(past_kv)):
        kv = past_kv[layer_idx]
        if kv is None:
            continue

        # Standard attention layers return (key, value) tensors
        # Shape: [batch, num_kv_heads, seq_len, head_dim]
        try:
            if isinstance(kv, (tuple, list)) and len(kv) >= 2:
                keys = kv[0]
                values = kv[1]

                if not isinstance(keys, torch.Tensor):
                    continue

                if keys.dim() < 3:
                    continue

                keys_np = keys.squeeze(0).float().numpy()
                values_np = values.squeeze(0).float().numpy()

                if keys_np.ndim == 2:
                    # [seq_len, hidden] → reshape to [1, seq_len, hidden]
                    keys_np = keys_np[np.newaxis, :, :]
                    values_np = values_np[np.newaxis, :, :]

                num_heads = keys_np.shape[0]
                seq_l = keys_np.shape[1]
                head_dim = keys_np.shape[2]

                print(f"  Layer {layer_idx}: heads={num_heads}, seq={seq_l}, dim={head_dim}")

                # Save keys
                fname_k = os.path.join(OUTPUT_DIR, f"layer{layer_idx}_keys.bin")
                with open(fname_k, "wb") as f:
                    f.write(struct.pack("<4I", MAGIC, layer_idx, num_heads, seq_l))
                    f.write(struct.pack("<I", head_dim))
                    for h in range(num_heads):
                        for s in range(seq_l):
                            f.write(keys_np[h, s, :].astype(np.float32).tobytes())

                # Save values
                fname_v = os.path.join(OUTPUT_DIR, f"layer{layer_idx}_values.bin")
                with open(fname_v, "wb") as f:
                    f.write(struct.pack("<4I", MAGIC, layer_idx, num_heads, seq_l))
                    f.write(struct.pack("<I", head_dim))
                    for h in range(num_heads):
                        for s in range(seq_l):
                            f.write(values_np[h, s, :].astype(np.float32).tobytes())

                layers_saved += 1
                if layers_saved >= 4:
                    break
        except Exception as e:
            print(f"  Layer {layer_idx}: skipped ({e})")
            continue

    if layers_saved == 0:
        print("No standard KV cache layers found. Generating synthetic data.")
        generate_qwen35_synthetic()
    else:
        print(f"\nSaved {layers_saved} layers to {OUTPUT_DIR}")

    # Save model info
    info_path = os.path.join(OUTPUT_DIR, "model_info.txt")
    with open(info_path, "w") as f:
        f.write(f"model: {model_name}\n")
        f.write(f"prompt_tokens: {seq_len}\n")
        f.write(f"layers_saved: {layers_saved}\n")
        f.write(f"architecture: hybrid (DeltaNet + Gated Attention)\n")
        f.write(f"kv_heads: 2\n")
        f.write(f"q_heads: 8\n")
        f.write(f"head_dim: 256\n")
        f.write(f"total_layers: 24\n")
        f.write(f"attention_layers: 6 (every 4th layer)\n")

    return layers_saved


def generate_qwen35_synthetic():
    """Generate realistic synthetic KV data matching Qwen3.5-0.8B statistics."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    # Qwen3.5-0.8B Gated Attention specs
    num_kv_heads = 2
    head_dim = 256
    seq_len = 64
    attention_layers = [3, 7, 11, 15]  # every 4th layer in the 24-layer hybrid

    for i, layer_idx in enumerate(attention_layers):
        # Real LLM keys: per-channel variance with log-normal distribution
        # Deeper layers have larger variance (typical in transformers)
        depth_factor = 1.0 + 0.5 * i
        channel_std = np.random.lognormal(
            mean=-1.5, sigma=0.8, size=head_dim
        ).astype(np.float32) * depth_factor

        keys = np.zeros((num_kv_heads, seq_len, head_dim), dtype=np.float32)
        values = np.zeros((num_kv_heads, seq_len, head_dim), dtype=np.float32)

        for h in range(num_kv_heads):
            for d in range(head_dim):
                keys[h, :, d] = np.random.normal(0, channel_std[d], seq_len)
                values[h, :, d] = np.random.normal(0, 0.1, seq_len)

            # RoPE-like outliers (sparse, large magnitude)
            n_outliers = max(1, seq_len // 16)
            outlier_pos = np.random.choice(seq_len, size=n_outliers, replace=False)
            outlier_dims = np.random.choice(head_dim, size=4, replace=False)
            for p in outlier_pos:
                for od in outlier_dims:
                    keys[h, p, od] *= 5.0

        # Save
        for name, data in [("keys", keys), ("values", values)]:
            fname = os.path.join(OUTPUT_DIR, f"layer{layer_idx}_{name}.bin")
            with open(fname, "wb") as f:
                f.write(struct.pack("<4I", MAGIC, layer_idx, num_kv_heads, seq_len))
                f.write(struct.pack("<I", head_dim))
                for h in range(num_kv_heads):
                    for s in range(seq_len):
                        f.write(data[h, s, :].astype(np.float32).tobytes())

        print(f"  Layer {layer_idx}: {num_kv_heads}h x {seq_len}seq x {head_dim}d")

    # Save model info
    info_path = os.path.join(OUTPUT_DIR, "model_info.txt")
    with open(info_path, "w") as f:
        f.write("model: Qwen/Qwen3.5-0.8B (synthetic)\n")
        f.write(f"prompt_tokens: {seq_len}\n")
        f.write("layers_saved: 4\n")
        f.write("architecture: hybrid (DeltaNet + Gated Attention)\n")
        f.write(f"kv_heads: {num_kv_heads}\n")
        f.write("q_heads: 8\n")
        f.write(f"head_dim: {head_dim}\n")
        f.write("total_layers: 24\n")
        f.write("attention_layers: 6 (every 4th)\n")
        f.write("note: synthetic data with realistic statistics\n")

    print(f"\nSynthetic data saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=" * 60)
    print("  TurboQuant — Qwen3.5-0.8B KV Cache Dump")
    print("=" * 60)
    print()

    try:
        import torch
        from transformers import AutoModelForCausalLM
        print("transformers + torch available, attempting real model dump...")
        layers = dump_with_transformers()
    except ImportError as e:
        print(f"transformers/torch not available ({e})")
        print("Generating synthetic data with Qwen3.5-0.8B statistics...")
        generate_qwen35_synthetic()
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Falling back to synthetic data...")
        generate_qwen35_synthetic()

    print("\nDone!")
