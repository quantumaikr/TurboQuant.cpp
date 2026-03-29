#!/usr/bin/env python3
"""
debug_deltanet.py — Capture per-layer activations from Qwen3.5-0.8B in PyTorch.

Saves numpy files to /tmp/tq_ref/ for comparison with the C engine.
Each file contains the hidden state after that layer's full processing
(attention/deltanet + FFN + residual).
"""

import torch
import numpy as np
import os
import warnings
import contextlib
import io
import sys

# Suppress noisy warnings from transformers
warnings.filterwarnings('ignore')

TOKEN_ID = 9419  # "Hello" — single token for easy debugging

print(f"Loading Qwen3.5-0.8B (token_id={TOKEN_ID})...")
with contextlib.redirect_stderr(io.StringIO()):
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3.5-0.8B',
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
model.eval()

# Print model structure summary
print(f"Model type: {type(model).__name__}")
lm = model.model
print(f"Number of layers: {len(lm.layers)}")

# Print config details relevant to our C implementation
# Qwen3.5 uses nested config: config.text_config has the details
tc = config.text_config if hasattr(config, 'text_config') else config
print(f"\nModel config:")
print(f"  hidden_size: {tc.hidden_size}")
print(f"  num_attention_heads: {tc.num_attention_heads}")
print(f"  num_key_value_heads: {tc.num_key_value_heads}")
print(f"  head_dim: {tc.head_dim}")
print(f"  intermediate_size: {tc.intermediate_size}")
print(f"  vocab_size: {tc.vocab_size}")
print(f"  rms_norm_eps: {tc.rms_norm_eps}")
print(f"  layer_types: {tc.layer_types}")
print(f"  linear_num_key_heads: {tc.linear_num_key_heads}")
print(f"  linear_key_head_dim: {tc.linear_key_head_dim}")
print(f"  linear_value_head_dim: {tc.linear_value_head_dim}")
print(f"  linear_conv_kernel_dim: {tc.linear_conv_kernel_dim}")
print(f"  attn_output_gate: {tc.attn_output_gate}")
if hasattr(tc, 'rope_parameters'):
    print(f"  rope_parameters: {tc.rope_parameters}")
if hasattr(tc, 'partial_rotary_factor'):
    print(f"  partial_rotary_factor: {tc.partial_rotary_factor}")

# Inspect layer types
for i, layer in enumerate(lm.layers):
    layer_type = "unknown"
    if hasattr(layer, 'self_attn') and layer.self_attn is not None:
        attn_cls = type(layer.self_attn).__name__
        layer_type = f"self_attn({attn_cls})"
    if hasattr(layer, 'linear_attn') and layer.linear_attn is not None:
        la_cls = type(layer.linear_attn).__name__
        layer_type = f"linear_attn({la_cls})"
    if i < 3 or i >= len(lm.layers) - 1:
        print(f"  layer {i:2d}: {layer_type}")
    elif i == 3:
        print(f"  ...")

# Register hooks to capture activations at each stage
activations = {}

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach().clone()
        elif isinstance(output, torch.Tensor):
            activations[name] = output.detach().clone()
        else:
            # BaseModelOutputWithPast or similar
            if hasattr(output, 'last_hidden_state'):
                activations[name] = output.last_hidden_state.detach().clone()
    return hook_fn

# Hook embedding
lm.embed_tokens.register_forward_hook(make_hook('embed'))

# Hook each transformer layer (captures output AFTER residual connections)
for i in range(len(lm.layers)):
    lm.layers[i].register_forward_hook(make_hook(f'layer{i:02d}'))

# Hook final norm
lm.norm.register_forward_hook(make_hook('final_norm'))

# Also capture intermediate states within layer 0 for detailed debugging
# Hook the attention/linear_attn sub-module of first few layers
for i in range(min(4, len(lm.layers))):
    layer = lm.layers[i]
    if hasattr(layer, 'linear_attn') and layer.linear_attn is not None:
        layer.linear_attn.register_forward_hook(make_hook(f'layer{i:02d}_linear_attn'))
    if hasattr(layer, 'self_attn') and layer.self_attn is not None:
        layer.self_attn.register_forward_hook(make_hook(f'layer{i:02d}_self_attn'))
    # Hook the MLP
    if hasattr(layer, 'mlp') and layer.mlp is not None:
        layer.mlp.register_forward_hook(make_hook(f'layer{i:02d}_mlp'))
    # Hook input_layernorm (pre-attention norm)
    if hasattr(layer, 'input_layernorm'):
        layer.input_layernorm.register_forward_hook(make_hook(f'layer{i:02d}_attn_norm'))
    # Hook post_attention_layernorm (pre-FFN norm)
    if hasattr(layer, 'post_attention_layernorm'):
        layer.post_attention_layernorm.register_forward_hook(make_hook(f'layer{i:02d}_ffn_norm'))

# Run forward pass
print(f"\nRunning forward pass with token_id={TOKEN_ID}...")
input_ids = torch.tensor([[TOKEN_ID]])

with torch.no_grad():
    out = model(input_ids, use_cache=False)

# Save activations
os.makedirs('/tmp/tq_ref', exist_ok=True)

print(f"\nActivations captured ({len(activations)} entries):")
print(f"{'Name':<30} {'Shape':<20} {'Mean':>12} {'Std':>12} {'[0:5]'}")
print("-" * 100)

for name in sorted(activations.keys()):
    t = activations[name]
    d = t.squeeze().float().numpy()
    np.save(f'/tmp/tq_ref/{name}.npy', d)
    vals = d.flatten()[:5]
    vals_str = ', '.join(f'{v:.4f}' for v in vals)
    print(f'{name:<30} {str(d.shape):<20} {d.mean():>12.6f} {d.std():>12.6f} [{vals_str}]')

# Save logits
logits = out.logits[0, -1, :].float().numpy()
np.save('/tmp/tq_ref/logits.npy', logits)

top_id = logits.argmax()
print(f'\nLogits: shape={logits.shape}, top_id={top_id}, top_val={logits[top_id]:.4f}')
print(f'logits[0:5] = [{", ".join(f"{v:.4f}" for v in logits[:5])}]')

# Also save embedding weights for first token to verify loading
embed_weight = lm.embed_tokens.weight[TOKEN_ID].detach().float().numpy()
np.save('/tmp/tq_ref/embed_weight_token.npy', embed_weight)
print(f'\nEmbed weight for token {TOKEN_ID}: [0:5]=[{", ".join(f"{v:.6f}" for v in embed_weight[:5])}]')

# Save some key DeltaNet weights for layer 0 to verify weight loading
layer0 = lm.layers[0]
if hasattr(layer0, 'linear_attn') and layer0.linear_attn is not None:
    la = layer0.linear_attn
    print(f"\nLayer 0 DeltaNet weight shapes:")
    for wname in ['A_log', 'dt_bias', 'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b',
                   'conv1d', 'norm', 'out_proj']:
        parts = wname.split('.')
        obj = la
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            if hasattr(obj, 'weight'):
                w = obj.weight
            elif isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
                w = obj
            else:
                w = None
            if w is not None:
                wd = w.detach().clone().float().numpy()
                np.save(f'/tmp/tq_ref/l0_{wname.replace(".", "_")}.npy', wd)
                print(f"  {wname}: shape={wd.shape} [0:3]={wd.flatten()[:3]}")

print(f"\nAll files saved to /tmp/tq_ref/")
print("Run the C comparison tool next to identify divergence point.")
