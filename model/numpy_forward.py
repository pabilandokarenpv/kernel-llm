"""
Pure-numpy forward pass for Kernel AI's model — the exact same
architecture as model.py (RMSNorm, RoPE, QK-norm attention, dense SwiGLU),
reimplemented without PyTorch so it can run inside a Cloudflare Python
Worker (Pyodide ships numpy; it does not ship PyTorch).

This is NOT a different model — it's the same trained weights, same math,
just a different (framework-free) execution path. Correctness is proven
by exact-match comparison against the PyTorch forward pass, not assumed.
"""

import numpy as np


def rms_norm(x, weight, eps=1e-6):
    norm = x * (1.0 / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps))
    return norm * weight


def precompute_rope(dim, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
    t = np.arange(max_seq_len).astype(np.float32)
    freqs = np.outer(t, inv_freq)  # (seq, dim/2)
    return np.cos(freqs), np.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    T = x.shape[2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[:T][None, None, :, :]
    s = sin[:T][None, None, :, :]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    out = np.stack([out1, out2], axis=-1)
    return out.reshape(x.shape)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def causal_self_attention(x, w, prefix, n_heads, cos, sin):
    B, T, C = x.shape
    head_dim = C // n_heads

    qkv = x @ w[f"{prefix}.qkv.weight"].T  # (B,T,3C)
    qkv = qkv.reshape(B, T, 3, n_heads, head_dim).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,T,D)

    q = rms_norm(q, w[f"{prefix}.q_norm.weight"])
    k = rms_norm(k, w[f"{prefix}.k_norm.weight"])
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)  # (B,H,T,T)
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal_mask[None, None, :, :], -1e9, scores)
    attn = softmax(scores, axis=-1)
    out = attn @ v  # (B,H,T,D)

    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    return out @ w[f"{prefix}.proj.weight"].T


def swiglu(x, w, prefix):
    a = x @ w[f"{prefix}.w1.weight"].T
    b = x @ w[f"{prefix}.w2.weight"].T
    return (silu(a) * b) @ w[f"{prefix}.w3.weight"].T


def forward(input_ids, w, n_layers, n_heads, max_seq_len):
    """input_ids: (B, T) int array. w: dict of numpy weight arrays. Returns logits (B, T, V)."""
    d_model = w["tok_emb.weight"].shape[1]
    head_dim = d_model // n_heads
    cos, sin = precompute_rope(head_dim, max_seq_len)

    x = w["tok_emb.weight"][input_ids]  # (B, T, C)

    for i in range(n_layers):
        p = f"blocks.{i}"
        n1 = rms_norm(x, w[f"{p}.norm1.weight"])
        x = x + causal_self_attention(n1, w, f"{p}.attn", n_heads, cos, sin)
        n2 = rms_norm(x, w[f"{p}.norm2.weight"])
        x = x + swiglu(n2, w, f"{p}.ffn")

    x = rms_norm(x, w["norm_f.weight"])
    logits = x @ w["head.weight"].T
    return logits


def generate(input_ids, w, n_layers, n_heads, max_seq_len, max_new_tokens=40,
             temperature=0.8, eos_id=None, rng=None):
    """Naive autoregressive generation: re-runs the full forward pass each
    step (no KV cache — the model is small enough that this is still fast,
    and it keeps numpy_forward.py's core `forward()` simple/verified).
    input_ids: (1, T) int array. Returns the full extended token list."""
    rng = rng or np.random.default_rng()
    ids = list(input_ids[0])
    for _ in range(max_new_tokens):
        if len(ids) >= max_seq_len:
            break
        cur = np.array([ids], dtype=np.int64)
        logits = forward(cur, w, n_layers, n_heads, max_seq_len)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = softmax(next_logits[None, :], axis=-1)[0]
        next_id = int(rng.choice(len(probs), p=probs))
        ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return ids
