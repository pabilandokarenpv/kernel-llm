"""
Kernel-LLM: a modernized nanoGPT-style decoder-only transformer.

Architecture choices (each a deliberate deviation from vanilla nanoGPT,
each citable, each testable via the ablation grid this file is designed
to support):
  - RoPE          (Su et al. 2021)      instead of learned absolute positions
  - RMSNorm       (Zhang & Sennrich 2019) instead of LayerNorm
  - QK-Norm                              normalize q/k pre-attention for stability
  - SwiGLU        (Shazeer 2020)         FFN activation
  - optional top-k MoE FFN (Shazeer et al. 2017 / Switch Transformer 2021)
    as a drop-in swap for the dense SwiGLU FFN, with load-balancing aux loss
  - zero-init output projection (muP-style) for small-scale training stability

This file is deliberately framework-light: pure PyTorch, no dependency on
a training framework, so it runs unmodified on a laptop CPU for a smoke
test and on a Kaggle T4/TPU for the real run.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(dim, max_seq_len, base=10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq, dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    # x: (B, H, T, D). Rotate pairs (x0,x1),(x2,x3),...
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[:x.size(2)].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.size(2)].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)
        q, k = self.q_norm(q), self.k_norm(k)  # QK-Norm
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MoEFFN(nn.Module):
    """Top-k routed MoE FFN with a load-balancing auxiliary loss.
    Each expert is a small SwiGLU. Returns (output, aux_loss)."""

    def __init__(self, d_model, d_ff, n_experts=4, top_k=1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        flat = x.view(-1, C)  # (N, C)
        logits = self.router(flat)  # (N, E)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = (topk_idx == e).any(dim=-1)
            if mask.any():
                expert_out = self.experts[e](flat[mask])
                weight = topk_probs[mask][topk_idx[mask] == e].unsqueeze(-1) if self.top_k > 1 else topk_probs[mask]
                out[mask] += expert_out * weight

        # load-balancing aux loss (Switch Transformer style): encourages
        # uniform routing — fraction-of-tokens * fraction-of-router-prob per expert
        frac_tokens = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            frac_tokens[e] = (topk_idx == e).any(dim=-1).float().mean()
        frac_prob = probs.mean(dim=0)
        aux_loss = self.n_experts * (frac_tokens * frac_prob).sum()

        return out.view(B, T, C), aux_loss


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.use_moe = cfg.use_moe
        if cfg.use_moe:
            self.ffn = MoEFFN(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
            x = x + ffn_out
            return x, aux_loss
        else:
            x = x + self.ffn(self.norm2(x))
            return x, torch.tensor(0.0, device=x.device)


class KernelLLMConfig:
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_heads=6,
                 d_ff=None, max_seq_len=1024, use_moe=False, n_experts=4, top_k=1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff or (4 * d_model * 2 // 3)  # SwiGLU convention
        self.max_seq_len = max_seq_len
        self.use_moe = use_moe
        self.n_experts = n_experts
        self.top_k = top_k


class KernelLLM(nn.Module):
    def __init__(self, cfg: KernelLLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.zeros_(self.head.weight)  # zero-init output (muP-style stability)

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        total_aux = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, aux = block(x, self.rope_cos, self.rope_sin)
            total_aux = total_aux + aux
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce + 0.01 * total_aux if self.cfg.use_moe else ce
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
