"""
Kernel AI v1 -- real training run on Kaggle T4x2 GPU, self-contained
(model + optimizer inlined since Kaggle script kernels only upload one
file). Pushed/monitored via the Kaggle API, not a manually-run notebook.
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ model.py (inlined) ============
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


# ============ muon.py (inlined) ============
def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate orthogonalization of G via a quintic Newton-Schulz
    iteration — avoids an explicit, expensive SVD."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.float()


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, ns_steps = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                if g.ndim == 2:
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                else:
                    scale = 1.0
                p.add_(g, alpha=-lr * scale)


def split_muon_adamw_params(model):
    """Muon for 2D hidden matrices, AdamW for everything else
    (embeddings, 1D norms/biases, the output head)."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "tok_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


# ============ training script ============
# Training data comes from a Kaggle Dataset (karenpearlvpabilando/kernel-ai-corpus),
# mounted read-only at this path when the dataset is attached to the kernel —
# not fetched over the network, so no GitHub size limits apply.
DATASET_DIR = "/kaggle/input/kernel-ai-corpus"

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# Sized for a real multi-hour Kaggle T4x2 session: enough capacity to be
# meaningfully more than a toy, small enough to actually finish real steps
# in a single session instead of timing out mid-run.
D_MODEL, N_LAYERS, N_HEADS = 384, 8, 6
SEQ_LEN = 256
BATCH_SIZE = 24
TOTAL_STEPS = 6000
CHECKPOINT_EVERY = 500
LR_MUON = 0.02
LR_ADAMW = 3e-4


def load_data(corpus_path, tokenizer_path, seq_len, val_fraction=0.02, max_chars=None):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    with open(corpus_path, "r", errors="ignore") as f:
        text = f.read(max_chars) if max_chars else f.read()
    print(f"  corpus text: {len(text)/1e6:.1f}M chars")
    ids = tok.encode(text).ids
    print(f"  tokenized: {len(ids)/1e6:.1f}M tokens, vocab={tok.get_vocab_size()}")
    data = torch.tensor(ids, dtype=torch.long)
    n_chunks = len(data) // seq_len
    data = data[: n_chunks * seq_len].view(n_chunks, seq_len)
    n_val = max(1, int(n_chunks * val_fraction))
    return data[n_val:], data[:n_val], tok.get_vocab_size()


def get_batch(data, batch_size, device):
    idx = torch.randint(0, data.shape[0], (batch_size,))
    batch = data[idx].to(device)
    return batch[:, :-1], batch[:, 1:]


def find_dataset_file(filename):
    """Kaggle's dataset mount path isn't always a flat /kaggle/input/<slug>/
    — walk the tree to find the file wherever it actually landed."""
    for root_dir in ("/kaggle/input", DATASET_DIR):
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _, filenames in os.walk(root_dir):
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"torch version: {torch.__version__}, cuda built: {torch.version.cuda}")

    if os.path.isdir("/kaggle/input"):
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            print(f"  {dirpath}: dirs={dirnames} files={filenames[:5]}")
    else:
        print("NO /kaggle/input DIR")

    corpus_path = find_dataset_file("kernel_ai_final_corpus.txt")
    tokenizer_path = find_dataset_file("kernel_ai_final_tokenizer.json")
    if not corpus_path or not tokenizer_path:
        raise FileNotFoundError(
            "could not locate kernel_ai_final_corpus.txt / kernel_ai_final_tokenizer.json "
            "anywhere under /kaggle/input — attach the 'kernel-ai-corpus' dataset to this kernel."
        )

    train_data, val_data, vocab_size = load_data(corpus_path, tokenizer_path, SEQ_LEN)
    print(f"train sequences: {train_data.shape[0]}, val: {val_data.shape[0]}")

    cfg = KernelLLMConfig(
        vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        max_seq_len=SEQ_LEN, use_moe=False,
    )
    model = KernelLLM(cfg).to(device)
    print(f"model: {model.num_params():,} params")

    muon_params, adamw_params = split_muon_adamw_params(model)
    muon_opt = Muon(muon_params, lr=LR_MUON)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=LR_ADAMW)

    log_path = os.path.join(OUT_DIR, "train_log.jsonl")
    t0 = time.time()
    for step in range(TOTAL_STEPS):
        model.train()
        x, y = get_batch(train_data, BATCH_SIZE, device)
        _, loss = model(x, y)
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        loss.backward()
        muon_opt.step()
        adamw_opt.step()

        if step % 50 == 0 or step == TOTAL_STEPS - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, min(BATCH_SIZE, val_data.shape[0]), device)
                _, val_loss = model(vx, vy)
            elapsed = time.time() - t0
            line = f"step {step:5d}  train_loss {loss.item():.4f}  val_loss {val_loss.item():.4f}  elapsed {elapsed:.0f}s"
            print(line)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "train_loss": loss.item(),
                                     "val_loss": val_loss.item(), "elapsed_s": elapsed}) + "\n")

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(OUT_DIR, f"kernel_ai_v1_step{step}.pt")
            torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step}, ckpt_path)
            print(f"  checkpoint saved: {ckpt_path}")

    final_path = os.path.join(OUT_DIR, "kernel_ai_v1_final.pt")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": TOTAL_STEPS}, final_path)
    print(f"training complete, final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

