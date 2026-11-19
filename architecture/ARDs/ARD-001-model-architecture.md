# ARD-001: Model architecture — RoPE, RMSNorm, QK-Norm, SwiGLU, Muon; dense FFN for the serving target

**Status:** Accepted

## Context

The model needs to be genuinely "ours" (see ARD-002) — not a fine-tune, not
a wrapped API — while still being small enough to train on free GPU compute
(Kaggle T4x2, ~30 GPU-hrs/week) and small enough to serve on Cloudflare's
free-tier compute budgets. Every architecture choice below is a deliberate,
citable deviation from a vanilla nanoGPT baseline, chosen for either
training stability at small scale or serving simplicity.

## Decision

`model/model.py` implements:

| Component | Choice | Citation / reason |
|---|---|---|
| Position encoding | RoPE (rotary) | Su et al. 2021 — no learned absolute position table to store/serve |
| Normalization | RMSNorm | Zhang & Sennrich 2019 — cheaper than LayerNorm, no mean-centering |
| Attention stability | QK-Norm | normalizes q/k pre-attention; stabilizes small-scale training |
| FFN activation | SwiGLU | Shazeer 2020 |
| Optional FFN variant | top-k MoE with load-balancing aux loss | Shazeer et al. 2017 / Switch Transformer 2021 — **available in training code, excluded from the serving path** |
| Output head init | zero-init | muP-style stability at small scale |
| Optimizer | Muon (orthogonalized momentum SGD) for 2D hidden matrices, AdamW for embeddings/norms/head | Keller Jordan et al. 2024 |

For the model that actually gets served (Kernel AI v1), **MoE is disabled**
(`use_moe=False`, dense SwiGLU only) — see ARD-003 for why.

## Alternatives considered

- **Vanilla nanoGPT (learned abs. position, LayerNorm, GELU MLP, AdamW
  only)** — rejected: wanted architecture choices that are each individually
  citable and testable, both for research credibility and because this
  project doubles as an interpretability-research substrate (see
  kernel-lang's PRD) where architecture choices matter for what gets
  probed later.
- **MoE as the default, not optional** — rejected for the *serving* model:
  MoE's dynamic top-k routing isn't traceable to a static computation graph,
  which blocks every serving path that needs a fixed graph (ONNX export,
  and implicitly any future non-Python-Worker backend). Dense SwiGLU has no
  such restriction. MoE is kept in `model.py` as an option for future
  *training-side* interpretability experiments (the original research
  question: does an MoE router specialize by program structure?), just not
  used for the model that ships.

## Consequences

- The served checkpoint (dense, no MoE) is directly portable across every
  serving path explored (ONNX, onnxruntime-node, pure numpy) with no
  architecture mismatch to debug — this constraint was discovered the hard
  way during the ONNX export work and then applied retroactively as a rule.
- MoE interpretability work (the original research question) requires a
  *separate* training run with `use_moe=True` if pursued — it's available
  in the codebase, just not on the serving critical path.
