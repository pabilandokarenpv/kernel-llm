# RFC-001: End-to-end system architecture

**Status:** Implemented (training in progress at time of writing)

## Summary

Kernel AI has two halves that run on different infrastructure, connected
by a GitHub-and-Kaggle-hosted checkpoint handoff: **training** (Kaggle,
GPU, batch) and **serving** (Cloudflare Workers, scale-to-zero, request-
driven). Both are agent-drivable end to end via API/CLI — no manual
notebook-clicking or dashboard-only steps anywhere in the loop except the
one-time account setup each platform requires.

## Full system diagram

```mermaid
flowchart TD
    subgraph SOURCES["Data sources"]
        TS["TinyStories"]
        OASST["OASST2"]
        CA["CodeAlpaca-20k"]
        DOLLY["Dolly15k"]
        GSM["GSM8K"]
        KTRACE["Kernel execution traces\n(ours, from kernel-lang)"]
        TEACHER["gpt-oss-120b\n(data generation ONLY — ARD-002)"]
    end

    TS & OASST & CA & DOLLY & GSM & KTRACE --> MERGE["build_final_corpus.py\n110,187 examples, 99MB"]
    TEACHER -->|"datagen-worker\n(Cloudflare Worker)"| DISTILL["distilled/*.jsonl\n440 examples"]
    DISTILL --> MERGE

    MERGE --> TOK["train_assistant_tokenizer.py\n24.5k vocab BPE"]
    MERGE --> KDATASET["Kaggle Dataset\nkernel-ai-corpus (98MB)"]
    TOK --> KDATASET

    KDATASET --> KTRAIN["kaggle_kernel_train_v1.py\nruns on Kaggle T4x2 GPU\n(pushed/polled via Kaggle API)"]
    KTRAIN --> CKPT["checkpoint (.pt)\nd_model=384, n_layers=8, n_heads=6\ndense SwiGLU, no MoE"]

    CKPT --> EXPORT["export_numpy.py\nflat binary weight format"]
    EXPORT --> GH["GitHub (raw.githubusercontent.com)\nweight blob host"]

    GH --> WORKER["Cloudflare Python Worker\nnumpy_forward.py\n(hand-written forward pass)"]
    WORKER --> KERNELAI["Kernel AI product (Slate)\nchat UI: sessions, Artifact panel, Kernel Code panel"]
    USER["user"] --> KERNELAI

    style TEACHER fill:#4a2e1a,stroke:#862,color:#fff
    style KERNELAI fill:#16213e,stroke:#0f3460,color:#fff
    style KTRAIN fill:#1a4a2e,stroke:#2a8,color:#fff
    style WORKER fill:#1a4a2e,stroke:#2a8,color:#fff
```

## Why two different cloud platforms

| Concern | Platform | Why |
|---|---|---|
| Training (batch, GPU-hungry, hours) | Kaggle | Free GPU quota (~30 hrs/week), cardless, real API — see ARD-004 |
| Serving (request-driven, must be cheap at near-zero traffic) | Cloudflare Workers | Scale-to-zero, matches en's standing infra preference, Python Workers support numpy — see ARD-003 |

Training and serving are explicitly treated as **separable concerns** —
committing to Cloudflare for serving never implied Cloudflare for training;
Kaggle was chosen for training purely on its own merits (free GPU,
cardless, automatable).

## The handoff: checkpoint → servable weights

A checkpoint trained on Kaggle isn't directly servable — it's a PyTorch
`.pt` file, and the Worker has no PyTorch. `export_numpy.py` flattens the
state dict into a manifest + raw float32 byte format (chosen over `.npz`
specifically because zip/pickle parsing was the actual CPU-budget
bottleneck on Worker cold start, not the matmuls — see ARD-003), and the
Worker fetches it live from GitHub at request time.

## Product surface

The trained model is wrapped in "Kernel AI" (built on the `slate`
Cloudflare Worker) — a chat interface with sessions, an Artifact panel, and
a Kernel Code panel, deliberately modeled on the ChatGPT/Claude/Claude Code
structural pattern. See `product/PRDs/PRD-001-kernel-ai-product.md` for the
product requirements and `engineering/RFCs/RFC-004-serving-pipeline.md`
for exactly how a chat request becomes a model response.
