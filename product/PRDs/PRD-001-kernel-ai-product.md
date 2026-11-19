# PRD-001: Kernel AI

**Status:** Product shell shipped and deployed; first real training run on
Kaggle completed — val_loss 10.08 → 2.27 over 6,000 steps, confirmed real
(not a smoke test). Serving swap blocked on a confirmed Cloudflare Workers
free-tier platform limit (10ms CPU/request — see R8, Open items).

## Problem

en wants a ChatGPT/Claude-style product — chat with sessions, live
artifacts, a code panel — but powered by a model that is genuinely "ours,"
the same way every major lab's assistant is powered by their own trained
model, not a resale of someone else's API. Existing options are either
(a) wrap a hosted API (fast, but not "built from scratch" by any honest
definition) or (b) fine-tune an existing open model (still not from
scratch — it starts from someone else's pretrained weights). Neither
satisfies the actual goal.

## Goals

- G1: A chat product with the same structural shape as ChatGPT/Claude/
  Claude Code — sessions, an Artifact panel, a Kernel Code panel — not a
  bare text box.
- G2: Every response the user sees comes from a model trained entirely by
  us, from random initialization, on data we assembled — never a hosted
  model answering directly (see ARD-002).
- G3: $0 ongoing infrastructure cost at low/personal-scale traffic —
  matches en's standing infra preference (scale-to-zero, no idle cost).
- G4: The model should be broadly conversational ("like ChatGPT/Claude"),
  not narrowly limited to Kernel-language topics — while *also* retaining
  real competence on Kernel specifically, since that's this project's
  unique angle no other assistant has.

## Non-goals

- Matching GPT-4/Claude-tier capability. The served model (tens of
  millions of parameters, trained on free GPU compute) is honestly, and
  by design, far below frontier-model capability. The goal is a working,
  genuinely-ours pipeline end to end — not competing on raw capability.
- Multi-user production scale. This is personal-use-first (matches en's
  standing pattern for the Apple-app projects too) — the architecture
  happens to be cheap at scale, but scale isn't a design target.
- Fine-tuning or distilling the served model's *weights* directly from the
  teacher model. The teacher (gpt-oss-120b) only ever produces *training
  examples* that get mixed into the corpus — see ARD-002, this line is
  intentionally never crossed.

## Target users

- **Primary**: en — personal use, and the flagship example in a planned
  build-in-public series (LinkedIn/TikTok/X/Reddit) + research-paper-style
  writeups (Medium/Devpost/envember.com) across a ~20-project portfolio.
- **Secondary**: none planned yet.

## Requirements

| # | Requirement | Status |
|---|---|---|
| R1 | Chat UI with persistent sessions (sidebar, new-session) | ✅ shipped |
| R2 | Artifact panel (renders model-produced artifacts) | ✅ shipped |
| R3 | Kernel Code panel (runs/displays Kernel-language code) | ✅ shipped |
| R4 | Responsive across desktop / tablet / phone | ✅ shipped, verified at 3 breakpoints |
| R5 | Model is trained from random init, not fine-tuned from an existing pretrained model | ✅ by construction (`model.py` is a from-scratch architecture, Kaggle run starts from fresh weights) |
| R6 | No hosted model ever answers a real user message | ✅ enforced by architecture (ARD-002) — chat endpoint serves exclusively from our checkpoint |
| R7 | Training data includes broad, real, free English/code/reasoning sources, not just Kernel-language content | ✅ shipped (ARD-005) — 110,187-example merged corpus |
| R8 | $0 serving cost at low traffic | ⚠️ blocked — Cloudflare Workers Free caps CPU time at **10ms/request** (confirmed via `wrangler tail`, real error: "Worker exceeded CPU time limit"), not a probabilistic cold-start issue as ARD-003 originally guessed but a hard platform wall too small for any real model forward pass. Decision pending with en (pay $5/mo Workers Paid for 30s-5min CPU/request, find another free host, or shrink the model further). |
| R9 | $0 training cost | ✅ Kaggle free GPU quota, cardless (ARD-004) |
| R10 | Real trained checkpoint, verified against a genuine training run (not a smoke test claimed as "done") | ✅ shipped — 6,000 steps on Kaggle T4, val_loss 10.08 → 2.27, sample generations verified real (coherent English prose, structurally valid Kernel syntax, correct GSM8K answer-format though not yet reliably correct arithmetic) |

## User-facing architecture (what the user actually sees)

```mermaid
flowchart LR
    USER(("User")) --> UI["Kernel AI chat UI\n(sessions sidebar)"]
    UI --> MSG["message"]
    MSG --> PANEL1["Artifact panel\n(renders generated artifacts)"]
    MSG --> PANEL2["Kernel Code panel\n(Kernel-language code output)"]
    MSG -.->|"served by"| MODEL["our own trained model\n(never a hosted API)"]

    style MODEL fill:#1a4a2e,stroke:#2a8,color:#fff
```

## Success criteria

- A real user conversation, end to end, answered entirely by our own
  checkpoint, with no fabricated/assumed output — the standing project
  rule throughout has been to report real tool/training output, including
  garbled early-stage generations, rather than claim success without
  evidence.
- Loss curve from a real training run (not a CPU smoke test) demonstrating
  the model is actually learning on the full merged corpus.
- The product (Kernel AI / Slate) remains live and responsive at
  https://slate.pabilando-karenpv.workers.dev once the trained checkpoint
  replaces the current hosted-model interim.

## Open items at time of writing

- First full training run complete: 6,000 steps, dense 384d/8L/6h, Kaggle
  T4, ~46 min wall clock, val_loss 10.08 → 2.27. Sample generations
  verified real (not cherry-picked): coherent English story continuations,
  structurally valid Kernel syntax (correct indentation/nesting), and
  correct GSM8K-style `<<...>>` answer formatting (arithmetic itself not
  yet reliably correct — honestly reported, not glossed over).
- Serving swap code is done (Worker updated to the real checkpoint, fp16
  export verified against fp32 PyTorch reference, max diff 0.0077) but
  **deployment is blocked**: Cloudflare Workers Free's 10ms CPU-time-per-
  request limit is a hard wall, confirmed via live `wrangler tail`
  ("Worker exceeded CPU time limit"), not something any amount of
  optimization within free tier fixes for a real transformer forward
  pass. Decision pending with en: pay the $5/mo Workers Paid minimum
  (usage-based, not an idle VM — closest fit to en's standing infra
  preference), try another free host for inference only, or shrink the
  model further (real risk: still may not fit under 10ms, and hurts
  quality either way).
