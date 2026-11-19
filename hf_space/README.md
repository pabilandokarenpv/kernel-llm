---
title: Kernel AI
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
---

# Kernel AI — a genuinely from-scratch LLM

Not a wrapper around a hosted model. This is a 33M-parameter transformer
trained from random initialization on a 110,187-example merged corpus
(TinyStories, OASST2, CodeAlpaca-20k, Dolly15k, GSM8K, plus a Kernel-
language execution-trace corpus and distilled synthetic examples),
6,000 training steps on a free Kaggle T4 GPU — val_loss 10.08 → 2.27.

Inference here runs on pure numpy (no PyTorch at serving time) — the
exact same hand-verified forward pass built for Cloudflare Workers,
which turned out to need more CPU time per request than Workers' free
tier allows (10ms hard limit). HF Spaces' free CPU Basic tier runs a
persistent container instead, so the same code just runs here.

Honest expectations: small model, short coherent prose, structurally
correct code/math formatting — not always factually or numerically
correct. See the kernel-llm repo's `product/PRDs/PRD-001-kernel-ai-product.md`
for the full story, architecture docs, and what's still in progress.

Source: https://github.com/pabilandokarenpv/kernel-llm
