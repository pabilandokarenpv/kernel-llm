# Architecture Decision Records — Kernel-LLM / Kernel AI

| ID | Title | Status |
|---|---|---|
| [ARD-001](./ARD-001-model-architecture.md) | Model architecture: RoPE + RMSNorm + QK-Norm + SwiGLU + Muon, dense (no MoE) for the serving target | Accepted |
| [ARD-002](./ARD-002-from-scratch-mandate.md) | From-scratch mandate: teacher model for data generation only, never serving | Accepted |
| [ARD-003](./ARD-003-serving-infrastructure.md) | Serving on Cloudflare Python Workers + pure numpy, after ONNX/onnxruntime-web failed | Accepted |
| [ARD-004](./ARD-004-training-infrastructure.md) | Training on Kaggle (cardless API-driven), after Modal's GPU card gate | Accepted |
| [ARD-005](./ARD-005-data-strategy.md) | Data strategy: real open datasets + narrow self-distillation, merged into one corpus | Accepted |
