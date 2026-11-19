# ARD-003: Serving on Cloudflare Python Workers + pure numpy, after ONNX/onnxruntime-web failed

**Status:** Accepted

## Context

The whole project is built on Cloudflare (Workers, scale-to-zero, no idle
cost — matches en's standing infra preference). Serving a PyTorch-trained
checkpoint inside a Cloudflare Worker needs some runtime that can execute a
forward pass without PyTorch itself (too heavy for a Worker).

## Decision path (what was actually tried, in order)

```mermaid
flowchart TD
    A["PyTorch checkpoint\n(kernel_ai_v0.pt)"] --> B{"ONNX export"}
    B -->|"onnx + onnxscript"| C["kernel_ai_v0.onnx\nverified vs PyTorch: diff 0.000003"]
    C --> D["onnxruntime-node\n(local verification)"]
    D -->|"exact match"| E["✅ proves the ONNX graph is correct"]
    C --> F["onnxruntime-web inside workerd"]
    F -->|"WebAssembly.instantiate from raw bytes\nnot allowed in workerd"| G["❌ wall #1"]
    G -->|"instantiateWasm hook + precompiled Module\n(web-searched fix)"| H{"does onnxruntime-web 1.31.1\nforward instantiateWasm?"}
    H -->|"grepped the installed bundle directly:\nNO — confirmed real upstream gap\nmicrosoft/onnxruntime#34202"| I["❌ wall #2"]
    I -->|"monkey-patched ort.wasm.bundle.min.mjs directly"| J{"literal 'dummy' module\nspecifier import"}
    J -->|"unresolved"| K["❌ wall #3 — path abandoned"]
    E -.->|"architecture proven correct,\njust needs a different runtime"| L
    K --> L["Cloudflare native Python Workers\n(Pyodide + numpy support)"]
    L --> M["numpy_forward.py\nhand-written RMSNorm/RoPE/attention/SwiGLU"]
    M --> N["verify_numpy.py\nverified vs PyTorch: diff 0.000007"]
    N --> O["✅ deployed: kernel-ai-py-test worker"]

    style G fill:#4a1a1a,stroke:#822,color:#fff
    style I fill:#4a1a1a,stroke:#822,color:#fff
    style K fill:#4a1a1a,stroke:#822,color:#fff
    style O fill:#1a4a2e,stroke:#2a8,color:#fff
```

The ONNX export and onnxruntime-node verification (steps A-E) were not
wasted work — they proved the model's computation graph exports and
matches PyTorch exactly, isolating the failure to "workerd can't run
onnxruntime-web," not "our model/export is broken."

## Decision

Serve via a Cloudflare **Python Worker** (Pyodide runtime, `numpy`
available as a package), running a hand-written pure-numpy reimplementation
of the forward pass (`model/numpy_forward.py`) — RMSNorm, RoPE, QK-norm
attention, SwiGLU, matching the dense (no-MoE) architecture from ARD-001.
Weights are exported to a flat binary format (`export_numpy.py` →
`.npz` initially, later a raw manifest+float32-bytes format to avoid
zip/pickle parsing overhead) and fetched from GitHub at request time
(`raw.githubusercontent.com`), since direct R2 upload was blocked (see
"Rejected alternatives" below).

## Rejected alternatives

- **onnxruntime-web inside Workers** — real, confirmed upstream gap
  (`microsoft/onnxruntime#34202`: `ort.env.wasm.instantiateWasm` doesn't
  forward to the low-level Emscripten factory in v1.31.1's public API).
  A direct monkey-patch of the installed bundle got one layer further but
  hit a second, unrelated wall (a literal `"dummy"` module specifier
  treated as a real import) — abandoned as not worth continued fragile
  patching with no guaranteed payoff.
- **Modal/HF Spaces for serving** — considered when the ONNX path failed,
  but en's explicit standing preference is to keep infra on Cloudflare
  when a working path exists there; Python Workers turned out to be
  exactly that path.
- **R2 for weight storage** — attempted (bucket created successfully) but
  blocked: the OAuth-scoped Cloudflare MCP connector has no
  token-creation permission, and the Cloudflare dashboard itself hit the
  same bot-check wall that blocked LinkedIn automation earlier in the
  project. GitHub raw-file hosting was the workaround — the deployed
  Worker has full internet access at actual runtime (unlike the local dev
  sandbox, which blocks external fetches).

## Consequences

- Serving is genuinely free/scale-to-zero, matching infra preference —
  for placeholder-sized checkpoints only (see below, now confirmed false
  for the real model).
- **Confirmed, not just probabilistic**: once the real checkpoint (384d/
  8L/6h, 33M params, replacing the earlier 192d/3L/4h placeholder) was
  wired in and deployed, every single request hit Cloudflare's **10ms
  CPU-time-per-request Free-tier wall** — verified via live `wrangler
  tail`, real error `Worker exceeded CPU time limit`, not a cold-start
  edge case. This is a hard platform limit (10ms is far too small for any
  real transformer forward pass, not a tuning problem) — superseding the
  earlier "probabilistic 503s" framing below, which was accurate for the
  smaller placeholder checkpoint but doesn't describe what happens with a
  model actually worth serving. Decision on how to proceed (pay for
  Workers Paid, find another free inference host, or shrink the model) is
  pending with en — see PRD-001's Open items.
- (Historical, placeholder-checkpoint-era) cold-start CPU budget (numpy +
  httpx import + first fetch) sat right at the edge of the free Worker's
  10ms budget, causing probabilistic 503s on cold edges.
- MoE is now a hard non-option for serving (static-graph requirement —
  see ARD-001), locking that decision in beyond just "nice to have."
