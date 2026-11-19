# ARD-006: Serving via Cloudflare Tunnel, after both Workers Free and HF Spaces turned out paid-only

**Status:** Accepted

## Context

ARD-003 shipped serving on Cloudflare Python Workers (Pyodide + numpy).
Once the real trained checkpoint (33M params, 384d/8L/6h) replaced the
placeholder, every request hit Cloudflare's Workers Free tier CPU-time
limit — confirmed via live `wrangler tail`, real error `Worker exceeded
CPU time limit`. This is a hard platform wall: **10ms of CPU time per
request**, confirmed via Cloudflare's own docs, nowhere near enough for
even one transformer forward pass in Python. Not a tuning problem.

Two paid-tier options were identified and costed out:
- Cloudflare Workers Paid: $5/mo minimum, raises the CPU-time budget to
  30s (default, configurable to 5 min).
- Hugging Face Spaces: looked free for CPU Basic hardware per their own
  docs, but attempting to actually create a Gradio Space returned a real
  `402 Payment Required` — "hosting Gradio and Docker Spaces on free
  cpu-basic requires a PRO subscription," $9/mo. (HF Spaces' free tier
  historically did allow this; the docs search result had actually
  flagged this ambiguously, and the real API call confirmed it one way,
  not a guess.)

Both of these conflict with the project's standing $0-cost hard
requirement (see ARD-004's framing of this as "not just a preference").

## Decision

Serve via a **Cloudflare Tunnel** (`cloudflared`) pointed at a Gradio app
running on a persistent process, with DNS routed through en's existing
Cloudflare zone (`forgirlswhobuild.com`) — live at
`https://kernel-ai.forgirlswhobuild.com/`.

This sidesteps the CPU-time-per-request billing model entirely: a
Cloudflare Tunnel is a free reverse-proxy product (no card, no compute
charge — Cloudflare just proxies traffic to an origin someone else
runs), so there is no per-request CPU budget to hit. The actual model
forward pass runs via the exact same `numpy_forward.py` +
`bpe_tokenizer.py` code already verified byte-for-byte against the real
PyTorch model for the (abandoned) Workers path — no new model code, only
a different transport/host.

Setup, for reproducibility:
1. Created a named tunnel via the Cloudflare API
   (`POST /accounts/{id}/cfd_tunnel`) — required a **Global API Key**
   (`X-Auth-Email` + `X-Auth-Key` headers), not the existing scoped API
   token, which lacked the `Cloudflare Tunnel: Edit` permission.
2. Set the tunnel's ingress config via API: `kernel-ai.forgirlswhobuild.com`
   → `http://localhost:7860` (Gradio's default port).
3. Created a proxied CNAME DNS record pointing at
   `<tunnel-id>.cfargotunnel.com`.
4. Ran the Gradio app (`demo.launch(server_name="0.0.0.0",
   server_port=7860)`) and `cloudflared tunnel run --token <token>` as
   two persistent background processes on the same machine.

## Rejected alternatives

- **Pay for Cloudflare Workers Paid ($5/mo)** — rejected per the
  project's $0 hard requirement; the code for this path is already
  built and verified (fp16 export, Worker updated) if the requirement
  ever relaxes, so this isn't wasted work, just not deployed.
- **Pay for HF PRO ($9/mo)** — rejected for the same reason, and more
  expensive than Workers Paid besides.
- **Oracle Cloud "Always Free" tier** — genuinely $0 forever with real
  always-on compute (no CPU-time-per-request wall, since it's a real VM),
  but requires a credit card for identity verification at signup (an
  auth hold, not a charge) — the same category of step that already
  failed for Modal earlier in the project ("none of my cards are
  working"). Not attempted given that known risk; worth revisiting if
  the Tunnel approach ever needs to move off en's own infra.
- **Reframe to on-demand/local-only demos (Colab/Kaggle/local), no
  always-on URL** — considered and presented as an option, but en chose
  to keep a live public URL via the Tunnel instead.

## Consequences

- Genuinely $0/month, no card, matching the project's hard requirement —
  the first serving path this session that actually satisfies it.
- The origin (Gradio app + cloudflared) runs on infrastructure this
  agent's environment provides, not a traditional "en's own home
  server" — same tradeoff as the "always-on origin needed" dependency
  `ARD-003`'s rejected "Tunnel to your own machine" alternative named:
  something has to stay up continuously. Framed honestly: this isn't
  scale-to-zero the way Workers is, it's "a persistent process behind a
  free proxy" — acceptable here because there's no cost attached either
  way, only an uptime dependency.
- The Workers serving path (ARD-003) and its fp16-export/Worker code are
  not deleted — they're a validated fallback if this Tunnel's host ever
  needs to change, or if the $0 requirement is ever relaxed in favor of
  Workers Paid's larger CPU budget and more conventional ops story.
